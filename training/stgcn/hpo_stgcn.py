import datetime
import logging
import os

import mlflow
import optuna
import torch
import torch.nn.functional as F
from dataset import build_sliding_dataset, load_graph_topology, load_traffic_series

# Import local modules
from model import SpatioTemporalGCN
from sqlalchemy import create_engine

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LyonFlow-STGCN-HPO")

# Environment and database configuration
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# MLflow configuration
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Cache data loading at module-level to avoid database overhead on each Optuna trial
topology_data = None
traffic_data = None


def get_db_url():
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    return create_engine(get_db_url())


def objective(trial):
    global topology_data, traffic_data

    # 1. Sample Hyperparameters via Bayesian Search (TPE)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    seq_len = trial.suggest_int("seq_len", 6, 24, step=6)  # 30 min, 1h, 1h30, or 2h of history
    batch_size = trial.suggest_categorical("batch_size", [8, 16])  # Safe for VRAM limits

    # Congestion penalization weights
    weight_jam = trial.suggest_float("weight_jam", 5.0, 20.0)
    weight_slow = trial.suggest_float("weight_slow", 2.0, 8.0)

    # 2. Get Cached Data (Loaded once on main entrypoint)
    num_nodes, edge_index = topology_data
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = traffic_data

    # 3. Build sliding window loaders for this trial's seq_len & batch_size
    train_loader, test_loader, scaler = build_sliding_dataset(
        vitesse_matrix_raw,
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
        seq_len=seq_len,
        edge_index_tensor=edge_index,
        num_nodes=num_nodes,
        test_split=0.2,
        batch_size=batch_size,
    )

    # 4. Device and Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(in_channels=5, hidden_channels=hidden_channels, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

    # 5. Rapid Training Loop (Max 15 epochs for quick HPO evaluations)
    max_epochs = 15
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            predictions = model(batch.x, batch.edge_index)

            # De-normalize to compute staircase loss penalties on km/h scale
            y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)

            # Penalize under-predictions
            weights = torch.where(
                y_kmh < 10.0,
                torch.tensor(weight_jam, device=device),
                torch.where(y_kmh < 30.0, torch.tensor(weight_slow, device=device), torch.tensor(1.0, device=device)),
            )

            loss = (((predictions - batch.y) ** 2) * weights).mean()
            loss.backward()
            # Gradient clipping to prevent gradient explosion (critical for GRUs/GNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluation
        model.eval()
        mae_metric = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                predictions = model(batch.x, batch.edge_index)

                B_curr = batch.num_graphs
                preds_kmh = predictions * scale_tensor.repeat(B_curr, 1) + mean_tensor.repeat(B_curr, 1)
                targets_kmh = batch.y * scale_tensor.repeat(B_curr, 1) + mean_tensor.repeat(B_curr, 1)

                mae_metric += F.l1_loss(preds_kmh, targets_kmh, reduction="sum").item()

        test_mae_kmh = mae_metric / (len(test_loader.dataset) * num_nodes)

        # Report trial intermediate metrics for Pruning decisions
        trial.report(test_mae_kmh, epoch)

        # Prune unpromising trials (e.g. median performance threshold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_mae_kmh


def run_hpo():
    global topology_data, traffic_data
    logger.info("🚀 Initializing STGCN Hyperparameter Tuning...")

    # Initialize SQL Alchemy engine
    engine = get_engine()

    # 1. Warm-up topology and traffic data cache
    logger.info("💾 Loading topology and timeseries database cache...")
    topology_data = load_graph_topology(engine)
    traffic_data = load_traffic_series(engine)
    logger.info("✅ Database cache warmed-up.")

    # 2. Setup MLflow Tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("LyonFlow-STGCN-Optuna-Tuning")
    except Exception as e:
        logger.warning(f"⚠️ MLflow server unavailable at {MLFLOW_URL} ({e}). Logging only in RDB.")

    # 3. Setup PostgreSQL storage for robust cross-worker Optuna studies
    postgres_storage = optuna.storages.RDBStorage(url=get_db_url(), engine_kwargs={"pool_pre_ping": True})

    # 4. Create or Load Study
    study = optuna.create_study(
        study_name="lyonflow_stgcn_tuning_v1",
        storage=postgres_storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # 5. Run Optimize
    run_name = f"Optuna_HPO_Study_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        logger.info("Bayesian hyperparameter optimization in progress...")
        study.optimize(objective, n_trials=20)

        # Log best results
        logger.info(f"🏆 Best trial finished with MAE: {study.best_value:.4f} km/h")
        logger.info(f"📊 Optimal hyperparams: {study.best_params}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mae_kmh", study.best_value)

        print("\n" + "=" * 60)
        print("🎉 OPTUNA BAYESIAN SEARCH COMPLETED SUCCESSFULLY!")
        print("To visualize hyperparameter importance, launch:")
        print(f"  optuna-dashboard postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    run_hpo()
