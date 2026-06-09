import datetime
import logging
import os

import mlflow
import numpy as np
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
pyg_data_list = None
scaler_data = None
split_idx_data = None


def get_db_url():
    """Construit l'URL SQLAlchemy PostgreSQL à partir des variables d'environnement.

    Returns:
        str: URL de la forme `postgresql+psycopg2://user:pass@host:port/db`.
    """
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    """Crée un SQLAlchemy `Engine` à partir de `get_db_url()`.

    Returns:
        sqlalchemy.Engine: Engine prêt à l'emploi.
    """
    return create_engine(get_db_url())


def objective(trial):
    """Fonction objectif Optuna — un trial d'optimisation bayésienne.

    Le sampler TPE explore l'espace des hyperparamètres suivant :
      - `lr` ∈ [1e-4, 1e-2] (log)
      - `hidden_channels` ∈ {64, 128, 256}
      - `weight_decay` ∈ [1e-6, 1e-4] (log)
      - `seq_len` ∈ {6, 12, 18, 24} pas (30 min, 1h, 1h30, 2h)
      - `batch_size` ∈ {8, 16}
      - `weight_jam` ∈ [5, 20] et `weight_slow` ∈ [2, 8] :
        pondérations de la loss pour les embouteillages et les ralentissements.

    Pour chaque trial :
      1. Construit les loaders PyG avec les hyperparamètres.
      2. Entraîne pendant `max_epochs=15` (rapide pour HPO).
      3. Dénormalise les prédictions vers km/h pour appliquer la loss
         pondérée « en escalier » et calcule la MAE en km/h.
      4. `trial.report(...)` + `trial.should_prune()` → élagage agressif
         des trials médiocres via `MedianPruner`.

    Args:
        trial (optuna.Trial): Trial Optuna courant.

    Returns:
        float: MAE (km/h) sur le set de test à la fin de l'entraînement.

    Raises:
        optuna.exceptions.TrialPruned: Si le MedianPruner décide d'arrêter
        le trial avant la fin.
    """
    global topology_data, traffic_data, pyg_data_list, scaler_data, split_idx_data

    # 1. Sample Hyperparameters via Bayesian Search (TPE)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Aligné sur l'inférence : seq_len = 120 (10 heures d'historique)
    seq_len = 120
    
    # Taille de lot adaptée pour éviter les débordements de mémoire (OOM VRAM) avec seq_len=120
    batch_size = trial.suggest_categorical("batch_size", [1, 2])

    # Congestion penalization weights
    weight_jam = trial.suggest_float("weight_jam", 5.0, 20.0)
    weight_slow = trial.suggest_float("weight_slow", 2.0, 8.0)

    # 2. Get Cached Data (Loaded once on main entrypoint)
    num_nodes, edge_index = topology_data
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = traffic_data

    # 3. Build sliding window loaders (use cached list of Data samples if available)
    if pyg_data_list is None:
        logger.info("⏱️ Constructing sliding window dataset for the first trial (will be cached)...")
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
            batch_size=1,  # dummy batch_size
            horizons=[6, 12, 36],  # 30 min, 1h, 3h
        )
        pyg_data_list = train_loader.dataset + test_loader.dataset
        split_idx_data = len(train_loader.dataset)
        scaler_data = scaler
        logger.info(f"✅ Sliding window dataset constructed with {len(pyg_data_list)} samples and cached.")
    else:
        scaler = scaler_data

    # Create PyG DataLoaders on the fly with the trial's batch_size
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(pyg_data_list[:split_idx_data], batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(pyg_data_list[split_idx_data:], batch_size=batch_size, shuffle=False)

    # 4. Device and Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(in_channels=5, hidden_channels=hidden_channels, out_channels=3).to(device)
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

        test_mae_kmh = mae_metric / (len(test_loader.dataset) * num_nodes * 3)  # Divisé par 3 horizons

        logger.info(f"  [Trial {trial.number}] Epoch {epoch}/{max_epochs} - Test MAE: {test_mae_kmh:.4f} km/h")

        # Report trial intermediate metrics for Pruning decisions
        trial.report(test_mae_kmh, epoch)

        # Prune unpromising trials (e.g. median performance threshold)
        if trial.should_prune():
            logger.info(f"  ⚠️ [Trial {trial.number}] Pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    return test_mae_kmh


def run_hpo():
    """Lance l'étude d'optimisation bayésienne Optuna (HPO).

    Étapes :
      1. **Warm-up du cache module-level** : on charge la topologie et les
         séries temporelles **une seule fois** dans `topology_data` /
         `traffic_data`, puis tous les trials réutilisent ce cache (gain
         massif sur le temps d'HPO : on évite N requêtes DB).
      2. Setup MLflow tracking (experiment `LyonFlow-STGCN-Optuna-Tuning`).
      3. Storage Optuna = PostgreSQL via `RDBStorage` pour permettre le
         scaling multi-worker et la reprise après crash.
      4. `optuna.create_study` avec :
         - `direction="minimize"` (on minimise la MAE),
         - `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` pour
           l'élagage précoce,
         - `load_if_exists=True` pour reprendre une étude existante.
      5. `study.optimize(objective, n_trials=20)` puis log des meilleurs
         hyperparamètres et de la MAE dans MLflow.

    Imprime à la fin la commande pour lancer le dashboard Optuna.
    """
    global topology_data, traffic_data
    logger.info("🚀 Initializing STGCN Hyperparameter Tuning...")

    USE_LOCAL_CSV = os.getenv("USE_LOCAL_CSV", "false").lower() == "true"
    DATA_FOLDER = os.getenv("DATA_FOLDER", "/home/ray/project/data/in")

    if USE_LOCAL_CSV:
        logger.info(f"📁 [CACHE] Chargement local depuis les fichiers CSV dans {DATA_FOLDER}...")
        from dataset import load_graph_topology_from_csv, load_traffic_series_from_csv
        topology_data = load_graph_topology_from_csv(DATA_FOLDER)
        traffic_data = load_traffic_series_from_csv(DATA_FOLDER)
        logger.info("✅ Cache local de données CSV initialisé.")
    else:
        logger.info("📡 [CACHE] Connexion PostgreSQL pour chargement des données...")
        engine = get_engine()
        topology_data = load_graph_topology(engine)
        num_nodes, edge_index = topology_data
        
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series(engine)
        
        # Aligner la matrice de vitesse pour qu'elle ait exactement num_nodes colonnes (prévention shape mismatch)
        if vitesse_matrix_raw.shape[1] != num_nodes:
            logger.warning(f"⚠️ Alignement de la matrice : {vitesse_matrix_raw.shape[1]} -> {num_nodes} colonnes.")
            default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
            padded_vitesse = np.full((vitesse_matrix_raw.shape[0], num_nodes), default_speed)
            cols_to_copy = min(vitesse_matrix_raw.shape[1], num_nodes)
            padded_vitesse[:, :cols_to_copy] = vitesse_matrix_raw[:, :cols_to_copy]
            vitesse_matrix_raw = padded_vitesse
            
        traffic_data = (vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos)
        engine.dispose()
        logger.info("✅ Cache de données PostgreSQL initialisé et aligné.")

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
        study_name="lyonflow_stgcn_tuning_seq120",
        storage=postgres_storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # 5. Run Optimize
    run_name = f"Optuna_HPO_Study_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n_trials = int(os.getenv("N_TRIALS", "20"))
    with mlflow.start_run(run_name=run_name):
        logger.info(f"Bayesian hyperparameter optimization in progress ({n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials)

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
