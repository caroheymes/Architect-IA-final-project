import logging
import os
import shutil
import sys

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LyonFlow-Validation-Promotion")

# Add current directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import build_sliding_dataset
from model import SpatioTemporalGCN

# Environment variables
USE_LOCAL_CSV = os.getenv("USE_LOCAL_CSV", "true").lower() == "true"
DATA_FOLDER = os.getenv("DATA_FOLDER", "data/in")
HORIZONS_STR = os.getenv("HORIZONS", "1")
HORIZONS = [int(h) for h in HORIZONS_STR.split(",") if h.strip()]

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")


def evaluate_model(model, loader, device, mean_tensor, scale_tensor):
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index)

            # De-normalize to km/h for true MAE
            pred_kmh = predictions * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)
            y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)

            mae = torch.abs(pred_kmh - y_kmh).mean().item()
            total_mae += mae * batch.num_graphs
            total_samples += batch.num_graphs

    return total_mae / total_samples if total_samples > 0 else float("inf")


def main():
    logger.info("🏁 Starting Model Validation & Promotion Process...")

    # Load optimal hyperparameters from Optuna/MLflow to get the correct model architecture
    logger.info("Retrieving optimal hyperparameters from Optuna/MLflow...")
    best_params = None
    try:
        import get_best_params

        best_params = get_best_params.get_params_from_optuna()
        if not best_params:
            best_params = get_best_params.get_params_from_mlflow()
    except Exception as e:
        logger.warning(f"⚠️ Could not load optimal hyperparameters script: {e}")

    if not best_params:
        logger.warning("⚠️ No optimal parameters found. Using default architecture.")
        best_params = {
            "hidden_channels": 128,
            "seq_len": 120,
            "batch_size": 2,
        }

    # Setup parameters
    hidden_channels = int(best_params.get("hidden_channels", 128))
    seq_len = int(best_params.get("seq_len", 120))
    batch_size = int(best_params.get("batch_size", 2))

    logger.info(f"Target Model Architecture -> HIDDEN_CHANNELS: {hidden_channels}, SEQ_LEN: {seq_len}")

    # Paths
    new_model_path = "models/stgcn_v2_latest.pt"
    new_scaler_path = "models/stgcn_v2_scaler.pkl"
    prod_model_path = "models/stgcn_prod_latest.pt"
    prod_scaler_path = "models/stgcn_scaler.pkl"

    if not os.path.exists(new_model_path):
        logger.error(f"❌ New model file not found: {new_model_path}")
        sys.exit(1)

    # 1. Load Data
    if USE_LOCAL_CSV:
        logger.info(f"📁 Loading local data from {DATA_FOLDER}...")
        from dataset import load_graph_topology_from_csv, load_traffic_series_from_csv

        num_nodes, edge_index = load_graph_topology_from_csv(DATA_FOLDER)
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series_from_csv(DATA_FOLDER)
    else:
        logger.info("📡 Connecting to PostgreSQL database...")
        from sqlalchemy import create_engine

        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        from dataset import load_graph_topology, load_traffic_series

        num_nodes, edge_index = load_graph_topology(engine)
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series(engine)
        engine.dispose()

    _, test_loader, scaler = build_sliding_dataset(
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
        horizons=HORIZONS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

    # 2. Evaluate New Model
    logger.info("Evaluating newly trained model...")
    new_model = SpatioTemporalGCN(in_channels=5, hidden_channels=hidden_channels, out_channels=len(HORIZONS)).to(device)
    new_model.load_state_dict(torch.load(new_model_path, map_location=device))
    mae_new = evaluate_model(new_model, test_loader, device, mean_tensor, scale_tensor)
    logger.info(f"🆕 New Model Validation MAE: {mae_new:.4f} km/h")

    # 3. Evaluate Current Prod Model (if exists)
    mae_prod = float("inf")
    if os.path.exists(prod_model_path):
        logger.info("Evaluating current production model...")
        try:
            # Note: The current production model might have been trained with a different hidden_channels size (e.g. 128)
            # We will try to load it dynamically by inspecting its state_dict keys if possible, or try the default/loaded ones.
            prod_state_dict = torch.load(prod_model_path, map_location=device)
            # Infer hidden channels from the spatial_gcn1.bias shape
            inferred_channels = 128
            if "spatial_gcn1.bias" in prod_state_dict:
                inferred_channels = prod_state_dict["spatial_gcn1.bias"].shape[0]
                logger.info(f"Inferred HIDDEN_CHANNELS of prod model: {inferred_channels}")

            prod_model = SpatioTemporalGCN(
                in_channels=5, hidden_channels=inferred_channels, out_channels=len(HORIZONS)
            ).to(device)
            prod_model.load_state_dict(prod_state_dict)
            mae_prod = evaluate_model(prod_model, test_loader, device, mean_tensor, scale_tensor)
            logger.info(f"🟢 Current Production Model Validation MAE: {mae_prod:.4f} km/h")
        except Exception as e:
            logger.warning(f"⚠️ Failed to evaluate current production model: {e}. Assuming new model is better.")
    else:
        logger.info("ℹ️ No active production model found.")

    # 4. Compare and Promote
    if mae_new < mae_prod:
        logger.info("🏆 New model is BETTER! Promoting to production...")
        shutil.copy(new_model_path, prod_model_path)
        shutil.copy(new_scaler_path, prod_scaler_path)

        # Also copy the error analysis plot if it exists
        new_plot_path = "models/stratified_error_analysis_v2.png"
        prod_plot_path = "models/stratified_error_analysis.png"
        if os.path.exists(new_plot_path):
            shutil.copy(new_plot_path, prod_plot_path)

        logger.info("🚀 Promotion successful! Production model updated.")
        print("PROMOTION_SUCCESSFUL: true")
    else:
        logger.info("❌ New model did NOT outperform the current production model. Keeping current model.")
        print("PROMOTION_SUCCESSFUL: false")


if __name__ == "__main__":
    main()
