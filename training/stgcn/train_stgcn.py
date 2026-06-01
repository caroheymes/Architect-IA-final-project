import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy import create_engine
import mlflow

# Import local modules
from model import SpatioTemporalGCN
from dataset import load_graph_topology, load_traffic_series, build_sliding_dataset

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("LyonFlow-STGCN-Train")

# Environment and database configuration
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# MLflow configuration
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Default Hyperparameters
SEQ_LEN = 12             # 1h of historical series (5min steps)
BATCH_SIZE = 16          # Keeping it safe for 4GB VRAM
HIDDEN_CHANNELS = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 20

# Staircase weighted loss penalties (for under-prediction of congestions)
WEIGHT_JAM = 15.0        # < 10 km/h
WEIGHT_SLOW = 5.0        # 10 - 30 km/h
WEIGHT_NORMAL = 1.0      # > 30 km/h

def train_model():
    # 1. Establish PostgreSQL Connection
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    logger.info(f"🔌 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
    
    # 2. Setup MLflow Tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("LyonFlow-STGCN-Production-Training")
    except Exception as e:
        logger.warning(f"⚠️ MLflow server unavailable at {MLFLOW_URL} ({e}). Running locally.")

    # 3. Load topology and facts
    logger.info("📡 Loading spatial topology and traffic timeseries from Postgres...")
    num_nodes, edge_index = load_graph_topology(engine)
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series(engine)
    logger.info(f"✅ Graphe chargé : {num_nodes} nœuds, {edge_index.shape[1]} arêtes.")

    # 4. Build PyG loaders
    train_loader, test_loader, scaler = build_sliding_dataset(
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos,
        seq_len=SEQ_LEN, edge_index_tensor=edge_index, num_nodes=num_nodes,
        test_split=0.2, batch_size=BATCH_SIZE
    )

    # 5. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatioTemporalGCN(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Push scaler params to GPU for fast on-the-fly denormalization in loss computation
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

    logger.info(f"🚀 Starting STGCN Training on device: {device.type.upper()}")
    
    with mlflow.start_run(run_name=f"STGCN_Prod_Train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log params
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_channels": HIDDEN_CHANNELS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "weight_jam": WEIGHT_JAM,
            "weight_slow": WEIGHT_SLOW
        })

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                predictions = model(batch.x, batch.edge_index)
                
                # De-normalize target values to apply km/h based staircase penalties
                y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)
                
                # Compute custom weighted penalties
                weights = torch.where(
                    y_kmh < 10.0,
                    torch.tensor(WEIGHT_JAM, device=device),
                    torch.where(
                        y_kmh < 30.0,
                        torch.tensor(WEIGHT_SLOW, device=device),
                        torch.tensor(WEIGHT_NORMAL, device=device)
                    )
                )
                
                # Weighted MSE Loss
                loss = (((predictions - batch.y) ** 2) * weights).mean()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
            
            epoch_loss = train_loss / len(train_loader.dataset)
            
            # Evaluate on chronological test set
            model.eval()
            mae_metric = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    predictions = model(batch.x, batch.edge_index)
                    
                    # Convert standard scores back to physical km/h
                    B_curr = batch.num_graphs
                    preds_kmh = predictions * scale_tensor.repeat(B_curr, 1) + mean_tensor.repeat(B_curr, 1)
                    targets_kmh = batch.y * scale_tensor.repeat(B_curr, 1) + mean_tensor.repeat(B_curr, 1)
                    
                    mae_metric += F.l1_loss(preds_kmh, targets_kmh, reduction='sum').item()
                    
            test_mae_kmh = mae_metric / (len(test_loader.dataset) * num_nodes)
            
            # Log metrics to console and MLflow
            logger.info(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss (std): {epoch_loss:.4f} | Test MAE (km/h): {test_mae_kmh:.4f}")
            mlflow.log_metric("train_loss_std", epoch_loss, step=epoch)
            mlflow.log_metric("test_mae_kmh", test_mae_kmh, step=epoch)
            
        # Save model checkpoint
        os.makedirs("models", exist_ok=True)
        model_path = "models/stgcn_prod_latest.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Model weights successfully saved to: {model_path}")
        
        # Log model artifact to MLflow
        mlflow.log_artifact(model_path, artifact_path="model_checkpoints")
        logger.info("🏆 Production training completed successfully!")

if __name__ == "__main__":
    train_model()
