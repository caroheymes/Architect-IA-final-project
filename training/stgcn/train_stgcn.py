import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy import create_engine
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
SEQ_LEN = int(os.getenv("SEQ_LEN", "120"))            # 10h of historical series (5min steps)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))           # Reduced to fit 10h seq_len (120) in 4GB VRAM
HIDDEN_CHANNELS = int(os.getenv("HIDDEN_CHANNELS", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
EPOCHS = int(os.getenv("EPOCHS", "100"))

# Staircase weighted loss penalties (for under-prediction of congestions)
WEIGHT_JAM = float(os.getenv("WEIGHT_JAM", "15.0"))        # < 10 km/h
WEIGHT_SLOW = float(os.getenv("WEIGHT_SLOW", "5.0"))        # 10 - 30 km/h
WEIGHT_NORMAL = float(os.getenv("WEIGHT_NORMAL", "1.0"))      # > 30 km/h

def train_model():
    # Capping PyTorch threads to avoid CPU thrashing on laptops
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # 1. Establish PostgreSQL Connection
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    logger.info(f"🔌 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
    
    # 2. Setup MLflow Tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("LyonFlow-STGCN-Production-Training-v2")
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

        # Early Stopping configuration
        best_test_mae = float('inf')
        patience = 10
        patience_counter = 0
        best_epoch = 0
        best_model_state = None

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
                # Gradient clipping to prevent gradient explosion (critical for GRUs/GNNs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
            # Early Stopping Check
            if test_mae_kmh < best_test_mae:
                best_test_mae = test_mae_kmh
                patience_counter = 0
                best_epoch = epoch
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
                # Save best model checkpoint
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/stgcn_prod_latest.pt")
                logger.info(f"🏆 New best model found at epoch {epoch:02d} with Test MAE: {best_test_mae:.4f} km/h.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch:02d}. Best Test MAE: {best_test_mae:.4f} km/h at epoch {best_epoch:02d}.")
                    break
            
        # Restore best model weights for final evaluation & registry
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            logger.info(f"♻️ Restored best model weights from epoch {best_epoch:02d} with Test MAE: {best_test_mae:.4f} km/h.")
        else:
            # Fallback if no training occurred
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/stgcn_prod_latest.pt")
            logger.info(f"💾 Saved last epoch model weights as fallback.")
        
        # =====================================================================
        # 6. Stratified Error Analysis (Speed Bins) & MLflow Logging
        # =====================================================================
        logger.info("📊 Performing stratified error analysis on test set...")
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                predictions = model(batch.x, batch.edge_index)
                
                # De-normalize vectorially on GPU
                B_curr = batch.num_graphs
                scale_batch = scale_tensor.repeat(B_curr, 1)
                mean_batch = mean_tensor.repeat(B_curr, 1)
                
                preds_kmh = predictions * scale_batch + mean_batch
                targets_kmh = batch.y * scale_batch + mean_batch
                
                all_preds.append(preds_kmh.cpu().numpy().flatten())
                all_targets.append(targets_kmh.cpu().numpy().flatten())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Define bins
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
        bin_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90+"]

        mae_by_bin = []
        bias_by_bin = []
        std_preds_by_bin = []
        std_errors_by_bin = []
        count_by_bin = []
        box_data = []

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i+1]
            indices = np.where((all_targets >= lower) & (all_targets < upper))[0]
            
            if len(indices) > 0:
                mae = float(np.mean(np.abs(all_preds[indices] - all_targets[indices])))
                bias = float(np.mean(all_preds[indices] - all_targets[indices]))
                std_pred = float(np.std(all_preds[indices]))
                std_err = float(np.std(all_preds[indices] - all_targets[indices]))
                count = int(len(indices))
                
                mae_by_bin.append(mae)
                bias_by_bin.append(bias)
                std_preds_by_bin.append(std_pred)
                std_errors_by_bin.append(std_err)
                count_by_bin.append(count)
                box_data.append(all_preds[indices])
                
                # In-memory logging to local lists done, skipping per-bin log_metric to keep the MLflow Metrics dashboard clean.
            else:
                mae_by_bin.append(0.0)
                bias_by_bin.append(0.0)
                std_preds_by_bin.append(0.0)
                std_errors_by_bin.append(0.0)
                count_by_bin.append(0)
                box_data.append([])

        df_analysis = pd.DataFrame({
            "Tranche (km/h)": bin_labels,
            "Nombre d'exemples": count_by_bin,
            "MAE (km/h)": mae_by_bin,
            "Biais (km/h)": bias_by_bin,
            "Écart-type Prédiction (km/h)": std_preds_by_bin,
            "Écart-type Erreur (km/h)": std_errors_by_bin
        })

        logger.info("\n📊 Tableau Récapitulatif des Erreurs par Tranche de Vitesse réelle :")
        logger.info(f"\n{df_analysis.to_string(index=False)}")

        # Save analysis table as CSV artifact
        os.makedirs("models", exist_ok=True)
        csv_path = "models/stratified_error_analysis.csv"
        df_analysis.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="analysis")
        logger.info(f"💾 Analysis CSV successfully saved and logged to MLflow as artifact: {csv_path}")

        # Plot figures
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        axs = axs.ravel()

        # --- Graphique 1 : MAE et Nombre d'exemples par tranche ---
        color = 'tab:blue'
        axs[0].set_title("1. Erreur Absolue Moyenne (MAE) et Volume d'exemples", fontsize=12, fontweight='bold')
        axs[0].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[0].set_ylabel("MAE (km/h)", color=color)
        axs[0].bar(bin_labels, mae_by_bin, color=color, alpha=0.6, label="MAE (km/h)")
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].grid(True, linestyle="--", alpha=0.3)

        ax1_sub = axs[0].twinx()
        color = 'tab:grey'
        ax1_sub.set_ylabel("Nombre d'exemples", color=color)
        ax1_sub.plot(bin_labels, count_by_bin, color=color, marker="o", linestyle=":", linewidth=2, label="Volume d'exemples")
        ax1_sub.tick_params(axis='y', labelcolor=color)

        # --- Graphique 2 : Biais de Prédiction Systématique ---
        axs[1].axhline(0, color="black", linestyle="--", alpha=0.7)
        axs[1].plot(bin_labels, bias_by_bin, marker="s", color="red", linewidth=2.5, label="Biais systématique (Prédit - Réel)")
        bias_by_bin_arr = np.array(bias_by_bin)
        axs[1].fill_between(bin_labels, bias_by_bin_arr, 0, where=bias_by_bin_arr > 0, color="orange", alpha=0.3, interpolate=True, label="Sur-évaluation (vitesse sur-estimée)")
        axs[1].fill_between(bin_labels, bias_by_bin_arr, 0, where=bias_by_bin_arr < 0, color="blue", alpha=0.15, interpolate=True, label="Sous-évaluation (vitesse sous-estimée)")
        axs[1].set_title("2. Biais de Prédiction Systématique", fontsize=12, fontweight='bold')
        axs[1].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[1].set_ylabel("Biais (km/h) [Positif = Sur-évaluation | Négatif = Sous-évaluation]")
        axs[1].set_ylim(-40, 40)
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].legend()

        # --- Graphique 3 : Écarts-types (Évaluation de la Dispersion) ---
        axs[2].plot(bin_labels, std_preds_by_bin, marker="o", color="purple", linewidth=2.5, label="Écart-type des Vitesses Prédites (Dispersion)")
        axs[2].plot(bin_labels, std_errors_by_bin, marker="^", color="teal", linewidth=2.5, linestyle="--", label="Écart-type des Erreurs (Incertitude des résidus)")
        axs[2].set_title("3. Écarts-types par Tranche de Vitesse", fontsize=12, fontweight='bold')
        axs[2].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[2].set_ylabel("Écart-type (km/h)")
        axs[2].grid(True, linestyle="--", alpha=0.5)
        axs[2].legend()

        # --- Graphique 4 : Distribution des Prédictions vs Réel (Boxplot) ---
        axs[3].boxplot(box_data, labels=bin_labels, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="#E6E6FA", color="#5D3FD3", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2))

        bin_centers = []
        for i in range(len(bins)-1):
            bin_centers.append((bins[i] + bins[i+1])/2)

        axs[3].plot(range(1, len(bin_labels) + 1), bin_centers, color="blue", linestyle="--", marker="x", linewidth=2, label="Vitesse Réelle Cible (Milieu de classe)")
        axs[3].set_title("4. Boîtes à Moustaches (Boxplots) des Vitesses Prédites", fontsize=12, fontweight='bold')
        axs[3].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[3].set_ylabel("Vitesse Prédite par le Modèle (km/h)")
        axs[3].grid(True, linestyle="--", alpha=0.5)
        axs[3].legend()

        plt.tight_layout()
        
        # Save figure locally and log to MLflow
        os.makedirs("models", exist_ok=True)
        local_plot_path = "models/stratified_error_analysis.png"
        fig.savefig(local_plot_path, dpi=300)
        logger.info(f"🎨 Analysis plot successfully saved locally to {local_plot_path}")

        mlflow.log_figure(fig, "plots/stratified_error_analysis.png")
        plt.close(fig)
        logger.info("🎨 Analysis plot successfully logged to MLflow under 'plots/stratified_error_analysis.png'!")

        # Log model checkpoint
        mlflow.log_artifact("models/stgcn_prod_latest.pt", artifact_path="model_checkpoints")
        logger.info("🏆 Production training completed successfully!")


if __name__ == "__main__":
    train_model()
