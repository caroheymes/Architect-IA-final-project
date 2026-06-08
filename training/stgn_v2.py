"""
train_stgcn_vps.py
==================
Variante VPS de train_stgcn.py : se connecte directement au PostgreSQL
distant (51.83.159.224:5432, base lyonflow) et lit les 3 tables `gold.*`
déjà préparées par build_gold_training_inputs.py :
  - gold.dim_spatial_grid_mapping
  - gold.dim_gnn_adjacency
  - gold.mv_fact_traffic_pivot

Dataset attendu : 1109 nœuds, ~4436 arêtes, 1.15M observations
(~1145 timesteps de 5 min ≈ 4 jours d'historique trafic).

USAGE (depuis M:\\Test) :
    .\\.venv\\Scripts\\python.exe .\train_stgcn_vps.py

Prérequis :
  - venv Python 3.12 (torch CUDA, torch-geometric, etc.) déjà créé
  - M:\\Test\\model.py et M:\\Test\\dataset.py en place (imports locaux)
  - Tables gold.* peuplées sur le VPS (script build_gold_training_inputs.py
    déjà exécuté au moins une fois)
"""

from __future__ import annotations

import datetime
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import build_sliding_dataset, load_graph_topology, load_traffic_series

# Import local modules
from model import SpatioTemporalGCN
from sqlalchemy import create_engine

# Logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LyonFlow-STGCN-VPS")

# ============================================================================
# CONFIGURATION POSTGRESQL — VPS LyonFlow
# ============================================================================
# Le .env du projet pointe vers localhost (stack Docker interne). On force
# la cible publique du VPS pour taper directement sur les gold tables.
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD", "xYjM0hwMgkXtOABJdl7NOWsimWOzvbQcLLx9cKSE"
)
DB_HOST = os.getenv("POSTGRES_HOST", "51.83.159.224")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# ============================================================================
# MLFLOW — backend fichier local (MLflow 3.x a déprécié le file-store par
# défaut : penser à exporter MLFLOW_ALLOW_FILE_STORE=true avant de lancer)
# ============================================================================
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# ============================================================================
# HYPERPARAMÈTRES — adaptés à la taille du dataset VPS
# ============================================================================
SEQ_LEN = int(os.getenv("SEQ_LEN", "120"))            # 10h d'historique
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))        # prudent sur gros graphes
HIDDEN_CHANNELS = int(os.getenv("HIDDEN_CHANNELS", "128"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
PATIENCE = int(os.getenv("PATIENCE", "10"))
MODEL_OUT = os.getenv("MODEL_OUT", "models/stgcn_vps_latest.pt")

# Pondérations de la loss (sous-prédiction des bouchons = sur-coût)
WEIGHT_JAM = float(os.getenv("WEIGHT_JAM", "15.0"))      # < 10 km/h
WEIGHT_SLOW = float(os.getenv("WEIGHT_SLOW", "5.0"))      # 10-30 km/h
WEIGHT_NORMAL = float(os.getenv("WEIGHT_NORMAL", "1.0"))  # > 30 km/h


def train_model():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # --- 1) Connexion PostgreSQL distante ----------------------------------
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url, connect_args={"connect_timeout": 15})
    logger.info(f"🔌 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}")

    # --- 2) MLflow Tracking (file store local) ------------------------------
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("LyonFlow-STGCN-Production-Training-VPS")
    except Exception as e:
        logger.warning(f"⚠️ MLflow setup failed ({e}). Continuing in best-effort mode.")

    # --- 3) Chargement direct depuis gold.* (transformations déjà appliquées
    #         par build_gold_training_inputs.py : jointure spatial/topologie
    #         et pivot long→large déjà effectués) ---------------------------
    logger.info("📡 Chargement direct depuis gold.* sur le VPS…")
    num_nodes, edge_index = load_graph_topology(engine)
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series(engine)
    logger.info(f"✅ Graphe chargé : {num_nodes} nœuds, {edge_index.shape[1]} arêtes.")
    engine.dispose()

    # --- 4) Horizons de prédiction (en pas de 5 min) -----------------------
    HORIZONS_STR = os.getenv("HORIZONS", "1")
    HORIZONS = [int(h) for h in HORIZONS_STR.split(",") if h.strip()]
    logger.info(
        f"🔮 Horizons configurés : {HORIZONS} "
        f"({[h * 5 for h in HORIZONS]} minutes)"
    )

    train_loader, test_loader, scaler = build_sliding_dataset(
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos,
        seq_len=SEQ_LEN, edge_index_tensor=edge_index, num_nodes=num_nodes,
        test_split=0.2, batch_size=BATCH_SIZE, horizons=HORIZONS,
    )

    # --- 5) Modèle & optimiseur -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(
        in_channels=5,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=len(HORIZONS),
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    mean_tensor = torch.tensor(
        scaler.mean_, dtype=torch.float, device=device
    ).view(-1, 1)
    scale_tensor = torch.tensor(
        scaler.scale_, dtype=torch.float, device=device
    ).view(-1, 1)

    logger.info(f"🚀 Starting STGCN VPS Training on device: {device.type.upper()}")

    with mlflow.start_run(
        run_name=f"STGCN_VPS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_channels": HIDDEN_CHANNELS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "weight_jam": WEIGHT_JAM,
            "weight_slow": WEIGHT_SLOW,
            "weight_normal": WEIGHT_NORMAL,
            "db_host": DB_HOST,
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[1]),
        })

        best_test_mae = float("inf")
        patience = PATIENCE
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

                # De-normalize target for km/h-based weighted penalties
                y_kmh = (
                    batch.y * scale_tensor.repeat(batch.num_graphs, 1)
                    + mean_tensor.repeat(batch.num_graphs, 1)
                )
                weights = torch.where(
                    y_kmh < 10.0,
                    torch.tensor(WEIGHT_JAM, device=device),
                    torch.where(
                        y_kmh < 30.0,
                        torch.tensor(WEIGHT_SLOW, device=device),
                        torch.tensor(WEIGHT_NORMAL, device=device),
                    ),
                )
                loss = (((predictions - batch.y) ** 2) * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs

            epoch_loss = train_loss / len(train_loader.dataset)

            # ---- Évaluation ------------------------------------------------
            model.eval()
            mae_metric = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    predictions = model(batch.x, batch.edge_index)
                    B_curr = batch.num_graphs
                    preds_kmh = (
                        predictions * scale_tensor.repeat(B_curr, 1)
                        + mean_tensor.repeat(B_curr, 1)
                    )
                    targets_kmh = (
                        batch.y * scale_tensor.repeat(B_curr, 1)
                        + mean_tensor.repeat(B_curr, 1)
                    )
                    mae_metric += F.l1_loss(
                        preds_kmh, targets_kmh, reduction="sum"
                    ).item()

            test_mae_kmh = mae_metric / (
                len(test_loader.dataset) * num_nodes * len(HORIZONS)
            )

            logger.info(
                f"Epoch {epoch:02d}/{EPOCHS} | Train Loss (std): {epoch_loss:.4f} "
                f"| Test MAE (km/h): {test_mae_kmh:.4f}"
            )
            mlflow.log_metric("train_loss_std", epoch_loss, step=epoch)
            mlflow.log_metric("test_mae_kmh", test_mae_kmh, step=epoch)

            if test_mae_kmh < best_test_mae:
                best_test_mae = test_mae_kmh
                patience_counter = 0
                best_epoch = epoch
                best_model_state = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }
                os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
                torch.save(model.state_dict(), MODEL_OUT)
                logger.info(
                    f"🏆 New best model at epoch {epoch:02d} "
                    f"with Test MAE: {best_test_mae:.4f} km/h → {MODEL_OUT}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"🛑 Early stopping at epoch {epoch:02d}. "
                        f"Best Test MAE: {best_test_mae:.4f} km/h at epoch {best_epoch:02d}."
                    )
                    break

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            logger.info(
                f"♻️ Restored best model weights from epoch {best_epoch:02d} "
                f"with Test MAE: {best_test_mae:.4f} km/h."
            )
        else:
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            logger.info(f"💾 Saved last epoch weights as fallback → {MODEL_OUT}")

        # =====================================================================
        # Analyse stratifiée + figures (même logique que train_stgcn.py)
        # =====================================================================
        logger.info("📊 Performing stratified error analysis on test set…")
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                predictions = model(batch.x, batch.edge_index)
                B_curr = batch.num_graphs
                scale_b = scale_tensor.repeat(B_curr, 1)
                mean_b = mean_tensor.repeat(B_curr, 1)
                preds_kmh = predictions * scale_b + mean_b
                targets_kmh = batch.y * scale_b + mean_b
                all_preds.append(preds_kmh.cpu().numpy().flatten())
                all_targets.append(targets_kmh.cpu().numpy().flatten())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
        bin_labels = ["0-10", "10-20", "20-30", "30-40", "40-50",
                      "50-60", "60-70", "70-80", "80-90", "90+"]
        mae_by_bin, bias_by_bin, std_preds_by_bin, std_err_by_bin, count_by_bin, box_data = (
            [], [], [], [], [], []
        )
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            idx = np.where((all_targets >= lo) & (all_targets < hi))[0]
            if len(idx) > 0:
                mae_by_bin.append(float(np.mean(np.abs(all_preds[idx] - all_targets[idx]))))
                bias_by_bin.append(float(np.mean(all_preds[idx] - all_targets[idx])))
                std_preds_by_bin.append(float(np.std(all_preds[idx])))
                std_err_by_bin.append(float(np.std(all_preds[idx] - all_targets[idx])))
                count_by_bin.append(int(len(idx)))
                box_data.append(all_preds[idx])
            else:
                mae_by_bin.append(0.0)
                bias_by_bin.append(0.0)
                std_preds_by_bin.append(0.0)
                std_err_by_bin.append(0.0)
                count_by_bin.append(0)
                box_data.append([])

        df_analysis = pd.DataFrame({
            "Tranche (km/h)": bin_labels,
            "Nombre d'exemples": count_by_bin,
            "MAE (km/h)": mae_by_bin,
            "Biais (km/h)": bias_by_bin,
            "Écart-type Prédiction (km/h)": std_preds_by_bin,
            "Écart-type Erreur (km/h)": std_err_by_bin,
        })
        logger.info("\n📊 Erreurs par tranche de vitesse réelle :")
        logger.info(f"\n{df_analysis.to_string(index=False)}")
        os.makedirs("models", exist_ok=True)
        csv_path = "models/stratified_error_analysis_vps.csv"
        df_analysis.to_csv(csv_path, index=False)
        try:
            mlflow.log_artifact(csv_path, artifact_path="analysis")
        except Exception as e:
            logger.warning(f"MLflow artifact log failed: {e}")
        logger.info(f"💾 Analysis CSV saved: {csv_path}")

        # ----- Figures 2×2 ------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        axs = axs.ravel()
        color = "tab:blue"
        axs[0].set_title("1. MAE et Volume d'exemples", fontsize=12, fontweight="bold")
        axs[0].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[0].set_ylabel("MAE (km/h)", color=color)
        axs[0].bar(bin_labels, mae_by_bin, color=color, alpha=0.6)
        axs[0].tick_params(axis="y", labelcolor=color)
        axs[0].grid(True, linestyle="--", alpha=0.3)
        ax1_sub = axs[0].twinx()
        ax1_sub.set_ylabel("Nombre d'exemples", color="tab:grey")
        ax1_sub.plot(bin_labels, count_by_bin, color="tab:grey", marker="o", linestyle=":")
        ax1_sub.tick_params(axis="y", labelcolor="tab:grey")

        axs[1].axhline(0, color="black", linestyle="--", alpha=0.7)
        axs[1].plot(bin_labels, bias_by_bin, marker="s", color="red", linewidth=2.5)
        b_arr = np.array(bias_by_bin)
        axs[1].fill_between(bin_labels, b_arr, 0, where=b_arr > 0, color="orange", alpha=0.3, interpolate=True)
        axs[1].fill_between(bin_labels, b_arr, 0, where=b_arr < 0, color="blue", alpha=0.15, interpolate=True)
        axs[1].set_title("2. Biais de Prédiction Systématique", fontsize=12, fontweight="bold")
        axs[1].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[1].set_ylabel("Biais (km/h)")
        axs[1].set_ylim(-40, 40)
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].legend()

        axs[2].plot(bin_labels, std_preds_by_bin, marker="o", color="purple", linewidth=2.5, label="Std Prédictions")
        axs[2].plot(bin_labels, std_err_by_bin, marker="^", color="teal", linewidth=2.5, linestyle="--", label="Std Erreurs")
        axs[2].set_title("3. Écarts-types par Tranche", fontsize=12, fontweight="bold")
        axs[2].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[2].set_ylabel("Écart-type (km/h)")
        axs[2].grid(True, linestyle="--", alpha=0.5)
        axs[2].legend()

        axs[3].boxplot(box_data, tick_labels=bin_labels, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="#E6E6FA", color="#5D3FD3", alpha=0.7),
                       medianprops=dict(color="red", linewidth=2))
        axs[3].set_title("4. Boxplots des Vitesses Prédites", fontsize=12, fontweight="bold")
        axs[3].set_xlabel("Tranche de Vitesse Réelle (km/h)")
        axs[3].set_ylabel("Vitesse Prédite (km/h)")
        axs[3].grid(True, linestyle="--", alpha=0.5)
        axs[3].legend()

        plt.tight_layout()
        local_plot = "models/stratified_error_analysis_vps.png"
        fig.savefig(local_plot, dpi=300)
        try:
            mlflow.log_figure(fig, "plots/stratified_error_analysis_vps.png")
        except Exception as e:
            logger.warning(f"MLflow figure log failed: {e}")
        plt.close(fig)
        logger.info(f"🎨 Plot saved: {local_plot}")

        try:
            mlflow.log_artifact(MODEL_OUT, artifact_path="model_checkpoints")
        except Exception as e:
            logger.warning(f"MLflow checkpoint log failed: {e}")
        logger.info(f"🏆 VPS production training completed successfully! Model → {MODEL_OUT}")


if __name__ == "__main__":
    train_model()
