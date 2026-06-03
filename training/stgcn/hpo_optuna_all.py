# -*- coding: utf-8 -*-
"""
hpo_optuna_all.py
=================
Recherche d'hyperparamètres (HPO) avec Optuna et comparaison de modèles sur GPU.

Ce script :
  1. Fixe seq_len = 200 pour les deux modèles (V1 et V2).
  2. Lance une étude Optuna pour le modèle STGCN V1 (Optimiseur Adam classique).
  3. Lance une étude Optuna pour le modèle STGCN V2 (Optimiseur AdamW moderne avec régularisation).
  4. Compare les performances (meilleure MAE) des deux modèles.
  5. Enregistre le grand gagnant ("Champion") dans MLflow et sauvegarde ses poids ainsi que son scaler.

Prend en charge USE_LOCAL_CSV (par défaut True) pour un entraînement ultra-rapide.
"""

from __future__ import annotations

import os
import logging
import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy import create_engine
import mlflow
import optuna
import numpy as np
import pandas as pd

# Import local modules
from model import SpatioTemporalGCN
from dataset import load_graph_topology, load_traffic_series, build_sliding_dataset

# Logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LyonFlow-STGCN-HPO-Comparison")

# Environment & DB Configs
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Cache data loading at module-level to avoid IO overhead on each Optuna trial
topology_data = None
traffic_data = None

# Global constant sequence length
SEQ_LEN = 200

# Number of trials for the demo run (can be adjusted via env)
N_TRIALS = int(os.getenv("N_TRIALS", "3"))


def get_db_url():
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_all_data():
    """Charge les données une seule fois (CSV ou Postgres) et les met en cache."""
    global topology_data, traffic_data
    USE_LOCAL_CSV = os.getenv("USE_LOCAL_CSV", "true").lower() == "true"
    DATA_FOLDER = os.getenv("DATA_FOLDER", "/home/ray/project/data/in")

    if USE_LOCAL_CSV:
        logger.info(f"📁 [CACHE] Chargement local depuis les fichiers CSV dans {DATA_FOLDER}...")
        from dataset import load_graph_topology_from_csv, load_traffic_series_from_csv
        num_nodes, edge_index = load_graph_topology_from_csv(DATA_FOLDER)
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series_from_csv(DATA_FOLDER)
    else:
        logger.info("📡 [CACHE] Connexion PostgreSQL pour chargement des données...")
        engine = create_engine(get_db_url(), connect_args={"connect_timeout": 15})
        num_nodes, edge_index = load_graph_topology(engine)
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series(engine)
        engine.dispose()

    topology_data = (num_nodes, edge_index)
    traffic_data = (vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos)
    logger.info(f"✅ Cache de données initialisé : {num_nodes} nœuds, {edge_index.shape[1]} arêtes.")


def objective_v1(trial):
    """Objectif Optuna pour STGCN V1 (Adam classique)"""
    global topology_data, traffic_data

    # Espace de recherche d'hyperparamètres
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])  # Limité pour seq_len=200 et VRAM
    weight_jam = trial.suggest_float("weight_jam", 5.0, 20.0)
    weight_slow = trial.suggest_float("weight_slow", 2.0, 8.0)

    # Récupération des données en cache
    num_nodes, edge_index = topology_data
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = traffic_data

    # Construction du dataset
    train_loader, test_loader, scaler = build_sliding_dataset(
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos,
        seq_len=SEQ_LEN, edge_index_tensor=edge_index, num_nodes=num_nodes,
        test_split=0.2, batch_size=batch_size, horizons=[1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(
        in_channels=5, hidden_channels=hidden_channels, out_channels=1
    ).to(device)

    # Optimiseur Adam classique pour la V1
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

    max_epochs = 5  # Rapide pour HPO
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch.x, batch.edge_index)

            y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)
            weights = torch.where(
                y_kmh < 10.0,
                torch.tensor(weight_jam, device=device),
                torch.where(y_kmh < 30.0, torch.tensor(weight_slow, device=device), torch.tensor(1.0, device=device)),
            )

            loss = (((predictions - batch.y) ** 2) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Évaluation
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

        test_mae = mae_metric / (len(test_loader.dataset) * num_nodes)
        trial.report(test_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_mae


def objective_v2(trial):
    """Objectif Optuna pour STGCN V2 (AdamW moderne)"""
    global topology_data, traffic_data

    # Espace de recherche d'hyperparamètres (légèrement décalé pour tester des plages différentes)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)  # Plus fort pour AdamW
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    weight_jam = trial.suggest_float("weight_jam", 5.0, 20.0)
    weight_slow = trial.suggest_float("weight_slow", 2.0, 8.0)

    num_nodes, edge_index = topology_data
    vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = traffic_data

    train_loader, test_loader, scaler = build_sliding_dataset(
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos,
        seq_len=SEQ_LEN, edge_index_tensor=edge_index, num_nodes=num_nodes,
        test_split=0.2, batch_size=batch_size, horizons=[1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(
        in_channels=5, hidden_channels=hidden_channels, out_channels=1
    ).to(device)

    # Optimiseur AdamW avec découplage de la décroissance des poids (V2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

    max_epochs = 5  # Rapide pour HPO
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch.x, batch.edge_index)

            y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)
            weights = torch.where(
                y_kmh < 10.0,
                torch.tensor(weight_jam, device=device),
                torch.where(y_kmh < 30.0, torch.tensor(weight_slow, device=device), torch.tensor(1.0, device=device)),
            )

            loss = (((predictions - batch.y) ** 2) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Évaluation
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

        test_mae = mae_metric / (len(test_loader.dataset) * num_nodes)
        trial.report(test_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_mae


def run_full_hpo_comparison():
    logger.info("🔥 Démarrage de l'optimisation Optuna & Détermination du Meilleur Modèle sur GPU 🔥")
    logger.info(f"Configuration : SEQ_LEN={SEQ_LEN}, N_TRIALS_PER_MODEL={N_TRIALS}")

    # 1. Warm-up cache
    load_all_data()

    # 2. Setup MLflow Tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("LyonFlow-STGCN-Optuna-Comparison")
    except Exception as e:
        logger.warning(f"⚠️ MLflow tracking indisponible sur {MLFLOW_URL} ({e}).")

    # 3. Create/Run Optuna Study for STGCN V1
    logger.info("🎯 Lancement de l'étude Optuna pour STGCN V1 (Adam)...")
    study_v1 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    with mlflow.start_run(run_name="HPO_STGCN_V1") as run_v1:
        study_v1.optimize(objective_v1, n_trials=N_TRIALS)
        best_mae_v1 = study_v1.best_value
        best_params_v1 = study_v1.best_params
        logger.info(f"🏆 Meilleur run STGCN V1 - MAE: {best_mae_v1:.4f} km/h")
        mlflow.log_params(best_params_v1)
        mlflow.log_param("model_type", "STGCN_V1_Adam")
        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_metric("best_mae_kmh", best_mae_v1)

    # 4. Create/Run Optuna Study for STGCN V2
    logger.info("🎯 Lancement de l'étude Optuna pour STGCN V2 (AdamW)...")
    study_v2 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    with mlflow.start_run(run_name="HPO_STGCN_V2") as run_v2:
        study_v2.optimize(objective_v2, n_trials=N_TRIALS)
        best_mae_v2 = study_v2.best_value
        best_params_v2 = study_v2.best_params
        logger.info(f"🏆 Meilleur run STGCN V2 - MAE: {best_mae_v2:.4f} km/h")
        mlflow.log_params(best_params_v2)
        mlflow.log_param("model_type", "STGCN_V2_AdamW")
        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_metric("best_mae_kmh", best_mae_v2)

    # 5. Determine the overall Champion model
    logger.info("⚔️ Comparaison des modèles et détermination du Champion en cours...")
    
    if best_mae_v1 < best_mae_v2:
        champion_name = "STGCN_V1_Adam"
        champion_mae = best_mae_v1
        champion_params = best_params_v1
    else:
        champion_name = "STGCN_V2_AdamW"
        champion_mae = best_mae_v2
        champion_params = best_params_v2

    logger.info(f"👑 CHAMPION : {champion_name} avec une MAE de {champion_mae:.4f} km/h !")

    # 6. Log the Champion Determination run in MLflow
    with mlflow.start_run(run_name="CHAMPION_DETERMINATION") as champion_run:
        mlflow.log_param("champion_model_type", champion_name)
        mlflow.log_param("champion_seq_len", SEQ_LEN)
        mlflow.log_metric("champion_mae_kmh", champion_mae)
        for k, v in champion_params.items():
            mlflow.log_param(f"champion_opt_{k}", v)

        # Entraîner le Champion final avec les hyperparamètres optimaux pour sauvegarder le modèle final
        logger.info(f"💾 Entraînement final du modèle Champion {champion_name} pour enregistrer les poids...")
        
        num_nodes, edge_index = topology_data
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = traffic_data

        train_loader, test_loader, scaler = build_sliding_dataset(
            vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos,
            seq_len=SEQ_LEN, edge_index_tensor=edge_index, num_nodes=num_nodes,
            test_split=0.2, batch_size=champion_params["batch_size"], horizons=[1],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpatioTemporalGCN(
            in_channels=5, hidden_channels=champion_params["hidden_channels"], out_channels=1
        ).to(device)

        if champion_name == "STGCN_V1_Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=champion_params["lr"], weight_decay=champion_params["weight_decay"]
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=champion_params["lr"], weight_decay=champion_params["weight_decay"]
            )

        mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float, device=device).view(-1, 1)
        scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float, device=device).view(-1, 1)

        # Entraînement sur 3 époques complètes pour avoir des poids viables
        weight_jam = champion_params["weight_jam"]
        weight_slow = champion_params["weight_slow"]
        for epoch in range(1, 4):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch.x, batch.edge_index)
                
                y_kmh = batch.y * scale_tensor.repeat(batch.num_graphs, 1) + mean_tensor.repeat(batch.num_graphs, 1)
                weights = torch.where(
                    y_kmh < 10.0,
                    torch.tensor(weight_jam, device=device),
                    torch.where(y_kmh < 30.0, torch.tensor(weight_slow, device=device), torch.tensor(1.0, device=device)),
                )
                
                loss = (((predictions - batch.y) ** 2) * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Sauvegarde locale du modèle champion
        os.makedirs("models", exist_ok=True)
        champion_model_path = "models/stgcn_champion.pt"
        champion_scaler_path = "models/stgcn_champion_scaler.pkl"
        
        torch.save(model.state_dict(), champion_model_path)
        with open(champion_scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        logger.info(f"💾 Champion model and scaler saved locally.")
        
        # Logged to MLflow as ultimate production assets
        try:
            mlflow.log_artifact(champion_model_path, artifact_path="champion_model")
            mlflow.log_artifact(champion_scaler_path, artifact_path="champion_model")
            logger.info("🏆 Champion model and scaler uploaded to MLflow Artifacts.")
        except Exception as e:
            logger.warning(f"MLflow champion artifact upload failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 OPTUNA MODELS COMPARISON AND CHAMPION SELECTION COMPLETED!")
    print(f"🥇 WINNER: {champion_name} (MAE: {champion_mae:.4f} km/h)")
    print(f"Optimal Params: {champion_params}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_full_hpo_comparison()
