"""
backfill_predictions.py

Ce script réalise un backfill (rattrapage historique) des prédictions STGCN.
Il identifie tous les pas temporels présents dans la table `gold.fact_traffic_series` qui ne
possèdent pas encore de prédictions correspondantes dans `gold.fact_predictions_traffic`,
reconstruit les séquences historiques de 120 pas temporels pour chacun de ces instants,
exécute l'inférence via le modèle STGCN sur GPU/CPU, et insère les prédictions dans la base de données.

Formulation impersonnelle pour la conformité de la documentation de projet.
"""

import datetime
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# Importation des modules locaux du GNN
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_graph_topology_from_csv
from model import SpatioTemporalGCN

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LyonFlow-STGCN-Backfill")

# Configuration de la base de données
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# Configuration du modèle
MODEL_PATH = os.getenv("MODEL_PATH", "models/stgcn_prod_latest.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/stgcn_scaler.pkl")

# Paramètres globaux
SEQ_LEN = int(os.getenv("SEQ_LEN", "120"))
HIDDEN_CHANNELS = int(os.getenv("HIDDEN_CHANNELS", "128"))
HORIZONS_STR = os.getenv("HORIZONS", "6,12,36")
HORIZONS = [int(h) for h in HORIZONS_STR.split(",") if h.strip()]


def init_database_table(engine):
    """Crée `gold.fact_predictions_traffic` et ses index si nécessaire.

    Args:
        engine (sqlalchemy.Engine): Engine SQLAlchemy.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        create_table_query = """
        CREATE TABLE IF NOT EXISTS gold.fact_predictions_traffic (
            id SERIAL PRIMARY KEY,
            prediction_timestamp TIMESTAMP NOT NULL,
            target_timestamp TIMESTAMP NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            node_idx INTEGER NOT NULL,
            properties_twgid INTEGER NOT NULL,
            predicted_speed REAL NOT NULL,
            real_speed REAL,
            geometry_wgs84_wkt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn.execute(text(create_table_query))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_pred_timestamp_horizon ON gold.fact_predictions_traffic(prediction_timestamp, horizon_minutes);"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_pred_twgid ON gold.fact_predictions_traffic(properties_twgid);")
        )


def run_backfill():
    """Rattrapage historique des prédictions STGCN sur tous les pas de temps manquants.

    Identifie les instants de `gold.fact_traffic_series` qui n'ont pas encore
    de prédiction dans `gold.fact_predictions_traffic`, reconstruit pour
    chacun la fenêtre glissante de `SEQ_LEN` pas précédents, exécute
    l'inférence et insère tous les enregistrements en base.

    Étapes principales :
      1. Crée la table cible (`init_database_table`).
      2. Charge la topologie (mapping + adjacency + self-loops).
      3. Charge **tout** l'historique de `gold.fact_traffic_series`,
         pivote en matrice `[T × N]`, remplace les NaN par `LYON_DEFAULT_SPEED`.
      4. Liste les `prediction_timestamp` déjà calculés et identifie ceux
         qui sont éligibles (≥ SEQ_LEN pas d'historique disponibles).
      5. Charge le scaler et le modèle (le modèle reste en `eval()` sur
         GPU/CPU).
      6. Pour chaque instant éligible, extrait la fenêtre, normalise,
         construit le tenseur PyG, infère, dénormalise et clip.
      7. Insère **en bloc** (`chunksize=10000`) tous les enregistrements
         `nœud × horizon` dans `gold.fact_predictions_traffic`.
    """
    logger.info("====================================================================")
    logger.info("🚀 Démarrage du pipeline de Backfill d'Inférence STGCN...")
    logger.info("====================================================================")

    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    # 1. Initialisation de la table cible
    init_database_table(engine)

    # 2. Récupération de la topologie
    logger.info("📡 Chargement de la topologie du réseau routier depuis la base de données...")
    query_mapping = """
        SELECT m.node_idx, m.properties_twgid, r.geometry_wgs84_wkt
        FROM gold.dim_spatial_grid_mapping m
        LEFT JOIN silver.ref_segments r ON m.properties_twgid = CAST(r.properties_twgid AS VARCHAR)
        ORDER BY m.node_idx ASC;
    """
    df_nodes = pd.read_sql(query_mapping, con=engine)
    num_nodes = len(df_nodes)
    logger.info(f"✅ Topologie chargée : {num_nodes} nœuds.")

    # Récupération de l'adjacence GNN
    df_edges = pd.read_sql("SELECT node_u, node_v FROM gold.dim_gnn_adjacency;", con=engine)
    edge_index_list = []
    for _, row in df_edges.iterrows():
        u, v = int(row["node_u"]), int(row["node_v"])
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])
    for i in range(num_nodes):
        edge_index_list.append([i, i])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # 3. Chargement de l'ensemble de l'historique des faits de trafic
    logger.info("📡 Extraction de l'historique complet des séries de trafic de la couche Gold...")
    query_traffic = """
        SELECT timestamp, node_idx, properties_vitesse
        FROM gold.fact_traffic_series
        ORDER BY timestamp ASC, node_idx ASC;
    """
    df_traffic_raw = pd.read_sql(query_traffic, con=engine)

    if df_traffic_raw.empty:
        logger.warning("⚠️ Table gold.fact_traffic_series vide. Impossible de procéder au backfill.")
        return

    # Pivotage des données pour obtenir un index temporel propre
    df_pivot = df_traffic_raw.pivot(index="timestamp", columns="node_idx", values="properties_vitesse")
    df_pivot.index = pd.to_datetime(df_pivot.index)

    # Remplacement des valeurs manquantes par la vitesse par défaut
    default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
    vitesse_matrix_raw = np.nan_to_num(df_pivot.values, nan=default_speed)

    all_timestamps = df_pivot.index.tolist()
    num_timestamps = len(all_timestamps)
    logger.info(f"✅ {num_timestamps} instants observés chargés de l'historique de trafic.")

    # 4. Identification des prédictions déjà existantes pour éviter les doublons
    query_existing = "SELECT DISTINCT prediction_timestamp FROM gold.fact_predictions_traffic;"
    df_existing = pd.read_sql(query_existing, con=engine)
    existing_timestamps = set(pd.to_datetime(df_existing["prediction_timestamp"]))
    logger.info(f"✅ {len(existing_timestamps)} instants possèdent déjà des prédictions enregistrées.")

    # Déterminer les instants éligibles pour l'inférence (nécessite au moins SEQ_LEN pas temporels d'historique)
    eligible_missing_timestamps = []
    for i in range(SEQ_LEN - 1, num_timestamps):
        ts = all_timestamps[i]
        if ts not in existing_timestamps:
            eligible_missing_timestamps.append((i, ts))

    total_to_predict = len(eligible_missing_timestamps)
    logger.info(f"🎯 Nombre d'instants éligibles à calculer (backfill) : {total_to_predict}")

    if total_to_predict == 0:
        logger.info("🟢 Aucune prédiction manquante à combler. Fin du script de backfill.")
        return

    # 5. Chargement du scaler et du modèle
    scaler = StandardScaler()
    if os.path.exists(SCALER_PATH):
        logger.info(f"💾 Chargement du scaler depuis {SCALER_PATH}...")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
            
        # Ajustement dynamique du scaler si le nombre de nœuds a changé
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != num_nodes:
            logger.warning(
                f"⚠️ Discordance du Scaler : entraîné sur {scaler.n_features_in_} nœuds, "
                f"mais {num_nodes} nœuds détectés actuellement. Ajustement dynamique en cours..."
            )
            old_n = scaler.n_features_in_
            if num_nodes > old_n:
                extra_count = num_nodes - old_n
                avg_mean = np.mean(scaler.mean_)
                avg_var = np.mean(scaler.var_)
                avg_scale = np.mean(scaler.scale_)
                
                scaler.mean_ = np.concatenate([scaler.mean_, np.full(extra_count, avg_mean)])
                scaler.var_ = np.concatenate([scaler.var_, np.full(extra_count, avg_var)])
                scaler.scale_ = np.concatenate([scaler.scale_, np.full(extra_count, avg_scale)])
                scaler.n_features_in_ = num_nodes
            else:
                scaler.mean_ = scaler.mean_[:num_nodes]
                scaler.var_ = scaler.var_[:num_nodes]
                scaler.scale_ = scaler.scale_[:num_nodes]
                scaler.n_features_in_ = num_nodes
    else:
        logger.error(f"❌ Scaler introuvable à {SCALER_PATH}. Un entraînement préalable est requis.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(MODEL_PATH):
        logger.info(f"🤖 Chargement du modèle STGCN depuis {MODEL_PATH} sur le périphérique {device}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Détection dynamique de hidden_channels pour éviter les erreurs de taille
        hidden_size = HIDDEN_CHANNELS
        if "temporal_gru.weight_ih_l0" in state_dict:
            detected_hidden = state_dict["temporal_gru.weight_ih_l0"].shape[0] // 3
            if detected_hidden != hidden_size:
                logger.info(f"🔄 Détection dynamique de HIDDEN_CHANNELS : {detected_hidden} (configuré : {hidden_size})")
                hidden_size = detected_hidden
                
        model = SpatioTemporalGCN(in_channels=5, hidden_channels=hidden_size, out_channels=len(HORIZONS)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        logger.error(f"❌ Modèle introuvable à {MODEL_PATH}.")
        sys.exit(1)

    # Pré-calcul des variables de temps globales
    hours = df_pivot.index.hour + df_pivot.index.minute / 60.0
    days = df_pivot.index.dayofweek

    # 6. Boucle d'inférence séquentielle optimisée
    logger.info("🤖 Exécution des prédictions de rattrapage...")

    all_predictions_records = []
    scale_vec = scaler.scale_.reshape(-1, 1)
    mean_vec = scaler.mean_.reshape(-1, 1)

    # Extraction optimisée des colonnes sous forme de tableaux numpy pour éviter les .iloc lents dans la boucle
    node_twgids = df_nodes["properties_twgid"].astype(int).values
    node_geoms = df_nodes["geometry_wgs84_wkt"].values

    # Utilisation d'un gestionnaire de contexte sans gradients pour accélérer l'inférence
    with torch.no_grad():
        edge_index = edge_index.to(device)

        for idx, (original_idx, ts) in enumerate(eligible_missing_timestamps):
            if (idx + 1) % 50 == 0 or idx + 1 == total_to_predict:
                logger.info(f"   - Progression : {idx + 1}/{total_to_predict} instants traités...")

            # Extraction de la fenêtre glissante temporelle
            latest_vitesse = vitesse_matrix_raw[original_idx - SEQ_LEN + 1 : original_idx + 1, :]
            latest_h_sin = np.sin(2 * np.pi * hours[original_idx - SEQ_LEN + 1 : original_idx + 1] / 24.0)
            latest_h_cos = np.cos(2 * np.pi * hours[original_idx - SEQ_LEN + 1 : original_idx + 1] / 24.0)
            latest_d_sin = np.sin(2 * np.pi * days[original_idx - SEQ_LEN + 1 : original_idx + 1] / 7.0)
            latest_d_cos = np.cos(2 * np.pi * days[original_idx - SEQ_LEN + 1 : original_idx + 1] / 7.0)

            # Normalisation
            vitesse_norm = scaler.transform(latest_vitesse)
            speeds_feat = vitesse_norm.T

            h_sin_exp = np.tile(latest_h_sin, (num_nodes, 1))
            h_cos_exp = np.tile(latest_h_cos, (num_nodes, 1))
            d_sin_exp = np.tile(latest_d_sin, (num_nodes, 1))
            d_cos_exp = np.tile(latest_d_cos, (num_nodes, 1))

            X = np.stack([speeds_feat, h_sin_exp, h_cos_exp, d_sin_exp, d_cos_exp], axis=-1)
            x_tensor = torch.tensor(X, dtype=torch.float).to(device)

            # Inférence GNN
            predictions = model(x_tensor, edge_index).cpu().numpy()

            # Dénormalisation et bornage
            preds_kmh = predictions * scale_vec + mean_vec
            preds_kmh = np.clip(preds_kmh, a_min=1.0, a_max=130.0)

            # Création des enregistrements pour insertion (version optimisée sans .iloc)
            for node_idx in range(num_nodes):
                twgid = int(node_twgids[node_idx])
                geom = node_geoms[node_idx]
                if pd.isna(geom):
                    geom = None
                last_real_speed = float(latest_vitesse[-1, node_idx])

                for h_idx, h in enumerate(HORIZONS):
                    horizon_minutes = h * 5
                    target_time = ts + datetime.timedelta(minutes=horizon_minutes)
                    predicted_speed = float(preds_kmh[node_idx, h_idx])

                    all_predictions_records.append(
                        {
                            "prediction_timestamp": ts,
                            "target_timestamp": target_time,
                            "horizon_minutes": horizon_minutes,
                            "node_idx": node_idx,
                            "properties_twgid": twgid,
                            "predicted_speed": round(predicted_speed, 2),
                            "real_speed": round(last_real_speed, 2),
                            "geometry_wgs84_wkt": geom,
                        }
                    )

            # Insertion intermédiaire périodique pour éviter d'accumuler trop d'enregistrements en RAM (OOM de Ray)
            if (idx + 1) % 15 == 0 and all_predictions_records:
                df_temp = pd.DataFrame(all_predictions_records)
                logger.info(f"📥 [Lots] Insertion intermédiaire de {len(df_temp)} lignes de prédictions (instants traités : {idx + 1}/{total_to_predict})...")
                try:
                    df_temp.to_sql(
                        name="fact_predictions_traffic",
                        con=engine,
                        schema="gold",
                        if_exists="append",
                        index=False,
                        chunksize=10000
                    )
                    all_predictions_records.clear()
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'insertion intermédiaire : {e}")
                    sys.exit(1)

    # 7. Insertion des derniers enregistrements restants
    if all_predictions_records:
        df_preds_all = pd.DataFrame(all_predictions_records)
        logger.info(f"📥 Insertion finale de {len(df_preds_all)} lignes de prédictions dans PostgreSQL (gold.fact_predictions_traffic)...")
        try:
            df_preds_all.to_sql(
                name="fact_predictions_traffic",
                con=engine,
                schema="gold",
                if_exists="append",
                index=False,
                chunksize=10000,
            )
            logger.info("✅ Insertion finale effectuée avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'insertion finale dans la base de données : {e}")
            sys.exit(1)

    logger.info("====================================================================")
    logger.info(f"🎉 Rattrapage complété : {total_to_predict} instants d'historique prédits.")
    logger.info("====================================================================")


if __name__ == "__main__":
    run_backfill()
