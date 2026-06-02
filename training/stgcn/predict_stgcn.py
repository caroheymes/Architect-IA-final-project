"""
predict_stgcn.py

Ce script réalise l'inférence en temps réel ou de test pour le modèle Spatio-Temporal GCN (STGCN).
Il charge le modèle entraîné, récupère les 120 derniers pas temporels (10h d'historique),
effectue des prédictions multi-horizons (par exemple 30 min, 1h, 3h), dénormalise les vitesses,
et associe chaque prédiction à sa géométrie de segment routier (LineString).

Les prédictions sont écrites :
1. Dans un fichier CSV de sortie dans le volume partagé (dossier data/out/predictions_traffic.csv).
2. Dans la table PostgreSQL gold.fact_predictions_traffic (si USE_LOCAL_CSV n'est pas activé).

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
# Ajout du chemin parent au PYTHONPATH pour pouvoir importer correctement
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_graph_topology_from_csv, load_traffic_series_from_csv
from model import SpatioTemporalGCN

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LyonFlow-STGCN-Predict")

# Configuration de la base de données
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# Configuration du modèle et des dossiers
USE_LOCAL_CSV = os.getenv("USE_LOCAL_CSV", "false").lower() == "true"
DATA_FOLDER_IN = os.getenv("DATA_FOLDER", "/home/ray/project/data/in")
DATA_FOLDER_OUT = os.getenv("DATA_FOLDER_OUT", "/home/ray/project/data/out")
MODEL_PATH = os.getenv("MODEL_PATH", "models/stgcn_prod_latest.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/stgcn_scaler.pkl")

# Paramètres globaux de prédiction
SEQ_LEN = int(os.getenv("SEQ_LEN", "120"))
HIDDEN_CHANNELS = int(os.getenv("HIDDEN_CHANNELS", "128"))
HORIZONS_STR = os.getenv("HORIZONS", "6,12,36")  # 30 min (6 pas), 1h (12 pas), 3h (36 pas)
HORIZONS = [int(h) for h in HORIZONS_STR.split(",") if h.strip()]


def init_database_table(engine):
    """Crée la table `gold.fact_predictions_traffic` et ses index si nécessaire.

    Schéma de la table :
      - `prediction_timestamp` : `t_0` (date d'origine de la prédiction).
      - `target_timestamp`     : `t_0 + horizon` (date cible).
      - `horizon_minutes`      : horizon en minutes (5 × nb de pas).
      - `node_idx`, `properties_twgid`, `geometry_wgs84_wkt` : identifiants nœud.
      - `predicted_speed`, `real_speed` (optionnel) : km/h.
      - `created_at`           : horodatage serveur.

    Index créés :
      - `(prediction_timestamp, horizon_minutes)` : accélère les requêtes
        « dernière prédiction pour chaque horizon ».
      - `(properties_twgid)` : accélère les jointures avec `silver.ref_segments`.

    Args:
        engine (sqlalchemy.Engine): Engine SQLAlchemy pointant sur la DB Gold.
    """
    with engine.begin() as conn:
        logger.info("🛠️ Initialisation de la table gold.fact_predictions_traffic...")
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

        # Création des index pour accélérer les visualisations géographiques et temporelles
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_pred_timestamp_horizon ON gold.fact_predictions_traffic(prediction_timestamp, horizon_minutes);"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_pred_twgid ON gold.fact_predictions_traffic(properties_twgid);")
        )
        logger.info("✅ Table d'historisation des prédictions prête.")


def run_prediction():
    """Pipeline complet d'inférence STGCN pour un cycle de prédiction.

    Étapes :
      1. Charge les 120 derniers pas de temps (`SEQ_LEN`) + topologie, soit
         depuis les CSV (`USE_LOCAL_CSV=true`) soit depuis PostgreSQL.
      2. Vérifie qu'on a assez d'historique (`sys.exit(1)` sinon).
      3. Charge le `StandardScaler` fitté à l'entraînement (fallback : fit
         temporaire sur l'échantillon courant, sous-optimal).
      4. Construit le tenseur PyG `[N, SEQ_LEN, 5]` puis charge le modèle
         et bascule en `eval()`.
      5. Inference + dénormalisation scaler → km/h + clip `[1, 130]`.
      6. Pour chaque nœud × horizon, construit un enregistrement
         (prediction_timestamp, target_timestamp, predicted_speed, real_speed, geom).
      7. Sauvegarde locale dans `$DATA_FOLDER_OUT/predictions_traffic.csv`.
      8. Si mode DB, crée la table si besoin (`init_database_table`) et
         `to_sql(..., if_exists="append")`.
      9. Affiche un échantillon console des 10 premières lignes.
    """
    logger.info("====================================================================")
    logger.info("🚀 Démarrage du pipeline de prédiction STGCN (Inférence)...")
    logger.info("====================================================================")

    # 1. Chargement des métadonnées spatiales et des séries temporelles (CSV ou Postgres)
    if USE_LOCAL_CSV:
        logger.info(f"📁 Mode hors-ligne : chargement des données depuis {DATA_FOLDER_IN}...")
        num_nodes, edge_index = load_graph_topology_from_csv(DATA_FOLDER_IN)
        vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos = load_traffic_series_from_csv(DATA_FOLDER_IN)

        # Chargement du fichier de mapping pour récupérer les géométries et twgid hors-ligne
        df_nodes = pd.read_csv(os.path.join(DATA_FOLDER_IN, "node_mapping.csv"))
    else:
        logger.info(f"📡 Mode en ligne : connexion à la base de données PostgreSQL sur {DB_HOST}:{DB_PORT}...")
        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)

        # Récupération de la topologie
        query_mapping = """
            SELECT m.node_idx, m.properties_twgid, r.geometry_wgs84_wkt
            FROM gold.dim_spatial_grid_mapping m
            LEFT JOIN silver.ref_segments r ON m.properties_twgid = CAST(r.properties_twgid AS VARCHAR)
            ORDER BY m.node_idx ASC;
        """
        df_nodes = pd.read_sql(query_mapping, con=engine)
        num_nodes = len(df_nodes)

        # Récupération des connexions
        df_edges = pd.read_sql("SELECT node_u, node_v FROM gold.dim_gnn_adjacency;", con=engine)
        edge_index_list = []
        for _, row in df_edges.iterrows():
            u, v = int(row["node_u"]), int(row["node_v"])
            edge_index_list.append([u, v])
            edge_index_list.append([v, u])  # Non-orienté
        for i in range(num_nodes):
            edge_index_list.append([i, i])  # Boucles sur soi-même
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # Récupération des faits (120 derniers pas temporels en base pour l'inférence)
        query_traffic = f"""
            SELECT timestamp, node_idx, properties_vitesse
            FROM gold.fact_traffic_series
            WHERE timestamp IN (
                SELECT DISTINCT timestamp
                FROM gold.fact_traffic_series
                ORDER BY timestamp DESC
                LIMIT {SEQ_LEN}
            )
            ORDER BY timestamp ASC, node_idx ASC;
        """
        df_traffic_raw = pd.read_sql(query_traffic, con=engine)
        df_pivot = df_traffic_raw.pivot(index="timestamp", columns="node_idx", values="properties_vitesse")

        default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
        vitesse_matrix_raw = np.nan_to_num(df_pivot.values, nan=default_speed)

        df_pivot.index = pd.to_datetime(df_pivot.index)
        hours = df_pivot.index.hour + df_pivot.index.minute / 60.0
        days = df_pivot.index.dayofweek

        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        day_sin = np.sin(2 * np.pi * days / 7.0)
        day_cos = np.cos(2 * np.pi * days / 7.0)

    logger.info(f"✅ Données chargées : {num_nodes} nœuds.")

    # 2. Vérification de la taille des données d'entrée
    num_timestamps = len(vitesse_matrix_raw)
    if num_timestamps < SEQ_LEN:
        logger.error(f"❌ Historique insuffisant : requis={SEQ_LEN}, disponible={num_timestamps}")
        sys.exit(1)

    # Extraction des 120 derniers pas de temps exacts pour l'inférence
    latest_vitesse = vitesse_matrix_raw[-SEQ_LEN:, :]
    latest_h_sin = hour_sin[-SEQ_LEN:]
    latest_h_cos = hour_cos[-SEQ_LEN:]
    latest_d_sin = day_sin[-SEQ_LEN:]
    latest_d_cos = day_cos[-SEQ_LEN:]

    # Détermination du timestamp d'origine de la prédiction (le dernier pas observé)
    if USE_LOCAL_CSV:
        # Lecture des timestamps réels depuis le fichier CSV d'entrée
        traffic_path = os.path.join(DATA_FOLDER_IN, "traffic_series.csv")
        df_ts = pd.read_csv(traffic_path)
        df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
        prediction_timestamp = df_ts["timestamp"].max()
    else:
        prediction_timestamp = df_pivot.index.max()

    logger.info(f"📅 Date d'origine de la prédiction (t_0) : {prediction_timestamp}")

    # 3. Chargement et application du StandardScaler
    scaler = StandardScaler()
    if os.path.exists(SCALER_PATH):
        logger.info(f"💾 Chargement du scaler entraîné depuis {SCALER_PATH}...")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    else:
        logger.warning(
            f"⚠️ Scaler introuvable à l'emplacement : {SCALER_PATH}. Ajustement temporaire sur l'échantillon courant."
        )
        scaler.fit(latest_vitesse)

    # Normalisation des vitesses d'entrée
    vitesse_norm = scaler.transform(latest_vitesse)

    # 4. Construction de la structure de tenseurs PyG pour le modèle
    speeds_feat = vitesse_norm.T  # shape: [N, SEQ_LEN]
    h_sin_exp = np.tile(latest_h_sin, (num_nodes, 1))
    h_cos_exp = np.tile(latest_h_cos, (num_nodes, 1))
    d_sin_exp = np.tile(latest_d_sin, (num_nodes, 1))
    d_cos_exp = np.tile(latest_d_cos, (num_nodes, 1))

    X = np.stack([speeds_feat, h_sin_exp, h_cos_exp, d_sin_exp, d_cos_exp], axis=-1)
    x_tensor = torch.tensor(X, dtype=torch.float)

    # 5. Instanciation du modèle et chargement des poids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalGCN(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=len(HORIZONS)).to(device)

    if os.path.exists(MODEL_PATH):
        logger.info(f"🤖 Chargement des poids du modèle STGCN depuis {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        logger.error(
            f"❌ Modèle introuvable à l'emplacement : {MODEL_PATH}. Un entraînement est nécessaire au préalable."
        )
        sys.exit(1)

    # 6. Exécution de l'inférence
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        edge_index = edge_index.to(device)
        predictions = model(x_tensor, edge_index)  # Shape: [num_nodes, len(HORIZONS)]
        predictions = predictions.cpu().numpy()

    # Dénormalisation des prédictions pour retrouver l'échelle des km/h
    scale_vec = scaler.scale_.reshape(-1, 1)
    mean_vec = scaler.mean_.reshape(-1, 1)
    preds_kmh = predictions * scale_vec + mean_vec
    # Seuil de sécurité : pas de vitesse négative possible
    preds_kmh = np.clip(preds_kmh, a_min=1.0, a_max=130.0)

    # 7. Restructuration et formatage des prédictions pour l'export
    prediction_records = []

    for node_idx in range(num_nodes):
        node_row = df_nodes.iloc[node_idx]
        twgid = int(node_row["properties_twgid"])
        geom = node_row.get("geometry_wgs84_wkt", None)

        # Récupération de la dernière vitesse réelle observée à t_0 à titre de comparaison
        last_real_speed = float(latest_vitesse[-1, node_idx])

        for h_idx, h in enumerate(HORIZONS):
            horizon_minutes = h * 5
            target_time = prediction_timestamp + datetime.timedelta(minutes=horizon_minutes)
            predicted_speed = float(preds_kmh[node_idx, h_idx])

            prediction_records.append(
                {
                    "prediction_timestamp": prediction_timestamp,
                    "target_timestamp": target_time,
                    "horizon_minutes": horizon_minutes,
                    "node_idx": node_idx,
                    "properties_twgid": twgid,
                    "predicted_speed": round(predicted_speed, 2),
                    "real_speed": round(
                        last_real_speed, 2
                    ),  # vitesse à t_0 utilisée comme base réelle d'évaluation immédiate
                    "geometry_wgs84_wkt": geom,
                }
            )

    df_preds = pd.DataFrame(prediction_records)

    # 8. Sauvegarde locale au format CSV (permet de récupérer le CSV sur la machine hôte)
    os.makedirs(DATA_FOLDER_OUT, exist_ok=True)
    csv_out_path = os.path.join(DATA_FOLDER_OUT, "predictions_traffic.csv")
    df_preds.to_csv(csv_out_path, index=False)
    logger.info(f"💾 Copie CSV locale sauvegardée avec succès ({len(df_preds)} lignes) -> {csv_out_path}")

    # 9. Écriture dans PostgreSQL (uniquement si le mode en ligne est activé)
    if not USE_LOCAL_CSV:
        try:
            init_database_table(engine)
            logger.info("📥 Insertion des prédictions dans PostgreSQL (gold.fact_predictions_traffic)...")
            df_preds.to_sql(name="fact_predictions_traffic", con=engine, schema="gold", if_exists="append", index=False)
            logger.info("✅ Insertion PostgreSQL effectuée avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'insertion dans la base de données : {e}")
            sys.exit(1)

    # 10. Affichage d'un extrait des prédictions pour contrôle visuel immédiat
    logger.info("\n✨ Aperçu des prédictions générées :")
    sample_df = df_preds.head(10)[
        [
            "prediction_timestamp",
            "target_timestamp",
            "horizon_minutes",
            "properties_twgid",
            "predicted_speed",
            "real_speed",
        ]
    ]
    logger.info(f"\n{sample_df.to_string(index=False)}")

    logger.info("====================================================================")
    logger.info("🎉 Inférence STGCN complétée avec succès.")
    logger.info("====================================================================")


if __name__ == "__main__":
    run_prediction()
