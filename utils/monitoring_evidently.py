"""
monitoring_evidently.py

Ce script réalise un suivi continu (continuous monitoring) de la précision du modèle STGCN
sur le créneau critique de la pointe du matin (07h00 à 10h00).
Il compare les performances d'inférence de la veille (J-1, Reference) avec celles du jour courant (J, Current)
à l'aide d'Evidently AI.

La jointure et l'arrondi des timestamps à la minute sont réalisés côté Pandas pour des performances optimales,
évitant d'appliquer des fonctions de date non indexées sur PostgreSQL.

"""
from datetime import datetime
import json
import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text


# Configuration du Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LyonFlow-Evidently-Monitoring")

# Configuration de la base de données
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

if not DB_PASSWORD:
    logger.error(
        "❌ La variable d'environnement POSTGRES_PASSWORD est manquante ou vide. Impossible de démarrer le monitoring."
    )
    sys.exit(1)

# Dossier d'export du rapport
DATA_FOLDER_OUT = os.getenv("DATA_FOLDER_OUT", "/home/ray/project/data/out")


def generate_report():
    logger.info("====================================================================")
    logger.info("🚀 Démarrage du pipeline de Monitoring Continu (Evidently AI)...")
    logger.info("====================================================================")

    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    # 1. Récupération des jours distincts disponibles ayant des données matinales (7h-9h)
    logger.info("📡 Analyse des jours disponibles dans la base de données...")
    query_preds_days = """
        SELECT DISTINCT target_timestamp::date AS day 
        FROM gold.fact_predictions_traffic 
        WHERE EXTRACT(HOUR FROM target_timestamp) BETWEEN 7 AND 9;
    """
    query_facts_days = """
        SELECT DISTINCT timestamp::date AS day 
        FROM gold.fact_traffic_series 
        WHERE EXTRACT(HOUR FROM timestamp) BETWEEN 7 AND 9;
    """
    try:
        df_preds_days = pd.read_sql(query_preds_days, con=engine)
        df_facts_days = pd.read_sql(query_facts_days, con=engine)
    except Exception as e:
        logger.error(f"❌ Impossible d'accéder aux tables de données : {e}")
        sys.exit(1)

    days_preds = set(df_preds_days["day"].tolist())
    days_facts = set(df_facts_days["day"].tolist())
    available_days = sorted(list(days_preds.intersection(days_facts)))

    if len(available_days) < 2:
        logger.warning(
            "⚠️ Nombre de jours avec données du matin insuffisant (minimum 2 requis) pour effectuer une comparaison. Monitoring annulé."
        )
        return

    day_j = available_days[-1]  # Jour le plus récent avec données du matin (Current)
    day_j_minus_1 = available_days[-2]  # Jour précédent avec données du matin (Reference)

    logger.info(f"📊 Jour de Référence (Reference, J-1) : {day_j_minus_1}")
    logger.info(f"📊 Jour d'Audit (Current, J)          : {day_j}")

    # 2. Extraction des données brutes de prédiction et de trafic réel
    # On filtre sur les jours J-1 et J pour minimiser le transfert de données
    query_preds = """
        SELECT prediction_timestamp, target_timestamp, horizon_minutes, node_idx, properties_twgid, predicted_speed
        FROM gold.fact_predictions_traffic
        WHERE target_timestamp::date IN (:day_ref, :day_curr)
          AND EXTRACT(HOUR FROM target_timestamp) BETWEEN 7 AND 9;
    """

    query_facts = """
        SELECT timestamp, node_idx, properties_vitesse AS actual_speed
        FROM gold.fact_traffic_series
        WHERE timestamp::date IN (:day_ref, :day_curr)
          AND EXTRACT(HOUR FROM timestamp) BETWEEN 7 AND 9;
    """

    logger.info("📡 Chargement des prédictions depuis PostgreSQL...")
    df_preds_raw = pd.read_sql(text(query_preds), con=engine, params={"day_ref": day_j_minus_1, "day_curr": day_j})

    logger.info("📡 Chargement du trafic réel observée depuis PostgreSQL...")
    df_facts_raw = pd.read_sql(text(query_facts), con=engine, params={"day_ref": day_j_minus_1, "day_curr": day_j})

    logger.info(f"✅ Prédictions chargées : {len(df_preds_raw)} lignes.")
    logger.info(f"✅ Trafic réel chargé   : {len(df_facts_raw)} lignes.")

    if df_preds_raw.empty or df_facts_raw.empty:
        logger.warning(
            "⚠️ L'une des tables extraites est vide. Vérifier que l'ingestion et les prédictions ont bien fonctionné."
        )
        return

    # 3. Alignement des formats temporels et arrondi côté Pandas
    logger.info("⚙️ Alignement et arrondi temporel à la minute sous Pandas...")

    # Suppression de la timezone si présente pour obtenir des naïfs datetimes
    df_preds_raw["target_dt_min"] = (
        pd.to_datetime(df_preds_raw["target_timestamp"]).dt.tz_localize(None).dt.floor("min")
    )

    # Recherche du shift optimal de timezone (entre -2h, -1h, 0h, 1h, 2h) pour maximiser la jointure
    best_shift = 0
    max_joined_rows = -1
    best_df_merged = None

    logger.info("⚙️ Recherche dynamique du décalage de timezone optimal pour la jointure...")
    for shift_hours in [0, 1, 2, -1, -2]:
        df_facts_temp = df_facts_raw.copy()
        df_facts_temp["fact_dt_min"] = (
            pd.to_datetime(df_facts_temp["timestamp"]).dt.tz_localize(None) + pd.Timedelta(hours=shift_hours)
        ).dt.floor("min")

        df_merged_temp = pd.merge(
            df_preds_raw,
            df_facts_temp,
            left_on=["target_dt_min", "node_idx"],
            right_on=["fact_dt_min", "node_idx"],
            how="inner",
        )

        num_rows = len(df_merged_temp)
        logger.info(f"   - Essai shift {shift_hours:+}h : {num_rows} lignes jointes.")
        if num_rows > max_joined_rows:
            max_joined_rows = num_rows
            best_shift = shift_hours
            best_df_merged = df_merged_temp

    if max_joined_rows <= 0 or best_df_merged is None:
        logger.warning(
            "⚠️ Aucun alignement temporel trouvé après examen de tous les décalages de timezone. Vérifier l'adéquation des horodatages."
        )
        return

    logger.info(
        f"🏆 Alignement temporel optimal sélectionné : shift de {best_shift:+}h ({max_joined_rows} lignes alignées)."
    )
    df_merged = best_df_merged

    # Découpage final en ensembles Reference et Current
    df_ref_full = df_merged[df_merged["target_dt_min"].dt.date == day_j_minus_1]
    df_curr_full = df_merged[df_merged["target_dt_min"].dt.date == day_j]

    # Sélection et renommage des colonnes requises par Evidently (prediction, target)
    df_ref_evidently = df_ref_full[["predicted_speed", "actual_speed"]].rename(
        columns={"predicted_speed": "prediction", "actual_speed": "target"}
    )
    df_curr_evidently = df_curr_full[["predicted_speed", "actual_speed"]].rename(
        columns={"predicted_speed": "prediction", "actual_speed": "target"}
    )

    logger.info(f"📊 Population Référence (J-1) : {len(df_ref_evidently)} lignes.")
    logger.info(f"📊 Population Audit     (J)   : {len(df_curr_evidently)} lignes.")

    if df_ref_evidently.empty or df_curr_evidently.empty:
        logger.warning("⚠️ L'un des deux jeux d'évaluation Evidently est vide.")
        return

    # 4. Import d'Evidently
    from evidently import DataDefinition, Dataset, Regression, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import RegressionPreset

    # 5. Exécution du diagnostic d'observabilité
    logger.info("🤖 Préparation des objets Dataset avec DataDefinition (Evidently v0.7+)...")
    data_def = DataDefinition(regression=[Regression(target="target", prediction="prediction")])
    ref_dataset = Dataset.from_pandas(df_ref_evidently, data_definition=data_def)
    curr_dataset = Dataset.from_pandas(df_curr_evidently, data_definition=data_def)

    logger.info("🤖 Calcul des métriques de régression et dérives...")
    report = Report(metrics=[RegressionPreset(), ValueDrift(column="target", method="ks", threshold=0.05)])
    result = report.run(reference_data=ref_dataset, current_data=curr_dataset)

    # 6. Sauvegarde des rapports
    os.makedirs(DATA_FOLDER_OUT, exist_ok=True)

    html_out_path = os.path.join(DATA_FOLDER_OUT, "monitoring_report_morning.html")
    result.save_html(html_out_path)
    logger.info(f"💾 Rapport interactif HTML exporté avec succès sous : {html_out_path}")

    # Post-traitement pour formater les titres selon les souhaits de l'utilisateur
    try:
        if os.path.exists(html_out_path):
            logger.info("Application du post-traitement sur les titres du rapport...")
            with open(html_out_path, encoding="utf-8") as f:
                html_content = f.read()

            # 1. Remplacements exacts de textes spécifiques
            replacements = {
                "Regression Model Performance. Target: 'target'": "Regression model performance. Target : 'target'",
                "Regression Model Performance. Target: 'target\\u2019": "Regression model performance. Target : 'target\\u2019",
                "Target: 'target'": "Target : 'target'",
                "Target: 'target\\u2019": "Target : 'target\\u2019",
                "Predicted vs Actual in Time": "Predicted vs actual in time",
                "Error (Predicted - Actual)": "Error (predicted - actual)",
                "Error Distribution": "Error distribution",
                "Mean Absolute Percentage Error: Current": "Mean absolute percentage error : current",
                "Mean Absolute Percentage Error: Reference": "Mean absolute percentage error : reference",
                "Absolute Percentage Error": "Absolute percentage error",
                "Absolute Max Error (current)": "Absolute max error (current)",
                "Absolute Max Error (reference)": "Absolute max error (reference)",
                "R2 Score (current)": "R2 score (current)",
                "R2 Score (reference)": "R2 score (reference)"
            }

            for old, new in replacements.items():
                html_content = html_content.replace(old, new)

            # 2. Remplacements par expression régulière pour les titres "Current: ..." et "Reference: ..." restants
            import re

            def format_title(match):
                prefix = match.group(1)      # "Current" ou "Reference"
                title_text = match.group(2)  # Le texte du titre, ex: "Model Quality (+/- std)"
                return f"{prefix} : {title_text.lower()}"

            # Regex capturant "Current:" ou "Reference:" suivi de lettres, espaces, parenthèses ou symboles de calcul
            pattern = r'\b(Current|Reference):\s*([a-zA-Z0-9\s()/\-+]+)'
            modified_content = re.sub(pattern, format_title, html_content)

            with open(html_out_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logger.info("🟢 Titres reformatés avec succès (ex: 'Current : model quality (+/- std)').")
    except Exception as e:
        logger.warning(f"⚠️ Échec du post-traitement des titres : {e}")

    # Export des métriques brutes au format JSON pour MLflow ou Streamlit
    json_out_path = os.path.join(DATA_FOLDER_OUT, "monitoring_metrics_morning.json")
    try:
        metrics_dict = result.dict()
        with open(json_out_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"💾 Métriques détaillées JSON exportées sous : {json_out_path}")
    except Exception as e:
        logger.warning(f"⚠️ Impossible d'exporter les métriques JSON : {e}")

    logger.info("====================================================================")
    logger.info("🎉 Fin du pipeline de Monitoring Continu avec succès !")
    logger.info("====================================================================")


if __name__ == "__main__":
    generate_report()
