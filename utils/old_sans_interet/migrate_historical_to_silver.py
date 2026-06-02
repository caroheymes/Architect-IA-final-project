"""
migrate_historical_to_silver.py

This script crawls the historical transformed GeoJSON datasets in /opt/airflow/data,
re-serializes them for database compatibility (mapping spatial fields to WKT and lists/dicts to JSON),
and appends them to the PostgreSQL silver.trafic_vitesse_propre table.
It parses the true snapshot date/time from each filename to preserve history correctly.
"""

import glob
import json
import logging
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytz
from sqlalchemy import create_engine, text

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Postgres Connection config
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"
DATA_DIR = "/opt/airflow/data"


def migrate_historical():
    """Ré-ingeste les GeoJSON historiques (`*_transformed.json`) vers la table Silver.

    Ce script est utile quand on dispose d'un historique de fichiers GeoJSON
    déjà transformés (snapshot du dossier `$DATA_DIR`, typiquement
    `/opt/airflow/data`) qu'on souhaite rétro-injecter dans la base pour
    peupler `silver.trafic_vitesse_propre`.

    Pipeline pour chaque fichier trouvé :
      1. Lecture via GeoPandas (`read_file`).
      2. Sérialisation des objets Shapely en WKT et des listes/dicts en JSON
         via la fonction interne `to_json_str`.
      3. Extraction du timestamp de capture depuis le nom de fichier
         (format attendu : `YYYY_MM_DD_HH_MM_transformed.json`), avec
         fallback à `datetime.now()` si le parse échoue.
      4. Sélection d'un sous-ensemble de colonnes aligné sur le schéma Silver
         puis `to_sql(..., if_exists="append", chunksize=500)`.

    Les erreurs fichier par fichier sont attrapées pour que l'exécution
    continue sur les snapshots restants.
    """
    logger.info("Connecting to PostgreSQL database...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # Ensure silver schema exists
    with engine.begin() as conn:
        logger.info("Ensuring silver schema exists...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))

    # Find all transformed GeoJSON files
    json_pattern = os.path.join(DATA_DIR, "*_transformed.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        logger.warning(f"No transformed JSON files found in {DATA_DIR}")
        return

    logger.info(f"Found {len(json_files)} historical files to process.")

    for i, file_path in enumerate(json_files, 1):
        filename = os.path.basename(file_path)
        logger.info(f"[{i}/{len(json_files)}] Processing file: {filename}")

        try:
            # Load GeoJSON into GeoDataFrame
            gdf_wgs84_copy = gpd.read_file(file_path)

            if gdf_wgs84_copy.empty:
                logger.warning(f"File {filename} is empty, skipping.")
                continue

            # Prepare DataFrame for SQL
            df_silver = pd.DataFrame(gdf_wgs84_copy)

            # Ensure id_rue exists
            df_silver["id_rue"] = df_silver.index

            # Convert Shapely geometry objects and lists/dicts to database-friendly formats (WKT/JSON)
            if "geometry" in df_silver.columns:
                df_silver["geometry_wgs84_wkt"] = df_silver["geometry"].apply(lambda geom: geom.wkt if geom else None)
            elif "geometry_coordinates_obj" in df_silver.columns:
                df_silver["geometry_wgs84_wkt"] = df_silver["geometry_coordinates_obj"].apply(
                    lambda geom: geom.wkt if geom else None
                )
            else:
                df_silver["geometry_wgs84_wkt"] = None

            # Parse or convert list/dict fields to valid JSON strings
            def to_json_str(val):
                """Convertit n'importe quelle valeur en chaîne JSON valide.

                Règles :
                  - `None` → `None`
                  - `list` / `dict` → sérialisation `json.dumps`.
                  - `str` : si la chaîne est déjà du JSON valide, on la
                    re-sérialise (pour normaliser les guillemets / escapes),
                    sinon on la conserve telle quelle.
                  - autres types → `str(val)`.
                """
                if val is None:
                    return None
                if isinstance(val, (list, dict)):
                    return json.dumps(val)
                if isinstance(val, str):
                    try:
                        # Test if it's already a valid JSON string
                        parsed = json.loads(val)
                        return json.dumps(parsed)
                    except ValueError:
                        return val
                return str(val)

            df_silver["points_json"] = df_silver["points"].apply(to_json_str) if "points" in df_silver.columns else None
            df_silver["hexes_json"] = df_silver["hexes"].apply(to_json_str) if "hexes" in df_silver.columns else None
            df_silver["merged_h3_geometry_json"] = (
                df_silver["merged_h3_geometry"].apply(to_json_str)
                if "merged_h3_geometry" in df_silver.columns
                else None
            )

            # Extract timestamp from filename (format: YYYY_MM_DD_HH_MM_transformed.json)
            try:
                ts_str = filename.replace("_transformed.json", "")
                dt = datetime.strptime(ts_str, "%Y_%m_%d_%H_%M")
                paris_tz = pytz.timezone("Europe/Paris")
                transformed_at = paris_tz.localize(dt)
            except Exception as ts_err:
                logger.warning(f"Could not parse timestamp from filename {filename}: {ts_err}. Using current time.")
                transformed_at = datetime.now(pytz.timezone("Europe/Paris"))

            df_silver["transformed_at"] = transformed_at

            # Select and clean columns to match silver table schema
            columns_to_write = [
                "id_rue",
                "properties_libelle",
                "properties_sens",
                "properties_etat",
                "properties_vitesse",
                "properties_last_update",
                "properties_est_a_jour",
                "speed_category",
                "speed_color_map",
                "geometry_wgs84_wkt",
                "points_json",
                "hexes_json",
                "merged_h3_geometry_json",
                "transformed_at",
            ]

            # Filter columns that actually exist to be perfectly safe
            columns_to_write = [col for col in columns_to_write if col in df_silver.columns]
            df_silver_clean = df_silver[columns_to_write].copy()

            # Execute push
            with engine.begin() as conn:
                df_silver_clean.to_sql(
                    name="trafic_vitesse_propre",
                    con=conn,
                    schema="silver",
                    if_exists="append",
                    index=False,
                    chunksize=500,
                )
            logger.info(
                f"🟢 Successfully migrated {len(df_silver_clean)} records from {filename} to silver.trafic_vitesse_propre"
            )

        except Exception as file_err:
            logger.error(f"🔴 Error processing {filename}: {file_err}")

    logger.info("🟢 Historical migration completed successfully!")
    engine.dispose()


if __name__ == "__main__":
    migrate_historical()
