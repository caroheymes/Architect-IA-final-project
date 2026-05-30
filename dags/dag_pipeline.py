# -*- coding: utf-8 -*-
"""
dag_pipeline.py

This is the unified, self-contained Airflow DAG that orchestrates the entire
real-time traffic pipeline for LyonFlow. It houses:
1. Ingestion: Downloads raw WFS traffic speed data and inserts it into postgres bronze.
2. Spatial Transformation: Pulls the latest bronze, interpolates street segments every 7m,
   maps to H3 res 13, and writes BOTH file backups to D: and clean SQL records to postgres silver.

Having everything in this single file ensures that 100% of your data engineering pipeline code
is fully visible and interactive directly in the Airflow UI.
"""

import os
import json
import glob
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import h3
import pyproj
import pytz
import requests
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform
from sqlalchemy import create_engine, text

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================================
# LOGGING & CORE CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Grand Lyon WFS API Credentials & URL
API_URL = "https://data.grandlyon.com/geoserver/metropole-de-lyon/ows?SERVICE=WFS&VERSION=2.0.0&request=GetFeature&typename=metropole-de-lyon:pvo_patrimoine_voirie.pvotrafic&outputFormat=application/json&SRSNAME=EPSG:2154&startIndex=0&sortby=gid"
API_LOGIN = os.getenv("API_LOGIN")
API_PASSWORD = os.getenv("API_PASSWORD")

# PostgreSQL database configuration
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"
OUTPUT_DIR = "/opt/airflow/data"

# ============================================================================
# HELPER SPATIAL FUNCTIONS
# ============================================================================
def transform_line_to_point(ligne_2154):
    """Transforms a LineString into a list of points every 7 meters."""
    if not ligne_2154 or ligne_2154.is_empty:
        return []
    
    # Convert from EPSG:2154 (meters) to EPSG:4326 (degrees - WGS84)
    proj_vers_4326 = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform

    # Generate points spaced every 7 meters
    distances = np.arange(0, ligne_2154.length, 7)
    points = [ligne_2154.interpolate(d) for d in distances]

    # Always include the exact end point
    if ligne_2154.length % 7 != 0:
        points.append(ligne_2154.interpolate(ligne_2154.length))

    # Project coordinates to EPSG:4326
    points = [transform(proj_vers_4326, p) for p in points]
    return points

def create_merged_polygon_from_hexes(h3_id_list):
    """Unifies a list of H3 cell IDs into a single Shapely Polygon."""
    if not h3_id_list:
        return None

    unique_hexes = list(set(h3_id_list))

    try:
        # Check for H3 v4 capabilities
        if hasattr(h3, 'cells_to_geo'):
            geojson_dict = h3.cells_to_geo(unique_hexes)
            return shape(geojson_dict)
        elif hasattr(h3, 'cells_to_geojson'):
            geojson_dict = h3.cells_to_geojson(unique_hexes)
            return shape(geojson_dict)
        else:
            # Fallback by generating individual polygons and merging
            polygons = []
            for h in unique_hexes:
                boundary = h3.cell_to_boundary(h)
                polygons.append(Polygon([(lon, lat) for lat, lon in boundary]))
            from shapely.ops import unary_union
            return unary_union(polygons)
    except Exception as e:
        logger.error(f"Error merging H3 hexagons: {e}")
        return None

def get_speed_category(speed):
    """Maps speed metrics to human-readable categories."""
    if pd.isna(speed):
        return "Unknown"
    elif speed <= 20:
        return "Slow (0-20 km/h)"
    elif speed > 50:
        return "Fast (>50 km/h)"
    else:
        return "Medium (20-50 km/h)"


# ============================================================================
# CORE PIPELINE PIPES (EXECUTED AS PYTHON TASKS)
# ============================================================================
def ingest_traffic_data():
    """Fetches real-time traffic data from Grand Lyon API and saves it to postgres bronze."""
    logger.info("Starting real-time traffic data ingestion...")
    try:
        response = requests.get(API_URL, auth=(API_LOGIN, API_PASSWORD), timeout=30)
        response.raise_for_status()
        raw_payload = response.json()
        logger.info("Successfully fetched data from Grand Lyon WFS API.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Grand Lyon API: {e}")
        raise Exception(f"Ingestion failed: {e}")

    # Track fetch timestamp in local Paris time
    timezone = pytz.timezone('Europe/Paris')
    fetched_at = datetime.now(timezone)

    # Push to local PostgreSQL bronze layer
    logger.info("Connecting to PostgreSQL container...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze;"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bronze.trafic_vitesse_brute (
                    id SERIAL PRIMARY KEY,
                    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    raw_data JSONB NOT NULL
                );
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_bronze_fetched_at 
                ON bronze.trafic_vitesse_brute (fetched_at DESC);
            """))

            # Execute parametrized insert
            logger.info("Inserting raw payload to bronze.trafic_vitesse_brute...")
            insert_query = text("""
                INSERT INTO bronze.trafic_vitesse_brute (fetched_at, raw_data)
                VALUES (:fetched_at, :raw_data);
            """)
            conn.execute(insert_query, {
                "fetched_at": fetched_at,
                "raw_data": json.dumps(raw_payload)
            })
            
        logger.info(f"🟢 Ingestion successfully saved to bronze at {fetched_at.strftime('%Y-%m-%d %H:%M:%S %Z')}!")
    finally:
        engine.dispose()


def transform_traffic_data():
    """Transforms raw bronze JSON payload, outputs file backups, and pushes to postgres silver."""
    logger.info("Starting spatial data transformation...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    try:
        # 1. Fetch the latest raw snapshot from bronze
        with engine.begin() as conn:
            logger.info("Fetching the latest raw record from bronze layer...")
            query = text("""
                SELECT raw_data FROM bronze.trafic_vitesse_brute 
                ORDER BY fetched_at DESC LIMIT 1;
            """)
            result = conn.execute(query).fetchone()

        if not result:
            logger.warning("No data found in bronze.trafic_vitesse_brute. Cannot transform.")
            raise Exception("No source data found in bronze schema.")

        raw_payload = result[0]
        features = raw_payload.get('features', [])
        if not features:
            logger.warning("Source payload contains no features.")
            raise Exception("Empty feature list inside bronze record.")

        # 2. DataFrame transformations
        trafic = pd.json_normalize(features)
        cols = [c.replace('.', '_') for c in trafic.columns]
        trafic.columns = cols

        selected_columns = [
            'geometry_coordinates', 'properties_libelle', 'properties_sens', 
            'properties_etat', 'properties_vitesse', 'properties_last_update', 
            'properties_est_a_jour'
        ]

        for col in selected_columns:
            if col not in trafic.columns:
                trafic[col] = None

        trafic = trafic[selected_columns]
        if 'properties_est_a_jour' in trafic.columns:
            trafic = trafic[trafic.properties_est_a_jour != False]

        # Form spatial attributes
        trafic['geometry_coordinates_obj'] = trafic['geometry_coordinates'].apply(LineString)
        gdf = GeoDataFrame(data=trafic, geometry='geometry_coordinates_obj')
        gdf.set_crs(epsg=2154, inplace=True, allow_override=True)

        logger.info("Interpolating segments every 7 meters...")
        gdf['points'] = [transform_line_to_point(elem) for elem in gdf.geometry_coordinates_obj]

        logger.info("Mapping interpolated coordinates to H3 resolution 13 cells...")
        gdf['hexes'] = gdf.points.apply(lambda pts: [h3.latlng_to_cell(p.y, p.x, 13) for p in pts])

        logger.info("Unifying H3 cell clusters into polygons...")
        gdf['merged_h3_geometry'] = gdf['hexes'].apply(create_merged_polygon_from_hexes)

        # Average speed sanitization and fallback computation
        gdf['properties_vitesse'] = gdf['properties_vitesse'].astype(str).str.split(' ').str[0]
        gdf['properties_vitesse'] = pd.to_numeric(gdf['properties_vitesse'], errors="coerce")

        mean_speed_df = gdf.groupby(by="properties_libelle").agg(mean_speed=('properties_vitesse', 'mean')).reset_index()
        gdf = gdf.merge(mean_speed_df, on='properties_libelle', how='left')
        gdf['properties_vitesse'] = [
            elem if not pd.isna(elem) else mean_speed if not pd.isna(mean_speed) else np.nan 
            for elem, mean_speed in zip(gdf.properties_vitesse, gdf.mean_speed)
        ]
        gdf = gdf.drop(columns=['mean_speed'], errors='ignore')

        gdf['speed_category'] = [get_speed_category(speed) for speed in gdf.properties_vitesse]
        gdf['speed_color_map'] = gdf['speed_category'].map({
            "Slow (0-20 km/h)": "red",
            "Medium (20-50 km/h)": "orange",
            "Fast (>50 km/h)": "green",
            "Unknown": "gray"
        })

        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf['id_rue'] = gdf.index

        # Define file outputs
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        now = datetime.now(pytz.timezone('Europe/Paris')).strftime(format="%Y_%m_%d_%H_%M")
        csv_path = f"{OUTPUT_DIR}/{now}_transformed.csv"
        json_path = f"{OUTPUT_DIR}/{now}_transformed.json"

        # 3a. Export to CSV (Stringify spatial objects for clean file writes)
        df_csv = gdf.copy()
        df_csv['geometry_coordinates_obj'] = df_csv['geometry_coordinates_obj'].apply(lambda x: x.wkt if x else None)
        df_csv['merged_h3_geometry'] = df_csv['merged_h3_geometry'].apply(lambda x: x.wkt if x else None)
        df_csv['points'] = df_csv['points'].apply(lambda lst: [p.wkt for p in lst] if lst else [])
        df_csv.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"🟢 Saved CSV output to backup storage: {csv_path}")

        # 3b. Export to GeoJSON
        gdf_wgs84_copy = gdf_wgs84.copy()
        gdf_wgs84_copy['points'] = gdf_wgs84_copy['points'].apply(lambda lst: [[p.x, p.y] for p in lst] if lst else [])
        gdf_wgs84_copy['merged_h3_geometry'] = gdf_wgs84_copy['merged_h3_geometry'].apply(lambda x: x.__geo_interface__ if x else None)
        
        gdf_wgs84_copy.to_file(json_path, driver="GeoJSON")
        logger.info(f"🟢 Saved GeoJSON output to backup storage: {json_path}")

        # 3c. Push to PostgreSQL silver schema
        logger.info("Formatting data and pushing to PostgreSQL silver schema...")
        df_silver = pd.DataFrame(gdf_wgs84_copy)
        df_silver['id_rue'] = df_silver.index
        
        # Serialize Shapely objects and nested datatypes for SQL integration
        df_silver['geometry_wgs84_wkt'] = df_silver['geometry_coordinates_obj'].apply(lambda geom: geom.wkt if geom else None)
        df_silver['points_json'] = df_silver['points'].apply(lambda lst: json.dumps(lst) if lst else None)
        df_silver['hexes_json'] = df_silver['hexes'].apply(lambda lst: json.dumps(lst) if lst else None)
        df_silver['merged_h3_geometry_json'] = df_silver['merged_h3_geometry'].apply(lambda d: json.dumps(d) if d else None)
        
        # Tracking metadata
        paris_timezone = pytz.timezone('Europe/Paris')
        df_silver['transformed_at'] = datetime.now(paris_timezone)
        
        columns_to_write = [
            'id_rue',
            'properties_libelle',
            'properties_sens',
            'properties_etat',
            'properties_vitesse',
            'properties_last_update',
            'properties_est_a_jour',
            'speed_category',
            'speed_color_map',
            'geometry_wgs84_wkt',
            'points_json',
            'hexes_json',
            'merged_h3_geometry_json',
            'transformed_at'
        ]
        
        columns_to_write = [col for col in columns_to_write if col in df_silver.columns]
        df_silver_clean = df_silver[columns_to_write].copy()
        
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))
            logger.info("Appending clean records to silver.trafic_vitesse_propre table...")
            df_silver_clean.to_sql(
                name="trafic_vitesse_propre",
                con=conn,
                schema="silver",
                if_exists="append",
                index=False,
                chunksize=500
            )
        logger.info("🟢 Successfully pushed transformed data to silver.trafic_vitesse_propre!")
    except Exception as e:
        logger.error(f"🔴 Transformation failed: {e}")
        raise e
    finally:
        engine.dispose()


# ============================================================================
# AIRFLOW DAG ORCHESTRATION LAYOUT
# ============================================================================
default_args = {
    'owner': 'lyonflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    dag_id='lyonflow_traffic_pipeline',
    default_args=default_args,
    description='Unified LyonFlow pipeline: Ingest real-time Grand Lyon API data, then apply spatial H3 transformations',
    schedule_interval='*/5 * * * *', # Executes the entire pipeline every 5 minutes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['lyonflow', 'pipeline', 'ingest', 'transform', 'h3', 'unified'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_grand_lyon_traffic',
        python_callable=ingest_traffic_data,
    )

    transform_task = PythonOperator(
        task_id='spatial_transformation_and_mapping',
        python_callable=transform_traffic_data,
    )

    # Define sequential dependency
    ingest_task >> transform_task
