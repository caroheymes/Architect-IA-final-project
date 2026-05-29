# -*- coding: utf-8 -*-
"""
transform_data.py

This script implements 'dag 2: transformation des données et préparation des données cartes'.
It fetches the latest raw JSON payload from the PostgreSQL 'bronze.trafic_vitesse_brute' table,
performs spatial coordinate interpolation every 7 meters, maps coordinates to H3 resolution 13 cells,
aggregates average speed and categories, and saves the final datasets to the mounted /opt/airflow/data/ volume.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import h3
import pyproj
import pytz
from datetime import datetime
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform
from sqlalchemy import create_engine, text

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

# ============================================================================
# HELPER SPATIAL FUNCTIONS
# ============================================================================
def transform_line_to_point(ligne_2154):
    """Transforme une LineString en liste de points tous les 7 mètres."""
    if not ligne_2154 or ligne_2154.is_empty:
        return []
    
    # Convertisseur de EPSG:2154 (mètres) vers EPSG:4326 (degrés - WGS84)
    proj_vers_4326 = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform

    # Générer les intervalles tous les 7 mètres
    distances = np.arange(0, ligne_2154.length, 7)
    points = [ligne_2154.interpolate(d) for d in distances]

    # Inclure le point de fin exact
    if ligne_2154.length % 7 != 0:
        points.append(ligne_2154.interpolate(ligne_2154.length))

    # Reprojeter les points générés en EPSG:4326
    points = [transform(proj_vers_4326, p) for p in points]
    return points

def create_merged_polygon_from_hexes(h3_id_list):
    """Regroupe une liste d'IDs H3 en un seul polygone unifié."""
    if not h3_id_list:
        return None

    unique_hexes = list(set(h3_id_list))

    try:
        # Support pour H3 v4 et fallback robuste
        if hasattr(h3, 'cells_to_geo'):
            geojson_dict = h3.cells_to_geo(unique_hexes)
            return shape(geojson_dict)
        elif hasattr(h3, 'cells_to_geojson'):
            geojson_dict = h3.cells_to_geojson(unique_hexes)
            return shape(geojson_dict)
        else:
            # Fallback en générant et unifiant les polygones individuels de chaque hexagone
            polygons = []
            for h in unique_hexes:
                boundary = h3.cell_to_boundary(h)
                polygons.append(Polygon([(lon, lat) for lat, lon in boundary]))
            from shapely.ops import unary_union
            return unary_union(polygons)
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des hexagones H3: {e}")
        return None

def get_speed_category(speed):
    """Catégorise la vitesse pour le mappage de couleurs."""
    if pd.isna(speed):
        return "Unknown"
    elif speed <= 20:
        return "Slow (0-20 km/h)"
    elif speed > 50:
        return "Fast (>50 km/h)"
    else:
        return "Medium (20-50 km/h)"

# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================
def transform_traffic_data():
    logger.info("Connecting to PostgreSQL database...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    try:
        # 1. Fetch latest raw payload from postgres
        with engine.begin() as conn:
            logger.info("Fetching the latest raw traffic payload from bronze layer...")
            query = text("""
                SELECT raw_data FROM bronze.trafic_vitesse_brute 
                ORDER BY fetched_at DESC LIMIT 1;
            """)
            result = conn.execute(query).fetchone()

        if not result:
            logger.warning("⚠️ No data found in bronze.trafic_vitesse_brute table. Cannot proceed with transformation.")
            return False

        raw_payload = result[0]
        logger.info("Successfully fetched raw payload from database.")

        # 2. Re-create dataframe logic
        features = raw_payload.get('features', [])
        if not features:
            logger.warning("⚠️ Raw payload contains no features.")
            return False

        trafic = pd.json_normalize(features)
        
        # Replace dots with underscores to match original colab logic
        cols = [c.replace('.', '_') for c in trafic.columns]
        trafic.columns = cols

        selected_columns = [
            'geometry_coordinates', 'properties_libelle', 'properties_sens', 
            'properties_etat', 'properties_vitesse', 'properties_last_update', 
            'properties_est_a_jour'
        ]

        # Check and handle missing selected columns
        for col in selected_columns:
            if col not in trafic.columns:
                trafic[col] = None

        trafic = trafic[selected_columns]

        # Filter out-of-date features
        if 'properties_est_a_jour' in trafic.columns:
            trafic = trafic[trafic.properties_est_a_jour != False]

        # Convert geometry coordinates to LineString objects
        trafic['geometry_coordinates_obj'] = trafic['geometry_coordinates'].apply(LineString)

        # Create the GeoDataFrame (EPSG:2154 as in original script)
        gdf = GeoDataFrame(data=trafic, geometry='geometry_coordinates_obj')
        gdf.set_crs(epsg=2154, inplace=True, allow_override=True)

        logger.info("Interpolating streets to points every 7 meters...")
        gdf['points'] = [transform_line_to_point(elem) for elem in gdf.geometry_coordinates_obj]

        logger.info("Mapping point list to H3 resolution 13 hex cells...")
        gdf['hexes'] = gdf.points.apply(lambda point_list: [h3.latlng_to_cell(p.y, p.x, 13) for p in point_list])

        logger.info("Generating merged polygon boundaries from H3 cells...")
        gdf['merged_h3_geometry'] = gdf['hexes'].apply(create_merged_polygon_from_hexes)

        # Speed normalization and clean-up
        gdf['properties_vitesse'] = gdf['properties_vitesse'].astype(str).str.split(' ').str[0]
        gdf['properties_vitesse'] = pd.to_numeric(gdf['properties_vitesse'], errors="coerce")

        # Average speed fallback aggregation per street name
        mean_speed_df = gdf.groupby(by="properties_libelle").agg(mean_speed=('properties_vitesse', 'mean')).reset_index()
        gdf = gdf.merge(mean_speed_df, left_on='properties_libelle', right_on='properties_libelle', how='left')
        gdf['properties_vitesse'] = [
            elem if not pd.isna(elem) else mean_speed if not pd.isna(mean_speed) else np.nan 
            for elem, mean_speed in zip(gdf.properties_vitesse, gdf.mean_speed)
        ]
        gdf = gdf.drop(columns=['mean_speed'], errors='ignore')

        # Speed categorization & mapping
        gdf['speed_category'] = [get_speed_category(speed) for speed in gdf.properties_vitesse]
        gdf['speed_color_map'] = gdf['speed_category'].map({
            "Slow (0-20 km/h)": "red",
            "Medium (20-50 km/h)": "orange",
            "Fast (>50 km/h)": "green",
            "Unknown": "gray"
        })

        # Final coordinate conversion to WGS84
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf['id_rue'] = gdf.index

        # Ensure export folder exists
        output_dir = "/opt/airflow/data"
        os.makedirs(output_dir, exist_ok=True)

        # Get formatted datetime according to Paris timezone
        now = datetime.now(pytz.timezone('Europe/Paris')).strftime(format="%Y_%m_%d_%H_%M")
        
        # Define output filepaths
        csv_path = f"{output_dir}/{now}_transformed.csv"
        json_path = f"{output_dir}/{now}_transformed.json"

        # 3a. Export to CSV (stringifying complex shapes to WKT so they write cleanly)
        df_csv = gdf.copy()
        df_csv['geometry_coordinates_obj'] = df_csv['geometry_coordinates_obj'].apply(lambda x: x.wkt if x else None)
        df_csv['merged_h3_geometry'] = df_csv['merged_h3_geometry'].apply(lambda x: x.wkt if x else None)
        df_csv['points'] = df_csv['points'].apply(lambda lst: [p.wkt for p in lst] if lst else [])
        df_csv.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"🟢 Saved CSV output to: {csv_path}")

        # 3b. Export to JSON (GeoJSON format is perfect for preserving map data structures!)
        gdf_wgs84_copy = gdf_wgs84.copy()
        # Serialize nested geometries & lists to standard coordinates for GeoJSON compatibility
        gdf_wgs84_copy['points'] = gdf_wgs84_copy['points'].apply(lambda lst: [[p.x, p.y] for p in lst] if lst else [])
        gdf_wgs84_copy['merged_h3_geometry'] = gdf_wgs84_copy['merged_h3_geometry'].apply(lambda x: x.__geo_interface__ if x else None)
        
        gdf_wgs84_copy.to_file(json_path, driver="GeoJSON")
        logger.info(f"🟢 Saved JSON/GeoJSON output to: {json_path}")

        logger.info("🟢 Transformation and map preparation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"🔴 Transformation failed: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    transform_traffic_data()
