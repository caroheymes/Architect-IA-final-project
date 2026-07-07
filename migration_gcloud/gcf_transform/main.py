import json
import logging
import os
from datetime import datetime

import functions_framework
import h3
import numpy as np
import pandas as pd
import pyproj
import pytz
from google.cloud import bigquery, storage
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("BQ_PROJECT_ID")
BRONZE_DATASET = os.environ.get("BQ_BRONZE_DATASET", "bronze")
BRONZE_TABLE = os.environ.get("BQ_BRONZE_TABLE", "trafic_vitesse_brute")
SILVER_DATASET = os.environ.get("BQ_SILVER_DATASET", "silver")
SILVER_TABLE = os.environ.get("BQ_SILVER_TABLE", "trafic_vitesse_propre")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

# Helper functions (copied from dag_pipeline.py)
def transform_line_to_point(ligne_2154):
    if not ligne_2154 or ligne_2154.is_empty:
        return []
    proj_vers_4326 = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
    distances = np.arange(0, ligne_2154.length, 7)
    points = [ligne_2154.interpolate(d) for d in distances]
    if ligne_2154.length % 7 != 0:
        points.append(ligne_2154.interpolate(ligne_2154.length))
    points = [transform(proj_vers_4326, p) for p in points]
    return points

def create_merged_polygon_from_hexes(h3_id_list):
    if not h3_id_list:
        return None
    unique_hexes = list(set(h3_id_list))
    try:
        if hasattr(h3, "cells_to_geo"):
            geojson_dict = h3.cells_to_geo(unique_hexes)
            return shape(geojson_dict)
        elif hasattr(h3, "cells_to_geojson"):
            geojson_dict = h3.cells_to_geojson(unique_hexes)
            return shape(geojson_dict)
        else:
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
    # Match user's new threshold logic: < 15 is red, 15-30 is orange, > 30 is green
    if pd.isna(speed):
        return "Inconnu"
    elif speed < 15:
        return "Lent (< 15 km/h)"
    elif speed > 30:
        return "Rapide (> 30 km/h)"
    else:
        return "Moyen (15-30 km/h)"

@functions_framework.http
def transform_traffic_data_gcf(request):
    logger.info("Starting spatial data transformation GCF...")
    if not PROJECT_ID:
        logger.error("BQ_PROJECT_ID environment variable is missing.")
        return "Error: Missing GCP project configuration.", 500
        
    bq_client = bigquery.Client(project=PROJECT_ID)
    
    # 1. Fetch latest raw snapshot from BigQuery
    logger.info("Fetching the latest raw record from bronze BigQuery table...")
    query = f"""
        SELECT raw_data FROM `{PROJECT_ID}.{BRONZE_DATASET}.{BRONZE_TABLE}` 
        ORDER BY fetched_at DESC LIMIT 1
    """
    
    try:
        query_job = bq_client.query(query)
        results = list(query_job.result())
    except Exception as e:
        logger.error(f"Error reading from BigQuery bronze layer: {e}")
        return f"Failed to read from BigQuery: {e}", 500
        
    if not results:
        logger.warning("No data found in bronze layer.")
        return "No source data found.", 200

    raw_data_field = results[0]["raw_data"]
    if isinstance(raw_data_field, str):
        raw_payload = json.loads(raw_data_field)
    else:
        raw_payload = raw_data_field
        
    features = raw_payload.get("features", [])
    if not features:
        logger.warning("Source payload contains no features.")
        return "Empty feature list.", 200

    # 2. DataFrame transformations
    import geopandas as gpd
    from geopandas import GeoDataFrame
    
    trafic = pd.json_normalize(features)
    cols = [c.replace(".", "_") for c in trafic.columns]
    trafic.columns = cols

    selected_columns = [
        "geometry_coordinates",
        "properties_libelle",
        "properties_sens",
        "properties_etat",
        "properties_vitesse",
        "properties_last_update",
        "properties_est_a_jour",
        "properties_twgid",
        "properties_gid",
    ]

    for col in selected_columns:
        if col not in trafic.columns:
            trafic[col] = None

    trafic = trafic[selected_columns]
    if "properties_est_a_jour" in trafic.columns:
        trafic = trafic[trafic.properties_est_a_jour != False]

    # Form spatial attributes
    trafic["geometry_coordinates_obj"] = trafic["geometry_coordinates"].apply(LineString)
    gdf = GeoDataFrame(data=trafic, geometry="geometry_coordinates_obj")
    gdf.set_crs(epsg=2154, inplace=True, allow_override=True)

    logger.info("Interpolating segments every 7 meters...")
    gdf["points"] = [transform_line_to_point(elem) for elem in gdf.geometry_coordinates_obj]

    logger.info("Mapping interpolated coordinates to H3 resolution 13 cells...")
    gdf["hexes"] = gdf.points.apply(lambda pts: [h3.latlng_to_cell(p.y, p.x, 13) for p in pts])

    logger.info("Unifying H3 cell clusters into polygons...")
    gdf["merged_h3_geometry"] = gdf["hexes"].apply(create_merged_polygon_from_hexes)

    # Average speed sanitization and fallback computation
    gdf["properties_vitesse"] = gdf["properties_vitesse"].astype(str).str.split(" ").str[0]
    gdf["properties_vitesse"] = pd.to_numeric(gdf["properties_vitesse"], errors="coerce")

    mean_speed_df = (
        gdf.groupby(by="properties_libelle").agg(mean_speed=("properties_vitesse", "mean")).reset_index()
    )
    gdf = gdf.merge(mean_speed_df, on="properties_libelle", how="left")
    gdf["properties_vitesse"] = [
        elem if not pd.isna(elem) else mean_speed if not pd.isna(mean_speed) else np.nan
        for elem, mean_speed in zip(gdf.properties_vitesse, gdf.mean_speed)
    ]
    gdf = gdf.drop(columns=["mean_speed"], errors="ignore")

    gdf["speed_category"] = [get_speed_category(speed) for speed in gdf.properties_vitesse]
    gdf["speed_color_map"] = gdf["speed_category"].map(
        {"Lent (< 15 km/h)": "red", "Moyen (15-30 km/h)": "orange", "Rapide (> 30 km/h)": "green", "Inconnu": "gray"}
    )

    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf["id_rue"] = gdf.index

    # Form formatted file names
    now_str = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y_%m_%d_%H_%M")
    csv_filename = f"{now_str}_transformed.csv"
    json_filename = f"{now_str}_transformed.json"

    # Export to CSV (Stringify spatial objects)
    df_csv = gdf.copy()
    df_csv["geometry_coordinates_obj"] = df_csv["geometry_coordinates_obj"].apply(lambda x: x.wkt if x else None)
    df_csv["merged_h3_geometry"] = df_csv["merged_h3_geometry"].apply(lambda x: x.wkt if x else None)
    df_csv["points"] = df_csv["points"].apply(lambda lst: [p.wkt for p in lst] if lst else [])
    csv_data = df_csv.to_csv(index=False, encoding="utf-8")

    # Export to GeoJSON
    gdf_wgs84_copy = gdf_wgs84.copy()
    gdf_wgs84_copy["points"] = gdf_wgs84_copy["points"].apply(lambda lst: [[p.x, p.y] for p in lst] if lst else [])
    gdf_wgs84_copy["merged_h3_geometry"] = gdf_wgs84_copy["merged_h3_geometry"].apply(
        lambda x: x.__geo_interface__ if x else None
    )
    json_data = gdf_wgs84_copy.to_json()

    # 3. Upload files to GCS
    if GCS_BUCKET_NAME:
        logger.info(f"Uploading files to GCS bucket: {GCS_BUCKET_NAME}...")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            
            # Upload CSV
            csv_blob = bucket.blob(f"historical/{csv_filename}")
            csv_blob.upload_from_string(csv_data, content_type="text/csv")
            logger.info(f"Uploaded CSV blob: historical/{csv_filename}")
            
            # Upload GeoJSON
            json_blob = bucket.blob(f"historical/{json_filename}")
            json_blob.upload_from_string(json_data, content_type="application/json")
            logger.info(f"Uploaded GeoJSON blob: historical/{json_filename}")
        except Exception as e:
            logger.error(f"Error uploading files to GCS: {e}")
    else:
        logger.warning("GCS_BUCKET_NAME environment variable is not set. Skipping GCS upload.")

    # 4. Push to BigQuery silver schema
    logger.info("Formatting data for BigQuery silver schema...")
    df_silver = pd.DataFrame(gdf_wgs84_copy)
    df_silver["id_rue"] = df_silver.index
    df_silver["geometry_wgs84_wkt"] = df_silver["geometry_coordinates_obj"].apply(
        lambda geom: geom.wkt if geom else None
    )
    df_silver["points_json"] = df_silver["points"].apply(lambda lst: json.dumps(lst) if lst else None)
    df_silver["hexes_json"] = df_silver["hexes"].apply(lambda lst: json.dumps(lst) if lst else None)
    df_silver["merged_h3_geometry_json"] = df_silver["merged_h3_geometry"].apply(
        lambda d: json.dumps(d) if d else None
    )
    df_silver["transformed_at"] = datetime.now(pytz.UTC).isoformat()

    columns_to_write = [
        "id_rue",
        "properties_twgid",
        "properties_gid",
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
    
    # Ensure types are correct for BigQuery JSON load
    df_bq = df_silver[columns_to_write].copy()
    
    # Fill NaN values to prevent BQ loading errors
    df_bq["properties_twgid"] = df_bq["properties_twgid"].fillna(0).astype(int)
    df_bq["properties_gid"] = df_bq["properties_gid"].fillna(0).astype(int)
    df_bq["properties_vitesse"] = df_bq["properties_vitesse"].fillna(0.0).astype(float)
    df_bq["properties_est_a_jour"] = df_bq["properties_est_a_jour"].fillna(False).astype(bool)
    
    rows_to_insert = df_bq.to_dict(orient="records")

    # Initialize silver dataset and table
    dataset_ref = bq_client.dataset(SILVER_DATASET)
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        logger.info(f"Creating dataset {SILVER_DATASET}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        bq_client.create_dataset(dataset)

    table_ref = dataset_ref.table(SILVER_TABLE)
    schema = [
        bigquery.SchemaField("id_rue", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("properties_twgid", "INTEGER"),
        bigquery.SchemaField("properties_gid", "INTEGER"),
        bigquery.SchemaField("properties_libelle", "STRING"),
        bigquery.SchemaField("properties_sens", "STRING"),
        bigquery.SchemaField("properties_etat", "STRING"),
        bigquery.SchemaField("properties_vitesse", "FLOAT"),
        bigquery.SchemaField("properties_last_update", "STRING"),
        bigquery.SchemaField("properties_est_a_jour", "BOOLEAN"),
        bigquery.SchemaField("speed_category", "STRING"),
        bigquery.SchemaField("speed_color_map", "STRING"),
        bigquery.SchemaField("geometry_wgs84_wkt", "STRING"),
        bigquery.SchemaField("points_json", "STRING"),
        bigquery.SchemaField("hexes_json", "STRING"),
        bigquery.SchemaField("merged_h3_geometry_json", "STRING"),
        bigquery.SchemaField("transformed_at", "TIMESTAMP"),
    ]
    
    try:
        bq_client.get_table(table_ref)
    except Exception:
        logger.info(f"Creating table {SILVER_TABLE}...")
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table)

    # Stream write rows
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
    if errors == []:
        logger.info("🟢 Successfully pushed transformed data to BigQuery silver.trafic_vitesse_propre!")
        return "Transformation and upload successful.", 200
    else:
        logger.error(f"Failed to insert rows into silver table: {errors}")
        return f"Failed to write to BigQuery: {errors}", 500
