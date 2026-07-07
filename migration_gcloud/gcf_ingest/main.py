import json
import logging
import os
from datetime import datetime

import functions_framework
import pytz
import requests
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
API_URL = "https://data.grandlyon.com/geoserver/metropole-de-lyon/ows?SERVICE=WFS&VERSION=2.0.0&request=GetFeature&typename=metropole-de-lyon:pvo_patrimoine_voirie.pvotrafic&outputFormat=application/json&SRSNAME=EPSG:2154&startIndex=0&sortby=gid"
API_LOGIN = os.environ.get("API_LOGIN")
API_PASSWORD = os.environ.get("API_PASSWORD")
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("BQ_PROJECT_ID")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "bronze")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "trafic_vitesse_brute")


@functions_framework.http
def ingest_traffic_data_gcf(request):
    logger.info("Starting real-time traffic data ingestion...")
    if not API_LOGIN or not API_PASSWORD:
        logger.error("API_LOGIN or API_PASSWORD environment variables are missing.")
        return "Error: Missing WFS credentials.", 500

    try:
        response = requests.get(API_URL, auth=(API_LOGIN, API_PASSWORD), timeout=30)
        response.raise_for_status()
        raw_payload = response.json()
        logger.info("Successfully fetched data from Grand Lyon WFS API.")
    except Exception as e:
        logger.error(f"Error querying Grand Lyon API: {e}")
        return f"Ingestion failed: {e}", 500

    # Get fetch timestamp in local Paris time and UTC
    timezone = pytz.timezone("Europe/Paris")
    fetched_at_local = datetime.now(timezone)
    fetched_at_utc = datetime.now(pytz.UTC)

    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)

    # Create dataset if not exists
    dataset_ref = client.dataset(DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {DATASET_ID} already exists.")
    except Exception:
        logger.info(f"Creating dataset {DATASET_ID}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"  # Lyon is in Europe, so EU location makes sense
        client.create_dataset(dataset)

    # Create table if not exists
    table_ref = dataset_ref.table(TABLE_ID)
    schema = [
        bigquery.SchemaField("fetched_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("raw_data", "JSON", mode="REQUIRED"),
    ]

    try:
        client.get_table(table_ref)
        logger.info(f"Table {TABLE_ID} already exists.")
    except Exception:
        logger.info(f"Creating table {TABLE_ID}...")
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)

    # Insert row
    rows_to_insert = [{"fetched_at": fetched_at_utc.isoformat(), "raw_data": json.dumps(raw_payload)}]

    errors = client.insert_rows_json(table_ref, rows_to_insert)
    if errors == []:
        logger.info(
            f"🟢 Ingestion successfully saved to BigQuery at {fetched_at_local.strftime('%Y-%m-%d %H:%M:%S %Z')}!"
        )
        return "Ingestion successful.", 200
    else:
        logger.error(f"Failed to insert rows: {errors}")
        return f"Database insert failed: {errors}", 500
