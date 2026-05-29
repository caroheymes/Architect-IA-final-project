# -*- coding: utf-8 -*-
"""
ingest_api.py

This script connects to the Grand Lyon WFS API, downloads the real-time 
traffic speed data, and inserts the raw JSON payload into the 
'bronze.trafic_vitesse_brute' table of your local PostgreSQL database running in Docker.
"""

import os
import json
import logging
import requests
from datetime import datetime
import pytz
from sqlalchemy import create_engine, text

from sqlalchemy.exc import SQLAlchemyError

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
# API Credentials & URL
API_URL = "https://data.grandlyon.com/geoserver/metropole-de-lyon/ows?SERVICE=WFS&VERSION=2.0.0&request=GetFeature&typename=metropole-de-lyon:pvo_patrimoine_voirie.pvotrafic&outputFormat=application/json&SRSNAME=EPSG:2154&startIndex=0&sortby=gid"
API_LOGIN = os.getenv("API_LOGIN")
API_PASSWORD = os.getenv("API_PASSWORD")

# PostgreSQL local Connection (Docker)
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

# ============================================================================
# MAIN INGESTION FUNCTION
# ============================================================================
def ingest_traffic_data():
    # 1. Fetch data from Grand Lyon API
    logger.info("Fetching real-time traffic data from Grand Lyon API...")
    try:
        response = requests.get(API_URL, auth=(API_LOGIN, API_PASSWORD), timeout=30)
        response.raise_for_status()
        raw_payload = response.json()
        logger.info("Successfully downloaded data from API.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from API: {e}")
        return False

    # Get local Paris time
    timezone = pytz.timezone('Europe/Paris')
    fetched_at = datetime.now(timezone)

    # 2. Connect to local PostgreSQL running in Docker
    logger.info("Connecting to local PostgreSQL database...")
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        # We use a context manager to handle connections cleanly
        with engine.begin() as conn:
            # Create schema and table if they do not exist
            logger.info("Ensuring schema and table exist...")
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

            # Insert raw JSON data using parameterized query (prevents injection and handles JSON conversion)
            logger.info("Inserting raw JSON data into bronze layer...")
            insert_query = text("""
                INSERT INTO bronze.trafic_vitesse_brute (fetched_at, raw_data)
                VALUES (:fetched_at, :raw_data);
            """)
            conn.execute(insert_query, {
                "fetched_at": fetched_at,
                "raw_data": json.dumps(raw_payload)
            })

        logger.info(f"🟢 Successfully ingested raw traffic data at {fetched_at.strftime('%Y-%m-%d %H:%M:%S %Z')}!")
        return True

    except SQLAlchemyError as e:
        logger.error(f"🔴 PostgreSQL Database Error: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    ingest_traffic_data()
