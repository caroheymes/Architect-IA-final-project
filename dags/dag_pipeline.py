# -*- coding: utf-8 -*-
"""
dag_pipeline.py

This unified Airflow DAG orchestrates the entire real-time traffic data pipeline.
It chains the two tasks sequentially:
1. Ingestion: Fetches raw JSON payload from the Grand Lyon WFS API and saves it to Postgres bronze.
2. Transformation: Reads the fresh bronze payload, performs spatial interpolations, H3 mappings,
   and exports the ready-to-map datasets to the local ./data/ volume as CSV and JSON.

Flow: ingest_grand_lyon_traffic >> spatial_transformation_and_mapping
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments for DAG tasks
default_args = {
    'owner': 'lyonflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}

def run_api_ingestion():
    """Import and execute the ingestion script."""
    from ingest_api import ingest_traffic_data
    success = ingest_traffic_data()
    if not success:
        raise Exception("Traffic data ingestion failed. Check task logs for details.")

def run_spatial_transformation():
    """Import and execute the transformation script."""
    from transform_data import transform_traffic_data
    success = transform_traffic_data()
    if not success:
        raise Exception("Traffic data transformation failed. Check task logs for details.")

with DAG(
    dag_id='lyonflow_traffic_pipeline',
    default_args=default_args,
    description='Unified LyonFlow pipeline: Ingest real-time Grand Lyon API data, then apply spatial H3 transformations',
    schedule_interval='*/5 * * * *', # Executes the entire pipeline every 5 minutes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['lyonflow', 'pipeline', 'ingest', 'transform', 'h3'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_grand_lyon_traffic',
        python_callable=run_api_ingestion,
    )

    transform_task = PythonOperator(
        task_id='spatial_transformation_and_mapping',
        python_callable=run_spatial_transformation,
    )

    # Define sequential dependency
    ingest_task >> transform_task
