"""
dag_monitoring.py

DAG Airflow dédié au suivi continu (continuous monitoring) de la performance prédictive du modèle STGCN.
Il s'exécute quotidiennement à 11h00, après la période de pointe du matin (07h00 - 10h00).
Le script d'analyse d'observabilité Evidently AI est soumis de façon isolée au cluster Ray.

Formulation impersonnelle pour la conformité de la documentation de projet.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Configuration du Logger d'Airflow
logger = logging.getLogger("airflow.task")


def trigger_evidently_monitoring_on_ray():
    """
    Soumet le script d'analyse d'observabilité Evidently AI au cluster Ray
    via l'API Jobs Dashboard et suit son exécution jusqu'à complétion.
    """
    import time

    import requests
    from airflow.hooks.base import BaseHook

    ray_dashboard_url = "http://ray-head:8265"
    submit_url = f"{ray_dashboard_url}/api/jobs/"

    # Récupération dynamique et sécurisée des secrets depuis les connexions Airflow
    try:
        conn = BaseHook.get_connection("postgres_default")
        db_host = conn.host or "postgres"
        db_port = str(conn.port) or "5432"
        db_user = conn.login or "lyonflow"
        db_password = conn.password or ""
        db_name = conn.schema or "lyonflow"
        logger.info("🔐 Secrets PostgreSQL récupérés avec succès depuis la connexion Airflow 'postgres_default'.")
    except Exception as e:
        logger.warning(
            f"⚠️ Impossible de récupérer la connexion Airflow 'postgres_default' ({e}). Repli sur l'environnement..."
        )
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_user = os.getenv("POSTGRES_USER", "lyonflow")
        db_password = os.getenv("POSTGRES_PASSWORD", "")
        db_name = os.getenv("POSTGRES_DB", "lyonflow")

    payload = {
        "entrypoint": "cd /home/ray/project && python utils/monitoring_evidently.py",
        "runtime_env": {
            "env_vars": {
                "POSTGRES_HOST": db_host,
                "POSTGRES_PORT": db_port,
                "POSTGRES_USER": db_user,
                "POSTGRES_PASSWORD": db_password,
                "POSTGRES_DB": db_name,
                "DATA_FOLDER_OUT": "/home/ray/project/data/out",
            }
        },
    }

    logger.info(f"Soumission du job de monitoring à Ray sur {submit_url}...")
    response = requests.post(submit_url, json=payload, timeout=30)
    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]
    logger.info(f"Job Ray de monitoring soumis avec succès. ID : {job_id}")

    status_url = f"{ray_dashboard_url}/api/jobs/{job_id}"
    while True:
        time.sleep(10)
        status_resp = requests.get(status_url, timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data["status"]
        logger.info(f"État du Job Ray de monitoring {job_id} : {status}")

        if status == "SUCCEEDED":
            logger.info("🟢 Le Job Ray de monitoring Evidently a été complété avec succès !")
            break
        elif status in ["FAILED", "STOPPED"]:
            error_msg = f"🔴 Le Job Ray de monitoring Evidently a échoué avec le statut : {status}."
            logger.error(error_msg)
            try:
                logs_url = f"{ray_dashboard_url}/api/jobs/{job_id}/logs"
                logs_resp = requests.get(logs_url, timeout=30)
                if logs_resp.status_code == 200:
                    logger.error(f"Logs du job de monitoring Ray :\n{logs_resp.json().get('logs', '')}")
            except Exception as e:
                logger.warning(f"Impossible de récupérer les logs du job : {e}")
            raise Exception(error_msg)


# Configuration par défaut du DAG
default_args = {
    "owner": "lyonflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="lyonflow_monitoring_pipeline",
    default_args=default_args,
    description="Pipeline quotidien de suivi de performance et d observabilite STGCN avec Evidently AI",
    schedule_interval="0 11 * * *",  # S'exécute chaque jour à 11h00 locale/UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["lyonflow", "monitoring", "evidently", "observability"],
) as dag:
    run_monitoring_task = PythonOperator(
        task_id="trigger_evidently_monitoring",
        python_callable=trigger_evidently_monitoring_on_ray,
    )
