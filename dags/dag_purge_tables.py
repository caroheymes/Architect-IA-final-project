# -*- coding: utf-8 -*-
"""
dag_purge_tables.py
===================
DAG Airflow de maintenance quotidienne exécuté chaque jour à 07h00.
Il purge de manière automatisée et sécurisée les données obsolètes (fenêtre glissante de 2 jours)
dans les tables PostgreSQL bronze.trafic_vitesse_brute et silver.trafic_vitesse_propre,
puis libère l'espace disque inutilisé à l'aide d'un VACUUM.
"""

import os
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text

# Configuration du Logger d'Airflow
logger = logging.getLogger("airflow.task")

def run_purge_and_vacuum():
    """
    Exécute la purge des anciennes lignes (fenêtre glissante de 2 jours)
    puis lance un VACUUM pour nettoyer l'espace disque.
    """
    # 1. Récupération des informations de connexion depuis les variables d'environnement
    db_user = os.getenv("POSTGRES_USER", "lyonflow")
    db_password = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_db = os.getenv("POSTGRES_DB", "lyonflow")

    database_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_db}"
    engine = create_engine(database_url, pool_pre_ping=True)

    # 2. Exécution de la purge (DELETE)
    logger.info("Début de la purge des tables Bronze et Silver (Rétention : 2 jours)...")
    try:
        with engine.begin() as conn:
            # Purge de la table Bronze (fetched_at)
            query_purge_bronze = text("""
                DELETE FROM bronze.trafic_vitesse_brute 
                WHERE fetched_at < NOW() - INTERVAL '2 days';
            """)
            result_bronze = conn.execute(query_purge_bronze)
            logger.info(f"🟢 Couche Bronze : {result_bronze.rowcount} lignes obsolètes purgées.")

            # Purge de la table Silver (transformed_at)
            query_purge_silver = text("""
                DELETE FROM silver.trafic_vitesse_propre 
                WHERE transformed_at < NOW() - INTERVAL '2 days';
            """)
            result_silver = conn.execute(query_purge_silver)
            logger.info(f"🟢 Couche Silver : {result_silver.rowcount} lignes obsolètes purgées.")
            
    except Exception as e:
        logger.error(f"🔴 Échec lors de l'exécution de la purge : {e}")
        raise e

    # 3. Exécution du VACUUM (nécessite d'être hors transaction avec autocommit)
    logger.info("Lancement du VACUUM pour nettoyer l'espace des lignes supprimées...")
    try:
        # Activer autocommit pour permettre le VACUUM hors bloc de transaction
        autocommit_engine = engine.execution_options(isolation_level="AUTOCOMMIT")
        with autocommit_engine.connect() as conn:
            logger.info("Exécution du VACUUM sur bronze.trafic_vitesse_brute...")
            conn.execute(text("VACUUM bronze.trafic_vitesse_brute;"))
            
            logger.info("Exécution du VACUUM sur silver.trafic_vitesse_propre...")
            conn.execute(text("VACUUM silver.trafic_vitesse_propre;"))
            
        logger.info("🟢 VACUUM complété avec succès ! L'espace de stockage est maintenant disponible.")
    except Exception as e:
        logger.warning(f"⚠️ Le VACUUM a échoué ou a été interrompu ({e}). Ce n'est pas bloquant pour la purge.")

# Configuration par défaut du DAG
default_args = {
    'owner': 'lyonflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='lyonflow_database_purge_pipeline',
    default_args=default_args,
    description='Purge automatique des couches Bronze et Silver de PostgreSQL (rétention 2 jours)',
    schedule_interval='0 7 * * *',  # S'exécute tous les jours à 07h00 locale/UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['lyonflow', 'maintenance', 'db_clean'],
) as dag:

    purge_task = PythonOperator(
        task_id='run_db_purge_and_vacuum',
        python_callable=run_purge_and_vacuum,
    )

    purge_task
