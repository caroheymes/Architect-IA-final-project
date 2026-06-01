# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text

st.set_page_code_info = {
    "page_title": "LyonFlow - Traffic Prediction",
    "page_icon": "🚦",
    "layout": "wide"
}

st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚦 LyonFlow</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plateforme temps réel de prédiction de trafic - Métropole de Lyon</div>', unsafe_allow_html=True)

# Database Connection Status
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

st.sidebar.header("Configuration & Status")
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT NOW()")).fetchone()
        st.sidebar.success("🟢 Connecté à PostgreSQL")
        st.sidebar.caption(f"Heure Serveur: {res[0]}")
except Exception as e:
    st.sidebar.error("🔴 Erreur de connexion PostgreSQL")
    st.sidebar.caption(str(e))

st.sidebar.markdown("---")
st.sidebar.info("""
    **LyonFlow Stack**:
    - **Orchestration**: Apache Airflow
    - **Calcul Distribué**: Ray Core
    - **Tracking ML**: MLflow & Skore
    - **Base de données**: PostgreSQL
""")

def get_mlflow_artifact_plot():
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)
    
    specific_run_id = "eb4789d2e3374056aede9faa588334c8"
    plot_subpath = "plots/stratified_error_analysis.png"
    
    # 1. Try downloading from the specific run ID
    try:
        local_path = client.download_artifacts(specific_run_id, plot_subpath)
        if os.path.exists(local_path):
            return local_path, specific_run_id, "STGCN Production Run"
    except Exception:
        pass
        
    # 2. Try searching the latest run in experiment 7
    try:
        runs = client.search_runs(
            experiment_ids=["7"],
            order_by=["attribute.start_time DESC"],
            max_results=5
        )
        for run in runs:
            run_id = run.info.run_id
            try:
                artifacts = client.list_artifacts(run_id, path="plots")
                if any(art.path == plot_subpath for art in artifacts):
                    local_path = client.download_artifacts(run_id, plot_subpath)
                    if os.path.exists(local_path):
                        return local_path, run_id, run.info.run_name or "STGCN Run"
            except Exception:
                continue
    except Exception:
        pass
        
    return None, None, None

# Main page columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <h3>Segments H3 (Res 13)</h3>
            <h2 style="color: #2563EB;">~2 500</h2>
            <p>Mises à jour toutes les 5 min</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <h3>Modèles de Prédiction</h3>
            <h2 style="color: #059669;">ST-GRU-GNN</h2>
            <p>Spatiotemporel (MLflow & Skore)</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <h3>Volume Ingestion</h3>
            <h2 style="color: #D97706;">En cours...</h2>
            <p>Couche Bronze lancée</p>
        </div>
    """, unsafe_allow_html=True)

st.write("---")

tab1, tab2 = st.tabs(["🗺️ Visualisation Temps Réel", "📊 Performances & Analyse du Modèle (STGCN)"])

with tab1:
    st.subheader("État du réseau routier en temps réel")
    st.info("Les données d'ingestion et de prédiction s'afficheront ici dès que le premier cycle d'orchestration Airflow aura complété l'ingestion bronze et la transformation silver.")

with tab2:
    st.subheader("Analyse d'Erreur Stratifiée du Modèle de Production")
    st.markdown("""
    Cette vue présente l'évaluation détaillée des performances réelles du dernier modèle de Deep Learning géométrique **Spatio-Temporal GNN (ST-GRU-GNN)** entraîné sur les données de la Métropole de Lyon.
    
    Le graphique est découpé en 4 analyses clés pour garantir l'honnêteté et la transparence scientifique de notre modèle :
    1. **MAE par tranche de vitesse** : Évaluation de la précision du modèle selon l'état du trafic (embouteillages/ralentissements vs trafic fluide).
    2. **Biais de prédiction systématique** : Identification des zones de sur-estimation (vitesse prédite > réelle) ou de sous-estimation.
    3. **Dispersion et Incertitude** : Analyse de l'écart-type des prédictions et de l'incertitude sur les résidus d'erreurs.
    4. **Boîtes à moustaches (Boxplots)** : Comparaison de la distribution statistique des prédictions du modèle par rapport à la réalité physique du terrain.
    """)

    latest_plot_path = "models/stratified_error_analysis.png"
    fallback_plot_path = "trash/model_metrics.png"
    ideal_plot_path = "trash/ideal.png"

    # Attempt to pull from MLflow dynamically
    mlflow_plot_path, run_id, run_name = get_mlflow_artifact_plot()

    if mlflow_plot_path and os.path.exists(mlflow_plot_path):
        st.success(f"🟢 Graphique de performance récupéré dynamiquement depuis MLflow (Run: `{run_id}` | `{run_name}`).")
        st.image(mlflow_plot_path, caption=f"Analyse d'erreur stratifiée - MLflow (Run ID: {run_id})", use_container_width=True)
    elif os.path.exists(latest_plot_path):
        st.success("🟢 Graphique de performance généré localement par le dernier entraînement de production.")
        st.image(latest_plot_path, caption="Analyse d'erreur stratifiée - Modèle Actuel (Généré en production locale)", use_container_width=True)
    elif os.path.exists(ideal_plot_path):
        st.info("💡 Aucun entraînement personnalisé n'a encore été finalisé sur cette machine. Affichage du graphique de performance idéal de référence.")
        st.image(ideal_plot_path, caption="Analyse d'erreur stratifiée de référence (Cible finale)", use_container_width=True)
    elif os.path.exists(fallback_plot_path):
        st.warning("⚠️ Aucun entraînement personnalisé n'a encore été finalisé. Affichage des métriques de référence initiales.")
        st.image(fallback_plot_path, caption="Analyse de performance initiale (STGCN de base)", use_container_width=True)
    else:
        st.error("❌ Aucun graphique de performance de référence n'a pu être localisé dans l'espace de travail.")
