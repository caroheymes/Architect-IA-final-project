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
            <h2 style="color: #059669;">XGBoost / Prophet</h2>
            <p>Suivi via MLflow & Skore</p>
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

st.header("🗺️ Visualisation temps réel")
st.info("Les données d'ingestion et de prédiction s'afficheront ici dès que le premier cycle d'orchestration Airflow aura complété l'ingestion bronze et la transformation silver.")
