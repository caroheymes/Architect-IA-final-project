# Dockerfile pour l'image unifiée "lyonflow-app"
# Cette image sert à la fois pour Airflow, Streamlit, et les scripts de transformation/calcul.
FROM apache/airflow:2.9.1-python3.12

USER root

# Installation des dépendances système nécessaires pour GeoPandas, Shapely et la compilation de Prophet/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copie et installation des dépendances Python du projet
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
