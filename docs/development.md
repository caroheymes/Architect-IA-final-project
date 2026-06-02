# Guide de Développement

## Prérequis

- Python 3.12+
- Docker Desktop (avec backend WSL2 sur Windows, ou Docker natif sur macOS/Linux)
- GPU NVIDIA + drivers CUDA 12.1+ (pour l'entraînement Ray Worker)

## Installation Locale

```bash
# Cloner le dépôt
git clone <repo-url>
cd FinalProjet

# Créer le fichier d'environnement
cp .env.example .env
# Éditer .env avec vos credentials Grand Lyon, etc.

# Lancer la stack complète
docker-compose up -d --build
```

## Commandes Fréquentes

### Qualité de Code

```bash
# Lint (vérification)
ruff check .

# Lint (auto-fix)
ruff check . --fix

# Format (vérification)
ruff format --check .

# Format (application)
ruff format .

# Type check (non-bloquant)
mypy dags/ training/ utils/ app.py --ignore-missing-imports --namespace-packages --explicit-package-bases
```

### Tests

```bash
# Tous les tests
pytest tests/ -v --tb=short

# Test spécifique
pytest tests/test_stgcn_model.py -v

# Avec rapport JUnit
pytest tests/ -v --tb=short --junitxml=reports/junit.xml
```

### Docker

```bash
# Démarrer la stack
docker-compose up -d --build

# Voir les logs d'un service
docker-compose logs -f airflow-scheduler
docker-compose logs -f ray-worker

# Exécuter une commande dans un conteneur
docker exec -it lyonflow-ray-worker bash

# État des conteneurs
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Entraînement ML

```bash
# Entraînement simple (dans le conteneur Ray)
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/train_stgcn.py

# HPO Optuna (dans le conteneur Ray)
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/hpo_stgcn.py

# Optuna Dashboard
optuna-dashboard postgresql://lyonflow:lyonflow_password@localhost:5432/lyonflow
```

### Utilitaires

```bash
# Export Gold → CSV
python utils/export_db_to_csv.py

# Inspecter la couche Silver
python utils/inspect_silver.py

# Lister toutes les tables
python utils/list_all_tables.py

# Migrer données historiques vers Silver
python utils/migrate_historical_to_silver.py

# Vérifier les NULLs
python utils/check_nulls.py
```

## Structure du Projet

```
FinalProjet/
├── .github/workflows/
│   ├── ci.yml                     # CI : lint + tests + Docker build + Trivy
│   ├── cd.yml                     # CD : push GHCR + K8s staging + smoke tests
│   └── ml-training.yml            # ML : HPO + train champion + promote
├── dags/
│   └── dag_pipeline.py            # DAG Airflow unique (5 tâches séquentielles)
├── training/stgcn/
│   ├── model.py                   # Architecture ST-GRU-GNN (PyTorch Geometric)
│   ├── dataset.py                 # Chargement données Gold → DataLoader PyG
│   ├── train_stgcn.py             # Entraînement + MLflow + analyse stratifiée
│   ├── hpo_stgcn.py               # HPO Optuna distribué via Ray
│   ├── predict_stgcn.py           # Script d'inférence
│   ├── backfill_predictions.py    # Rétro-prédictions historiques
│   ├── get_best_params.py         # Extraction meilleurs hyperparamètres
│   └── test_perf.py               # Benchmarks de performance
├── utils/
│   ├── export_db_to_csv.py        # Export Gold → fichiers CSV plats
│   ├── migrate_historical_to_silver.py  # Migration données historiques
│   ├── rebuild_silver_from_bronze.py    # Reconstruction Silver depuis Bronze
│   ├── inspect_silver.py          # Inspection couche Silver
│   ├── check_nulls.py             # Vérification valeurs manquantes
│   ├── check_unmatched_detail.py  # Détail des segments non appariés
│   ├── inspect_unmatched_geoms.py # Inspection géométries non matchées
│   ├── create_ref_segments.py     # Création segments de référence
│   ├── backfill_rounded_wkt.py    # Backfill WKT arrondis
│   ├── list_all_tables.py         # Liste toutes les tables PostgreSQL
│   ├── kill_backends.py           # Kill des connexions PostgreSQL
│   └── profile_rebuild.py         # Profilage reconstruction Silver
├── tests/
│   ├── test_ingest.py             # Tests ingestion Bronze
│   ├── test_transform.py          # Tests transformation Silver
│   ├── test_stgcn_model.py        # Tests architecture modèle
│   ├── test_migrate_historical.py # Tests migration historique
│   ├── test_bronze_fields.py      # Tests champs Bronze
│   ├── test_query.py              # Tests requêtes SQL
│   ├── test_pg_status.py          # Tests statut PostgreSQL
│   ├── test_cache_hits.py         # Tests cache hits
│   └── test_super_cache.py        # Tests super cache
├── kubernetes/                     # Manifests K8s (6 fichiers)
├── data/
│   ├── in/                        # Données d'entrée (CSV pour entraînement)
│   ├── raw/                       # Données brutes exportées
│   └── vps_export/                # Exports VPS
├── models/                        # Artéfacts modèle (scaler.pkl, poids .pt)
├── docs/                          # Documentation technique
├── app.py                         # Dashboard Streamlit
├── docker-compose.yml             # Stack complète (8 services)
├── Dockerfile                     # Image Airflow/Streamlit
├── Dockerfile.ray                 # Image Ray (GPU + PyTorch)
├── init-db.sql                    # Init bases airflow + mlflow
├── requirements.txt               # Dépendances Python
├── pyproject.toml                 # Config ruff, mypy, pytest
└── .env.example                   # Template variables d'environnement
```

## Conventions de Code

### Langue

- **Pipeline de données** (dags/, utils/) : Commentaires et docstrings en **français**
- **Code ML** (training/, model) : Commentaires en **anglais**
- **Variables et noms de fonctions** : Toujours en **anglais**

### Accès Base de Données

- SQLAlchemy `create_engine()` + `text()` pour toutes les requêtes
- Pas d'ORM : requêtes SQL brutes parametrées (`text("... :param ...")`)
- Pandas `read_sql()` / `to_sql()` pour les lectures/écritures bulk
- Transactions via `engine.begin()` (context manager)

### Données Spatiales

- GeoPandas (`GeoDataFrame`) pour toutes les opérations géospatiales
- Shapely pour les géométries (LineString, Polygon, interpolate)
- pyproj pour les reprojections (EPSG:2154 ↔ EPSG:4326)
- H3 v4 avec fallbacks v3 (`grid_disk`/`k_ring`, `cell_to_boundary`/`h3_to_geo_boundary`)

### ML / PyTorch

- PyTorch Geometric `Data` objects pour le batching de graphes
- `StandardScaler` (scikit-learn) pour la normalisation des vitesses
- Dénormalisation on-the-fly sur GPU pour le calcul de la loss en km/h
- `torch.set_num_threads(1)` pour éviter le CPU thrashing

### Configuration

- Tout par `os.getenv()` avec valeurs par défaut
- Pas de fichiers de config (YAML, TOML, etc.) pour les paramètres runtime
- `.env` pour le développement local (jamais committé)

## Accès aux Services (Local)

| Service | URL | Credentials par défaut |
|---------|-----|------------------------|
| Streamlit | http://localhost:8501 | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Optuna Dashboard | http://localhost:8085 | — |
| Ray Dashboard | http://localhost:8265 | — |
| PostgreSQL | localhost:5432 | lyonflow / lyonflow_password |
| ngrok Inspector | http://localhost:4040 | — |
