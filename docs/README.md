# Documentation — LyonFlow

Documentation technique de la plateforme MLOps **LyonFlow** : prédiction spatio-temporelle du trafic routier de la Métropole de Lyon.

> **Vue d'ensemble** : pipeline de données médaillon (Bronze/Silver/Gold) + modèle GNN (`SpatioTemporalGCN`) entraîné via Optuna/Ray et servi via Airflow/Streamlit.

## 🗺️ Par où commencer ?

| Vous êtes… | Commencez par… |
|---|---|
| 🆕 Nouveau sur le projet | [Architecture globale](architecture.md) → [Démarrage rapide](development.md#installation-locale) |
| 🛠️ Développeur / contributeur | [Guide de développement](development.md) → [Conventions](development.md#conventions-de-code) |
| 🚀 Ops / SRE | [Infrastructure & Déploiement](infrastructure.md) → [Runbook opérationnel](operational-runbook.md) |
| 🧠 Data scientist | [Modèle ML](model.md) → [Référence API](api-reference.md) |
| 🐛 Vous avez un bug | [Runbook opérationnel](operational-runbook.md) → [Dette technique](tech-debt.md) |

## 📚 Sommaire

### Concepts

| Document | Description |
|----------|-------------|
| [Architecture Globale](architecture.md) | Diagrammes, architecture médaillon, services, flux de données |
| [Pipeline de Données](data-pipeline.md) | Schémas SQL, transformations spatiales Bronze→Silver→Gold |
| [Modèle ML (ST-GRU-GNN)](model.md) | Architecture du réseau, features, loss personnalisée, multi-horizon |

### Deep Dives

| Document | Description |
|----------|-------------|
| [Ray Cluster](ray-cluster.md) | Architecture head/worker, config Docker/K8s, modes d'utilisation, Jobs API, monitoring |
| [Optuna HPO](optuna-hpo.md) | Optimisation bayésienne TPE, pruning, stockage PostgreSQL, dashboard, get_best_params |
| [Entraînement](training.md) | Pipeline complet, staircase loss, early stopping, analyse stratifiée, MLflow |
| [Inférence & Prédiction](inference.md) | Prédiction temps réel, backfill historique, table de sortie, benchmarks |

### Mise en oeuvre

| Document | Description |
|----------|-------------|
| [Infrastructure & Déploiement](infrastructure.md) | Docker Compose, Kubernetes, CI/CD GitHub Actions, variables d'env |
| [Guide de Développement](development.md) | Setup local, conventions, tests, linting, structure du projet |

### Exploitation

| Document | Description |
|----------|-------------|
| [Dette Technique & Roadmap](tech-debt.md) | Problèmes connus, améliorations planifiées |

## 🏗️ Architecture en 30 secondes

```
API WFS Grand Lyon
    ↓ (toutes les 5 min, Airflow)
bronze.trafic_vitesse_brute     →  Payload JSONB brut
    ↓
silver.trafic_vitesse_propre    →  WGS84 + H3 rés.13 + catégorisation
    ↓
gold.{dim_spatial_grid_mapping, dim_gnn_adjacency, fact_traffic_series}
    ↓ (toutes les 5 min, Airflow → Ray)
SpatioTemporalGCN (GRU + GCN + Skip Connections)
    ↓
Streamlit Dashboard + PostgreSQL fact_predictions_traffic
```

Pour le détail : [architecture.md](architecture.md).

## ⚡ Démarrage rapide (TL;DR)

```bash
# 1. Cloner et configurer
git clone <repo>
cd FinalProjet
cp .env.example .env
# Éditer .env : API_LOGIN, API_PASSWORD, NGROK_AUTHTOKEN, AIRFLOW_FERNET_KEY

# 2. Lancer la stack
docker-compose up -d --build

# 3. Vérifier que tout est UP
docker ps --format "table {{.Names}}\t{{.Status}}"

# 4. Accéder aux services
#    Streamlit       http://localhost:8501
#    Airflow         http://localhost:8080  (admin / admin)
#    MLflow          http://localhost:5000
#    Ray Dashboard   http://localhost:8265
```

Le DAG `lyonflow_traffic_pipeline` s'exécute automatiquement toutes les 5 minutes. Voir [development.md](development.md#installation-locale) pour le détail.

## 🔧 Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| **Stockage** | PostgreSQL 15 (médaillon) | Bronze / Silver / Gold + métadonnées Airflow/MLflow/Optuna |
| **Orchestration** | Apache Airflow 2.9 | DAG toutes les 5 min (Bronze→Silver→Gold→CSV→Prédiction) |
| **Tracking ML** | MLflow | Expériences, métriques, artéfacts (`.pt`, scaler, plots) |
| **Calcul distribué** | Ray 2.35 | Cluster head + worker GPU, HPO parallèle, inférence |
| **Optimisation** | Optuna | TPE bayésien + MedianPruner, 20 trials en ~8 min |
| **Modèle** | PyTorch Geometric | ST-GRU-GNN (GRU + 2× GCNConv + Skip Connections) |
| **Géospatial** | GeoPandas, Shapely, H3, pyproj | EPSG:2154→4326, H3 rés.13, polygones |
| **Visualisation** | Streamlit + Plotly | Dashboard 3 onglets (courbes, analyse erreur, carte) |
| **Tunnel** | ngrok | Accès distant à PostgreSQL (Colab, debug externe) |

## 📂 Structure du dépôt

Voir [development.md § Structure du Projet](development.md#structure-du-projet) pour l'arbre complet.

```
FinalProjet/
├── app.py                 # Dashboard Streamlit
├── dags/                  # DAG Airflow (Bronze→Silver→Gold)
├── training/stgcn/        # Modèle, dataset, train, HPO, predict
├── utils/                 # Outils ops + scripts de maintenance DB
├── tests/                 # Tests unitaires pytest + benchmarks cache
├── kubernetes/            # Manifests K8s (deployments + secrets)
├── docs/                  # 📍 Vous êtes ici
├── docker-compose.yml     # Stack locale 8 services
├── Dockerfile             # Image app (Airflow/Streamlit/scripts)
├── Dockerfile.ray         # Image Ray (GPU + PyTorch + PyG)
└── init-db.sql            # Init bases airflow + mlflow
```

## 📝 Conventions du projet

| Aspect | Convention |
|---|---|
| **Langue des commentaires** | Pipeline de données (`dags/`, `utils/`) en **français** · Code ML (`training/`, `model`) en **anglais** |
| **Identifiants & fonctions** | Toujours en **anglais** |
| **Python** | 3.12+, type hints encouragés, `ruff` + `mypy` (non-bloquant) |
| **Tests** | `pytest` + `unittest.mock` (mocks explicites, pas de réseau) |
| **BDD** | SQLAlchemy `create_engine` + `text()` + `read_sql` / `to_sql` |
| **Géospatial** | GeoPandas + Shapely + pyproj, H3 v4 avec fallbacks v3 |
| **Configuration** | `os.getenv()` partout, pas de fichiers de config runtime |
| **Secrets** | `.env` (local) / `kubernetes/secrets.yaml` (prod), jamais commités |

Plus de détails : [development.md § Conventions](development.md#conventions-de-code).

## 🤝 Contribution

1. Créer une branche depuis `main` : `git checkout -b feat/ma-feature`
2. Implémenter + tester : `pytest tests/ -v`
3. Vérifier le linting : `ruff check . && ruff format --check .`
4. Ouvrir une Pull Request (CI automatique : lint + tests + Docker build + Trivy scan)

## 📊 État du projet

| Aspect | État |
|---|---|
| Couverture tests | ~8 fichiers pytest, focus sur ingest, transform, modèle, migration |
| CI | ✅ Lint + tests + Docker build + Trivy scan |
| CD | ✅ Auto-build images vers GHCR + déploiement K8s staging |
| HPO automatisé | ✅ Workflow `ml-training.yml` dimanche 3h + manuel |
| Documentation | ✅ Ce dossier |
| Dette technique connue | Voir [tech-debt.md](tech-debt.md) |

---

_Documentation maintenue manuellement. Pour les questions, ouvrir une issue GitHub._
