# Infrastructure & Déploiement

## Docker Compose (Développement Local)

### Images Personnalisées

| Image | Dockerfile | Base | Usage |
|-------|-----------|------|-------|
| `lyonflow-app` | `Dockerfile` | `apache/airflow:2.9.1-python3.12` | Airflow, Streamlit, scripts de transformation |
| `lyonflow-ray` | `Dockerfile.ray` | `rayproject/ray:2.35.0-py312` | Ray head/worker, entraînement GPU (CUDA 12.1) |

### Limites de Ressources

| Service | CPU | RAM | GPU |
|---------|-----|-----|-----|
| PostgreSQL | 0.50 | 2 Go | — |
| MLflow | 1.00 | 2 Go | — |
| Ray Head | 1.00 | 3 Go | — |
| Ray Worker | 1.00 | **9 Go** | 1× NVIDIA (driver) |
| Airflow Webserver | 0.50 | 1.5 Go | — |
| Airflow Scheduler | 0.50 | 1.5 Go | — |
| Streamlit | 0.25 | 1 Go | — |
| ngrok | 0.10 | 256 Mo | — |

### Volumes

| Volume | Contenu |
|--------|---------|
| `postgres-data` | Données PostgreSQL persistantes |
| `mlflow-artifacts` | Artéfacts MLflow (modèles, plots) |

### Réseau Interne

Tous les services communiquent via le réseau Docker par défaut. Noms DNS internes :
- `postgres` (port 5432)
- `mlflow` (port 5000)
- `ray-head` (ports 6379, 10001, 8265)

## Initialisation de la Base de Données

Le fichier `init-db.sql` crée deux bases supplémentaires au démarrage :
- `airflow` — métadonnées d'Airflow
- `mlflow` — backend store MLflow

Les schémas `bronze`, `silver`, `gold` sont créés dynamiquement par le DAG.

## Kubernetes (Staging / Production)

### Manifests disponibles (`kubernetes/`)

| Fichier | Ressource |
|---------|-----------|
| `secrets-template.yaml` | Secrets K8s (credentials DB, API, MLflow) |
| `postgres-deployment.yaml` | Deployment + Service PostgreSQL |
| `mlflow-deployment.yaml` | Deployment + Service MLflow |
| `ray-cluster-deployment.yaml` | Ray Head + Worker Deployments |
| `airflow-deployment.yaml` | Airflow Webserver + Scheduler |
| `streamlit-deployment.yaml` | Deployment + Service Streamlit |

### Ordre de Déploiement

```
1. secrets-template.yaml
2. postgres-deployment.yaml
3. mlflow-deployment.yaml
4. ray-cluster-deployment.yaml
5. airflow-deployment.yaml
6. streamlit-deployment.yaml
```

## CI/CD (GitHub Actions)

### Pipeline CI (`ci.yml`)

Déclenché sur push/PR vers `main` ou `master`.

```
lint (ruff) ──────────┐
                      ├──▶ docker-build ──▶ security-scan (Trivy)
tests (pytest) ───────┘
typecheck (mypy) ─── (non-bloquant, indépendant)
```

| Job | Description | Bloquant |
|-----|-------------|----------|
| `lint` | ruff check + ruff format | Oui |
| `typecheck` | mypy (continue-on-error) | Non |
| `tests` | pytest + upload JUnit XML | Oui |
| `docker-build` | Build des 2 images (cache GHA) | Oui |
| `security-scan` | Trivy CRITICAL/HIGH (exit-code: 0) | Non |

### Pipeline CD (`cd.yml`)

Déclenché sur push vers `main` uniquement.

```
build-push (GHCR) ──▶ auto-tag (semver) ──▶ deploy-staging (K8s) ──▶ smoke-test
```

| Job | Description |
|-----|-------------|
| `build-push` | Build + push vers GitHub Container Registry |
| `tag` | Auto-bump version sémantique (patch par défaut) |
| `deploy-staging` | Apply manifests K8s + update image tags |
| `smoke-test` | Healthcheck PostgreSQL, Airflow, Ray + vérification DAG |

### Pipeline ML Training (`ml-training.yml`)

Déclenché manuellement (`workflow_dispatch`) ou chaque dimanche à 3h (`cron: 0 3 * * 0`).

```
test-model ──▶ export-data ──▶ hpo (conditionnel) ──▶ train-champion ──▶ promote
```

| Job | Runner | Description |
|-----|--------|-------------|
| `test-model` | ubuntu-latest | Tests unitaires STGCN |
| `export-data` | ubuntu-latest | Export Gold → CSV + upload artifact |
| `hpo` | self-hosted, gpu | Optuna HPO (sauf si `skip_hpo=true`) |
| `train-champion` | self-hosted, gpu | Entraînement avec meilleurs hyperparamètres |
| `promote` | ubuntu-latest | Évaluation MAE → GitHub Release si seuil respecté |

**Paramètres manuels** :
- `n_trials` : Nombre de trials HPO (défaut: 20)
- `mae_threshold` : Seuil MAE pour promotion (défaut: 5.0 km/h)
- `skip_hpo` : Sauter l'optimisation et utiliser les params existants

## Variables d'Environnement

### Credentials & Connexions

| Variable | Défaut | Contexte |
|----------|--------|----------|
| `API_LOGIN` | — | Identifiant API Grand Lyon |
| `API_PASSWORD` | — | Mot de passe API Grand Lyon |
| `POSTGRES_USER` | `lyonflow` | Utilisateur PostgreSQL |
| `POSTGRES_PASSWORD` | `lyonflow_password` | Mot de passe PostgreSQL |
| `POSTGRES_HOST` | `postgres` (Docker) | Hôte PostgreSQL |
| `POSTGRES_PORT` | `5432` | Port PostgreSQL |
| `POSTGRES_DB` | `lyonflow` | Nom de la base |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Serveur MLflow |
| `NGROK_AUTHTOKEN` | — | Token pour tunnel ngrok |
| `AIRFLOW_FERNET_KEY` | (généré) | Clé de chiffrement Airflow |

### Configuration ML

| Variable | Défaut | Description |
|----------|--------|-------------|
| `USE_LOCAL_CSV` | `false` | `true` = lire CSV au lieu de PostgreSQL |
| `DATA_FOLDER` | `/home/ray/project/data/in` | Répertoire des CSV |
| `SEQ_LEN` | `120` | Longueur séquence d'entrée (120 × 5min = 10h) |
| `HORIZONS` | `1` | Horizons de prédiction (ex: `"6,12,36"`) |
| `BATCH_SIZE` | `2` | Taille de batch (contraint par VRAM) |
| `HIDDEN_CHANNELS` | `128` | Dimension cachée GRU/GCN |
| `LEARNING_RATE` | `0.001` | Taux d'apprentissage Adam |
| `WEIGHT_DECAY` | `1e-5` | Régularisation L2 |
| `EPOCHS` | `100` | Nombre max d'epochs |
| `WEIGHT_JAM` | `15.0` | Poids perte < 10 km/h |
| `WEIGHT_SLOW` | `5.0` | Poids perte 10-30 km/h |
| `WEIGHT_NORMAL` | `1.0` | Poids perte > 30 km/h |
| `LYON_DEFAULT_SPEED` | `30.0` | Vitesse d'imputation par défaut (km/h) |
