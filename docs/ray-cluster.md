# Ray Cluster — Configuration & Fonctionnement

## Pourquoi Ray ?

L'entraînement d'un GNN spatio-temporel sur 1 520 nœuds avec des séquences de 120 pas de temps est extrêmement coûteux en mémoire et en calcul. Ray résout trois problèmes :

1. **Contrôle strict des ressources** : Bridage CPU/GPU via Docker pour éviter surchauffe et blocage de la machine hôte
2. **Parallélisme HPO** : Exécution de plusieurs trials Optuna simultanément sur les ressources partagées
3. **Portabilité** : Le même code s'exécute en local (Docker Compose) ou sur un cluster Kubernetes multi-nœuds sans modification

## Architecture du Cluster

```
┌──────────────────────────────────────────────────────────────┐
│                    Réseau Docker interne                      │
│                                                              │
│  ┌─────────────────────┐      ┌────────────────────────────┐ │
│  │   RAY HEAD           │      │   RAY WORKER               │ │
│  │   lyonflow-ray-head  │◄────▶│   lyonflow-ray-worker      │ │
│  │                      │      │                            │ │
│  │   Rôle :             │      │   Rôle :                   │ │
│  │   - GCS (port 6379)  │      │   - Exécution des tasks    │ │
│  │   - Dashboard (8265) │      │   - Entraînement GPU       │ │
│  │   - Client API(10001)│      │   - HPO trials             │ │
│  │   - Scheduling       │      │   - Inférence              │ │
│  │                      │      │                            │ │
│  │   CPU: 1 core        │      │   CPU: 2 cores             │ │
│  │   RAM: 3 Go          │      │   RAM: 9 Go                │ │
│  │   GPU: aucun         │      │   GPU: 1× NVIDIA           │ │
│  └─────────────────────┘      └────────────────────────────┘ │
│           │                              │                    │
│           ▼                              ▼                    │
│  ┌─────────────────────┐      ┌────────────────────────────┐ │
│  │   PostgreSQL         │      │   MLflow                   │ │
│  │   (données + Optuna) │      │   (tracking expériences)   │ │
│  └─────────────────────┘      └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Configuration Docker Compose

### Ray Head (Coordinateur)

```yaml
ray-head:
  image: lyonflow-ray:latest          # Image custom (Dockerfile.ray)
  container_name: lyonflow-ray-head
  command: ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=1 --block
  ports:
    - "8265:8265"   # Dashboard web
    - "10001:10001"  # Ray Client Protocol (soumission de jobs distante)
    - "6379:6379"    # GCS (Global Control Store) — registre interne du cluster
  deploy:
    resources:
      limits:
        cpus: '1.00'
        memory: 3G
```

Le Head **ne fait pas de calcul**. Il coordonne :
- **GCS (port 6379)** : Registre central — stocke l'état des nœuds, la file d'attente des tâches, les références d'objets partagés
- **Dashboard (port 8265)** : Interface web monitoring (CPU, RAM, GPU, jobs, logs)
- **Client (port 10001)** : Point d'entrée pour `ray://ray-head:10001` (soumission depuis Airflow ou CLI)

### Ray Worker (Exécutant)

```yaml
ray-worker:
  image: lyonflow-ray:latest
  container_name: lyonflow-ray-worker
  command: ray start --address=ray-head:6379 --num-cpus=2 --block
  deploy:
    resources:
      limits:
        cpus: '1.00'
        memory: 9G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Le Worker **fait tout le calcul lourd** :
- **9 Go RAM** : Nécessaire pour charger le graphe complet (~1 520 nœuds × 120 pas × 5 features) + modèle + gradients
- **GPU NVIDIA** : Réservé via `nvidia-container-runtime`. Accéléré CUDA 12.1
- **`--num-cpus=2`** : Ray voit 2 CPU logiques pour le scheduling, mais Docker bride à 1 core physique (sécurité thermique)
- **`--block`** : Le processus Ray reste au premier plan (le conteneur ne s'arrête pas)

### Pourquoi `--num-cpus` Diffère du Limit Docker

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Docker `cpus: '1.00'` | 1 core physique | **Limite réelle** — protection thermique de la machine hôte |
| Ray `--num-cpus=2` | 2 CPU logiques | **Budget de scheduling** — Ray planifie jusqu'à 2 tâches CPU en parallèle |

En pratique, Ray alloue 2 tâches qui se partagent 1 core physique via time-slicing. C'est intentionnel : les tâches sont majoritairement GPU-bound, pas CPU-bound.

## Image Docker Ray (`Dockerfile.ray`)

```dockerfile
FROM rayproject/ray:2.35.0-py312

# Dépendances système pour PostgreSQL
RUN apt-get update && apt-get install -y build-essential libpq-dev

# Dépendances Python
RUN pip install psycopg2-binary scikit-learn mlflow matplotlib optuna optuna-dashboard

# PyTorch avec CUDA 12.1
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
RUN pip install torch-geometric
```

**Base** : `rayproject/ray:2.35.0-py312` — Python 3.12, Ray 2.35.0 préinstallé.

**CUDA 12.1** : Compatible avec les drivers hôtes CUDA 12.1+ (testé avec CUDA 12.8 sur la machine de développement).

**Volume monté** : Le projet entier est monté dans `/home/ray/project` via `docker-compose.yml`, permettant d'exécuter directement les scripts d'entraînement sans rebuild d'image.

## Modes d'Utilisation

### 1. Exécution Directe (docker exec)

Commande la plus simple — exécuter un script Python dans le Worker :

```bash
# Entraînement
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/train_stgcn.py

# HPO
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/hpo_stgcn.py

# Inférence
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/predict_stgcn.py

# Backfill historique
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/backfill_predictions.py
```

Le script s'exécute dans le Worker, détecte automatiquement le GPU via `torch.cuda.is_available()`, et se connecte aux services internes (PostgreSQL, MLflow) via le réseau Docker.

### 2. Soumission via Ray Jobs API (Airflow)

Airflow soumet les jobs via l'API REST du dashboard Ray (port 8265) :

```python
# Extrait de dag_pipeline.py — trigger_stgcn_prediction_on_ray()
payload = {
    "entrypoint": "cd /home/ray/project && python training/stgcn/predict_stgcn.py",
    "runtime_env": {
        "env_vars": {
            "USE_LOCAL_CSV": "false",
            "DATA_FOLDER": "/home/ray/project/data/in",
            "MODEL_PATH": "/home/ray/project/models/stgcn_prod_latest.pt",
            "SCALER_PATH": "/home/ray/project/models/stgcn_scaler.pkl",
            "SEQ_LEN": "120",
            "HORIZONS": "6,12,36",
            "POSTGRES_HOST": "postgres",
            ...
        }
    },
}

# POST http://ray-head:8265/api/jobs/
response = requests.post(submit_url, json=payload, timeout=30)
job_id = response.json()["job_id"]

# Polling du statut jusqu'à SUCCEEDED ou FAILED
while True:
    status = requests.get(f"{ray_dashboard_url}/api/jobs/{job_id}").json()["status"]
    if status == "SUCCEEDED": break
    if status in ["FAILED", "STOPPED"]: raise Exception(...)
    time.sleep(10)
```

**Flux** :
1. Airflow POST le payload JSON sur `http://ray-head:8265/api/jobs/`
2. Ray Head dispatche le job vers un Worker disponible
3. Airflow poll le statut toutes les 10 secondes
4. En cas d'échec, Airflow récupère les logs via `/api/jobs/{job_id}/logs`

### 3. Ray Client Protocol (depuis un notebook Colab)

Pour soumettre des jobs depuis l'extérieur (ex: Google Colab via ngrok) :

```python
import ray
ray.init("ray://localhost:10001")  # ou ray://<ngrok-tcp-url>:10001
```

C'est le cas d'usage du tunnel ngrok : exposer le port 10001 pour que des notebooks distants puissent soumettre des tâches au cluster.

## Configuration Kubernetes

### Manifeste K8s (`kubernetes/ray-cluster-deployment.yaml`)

En K8s, le cluster Ray est déployé avec deux Deployments séparés :

**Ray Head** :
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ray-head
spec:
  replicas: 1                      # Toujours 1 seul Head
  containers:
    - name: ray-head
      image: rayproject/ray:2.35.0-py312
      command: ["sh", "-c"]
      args: ["ray start --head --port=6379 --dashboard-host=0.0.0.0 --block"]
      ports:
        - containerPort: 8265      # Dashboard
        - containerPort: 10001     # Client
        - containerPort: 6379      # GCS
```

**Ray Worker** :
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ray-worker
spec:
  replicas: 1                      # Scalable horizontalement
  containers:
    - name: ray-worker
      image: rayproject/ray:2.35.0-py312
      command: ["sh", "-c"]
      args: ["ray start --address=ray-head:6379 --block"]
```

**Service** exposant les 3 ports du Head :
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ray-head
spec:
  ports:
    - name: dashboard  → 8265
    - name: client     → 10001
    - name: gcs        → 6379
  selector:
    app: ray-head
```

### Scaling en Production

Pour ajouter des Workers GPU en K8s :

```bash
kubectl scale deployment ray-worker --replicas=3
```

Les Workers supplémentaires se connectent automatiquement au Head via `ray start --address=ray-head:6379`. Optuna distribue les trials HPO sur les Workers disponibles via le stockage partagé PostgreSQL.

## Dashboard Ray

Accessible sur `http://localhost:8265` (local) ou via le Service K8s.

### Informations Disponibles

| Section | Contenu |
|---------|---------|
| **Cluster** | Nœuds actifs, CPU/GPU disponibles et utilisés, mémoire |
| **Jobs** | Liste des jobs soumis, statut, logs, durée |
| **Actors** | Acteurs Ray en cours (utilisé par Optuna internalement) |
| **Logs** | Logs centralisés de tous les Workers |
| **Metrics** | Graphiques temps réel CPU, RAM, GPU utilization |

### Monitoring d'un Entraînement

Pendant un `train_stgcn.py` ou `hpo_stgcn.py`, le dashboard montre :
- **GPU Utilization** : Devrait être >80% pendant les passes forward/backward
- **GPU Memory** : ~3.5 Go sur 4 Go (NVIDIA T600) avec batch_size=2 et seq_len=120
- **CPU** : Faible (data loading, logging) — le calcul est sur GPU

## Interactions avec les Autres Services

### Ray ↔ PostgreSQL

- **Entraînement/Inférence** : `train_stgcn.py` et `predict_stgcn.py` se connectent directement à `postgres:5432` via SQLAlchemy pour charger les données Gold
- **Optuna Storage** : `hpo_stgcn.py` utilise `optuna.storages.RDBStorage(url=...)` pointant vers PostgreSQL — cela permet la coordination multi-worker et la reprise après crash
- **Résultat prédictions** : `predict_stgcn.py` écrit dans `gold.fact_predictions_traffic`

### Ray ↔ MLflow

- Tous les scripts d'entraînement se connectent à `http://mlflow:5000`
- Métriques loguées par epoch : `train_loss_std`, `test_mae_kmh`
- Artéfacts sauvegardés : poids modèle (`.pt`), scaler (`.pkl`), plots d'analyse (`.png`, `.csv`)
- HPO : meilleurs hyperparamètres + `best_mae_kmh`

### Ray ↔ Airflow

- Airflow soumet les jobs via HTTP REST (`POST /api/jobs/`)
- Le runtime_env spécifie les variables d'environnement (chemins modèle, horizons, DB)
- Airflow poll le statut et récupère les logs en cas d'erreur
- Pas de dépendance Python directe — communication pure HTTP

## Gestion des Erreurs

### Timeout d'Entraînement

Ray n'impose pas de timeout par défaut. Pour un trial HPO qui diverge :
- **Optuna MedianPruner** : Coupe les trials sous-performants après 3 epochs de warmup
- **Early stopping** (train_stgcn.py) : Arrête après 10 epochs sans amélioration

### Worker Crash

- Docker `restart: always` relance automatiquement le conteneur
- Le Worker se reconnecte au Head via `--address=ray-head:6379`
- Optuna reprend l'étude depuis PostgreSQL (`load_if_exists=True`)

### GPU Out of Memory

- `batch_size=2` dimensionné pour 4 Go VRAM (NVIDIA T600)
- `torch.set_num_threads(1)` réduit la consommation mémoire CPU
- `get_best_params.py` plafonne `batch_size` à 16 max pour les modèles champions
- Si OOM : réduire `BATCH_SIZE` ou `SEQ_LEN`, ou augmenter la RAM Worker

## Variables d'Environnement Spécifiques à Ray

| Variable | Défaut | Effet |
|----------|--------|-------|
| `RAY_ADDRESS` | `ray://ray-head:10001` | Adresse du cluster pour les clients Airflow |
| `USE_LOCAL_CSV` | `false` | `true` = charger depuis CSV au lieu de PostgreSQL (mode déconnecté) |
| `DATA_FOLDER` | `/home/ray/project/data/in` | Répertoire des CSV (si `USE_LOCAL_CSV=true`) |
| `DATA_FOLDER_OUT` | `/home/ray/project/data/out` | Répertoire de sortie des prédictions CSV |
| `MODEL_PATH` | `models/stgcn_prod_latest.pt` | Chemin du modèle entraîné |
| `SCALER_PATH` | `models/stgcn_scaler.pkl` | Chemin du StandardScaler fitté |
