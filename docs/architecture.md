# Architecture Globale

## Vue d'ensemble

LyonFlow est une plateforme MLOps de bout en bout conçue pour ingérer, transformer, modéliser et prédire en temps réel la vitesse du trafic routier de la Métropole de Lyon. Elle combine un pipeline de données géospatiales avec un modèle de Deep Learning sur graphes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API Grand Lyon (WFS)                        │
│                    Flux temps réel toutes les 5 min                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION — Apache Airflow                                      │
│  DAG : lyonflow_traffic_pipeline (*/5 * * * *)                       │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ Ingestion│──▶│Transformation│──▶│   Gold   │──▶│Export CSV +  │  │
│  │ (Bronze) │   │   (Silver)   │   │Matérialis│   │Prédiction Ray│  │
│  └──────────┘   └──────────────┘   └──────────┘   └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STOCKAGE — PostgreSQL 15                                            │
│                                                                      │
│  bronze.trafic_vitesse_brute    (JSONB brut)                         │
│  silver.trafic_vitesse_propre   (nettoyé, H3, WGS84)                │
│  gold.dim_spatial_grid_mapping  (grille spatiale → node_idx)         │
│  gold.dim_gnn_adjacency        (arêtes du graphe routier)            │
│  gold.fact_traffic_series       (séries temporelles imputées)        │
└──────────────────────────────────────────────────────────────────────┘
         │                                              │
         ▼                                              ▼
┌────────────────────┐                    ┌────────────────────────────┐
│  TRACKING — MLflow │                    │  CALCUL — Ray Cluster      │
│  Expériences,      │                    │  Entraînement distribué,   │
│  métriques,        │                    │  HPO Optuna parallèle      │
│  artéfacts modèle  │                    └────────────────────────────┘
└────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INTERFACE — Streamlit Dashboard                                     │
│  Courbes d'apprentissage, analyse d'erreur stratifiée,               │
│  visualisation temps réel du réseau routier                          │
└──────────────────────────────────────────────────────────────────────┘
```

## Architecture Medallion (PostgreSQL)

Le stockage suit une architecture en trois couches (Medallion Architecture) dans une seule instance PostgreSQL 15 :

| Couche | Schéma | Table(s) | Contenu |
|--------|--------|----------|---------|
| **Bronze** | `bronze` | `trafic_vitesse_brute` | Payload JSONB brut de l'API WFS, horodaté |
| **Silver** | `silver` | `trafic_vitesse_propre` | Données nettoyées : reprojection EPSG:2154→4326, interpolation 7m, indexation H3 résolution 13, catégorisation des vitesses |
| **Gold** | `gold` | `dim_spatial_grid_mapping` | Correspondance capteur → node_idx + coordonnées grille (i, j) |
| | | `dim_gnn_adjacency` | Liste d'arêtes du graphe routier (rayon K=2 cellules H3) |
| | | `fact_traffic_series` | Vitesses par timestamp × node_idx, avec imputation (moyenne historique ou 30 km/h par défaut) |

## Services et Ports

| Service | Conteneur | Port(s) | Rôle |
|---------|-----------|---------|------|
| PostgreSQL 15 | `lyonflow-postgres` | 5432 | Stockage medallion + métadonnées Airflow/MLflow/Optuna |
| Apache Airflow (Webserver) | `lyonflow-airflow-webserver` | 8080 | Interface d'administration des pipelines |
| Apache Airflow (Scheduler) | `lyonflow-airflow-scheduler` | — | Exécution planifiée des DAGs |
| MLflow | `lyonflow-mlflow` | 5000 | Tracking des expériences ML, stockage artéfacts |
| Ray Head | `lyonflow-ray-head` | 8265 (dashboard), 10001 (client), 6379 (GCS) | Coordinateur du cluster de calcul |
| Ray Worker | `lyonflow-ray-worker` | — | Worker GPU (9 Go RAM, NVIDIA) |
| Streamlit | `lyonflow-streamlit` | 8501 | Dashboard trafic temps réel |
| ngrok | `ngrok_tunnel` | 4040 | Tunnel TCP pour accès distant à PostgreSQL |
| Optuna Dashboard | — | 8085 | Visualisation HPO (lancé manuellement) |

## Flux de Données Détaillé

### 1. Ingestion (Bronze)

```
API WFS Grand Lyon (EPSG:2154)
  → Requête HTTP GET avec authentification
  → Payload GeoJSON complet
  → INSERT dans bronze.trafic_vitesse_brute (fetched_at, raw_data JSONB)
```

### 2. Transformation (Silver)

```
Dernier enregistrement Bronze
  → Extraction des features GeoJSON
  → Construction GeoDataFrame (geopandas, CRS=EPSG:2154)
  → Interpolation des segments tous les 7 mètres (Shapely)
  → Reprojection vers EPSG:4326 (WGS84)
  → Indexation H3 résolution 13 (h3-py v4)
  → Fusion des cellules H3 en polygones (unary_union)
  → Normalisation des vitesses + catégorisation (Slow/Medium/Fast)
  → Imputation fallback (moyenne par capteur → 30 km/h global)
  → Export CSV + GeoJSON (backup fichier)
  → INSERT dans silver.trafic_vitesse_propre
```

### 3. Matérialisation Gold

```
silver.trafic_vitesse_propre
  → Identification des capteurs actifs (<90% NaN)
  → Construction grille spatiale (H3 → local IJ projection)
  → Construction matrice d'adjacence (voisinage K=2 via grid_disk)
  → TRUNCATE + INSERT dim_spatial_grid_mapping
  → TRUNCATE + INSERT dim_gnn_adjacency
  → Snapshot latest timestamp + imputation → INSERT fact_traffic_series
```

### 4. Prédiction

```
Gold CSV export → Soumission job Ray (predict_stgcn.py)
  → Chargement modèle (.pt) + scaler (.pkl)
  → Inférence multi-horizon (30min, 1h, 3h)
  → Résultats vers PostgreSQL / fichiers
```

## Cycle de Vie d'un Run d'Entraînement

```
1. Export Gold (PostgreSQL) → CSV plats (data/in/)
2. Lancement hpo_stgcn.py    → Optuna TPE × N trials
   └─ Cache chaud (topology_data, traffic_data) ← partagé entre trials
3. Récupération best_params  → get_params_from_optuna() puis get_params_from_mlflow()
4. Lancement train_stgcn.py  → Modèle final, best_model_state sauvegardé
5. Analyse stratifiée        → models/stratified_error_analysis.{csv,png}
6. Promotion                 → mlflow.log_artifact() (auto-promote si MAE < seuil)
```

## Observabilité et Monitoring

### Sources de Logs

| Source | Localisation | Rétention |
|---|---|---|
| **Airflow** | `logs/` (monté sur hôte) | Configurable (défaut Airflow : 30 jours) |
| **MLflow** | `mlflow-artifacts/` volume Docker | Illimitée tant que le volume existe |
| **Ray** | Dashboard web (port 8265) + stdout | Pendant la durée de vie du conteneur |
| **PostgreSQL** | Logs internes + `pg_stat_activity` | Illimitée |
| **Streamlit** | `streamlit.log` (stdout conteneur) | Taille du fichier |

### Métriques Trackées

- **MLflow** : `train_loss_std`, `test_mae_kmh` par epoch, hyperparamètres, artéfacts (`.pt`, scaler, plots, CSV)
- **Optuna** : `value` (= MAE), `params` par trial, pruner state
- **PostgreSQL** : taille des schémas, nombre de NaN, taux d'imputation (`utils/inspect_silver.py`)
- **Airflow** : durée d'exécution par tâche, statut de chaque DAG run

### Healthchecks Docker

| Service | Healthcheck | Sensibilité |
|---|---|---|
| `postgres` | `pg_isready` | 5s × 5 retries |
| `mlflow` | HTTP 200 sur `/` | ❌ Non configuré (TODO) |
| `airflow-webserver` | ❌ Non configuré (TODO) | — |
| `ray-head` | ❌ Non configuré (TODO) | — |
| `streamlit` | ❌ Non configuré (TODO) | — |

Voir [tech-debt.md](tech-debt.md#infrastructure) pour la roadmap d'amélioration.

## Sécurité

### Surface d'Attaque

| Élément | Exposition | Mitigation |
|---|---|---|
| **API Grand Lyon** | Réseau Internet | Auth HTTP Basic, credentials en `.env` (jamais commités) |
| **PostgreSQL** | Port 5432 exposé hôte + tunnel ngrok | Mot de passe + tunnel chiffré ngrok (TCP) |
| **MLflow** | Port 5000 exposé | `--disable-security-middleware` (dev only — à protéger en prod) |
| **Airflow** | Port 8080 exposé | Authentification web (admin / admin par défaut — **à changer**) |
| **Ray Dashboard** | Port 8265 exposé | Pas d'auth en mode dev (TODO : activer `auth-mode=token`) |
| **Streamlit** | Port 8501 exposé | Pas d'auth (intranet seulement) |

### Secrets

- **Dev** : fichier `.env` (gitignore'd), template `.env.example`
- **Prod (K8s)** : `kubernetes/secrets.yaml` (gitignore'd), template `kubernetes/secrets-template.yaml` à dupliquer
- **CI/CD** : GitHub Actions Secrets (`POSTGRES_*`, `MLFLOW_*`)

### Données Sensibles

- **Credentials API** : nécessaires à l'ingestion, rotation manuelle
- **Données de trafic** : publiques (open data Grand Lyon), pas de PII
- **Logs Airflow** : peuvent contenir des messages d'erreur verbeux — ne pas exposer publiquement

## Réseau et Communication Inter-Services

### Résolution DNS (Docker Compose)

Tous les services partagent le réseau par défaut. Les noms DNS internes sont :

```
postgres:5432        ← source de vérité (Bronze/Silver/Gold + métadonnées)
mlflow:5000          ← tracking (MLflow server)
ray-head:6379        ← GCS (Global Control Store)
ray-head:10001       ← Ray Client API
ray-head:8265        ← Ray Dashboard
postgres:5432 (via ngrok) ← tunnel TCP externe
```

### Appels Sortants

| Service | Destination | Protocole | Fréquence |
|---|---|---|---|
| `airflow-webserver` / `airflow-scheduler` | `https://data.grandlyon.com` | HTTPS | Toutes les 5 min (Bronze) |
| `airflow-scheduler` | `ray-head:8265` | HTTP | Toutes les 5 min (déclenche job Ray) |
| `ray-head` / `ray-worker` | `postgres:5432` | TCP | Continu (entraînement) |
| `ray-head` / `ray-worker` | `mlflow:5000` | HTTP | Continu (tracking) |
| `streamlit` | `postgres:5432` + `mlflow:5000` | TCP + HTTP | À chaque rafraîchissement UI |

## Conventions Architecturales

### Idempotence

Toutes les écritures DB sont conçues pour être **idempotentes** :

- **Bronze** : `INSERT` uniquement (pas de clé d'unicité, chaque fetch est unique par `fetched_at`)
- **Silver** : `INSERT ... if_exists="append"` (clé composite `transformed_at + id_rue`)
- **Gold dimensions** : `TRUNCATE` puis `INSERT` (recalcul complet à chaque exécution)
- **Gold faits** : `DELETE WHERE timestamp = :ts` puis `INSERT` (idempotent pour un même timestamp)
- **Predictions** : `INSERT ... if_exists="append"` (historisation)

### Fail-Fast

Toutes les couches lèvent une `Exception` explicite en cas d'erreur :

- API WFS KO → ingestion échoue, le DAG retry 2× avec backoff 30s
- Bronze vide → `materialize_gold_layer` retourne proprement sans erreur
- Silver vide → Gold retourne sans erreur (mais avec un warning)
- Scaler manquant → `predict_stgcn.py` sys.exit(1) explicite
- Modèle manquant → `sys.exit(1)` explicite

### Conventions de Nommage

| Contexte | Pattern | Exemple |
|---|---|---|
| Tables Bronze | `bronze.<source>_<type>` | `bronze.trafic_vitesse_brute` |
| Tables Silver | `silver.<source>_<qualifier>` | `silver.trafic_vitesse_propre`, `silver.ref_segments` |
| Tables Gold — dimensions | `gold.dim_<concept>` | `dim_spatial_grid_mapping`, `dim_gnn_adjacency` |
| Tables Gold — faits | `gold.fact_<concept>` | `fact_traffic_series`, `fact_predictions_traffic` |
| Métadonnées | `<source>_<column>` (préfixe plat) | `properties_twgid`, `properties_libelle` |
| Fichiers Airflow | `<DAG>_<task>` | `lyonflow_traffic_pipeline.ingest_grand_lyon_traffic` |
| Artéfacts MLflow | `<concept>.<ext>` | `stgcn_prod_latest.pt`, `stratified_error_analysis.png` |

