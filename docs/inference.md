# Inférence & Prédiction

## Vue d'Ensemble

Deux modes d'inférence existent dans LyonFlow :

| Script | Usage | Déclencheur |
|--------|-------|-------------|
| `predict_stgcn.py` | Prédiction en temps réel sur le dernier snapshot | DAG Airflow (toutes les 5 min) |
| `backfill_predictions.py` | Rattrapage historique sur tous les timestamps manquants | Exécution manuelle |

## Prédiction Temps Réel (`predict_stgcn.py`)

### Pipeline

```
1. Charger données (CSV ou PostgreSQL)
   ├── Topologie : node_mapping + adjacency + self-loops
   └── Séries : 120 derniers pas de temps (10h d'historique)
                  │
2. Vérifier historique suffisant (≥ SEQ_LEN timestamps)
                  │
3. Charger StandardScaler (models/stgcn_scaler.pkl)
   └── Fallback : fit temporaire sur l'échantillon courant (sous-optimal)
                  │
4. Construire tenseur PyG [N, 120, 5]
   └── 5 canaux : speed_norm, hour_sin, hour_cos, day_sin, day_cos
                  │
5. Charger modèle (models/stgcn_prod_latest.pt) → model.eval()
                  │
6. Inférence (torch.no_grad)
   └── Sortie : [N, len(HORIZONS)] valeurs normalisées
                  │
7. Dénormalisation (scaler) + clip [1, 130] km/h
                  │
8. Construire enregistrements (node × horizon)
   └── prediction_timestamp, target_timestamp, horizon_minutes,
       node_idx, properties_twgid, predicted_speed, real_speed, geometry_wgs84_wkt
                  │
9. Export CSV → data/out/predictions_traffic.csv
                  │
10. Insert PostgreSQL → gold.fact_predictions_traffic (si mode DB)
```

### Table de Sortie : `gold.fact_predictions_traffic`

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | SERIAL PK | Auto-incrémenté |
| `prediction_timestamp` | TIMESTAMP | t₀ — instant d'origine de la prédiction |
| `target_timestamp` | TIMESTAMP | t₀ + horizon — instant cible prédit |
| `horizon_minutes` | INTEGER | Horizon en minutes (30, 60, 180) |
| `node_idx` | INTEGER | Index du nœud dans le graphe |
| `properties_twgid` | INTEGER | Identifiant technique du segment routier |
| `predicted_speed` | REAL | Vitesse prédite (km/h), clippée [1, 130] |
| `real_speed` | REAL | Vitesse réelle à t₀ (comparaison immédiate) |
| `geometry_wgs84_wkt` | TEXT | LineString WGS84 du segment (pour cartographie) |
| `created_at` | TIMESTAMP | Horodatage d'insertion |

**Index** :
- `(prediction_timestamp, horizon_minutes)` — requêtes "dernière prédiction par horizon"
- `(properties_twgid)` — jointures avec `silver.ref_segments`

### Volume de Données Généré

Par cycle de prédiction :
- **1 520 nœuds × 3 horizons = 4 560 enregistrements** par exécution
- Toutes les 5 minutes → **~1 314 720 enregistrements/jour**

### Mode Hors-Ligne (CSV)

Quand `USE_LOCAL_CSV=true` :
- Lit `node_mapping.csv`, `edges.csv`, `traffic_series.csv` depuis `DATA_FOLDER`
- N'écrit **pas** dans PostgreSQL
- Utile pour tester sur Google Colab ou en CI sans base de données

### Sécurité des Prédictions

- **Clip [1, 130]** : Aucune vitesse négative ni irréaliste (>130 km/h)
- **Scaler fallback** : Si le fichier `.pkl` n'existe pas, fit temporaire sur l'échantillon courant — produit des prédictions moins précises mais fonctionnelles
- **Modèle absent** : `sys.exit(1)` — pas d'inférence sans poids entraînés

## Backfill Historique (`backfill_predictions.py`)

### Objectif

Combler rétroactivement toutes les prédictions manquantes dans `gold.fact_predictions_traffic` pour les timestamps qui existent dans `gold.fact_traffic_series` mais n'ont pas encore de prédiction.

### Pipeline

```
1. Charger topologie complète depuis PostgreSQL (mapping + adjacency)
                  │
2. Charger TOUT l'historique de gold.fact_traffic_series
   └── Pivoter en matrice [T × N], remplacer NaN par LYON_DEFAULT_SPEED
                  │
3. Identifier timestamps éligibles :
   ├── Présent dans fact_traffic_series
   ├── Absent de fact_predictions_traffic
   └── A au moins SEQ_LEN pas d'historique avant lui
                  │
4. Pour chaque timestamp éligible :
   ├── Extraire fenêtre glissante [t-119 : t] (120 pas)
   ├── Calculer features temporelles cycliques
   ├── Normaliser avec scaler
   ├── Construire tenseur PyG
   ├── Inférence → dénormaliser → clip
   └── Accumuler les enregistrements
                  │
5. Insertion massive en base (chunksize=10000)
```

### Optimisations

| Technique | Gain |
|-----------|------|
| Pré-extraction numpy des `node_twgids` et `node_geoms` | Évite `.iloc` lent dans la boucle interne |
| `torch.no_grad()` global | Un seul contexte pour toutes les inférences |
| `edge_index.to(device)` une seule fois | Pas de transfert CPU→GPU répété |
| Insertion bulk avec `chunksize=10000` | Moins de roundtrips PostgreSQL |
| Logs de progression toutes les 50 itérations | Monitoring sans overhead |

### Volume

Pour 1 000 timestamps manquants : 1 000 × 1 520 nœuds × 3 horizons = **4 560 000 enregistrements** insérés.

## Benchmarks (`test_perf.py`)

Script de micro-benchmark pour valider les performances :

| Test | Configuration | Ce qu'il mesure |
|------|---------------|-----------------|
| GNN forward pass | 1 536 nœuds, 9 000 arêtes, 20 itérations | Latence inférence GPU/CPU |
| Boucle non-optimisée | 10 timestamps × 1 536 nœuds × 3 horizons | Temps de construction des enregistrements |
| Boucle optimisée | Idem, sans `pd.isna()` | Gain de l'optimisation |

Exécution :
```bash
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/test_perf.py
```
