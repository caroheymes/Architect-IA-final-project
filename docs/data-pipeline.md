# Pipeline de Données

## Source : API Grand Lyon WFS

- **URL** : `https://data.grandlyon.com/geoserver/metropole-de-lyon/ows`
- **Type** : WFS 2.0.0 (Web Feature Service)
- **Couche** : `pvo_patrimoine_voirie.pvotrafic`
- **Format** : GeoJSON (application/json)
- **CRS source** : EPSG:2154 (Lambert-93)
- **Authentification** : HTTP Basic (`API_LOGIN` / `API_PASSWORD`)
- **Fréquence d'ingestion** : Toutes les 5 minutes (cron Airflow)

## Couche Bronze

### Table : `bronze.trafic_vitesse_brute`

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | SERIAL PK | Identifiant auto-incrémenté |
| `fetched_at` | TIMESTAMP WITH TIME ZONE | Horodatage de l'ingestion (Europe/Paris) |
| `raw_data` | JSONB | Payload GeoJSON complet de l'API |

**Index** : `idx_bronze_fetched_at` sur `fetched_at DESC`

### Logique

1. Requête GET vers l'API WFS avec timeout de 30s
2. Stockage intégral du payload JSON sans transformation
3. Pas de déduplication — chaque appel produit un enregistrement

## Couche Silver

### Table : `silver.trafic_vitesse_propre`

| Colonne | Type | Description |
|---------|------|-------------|
| `id_rue` | INT | Index de la rue dans le snapshot |
| `properties_twgid` | VARCHAR | Identifiant technique du segment (clé de jointure Gold) |
| `properties_gid` | VARCHAR | Identifiant GID Grand Lyon |
| `properties_libelle` | VARCHAR | Nom de la voie |
| `properties_sens` | VARCHAR | Sens de circulation |
| `properties_etat` | VARCHAR | État du capteur |
| `properties_vitesse` | FLOAT | Vitesse mesurée (km/h), imputée si manquante |
| `properties_last_update` | VARCHAR | Dernier update côté API |
| `properties_est_a_jour` | VARCHAR | Indicateur de fraîcheur |
| `speed_category` | VARCHAR | Slow (0-20) / Medium (20-50) / Fast (>50) |
| `speed_color_map` | VARCHAR | Couleur cartographique (red/orange/green/gray) |
| `geometry_wgs84_wkt` | TEXT | Géométrie LineString en WKT (EPSG:4326) |
| `points_json` | TEXT | Points interpolés tous les 7m (JSON array de [lon, lat]) |
| `hexes_json` | TEXT | Cellules H3 résolution 13 couvrant le segment (JSON array) |
| `merged_h3_geometry_json` | TEXT | Polygone fusionné des cellules H3 (GeoJSON) |
| `transformed_at` | TIMESTAMP WITH TIME ZONE | Horodatage de la transformation |

### Transformations Appliquées

1. **Filtrage** : Exclusion des capteurs marqués `est_a_jour = False`
2. **Géométrie** : Construction de `LineString` depuis les coordonnées brutes
3. **Interpolation spatiale** : Points générés tous les 7 mètres le long de chaque segment (Shapely `interpolate`)
4. **Reprojection** : EPSG:2154 (Lambert-93, métrique) → EPSG:4326 (WGS84, degrés) via `pyproj`
5. **Indexation H3** : Chaque point interpolé → cellule H3 résolution 13 (`h3.latlng_to_cell`)
6. **Fusion polygonale** : Cellules H3 d'un segment → polygone unique (`unary_union`)
7. **Imputation vitesse** : Si NaN → moyenne historique du capteur → fallback global (`LYON_DEFAULT_SPEED`, défaut 30 km/h)
8. **Catégorisation** : Slow (0-20 km/h), Medium (20-50), Fast (>50)

## Couche Gold

### Table : `gold.dim_spatial_grid_mapping`

| Colonne | Type | Description |
|---------|------|-------------|
| `node_idx` | INT | Index séquentiel du noeud pour le GNN (0 à N-1) |
| `properties_twgid` | VARCHAR PK | Identifiant technique du capteur |
| `matrix_i` | INT | Coordonnée i dans la grille relative H3 |
| `matrix_j` | INT | Coordonnée j dans la grille relative H3 |
| `h3_id` | VARCHAR(15) | Cellule H3 de référence du capteur |
| `updated_at` | TIMESTAMP WITH TIME ZONE | Dernier recalcul |

### Table : `gold.dim_gnn_adjacency`

| Colonne | Type | Description |
|---------|------|-------------|
| `node_u` | INT | Noeud source (PK composite) |
| `node_v` | INT | Noeud destination (PK composite) |
| `is_connected` | BOOLEAN | Toujours `true` |
| `updated_at` | TIMESTAMP WITH TIME ZONE | Dernier recalcul |

**Construction** : Deux capteurs sont adjacents si leurs cellules H3 sont à distance ≤ 2 (`grid_disk(cell, 2)`). Graphe non orienté (`min(u,v), max(u,v)`).

### Table : `gold.fact_traffic_series`

| Colonne | Type | Description |
|---------|------|-------------|
| `timestamp` | TIMESTAMP WITH TIME ZONE | Horodatage du snapshot (PK composite) |
| `node_idx` | INT | Index du noeud (PK composite) |
| `properties_vitesse` | FLOAT | Vitesse en km/h (imputée si nécessaire) |
| `imputed` | BOOLEAN | `true` si la valeur est une imputation |

**Garantie** : Chaque snapshot contient exactement N enregistrements (un par capteur actif), même si des mesures manquent. Hiérarchie d'imputation :
1. Moyenne historique du capteur
2. Vitesse par défaut (`LYON_DEFAULT_SPEED` = 30 km/h)

### Logique de Matérialisation

- Les dimensions (`dim_spatial_grid_mapping`, `dim_gnn_adjacency`) sont recalculées intégralement à chaque exécution (TRUNCATE + INSERT)
- Les faits (`fact_traffic_series`) utilisent un upsert idempotent (DELETE du timestamp puis INSERT)
- Seuls les capteurs avec <90% de NaN historiques sont considérés "actifs"

## DAG Airflow

**ID** : `lyonflow_traffic_pipeline`
**Schedule** : `*/5 * * * *` (toutes les 5 minutes)
**Max active runs** : 1 (pas de parallélisme)
**Retries** : 2 (délai 30s)

```
ingest_grand_lyon_traffic
  → spatial_transformation_and_mapping
    → materialize_gold_layer
      → export_gold_to_csv
        → stgcn_predict_on_ray
```

| Task | Fonction | Description |
|------|----------|-------------|
| `ingest_grand_lyon_traffic` | `ingest_traffic_data()` | Appel API WFS → Bronze |
| `spatial_transformation_and_mapping` | `transform_traffic_data()` | Bronze → Silver + exports fichier |
| `materialize_gold_layer` | `materialize_gold_layer()` | Silver → Gold (dimensions + faits) |
| `export_gold_to_csv` | `export_to_csv_task()` | Gold → CSV plats pour entraînement |
| `stgcn_predict_on_ray` | `trigger_stgcn_prediction_on_ray()` | Soumission job d'inférence via Ray Jobs API |

## Volumes et Rétention

### Couche Bronze

- **Politique** : aucune déduplication, conservation intégrale de chaque snapshot
- **Croissance** : ~1 snapshot / 5 min = ~288 snapshots / jour
- **Taille typique par snapshot** : ~50-200 Ko (JSONB compressé)
- **Croissance quotidienne estimée** : 14-58 Mo
- **Rétention recommandée** : 30 jours (purge manuelle via `DELETE FROM bronze.trafic_vitesse_brute WHERE fetched_at < NOW() - INTERVAL '30 days';`)

### Couche Silver

- **Politique** : append-only, chaque snapshot Bronze produit ~1 ligne par capteur (≈1500-2500 lignes)
- **Croissance quotidienne** : ~5-8 Mo / jour
- **Index actuels** : `idx_bronze_fetched_at` sur Bronze uniquement. Silver n'a pas d'index → scans séquentiels si volumineux
- **TODO** : ajouter index sur `(transformed_at, properties_twgid)` et `(properties_gid, transformed_at)`

### Couche Gold

- **Dimensions** : `TRUNCATE` + INSERT à chaque exécution → taille constante (≈1500 nœuds, ≈10 000 arêtes)
- **Faits** : `DELETE/INSERT` idempotent par timestamp → 1 ligne par (timestamp × nœud) = ~1500 lignes / snapshot
- **Croissance quotidienne** : ~12-25 Mo / jour
- **Index existants** : PK composite sur `(timestamp, node_idx)` pour `fact_traffic_series`

## Données Manquantes et Stratégies d'Imputation

### Sources de NaN

1. **Mesure absente** : capteur en panne, travaux, hors plage horaire
2. **Mesure non mise à jour** : `est_a_jour = False` (filtré en Silver, donc exclu)
3. **Capteur jamais vu** : nouveau segment routier dans la géométrie

### Pipeline d'Imputation (3 niveaux)

```
Mesure réelle disponible ?
  ├─ OUI → valeur réelle (km/h)
  └─ NON → Mesure historique moyenne du capteur (SQL AVG par twgid)
              ├─ DISPONIBLE → moyenne historique
              └─ NON → LYON_DEFAULT_SPEED (défaut 30 km/h)
```

Implémentation : `materialize_gold_layer()` dans `dags/dag_pipeline.py`.

### Capteurs Actifs vs Inactifs

Un capteur est **actif** si son taux historique de NaN est **< 90%**. Les inactifs sont exclus du Gold (donc du modèle). Conséquence : le graphe est **stable** entre les exécutions tant que l'historique ne change pas drastiquement.

## H3 — Indexation Spatiale

### Pourquoi H3 Résolution 13 ?

- Résolution 13 → cellules d'environ **14 m²** (taille idéale pour un segment de route)
- Indexation rapide via `h3.latlng_to_cell(lat, lon, 13)` (O(1))
- Compatible avec les opérations de voisinage (`grid_disk`) pour la construction du graphe
- Format textuel stable de 15 caractères

### Compatibilité H3 v3 et v4

La lib H3 a changé d'API entre v3 et v4. Le projet gère les deux grâce à des `hasattr` checks :

```python
if hasattr(h3, "cell_to_local_ij"):  # v4
    h3_to_ij_func = h3.cell_to_local_ij
elif hasattr(h3, "experimental_h3_to_local_ij"):  # v3
    h3_to_ij_func = h3.experimental_h3_to_local_ij
```

Vérifier la version installée : `python -c "import h3; print(h3.__version__)"`.

## Performance du Pipeline

### Benchmarks Observés (M1 MacBook, 8 Go RAM)

| Étape | Durée typique | Pic mémoire |
|---|---|---|
| `ingest_traffic_data` | 1-3 s | < 100 Mo |
| `transform_traffic_data` (1er snapshot) | 8-12 s | 600-900 Mo |
| `transform_traffic_data` (snapshots suivants, cache chaud) | 0.1-0.5 s | < 200 Mo |
| `materialize_gold_layer` | 2-5 s | < 300 Mo |
| `export_to_csv_task` | 1-3 s | < 200 Mo |
| `trigger_stgcn_prediction_on_ray` | 30-60 s (chargement + inférence) | 1-2 Go (GPU) |

### Optimisations Mises en Place

- **Cache `_h3shape_cache`** (dict module-level) : évite de recalculer `cells_to_h3shape` pour les mêmes ensembles de cellules H3
- **Super Cache `_super_segment_spatial_cache`** : stocke les représentations **déjà sérialisées** (WKT + JSON) → speedup ×350 sur les snapshots consécutifs
- **Reprojection Lambert-93 → WGS84 globale** : `GLOBAL_TRANSFORMER` réutilisé entre snapshots
- **Interpolation vectorisée** : `np.arange(0, length, 7)` puis `geom.interpolate(dists)` au lieu d'une boucle Python

Pour le profilage détaillé : `utils/profile_rebuild.py` (mesure étape par étape) et `tests/test_super_cache.py` (mesure du speedup cache).

