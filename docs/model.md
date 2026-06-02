# Modèle ML — ST-GRU-GNN

## Clarification Architecturale

Le code utilise le nom de classe `SpatioTemporalGCN` et le préfixe `stgcn` par simplicité, mais l'architecture **diffère** du STGCN original (Yu et al., 2018).

| | STGCN Original (Yu 2018) | Notre implémentation (ST-GRU-GNN) |
|---|---|---|
| **Composante temporelle** | Convolutions causales 1D avec GLU | GRU (Gated Recurrent Unit) |
| **Composante spatiale** | Convolutions spectrales de graphes | 2× GCNConv (PyTorch Geometric) |
| **Connexions résiduelles** | Entre blocs ST | Entre couches spatiales |
| **Justification** | Données régulières et propres | Robustesse au bruit et échantillonnage irrégulier |

Ce choix est **délibéré** : le GRU est plus robuste face au bruit et aux trous dans le flux réel de la Métropole de Lyon.

## Architecture du Réseau

```
Entrée : x [B×N, SEQ_LEN, 5]
  │
  ▼
┌─────────────────────────────────┐
│  GRU (Temporal Encoder)         │
│  input_size=5, hidden=H         │
│  → Dernière sortie cachée       │
│  Sortie : h_temp [B×N, H]      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  GCNConv 1 (Spatial)            │
│  H → H + LeakyReLU(0.2)        │
│  + Skip Connection (h_temp)     │   h_space1 = ReLU(GCN1(h_temp)) + h_temp
│  Sortie : h_space1 [B×N, H]    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  GCNConv 2 (Spatial)            │
│  H → H + LeakyReLU(0.2)        │
│  + Skip Connection (h_space1)   │   h_space2 = ReLU(GCN2(h_space1)) + h_space1
│  Sortie : h_space2 [B×N, H]    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Linear (Regression Head)       │
│  H → out_channels               │
│  Sortie : [B×N, n_horizons]    │
└─────────────────────────────────┘
```

### Paramètres Clés

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `in_channels` | 5 | speed + hour_sin + hour_cos + day_sin + day_cos |
| `hidden_channels` | 128 | Dimension cachée GRU et GCN |
| `out_channels` | len(HORIZONS) | Nombre d'horizons de prédiction |
| `SEQ_LEN` | 120 | Fenêtre temporelle d'entrée (120 × 5min = 10h) |

## Features d'Entrée (5 canaux)

Pour chaque noeud du graphe et chaque pas de temps :

| Canal | Description | Plage |
|-------|-------------|-------|
| `speed` | Vitesse standardisée (StandardScaler) | ~ [-3, +3] |
| `hour_sin` | sin(2π × heure / 24) | [-1, 1] |
| `hour_cos` | cos(2π × heure / 24) | [-1, 1] |
| `day_sin` | sin(2π × jour_semaine / 7) | [-1, 1] |
| `day_cos` | cos(2π × jour_semaine / 7) | [-1, 1] |

Les features cycliques encodent le temps sans discontinuité (minuit ≈ 23h59, dimanche ≈ lundi).

## Graphe Routier

- **Noeuds** : ~1 520 capteurs actifs (segments avec <90% NaN)
- **Arêtes** : ~9 540 (non orienté, voisinage H3 K=2 + boucles sur soi-même)
- **Format** : `edge_index` PyTorch Geometric [2, E] (COO sparse)

## Fonction de Perte Personnalisée (Staircase Weighting)

MSE pondérée par tranche de vitesse réelle (dénormalisée en km/h) :

| Tranche de vitesse | Poids | Raison |
|---------------------|-------|--------|
| < 10 km/h (embouteillage) | **15.0** | Erreur critique — impact sécurité et décision |
| 10 – 30 km/h (ralentissement) | **5.0** | Erreur significative — congestion probable |
| > 30 km/h (fluide) | **1.0** | Erreur tolérable — trafic normal |

```python
loss = mean( (pred - target)² × weight_by_speed_bin )
```

Cette pondération force le modèle à être précis là où ça compte : les congestions.

## Entraînement

### Configuration

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Optimiseur | Adam | lr=0.001, weight_decay=1e-5 |
| Gradient clipping | max_norm=1.0 | Prévention explosion de gradients (GRU+GNN) |
| Early stopping | patience=10 | Arrêt si MAE test ne s'améliore plus |
| Epochs max | 100 | Limite haute |
| Batch size | 2 | Contraint par 4 Go VRAM |
| Split | 80/20 | Chronologique (pas aléatoire) |
| Threads PyTorch | 1 | Bridé pour éviter surchauffe CPU |

### Métriques Trackées (MLflow)

| Métrique | Unité | Logging |
|----------|-------|---------|
| `train_loss_std` | Sans unité (MSE standardisée) | Par epoch |
| `test_mae_kmh` | km/h | Par epoch |

### Analyse d'Erreur Stratifiée

Après entraînement, évaluation fine par tranche de vitesse (0-10, 10-20, ..., 90+) :
- MAE par tranche
- Biais systématique (sur/sous-estimation)
- Écart-type des prédictions et des erreurs
- Boxplots des distributions

Résultats sauvés en CSV + PNG et loggés comme artéfacts MLflow.

## HPO (Optuna + Ray)

- **Sampler** : TPE (Tree-structured Parzen Estimator) — optimisation bayésienne
- **Pruner** : MedianPruner — élimination précoce des mauvais trials
- **Stockage** : PostgreSQL (coordination multi-worker)
- **Parallélisme** : Via Ray Tune sur le cluster
- **Accélération mesurée** : ×5.3 (20 trials en ~8 min sur machine locale)

### Hyperparamètres Recherchés

| Paramètre | Espace de recherche |
|-----------|---------------------|
| `hidden_channels` | Catégorique |
| `seq_len` | Catégorique |
| `learning_rate` | Log-uniforme |
| `weight_decay` | Log-uniforme |
| `weight_jam` | Uniforme |
| `weight_slow` | Uniforme |

## Prédiction Multi-Horizon

Le modèle peut prédire simultanément à plusieurs horizons temporels :

| Horizon | Pas de temps | Durée |
|---------|-------------|-------|
| 6 | 6 × 5 min | 30 minutes |
| 12 | 12 × 5 min | 1 heure |
| 36 | 36 × 5 min | 3 heures |

Configurable via la variable `HORIZONS` (ex: `"6,12,36"`).

## Inférence

### Pipeline Complet

```
predict_stgcn.py
  1. Chargement topologie + 120 derniers pas (DB ou CSV)
  2. Chargement scaler (stgcn_scaler.pkl)
  3. Normalisation → tenseur PyG [N, 120, 5]
  4. model.forward() → [N, n_horizons] (en km/h standardisé)
  5. Dénormalisation scaler → km/h
  6. Clip [1, 130] km/h (sécurité physique)
  7. Écriture CSV (data/out/predictions_traffic.csv) + INSERT PostgreSQL
```

### Mode Local (CSV) vs Online (PostgreSQL)

| Mode | Variable `USE_LOCAL_CSV` | Cas d'usage |
|---|---|---|
| **Local** | `true` | Test unitaire, debug sans DB, exécution CI/CD GitHub Actions |
| **Online** | `false` (défaut) | Production : lit la dernière snapshot Gold + écrit les prédictions |

### Performance d'Inférence (M1 Pro, GPU désactivé)

| Phase | Durée |
|---|---|
| Chargement modèle + scaler | 1-2 s |
| Préparation tenseur | 0.5-1 s |
| Forward pass (batch=1) | 50-200 ms |
| Dénormalisation + formatage | 0.5-1 s |
| **Total par cycle de prédiction** | **3-6 s** |

### Backfill Historique

`backfill_predictions.py` permet de prédire rétroactivement sur tous les pas de temps où aucune prédiction n'existe encore. Utile pour :

- Repeupler la table après une coupure de service
- Générer un dataset d'évaluation historique
- Tester la stabilité temporelle du modèle

**Algorithme** :
1. Charge tous les `prediction_timestamp` existants dans `gold.fact_predictions_traffic`
2. Liste tous les timestamps éligibles dans `gold.fact_traffic_series` (≥ `SEQ_LEN` pas d'historique)
3. Pour chaque timestamp manquant : reconstruit la fenêtre, infère, INSERT

## Versionnage et Reproductibilité

### Artéfacts Sauvegardés

| Artéfact | Localisation | Rôle |
|---|---|---|
| `stgcn_prod_latest.pt` | `models/` (monté Docker) | Poids du modèle champion |
| `stgcn_scaler.pkl` | `models/` | `StandardScaler` fitté (nécessaire à l'inférence) |
| `stratified_error_analysis.{csv,png}` | `models/` | Diagnostic par tranche de vitesse |
| MLflow run | `mlflow-artifacts/` (volume) | Historique complet (paramètres, métriques, artéfacts) |

### Convention de Tagging MLflow

- **Experiment name** : `LyonFlow-STGCN-Production-Training-v2`
- **Run name** : `STGCN_Prod_Train_{YYYYMMDD_HHMMSS}`
- **Tag** : `mlflow.runName` pour la lisibilité dans l'UI

### Promouvoir un Modèle

1. Le workflow `ml-training.yml` calcule la MAE finale
2. Si `mae < mae_threshold` (défaut 5.0 km/h) → création d'une **GitHub Release** automatique
3. Le tag de release (`vX.Y.Z`) référence le commit + les artifacts

## Limites Connues et Trade-offs

| Limite | Impact | Mitigation actuelle |
|---|---|---|
| Mémoire GPU limitée (4-9 Go) | `batch_size=2`, `hidden=128` max | Gradient clipping, threads bridés |
| Capteurs inactifs exclus | Graphe peut rétrécir si beaucoup de pannes | Filtre < 90% NaN configurable |
| Capteurs jamais vus | Pas de prédiction possible | Capteurs inconnus → exclus du modèle |
| Imputation à 30 km/h | Biais sur capteurs très atypiques | Amélioration possible via ML par capteur |
| Pas de validation croisée temporelle | Risque d'overfit sur une période | Split chronologique strict + early stopping |
| Pas de monitoring de drift | Modèle peut dériver sans alerte | TODO — voir [tech-debt.md](tech-debt.md) |

## Pistes d'Amélioration

- **Transformer temporel** : remplacer le GRU par un Transformer avec attention causale
- **Attention spatiale** : `GATConv` au lieu de `GCNConv` pour pondérer les voisins
- **Perte multi-tâches** : prédire simultanément vitesse + état (fluide/lent/bouchon)
- **Apprentissage incrémental** : réentraîner sur les N derniers jours chaque semaine
- **Model registry MLflow** : formaliser le staging → production (staging/archived/production)
