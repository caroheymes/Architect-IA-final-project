# Entraînement du Modèle

## Modes d'Entraînement

| Mode | Script | Epochs | Objectif |
|------|--------|--------|----------|
| **HPO** (exploration) | `hpo_stgcn.py` | 15 par trial | Trouver les meilleurs hyperparamètres |
| **Champion** (production) | `train_stgcn.py` | 100 max (early stopping) | Entraîner le modèle final |

## Entraînement Champion (`train_stgcn.py`)

### Pipeline Complet

```
1. Connexion PostgreSQL + setup MLflow
                │
2. Chargement des données (CSV ou DB)
   ├── Topologie : num_nodes, edge_index
   └── Séries : vitesse_matrix_raw + features temporelles cycliques
                │
3. Construction du dataset PyG (sliding window)
   ├── Normalisation StandardScaler
   ├── Fenêtres glissantes [t : t+SEQ_LEN] → X, [t+SEQ_LEN+h] → Y
   ├── Multi-horizon : Y shape [N, len(HORIZONS)]
   └── Split chronologique 80/20 → train_loader, test_loader
                │
4. Initialisation modèle + optimiseur
   ├── SpatioTemporalGCN(in=5, hidden=128, out=len(HORIZONS))
   ├── Adam(lr=0.001, weight_decay=1e-5)
   └── Pré-calcul mean_tensor, scale_tensor sur GPU
                │
5. Boucle d'entraînement (100 epochs max)
   │   Pour chaque batch :
   │   ├── Forward pass
   │   ├── Dénormalisation target → km/h (sur GPU)
   │   ├── Calcul poids staircase par tranche de vitesse
   │   ├── Weighted MSE loss
   │   ├── Backward + gradient clipping (max_norm=1.0)
   │   └── Optimizer step
   │
   │   Évaluation sur test set :
   │   ├── Forward pass (no_grad)
   │   ├── Dénormalisation pred + target → km/h
   │   └── MAE en km/h = sum(|pred - target|) / (N_samples × N_nodes × N_horizons)
   │
   │   Early stopping :
   │   ├── Si MAE < best → sauvegarder modèle + scaler, reset compteur
   │   └── Si 10 epochs sans amélioration → arrêter
   │
   │   MLflow logging : train_loss_std + test_mae_kmh par epoch
                │
6. Analyse d'erreur stratifiée (post-entraînement)
   ├── Évaluation sur test set complet
   ├── Découpage en 10 tranches de vitesse (0-10, 10-20, ..., 90+)
   ├── Par tranche : MAE, biais, écart-type prédiction, écart-type erreur
   ├── 4 graphiques : MAE+volume, biais systématique, dispersions, boxplots
   └── Export CSV + PNG (local + MLflow artifacts)
                │
7. Log du checkpoint modèle dans MLflow
```

### Fonction de Perte Détaillée

La loss est un MSE pondéré calculé **en espace standardisé** mais avec des poids déterminés **en km/h** :

```python
# 1. Dénormaliser les targets pour déterminer les poids
y_kmh = batch.y * scale_tensor + mean_tensor

# 2. Attribuer les poids par tranche de vitesse réelle
weights = where(y_kmh < 10,  WEIGHT_JAM,      # Embouteillage  → 15×
          where(y_kmh < 30,  WEIGHT_SLOW,      # Ralentissement → 5×
                             WEIGHT_NORMAL))    # Fluide         → 1×

# 3. MSE pondéré en espace standardisé
loss = mean( (pred_std - target_std)² × weights )
```

**Pourquoi pondérer en km/h mais calculer en espace standardisé** : Les poids reflètent l'importance métier (congestions critiques). La MSE en espace standardisé est numériquement stable pour l'optimisation.

### Early Stopping

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `patience` | 10 | Nombre d'epochs sans amélioration avant arrêt |
| `best_test_mae` | ∞ (init) | Meilleure MAE observée |
| Sauvegarde | À chaque nouveau best | `models/stgcn_prod_latest.pt` + `models/stgcn_scaler.pkl` |
| Restauration | Après arrêt | Les poids du meilleur epoch sont rechargés |

### Analyse d'Erreur Stratifiée

Produit **4 graphiques** sauvés en un seul PNG (2×2) :

| Graphique | Axes X / Y | Ce qu'il montre |
|-----------|-----------|-----------------|
| **1. MAE + Volume** | Tranche vitesse / MAE (km/h) + count | Où le modèle est bon/mauvais + combien de données par tranche |
| **2. Biais systématique** | Tranche / Biais (prédit - réel) | Sur-estimation (orange) vs sous-estimation (bleu) par tranche |
| **3. Écarts-types** | Tranche / σ (km/h) | Dispersion des prédictions + incertitude des résidus |
| **4. Boxplots** | Tranche / Vitesse prédite | Distribution des prédictions vs ligne "vitesse réelle cible" |

## Construction du Dataset (`dataset.py`)

### Sources de Données

| Mode | Fonctions | Source |
|------|-----------|--------|
| **PostgreSQL** | `load_graph_topology()` + `load_traffic_series()` | Tables Gold en direct |
| **CSV local** | `load_graph_topology_from_csv()` + `load_traffic_series_from_csv()` | Fichiers `data/in/` |

### Topologie du Graphe

```python
# Depuis PostgreSQL :
df_mapping = read_sql("SELECT node_idx, properties_twgid FROM gold.dim_spatial_grid_mapping")
df_edges = read_sql("SELECT node_u, node_v FROM gold.dim_gnn_adjacency")

# Construction edge_index PyG :
# 1. Arêtes bidirectionnelles (u→v et v→u)
# 2. Self-loops (i→i pour chaque nœud)
# → Tensor [2, E] en format COO sparse
```

### Fenêtre Glissante (Sliding Window)

```
Entrée : matrice [T, N] (timestamps × nœuds) + features temporelles

Pour t = 0, 1, ..., T - SEQ_LEN - max(HORIZONS) :
  X[t] = matrice[t : t+SEQ_LEN, :].T        → shape [N, SEQ_LEN]
       + hour_sin, hour_cos, day_sin, day_cos → shape [N, SEQ_LEN] chacun
       = stack → [N, SEQ_LEN, 5]

  Y[t] = pour chaque horizon h dans HORIZONS :
           matrice[t + SEQ_LEN - 1 + h, :].reshape(-1, 1)
         = concatenate → [N, len(HORIZONS)]

  → Data(x=X, edge_index=edge_index, y=Y)

Split chronologique :
  train = data_list[:80%]   (shuffle=True dans DataLoader)
  test  = data_list[80%:]   (shuffle=False)
```

**Note** : Le shuffle dans le DataLoader mélange les **fenêtres temporelles**, pas les timestamps à l'intérieur d'une fenêtre. L'ordre temporel intra-fenêtre est préservé pour le GRU.

### Normalisation

```python
scaler = StandardScaler()
vitesse_matrix = scaler.fit_transform(vitesse_matrix_raw)
# scaler.mean_ : [N] moyennes par nœud
# scaler.scale_ : [N] écarts-types par nœud
```

Le scaler est fitté sur **tout** le dataset (train+test) pour que la normalisation soit cohérente. Les poids sont sauvés dans `models/stgcn_scaler.pkl` pour l'inférence.

## Artéfacts Produits

| Fichier | Contenu | Destination |
|---------|---------|-------------|
| `models/stgcn_prod_latest.pt` | Poids du meilleur epoch (state_dict) | Local + MLflow |
| `models/stgcn_scaler.pkl` | StandardScaler fitté | Local + MLflow |
| `models/stratified_error_analysis.csv` | Tableau MAE/biais/σ par tranche | Local + MLflow |
| `models/stratified_error_analysis.png` | 4 graphiques d'analyse (2×2) | Local + MLflow |

## MLflow Tracking

### Expériences

| Nom | Usage |
|-----|-------|
| `LyonFlow-STGCN-Production-Training-v2` | Entraînements champions |
| `LyonFlow-STGCN-Optuna-Tuning` | Études HPO |

### Paramètres Loggés (Champion)

```
seq_len, batch_size, hidden_channels, lr, weight_decay, epochs, weight_jam, weight_slow
```

### Métriques Loggées

| Métrique | Par epoch | Unité |
|----------|-----------|-------|
| `train_loss_std` | Oui | Sans unité (MSE standardisée) |
| `test_mae_kmh` | Oui | km/h |

### Artéfacts Loggés

```
model_checkpoints/stgcn_prod_latest.pt
analysis/stratified_error_analysis.csv
plots/stratified_error_analysis.png
```
