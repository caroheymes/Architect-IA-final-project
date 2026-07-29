# Optuna HPO — Optimisation Bayésienne des Hyperparamètres

## Principe

Trouver les meilleurs hyperparamètres du modèle ST-GRU-GNN par **optimisation bayésienne** (pas de grid search ni random search). Optuna construit un modèle probabiliste des performances pour échantillonner intelligemment les prochains jeux de paramètres à tester.

## Espace de Recherche

| Paramètre | Type | Plage | Distribution |
|-----------|------|-------|-------------|
| `lr` | float | [1e-4, 1e-2] | Log-uniforme |
| `hidden_channels` | catégorique | {64, 128, 256} | Uniforme |
| `weight_decay` | float | [1e-6, 1e-4] | Log-uniforme |
| `seq_len` | int | {6, 12, 18, 24} | Step=6 (30 min, 1h, 1h30, 2h) |
| `batch_size` | catégorique | {8, 16} | Uniforme |
| `weight_jam` | float | [5.0, 20.0] | Uniforme |
| `weight_slow` | float | [2.0, 8.0] | Uniforme |

**Note** : Pendant HPO, `seq_len` varie entre 6 et 24 pas (30 min – 2h) pour trouver la fenêtre optimale. Le modèle champion est ensuite réentraîné avec `seq_len=120` (10h) pour une meilleure performance en production.

## Algorithmes

### Sampler : TPE (Tree-structured Parzen Estimator)

```python
# Optuna utilise TPE par défaut
study = optuna.create_study(direction="minimize")
```

TPE modélise `P(hyperparams | score < seuil)` et `P(hyperparams | score >= seuil)`, puis échantillonne les configurations maximisant le ratio. Plus efficace que random search, surtout avec 7 hyperparamètres.

### Pruner : MedianPruner

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,  # Pas de pruning sur les 5 premiers trials
    n_warmup_steps=3,  # Pas de pruning sur les 3 premières epochs d'un trial
)
```

Fonctionnement :
1. Un trial rapporte sa MAE à chaque epoch via `trial.report(test_mae_kmh, epoch)`
2. Si la MAE à l'epoch E est pire que la **médiane** des MAE des trials précédents à la même epoch → **pruned** (arrêté)
3. `n_startup_trials=5` : Les 5 premiers trials ne sont jamais prunés (pas assez de référence)
4. `n_warmup_steps=3` : Un trial n'est jamais pruné avant l'epoch 3 (laisser le modèle converger)

**Impact mesuré** : Accélération ×5.3 — 20 trials terminés en 8 minutes sur machine locale.

## Stockage Distribué (PostgreSQL)

```python
postgres_storage = optuna.storages.RDBStorage(
    url="postgresql+psycopg2://lyonflow:pass@postgres:5432/lyonflow", engine_kwargs={"pool_pre_ping": True}
)

study = optuna.create_study(
    study_name="lyonflow_stgcn_tuning_v1",
    storage=postgres_storage,
    load_if_exists=True,  # Reprendre une étude existante
    direction="minimize",
    pruner=pruner,
)
```

**Pourquoi PostgreSQL** (pas SQLite) :
- **Multi-worker** : Plusieurs Workers Ray peuvent exécuter des trials en parallèle sur la même étude
- **Persistance** : L'étude survit aux redémarrages de conteneurs
- **Reprise après crash** : `load_if_exists=True` reprend exactement là où ça s'est arrêté
- **Optuna Dashboard** : Le dashboard web lit directement depuis PostgreSQL

## Cache de Données (Module-Level)

```python
# Variables globales partagées par tous les trials
topology_data = None
traffic_data = None


def run_hpo():
    global topology_data, traffic_data
    # Chargé UNE SEULE FOIS depuis PostgreSQL
    topology_data = load_graph_topology(engine)
    traffic_data = load_traffic_series(engine)
    # Les 20 trials réutilisent ce cache
    study.optimize(objective, n_trials=20)
```

Sans ce cache, chaque trial rechargerait ~1 520 nœuds × des milliers de timestamps depuis PostgreSQL. Le cache évite N requêtes DB lourdes.

## Boucle d'Entraînement HPO

Chaque trial exécute un entraînement rapide de **15 epochs** (vs 100 pour le champion) :

```
Pour chaque trial (1 à 20) :
  1. Sampler TPE → nouveau jeu d'hyperparamètres
  2. build_sliding_dataset() avec seq_len et batch_size du trial
  3. Instancier SpatioTemporalGCN(hidden_channels=...)
  4. Pour chaque epoch (1 à 15) :
     a. Entraîner (forward + staircase loss + backward + clip grad)
     b. Évaluer MAE en km/h sur test set
     c. trial.report(mae, epoch)
     d. Si should_prune() → lever TrialPruned → passer au trial suivant
  5. Retourner MAE finale → Optuna met à jour son modèle TPE
```

## Visualisation (Optuna Dashboard)

```bash
optuna-dashboard postgresql://lyonflow:lyonflow_password@localhost:5432/lyonflow
# → http://localhost:8085
```

### Graphiques Disponibles

| Visualisation | Description |
|---------------|-------------|
| **Optimization History** | MAE par trial (chronologique), avec best-so-far |
| **Parameter Importances** | Classement ANOVA des hyperparamètres par impact sur la MAE |
| **Parallel Coordinate** | Vue multi-dimensionnelle de tous les trials |
| **Contour Plots** | Heatmaps 2D (ex: lr × hidden_channels → MAE) |
| **Slice Plots** | Effet marginal de chaque hyperparamètre |
| **Trial Timeline** | Durée de chaque trial (montre l'effet du pruning) |

## Récupération des Meilleurs Paramètres

Le script `get_best_params.py` extrait les hyperparamètres du champion :

```
Priorité :
  1. Optuna (PostgreSQL) → study.best_params  ← source de vérité
  2. MLflow (API)        → meilleur run de l'expérience "Optuna-Tuning"
  3. Défauts hardcodés   → valeurs conservatrices
```

**Post-traitements** :
- `seq_len` forcé à **120** (le HPO explore 6-24, mais le champion utilise 10h d'historique)
- `batch_size` plafonné à **16** (sécurité VRAM pour le seq_len=120 plus long)
- `lr` renommé en `LEARNING_RATE` (convention de `train_stgcn.py`)

**Sortie** : Lignes `export KEY=VALUE` à sourcer dans un shell :
```bash
source <(python training/stgcn/get_best_params.py)
python training/stgcn/train_stgcn.py
```

## Intégration CI/CD (ml-training.yml)

Dans le pipeline GitHub Actions ML :

```
export-data → hpo (si skip_hpo=false) → train-champion → promote
```

1. **export-data** : Gold → CSV (artéfact GitHub)
2. **hpo** : Exécuté sur runner `[self-hosted, gpu]`, télécharge les CSV, lance `hpo_stgcn.py`, exporte `best_params.sh`
3. **train-champion** : Source `best_params.sh`, entraîne avec `EPOCHS=100` et `HORIZONS="6,12,36"`
4. **promote** : Évalue MAE moyenne → GitHub Release si sous le seuil (défaut: 5.0 km/h)

**Skip HPO** : Si `skip_hpo=true` (paramètre workflow_dispatch), le step HPO est sauté et le champion est entraîné avec les paramètres par défaut ou ceux du dernier HPO.
