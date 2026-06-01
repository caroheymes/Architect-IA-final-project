# 🕸️ LyonFlow : Modélisation Spatio-Temporelle (STGCN)

Ce dossier contient l'implémentation industrialisée et modulaire du modèle de réseau de neurones sur graphes spatio-temporels (**STGCN**) pour prédire la vitesse de circulation sur la métropole de Lyon.

## 📁 Architecture des Fichiers

L'implémentation est découpée de manière modulaire selon les meilleures pratiques du MLOps :

*   **`model.py`** : Définition de la classe PyTorch `SpatioTemporalGCN`. Combine un encodeur temporel recurrent (GRU) avec des convolutions de graphes spatiales (`GCNConv`) de PyTorch Geometric, connectées par des sauts de connexion (Skip Connections).
*   **`dataset.py`** : Logique d'extraction des données depuis la couche **Gold** de PostgreSQL. Gère le chargement de la topologie (graphe de 1 520 nœuds et 9 540 arêtes), le pivotement des séries temporelles, le calcul des features temporelles cycliques (sin/cos de l'heure et du jour de la semaine) et la construction du dataset glissant normalisé.
*   **`train_stgcn.py`** : Script d'entraînement pour un essai unique avec des hyperparamètres fixes. Il intègre une fonction de perte personnalisée de type "escalier" (Staircase Weighting Loss) pour pénaliser lourdement les erreurs sur les faibles vitesses (congestions), et logue les métriques dans **MLflow**.
*   **`hpo_stgcn.py`** : Script d'optimisation d'hyperparamètres (HPO) distribué utilisant **Optuna**. Il effectue une recherche bayésienne (TPE), implémente un élagage intelligent des mauvais essais (MedianPruner), persiste l'avancement dans PostgreSQL pour permettre la coordination entre plusieurs workers Ray, et logue les résultats dans **MLflow**.

---

## 🚦 Comment Lancer l'Entraînement ou le Tuning

### 1. Entraînement Simple (Single Run)

Pour entraîner un modèle STGCN avec des paramètres standards et enregistrer les poids dans le dossier `models/stgcn_prod_latest.pt` :

```bash
python train_stgcn.py
```

### 2. Tuning d'Hyperparamètres (Optuna)

Pour lancer la recherche bayésienne d'Optuna sur 20 essais (en parallélisant ou en exécutant sur vos containers) :

```bash
python hpo_stgcn.py
```

Pour suivre l'avancement de l'optimisation en temps réel via l'interface graphique :
```bash
optuna-dashboard postgresql://lyonflow:lyonflow_password@localhost:5432/lyonflow
```
*Le dashboard sera alors consultable à l'adresse [http://localhost:8080](http://localhost:8080)*.
