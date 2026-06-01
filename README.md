# 🟢 LyonFlow : Spatio-Temporal Traffic Prediction Platform (Grand Lyon)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8.1-017CE2?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0?logo=ray&logoColor=white)](https://www.ray.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_HPO-004B87?logo=optuna&logoColor=white)](https://optuna.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-WSL2-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**LyonFlow** est une plateforme MLOps de bout en bout conçue pour ingérer, transformer, modéliser et prédire en temps réel l'état du trafic routier de la Métropole de Lyon. Ce projet d'architecture IA combine un pipeline de données  géospatiales et un modèle de Deep Learning géométrique avancé.

---

## 🗺️ Architecture de Données (Médaillon)

La plateforme repose sur une architecture de stockage structurée en trois couches dans **PostgreSQL** :

```mermaid
graph TD
    API[API Grand Lyon WFS] -->|Ingestion Temps Réel - 5 min| Bronze[Layer Bronze: Raw JSON]
    Bronze -->|Spatial Interpolation & H3 Indexing| Silver[Layer Silver: Cleaned & Categorized Sensors]
    Silver -->|Graph Extraction & Historical Fallback| Gold[Layer Gold: GNN Spatial Grid & Imputed Facts]
    Gold -->|Distributed HPO & Training| STGCN[STGCN Model - PyG]
    Gold -->|Visualisation| Streamlit[Streamlit Live Dashboard]
```

1. **Layer Bronze (`bronze.trafic_vitesse_brute`)** : Ingestion brute au format JSONB du flux WFS de la Métropole de Lyon toutes les 5 minutes via Airflow.
2. **Layer Silver (`silver.trafic_vitesse_propre`)** :
   * Nettoyage spatial et géospatial (EPSG:2154 Lambert-93 vers EPSG:4326 WGS84).
   * Interpolation des segments de route tous les **7 mètres** pour générer des coordonnées précises.
   * Indexation spatiale à l'aide de cellules **H3 (Résolution 13)**.
   * Normalisation et catégorisation des vitesses en temps réel.
3. **Layer Gold (`gold`)** :
   * `dim_spatial_grid_mapping` : Alignement des capteurs géographiques sur une grille relative $(i, j)$ constante.
   * `dim_gnn_adjacency` : Construction dynamique de la matrice d'adjacence du graphe routier (connexion des capteurs adjacents).
   * `fact_traffic_series` : Table temporelle de faits de vitesse interpolée et imputée (avec repli sur la moyenne historique par capteur en cas de valeur manquante) garantissant une entrée de taille constante $N$ pour l'apprentissage.

---

## 🧠 Pourquoi ce choix technologique ? (STGCN + Ray + Optuna)

Pour résoudre le défi de la prédiction du trafic, une simple approche par séries temporelles classiques (LSTM, Prophet) ou par apprentissage tabulaire (XGBoost) est insuffisante car **le trafic routier est intrinsèquement spatio-temporel**.

### 1. Architecture du Modèle : Spatio-Temporal GRU-GNN (ST-GRU-GNN)
> [!IMPORTANT]
> **Clarification Architecturale (Honnêteté Scientifique) :**
> Notre implémentation dans `model.py` différe de la structure STGCN originale proposée par Yu et al. (2018). Alors que l'article original préconise des convolutions temporelles causales 1D (Gated CNNs avec GLU), nous implémentons un modèle hybride récurrent-spatial : **GRU + GCN avec Skip Connections** (que nous désignons sous le nom de **ST-GRU-GNN**). Bien que le nom de classe `SpatioTemporalGCN` et les scripts conservent par simplicité le préfixe `stgcn`, l'architecture récurrente est ici privilégiée pour sa robustesse face au bruit et à l'échantillonnage irrégulier du flux réel de la Métropole de Lyon.

Le trafic dépend à la fois de son propre historique (temporel) et du trafic des rues adjacentes qui s'y déverse (spatial) :
* **Composante Temporelle (Temporal GRU Encoder) :** Utilise un réseau récurrent **GRU (Gated Recurrent Unit)** pour modéliser les dépendances temporelles séquentielles de vitesse sur chaque nœud du graphe. Cela capte efficacement les dynamiques temporelles à court terme sans souffrir des limitations des convolutions causales sur des séries réelles bruitées.
* **Composante Spatiale (Spatial GNN Decoder) :** Utilise deux couches de convolutions spectrales de graphes (`GCNConv` de PyTorch Geometric) sur la matrice d'adjacence du réseau routier pour propager l'influence physique des flux entre segments adjacents.
* **Sauts de Connexion (Residual Skip Connections) :** Des connexions résiduelles relient les étapes de convolution spatiale pour stabiliser le calcul des gradients lors de la backpropagation.
* **Framework :** Écrit sous **PyTorch Geometric (PyG)**.

### 2. Moteur : Ray Cluster (Apprentissage Distribué)
L'entraînement d'un modèle de Deep Learning sur des graphes (GNN) est extrêmement gourmand en ressources de calcul.
* **Gestion des ressources locales :** Ray permet de brider strictement l'usage CPU/GPU (ex. allocation stricte de 1 CPU et du GPU T600) via Docker pour éviter toute surchauffe ou blocage de la machine hôte.
* **Évolutivité (Scale-out) :** Ray abstrait l'infrastructure. Le même code d'entraînement s'exécute de manière transparente sur en local ou sur un cluster Kubernetes de plusieurs nœuds dans le Cloud, sans modification.
* **Parallélisme :** Les trials d'optimisation s'exécutent en parallèle, gérant dynamiquement la file d'attente sur les ressources partagées.

### 3. Optimiseur : Optuna (Bayesian Hyperparameter Tuning)
Le comportement de STGCN est hautement sensible à ses hyperparamètres (nombre de canaux cachés, taille de la fenêtre temporelle d'entrée, taux d'apprentissage, et pondérations spécifiques pour pénaliser les ralentissements et embouteillages).
* **Optimisation Bayésienne (TPE Sampler) :** Contrairement à une recherche aléatoire (Random Search) ou sur grille (Grid Search), Optuna construit un modèle probabiliste des performances pour échantillonner intelligemment les meilleurs jeux de paramètres futurs.
* **Élimination Précoce (Pruning) :** Grâce au pruner d'Optuna (MedianPruner), les trials qui affichent des performances initiales médiocres après quelques époques sont arrêtés immédiatement. Cela a permis d'accélérer notre recherche d'un **facteur de x5.3** (20 trials terminés avec succès en seulement 8 minutes sur machine locale !).

---

## 🏗️ Stack Technologique & Écosystème MLOps

La plateforme orchestre plusieurs conteneurs isolés via Docker Compose :

* **Orchestrateur (Apache Airflow) :** Gère le déclenchement périodique de l'ingestion brute (Bronze), de la transformation spatiale (Silver) et de la matérialisation du graphe (Gold).
* **Moteur d'Expérimentation (MLflow) :** Enregistre automatiquement chaque trial d'optimisation d'Optuna (paramètres, métriques de perte d'entraînement/validation, et artéfacts du modèle).
* **Visualiseur HPO (Optuna Dashboard) :** Dashboard web complet dédié à l'étude d'hyperparamètres, affichant les analyses d'importance des paramètres, les coordonnées parallèles et les courbes de convergence.
* **Interface Utilisateur (Streamlit) :** Une application interactive permettant aux décideurs d'observer l'état du réseau routier et d'accéder aux prévisions en temps réel.
* **Réseau (ngrok) :** Fournit un tunnel sécurisé TCP pour exposer à distance la base PostgreSQL.

---

## ⚡ Guide de Démarrage Rapide

### 1. Lancement de la plateforme MLOps
Assurez-vous que Docker Desktop (avec backend WSL2) est démarré, puis exécutez à la racine du projet :
```powershell
docker-compose up -d --build
```

### 2. Accéder aux différents services

| Service | URL locale | Description |
| :--- | :--- | :--- |
| **📊 Streamlit App** | [http://localhost:8501](http://localhost:8501) | Dashboard trafic en temps réel |
| **🧪 MLflow Tracking** | [http://localhost:5000](http://localhost:5000) | Suivi des entraînements et des modèles |
| **⚙️ Apache Airflow** | [http://localhost:8080](http://localhost:8080) | Interface d'administration des pipelines |
| **📈 Optuna Dashboard** | [http://localhost:8085](http://localhost:8085) | Visualisation bayésienne des hyperparamètres |
| **⚡ Ray Dashboard** | [http://localhost:8265](http://localhost:8265) | Monitoring du cluster de calcul distribué |

### 3. Exécuter l'entraînement du modèle
Pour lancer l'entraînement final du modèle STGCN optimisé avec les meilleurs paramètres identifiés :
```powershell
docker exec -it lyonflow-ray-worker python /home/ray/project/training/stgcn/train_stgcn.py
```
docker ps --format "table {{.Names}}\t{{.Status}}"
docker ps -f "name=ray-worker"
docker exec -it f8db0d184d73_lyonflow-ray-worker python /home/ray/project/training/stgcn/train_stgcn.py

docker exec -it f8db0d184d73_lyonflow-ray-worker /home/ray/project/training/stgcn/train_stgcn.py