import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_graph_topology(engine):
    """Charge la topologie du graphe routier depuis la couche Gold (PostgreSQL).

    Lit deux tables :
      - `gold.dim_spatial_grid_mapping` pour la liste ordonnée des nœuds
        (permet d'indexer les features).
      - `gold.dim_gnn_adjacency` pour les arêtes du graphe (sens `u → v`).

    Construit l'`edge_index` au format PyG :
      - ajoute les arêtes dans les deux sens (`u→v` et `v→u`) car
        l'adjacence Gold est non-dirigée ;
      - ajoute une self-loop par nœud pour stabiliser la propagation GCN.

    Args:
        engine (sqlalchemy.Engine): Engine SQLAlchemy pointant sur la base.

    Returns:
        tuple[int, torch.Tensor]: `(num_nodes, edge_index)` où `edge_index`
        est de shape `[2, 2*E + N]` (`E` = nb d'arêtes, `N` = nb de nœuds).
    """
    df_mapping = pd.read_sql(
        "SELECT node_idx, properties_twgid FROM gold.dim_spatial_grid_mapping ORDER BY node_idx ASC;", con=engine
    )
    num_nodes = len(df_mapping)

    df_edges = pd.read_sql("SELECT node_u, node_v FROM gold.dim_gnn_adjacency;", con=engine)

    edge_index_list = []
    for _, row in df_edges.iterrows():
        u, v = int(row["node_u"]), int(row["node_v"])
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])  # undirected / bidirectional
    for i in range(num_nodes):
        edge_index_list.append([i, i])  # self-loops

    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return num_nodes, edge_index_tensor


def load_traffic_series(engine):
    """Charge les séries de vitesse et calcule les features temporelles cycliques.

    Pipeline :
      1. Lit `gold.fact_traffic_series` (long format).
      2. Pivote en matrice `[Timestamps × Nodes]` (`vitesse_matrix_raw`).
      3. Remplace les NaN par la vitesse par défaut (`LYON_DEFAULT_SPEED`,
         30 km/h) pour garantir une matrice dense.
      4. Calcule 4 features temporelles cycliques (encodage sin/cos pour
         éviter la discontinuité entre 23h et 0h, et entre dimanche et lundi) :
         - `hour_sin`, `hour_cos` (période 24 h)
         - `day_sin`,  `day_cos`  (période 7 jours)

    Args:
        engine (sqlalchemy.Engine): Engine SQLAlchemy.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        `(vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos)`.
        `vitesse_matrix_raw` est de shape `[T, N]`, les 4 autres de shape `[T]`.
    """
    df_facts = pd.read_sql(
        "SELECT timestamp, node_idx, properties_vitesse FROM gold.fact_traffic_series ORDER BY timestamp ASC, node_idx ASC;",
        con=engine,
    )
    df_pivot = df_facts.pivot(index="timestamp", columns="node_idx", values="properties_vitesse")
    import os

    default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
    vitesse_matrix_raw = np.nan_to_num(df_pivot.values, nan=default_speed)

    # Extract cyclical temporal variables
    df_pivot.index = pd.to_datetime(df_pivot.index)
    hours = df_pivot.index.hour + df_pivot.index.minute / 60.0
    days = df_pivot.index.dayofweek

    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    day_sin = np.sin(2 * np.pi * days / 7.0)
    day_cos = np.cos(2 * np.pi * days / 7.0)

    return vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos


def build_sliding_dataset(
    vitesse_matrix_raw,
    hour_sin,
    hour_cos,
    day_sin,
    day_cos,
    seq_len,
    edge_index_tensor,
    num_nodes,
    test_split=0.2,
    batch_size=16,
    horizons=[1],
):
    """Construit le dataset glissant multi-horizon au format PyTorch Geometric.

    Pour chaque pas de départ `t`, on crée un objet `Data` PyG avec :
      - `x` de shape `[N, seq_len, 5]` : `[speed, hour_sin, hour_cos, day_sin, day_cos]`
        (la vitesse est standardisée via `StandardScaler.fit_transform`).
      - `edge_index` partagé (réutilisé sur tous les samples).
      - `y` de shape `[N, len(horizons)]` : la vitesse future à chaque horizon
        (`t + seq_len - 1 + h` pour chaque `h` dans `horizons`).

    Le split train/test est **chronologique** (pas aléatoire) pour respecter
    la nature temporelle du problème.

    Args:
        vitesse_matrix_raw (np.ndarray): Matrice `[T, N]` des vitesses brutes (km/h).
        hour_sin/cos, day_sin/cos (np.ndarray): Vecteurs `[T]` de features cycliques.
        seq_len (int): Longueur de la fenêtre d'entrée.
        edge_index_tensor (torch.Tensor): `edge_index` PyG partagé.
        num_nodes (int): Nombre de nœuds du graphe.
        test_split (float): Proportion du set de test (défaut 0.2 = 80/20).
        batch_size (int): Taille de batch pour les DataLoaders.
        horizons (list[int]): Liste des horizons futurs à prédire (défaut `[1]`).

    Returns:
        tuple[DataLoader, DataLoader, StandardScaler]:
        `(train_loader, test_loader, scaler)`. Le `scaler` est fitté sur tout
        le dataset et doit être réutilisé à l'inférence pour déstandardiser.
    """
    import os

    if vitesse_matrix_raw.shape[1] != num_nodes:
        print(
            f"[LyonFlow-STGCN] ⚠️ Warning: aligning vitesse_matrix_raw shape from {vitesse_matrix_raw.shape[1]} to {num_nodes} columns."
        )
        default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
        padded_vitesse = np.full((vitesse_matrix_raw.shape[0], num_nodes), default_speed)
        cols_to_copy = min(vitesse_matrix_raw.shape[1], num_nodes)
        padded_vitesse[:, :cols_to_copy] = vitesse_matrix_raw[:, :cols_to_copy]
        vitesse_matrix_raw = padded_vitesse

    scaler = StandardScaler()
    vitesse_matrix = scaler.fit_transform(vitesse_matrix_raw)

    pyg_data_list = []
    num_timestamps = len(vitesse_matrix)

    max_horizon = max(horizons) if horizons else 1

    for t in range(num_timestamps - seq_len - max_horizon + 1):
        speeds = vitesse_matrix[t : t + seq_len, :].T  # shape: [N, SEQ_LEN]
        h_sin_expanded = np.tile(hour_sin[t : t + seq_len], (num_nodes, 1))
        h_cos_expanded = np.tile(hour_cos[t : t + seq_len], (num_nodes, 1))
        d_sin_expanded = np.tile(day_sin[t : t + seq_len], (num_nodes, 1))
        d_cos_expanded = np.tile(day_cos[t : t + seq_len], (num_nodes, 1))

        # X shape: [N, SEQ_LEN, 5] (speed, hour_sin, hour_cos, day_sin, day_cos)
        X = np.stack([speeds, h_sin_expanded, h_cos_expanded, d_sin_expanded, d_cos_expanded], axis=-1)

        # Construction de la cible multi-horizon de forme [N, len(horizons)]
        Y_list = []
        for h in horizons:
            # h pas temporels après le dernier pas d'entrée (t + seq_len - 1)
            y_h = vitesse_matrix[t + seq_len - 1 + h, :].reshape(-1, 1)
            Y_list.append(y_h)
        Y = np.concatenate(Y_list, axis=-1)

        data = Data(
            x=torch.tensor(X, dtype=torch.float), edge_index=edge_index_tensor, y=torch.tensor(Y, dtype=torch.float)
        )
        pyg_data_list.append(data)

    split_idx = int(len(pyg_data_list) * (1 - test_split))

    train_loader = DataLoader(pyg_data_list[:split_idx], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(pyg_data_list[split_idx:], batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def load_graph_topology_from_csv(folder_path):
    """Variante CSV de `load_graph_topology` (mode fichier, sans PostgreSQL).

    Lit `node_mapping.csv` et `edges.csv` produits par
    `utils/export_db_to_csv.py`, puis construit l'`edge_index` PyG de la
    même manière (arêtes bidirectionnelles + self-loops).

    Args:
        folder_path (str): Chemin du dossier contenant les deux CSV.

    Returns:
        tuple[int, torch.Tensor]: `(num_nodes, edge_index)`.
    """
    import os

    mapping_path = os.path.join(folder_path, "node_mapping.csv")
    edges_path = os.path.join(folder_path, "edges.csv")

    df_mapping = pd.read_csv(mapping_path)
    df_mapping = df_mapping.sort_values("node_idx")
    num_nodes = len(df_mapping)

    df_edges = pd.read_csv(edges_path)

    edge_index_list = []
    for _, row in df_edges.iterrows():
        u, v = int(row["node_u"]), int(row["node_v"])
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])  # Non orienté / bidirectionnel
    for i in range(num_nodes):
        edge_index_list.append([i, i])  # Boucles sur soi-même

    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return num_nodes, edge_index_tensor


def load_traffic_series_from_csv(folder_path):
    """Variante CSV de `load_traffic_series` (mode fichier).

    Lit `traffic_series.csv` (produit par `export_db_to_csv.py`), pivote
    en matrice `[T × N]` et calcule les 4 features temporelles cycliques.

    Args:
        folder_path (str): Chemin du dossier contenant `traffic_series.csv`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        `(vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos)`.
    """
    import os

    traffic_path = os.path.join(folder_path, "traffic_series.csv")
    df_facts = pd.read_csv(traffic_path)

    # Tri par timestamp et node_idx pour assurer la cohérence matricielle
    df_facts = df_facts.sort_values(by=["timestamp", "node_idx"])
    df_pivot = df_facts.pivot(index="timestamp", columns="node_idx", values="properties_vitesse")

    default_speed = float(os.getenv("LYON_DEFAULT_SPEED", 30.0))
    vitesse_matrix_raw = np.nan_to_num(df_pivot.values, nan=default_speed)

    # Extraction des variables temporelles cycliques
    df_pivot.index = pd.to_datetime(df_pivot.index)
    hours = df_pivot.index.hour + df_pivot.index.minute / 60.0
    days = df_pivot.index.dayofweek

    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    day_sin = np.sin(2 * np.pi * days / 7.0)
    day_cos = np.cos(2 * np.pi * days / 7.0)

    return vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos
