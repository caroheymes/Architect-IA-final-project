import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

def load_graph_topology(engine):
    """
    Loads nodes mapping and GNN adjacency from PostgreSQL gold layer
    and builds PyTorch Geometric edge index with self-loops.
    """
    df_mapping = pd.read_sql(
        "SELECT node_idx, properties_twgid FROM gold.dim_spatial_grid_mapping ORDER BY node_idx ASC;", 
        con=engine
    )
    num_nodes = len(df_mapping)
    
    df_edges = pd.read_sql(
        "SELECT node_u, node_v FROM gold.dim_gnn_adjacency;", 
        con=engine
    )
    
    edge_index_list = []
    for _, row in df_edges.iterrows():
        u, v = int(row["node_u"]), int(row["node_v"])
        edge_index_list.append([u, v])
        edge_index_list.append([v, u]) # undirected / bidirectional
    for i in range(num_nodes):
        edge_index_list.append([i, i]) # self-loops
        
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return num_nodes, edge_index_tensor

def load_traffic_series(engine):
    """
    Loads full speed facts series, pivots them into a (timestamps, nodes) matrix,
    and extracts cyclical temporal features (sine/cosine of hour and day of week).
    """
    df_facts = pd.read_sql(
        "SELECT timestamp, node_idx, properties_vitesse FROM gold.fact_traffic_series ORDER BY timestamp ASC, node_idx ASC;", 
        con=engine
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

def build_sliding_dataset(vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos, seq_len, edge_index_tensor, num_nodes, test_split=0.2, batch_size=16):
    """
    Builds a sliding window Spatio-Temporal PyTorch Geometric dataset.
    Splits chronologically (e.g. 80/20) and returns train/test DataLoaders and the fitted StandardScaler.
    """
    scaler = StandardScaler()
    vitesse_matrix = scaler.fit_transform(vitesse_matrix_raw)
    
    pyg_data_list = []
    num_timestamps = len(vitesse_matrix)
    for t in range(num_timestamps - seq_len):
        speeds = vitesse_matrix[t : t + seq_len, :].T # shape: [N, SEQ_LEN]
        h_sin_expanded = np.tile(hour_sin[t : t + seq_len], (num_nodes, 1))
        h_cos_expanded = np.tile(hour_cos[t : t + seq_len], (num_nodes, 1))
        d_sin_expanded = np.tile(day_sin[t : t + seq_len], (num_nodes, 1))
        d_cos_expanded = np.tile(day_cos[t : t + seq_len], (num_nodes, 1))
        
        # X shape: [N, SEQ_LEN, 5] (speed, hour_sin, hour_cos, day_sin, day_cos)
        X = np.stack([speeds, h_sin_expanded, h_cos_expanded, d_sin_expanded, d_cos_expanded], axis=-1)
        Y = vitesse_matrix[t + seq_len, :].reshape(-1, 1) # shape: [N, 1]
        
        data = Data(
            x=torch.tensor(X, dtype=torch.float), 
            edge_index=edge_index_tensor, 
            y=torch.tensor(Y, dtype=torch.float)
        )
        pyg_data_list.append(data)
        
    split_idx = int(len(pyg_data_list) * (1 - test_split))
    
    train_loader = DataLoader(pyg_data_list[:split_idx], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(pyg_data_list[split_idx:], batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler
