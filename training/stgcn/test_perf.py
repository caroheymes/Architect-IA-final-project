# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import torch
import datetime

num_nodes = 1536
seq_len = 120
in_channels = 5
out_channels = 3 # horizons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Mock GNN
from model import SpatioTemporalGCN
model = SpatioTemporalGCN(in_channels=in_channels, hidden_channels=128, out_channels=out_channels).to(device)
model.eval()

edge_index = torch.randint(0, num_nodes, (2, 9000), dtype=torch.long).to(device)

x_tensor = torch.randn(num_nodes, seq_len, in_channels).to(device)

# Benchmark forward pass
t0 = time.time()
with torch.no_grad():
    for _ in range(20):
        out = model(x_tensor, edge_index)
t1 = time.time()
print(f"GNN forward pass (20 iterations): {t1 - t0:.4f} seconds (Avg: {(t1 - t0)/20:.4f}s)")

# Benchmark loop
node_twgids = np.random.randint(10000, 99999, num_nodes)
node_geoms = np.array(["LINESTRING(0 0, 1 1)"] * num_nodes, dtype=object)
latest_vitesse = np.random.rand(seq_len, num_nodes) * 50
preds_kmh = np.random.rand(num_nodes, out_channels) * 50
HORIZONS = [6, 12, 36]
deltas = [datetime.timedelta(minutes=h * 5) for h in HORIZONS]
ts = datetime.datetime.now()

all_predictions_records = []

t0 = time.time()
for _ in range(10): # 10 timestamps
    for node_idx in range(num_nodes):
        twgid = int(node_twgids[node_idx])
        geom = node_geoms[node_idx]
        if pd.isna(geom):
            geom = None
        last_real_speed = float(latest_vitesse[-1, node_idx])

        for h_idx, h in enumerate(HORIZONS):
            horizon_minutes = h * 5
            target_time = ts + deltas[h_idx]
            predicted_speed = float(preds_kmh[node_idx, h_idx])

            all_predictions_records.append({
                "prediction_timestamp": ts,
                "target_timestamp": target_time,
                "horizon_minutes": horizon_minutes,
                "node_idx": node_idx,
                "properties_twgid": twgid,
                "predicted_speed": round(predicted_speed, 2),
                "real_speed": round(last_real_speed, 2),
                "geometry_wgs84_wkt": geom
            })
t1 = time.time()
print(f"Loop (10 timestamps, unoptimized): {t1 - t0:.4f} seconds (Avg: {(t1 - t0)/10:.4f}s)")

# Optimized loop
all_predictions_records = []
t0 = time.time()
for _ in range(10): # 10 timestamps
    for node_idx in range(num_nodes):
        twgid = int(node_twgids[node_idx])
        geom = node_geoms[node_idx]
        if geom == "":
            geom = None
        last_real_speed = float(latest_vitesse[-1, node_idx])

        for h_idx, h in enumerate(HORIZONS):
            target_time = ts + deltas[h_idx]
            predicted_speed = float(preds_kmh[node_idx, h_idx])

            all_predictions_records.append({
                "prediction_timestamp": ts,
                "target_timestamp": target_time,
                "horizon_minutes": HORIZONS[h_idx] * 5,
                "node_idx": node_idx,
                "properties_twgid": twgid,
                "predicted_speed": round(predicted_speed, 2),
                "real_speed": round(last_real_speed, 2),
                "geometry_wgs84_wkt": geom
            })
t1 = time.time()
print(f"Loop (10 timestamps, optimized): {t1 - t0:.4f} seconds (Avg: {(t1 - t0)/10:.4f}s)")
