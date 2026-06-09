import time
import os

print("0. Starting step-by-step profiling...")
t_start = time.time()

# 1. Imports
t0 = time.time()
import numpy as np
import pandas as pd
import streamlit as st

print(f"   Step 1: Imports took {time.time() - t0:.3f}s")

# 2. Page config (simulate or run)
t0 = time.time()
try:
    st.set_page_config(page_title="LyonFlow - Traffic Prediction", page_icon="🚦", layout="wide")
    print(f"   Step 2: Page config took {time.time() - t0:.3f}s")
except Exception as e:
    print(f"   Step 2: Page config failed (expected in bare mode) in {time.time() - t0:.3f}s: {e}")

# 3. CSS markdown
t0 = time.time()
try:
    st.markdown("CSS STYLING PLACEHOLDER")
    print(f"   Step 3: CSS markdown took {time.time() - t0:.3f}s")
except Exception as e:
    print(f"   Step 3: CSS failed in {time.time() - t0:.3f}s: {e}")

# 4. Functions definitions (simulated by executing import / parse)
t0 = time.time()
# (We don't need to do anything, function def is instantaneous in Python)
print(f"   Step 4: Function definitions took {time.time() - t0:.3f}s")

# 5. Load predictions cached (the actual core call!)
t0 = time.time()
csv_path = "data/out/predictions_traffic.csv"
streets_path = "data/out/street_names.csv"

# Inline load_predictions_data
df_preds = None
if os.path.exists(csv_path):
    df_preds = pd.read_csv(csv_path)
print(
    f"   Step 5a: Load predictions CSV took {time.time() - t0:.3f}s (Rows: {len(df_preds) if df_preds is not None else 0})"
)

# Inline load_street_names
t_sub = time.time()
df_streets = pd.DataFrame(columns=["properties_twgid", "properties_libelle"])
if os.path.exists(streets_path):
    df_streets = pd.read_csv(streets_path)
    df_streets["properties_twgid"] = df_streets["properties_twgid"].astype(int)
    df_streets = df_streets[["properties_twgid", "properties_libelle"]]
print(f"   Step 5b: Load street names CSV took {time.time() - t_sub:.3f}s (Rows: {len(df_streets)})")

# Inline processing
t_sub = time.time()
if df_preds is not None and not df_preds.empty:
    df_preds["prediction_timestamp"] = pd.to_datetime(df_preds["prediction_timestamp"])
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
    df_preds["speed_diff"] = df_preds["predicted_speed"] - df_preds["real_speed"]
    latest_run = df_preds["prediction_timestamp"].max()
    df_preds = df_preds[df_preds["prediction_timestamp"] == latest_run].copy()

    def parse_wkt_centroid(wkt_str):
        try:
            if not wkt_str or not isinstance(wkt_str, str):
                return None, None
            content = wkt_str.replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
            coords = [c.strip().split() for c in content.split(",")]
            lats = [float(c[1]) for c in coords if len(c) >= 2]
            lons = [float(c[0]) for c in coords if len(c) >= 2]
            if lats and lons:
                return sum(lats) / len(lats), sum(lons) / len(lons)
        except Exception:
            pass
        return None, None

    centroids = df_preds["geometry_wgs84_wkt"].apply(parse_wkt_centroid)
    df_preds["latitude"] = [c[0] for c in centroids]
    df_preds["longitude"] = [c[1] for c in centroids]
    df_preds = df_preds.dropna(subset=["latitude", "longitude"])

    df_preds["properties_twgid"] = df_preds["properties_twgid"].astype(int)
    df_preds = df_preds.merge(df_streets, on="properties_twgid", how="left")
    df_preds["nom_rue"] = df_preds["properties_libelle"].fillna(
        df_preds["properties_twgid"].apply(lambda x: f"Segment {x}")
    )

    def parse_linestring_coords(wkt_str):
        try:
            if not wkt_str or not isinstance(wkt_str, str):
                return None
            content = wkt_str.replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
            coords = [c.strip().split() for c in content.split(",")]
            return [[float(c[0]), float(c[1])] for c in coords if len(c) >= 2]
        except Exception:
            return None

    df_preds["parsed_path_coords"] = df_preds["geometry_wgs84_wkt"].apply(parse_linestring_coords)
print(
    f"   Step 5c: Predictions processing took {time.time() - t_sub:.3f}s (Final Rows: {len(df_preds) if df_preds is not None else 0})"
)

# 6. Tab Obs and Evidently metrics JSON parsing
t0 = time.time()
report_json_path = "data/out/monitoring_metrics_morning.json"
mae_val = None
if os.path.exists(report_json_path):
    import json

    with open(report_json_path, "r", encoding="utf-8") as fj:
        metrics_data = json.load(fj)
print(f"   Step 6: Evidently metrics parsing took {time.time() - t0:.3f}s")

# 7. Rendering simulations (we simulate the components that might run or be created)
t0 = time.time()
import pydeck as pdk
import plotly.express as px
import shapely.wkt

print(f"   Step 7: Render imports took {time.time() - t0:.3f}s")

print(f"Total profiling completed in {time.time() - t_start:.3f}s")
