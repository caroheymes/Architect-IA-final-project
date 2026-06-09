import time
import os
import sys

# Set database URL
os.environ["POSTGRES_USER"] = "lyonflow"
os.environ["POSTGRES_PASSWORD"] = "lyonflow_password"
os.environ["POSTGRES_HOST"] = "postgres"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_DB"] = "lyonflow"

DB_USER = "lyonflow"
DB_PASSWORD = "lyonflow_password"
DB_HOST = "postgres"
DB_PORT = "5432"
DB_DB = "lyonflow"
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

print("1. Profiling database connection check...")
t0 = time.time()
from sqlalchemy import create_engine, text

try:
    engine_check = create_engine(
        DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3, "options": "-c statement_timeout=3000"}
    )
    with engine_check.connect() as conn:
        res = conn.execute(text("SELECT NOW()")).fetchone()
        print(f"   Database check OK in {time.time() - t0:.3f}s: {res[0]}")
except Exception as e:
    print(f"   Database check FAILED in {time.time() - t0:.3f}s: {e}")

print("2. Profiling MLflow client creation...")
t0 = time.time()
try:
    from mlflow.tracking import MlflowClient

    # Simulate internal check
    import socket

    mlflow_host = "mlflow"
    mlflow_port = 5000
    socket.create_connection((mlflow_host, mlflow_port), timeout=1.0)
    print(f"   MLflow socket OK in {time.time() - t0:.3f}s")
except Exception as e:
    print(f"   MLflow socket FAILED in {time.time() - t0:.3f}s: {e}")

print("3. Profiling MLflow runs retrieval...")
t0 = time.time()
try:
    import mlflow

    client = MlflowClient(tracking_uri="http://mlflow:5000")
    runs = client.search_runs(experiment_ids=["6", "7", "8"], order_by=["attribute.start_time DESC"], max_results=30)
    print(f"   Retrieved {len(runs)} MLflow runs in {time.time() - t0:.3f}s")
except Exception as e:
    print(f"   MLflow runs retrieval FAILED in {time.time() - t0:.3f}s: {e}")

print("4. Profiling predictions data loading...")
t0 = time.time()
csv_path = "data/out/predictions_traffic.csv"
if os.path.exists(csv_path):
    import pandas as pd

    df_preds = pd.read_csv(csv_path)
    print(f"   Loaded local predictions CSV in {time.time() - t0:.3f}s. Rows: {len(df_preds)}")
else:
    print("   Local CSV does not exist!")

print("5. Profiling street names loading...")
t0 = time.time()
streets_csv = "data/out/street_names.csv"
if os.path.exists(streets_csv):
    df_streets = pd.read_csv(streets_csv)
    print(f"   Loaded local street names CSV in {time.time() - t0:.3f}s. Rows: {len(df_streets)}")
else:
    print("   Local streets CSV does not exist!")

print("6. Profiling predictions processing (Centroids + Merges)...")
t0 = time.time()
if "df_preds" in locals() and not df_preds.empty:
    df_preds["prediction_timestamp"] = pd.to_datetime(df_preds["prediction_timestamp"])
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
    df_preds["speed_diff"] = df_preds["predicted_speed"] - df_preds["real_speed"]
    latest_run = df_preds["prediction_timestamp"].max()
    df_preds = df_preds[df_preds["prediction_timestamp"] == latest_run].copy()

    # centroid WKT parsing
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

    if "df_streets" in locals() and not df_streets.empty:
        df_streets["properties_twgid"] = df_streets["properties_twgid"].astype(int)
        df_preds["properties_twgid"] = df_preds["properties_twgid"].astype(int)
        df_preds = df_preds.merge(df_streets, on="properties_twgid", how="left")
        df_preds["nom_rue"] = df_preds["properties_libelle"].fillna(
            df_preds["properties_twgid"].apply(lambda x: f"Segment {x}")
        )
    print(f"   Processed predictions in {time.time() - t0:.3f}s. Final Rows: {len(df_preds)}")

print("7. Profiling line geometry preparation...")
t0 = time.time()
import shapely.wkt

max_segments = 1400
df_map_data = df_preds.head(max_segments).copy()
path_data = []
for idx, row in df_map_data.iterrows():
    wkt_str = row.get("geometry_wgs84_wkt")
    if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
        try:
            geom = shapely.wkt.loads(wkt_str)
            coords = [[float(pt[0]), float(pt[1])] for pt in geom.coords]
            path_data.append(
                {
                    "path": coords,
                    "color": [128, 128, 128],
                    "name": str(row.get("nom_rue", "")),
                    "val": float(row.get("predicted_speed", 0)),
                    "val_str": "30.0",
                }
            )
        except Exception:
            pass
print(f"   Prepared {len(path_data)} line geoms in {time.time() - t0:.3f}s")
