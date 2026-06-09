import time
import pandas as pd
import shapely.wkt

print("Starting benchmark...")
t0 = time.time()
df = pd.read_csv("data/out/predictions_traffic.csv")
print(f"Loaded CSV in {time.time() - t0:.3f} seconds. Rows: {len(df)}")

# Filter the latest prediction_timestamp
t0 = time.time()
df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
latest = df["prediction_timestamp"].max()
df_latest = df[df["prediction_timestamp"] == latest].copy()
print(f"Filtered latest run in {time.time() - t0:.3f} seconds. Rows: {len(df_latest)}")

# Benchmark WKT Centroid parsing
t0 = time.time()


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


centroids = df_latest["geometry_wgs84_wkt"].apply(parse_wkt_centroid)
print(f"Parsed {len(centroids)} centroids in {time.time() - t0:.3f} seconds")

# Benchmark Shapely loads on 1400 segments
t0 = time.time()
sample_wkt = df_latest["geometry_wgs84_wkt"].dropna().head(1400).tolist()
path_data = []
for wkt_str in sample_wkt:
    if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
        try:
            geom = shapely.wkt.loads(wkt_str)
            coords = [[float(pt[0]), float(pt[1])] for pt in geom.coords]
            path_data.append(coords)
        except Exception:
            pass
print(f"Parsed 1400 Shapely LINESTRINGs in {time.time() - t0:.3f} seconds")

# Benchmark string-based parsing on 1400 segments
t0 = time.time()


def parse_linestring_coords(wkt_str):
    try:
        if not wkt_str or not isinstance(wkt_str, str):
            return None
        content = wkt_str.replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
        coords = [c.strip().split() for c in content.split(",")]
        return [[float(c[0]), float(c[1])] for c in coords if len(c) >= 2]
    except Exception:
        return None


path_data_str = []
for wkt_str in sample_wkt:
    coords = parse_linestring_coords(wkt_str)
    if coords:
        path_data_str.append(coords)
print(f"Parsed 1400 string-based LINESTRINGs in {time.time() - t0:.3f} seconds")
