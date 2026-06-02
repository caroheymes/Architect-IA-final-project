import time

import h3
import numpy as np
import pyproj
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union
from sqlalchemy import create_engine, text

GLOBAL_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform

_h3shape_cache = {}


def h3shape_merge_cached(h3_id_list):
    if not h3_id_list:
        return None
    # Sort to ensure tuple key is stable
    unique_hexes = sorted(list(set(h3_id_list)))
    key = tuple(unique_hexes)
    if key in _h3shape_cache:
        return _h3shape_cache[key]

    try:
        val = shape(h3.cells_to_h3shape(unique_hexes))
    except Exception:
        polygons = []
        for h in unique_hexes:
            boundary = h3.cell_to_boundary(h)
            polygons.append(Polygon([(lon, lat) for lat, lon in boundary]))
        val = unary_union(polygons)

    _h3shape_cache[key] = val
    return val


engine = create_engine("postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow")
with engine.connect() as conn:
    snap = conn.execute(text("SELECT raw_data FROM bronze.trafic_vitesse_brute LIMIT 1")).fetchone()
    raw_payload = snap[0]
    features = raw_payload.get("features", [])
    geoms = [LineString(f["geometry"]["coordinates"]) for f in features if f.get("geometry")]

    print("Running pipeline to get H3 cells...", flush=True)
    num_pts_per_segment = []
    xs, ys = [], []
    for geom in geoms:
        length = geom.length
        dists = np.arange(0, length, 7)
        if length % 7 != 0:
            dists = np.append(dists, length)
        pts = geom.interpolate(dists)
        num_pts_per_segment.append(len(pts))
        for p in pts:
            xs.append(p.x)
            ys.append(p.y)

    lons, lats = GLOBAL_TRANSFORMER(xs, ys)
    flat_hexes = [h3.latlng_to_cell(lat, lon, 13) for lat, lon in zip(lats, lons)]

    hexes_per_segment = []
    start_idx = 0
    for count in num_pts_per_segment:
        hexes_per_segment.append(flat_hexes[start_idx : start_idx + count])
        start_idx += count

    print("\n--- Running 1st Snapshot (Cache cold) ---", flush=True)
    t0 = time.time()
    results_1 = [h3shape_merge_cached(h) for h in hexes_per_segment]
    print(f"Cold run took {time.time() - t0:.4f}s", flush=True)

    print("\n--- Running 2nd Snapshot (Cache hot) ---", flush=True)
    t0 = time.time()
    results_2 = [h3shape_merge_cached(h) for h in hexes_per_segment]
    print(f"Hot run took {time.time() - t0:.4f}s", flush=True)
