import json
import time

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

GLOBAL_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
_h3shape_cache = {}
_segment_spatial_cache = {}


def h3shape_merge_cached(h3_id_list):
    if not h3_id_list:
        return None
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


def get_speed_category(speed):
    if pd.isna(speed):
        return "Unknown"
    elif speed <= 20:
        return "Slow (0-20 km/h)"
    elif speed > 50:
        return "Fast (>50 km/h)"
    else:
        return "Medium (20-50 km/h)"


def run_profile():
    with engine.begin() as conn:
        snap = conn.execute(
            text("SELECT id, fetched_at, raw_data FROM bronze.trafic_vitesse_brute ORDER BY fetched_at ASC LIMIT 2;")
        ).fetchall()

    # Process Snapshot 1 to warm up cache
    snap1_id, fetched_at1, raw_payload1 = snap[0]
    features = raw_payload1.get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        if props.get("est_a_jour") is False:
            continue
        rows.append(
            {
                "properties_gid": props.get("gid"),
                "geometry_coordinates": geom.get("coordinates"),
                "properties_libelle": props.get("libelle"),
                "properties_sens": props.get("sens"),
                "properties_etat": props.get("etat"),
                "properties_vitesse": props.get("vitesse"),
                "properties_last_update": props.get("last_update"),
                "properties_est_a_jour": props.get("est_a_jour"),
            }
        )
    trafic1 = pd.DataFrame(rows)
    trafic1["geometry_coordinates_obj"] = trafic1["geometry_coordinates"].apply(LineString)

    points_col = []
    hexes_col = []
    merged_geom_col = []
    uncached_idxs = []
    uncached_geoms = []
    for row_idx, (gid, coords) in enumerate(zip(trafic1["properties_gid"], trafic1["geometry_coordinates"])):
        if not coords or len(coords) < 2:
            points_col.append([])
            hexes_col.append([])
            merged_geom_col.append(None)
            continue
        key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
        points_col.append(None)
        hexes_col.append(None)
        merged_geom_col.append(None)
        uncached_idxs.append(row_idx)
        uncached_geoms.append(trafic1["geometry_coordinates_obj"].iloc[row_idx])

    num_pts_per_segment = []
    xs, ys = [], []
    for geom in uncached_geoms:
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
    lons = list(lons)
    lats = list(lats)
    flat_hexes = [h3.latlng_to_cell(lat, lon, 13) for lat, lon in zip(lats, lons)]

    start_idx = 0
    for u_idx, count in zip(uncached_idxs, num_pts_per_segment):
        hex_list = flat_hexes[start_idx : start_idx + count]
        pts_list = [
            [lon, lat] for lon, lat in zip(lons[start_idx : start_idx + count], lats[start_idx : start_idx + count])
        ]
        start_idx += count
        merged_geom = h3shape_merge_cached(hex_list)

        coords = trafic1["geometry_coordinates"].iloc[u_idx]
        gid = trafic1["properties_gid"].iloc[u_idx]
        key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
        _segment_spatial_cache[key] = (pts_list, hex_list, merged_geom)

    print("--- Cache warmed up! ---", flush=True)

    # Process Snapshot 2 and profile each step
    snap2_id, fetched_at2, raw_payload2 = snap[1]
    features = raw_payload2.get("features", [])

    t0 = time.time()
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        if props.get("est_a_jour") is False:
            continue
        rows.append(
            {
                "properties_gid": props.get("gid"),
                "geometry_coordinates": geom.get("coordinates"),
                "properties_libelle": props.get("libelle"),
                "properties_sens": props.get("sens"),
                "properties_etat": props.get("etat"),
                "properties_vitesse": props.get("vitesse"),
                "properties_last_update": props.get("last_update"),
                "properties_est_a_jour": props.get("est_a_jour"),
            }
        )
    print(f"1. Dict parsing: {time.time() - t0:.4f}s", flush=True)

    t0 = time.time()
    trafic = pd.DataFrame(rows)
    trafic["geometry_coordinates_obj"] = trafic["geometry_coordinates"].apply(LineString)
    print(f"2. pd.DataFrame build & LineString: {time.time() - t0:.4f}s", flush=True)

    t0 = time.time()
    points_col = []
    hexes_col = []
    merged_geom_col = []
    uncached_idxs = []
    uncached_geoms = []
    for row_idx, (gid, coords) in enumerate(zip(trafic["properties_gid"], trafic["geometry_coordinates"])):
        if not coords or len(coords) < 2:
            points_col.append([])
            hexes_col.append([])
            merged_geom_col.append(None)
            continue
        key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
        if key in _segment_spatial_cache:
            cached_pts, cached_hexes, cached_geom = _segment_spatial_cache[key]
            points_col.append(cached_pts)
            hexes_col.append(cached_hexes)
            merged_geom_col.append(cached_geom)
        else:
            points_col.append(None)
            hexes_col.append(None)
            merged_geom_col.append(None)
            uncached_idxs.append(row_idx)
            uncached_geoms.append(trafic["geometry_coordinates_obj"].iloc[row_idx])
    print(
        f"3. Cache lookup ({len(trafic) - len(uncached_idxs)} hits, {len(uncached_idxs)} misses): {time.time() - t0:.4f}s",
        flush=True,
    )

    t0 = time.time()
    if uncached_geoms:
        num_pts_per_segment = []
        xs, ys = [], []
        for geom in uncached_geoms:
            if not geom or geom.is_empty:
                num_pts_per_segment.append(0)
                continue
            length = geom.length
            dists = np.arange(0, length, 7)
            if length % 7 != 0:
                dists = np.append(dists, length)
            pts = geom.interpolate(dists)
            num_pts_per_segment.append(len(pts))
            for p in pts:
                xs.append(p.x)
                ys.append(p.y)

        if xs:
            lons, lats = GLOBAL_TRANSFORMER(xs, ys)
            lons = list(lons)
            lats = list(lats)
            flat_hexes = [h3.latlng_to_cell(lat, lon, 13) for lat, lon in zip(lats, lons)]
        else:
            lons, lats = [], []
            flat_hexes = []

        start_idx = 0
        for u_idx, count in zip(uncached_idxs, num_pts_per_segment):
            if count == 0:
                pts_list = []
                hex_list = []
            else:
                hex_list = flat_hexes[start_idx : start_idx + count]
                pts_list = [
                    [lon, lat]
                    for lon, lat in zip(lons[start_idx : start_idx + count], lats[start_idx : start_idx + count])
                ]
                start_idx += count

            merged_geom = h3shape_merge_cached(hex_list)
            points_col[u_idx] = pts_list
            hexes_col[u_idx] = hex_list
            merged_geom_col[u_idx] = merged_geom

            coords = trafic["geometry_coordinates"].iloc[u_idx]
            gid = trafic["properties_gid"].iloc[u_idx]
            key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
            _segment_spatial_cache[key] = (pts_list, hex_list, merged_geom)
    print(f"4. Interpolation / H3 mapping: {time.time() - t0:.4f}s", flush=True)

    t0 = time.time()
    trafic["points"] = points_col
    trafic["hexes"] = hexes_col
    trafic["merged_h3_geometry"] = merged_geom_col

    trafic["properties_vitesse"] = trafic["properties_vitesse"].astype(str).str.split(" ").str[0]
    trafic["properties_vitesse"] = pd.to_numeric(trafic["properties_vitesse"], errors="coerce")
    mean_speed_df = trafic.groupby(by="properties_libelle").agg(mean_speed=("properties_vitesse", "mean")).reset_index()
    trafic = trafic.merge(mean_speed_df, on="properties_libelle", how="left")
    trafic["properties_vitesse"] = [
        elem if not pd.isna(elem) else mean_speed if not pd.isna(mean_speed) else np.nan
        for elem, mean_speed in zip(trafic.properties_vitesse, trafic.mean_speed)
    ]
    trafic = trafic.drop(columns=["mean_speed"], errors="ignore")
    trafic["speed_category"] = [get_speed_category(speed) for speed in trafic.properties_vitesse]
    trafic["speed_color_map"] = trafic["speed_category"].map(
        {"Slow (0-20 km/h)": "red", "Medium (20-50 km/h)": "orange", "Fast (>50 km/h)": "green", "Unknown": "gray"}
    )
    print(f"5. Speed sanitization & mapping: {time.time() - t0:.4f}s", flush=True)

    t0 = time.time()
    gdf = GeoDataFrame(data=trafic, geometry="geometry_coordinates_obj")
    gdf.set_crs(epsg=2154, inplace=True, allow_override=True)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    print(f"6. GeoDataFrame CRS transform: {time.time() - t0:.4f}s", flush=True)

    t0 = time.time()
    df_silver = pd.DataFrame(gdf_wgs84)
    df_silver["id_rue"] = df_silver.index
    df_silver["geometry_wgs84_wkt"] = df_silver["geometry_coordinates_obj"].apply(
        lambda geom: geom.wkt if geom else None
    )
    df_silver["points_json"] = df_silver["points"].apply(lambda lst: json.dumps(lst) if lst else None)
    df_silver["hexes_json"] = df_silver["hexes"].apply(lambda lst: json.dumps(lst) if lst else None)
    df_silver["merged_h3_geometry_json"] = df_silver["merged_h3_geometry"].apply(
        lambda geom: json.dumps(geom.__geo_interface__) if geom else None
    )
    print(f"7. Serialization (WKT/JSON): {time.time() - t0:.4f}s", flush=True)


if __name__ == "__main__":
    run_profile()
