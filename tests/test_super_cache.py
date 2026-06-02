import json
import time

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform, unary_union
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

GLOBAL_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
_h3shape_cache = {}
_super_cache = {}  # maps GID to (geometry_wgs84_wkt, points_json, hexes_json, merged_h3_geometry_json)


def h3shape_merge_cached(h3_id_list):
    """Variante cachée de la fusion H3 — voir `profile_rebuild.py`.

    Args:
        h3_id_list (list[str]): Identifiants H3 (rés. 13).

    Returns:
        shapely.geometry.Polygon | None: Polygone fusionné, ou `None`.
    """
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


def run_test():
    """Mesure les performances du **Super Cache** sur les 5 plus vieux snapshots Bronze.

    Le Super Cache `_super_cache` stocke par segment la valeur **déjà
    sérialisée** (`wkt`, `points_json`, `hexes_json`, `merged_h3_geometry_json`),
    ce qui permet sur cache hit d'éviter la reprojection et la sérialisation.

    Pour chaque snapshot, on :
      1. charge les features ;
      2. tente de récupérer la ligne pré-sérialisée dans `_super_cache` ;
      3. sur miss, recalcule la géométrie WGS84 + interpolation + H3 +
         sérialisation, puis peuple le cache ;
      4. log le temps total et les compteurs `hits/misses`.

    Le but est de mesurer le speedup réel du Super Cache vs la version
    sans cache (cf. `profile_rebuild.py`).
    """
    with engine.begin() as conn:
        snapshots = conn.execute(
            text("SELECT id, fetched_at, raw_data FROM bronze.trafic_vitesse_brute ORDER BY fetched_at ASC LIMIT 5;")
        ).fetchall()

    print(f"Loaded {len(snapshots)} snapshots for test.", flush=True)

    for idx, snap in enumerate(snapshots, 1):
        t_start = time.time()
        snap_id, fetched_at, raw_payload = snap
        features = raw_payload.get("features", [])

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

        trafic = pd.DataFrame(rows)

        # We need to construct:
        # geometry_wgs84_wkt, points_json, hexes_json, merged_h3_geometry_json

        geometry_wgs84_wkt_col = []
        points_json_col = []
        hexes_json_col = []
        merged_h3_geometry_json_col = []

        uncached_idxs = []

        # Warm cache lookup
        for row_idx, (gid, coords) in enumerate(zip(trafic["properties_gid"], trafic["geometry_coordinates"])):
            if not coords or len(coords) < 2:
                geometry_wgs84_wkt_col.append(None)
                points_json_col.append(None)
                hexes_json_col.append(None)
                merged_h3_geometry_json_col.append(None)
                continue

            key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
            if key in _super_cache:
                wkt, pts_js, hex_js, merged_js = _super_cache[key]
                geometry_wgs84_wkt_col.append(wkt)
                points_json_col.append(pts_js)
                hexes_json_col.append(hex_js)
                merged_h3_geometry_json_col.append(merged_js)
            else:
                geometry_wgs84_wkt_col.append(None)
                points_json_col.append(None)
                hexes_json_col.append(None)
                merged_h3_geometry_json_col.append(None)
                uncached_idxs.append(row_idx)

        hits = len(trafic) - len(uncached_idxs)
        print(f"Snapshot {idx}: total={len(trafic)}, hits={hits}, misses={len(uncached_idxs)}", flush=True)

        if uncached_idxs:
            # For uncached rows, we do the full interpolation and projection
            uncached_geoms = [LineString(trafic["geometry_coordinates"].iloc[i]) for i in uncached_idxs]

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

            if xs:
                lons, lats = GLOBAL_TRANSFORMER(xs, ys)
                lons = list(lons)
                lats = list(lats)
                flat_hexes = [h3.latlng_to_cell(lat, lon, 13) for lat, lon in zip(lats, lons)]
            else:
                lons, lats = [], []
                flat_hexes = []

            start_idx = 0
            for u_idx, count, geom_local in zip(uncached_idxs, num_pts_per_segment, uncached_geoms):
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

                # Project the segment LineString to WGS84 directly with shapely.ops.transform
                # This is extremely fast because we are projecting just the LineString geometry object
                geom_wgs84 = transform(GLOBAL_TRANSFORMER, geom_local)

                # Pre-serialize to strings
                wkt = geom_wgs84.wkt
                pts_js = json.dumps(pts_list)
                hex_js = json.dumps(hex_list)
                merged_js = json.dumps(merged_geom.__geo_interface__) if merged_geom else None

                geometry_wgs84_wkt_col[u_idx] = wkt
                points_json_col[u_idx] = pts_js
                hexes_json_col[u_idx] = hex_js
                merged_h3_geometry_json_col[u_idx] = merged_js

                coords = trafic["geometry_coordinates"].iloc[u_idx]
                gid = trafic["properties_gid"].iloc[u_idx]
                key = gid if gid is not None else tuple(tuple(pt) for pt in coords)
                _super_cache[key] = (wkt, pts_js, hex_js, merged_js)

        # Fill Columns directly with our string lists (already serialized!)
        trafic["geometry_wgs84_wkt"] = geometry_wgs84_wkt_col
        trafic["points_json"] = points_json_col
        trafic["hexes_json"] = hexes_json_col
        trafic["merged_h3_geometry_json"] = merged_h3_geometry_json_col

        # Clean speed and metadata mapping
        trafic["properties_vitesse"] = trafic["properties_vitesse"].astype(str).str.split(" ").str[0]
        trafic["properties_vitesse"] = pd.to_numeric(trafic["properties_vitesse"], errors="coerce")
        # Mean speed filled
        mean_speed_df = (
            trafic.groupby(by="properties_libelle").agg(mean_speed=("properties_vitesse", "mean")).reset_index()
        )
        trafic = trafic.merge(mean_speed_df, on="properties_libelle", how="left")
        trafic["properties_vitesse"] = [
            elem if not pd.isna(elem) else mean_speed if not pd.isna(mean_speed) else np.nan
            for elem, mean_speed in zip(trafic.properties_vitesse, trafic.mean_speed)
        ]
        trafic = trafic.drop(columns=["mean_speed"], errors="ignore")

        print(f"Snapshot {idx} completed in {time.time() - t_start:.4f} seconds!", flush=True)


if __name__ == "__main__":
    run_test()
