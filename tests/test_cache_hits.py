import json
import logging
import os

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform, unary_union
from sqlalchemy import create_engine, text

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_USER = "lyonflow"
DB_PASSWORD = "lyonflow_password"
DB_HOST = "postgres"
DB_PORT = "5432"
DB_DB = "lyonflow"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"
GLOBAL_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform

_h3shape_cache = {}
_segment_spatial_cache = {}


def h3shape_merge_cached(h3_id_list):
    """Variante cachée de la fusion de cellules H3 (cf. autres modules).

    Args:
        h3_id_list (list[str]): Liste d'IDs H3 (rés. 13).

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


def test_runs():
    """Mesure le hit rate du cache spatial `_segment_spatial_cache` sur les 5 plus vieux snapshots Bronze.

    Pour chaque snapshot :
      1. Charge les features.
      2. Filtre `est_a_jour != False`, construit le DataFrame.
      3. Pour chaque segment, tente de récupérer
         `(points, hexes, merged_geom)` dans le cache (clé = `gid` ou
         tuple de coords).
      4. Sur cache miss, recalcule l'interpolation, l'indexation H3 et la
         fusion, puis stocke le résultat dans le cache.
      5. Log un récapitulatif `hits / misses / hit_rate` par snapshot.

    Sert à valider que le gain x350 observé par `profile_rebuild.py` est
    bien réel sur des snapshots consécutifs (test terrain).
    """
    engine = create_engine(DATABASE_URL)
    query = text("SELECT id, fetched_at, raw_data FROM bronze.trafic_vitesse_brute ORDER BY fetched_at ASC LIMIT 5;")
    with engine.begin() as conn:
        snapshots = conn.execute(query).fetchall()

    logger.info(f"Loaded {len(snapshots)} snapshots for testing.")

    for idx, snap in enumerate(snapshots, 1):
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
                }
            )

        trafic = pd.DataFrame(rows)
        trafic["geometry_coordinates_obj"] = trafic["geometry_coordinates"].apply(LineString)

        # Track cache hits
        hits = 0
        misses = 0

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

            # Key options
            # Options 1: GID
            # Options 2: Coordinates Tuple
            key = gid if gid is not None else tuple(tuple(pt) for pt in coords)

            if key in _segment_spatial_cache:
                cached_pts, cached_hexes, cached_geom = _segment_spatial_cache[key]
                points_col.append(cached_pts)
                hexes_col.append(cached_hexes)
                merged_geom_col.append(cached_geom)
                hits += 1
            else:
                points_col.append(None)
                hexes_col.append(None)
                merged_geom_col.append(None)

                uncached_idxs.append(row_idx)
                uncached_geoms.append(trafic["geometry_coordinates_obj"].iloc[row_idx])
                misses += 1

        logger.info(
            f"Snapshot {snap_id}: total_segments={len(trafic)}, hits={hits}, misses={misses} (Hit Rate: {hits / (hits + misses) * 100:.2f}%)"
        )

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


if __name__ == "__main__":
    test_runs()
