import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("--- Searching for PONT DE LA FEUILLEE in silver table ---")
    rows_silver = conn.execute(
        text("""
        SELECT DISTINCT geometry_wgs84_wkt, properties_gid, properties_twgid
        FROM silver.trafic_vitesse_propre 
        WHERE properties_libelle = 'PONT DE LA FEUILLEE';
    """)
    ).fetchall()
    for r in rows_silver:
        print(f"Silver - GID: {r[1]}, TWGID: {r[2]}, WKT: {r[0][:80]}...")

    print("\n--- Searching for PONT DE LA FEUILLEE in ref_segments ---")
    # Let's see if there is any row in ref_segments that matches the libelle if we can extract it from bronze
    # Wait, ref_segments does not have libelle, but we can see if its coordinates exist in bronze snapshots
    # Let's search inside the bronze table JSON payloads
    rows_bronze = conn.execute(
        text("""
        SELECT DISTINCT 
            (f->'properties'->>'libelle') as libelle,
            (f->'properties'->>'gid') as gid,
            (f->'properties'->>'twgid') as twgid,
            (f->'geometry'->'coordinates') as coords
        FROM (
            SELECT jsonb_array_elements(raw_data->'features') as f
            FROM bronze.trafic_vitesse_brute
        ) t
        WHERE f->'properties'->>'libelle' = 'PONT DE LA FEUILLEE';
    """)
    ).fetchall()
    for r in rows_bronze:
        print(f"Bronze - GID: {r[1]}, TWGID: {r[2]}, Coords count: {len(r[3]) if r[3] else 0}")
