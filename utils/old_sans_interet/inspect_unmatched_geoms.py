import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("--- Inspecting a few unmatched rows in silver ---")
    query = text("""
        SELECT id_rue, properties_libelle, geometry_wgs84_wkt 
        FROM silver.trafic_vitesse_propre 
        WHERE properties_gid IS NULL 
        LIMIT 5;
    """)
    for r in conn.execute(query).fetchall():
        print(f"ID Rue: {r[0]}, Libelle: {r[1]}, WKT: {r[2][:80]}...")
