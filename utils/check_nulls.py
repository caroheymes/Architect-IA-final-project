import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("Checking counts in silver table...")
    total = conn.execute(text("SELECT count(*) FROM silver.trafic_vitesse_propre")).fetchone()[0]
    null_gid = conn.execute(
        text("SELECT count(*) FROM silver.trafic_vitesse_propre WHERE properties_gid IS NULL")
    ).fetchone()[0]
    null_twgid = conn.execute(
        text("SELECT count(*) FROM silver.trafic_vitesse_propre WHERE properties_twgid IS NULL")
    ).fetchone()[0]

    print(f"Total rows: {total}")
    print(f"Rows with properties_gid IS NULL: {null_gid}")
    print(f"Rows with properties_twgid IS NULL: {null_twgid}")
