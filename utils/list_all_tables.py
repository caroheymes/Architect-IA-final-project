import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("--- Listing All Tables in Database ---")
    query = text("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """)
    for r in conn.execute(query).fetchall():
        print(f"Schema: {r[0]}, Table: {r[1]}")

    print("\n--- Listing Columns for bronze.trafic_vitesse_brute ---")
    bronze_cols = conn.execute(
        text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema='bronze' AND table_name='trafic_vitesse_brute';
    """)
    ).fetchall()
    for r in bronze_cols:
        print(f"  {r[0]}: {r[1]}")
