import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("Columns of silver.trafic_vitesse_propre:")
    query = text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema='silver' AND table_name='trafic_vitesse_propre';
    """)
    for r in conn.execute(query).fetchall():
        print(f"  {r[0]}: {r[1]}")
