import os
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("Terminating locks and other connections...")
    query = text("""
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE pid != pg_backend_pid() AND state IS NOT NULL;
    """)
    res = conn.execute(query).fetchall()
    print(f"Terminated {len(res)} backend connections.")
