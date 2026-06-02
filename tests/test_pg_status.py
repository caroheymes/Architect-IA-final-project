import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("--- Active Connections ---")
    query = text("""
        SELECT pid, usename, client_addr, state, wait_event_type, wait_event, query 
        FROM pg_stat_activity 
        WHERE state IS NOT NULL AND state != 'idle';
    """)
    for row in conn.execute(query).fetchall():
        print(row)

    print("\n--- Locks ---")
    lock_query = text("""
        SELECT t.schemaname, t.relname, l.mode, l.granted, l.pid, a.query
        FROM pg_locks l
        JOIN pg_stat_all_tables t ON l.relation = t.relid
        JOIN pg_stat_activity a ON l.pid = a.pid
        ORDER BY t.schemaname, t.relname;
    """)
    for row in conn.execute(lock_query).fetchall():
        print(row)
