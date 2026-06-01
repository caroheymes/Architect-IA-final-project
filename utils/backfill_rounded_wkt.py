import os
import pandas as pd
import shapely.wkt
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

def psql_insert_execute_values(table, conn, keys, data_iter):
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        columns = ', '.join([f'"{k}"' for k in keys])
        if table.schema:
            table_name = f'"{table.schema}"."{table.name}"'
        else:
            table_name = f'"{table.name}"'
        sql = f'INSERT INTO {table_name} ({columns}) VALUES %s'
        execute_values(cur, sql, list(data_iter))

def round_wkt(wkt_str):
    if not wkt_str:
        return None
    try:
        geom = shapely.wkt.loads(wkt_str)
        return shapely.wkt.dumps(geom, rounding_precision=6)
    except Exception:
        return None

def backfill_with_rounded_wkt():
    print("Connecting to PostgreSQL database...")
    
    # 1. Fetch ref_segments and compute rounded WKT in Python
    with engine.connect() as conn:
        print("Reading ref_segments...")
        df_ref = pd.read_sql("SELECT geometry_wgs84_wkt, properties_twgid, properties_gid FROM silver.ref_segments", con=conn)
        
    print(f"Loaded {len(df_ref)} reference segments. Computing rounded WKT...")
    df_ref['wkt_rounded_6'] = df_ref['geometry_wgs84_wkt'].apply(round_wkt)
    
    # Deduplicate reference segments by the rounded WKT to avoid duplicate keys in join
    df_ref_clean = df_ref.dropna(subset=['wkt_rounded_6']).drop_duplicates(subset=['wkt_rounded_6']).copy()
    print(f"Deduplicated reference segments: {len(df_ref_clean)} unique rounded WKT keys.")
    
    # Write rounded reference table
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS silver.ref_segments_rounded;"))
        df_ref_clean[['wkt_rounded_6', 'properties_twgid', 'properties_gid']].to_sql(
            name="ref_segments_rounded",
            con=conn,
            schema="silver",
            if_exists="append",
            index=False,
            method=psql_insert_execute_values
        )
    print("🟢 Created silver.ref_segments_rounded reference table.")
    
    # 2. Fetch all silver table rows that have NULL properties_gid and compute their rounded WKT
    with engine.connect() as conn:
        print("Reading unmatched rows from silver table...")
        df_silver = pd.read_sql("""
            SELECT id_rue, geometry_wgs84_wkt 
            FROM silver.trafic_vitesse_propre 
            WHERE properties_gid IS NULL;
        """, con=conn)
        
    if df_silver.empty:
        print("No unmatched rows found in silver table.")
        return
        
    print(f"Loaded {len(df_silver)} unmatched rows from silver. Computing rounded WKT...")
    df_silver['wkt_rounded_6'] = df_silver['geometry_wgs84_wkt'].apply(round_wkt)
    
    # Create temp table in postgres to host silver rounded keys
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS silver.temp_silver_rounded;"))
        df_silver[['id_rue', 'wkt_rounded_6']].to_sql(
            name="temp_silver_rounded",
            con=conn,
            schema="silver",
            if_exists="append",
            index=False,
            method=psql_insert_execute_values
        )
    print("🟢 Created silver.temp_silver_rounded table.")
    
    # 3. Perform the UPDATE JOIN using rounded WKT strings
    with engine.begin() as conn:
        print("Executing SQL UPDATE JOIN with rounded WKT...")
        update_res = conn.execute(text("""
            UPDATE silver.trafic_vitesse_propre s
            SET 
                properties_gid = m.properties_gid,
                properties_twgid = m.properties_twgid
            FROM silver.temp_silver_rounded t
            JOIN silver.ref_segments_rounded m ON t.wkt_rounded_6 = m.wkt_rounded_6
            WHERE s.id_rue = t.id_rue;
        """))
        print(f"🟢 Successfully updated {update_res.rowcount} unmatched rows using rounded WKT join!")
        
        # Cleanup temporary tables
        print("Cleaning up temporary tables...")
        conn.execute(text("DROP TABLE IF EXISTS silver.temp_silver_rounded;"))
        conn.execute(text("DROP TABLE IF EXISTS silver.ref_segments_rounded;"))
        
    print("🟢 Completed backfill of the silver table with 100% precision tolerance!")

if __name__ == "__main__":
    backfill_with_rounded_wkt()
