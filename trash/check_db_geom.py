import os
import pandas as pd
from sqlalchemy import create_engine

# Use container env variables
DB_USER = os.environ.get("DB_USER", "lyonflow")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "lyonflow_password")
DB_HOST = os.environ.get("DB_HOST", "lyonflow-postgres")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_DB = os.environ.get("DB_DB", "lyonflow")

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

try:
    engine = create_engine(db_url)
    query = """
        SELECT 
            geometry_wgs84_wkt, 
            COUNT(*) as count
        FROM gold.fact_predictions_traffic 
        WHERE geometry_wgs84_wkt IS NOT NULL
        GROUP BY geometry_wgs84_wkt
        LIMIT 5;
    """
    df = pd.read_sql(query, con=engine)
    print("Sample geometries from gold:")
    print(df)
    
    # Check if there are any LINESTRINGs
    query_line = "SELECT COUNT(*) FROM gold.fact_predictions_traffic WHERE geometry_wgs84_wkt LIKE 'LINESTRING%';"
    df_line = pd.read_sql(query_line, con=engine)
    print("\nNumber of LINESTRING geometries in gold:")
    print(df_line)
    
    # Check total rows
    query_total = "SELECT COUNT(*) FROM gold.fact_predictions_traffic;"
    df_total = pd.read_sql(query_total, con=engine)
    print("\nTotal rows in gold:")
    print(df_total)
    
except Exception as e:
    print("Error querying database inside container:", e)
