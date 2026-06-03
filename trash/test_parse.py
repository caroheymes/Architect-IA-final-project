import os
import pandas as pd
import shapely.wkt
from sqlalchemy import create_engine

DB_USER = os.environ.get("DB_USER", "lyonflow")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "lyonflow_password")
DB_HOST = os.environ.get("DB_HOST", "lyonflow-postgres")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_DB = os.environ.get("DB_DB", "lyonflow")

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

print("=== CHECKING POSTGRESQL ===")
try:
    engine = create_engine(db_url)
    query = """
        SELECT 
            geometry_wgs84_wkt 
        FROM gold.fact_predictions_traffic 
        LIMIT 100;
    """
    df = pd.read_sql(query, con=engine)
    print("Fetched rows:", len(df))
    count_ls = 0
    for idx, row in df.iterrows():
        wkt_str = row.get("geometry_wgs84_wkt")
        if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
            count_ls += 1
    print("LINESTRING count in DB limit 100:", count_ls)
    if len(df) > 0:
        print("First WKT:", df.iloc[0]["geometry_wgs84_wkt"])
except Exception as e:
    print("DB error:", e)

print("\n=== CHECKING FALLBACK CSV ===")
csv_path = "data/out/predictions_traffic.csv"
if os.path.exists(csv_path):
    try:
        df_csv = pd.read_csv(csv_path)
        print("Fetched CSV rows:", len(df_csv))
        count_ls_csv = 0
        for idx, row in df_csv.iterrows():
            wkt_str = row.get("geometry_wgs84_wkt")
            if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
                count_ls_csv += 1
        print("LINESTRING count in CSV:", count_ls_csv)
    except Exception as e:
        print("CSV error:", e)
else:
    print("CSV file does not exist")
