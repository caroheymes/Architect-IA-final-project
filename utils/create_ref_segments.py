import os
import json
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)

def generate_mapping_exhaustive():
    print("Connecting to PostgreSQL database...")
    
    # 1. Fetch ALL snapshots from bronze to build an exhaustive dictionary of all historical segments
    with engine.begin() as conn:
        print("Fetching ALL historical snapshots from bronze...")
        snaps = conn.execute(text("""
            SELECT raw_data 
            FROM bronze.trafic_vitesse_brute;
        """)).fetchall()
        
    if not snaps:
        print("No snapshots found in bronze table.")
        return
        
    print(f"Retrieved {len(snaps)} total snapshots. Parsing features...")
    
    # 2. Extract features and deduplicate by twgid across all history
    segments_dict = {}
    for idx, snap in enumerate(snaps, 1):
        raw_payload = snap[0] or {}
        features = raw_payload.get('features', []) or []
        for f in features:
            props = f.get('properties', {}) or {}
            geom = f.get('geometry', {}) or {}
            coords = geom.get('coordinates')
            twgid = props.get('twgid')
            gid = props.get('gid')
            
            if not coords or len(coords) < 2 or twgid is None:
                continue
                
            # If we haven't seen this twgid yet, or if its coordinates/gid are missing
            if twgid not in segments_dict:
                segments_dict[twgid] = {
                    'properties_gid': gid,
                    'properties_twgid': twgid,
                    'geometry_coordinates': coords
                }
                
    print(f"Extraction complete. Found {len(segments_dict)} unique historical segments across all snapshots.")
    
    # Convert to DataFrame
    df_unique = pd.DataFrame(list(segments_dict.values()))
    
    # 3. Project geometry coordinates to WGS84 WKT using geopandas to match silver table exactly
    print("Projecting segment geometries to WGS84 WKT...")
    df_unique['geometry_obj'] = df_unique['geometry_coordinates'].apply(LineString)
    gdf = GeoDataFrame(data=df_unique, geometry='geometry_obj')
    gdf.set_crs(epsg=2154, inplace=True)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    
    df_unique['geometry_wgs84_wkt'] = gdf_wgs84['geometry_obj'].apply(lambda g: g.wkt if g else None)
    
    # Select final mapping columns
    df_mapping = df_unique[['geometry_wgs84_wkt', 'properties_twgid', 'properties_gid']].copy()
    
    # Write to database
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))
        conn.execute(text("DROP TABLE IF EXISTS silver.ref_segments;"))
        
        df_mapping.to_sql(
            name="ref_segments",
            con=conn,
            schema="silver",
            if_exists="append",
            index=False
        )
        print("🟢 Recreated reference mapping table silver.ref_segments with all historical segments.")
        
        # 4. Perform direct database-native update on silver table
        print("Checking if silver table exists...")
        table_exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'silver' AND table_name = 'trafic_vitesse_propre'
            );
        """)).fetchone()[0]
        
        if table_exists:
            print("Alter silver table to add properties_gid and properties_twgid columns if not exist...")
            conn.execute(text("ALTER TABLE silver.trafic_vitesse_propre ADD COLUMN IF NOT EXISTS properties_gid integer;"))
            conn.execute(text("ALTER TABLE silver.trafic_vitesse_propre ADD COLUMN IF NOT EXISTS properties_twgid integer;"))
            
            print("Performing ultra-fast UPDATE JOIN to populate the columns for all historical rows...")
            update_res = conn.execute(text("""
                UPDATE silver.trafic_vitesse_propre s
                SET 
                    properties_gid = m.properties_gid,
                    properties_twgid = m.properties_twgid
                FROM silver.ref_segments m
                WHERE s.geometry_wgs84_wkt = m.geometry_wgs84_wkt;
            """))
            print(f"🟢 Successfully updated {update_res.rowcount} rows in silver.trafic_vitesse_propre!")
        else:
            print("Silver table does not exist yet. It will be created with correct columns in the next run.")

if __name__ == "__main__":
    generate_mapping_exhaustive()
