import os

import pandas as pd
import shapely.wkt
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)


def psql_insert_execute_values(table, conn, keys, data_iter):
    """Méthode d'insertion `to_sql` optimisée via `execute_values` (insert batch multi-lignes).

    Pandas `to_sql(method=...)` attend une fonction `(table, conn, keys, data_iter) -> None`.
    On construit ici un `INSERT INTO ... VALUES %s` qui profite de la sérialisation
    groupée de `psycopg2.extras.execute_values` (bien plus rapide que la boucle
    par défaut de `to_sql`).

    Args:
        table (sqlalchemy.Table): Table cible (incluant le schéma).
        conn (sqlalchemy.Connection): Connexion SQLAlchemy ouverte.
        keys (list[str]): Noms des colonnes à insérer.
        data_iter (iterable): Itérable de tuples de valeurs alignés sur `keys`.
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        columns = ", ".join([f'"{k}"' for k in keys])
        if table.schema:
            table_name = f'"{table.schema}"."{table.name}"'
        else:
            table_name = f'"{table.name}"'
        sql = f"INSERT INTO {table_name} ({columns}) VALUES %s"
        execute_values(cur, sql, list(data_iter))


def round_wkt(wkt_str):
    """Arrondit toutes les coordonnées d'un WKT à 6 décimales (~ 11 cm à l'équateur).

    Sert à « normaliser » deux géométries qui devraient être identiques mais
    diffèrent de quelques décimales dues à des transformations successives de
    CRS, afin de pouvoir les joindre par égalité stricte de chaîne.

    Args:
        wkt_str (str | None): WKT en entrée.

    Returns:
        str | None: WKT réécrit avec précision=6, ou `None` si l'entrée est
        vide ou si le parsing échoue.
    """
    if not wkt_str:
        return None
    try:
        geom = shapely.wkt.loads(wkt_str)
        return shapely.wkt.dumps(geom, rounding_precision=6)
    except Exception:
        return None


def backfill_with_rounded_wkt():
    """Backfill des colonnes `properties_gid` / `properties_twgid` par jointure WKT.

    Problème résolu : après ingestion Bronze→Silver, certains segments n'ont
    pas pu être rattachés à `silver.ref_segments` car leurs WKT diffèrent
    de quelques décimales. Ce script :
      1. Calcule un WKT « arrondi à 6 décimales » pour chaque ligne de
         `silver.ref_segments` et de `silver.trafic_vitesse_propre`
         (lignes où `properties_gid IS NULL`).
      2. Dépose ces deux jeux de clés dans des tables temporaires
         (`ref_segments_rounded`, `temp_silver_rounded`).
      3. Exécute un `UPDATE ... FROM ... JOIN` qui réconcilie les lignes
         Silver non matchées avec leur référence par égalité sur le WKT
         arrondi.
      4. Supprime les tables temporaires.

    Affiche un récapitulatif du nombre de lignes mises à jour et s'arrête
    proprement si aucune ligne à backfiller n'est trouvée.
    """
    print("Connecting to PostgreSQL database...")

    # 1. Fetch ref_segments and compute rounded WKT in Python
    with engine.connect() as conn:
        print("Reading ref_segments...")
        df_ref = pd.read_sql(
            "SELECT geometry_wgs84_wkt, properties_twgid, properties_gid FROM silver.ref_segments", con=conn
        )

    print(f"Loaded {len(df_ref)} reference segments. Computing rounded WKT...")
    df_ref["wkt_rounded_6"] = df_ref["geometry_wgs84_wkt"].apply(round_wkt)

    # Deduplicate reference segments by the rounded WKT to avoid duplicate keys in join
    df_ref_clean = df_ref.dropna(subset=["wkt_rounded_6"]).drop_duplicates(subset=["wkt_rounded_6"]).copy()
    print(f"Deduplicated reference segments: {len(df_ref_clean)} unique rounded WKT keys.")

    # Write rounded reference table
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS silver.ref_segments_rounded;"))
        df_ref_clean[["wkt_rounded_6", "properties_twgid", "properties_gid"]].to_sql(
            name="ref_segments_rounded",
            con=conn,
            schema="silver",
            if_exists="append",
            index=False,
            method=psql_insert_execute_values,
        )
    print("🟢 Created silver.ref_segments_rounded reference table.")

    # 2. Fetch all silver table rows that have NULL properties_gid and compute their rounded WKT
    with engine.connect() as conn:
        print("Reading unmatched rows from silver table...")
        df_silver = pd.read_sql(
            """
            SELECT id_rue, geometry_wgs84_wkt 
            FROM silver.trafic_vitesse_propre 
            WHERE properties_gid IS NULL;
        """,
            con=conn,
        )

    if df_silver.empty:
        print("No unmatched rows found in silver table.")
        return

    print(f"Loaded {len(df_silver)} unmatched rows from silver. Computing rounded WKT...")
    df_silver["wkt_rounded_6"] = df_silver["geometry_wgs84_wkt"].apply(round_wkt)

    # Create temp table in postgres to host silver rounded keys
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS silver.temp_silver_rounded;"))
        df_silver[["id_rue", "wkt_rounded_6"]].to_sql(
            name="temp_silver_rounded",
            con=conn,
            schema="silver",
            if_exists="append",
            index=False,
            method=psql_insert_execute_values,
        )
    print("🟢 Created silver.temp_silver_rounded table.")

    # 3. Perform the UPDATE JOIN using rounded WKT strings
    with engine.begin() as conn:
        print("Executing SQL UPDATE JOIN with rounded WKT...")
        update_res = conn.execute(
            text("""
            UPDATE silver.trafic_vitesse_propre s
            SET 
                properties_gid = m.properties_gid,
                properties_twgid = m.properties_twgid
            FROM silver.temp_silver_rounded t
            JOIN silver.ref_segments_rounded m ON t.wkt_rounded_6 = m.wkt_rounded_6
            WHERE s.id_rue = t.id_rue;
        """)
        )
        print(f"🟢 Successfully updated {update_res.rowcount} unmatched rows using rounded WKT join!")

        # Cleanup temporary tables
        print("Cleaning up temporary tables...")
        conn.execute(text("DROP TABLE IF EXISTS silver.temp_silver_rounded;"))
        conn.execute(text("DROP TABLE IF EXISTS silver.ref_segments_rounded;"))

    print("🟢 Completed backfill of the silver table with 100% precision tolerance!")


if __name__ == "__main__":
    backfill_with_rounded_wkt()
