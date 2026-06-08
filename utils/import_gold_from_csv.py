"""
Importe les 3 fichiers CSV locaux dans les tables gold.* du VPS LyonFlow,
puis rafraichit la vue materialisee ``gold.mv_fact_traffic_pivot``.

Schema CSV attendu (data/raw/ par defaut)::

    edges.csv          : node_u,node_v
    node_mapping.csv   : node_idx,properties_twgid
    traffic_series.csv : timestamp,node_idx,properties_vitesse

Ciblees Gold :

    gold.dim_spatial_grid_mapping  <-- node_mapping.csv (+ colonnes synthetiques)
    gold.dim_gnn_adjacency         <-- edges.csv
    gold.fact_traffic_series       <-- traffic_series.csv
    gold.mv_fact_traffic_pivot     (REFRESH MATERIALIZED VIEW apres import)

Usage
-----
    # Avec le .env a la racine du repo et data/raw/ par defaut
    python utils/import_gold_from_csv.py

    # Dossier de CSV personnalise
    python utils/import_gold_from_csv.py --data-folder /tmp/csv

    # Mode lecture seule (aucune ecriture en base)
    python utils/import_gold_from_csv.py --dry-run

    # Ajout incremental (pas de TRUNCATE)
    python utils/import_gold_from_csv.py --no-truncate

Prerequis
---------
    pip install psycopg2-binary pandas python-dotenv pyarrow
    # .env a la racine : POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & logs
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("LyonFlow-ImportGold")

# Ciblee Gold -- une seule source de verite
GOLD_SCHEMA = "gold"
TABLE_MAPPING = "dim_spatial_grid_mapping"
TABLE_ADJACENCY = "dim_gnn_adjacency"
TABLE_FACT = "fact_traffic_series"
MV_FACT_PIVOT = "mv_fact_traffic_pivot"

BATCH_SIZE = int(os.getenv("IMPORT_BATCH_SIZE", "10000"))


# ---------------------------------------------------------------------------
# Connexion
# ---------------------------------------------------------------------------
def get_db_config() -> dict:
    """Lit la conf DB depuis l'env (MVS : POSTGRES_PASSWORD du .env)."""
    host = os.getenv("POSTGRES_HOST", "51.83.159.224")
    cfg = {
        "host": host,
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "lyonflow"),
        "user": os.getenv("POSTGRES_USER", "lyonflow"),
        "password": os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD"),
    }
    if not cfg["password"]:
        logger.error(
            "POSTGRES_PASSWORD (ou DB_PASSWORD) absent. "
            "Renseignez-le dans .env ou via la variable d'environnement."
        )
        sys.exit(1)
    return cfg


def connect():
    cfg = get_db_config()
    conn = psycopg2.connect(connect_timeout=15, **cfg)
    conn.autocommit = False
    logger.info("Connecte a %s:%s/%s en tant que %s", cfg["host"], cfg["port"], cfg["dbname"], cfg["user"])
    return conn


# ---------------------------------------------------------------------------
# Utilitaires COPY
# ---------------------------------------------------------------------------
def _copy_from_dataframe(
    conn,
    table: str,
    columns: list[str],
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> int:
    """COPY par lots via un buffer en memoire. Retourne le nombre de lignes."""
    if df.empty:
        return 0
    cols_sql = ",".join(f'"{c}"' for c in columns)
    total = 0
    with conn.cursor() as cur:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            buf = io.StringIO()
            chunk.to_csv(buf, index=False, header=False)
            buf.seek(0)
            sql = f'COPY "{GOLD_SCHEMA}"."{table}" ({cols_sql}) FROM STDIN WITH (FORMAT CSV)'
            cur.copy_expert(sql, buf)
            total += len(chunk)
            logger.info("  .. %s: %d/%d", table, total, len(df))
    return total


def _execute_values(
    conn,
    table: str,
    columns: list[str],
    rows: Iterator[tuple],
    batch_size: int = BATCH_SIZE,
) -> int:
    """Insertion par lots via execute_values (gere les colonnes NOT NULL synthetiques)."""
    cols_sql = ",".join(f'"{c}"' for c in columns)
    placeholders = ",".join(["%s"] * len(columns))
    template = f"({placeholders})"
    total = 0
    batch: list[tuple] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                psycopg2.extras.execute_values(
                    cur,
                    f'INSERT INTO "{GOLD_SCHEMA}"."{table}" ({cols_sql}) VALUES %s '
                    f'ON CONFLICT DO NOTHING',
                    batch,
                    template=template,
                    page_size=batch_size,
                )
                total += len(batch)
                logger.info("  .. %s: %d inseres", table, total)
                batch = []
        if batch:
            psycopg2.extras.execute_values(
                cur,
                f'INSERT INTO "{GOLD_SCHEMA}"."{table}" ({cols_sql}) VALUES %s '
                f'ON CONFLICT DO NOTHING',
                batch,
                template=template,
                page_size=batch_size,
                )
            total += len(batch)
            logger.info("  .. %s: %d inseres", table, total)
    return total


# ---------------------------------------------------------------------------
# Import dim_spatial_grid_mapping  (colonnes NOT NULL synthetiques)
# ---------------------------------------------------------------------------
def _grid_for_node(node_idx: int, width: int = 40) -> tuple[int, int, str]:
    """Genere matrix_i/matrix_j/h3_id deterministes a partir de node_idx.

    Les coordonnees reelles (lat/lon) ne sont pas dans le CSV : laissees NULL.
    Le h3_id est un placeholder stable, a remplacer ulterieurement par le vrai
    h3_id issu de la geolocalisation du capteur.
    """
    return (node_idx // width, node_idx % width, f"h3_pending_{node_idx:04d}")


def import_node_mapping(conn, csv_path: Path, truncate: bool, dry_run: bool) -> int:
    logger.info("Lecture %s ...", csv_path.name)
    df = pd.read_csv(csv_path, dtype={"node_idx": int, "properties_twgid": str})
    df["properties_twgid"] = df["properties_twgid"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["properties_twgid"])

    # Colonnes NOT NULL a fournir : matrix_i, matrix_j, h3_id
    grid = df["node_idx"].map(lambda n: _grid_for_node(int(n)))
    df["matrix_i"] = [g[0] for g in grid]
    df["matrix_j"] = [g[1] for g in grid]
    df["h3_id"] = [g[2] for g in grid]
    # lat, lon, updated_at -> defaut NULL / CURRENT_TIMESTAMP
    logger.info(
        "  %d lignes (node_idx 0..%d, twgid uniques: %d)",
        len(df), int(df["node_idx"].max()), df["properties_twgid"].nunique(),
    )

    if dry_run:
        logger.info("[dry-run] %s.%s : %d lignes pretes (non ecrites).", GOLD_SCHEMA, TABLE_MAPPING, len(df))
        return len(df)

    with conn.cursor() as cur:
        if truncate:
            cur.execute(f'TRUNCATE TABLE "{GOLD_SCHEMA}"."{TABLE_MAPPING}" CASCADE')
            logger.info("TRUNCATE %s.%s OK", GOLD_SCHEMA, TABLE_MAPPING)

    rows = (
        (
            int(r.node_idx), r.properties_twgid,
            int(r.matrix_i), int(r.matrix_j), r.h3_id,
        )
        for r in df.itertuples(index=False)
    )
    written = _execute_values(
        conn, TABLE_MAPPING,
        ["node_idx", "properties_twgid", "matrix_i", "matrix_j", "h3_id"],
        rows,
    )
    conn.commit()
    logger.info("OK %s.%s : %d lignes importees.", GOLD_SCHEMA, TABLE_MAPPING, written)
    return written


# ---------------------------------------------------------------------------
# Import dim_gnn_adjacency
# ---------------------------------------------------------------------------
def import_edges(conn, csv_path: Path, truncate: bool, dry_run: bool) -> int:
    logger.info("Lecture %s ...", csv_path.name)
    df = pd.read_csv(csv_path, dtype={"node_u": int, "node_v": int})
    df = df.drop_duplicates(subset=["node_u", "node_v"])
    logger.info("  %d aretes uniques.", len(df))

    if dry_run:
        logger.info("[dry-run] %s.%s : %d lignes pretes (non ecrites).", GOLD_SCHEMA, TABLE_ADJACENCY, len(df))
        return len(df)

    with conn.cursor() as cur:
        if truncate:
            cur.execute(f'TRUNCATE TABLE "{GOLD_SCHEMA}"."{TABLE_ADJACENCY}" CASCADE')
            logger.info("TRUNCATE %s.%s OK", GOLD_SCHEMA, TABLE_ADJACENCY)

    written = _copy_from_dataframe(
        conn, TABLE_ADJACENCY,
        ["node_u", "node_v"], df,
    )
    conn.commit()
    logger.info("OK %s.%s : %d lignes importees.", GOLD_SCHEMA, TABLE_ADJACENCY, written)
    return written


# ---------------------------------------------------------------------------
# Import fact_traffic_series
# ---------------------------------------------------------------------------
def import_traffic_series(conn, csv_path: Path, truncate: bool, dry_run: bool) -> tuple[int, pd.DataFrame]:
    logger.info("Lecture %s ...", csv_path.name)
    df = pd.read_csv(csv_path, dtype={"node_idx": int, "properties_vitesse": float})
    # timestamp peut etre ISO avec timezone -> on laisse pandas parser
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "node_idx", "properties_vitesse"])
    df = df.drop_duplicates(subset=["timestamp", "node_idx"])
    logger.info(
        "  %d observations, %d timestamps, %d noeuds, plage [%s -> %s]",
        len(df), df["timestamp"].nunique(), df["node_idx"].nunique(),
        df["timestamp"].min(), df["timestamp"].max(),
    )

    if dry_run:
        logger.info("[dry-run] %s.%s : %d lignes pretes (non ecrites).", GOLD_SCHEMA, TABLE_FACT, len(df))
        return len(df), df

    with conn.cursor() as cur:
        if truncate:
            cur.execute(f'TRUNCATE TABLE "{GOLD_SCHEMA}"."{TABLE_FACT}"')
            logger.info("TRUNCATE %s.%s OK", GOLD_SCHEMA, TABLE_FACT)

    # Format ISO8601 sans Z (Postgres timestamptz attend +00:00)
    out = df[["timestamp", "node_idx", "properties_vitesse"]].copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out["timestamp"] = out["timestamp"].str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)

    written = _copy_from_dataframe(
        conn, TABLE_FACT,
        ["timestamp", "node_idx", "properties_vitesse"], out,
    )
    conn.commit()
    logger.info("OK %s.%s : %d lignes importees.", GOLD_SCHEMA, TABLE_FACT, written)
    return written, df


# ---------------------------------------------------------------------------
# Refresh vue materialisee
# ---------------------------------------------------------------------------
def refresh_mv(conn, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] REFRESH MATERIALIZED VIEW %s.%s (ignore).", GOLD_SCHEMA, MV_FACT_PIVOT)
        return
    logger.info("REFRESH MATERIALIZED VIEW %s.%s ...", GOLD_SCHEMA, MV_FACT_PIVOT)
    with conn.cursor() as cur:
        cur.execute(f'REFRESH MATERIALIZED VIEW "{GOLD_SCHEMA}"."{MV_FACT_PIVOT}"')
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{GOLD_SCHEMA}"."{MV_FACT_PIVOT}"')
        n = cur.fetchone()[0]
    logger.info("MV %s.%s : %d lignes apres refresh.", GOLD_SCHEMA, MV_FACT_PIVOT, n)


# ---------------------------------------------------------------------------
# Sauvegarde parquet pour usage ML downstream
# ---------------------------------------------------------------------------
def save_parquet_backup(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"traffic_series_{stamp}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Backup parquet ecrit : %s (%d lignes)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Import CSV -> gold.* sur VPS LyonFlow")
    p.add_argument(
        "--data-folder",
        default=str(REPO_ROOT / "data" / "raw"),
        help="Dossier contenant edges.csv, node_mapping.csv, traffic_series.csv",
    )
    p.add_argument("--no-truncate", action="store_true", help="Garde les donnees existantes (ajout / ON CONFLICT DO NOTHING)")
    p.add_argument("--dry-run", action="store_true", help="Lit les CSV, ne touche pas la base")
    p.add_argument("--no-parquet", action="store_true", help="Ne pas ecrire la copie parquet")
    p.add_argument("--no-refresh-mv", action="store_true", help="Ne pas rafraichir mv_fact_traffic_pivot")
    args = p.parse_args()

    data_folder = Path(args.data_folder)
    edges_p = data_folder / "edges.csv"
    mapping_p = data_folder / "node_mapping.csv"
    series_p = data_folder / "traffic_series.csv"
    for p_ in (edges_p, mapping_p, series_p):
        if not p_.exists():
            logger.error("CSV introuvable : %s", p_)
            return 1

    truncate = not args.no_truncate
    dry_run = args.dry_run
    logger.info("Datasource : %s", data_folder)
    logger.info("Mode : %s%s", "DRY-RUN " if dry_run else "", "TRUNCATE+INSERT" if truncate else "INSERT ONLY")

    conn = connect() if not dry_run else None
    try:
        import_node_mapping(conn, mapping_p, truncate, dry_run)
        import_edges(conn, edges_p, truncate, dry_run)
        n_series, df_series = import_traffic_series(conn, series_p, truncate, dry_run)

        if conn is not None:
            if not args.no_refresh_mv:
                refresh_mv(conn, dry_run)
            conn.close()

        if not dry_run and not args.no_parquet:
            save_parquet_backup(df_series, REPO_ROOT / "data" / "processed")

        logger.info("Termine.")
        return 0
    except Exception as exc:
        logger.exception("Echec import : %s", exc)
        if conn is not None:
            conn.rollback()
            conn.close()
        return 2


if __name__ == "__main__":
    sys.exit(main())
