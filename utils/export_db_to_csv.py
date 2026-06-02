import os
import sys

import pandas as pd
from sqlalchemy import create_engine

# Configuration de la base de données via variables d'environnement
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")

# Dossier d'export de destination
OUTPUT_DIR = os.getenv("DATA_FOLDER", "data/in")

# Nombre de pas temporels à exporter (par défaut 150 pour permettre un sliding window de 120)
SEQ_LEN = int(os.getenv("SEQ_LEN_EXPORT", "150"))


def run_export():
    """
    Exécute l'exportation des données de la couche Gold de PostgreSQL vers des fichiers CSV locaux.
    Limite les données de trafic aux 120 derniers pas temporels (seq_len = 120) pour tester l'inférence
    ou un entraînement unitaire ultra-court sans charger toute la base.
    """
    print("====================================================================")
    print("🚀 Démarrage du script d'export de données PostgreSQL vers CSV...")
    print("====================================================================")

    # 1. Connexion à la base de données
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(db_url)
        # Test de connexion rapide
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        print(f"🔌 Connexion réussie à PostgreSQL sur {DB_HOST}:{DB_PORT} (Base: {DB_NAME})")
    except Exception as e:
        print(f"❌ Erreur de connexion à la base de données : {e}")
        sys.exit(1)

    # 2. Création du dossier d'export s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Dossier cible configuré : {OUTPUT_DIR}")

    # 3. Exportation du mapping spatial des nœuds
    try:
        print("\n📦 1/3 Exportation du mapping spatial (dim_spatial_grid_mapping)...")
        query_mapping = """
            SELECT m.node_idx, m.properties_twgid, r.geometry_wgs84_wkt
            FROM gold.dim_spatial_grid_mapping m
            LEFT JOIN silver.ref_segments r ON m.properties_twgid = CAST(r.properties_twgid AS VARCHAR)
            ORDER BY m.node_idx ASC;
        """
        df_mapping = pd.read_sql(query_mapping, con=engine)
        mapping_path = os.path.join(OUTPUT_DIR, "node_mapping.csv")
        df_mapping.to_csv(mapping_path, index=False)
        print(f"✅ Mapping spatial avec géométries sauvegardé ({len(df_mapping)} nœuds) -> {mapping_path}")
    except Exception as e:
        print(f"❌ Erreur lors de l'export du mapping spatial : {e}")
        sys.exit(1)

    # 4. Exportation de la topologie du graphe (les arêtes)
    try:
        print("\n📦 2/3 Exportation de la topologie du graphe (dim_gnn_adjacency)...")
        query_edges = "SELECT node_u, node_v FROM gold.dim_gnn_adjacency;"
        df_edges = pd.read_sql(query_edges, con=engine)
        edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
        df_edges.to_csv(edges_path, index=False)
        print(f"✅ Topologie sauvegardée ({len(df_edges)} arêtes) -> {edges_path}")
    except Exception as e:
        print(f"❌ Erreur lors de l'export de l'adjacence GNN : {e}")
        sys.exit(1)

    # 5. Exportation de la table des faits filtrée sur les 120 derniers pas de temps (seq_len = 120)
    try:
        print(f"\n📦 3/3 Exportation des séries temporelles de trafic (derniers {SEQ_LEN} pas de temps)...")
        # Cette requête récupère uniquement les données associées aux 120 timestamps les plus récents en base
        query_traffic = f"""
            SELECT timestamp, node_idx, properties_vitesse
            FROM gold.fact_traffic_series
            WHERE timestamp IN (
                SELECT DISTINCT timestamp
                FROM gold.fact_traffic_series
                ORDER BY timestamp DESC
                LIMIT {SEQ_LEN}
            )
            ORDER BY timestamp ASC, node_idx ASC;
        """
        df_traffic = pd.read_sql(query_traffic, con=engine)

        # Vérification du nombre de pas de temps réellement exportés
        unique_timestamps = df_traffic["timestamp"].nunique()
        print(f"ℹ️ Nombre de pas temporels récupérés : {unique_timestamps} (demandé : {SEQ_LEN})")

        traffic_path = os.path.join(OUTPUT_DIR, "traffic_series.csv")
        df_traffic.to_csv(traffic_path, index=False)
        print(f"✅ Séries de trafic sauvegardées ({len(df_traffic)} lignes) -> {traffic_path}")
    except Exception as e:
        print(f"❌ Erreur lors de l'export des séries temporelles : {e}")
        sys.exit(1)

    print("\n====================================================================")
    print("🎉 Synthèse de l'exportation :")
    print(f"   - Fichiers écrits dans : {OUTPUT_DIR}")
    print("   - Fichiers générés : node_mapping.csv, edges.csv, traffic_series.csv")
    print(f"   - Configuration de test prête pour seq_len = {SEQ_LEN}")
    print("====================================================================")


if __name__ == "__main__":
    run_export()
