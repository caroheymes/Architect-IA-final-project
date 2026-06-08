import os

from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://lyonflow:lyonflow_password@postgres:5432/lyonflow"
engine = create_engine(DATABASE_URL)


def run_bronze_fields_check():
    with engine.connect() as conn:
        print("Checking total snapshots in bronze...")
        total_snaps = conn.execute(text("SELECT count(*) FROM bronze.trafic_vitesse_brute")).fetchone()[0]
        print(f"Total snapshots: {total_snaps}")

        print("Inspecting a few snapshots...")
        snaps = conn.execute(
            text("SELECT id, fetched_at, raw_data FROM bronze.trafic_vitesse_brute ORDER BY fetched_at ASC LIMIT 5")
        ).fetchall()

        for snap_id, fetched_at, raw_data in snaps:
            features = raw_data.get("features", [])
            if features:
                f0 = features[0]
                geom = f0.get("geometry", {})
                coords = geom.get("coordinates")
                props = f0.get("properties", {})
                gid = props.get("gid")
                twgid = props.get("twgid")
                print(
                    f"Snap {snap_id} (fetched {fetched_at}): coords_len={len(coords) if coords else 0}, properties.gid={gid}, properties.twgid={twgid}"
                )
            else:
                print(f"Snap {snap_id} (fetched {fetched_at}): No features")


if __name__ == "__main__":
    run_bronze_fields_check()
