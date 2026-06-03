import os
import mlflow
from mlflow.tracking import MlflowClient

def main():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)
    try:
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=100)
            for r in runs:
                # search keys and values of params
                for k, v in r.data.params.items():
                    if "ST-GRU-GNN" in str(v) or "ST-GRU-GNN" in str(k):
                        print(f"Found in Run {r.info.run_id} ({r.data.tags.get('mlflow.runName')}) - Param {k}: {v}")
                # search keys and values of tags
                for k, v in r.data.tags.items():
                    if "ST-GRU-GNN" in str(v) or "ST-GRU-GNN" in str(k):
                        print(f"Found in Run {r.info.run_id} ({r.data.tags.get('mlflow.runName')}) - Tag {k}: {v}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
