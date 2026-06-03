import os
import mlflow
from mlflow.tracking import MlflowClient

def get_mlflow_runs():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)
    try:
        runs = client.search_runs(experiment_ids=["6", "7", "8"], order_by=["attribute.start_time DESC"], max_results=30)
        return runs
    except Exception as e:
        print("Error getting runs:", e)
        return []

def main():
    runs = get_mlflow_runs()
    for r in runs:
        run_name = r.data.tags.get("mlflow.runName", "STGCN Run")
        status = r.info.status
        label = f"{run_name} ({r.info.run_id[:8]}) [{status}]"
        
        # Get the specific model architecture parameters logged
        model_type = r.data.params.get("model_type", r.data.params.get("champion_model_type"))
        if not model_type:
            hidden = r.data.params.get("hidden_channels")
            if hidden == "64" or run_name.lower().startswith("stgcn_v2_"):
                model_type = "STGCN_V2_AdamW"
            elif hidden == "128" or run_name.lower().startswith("stgcn_prod_train_"):
                model_type = "STGCN_V1_Adam"
            else:
                model_type = "STGCN"
                
        print(f"Label: {label}")
        print(f"  Run ID: {r.info.run_id}")
        print(f"  Run Name: {run_name}")
        print(f"  Status: {status}")
        print(f"  Model Type: {model_type}")
        print("-" * 50)

if __name__ == "__main__":
    main()
