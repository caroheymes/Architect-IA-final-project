import os
import sys

# Import tracking libraries if available
try:
    import optuna
except ImportError:
    optuna = None

try:
    import mlflow
except ImportError:
    mlflow = None

# Database & tracking settings
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "lyonflow")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def get_params_from_optuna():
    if optuna is None:
        print("# optuna is not installed on this host. Skipping database check.")
        return None
    
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        study = optuna.load_study(study_name="lyonflow_stgcn_tuning_v1", storage=db_url)
        return study.best_params
    except Exception as e:
        print(f"# Optuna database reading skipped or failed: {e}")
        return None

def get_params_from_mlflow():
    if mlflow is None:
        print("# mlflow is not installed on this host. Skipping MLflow tracking check.")
        return None
        
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.client.MlflowClient()
        experiment = client.get_experiment_by_name("LyonFlow-STGCN-Optuna-Tuning")
        if experiment is None:
            print("# MLflow experiment 'LyonFlow-STGCN-Optuna-Tuning' not found.")
            return None
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_mae_kmh ASC"],
            max_results=1
        )
        if len(runs) > 0:
            best_run = runs[0]
            print(f"# Found best run in MLflow: {best_run.info.run_id}")
            params = {}
            for k, v in best_run.data.params.items():
                # Clean and parse types dynamically
                try:
                    if '.' in v or 'e' in v.lower():
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v
            return params
        else:
            print("# No runs found in MLflow experiment.")
    except Exception as e:
        print(f"# MLflow tracking reading skipped or failed: {e}")
    return None

def main():
    print("# Searching for the Champion Model's hyperparameters...")
    
    # 1. Try to read directly from PostgreSQL Optuna storage (the direct source of truth)
    params = get_params_from_optuna()
    
    # 2. Try to fallback to MLflow Server API
    if not params:
        params = get_params_from_mlflow()
        
    if params:
        print("# " + "="*50)
        print("# OPTIMAL PARAMETERS RETRIEVED SUCCESSFULLY")
        print("# " + "="*50)
        
        # We enforce seq_len to 120 as specified for the Champion model
        params["seq_len"] = 120
        # # Ensure batch_size is safe for sequence length of 120 (max 4 to fit in GPU VRAM)
        # if "batch_size" in params and params["batch_size"] > 4:
        #     print(f"# Scaling down batch_size from {params['batch_size']} to 16 to prevent GPU VRAM Out Of Memory errors on seq_len 120.")
        #     params["batch_size"] = 16
        # Limit batch_size to a maximum of 16 to fit smoothly in 24GB VRAM
        if "batch_size" in params and params["batch_size"] > 16:
            print(f"# Scaling down batch_size from {params['batch_size']} to 16 to prevent GPU VRAM Out Of Memory errors on seq_len 120.")
            params["batch_size"] = 16
            
        for k, v in params.items():
            env_key = k.upper()
            if env_key == "LR":
                env_key = "LEARNING_RATE"
            print(f"export {env_key}={v}")
    else:
        print("# " + "="*50)
        print("# WARNING: NO PARAMETERS FOUND, USING CODE DEFAULTS")
        print("# " + "="*50)
        # Safe fallback standard params
        print("export LEARNING_RATE=0.001")
        print("export HIDDEN_CHANNELS=128")
        print("export WEIGHT_DECAY=1e-5")
        print("export BATCH_SIZE=2")
        print("export SEQ_LEN=120")
        print("export WEIGHT_JAM=15.0")
        print("export WEIGHT_SLOW=5.0")

if __name__ == "__main__":
    main()
