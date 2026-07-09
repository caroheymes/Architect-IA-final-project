import logging
import os
import subprocess
import sys

import ray

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LyonFlow-Run-Retraining")

# Add project root and current directory to sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(project_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Define a Ray remote task that requires 1 GPU
# This forces Ray to schedule the task on the ray-worker node where the GPU is available
@ray.remote(num_gpus=1)
def run_training_on_worker_gpu(env):
    import logging
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    task_logger = logging.getLogger("LyonFlow-GPU-Training-Task")

    train_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_stgcn_v2.py"))
    project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    task_logger.info(
        f"🚀 Launching train_stgcn_v2.py on the GPU worker node in cwd={project_dir} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})..."
    )

    result = subprocess.run([sys.executable, train_script], env=env, cwd=project_dir)
    return result.returncode


def main():
    # 0. Refresh traffic_series.csv from PostgreSQL to train on the drifted data
    logger.info("🔄 Refreshing traffic_series.csv from PostgreSQL database...")
    try:
        # Set default to 600 timestamps (~2 days of history) to cover the new drifted patterns
        os.environ["SEQ_LEN_EXPORT"] = os.getenv("SEQ_LEN_EXPORT", "800")
        from utils.export_db_to_csv import run_export
        run_export()
        logger.info("🟢 Traffic series successfully updated.")
    except Exception as e:
        logger.error(f"❌ Failed to refresh CSV from database: {e}. Falling back to existing files.")

    logger.info("🏁 Initializing Ray client connection...")
    ray.init(address="auto", ignore_reinit_error=True)

    logger.info("🚀 Loading optimal hyperparameters from Optuna...")
    try:
        import get_best_params

        best_params = get_best_params.get_params_from_optuna()
        if not best_params:
            logger.warning("⚠️ No parameters found in Optuna. Trying MLflow...")
            best_params = get_best_params.get_params_from_mlflow()
    except Exception as e:
        logger.error(f"❌ Failed to load best params script: {e}")
        best_params = None

    if not best_params:
        logger.warning("⚠️ No optimal parameters found. Using safe defaults.")
        best_params = {
            "learning_rate": 0.001,
            "hidden_channels": 128,
            "weight_decay": 1e-5,
            "batch_size": 2,
            "seq_len": 120,
            "weight_jam": 15.0,
            "weight_slow": 5.0,
        }

    # Post-processing security checks (same as get_best_params.py)
    best_params["seq_len"] = 120
    if "batch_size" in best_params and best_params["batch_size"] > 16:
        best_params["batch_size"] = 16

    logger.info("🎯 Selected hyperparameters for training:")
    for k, v in best_params.items():
        logger.info(f"  • {k.upper()} = {v}")

    # Prepare environment variables
    env = os.environ.copy()
    env["USE_LOCAL_CSV"] = "true"
    env["DATA_FOLDER"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/in"))
    env["EPOCHS"] = os.getenv("EPOCHS", "100")
    env["HORIZONS"] = os.getenv("HORIZONS", "6,12,36")
    env["SEQ_LEN"] = str(best_params.get("seq_len", 120))
    env["BATCH_SIZE"] = str(best_params.get("batch_size", 2))
    env["HIDDEN_CHANNELS"] = str(best_params.get("hidden_channels", 128))
    env["DROPOUT"] = str(best_params.get("dropout", 0.1))

    # Handle learning rate naming differences
    lr_val = best_params.get("learning_rate") or best_params.get("lr") or 0.001
    env["LEARNING_RATE"] = str(lr_val)

    if "weight_decay" in best_params:
        env["WEIGHT_DECAY"] = str(best_params["weight_decay"])
    if "weight_jam" in best_params:
        env["WEIGHT_JAM"] = str(best_params["weight_jam"])
    if "weight_slow" in best_params:
        env["WEIGHT_SLOW"] = str(best_params["weight_slow"])

    logger.info("Scheduling training task on Ray cluster (GPU)...")
    # Submit the task to the Ray cluster
    future = run_training_on_worker_gpu.remote(env)

    # Wait for the task to finish and get the return code
    return_code = ray.get(future)

    if return_code != 0:
        logger.error(f"❌ GPU Training task failed with exit code {return_code}")
        sys.exit(return_code)

    logger.info("🟢 GPU Training wrapper finished successfully.")


if __name__ == "__main__":
    main()
