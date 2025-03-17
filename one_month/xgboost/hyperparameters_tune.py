import os
import json
import itertools
import nbformat
import logging
from nbclient import NotebookClient

log_file = "hyperparameter_tuning.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting Hyperparameter Tuning")

# Define hyperparameter ranges
# n_estimators_range = [5, 10, 50, 100, 200, 300, 500, 1000]
n_estimators_range = [5, 10, 50, 100, 200]
max_depth_range = [3, 5, 6, 7, 10, 15, 20]

results_folder = "./results"
os.makedirs(results_folder, exist_ok=True)

results_file = os.path.join(results_folder, "hyperparameter_results.json")

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
    logging.info(f"Loaded existing results from {results_file}.")
else:
    results = []
    with open(results_file, "w") as f:
        json.dump([], f, indent=4)
    logging.info(f"Initialized {results_file} with an empty list.")

notebook_path = "xgboost.ipynb"

hyperparameter_combinations = list(itertools.product(n_estimators_range, max_depth_range))
logging.info(f"Total combinations to test: {len(hyperparameter_combinations)}")

for n_estimators, max_depth in hyperparameter_combinations:
    if any(r["n_estimators"] == n_estimators and r["max_depth"] == max_depth for r in results):
        logging.info(f"Skipping already tested combination: n_estimators={n_estimators}, max_depth={max_depth}")
        continue

    config_path = os.path.join(".", "config.json")
    config = {"n_estimators": n_estimators, "max_depth": max_depth}

    with open(config_path, "w") as f:
        json.dump(config, f)

    logging.info(f"Testing: n_estimators={n_estimators}, max_depth={max_depth}")

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=3600, kernel_name="python3")
        client.execute()

        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    except Exception as e:
        logging.error(f"Error executing notebook for n_estimators={n_estimators}, max_depth={max_depth}: {e}")
        error_log = os.path.join(results_folder, f"error_{n_estimators}_{max_depth}.log")
        with open(error_log, "w") as error_file:
            error_file.write(str(e))
        continue

    metrics_file = os.path.join(results_folder, "metrics_XGBoost.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        result_entry = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "MAE_Delay": metrics.get("MAE_Delay"),
            "MAE_Horizon": metrics.get("MAE_Horizon"),
            "Training_Time": metrics.get("Training_Time"),
        }

        results.append(result_entry)
        logging.info(f"Results stored for n_estimators={n_estimators}, max_depth={max_depth}: {result_entry}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
            logging.info(f"Results updated in {results_file}")

    else:
        logging.warning(f"No metrics found for n_estimators={n_estimators}, max_depth={max_depth}")
        results.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "error": "Metrics file not found"
        })

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
            logging.info(f"Error results updated in {results_file}")
