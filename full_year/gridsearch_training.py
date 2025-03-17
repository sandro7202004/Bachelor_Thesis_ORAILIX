import os
import time
import json
import logging
import numpy as np
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(filename='gridsearch_training.log', level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Configurable Parameters
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Specify which models to run; for example, ["xgboost"] or ["xgboost", "MLP"]
RUN_MODELS = ["xgboost"]
PERCENTAGE_OF_DATA_USAGE = 0.1

# Progressive training sets:
# Train months will be progressively [1], [1,2], [1,2,3], … up to [1,2,...,11]
progressive_train_sets = [list(range(1, i + 1)) for i in range(1, 12)]
FIXED_TEST_MONTHS = [12]

combined_results = defaultdict(dict)

# ---------------------------
# Helper Functions
# ---------------------------
def update_notebook_parameters(notebook, percentage, train_months, test_months, suffix):
    """
    Finds (or creates) a parameters cell that starts with "# PARAMETERS" and updates it.
    """
    param_cell_found = False
    new_source = f"""# PARAMETERS
percentage_of_data_usage = {percentage}
train_months = {train_months}
test_months = {test_months}
suffix = "{suffix}"
"""
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.source.strip().startswith("# PARAMETERS"):
            cell.source = new_source
            param_cell_found = True
            break
    if not param_cell_found:
        new_cell = nbformat.v4.new_code_cell(source=new_source)
        notebook.cells.insert(0, new_cell)
    return notebook

def run_notebook(notebook_path, updated_notebook):
    """
    Executes the given notebook (already updated with parameters) and saves an executed copy.
    """
    logging.info(f"Running notebook: {notebook_path}")
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
    try:
        ep.preprocess(updated_notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        logging.error(f"Error executing notebook {notebook_path}: {e}")
    logging.info(f"Completed notebook: {notebook_path}")

    output_notebook_path = notebook_path.replace(".ipynb", f"_executed.ipynb")
    with open(output_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(updated_notebook, f)

def is_evaluation_complete(log_path, model_name, progressive_suffix):
    """
    Checks the model’s evaluation log for markers indicating that evaluation for the current progressive run is complete.
    We assume that the notebook writes log messages such as:
      "Starting evaluation {suffix}"
      "Evaluation complete {suffix}"
    """
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r") as f:
        logs = f.readlines()
    complete_lines = [line for line in logs if f"Evaluation complete {progressive_suffix}" in line]
    start_lines = [line for line in logs if f"Starting evaluation {progressive_suffix}" in line]
    return len(complete_lines) >= len(start_lines) and len(start_lines) > 0

def extract_results(json_path):
    """
    Reads the JSON metrics file and extracts the MAE horizon values and training time.
    """
    if not os.path.exists(json_path):
        logging.error(f"Missing results file: {json_path}")
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    mae_horizon = data.get("MAE_Horizon", [])
    training_time = data.get("Training_Time", None)
    return mae_horizon, training_time

# ---------------------------
# Main Execution Loop
# ---------------------------
for model in RUN_MODELS:
    model_dir = os.path.join(BASE_DIR, model)
    notebook_path = os.path.join(model_dir, f"{model}.ipynb")
    log_path = os.path.join(model_dir, f"{model}_evaluation.log")

    model_results = {}

    training_months_list = []
    model_mae_list = []
    translation_mae_list = []
    improvement_list = []
    
    for i, train_months in enumerate(progressive_train_sets, start=1):
        # Use a suffix to uniquely mark this progressive run
        suffix = f"_prog_{i}"
        logging.info(f"Starting grid search for {model} with training months: {train_months} and suffix: {suffix}")
        
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        updated_nb = update_notebook_parameters(nb, PERCENTAGE_OF_DATA_USAGE, train_months, FIXED_TEST_MONTHS, suffix)
        
        # Save updated notebook temporarily
        temp_notebook_path = notebook_path.replace(".ipynb", f"_temp{suffix}.ipynb")
        with open(temp_notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(updated_nb, f)
        
        run_notebook(temp_notebook_path, updated_nb)

        logging.info(f"Waiting for evaluation to complete for {model} with suffix {suffix}...")
        timeout = 600  # seconds
        start_wait = time.time()
        while not is_evaluation_complete(log_path, model, suffix):
            time.sleep(30)
            if time.time() - start_wait > timeout:
                logging.warning(f"Timeout waiting for evaluation for {model} with suffix {suffix}.")
                break
        
        results_json_path = os.path.join(model_dir, "results", f"metrics{suffix}_{model}.json")
        extraction = extract_results(results_json_path)
        if extraction is None:
            logging.error(f"Could not extract results for {model} with suffix {suffix}.")
            continue
        mae_horizon, train_time = extraction
        
        if mae_horizon and isinstance(mae_horizon, list) and len(mae_horizon) > 0:
            model_mae = np.mean([bin_result[0] for bin_result in mae_horizon[:4]])
            translation_mae = np.mean([bin_result[1] for bin_result in mae_horizon[:4]])
            improvement = translation_mae - model_mae
        else:
            model_mae = None
            translation_mae = None
            improvement = None
        
        model_results[suffix] = {
            "train_months": train_months,
            "test_months": FIXED_TEST_MONTHS,
            "mae_horizon": mae_horizon,
            "training_time": train_time,
            "model_mae": model_mae,
            "translation_mae": translation_mae,
            "improvement": improvement
        }
        training_months_list.append(len(train_months))
        model_mae_list.append(model_mae)
        translation_mae_list.append(translation_mae)
        improvement_list.append(improvement)
        
        # Remove the temporary notebook file
        os.remove(temp_notebook_path)
    
    combined_json_path = os.path.join(model_dir, "results", f"combined_results_{model}_{int(time.time())}.json")
    with open(combined_json_path, "w") as f:
        json.dump(model_results, f, indent=4)
    combined_results[model] = model_results
    
    plt.figure(figsize=(10, 6))
    plt.plot(training_months_list, improvement_list, marker='o')
    plt.xlabel("Number of Training Months Used")
    plt.ylabel("Improvement (Translation MAE - Model MAE)")
    plt.title(f"Progressive Forecasting Improvement for {model}")
    plt.grid(True)
    plot_filename = os.path.join(model_dir, "results", "plots", f"progressive_improvement_{model}_{int(time.time())}.png")
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(training_months_list, model_mae_list, marker='o', label="Model MAE")
    plt.plot(training_months_list, translation_mae_list, marker='o', label="Translation MAE")
    plt.xlabel("Number of Training Months Used")
    plt.ylabel("Average MAE (over horizon bins)")
    plt.title(f"Progressive Forecasting MAE for {model}")
    plt.legend()
    plt.grid(True)
    plot_filename2 = os.path.join(model_dir, "results", "plots", f"progressive_mae_{model}_{int(time.time())}.png")
    plt.savefig(plot_filename2)
    plt.close()
    
    logging.info(f"Combined results and plots generated for {model}.")

overall_combined_json_path = os.path.join(BASE_DIR, f"combined_results_overall_{int(time.time())}.json")
with open(overall_combined_json_path, "w") as f:
    json.dump(combined_results, f, indent=4)

logging.info("Grid search training and evaluation complete.")
