# -----------------------------------------------
# Imports
# -----------------------------------------------
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import networkx as nx
import logging
import json
import time
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
def transform_to_seconds(data):
    """
    Transforms data to seconds using a custom transformation function.
    
    Args:
        data (np.ndarray): Input data.
        
    Returns:
        np.ndarray: Transformed data.
    """
    return np.sign(data) * (np.power((data * 6), 2))

def inverse_transform(data):
    """
    Inverse transforms data from seconds using a custom transformation function.
    
    Args:
        data (np.ndarray): Transformed data.
        
    Returns:
        np.ndarray: Original data.
    """
    return np.sign(data) * (np.sqrt(np.abs(data)) / 6)

def display_all_pkl_contents(results_folder="./results"):
    """
    Opens and displays the contents of all PKL files in the specified results folder.
    
    Args:
        results_folder (str): Path to the folder containing PKL files.
    
    Returns:
        None
    """
    pkl_files = [file for file in os.listdir(results_folder) if file.endswith(".pkl")]
    
    if not pkl_files:
        logging.info("No PKL files found in the results folder.")
        return
    
    for pkl_file in tqdm(pkl_files, desc="Displaying PKL files"):
        file_path = os.path.join(results_folder, pkl_file)
        try:
            with open(file_path, "rb") as f:
                data = pkl.load(f)
                print(f"Contents of {pkl_file}:")
                print(json.dumps(data, indent=4))
                print("\n" + "-"*80 + "\n")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            
def convert_pkl_to_json(results_folder="./results"):
    """
    Converts all PKL files in the specified results folder into JSON files with the same name.
    
    Args:
        results_folder (str): Path to the folder containing PKL files.
    
    Returns:
        None
    """
    pkl_files = [file for file in os.listdir(results_folder) if file.endswith(".pkl")]
    
    if not pkl_files:
        logging.info("No PKL files found in the results folder.")
        return
    
    for pkl_file in tqdm(pkl_files, desc="Converting PKL to JSON"):
        pkl_path = os.path.join(results_folder, pkl_file)
        json_path = os.path.join(results_folder, os.path.splitext(pkl_file)[0] + ".json")
        
        try:
            with open(pkl_path, "rb") as f:
                data = pkl.load(f)
            
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            
            logging.info(f"Converted {pkl_file} to {json_path}")
        except Exception as e:
            logging.error(f"Error converting {pkl_path} to JSON: {e}")
            
def log_time(message):
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def compute_translation_based_estimates(X_test, past_delay_index, futur_planned_start, futur_planned_end, futur_planned_extended_end, y_test):
    past_delay_1_test = transform_to_seconds(X_test[:, past_delay_index])
    futur_planned_test = transform_to_seconds(X_test[:, futur_planned_start:futur_planned_end])
    
    futur_planned_test_extended = (
        transform_to_seconds(X_test[:, 390:405]) if futur_planned_extended_end is not None 
        else futur_planned_test
    )

    corrected_translate_test = past_delay_1_test - ((futur_planned_test[:, 0] + past_delay_1_test) < 0) * (futur_planned_test[:, 0] + past_delay_1_test)
    corrected_translate_test = np.repeat(corrected_translate_test[:, np.newaxis], y_test.shape[1], axis=1)

    log_time("Computed translation-based estimates.")

    return futur_planned_test_extended, corrected_translate_test
   
def compute_mae_bins(y_test, corrected_translate_test, mae_model, mae_translation, delay_delta_bins, horizon_obs_bins, futur_planned_test_extended):
    def compute_mae(bins, values):
        return np.array([
            [np.mean(mae_model[(bins[i] <= values) & (values < bins[i+1])]), 
             np.mean(mae_translation[(bins[i] <= values) & (values < bins[i+1])])]
            for i in range(len(bins) - 1)
        ])
    
    mae_delay = compute_mae(delay_delta_bins, y_test - corrected_translate_test)
    log_time("Computed MAE per delay bin.")
    
    mae_horizon = compute_mae(horizon_obs_bins, futur_planned_test_extended + y_test)
    log_time("Computed MAE per horizon bin.")

    return mae_delay, mae_horizon

def compute_and_save_metrics(y_test, corrected_translate_test, mae_model, mae_translation, 
                             delay_delta_bins, horizon_obs_bins, futur_planned_test_extended, 
                             trained_model, results_folder, model_name, suffix):
    mae_delay, mae_horizon = compute_mae_bins(
        y_test, corrected_translate_test, mae_model, mae_translation, 
        delay_delta_bins, horizon_obs_bins, futur_planned_test_extended
    )

    metrics = {
        "MAE_Delay": mae_delay.tolist(),
        "MAE_Horizon": mae_horizon.tolist(),
        "Training_Time": trained_model["training_time"],
    }

    json_path = f"{results_folder}/metrics{suffix}_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log_time(f"Saved metrics to JSON file: {json_path}")

    pkl_path = f"{results_folder}/metrics{suffix}_{model_name}.pkl"
    with open(pkl_path, "wb") as f:
        pkl.dump(metrics, f)
    log_time(f"Saved metrics to PKL file: {pkl_path}")

    return metrics, mae_delay, mae_horizon

def plot_mae(mae_delay, mae_horizon, delay_delta_bins, horizon_obs_bins, plots_folder, model_name, suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(delay_delta_bins)-1)
    ax1.barh(x - 0.2, mae_delay[:, 0], height=0.4, label='Model MAE', align='center')
    ax1.barh(x + 0.2, mae_delay[:, 1], height=0.4, label='Translation MAE', align='center')
    ax1.set_yticks(x)
    ax1.set_yticklabels([f'{delay_delta_bins[i]/60} < {delay_delta_bins[i+1]/60}' for i in range(len(delay_delta_bins)-1)])
    ax1.set_xlabel('MAE')
    ax1.set_ylabel('Delay Delta Bins')
    ax1.set_title(f'MAE per Delay Delta - {model_name}')
    ax1.legend()

    x = np.arange(len(horizon_obs_bins)-1)
    ax2.barh(x - 0.2, mae_horizon[:, 0], height=0.4, label='Model MAE', align='center')
    ax2.barh(x + 0.2, mae_horizon[:, 1], height=0.4, label='Translation MAE', align='center')
    ax2.set_yticks(x)
    ax2.set_yticklabels([f'{horizon_obs_bins[i]/60} < {horizon_obs_bins[i+1]/60}' for i in range(len(horizon_obs_bins)-1)])
    ax2.set_xlabel('MAE')
    ax2.set_ylabel('Horizon Bins')
    ax2.set_title(f'MAE per Horizon - {model_name}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{plots_folder}/mae_plots{suffix}_{model_name}.png")
    plt.close()
    logging.info("Saved MAE plots.")

# -----------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------
def load_data_aux(local_path_to_data, percentage_of_data_usage, train_months=None, test_months=None, adjust_target=False):
    """
    Auxiliary function to load and preprocess the dataset with explicit training and testing months.
    
    Args:
        local_path_to_data (str): Path to the data folder.
        percentage_of_data_usage (float): The percentage of the dataset to load (0.0 to 1.0).
        train_months (list, optional): List of month identifiers for training.
        test_months (list, optional): List of month identifiers for testing.
        adjust_target (bool, optional): Whether to adjust the target variable (default: False).
    
    Returns:
        dict: A dictionary containing preprocessed data splits and metadata.
    """
    logging.info("Loading data...")

    with open(os.path.join(local_path_to_data, "metadata/columns_scheme.pkl"), "rb") as f:
        columns_scheme = pkl.load(f)

    if train_months is None or test_months is None:
        raise ValueError("Both train_months and test_months must be specified.")

    train_idx, test_idx = None, None
    if len(train_months) == 1 and len(test_months) == 1 and train_months == test_months:
        md_list = []

        logging.info("Single month training and testing detected. Applying 80-20 split.")

        month_str = f"{int(train_months[0]):02d}"

        X_month = np.load(os.path.join(local_path_to_data, f"x/x_2023{month_str}.npy"))
        y_month = np.load(os.path.join(local_path_to_data, f"y_delays/y_delays_2023{month_str}.npy"))
        md = np.load(os.path.join(local_path_to_data, f"metadata/md_2023{month_str}.npy"), allow_pickle=True)

        month_sample_size = max(1, int(percentage_of_data_usage * len(X_month)))
        X_month, y_month, md = X_month[:month_sample_size], y_month[:month_sample_size], md[:month_sample_size]

        datdep_col = md[:, 1]
        unique_dates = np.unique(datdep_col)

        split_index = int(0.8 * len(unique_dates))
        train_dates, test_dates = unique_dates[:split_index], unique_dates[split_index:]

        train_idx = np.isin(datdep_col, train_dates)
        test_idx = np.isin(datdep_col, test_dates)

        X_train, y_train = X_month[train_idx], y_month[train_idx]
        X_test, y_test = X_month[test_idx], y_month[test_idx]

        md_list.append(md)

    else:
        logging.info("Processing multiple months separately.")

        X_train_list, y_train_list, md_list = [], [], []
        X_test_list, y_test_list = [], []

        for month in tqdm(train_months, desc="Loading train months"):
            month_str = f"{int(month):02d}"
            X_month = np.load(os.path.join(local_path_to_data, f"x/x_2023{month_str}.npy"), mmap_mode='r')
            y_month = np.load(os.path.join(local_path_to_data, f"y_delays/y_delays_2023{month_str}.npy"), mmap_mode='r')
            md_month = np.load(os.path.join(local_path_to_data, f"metadata/md_2023{month_str}.npy"), allow_pickle=True)

            month_sample_size = max(1, int(percentage_of_data_usage * len(X_month)))
            X_train_list.append(X_month[:month_sample_size].copy())
            y_train_list.append(y_month[:month_sample_size].copy())
            md_list.append(md_month[:month_sample_size].copy())

        for month in tqdm(test_months, desc="Loading test months"):
            month_str = f"{int(month):02d}"
            X_month = np.load(os.path.join(local_path_to_data, f"x/x_2023{month_str}.npy"), mmap_mode='r')
            y_month = np.load(os.path.join(local_path_to_data, f"y_delays/y_delays_2023{month_str}.npy"), mmap_mode='r')
            md_month = np.load(os.path.join(local_path_to_data, f"metadata/md_2023{month_str}.npy"), allow_pickle=True)

            month_sample_size = max(1, int(percentage_of_data_usage * len(X_month)))
            X_test_list.append(X_month[:month_sample_size].copy())
            y_test_list.append(y_month[:month_sample_size].copy())
            md_list.append(md_month[:month_sample_size].copy())

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        train_dates = np.unique(md[:, 1])
        test_dates = np.unique(md[:, 1])

    md = np.concatenate(md_list, axis=0)

    logging.info(f"Train dates: {train_dates}")
    logging.info(f"Test dates: {test_dates}")
    logging.info(f"Number of unique train dates: {len(train_dates)}")
    logging.info(f"Number of unique test dates: {len(test_dates)}")

    if adjust_target:
        # y_delays = inverse_transform(transform_to_seconds(y_delays) - transform_to_seconds(past_delays))
        # y_train = inverse_transform(transform_to_seconds(y_train) - transform_to_seconds(past_delays))
        # y_test = inverse_transform(transform_to_seconds(y_test) - transform_to_seconds(past_delays))
        past_delays = np.repeat(X_train[:, 4][:, np.newaxis], 5, axis=1)
        y_train -= past_delays
        y_test -= np.repeat(X_test[:, 4][:, np.newaxis], 5, axis=1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": md,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "columns_scheme": columns_scheme,
        "X": np.vstack([X_train, X_test]),
        "y_delays": np.vstack([y_train, y_test]),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }

def load_data(local_path_to_data="/Users/mac/Desktop/train_delay_prediction/delay_pred_5_5_v1/", percentage_of_data_usage=1.0):
    return load_data_aux(local_path_to_data, percentage_of_data_usage, train_months=[3], test_months=[3])

def load_data_newtarget(local_path_to_data="/Users/mac/Desktop/train_delay_prediction/delay_pred_5_5_v1/", percentage_of_data_usage=1.0):
    return load_data_aux(local_path_to_data, percentage_of_data_usage, train_months=[3], test_months=[3], adjust_target=True)

def load_full_year_data(local_path_to_data="/Users/mac/Desktop/train_delay_prediction/delay_pred_5_15_numpy_v1/", percentage_of_data_usage=1.0, train_months=None, test_months=None):
    return load_data_aux(local_path_to_data, percentage_of_data_usage, train_months=train_months, test_months=test_months)

def load_full_year_data_newtarget(local_path_to_data="/Users/mac/Desktop/train_delay_prediction/delay_pred_5_15_numpy_v1/", percentage_of_data_usage=1.0, train_months=None, test_months=None):
    return load_data_aux(local_path_to_data, percentage_of_data_usage, train_months=train_months, test_months=test_months, adjust_target=True)

def load_data_more_features(percentage_of_data_usage, train_months=[3], test_months=[3]):
    logging.info("Loading data...")

    data = load_data(percentage_of_data_usage=percentage_of_data_usage)

    y_delays_org = data["y_delays"]

    X_train_org = data["X_train"]
    X_test_org = data["X_test"]
    y_test = data["y_test"]
    y_train = data["y_train"]

    train_dates = data["train_dates"]
    test_dates = data["test_dates"]

    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    md = data["metadata"]

    path = "/Users/mac/Desktop/train_delay_prediction/delay_pred_5_5_v1_more_features/"

    with open(os.path.join(path, "metadata/columns_scheme.pkl"), "rb") as f:
        columns_scheme = pkl.load(f)

    if len(train_months) == 1 and len(test_months) == 1 and train_months == test_months:
        logging.info("Single month training and testing detected. Applying 80-20 split.")

        month_str = f"{int(train_months[0]):02d}"

        X_morefeatures = np.load(os.path.join(path, f"x/x_2023{month_str}.npy"))

        month_sample_size = max(1, int(percentage_of_data_usage * len(X_morefeatures)))
        X_morefeatures, y_delays_org = X_morefeatures[:month_sample_size], y_delays_org[:month_sample_size]

        X_morefeatures = X_morefeatures[:, -21:]

        X_train_morefeatures = X_morefeatures[train_idx]
        X_test_morefeature = X_morefeatures[test_idx]

        X_train = np.hstack((X_train_org, X_train_morefeatures))
        X_test= np.hstack((X_test_org, X_test_morefeature))

    logging.info(f"Train dates: {train_dates}")
    logging.info(f"Test dates: {test_dates}")
    logging.info(f"Number of unique train dates: {len(train_dates)}")
    logging.info(f"Number of unique test dates: {len(test_dates)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": md,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "columns_scheme": columns_scheme,
        "X": np.vstack([X_train, X_test]),
        "y_delays": y_delays_org,
    }

# -----------------------------------------------
# Train Models
# -----------------------------------------------
def train(model, X_train, y_train, model_name, models_folder="./models", savemodel=True, verbose=1):
    """
    Trains a model and saves it to a dictionary and disk.

    Args:
        model: Model instance with `fit` and `predict` methods.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target matrix.
        model_name (str): Name of the model for saving.
        models_folder (str): Folder to save the trained models. Default is './models'.

    Returns:
        dict: Contains the trained model and its training time.
    """
    os.makedirs(models_folder, exist_ok=True)

    if hasattr(model, "verbose"):
        model.verbose = verbose

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f"Model '{model_name}' trained in {training_time:.2f} seconds.")

    model_path = f"{models_folder}/{model_name.replace(' ', '_')}_model.pkl"
    
    if savemodel:
        with open(model_path, "wb") as f:
            pkl.dump(model, f)
        logging.info(f"Model '{model_name}' saved at '{model_path}'.")

    return {
        "model": model,
        "training_time": training_time,
    }

# -----------------------------------------------
# Evaluation Metrics
# -----------------------------------------------
def evaluate(trained_model, X_test, y_test, transform_func=transform_to_seconds, results_folder="./results", model_name="model"):
    """
    Evaluates a trained model, calculates MAE, corrected translation MAE, and saves metrics.

    Args:
        trained_model (dict): Contains the trained model and its metadata.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test target matrix.
        transform_func (callable): Function to transform predictions and targets.
        results_folder (str): Folder to save evaluation metrics. Default is './results'.
        model_name (str): Name of the model for saving metrics.

    Returns:
        dict: Evaluation metrics.
    """
    os.makedirs(results_folder, exist_ok=True)

    model = trained_model["model"]

    y_pred = model.predict(X_test)
    y_pred_sec = transform_func(y_pred)
    y_test_sec = transform_func(y_test)

    mae_list = []

    for i in range(y_test.shape[1]):
        mae = mean_absolute_error(y_test_sec[:, i], y_pred_sec[:, i])
        mae_list.append(mae)

    mean_mae = np.mean(mae_list)

    metrics = {
        "Mean MAE": mean_mae,
        "MAE per Output": mae_list,
        "Training Time": trained_model["training_time"],
    }

    metrics_json_path = f"{results_folder}/metrics_{model_name}.json"
    metrics_pkl_path = f"{results_folder}/metrics_{model_name}.pkl"

    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(metrics_pkl_path, "wb") as f:
        pkl.dump(metrics, f)

    logging.info(f"Metrics saved to '{metrics_json_path}' and '{metrics_pkl_path}'.")
    return metrics

def evaluate_2_aux(trained_model, X_test, y_test, delay_delta_bins=np.array([-np.inf,-5,0,5,10,15,20,25,30,np.inf])*60, horizon_obs_bins=np.array([0,5,10,15,np.inf])*60, results_folder="./results", model_name="model", plots_folder="./plots", past_delay_index=4, futur_planned_start=200, futur_planned_end=205, futur_planned_extended_end=None, add_past_delay=False, suffix=""):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log_time("Starting evaluation.")
    
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    log_time("Created results and plots directories if not already existing.")

    model = trained_model["model"]
    log_time("Loaded trained model.")
    
    y_pred = model.predict(X_test)
    log_time("Model predictions completed.")

    if add_past_delay:
        past_delay = np.repeat(X_test[:, past_delay_index][:, np.newaxis], 5, axis=1)
        y_pred += past_delay
        y_test += past_delay

    y_pred = transform_to_seconds(y_pred)
    y_test = transform_to_seconds(y_test)

    # if add_past_delay:
    #     past_delay = np.repeat(X_test[:, past_delay_index][:, np.newaxis], 5, axis=1)
    #     y_pred += transform_to_seconds(past_delay)
    #     y_test += transform_to_seconds(past_delay)
    
    log_time("Transformed inputs to seconds.")
    
    futur_planned_test_extended, corrected_translate_test = compute_translation_based_estimates(X_test, past_delay_index, futur_planned_start, futur_planned_end, futur_planned_extended_end, y_test)

    mae_model = np.abs(y_pred - y_test)
    mae_translation = np.abs(corrected_translate_test - y_test)

    log_time("MAE Translation Mean: {}".format(mae_translation.mean(axis=0)))
    log_time("Computed MAE for model and translation approach.")

    if y_test.shape[1] > 5:
        # Extended prediction horizon expected so we use extended bins.
        delay_delta_bins = np.array([-np.inf, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, np.inf]) * 60
        horizon_obs_bins = np.array([0, 5, 10, 15, 20, 25, 30, np.inf]) * 60
        log_time("Using extended bins for multi-month evaluation.")

    metrics, mae_delay, mae_horizon = compute_and_save_metrics(y_test, corrected_translate_test, mae_model, mae_translation, delay_delta_bins, horizon_obs_bins, futur_planned_test_extended, trained_model, results_folder, model_name, suffix)

    plot_mae(mae_delay, mae_horizon, delay_delta_bins, horizon_obs_bins, plots_folder, model_name, suffix)
    log_time("Evaluation complete.")

    return metrics

def evaluate_2(trained_model, X_test, y_test, model_name):
    return evaluate_2_aux(trained_model, X_test, y_test, model_name=model_name)

def evaluate_2_newtarget(trained_model, X_test, y_test, model_name):
    return evaluate_2_aux(trained_model, X_test, y_test, model_name=model_name, add_past_delay=True, suffix="_newtarget")

def evaluate_2_fullyear(trained_model, X_test, y_test, model_name, suffix=""):
    return evaluate_2_aux(trained_model, X_test, y_test, suffix, model_name=model_name, futur_planned_start=390, futur_planned_end=395, futur_planned_extended_end=405)

def evaluate_2_fullyear_newtarget(trained_model, X_test, y_test, model_name):
    return evaluate_2_aux(trained_model, X_test, y_test, model_name=model_name, futur_planned_start=390, futur_planned_end=395, futur_planned_extended_end=405, add_past_delay=True, suffix="_fullyear_newtarget")

def combine_metrics(delete: bool, results_folder="./results", combined_file_name="combined_metrics.json", combined_pkl_name="combined_metrics.pkl"):
    """
    Combines all JSON and PKL metric files in the specified folder into a single JSON and PKL file
    without deleting the combined files.

    Args:
        results_folder (str): The folder containing the individual metric files.
        combined_file_name (str): The name of the combined JSON file.
        combined_pkl_name (str): The name of the combined PKL file.

    Returns:
        None
    """
    combined_metrics = {}
    json_files = [file for file in os.listdir(results_folder) if file.endswith(".json")]
    pkl_files = [file for file in os.listdir(results_folder) if file.endswith(".pkl")]
    
    if combined_file_name in json_files:
        json_files.remove(combined_file_name)
    if combined_pkl_name in pkl_files:
        pkl_files.remove(combined_pkl_name)

    for json_file in tqdm(json_files, desc="Combining JSON metrics"):
        file_path = os.path.join(results_folder, json_file)
        try:
            with open(file_path, "r") as f:
                metrics = json.load(f)
                model_name = os.path.splitext(json_file)[0]
                combined_metrics[model_name] = metrics
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue

    combined_json_path = os.path.join(results_folder, combined_file_name)
    with open(combined_json_path, "w") as f:
        json.dump(combined_metrics, f, indent=4)
    logging.info(f"Combined metrics saved to '{combined_json_path}'.")

    combined_pkl_path = os.path.join(results_folder, combined_pkl_name)
    with open(combined_pkl_path, "wb") as f:
        pkl.dump(combined_metrics, f)
    logging.info(f"Combined metrics saved to '{combined_pkl_path}'.")

    if delete:
        for json_file in tqdm(json_files, desc="Deleting JSON files"):
            file_path = os.path.join(results_folder, json_file)
            try:
                os.remove(file_path)
                logging.info(f"Deleted '{file_path}'.")
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")

        for pkl_file in tqdm(pkl_files, desc="Deleting PKL files"):
            file_path = os.path.join(results_folder, pkl_file)
            try:
                os.remove(file_path)
                logging.info(f"Deleted '{file_path}'.")
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")

# -----------------------------------------------
# Feature Importance Calculation and Model Visualization
# -----------------------------------------------
def calculate_feature_importance(trained_models, X_test, y_test, feature_mapping, plots_folder="./plots", results_folder="./results", top_features_threshold=0.01, n_repeats=10):
    """
    Calculates and plots feature importance for trained models using permutation importance.
    Saves feature importance rankings in a JSON file.
    
    Args:
        trained_models (dict): Dictionary containing models and their data.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test target vector.
        feature_mapping (dict): Dictionary mapping feature indices to feature names.
        plots_folder (str): Folder to save feature importance plots. Default is './plots'.
        results_folder (str): Folder to save feature importance rankings JSON file. Default is './results'.
        top_features_threshold (float): Minimum importance threshold to display a feature.
        n_repeats (int): Number of permutations for calculating importance. Default is 10.
        
    Returns:
        None
    """
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    logging.info(f"Saving feature importance plots to: {plots_folder}")
    logging.info(f"Saving feature importance rankings to: {results_folder}/feature_importance.json")
    
    feature_importance_results = {}
    
    index_to_feature = {v: k for k, v in feature_mapping.items()}
    
    for arch_name, model_data in tqdm(trained_models.items(), desc="Calculating feature importance"):
        model = model_data["model"]
        logging.info(f"Calculating feature importance for {arch_name}...")

        try:
            r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
            sorted_idx = np.argsort(r.importances_mean)[::-1]  # Sort in descending order
            
            all_features = {
                index_to_feature[i]: {"feature_index": int(i), "importance": float(r.importances_mean[i])}
                for i in sorted_idx
            }
            
            important_features = {
                index_to_feature[i]: {"feature_index": int(i), "importance": float(r.importances_mean[i])}
                for i in sorted_idx if r.importances_mean[i] > top_features_threshold
            }
            
            feature_importance_results[arch_name] = all_features
            
            logging.info(f"Top features for {arch_name}:")
            if important_features:
                for feature, data in important_features.items():
                    logging.info(f"  {feature} (Feature {data['feature_index']}): {data['importance']:.4f}")
            else:
                logging.info(f"No features above the threshold of {top_features_threshold} for {arch_name}.")
            
            plt.figure(figsize=(10, 6))
            feature_labels = [f"{feature} (Feature {data['feature_index']})" for feature, data in important_features.items()][::-1]
            feature_values = [data["importance"] for data in important_features.values()][::-1]
            plt.barh(feature_labels, feature_values, align="center")
            plt.xlabel("Mean Decrease in Accuracy")
            plt.title(f"Feature Importance: {arch_name}")
            plt.tight_layout()
            plt.savefig(f"{plots_folder}/feature_importance_{arch_name.replace(' ', '_')}.png")
            plt.close()
            logging.info(f"Feature importance plot saved for {arch_name}.")
        
        except Exception as e:
            logging.error(f"Error calculating feature importance for {arch_name}: {e}")
    
    json_file_path = os.path.join(results_folder, "feature_importance.json")
    with open(json_file_path, "w") as f:
        json.dump(feature_importance_results, f, indent=4)
    
    logging.info("Feature importance calculation complete. Results saved.")

def visualize_network_architecture(architectures, X_train, y_train, plots_folder="./plots"):
    """
    Visualizes neural network architectures as directed graphs and saves the plots.

    Args:
        architectures (list): List of architectures. Each architecture is a dictionary with:
            - 'name' (str): Name of the architecture.
            - 'hidden_layer_sizes' (tuple): Sizes of the hidden layers.
        X_train (np.ndarray): Training feature matrix (used to determine input size).
        y_train (np.ndarray): Training target matrix (used to determine output size).
        plots_folder (str): Folder to save network architecture plots. Default is './plots'.

    Returns:
        None
    """
    os.makedirs(plots_folder, exist_ok=True)
    logging.info(f"Network visualizations will be saved in the folder: {plots_folder}")

    for arch in tqdm(architectures, desc="Visualizing architectures"):
        layer_sizes = [X_train.shape[1]] + list(arch["hidden_layer_sizes"]) + [y_train.shape[1]]

        G = nx.DiGraph()
        pos = {}
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            for i in range(layer_size):
                node_name = f"L{layer_idx}N{i}"
                G.add_node(node_name, layer=layer_idx)
                pos[node_name] = (layer_idx, -i)
                
                if layer_idx > 0:
                    for prev_idx in range(layer_sizes[layer_idx - 1]):
                        prev_node_name = f"L{layer_idx - 1}N{prev_idx}"
                        G.add_edge(prev_node_name, node_name)

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=False, node_size=500, node_color="skyblue", edge_color="gray")
        plt.title(f"Architecture: {arch['name']}")
        plt.savefig(f"{plots_folder}/architecture_{arch['name'].replace(' ', '_')}.png")
        plt.close()

    logging.info("Network visualizations saved.")

# -------------------------------
# Feature Computation Functions
# -------------------------------
def indices_retrieval(columns_scheme):
    x_dict = columns_scheme.get("x", {})
    
    past_station_base = x_dict.get("PAST_STATIONS_1_embedding_0")
    future_station_base = x_dict.get("FUTURE_STATIONS_1_embedding_0")
    past_delay_index = x_dict.get("PAST_DELAYS_1")
    
    if past_station_base is None or future_station_base is None or past_delay_index is None:
        raise ValueError("Missing required keys in columns_scheme['x']")
    
    past_station_1_indices = [past_station_base + i * 10 for i in range(8)]
    future_station_1_indices = [future_station_base + i * 10 for i in range(8)]
    
    features_config = {
        "past_station_1_indices": past_station_1_indices,
        "future_station_1_indices": future_station_1_indices,
        "past_delay_index": past_delay_index
    }
    
    return features_config

def compute_train_position_embedding(X, past_indices, future_indices):
    logging.info("Computing train position embedding (midpoint of past and future stations).")
    return ((X[:, past_indices] + X[:, future_indices]) / 2.0)

def euclidean_distance_matrix(positions):
    positions = positions.astype(np.float32)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def count_trains(distance_matrix):
    num_trains = distance_matrix.shape[0]
    count_trains = np.zeros(num_trains)

    for i in range(num_trains):
        row_values = [
            distance_matrix[i][j] for j in range(num_trains)
            if i != j  # Ignore diagonal to evaluate other trains
        ]
        count_trains[i] = np.sum(row_values)
    
    return count_trains
  
def min_non_diagonal_distances(distance_matrix):
    num_trains = distance_matrix.shape[0]
    min_distances = np.full(num_trains, np.inf)

    for i in range(num_trains):
        row_values = [
            distance_matrix[i][j] for j in range(num_trains)
            if i != j and distance_matrix[i][j] != np.inf  # Ignore diagonal and infinite distances
        ]
        
        if row_values:
            min_distances[i] = min(row_values)
        else:
            # Sets a default value if no valid neighbor is found
            min_distances[i] = 0.0

    return min_distances

def mean_distances(distance_matrix):
    num_trains = distance_matrix.shape[0]
    mean_distances = np.zeros(num_trains)

    for i in range(num_trains):
        row_values = [
            distance_matrix[i][j] for j in range(num_trains)
            if i != j and distance_matrix[i][j] != np.inf
        ]

        if not row_values:
            mean_distances[i] = 0
        else:
            mean_distances[i] = np.mean(row_values)

    return mean_distances

def mean_delays(distance_matrix, past_delays):
    num_trains = distance_matrix.shape[0]
    mean_delays = np.zeros(num_trains)

    for i in range(num_trains):
        valid_neighbors = [
            j for j in range(num_trains) if i != j and distance_matrix[i][j] != np.inf
        ]

        if valid_neighbors:
            neighbor_delays = past_delays[valid_neighbors]
            mean_delays[i] = np.mean(neighbor_delays)
        else:
            mean_delays[i] = 0 

    return mean_delays

def compute_new_features(radii, X, past_delay_index):
    """
    Compute new features per batches assigned before.
    
    Args:
        - radii (list): List of radii for range search.
        - X (np.ndarray): Data array containing train positions and other features.
        - past_delay_index (int): Index for the past delay feature in X.
    
    Returns:
        dict: A dictionary with computed features including number of trains, minimum distance,
              mean distance, mean delay, and global number of trains.
    """
    logging.basicConfig(filename='more_features.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    nb_trains = X.shape[0]
    num_radii = len(radii)
    
    num_trains_arr = np.zeros((nb_trains, num_radii))
    min_dist = np.full((nb_trains, num_radii), np.nan)
    mean_dist = np.zeros((nb_trains, num_radii))
    mean_delay = np.zeros((nb_trains, num_radii))
    
    # New feature: total number of trains in the current state time
    nb_trains_global = np.full((nb_trains, 1), nb_trains)

    # Euclidean distances for each pair of trains -> matrix nb_trains x nb_train
    distance_matrix = euclidean_distance_matrix(X[:, -8:])

    nb_trains_global[0] = distance_matrix.shape[0]

    # Loop over radii
    # Add to diagonal a value > than my radius to create a matrix of True and False, so a mask. 
    # Then we sum over each row to get the vector with the number of trains in this specific radius
    count = 0
    for radius in radii:
        # In the matrix keep distances < radius, set others (and diagonal) to np.inf
        masked_distance_matrix = distance_matrix.copy()
        np.fill_diagonal(masked_distance_matrix, np.inf)
        masked_distance_matrix[masked_distance_matrix >= radius] = np.inf

        # In the matrix the values are: True if distance is less than radius, False otherwise
        bool_distance_matrix = (masked_distance_matrix != np.inf)

        # Count the number of trains in this radius and put it in the variable
        active_trains_in_radius = count_trains(bool_distance_matrix)

        # Take the minimum distance for each train
        min_distances = min_non_diagonal_distances(masked_distance_matrix)

        # Take the average distance for each train
        mean_distance = mean_distances(masked_distance_matrix)

        # Take the average delay for each trains neighboring the current train
        past_delays = X[:, past_delay_index]
        avg_delay = mean_delays(masked_distance_matrix, past_delays)

        num_trains_arr[:, count] = active_trains_in_radius
        min_dist[:, count] = min_distances
        mean_dist[:, count] = mean_distance
        mean_delay[:, count] = avg_delay
        count += 1

    result = {
        "num_trains": num_trains_arr, # For each radius a separate one
        "min_distance": min_dist, # For each radius a separate one
        "mean_distance": mean_dist, # For each radius a separate one
        "mean_delay": mean_delay, # For each radius a separate one
        "nb_trains_global": nb_trains_global # Independent for radii, basically at each state_time, the number of active trains in the network (in the batch) is recorded.
    }

    return result

def process_state_time(args):
    logging.basicConfig(filename='more_features.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    st, indices, X_augmented, radii, features_config = args
    logging.info(f"Processing STATE_TIME: {st} ({len(indices)} samples)")

    X_filtered = X_augmented[indices, :]
    res = compute_new_features(radii, X_filtered, features_config["past_delay_index"])

    features_st = np.hstack([
        res["num_trains"],
        res["min_distance"],
        res["mean_distance"],
        res["mean_delay"],
        res["nb_trains_global"]
    ])
    
    return indices, features_st

