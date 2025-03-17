import json
import numpy as np
import matplotlib.pyplot as plt

def plot_improvements_from_combined(json_file_path, output_dir="."):
    """
    Reads a combined metrics JSON file and produces two plots:
    
    1. A plot showing, for each delay-delta bin (each row in "MAE_Delay"),
       the improvement (translation MAE - model MAE) vs. the number of training months used.
    2. A plot showing the average improvement over horizon bins greater than 30 min 
       (assumed to be bins with index >= 1 in "MAE_Horizon") vs. the number of training months used.
       
    Parameters:
      json_file_path (str): Path to the combined JSON metrics file.
      output_dir (str): Directory where the plots will be saved.
      
    The function saves two plots:
      - progressive_delay_improvement.png
      - progressive_horizon_improvement.png
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # Lists to store:
    #   - progressive run number (assumed to be embedded in the key as "prog_<n>")
    #   - delay_improvements: a list of lists, one per progressive run,
    #     where each inner list contains improvements for each delay-delta bin.
    #   - horizon_improvements: average improvement (translation - model) computed over horizon bins (index>=1)
    prog_numbers = []
    delay_improvements_list = []  
    horizon_improvements = []  
    
    for key, value in data.items():
        # Expect key format like "metrics_XGBoost_prog_3"
        try:
            prog_num = int(key.split("_prog_")[-1])
        except Exception as e:
            print(f"Skipping key {key} due to error: {e}")
            continue
        prog_numbers.append(prog_num)
        
        # Process MAE_Delay: each row is [model_MAE, translation_MAE]
        mae_delay = value.get("MAE_Delay", [])
        # Compute improvement for each delay bin: (translation - model)
        improvements_delay = [row[1] - row[0] for row in mae_delay]
        delay_improvements_list.append(improvements_delay)
        
        # Process MAE_Horizon: assume that the first bin corresponds to <= 30 min.
        # For horizons > 30 min, we take bins with index >= 1.
        mae_horizon = value.get("MAE_Horizon", [])
        if len(mae_horizon) < 2:
            horizon_improvements.append(None)
        else:
            improvements_horizon = [row[1] - row[0] for row in mae_horizon[1:]]
            avg_improvement_horizon = np.mean(improvements_horizon)
            horizon_improvements.append(avg_improvement_horizon)
    
    prog_numbers = np.array(prog_numbers)
    delay_improvements = np.array(delay_improvements_list) 
    horizon_improvements = np.array(horizon_improvements)
    
    sorted_idx = np.argsort(prog_numbers)
    prog_numbers_sorted = prog_numbers[sorted_idx]
    delay_improvements_sorted = delay_improvements[sorted_idx, :]
    horizon_improvements_sorted = horizon_improvements[sorted_idx]
    
    # ---------------------------
    # Plot 1: Delay Improvement per Delay-Delta Bin
    # ---------------------------
    plt.figure(figsize=(10, 6))
    num_delay_bins = delay_improvements_sorted.shape[1]
    for j in range(num_delay_bins):
        plt.plot(prog_numbers_sorted, delay_improvements_sorted[:, j], marker='o', label=f"Delay Bin {j+1}")
    plt.xlabel("Number of Training Months Used")
    plt.ylabel("Improvement (Translation MAE - Model MAE)")
    plt.title("Progressive Improvement per Delay Delta Bin")
    plt.legend()
    plt.grid(True)
    delay_plot_path = f"{output_dir}/progressive_delay_improvement.png"
    plt.savefig(delay_plot_path)
    plt.close()
    
    # ---------------------------
    # Plot 2: Average Horizon Improvement (for Horizons > 30 min)
    # ---------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(prog_numbers_sorted, horizon_improvements_sorted, marker='o', color='red', label="Horizon > 30 min")
    plt.xlabel("Number of Training Months Used")
    plt.ylabel("Average Improvement (Translation MAE - Model MAE)")
    plt.title("Progressive Improvement over Horizons > 30 min")
    plt.legend()
    plt.grid(True)
    horizon_plot_path = f"{output_dir}/progressive_horizon_improvement.png"
    plt.savefig(horizon_plot_path)
    plt.close()
    
    print(f"Delay improvement plot saved to: {delay_plot_path}")
    print(f"Horizon improvement plot saved to: {horizon_plot_path}")

plot_improvements_from_combined("/Users/mac/Desktop/train_delay_prediction/full_year/xgboost/results/combined_metrics.json", output_dir="/Users/mac/Desktop/train_delay_prediction/full_year/xgboost/plots")
