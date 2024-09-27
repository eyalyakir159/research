import wandb
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
# Initialize a W&B run
import wandb

# Login to wandb
wandb.login(key="c539e6a7d6f8cab8c34d65af4fc680be5e3fe58d")

# Initialize the wandb API
api = wandb.Api()

# Define your project and entity
entity = "eyalyak-ben-gurion-university-of-the-negev"  # Replace with your wandb username or team
project = "thesis"  # Replace with your project name

# Function to run a 1D parameter sweep and save MAE and MSE arrays to a text file
def one_d_sweep(runs, param_sets, sweep_over_params):
    # Open a text file to write the results
    with open('sweep_results.txt', 'w') as file:
        # Loop through both param_sets and sweep_over_params in parallel
        for param_set, sweep_over_param in zip(param_sets, sweep_over_params):
            sweep_values = set()
            mae_values = []
            mse_values = []

            # Convert param_set to string and write it to the file
            file.write(f"Parameter Set: {str(param_set)}\n")
            file.write(f"Sweep Parameter: {sweep_over_param}\n")

            # Iterate through runs
            for run in runs:
                # Check if run matches the current parameter set (excluding the sweep parameter)
                if all(run.config.get(k) == v for k, v in param_set.items() if k != sweep_over_param):
                    sweep_value = run.config.get(sweep_over_param)
                    mae = run.summary.get("mae")
                    mse = run.summary.get("mse")

                    # If values exist, collect them
                    if sweep_value is not None and mae is not None and mse is not None:
                        sweep_values.add(sweep_value)
                        mae_values.append((sweep_value, mae))
                        mse_values.append((sweep_value, mse))

            # Sort values by the sweep parameter
            mae_values.sort(key=lambda x: x[0])
            mse_values.sort(key=lambda x: x[0])

            # Unpack values for writing
            sweep_values_sorted = [v[0] for v in mae_values]
            maes_sorted = [v[1] for v in mae_values]
            mses_sorted = [v[1] for v in mse_values]

            # Write sweep values, MAE, and MSE to the file
            file.write(f"Sweep Parameter Array: {sweep_values_sorted}\n")
            file.write(f"MAE: {maes_sorted}\n")
            file.write(f"MSE: {mses_sorted}\n\n")

    print("Results have been saved to 'sweep_results.txt'")




# Function to run a 2D parameter sweep and return hashmaps of MAE and RMSE
def two_d_sweep(runs, Paramaters, sweep_param_1, sweep_param_2):
    mae_map = defaultdict(dict)
    rmse_map = defaultdict(dict)

    # Iterate through runs
    for run in runs:
        if all(run.config.get(k) == v for k, v in Paramaters.items() if k != sweep_param_1 and k != sweep_param_2):
            param1_value = run.config.get(sweep_param_1)
            param2_value = run.config.get(sweep_param_2)
            mae = run.summary.get("mae")
            rmse = run.summary.get("rmse")

            if param1_value is not None and param2_value is not None and mae is not None and rmse is not None:
                mae_map[param1_value][param2_value] = mae
                rmse_map[param1_value][param2_value] = rmse

    # Convert to pandas DataFrame for heatmap plotting
    mae_df = pd.DataFrame(mae_map).T  # Transpose to match the heatmap structure
    rmse_df = pd.DataFrame(rmse_map).T

    # Plot the heatmaps
    plt.figure(figsize=(12, 6))

    # Plot MAE heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(mae_df, annot=True, cmap="YlGnBu")
    plt.title(f"MAE Heatmap ({sweep_param_1} vs {sweep_param_2})")
    plt.xlabel(sweep_param_2)
    plt.ylabel(sweep_param_1)

    # Plot RMSE heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(rmse_df, annot=True, cmap="YlGnBu")
    plt.title(f"RMSE Heatmap ({sweep_param_1} vs {sweep_param_2})")
    plt.xlabel(sweep_param_2)
    plt.ylabel(sweep_param_1)

    plt.tight_layout()
    plt.show()

    # Convert to standard dictionary to avoid defaultdict behavior during display
    return dict(mae_map), dict(rmse_map)


runs = api.runs(f"{entity}/{project}")

# Example usage
param_sets = [
    {
        "model": "HighDimLearner",
        "dataset": "ETTh1",
        "TopM": 0,
        "prediction_length": 96,
        "sequence_length": 96,
        "run_version": "Embedding Size Experiment",
    },
    {
        "model": "HighDimLearner",
        "dataset": "electricity",
        "TopM": 0,
        "prediction_length": 96,
        "sequence_length": 96,
        "run_version": "Embedding Size Experiment",
    },
    {
        "model": "HighDimLearner",
        "dataset": "ETTh1",
        "TopM": 0,
        "prediction_length": 336,
        "sequence_length": 96,
        "run_version": "Embedding Size Experiment",
    },
    {
        "model": "HighDimLearner",
        "dataset": "electricity",
        "TopM": 0,
        "prediction_length": 336,
        "sequence_length": 96,
        "run_version": "Embedding Size Experiment",
    },
    {
        "model": "HighDimLearner",
        "dataset": "ETTh1",
        "TopM": 0,
        "prediction_length": 336,
        "sequence_length": 96,
        "run_version": "Embedding Size Experiment",
    }

]
sweep_over_params = ["embed_size", "embed_size","embed_size","embed_size","TopM","TopM","TopM","TopM"]

# Call the function
one_d_sweep(runs, param_sets, sweep_over_params)