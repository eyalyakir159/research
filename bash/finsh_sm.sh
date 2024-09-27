#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define dataset and its corresponding path and name
dataset="electricity.csv"
data_path="./data/electricity/"
data_name="custom"

# Define the model
model="HighDimLearner"

# Define fixed parameters
batch_size=8
nfft=48
hop_length=6
seq_len=96

# Prediction lengths and corresponding M values
pred_lens=(96 336)
topM_values_96=(16 20)  # Only 16 and 20 for pred_len 96
M_values_336=(1 2 4 8 12 16 20 0)  # All M values for pred_len 336

# List of GPUs to use (assuming you have 7 GPUs)
gpus=(0 1 2 3 4 5 6)
gpu_count=${#gpus[@]}

gpu_task_counter=0  # Counter to track tasks per GPU

# Function to select the current GPU in a round-robin fashion
get_next_gpu() {
    gpu_idx=$(( gpu_task_counter % gpu_count ))
    echo "${gpus[$gpu_idx]}"
}

# Loop over prediction lengths
for pred_len in "${pred_lens[@]}"; do
    label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

    # Set the appropriate M values depending on pred_len
    if [ "$pred_len" -eq 96 ]; then
        M_values=("${topM_values_96[@]}")  # Use topM values for pred_len 96
    elif [ "$pred_len" -eq 336 ]; then
        M_values=("${M_values_336[@]}")  # Use full M_values for pred_len 336
    fi

    # Loop over M values
    for M in "${M_values[@]}"; do
        # Get the next GPU to use
        gpu=$(get_next_gpu)

        echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $M, hop_length $hop_length, nfft $nfft, embed_size 64 on GPU $gpu"

        # Run the Python script on the selected GPU
        python -u "$SCRIPT_PATH" \
          --gpu "$gpu" \
          --is_training 1 \
          --root_path "$data_path" \
          --data_path "$dataset" \
          --model "$model" \
          --model_id "${data_name}_${pred_len}" \
          --data "$data_name" \
          --features "M" \
          --seq_len "$seq_len" \
          --label_len "$label_len" \
          --pred_len "$pred_len" \
          --itr 1 \
          --batch_size "$batch_size" \
          --train_epochs 10 \
          --embed_size 64 \
          --TopM "$M" \
          --n_fft "$nfft" \
          --hop_length "$hop_length" \
          --run_version "M Experiment" \
          --hide "None" &

        # Increment the task counter and manage GPU assignments
        gpu_task_counter=$((gpu_task_counter + 1))

        # If we've launched one job per GPU, wait for all to finish before launching new ones
        if [ "$gpu_task_counter" -ge "$gpu_count" ]; then
            wait  # Wait for all background processes to complete
            gpu_task_counter=0  # Reset the counter after waiting
        fi
    done
done

wait  # Wait for all remaining background processes to finish before exiting
