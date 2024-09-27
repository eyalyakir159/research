#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets and their corresponding names
datasets=("ETTh1.csv" "electricity.csv")
data_path=("./data/ETT/" "./data/electricity/")
data_names=("ETTh1" "custom")

# Define the model
model="HighDimLearner"

# Define fixed parameters
batch_size=8
nfft=48
hop_length=6
# Define sequence length and prediction lengths
seq_len=96
pred_lens=(96 336)

# Define M values to evaluate
M_values=(1 2 4 8 12 16 20 0)

# List of GPUs to use
gpus=(1 2 3 4)
gpu_count=${#gpus[@]}

gpu_task_counter=0  # Counter to track tasks per GPU

# Loop over datasets and prediction lengths
for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${data_names[i]}"

    # Set embed_size based on the dataset
    if [ "$dataset" == "electricity.csv" ]; then
        embed_size=64
    else
        embed_size=128
    fi

    for pred_len in "${pred_lens[@]}"; do
        label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

        # Loop over M values
        for M in "${M_values[@]}"; do

            # Select the current GPU by rotating over the available GPUs
            gpu_idx=$(( gpu_task_counter / 2 % gpu_count ))
            gpu="${gpus[$gpu_idx]}"

            echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $M, hop_length $hop_length, nfft $nfft, embed_size $embed_size, M $M on GPU $gpu"

            # Run the Python script on the selected GPU
            python -u "$SCRIPT_PATH" \
              --gpu "$gpu" \
              --is_training 1 \
              --root_path "${data_path[i]}" \
              --data_path "$dataset" \
              --model "$model" \
              --model_id "${dataset_name}_$pred_len" \
              --data "$dataset_name" \
              --features "M" \
              --seq_len "$seq_len" \
              --label_len "$label_len" \
              --pred_len "$pred_len" \
              --itr 1 \
              --batch_size "$batch_size" \
              --train_epochs 10 \
              --embed_size "$embed_size" \
              --TopM "$M" \
              --n_fft "$nfft" \
              --hop_length "$hop_length" \
              --run_version "M Experiment" \
              --hide "None" &

            # Increment task counter
            gpu_task_counter=$((gpu_task_counter + 1))

            # If the number of parallel runs reaches 8, wait for them to finish
            if [ "$gpu_task_counter" -ge 8 ]; then
                wait  # Wait for all background processes to complete
                gpu_task_counter=0  # Reset the counter after waiting
            fi
        done
    done
done

wait  # Wait for all remaining background processes to finish
