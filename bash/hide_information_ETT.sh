#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets and their corresponding names
datasets=("ETTh1.csv" "ETTm1.csv")
data_path=("./data/ETT/" "./data/ETT/")
dataset_names=("ETTh1" "ETTm1")
# Define the model
model="HighDimLearner"

# Define batch size
batch_size=8

# Define topM
topM=0

# Define sequence length and prediction lengths
seq_len=96
pred_lens=(96 192 336 720)

# Define the hide argument (real/imaginary outputs or weights)
hide_options=("MIXR" "MIXI")

# Total number of GPUs
num_gpus=7

# GPU counter
gpu=0

# Total run counter
run_counter=0

# Maximum number of parallel runs before waiting
max_parallel_runs=7

# Loop over datasets, models, prediction lengths, and hide options
for hide_option in "${hide_options[@]}"; do
  for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${dataset_names[i]}"  # Get the corresponding dataset name
    dmodel="${d_model[i]}"
    for pred_len in "${pred_lens[@]}"; do
      label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

      echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $topM, hide_option $hide_option on GPU $gpu"

      # Run the Python script on the assigned GPU
      python -u "$SCRIPT_PATH" \
        --gpu "$gpu" \
        --is_training 1 \
        --root_path "./data/ETT/" \
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
        --embed_size 128 \
        --TopM "$topM" \
        --run_version "Hide Parts" \
        --hide "$hide_option" &

      # Increment GPU counter and run counter
      gpu=$(( (gpu + 1) % num_gpus ))
      run_counter=$((run_counter + 1))

      # If 21 runs have been started, wait for them to finish
      if [ "$run_counter" -ge "$max_parallel_runs" ]; then
        echo "Waiting for $run_counter runs to finish..."
        wait  # Wait for all processes to complete
        run_counter=0  # Reset the run counter after waiting
      fi
    done
  done
done

wait  # Wait for all remaining background processes to finish
