#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define dataset and their corresponding paths
dataset="electricity.csv"
data_path="./data/electricity/"
data_name="custom"

# Define the model
model="HighDimLearner"

# Define fixed batch size
batch_size=8

# Define prediction lengths and corresponding hop lengths for the electricity dataset
pred_len_96=(3)      # Hop lengths for seq_len=96
pred_len_336=(32 24 16 12 8 6 4 3)  # Hop lengths for seq_len=336

# Function to choose nfft based on hop_length
choose_nfft() {
  if [ "$1" -eq 32 ]; then
    echo 48  # Set nfft to 48 when hop_length is 32
  else
    echo $((2 * $1))  # Set nfft to 2 times hop_length otherwise
  fi
}

topM=0  # Assuming TopM is a fixed value in this case

# Run only on GPU 4
gpu=2

# Initialize a counter to limit the number of parallel runs
run_counter=0
max_parallel_runs=3

# Function to manage parallel runs and limit to 2 at a time
manage_parallel_runs() {
    run_counter=$((run_counter + 1))
    if [ "$run_counter" -ge "$max_parallel_runs" ]; then
        wait  # Wait for all background processes to complete
        run_counter=0  # Reset the counter after waiting
    fi
}

# First do seq_len 96 with corresponding hop lengths
seq_len=96
label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

for hop_length in "${pred_len_96[@]}"; do
    nfft=$(choose_nfft "$hop_length")  # Choose nfft based on hop_length

    echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $seq_len, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size 16 on GPU $gpu"

    python -u "$SCRIPT_PATH" \
      --gpu "$gpu" \
      --is_training 1 \
      --root_path "$data_path" \
      --data_path "$dataset" \
      --model "$model" \
      --model_id "${data_name}_$seq_len" \
      --data "$data_name" \
      --features "M" \
      --seq_len "$seq_len" \
      --label_len "$label_len" \
      --pred_len 96 \
      --itr 1 \
      --batch_size "$batch_size" \
      --train_epochs 10 \
      --embed_size 16 \
      --TopM "$topM" \
      --n_fft "$nfft" \
      --hop_length "$hop_length" \
      --run_version "Sweep for Hop Length" \
      --hide "None" &

    manage_parallel_runs  # Check and manage parallelism
done

echo "done with 96"

for hop_length in "${pred_len_336[@]}"; do
    nfft=$(choose_nfft "$hop_length")  # Choose nfft based on hop_length

    echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len 336, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size 16 on GPU $gpu"

    python -u "$SCRIPT_PATH" \
      --gpu "$gpu" \
      --is_training 1 \
      --root_path "$data_path" \
      --data_path "$dataset" \
      --model "$model" \
      --model_id "${data_name}_$seq_len" \
      --data "$data_name" \
      --features "M" \
      --seq_len "$seq_len" \
      --label_len "$label_len" \
      --pred_len 336 \
      --itr 1 \
      --batch_size "$batch_size" \
      --train_epochs 10 \
      --embed_size 16 \
      --TopM "$topM" \
      --n_fft "$nfft" \
      --hop_length "$hop_length" \
      --run_version "Sweep for Hop Length" \
      --hide "None" &

    manage_parallel_runs  # Check and manage parallelism
done

wait  # Wait for all remaining background processes to finish
