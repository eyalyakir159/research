#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets and their corresponding paths
datasets=("traffic.csv")
data_paths=("./data/traffic/")
data_names=("custom")

# Define the model
model="HighDimLearner"

# Define fixed batch size
batch_size=8

# Define sequence lengths for failed runs
seqlens_720=(288 192 96 48 24 12)  # Failed for pred_len=720

# Define prediction lengths to run (only 336 and 720)
pred_lens=(720)

# Function to choose embedding size based on dataset
choose_embed_size() {
  if [ "$1" == "traffic.csv" ]; then
    echo 8  # Embed size for traffic dataset
  else
    echo 128  # Embed size for other datasets
  fi
}

topM=0  # Assuming TopM is a fixed value in this case

# Function to run the experiment on a specific GPU with the given seq_len
run_experiment() {
  local seq_len=$1
  local dataset=$2
  local data_path=$3
  local dataset_name=$4
  local pred_len=$5
  local embed_size=$6

  local nfft=$((seq_len / 4))  # Set nfft as seq_len / 4
  local hop_length=$((seq_len / 8))  # Set hop_length as seq_len / 8
  local label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

  echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size $embed_size on GPU 3"

  python -u "$SCRIPT_PATH" \
    --gpu 3 \
    --is_training 1 \
    --root_path "$data_path" \
    --data_path "$dataset" \
    --model "$model" \
    --model_id "${dataset_name}_seq${seq_len}_pred${pred_len}" \
    --data "$dataset_name" \
    --features "M" \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --itr 1 \
    --batch_size "$batch_size" \
    --train_epochs 10 \
    --embed_size "$embed_size" \
    --TopM "$topM" \
    --n_fft "$nfft" \
    --hop_length "$hop_length" \
    --run_version "Sweep for Seq Len" \
    --hide "None" &
}

# Function to wait until less than 3 jobs are running on GPU 3
wait_for_free_gpu_slots() {
  while [ "$(pgrep -f 'gpu 3' | wc -l)" -ge 3 ]; do
    sleep 5
  done
}

# Assign jobs for failed runs, distributing across GPU 3 with concurrency control
for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${data_names[i]}"
    data_path="${data_paths[i]}"

    embed_size=$(choose_embed_size "$dataset")  # Choose embed_size based on dataset

    # Run for pred_len=720 on GPU 3
    pred_len=720
    for idx in "${!seqlens_720[@]}"; do
        wait_for_free_gpu_slots  # Ensure only 3 concurrent jobs
        run_experiment "${seqlens_720[idx]}" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
    done
done

# Wait for all background processes to complete before exiting
wait
