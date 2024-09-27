#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets and their corresponding paths
datasets=("ETTh1.csv")
data_paths=("./data/ETT/")
data_names=("ETTh1")

# Define the model
model="HighDimLearner"

# Define fixed batch size
batch_size=8

# Define sequence lengths to test and their assigned GPUs
gpu_1_seqlens=(672)
gpu_2_seqlens=(576)
gpu_3_seqlens=(12 48)
gpu_4_seqlens=(192)
gpu_5_seqlens=(480 24)
gpu_6_seqlens=(96 288)

# Define prediction lengths to test
pred_lens=(96 192 336 720)

# Function to choose embedding size based on dataset
choose_embed_size() {
  if [ "$1" == "electricity.csv" ]; then
    echo 16  # Embed size for electricity dataset
  else
    echo 128  # Embed size for ETTh1 and ETTm1
  fi
}

topM=0  # Assuming TopM is a fixed value in this case

# Function to run the experiment on a specific GPU with the given seq_len
run_experiment() {
  local gpu=$1
  local seq_len=$2
  local dataset=$3
  local data_path=$4
  local dataset_name=$5
  local pred_len=$6
  local embed_size=$7

  local nfft=$((seq_len / 4))  # Set nfft as seq_len / 4
  local hop_length=$((seq_len / 8))  # Set hop_length as seq_len / 8
  local label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

  echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size $embed_size on GPU $gpu"

  python -u "$SCRIPT_PATH" \
    --gpu "$gpu" \
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

# Assign jobs to each GPU
for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${data_names[i]}"
    data_path="${data_paths[i]}"

    embed_size=$(choose_embed_size "$dataset")  # Choose embed_size based on dataset

    # Loop through prediction lengths
    for pred_len in "${pred_lens[@]}"; do


        # Run jobs on GPU 1 (seq_len 672)
        for seq_len in "${gpu_1_seqlens[@]}"; do
            run_experiment 1 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        # Run jobs on GPU 2 (seq_len 576)
        for seq_len in "${gpu_2_seqlens[@]}"; do
            run_experiment 2 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        # Run jobs on GPU 3 (seq_len 12, 48)
        for seq_len in "${gpu_3_seqlens[@]}"; do
            run_experiment 3 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        # Run jobs on GPU 4 (seq_len 192)
        for seq_len in "${gpu_4_seqlens[@]}"; do
            run_experiment 4 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        # Run jobs on GPU 5 (seq_len 480)
        for seq_len in "${gpu_5_seqlens[@]}"; do
            run_experiment 5 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        # Run jobs on GPU 6 (seq_len 24, 96)
        for seq_len in "${gpu_6_seqlens[@]}"; do
            run_experiment 6 "$seq_len" "$dataset" "$data_path" "$dataset_name" "$pred_len" "$embed_size"
        done

        wait  # Wait for all background processes to complete before starting the next pred_len

    done
done

wait  # Wait for all remaining background processes to finish
