#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets

datasets=("ETTh1.csv" "ETTm1.csv" "weather.csv" "exchange_rate.csv" "electricity.csv" "traffic.csv")
d_model=(7 7 21 8 321 862)

# Define sequence lengths
seq_lens=(96 $((96*2)) $((96*4)) $((96*6)))
predslen=(96 192 336 720)

# Define batch sizes
batch_sizes=(4 8 16 32)

# Define model-specific parameters
model="YakirHighDimLinear2"
gpu=3
nfft_values=(24 32 48)
hop_lengths=(8 6 4 16 32)

# Define topM values for different nfft values
declare -A topM_values
topM_values[24]="0 4 8"
topM_values[32]="0 8 12"
topM_values[48]="0 8 12 16"

# Loop over all combinations of datasets, sequence lengths, batch sizes, nfft, and hop lengths
for dataset_index in "${!datasets[@]}"; do
  dataset="${datasets[$dataset_index]}"
  model_dim="${d_model[$dataset_index]}"

  for seq_len in "${seq_lens[@]}"; do
    for pre_len in "${predslen[@]}"; do
      label_len=$((seq_len / 2))  # Compute label_len as half of seq_len
      for batch_size in "${batch_sizes[@]}"; do
        for nfft in "${nfft_values[@]}"; do
          for topM in ${topM_values[$nfft]}; do
            for hop_length in "${hop_lengths[@]}"; do
              echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, batch_size $batch_size, nfft $nfft, topM $topM, hop_length $hop_length"
              # Run the Python script
              python -u "$SCRIPT_PATH" \
                --gpu "$gpu" \
                --is_training 1 \
                --root_path "./dataset/" \
                --data_path "$dataset" \
                --model "$model" \
                --model_id "${dataset%.csv}" \
                --data "${dataset%.csv}" \
                --features "M" \
                --seq_len "$seq_len" \
                --label_len "$label_len" \
                --pred_len "$pre_len" \
                --itr 1 \
                --batch_size "$batch_size" \
                --enc_in "$model_dim" \
                --dec_in "$model_dim" \
                --c_out "$model_dim" \
                --train_epochs 10 \
                --n_fft "$nfft" \
                --TopM "$topM" \
                --hop_length "$hop_length" \
                --channel_independence 0 \
                --run_version "Last SWAP" &
            done
            wait  # Wait for all background processes to complete before starting the next round
          done
        done
      done
    done
  done
done
