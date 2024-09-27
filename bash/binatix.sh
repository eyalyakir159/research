#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets
data="Binatix"
d_model=(252)

# Define sequence lengths
seq_len=96
predslen=(1)

# Define batch sizes
batch_sizes=(64)

# Define model-specific parameters
models=("FreLinear" "test_Linear")
gpu=5

# Loop over models
for model in "${models[@]}"; do
    # Loop over prediction lengths
    for pre_len in "${predslen[@]}"; do
        # Loop over batch sizes
        for batch_size in "${batch_sizes[@]}"; do
            # Run the Python script with the current parameters
            python -u "$SCRIPT_PATH" \
                --gpu "$gpu" \
                --is_training 1 \
                --root_path "./dataset/" \
                --data_path "ccver11a" \
                --model "$model" \
                --model_id "${data}_ccver11a_${model}" \
                --data "$data" \
                --features "MS" \
                --seq_len "$seq_len" \
                --label_len 48 \
                --pred_len "$pre_len" \
                --itr 1 \
                --batch_size "$batch_size" \
                --enc_in "${d_model[0]}" \
                --dec_in "${d_model[0]}" \
                --c_out "${d_model[0]}" \
                --train_epochs 5 \
                --channel_independence 0 \
                --wandb True \
                --run_version "${data} Run for ${model}" &
        done
    done
    wait
done
wait

