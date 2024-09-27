#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/common_space_docker/network_ssd/eyal/thesis/run.py"

# Define datasets and their corresponding names
datasets=("electricity.csv")
data_path=("./data/electricity/")
data_names=("custom")

# Define the model
model="HighDimLearner"

# Define fixed batch size
batch_size=8

# Define sequence length and prediction lengths
seq_len=96
pred_lens=(96 336)

# Define NFFT values to evaluate
nfft_values=(8 12 16 24 32 48)

# Define fixed hop length and embedding sizes
hop_length=6

topM=0  # Assuming TopM is a fixed value in this case

# Run only on GPU 4
gpu=5

# Loop over datasets and prediction lengths
for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${data_names[i]}"

    # Set embed_size based on the dataset
    if [ "$dataset" == "electricity.csv" ]; then
        embed_size=16
        max_parallel_runs=2  # Only one run at a time for electricity
    else
        embed_size=128
        max_parallel_runs=6  # Up to 4 runs for ETTh1
    fi

    run_counter=0  # Counter to track the number of parallel runs

    for pred_len in "${pred_lens[@]}"; do
        label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

        # Loop over NFFT values
        for nfft in "${nfft_values[@]}"; do
            echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size $embed_size on GPU $gpu"

            # Run the Python script on GPU 4
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
              --TopM "$topM" \
              --n_fft "$nfft" \
              --hop_length "$hop_length" \
              --run_version "NFFT Experiment" \
              --hide "None" &

            # Increment run counter
            run_counter=$((run_counter + 1))

            # If the number of parallel runs reaches max_parallel_runs, wait for them to finish
            if [ "$run_counter" -ge "$max_parallel_runs" ]; then
                wait  # Wait for all background processes to complete
                run_counter=0  # Reset the counter after waiting
            fi
        done
    done
done

wait  # Wait for all remaining background processes to finish
