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
pred_lens=(336)

embed_sizes_electricity_336=(1 16 32 64 128 256)

# Define fixed FFT points and hop length
nfft=16
hop_length=8

topM=0  # Assuming TopM is a fixed value in this case

# Define the available GPUs
gpus=(1 2 3 4 5 6)

# Initialize GPU counter
gpu_counter=0

# Loop over datasets and prediction lengths
for i in "${!datasets[@]}"; do
    dataset="${datasets[i]}"
    dataset_name="${data_names[i]}"

    run_counter=0  # Counter to track the number of parallel runs

    for pred_len in "${pred_lens[@]}"; do
        label_len=$((seq_len / 2))  # Compute label_len as half of seq_len

        # Define appropriate embedding sizes based on dataset and prediction length
        if [[ "$dataset_name" == "custom" && "$pred_len" == 96 ]]; then
            embed_sizes=("${embed_sizes_electricity_96[@]}")
        elif [[ "$dataset_name" == "custom" && "$pred_len" == 336 ]]; then
            embed_sizes=("${embed_sizes_electricity_336[@]}")
        else
            continue  # Skip cases that don't need to be run
        fi

        # Loop over embedding sizes
        for embed_size in "${embed_sizes[@]}"; do
            # Assign task to a GPU in round-robin fashion
            gpu="${gpus[gpu_counter]}"

            echo "Running model $model on dataset $dataset with seq_len $seq_len, label_len $label_len, pred_len $pred_len, batch_size $batch_size, topM $topM, hop_length $hop_length, nfft $nfft, embed_size $embed_size on GPU $gpu"

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
              --run_version "Embedding Size Experiment" \
              --hide "None" &

            # Update GPU counter in a round-robin manner
            gpu_counter=$(( (gpu_counter + 1) % 6 ))

            # Increment run counter based on embedding size
            if [[ "$embed_size" -eq 128 || "$embed_size" -eq 256 || "$embed_size" -eq 512 ]]; then
                run_counter=$((run_counter + 1))  # Larger embed sizes count as 3 runs
            else
                run_counter=$((run_counter + 1))  # Smaller embed sizes count as 2 runs
            fi

            # Wait if total runs reach 6 (as per your GPU count)
            if [ "$run_counter" -ge 6 ]; then
                wait  # Wait for all background processes to complete
                run_counter=0  # Reset the counter after waiting
            fi
        done
    done
done

wait  # Wait for all remaining background processes to finish
