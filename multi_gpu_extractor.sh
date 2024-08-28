#!/bin/bash

audio_dir="/work/u3937558/soundon-asr/soundon-kmeans-100h"
output_dir="km_data_orig"
temp_dir="temp_audio_splits"  # Directory to store temporary lists of files for each GPU
log_dir="logdir"
num_gpus=4

# Function to clean up background processes
cleanup() {
    echo "Terminating background processes..."
    pkill -P $$
    rm -r $temp_dir
    exit 1
}
trap cleanup INT

# Create temporary directory for splits
mkdir -p $temp_dir
mkdir -p $log_dir
mkdir -p $output_dir

# Get a list of all audio files
audio_files=($audio_dir/*.wav)
total_files=${#audio_files[@]}

# Calculate the number of files per GPU
files_per_gpu=$(( (total_files + num_gpus - 1) / num_gpus ))

for i in $(seq 0 $((num_gpus - 1)))
do
    start_index=$(( i * files_per_gpu ))
    end_index=$(( (i + 1) * files_per_gpu ))

    # Create a list of files for this GPU
    gpu_files=("${audio_files[@]:start_index:files_per_gpu}")

    # Save the list to a temporary file
    printf "%s\n" "${gpu_files[@]}" > "$temp_dir/gpu_$i.txt"

    # Run the feature extraction on the files assigned to this GPU
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$i python main.py \
      --file_list "$temp_dir/gpu_$i.txt" \
      --output_dir "$output_dir" \
      > "$log_dir/gpu_$i.log" 2>&1 &
done
# Wait for all processes to finish
wait

# Clean up temporary files
rm -r $temp_dir

