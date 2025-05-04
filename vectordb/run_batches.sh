#!/bin/bash

echo "Starting batch processing..."

# Directory for output files (relative to project root)
OUTPUT_DIR="../embeddings"

# Clean up existing chunked embedding files in root directory
echo "Cleaning up existing chunked embedding files in root directory..."
rm -f ../chunked_embeddings_*.json

# Create output directory if it doesn't exist
mkdir -p $(dirname $(dirname $0))/$OUTPUT_DIR

# Run batches sequentially
step=0
while true; do
    echo "Running batch step: $step"
    
    # Run the Python script for this batch
    python ./vectordb/main.py --step=$step --output_dir=$OUTPUT_DIR
    
    # Check exit code
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "No more batches left. Exiting."
        break
    fi
    
    # Increment step
    ((step++))
done

echo "Batch processing complete."