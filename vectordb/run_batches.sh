#!/bin/bash

# CONFIGURABLE
TOTAL_STEPS=100     # Set this high; script will exit early if no more docs
BATCH_SIZE=10       # Number of documents per batch
SCRIPT_PATH="main.py"  # Path to your Python script

for (( STEP=0; STEP<$TOTAL_STEPS; STEP++ ))
do
    echo "Running batch step: $STEP"

    # Capture output
    OUTPUT=$(python "$SCRIPT_PATH" --step=$STEP --batch_size=$BATCH_SIZE)

    echo "$OUTPUT"

    # Check for termination signal
    if echo "$OUTPUT" | grep -q "No more documents to process"; then
        echo "No more batches left. Exiting."
        break
    fi

    # Optional short pause
    # sleep 1
done

echo "Batch processing complete."
