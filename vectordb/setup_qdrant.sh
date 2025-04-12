#!/bin/bash

# Set default values for arguments
CHUNK_DATA="no"
OVERWRITE_DB="no"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunk_data)
      if [[ "$2" == "yes" || "$2" == "no" ]]; then
        CHUNK_DATA="$2"
        shift 2
      else
        echo "Invalid value for --chunk_data. Use 'yes' or 'no'."
        exit 1
      fi
      ;;
    --overwrite_db)
      if [[ "$2" == "yes" || "$2" == "no" ]]; then
        OVERWRITE_DB="$2"
        shift 2
      else
        echo "Invalid value for --overwrite_db. Use 'yes' or 'no'."
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set variables
QDRANT_CONTAINER_NAME="qdrant-local"
QDRANT_PORT=6333

# Function to check and start Qdrant Docker container if not running
function setup_qdrant_docker() {
  echo "Checking for Qdrant container..."

  # Check if the Qdrant container is already running
  if [[ "$(docker ps -q -f name=$QDRANT_CONTAINER_NAME)" ]]; then
    echo "Qdrant is already running."
  else
    echo "Starting Qdrant Docker container..."
    docker run -d \
      --name $QDRANT_CONTAINER_NAME \
      -p $QDRANT_PORT:6333 \
      qdrant/qdrant
    echo "Waiting for Qdrant to be ready..."
    sleep 5  # Give it time to initialize
  fi
}

# Function to chunk data (run run_batches.sh)
function chunk_data() {
  if [[ $CHUNK_DATA == "yes" ]]; then
    echo "Chunking data by running run_batches.sh..."
    ./run_batches.sh
  else
    echo "Skipping data chunking (run_batches.sh)..."
  fi
}

# Function to run Qdrant load script
function run_qdrant_load() {
  echo "Running qdrant_load.py with --overwrite_db=$OVERWRITE_DB"
  python3 qdrant_load.py --rewrite "$OVERWRITE_DB"
}

# Main execution flow
echo "Setting up Qdrant Docker container..."
setup_qdrant_docker

# Run data chunking if argument is passed
chunk_data

# Run Qdrant loading process
run_qdrant_load

echo "Setup complete!"
