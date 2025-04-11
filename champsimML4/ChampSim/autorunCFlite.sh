#!/bin/bash

# CruiseFetchPro Automated Training and Testing Script
# This script automates the process of training, generating prefetches, building, and testing for CruiseFetchPro
# 
# INSTRUCTIONS: Edit the variables below to customize your training and testing process

#------------------------------------
# CONFIGURABLE PATHS AND PARAMETERS
#------------------------------------

# Trace files
TRACE_TRAIN="./traces/605.mcf-s0.txt.xz"     # Trace file for training
TRACE_GENERATE="./traces/605.mcf-s1.txt.xz"  # Trace file for generating prefetches
TRACE_TEST="./traces/605.mcf-s1.trace.xz"    # Trace file for testing

# Model and prefetch files
MODEL="./prefetch_files/model605-cruisefetchpro-sen-cluster-605s0"  # Path to save/load model
PREFETCH_FILE="./prefetch_files/prefetches_605s1-cruisefetchpro-sen-cluster.txt"  # Path for generated prefetches

# Configuration file
CONFIG_FILE="./model_config/cruisefetch_config_sen_cluster.yml"  # YAML configuration file path

# Parameters
WARMUP_TRAIN=20     # Number of warmup instructions for training
WARMUP_GENERATE=20  # Number of warmup instructions for generating
USE_NO_BASE=true    # Whether to use --no-base option in testing (true/false)

#------------------------------------
# UTILITY FUNCTIONS - DO NOT MODIFY
#------------------------------------

# Function to clear memory caches
clear_memory() {
    echo "Clearing memory cache..."
    sync                       # Synchronize cached writes to persistent storage
    echo 3 > /proc/sys/vm/drop_caches   # Clear pagecache, dentries and inodes
    echo "Memory cache cleared."
}

# Function to wait with a countdown
countdown() {
    local seconds=$1
    echo "Waiting $seconds seconds before next step..."
    for (( i=$seconds; i>0; i-- )); do
        echo -ne "$i...\r"
        sleep 1
    done
    echo -e "\nContinuing to next step."
}

# Check if a command succeeded
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: Previous command failed with exit code $?. Stopping execution."
        exit 1
    else
        echo "Success: Command completed successfully."
    fi
}

# Function to create a symbolic link from custom config to default name expected by model.py
create_config_symlink() {
    # Path to default config file expected by model.py
    DEFAULT_CONFIG_PATH="../cruisefetch_config.yml"
    
    # Remove any existing symlink or file
    if [ -e "$DEFAULT_CONFIG_PATH" ] || [ -L "$DEFAULT_CONFIG_PATH" ]; then
        echo "Removing existing default config file or symlink"
        rm "$DEFAULT_CONFIG_PATH"
    fi
    
    # Create a symbolic link from your custom config to the default location
    echo "Creating symbolic link from $CONFIG_FILE to $DEFAULT_CONFIG_PATH"
    ln -sf "$(realpath $CONFIG_FILE)" "$DEFAULT_CONFIG_PATH"
    
    # Verify the link was created successfully
    if [ -L "$DEFAULT_CONFIG_PATH" ]; then
        echo "Symbolic link created successfully"
    else
        echo "Failed to create symbolic link"
        exit 1
    fi
}

# Prepare no-base option for testing
NO_BASE_OPTION=""
if [ "$USE_NO_BASE" = true ]; then
    NO_BASE_OPTION="--no-base"
    echo "Using --no-base option for testing"
else
    echo "Not using --no-base option for testing"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Configuration file $CONFIG_FILE not found. Will use default configuration."
else
    echo "Using configuration file: $CONFIG_FILE"
    # Copy the configuration file to the model directory for reference
    CONFIG_DIR=$(dirname "$MODEL")
    mkdir -p "$CONFIG_DIR"
    cp "$CONFIG_FILE" "${CONFIG_DIR}/$(basename ${CONFIG_FILE})"
    echo "Configuration file copied to model directory for reference."
    
    # Create symbolic link to default config name for model.py
    create_config_symlink
fi

#------------------------------------
# MAIN EXECUTION
#------------------------------------

echo "===== STEP 1: TRAINING MODEL ====="
echo "Running: python3 ml_prefetch_sim.py train $TRACE_TRAIN --model $MODEL --config $CONFIG_FILE --num-prefetch-warmup-instructions $WARMUP_TRAIN"
python3 ml_prefetch_sim.py train $TRACE_TRAIN --model $MODEL --config $CONFIG_FILE --num-prefetch-warmup-instructions $WARMUP_TRAIN
check_success
clear_memory
countdown 10

echo "===== STEP 2: GENERATING PREFETCHES ====="
echo "Running: python3 ml_prefetch_sim.py generate $TRACE_GENERATE $PREFETCH_FILE --model $MODEL --config $CONFIG_FILE --num-prefetch-warmup-instructions $WARMUP_GENERATE"
python3 ml_prefetch_sim.py generate $TRACE_GENERATE $PREFETCH_FILE --model $MODEL --config $CONFIG_FILE --num-prefetch-warmup-instructions $WARMUP_GENERATE
check_success
clear_memory
countdown 10

echo "===== STEP 3: BUILDING ====="
echo "Running: python3 ml_prefetch_sim.py build"
python3 ml_prefetch_sim.py build
check_success
clear_memory
countdown 10

echo "===== STEP 4: TESTING ====="
echo "Running: python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE $NO_BASE_OPTION"
python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE $NO_BASE_OPTION
check_success

#echo "===== STEP 5: MODEL EVALUATION ====="
#echo "Running: python3 ml_prefetch_sim.py eval --results-dir ./results"
#python3 ml_prefetch_sim.py eval --results-dir ./results
#check_success

echo "===== ALL STEPS COMPLETED SUCCESSFULLY ====="
echo "Training and evaluation results for CruiseFetchPro with configuration file $CONFIG_FILE"
echo "Model saved to: $MODEL"
echo "Prefetches generated to: $PREFETCH_FILE"