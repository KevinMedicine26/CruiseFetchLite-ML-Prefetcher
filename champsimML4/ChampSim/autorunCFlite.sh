#!/bin/bash

# ChampSim Automated Training and Testing Script
# This script automates the process of training, generating prefetches, building, and testing in ChampSim
#605.mcf-s0.trace.xz --no-base
#docker-compose build

#docker-compose up -d

#docker-compose exec champsim bash

#chmod +x autorunCFlite.sh

#./autorunCFlite.sh


# Trace file to use
TRACE="./traces/605.mcf-s0.txt.xz"
TRACE_TEST="./traces/605.mcf-s0.trace.xz"
MODEL="./prefetch_files/model605-WM20-PCembd128-CLUSembd48-Ncadi3-dpf2-sensitive"
PREFETCH_FILE="./prefetch_files/prefetches_605-WM20-PCembd128-CLUSembd48-Ncadi3-dpf2-sensitive.txt"
WARMUP_INSTRUCTIONS=20

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

# Main execution

echo "===== STEP 1: TRAINING MODEL ====="
echo "Running: python3 ml_prefetch_sim.py train $TRACE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_INSTRUCTIONS"
python3 ml_prefetch_sim.py train $TRACE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_INSTRUCTIONS
check_success
clear_memory
countdown 10

echo "===== STEP 2: GENERATING PREFETCHES ====="
echo "Running: python3 ml_prefetch_sim.py generate $TRACE $PREFETCH_FILE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_INSTRUCTIONS"
python3 ml_prefetch_sim.py generate $TRACE $PREFETCH_FILE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_INSTRUCTIONS
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
echo "Running: python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE --no-base"
python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE --no-base
check_success

echo "===== ALL STEPS COMPLETED SUCCESSFULLY ====="