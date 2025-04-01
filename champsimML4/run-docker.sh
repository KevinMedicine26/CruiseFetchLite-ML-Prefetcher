#!/bin/bash

# Helper script for running ChampSim in Docker

# Create required directories if they don't exist
mkdir -p ChampSim/traces ChampSim/results ChampSim/prefetch_files

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to show help message
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build      - Build the Docker container"
    echo "  start      - Start the Docker container"
    echo "  stop       - Stop the Docker container"
    echo "  shell      - Open a shell in the running container"
    echo "  train      - Train the ML model (args: <trace>)"
    echo "  generate   - Generate prefetches (args: <trace>)"
    echo "  run        - Run a simulation (args: <trace>)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 shell"
    echo "  $0 train traces/trace1.trace"
}

# Process commands
case "$1" in
    build)
        docker-compose build
        ;;
    start)
        docker-compose up -d
        echo "Container started. Use '$0 shell' to open a shell."
        ;;
    stop)
        docker-compose down
        ;;
    shell)
        docker-compose exec champsim bash
        ;;
    train)
        if [ -z "$2" ]; then
            echo "Error: No trace file specified"
            echo "Usage: $0 train <trace_file>"
            exit 1
        fi
        docker-compose exec champsim python3 /app/ChampSim/ml_prefetch_sim.py train "$2"
        ;;
    generate)
        if [ -z "$2" ]; then
            echo "Error: No trace file specified"
            echo "Usage: $0 generate <trace_file>"
            exit 1
        fi
        docker-compose exec champsim python3 /app/ChampSim/ml_prefetch_sim.py generate "$2"
        ;;
    run)
        if [ -z "$2" ]; then
            echo "Error: No trace file specified"
            echo "Usage: $0 run <trace_file>"
            exit 1
        fi
        docker-compose exec champsim python3 /app/ChampSim/ml_prefetch_sim.py run "$2"
        ;;
    help|*)
        show_help
        ;;
esac
