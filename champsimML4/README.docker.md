# ChampSim Docker Instructions

This document explains how to build and run the ChampSim project with the CruiseFetchLITE model using Docker.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (included in Docker Desktop)

## Directory Structure

The Docker configuration maintains the following directory structure:

```
champsimML4/
├── ChampSim/            # ChampSim simulation framework
│   ├── model.py         # Your CruiseFetchLITE model implementation
│   ├── ml_prefetch_sim.py # ML prefetcher simulation script
│   └── ...              # Other ChampSim files
├── Dockerfile           # Docker configuration
└── docker-compose.yml   # Docker Compose configuration
```

## Building the Docker Container

1. Open a terminal or command prompt
2. Navigate to the project directory where `Dockerfile` is located
3. Run the following command to build the Docker image:

```bash
docker-compose build
```

## Running the Docker Container

To start the Docker container, run:

```bash
docker-compose up -d
```

To enter the container in interactive mode:

```bash
docker-compose exec champsim bash
```

If you encounter any issues with the container not running or not being accessible, try the following:

1. Stop the container: `docker-compose down`
2. Rebuild the container: `docker-compose build`
3. Start the container again: `docker-compose up -d`
4. Then connect to it: `docker-compose exec champsim bash`

## Using ChampSim inside the Docker Container

Once inside the container, you can use ChampSim as normal:

### Training the Model

```bash
cd /app/ChampSim
python3 ml_prefetch_sim.py train ./traces/471.omnetpp-s0.txt.xz --model ./models/model_471.h5
```


### Generating Prefetches

```bash
cd /app/ChampSim
python3 ml_prefetch_sim.py generate ./traces/471.omnetpp-s0.txt.xz prefetches_471.txt --model ./models/model_471.h5
```

### Building ChampSim

```bash
cd /app/ChampSim
python3 ml_prefetch_sim.py build
```

### Running Simulations

```bash
cd /app/ChampSim
python3 ml_prefetch_sim.py run ./traces/471.omnetpp-s0.gz
```

### Building Different ChampSim Configurations

```bash
cd /app/ChampSim
./build_champsim.sh [branch_pred] [l1i_pref] [l1d_pref] [l2c_pref] [llc_pref] [llc_repl] [num_core]
```

## Directory Mapping

The following directories are mapped between your host machine and the Docker container:

- `./ChampSim/traces` → `/app/ChampSim/traces` - Place your traces here
- `./ChampSim/results` → `/app/ChampSim/results` - Simulation results will be stored here
- `./ChampSim/prefetch_files` → `/app/ChampSim/prefetch_files` - Prefetch files will be stored here

## Stopping the Docker Container

To stop the Docker container:

```bash
docker-compose down
