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

运行指南
docker前三步骤（environment）
docker-compose build
## Running the Docker Container

To start the Docker container, run:
docker-compose up -d

To enter the container in interactive mode:
docker-compose exec champsim bash

进入容器并使脚本可执行：


chmod +x autorunCFlite.sh


在容器内运行脚本：
./autorunCFlite.sh


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
python3 ml_prefetch_sim.py train ./traces/471.omnetpp-s0.txt.xz --model ./model/model_471
```


### Generating Prefetches

```bash
cd /app/ChampSim
python3 ml_prefetch_sim.py generate ./traces/471.omnetpp-s0.txt.xz prefetches_471.txt --model ./model/model_471
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

### Mounting Additional Trace Folders

To mount the "D:\ChampsimML4\traces" folder to the Docker container, modify the `docker-compose.yml` file to add the following volume mapping:

```yaml
volumes:
  - ./ChampSim:/app/ChampSim
  - ./ChampSim/traces:/app/ChampSim/traces
  - ./ChampSim/results:/app/ChampSim/results
  - ./ChampSim/prefetch_files:/app/ChampSim/prefetch_files
  - D:\ChampsimML4\externaldata:/app/ChampSim/externaldata  # Mount external traces folder
```

Then you can access these traces inside the container at `/app/ChampSim/external_traces`.

For Windows users, you may need to use the following format instead:

```yaml
volumes:
  - ./ChampSim:/app/ChampSim
  - ./ChampSim/traces:/app/ChampSim/traces
  - ./ChampSim/results:/app/ChampSim/results
  - ./ChampSim/prefetch_files:/app/ChampSim/prefetch_files
  - D:/ChampsimML4/traces:/app/ChampSim/external_traces  # Windows path format
```

## Stopping the Docker Container

To stop the Docker container:

```bash
docker-compose down
