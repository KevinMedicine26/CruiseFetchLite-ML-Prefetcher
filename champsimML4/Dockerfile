FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    make \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    vim \
    wget \
    dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the ML model
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    tensorflow \
    numpy \
    matplotlib \
    pandas

# Copy ChampSim repository
COPY ChampSim /app/ChampSim

# Fix line endings and make scripts executable
RUN find /app/ChampSim -name "*.sh" -exec dos2unix {} \; && \
    chmod +x /app/ChampSim/*.sh

# Build ChampSim
WORKDIR /app/ChampSim
RUN ./build_champsim.sh hashed_perceptron no no no no lru 1

# Create directory for traces
RUN mkdir -p /app/ChampSim/traces

# Set environment variables for CUDA
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set default command
CMD ["bash"]
