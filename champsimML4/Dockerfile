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
    gnupg2 \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8 \
    libnvonnxparsers8=8.5.3-1+cuda11.8 \
    libnvparsers8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
    libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
    libnvparsers-dev=8.5.3-1+cuda11.8 \
    python3-libnvinfer=8.5.3-1+cuda11.8 \
    python3-libnvinfer-dev=8.5.3-1+cuda11.8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the ML model
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    tensorflow \
    numpy \
    matplotlib \
    pandas \
    pyyaml \
    scikit-learn \
    tensorrt==8.5.3.1

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

# Set environment variables for CUDA (fixed legacy format)
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Set default command
CMD ["bash"]