FROM ubuntu:22.04

# Avoid tzdata interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install base tools + Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-venv python3.9-dev python3-pip \
    wget gnupg curl ca-certificates \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install CUDA repo key
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install CUDA 12.3 + cuDNN 9.x
RUN apt-get install -y \
    cuda-toolkit-12-3 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12

# Set working directory
WORKDIR /app

# Copy project files
COPY buggy_code.py .
COPY test_issue_135428.py .
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Run pytest by default
CMD ["pytest", "-sx"]
