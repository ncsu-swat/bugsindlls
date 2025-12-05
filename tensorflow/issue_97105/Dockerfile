FROM ubuntu:22.04

# Needed to avoid tzdata interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install base tools + Python 3.10
RUN apt-get update && apt-get install -y \
   python3 python3-pip python3-venv python3-dev \
   wget gnupg curl ca-certificates

# Install CUDA repo key (Ubuntu 22.04 version)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
   && dpkg -i cuda-keyring_1.1-1_all.deb \
   && apt-get update

# Install CUDA 12.3 + cuDNN 9.x
RUN apt-get install -y \
   cuda-toolkit-12-3 \
   libcudnn9-cuda-12 \
   libcudnn9-dev-cuda-12

# Copy project
WORKDIR /app
COPY test_issue_97105.py .
COPY requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD ["pytest", "-sx"]