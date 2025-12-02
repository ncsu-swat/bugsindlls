FROM python:3.12.3

# Install CUDA 12.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-1

# Install cuDNN 9.x (CUDA 12 variant — compatible with CUDA 12.1)
RUN apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Copy application files
COPY test_issue_153276.py app/
COPY requirements.txt app/

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]