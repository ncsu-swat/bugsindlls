FROM python:3.12.3

# Install CUDA 12.4
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-4

# cuDNN version compatible with CUDA 12.x
RUN apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Copy application files
COPY test_issue_153133.py app/
COPY requirements.txt app/

WORKDIR /app

# Install Python requirements
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]
