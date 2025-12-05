FROM python:3.8

RUN apt-get update && \
    apt-get install -y wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-3

# cuDNN version compatible with CUDA 11.x
RUN apt-get install -y libcudnn8 libcudnn8-dev

COPY test_issue_88647.py app/
COPY requirements.txt app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]
