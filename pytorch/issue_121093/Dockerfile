FROM python:3.8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-11-8

ENV cudnn_version=8.9.4.25
ENV cuda_version=cuda11.8

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
COPY test_issue_121093.py app/
COPY buggy_code.py app/
COPY requirements.txt app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["pytest", "-sx"]