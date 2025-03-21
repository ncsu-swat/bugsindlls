FROM mongodb/mongodb-community-server:latest
USER root
RUN apt-get update
RUN apt-get install -y python3.10
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-12-1
ENV cudnn_version=8.9.2.26
ENV cuda_version=cuda12.1
RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

COPY FreeFuzz FreeFuzz/
COPY postprocess.py FreeFuzz/
COPY preprocess.py FreeFuzz/
WORKDIR /FreeFuzz
RUN pip install -r requirements.txt
RUN pip install rapidfuzz