FROM python:3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-11-8

ENV cudnn_version=8.9.4.25
ENV cuda_version=cuda11.8

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
RUN pip3 install numpy==1.26.3
RUN pip3 install --pre torch==2.3.0.dev20240301+cu118 --index-url https://download.pytorch.org/whl/nightly
RUN pip3 install --pre torchaudio==2.2.0.dev20240301+cu118 --index-url https://download.pytorch.org/whl/nightly
RUN pip3 install --pre torchvision==0.18.0.dev20240301+cu118 --index-url https://download.pytorch.org/whl/nightly
RUN pip install -U pytest
COPY test_issue_121253.py app/
WORKDIR /app
CMD ["pytest", "-sx"]