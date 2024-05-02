FROM python:3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-12-1

ENV cudnn_version=8.9.2.26
ENV cuda_version=cuda12.1

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
COPY test_issue_120085.py app/
COPY requirements.txt app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --pre torch==2.4.0.dev20240317+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install --pre pytorch-triton==3.0.0+989adb9a29 --index-url https://download.pytorch.org/whl/nightly/cu121 
CMD ["pytest", "-sx"]
