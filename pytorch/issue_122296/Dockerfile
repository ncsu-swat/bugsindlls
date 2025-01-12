FROM python:3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-12-1

ENV cudnn_version=8.9.2.26
ENV cuda_version=cuda12.1

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
RUN pip install gdown==5.2.0
RUN gdown --fuzzy https://drive.google.com/file/d/1c-TWGjCe9xQV4q8s0hfz9k-pwNCQQPmb/view?usp=sharing -O /tmp/
COPY test_issue_122296.py app/
COPY requirements.txt app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["pytest", "-sx"]