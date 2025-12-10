FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3.9 python3.9-distutils python3-pip

WORKDIR /app
COPY test_issue_71636.py requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["pytest", "-sx"]
