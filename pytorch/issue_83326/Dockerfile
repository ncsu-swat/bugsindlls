FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04


# Refresh certificates and GPG keys
RUN apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates \
       gnupg \
       wget \
   && rm -rf /var/lib/apt/lists/*


# Install Python 3.8 and pip
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3.8 \
       python3.8-venv \
       python3-pip \
   && ln -s /usr/bin/python3.8 /usr/bin/python \
   && pip3 install --upgrade pip \
   && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY test_issue_83326.py requirements.txt ./
RUN pip install -r requirements.txt


CMD ["pytest", "-sx"]