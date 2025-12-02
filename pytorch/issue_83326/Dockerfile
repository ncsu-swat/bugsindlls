FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata python3.9 python3.9-distutils python3-pip wget && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    python -m pip install --upgrade pip

# Copy application files
WORKDIR /app
COPY test_issue_83326.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
CMD ["pytest", "-sx"]