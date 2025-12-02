FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python 3.10 and tools
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip wget && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip

# Copy application files
WORKDIR /app
COPY test_issue_112494.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]
