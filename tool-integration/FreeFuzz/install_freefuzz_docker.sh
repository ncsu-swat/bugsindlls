#!/bin/bash

git clone https://github.com/ise-uiuc/FreeFuzz.git
docker build -t freefuzz .
docker run --name freefuzz --gpus all -d freefuzz:latest
docker exec -it freefuzz mongorestore dump/