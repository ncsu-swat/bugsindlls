#!/bin/bash

if [ ! -d FreeFuzz ] ; then
    git clone https://github.com/ise-uiuc/FreeFuzz.git FreeFuzz
else
    cd FreeFuzz
    git pull https://github.com/ise-uiuc/FreeFuzz.git
    cd ..
fi

docker build -t freefuzz .
docker run --name freefuzz --gpus all -d freefuzz:latest
docker exec -it freefuzz mongorestore dump/