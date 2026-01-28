#!/bin/bash

if [ ! -d DeepREL ] ; then
    git clone https://github.com/ise-uiuc/DeepREL.git DeepREL
else
    cd DeepREL
    git pull https://github.com/ise-uiuc/DeepREL.git
    cd ..
fi

docker build -t deeprel .
docker run --name deeprel --gpus all -d deeprel:latest
docker exec -it deeprel mongorestore -h 127.0.0.1:27017 --db tf dump/tf/
docker exec -it deeprel mongorestore -h 127.0.0.1:27017 --db torch dump/torch/
