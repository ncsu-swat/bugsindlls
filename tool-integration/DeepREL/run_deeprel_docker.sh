#!/bin/bash

libname=${1:-torch}

if [[ "$libname" == "torch" ]] || [[ "$libname" == "pytorch" ]]; then
    docker exec -it deeprel bash -c "cd pytorch/src && python DeepREL.py"
elif [[ "$libname" == "tensorflow" ]] || [[ "$libname" == "tf" ]]; then
    docker exec -it deeprel bash -c "cd tensorflow/src && python DeepREL.py"
else
    echo "$libname not supported"
fi