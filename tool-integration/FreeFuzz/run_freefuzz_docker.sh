#!/bin/bash

libname=${1:-torch}

if [[ "$libname" == "torch" ]] || [[ "$libname" == "pytorch" ]]; then
    docker exec -it freefuzz bash -c "python3 preprocess.py torch && cd src && python3 FreeFuzz.py --conf demo_torch.conf && cd .. && python3 postprocess.py demo_torch.conf"
elif [[ "$libname" == "tensorflow" ]] || [[ "$libname" == "tf" ]]; then
    docker exec -it freefuzz bash -c "python3 preprocess.py tf && cd src && python3 FreeFuzz.py --conf demo_tf.conf && cd .. && python3 postprocess.py demo_tf.conf"
else
    echo "$libname not supported"
fi