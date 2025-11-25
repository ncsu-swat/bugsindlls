#!/bin/bash

# Minimum required nvidia driver version:
reqmajor=450
reqminor=80
reqpatch=02

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

major=$(echo "$driver_version" | cut -d. -f1)
minor=$(echo "$driver_version" | cut -d. -f2)
patch=$(echo "$driver_version" | cut -d. -f3)

nvidiadrivermsg="Broken assumption: Script requires an nvidia driver with a version >=${reqmajor}.${reqminor}.${reqpatch}"
re='^[0-9]+$'

if ! [[ "$major" =~ $re ]]
then
    # Does not have an Nvidia gpu/driver
    conda init
    conda create --name issue_58273 python==3.8 pip -y
    eval "$(conda shell.bash hook)"
    conda activate issue_58273
    pip install -r requirements.txt
    pytest -sx
    returncode=$?
    conda deactivate
    conda env remove --name issue_58273 -y
    rm custom_model.keras
    exit ${returncode}
elif [ "$major" -lt "$reqmajor" ]
then
    echo ${nvidiadrivermsg}
    exit 2
elif [ "$major" -eq "$reqmajor" ]
then
    if [ "$minor" -lt "$reqminor" ]
    then
        echo ${nvidiadrivermsg}
        exit 2
    elif [ "$minor" -eq "$reqminor" ]
    then
        if [ "$patch" -lt "$reqpatch" ]
        then
            echo ${nvidiadrivermsg}
            exit 2
        fi
    fi
fi

docker build -t issue_58273 .
docker run -it --rm --gpus all issue_58273
exit $?