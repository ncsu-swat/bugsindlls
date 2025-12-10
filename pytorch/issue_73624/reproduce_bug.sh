#!/bin/bash
set -euo pipefail

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

# Minimum NVIDIA driver version
reqmajor=525
reqminor=60
reqpatch=13

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
major=$(echo "$driver_version" | cut -d. -f1)
minor=$(echo "$driver_version" | cut -d. -f2)
patch=$(echo "$driver_version" | cut -d. -f3)

nvidiadrivermsg="Broken assumption: Script requires an NVIDIA driver >= ${reqmajor}.${reqminor}.${reqpatch}"

if ! [[ "$major" =~ ^[0-9]+$ ]] || (( major < reqmajor )) || (( major == reqmajor && minor < reqminor )) || (( major == reqmajor && minor == reqminor && patch < reqpatch )); then
    echo "$nvidiadrivermsg"
    exit 2
fi

docker build -t issue_73624 .
docker run -it --rm --gpus all issue_73624
exit $?
