#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Bug reproducible only on Linux."
    exit 2
fi

conda init
conda create --name issue_120803 python==3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_120803
curl -OL https://download.pytorch.org/whl/nightly/cpu/torch-2.3.0.dev20240220%2Bcpu-cp38-cp38-linux_x86_64.whl
pip install torch-2.3.0.dev20240220%2Bcpu-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_120803 -y
rm torch-2.3.0.dev20240220%2Bcpu-cp38-cp38-linux_x86_64.whl
exit ${returncode}