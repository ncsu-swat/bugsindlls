#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Bug reproducible only on Linux."
    exit 2
fi

conda init
conda create --name issue_113013 python==3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_113013
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_113013 -y
exit ${returncode}