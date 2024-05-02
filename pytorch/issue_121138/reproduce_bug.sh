#!/bin/bash

conda init
conda create --name issue_121138 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_121138
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
rm testingmemleak.h5
conda deactivate
conda env remove --name issue_121138 -y
exit ${returncode}