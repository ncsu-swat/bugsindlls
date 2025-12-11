#!/bin/bash

conda init
conda create --name issue_71059 python==3.9.5 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_71059
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_71059 -y
exit ${returncode}