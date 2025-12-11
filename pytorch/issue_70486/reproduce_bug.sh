#!/bin/bash

conda init
conda create --name issue_70486 python==3.9.5 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_70486
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_70486 -y
exit ${returncode}
