#!/bin/bash

conda init
conda create --name issue_94594 python==3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_94594
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_94594 -y
exit ${returncode}
