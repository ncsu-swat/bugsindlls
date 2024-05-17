#!/bin/bash

conda init
conda create --name issue_120762 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_120762
pip install -r  requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_120762 -y
exit ${returncode}
