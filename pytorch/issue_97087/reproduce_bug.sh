#!/bin/bash

conda init
<<<<<<<< HEAD:pytorch/issue_94669/reproduce_bug.sh
conda create --name issue_94669 python==3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_94669
========
conda create --name issue_97087 python==3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_97087
>>>>>>>> origin/main:pytorch/issue_97087/reproduce_bug.sh
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
<<<<<<<< HEAD:pytorch/issue_94669/reproduce_bug.sh
conda env remove --name issue_94669 -y
========
conda env remove --name issue_97087 -y
>>>>>>>> origin/main:pytorch/issue_97087/reproduce_bug.sh
exit ${returncode}