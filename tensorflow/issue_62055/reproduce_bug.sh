#!/bin/bash

conda init
conda create --name issue_62055 python=3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_62055
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_62055 -y
exit ${returncode}
