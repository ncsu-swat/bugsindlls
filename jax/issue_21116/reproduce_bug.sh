#!/bin/bash

conda init
conda create --name issue_21116 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_21116
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_21116 -y
exit ${returncode}
