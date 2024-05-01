#!/bin/bash

conda init
conda create --name issue_62072 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_62072
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_62072 -y
exit ${returncode}
