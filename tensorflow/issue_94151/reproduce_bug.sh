#!/bin/bash

conda init
conda create --name issue_94151 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_94151
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_94151 -y
exit ${returncode}