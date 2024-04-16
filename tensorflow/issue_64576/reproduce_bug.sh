#!/bin/bash
conda init
conda create --name issue_64576 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_64576
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_64576 -y
rm custom_model.keras
exit ${returncode}