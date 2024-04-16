#!/bin/bash
conda init
conda create --name issue_61974 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_61974
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_61974 -y
exit ${returncode}