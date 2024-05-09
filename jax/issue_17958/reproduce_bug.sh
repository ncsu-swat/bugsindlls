#!/bin/bash
conda init
conda create --name issue_17958 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_17958
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_17958 -y
exit ${returncode}
