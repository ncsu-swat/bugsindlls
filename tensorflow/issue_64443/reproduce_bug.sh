#!/bin/bash
conda init
conda create --name issue_64443 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_64443
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_64443 -y
rm -r my_model/
exit ${returncode}