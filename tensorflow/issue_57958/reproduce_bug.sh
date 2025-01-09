#!/bin/bash

conda init
conda create --name issue_57958 python=3.7.6 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_57958
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_57958 -y
exit ${returncode}
