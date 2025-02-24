#!/bin/bash

conda init
conda create --name issue_53660 python=3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_53660
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_53660 -y
exit ${returncode}
