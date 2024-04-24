#!/bin/bash

conda init
conda create --name issue_61163 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_61163
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_61163 -y
exit ${returncode}
