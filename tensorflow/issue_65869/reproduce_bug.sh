#!/bin/bash

conda init
conda create --name issue_65869 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_65869
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_65869 -y
exit ${returncode}