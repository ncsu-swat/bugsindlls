#!/bin/bash

conda init
conda create --name issue_93903 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_93903
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_93903 -y
exit ${returncode}