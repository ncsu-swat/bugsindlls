#!/bin/bash

conda init
conda create --name issue_93733 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_93733
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_93733 -y
exit ${returncode}