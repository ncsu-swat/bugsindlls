#!/bin/bash

conda init
conda create --name issue_61761 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_61761
pip install -r requirements.txt
pip install ml_dtypes==0.2.0
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_61761 -y
exit ${returncode}
