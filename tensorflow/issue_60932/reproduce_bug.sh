#!/bin/bash

conda init
conda create --name issue_60932 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_60932
pip install -r requirements.txt
rm ~/.keras/datasets/auto-mpg.csv
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_60932 -y
exit ${returncode}
