#!/bin/bash

conda init
conda create --name issue_20856 python==3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_20856
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_20856 -y
exit ${returncode}
