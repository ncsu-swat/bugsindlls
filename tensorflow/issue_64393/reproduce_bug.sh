#!/bin/bash

conda init
conda create --name issue_64393 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_64393
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_64393 -y
rm *.keras
exit ${returncode}
