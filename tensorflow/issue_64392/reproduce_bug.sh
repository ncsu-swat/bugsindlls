#!/bin/bash

conda init
conda create --name issue_64392 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_64392
pip install -r requirements.txt
pip install keras==3.1.1
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_64392 -y
exit ${returncode}
