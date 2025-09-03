#!/bin/bash

conda init
conda create --name issue_57496 python=3.9 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_57496
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_57496 -y
exit ${returncode}
