#!/bin/bash
conda init
conda create --name issue_18004 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_18004
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_18004 -y
exit ${returncode}
