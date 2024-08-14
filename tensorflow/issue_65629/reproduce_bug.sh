#!/bin/bash
conda init
conda create --name issue_65629 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_65629
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_65629 -y
exit ${returncode}
