#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

conda init
conda create --name issue_tfsa2022090 python=3.8.0 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_tfsa2022090
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_tfsa2022090 -y
exit ${returncode}
