#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

conda init
conda create --name issue_58352 python=3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_58352
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_58352 -y
exit ${returncode}
