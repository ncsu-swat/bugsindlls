#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

conda init
conda create --name issue_48589 python=3.7.6 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_48589
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_48589 -y
exit ${returncode}
