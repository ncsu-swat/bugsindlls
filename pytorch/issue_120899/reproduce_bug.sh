#!/bin/bash

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Broken assumption: The bug can only be reproduced on MacOS"
    exit 2
fi

conda init
conda create --name issue_120899 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_120899
pip install -r requirements.txt
brew install libomp
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_120899 -y
exit ${returncode}

