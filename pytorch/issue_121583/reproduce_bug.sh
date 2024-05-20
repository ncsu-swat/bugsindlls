#!/bin/bash

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Broken assumption: The bug can only be reproduced on MacOS"
    exit 2
fi

conda init
conda create --name issue_121583 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_121583
pip install -r requirements.txt
curl -OL https://download.pytorch.org/whl/nightly/cpu/torch-2.3.0.dev20240309-cp310-none-macosx_11_0_arm64.whl
pip install torch-2.3.0.dev20240309-cp310-none-macosx_11_0_arm64.whl
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_121583 -y
rm torch-2.3.0.dev20240309-cp310-none-macosx_11_0_arm64.whl
exit ${returncode}
