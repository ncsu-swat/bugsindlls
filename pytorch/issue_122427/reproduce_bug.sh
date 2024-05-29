#!/bin/bash

if [[ $OSTYPE != 'darwin'* ]]; then
    echo "Broken assumption: Bug reproducible only on MacOS."
    exit 2
fi

conda init
conda create --name issue_122427 python=3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122427
curl -OL https://download.pytorch.org/whl/nightly/cpu/torch-2.4.0.dev20240321-cp311-none-macosx_11_0_arm64.whl
pip install torch-2.4.0.dev20240321-cp311-none-macosx_11_0_arm64.whl
pip install -r requirements.txt
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122427 -y
rm torch-2.4.0.dev20240321-cp311-none-macosx_11_0_arm64.whl
exit ${returncode}