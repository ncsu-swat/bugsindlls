#!/bin/bash

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Bug reproducible only on Linux."
    exit 2
fi

conda init
conda create --name issue_120803 python==3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_120803
pip install gdown==5.2.0
gdown --fuzzy https://drive.google.com/file/d/1ombH9bjC3S9U6UybuvmfCZoJJPHx-U_K/view?usp=sharing -O /tmp/
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_120803 -y
exit ${returncode}