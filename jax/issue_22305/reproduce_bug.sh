#!/bin/bash

conda init
conda create --name issue_22305 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_22305
pip install gdown==5.2.0
gdown --fuzzy https://drive.google.com/file/d/1KkQqFT8J2d82wabZT8efqFNX4Jud-RuA/view?usp=sharing -O /tmp/
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_22305 -y
exit ${returncode}
