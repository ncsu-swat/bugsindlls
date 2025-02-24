#!/bin/bash

conda init
conda create --name issue_48609 python=3.7.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_48609
pip install gdown==4.7.3
gdown --fuzzy https://drive.google.com/file/d/1Ki2VVIpN7B9BDxse-elqYC8QnAQ9t2PJ/view?usp=sharing -O /tmp/
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_48609 -y
exit ${returncode}
