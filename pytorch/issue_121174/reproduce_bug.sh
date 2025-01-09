#!/bin/bash
conda init
conda create --name issue_121174 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_121174
pip install gdown==5.2.0
gdown --fuzzy https://drive.google.com/file/d/1mQ5N7AAJH5ismHhhoJyluyMRsXFOqqSu/view?usp=sharing -O /tmp/
pip install -r requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_121174 -y
exit ${returncode}