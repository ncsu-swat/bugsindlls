#!/bin/bash

conda init
conda create --name issue_120188 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_120188
pip install gdown==5.2.0
gdown --fuzzy https://drive.google.com/file/d/10x74XH7VSd5JGPWSPYTwwG9S0X4jwuXC/view?usp=sharing -O /tmp/
pip install -r  requirements.txt
if [[ $OSTYPE == 'darwin'* ]]
then
    brew install libomp
fi
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_120188 -y
exit ${returncode}
