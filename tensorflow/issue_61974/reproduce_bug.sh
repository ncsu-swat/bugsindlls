conda init
conda create --name issue_61974 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_61974
pip3 install --no-cache-dir tensorflow==2.13.0
pip install -U pytest
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_61974 -y
exit ${returncode}