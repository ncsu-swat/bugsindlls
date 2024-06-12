conda init
conda create --name issue_17294 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_17294
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_17294 -y
exit ${returncode}
