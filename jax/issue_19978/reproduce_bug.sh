conda init
conda create --name issue_19978 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_19978
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_19978 -y
exit ${returncode}
