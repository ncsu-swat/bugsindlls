conda init
conda create --name issue_19076 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_19076
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_19076 -y
exit ${returncode}
