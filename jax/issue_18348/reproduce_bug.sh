conda init
conda create --name issue_18348 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_18348
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_18348 -y
exit ${returncode}
