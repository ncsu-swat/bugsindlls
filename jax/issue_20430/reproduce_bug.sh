conda init
conda create --name issue_20430 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_20430
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_20430 -y
exit ${returncode}
