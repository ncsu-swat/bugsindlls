conda init
conda create --name issue_20620 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_20620
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_20620 -y
exit ${returncode}