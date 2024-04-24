conda init
conda create --name issue_121583 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_121583
pip install -r requirements.txt
pip install --pre torch==2.3.0.dev20240309 --index-url https://download.pytorch.org/whl/nightly/cpu
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_121583 -y
exit ${returncode}
