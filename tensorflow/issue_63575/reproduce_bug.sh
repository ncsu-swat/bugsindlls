conda init
conda create --name issue_63575 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_63575
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_63575 -y
exit ${returncode}