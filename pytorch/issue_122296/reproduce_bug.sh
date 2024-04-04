conda init
conda create --name issue_122296 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122296
pip install -r requirements.txt
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122296 -y
exit ${returncode}