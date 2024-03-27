conda init
conda create --name issue_122126 python==3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122126
pip3 install --pre torch==2.4.0.dev20240317 --index-url https://download.pytorch.org/whl/nightly
pip3 install numpy
pip install -U pytest
python -m torch.utils.collect_env
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122126 -y
exit ${returncode}