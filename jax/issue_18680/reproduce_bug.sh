conda init
conda create --name issue_18680 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_18680
pip install jax==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -U pytest
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_18680 -y
exit ${returncode}
