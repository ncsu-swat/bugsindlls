conda init
conda create --name issue_19978 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_19978
pip install jax==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpy==1.24.3
pip install -U pytest
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_19978 -y
exit ${returncode}
