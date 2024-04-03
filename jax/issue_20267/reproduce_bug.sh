conda init
conda create --name issue_20267 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_20267
pip install jax==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpy==1.25.2
pip install -U pytest
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_20267 -y
exit ${returncode}
