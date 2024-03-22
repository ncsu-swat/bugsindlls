conda create --name issue_18218 python==3.11 pip -y
source activate issue_18218
pip install jax==0.4.18 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.4.18 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -U pytest
pytest -sx
conda deactivate
conda env remove --name issue_18218