conda init
conda create --name issue_122771 python=3.10 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122771
pip install gdown==5.2.0
gdown --fuzzy https://drive.google.com/file/d/1JNpkDBx28ln2mKaoCvISvLBvFWCjboHT/view?usp=sharing -O /tmp/
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122771 -y
exit ${returncode}
