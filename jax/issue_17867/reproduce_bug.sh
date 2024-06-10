if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Broken assumption: Script was tested only on MacOS."
    exit 2
fi

conda init
conda create --name issue_17867 python==3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_17867
pip install -r requirements.txt
git clone git@github.com:google/jax.git
cd jax
git checkout d477b92
cd ..
pytest test_issue_17867.py
returncode=$?
conda deactivate
conda env remove --name issue_17867 -y
rm -rf jax
exit ${returncode}
