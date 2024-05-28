if [[ "$(uname)" != "Darwin" ]]; then
    echo "This is not macOS. Exiting..."
    exit 2
fi

conda init
conda create --name issue_122705 python=3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122705
pip install -r requirements.txt
pip install --pre torch==2.4.0.dev20240401 --index-url https://download.pytorch.org/whl/nightly/cpu
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122705 -y
exit ${returncode}
