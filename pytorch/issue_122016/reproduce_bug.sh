if [[ "$(uname)" != "Darwin" ]]; then
    echo "This is not macOS. Exiting..."
    exit 2
fi

conda init
conda create --name issue_122016 python==3.12.2 pip -y
eval "$(conda shell.bash hook)"
conda activate issue_122016
pip install -r requirements.txt
pytest -sx
returncode=$?
conda deactivate
conda env remove --name issue_122016 -y
exit ${returncode}
