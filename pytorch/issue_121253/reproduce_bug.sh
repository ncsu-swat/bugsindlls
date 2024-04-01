if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

docker build -t issue_121253 .
docker run -it --rm --gpus all issue_121253
exit $?