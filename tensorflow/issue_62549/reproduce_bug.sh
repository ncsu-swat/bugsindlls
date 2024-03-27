if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

docker build -t issue_62549 .
docker run -it --rm --gpus all issue_62549
exit $?