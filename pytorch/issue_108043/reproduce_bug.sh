if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Broken assumption: Script was tested only on Linux."
    exit 2
fi

# Minimum required nvidia driver version:
reqmajor=525
reqminor=60
reqpatch=13

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

major=$(echo "$driver_version" | cut -d. -f1)
minor=$(echo "$driver_version" | cut -d. -f2)
patch=$(echo "$driver_version" | cut -d. -f3)

nvidiadrivermsg="Broken assumption: Script requires an nvidia driver with a version >=${reqmajor}.${reqminor}.${reqpatch}"
re='^[0-9]+$'

if ! [[ "$major" =~ $re ]]
then
    echo ${nvidiadrivermsg}
    exit 2
elif [ "$major" -lt "$reqmajor" ]
then
    echo ${nvidiadrivermsg}
    exit 2
elif [ "$major" -eq "$reqmajor" ]
then
    if [ "$minor" -lt "$reqminor" ]
    then
        echo ${nvidiadrivermsg}
        exit 2
    elif [ "$minor" -eq "$reqminor" ]
    then
        if [ "$patch" -lt "$reqpatch" ]
        then
            echo ${nvidiadrivermsg}
            exit 2
        fi
    fi
fi

docker build -t issue_108043 .
docker run -it --rm --gpus all issue_108043
exit $?