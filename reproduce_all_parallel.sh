#!/bin/bash

libraries=(jax pytorch tensorflow)
NOW=$(date +"%m-%d-%Y-%T")
LOG_DIR=$(mktemp -d -q /tmp/dnnbugs.$NOW.XXX || exit 1)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

function run() {
    dir=$1
    bug_id=$2
    log_dir_library=$3

    cd $dir

    mkdir -p $log_dir_library # create directory
    
    LOG_FILE=$(mktemp -q ${log_dir_library}/${bug_id}.XXX || exit 1)

    echo "Start bug id $bug_id. Log: ${LOG_FILE}"
    #avoid verbose output on screen (make it optional later)
    
    ./reproduce_bug.sh >> $LOG_FILE 2>&1
    
    returncode=$?
    if [ $returncode -eq 0 ]; then
        msg="Bug reproduction successful" 
    elif [ $returncode -eq 2 ]; then
        msg="Bug reproduction unsat prereqs" 
    else
        msg="Bug reproduction failed" 
    fi
    echo "End bug id $bug_id: $msg" | tee -a $LOG_FILE
}

###
# reproducing bugs
###
echo "Logs in $LOG_DIR"
tasks=$(mktemp)
for library in "${libraries[@]}"; do
    (cd "${SCRIPT_DIR}/${library}";
        parentdirectory=$(pwd)
        for subdir in "$parentdirectory"/*; do
            if [ -d "$subdir" ]; then
                (cd "$subdir";
                    bug_id=$(basename $subdir)                    
                    log_dir_library=${LOG_DIR}/$library
                    echo "$subdir $bug_id $log_dir_library" >> $tasks 2>&1
                )
                # leaving bug id
            fi
        done
    )
    # leaving library
done

# read tasks from file and run them in background (in parallel)
while read -r line; do
    run $line &
    array+=($!)
done < $tasks
# wait for them to finish
wait ${array[@]}

###
# reporting results
###
printf "\n---- %s ----\n" "Summary"
printf "%-10s | %-5s | %-5s | %-5s | %-10s\n" "Library" "Total" "Pass" "Fail" "Broken"
for library in "${libraries[@]}"; do
    passcount=`grep "Bug reproduction successful" ${LOG_DIR}/$library/issue* | wc -l | sed 's/ //g'`
    failcount=`grep "Bug reproduction failed" ${LOG_DIR}/$library/issue* | wc -l | sed 's/ //g'`
    breakcount=`grep "Bug reproduction unsat prereqs" ${LOG_DIR}/$library/issue* | wc -l | sed 's/ //g'`
    totalcount=$((passcount+failcount+breakcount))
    printf "%-10s | %-5s | %-5s | %-5s | %-5s\n" "${library}" "${totalcount}" "${passcount}" "${failcount}" "${breakcount}"
done
