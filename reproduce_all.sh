#!/bin/bash

libraries=(jax pytorch tensorflow)
NOW=$(date +"%m-%d-%Y-%T")
LOG_DIR=$(mktemp -d -q /tmp/dnnbugs.$NOW.XXX || exit 1)

###
# reproducing bugs
###
for library in "${libraries[@]}"; do

    (cd "${library}";
     
        parentdirectory=$(pwd)
        echo "---------- Starting $library ----------" 

        for subdir in "$parentdirectory"/*; do
            if [ -d "$subdir" ]; then

                (cd "$subdir";

                    bug_id=$(basename $subdir)                    
                    log_dir_library=${LOG_DIR}/$library
                    mkdir -p $log_dir_library # create directory
                    LOG_FILE=$(mktemp -q ${log_dir_library}/${bug_id}.XXX || exit 1)

                    echo "Start bug id $bug_id. Log: ${LOG_FILE}"
                    # avoid verbose output on screen (make it optional later)
                    ./reproduce_bug.sh >> $LOG_FILE 2>&1
                    
                    if [ $? -eq 0 ]; then
                        msg="Bug reproduction successful" 
                    elif [ $? -eq 2 ]; then
                        msg="Bug reproduction unsat prereqs" 
                    else
                        msg="Bug reproduction failed" 
                    fi
                    echo "End bug id $bug_id: $msg" | tee -a $LOG_FILE
                ) 
                # leaving bug id
            fi
        done
    )
    # leaving library
done

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