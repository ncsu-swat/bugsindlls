#!/bin/bash

libraries=(jax pytorch tensorflow)
declare -A results

for library in "${libraries[@]}"; do
    cd "${library}"
    parentdirectory=$(pwd)

    passcount=0
    failcount=0
    breakcount=0

    for subdir in "$parentdirectory"/*; do
        if [ -d "$subdir" ]; then

            cd "$subdir"

            ./reproduce_bug.sh

            if [ $? -eq 0 ]; then
                echo "Bug reproduction successful"
                passcount=$((passcount+1))
            elif [ $? -eq 2 ]; then
                echo "Assumption Broken"
                breakcount=$((breakcount+1))
            else
                echo "Bug reproduction failed"
                failcount=$((failcount+1))
            fi

            cd ..
        fi
    done

    totalcount=$((passcount+failcount+breakcount))

    results["${library}"]=$(printf "%-10s | %-10s | %-10s | %-10s | %-10s" "${library}" "${totalcount}" "${passcount}" "${failcount}" "${breakcount}")
    cd ..
done

printf "\n---- %s ----\n" "Summary"
printf "%-10s | %-10s | %-10s | %-10s | %-10s\n" "Library" "Total" "Pass" "Fail" "Broken"
for library in "${libraries[@]}"; do
    echo "${results["${library}"]}"
done