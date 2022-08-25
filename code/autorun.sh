#!/bin/bash
####################
# ${1}: gresidx
# ${2}: gresnum
# ${3}: config_name
# ${4}: repeats
# ${5}: dir_name
####################
while true
do
    timestamp=$(date +%s)
    if [[ $((${timestamp}%${2})) == ${1} ]]; then
        cd '..'
        value=$(python automl.py --allocate --config_name ${3} --repeats ${4} --dir_name ${5})
        if [[ ${value} == 'Finished' ]]; then
            break
        fi
        cd 'shells'
        echo "${value} is running"
        chmod +x "${value}"
        sh "${value}"
    fi
done
