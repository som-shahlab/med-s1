#!/bin/bash

# Get all jobs with names starting with med-s1
running_jobs=$(squeue -h -u $USER -o "%i %j" | grep "med-s1" | awk '{print $1}')

if [ -n "$running_jobs" ]; then
    echo "Cancelling med-s1 jobs:"
    echo "$running_jobs"
    for job in $running_jobs; do
        scancel $job
    done
    echo "All med-s1 jobs cancelled"
else
    echo "No med-s1 jobs running"
fi