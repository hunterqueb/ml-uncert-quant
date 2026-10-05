#!/bin/bash

# timing-only training runs for mamba and lstm on the jetson
# nothing is saved except the appended train/test times in data/results/2bp_times_<model>_<orbit>.csv
# each config is rerun until its csv has N entries (default 20): N=100 ./jetsonTiming.sh
# extra args are passed through to every run, e.g. ./jetsonTiming.sh --float

N=${N:-20}
flags="--jetson --traj-chunk 250 --time --no-save --no-plots $@"

# number of data rows (minus header) in a csv, 0 if it doesn't exist
count() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

# usage: run_until <model> <orbit> <other reachability2BP.py args>
run_until() {
    local model=$1 orbit=$2; shift 2
    local file="data/results/2bp_times_${model}_${orbit}.csv"
    while [ "$(count "$file")" -lt "$N" ]; do
        echo "$file: $(count "$file")/$N entries, running $model $orbit"
        # stop instead of looping forever if a run crashes or gets OOM-killed
        python scripts/reachability2BP.py --model "$model" --orbit "$orbit" "$@" $flags || {
            echo "$model $orbit run failed (exit $?), skipping"; return 1; }
    done
    echo "$file: $(count "$file")/$N entries, done"
}

# 2bp leo
# run_until mamba leo --train-ratio 0.1 --train-timesteps 80 --propMin 450 --batch 8 --batch-test 16
# run_until lstm leo --train-ratio 0.1 --train-timesteps 80 --propMin 450

# 2bp elliptical
run_until mamba heo --train-ratio 0.1 --train-timesteps 70 --propMin 1750 --batch 8 --batch-test 16 --n 3000
# lstm batches match mamba sequences per pass: 8 windows x 300 train trajs = 2400, 16 windows x 250 traj-chunk = 4000
run_until lstm heo --train-ratio 0.1 --train-timesteps 70 --propMin 1750 --batch 2400 --batch-test 4000 --n 3000
