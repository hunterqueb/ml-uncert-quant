#!/bin/bash

# if pdf is specified, the script will pass --pdf flag to scripts.

if [ "$1" == "pdf" ]; then
    echo "PDF flag is set. Passing --pdf to scripts."
    pdf_flag="--pdf"
else
    echo "PDF flag is not set. Running without --pdf."
    pdf_flag=""
fi

# 2bp elliptical 
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 70 --propMin 1750 --batch 8 $pdf_flag --orbit heo --n 3000
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 70 --propMin 1750 $pdf_flag --orbit heo --n 3000

python scripts/plotReachComparison.py \
    --mamba data/results/2bp_mamba_orbit_heo_prop1750min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_70.npz \
    --lstm data/results/2bp_lstm_orbit_heo_prop1750min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_70.npz \
    --pdf

# move all pdf files to a separate directory + timestamp of execution
directory=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p plots/DDDAS/$directory
mv plots/*.pdf plots/DDDAS/$directory