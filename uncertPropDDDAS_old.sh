#!/bin/bash

# if pdf is specified, the script will pass --pdf flag to scripts.

if [ "$1" == "pdf" ]; then
    echo "PDF flag is set. Passing --pdf to scripts."
    pdf_flag="--pdf"
else
    echo "PDF flag is not set. Running without --pdf."
    pdf_flag=""
fi

# 2bp
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 80 --propMin 450 --batch 8 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 80 --propMin 450 $pdf_flag

# 3bp -- still need to find appropriate parameters
# 200-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 100 --lookback 10 --batch 8 $pdf_flag
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 100 --lookback 10 --model lstm $pdf_flag

python scripts/plotReachComparison.py \
    --mamba data/results/2bp_mamba_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npz \
    --lstm data/results/2bp_lstm_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npz \
    --pdf

python scripts/plotReachComparison.py \
    --mamba data/results/3bp_mamba_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_100.npz \
    --lstm data/results/3bp_lstm_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_100.npz \
    --pdf

# move all pdf files to a separate directory + timestamp of execution
directory=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p plots/DDDAS/$directory
mv plots/*.pdf plots/DDDAS/$directory