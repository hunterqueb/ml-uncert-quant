#!/bin/bash

# if pdf is specified, the script will pass --pdf flag to scripts and move the
# resulting pdfs into a timestamped directory. Otherwise the comparison script also
# renders the PDF evolution animations and outputs are left in plots/.

if [ "$1" == "pdf" ]; then
    echo "PDF flag is set. Passing --pdf to scripts."
    pdf_flag="--pdf"
    evolution_flag=""
else
    echo "PDF flag is not set. Running without --pdf and rendering PDF evolution animations."
    pdf_flag=""
    evolution_flag="--pdf-evolution"
fi

# 2bp
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 80 --propMin 450 --batch 8 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 80 --propMin 450 $pdf_flag

# 2bp elliptical 
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 70 --propMin 1750 --batch 8 $pdf_flag --orbit heo --n 3000
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 70 --propMin 1750 $pdf_flag --orbit heo --n 3000

python scripts/plotReachComparison.py \
    --mamba data/results/2bp_mamba_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npz \
    --lstm data/results/2bp_lstm_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npz \
    $pdf_flag $evolution_flag

python scripts/plotReachComparison.py \
    --mamba data/results/2bp_mamba_orbit_heo_prop1750min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_70.npz \
    --lstm data/results/2bp_lstm_orbit_heo_prop1750min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_70.npz \
    $pdf_flag $evolution_flag

# move all pdf files to a separate directory + timestamp of execution
if [ "$1" == "pdf" ]; then
    directory=$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p plots/DDDAS/$directory
    mv plots/*.pdf plots/DDDAS/$directory
fi