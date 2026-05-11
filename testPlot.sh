#!/bin/bash

python scripts/plotReachComparison.py \
    --mamba data/results/2bp_mamba_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npy \
    --lstm data/results/2bp_lstm_orbit_leo_prop450min_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_80.npy \
    --pdf


python scripts/plotReachComparison.py \
    --mamba data/results/3bp_mamba_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_100.npy \
    --lstm data/results/3bp_lstm_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.1_epoch_10_lr_0.01_train_timesteps_100.npy \
    --pdf
