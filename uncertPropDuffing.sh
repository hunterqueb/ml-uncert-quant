#!/bin/bash

# run dataset generation for the Duffing system if files do not exist
if [ ! -f "data/test/duffing_monte_carlo_trajectories_sigma_0.2_dt_0.02_n_20000.npz" ]; then
    echo "Generating dataset for the Duffing system..."
    python scripts/datagen/duffing_gen.py
    echo "Dataset generation complete."
else
    echo "Dataset already exists. Skipping generation."
fi

# 1 second train time
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 50 --jetson
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 50 --model lstm --jetson 


# 4 second train time
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 200 --jetson --batch 32
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 200 --model lstm --jetson 


# 8 second train time
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 400 --jetson --batch 32
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 400 --model lstm --jetson 

# 12 second train time
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 600 --jetson --batch 32
python scripts/reachabilityDuffing.py --train-ratio 0.8 --train-timesteps 600 --model lstm --jetson
