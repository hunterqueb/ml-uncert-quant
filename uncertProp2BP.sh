#!/bin/bash

# # 30 min dataset
# # 10 min train time
# python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 10
# python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 10  

# # 20 min train time
# python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 20
# python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 20 

# 90 min dataset
# 10 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 10 --propMin 90
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 10 --propMin 90  

# 20 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 20 --propMin 90
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 20 --propMin 90 

# 30 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 30 --propMin 90 --batch 16
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 30 --propMin 90

# 45 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 45 --propMin 90 --batch 8
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 45 --propMin 90