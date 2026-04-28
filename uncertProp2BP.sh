#!/bin/bash

# # 30 min dataset
# 10 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 10 --pdf 
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 10 --pdf

# 20 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 20  --pdf  
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 20  --pdf

# 90 min dataset
# 10 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 10 --propMin 90 --pdf
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 10 --propMin 90 --pdf  

# 20 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 20 --propMin 90 --pdf
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 20 --propMin 90 --pdf 

# 30 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 30 --propMin 90 --batch 16 --pdf
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 30 --propMin 90 --pdf

# 45 min train time
python scripts/reachability2BP.py --train-ratio 0.8 --train-timesteps 45 --propMin 90 --batch 8 --pdf
python scripts/reachability2BP.py --train-ratio 0.8 --model lstm --train-timesteps 45 --propMin 90 --pdf
