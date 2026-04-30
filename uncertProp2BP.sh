#!/bin/bash

# if pdf is specified, the script will pass --pdf flag to scripts.

if [ "$1" == "pdf" ]; then
    echo "PDF flag is set. Passing --pdf to scripts."
    pdf_flag="--pdf"
else
    echo "PDF flag is not set. Running without --pdf."
    pdf_flag=""
fi

# # 30 min dataset
# 10 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 10 $pdf_flag
# 20 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 20  $pdf_flag  
# 90 min dataset
# 10 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 10 --propMin 90 --batch 8 $pdf_flag --jetson
# 20 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 20 --propMin 90 --batch 8 $pdf_flag --jetson
# 30 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 30 --propMin 90 --batch 8 $pdf_flag --jetson
# 45 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 45 --propMin 90 --batch 8 $pdf_flag --jetson
# 80 min train time
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 80 --propMin 90 --batch 8 $pdf_flag --jetson


python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 10 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 20  $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 10 --propMin 90 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 20 --propMin 90 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 30 --propMin 90 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 45 --propMin 90 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 80 --propMin 90 $pdf_flag





