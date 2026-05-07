#!/bin/bash

if [ "$1" == "pdf" ]; then
    echo "PDF flag is set. Passing --pdf to scripts."
    pdf_flag="--pdf"
else
    echo "PDF flag is not set. Running without --pdf."
    pdf_flag=""
fi


# 80-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 80 --lookback 10 --jetson $pdf_flag 
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 80 --lookback 10 --model lstm --jetson $pdf_flag


# 100-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 100 --lookback 10 --jetson --batch 8 $pdf_flag
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 100 --lookback 10 --model lstm --jetson $pdf_flag

# 150-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 150 --lookback 10 --jetson --batch 8 $pdf_flag
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 150 --lookback 10 --model lstm --jetson $pdf_flag

# 200-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 200 --lookback 10 --jetson --batch 8 $pdf_flag
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 200 --lookback 10 --model lstm --jetson $pdf_flag

