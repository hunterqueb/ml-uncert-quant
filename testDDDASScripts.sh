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
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 10 --propMin 450 --batch 8 $pdf_flag --jetson

# 3bp -- still need to find appropriate parameters
# 200-2hr train time
python scripts/reachability3BP.py --train-ratio 0.1 --train-timesteps 20 --lookback 10 --batch 8 $pdf_flag --jetson
