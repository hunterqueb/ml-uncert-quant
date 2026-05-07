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
python scripts/reachability2BP.py --train-ratio 0.1 --train-timesteps 90 --propMin 450 --batch 8 $pdf_flag
python scripts/reachability2BP.py --train-ratio 0.1 --model lstm --train-timesteps 80 --propMin 450 $pdf_flag

# 3bp -- still need to find appropriate parameters