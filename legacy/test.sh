#!/bin/bash

set -o xtrace

source .env/bin/activate

#tucson
MODEL=model/20171212_182429
./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn \
  --conv_model=1 \
  --maxGameIter=1 \
  --load --model_dir $MODEL/
set +o xtrace
