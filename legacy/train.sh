#!/bin/bash

set -o xtrace

source .env/bin/activate

# tucson
MODEL=saved_model/CNN_action
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

# Cannot load parameter for some reason
MODEL=model/20171210_154650
#./xvfb-run-safe -s "-screen 1 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

MODEL=model/20171212_181553
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--batchSize=64 \
  #--batchPerFeedback=16 \
  #--updateInterval=64 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

MODEL=model/20171212_210533
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--batchSize=128 \
  #--batchPerFeedback=16 \
  #--updateInterval=256 \
  #--save_period=500 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

#tucson
MODEL=model/20171212_182429
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

MODEL=model/20171212_182700
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--lr=1e-3 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

MODEL=model/20171212_182911
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--lr=1e-2 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

# tucson
MODEL=model/20171213_021909
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--partial_reward \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

# tucson
MODEL=model/20171214_160447
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--dist_reward_only \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

# tucson
MODEL=model/20171213_164419
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=manual --train \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

# tucson
MODEL=model/20171213_165416
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=manual --train \
  #--fix_exprate \
  #--explorationProb=0.3 \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

#tucson
MODEL=model/20171214_161615
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--explorationProb=0.3 \
  #--maxGameIter=3000 \
  #--load --model_dir $MODEL/
#python tools/plot_scores.py --score_log_path $MODEL/score_log 
git add -f $MODEL/score_log

set +o xtrace
