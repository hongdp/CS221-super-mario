#!/usr/bin/env bash
# Few-shot evaluation (goal-B metric, E30 protocol): fine-tune a pretrained policy on each held-out level for 100k steps
# (checkpoints every 25k) and evaluate it on that level (10 stochastic episodes per checkpoint, + greedy at the end).
# usage: fewshot.sh OUTDIR "PRETRAINED_GLOB_WITH_SEED" "TRAIN_ARGS" [seeds]
#   e.g. fewshot.sh experiments/E37_x "experiments/E36-*-seedSEED_*/latest.pt" "--obs tiles2 --ego-back 2 --ego-ahead 8"
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
D=$1; PRE=$2; ARGS=$3; SEEDS=${4:-"1 2 3"}
mkdir -p $D/runs $D/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a experiments/queue.log; }
log "fewshot start $(basename $D)"
for s in $SEEDS; do
  pre=$(ls ${PRE//SEED/$s} | head -1)
  for L in 2-1 3-3 4-2 5-1 6-2 6-4 7-1; do
    name=s${s}_${L}_pretrained; out=$D/runs/$name
    [ -f $out/eval_final_greedy.json ] && continue
    mario-rl train --out-dir $D/runs --run-name $name $ARGS --levels $L --total-steps 100000 --num-envs 8 \
      --num-steps 256 --eval-interval 0 --save-interval 25000 --torch-threads 4 --seed $s --init-from $pre \
      > $D/logs/$name.train.log 2>&1
    for c in $out/ckpt_*.pt $out/latest.pt; do
      mario-rl eval $c --levels $L --episodes 10 --num-envs 10 --output ${c%.pt}.eval.json >> $D/logs/$name.eval.log 2>&1
    done
    mario-rl eval $out/latest.pt --levels $L --episodes 10 --num-envs 10 --greedy --output $out/eval_final_greedy.json >> $D/logs/$name.eval.log 2>&1
    rm -f $out/ckpt_*.pt
  done
done
log "fewshot done $(basename $D)"
