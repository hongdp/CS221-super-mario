#!/usr/bin/env bash
# E30 driver: 3 seeds x 7 held-out levels x {pretrained, scratch}, 100k steps each, then evaluate checkpoints.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
D=experiments/E30-goalB-fewshot-finetune_20261008_003814
log() { echo "[$(date '+%F %T')] $*" | tee -a experiments/queue.log; }
until [ $(grep -c "evals done E31.*" experiments/queue.log) -ge 2 ]; do sleep 30; done
log "E30 start"
for s in 1 2 3; do
  pre=$(ls -d experiments/E28-goalB-ego2x8-fixedenv-10m-seed${s}_*)/latest.pt
  for L in 2-1 3-3 4-2 5-1 6-2 6-4 7-1; do
    for arm in pretrained scratch; do
      name=s${s}_${L}_${arm}; out=$D/runs/$name
      [ -f $out/eval_final_greedy.json ] && continue
      init=""; [ $arm = pretrained ] && init="--init-from $pre"
      mario-rl train --out-dir $D/runs --run-name $name --obs tiles2 --ego-back 2 --ego-ahead 8 --levels $L \
        --total-steps 100000 --num-envs 8 --num-steps 256 --eval-interval 0 --save-interval 25000 --torch-threads 4 \
        --seed $s $init > $D/logs/$name.train.log 2>&1
      for c in $out/ckpt_*.pt $out/latest.pt; do
        mario-rl eval $c --levels $L --episodes 10 --num-envs 10 --output ${c%.pt}.eval.json >> $D/logs/$name.eval.log 2>&1
      done
      mario-rl eval $out/latest.pt --levels $L --episodes 10 --num-envs 10 --greedy --output $out/eval_final_greedy.json >> $D/logs/$name.eval.log 2>&1
      rm -f $out/ckpt_*.pt   # keep latest.pt only (42 runs x 4 checkpoints x 11 MB otherwise)
    done
  done
  log "E30 seed $s done"
done
log "E30 done"
