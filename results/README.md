# Results

## `tiles-uniform-8m`

The default configuration, trained for 8M env steps on the 22 training levels.

```bash
mario-rl train --run-name tiles-uniform-8m --total-steps 8000000 --num-envs 8 --num-steps 256 \
  --eval-interval 500000 --eval-episodes 2 --save-interval 1000000 --seed 1
# re-evaluate every checkpoint with 10 episodes per level
for c in runs/tiles-uniform-8m/ckpt_*.pt runs/tiles-uniform-8m/latest.pt; do
  mario-rl eval $c --levels test train --episodes 10 --num-envs 8 --output <dir>/$(basename $c .pt).json
done
mario-rl plot <dir> --evals                    # generalization.png
mario-rl plot runs/tiles-uniform-8m            # training_curves.png
mario-rl play runs/tiles-uniform-8m/latest.pt --level 4-1 --seed 0 --output play_4-1_train.gif
```

| file | content |
|---|---|
| `model.pt` | final weights + configs (`mario-rl play/eval results/tiles-uniform-8m/model.pt ...` works) |
| `eval_final.json` | per-level results of the final model, 10 episodes per level |
| `checkpoint_evals/` | the same evaluation for every 1M-step checkpoint (`ckpt_0.json` is the untrained policy) |
| `eval.jsonl` | in-training evaluations (2 episodes per level, noisier) |
| `metrics.jsonl.gz` | per-update training metrics (losses, entropy, KL, SPS, episode stats) |
| `config.json` | the full `PPOConfig` / `EnvConfig` |
| `*.png`, `*.gif` | figures and recordings referenced from the main README |
