# Results

[`EXPERIMENTS.md`](EXPERIMENTS.md) is the chronological log of all experiments (E1-E10).

| folder | experiment |
|---|---|
| `tiles-uniform-8m/` | E6: main run, `--obs tiles`, 8M steps |
| `tiles-sticky-3m/` | E8: main config + sticky actions (p=0.25), 3M steps |
| `tiles2-uniform-3m/` | E9: `--obs tiles2`, 3M steps (weights in `model.pt`) |
| `pixels-uniform-3m/` | E10: `--obs pixels`, 3M steps (weights not committed) |
| `comparisons/` | overlays of several runs (`mario-rl compare`) |
| `diagnostics/old-shared-encoder-1m/` | E5: first multi-level run with a shared actor/critic trunk (policy collapse) |
| `diagnostics/level-1-1-ab/` | E4: 200k-step A/B tests on level 1-1 (γ, reward scale, decoupling) |

## `tiles-uniform-8m`

`--obs tiles` (the default at the time), trained for 8M env steps on the 22 training levels.

```bash
mario-rl train --run-name tiles-uniform-8m --obs tiles --total-steps 8000000 --num-envs 8 --num-steps 256 \
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

## `tiles-sticky-3m`

```bash
mario-rl train --run-name tiles-sticky-3m --obs tiles --total-steps 3000000 --num-envs 8 --num-steps 256 \
  --sticky-prob 0.25 --eval-interval 500000 --eval-episodes 2 --save-interval 1000000 --seed 1
```

`checkpoint_evals/` holds the 10-episodes-per-level evaluations at 1M, 2M and 3M steps;
`eval.jsonl` and `config.json` are as above.

## `tiles2-uniform-3m`

```bash
mario-rl train --run-name tiles2-uniform-3m --obs tiles2 --total-steps 3000000 --num-envs 8 \
  --num-steps 256 --eval-interval 500000 --eval-episodes 2 --save-interval 1000000 --seed 1
mario-rl compare comparisons/obs_3m.png "tiles (main run)=tiles-uniform-8m/checkpoint_evals" \
  "tiles + sticky 0.25=tiles-sticky-3m/checkpoint_evals" "tiles2=tiles2-uniform-3m/checkpoint_evals" \
  "pixels=pixels-uniform-3m/checkpoint_evals" --max-step 3000000
```

The GIFs are the seed-0 episodes, not selected: 1-1 reaches 84% (the level is completed in 6 of 10
evaluation episodes), and the held-out castle 6-4 reaches 24% (mean 41%).

## `pixels-uniform-3m`

```bash
mario-rl train --run-name pixels-uniform-3m --obs pixels --total-steps 3000000 --num-envs 8 \
  --num-steps 256 --eval-interval 500000 --eval-episodes 2 --save-interval 1000000 --seed 1
```

`checkpoint_evals/` holds the 10-episodes-per-level evaluations at 1M, 2M and 3M steps. The weights
(~13 MB) are not committed.
