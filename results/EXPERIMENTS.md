# Experiment log

This is a chronological record of every experiment run during the 2026 port, including the
diagnostics and dead ends behind the design decisions. Raw data lives next to this file:

- `tiles-uniform-8m/`: main run
- `tiles-sticky-3m/`: sticky actions
- `diagnostics/`: early runs and A/B tests

**Machine**: 4 vCPU Intel Xeon @ 2.8 GHz, 15 GB RAM, no GPU. **Software**: Python 3.11, PyTorch 2.14
(CPU), Gymnasium 1.3, stable-retro 1.0.1, NumPy 2.4.

**Evaluation protocol**: unless noted, a checkpoint is evaluated with 10 stochastic episodes on
each of the 22 training and 7 held-out levels (`mario-rl eval --episodes 10`). Each episode uses
0-30 random no-op frames at the start. *Progress* is the fraction of the level reached (1.0 at the
flag), and *completion* is the fraction of episodes that reach the flag or axe. Split numbers
average over levels. At 10 episodes per level, differences of about ±2-3 points in mean progress
are noise. The in-training evaluations (`eval.jsonl`, 2 episodes per level) are noisier.

---

## E1: Emulator backend throughput (2026-10-03)

| backend | raw frames/s, 1 core |
|---|---|
| nes-py 8.2.1 (LaiNES, built with `-O3`) | ~465 |
| stable-retro 1.0.1 (FCEUmm) | ~3100 |

`MarioEnv` in one process: 522 env steps/s (frame skip 4, about 2090 frames/s). The emulator takes
85% of the step time. nes-py also fails under NumPy 2 (uint8 overflow in its ROM parser) and
imports the unmaintained `gym` package.
**Decision**: use stable-retro's `RetroEmulator` directly, with our own RAM logic.

## E2: Parallel sampling architecture

| setup (tiles, random actions) | env steps/s |
|---|---|
| 4 independent processes (upper bound) | ~2150 (4 x 540) |
| Gymnasium `AsyncVectorEnv`, 4 envs, SAME_STEP autoreset | 841 |
| `AsyncVectorEnv`, 8 envs | 900 |
| `AsyncVectorEnv`, `shared_memory=False` | 934 |

Inside `AsyncVectorEnv` a worker's `env.step` takes ~1.7 ms, but a vector step takes 4.6 ms. A pipe
round trip is only 91 µs, so the loss comes from lock-step synchronization, not from data transfer.
stable-retro allows one emulator per process, so a worker cannot batch several envs.
**Decision**: each worker keeps a reference to the learner's shared-memory policy and collects a
whole rollout locally, sending one message per PPO iteration.

End-to-end PPO throughput (shared-trunk tile model): 4 workers 807 env steps/s (rollout phase
1355), 8 workers 887 (rollout 1344). With the decoupled actor/critic (E5) it is ~650; with tiles2
(E9) ~450.

## E3: Model cost on this CPU

| change | single-sample act (1 thread) | PPO update |
|---|---|---|
| tile CNN 32/64/64, 5.2M MACs/sample | 1.17 ms | 1.8 s per 2048 samples x 4 epochs |
| tile CNN 16/32/32, 1.8M MACs/sample | 0.63 ms (Gumbel-max sampling) | 0.94 s |
| TorchScript trace + freeze | no gain | - |
| tiles2 first version (26x32, 2.7M MACs) | - | 332 ms per 512-sample minibatch |
| + broadcast-compare one-hot, shared by actor and critic | - | 217 ms |
| + extra early downsampling (~2M MACs) | 1.25 ms (actor + critic) | 173 ms |
| pixels (Nature CNN) vs tiles, measured under load | 2.4x slower | 3x slower |

## E4: Level 1-1 A/B tests, 200k steps each

Folder: `diagnostics/level-1-1-ab/`. 8 workers x 256 steps, LR annealed to 0 over 200k steps.
"Progress" is the training-episode progress averaged over the last 5 updates.

| variant | progress | last update | best completion | final entropy |
|---|---|---|---|---|
| shared trunk, γ=0.99, reward scale 0.1 | 25.5% | 31.6% | 6.7% | 1.34 |
| shared trunk, γ=0.9 | 27.4% | 29.4% | 0.0% | 1.01 |
| shared trunk, reward scale 0.01 | 26.7% | 22.8% | 5.6% | 1.53 |
| decoupled actor/critic + reward normalization | 32.7% | 29.3% | 5.9% | 1.08 |

**Conclusion**: 200k steps is too short to separate the variants. All runs cleared 1-1 at most
occasionally. γ and reward scale were kept at 0.99 / 0.1.

## E5: First multi-level run with a shared trunk, stopped at 1.1M steps

Folder: `diagnostics/old-shared-encoder-1m/`. Run 2026-10-03 20:46-21:13 (paused twice for E4).

| step | train progress | held-out progress |
|---|---|---|
| 0.5M | 17.6% | 16.1% |
| 1.0M | 17.7% | 16.9% |

(in-training evaluation, 2 episodes per level)

The policy collapsed to a state-independent distribution. The 1M checkpoint chose right+A about
55% and right+B about 40% of the time, with entropy ≈ 1.0 on every level it was tested on. Gradient norms on a
1024-step batch were value loss 145, policy loss 1.1 and entropy 0.11 (value targets: mean 24,
std 13). The shared trunk was effectively trained only by the critic.
**Decision**: separate actor and critic encoders with per-network gradient clipping, plus reward
normalization by the running std of the discounted return (now the defaults).

At ~1M steps the decoupled main run (E6) reached 31.3% training-episode progress (old run: 19.7%)
and 28.9% in-training-eval train progress (old run: 17.7%). Held-out was unchanged (17.0% vs 16.9%).

## E6: Main run `tiles-uniform-8m`

Folder: `tiles-uniform-8m/`. Default config, 22 training levels, uniform sampling, 8M steps.
Run 2026-10-03 21:22 to 2026-10-04 00:49 (3.4 h, ~650 env steps/s).

| steps | train progress | train completion | held-out progress | held-out completion |
|---|---|---|---|---|
| untrained policy | 15.5% | 0.0% | 10.7% | 0.0% |
| 1M | 29.7% | 0.5% | 19.5% | 0.0% |
| 2M | 35.5% | 1.4% | 20.2% | 0.0% |
| 3M | 38.5% | 1.8% | 20.8% | 0.0% |
| 4M | 45.7% | 6.8% | 19.7% | 0.0% |
| 5M | 51.1% | 12.3% | 22.4% | 0.0% |
| 6M | 55.4% | 18.2% | 20.9% | 0.0% |
| 7M | 55.9% | 20.0% | 22.1% | 0.0% |
| 8M | 61.7% | 23.2% | 21.6% | 0.0% |

Completion at 8M: 4-1 10/10, 3-2 9/10, 1-1 8/10, 6-1 8/10, 2-3 6/10, 1-4 4/10, 7-3 3/10.
Weakest training levels: 4-3 24%, 5-3 27.5%, 1-2 29%, 1-3 30%. Held-out at 8M: 7-1 41%,
2-1 25%, 6-4 24%, 3-3 19%, 5-1 17.5%, 6-2 16%, 4-2 10%.
**Conclusion**: the training levels are learned steadily, but held-out progress gains almost all of
its improvement over the untrained policy in the first 1M steps and then stays flat. The policy
memorizes level-specific behaviour.

## E7: Failure analysis of the 8M model and observation audit

- **1-3**: all 4 sampled episodes end at x ≈ 771-783, the jump from the tall treetop onto a moving
  lift. 5-3 (same layout) ends at 381-783, 4-3 at 249-595, and 1-2 at 899-1560 (piranha plants
  and koopas). The weakest training levels are the lift-heavy athletic x-3 levels.
- **Lift representation in `tiles`**: the game's own collision boxes (`$04B0 + 4·slot`) show lift
  widths of 32 px (types 0x24, 0x25, 0x28) and 48 px (0x26, 0x27). `tiles` assumed 48 px for all.
  The boxes match the screen exactly (overlay checked on a 1-2 lift).
- **Other gaps in `tiles`**: all enemies share one class; firebars show only their pivot (segments
  have no collision box and exist only as OAM sprites, tile 0x64); hammers are misc objects (8x8
  boxes at `$04D0`) and are not shown at all; there is no water, power-up or velocity information.
- **Pixels instead?** They would add palette and background differences between worlds, which is a
  liability with only 22 levels, and cost about 2.5x the compute here. **Decision**: build `tiles2`
  first (E9), then run a pixel baseline (E10).

## E8: Sticky actions, `tiles-sticky-3m`

Folder: `tiles-sticky-3m/`. Main-run config plus `--sticky-prob 0.25`, 3M steps.
Run 2026-10-04 01:01-02:24.

| steps | train progress | train completion | held-out progress | (main run held-out) |
|---|---|---|---|---|
| 1M | 29.8% | 0.5% | 18.4% | 19.5% |
| 2M | 37.4% | 1.4% | 20.2% | 20.2% |
| 3M | 40.3% | 4.1% | 17.3% | 20.8% |

**Conclusion**: there is no held-out gain, and training levels are slightly better. The
memorization is not just open-loop button sequences.

## E9: `tiles2` observation, `tiles2-uniform-3m`

Folder: `tiles2-uniform-3m/`. Main-run config with `--obs tiles2` (8px grid built from collision
boxes, stompable/hazard classes, firebar segments, hammers and a Mario state vector), 3M steps.
Run 2026-10-04 02:34-04:28 (~450 env steps/s).

| steps | train progress | train completion | held-out progress | held-out completion |
|---|---|---|---|---|
| 1M | 34.9% (tiles: 29.7%) | 2.3% (0.5%) | 16.2% (19.5%) | 0.0% |
| 2M | 45.3% (35.5%) | 8.6% (1.4%) | 23.3% (20.2%) | 0.0% |
| 3M | **51.2%** (38.5%) | **15.5%** (1.8%) | **24.4%** (20.8%) | 0.0% |

(main-run `tiles` numbers at the same step in parentheses; overlay in `comparisons/obs_3m.png`)

Per level at 3M (tiles → tiles2):
- **Castles improve most**, consistent with firebars now being visible: 1-4 47→74%, 2-4 59→70%,
  5-4 58→69%, and the held-out castle **6-4 21→41%**.
- **Athletic (lift) levels improve**: 1-3 30→38%, 4-3 22→35%, 5-3 24→31%.
- **Completions** at 3M: 4-1 9/10, 3-2 8/10, 1-1 6/10, 2-3 5/10, 6-1 4/10.
- Held-out: 3-3 18→26%, 5-1 19→22%, 7-1 32→35%, 2-1 34→26%, 4-2 12→9%, 6-2 11→12%.

**Conclusion**: the richer observation makes the training levels much easier to learn. tiles2 at
3M roughly matches tiles at 6M, with 8x the completion rate at equal steps. It also raises held-out
progress above anything `tiles` reached in 8M steps (24.4% vs 21.6%), and the held-out curve is
still rising. The held-out gain is small next to the training gain, and no held-out level is
completed yet, so most of the generalization gap remains.

## E10: Pixel baseline, `pixels-uniform-3m` (running)

Folder: `pixels-uniform-3m/`. Main-run config with `--obs pixels` (84x84 grayscale, Nature CNN for
both actor and critic), 3M steps, started 2026-10-04 04:30. Throughput is ~185 env steps/s
(tiles2: ~450): the rollout phase runs at ~600 steps/s, and each PPO update takes ~8 s. The full run
will take ~4.6 h.

| steps | train progress | train completion | held-out progress | held-out completion |
|---|---|---|---|---|
| 1M | 29.3% (tiles 29.7%, tiles2 34.9%) | 0.9% | 14.2% (tiles 19.5%, tiles2 16.2%) | 0.0% |
| 2M | _pending_ | | | |
| 3M | _pending_ | | | |

Held-out per level at 1M: 2-1 20%, 3-3 13%, 4-2 6%, 5-1 10%, 6-2 16%, 6-4 18%, 7-1 16%.
Interim reading: at 1M steps pixels learn the training levels as fast as `tiles` but transfer the
least. The 1M point is not decisive, though: tiles2 was also low at 1M (16.2%) and jumped to 23.3% at 2M.
