# Experiment log

This is a chronological record of every experiment run during the 2026 port, including the
diagnostics and dead ends behind the design decisions. Raw data lives next to this file:

- `tiles-uniform-8m/`: main run
- `tiles-sticky-3m/`: sticky actions
- `diagnostics/`: early runs and A/B tests

**Machine**: 4 vCPU Intel Xeon @ 2.8 GHz, 15 GB RAM, no GPU. **Software**: Python 3.11, PyTorch 2.14
(CPU), Gymnasium 1.3, stable-retro 1.0.1, NumPy 2.4.

**Note**: `tiles2` became the default observation after E10. All runs before E9 used `--obs tiles`,
which was the default then.

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

Folder: `tiles-uniform-8m/`. `--obs tiles`, otherwise default config; 22 training levels, uniform sampling, 8M steps.
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

## E10: Pixel baseline, `pixels-uniform-3m`

Folder: `pixels-uniform-3m/`. Main-run config with `--obs pixels` (84x84 grayscale, Nature CNN for
both actor and critic), 3M steps. Run 2026-10-04 04:30-09:10 (4.6 h, ~185 env steps/s vs ~450 for
tiles2). The rollout phase runs at ~600 steps/s, and each PPO update takes ~8 s. The weights
(~13 MB) are not committed.

| steps | train progress | train completion | held-out progress | held-out completion |
|---|---|---|---|---|
| 1M | 29.3% (tiles 29.7%, tiles2 34.9%) | 0.9% | 14.2% (tiles 19.5%, tiles2 16.2%) | 0.0% |
| 2M | 36.2% (tiles 35.5%, tiles2 45.3%) | 2.3% | 15.2% (tiles 20.2%, tiles2 23.3%) | 0.0% |
| 3M | 40.6% (tiles 38.5%, tiles2 51.2%) | 4.1% | **16.5%** (tiles 20.8%, tiles2 24.4%) | 0.0% |

Held-out per level at 3M: 2-1 18%, 3-3 24%, 4-2 6%, 5-1 11%, 6-2 23%, 6-4 18%, 7-1 16%.
Training-level completions at 3M: 3-2 7/10, 1-4 2/10.

**Conclusion**: pixels learn the training levels at the same rate as `tiles` but transfer the least.
Held-out progress stays at 14-17%, below both grid observations at every checkpoint, while
costing ~2.5x the compute per step. With only 22 training levels, raw pixels give the network
level-specific appearance to latch onto (palettes and backgrounds differ between worlds).
**Answer to "should we switch to pixels?"**: no. The grid does lose information, and the fix is a
richer grid (`tiles2`), not raw frames. Pixels would need heavy augmentation (random crops,
palette jitter) to compete, if they can at all.

## Summary at 3M steps (10 episodes per level)

| observation | train progress | train completion | held-out progress | env steps/s |
|---|---|---|---|---|
| `tiles` | 38.5% | 1.8% | 20.8% | ~650 |
| `tiles` + sticky actions | 40.3% | 4.1% | 17.3% | ~650 |
| **`tiles2`** | **51.2%** | **15.5%** | **24.4%** | ~450 |
| `pixels` | 40.6% | 4.1% | 16.5% | ~185 |

Plot: `comparisons/obs_3m.png`. No variant has completed a held-out level yet. The biggest
remaining problem is generalization from only 22 training levels.

# 10M-step study (E11-E20), 2026-10-05/06

All runs: `tiles2`, 8 envs × 256 steps, 10M env steps, every 1M checkpoint re-evaluated with 10 stochastic
episodes per level. "Held-out late" = mean held-out progress over the 6-10M checkpoints, which smooths the ~2-point
noise of single checkpoints. Per-checkpoint evals and configs are in `10m-study/<run>/`. Full per-run records
(purpose, pre-registered criteria, progress) were kept locally in `experiments/` (gitignored).
From E12's second half on, the PPO update runs on the GPU (`--device`). From E15 on, actions are also chosen on the
GPU (`--inference batched`). A 100-150k-step A/B showed matching learning curves for both changes.

## E11: tiles2 uniform to 10M — does held-out keep rising?
No. Held-out plateaus at 20-24% from 2M on, while train progress climbs from 33% to 68%, so the gap grows from 13 to
47 points. Longer training only adds memorization. (E9's apparent "still rising" at 3M was noise.)

## E12, E15, E16: Prioritized Level Replay (β=0.1) vs uniform, 3 seeds

| seed | uniform held-out late | PLR held-out late | Δ |
|---|---|---|---|
| 1 (E11 / E12) | 20.3 | 28.0 | +7.7 |
| 2 (E15 / E16) | 21.8 | 22.0 | +0.1 |
| 3 (E15 / E16) | 22.3 | 23.3 | +1.1 |
| mean | 21.5 ± 1.0 | 24.4 ± 3.2 | +2.95 |

A small, consistent gain (pre-registered bar: +3). E12's +7.7 was a lucky seed. PLR costs ~3 points of train
completion. At β=0.1 PLR is near-greedy: rank^-10 puts ~90% of samples on one level.

## E13: DrAC random-shift augmentation (±4 cells)
Held-out late 23.3 vs 20.3 (seed 1), mostly from the held-out castle 6-4 (~20 → 30-45%). Train completion went up
(35% vs 28%). The policy became shift-robust but kept memorizing. Missed its bar by 0.5 points.

## E14: IMPALA-style residual encoder
Refuted. It learns training levels much more slowly at equal steps (56% vs 68% at 10M), gives no held-out gain
(20.4 late), and is 1.7x slower. Its smaller gap only reflects underfitting.

## E17: PLR temperature (seed 1)
β=0.1 / 0.3 / 1.0 → held-out late 28.0 / 24.1 / 26.0, all within PLR's ±3 seed spread. β=1.0 is the only soft
setting: max level weight 24%, no level below 1%. If PLR is used, β=1.0 is the safer default.

## E18: PLR + DrAC (seed 1)
Held-out late 29.4 (PLR alone 28.0): the best single run, but +1.4 is within seed noise.

## E19: held-out failure analysis — is the grid the problem?
140 episodes of the best checkpoint on held-out levels, with screen | grid | overlay snapshots before each failure
(`10m-study/E19-failure-analysis/`). The grid is exactly aligned, and the cause of death was visible in it in every
inspected case. Failures cluster at single obstacles: all 20 episodes of 3-3 die in one gap, 11/20 of 4-2 in one gap,
and piranha plants on pipes in 2-1/5-1/6-2. The agent sees these obstacles but has memorized training-level timing
instead of learning obstacle skills. One encoding flaw: classes are exclusive, so an object overwrites terrain (a
piranha erases the solid pipe top). Power-ups and ?-blocks are not represented (by design).

## E20: egocentric window (4 cells behind / 12 ahead of Mario), uniform, 3 seeds
Hypothesis from E19: the full-screen grid gives the network enough layout to fingerprint its position, and a
Mario-anchored window removes that while keeping what the decision needs.

| seed | full screen held-out late | ego held-out late | Δ |
|---|---|---|---|
| 1 | 20.3 | 26.2 | +5.8 |
| 2 | 21.8 | 26.9 | +5.0 |
| 3 | 22.3 | 23.3 | +1.0 |
| mean | 21.5 ± 1.0 | **25.4 ± 1.9** | **+3.95** |

Supported, meeting every pre-registered criterion. Train progress is unchanged (62.6 vs 63.9), so the gap shrinks from 42 to
37 points. Gains are broad (2-1 25→32, 3-3 18→26, 6-4 16→21, 7-1 34→39). This is the first intervention that improves
transfer without trading anything away. Plot: `10m-study/families_10m.png`.

## Summary

| intervention (vs full-screen uniform) | held-out late | train cost | seeds |
|---|---|---|---|
| longer training (3M → 10M) | no gain | — | 1 |
| PLR β=0.1 | +2.95 | −3 pts completion | 3 |
| DrAC shift | +3.0 | none (train up) | 1 |
| IMPALA encoder | +0.1 | −12 pts progress | 1 |
| **egocentric window 4/12** | **+3.95** | **none** | **3** |


## E21, E22: window size and ego + PLR (3 seeds each)

| config | held-out late | per-seed Δ vs ego 4/12 | train completion @10M | gap |
|---|---|---|---|---|
| full screen | 21.5 ± 1.0 | −5.8 / −5.0 / −1.0 | 29.1 | 42.4 |
| ego 4/12 (E20) | 25.4 ± 1.9 | — | 29.2 | 37.1 |
| **ego 2/8** | **30.2 ± 1.2** | **+4.3 / +4.4 / +5.6** | **20.8** | **28.8** |
| ego 6/16 | 27.7 ± 3.5 | −1.6 / 0.0 / +8.3 | 29.2 | 36.3 |
| ego 4/12 + PLR β=1.0 | 25.1 ± 3.4 | −0.7 / +1.5 / −1.7 | 25.8 | 36.5 |

- **2/8 gives the best transfer of the project**: +8.7 over full screen, better than 4/12 for every seed, with the
  smallest spread. It costs 8 points of training-level completion, because 64 px of lookahead is less than a full
  running jump.
- 6/16 is inconclusive: one seed at +8.3 and the other two at or below 4/12.
- PLR adds nothing on top of the window (refuted) and again costs train completion. It is dropped from the recipe.
- **Trade-off**: a narrow view is best for unseen levels (goal B), while a wider lookahead is needed to clear levels (goal A).


# Goal A: clearing the training levels

## E23: mid-level starts (`--mid-start-prob 0.5`), ego 4/12, 3 seeds
Every episode used to start at the level start and die around 50-60% of the level, so the late sections got little
practice. E23 started half of the training episodes from archived mid-level states. **Refuted**: training-level
completion fell from 29.2 (E20) to 26.1, lower for every seed. Static levels gained (1-4 50→73%, 5-2 3→17%, first
completions of 1-2, 2-2, 8-2 and 8-3), but athletic lift levels collapsed (1-3 67→0%, 4-3 17→0%). Diagnosis: after
loading a state, reset still ran 0-30 random idle frames, and states were archived while Mario stood on lifts. Restarts on
lift levels were therefore often doomed: death within 40 steps of a 1-3 restart was 41% as trained, 19% without idle
frames, and 7% after fixing both flaws. E24 re-tests the fixed version.

## E24: fixed mid-level starts, 3 seeds
Same as E23 with both flaws fixed. **Refuted again**: train completion is 28.0 vs 29.2 for E20 (+4.5 / −2.7 / −5.5 per seed).
The fix rescued 4-3 (0 → 23%), but 1-3 stays at 0/30 (E20: 19/30). Its lift sections are never archived, so they are
only practised from level starts, and those are halved. At this budget the static-level gains (1-4 +20, 3-4 +13)
are cancelled by the lost level-start practice. For goal A, mid-level starts are shelved. The next step is a long
run (100M steps) with the E20 recipe on all 29 non-maze levels, since train progress was still rising 2.5-4.3 points/M at 10M.

## E25: goal A long run — 29 non-maze levels, ego 4/12, 100M steps, 2 seeds
The E20 recipe unchanged (uniform sampling, no mid-starts), trained on every non-maze level for 100M env steps (~9 h,
two seeds side by side on one RTX 4080 + 12-core CPU). Evaluation from the level start, 10 episodes per level:

| steps | completion (seed 1 / 2) | levels completed ≥1× | levels ≥50% |
|---|---|---|---|
| 10M | 14.1 / 10.7 | 13 / 8 | 3 / 3 |
| 50M | 44.1 / 36.9 | 22 / 21 | 13 / 11 |
| 100M | **66.2 / 63.8** | 24 / 26 | 21 / 19 |
| 100M, greedy actions | **70.7 / 71.0** | 24 / 23 | 22 / 21 (13 / 16 at 100%) |

All pre-registered targets were met (≥50% completion, ≥25 levels cleared at least once, stretch ≥20 levels ≥50%). Training
length was the bottleneck for goal A, and the curve was still rising at 100M. Remaining failures: **1-2 and 4-2 are an env
trap**. The agent runs on top of the ceiling into the warp zone (90-100% progress, never the flag), because warp pipes
need DOWN and lead to other worlds. This is now blocked by `block_ceiling_route`. Also failing: the water levels 2-2 and
7-2 (0%), the 6-4 castle, and the seed-dependent 8-1 and 8-3. Per-checkpoint evals: `goalA-100m/`.

**Phases 2-3**: both seeds resumed to 120M with ceiling fix v1, then to 140M with v2. In v2, progress above the screen
earns nothing on 1-2/4-2. In v1, Mario bobbing on the top edge kept resetting the counter, so the ceiling run still
outpaid the exit pipe.

| checkpoint | greedy completion (seed 1 / 2) | levels ≥50% | 1-2 | 4-2 |
|---|---|---|---|---|
| 100M | 70.7 / 71.0 | 22 / 21 | 0 / 0 | 0 / 0 |
| 120M (fix v1) | 75.5 / 76.6 | 22 / 22 | 0 / 0 | 100 / 70 (stoch.) |
| **140M (fix v2)** | **80.7 / 79.3** | **25 / 25** | **90 / 70** | **100 / 100** |

Still below 50% (greedy): the water levels 2-2 and 7-2, the castle 6-4, and 8-1 / 7-1 (one seed each).

## E26: why 2-2, 7-2 and 6-4 still fail (140M, seed 1)
The grid is accurate in all three. **6-4**: 13/16 failures happen at the Bowser bridge (90% progress), a skill gap at one spot.
**Water**: a control flaw. Swim strokes trigger on a new A press, but the 4-frame action repeat makes consecutive A actions
one long hold, so holding "A" sinks Mario to the floor of 2-2 within ~20 steps unless the agent alternates A and no-op.
Fixed by `EnvConfig.swim_tap` (default on): A is released on the last frame of each step while swimming. Land jumps are unchanged.

## E25 phases 4-5 (140M → 180M)
Phase 4 resumed with `swim_tap`. Phase 5 used weighted level sampling, weight = 0.2 + (1 − greedy completion) per level,
which gives the weakest levels 8-11% of episodes each instead of 3.4%.

| checkpoint | greedy completion (seed 1 / 2) | levels ≥50% | below 50% |
|---|---|---|---|
| 160M (swim_tap) | 81.4 / 81.4 | 26 / 25 | 2-2, 7-2, 8-1 (+ 6-4 seed 2) |
| **180M (weighted)** | **87.2 / 83.8** | **27 / 26** | 2-2, 7-2 (+ 8-3 seed 2) |

Targeted practice fixed 8-1 (0 → 70 / 30 → 90) and 6-4 (seed 2: 20 → 80). **The water levels stay at 0-30%** despite the swim
fix and ~3x the practice.

**Maze castles (not yet trained).** A probe with the 140M policy shows that in 7-4 every wrong path teleports Mario back
and the walk forward is paid again, so looping farms reward until the stuck limit. The correct 8-4 route needs DOWN
(pipes), which `simple` lacks. Clearing the mazes needs (1) a new-ground-only progress reward on maze levels and (2) an
action set with DOWN, which means a new training run because the policy output changes.

# Goal B: unseen levels, continued (corrected env)

## E28, E29: terrain perturbation (3 seeds each)
E28 re-establishes the baseline under the corrected env (hammers visible, swim fix, ceiling fix): ego 2/8 window,
held-out **28.2 ± 2.2** (E21 under the old env: 30.2 ± 1.2). E29 adds `terrain_noise 0.1`, random extra blocks (floating platforms,
1-block steps) written into the block buffer as columns scroll in, so the agent sees and collides with them. **Refuted**:
held-out 25.9 ± 3.1 (−2.2, worse for every seed), and train progress drops by 7. The extra blocks make training harder but are not the kind of
hazard that kills on held-out levels (gaps, piranha plants, lifts).

## E31: 16-frame history (3 seeds)
Meant to give the policy timing information (piranha cycles take 2-3 s, while 4 frames cover 0.27 s). **Refuted**: held-out 21.5 ± 3.3
(−6.7 vs E28, worse for every seed), worse on the piranha levels too. Across E10/E20/E21/E31, every increase in input size hurt
transfer and every reduction helped.

## E30: few-shot fine-tuning on held-out levels (pretrained E28 vs scratch, 3 seeds × 7 levels)
| practice steps | pretrained progress / completion | scratch progress / completion |
|---|---|---|
| 0 | 30.0 / 2.4 | — |
| 25k (≈ 28 min of play) | **40.5** / 3.3 | 14.3 / 0 |
| 100k (≈ 1.9 h) | **50.8** / 7.1 (greedy 16.2) | 20.3 / 0 |

The 22-level pretraining transfers. It gives a 2.5x head start in progress at every budget. But it is shallow: even after
100k steps on a level the agent rarely clears it (best: 5-1 at 30%). Per-run evals are in `goalB/E30-fewshot/`.
