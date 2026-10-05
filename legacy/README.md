# Legacy: 2017 CS221 project (Deep Q-Learning, TensorFlow 1, Python 2)

This folder keeps the original Stanford CS221 project (December 2017) for reference.
It is **not runnable** on a modern stack (Python 2, TensorFlow 1.x `tf.contrib`, `gym==0.8`,
FCEUX + Lua pipe bridge) and has been superseded by the `mario_rl` package at the repository root.

What is here:

- `bot/` – agents: human, rule-based baseline, and Q-learning agents with hand-crafted features
  (`ManualFeatureAgent`) or a CNN over the 13x16 tile grid (`CNNFeatureAgent`), with experience
  replay and a target network (`bot/QLearnAlgo.py`, `bot/nn/q_model.py`).
- `run.py`, `train.sh`, `test.sh` – entry point and the exact commands used for each experiment.
- `tools/` – plotting scripts; `figs/` – the final comparison plots (learning rate, batch size,
  reward shaping, exploration, architecture).
- `results/` – per-run score logs (distance reached per game) and plots. The TensorFlow 1
  checkpoints, TensorBoard event files and the vendored FCEUX emulator / gym environment were
  removed from the tree; they remain available in git history (commit `541f9b6`).

Best 2017 result: `results/model/20171212_182429` (2 x 3x3 conv CNN, lr 1e-4, adaptive
exploration, distance + death + stuck reward) completed level 1-1 in 374 of 6376 training games,
averaging ~1880 distance over the last 100 games. It only ever trained and tested on level 1-1.

Original README:

> # Super Mario Bro AI with Deep Q Learning
> #### **Developed using Gym Super Mario environment bundle**
> Game play recording: <https://youtu.be/p8vFNvz5ggA>
