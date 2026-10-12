"""Widen a tiles2 checkpoint's vec input with zero weights (new inputs start with no effect), for --init-from.

usage: python scripts/widen_vec.py SRC.pt DST.pt EXTRA ENV_OVERRIDES_JSON
"""

import json
import sys

import torch

from mario_rl.env import EnvConfig, observation_spec
from mario_rl.models import ActorCritic
from mario_rl.ppo import load_policy

src, dst, extra, overrides = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
old, ck = load_policy(src)
env_cfg = EnvConfig(**{**ck["env_config"], **overrides})
obs_type, shape, n = observation_spec(env_cfg)
assert shape["vec"][0] == old.obs_shape["vec"][0] + extra, (shape, old.obs_shape)
new = ActorCritic(obs_type, shape, n, old.shared, old.arch, old.aux)
state = old.state_dict()
for k, v in new.state_dict().items():
    w = state[k]
    if (
        w.shape != v.shape
    ):  # the FC layer fed by [grid features, vec]: append zero columns for the new vec inputs
        assert k.endswith("fc.0.weight") and w.shape[1] + extra == v.shape[1], k
        w = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    state[k] = w
new.load_state_dict(state)
# the widened model must act exactly like the old one
g = torch.randint(0, 6, (5, *old.obs_shape["grid"]), dtype=torch.uint8)
v = torch.randn(5, old.obs_shape["vec"][0])
with torch.no_grad():
    lo, vo = old({"grid": g, "vec": v})
    ln, vn = new({"grid": g, "vec": torch.cat([v, torch.randn(5, extra)], 1)})
assert torch.allclose(lo, ln, atol=1e-5) and torch.allclose(vo, vn, atol=1e-5)
ck["model"], ck["model_spec"], ck["env_config"] = new.state_dict(), new.spec(), env_cfg.to_dict()
torch.save(ck, dst)
print("widened", src, "->", dst, "vec", old.obs_shape["vec"], "->", shape["vec"], "| outputs identical: True")
