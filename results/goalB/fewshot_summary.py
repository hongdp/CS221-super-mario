"""Summarize few-shot runs: python experiments/fewshot_summary.py DIR [DIR ...] (also reads zero-shot from the pretrained run)."""
import glob, json, re, sys
import numpy as np

for D in sys.argv[1:]:
    rows = {}
    for d in sorted(glob.glob(D + "/runs/*")):
        m = re.match(r"s(\d)_(\d-\d)_(\w+)", d.split("/")[-1])
        if not m:
            continue
        s, L, arm = m.groups()
        ev = {round(json.load(open(f))["global_step"] / 1000): json.load(open(f))[L]["levels"][L] for f in glob.glob(d + "/*.eval.json")}
        gf = d + "/eval_final_greedy.json"
        g = json.load(open(gf))[L]["levels"][L] if glob.glob(gf) else None
        rows.setdefault(arm, []).append((int(s), L, ev, g))
    print(f"== {D}")
    for arm, rs in rows.items():
        ks = sorted({k for _, _, ev, _ in rs for k in ev})
        for k in ks:
            v = [ev[k] for _, _, ev, _ in rs if k in ev]
            print(f"  {arm:10s} {k:3d}k: progress {100*np.mean([x['progress'] for x in v]):5.1f}  completion {100*np.mean([x['completion'] for x in v]):5.1f}  (n={len(v)})")
        gs = [g for *_, g in rs if g]
        if gs:
            print(f"  {arm:10s} final greedy: progress {100*np.mean([x['progress'] for x in gs]):5.1f}  completion {100*np.mean([x['completion'] for x in gs]):5.1f}")
