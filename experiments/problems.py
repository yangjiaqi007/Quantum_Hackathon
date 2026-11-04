"""Test windows for the experiments.

Until compiler/ produces real costs, experiments run on random windows that
follow the same h/W contract. Once it does, export a window with

    json.dump({"h": h, "W": W, "chords": [[v.chord for v in slot] for slot in cands]}, f)

and pass the file with --problems path1.json path2.json ...
"""
import json

import numpy as np

CHORDS = ["I", "ii", "iii", "IV", "V", "V7", "vi", "viio"]


def random_window(sizes, seed=None):
    """Random h, W in [0, 1) and random chord labels for a window of the given slot sizes.

    W gets a sparse set of large "rule violation" penalties on top of small
    smoothness costs, which is closer to what the rule compiler emits than
    uniform noise.
    """
    rng = np.random.default_rng(seed)
    h = [rng.random(k) * 0.5 for k in sizes]
    W = []
    for a, b in zip(sizes, sizes[1:]):
        w = rng.random((a, b)) * 0.3
        w += (rng.random((a, b)) < 0.25) * rng.uniform(1.0, 2.0, (a, b))  # parallel 5ths etc.
        W.append(w)
    chords = [list(rng.choice(CHORDS, k)) for k in sizes]
    return [x.tolist() for x in h], [w.tolist() for w in W], chords


def load_window(path):
    with open(path) as f:
        d = json.load(f)
    chords = d.get("chords") or [[f"v{k}" for k in range(len(x))] for x in d["h"]]
    return d["h"], d["W"], chords


def windows(args_problems=None, n=8, sizes=(4, 4, 4), seed=0):
    """List of (name, h, W, chords): from JSON files if given, else random."""
    if args_problems:
        return [(p, *load_window(p)) for p in args_problems]
    return [(f"random_{seed + i}", *random_window(list(sizes), seed + i)) for i in range(n)]
