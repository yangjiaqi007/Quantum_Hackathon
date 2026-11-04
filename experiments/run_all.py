"""Person 3's experiments: evidence for the judges.

    python -m experiments.run_all            # full run, ~minutes
    python -m experiments.run_all --quick    # smoke test
    python -m experiments.run_all --problems w1.json w2.json   # real compiler windows

Writes experiments/results/results.json and one PNG per experiment.

  1. mixer      plain X-mixer QAOA (several penalties) vs XY-mixer QAOA:
                P(valid), P(optimum), approximation ratio, per depth p
  2. depth      P(optimum) vs p for both mixers, against uniform random guessing
  3. dephasing  Z noise of strength q before every cost layer, angles re-optimized
                at each q. At q = 0.5 no interference is possible; if P(optimum)
                falls toward uniform as q grows, interference is what helps.
  4. sampling   Aer shots from the optimized XY circuit vs the checker's optimum:
                did we hit it, and how many distinct good chord sequences did we see
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.metrics import distribution_metrics, sample_metrics
from experiments.problems import windows
from quantum.xy_mixer import energy_tensor
from quantum.xy_solve import XModel, XYModel, depth_sweep, normalize, optimize, sample_circuit
from quantum.xy_mixer import xy_qaoa_circuit

OUT = Path(__file__).parent / "results"

# Validated with the dataviz palette checker (all-pairs, light mode).
XY_COLOR, X_COLOR, REF_COLOR = "#2a78d6", "#eb6834", "#898781"
X_SHADES = ["#f2a27f", "#eb6834", "#a8431c"]  # one hue, light -> dark, for penalty values
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def mean_sem(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- experiments

def exp_mixer_and_depth(problems, p_max, penalties, restarts):
    rows = []
    for name, h, W, chords in problems:
        n = normalize(h, W)
        E = energy_tensor(n.h, n.W)
        models = [("xy", None, XYModel(n.h, n.W))]
        models += [("x", P, XModel(n.h, n.W, P)) for P in penalties]
        for mixer, penalty, model in models:
            for p, params in depth_sweep(model, p_max, n_restarts=restarts).items():
                m = distribution_metrics(model.assignment_probs(params), E, chords)
                rows.append({"problem": name, "mixer": mixer, "penalty": penalty, "p": p,
                             "params": params.tolist(), **m})
                print(f"  {name:>12} {mixer:>2} P={penalty!s:>4} p={p}  "
                      f"valid={m['p_valid']:.3f}  p_opt={m['p_opt']:.3f}  ratio={m['approx_ratio_all']:.3f}")
    return rows


def exp_dephasing(problems, p, qs, restarts):
    rows = []
    for name, h, W, chords in problems:
        n = normalize(h, W)
        E = energy_tensor(n.h, n.W)
        clean, _ = optimize(XYModel(n.h, n.W), p, n_restarts=restarts)
        for q in qs:
            model = XYModel(n.h, n.W, dephasing=q)
            fixed = distribution_metrics(model.assignment_probs(clean), E, chords)
            x, _ = optimize(model, p, init=clean, n_restarts=1)
            reopt = distribution_metrics(model.assignment_probs(x), E, chords)
            rows.append({"problem": name, "q": q, "p": p,
                         "p_opt_fixed_angles": fixed["p_opt"],
                         "p_opt_reoptimized": reopt["p_opt"],
                         "approx_ratio_reoptimized": reopt["approx_ratio_all"],
                         "p_opt_uniform": reopt["p_opt_uniform"]})
            print(f"  {name:>12} q={q:.3f}  p_opt fixed={fixed['p_opt']:.3f}  reopt={reopt['p_opt']:.3f}")
    return rows


def exp_sampling(problems, p, shots, restarts, seed=0):
    rows = []
    for i, (name, h, W, chords) in enumerate(problems):
        n = normalize(h, W)
        E = energy_tensor(n.h, n.W)
        params, _ = optimize(XYModel(n.h, n.W), p, n_restarts=restarts)
        samples, bad = sample_circuit(xy_qaoa_circuit(n.h, n.W, p), params, shots, seed=seed + i)
        m = sample_metrics(samples, E, chords, n_shots=shots)
        rows.append({"problem": name, "p": p, "n_infeasible": bad, **m})
        print(f"  {name:>12} hit_opt={m['hit_optimum']}  p_opt={m['empirical_p_opt']:.3f}  "
              f"good chord seqs {m['distinct_good_chord_sequences']}/{m['reachable_good_chord_sequences']}")
    return rows


# ---------------------------------------------------------------- plots

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def _series(rows, key, **filt):
    ps = sorted({r["p"] for r in rows})
    out = []
    for p in ps:
        vals = [r[key] for r in rows if r["p"] == p and all(r[k] == v for k, v in filt.items())]
        out.append(mean_sem(vals))
    return ps, np.array(out)


def plot_mixer(rows, penalties, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("p_valid", "Valid outputs (one voicing per slot)", "P(valid)"),
              ("p_opt", "Probability of sampling the optimum", "P(optimum)"),
              ("approx_ratio_all", "Approximation ratio (invalid = 0)", "ratio")]
    for ax, (key, title, ylab) in zip(axes, panels):
        ps, s = _series(rows, key, mixer="xy")
        ax.errorbar(ps, s[:, 0], yerr=s[:, 1], color=XY_COLOR, lw=2, marker="o", ms=7,
                    capsize=3, label="XY mixer + W state")
        for P, c in zip(penalties, X_SHADES):
            ps, s = _series(rows, key, mixer="x", penalty=P)
            ax.errorbar(ps, s[:, 0], yerr=s[:, 1], color=c, lw=2, marker="s", ms=6,
                        capsize=3, label=f"X mixer, penalty {P}")
        if key == "p_opt":
            ax.axhline(np.mean([r["p_opt_uniform"] for r in rows]), color=REF_COLOR, lw=1.5,
                       ls="--", label="uniform over valid")
        _style(ax, title, "QAOA depth p", ylab)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(ps)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_depth(rows, penalties, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ps, s = _series(rows, "p_opt", mixer="xy")
    ax.errorbar(ps, s[:, 0], yerr=s[:, 1], color=XY_COLOR, lw=2, marker="o", ms=7, capsize=3,
                label="XY mixer")
    ax.annotate("XY mixer", (ps[-1], s[-1, 0]), xytext=(6, 0), textcoords="offset points",
                color=INK, va="center", fontsize=9)
    best_P = max(penalties, key=lambda P: _series(rows, "p_opt", mixer="x", penalty=P)[1][-1, 0])
    ps, s = _series(rows, "p_opt", mixer="x", penalty=best_P)
    ax.errorbar(ps, s[:, 0], yerr=s[:, 1], color=X_COLOR, lw=2, marker="s", ms=6, capsize=3,
                label=f"X mixer (best penalty {best_P})")
    ax.annotate("X mixer", (ps[-1], s[-1, 0]), xytext=(6, 0), textcoords="offset points",
                color=INK, va="center", fontsize=9)
    u = np.mean([r["p_opt_uniform"] for r in rows])
    ax.axhline(u, color=REF_COLOR, lw=1.5, ls="--", label="uniform over valid")
    _style(ax, "P(optimum) vs depth", "QAOA depth p", "P(optimum)")
    ax.set_xticks(ps)
    ax.set_xlim(ps[0] - 0.3, ps[-1] + 0.8)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dephasing(rows, path):
    import matplotlib.pyplot as plt
    qs = sorted({r["q"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for key, color, marker, label in [
            ("p_opt_reoptimized", XY_COLOR, "o", "angles re-optimized under noise"),
            ("p_opt_fixed_angles", X_COLOR, "s", "noiseless angles")]:
        s = np.array([mean_sem([r[key] for r in rows if r["q"] == q]) for q in qs])
        ax.errorbar(qs, s[:, 0], yerr=s[:, 1], color=color, lw=2, marker=marker, ms=7,
                    capsize=3, label=label)
    ax.axhline(np.mean([r["p_opt_uniform"] for r in rows]), color=REF_COLOR, lw=1.5, ls="--",
               label="uniform (no interference)")
    _style(ax, f"Coherence ablation, XY-QAOA p={rows[0]['p']}",
           "Z-error probability q per qubit per layer", "P(optimum)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main

def summarize(results):
    mix = results["mixer"]
    p_max = max(r["p"] for r in mix)
    xy = [r for r in mix if r["mixer"] == "xy" and r["p"] == p_max]
    lines = [f"At p={p_max}, over {len(xy)} windows:",
             f"  XY mixer:  P(valid) = 1.000, P(optimum) = {mean_sem([r['p_opt'] for r in xy])[0]:.3f}, "
             f"ratio = {mean_sem([r['approx_ratio_all'] for r in xy])[0]:.3f}"]
    for P in results["config"]["penalties"]:
        x = [r for r in mix if r["mixer"] == "x" and r["penalty"] == P and r["p"] == p_max]
        lines.append(f"  X, P={P}:  P(valid) = {mean_sem([r['p_valid'] for r in x])[0]:.3f}, "
                     f"P(optimum) = {mean_sem([r['p_opt'] for r in x])[0]:.3f}, "
                     f"ratio = {mean_sem([r['approx_ratio_all'] for r in x])[0]:.3f}")
    lines.append(f"  uniform guess: P(optimum) = {mean_sem([r['p_opt_uniform'] for r in xy])[0]:.3f}")
    deph = results["dephasing"]
    for q in sorted({r["q"] for r in deph}):
        v = mean_sem([r["p_opt_reoptimized"] for r in deph if r["q"] == q])[0]
        lines.append(f"  dephasing q={q:.3f}: P(optimum) = {v:.3f}")
    samp = results["sampling"]
    lines.append(f"  sampling: hit the optimum in {sum(r['hit_optimum'] for r in samp)}/{len(samp)} windows "
                 f"with {samp[0]['shots']} shots; mean good chord sequences found "
                 f"{np.mean([r['distinct_good_chord_sequences'] for r in samp]):.1f} of "
                 f"{np.mean([r['reachable_good_chord_sequences'] for r in samp]):.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small run for testing")
    ap.add_argument("--problems", nargs="*", help="JSON windows exported from compiler/")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--replot", action="store_true", help="redraw plots from <out>/results.json")
    args = ap.parse_args()

    if args.replot:
        out = Path(args.out)
        results = json.loads((out / "results.json").read_text())
        plot_mixer(results["mixer"], results["config"]["penalties"], out / "mixer_comparison.png")
        plot_depth(results["mixer"], results["config"]["penalties"], out / "p_opt_vs_depth.png")
        plot_dephasing(results["dephasing"], out / "dephasing_ablation.png")
        return

    if args.quick:
        cfg = dict(n=2, sizes=[3, 3, 3], p_max=2, penalties=[0.5, 1.0], restarts=2,
                   qs=[0.0, 0.05, 0.5], deph_p=2, shots=500)
    else:
        cfg = dict(n=8, sizes=[4, 4, 4], p_max=5, penalties=[0.5, 1.0, 2.0], restarts=6,
                   qs=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], deph_p=3, shots=2000)
    problems = windows(args.problems, n=cfg["n"], sizes=cfg["sizes"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("1+2. mixer comparison and depth sweep")
    mixer_rows = exp_mixer_and_depth(problems, cfg["p_max"], cfg["penalties"], cfg["restarts"])
    print("3. dephasing ablation")
    deph_rows = exp_dephasing(problems, cfg["deph_p"], cfg["qs"], cfg["restarts"])
    print("4. sampling vs checker")
    samp_rows = exp_sampling(problems, cfg["deph_p"], cfg["shots"], cfg["restarts"])

    results = {"config": cfg, "problems": [p[0] for p in problems],
               "mixer": mixer_rows, "dephasing": deph_rows, "sampling": samp_rows}
    results["summary"] = summarize(results)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    plot_mixer(mixer_rows, cfg["penalties"], out / "mixer_comparison.png")
    plot_depth(mixer_rows, cfg["penalties"], out / "p_opt_vs_depth.png")
    plot_dephasing(deph_rows, out / "dephasing_ablation.png")
    (out / "summary.txt").write_text(results["summary"])
    print(results["summary"])
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
