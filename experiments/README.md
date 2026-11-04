# Person 3: constraint-preserving QAOA and experiments

## What's here

| File | What it does |
|---|---|
| `quantum/xy_mixer.py` | One-hot qubit layout, W-state preparation, XY ring mixer, cost layer, and the two circuits: `xy_qaoa_circuit` (constraint-preserving) and `x_qaoa_circuit` (plain baseline with one-hot penalty). |
| `quantum/fast_sim.py` | Exact simulators used inside the optimizer loop. XY-QAOA is simulated on the 64 valid states instead of 4096 amplitudes; dephasing is an exact 64x64 density matrix. |
| `quantum/xy_solve.py` | Energy normalization, angle optimization (grid + ramp start, random restarts, INTERP warm start across depths), Aer sampling, and `xy_solve(h, W, p, shots, init=...)`. |
| `experiments/metrics.py` | P(valid), P(optimum), approximation ratio, shots needed to hit the optimum, and distinct good chord sequences. |
| `experiments/problems.py` | Random test windows, or windows loaded from JSON exported by `compiler/`. |
| `experiments/run_all.py` | Runs all four experiments and writes `results/*.png`, `results.json`, `summary.txt`. |
| `tests/` | Circuit vs. Qiskit `Statevector`, dephasing vs. Aer density matrix, QUBO vs. energy, metrics. |

## How it works

Each qubit `x[t,k]` means "slot t uses candidate voicing k" (the same variables as Person 2's `QuadraticProgram`).

- **Plain QAOA** starts in `|+>^n` and mixes with X on every qubit. It explores all 2^12 bitstrings, so it needs a penalty `P·Σ_t(Σ_k x_tk − 1)²` to discourage slots with zero or two voicings. Most of its shots are still invalid.
- **XY-QAOA** starts each slot in a W state (one excitation spread evenly over its K qubits). Its mixer applies `exp(−iβ(XX+YY)/2)` around a ring of each slot's qubits. That gate moves the excitation between qubits but never creates or destroys one. Every shot therefore has exactly one voicing per slot, and the cost layer contains only the real energy, with no penalty.
- **Normalization:** energies are rescaled to [0, 1] per window before optimizing. QAOA angles only make sense relative to the energy scale, and the argmin is unchanged.

## Running

```bash
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m experiments.run_all            # full run
.venv/Scripts/python.exe -m experiments.run_all --quick    # 10-second smoke test
.venv/Scripts/python.exe -m experiments.run_all --problems w1.json w2.json
```

A window file is `{"h": [[...]], "W": [[[...]]], "chords": [["I", "V", ...], ...]}`. Until `compiler/` produces real windows, the experiments use random windows of 3 slots × 4 voicings (12 qubits, 64 valid assignments). The random windows have small smoothness costs plus a sparse set of large "rule violation" penalties.

## Hooks for the rest of the team

- **Person 2 (windowing):** `xy_solve(h, W, p, init=prev.params)` returns `.params` for warm-starting the next window. To fix a boundary voicing, restrict that slot to the one fixed candidate (K=1). The layout handles uneven slot sizes, and a K=1 slot needs no mixer gates.
- **Person 4 (hardware):** `sample_circuit(qaoa, params, shots, backend=...)` accepts any backend with `.run`, including a fake noisy backend.
- **Person 1 (checker):** metrics default to `E.min()` as the optimum, which is exact on a window. Pass `e_opt=` to use the Viterbi checker's value instead.

## Findings

Full run: 8 random windows of 3 slots × 4 voicings, depth p = 1 to 5, about 5.5 minutes. Error bars are the standard error over windows. Raw numbers are in `results/results.json` and `results/summary.txt`.

**1. XY mixer vs. plain QAOA** (`results/mixer_comparison.png`). Values at p = 5:

| | P(valid) | P(optimum) | Approx. ratio (invalid = 0) |
|---|---|---|---|
| XY mixer + W state | 1.000 | 0.208 | 0.933 |
| X mixer, penalty 0.5 | 0.755 | 0.026 | 0.618 |
| X mixer, penalty 1.0 | 0.902 | 0.020 | 0.646 |
| X mixer, penalty 2.0 | 0.621 | 0.016 | 0.455 |
| Uniform guess over the 64 valid assignments | 1.000 | 0.016 | |

XY-QAOA never produces an invalid output, at every depth. Plain QAOA wastes 10 to 80% of its shots on invalid bitstrings, and its P(optimum) stays close to random guessing. A larger penalty does not help. It stretches the energy range, so the real costs become a small part of the landscape (the design doc predicted this).

**2. Depth** (`results/p_opt_vs_depth.png`). XY-QAOA's P(optimum) rises steadily from 0.056 at p = 1 to 0.208 at p = 5, about 13× uniform. At p = 5, a 99% chance of seeing the optimum takes about 20 shots, against about 293 for uniform guessing.

**3. Coherence ablation** (`results/dephasing_ablation.png`). Adding Z errors before each cost layer lowers P(optimum) steadily, from 0.105 at q = 0 to 0.016 at q = 0.5. At q = 0.5 no interference is possible, and the result equals uniform guessing exactly (tested in `tests/`). Re-optimizing the angles under noise recovers very little. The gain therefore comes from interference, not from some classical effect of the circuit. On hardware this means the circuit has to stay coherent: about 2% dephasing per layer already removes a fifth of the gain.

**4. Sampling vs. the checker.** With 2000 Aer shots of the optimized p = 3 circuit, the sampled set contained the checker's optimum in 8 of 8 windows. It also contained every chord sequence within 5% of the optimum (4.0 of 4.0 on average).

### Caveats to say out loud

- **No speedup is claimed.** Each window has 64 valid assignments, so the classical checker solves it exactly and instantly. The point here is that the constraint-preserving circuit works and depends on interference, not that it is faster.
- **Finding 4 is weak evidence.** 2000 uniform guesses over 64 states would also find everything. QAOA's real gain shows in finding 2: far fewer shots for the same confidence.
- **Results are averaged.** P(optimum) varies a lot between windows. On one 3×3 window the optimized p = 3 circuit put 96% of its probability on the 5th-best voicing and missed the optimum. This happens because the optimizer minimizes *expected* energy, not P(optimum). Always keep the best sample, not the most frequent one.
- **The optimizer is non-convex.** The dephasing run re-optimized p = 3 from scratch and got P(optimum) = 0.105 at q = 0. The depth sweep reached 0.137 at p = 3 because it also started from the p = 2 angles (INTERP).
- **These are random windows.** Rerun with `--problems` once `compiler/` exports real windows.
