# Quantum Four-Part Harmony

**Bridging music theory and QAOA with Qiskit** — Qiskit Fall Fest 2026, University of Waterloo.

Give the tool a soprano melody. It writes the alto, tenor and bass: the rules of four-part harmony are compiled into a
penalty Hamiltonian, a swappable solver (classical Viterbi, QAOA, or constraint-preserving XY-mixer QAOA) finds a
low-energy voicing, and you get a score, audio you can listen to, MIDI/MusicXML files and an honest quantum resource report.
Nobody has to write a circuit.

| Hackathon prompt | This project |
|---|---|
| **A real problem outside quantum computing** | Four-part harmonization: a standard music-theory exercise taught for centuries, with codified rules and Bach's chorales as expert ground truth. |
| **A useful bridge** | A rules → Hamiltonian compiler plus an interactive notebook interface. Musicians state rules in musical terms; the tool builds candidates, penalty terms (QUBO / Ising) and decodes the answer back to a score. |
| **A quantum idea in Qiskit** | Windowed, constraint-preserving QAOA (W-state start + XY mixer keeps exactly one voicing per chord slot), run through the Qiskit V2 primitives on a simulator, a noisy fake backend, or hardware. |

## Quick start

```bash
pip install qiskit qiskit-optimization qiskit-aer music21 ipywidgets matplotlib numpy pylatexenc
jupyter lab project.ipynb      # or open it in VS Code
```

1. Run the notebook top to bottom.
2. Run `p5_selfcheck()` — it prints a ✔/✘ checklist (dependencies, pipeline hook, score / audio / export, 4-note end-to-end run).
3. Use the panel from `p5_launch()`: choose *Ode to Joy*, *O Canada*, type notes (`E4 E4 F4 G4`, durations as `E4:1.5`), click the
   on-screen note buttons, or upload a MIDI file. Pick a style, a solver and press **Run harmonization**.
4. The *Score & audio*, *Quantum report* and *Export* tabs fill in. Files are also saved to `p5_output/`.

`p5_launch("zh")` switches the interface to Chinese. `p5_demo_showcase()` renders both demo melodies as static outputs that stay
in the saved notebook.

## How it works

```
melody (+ style, solver)
  → candidate voicings per chord slot           (Part 1)
  → rules as penalty terms  h[t][k], W[t][k][l] (Part 1)  → QuadraticProgram → QUBO → Ising   (Qiskit Optimization)
  → solver: viterbi | qaoa | xy_qaoa, windows of 3-4 slots, warm-started angles (Parts 2-3)
  → resource report: qubits, depth, two-qubit gates, energy per window, violations            (Part 4)
  → decode → score (PNG) + audio (WAV) + MIDI + MusicXML + report                             (Part 5)
```

| Part | Owner's job |
|---|---|
| 1 | Melody data, candidate generator, rules, style presets, rule checker |
| 2 | QUBO/Ising construction, QAOA, sliding windows with parameter transfer |
| 3 | XY-mixer QAOA, experiments |
| 4 | Backends / hardware, resource and honesty report |
| 5 | Interface, score rendering, audio, export, report display, README, demo, slides |

### Part 5 in detail (cells `P5-00 … P5-14`)

| Cell | What it does |
|---|---|
| P5-01 | Parses note names (`C4 F#4 Bb3 E4:1.5`), validates range / length, guesses the key, reads MIDI files, holds the Ode to Joy preset |
| P5-02 | `P5Score` data model; turns whatever the pipeline returns (tuples, dict, music21 Stream) into one normalized score; sanity checks (soprano preserved, voice order, ranges) |
| P5-03 | music21 → 4-part score with measures, ties and pickup → MIDI and MusicXML; saves files; download links |
| P5-04 | Synthesizes the chorale with numpy (additive synthesis → WAV). A browser cannot play `.mid`, so the notebook does not depend on a soundfont |
| P5-05 | Draws the grand staff with matplotlib (no MuseScore/LilyPond needed): accidentals, ties, dots, flags, chord labels, violations in red, QAOA windows as shaded bands |
| P5-06 | Report card, energy-per-window chart, circuit diagram, recorded-hardware box |
| P5-07 | The only place that calls the team pipeline; tolerant to different parameter names |
| P5-08 / 09 | Display panels and the ipywidgets interface |
| P5-10 | **Mock** classical harmonizer for testing — delete once the real pipeline is in the notebook |
| P5-11 | Static showcase and `p5_selfcheck()` |

All names defined by Part 5 start with `p5_` / `P5` (including import aliases), so the cells can be pasted into the team notebook
without colliding with other parts.

## Integration contract (Parts 1–4 ↔ Part 5)

```python
def harmonize_pipeline(melody, style="bach_chorale", style_weight=1.0, solver="xy_qaoa", progress=None,
                       layers_p=2, shots=2048, window_size=3, noisy=False):
    ...
    return result, report
```

Only the parameters the function declares are passed (a `**kwargs` function receives all of them).

* `melody` — dict: `title`, `preset` (`"ode_to_joy"`, `"o_canada"` or `None`), `notes` (`[(midi, beats), …]`, **`None` for O Canada: Part 1 resolves
  the preset id**), `key` (major tonic: C G D A E F Bb Eb Ab), `time_sig` `(4, 4)`, `pickup` (beats in an incomplete first bar), `tempo`.
* `result` — one voicing per soprano note, any of: list of `(bass, tenor, alto, soprano)` MIDI tuples · dict `{"voicings": [...],
  "chords": ["I", …], "durations": [...]}` · 4-part music21 Stream (S, A, T, B from top). The soprano must be unchanged.
* `report` — any subset of: `solver, backend, qubits, depth, two_qubit_gates, shots, layers_p, runtime_s, violations,
  violation_details [{rule, slot (0-based), voices (0 = bass … 3 = soprano)}], energy, classical_energy, approx_ratio, p_optimum,
  feasible_fraction, windows [{start, end (exclusive), qubits, depth, two_qubit_gates, energy, classical_energy}],
  circuit (QuantumCircuit or list), hardware {…recorded real-device run…}, verdict, notes`. Missing keys are simply not shown.
* Call `progress(fraction, "message")` (optional) to drive the progress bar.

## Honest assessment — why quantum?

Four-part harmony with nearest-neighbour rules is a chain, which classical dynamic programming (Viterbi) solves **exactly and
efficiently**. The project ships Viterbi as the baseline and **does not claim a quantum advantage**. What it does offer:

* a bridge that lets domain experts express rules in their own vocabulary and run them on classical or quantum solvers interchangeably;
* an interpretable benchmark family for constraint-preserving QAOA — you can *hear* a violated rule;
* experiments (Part 3) that compare the plain and XY mixers, sweep QAOA depth, and test whether coherence matters.

## Known limitations

* The score renderer is deliberately simple: no beams, rests or tuplets; notes crossing a barline are tied.
* Melodies typed in the interface are capped at 40 notes and must be monophonic.
* The audio is a simple synthesized timbre, not a recording.
* If the pipeline is not defined, the interface falls back to the labelled mock and says so in the report.
