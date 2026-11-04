# Person 3: Constraint-preserving QAOA and experiments

`quantum/` contains the XY-mixer implementation and simulator/solver helpers. `experiments/` compares plain and XY mixers, studies depth and dephasing, and evaluates sampling against checker optima. Run a full experiment with `python -m person3.experiments.run_all`; use `--quick` for a smaller run. The checked-in results are in `experiments/results/` and may be stale relative to current code or compiler inputs.
