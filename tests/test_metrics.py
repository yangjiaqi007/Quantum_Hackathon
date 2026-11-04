import math

import numpy as np
import pytest

from experiments.metrics import (distribution_metrics, good_chord_sequences, sample_metrics,
                                 shots_to_hit)


def test_distribution_metrics_point_mass_on_optimum():
    E = np.array([[0.0, 1.0], [0.5, 0.2]])
    P = np.array([[1.0, 0.0], [0.0, 0.0]])
    m = distribution_metrics(P, E)
    assert m["p_opt"] == 1.0 and m["approx_ratio"] == 1.0 and m["p_valid"] == 1.0
    assert m["p_opt_uniform"] == 0.25


def test_invalid_mass_counts_as_zero_ratio():
    E = np.array([0.0, 1.0])
    P = np.array([0.4, 0.0])  # 60% of shots were invalid bitstrings
    m = distribution_metrics(P, E)
    assert m["p_valid"] == pytest.approx(0.4)
    assert m["approx_ratio"] == pytest.approx(1.0)
    assert m["approx_ratio_all"] == pytest.approx(0.4)


def test_shots_to_hit():
    assert shots_to_hit(0.0) == math.inf
    assert shots_to_hit(1.0) == 1
    assert 1 - (1 - 0.1) ** shots_to_hit(0.1) >= 0.99


def test_good_chord_sequences_and_sampling():
    E = np.array([[0.0, 0.01], [1.0, 0.02]])
    chords = [["I", "V"], ["IV", "I"]]
    seqs = good_chord_sequences(E, chords, within=0.05)
    assert seqs == {("I", "IV"), ("I", "I"), ("V", "I")}
    m = sample_metrics([(0, 0), (0, 0), (1, 1)], E, chords, n_shots=4)
    assert m["hit_optimum"] and m["feasible_fraction"] == 0.75
    assert m["empirical_p_opt"] == 0.5
    assert m["distinct_good_chord_sequences"] == 2
    assert m["reachable_good_chord_sequences"] == 3
