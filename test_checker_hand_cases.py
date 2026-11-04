"""Hand-computed cases for the checker.

The energy test only proves the compiled costs and the checker agree. These
cases pin the checker to musical ground truth so they cannot both be wrong
together. ``sv(S, A, T, B, chord)`` takes pitches soprano-first.
"""
from __future__ import annotations

import pytest

from schema import ALTO, BASS, SOPRANO, TENOR

from compiler import Key, StyleWeights, check, chord_degree, explain, get_preset

from .helpers import sv

C_MAJOR = Key(0, "major")
A_MINOR = Key(9, "minor")
BACH = get_preset("bach_chorale")


def count(report, rule):
    return report.count[rule]


def test_parallel_fifths_between_tenor_and_bass():
    r = check([sv(76, 64, 55, 48), sv(76, 64, 57, 50)], C_MAJOR, BACH)  # C3/G3 -> D3/A3, upper voices hold
    assert count(r, "parallel_fifths_octaves") == 1
    assert r.by_rule("parallel_fifths_octaves")[0].voices == (TENOR, BASS)


def test_parallel_octaves_soprano_bass_are_not_also_hidden():
    r = check([sv(72, 64, 55, 48), sv(74, 64, 55, 50)], C_MAJOR, BACH)
    assert count(r, "parallel_fifths_octaves") == 1
    assert r.by_rule("parallel_fifths_octaves")[0].voices == (SOPRANO, BASS)
    assert count(r, "hidden_fifths") == 0


def test_held_voices_are_not_parallel():
    r = check([sv(72, 67, 55, 48), sv(72, 67, 55, 48)], C_MAJOR, BACH)  # same sonority repeated
    assert count(r, "parallel_fifths_octaves") == 0


def test_hidden_fifth_needs_similar_motion_and_soprano_leap():
    r = check([sv(72, 64, 60, 55), sv(76, 64, 60, 57)], C_MAJOR, BACH)  # P4 -> P12, both up, soprano leaps a third
    assert count(r, "hidden_fifths") == 1
    assert count(r, "parallel_fifths_octaves") == 0

    stepwise = check([sv(74, 64, 60, 55), sv(76, 64, 60, 57)], C_MAJOR, BACH)  # soprano moves by step
    assert count(stepwise, "hidden_fifths") == 0

    contrary = check([sv(72, 64, 60, 59), sv(76, 64, 60, 57)], C_MAJOR, BACH)  # soprano up, bass down
    assert count(contrary, "hidden_fifths") == 0


def test_voice_crossing():
    r = check([sv(60, 64, 55, 48)], C_MAJOR, BACH)  # soprano below alto
    assert count(r, "voice_crossing") == 1
    assert r.by_rule("voice_crossing")[0].voices == (SOPRANO, ALTO)


def test_unison_is_not_crossing():
    assert count(check([sv(64, 64, 55, 48)], C_MAJOR, BACH), "voice_crossing") == 0


def test_spacing():
    r = check([sv(84, 64, 55, 48)], C_MAJOR, BACH)  # soprano-alto 20 semitones, limit 12
    assert count(r, "spacing") == 1
    assert r.severity["spacing"] == pytest.approx(1 + 8 / 12)

    r = check([sv(72, 64, 60, 36)], C_MAJOR, BACH)  # tenor-bass 24 semitones, limit 19
    assert count(r, "spacing") == 1
    assert r.severity["spacing"] == pytest.approx(1 + 5 / 12)

    # exactly at the limits is fine: S-A 12, A-T 12, T-B 19
    assert count(check([sv(79, 67, 55, 36)], C_MAJOR, BACH), "spacing") == 0


def test_leading_tone_resolution():
    unresolved = check([sv(71, 64, 55, 48), sv(69, 64, 55, 48)], C_MAJOR, BACH)  # B4 -> A4
    assert count(unresolved, "leading_tone_resolution") == 1
    assert unresolved.severity["leading_tone_resolution"] == pytest.approx(1.0)

    resolved = check([sv(71, 64, 55, 48), sv(72, 64, 55, 48)], C_MAJOR, BACH)
    assert count(resolved, "leading_tone_resolution") == 0

    inner = check([sv(72, 71, 55, 48), sv(72, 69, 55, 48)], C_MAJOR, BACH)  # alto leading tone left unresolved
    assert inner.severity["leading_tone_resolution"] == pytest.approx(0.4)

    bass = check([sv(72, 64, 55, 47), sv(72, 64, 55, 43)], C_MAJOR, BACH)  # bass B2 -> G2
    assert bass.severity["leading_tone_resolution"] == pytest.approx(1.0)


def test_leading_tone_in_minor_is_the_raised_seventh():
    resolved = check([sv(68, 64, 57, 45), sv(69, 64, 57, 45)], A_MINOR, BACH)  # G#4 -> A4 in A minor
    assert count(resolved, "leading_tone_resolution") == 0
    natural = check([sv(67, 64, 57, 45), sv(65, 64, 57, 45)], A_MINOR, BACH)  # natural G is not a leading tone
    assert count(natural, "leading_tone_resolution") == 0
    unresolved = check([sv(68, 64, 57, 45), sv(65, 64, 57, 45)], A_MINOR, BACH)
    assert count(unresolved, "leading_tone_resolution") == 1


def test_leap_size_values_come_from_the_table():
    fifth_up = check([sv(60, 64, 55, 48), sv(67, 64, 55, 48)], C_MAJOR, BACH)
    assert fifth_up.severity["leap_size"] == pytest.approx(0.4)  # 7 semitones, soprano

    bass_octave = check([sv(72, 64, 55, 48), sv(72, 64, 55, 60)], C_MAJOR, BACH)
    assert bass_octave.severity["leap_size"] == pytest.approx(1.2 * 0.5)

    bass_ninth = check([sv(72, 64, 55, 48), sv(72, 64, 55, 62)], C_MAJOR, BACH)  # 14 semitones
    assert bass_ninth.severity["leap_size"] == pytest.approx(2.5 * 0.5)

    steps = check([sv(72, 64, 55, 48), sv(74, 65, 55, 48)], C_MAJOR, BACH)  # seconds are free
    assert steps.severity["leap_size"] == 0.0


def test_chord_label_parsing():
    expected = {
        "I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii": 6,
        "V7": 4, "ii65": 1, "I64": 0, "viio7": 6, "vii\u00b0": 6, "vii\u00f8": 6, "bVII": 6,
        "Imaj7": 0, "iv": 3, "VI": 5, "III": 2,
    }
    for label, degree in expected.items():
        assert chord_degree(label) == degree, label
    for label in ("V/V", "N6", "It6", "Ger65", "", "X"):
        assert chord_degree(label) is None, label


def test_chord_progression_prior():
    v_i = check([sv(72, 64, 55, 48, "V"), sv(72, 64, 55, 48, "I")], C_MAJOR, BACH)
    assert v_i.severity["chord_progression_prior"] == pytest.approx(1.0)  # opens on V; V->I and close on I are free
    assert count(v_i, "chord_progression_prior") == 1

    i_v_i = check([sv(72, 64, 55, 48, c) for c in ("I", "V7", "I")], C_MAJOR, BACH)
    assert i_v_i.severity["chord_progression_prior"] == pytest.approx(0.3)  # only I->V costs

    unknown = check([sv(72, 64, 55, 48, "I"), sv(72, 64, 55, 48, "N6")], C_MAJOR, BACH)
    assert unknown.severity["chord_progression_prior"] == pytest.approx(2.0)  # I->N6 and closing on N6
    assert count(unknown, "chord_progression_prior") == 2


def test_consonance_and_profiles():
    h = [sv(61, 60, 55, 48)]
    # S-A m2 1.0, S-T tritone 0.8, S-B m2 (bass table) 1.0, A-T P4 between upper voices 0.2; A-B and T-B are perfect
    r = check(h, C_MAJOR, BACH)
    assert r.severity["consonance"] == pytest.approx(3.0)
    assert count(r, "consonance") == 4

    jazz = check(h, C_MAJOR, get_preset("jazz_loose"))
    assert jazz.severity["consonance"] == pytest.approx(0.5 + 0.2 + 0.5)
    assert count(jazz, "consonance") == 3


def test_fourth_above_bass_costs_more_than_between_upper_voices():
    above_bass = check([sv(72, 67, 60, 55)], C_MAJOR, BACH)  # tenor C4 over bass G3: a fourth
    fourth_bass = [v for v in above_bass.by_rule("consonance") if v.voices == (TENOR, BASS)]
    assert fourth_bass and fourth_bass[0].severity == pytest.approx(0.7)

    # alto F4 over tenor C4 is a fourth between upper voices; every other pair is consonant
    upper_only = check([sv(72, 65, 60, 45)], C_MAJOR, BACH)
    assert count(upper_only, "consonance") == 1
    assert upper_only.by_rule("consonance")[0].voices == (ALTO, TENOR)
    assert upper_only.severity["consonance"] == pytest.approx(0.2)


def test_explanation_adds_up_and_is_sorted():
    voicings = [sv(71, 64, 55, 48, "V"), sv(69, 66, 58, 50, "vi"), sv(72, 67, 60, 48, "I")]
    weights = get_preset("strict_counterpoint")
    e = explain(voicings, C_MAJOR, weights, scale=4.0)
    report = check(voicings, C_MAJOR, weights)
    assert e.total == pytest.approx(report.weighted_total(weights))
    assert sum(r.contribution for r in e.rules) == pytest.approx(e.total)
    assert sum(r.share for r in e.rules) == pytest.approx(1.0)
    contributions = [r.contribution for r in e.rules]
    assert contributions == sorted(contributions, reverse=True)
    assert e.normalized_total == pytest.approx(e.total / 4.0)
    text = e.to_text()
    for r in e.violated():
        assert r.rule in text
    assert "leading_tone_resolution" in text  # B4 -> A4 in the soprano


def test_zero_weight_rule_is_reported_but_costs_nothing():
    weights = get_preset("jazz_loose")  # hidden_fifths weight is 0
    e = explain([sv(72, 64, 60, 55), sv(76, 64, 60, 57)], C_MAJOR, weights)
    row = next(r for r in e.rules if r.rule == "hidden_fifths")
    assert row.count == 1 and row.contribution == 0.0
    assert row not in e.violated()
    assert "weight 0" in e.to_text()


def test_weights_validation_and_presets():
    for name in ("strict_counterpoint", "bach_chorale", "jazz_loose"):
        w = get_preset(name)
        assert w.name == name
        assert max(w.normalized().as_dict().values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        StyleWeights(spacing=-1.0)
    with pytest.raises(ValueError):
        StyleWeights(consonance_profile="nope")
    with pytest.raises(KeyError):
        get_preset("nope")
