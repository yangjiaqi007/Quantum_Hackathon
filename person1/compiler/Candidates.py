"""
candidates.py: for every chord slot, build the short list of K voicings the
quantum solver will choose between.

Pipeline per slot
-----------------
1. Which chords can carry the soprano note?   (theory.chords_containing)
2. Enumerate every alto/tenor/bass assignment for those chords.
   HARD filters (a voicing failing any of these is never a candidate):
     - alto, tenor, bass inside their voice ranges (Fig. 3)
     - chord membership of the three lower voices
     - no crossing: bass < tenor < alto < soprano (unisons allowed only as a fallback)
     - spacing: soprano-alto and alto-tenor <= an octave, bass-tenor <= max_bass_gap
     - the chord is recognisable: root + third present (plus the fifth for
       diminished chords, plus the seventh for V7)
3. Score each voicing:  weighted_unary (the real Hamiltonian terms)
                      + prior_penalty  (PRUNING ONLY, see below)
4. Prune to K, keeping DIFFERENT CHORDS first (round robin), and within one chord
   keeping voicings that differ from each other.

prior_penalty
-------------
Doubling and inversion preferences are pruning-only. Spacing is a hard candidate
filter and is also represented by VoiceSpacing for explicit scoring and checking.

Opening and closing slots keep only tonic chords (those rules are weighted, and
no non-tonic candidate could ever win there).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from Schema import ALTO, BASS, SOPRANO, TENOR, Melody, Voicing
from .Theory import (Chord, chord_vocabulary, chords_containing, inversion,
                     leading_tone_pc, midi_to_name)
from .Rules import (DEFAULT_RANGES, MAX_BASS_GAP, MAX_UPPER_SPACING, TONICS,
                    RuleContext, build_context)
from .Weights import get_weights, weighted_unary

DEFAULT_K = 4
# --------------------------------------------------------------------------
# Step 2: enumeration with hard filters
# --------------------------------------------------------------------------
def _required_pcs(chord: Chord) -> set:
    """Chord tones that must be sounded for the chord to be recognisable."""
    if chord.quality == "dim":
        idx = (0, 1, 2)
    elif chord.quality == "dom7":
        idx = (0, 1, 3)            # root, third, seventh (fifth may be omitted)
    else:
        idx = (0, 1)               # root, third (fifth may be omitted)
    return {chord.pcs[i] for i in idx}


def enumerate_voicings(soprano: int, chord: Chord, ctx: RuleContext,
                       strict: bool = True,
                       max_upper_spacing: int = MAX_UPPER_SPACING,
                       max_bass_gap: int = MAX_BASS_GAP) -> List[Voicing]:
    """Every hard-filter-passing voicing of `chord` under the given soprano pitch.
    strict=True forbids unisons between voices."""
    chord_pcs = set(chord.pcs)
    required = _required_pcs(chord)

    def options(voice: int) -> List[int]:
        lo, hi = ctx.ranges[voice]
        return [p for p in range(lo, hi + 1) if p % 12 in chord_pcs]

    out: List[Voicing] = []
    for a in options(ALTO):
        if a > soprano or (strict and a == soprano) or soprano - a > max_upper_spacing:
            continue
        for tn in options(TENOR):
            if tn > a or (strict and tn == a) or a - tn > max_upper_spacing:
                continue
            for b in options(BASS):
                if b > tn or (strict and b == tn) or tn - b > max_bass_gap:
                    continue
                covered = {b % 12, tn % 12, a % 12, soprano % 12}
                if not required <= covered:
                    continue
                out.append(Voicing((b, tn, a, soprano), chord.name))
    return out


# --------------------------------------------------------------------------
# Step 3: scoring (pruning prior is NOT part of the Hamiltonian)
# --------------------------------------------------------------------------
def prior_penalty(v: Voicing, chord: Chord, t: int, ctx: RuleContext) -> float:
    counts = Counter(p % 12 for p in v.pitches)
    pen = 0.0

    if counts.get(leading_tone_pc(ctx.tonic), 0) > 1:        # doubled leading tone
        pen += 5.0
    if chord.has_seventh and counts.get(chord.pcs[3], 0) > 1:  # doubled seventh
        pen += 5.0

    pen += {0: 0.0, 1: 0.5, 2: 3.0, 3: 1.0}.get(inversion(v.bass, chord), 0.0)

    doubled = [i for i, pc in enumerate(chord.pcs) if counts.get(pc, 0) > 1]
    fifth_missing = counts.get(chord.pcs[2], 0) == 0
    if chord.quality == "dim":
        if doubled and doubled != [1]:
            pen += 1.5                                        # prefer doubling the third
    elif chord.quality == "dom7":
        if fifth_missing:
            pen += 1.0                                        # root doubled, fifth omitted
    else:
        if fifth_missing:
            pen += 1.5
        elif doubled == [1]:
            pen += 2.0                                        # doubled third
        elif doubled == [2]:
            pen += 1.0                                        # doubled fifth

    pen += 0.02 * (v.pitches[SOPRANO] - v.pitches[TENOR])    # prefer closer spacing

    c = ctx.final_run_start                                   # steer the cadence slot
    if c >= 1 and t == c - 1:
        if v.chord not in ("V", "V7"):
            pen += 4.0
        if v.bass % 12 != (ctx.tonic + 7) % 12:
            pen += 2.0
    return pen


# --------------------------------------------------------------------------
# Step 4: pruning that keeps different chords
# --------------------------------------------------------------------------
def _hamming(u: Voicing, v: Voicing) -> int:
    return sum(1 for i in (BASS, TENOR, ALTO) if u.pitches[i] != v.pitches[i])


def prune(scored: Sequence[Tuple[float, Voicing]], K: int) -> List[Voicing]:
    """Round robin over chords (best-scoring chord first): one voicing per chord,
    then a second per chord, and so on, until K are chosen. Extra voicings of the
    same chord must differ from those already chosen in >= 2 voices if possible."""
    by_chord: Dict[str, List[Tuple[float, Voicing]]] = {}
    for s, v in sorted(scored, key=lambda x: (x[0], x[1].pitches)):
        by_chord.setdefault(v.chord, []).append((s, v))
    chord_order = sorted(by_chord, key=lambda c: by_chord[c][0][0])

    chosen: List[Voicing] = []
    min_diff = 2
    while len(chosen) < K:
        progressed = False
        for c in chord_order:
            if len(chosen) >= K:
                break
            same = [u for u in chosen if u.chord == c]
            for _, v in by_chord[c]:
                if v in chosen:
                    continue
                if all(_hamming(u, v) >= min_diff for u in same):
                    chosen.append(v)
                    progressed = True
                    break
        if not progressed:
            if min_diff > 1:
                min_diff -= 1
            else:
                break
    return chosen


# --------------------------------------------------------------------------
# One slot, and the whole melody
# --------------------------------------------------------------------------
def _chords_for(soprano: int, t: int, ctx: RuleContext, on_unfit: str) -> List[Chord]:
    chords = chords_containing(soprano % 12, ctx.tonic, ctx.mode)
    if chords:
        return chords
    if on_unfit == "nearest":
        for off in (-1, 1, -2, 2):                  # treat it as a neighbouring note
            chords = chords_containing((soprano + off) % 12, ctx.tonic, ctx.mode)
            if chords:
                return chords
    raise ValueError(
        f"slot {t}: soprano note {midi_to_name(soprano)} fits no chord in this key's "
        f"vocabulary. Check the key, or use on_unfit='nearest' to harmonize it as a "
        f"non-chord tone."
    )


def generate_slot_candidates(soprano: int, t: int, ctx: RuleContext,
                             weights: Dict[str, float], K: int = DEFAULT_K,
                             on_unfit: str = "nearest") -> List[Voicing]:
    chords = _chords_for(soprano, t, ctx, on_unfit)
    chord_by_name = {c.name: c for c in chords}

    scored: List[Tuple[float, Voicing]] = []
    for strict in (True, False):                    # relax the no-unison rule only if needed
        for chord in chords:
            for v in enumerate_voicings(soprano, chord, ctx, strict=strict):
                score = (weighted_unary(v, t, ctx, weights)
                         + prior_penalty(v, chord_by_name[v.chord], t, ctx))
                scored.append((score, v))
        if scored:
            break
    if not scored:
        raise ValueError(f"slot {t}: no voicing of soprano {midi_to_name(soprano)} "
                         f"satisfies the range and spacing limits")

    needs_tonic = ((t == 0 and weights.get("OpeningChord", 0) > 0)
                   or (t >= ctx.final_run_start and weights.get("ClosingChord", 0) > 0))
    if needs_tonic:
        tonic_only = [x for x in scored if x[1].chord in TONICS]
        scored = tonic_only or scored

    return prune(scored, K)


def generate_candidates(melody: Melody, style: str = "chorale", K: int = DEFAULT_K,
                        ctx: RuleContext = None, weights: Dict[str, float] = None,
                        on_unfit: str = "nearest") -> List[List[Voicing]]:
    """candidates[t][k] for the whole melody."""
    ctx = ctx or build_context(melody)
    weights = weights or get_weights(style)
    return [generate_slot_candidates(s, t, ctx, weights, K, on_unfit)
            for t, s in enumerate(ctx.soprano)]


# --------------------------------------------------------------------------
# Verification and display
# --------------------------------------------------------------------------
def check_hard_filters(candidates: List[List[Voicing]], ctx: RuleContext) -> None:
    """Raise AssertionError if any candidate breaks a hard filter."""
    vocab = chord_vocabulary(ctx.tonic, ctx.mode)
    for t, slot in enumerate(candidates):
        assert slot, f"slot {t} has no candidates"
        for v in slot:
            b, tn, a, s = v.pitches
            assert s == ctx.soprano[t], f"slot {t}: wrong soprano"
            assert b <= tn <= a <= s, f"slot {t}: voices crossed {v.pitches}"
            for i, p in enumerate(v.pitches):
                lo, hi = ctx.ranges[i]
                assert lo <= p <= hi, f"slot {t}: voice {i} out of range"
            assert v.chord in vocab, f"slot {t}: unknown chord {v.chord}"
            for p in (b, tn, a):
                assert vocab[v.chord].contains(p), f"slot {t}: {p} not in {v.chord}"
            assert s - a <= MAX_UPPER_SPACING and a - tn <= MAX_UPPER_SPACING


def summarize(candidates: List[List[Voicing]], ctx: RuleContext) -> str:
    lines = []
    for t, slot in enumerate(candidates):
        chords = " ".join(v.chord for v in slot)
        lines.append(f"slot {t:2d}  S={midi_to_name(ctx.soprano[t]):4s}  K={len(slot)}  {chords}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-check:  python candidates.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .Melodies import melody_from_text, ode_to_joy

    m = ode_to_joy()
    ctx = build_context(m)
    for style in ("strict", "chorale", "jazz_loose"):
        cands = generate_candidates(m, style=style)
        check_hard_filters(cands, ctx)
        assert len(cands) == 32
        assert all(1 <= len(slot) <= DEFAULT_K for slot in cands)
        assert all(v.chord in TONICS for v in cands[0])                       # opening
        assert all(v.chord in TONICS for v in cands[-1] + cands[-2])          # closing run
        assert any(v.chord in ("V", "V7") for v in cands[ctx.final_run_start - 1])  # cadence
        n_qubits = sum(len(s) for s in cands)
        print(f"{style:11s} slots=32  candidates={n_qubits}  "
              f"one-hot qubits for a 4-slot window ~ {4 * DEFAULT_K}")

    print()
    print(summarize(generate_candidates(m, style="chorale"), ctx))

    # A note outside the key: error by default, or harmonized as a non-chord tone
    odd = melody_from_text("F#4 D#4 D4", key="D")
    odd_ctx = build_context(odd)
    try:
        generate_candidates(odd, ctx=odd_ctx, on_unfit="error")
        raise AssertionError("should have rejected the chromatic note")
    except ValueError as e:
        print("\nchromatic note, on_unfit='error':", e)
    fallback = generate_candidates(odd, ctx=odd_ctx, on_unfit="nearest")
    assert all(len(s) >= 1 for s in fallback)
    print("chromatic note, on_unfit='nearest': ok")
    print("candidates.py: all checks passed")
