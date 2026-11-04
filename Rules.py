"""
rules.py: the 15 rules from "Evolving Four-Part Harmony Using Genetic
Algorithms" as small classes.

Each rule returns a VIOLATION AMOUNT (0 = fine, 1 = one violation, ...), NOT a
weighted cost. weights.py multiplies by the style weight and normalizes, and
compile.py assembles h and W.

    unary(v, t, ctx)          cost for voicing v at slot t           -> goes in h
    pairwise(a, b, t, ctx)    cost of going a (slot t) -> b (slot t+1) -> goes in W

Voice order everywhere: pitches = (bass, tenor, alto, soprano).
The soprano is FIXED by the user's melody, so melodic rules look at the three
free voices only (bass, tenor, alto); the soprano would just add a constant.

EXACT vs SURROGATE
------------------
Rules 6, 9 and 10 involve 3+ slots or a whole-part count, so they cannot be
written exactly as pairwise terms. Each has a pairwise SURROGATE (exact = False).
The checker must count the surrogate version for the energy test and may also
report the paper-exact count to users.

NOT IN THE HAMILTONIAN
----------------------
Rule 4 (part range) is implied by rule 8 when every voice range spans <= 21
semitones (checked in build_context). Rules 14 and 15 are about rhythm, which
the optimizer does not choose, so they are diagnostics only (rhythm_diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from Schema import ALTO, BASS, SOPRANO, TENOR, Melody, Voicing
from Melodies import slot_sopranos

TONICS = ("I", "i")
FREE_VOICES = (BASS, TENOR, ALTO)            # soprano is fixed, so excluded

# TODO: replace with the ranges from Fig. 3 of the paper (not visible to me).
# These are standard SATB ranges, as MIDI (low, high).
DEFAULT_RANGES: Dict[int, Tuple[int, int]] = {
    BASS: (40, 60),       # E2 - C4
    TENOR: (48, 67),      # C3 - G4
    ALTO: (55, 74),       # G3 - D5
    SOPRANO: (60, 81),    # C4 - A5 (matches validate_melody)
}

PART_RANGE_LIMIT = 21     # a 13th (octave + fifth), in semitones
LEAP_HEIGHT_LIMIT = 14    # a 9th (octave + second)
MAJOR_SIXTH = 9
DISSONANT_CLASSES = {1, 2, 5, 10, 11}   # seconds, fourths, sevenths (paper's list)


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------
@dataclass
class RuleContext:
    tonic: int
    mode: str
    n_slots: int
    soprano: List[int]
    ranges: Dict[int, Tuple[int, int]]
    final_run_start: int           # first slot of the final run of equal soprano pitches
    part_range_implied: bool = True


def final_run_start(soprano: Sequence[int]) -> int:
    """First slot of the trailing run of identical soprano pitches.
    The final cadence is the transition INTO this slot (a held final note
    spans several beat-slots, so 'the last two slots' would be wrong)."""
    i = len(soprano) - 1
    while i > 0 and soprano[i - 1] == soprano[-1]:
        i -= 1
    return i


def build_context(melody: Melody, ranges: Dict[int, Tuple[int, int]] = None) -> RuleContext:
    ranges = dict(ranges or DEFAULT_RANGES)
    sop = slot_sopranos(melody)
    implied = all(hi - lo <= PART_RANGE_LIMIT for lo, hi in ranges.values())
    return RuleContext(
        tonic=melody.tonic,
        mode=melody.mode,
        n_slots=len(sop),
        soprano=sop,
        ranges=ranges,
        final_run_start=final_run_start(sop),
        part_range_implied=implied,
    )


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class Rule:
    name = "rule"
    paper_rule = ""            # which numbered rule of the paper
    exact = True               # False if this is a pairwise surrogate
    in_hamiltonian = True

    def unary(self, v: Voicing, t: int, ctx: RuleContext) -> float:
        return 0.0

    def pairwise(self, a: Voicing, b: Voicing, t: int, ctx: RuleContext) -> float:
        return 0.0


def _motions(a: Voicing, b: Voicing) -> List[int]:
    """Absolute pitch change (semitones) of each free voice between two slots."""
    return [abs(b.pitches[i] - a.pitches[i]) for i in FREE_VOICES]


# --------------------------------------------------------------------------
# Melodic rules (paper's 1-10)
# --------------------------------------------------------------------------
class OpeningChord(Rule):                      # rule 1
    name, paper_rule = "OpeningChord", "melodic 1"

    def unary(self, v, t, ctx):
        return 1.0 if t == 0 and v.chord not in TONICS else 0.0


class ClosingChord(Rule):                      # rule 2
    """Applies to every slot of the final run (a held final note covers several slots)."""
    name, paper_rule = "ClosingChord", "melodic 2"

    def unary(self, v, t, ctx):
        return 1.0 if t >= ctx.final_run_start and v.chord not in TONICS else 0.0


class FinalCadence(Rule):                      # rule 3
    """Transition into the final run must be V/V7 -> tonic, bass dominant -> tonic,
    tenor and alto closing by step. (Soprano excluded: fixed. The tonic arrival
    itself is charged by ClosingChord, so it is not double counted here.)"""
    name, paper_rule = "FinalCadence", "melodic 3"

    def pairwise(self, a, b, t, ctx):
        if ctx.final_run_start < 1 or t != ctx.final_run_start - 1:
            return 0.0
        v = 0.0
        if a.chord not in ("V", "V7"):
            v += 1
        if not (a.pitches[BASS] % 12 == (ctx.tonic + 7) % 12
                and b.pitches[BASS] % 12 == ctx.tonic):
            v += 1
        for i in (TENOR, ALTO):
            if abs(b.pitches[i] - a.pitches[i]) > 2:
                v += 1
        return v


class PartRange(Rule):                         # rule 4
    """No terms: implied by VoiceRange when each range spans <= 13th
    (see RuleContext.part_range_implied)."""
    name, paper_rule = "PartRange", "melodic 4"
    in_hamiltonian = False


class LeapHeight(Rule):                        # rule 5
    name, paper_rule = "LeapHeight", "melodic 5"

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d > LEAP_HEIGHT_LIMIT))


class LeapResolution(Rule):                    # rule 6 (SURROGATE)
    """Paper: leaps > major sixth must resolve by step in the opposite direction
    (needs 3 slots). Surrogate: penalize any leap > major sixth."""
    name, paper_rule, exact = "LeapResolution", "melodic 6", False

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d > MAJOR_SIXTH))


class VoiceCrossing(Rule):                     # rule 7
    """Unary: a lower voice above a higher voice in the same chord.
    Pairwise: overlap (a voice moving past where its neighbour just was)."""
    name, paper_rule = "VoiceCrossing", "melodic 7"

    def unary(self, v, t, ctx):
        p = v.pitches
        return float(sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]))

    def pairwise(self, a, b, t, ctx):
        n = 0
        for i in range(3):                       # i = lower voice, i+1 = upper voice
            if b.pitches[i] > a.pitches[i + 1] or b.pitches[i + 1] < a.pitches[i]:
                n += 1
        return float(n)


class VoiceRange(Rule):                        # rule 8
    """Also enforced as a hard filter in candidates.py; kept here for the checker."""
    name, paper_rule = "VoiceRange", "melodic 8"

    def unary(self, v, t, ctx):
        return float(sum(1 for i, p in enumerate(v.pitches)
                         if not ctx.ranges[i][0] <= p <= ctx.ranges[i][1]))


class RepeatingPitch(Rule):                    # rule 9 (SURROGATE)
    """Paper: no pitch repeated more than 3 times in a row (needs 4 slots).
    Surrogate: small penalty for repeating a pitch between adjacent slots."""
    name, paper_rule, exact = "RepeatingPitch", "melodic 9", False

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d == 0))


class StepwiseMotion(Rule):                    # rule 10 (SURROGATE)
    """Paper: at least half of each part's intervals are steps (whole-part count).
    Surrogate: penalize each leap (>= 3 semitones); steps and repeats cost nothing."""
    name, paper_rule, exact = "StepwiseMotion", "melodic 10", False

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d >= 3))


# --------------------------------------------------------------------------
# Harmonic rules (paper's 1-3)
# --------------------------------------------------------------------------
def _parallel_count(a: Voicing, b: Voicing, target_class: int) -> float:
    """Pairs of voices forming the same perfect interval (mod octave) before and
    after, where at least one of the two moved."""
    n = 0
    for i in range(4):
        for j in range(i + 1, 4):
            ica = (a.pitches[j] - a.pitches[i]) % 12
            icb = (b.pitches[j] - b.pitches[i]) % 12
            moved = (a.pitches[i], a.pitches[j]) != (b.pitches[i], b.pitches[j])
            if ica == icb == target_class and moved:
                n += 1
    return float(n)


class ParallelOctaves(Rule):                   # harmonic 1
    name, paper_rule = "ParallelOctaves", "harmonic 1"

    def pairwise(self, a, b, t, ctx):
        return _parallel_count(a, b, 0)


class ParallelFifths(Rule):                    # harmonic 2
    name, paper_rule = "ParallelFifths", "harmonic 2"

    def pairwise(self, a, b, t, ctx):
        return _parallel_count(a, b, 7)


class VerticalConsonance(Rule):                # harmonic 3
    """Counts voice pairs forming a second, fourth or seventh (the paper's list).
    Keep the weight small: V7 always has a seventh, and fourths between upper
    voices are normal in practice."""
    name, paper_rule = "VerticalConsonance", "harmonic 3"

    def unary(self, v, t, ctx):
        p = v.pitches
        return float(sum(1 for i in range(4) for j in range(i + 1, 4)
                         if abs(p[j] - p[i]) % 12 in DISSONANT_CLASSES))


# --------------------------------------------------------------------------
# Rhythm rules (paper's 1-2): diagnostics only, constant for every harmonization
# --------------------------------------------------------------------------
class RhythmicVariation(Rule):                 # rhythm 1
    name, paper_rule, in_hamiltonian = "RhythmicVariation", "rhythm 1", False


class NoteRepetition(Rule):                    # rhythm 2
    name, paper_rule, in_hamiltonian = "NoteRepetition", "rhythm 2", False


def rhythm_diagnostics(melody: Melody) -> Dict[str, object]:
    """Report rules 14 and 15 for the user's melody. These cannot change with
    the harmonization, so they are reported, not optimized."""
    wanted = {0.5: "eighth", 1.0: "quarter", 2.0: "half", 4.0: "whole"}
    present = {n.duration for n in melody.notes}
    missing = [name for d, name in wanted.items() if d not in present]

    longest_run, run = 1, 1
    for prev, cur in zip(melody.notes, melody.notes[1:]):
        run = run + 1 if cur.duration == prev.duration else 1
        longest_run = max(longest_run, run)
    return {
        "rhythmic_variation_missing_types": missing,           # empty = satisfied
        "longest_same_duration_run": longest_run,
        "note_repetition_ok": longest_run <= 3,
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
ALL_RULES: List[Rule] = [
    OpeningChord(), ClosingChord(), FinalCadence(), PartRange(), LeapHeight(),
    LeapResolution(), VoiceCrossing(), VoiceRange(), RepeatingPitch(),
    StepwiseMotion(), ParallelOctaves(), ParallelFifths(), VerticalConsonance(),
    RhythmicVariation(), NoteRepetition(),
]
RULES_BY_NAME: Dict[str, Rule] = {r.name: r for r in ALL_RULES}
assert len(ALL_RULES) == 15


# --------------------------------------------------------------------------
# Self-check:  python rules.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ctx = RuleContext(tonic=2, mode="major", n_slots=4, soprano=[66, 66, 64, 62],
                      ranges=DEFAULT_RANGES, final_run_start=3)
    assert final_run_start([66, 64, 64, 62, 62]) == 3
    assert final_run_start([66, 66, 64, 62]) == 3

    I = Voicing((50, 57, 62, 66), "I")            # D3 A3 D4 F#4
    # bass D->E and tenor A->B: fifth to fifth
    ii = Voicing((52, 59, 62, 66), "ii")
    assert ParallelFifths().pairwise(I, ii, 0, ctx) == 1
    # bass D3->E3 and alto D4->E4: octave to octave
    ii2 = Voicing((52, 59, 64, 66), "ii")
    assert ParallelOctaves().pairwise(I, ii2, 0, ctx) == 1

    # Voice crossing and range
    assert VoiceCrossing().unary(Voicing((50, 57, 55, 66), "I"), 0, ctx) == 1  # tenor above alto
    assert VoiceRange().unary(Voicing((30, 57, 62, 66), "I"), 0, ctx) == 1     # bass too low

    # Opening / closing
    assert OpeningChord().unary(Voicing((52, 59, 62, 66), "ii"), 0, ctx) == 1
    assert OpeningChord().unary(I, 0, ctx) == 0
    assert ClosingChord().unary(Voicing((52, 59, 62, 66), "ii"), 3, ctx) == 1

    # Final cadence V -> I (soprano E4 -> D4), transition t = final_run_start - 1 = 2
    V = Voicing((45, 57, 61, 64), "V")             # A2 A3 C#4 E4
    I_end = Voicing((50, 57, 62, 62), "I")         # D3 A3 D4 D4
    assert FinalCadence().pairwise(V, I_end, 2, ctx) == 0
    IV = Voicing((45, 57, 61, 64), "IV")
    assert FinalCadence().pairwise(IV, I_end, 2, ctx) == 1
    assert FinalCadence().pairwise(V, I_end, 1, ctx) == 0   # wrong transition: ignored

    # Leaps
    big = Voicing((38, 57, 62, 66), "I")           # bass leaps 12 semitones down
    assert LeapHeight().pairwise(I, big, 0, ctx) == 0       # 12 <= 14
    assert LeapResolution().pairwise(I, big, 0, ctx) == 1   # 12 > 9 (surrogate)
    assert StepwiseMotion().pairwise(I, big, 0, ctx) == 1

    # Vertical consonance: D3 + A3 + D4 + F#4 has no 2nd/4th/7th
    assert VerticalConsonance().unary(I, 0, ctx) == 0
    assert VerticalConsonance().unary(Voicing((50, 55, 62, 66), "I"), 0, ctx) == 1  # D-G fourth

    from Melodies import ode_to_joy
    print(rhythm_diagnostics(ode_to_joy()))
    print("rules.py: all checks passed;", len(ALL_RULES), "rules registered")