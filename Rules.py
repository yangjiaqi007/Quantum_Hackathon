"""
rules.py: the rules from "Evolving Four-Part Harmony Using Genetic Algorithms",
as small classes. 11 of the paper's 15 rules enter the Hamiltonian.

Each rule returns a VIOLATION AMOUNT (0 = fine, 1 = one violation, ...), NOT a
weighted cost. weights.py applies the style weights; compile.py builds h and W.

    unary(v, t, ctx)          cost for voicing v at slot t           -> goes in h
    pairwise(a, b, t, ctx)    cost of going a (slot t) -> b (slot t+1) -> goes in W

Voice order everywhere: pitches = (bass, tenor, alto, soprano).
Most melodic motion rules look at the three free voices (bass, tenor, alto).
The project also scores leading-tone resolution across all four voices and
spacing between adjacent voices.

WHICH PAPER RULES ARE IN, AND WHICH ARE NOT
-------------------------------------------
In:      melodic 1, 2, 3, 5, 6*, 7, 8, 10*     harmonic 1, 2, 3
         (* = pairwise SURROGATE of a rule that needs 3+ slots; exact = False)
Dropped: melodic 4  Part Range     implied by melodic 8 (see build_context)
         melodic 9  Repeating Pitch   its surrogate punishes keeping a common
                                      tone, which is GOOD voice leading
         rhythm 1, 2 (variation, note repetition)   the optimizer does not
                                      choose rhythm, so these are constants;
                                      reported by rhythm_diagnostics() instead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from Schema import ALTO, BASS, SOPRANO, TENOR, Melody, Voicing
from Melodies import final_run_start, slot_sopranos

TONICS = ("I", "i")
FREE_VOICES = (BASS, TENOR, ALTO)            # soprano is fixed, so excluded

# Fig. 3 of the paper (read from a small image; double-check against the paper).
# MIDI (low, high): Soprano C4-A5, Alto G3-D5, Tenor C3-G4, Bass E2-C4.
DEFAULT_RANGES: Dict[int, Tuple[int, int]] = {
    BASS: (40, 60),
    TENOR: (48, 67),
    ALTO: (55, 74),
    SOPRANO: (60, 81),
}

PART_RANGE_LIMIT = 21     # a 13th (octave + fifth), in semitones
LEAP_HEIGHT_LIMIT = 14    # a 9th (octave + second)
MAJOR_SIXTH = 9
MAX_UPPER_SPACING = 12    # adjacent upper voices: at most an octave
MAX_BASS_GAP = 19         # bass to tenor: at most an octave and a fifth
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


def build_context(melody: Melody, ranges: Dict[int, Tuple[int, int]] = None) -> RuleContext:
    ranges = dict(ranges or DEFAULT_RANGES)
    sop = slot_sopranos(melody)
    # Paper rule "Part Range" (<= 13th) is implied when every range spans <= 21 semitones.
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

    def unary(self, v: Voicing, t: int, ctx: RuleContext) -> float:
        return 0.0

    def pairwise(self, a: Voicing, b: Voicing, t: int, ctx: RuleContext) -> float:
        return 0.0


def _motions(a: Voicing, b: Voicing) -> List[int]:
    """Absolute pitch change (semitones) of each free voice between two slots."""
    return [abs(b.pitches[i] - a.pitches[i]) for i in FREE_VOICES]


# --------------------------------------------------------------------------
# Melodic rules
# --------------------------------------------------------------------------
class OpeningChord(Rule):                      # melodic 1
    name, paper_rule = "OpeningChord", "melodic 1"

    def unary(self, v, t, ctx):
        return 1.0 if t == 0 and v.chord not in TONICS else 0.0


class ClosingChord(Rule):                      # melodic 2
    """Applies to every slot of the final run (a held final note covers several slots)."""
    name, paper_rule = "ClosingChord", "melodic 2"

    def unary(self, v, t, ctx):
        return 1.0 if t >= ctx.final_run_start and v.chord not in TONICS else 0.0


class FinalCadence(Rule):                      # melodic 3
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


class LeapHeight(Rule):                        # melodic 5
    name, paper_rule = "LeapHeight", "melodic 5"

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d > LEAP_HEIGHT_LIMIT))


class LeapResolution(Rule):                    # melodic 6 (SURROGATE)
    """Paper: leaps > major sixth must resolve by step in the opposite direction
    (needs 3 slots). Surrogate: penalize any leap > major sixth."""
    name, paper_rule, exact = "LeapResolution", "melodic 6", False

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d > MAJOR_SIXTH))


class VoiceCrossing(Rule):                     # melodic 7
    """Unary: a lower voice above a higher voice in the same chord.
    Pairwise: overlap (a voice moving past where its neighbour just was).
    Candidates are filtered for in-chord crossing, so the unary part is 0 for
    them; it is kept so the independent checker can verify it."""
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


class VoiceRange(Rule):                        # melodic 8
    """Also a hard filter in candidates.py; kept here for the checker."""
    name, paper_rule = "VoiceRange", "melodic 8"

    def unary(self, v, t, ctx):
        return float(sum(1 for i, p in enumerate(v.pitches)
                         if not ctx.ranges[i][0] <= p <= ctx.ranges[i][1]))


class VoiceSpacing(Rule):
    """Penalty for spacing beyond the SATB limits used by candidate generation."""
    name, paper_rule = "VoiceSpacing", "harmony spacing"

    def unary(self, v, t, ctx):
        b, tenor, alto, soprano = v.pitches
        return float(
            (soprano - alto > MAX_UPPER_SPACING)
            + (alto - tenor > MAX_UPPER_SPACING)
            + (tenor - b > MAX_BASS_GAP)
        )


class LeadingToneResolution(Rule):
    """Each leading tone must resolve upward by semitone to the tonic."""
    name, paper_rule = "LeadingToneResolution", "tonal voice leading"

    def pairwise(self, a, b, t, ctx):
        return float(sum(
            pitch % 12 == (ctx.tonic + 11) % 12 and next_pitch - pitch != 1
            for pitch, next_pitch in zip(a.pitches, b.pitches)
        ))


class StepwiseMotion(Rule):                    # melodic 10 (SURROGATE)
    """Paper: at least half of each part's intervals are steps (whole-part count).
    Surrogate: penalize each leap (>= 3 semitones); steps and repeats cost nothing."""
    name, paper_rule, exact = "StepwiseMotion", "melodic 10", False

    def pairwise(self, a, b, t, ctx):
        return float(sum(1 for d in _motions(a, b) if d >= 3))


# --------------------------------------------------------------------------
# Harmonic rules
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
# Rhythm rules: diagnostics only (constant for every harmonization)
# --------------------------------------------------------------------------
def rhythm_diagnostics(melody: Melody) -> Dict[str, object]:
    """Report the paper's two rhythm rules for the user's melody."""
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
    OpeningChord(), ClosingChord(), FinalCadence(), LeapHeight(), LeapResolution(),
    VoiceCrossing(), VoiceRange(), VoiceSpacing(), LeadingToneResolution(),
    StepwiseMotion(), ParallelOctaves(), ParallelFifths(), VerticalConsonance(),
]
RULES_BY_NAME: Dict[str, Rule] = {r.name: r for r in ALL_RULES}
assert len(ALL_RULES) == 13


# --------------------------------------------------------------------------
# Self-check:  python rules.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ctx = RuleContext(tonic=2, mode="major", n_slots=4, soprano=[66, 66, 64, 62],
                      ranges=DEFAULT_RANGES, final_run_start=3)
    assert final_run_start([66, 64, 64, 62, 62]) == 3
    assert final_run_start([66, 66, 64, 62]) == 3

    I = Voicing((50, 57, 62, 66), "I")            # D3 A3 D4 F#4
    ii = Voicing((52, 59, 62, 66), "ii")          # bass D->E, tenor A->B: fifth to fifth
    assert ParallelFifths().pairwise(I, ii, 0, ctx) == 1
    ii2 = Voicing((52, 59, 64, 66), "ii")         # bass D3->E3, alto D4->E4: octave to octave
    assert ParallelOctaves().pairwise(I, ii2, 0, ctx) == 1

    assert VoiceCrossing().unary(Voicing((50, 57, 55, 66), "I"), 0, ctx) == 1  # tenor above alto
    assert VoiceRange().unary(Voicing((30, 57, 62, 66), "I"), 0, ctx) == 1     # bass too low

    assert OpeningChord().unary(ii, 0, ctx) == 1
    assert OpeningChord().unary(I, 0, ctx) == 0
    assert ClosingChord().unary(ii, 3, ctx) == 1

    V = Voicing((45, 57, 61, 64), "V")             # A2 A3 C#4 E4
    I_end = Voicing((50, 57, 62, 62), "I")         # D3 A3 D4 D4
    assert FinalCadence().pairwise(V, I_end, 2, ctx) == 0
    assert FinalCadence().pairwise(Voicing((45, 57, 61, 64), "IV"), I_end, 2, ctx) == 1
    assert FinalCadence().pairwise(V, I_end, 1, ctx) == 0   # wrong transition: ignored

    big = Voicing((38, 57, 62, 66), "I")           # bass leaps 12 semitones down
    assert LeapHeight().pairwise(I, big, 0, ctx) == 0       # 12 <= 14
    assert LeapResolution().pairwise(I, big, 0, ctx) == 1   # 12 > 9 (surrogate)
    assert StepwiseMotion().pairwise(I, big, 0, ctx) == 1

    assert VerticalConsonance().unary(I, 0, ctx) == 1   # tenor A3 - alto D4 is a fourth
    # D3 G3 D4 F#4: bass-tenor fourth (D-G) and tenor-soprano major seventh (G-F#)
    assert VerticalConsonance().unary(Voicing((50, 55, 62, 66), "I"), 0, ctx) == 2

    from Melodies import ode_to_joy
    c = build_context(ode_to_joy())
    assert c.n_slots == 32 and c.final_run_start == 30 and c.part_range_implied
    print(rhythm_diagnostics(ode_to_joy()))
    print("rules.py: all checks passed;", len(ALL_RULES), "rules in the Hamiltonian")
