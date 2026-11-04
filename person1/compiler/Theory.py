"""
theory.py: key, scale and chord helpers. No costs and no rules in here, only
facts about music that candidates.py and rules.py both lean on.

Conventions
-----------
- Pitches are MIDI numbers; pitch classes (pc) are 0-11 with C=0, D=2, ...
- A key is (tonic_pc, mode) with mode "major" or "minor".
- Chord labels are ASCII Roman numerals. Uppercase = major, lowercase = minor,
  a trailing 'o' = diminished (so "viio", "iio"), "V7" = dominant seventh.
- Minor mode uses the HARMONIC minor for V, V7 and viio (raised 7th = leading
  tone), which is standard in four-part writing.
- A chord's `pcs` tuple is ordered (root, third, fifth[, seventh]) in ABSOLUTE
  pitch classes for the given key. Position in that tuple = chord-tone role.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Scales
# --------------------------------------------------------------------------
# semitone offset from tonic -> scale degree (1-7)
_MAJOR_DEGREES = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
# natural minor, plus the raised 7th (11) used as the leading tone
_MINOR_DEGREES = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}

PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ROLE_NAMES = ("root", "third", "fifth", "seventh")


def _check_mode(mode: str) -> None:
    if mode not in ("major", "minor"):
        raise ValueError(f"mode must be 'major' or 'minor', got {mode!r}")


def scale_degree(pc: int, tonic: int, mode: str) -> Optional[int]:
    """Scale degree 1-7 of a pitch class in the key, or None if chromatic.
    In minor, both the natural and raised 7th count as degree 7."""
    _check_mode(mode)
    table = _MAJOR_DEGREES if mode == "major" else _MINOR_DEGREES
    return table.get((pc - tonic) % 12)


def is_in_key(pc: int, tonic: int, mode: str) -> bool:
    return scale_degree(pc, tonic, mode) is not None


def leading_tone_pc(tonic: int) -> int:
    """The leading tone is a semitone below the tonic in both modes."""
    return (tonic + 11) % 12


def is_leading_tone(midi: int, tonic: int) -> bool:
    return midi % 12 == leading_tone_pc(tonic)


# --------------------------------------------------------------------------
# Chords
# --------------------------------------------------------------------------
_QUALITIES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "dom7": (0, 4, 7, 10),
}

# label -> (root offset in semitones from tonic, quality)
_MAJOR_VOCAB = {
    "I": (0, "maj"),
    "ii": (2, "min"),
    "iii": (4, "min"),
    "IV": (5, "maj"),
    "V": (7, "maj"),
    "V7": (7, "dom7"),
    "vi": (9, "min"),
    "viio": (11, "dim"),
}
_MINOR_VOCAB = {
    "i": (0, "min"),
    "iio": (2, "dim"),
    "III": (3, "maj"),
    "iv": (5, "min"),
    "V": (7, "maj"),      # harmonic minor: major triad on the 5th degree
    "V7": (7, "dom7"),
    "VI": (8, "maj"),
    "viio": (11, "dim"),  # harmonic minor: built on the raised 7th
}


@dataclass(frozen=True)
class Chord:
    name: str                      # Roman numeral label, e.g. "V7"
    pcs: Tuple[int, ...]           # absolute pitch classes (root, third, fifth[, seventh])
    quality: str                   # "maj", "min", "dim", "dom7"

    @property
    def root_pc(self) -> int:
        return self.pcs[0]

    @property
    def has_seventh(self) -> bool:
        return len(self.pcs) == 4

    def contains(self, pc: int) -> bool:
        return pc % 12 in self.pcs


@lru_cache(maxsize=None)
def chord_vocabulary(tonic: int, mode: str) -> Dict[str, Chord]:
    """All chords the system may use in this key, keyed by label."""
    _check_mode(mode)
    vocab = _MAJOR_VOCAB if mode == "major" else _MINOR_VOCAB
    out: Dict[str, Chord] = {}
    for name, (offset, quality) in vocab.items():
        pcs = tuple((tonic + offset + iv) % 12 for iv in _QUALITIES[quality])
        out[name] = Chord(name=name, pcs=pcs, quality=quality)
    return out


def get_chord(label: str, tonic: int, mode: str) -> Chord:
    vocab = chord_vocabulary(tonic, mode)
    if label not in vocab:
        raise KeyError(f"chord {label!r} is not in the {mode} vocabulary: {sorted(vocab)}")
    return vocab[label]


def chords_containing(pc: int, tonic: int, mode: str) -> List[Chord]:
    """Every vocabulary chord that contains this pitch class.
    An EMPTY list means the note fits no chord (chromatic note, or one the
    vocabulary doesn't cover): candidates.py must handle that case."""
    return [c for c in chord_vocabulary(tonic, mode).values() if c.contains(pc)]


# --------------------------------------------------------------------------
# Roles and inversions
# --------------------------------------------------------------------------
def pitch_role(midi: int, chord: Chord) -> Optional[str]:
    """'root', 'third', 'fifth' or 'seventh' for a pitch in a chord, else None."""
    pc = midi % 12
    for i, chord_pc in enumerate(chord.pcs):
        if pc == chord_pc:
            return ROLE_NAMES[i]
    return None


def inversion(bass_midi: int, chord: Chord) -> Optional[int]:
    """0 = root position, 1 = first inversion (third in bass),
    2 = second inversion (fifth in bass), 3 = third inversion (seventh in bass).
    None if the bass note is not in the chord."""
    pc = bass_midi % 12
    for i, chord_pc in enumerate(chord.pcs):
        if pc == chord_pc:
            return i
    return None


def role_counts(pitches: Tuple[int, ...], chord: Chord) -> Dict[str, int]:
    """How many voices sing each chord tone, e.g. {'root': 2, 'third': 1, ...}.
    Voices on notes outside the chord are counted under 'other'."""
    counts = {r: 0 for r in ROLE_NAMES}
    counts["other"] = 0
    for p in pitches:
        role = pitch_role(p, chord)
        counts[role if role else "other"] += 1
    return counts


# --------------------------------------------------------------------------
# Intervals and names
# --------------------------------------------------------------------------
def interval_class(a: int, b: int) -> int:
    """Interval between two MIDI notes reduced to 0-11 semitones
    (octave and direction ignored). 0 = unison/octave, 7 = fifth, etc."""
    return abs(a - b) % 12


def midi_to_name(midi: int) -> str:
    return f"{PC_NAMES[midi % 12]}{midi // 12 - 1}"


# --------------------------------------------------------------------------
# Self-check:  python theory.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    D = 2

    # Scales
    assert scale_degree(2, D, "major") == 1       # D is degree 1 in D major
    assert scale_degree(6, D, "major") == 3       # F# is degree 3
    assert scale_degree(3, D, "major") is None    # D# is chromatic
    assert leading_tone_pc(D) == 1                # C#
    assert scale_degree(0, D, "minor") == 7       # C natural: natural 7th
    assert scale_degree(1, D, "minor") == 7       # C#: raised 7th

    # Chords in D major
    vocab = chord_vocabulary(D, "major")
    assert vocab["I"].pcs == (2, 6, 9)            # D F# A
    assert vocab["V"].pcs == (9, 1, 4)            # A C# E
    assert vocab["V7"].pcs == (9, 1, 4, 7)        # A C# E G
    assert vocab["viio"].pcs == (1, 4, 7)         # C# E G

    # Chords in D minor (harmonic)
    minor = chord_vocabulary(D, "minor")
    assert minor["i"].pcs == (2, 5, 9)            # D F A
    assert minor["V"].pcs == (9, 1, 4)            # A C# E (raised 7th)
    assert minor["iio"].pcs == (4, 7, 10)         # E G Bb

    # Which chords contain a given note (matches Ode to Joy's soprano notes)
    assert {c.name for c in chords_containing(6, D, "major")} == {"I", "iii", "vi"}   # F#
    assert {c.name for c in chords_containing(9, D, "major")} == {"I", "iii", "V", "V7"}  # A
    assert chords_containing(3, D, "major") == []                                     # D#

    # Roles and inversions
    I = vocab["I"]
    assert pitch_role(66, I) == "third"           # F#4 in D major tonic chord
    assert inversion(50, I) == 0                  # D3 in bass: root position
    assert inversion(54, I) == 1                  # F#3 in bass: first inversion
    assert inversion(57, I) == 2                  # A3 in bass: second inversion (6/4)
    assert role_counts((50, 57, 62, 66), I)["root"] == 2   # D3 A3 D4 F#4: doubled root

    # Intervals and names
    assert interval_class(60, 67) == 7 and interval_class(67, 60) == 7
    assert interval_class(60, 72) == 0
    assert midi_to_name(66) == "F#4" and midi_to_name(60) == "C4"

    print("theory.py: all checks passed")