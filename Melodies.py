"""
melodies.py: every way of getting a melody INTO the pipeline.

Rule: whatever the source (preset, typed note names, MIDI upload later), the
result is a `Melody` from schema.py. compile() never needs to know where it came
from.

    presets ──────┐
    note text ────┼──→  Melody  ──→  compile()  ──→  CompiledProblem
    MIDI upload ──┘     (MIDI upload: TODO)

Also provides slot_sopranos(): which soprano pitch sounds at each chord slot.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Tuple

# Adjust this import to match where schema.py lives in your repo
# (e.g. `from schema import ...` if it's at the top level).
from Schema import Melody, Note

# --------------------------------------------------------------------------
# Pitch helpers
# --------------------------------------------------------------------------
_NATURALS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]*)(-?\d+)$")
_TONIC_RE = re.compile(r"^([A-Ga-g])([#b]*)$")


def note_to_midi(name: str) -> int:
    """'C4' -> 60, 'F#4' -> 66, 'Bb3' -> 58. Middle C = C4 = 60."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"cannot parse note name {name!r} (expected e.g. 'F#4')")
    letter, accidentals, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    shift = accidentals.count("#") - accidentals.count("b")
    return 12 * (octave + 1) + _NATURALS[letter] + shift


def tonic_to_pc(name: str) -> int:
    """'D' -> 2, 'F#' -> 6, 'Bb' -> 10."""
    m = _TONIC_RE.match(name.strip())
    if not m:
        raise ValueError(f"cannot parse key {name!r} (expected e.g. 'D', 'F#', 'Bb')")
    shift = m.group(2).count("#") - m.group(2).count("b")
    return (_NATURALS[m.group(1).upper()] + shift) % 12


# --------------------------------------------------------------------------
# Note-name text input
# --------------------------------------------------------------------------
def melody_from_text(
    text: str,
    key: str,
    mode: str = "major",
    title: str = "User melody",
    time_signature: Tuple[int, int] = (4, 4),
    harmonic_rhythm: float = 1.0,
) -> Melody:
    """
    Parse 'pitch:beats' tokens separated by spaces. Bar lines '|' are ignored.
    Beats default to 1 if omitted. Example:
        "F#4 F#4 G4 A4 | A4 G4 F#4 E4"
        "F#4:1.5 E4:0.5 D4:2"
    Rests are not supported yet.
    """
    notes: List[Note] = []
    for token in text.replace("|", " ").split():
        if ":" in token:
            pitch_s, dur_s = token.split(":", 1)
            try:
                dur = float(dur_s)
            except ValueError:
                raise ValueError(f"bad duration in token {token!r}")
        else:
            pitch_s, dur = token, 1.0
        notes.append(Note(midi=note_to_midi(pitch_s), duration=dur))
    mel = Melody(
        title=title,
        tonic=tonic_to_pc(key),
        mode=mode,
        time_signature=time_signature,
        notes=notes,
        harmonic_rhythm=harmonic_rhythm,
    )
    validate_melody(mel)
    return mel


# --------------------------------------------------------------------------
# Slotting: melody notes -> one soprano pitch per chord slot
# --------------------------------------------------------------------------
def slot_sopranos(melody: Melody) -> List[int]:
    """
    Soprano pitch sounding at the START of each chord slot.
    A slot is `harmonic_rhythm` beats long. If a note is held across several
    slots, those slots share its pitch. If several short notes fall inside one
    slot, the one sounding at the slot's start is used (passing tones are
    ignored for now).
    """
    eps = 1e-9
    starts, t = [], 0.0
    for n in melody.notes:
        starts.append(t)
        t += n.duration
    total = t
    n_slots = max(1, math.ceil(total / melody.harmonic_rhythm - eps))

    out: List[int] = []
    idx = 0
    for s in range(n_slots):
        onset = s * melody.harmonic_rhythm
        while idx + 1 < len(melody.notes) and starts[idx + 1] <= onset + eps:
            idx += 1
        out.append(melody.notes[idx].midi)
    return out

def final_run_start(soprano):
    """First slot of the trailing run of identical soprano pitches.
    The final cadence is the transition INTO this slot."""
    i = len(soprano) - 1
    while i > 0 and soprano[i - 1] == soprano[-1]:
        i -= 1
    return i

# --------------------------------------------------------------------------
# Validation (grow this as you meet bad user input)
# --------------------------------------------------------------------------
def validate_melody(
    melody: Melody,
    max_slots: int = 64,
    soprano_range: Tuple[int, int] = (60, 81),   # C4..A5
) -> None:
    """Raise ValueError with a human-readable message if the melody is unusable."""
    if not melody.notes:
        raise ValueError("melody has no notes")
    for i, n in enumerate(melody.notes):
        if n.duration <= 0:
            raise ValueError(f"note {i + 1} has non-positive duration")
        lo, hi = soprano_range
        if not lo <= n.midi <= hi:
            raise ValueError(
                f"note {i + 1} (MIDI {n.midi}) is outside the soprano range {lo}-{hi}"
            )
    if melody.harmonic_rhythm <= 0:
        raise ValueError("harmonic_rhythm must be positive")
    n_slots = len(slot_sopranos(melody))
    if n_slots > max_slots:
        raise ValueError(f"melody has {n_slots} chord slots; maximum is {max_slots}")
    # TODO: key sanity check, e.g. warn if many notes fall outside the stated key.


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
def ode_to_joy() -> Melody:
    """
    Ode to Joy (Beethoven, Symphony No. 9, finale), first 8 bars, D major, 4/4.
    The melody is public domain. CHECK against a real score before relying on it.
    """
    phrase_a = (
        "F#4 F#4 G4 A4 | A4 G4 F#4 E4 | D4 D4 E4 F#4 | F#4:1.5 E4:0.5 E4:2"
    )
    phrase_b = (
        "F#4 F#4 G4 A4 | A4 G4 F#4 E4 | D4 D4 E4 F#4 | E4:1.5 D4:0.5 D4:2"
    )
    return melody_from_text(
        f"{phrase_a} | {phrase_b}",
        key="D",
        mode="major",
        title="Ode to Joy (first 8 bars)",
    )


def o_canada() -> Melody:
    """
    TODO (Person 1): enter O Canada from a VERIFIED public-domain score
    (melody by Calixa Lavallee, 1880), then delete the raise below.

    Use melody_from_text(...) like ode_to_joy(). Pick a key where the melody
    sits inside the soprano range (60-81), and note any accidentals and the
    pickup (upbeat) at the start. Decide how to handle the pickup: either
    start the melody at the first full bar or add a short first note.
    """
    raise NotImplementedError("Fill in O Canada from a verified score")


PRESETS: Dict[str, Callable[[], Melody]] = {
    "ode_to_joy": ode_to_joy,
    "o_canada": o_canada,
}


def get_preset(name: str) -> Melody:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")


# --------------------------------------------------------------------------
# Quick self-check:  python -m compiler.melodies   (adjust to your layout)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    m = ode_to_joy()
    print(m.title, "| key pc:", m.tonic, m.mode, "| notes:", len(m.notes))
    s = slot_sopranos(m)
    print("slots:", len(s))
    print("soprano per slot:", s)
    assert len(s) == 32, "8 bars x 4 beats should give 32 slots"
    assert s[0] == note_to_midi("F#4")
    assert note_to_midi("C4") == 60 and note_to_midi("Bb3") == 58
    print("ok")