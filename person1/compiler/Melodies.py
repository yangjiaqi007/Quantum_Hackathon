"""
melodies.py: every way of getting a melody INTO the pipeline.

Rule: whatever the source (preset, typed note names, MIDI upload), the
result is a `Melody` from schema.py. compile() never needs to know where it came
from.

    presets ──────┐
    note text ────┼──→  Melody  ──→  compile()  ──→  CompiledProblem
    MIDI upload ──┘

Also provides slot_sopranos(): which soprano pitch sounds at each chord slot.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
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


def melody_from_midi(path, key=None, mode=None, title=None,
                     time_signature=None, harmonic_rhythm=1.0,
                     track=None) -> Melody:
    """Read a monophonic melody track from a Standard MIDI File.

    Requires ``mido``. If ``track`` is omitted, uses the track with
    the most complete notes. Durations are quarter-note beats. Since Melody
    has no rest event, gaps are treated as sustained preceding notes.
    """
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("MIDI import requires the 'mido' package") from exc

    midi = mido.MidiFile(str(path))
    if not midi.tracks:
        raise ValueError("MIDI file contains no tracks")
    if track is not None and not 0 <= track < len(midi.tracks):
        raise ValueError(f"track index must be between 0 and {len(midi.tracks) - 1}")

    def read_track(messages):
        tick, active, events, signature, meter = 0, {}, [], None, None
        for msg in messages:
            tick += msg.time
            if msg.type == "key_signature":
                signature = msg.key
            elif msg.type == "time_signature":
                meter = (msg.numerator, msg.denominator)
            if msg.type not in ("note_on", "note_off"):
                continue
            k = (getattr(msg, "channel", 0), msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(k, []).append(tick)
            elif active.get(k):
                start = active[k].pop(0)
                if tick > start:
                    events.append((start, tick, msg.note))
        if any(starts for starts in active.values()):
            raise ValueError("MIDI track has note-on events without matching note-off")
        return events, signature, meter

    tracks = [read_track(t) for t in midi.tracks]
    if track is None:
        candidates = [
            (i, [event for event in e if 60 <= event[2] <= 81], s, meter)
            for i, (e, s, meter) in enumerate(tracks)
        ]
        candidates = [item for item in candidates if item[1]]
        if not candidates:
            raise ValueError("MIDI file contains no complete notes in the soprano range")
        _, events, signature, meter = max(
            candidates,
            key=lambda x: (len(x[1]), sum(ev[2] for ev in x[1]) / len(x[1])),
        )
    else:
        events, signature, meter = tracks[track]
        if not events:
            raise ValueError(f"MIDI track {track} contains no complete notes")
    events.sort()
    for previous, current in zip(events, events[1:]):
        if current[0] < previous[1]:
            raise ValueError("selected MIDI track is polyphonic; choose a monophonic melody track")
        if current[0] == previous[0]:
            raise ValueError("selected MIDI track has simultaneous melody notes")

    key_names = {"C": "C", "G": "G", "D": "D", "A": "A", "E": "E",
                 "B": "B", "F#": "F#", "C#": "C#", "F": "F",
                 "Bb": "Bb", "Eb": "Eb", "Ab": "Ab", "Db": "Db",
                 "Gb": "Gb", "Cb": "Cb"}
    if key is None:
        source_signature = signature or next(
            (s for _, s, _ in tracks if s is not None), "C"
        )
        if source_signature.endswith("m"):
            key = source_signature[:-1]
            mode = mode or "minor"
        else:
            key = key_names.get(source_signature, "C")
            mode = mode or "major"
    else:
        mode = mode or "major"
    if time_signature is None:
        time_signature = meter or next(
            (m for _, _, m in tracks if m is not None), (4, 4)
        )
    notes = []
    for i, (start, end, pitch) in enumerate(events):
        next_start = events[i + 1][0] if i + 1 < len(events) else end
        notes.append(Note(pitch, (next_start - start) / midi.ticks_per_beat))
    melody = Melody(
        title=title or Path(path).stem.replace("_", " ").title(),
        tonic=tonic_to_pc(key), mode=mode, time_signature=time_signature,
        notes=notes, harmonic_rhythm=harmonic_rhythm,
    )
    validate_melody(melody)
    return melody


# --------------------------------------------------------------------------
# Slotting: melody notes -> one soprano pitch per chord slot
# --------------------------------------------------------------------------
def slot_sopranos(melody: Melody) -> List[int]:
    """
    Soprano pitch sounding at the START of each chord slot.
    Slots occur at the harmonic-rhythm pulse and at every melody-note boundary,
    so no slot crosses an input note. A note held across several slots repeats
    its pitch in each of those slots.
    """
    starts, total = [], 0.0
    for note in melody.notes:
        starts.append(total)
        total += note.duration
    boundaries = _slot_boundaries(melody)
    out: List[int] = []
    idx = 0
    for onset in boundaries[:-1]:
        while idx + 1 < len(melody.notes) and starts[idx + 1] <= onset + 1e-9:
            idx += 1
        out.append(melody.notes[idx].midi)
    return out


def slot_durations(melody: Melody) -> List[float]:
    """Duration of each compiler slot; slots partition notes without crossing boundaries."""
    boundaries = _slot_boundaries(melody)
    return [b - a for a, b in zip(boundaries, boundaries[1:])]


def _slot_boundaries(melody: Melody) -> List[float]:
    """Union of melody-note boundaries and harmonic-rhythm pulse boundaries."""
    if melody.harmonic_rhythm <= 0:
        raise ValueError("harmonic_rhythm must be positive")
    total = sum(note.duration for note in melody.notes)
    points = {0.0, total}
    elapsed = 0.0
    for note in melody.notes[:-1]:
        elapsed += note.duration
        points.add(elapsed)
    pulse = melody.harmonic_rhythm
    n_pulses = int(math.ceil(total / pulse - 1e-9))
    points.update(min(i * pulse, total) for i in range(1, n_pulses))
    return sorted(points)

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
    max_slots: int = 256,
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
    """O Canada in F major, based on the official Canadian Heritage score.

    The pickup begins at beat 0 because the shared Melody schema has no
    anacrusis offset field.
    """
    return melody_from_text(
        "A4:2 C5:1.5 C5:0.5 | F4:3 G4:1 | A4:1 Bb4:1 C5:1 D5:1 | G4:3 G4:1 | A4:2 Bb4:1.5 Bb4:0.5 | C5:3 D5:1 | E5:1 E5:1 D5:1 D5:1 | C5:3 G4:0.75 A4:0.25 | Bb4:1.5 A4:0.5 G4:1 A4:0.75 Bb4:0.25 | C5:1.5 Bb4:0.5 A4:1 Bb4:0.75 C5:0.25 | D5:1 C5:1 Bb4:1 A4:1 | G4:3 G4:0.75 A4:0.25 | Bb4:1.5 A4:0.5 G4:1 A4:0.75 Bb4:0.25 | C5:1.5 Bb4:0.5 A4:1 A4:1 | G4:1 C5:1 C5:0.5 Bb4:0.5 A4:0.5 Bb4:0.5 | C5:3 C5:1 | A4:2 C5:1.5 C5:0.5 | F4:3 F4:1 | Bb4:2 D5:1.5 D5:0.5 | G4:3 G4:1 | C5:2 C#5:1.5 C#5:0.5 | D5:1 Bb4:1 A4:1 G4:1 | F4:2 G4:2 | A4:3 A4:1 | C5:2 F5:1.5 F5:0.5 | D5:1 Bb4:1 A4:1 G4:1 | C5:2 E4:2 | F4:4",
        key="F", mode="major", title="O Canada",
    )


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
