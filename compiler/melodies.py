"""Public melody input API; implementation is shared with legacy root imports."""

from Melodies import (
    get_preset,
    melody_from_midi,
    melody_from_text,
    note_to_midi,
    o_canada,
    ode_to_joy,
    slot_durations,
    slot_sopranos,
    tonic_to_pc,
    validate_melody,
)

__all__ = [
    "get_preset", "melody_from_midi", "melody_from_text", "note_to_midi",
    "o_canada", "ode_to_joy", "slot_durations", "slot_sopranos",
    "tonic_to_pc", "validate_melody",
]
