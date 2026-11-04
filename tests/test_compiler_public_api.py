from compiler import get_preset, slot_durations, compile_melody
from Melodies import melody_from_text
from Checker import check_energy, count_violations, viterbi
from Rules import LeadingToneResolution, VoiceSpacing, build_context
from Schema import Voicing
from Weights import get_weights
import mido
from itertools import product


def test_presets_are_usable_and_fully_slotted():
    for name in ("ode_to_joy", "o_canada"):
        melody = get_preset(name)
        durations = slot_durations(melody)
        problem = compile_melody(melody, style="chorale", K=2)
        assert len(problem.h) == len(durations)
        assert len(problem.W) == len(durations) - 1
        assert abs(sum(durations) - sum(n.duration for n in melody.notes)) < 1e-9


def test_short_note_boundaries_are_not_crossed_by_slots():
    melody = melody_from_text("C4:0.5 D4:1.5 E4", key="C")
    assert slot_durations(melody) == [0.5, 0.5, 1.0, 1.0]


def test_midi_import_reads_track_key_and_quarter_note_durations(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("key_signature", key="F", time=0))
    track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=62, velocity=80, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=960))
    path = tmp_path / "two_notes.mid"
    midi.save(path)

    from compiler import melody_from_midi
    melody = melody_from_midi(path)
    assert melody.tonic == 5
    assert melody.time_signature == (3, 4)
    assert [note.midi for note in melody.notes] == [60, 62]
    assert [note.duration for note in melody.notes] == [1.0, 2.0]


def test_midi_import_detects_minor_key_signature(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("key_signature", key="Am", time=0))
    track.append(mido.Message("note_on", note=69, velocity=80, time=0))
    track.append(mido.Message("note_off", note=69, velocity=0, time=480))
    path = tmp_path / "minor.mid"
    midi.save(path)

    from compiler import melody_from_midi
    melody = melody_from_midi(path)
    assert melody.tonic == 9
    assert melody.mode == "minor"


def test_compiled_energy_matches_independent_violation_counter():
    melody = melody_from_text("C4 D4 E4 F4", key="C")
    problem = compile_melody(melody, style="chorale", K=2)
    selections = product(*(range(len(slot)) for slot in problem.candidates))
    for selection in selections:
        assert check_energy(problem, selection, melody.tonic, get_weights("chorale"))

    best, energy = viterbi(problem)
    brute_energy = min(
        problem.energy(selection)
        for selection in product(*(range(len(slot)) for slot in problem.candidates))
    )
    assert abs(energy - brute_energy) < 1e-9
    assert abs(problem.energy(best) - energy) < 1e-9


def test_spacing_and_leading_tone_rules_have_matching_independent_counts():
    melody = melody_from_text("C4 B4 C5", key="C")
    context = build_context(melody)
    too_wide = Voicing((40, 60, 73, 81), "I")
    assert VoiceSpacing().unary(too_wide, 0, context) == 2
    assert count_violations([too_wide], melody.tonic)["VoiceSpacing"] == 2

    leading_tone = Voicing((48, 55, 60, 71), "V")
    resolves = Voicing((48, 55, 60, 72), "I")
    fails = Voicing((48, 55, 60, 69), "I")
    assert LeadingToneResolution().pairwise(leading_tone, resolves, 0, context) == 0
    assert LeadingToneResolution().pairwise(leading_tone, fails, 0, context) == 1
    assert count_violations([leading_tone, fails], melody.tonic)["LeadingToneResolution"] == 1
