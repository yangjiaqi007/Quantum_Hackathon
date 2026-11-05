# %% P5-01 Melody input: note names, validation, key guess, presets, MIDI upload
P5_LETTERS = "CDEFGAB"
P5_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
P5_KEY_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "F": -1, "Bb": -2, "Eb": -3, "Ab": -4}
P5_SHARP_ORDER = "FCGDAEB"
P5_FLAT_ORDER = "BEADGCF"
P5_SOPRANO_SOFT = (60, 81)     # C4 .. A5: outside -> warning
P5_SOPRANO_HARD = (55, 88)     # outside -> error
P5_DUR_LETTERS = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}

# Preset melodies owned by the UI. "text" is in the same syntax users type.
# O Canada follows the 28-bar score supplied for this project. Rests are preserved in the UI.
P5_PRESETS = {
    "ode_to_joy": {
        "label_key": "opt_ode", "title": "Ode to Joy", "key": "C", "time_sig": (4, 4), "pickup": 0.0, "tempo": 100,
        "text": ("E4 E4 F4 G4 | G4 F4 E4 D4 | C4 C4 D4 E4 | E4:1.5 D4:0.5 D4:2 | "
                 "E4 E4 F4 G4 | G4 F4 E4 D4 | C4 C4 D4 E4 | D4:1.5 C4:0.5 C4:2"),
    },
    "o_canada": {
        "label_key": "opt_canada", "title": "O Canada", "key": "F", "time_sig": (4, 4), "pickup": 0.0, "tempo": 80,
        "text": ("A4:h C5:q. C5:e | F4:h. G4:q | A4:q Bb4:q C5:q D5:q | G4:h. R:q | "
                 "A4:h Bb4:q. Bb4:e | C5:h. D5:q | E5:q E5:q D5:q D5:q | C5:h. G4:e. A4:s | "
                 "Bb4:q. A4:e G4:q A4:e. Bb4:s | C5:q. Bb4:e A4:q Bb4:e. C5:s | D5:q C5:q Bb4:q A4:q | "
                 "G4:h. G4:e. A4:s | Bb4:q. A4:e G4:q A4:e. Bb4:s | C5:q. Bb4:e A4:q A4:q | "
                 "G4:q C5:q C5:e Bb4:e A4:e Bb4:e | C5:h. R:q | A4:h C5:q. C5:e | F4:h. R:q | "
                 "Bb4:h D5:q. D5:e | G4:h. R:q | C5:h C#5:q. C#5:e | D5:q Bb4:q A4:q G4:q | "
                 "F4:h G4:h | A4:h. R:q | C5:h F5:q. F5:e | D5:q Bb4:q A4:q G4:q | C5:h E4:h | F4:w"),
    },
}


def p5_key_tonic_pc(key):
    """Pitch class (0-11) of a major-key tonic given as 'C', 'F#', 'Bb' ..."""
    alt = {"#": 1, "b": -1}.get(key[1:2], 0)
    return (P5_LETTER_PC[key[0].upper()] + alt) % 12


def p5_key_signature(key):
    """{letter: +1/-1} altered by the key signature of a major key."""
    n = P5_KEY_FIFTHS[key]
    if n > 0:
        return {L: 1 for L in P5_SHARP_ORDER[:n]}
    if n < 0:
        return {L: -1 for L in P5_FLAT_ORDER[:-n]}
    return {}


def p5_spell(midi, key="C"):
    """Spell a MIDI pitch as (letter, alteration, octave) using the key signature."""
    sig = p5_key_signature(key)
    pc = midi % 12
    for L in P5_LETTERS:                                   # 1) diatonic spelling
        alt = sig.get(L, 0)
        if (P5_LETTER_PC[L] + alt) % 12 == pc:
            return L, alt, (midi - alt) // 12 - 1
    for L in P5_LETTERS:                                   # 2) natural white key (needs a natural sign)
        if P5_LETTER_PC[L] == pc:
            return L, 0, midi // 12 - 1
    fifths = P5_KEY_FIFTHS[key]                            # 3) black key outside the key
    if fifths > 0:
        table = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)}
    elif fifths < 0:
        table = {1: ("C", 1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)} if key == "F" else {1: ("D", -1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)}
    else:
        table = {1: ("C", 1), 3: ("E", -1), 6: ("F", 1), 8: ("G", 1), 10: ("B", -1)}
    L, alt = table[pc]
    return L, alt, (midi - alt) // 12 - 1


def p5_note_name(midi, key="C"):
    """Human-readable name like 'F#4' or 'Bb3'."""
    L, alt, o = p5_spell(midi, key)
    return f"{L}{ {1: '#', -1: 'b', 0: ''}[alt] }{o}"


def p5_parse_dur(txt):
    """'1.5' / '1/2' / 'q' / 'e.' -> beats (quarter note = 1)."""
    txt = txt.strip().lower()
    m = p5_re.fullmatch(r"([whqes])(\.*)", txt)
    if m:
        d = P5_DUR_LETTERS[m.group(1)]
        add = d / 2
        for _ in m.group(2):
            d += add
            add /= 2
        return d
    if "/" in txt:
        a, b = txt.split("/", 1)
        return float(a) / float(b)
    return float(txt)


def p5_parse_melody_text(text):
    """Parse 'E4 E4 F4:1.5 ...' -> (notes, errors); notes = [(midi, beats), ...]."""
    notes, errors = [], []
    toks = [t for t in p5_re.split(r"[\s,;|]+", (text or "").strip()) if t]
    acc_val = {"#": 1, "♯": 1, "b": -1, "♭": -1, "-": -1}
    for k, tok in enumerate(toks, 1):
        rm = p5_re.fullmatch(r"(?:R|REST)(?::(\S+))?", tok, p5_re.I)
        if rm:
            try:
                beats = 1.0 if rm.group(1) is None else p5_parse_dur(rm.group(1))
            except (ValueError, ZeroDivisionError):
                errors.append(f"#{k} '{tok}': cannot read the rest duration")
                continue
            if not (0 < beats <= 8):
                errors.append(f"#{k} '{tok}': duration must be between 0 and 8 beats")
                continue
            notes.append((None, float(beats)))
            continue
        m = p5_re.fullmatch(r"([A-Ga-g])([#♯b♭-]{0,2})(-?\d)(?::(\S+))?", tok)
        if not m:
            errors.append(f"#{k} '{tok}' is not a note (examples: C4, F#4, Bb3, E4:1.5)")
            continue
        alt = sum(acc_val[c] for c in m.group(2))
        midi = 12 * (int(m.group(3)) + 1) + P5_LETTER_PC[m.group(1).upper()] + alt
        try:
            beats = 1.0 if m.group(4) is None else p5_parse_dur(m.group(4))
        except (ValueError, ZeroDivisionError):
            errors.append(f"#{k} '{tok}': cannot read the duration '{m.group(4)}'")
            continue
        if not (0 < beats <= 8):
            errors.append(f"#{k} '{tok}': duration must be between 0 and 8 beats")
            continue
        notes.append((midi, float(beats)))
    return notes, errors


def p5_validate_melody(notes, time_sig=(4, 4)):
    """Return (errors, warnings) for a list of (midi, beats)."""
    errors, warnings = [], []
    if sum(m is not None for m, _ in notes) < 2:
        errors.append("Enter at least 2 pitched notes.")
    if len(notes) > P5_MAX_NOTES:
        errors.append(f"Too many notes ({len(notes)}); the limit is {P5_MAX_NOTES}.")
    lo, hi = P5_SOPRANO_HARD
    slo, shi = P5_SOPRANO_SOFT
    for i, (m, _) in enumerate(notes, 1):
        if m is None:
            continue
        if not lo <= m <= hi:
            errors.append(f"Note #{i} ({p5_note_name(m)}) is outside the supported soprano range.")
        elif not slo <= m <= shi:
            warnings.append(f"Note #{i} ({p5_note_name(m)}) is outside the usual soprano range C4-A5.")
    for i in range(1, len(notes)):
        if notes[i][0] is not None and notes[i - 1][0] is not None and abs(notes[i][0] - notes[i - 1][0]) > 12:
            warnings.append(f"Leap larger than an octave between notes #{i} and #{i + 1}.")
    return errors, warnings


def p5_guess_key(notes):
    """Best-fitting major key among C G D A E F Bb Eb Ab (diatonic coverage, tonic-at-the-end bonus)."""
    pcs = [m % 12 for m, _ in notes if m is not None]
    if not pcs:
        return "C"
    best, best_score = "C", -1.0
    for name, fifths in P5_KEY_FIFTHS.items():
        tonic = p5_key_tonic_pc(name)
        scale = {(tonic + i) % 12 for i in (0, 2, 4, 5, 7, 9, 11)}
        score = sum(p in scale for p in pcs) / max(1, len(pcs))
        score += 0.03 * (pcs[-1] == tonic) + 0.01 * (pcs[0] == tonic) - 0.001 * abs(fifths)
        if score > best_score:
            best, best_score = name, score
    return best


def p5_make_melody(notes, key=None, time_sig=(4, 4), pickup=0.0, title="", preset=None, tempo=90, source="text"):
    """Normalized melody dict handed to the pipeline (see README: integration contract)."""
    return {
        "title": title, "preset": preset, "source": source,
        "notes": [(None if m is None else int(m), float(b)) for m, b in notes] if notes else None,
        "key": key or (p5_guess_key(notes) if notes else None), "mode": "major",
        "time_sig": tuple(time_sig), "pickup": float(pickup), "tempo": int(tempo),
    }


def p5_preset_melody(preset_id):
    """Melody dict for a preset id, resolving canonical Part 1 presets when available."""
    spec = P5_PRESETS[preset_id]
    notes = None
    if spec["text"]:
        notes, errs = p5_parse_melody_text(spec["text"])
        if errs:
            raise ValueError("; ".join(errs))
    return p5_make_melody(notes, key=spec["key"], time_sig=spec["time_sig"], pickup=spec["pickup"],
                          title=spec["title"], preset=preset_id, tempo=spec["tempo"], source="preset")


def p5_parse_midi_bytes(data, title="MIDI upload"):
    """Read a monophonic melody from MIDI bytes (needs music21). Returns a melody dict."""
    if p5_m21 is None:
        raise RuntimeError("music21 is required to read MIDI files (pip install music21).")
    with p5_tmp.TemporaryDirectory() as d:
        path = p5_os.path.join(d, "in.mid")
        with open(path, "wb") as f:
            f.write(data)
        s = p5_m21.converter.parse(path)
    flat = s.flatten()
    items = []
    for el in flat.notes:
        top = max(p.midi for p in el.pitches)                 # chord -> keep the top pitch
        items.append((float(el.offset), top, float(el.quarterLength)))
    items.sort()
    notes, end = [], -1.0
    for off, m, ql in items:                                   # drop overlapping notes
        if off >= end - 1e-6:
            notes.append((m, max(0.25, round(ql * 4) / 4)))
            end = off + ql
    ts, key = (4, 4), None
    for t in flat.getElementsByClass("TimeSignature"):
        ts = (int(t.numerator), int(t.denominator))
        break
    for k in flat.getElementsByClass("KeySignature"):
        try:
            cand = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E", -1: "F", -2: "Bb", -3: "Eb", -4: "Ab"}[int(k.sharps)]
            key = cand
        except KeyError:
            pass
        break
    return p5_make_melody(notes, key=key, time_sig=ts, title=title, source="midi")
