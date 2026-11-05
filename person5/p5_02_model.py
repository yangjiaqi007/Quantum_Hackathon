# %% P5-02 Data model: P5Score, measure splitting, normalizing whatever the pipeline returns
# Voice order everywhere in Part 5 follows the team contract: pitches = (bass, tenor, alto, soprano).
P5_VOICE_NAMES = ("Bass", "Tenor", "Alto", "Soprano")
P5_RANGES = {0: (40, 64), 1: (48, 69), 2: (55, 74), 3: (60, 81)}     # soft SATB ranges (E2-E4, C3-A4, G3-D5, C4-A5)


@p5_dataclass
class P5Slot:
    pitches: tuple                 # (bass, tenor, alto, soprano) as MIDI numbers
    beats: float = 1.0             # duration in quarter notes
    chord: str = ""                # optional Roman-numeral label such as "V7"


@p5_dataclass
class P5Score:
    slots: list
    key: str = "C"                 # major-key tonic: C G D A E F Bb Eb Ab
    time_sig: tuple = (4, 4)
    tempo: int = 90
    pickup: float = 0.0            # beats in an incomplete first measure (0 = none)
    title: str = ""
    warnings: list = p5_field(default_factory=list)

    @property
    def bar_len(self):
        return self.time_sig[0] * 4.0 / self.time_sig[1]


def p5_split_measures(beats_list, bar_len, pickup=0.0):
    """Group slots into measures. Each measure is a list of pieces (slot_index, beats, tie_in, tie_out);
    a note that crosses a barline is split into tied pieces."""
    eps = 1e-6
    measures, cur = [], []
    room = pickup if pickup > eps else bar_len
    for i, b in enumerate(beats_list):
        left, tie_in = float(b), False
        while left > eps:
            piece = min(left, room)
            tie_out = (left - piece) > eps
            cur.append((i, piece, tie_in, tie_out))
            left -= piece
            room -= piece
            tie_in = True
            if room <= eps:
                measures.append(cur)
                cur, room = [], bar_len
    if cur:
        measures.append(cur)
    return measures


def p5_is_m21_stream(obj):
    return p5_m21 is not None and isinstance(obj, p5_m21.stream.Stream)


def p5_voicing_pitches(v):
    """Pitches (B,T,A,S) from a tuple, a dict or an object with .pitches."""
    if isinstance(v, dict):
        v = v.get("pitches", v)
    elif hasattr(v, "pitches"):
        v = v.pitches
    p = tuple(int(round(float(x))) for x in v)
    if len(p) != 4:
        raise ValueError(f"A voicing needs 4 pitches (bass, tenor, alto, soprano); got {len(p)}.")
    return p


def p5_from_m21(stream, melody):
    """Stream with >=4 parts (S,A,T,B top to bottom) or chords of 4 notes -> P5Score."""
    parts = list(stream.parts) if hasattr(stream, "parts") else []
    slots = []
    if len(parts) >= 4:
        sop = list(parts[0].flatten().notes)
        others = [p.flatten().notes for p in parts[1:4]]
        for n in sop:
            top = max(x.midi for x in n.pitches)
            row = [top]
            for notes in others:
                el = notes.getElementAtOrBefore(n.offset) if hasattr(notes, "getElementAtOrBefore") else None
                row.append(max(x.midi for x in el.pitches) if el is not None else top)
            s, a, t, b = row
            slots.append(P5Slot((b, t, a, s), float(n.quarterLength)))
    else:
        for n in stream.flatten().notes:
            ps = sorted(x.midi for x in n.pitches)
            if len(ps) < 4:
                raise ValueError("music21 result needs 4 parts or 4-note chords.")
            slots.append(P5Slot(tuple(ps[:3] + [ps[-1]]), float(n.quarterLength)))
    return slots


def p5_normalize_result(result, melody=None):
    """Accept P5Score | dict{'voicings',...} | list of voicings | music21 Stream  ->  P5Score."""
    melody = melody or {}
    if isinstance(result, P5Score):
        return result
    meta = {"key": melody.get("key") or "C", "time_sig": tuple(melody.get("time_sig") or (4, 4)),
            "pickup": float(melody.get("pickup") or 0.0), "tempo": int(melody.get("tempo") or 90),
            "title": melody.get("title") or ""}
    notes = melody.get("notes")
    durations = chords = None
    warnings = []
    if p5_is_m21_stream(result):
        slots = p5_from_m21(result, melody)
    else:
        if isinstance(result, dict):
            voicings = result.get("voicings", result.get("slots"))
            durations, chords = result.get("durations"), result.get("chords")
            for k in meta:
                if result.get(k) is not None:
                    meta[k] = tuple(result[k]) if k == "time_sig" else result[k]
        else:
            voicings = result
        if voicings is None:
            raise ValueError("Pipeline result has no 'voicings'. Return a list of (bass, tenor, alto, soprano) "
                             "tuples, a dict with 'voicings', or a 4-part music21 Stream.")
        slots = []
        for i, v in enumerate(voicings):
            p = p5_voicing_pitches(v)
            if p[0] > p[3]:                                   # looks like (S, A, T, B)
                p = tuple(reversed(p))
                if i == 0:
                    warnings.append("Voicings looked like (soprano, alto, tenor, bass); reversed to (bass, tenor, alto, soprano).")
            chord = chords[i] if chords and i < len(chords) else (getattr(v, "chord", "") if not isinstance(v, (tuple, list, dict)) else (v.get("chord", "") if isinstance(v, dict) else ""))
            if durations and i < len(durations):
                b = float(durations[i])
            elif notes and len(notes) == len(voicings):
                b = float(notes[i][1])
            else:
                b = 1.0
            slots.append(P5Slot(p, b, str(chord or "")))
    if notes and len(notes) == len(slots):                    # durations from the input melody
        for s, (_, b) in zip(slots, notes):
            if s.beats == 1.0 and b != 1.0 and not durations:
                s.beats = float(b)
    sc = P5Score(slots=slots, warnings=warnings, **meta)
    sc.warnings += p5_sanity(sc, melody)
    return sc


def p5_sanity(sc, melody=None, limit=5):
    """Cheap integration checks (not harmony rules): soprano preserved, voice order, SATB ranges."""
    w = []
    notes = (melody or {}).get("notes")
    if notes and len(notes) == len(sc.slots):
        bad = [i + 1 for i, ((m, _), s) in enumerate(zip(notes, sc.slots)) if s.pitches[3] != m]
        if bad:
            w.append(f"Soprano differs from the input melody at slot(s) {bad[:8]} - the solver must keep the soprano fixed.")
    elif notes:
        w.append(f"Input melody has {len(notes)} notes but the result has {len(sc.slots)} slots.")
    order = [i + 1 for i, s in enumerate(sc.slots) if not (s.pitches[0] <= s.pitches[1] <= s.pitches[2] <= s.pitches[3])]
    if order:
        w.append(f"Voices are not in bass ≤ tenor ≤ alto ≤ soprano order at slot(s) {order[:8]}.")
    for v in range(4):
        lo, hi = P5_RANGES[v]
        out = [i + 1 for i, s in enumerate(sc.slots) if not lo <= s.pitches[v] <= hi]
        if out:
            w.append(f"{P5_VOICE_NAMES[v]} leaves its usual range at slot(s) {out[:8]}.")
    return w[:limit]
