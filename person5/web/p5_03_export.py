# %% P5-03 Export: P5Score -> music21 -> MIDI / MusicXML, saving files, download links
P5_VOICE_ORDER_TOP_DOWN = (3, 2, 1, 0)       # score order: soprano, alto, tenor, bass
P5_RED = "#d62728"


def p5_m21_name(midi, key="C"):
    """Pitch name in music21 syntax ('F#4', 'B-3'), spelled for the key."""
    L, alt, o = p5_spell(midi, key)
    acc = {1: "#", -1: "-", 0: ""}[alt]
    return f"{L}{acc}{o}"


def p5_to_m21(sc, marks=None):
    """Build a 4-part music21 Score (open score, with measures, ties, chord labels under the bass).
    marks: {slot_index: set(voice indices) or None(all voices)} -> coloured red."""
    if p5_m21 is None:
        raise RuntimeError(p5_t("no_m21"))
    m21 = p5_m21
    marks = marks or {}
    m21_key = sc.key.replace("b", "-")
    groups = p5_split_measures([s.beats for s in sc.slots], sc.bar_len, sc.pickup)
    classes = {3: m21.instrument.Soprano, 2: m21.instrument.Alto, 1: m21.instrument.Tenor, 0: m21.instrument.Bass}
    clefs = {3: m21.clef.TrebleClef, 2: m21.clef.TrebleClef, 1: m21.clef.Treble8vbClef, 0: m21.clef.BassClef}
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata()
    score.metadata.title = sc.title or "Quantum four-part harmony"
    for v in P5_VOICE_ORDER_TOP_DOWN:
        part = m21.stream.Part(id=P5_VOICE_NAMES[v])
        part.partName = P5_VOICE_NAMES[v]
        for mi, pieces in enumerate(groups):
            meas = m21.stream.Measure(number=mi if sc.pickup > 1e-6 else mi + 1)
            if mi == 0:
                meas.append(classes[v]())
                meas.append(clefs[v]())
                meas.append(m21.key.Key(m21_key))
                meas.append(m21.meter.TimeSignature(f"{sc.time_sig[0]}/{sc.time_sig[1]}"))
                if v == 3:
                    meas.append(m21.tempo.MetronomeMark(number=sc.tempo))
                if sc.pickup > 1e-6:
                    meas.paddingLeft = sc.bar_len - sc.pickup
            for si, beats, tie_in, tie_out in pieces:
                slot = sc.slots[si]
                n = m21.note.Note(p5_m21_name(slot.pitches[v], sc.key))
                n.quarterLength = beats
                if tie_in and tie_out:
                    n.tie = m21.tie.Tie("continue")
                elif tie_in:
                    n.tie = m21.tie.Tie("stop")
                elif tie_out:
                    n.tie = m21.tie.Tie("start")
                if si in marks and (marks[si] is None or v in marks[si]):
                    n.style.color = P5_RED
                if v == 0 and slot.chord and not tie_in:
                    n.addLyric(slot.chord)
                meas.append(n)
            if mi == len(groups) - 1:
                meas.rightBarline = "final"
            part.append(meas)
        score.insert(0, part)
    return score


def p5_m21_bytes(sc, fmt, marks=None):
    """fmt = 'midi' or 'musicxml' -> file bytes."""
    score = p5_to_m21(sc, marks)
    with p5_tmp.TemporaryDirectory() as d:
        path = score.write(fmt, fp=p5_os.path.join(d, "out." + ("mid" if fmt == "midi" else "musicxml")))
        with open(path, "rb") as f:
            return f.read()


def p5_safe_stem(title):
    stem = p5_re.sub(r"[^A-Za-z0-9_-]+", "_", title or "harmony").strip("_") or "harmony"
    return f"{stem}_{p5_dt.datetime.now():%Y%m%d_%H%M%S}"


def p5_save_all(sc, marks=None, score_png=None, wav=None, out_dir=None):
    """Write .mid / .musicxml / .wav / .png into P5_OUT_DIR. Returns {label: (path, bytes, mime)} and errors."""
    out_dir = out_dir or P5_OUT_DIR
    p5_os.makedirs(out_dir, exist_ok=True)
    stem = p5_safe_stem(sc.title)
    files, errors = {}, []
    jobs = [("midi", ".mid", "audio/midi", lambda: p5_m21_bytes(sc, "midi", marks)),
            ("xml", ".musicxml", "application/vnd.recordare.musicxml+xml", lambda: p5_m21_bytes(sc, "musicxml", marks)),
            ("wav", ".wav", "audio/wav", lambda: wav if wav is not None else p5_synth_wav(sc)),
            ("png", ".png", "image/png", lambda: score_png)]
    for label, ext, mime, make in jobs:
        try:
            data = make()
            if data is None:
                continue
            path = p5_os.path.join(out_dir, stem + ext)
            with open(path, "wb") as f:
                f.write(data)
            files[label] = (path, data, mime)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    return files, errors


def p5_download_links_html(files):
    """Base64 download buttons (work in JupyterLab / classic Notebook; the saved paths are the fallback)."""
    names = {"midi": "dl_midi", "xml": "dl_xml", "wav": "dl_wav", "png": "dl_png"}
    style = ("display:inline-block;margin:4px 8px 4px 0;padding:6px 12px;border-radius:6px;"
             "background:#1565c0;color:#fff;text-decoration:none;font-family:system-ui,Arial,sans-serif;font-size:13px")
    out = []
    for label, (path, data, mime) in files.items():
        b64 = p5_b64.b64encode(data).decode("ascii")
        fname = p5_html.escape(p5_os.path.basename(path))
        out.append(f'<a style="{style}" download="{fname}" href="data:{mime};base64,{b64}">{p5_html.escape(p5_t(names[label]))}</a>')
    return "".join(out)
