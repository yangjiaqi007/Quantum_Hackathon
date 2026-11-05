# %% P5-11 Optional demo showcase and post-merge self-check
P5_DEFAULT_OPTIONS = {"layers_p": 2, "shots": 2048, "window_size": 3, "noisy": False}


def p5_demo_showcase(presets=("ode_to_joy", "o_canada"), solver="qaoa", style="bach_chorale", style_weight=1.0, **options):
    """Run the demo melodies through the pipeline and show score, audio, report and export for each one.
    Render both demo melodies in the current notebook session."""
    opts = {**P5_DEFAULT_OPTIONS, **options}
    for pid in presets:
        spec = P5_PRESETS[pid]
        p5_display(p5_HTML(f'<h3 style="font-family:system-ui,Arial,sans-serif;margin:18px 0 4px 0;color:#222">🎼 {p5_html.escape(p5_t(spec["label_key"]))} '
                           f'- solver: <code>{p5_html.escape(solver)}</code></h3>'))
        melody = p5_preset_melody(pid)
        req = P5Request(melody=melody, style=style, style_weight=style_weight, solver=solver, options=opts, tempo=melody["tempo"])
        run = p5_run_request(req)
        P5_STATE["last_run"] = run
        p5_show_run(run)


def p5_selfcheck(run_pipeline=True):
    """Run after merging cells: checks dependencies, the pipeline hook and every Part-5 stage; prints a checklist."""
    results = []

    def step(name, fn):
        try:
            detail = fn()
            results.append((True, name, detail or ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))

    def deps():
        import importlib
        found = []
        for mod in ("music21", "qiskit", "qiskit_optimization", "qiskit_aer", "pylatexenc"):
            try:
                importlib.import_module(mod)
                found.append(mod)
            except Exception:
                pass
        return "available: " + ", ".join(found) + ("" if p5_m21 is not None else "  (music21 missing -> no MIDI/MusicXML export)")

    def hook():
        name, _ = p5_find_pipeline()
        return name + ("  <- MOCK: real harmonize_pipeline not defined yet" if name == "p5_mock_pipeline" else "")

    def parse():
        notes, errs = p5_parse_melody_text("C4 D4 E4:1.5 F4:0.5 G4 | Bb4")
        assert not errs and len(notes) == 6, errs
        return f"{len(notes)} notes, key guess {p5_guess_key(notes)}"

    def stages():
        sc = P5Score(slots=[P5Slot(v, 1.0, "I") for v in ((48, 55, 64, 72), (43, 55, 62, 71), (48, 55, 64, 72))], key="C", title="selfcheck")
        png = p5_engrave_png(sc)
        wav = p5_synth_wav(sc)
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and wav[:4] == b"RIFF"
        extra = ""
        if p5_m21 is not None:
            assert len(p5_m21_bytes(sc, "midi")) > 50 and b"<score-partwise" in p5_m21_bytes(sc, "musicxml")
            extra = ", MIDI + MusicXML"
        return f"score PNG {len(png) // 1024} KB, WAV {len(wav) // 1024} KB{extra}"

    def smoke():
        melody = p5_make_melody([(60, 1), (62, 1), (64, 1), (60, 1)], key="C", title="smoke test", tempo=90)
        run = p5_run_request(P5Request(melody=melody, solver="viterbi", options=dict(P5_DEFAULT_OPTIONS), tempo=90))
        assert not run.error, run.error
        p5_render_artifacts(run)
        note = f"; warnings: {run.warnings}" if run.warnings else ""
        return f"{len(run.score.slots)} slots via {run.pipeline_name} in {run.seconds:.1f}s{note}"

    step("dependencies", deps)
    step("pipeline hook", hook)
    step("melody parsing", parse)
    step("score / audio / export", stages)
    if run_pipeline:
        step("end-to-end smoke test (4 notes)", smoke)
    for ok, name, detail in results:
        print(("✔ " if ok else "✘ ") + name + (f" - {detail}" if detail else ""))
    return all(ok for ok, _, _ in results)
