# %% P5-11 Optional demo showcase and post-merge self-check
P5_DEFAULT_OPTIONS = {"layers_p": 1, "shots": 512, "window_size": 3, "noisy": False,
                      "use_api": True, "backend": "aer"}


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
    """Check dependencies, real API wiring, rendering/export and both real quantum solvers."""
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
        caps = p5_api_capabilities()
        assert caps.get("team_adapters_configured"), "Team compiler / QAOA adapter is not configured"
        assert "aer" in caps.get("configured_backends", []), "Aer simulator is not configured"
        assert caps.get("xy_solver_configured"), "XY-QAOA adapter is not configured"
        return "team compiler, QAOA, XY-QAOA and Aer simulator connected"

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
        melody = p5_make_melody([(60, 1), (62, 1), (64, 1), (60, 1)], key="C", title="Four-note QAOA demo", tempo=90)
        runs = []
        for solver in ("qaoa", "xy_qaoa"):
            options = {**P5_DEFAULT_OPTIONS, "window_size": 2}
            run = p5_run_request(P5Request(melody=melody, solver=solver, options=options, tempo=90))
            assert not run.error, f"{solver}: {run.error}"
            assert not run.report.get("is_mock"), f"{solver} returned a mock result"
            assert run.report.get("solver") == solver, f"Expected {solver}, got {run.report.get('solver')}"
            assert run.score and run.score.slots, f"{solver} returned no score"
            p5_render_artifacts(run)
            runs.append(run)
        P5_STATE["last_run"] = runs[0]
        for run in runs:
            print(f"{run.report['solver']} on {run.report.get('backend')}: {len(run.score.slots)} slots, "
                  f"energy {run.report.get('energy')}, {run.seconds:.1f}s, mock={run.report.get('is_mock')}")
            p5_show_score_panel(run)
            p5_show_report_panel(run)
        return "both QAOA and XY-QAOA produced non-mock Aer results"

    step("dependencies", deps)
    step("pipeline hook", hook)
    step("melody parsing", parse)
    step("score / audio / export", stages)
    if run_pipeline:
        step("end-to-end smoke test (4 notes)", smoke)
    for ok, name, detail in results:
        print(("✔ " if ok else "✘ ") + name + (f" - {detail}" if detail else ""))
    return all(ok for ok, _, _ in results)
