# %% P5-09 Interactive panel (ipywidgets): p5_launch() builds and shows it
P5_UI = {}                                   # widgets of the last panel, by name (handy for debugging)
P5_STATE = {"last_run": None, "status": ""}  # P5_STATE['last_run'] is the most recent P5Run


def p5_ui_read_upload(w):
    """FileUpload value -> (name, bytes) for ipywidgets 7 (dict) and 8 (tuple of dicts); None if empty."""
    v = w.value
    if not v:
        return None
    if isinstance(v, dict):
        name = next(iter(v))
        return name, bytes(v[name]["content"])
    item = v[0]
    return item["name"], bytes(item["content"])


def p5_ui_collect(ui):
    """Read the widgets and build a P5Request. Raises ValueError with a readable message on bad input."""
    src = ui["melody"].value
    tempo = int(ui["tempo"].value)
    if src in P5_PRESETS:
        melody = p5_preset_melody(src)
        melody["tempo"] = tempo
    else:
        ts = tuple(int(x) for x in ui["ts"].value.split("/"))
        key = None if ui["key"].value == "auto" else ui["key"].value
        if src == "text":
            notes, errs = p5_parse_melody_text(ui["text"].value)
            if errs:
                raise ValueError("\n".join(errs[:6]))
            melody = p5_make_melody(notes, key=key, time_sig=ts, pickup=ui["pickup"].value, title="Custom melody",
                                    tempo=tempo, source="text")
        else:
            up = p5_ui_read_upload(ui["upload"])
            if up is None:
                raise ValueError("Choose a .mid file first.")
            melody = p5_parse_midi_bytes(up[1], title=p5_os.path.splitext(up[0])[0])
            if key:
                melody["key"] = key
            melody["tempo"] = tempo
        errs, _ = p5_validate_melody(melody["notes"], melody["time_sig"])
        if errs:
            raise ValueError("\n".join(errs[:6]))
    options = {"layers_p": int(ui["p"].value), "shots": int(ui["shots"].value),
               "window_size": int(ui["window"].value), "noisy": bool(ui["noisy"].value)}
    return P5Request(melody=melody, style=ui["style"].value, style_weight=float(ui["strict"].value),
                     solver=ui["solver"].value, options=options, tempo=tempo)


def p5_safe_panel(out, fn, run):
    """Show a panel inside an Output widget; a bug in one panel must not blank the others."""
    with out:
        try:
            fn(run)
        except Exception:
            p5_display(p5_HTML(p5_banner_html("<b>Display error</b><pre style='font-size:11px;overflow-x:auto'>"
                                              + p5_html.escape(p5_tb.format_exc()) + "</pre>", "err")))


def p5_build_ui():
    """Create all widgets and wire the callbacks. Returns the root widget (also stored in P5_UI['root'])."""
    W = p5_w
    lay = lambda w="520px": W.Layout(width=w)
    sty = {"description_width": "130px"}
    ui = {}
    ui["header"] = W.HTML(f'<div style="font-family:system-ui,Arial,sans-serif"><h3 style="margin:0 0 4px 0">🎵 {p5_html.escape(p5_t("title"))}</h3>'
                          f'<div style="color:#666;font-size:13px">{p5_html.escape(p5_t("subtitle"))}</div></div>')
    options = [(p5_t(spec["label_key"]), pid) for pid, spec in P5_PRESETS.items()]
    options += [(p5_t("opt_custom"), "text"), (p5_t("opt_midi"), "midi")]
    ui["melody"] = W.Dropdown(options=options, value="ode_to_joy", description=p5_t("melody"), layout=lay("470px"), style=sty)

    ui["text"] = W.Textarea(value="E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4:1.5 D4:0.5 D4:2", layout=W.Layout(width="620px", height="64px"),
                            continuous_update=False)
    ui["hint"] = W.HTML(f'<div style="color:#666;font-size:12px">{p5_html.escape(p5_t("custom_hint"))}</div>')
    ui["pal_dur"] = W.Dropdown(options=[("1", 1.0), ("½", 0.5), ("¼", 0.25), ("1½", 1.5), ("2", 2.0), ("3", 3.0), ("4", 4.0)], value=1.0,
                               description=p5_t("pal_dur"), layout=lay("230px"), style={"description_width": "110px"})
    ui["pal_acc"] = W.Dropdown(options=[("♮", ""), ("♯", "#"), ("♭", "b")], value="", description=p5_t("pal_acc"),
                               layout=lay("200px"), style={"description_width": "90px"})
    ui["undo"] = W.Button(description=p5_t("btn_undo"), layout=lay("80px"))
    ui["clear"] = W.Button(description=p5_t("btn_clear"), layout=lay("80px"))
    keys = [f"{n}{o}" for o in (4, 5) for n in "CDEFGAB"] + ["C6"]
    ui["keys"] = [W.Button(description=k, layout=W.Layout(width="46px", height="30px", padding="0")) for k in keys]
    ui["palette"] = W.HBox(ui["keys"], layout=W.Layout(flex_flow="row wrap", width="760px"))
    ui["text_box"] = W.VBox([ui["text"], ui["hint"], W.HBox([ui["pal_dur"], ui["pal_acc"], ui["undo"], ui["clear"]]), ui["palette"]])

    ui["upload"] = W.FileUpload(accept=".mid,.midi", multiple=False, description=p5_t("opt_midi"), layout=lay("220px"))
    ui["midi_box"] = W.VBox([ui["upload"]])
    ui["key"] = W.Dropdown(options=[(p5_t("key_auto"), "auto")] + [(k.replace("b", "♭"), k) for k in P5_KEY_FIFTHS],
                           value="auto", description=p5_t("key_label"), layout=lay("260px"), style={"description_width": "110px"})
    ui["ts"] = W.Dropdown(options=["4/4", "3/4", "2/4", "6/8"], value="4/4", description=p5_t("ts_label"), layout=lay("230px"),
                          style={"description_width": "100px"})
    ui["pickup"] = W.BoundedFloatText(value=0.0, min=0.0, max=7.0, step=0.5, description=p5_t("pickup_label"), layout=lay("210px"),
                                      style={"description_width": "100px"})
    ui["custom_opts"] = W.HBox([ui["key"], ui["ts"], ui["pickup"]])

    ui["style"] = W.Dropdown(options=["strict_counterpoint", "bach_chorale", "jazz_loose"], value="bach_chorale",
                             description=p5_t("style_label"), layout=lay("320px"), style=sty)
    ui["solver"] = W.Dropdown(options=[(p5_t("s_viterbi"), "viterbi"), (p5_t("s_qaoa"), "qaoa"), (p5_t("s_xy"), "xy_qaoa")],
                              value="xy_qaoa", description=p5_t("solver_label"), layout=lay("470px"), style=sty)
    ui["strict"] = W.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description=p5_t("strict_label"), layout=lay("470px"), style=sty)
    ui["tempo"] = W.IntSlider(value=100, min=50, max=160, step=2, description=p5_t("tempo_label"), layout=lay("470px"), style=sty)
    ui["p"] = W.IntSlider(value=2, min=1, max=5, description=p5_t("adv_p"), layout=lay("420px"), style=sty)
    ui["shots"] = W.Dropdown(options=[512, 1024, 2048, 4096, 8192], value=2048, description=p5_t("adv_shots"), layout=lay("300px"), style=sty)
    ui["window"] = W.IntSlider(value=3, min=2, max=4, description=p5_t("adv_window"), layout=lay("420px"), style=sty)
    ui["noisy"] = W.Checkbox(value=False, description=p5_t("adv_noisy"), indent=False)
    ui["adv"] = W.Accordion(children=[W.VBox([ui["p"], ui["shots"], ui["window"], ui["noisy"]])], selected_index=None)
    ui["adv"].set_title(0, p5_t("adv_title"))

    ui["preview"] = W.Button(description=p5_t("btn_preview"), icon="volume-up", layout=lay("190px"))
    ui["run"] = W.Button(description=p5_t("btn_run"), button_style="success", icon="play", layout=lay("230px"))
    ui["progress"] = W.IntProgress(value=0, min=0, max=100, layout=lay("260px"))
    ui["status"] = W.Label(value=p5_t("status_ready"))
    ui["out_score"], ui["out_report"], ui["out_export"] = W.Output(), W.Output(), W.Output()
    ui["tabs"] = W.Tab(children=[ui["out_score"], ui["out_report"], ui["out_export"]])
    for i, k in enumerate(("tab_score", "tab_report", "tab_export")):
        ui["tabs"].set_title(i, p5_t(k))

    sec = lambda k: W.HTML(f'<div style="font-family:system-ui,Arial,sans-serif;font-weight:600;margin-top:8px">{p5_html.escape(p5_t(k))}</div>')
    ui["root"] = W.VBox([ui["header"], sec("sec_melody"), ui["melody"], ui["text_box"], ui["midi_box"], ui["custom_opts"],
                         sec("sec_style"), ui["style"], ui["solver"], ui["strict"], ui["tempo"], ui["adv"],
                         W.HBox([ui["preview"], ui["run"]]), W.HBox([ui["progress"], ui["status"]]), ui["tabs"]])

    def show_hide(_=None):
        v = ui["melody"].value
        ui["text_box"].layout.display = None if v == "text" else "none"
        ui["midi_box"].layout.display = None if v == "midi" else "none"
        ui["custom_opts"].layout.display = None if v in ("text", "midi") else "none"
        if v in P5_PRESETS:
            ui["tempo"].value = P5_PRESETS[v]["tempo"]
    ui["melody"].observe(show_hide, names="value")
    show_hide()

    def add_note(btn):
        name = btn.description
        tok = name[0] + ui["pal_acc"].value + name[1:]
        if ui["pal_dur"].value != 1.0:
            tok += f":{ui['pal_dur'].value:g}"
        ui["text"].value = (ui["text"].value.strip() + " " + tok).strip()
    for b in ui["keys"]:
        b.on_click(add_note)
    ui["undo"].on_click(lambda _b: setattr(ui["text"], "value", " ".join(ui["text"].value.split()[:-1])))
    ui["clear"].on_click(lambda _b: setattr(ui["text"], "value", ""))

    def on_preview(_b):
        with ui["out_score"]:
            p5_clear_output_widget(ui["out_score"])
            try:
                req = p5_ui_collect(ui)
                m = req.melody
                if not m["notes"]:
                    p5_display(p5_HTML(p5_banner_html(p5_html.escape(p5_t("no_notes_preset")), "warn")))
                    return
                sc = P5Score(slots=[P5Slot((n, n, n, n), b) for n, b in m["notes"]], key=m["key"], time_sig=m["time_sig"],
                             tempo=m["tempo"], pickup=m["pickup"], title=m["title"])
                names = " ".join(p5_note_name(n, m["key"]) for n, _ in m["notes"])
                info = p5_t("parsed", n=len(m["notes"]), key=m["key"], ts=f"{m['time_sig'][0]}/{m['time_sig'][1]}",
                            beats=sum(b for _, b in m["notes"]))
                _, warns = p5_validate_melody(m["notes"], m["time_sig"])
                p5_display(p5_HTML(f'<div style="font-family:system-ui,Arial,sans-serif;color:#222"><b>{p5_html.escape(p5_t("preview_title"))}</b> - '
                                   f'{p5_html.escape(info)}<br><code>{p5_html.escape(names)}</code></div>'
                                   + "".join(p5_banner_html(p5_html.escape(w), "warn") for w in warns[:4])
                                   + p5_audio_html(p5_synth_wav(sc, voices=(3,)), p5_t("audio_sop"))))
            except Exception as e:
                p5_display(p5_HTML(p5_banner_html(p5_html.escape(f"{type(e).__name__}: {e}").replace("\n", "<br>"), "err")))
        ui["tabs"].selected_index = 0

    def on_run(_b):
        ui["run"].disabled = ui["preview"].disabled = True
        ui["progress"].value = 0
        ui["status"].value = p5_t("status_running")
        for o in (ui["out_score"], ui["out_report"], ui["out_export"]):
            o.clear_output(wait=True)
        try:
            try:
                req = p5_ui_collect(ui)
            except Exception as e:
                run = P5Run(request=None, error=f"{type(e).__name__}: {e}")
            else:
                def progress(frac, msg=""):
                    ui["progress"].value = int(100 * max(0.0, min(1.0, float(frac))))
                    if msg:
                        ui["status"].value = str(msg)
                run = p5_run_request(req, progress)
            P5_STATE["last_run"] = run
            p5_safe_panel(ui["out_score"], p5_show_score_panel, run)
            p5_safe_panel(ui["out_report"], p5_show_report_panel, run)
            p5_safe_panel(ui["out_export"], p5_show_export_panel, run)
            P5_STATE["status"] = "ok" if not run.error else "error: " + run.error
            ui["status"].value = p5_t("status_done", sec=run.seconds) if not run.error else p5_t("status_error")
            ui["tabs"].selected_index = 0
        finally:
            ui["progress"].value = 100
            ui["run"].disabled = ui["preview"].disabled = False
    ui["preview"].on_click(on_preview)
    ui["run"].on_click(on_run)
    P5_UI.clear()
    P5_UI.update(ui)
    return ui["root"]


def p5_clear_output_widget(out):
    out.clear_output(wait=True)


def p5_launch(lang=None):
    """Build and display the interactive panel. lang='zh' switches the interface to Chinese."""
    global P5_LANG
    if lang:
        P5_LANG = lang
    p5_display(p5_build_ui())          # widgets live in P5_UI; nothing is returned so the cell shows the panel once
