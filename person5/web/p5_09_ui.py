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
               "window_size": int(ui["window"].value), "noisy": ui["backend"].value == "noisy",
               "use_api": bool(ui["use_api"].value), "backend": ui["backend"].value}
    return P5Request(melody=melody, style=ui["style"].value, style_weight=1.0,
                     solver="qaoa", options=options, tempo=tempo)


def p5_safe_panel(out, fn, run):
    """Assign a complete HTML value; avoids Output capture races in callbacks."""
    try:
        out.value = fn(run)
    except Exception:
        out.value = p5_banner_html("<b>Display error</b><pre style='font-size:11px;overflow-x:auto'>"
                                   + p5_html.escape(p5_tb.format_exc()) + "</pre>", "err")


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

    ui["style"] = W.Dropdown(options=[("Bach chorale", "bach_chorale")], value="bach_chorale", disabled=True,
                             description=p5_t("style_label"), layout=lay("320px"), style=sty)
    ui["solver"] = W.HTML(f'<div style="font-family:system-ui,Arial,sans-serif;margin:4px 0 8px 0">'
                          f'<span style="display:inline-block;width:130px">{p5_html.escape(p5_t("solver_label"))}</span>'
                          f'<b>{p5_html.escape(p5_t("s_qaoa"))}</b></div>')
    ui["tempo"] = W.IntSlider(value=100, min=50, max=160, step=2, description=p5_t("tempo_label"), layout=lay("470px"), style=sty)
    ui["p"] = W.IntSlider(value=2, min=1, max=5, description=p5_t("adv_p"), layout=lay("420px"), style=sty)
    ui["shots"] = W.Dropdown(options=[512, 1024, 2048, 4096, 8192], value=2048, description=p5_t("adv_shots"), layout=lay("300px"), style=sty)
    ui["window"] = W.IntSlider(value=3, min=2, max=4, description=p5_t("adv_window"), layout=lay("420px"), style=sty)
    ui["use_api"] = W.Checkbox(value=True, description="Use local quantum API", indent=False)
    ui["backend"] = W.Dropdown(options=[("Aer simulator", "aer"), ("Noisy fake backend", "noisy"),
                                        ("IBM Quantum hardware", "ibm")], value="aer",
                                description="Quantum backend", layout=lay("320px"), style=sty)
    ui["adv"] = W.Accordion(children=[W.VBox([ui["use_api"], ui["backend"], ui["p"], ui["shots"], ui["window"]])], selected_index=0)
    ui["adv"].set_title(0, p5_t("adv_title"))

    ui["preview"] = W.Button(description=p5_t("btn_preview"), icon="volume-up", layout=lay("190px"))
    ui["run"] = W.Button(description=p5_t("btn_run"), button_style="success", icon="play", layout=lay("230px"))
    ui["progress"] = W.IntProgress(value=0, min=0, max=100, layout=lay("260px"))
    ui["status"] = W.Label(value=p5_t("status_ready"))
    ui["variant"] = W.Dropdown(options=[(p5_t("variant_best"), 0)], value=0,
                               description=p5_t("variant_label"), disabled=True,
                               layout=lay("420px"), style=sty)
    ui["out_score"], ui["out_report"], ui["out_export"] = W.HTML(), W.HTML(), W.HTML()
    ui["tabs"] = W.Tab(children=[ui["out_score"], ui["out_report"], ui["out_export"]])
    for i, k in enumerate(("tab_score", "tab_report", "tab_export")):
        ui["tabs"].set_title(i, p5_t(k))

    sec = lambda k: W.HTML(f'<div style="font-family:system-ui,Arial,sans-serif;font-weight:600;margin-top:8px">{p5_html.escape(p5_t(k))}</div>')
    ui["root"] = W.VBox([ui["header"], sec("sec_melody"), ui["melody"], ui["text_box"], ui["midi_box"], ui["custom_opts"],
                         sec("sec_style"), ui["style"], ui["solver"], ui["tempo"], ui["adv"],
                         W.HBox([ui["preview"], ui["run"]]), W.HBox([ui["progress"], ui["status"]]), ui["variant"], ui["tabs"]])

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
        dur = float(ui["pal_dur"].value)
        dur_text = {0.25: "s", 0.5: "e", 0.75: "e.", 1.5: "q.", 2.0: "h", 3.0: "h.", 4.0: "w"}.get(dur)
        if dur_text:
            tok += f":{dur_text}"
        elif dur != 1.0:
            tok += f":{dur:g}"
        ui["text"].value = (ui["text"].value.strip() + " " + tok).strip()
    for b in ui["keys"]:
        b.on_click(add_note)
    ui["undo"].on_click(lambda _b: setattr(ui["text"], "value", " ".join(ui["text"].value.split()[:-1])))
    ui["clear"].on_click(lambda _b: setattr(ui["text"], "value", ""))

    def on_preview(_b):
        try:
            req = p5_ui_collect(ui)
            m = req.melody
            if not m["notes"]:
                ui["out_score"].value = p5_banner_html(p5_html.escape(p5_t("no_notes_preset")), "warn")
                ui["tabs"].selected_index = 0
                return
            prev = next(n for n, _ in m["notes"] if n is not None)
            slots = []
            for n, b in m["notes"]:
                rest = n is None
                if not rest:
                    prev = n
                slots.append(P5Slot((prev, prev, prev, prev), b, soprano_rest=rest))
            sc = P5Score(slots=slots, key=m["key"], time_sig=m["time_sig"],
                         tempo=m["tempo"], pickup=m["pickup"], title=m["title"])
            names = " ".join("R" if n is None else p5_note_name(n, m["key"]) for n, _ in m["notes"])
            info = p5_t("parsed", n=len(m["notes"]), key=m["key"], ts=f"{m['time_sig'][0]}/{m['time_sig'][1]}",
                        beats=sum(b for _, b in m["notes"]))
            _, warns = p5_validate_melody(m["notes"], m["time_sig"])
            ui["out_score"].value = (f'<div style="font-family:system-ui,Arial,sans-serif;color:#222"><b>{p5_html.escape(p5_t("preview_title"))}</b> - '
                                     f'{p5_html.escape(info)}<br><code>{p5_html.escape(names)}</code></div>'
                                     + "".join(p5_banner_html(p5_html.escape(w), "warn") for w in warns[:4])
                                     + p5_audio_html(p5_synth_wav(sc, voices=(3,)), p5_t("audio_sop")))
        except Exception as e:
            ui["out_score"].value = p5_banner_html(p5_html.escape(f"{type(e).__name__}: {e}").replace("\n", "<br>"), "err")
        ui["tabs"].selected_index = 0

    def show_variant(index=None):
        best_run = P5_STATE.get("last_run")
        if best_run is None or best_run.error:
            return
        choices = [best_run] + list(best_run.alternatives or [])
        idx = int(ui["variant"].value if index is None else index)
        if not 0 <= idx < len(choices):
            return
        selected = choices[idx]
        p5_safe_panel(ui["out_score"], p5_score_panel_html, selected)
        p5_safe_panel(ui["out_report"], p5_report_panel_html, selected)
        p5_safe_panel(ui["out_export"], p5_export_panel_html, selected)

    def on_variant_change(change):
        if change.get("name") == "value" and P5_STATE.get("last_run") is not None:
            show_variant(change.get("new"))

    ui["variant"].observe(on_variant_change, names="value")

    def on_run(_b):
        ui["run"].disabled = ui["preview"].disabled = True
        ui["progress"].value = 0
        ui["status"].value = p5_t("status_running")
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
            variants = [run] + list(run.alternatives or [])
            options = []
            for i, variant in enumerate(variants):
                energy = variant.report.get("energy")
                progression = []
                slots = variant.score.slots if variant.score is not None else []
                for slot in slots:
                    chord = slot.chord
                    chord = str(chord)
                    if chord and (not progression or progression[-1] != chord):
                        progression.append(chord)
                chord_path = " – ".join(progression[:10])
                if len(progression) > 10:
                    chord_path += " – …"
                label = f"Progression {i + 1}: {chord_path}" if chord_path else f"Progression {i + 1}"
                if energy is not None:
                    label += f" (energy {float(energy):.3g})"
                options.append((label, i))
            ui["variant"].options = options or [(p5_t("variant_best"), 0)]
            ui["variant"].value = 0
            ui["variant"].disabled = bool(run.error) or len(variants) < 2
            if run.error:
                p5_safe_panel(ui["out_score"], p5_panel_error_html, run)
            else:
                show_variant(0)
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
    out.value = ""


def p5_launch(lang=None):
    """Build and display the interactive panel. lang='zh' switches the interface to Chinese."""
    global P5_LANG
    if lang:
        P5_LANG = lang
    p5_display(p5_build_ui())          # widgets live in P5_UI; nothing is returned so the cell shows the panel once
