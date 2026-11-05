# %% P5-08 Display panels: score + audio, export. Used by the interactive UI and by the static showcase.
def p5_banner_html(text_html, kind="warn"):
    bg, fg = {"warn": ("#fff8e1", "#8d6e00"), "err": ("#ffebee", "#b71c1c"), "mock": ("#ffebee", "#b71c1c"),
              "ok": ("#e8f5e9", "#1b5e20")}[kind]
    return (f'<div style="font-family:system-ui,Segoe UI,Arial,sans-serif;background:{bg};color:{fg};border-radius:6px;'
            f'padding:8px 12px;margin:6px 0;max-width:920px;line-height:1.45">{text_html}</div>')


def p5_render_artifacts(run):
    """Render (once) the score PNG and the WAV files of a successful run."""
    if run.artifacts:
        return run.artifacts
    sc = run.score
    marks = p5_marks_from_report(run.report)
    wins = p5_windows_from_report(run.report)
    title = f"{sc.title or p5_t('score_title')} · {sc.key} major · {sc.time_sig[0]}/{sc.time_sig[1]}"
    run.artifacts = {
        "marks": marks, "windows": wins,
        "png": p5_engrave_png(sc, marks=marks, windows=wins, title=title),
        "wav_full": p5_synth_wav(sc),
        "wav_sop": p5_synth_wav(sc, voices=(3,)),
    }
    return run.artifacts


def p5_show_error(run):
    msg = p5_html.escape(run.error)
    p5_display(p5_HTML(p5_banner_html(f"<b>{p5_html.escape(p5_t('err_title'))}</b><br>{msg}", "err")
                       + f'<details style="max-width:920px"><summary>traceback</summary><pre style="font-size:11px;overflow-x:auto;'
                         f'color:#222;background:#fff">{p5_html.escape(run.traceback)}</pre></details>'))


def p5_show_score_panel(run):
    """Banners, score image and audio players."""
    if run.error:
        p5_show_error(run)
        return
    if run.report.get("is_mock"):
        p5_display(p5_HTML(p5_banner_html("⚠ " + p5_html.escape(p5_t("mock_banner")), "mock")))
    if run.warnings:
        items = "".join(f"<li>{p5_html.escape(w)}</li>" for w in run.warnings)
        p5_display(p5_HTML(p5_banner_html(f"<b>{p5_html.escape(p5_t('warn_title'))}</b><ul style='margin:4px 0 0 18px'>{items}</ul>", "warn")))
    art = p5_render_artifacts(run)
    p5_display(p5_Image(data=art["png"], format="png"))
    if art["marks"]:
        p5_display(p5_HTML(p5_banner_html(p5_html.escape(p5_t("viol_marked")), "warn")))
    p5_display(p5_HTML(p5_audio_html(art["wav_sop"], p5_t("audio_sop")) + p5_audio_html(art["wav_full"], p5_t("audio_full"))))


def p5_show_report_panel(run):
    if run.error:
        p5_show_error(run)
        return
    p5_show_report(run.report, run.score, run.pipeline_name, run.seconds)


def p5_show_export_panel(run):
    """Save MIDI / MusicXML / WAV / PNG to P5_OUT_DIR and show download buttons."""
    if run.error:
        return
    art = p5_render_artifacts(run)
    if p5_m21 is None:
        p5_display(p5_HTML(p5_banner_html(p5_html.escape(p5_t("no_m21")), "warn")))
    files, errors = p5_save_all(run.score, marks=art["marks"], score_png=art["png"], wav=art["wav_full"])
    run.artifacts["files"] = {k: v[0] for k, v in files.items()}
    parts = [p5_download_links_html(files)]
    if files:
        folder = p5_html.escape(p5_os.path.abspath(p5_os.path.dirname(next(iter(files.values()))[0])))
        names = "".join(f"<li><code>{p5_html.escape(p5_os.path.basename(v[0]))}</code></li>" for v in files.values())
        parts.append(f'<div style="font-family:system-ui,Arial,sans-serif;font-size:13px;color:#222;margin-top:6px">'
                     f'{p5_html.escape(p5_t("export_saved"))} <code>{folder}</code><ul style="margin:4px 0 0 18px">{names}</ul></div>')
    for e in errors:
        parts.append(p5_banner_html(p5_html.escape(e), "warn"))
    p5_display(p5_HTML("".join(parts)))


def p5_show_run(run):
    """Show everything for one run, one after another (used by the static showcase)."""
    p5_show_score_panel(run)
    if not run.error:
        p5_show_report_panel(run)
        p5_show_export_panel(run)
