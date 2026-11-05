# %% P5-06 Quantum report: resource table, violations, energy-per-window chart, circuit diagram
# The report is a plain dict produced by the pipeline (Parts 2-4). Every key is optional - Part 5 shows what exists.
# Keys (aliases in brackets):  solver, backend, is_mock, qubits, depth, two_qubit_gates [gates], shots, layers_p [p],
#   runtime_s, violations (int), violation_details [{rule, slot (0-based), voices (0=bass..3=soprano, optional)}],
#   energy, classical_energy, approx_ratio, p_optimum, feasible_fraction,
#   windows [{start, end (exclusive), qubits, depth, two_qubit_gates, energy, classical_energy}],
#   circuit (qiskit QuantumCircuit or list of them), hardware {dict of a recorded real-device run}, verdict, notes [str]
P5_CARD = ("font-family:system-ui,Segoe UI,Arial,sans-serif;border:2px solid {c};border-radius:10px;padding:14px 18px;"
           "background:#fafafa;color:#222;max-width:920px;margin:6px 0;line-height:1.5")
P5_TD = "padding:3px 14px 3px 0;vertical-align:top"


def p5_pick(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def p5_fmt(v, nd=3):
    if v is None:
        return p5_t("rep_na")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer() and abs(v) < 1e9):
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return p5_html.escape(str(v))


def p5_marks_from_report(report):
    """violation_details -> {slot: set(voices) | None}, used to colour notes red in the score."""
    marks = {}
    for d in (report or {}).get("violation_details") or []:
        try:
            slot = int(d["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        voices = d.get("voices")
        if voices:
            if marks.get(slot, set()) is not None:
                marks.setdefault(slot, set()).update(int(v) for v in voices)
        else:
            marks[slot] = None
    return marks


def p5_windows_from_report(report):
    """windows -> [(start, end_exclusive)] for the shaded bands in the score."""
    out = []
    for w in (report or {}).get("windows") or []:
        if "start" in w and "end" in w:
            out.append((int(w["start"]), int(w["end"])))
        elif w.get("slots"):
            out.append((int(min(w["slots"])), int(max(w["slots"])) + 1))
    return out


def p5_report_html(report, sc=None, pipeline_name="", seconds=None):
    r = report or {}
    is_mock = bool(r.get("is_mock"))
    color = "#c62828" if is_mock else "#2e7d32"
    n_slots = len(sc.slots) if sc is not None else None
    rows = [("rep_pipeline", pipeline_name or None), ("rep_solver", r.get("solver")), ("rep_backend", r.get("backend")),
            ("rep_slots", n_slots), ("rep_qubits", r.get("qubits")), ("rep_depth", r.get("depth")),
            ("rep_2q", p5_pick(r, "two_qubit_gates", "gates", "n_2q_gates")), ("rep_shots", r.get("shots")),
            ("rep_layers", p5_pick(r, "layers_p", "p")), ("rep_runtime", p5_pick(r, "runtime_s", default=seconds)),
            ("rep_viol", r.get("violations")), ("rep_energy", r.get("energy")),
            ("rep_classical", r.get("classical_energy")), ("rep_ratio", r.get("approx_ratio")),
            ("rep_popt", r.get("p_optimum")), ("rep_feas", r.get("feasible_fraction"))]
    h = [f'<div style="{P5_CARD.format(c=color)}">',
         f'<div style="font-size:1.15em;font-weight:700;color:{color};margin-bottom:6px">📊 {p5_html.escape(p5_t("rep_title"))}</div>']
    if is_mock:
        h.append('<div style="background:#ffebee;color:#b71c1c;border-radius:6px;padding:6px 10px;margin:6px 0;font-weight:600">'
                 f'⚠ {p5_html.escape(p5_t("mock_banner"))}</div>')
    h.append('<table style="border-collapse:collapse">')
    for key, val in rows:
        if val is not None:
            h.append(f'<tr><td style="{P5_TD};color:#555">{p5_html.escape(p5_t(key))}</td><td style="{P5_TD};font-weight:600">{p5_fmt(val)}</td></tr>')
    h.append("</table>")
    details = r.get("violation_details") or []
    if details:
        h.append(f'<div style="margin-top:8px"><b>{p5_html.escape(p5_t("rep_viol_head"))}</b><ul style="margin:4px 0 0 18px">')
        for d in details[:12]:
            v = d.get("voices")
            h.append(f'<li>{p5_html.escape(str(d.get("rule", "?")))} - slot {int(d.get("slot", -1)) + 1}'
                     + (f' (voices {", ".join(P5_VOICE_NAMES[int(i)] for i in v)})' if v else "") + "</li>")
        h.append("</ul></div>")
    elif r.get("violations") == 0:
        h.append(f'<div style="margin-top:8px;color:#2e7d32">✔ {p5_html.escape(p5_t("rep_viol_none"))}</div>')
    wins = r.get("windows") or []
    if wins:
        h.append(f'<div style="margin-top:10px"><b>{p5_html.escape(p5_t("rep_windows"))}</b></div>'
                 '<table style="border-collapse:collapse;font-size:.92em"><tr style="color:#555">')
        heads = [p5_t("rep_win_no"), p5_t("rep_win_slots"), p5_t("rep_qubits").split(" (")[0], p5_t("rep_depth"),
                 p5_t("rep_2q"), p5_t("rep_energy"), p5_t("rep_classical"), p5_t("rep_win_gap")]
        h.append("".join(f'<th style="{P5_TD};text-align:left">{p5_html.escape(x)}</th>' for x in heads) + "</tr>")
        for i, w in enumerate(wins, 1):
            e, c = w.get("energy"), w.get("classical_energy")
            gap = (e - c) if isinstance(e, (int, float)) and isinstance(c, (int, float)) else None
            span = f'{int(w["start"]) + 1}-{int(w["end"])}' if "start" in w and "end" in w else p5_t("rep_na")
            cells = [f"W{i}", span, w.get("qubits"), w.get("depth"), p5_pick(w, "two_qubit_gates", "gates"), e, c, gap]
            h.append("<tr>" + "".join(f'<td style="{P5_TD}">{p5_fmt(x) if j else x}</td>' for j, x in enumerate(cells)) + "</tr>")
        h.append("</table>")
    hw = r.get("hardware")
    if isinstance(hw, dict) and hw:
        h.append(f'<div style="margin-top:10px"><b>{p5_html.escape(p5_t("rep_hardware"))}</b><table style="border-collapse:collapse">')
        for k, v in hw.items():
            h.append(f'<tr><td style="{P5_TD};color:#555">{p5_html.escape(str(k))}</td><td style="{P5_TD}">{p5_fmt(v)}</td></tr>')
        h.append("</table></div>")
    if r.get("verdict"):
        h.append(f'<div style="margin-top:10px"><b>{p5_html.escape(p5_t("rep_verdict"))}:</b> {p5_html.escape(str(r["verdict"]))}</div>')
    for note in r.get("notes") or []:
        h.append(f'<div style="margin-top:4px;color:#555">• {p5_html.escape(str(note))}</div>')
    h.append('<hr style="border:0;border-top:1px solid #ddd;margin:12px 0 8px">'
             f'<div style="font-size:.93em;color:#444"><b>💡 {p5_html.escape(p5_t("rep_honest_title"))}</b><br>{p5_html.escape(p5_t("rep_honest"))}</div></div>')
    return "".join(h)


def p5_energy_chart_png(report, width_in=8.0):
    """Grouped bars: quantum energy vs classical optimum for every window (None if there is nothing to plot)."""
    wins = [w for w in (report or {}).get("windows") or [] if isinstance(w.get("energy"), (int, float))]
    if not wins:
        return None
    n = len(wins)
    fig = p5_Figure(figsize=(max(5.0, min(width_in, 1.0 + 0.9 * n)), 3.2), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)
    xs = p5_np.arange(n)
    ax.bar(xs - 0.2, [w["energy"] for w in wins], 0.4, label=p5_t("rep_q_series"), color="#1565c0")
    cl = [w.get("classical_energy") for w in wins]
    if any(isinstance(c, (int, float)) for c in cl):
        ax.bar(xs + 0.2, [c if isinstance(c, (int, float)) else 0 for c in cl], 0.4, label=p5_t("rep_c_series"), color="#9e9e9e")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"W{i + 1}" for i in range(n)])
    ax.set_title(p5_t("rep_chart"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = p5_io.BytesIO()
    p5_Canvas(fig).print_png(buf)
    return buf.getvalue()


def p5_circuit_png(circ):
    """PNG of a Qiskit circuit via circ.draw('mpl'); returns None if drawing is not possible."""
    try:
        import matplotlib.pyplot as p5_plt_local
        first = circ[0] if isinstance(circ, (list, tuple)) else circ
        fig = first.draw("mpl", fold=40, scale=0.6)
        buf = p5_io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        p5_plt_local.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def p5_circuit_text(circ, max_chars=3000):
    try:
        first = circ[0] if isinstance(circ, (list, tuple)) else circ
        txt = str(first.draw("text", fold=120))
        return txt[:max_chars] + ("\n..." if len(txt) > max_chars else "")
    except Exception:
        return None


def p5_show_report(report, sc=None, pipeline_name="", seconds=None):
    """Display the report card, the energy chart and the circuit diagram."""
    p5_display(p5_HTML(p5_report_html(report, sc, pipeline_name, seconds)))
    png = p5_energy_chart_png(report)
    if png:
        p5_display(p5_Image(data=png, format="png"))
    circ = (report or {}).get("circuit")
    if circ is not None:
        p5_display(p5_HTML(f'<div style="font-family:system-ui,Arial,sans-serif;margin-top:8px"><b>{p5_html.escape(p5_t("rep_circuit"))}</b></div>'))
        cpng = p5_circuit_png(circ)
        if cpng:
            p5_display(p5_Image(data=cpng, format="png"))
        else:
            txt = p5_circuit_text(circ)
            if txt:
                p5_display(p5_HTML(f'<pre style="font-size:10px;line-height:1.1;overflow-x:auto;color:#222;background:#fff">{p5_html.escape(txt)}</pre>'))
