# %% P5-05 Score renderer: grand staff drawn with matplotlib (offline, no MuseScore / LilyPond needed)
P5_SYS_H = 23.5                                   # vertical size of one system (in staff spaces)
P5_SHARP_STEPS = ([38, 35, 39, 36, 33, 37, 34], [24, 21, 25, 22, 19, 23, 20])   # (treble, bass): F C G D A E B
P5_FLAT_STEPS = ([34, 37, 33, 36, 32, 35, 31], [20, 23, 19, 22, 18, 21, 17])    # B E A D G C F
P5_TREBLE_CLEF = [(0.55, -1.35), (0.75, -1.75), (1.2, -1.9), (1.55, -1.55), (1.55, -0.6), (1.45, 1.0), (1.45, 2.6),
                  (1.55, 4.2), (1.7, 5.2), (2.0, 5.5), (2.35, 5.0), (2.35, 4.1), (1.95, 3.3), (1.2, 2.9), (0.45, 2.4),
                  (0.2, 1.5), (0.5, 0.55), (1.15, 0.25), (1.9, 0.55), (2.3, 1.2), (2.05, 1.9), (1.45, 2.1), (0.95, 1.7),
                  (0.95, 1.2), (1.3, 0.95), (1.6, 1.2)]
P5_BASS_CLEF = [(0.7, 3.0), (0.95, 3.75), (1.6, 3.95), (2.25, 3.45), (2.35, 2.5), (1.9, 1.4), (1.0, 0.6), (0.35, 0.1)]


def p5_spline(points, n=14):
    """Catmull-Rom spline through the points -> dense polyline (N x 2 array)."""
    P = p5_np.array(points, float)
    P = p5_np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in p5_np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return p5_np.array(out)


def p5_step_y(step):
    """Vertical position (staff spaces) of a diatonic step; E4 (step 30) is the bottom treble line, y = 0."""
    return (step - 30) / 2.0


def p5_ledger_ys(y, treble):
    if treble:
        low = list(range(int(p5_math.ceil(y)), 0)) if y <= -1 else []
        return list(range(5, int(p5_math.floor(y)) + 1)) + low
    low = list(range(int(p5_math.ceil(y)), -6)) if y <= -7 else []
    return low + (list(range(-1, int(p5_math.floor(y)) + 1)) if y >= -1 else [])


def p5_draw_clefs(ax, x, oy, pt):
    xy = p5_spline(P5_TREBLE_CLEF)
    ax.plot(x + xy[:, 0], oy + xy[:, 1], color="black", lw=0.17 * pt, solid_capstyle="round", zorder=3)
    ax.add_patch(p5_Circle((x + 0.58, oy - 1.3), 0.2, color="black", zorder=3))
    xy = p5_spline(P5_BASS_CLEF)
    by = oy - 6
    ax.plot(x + xy[:, 0], by + xy[:, 1], color="black", lw=0.2 * pt, solid_capstyle="round", zorder=3)
    ax.add_patch(p5_Circle((x + 0.7, by + 3.0), 0.4, color="black", zorder=3))
    for dy in (3.5, 2.5):
        ax.add_patch(p5_Circle((x + 3.0, by + dy), 0.22, color="black", zorder=3))


def p5_engrave_png(sc, marks=None, windows=None, title=None, width_in=13.5, dpi=110):
    """Render the 4-part score as PNG bytes.
    marks   {slot_index: set(voice idx) | None}  -> notes drawn in red (rule violations)
    windows [(start, end_exclusive), ...]        -> shaded bands labelled W1, W2, ... (QAOA windows)"""
    key, sig = sc.key, p5_key_signature(sc.key)
    n_acc = len(sig)
    marks = marks or {}
    groups = p5_split_measures([s.beats for s in sc.slots], sc.bar_len, sc.pickup)

    plan = {}                                            # accidentals needed: (measure, piece, voice) -> alteration
    for mi, pieces in enumerate(groups):
        state = {}
        for pi, (si, b, tie_in, tie_out) in enumerate(pieces):
            if tie_in:
                continue
            for v in range(4):
                if v == 3 and getattr(sc.slots[si], "soprano_rest", False):
                    continue
                L, alt, o = p5_spell(sc.slots[si].pitches[v], key)
                if alt != state.get((L, o), sig.get(L, 0)):
                    plan[(mi, pi, v)] = alt
                    state[(L, o)] = alt

    wmax = 88.0
    hdr = lambda first: 4.6 + 1.25 * n_acc + (3.4 if first else 0.0)
    meas = []
    for mi, pieces in enumerate(groups):
        ws = [2.6 + 1.8 * min(b, 4.0) ** 0.75 + (1.5 if any((mi, pi, v) in plan for v in range(4)) else 0.0)
              for pi, (si, b, _, _) in enumerate(pieces)]
        meas.append((ws, sum(ws) + 2.0))
    systems, cur, cw = [], [], 0.0
    for mi, (ws, w) in enumerate(meas):
        if cur and cw + w > wmax - hdr(len(systems) == 0):
            systems.append((cur, cw))
            cur, cw = [], 0.0
        cur.append(mi)
        cw += w
    if cur:
        systems.append((cur, cw))
    n_sys = len(systems)

    x0, x1 = -1.0, wmax + 2.0
    ytop, ybot = 13.0, -(n_sys - 1) * P5_SYS_H - 13.2
    fig = p5_Figure(figsize=(width_in, width_in * (ytop - ybot) / (x1 - x0)), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(ybot, ytop)
    ax.axis("off")
    pt = width_in * 72.0 / (x1 - x0)                     # points per staff space
    lw_staff, lw_stem = 0.09 * pt, 0.11 * pt
    Y = lambda y, s: y - s * P5_SYS_H
    ax.text(0, ytop - 0.4, title if title is not None else (sc.title or p5_t("score_title")),
            fontsize=1.3 * pt, fontweight="bold", va="top", color="#222")

    pos = {}                                             # slot -> [(system, x_center, width)]
    prev_head = {}                                       # (slot, voice) -> (x, y, system, stem_up) for ties
    last_measure = len(groups) - 1
    for s, (mlist, cwid) in enumerate(systems):
        hd = hdr(s == 0)
        stretch = 1.0 if s == n_sys - 1 else min(1.6, (wmax - hd) / cwid)
        x_end = hd + cwid * stretch
        for k in range(5):
            for base in (0, -6):
                ax.plot([0, x_end], [Y(base + k, s)] * 2, color="black", lw=lw_staff, solid_capstyle="butt", zorder=1)
        ax.plot([0, 0], [Y(4, s), Y(-6, s)], color="black", lw=lw_stem, zorder=1)
        p5_draw_clefs(ax, 0.5, Y(0, s), pt)
        xk = 4.4
        steps = (P5_SHARP_STEPS if P5_KEY_FIFTHS[key] > 0 else P5_FLAT_STEPS)
        glyph = "♯" if P5_KEY_FIFTHS[key] > 0 else "♭"
        for i in range(n_acc):
            for st in (steps[0][i], steps[1][i]):
                ax.text(xk, Y(p5_step_y(st), s) + (0.35 if glyph == "♭" else 0), glyph, fontsize=3.4 * pt,
                        ha="center", va="center", zorder=3)
            xk += 1.25
        if s == 0:
            for (cy, num, den) in ((3, sc.time_sig[0], sc.time_sig[1]), (1, None, None), (-3, sc.time_sig[0], sc.time_sig[1]), (-5, None, None)):
                txt = str(num) if num is not None else str(sc.time_sig[1])
                ax.text(xk + 1.0, Y(cy, s), txt, fontsize=2.6 * pt, fontweight="bold", ha="center", va="center", zorder=3)
        x = hd
        for mi in mlist:
            ws, _ = meas[mi]
            x += 1.0
            for pi, (si, beats, tie_in, tie_out) in enumerate(groups[mi]):
                w = ws[pi] * stretch
                has_acc = any((mi, pi, v) in plan for v in range(4))
                xc = x + w / 2 + (0.6 if has_acc else 0.0)
                pos.setdefault(si, []).append((s, xc, w))
                marked = si in marks
                if marked:
                    ax.add_patch(p5_Rect((x - 0.1, Y(-10.8, s)), w + 0.2, 19.6, facecolor="#ffe3e3", edgecolor="none", zorder=0.5))
                if not tie_in:
                    ax.text(xc, Y(8.0, s), str(si + 1), fontsize=0.8 * pt, color="#999", ha="center", va="center")
                    if sc.slots[si].chord:
                        ax.text(xc, Y(-11.4, s), sc.slots[si].chord, fontsize=1.0 * pt, color="#444", ha="center", va="center")
                p5_engrave_piece(ax, sc, plan, mi, pi, si, beats, tie_in, tie_out, xc, s, Y, pt, marks, prev_head, lw_stem)
                x += w
            x += 1.0
            thick = (mi == last_measure)
            ax.plot([x, x], [Y(4, s), Y(-6, s)], color="black", lw=(0.4 if thick else 0.11) * pt, zorder=1)
            if thick:
                ax.plot([x - 0.5, x - 0.5], [Y(4, s), Y(-6, s)], color="black", lw=0.11 * pt, zorder=1)

    for k, (a, b) in enumerate(windows or []):
        for s in range(n_sys):
            spans = [(xc - w / 2, xc + w / 2) for si in range(a, b) for (ss, xc, w) in pos.get(si, []) if ss == s]
            if spans:
                xa, xb = min(p[0] for p in spans), max(p[1] for p in spans)
                ax.add_patch(p5_Rect((xa - 0.1, Y(-12.6, s)), xb - xa + 0.2, 23.0, facecolor="#e3f0ff" if k % 2 == 0 else "#e8f5e9",
                                     edgecolor="none", zorder=0))
                ax.text(xa + 0.2, Y(9.7, s), f"W{k + 1}", fontsize=0.9 * pt, color="#1565c0", va="center")
    buf = p5_io.BytesIO()
    p5_Canvas(fig).print_png(buf)
    return buf.getvalue()


def p5_engrave_piece(ax, sc, plan, mi, pi, si, beats, tie_in, tie_out, xc, s, Y, pt, marks, prev_head, lw_stem):
    """Draw the four noteheads (with stems, flags, dots, ledger lines, accidentals, ties) of one piece."""
    key = sc.key
    slot = sc.slots[si]
    base = 2.0 ** p5_math.floor(p5_math.log2(beats + 1e-9))
    ratio = beats / base
    dots = 1 if abs(ratio - 1.5) < 0.05 else (2 if abs(ratio - 1.75) < 0.05 else 0)
    flags = 0 if base > 0.5 + 1e-9 else int(round(-p5_math.log2(base)))
    whole = beats >= 4.0 - 1e-6
    open_head = beats >= 2.0 - 1e-6
    for vu, vd, mid, treble in ((3, 2, 2.0, True), (1, 0, -4.0, False)):
        info = {}
        for v in (vu, vd):
            if v == 3 and getattr(slot, "soprano_rest", False):
                continue
            L, alt, o = p5_spell(slot.pitches[v], key)
            step = 7 * o + P5_LETTERS.index(L)
            info[v] = (step, p5_step_y(step))
        dx = {vu: 0.0, vd: 0.0}
        unison = vu in info and vd in info and info[vu][0] == info[vd][0]
        if vu in info and vd in info and abs(info[vu][0] - info[vd][0]) == 1:
            dx[vu if info[vu][0] > info[vd][0] else vd] = 1.15
        accs = []
        if vu == 3 and getattr(slot, "soprano_rest", False):
            rest_col = "#d62728" if si in marks else "black"
            rest_path = p5_Path([(xc - 0.2, Y(3.0, s)), (xc + 0.55, Y(2.2, s)),
                                 (xc - 0.25, Y(1.45, s)), (xc + 0.35, Y(0.6, s)),
                                 (xc + 0.05, Y(0.15, s)), (xc + 0.75, Y(-0.35, s)),
                                 (xc + 0.15, Y(-0.5, s))],
                                [p5_Path.MOVETO, p5_Path.CURVE3, p5_Path.CURVE3,
                                 p5_Path.CURVE3, p5_Path.CURVE3, p5_Path.CURVE3, p5_Path.CURVE3])
            ax.add_patch(p5_PathPatch(rest_path, facecolor="none", edgecolor=rest_col,
                                      lw=0.28 * pt, zorder=4))
        for v in (vu, vd):
            if v not in info:
                continue
            step, y = info[v]
            red = si in marks and (marks[si] is None or v in marks[si])
            col = "#d62728" if red else "black"
            up = v in (3, 1)
            if not (unison and v == vd):                                  # one head for a unison
                xh = xc + dx[v]
                for ly in p5_ledger_ys(y, treble):
                    ax.plot([xh - 1.05, xh + 1.05], [Y(ly, s)] * 2, color="black", lw=0.09 * pt, zorder=2)
                ax.add_patch(p5_Ellipse((xh, Y(y, s)), 1.7 if whole else 1.28, 0.95 if whole else 0.92,
                                        angle=0 if whole else 22, facecolor="white" if open_head else col,
                                        edgecolor=col, lw=0.11 * pt, zorder=4))
                if dots:
                    for d in range(dots):
                        ax.add_patch(p5_Circle((xh + 1.0 + 0.55 * d, Y(y + (0.5 if step % 2 == 0 else 0.0), s)), 0.17, color=col, zorder=4))
            xh = xc + dx[v]
            if not whole:
                sx = xh + 0.6 if up else xh - 0.6
                ye = max(y + 3.3, mid) if up else min(y - 3.3, mid)
                ax.plot([sx, sx], [Y(y + (0.1 if up else -0.1), s), Y(ye, s)], color=col, lw=lw_stem, zorder=3, solid_capstyle="butt")
                sg = 1 if up else -1
                for f in range(flags):
                    fy = ye - sg * 0.9 * f
                    path = p5_Path([(sx, Y(fy, s)), (sx + 0.2, Y(fy - sg * 0.9, s)), (sx + 1.4, Y(fy - sg * 1.3, s)), (sx + 0.95, Y(fy - sg * 2.5, s))],
                                   [p5_Path.MOVETO, p5_Path.CURVE4, p5_Path.CURVE4, p5_Path.CURVE4])
                    ax.add_patch(p5_PathPatch(path, facecolor="none", edgecolor=col, lw=0.13 * pt, zorder=3))
            if (mi, pi, v) in plan:
                accs.append((y, plan[(mi, pi, v)], col))
            if tie_in and (si, v) in prev_head and prev_head[(si, v)][2] == s:
                x1_, y1_, _, up1 = prev_head[(si, v)]
                d = -0.55 if up1 else 0.55
                path = p5_Path([(x1_ + 0.7, Y(y1_ + d, s)), ((x1_ + xh) / 2, Y(y1_ + d + (-1.3 if up1 else 1.3), s)), (xh - 0.7, Y(y + d, s))],
                               [p5_Path.MOVETO, p5_Path.CURVE3, p5_Path.CURVE3])
                ax.add_patch(p5_PathPatch(path, facecolor="none", edgecolor="black", lw=0.1 * pt, zorder=3))
            if tie_out:
                prev_head[(si, v)] = (xh, y, s, up)
        placed = []
        for y, alt, col in sorted(accs, key=lambda a: -a[0]):
            c = 0
            while any(abs(y - y2) < 3.0 and c2 == c for y2, c2 in placed):
                c += 1
            placed.append((y, c))
            glyph = {1: "♯", -1: "♭", 0: "♮"}[alt]
            ax.text(xc - 1.2 - 1.1 * c, Y(y + (0.35 if alt == -1 else 0.0), s), glyph, fontsize=3.4 * pt, color=col,
                    ha="center", va="center", zorder=4)
