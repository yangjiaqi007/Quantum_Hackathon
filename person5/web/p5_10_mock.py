# %% P5-10 MOCK pipeline - DELETE THIS CELL once the team's harmonize_pipeline exists
# A tiny classical dynamic-programming harmonizer so the interface, score, audio, export and report can be
# tested today. It is NOT quantum and the report says so (is_mock=True).
P5_MOCK_CHORDS = {"I": (0, 4, 7), "ii": (2, 5, 9), "IV": (5, 9, 0), "V": (7, 11, 2), "vi": (9, 0, 4)}


def p5_mock_candidates(sop, tonic, k_keep=8, want_tonic=False):
    rel = (sop - tonic) % 12
    out = []
    for name, cp in P5_MOCK_CHORDS.items():
        if rel not in cp:
            continue
        tones = [(tonic + c) % 12 for c in cp]
        for b in [x for x in range(40, 61) if x % 12 in tones[:2]]:
            for t in [x for x in range(48, 70) if x % 12 in tones and x > b]:
                for a in [x for x in range(55, 75) if x % 12 in tones and t < x < sop]:
                    if sop - a > 12 or a - t > 12:
                        continue
                    pcs = {b % 12, t % 12, a % 12, sop % 12}
                    cost = (0 if tones[0] in pcs and tones[1] in pcs else 6) + (0 if b % 12 == tones[0] else 2)
                    cost += 2 * [b, t, a, sop].count(tones[1]) % 3
                    cost += 8 if (want_tonic and name != "I") else 0     # start and end on the tonic chord
                    out.append((cost, name, (b, t, a, sop)))
    out.sort()
    return out[:k_keep]


def p5_mock_parallels(p, q):
    """Voice pairs moving in parallel perfect fifths / octaves from voicing p to q."""
    hits = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = (p[j] - p[i]) % 12, (q[j] - q[i]) % 12
            moved = p[i] != q[i] and p[j] != q[j]
            same_dir = (q[i] - p[i]) * (q[j] - p[j]) > 0
            if moved and same_dir and a == b and a in (0, 7):
                hits.append((i, j))
    return hits


def p5_mock_pipeline(melody, style_weight=1.0, solver="viterbi", progress=None):
    notes = melody.get("notes") if isinstance(melody, dict) else None
    if not notes:
        raise ValueError(f"The mock backend has no notes for preset '{(melody or {}).get('preset')}'. "
                         "Its data comes from Part 1 - use Ode to Joy or a custom melody until the real pipeline is in.")
    key = melody.get("key") or "C"
    tonic = p5_key_tonic_pc(key)
    sops = [m for m, _ in notes]
    cands = []
    for i, s in enumerate(sops):
        c = p5_mock_candidates(s, tonic, want_tonic=(i in (0, len(sops) - 1)))
        if not c:
            raise ValueError(f"Mock backend: no voicing found for slot {i + 1} ({p5_note_name(s, key)}).")
        cands.append(c)
        if progress:
            progress(0.4 * (i + 1) / len(sops), f"mock: candidates {i + 1}/{len(sops)}")
    best = [[(c[0], None) for c in cands[0]]]                      # DP table: (cost, backpointer)
    for t in range(1, len(sops)):
        row = []
        for cur in cands[t]:
            opts = []
            for j, prev in enumerate(cands[t - 1]):
                p, q = prev[2], cur[2]
                move = 0.5 * abs(q[0] - p[0]) + abs(q[1] - p[1]) + abs(q[2] - p[2])
                par = 20.0 * style_weight * len(p5_mock_parallels(p, q))
                retro = 3.0 if (prev[1] == "V" and cur[1] in ("IV", "ii")) else (-1.5 if (prev[1] == "V" and cur[1] == "I") else 0.0)
                opts.append((best[t - 1][j][0] + cur[0] + move + par + retro, j))
            row.append(min(opts))
        best.append(row)
        if progress:
            progress(0.4 + 0.6 * t / len(sops), f"mock: dynamic programming {t + 1}/{len(sops)}")
    j = min(range(len(best[-1])), key=lambda k: best[-1][k][0])
    total = best[-1][j][0]
    path = [j]
    for t in range(len(sops) - 1, 0, -1):
        j = best[t][j][1]
        path.append(j)
    path.reverse()
    chosen = [cands[t][path[t]] for t in range(len(sops))]
    voicings = [c[2] for c in chosen]
    details = []
    for t in range(1, len(voicings)):
        for i, j2 in p5_mock_parallels(voicings[t - 1], voicings[t]):
            details.append({"rule": "parallel fifths/octaves", "slot": t, "voices": [i, j2]})
    result = {"voicings": voicings, "chords": [c[1] for c in chosen], "durations": [b for _, b in notes],
              "key": key, "time_sig": melody.get("time_sig"), "pickup": melody.get("pickup"), "tempo": melody.get("tempo"),
              "title": melody.get("title")}
    report = {"is_mock": True, "solver": f"mock classical DP (requested: {solver})", "backend": "none (no quantum circuit)",
              "violations": len(details), "violation_details": details, "energy": round(float(total), 2),
              "notes": ["Placeholder harmonizer used while the real pipeline is not defined."]}
    return result, report
