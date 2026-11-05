# %% P5-07 Pipeline adapter: the ONE place where Part 5 calls the team's code
# Canonical contract (Parts 1-4 provide it):
#     harmonize_pipeline(melody, style="bach_chorale", style_weight=1.0, solver="xy_qaoa", progress=None, **options)
#         -> (result, report)
# melody  : dict from p5_make_melody (title, preset, notes=[(midi, beats)] or None, key, time_sig, pickup, tempo)
# result  : list of (bass, tenor, alto, soprano) MIDI tuples | dict {'voicings': [...], 'chords': [...], ...} | 4-part music21 Stream
# report  : dict, see cell P5-06.
# Parameter names may differ: only the parameters the function actually accepts are passed (see P5_ARG_ALIASES).
P5_ARG_ALIASES = {
    "style": ("style", "style_name"),
    "style_weight": ("style_weight", "weight", "strictness"),
    "solver": ("solver", "solver_type", "method"),
    "progress": ("progress", "progress_cb", "callback", "on_progress"),
    "layers_p": ("layers_p", "p", "reps", "layers"),
    "shots": ("shots", "n_shots"),
    "window_size": ("window_size", "window", "window_slots"),
    "noisy": ("noisy", "noise", "use_noise"),
}


@p5_dataclass
class P5Request:
    melody: dict
    style: str = "bach_chorale"
    style_weight: float = 1.0
    solver: str = "xy_qaoa"
    options: dict = p5_field(default_factory=dict)      # layers_p, shots, window_size, noisy
    tempo: int = 90


@p5_dataclass
class P5Run:
    request: P5Request
    score: object = None                                 # P5Score
    report: dict = p5_field(default_factory=dict)
    warnings: list = p5_field(default_factory=list)
    error: str = ""
    traceback: str = ""
    seconds: float = 0.0
    pipeline_name: str = ""
    artifacts: dict = p5_field(default_factory=dict)      # rendered PNG / WAV, filled by P5-08


def p5_find_pipeline():
    """(name, function): the team's harmonize_pipeline / harmonize if defined, else the mock."""
    g = globals()
    for name in ("harmonize_pipeline", "harmonize"):
        fn = g.get(name)
        if callable(fn):
            return name, fn
    fn = g.get("p5_mock_pipeline")
    if callable(fn):
        return "p5_mock_pipeline", fn
    raise RuntimeError("No pipeline found: define harmonize_pipeline(melody, ...) or keep cell P5-10 (mock).")


def p5_invoke_pipeline(fn, req, progress=None):
    values = {"style": req.style, "style_weight": req.style_weight, "solver": req.solver, "progress": progress, **req.options}
    params = p5_inspect.signature(fn).parameters
    first = next(iter(params), None)
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    ok_kinds = (p5_inspect.Parameter.POSITIONAL_OR_KEYWORD, p5_inspect.Parameter.KEYWORD_ONLY)
    kwargs = {}
    for canon, aliases in P5_ARG_ALIASES.items():
        val = values.get(canon)
        if val is None:
            continue
        for al in aliases:
            if al in params and al != first and params[al].kind in ok_kinds and al not in kwargs:
                kwargs[al] = val
                break
        else:
            if has_var_kw:
                kwargs[canon] = val
    return fn(req.melody, **kwargs)


def p5_split_return(raw):
    """Accept (result, report) | {'result':..., 'report':...} | result."""
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict) and "pitches" not in raw[1]:
        return raw[0], raw[1]
    if isinstance(raw, dict) and "report" in raw:
        rep = raw["report"]
        res = raw.get("result", {k: v for k, v in raw.items() if k != "report"})
        return res, rep
    return raw, {}


def p5_run_request(req, progress=None):
    """Call the pipeline and normalize its answer. Never raises: errors are stored in the returned P5Run."""
    run = P5Run(request=req)
    t0 = p5_time.time()
    try:
        name, fn = p5_find_pipeline()
        run.pipeline_name = name
        raw = p5_invoke_pipeline(fn, req, progress)
        result, report = p5_split_return(raw)
        sc = p5_normalize_result(result, req.melody)
        sc.tempo = int(req.tempo or sc.tempo)
        run.score, run.report, run.warnings = sc, dict(report or {}), list(sc.warnings)
        if name == "p5_mock_pipeline":
            run.report["is_mock"] = True
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        run.traceback = p5_tb.format_exc()
    run.seconds = p5_time.time() - t0
    return run
