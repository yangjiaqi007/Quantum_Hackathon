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
    solver: str = "qaoa"
    options: dict = p5_field(default_factory=dict)      # layers_p, shots, window_size, noisy
    tempo: int = 90
    inversions: list = p5_field(default_factory=list)


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
    alternatives: list = p5_field(default_factory=list)   # ranked sampled harmony variants


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
        if req.options.get("use_api"):
            name = "person4 FastAPI QAOA service"
            run.pipeline_name = name
            raw = p5_api_harmonize_pipeline(req, progress)
        else:
            name, fn = p5_find_pipeline()
            run.pipeline_name = name
            solver_melody = dict(req.melody)
            rest_flags = [m is None for m, _ in (req.melody.get("notes") or [])]
            if any(rest_flags):
                carried, previous = [], None
                for midi, duration in req.melody["notes"]:
                    if midi is not None:
                        previous = int(midi)
                    if previous is None:
                        raise ValueError("A melody cannot begin with a rest.")
                    carried.append((previous, duration))
                solver_melody["notes"] = carried
            solver_req = P5Request(melody=solver_melody, style=req.style, style_weight=req.style_weight,
                                   solver=req.solver, options=req.options, tempo=req.tempo,
                                   inversions=list(req.inversions))
            raw = p5_invoke_pipeline(fn, solver_req, progress)
        result, report = p5_split_return(raw)
        score_melody = req.melody
        if req.options.get("use_api") and isinstance(result, dict) and result.get("_p5_input_notes") is not None:
            score_melody = dict(req.melody)
            score_melody["notes"] = result["_p5_input_notes"]
        elif any(m is None for m, _ in (req.melody.get("notes") or [])):
            score_melody = dict(req.melody)
            score_melody["notes"] = solver_melody["notes"]
        sc = p5_normalize_result(result, score_melody)
        if req.options.get("use_api") and isinstance(result, dict):
            rest_flags = result.get("_p5_soprano_rests") or []
            if len(rest_flags) != len(sc.slots):
                raise RuntimeError("API soprano rest map does not match the returned chord slots.")
        if len(rest_flags) == len(sc.slots):
            for slot, is_rest in zip(sc.slots, rest_flags):
                slot.soprano_rest = bool(is_rest)
        if req.options.get("use_api") and isinstance(result, dict):
            for alternative in result.get("_p5_alternatives") or []:
                alt_melody = dict(req.melody)
                alt_melody["notes"] = alternative["_p5_input_notes"]
                alt_score = p5_normalize_result(alternative, alt_melody)
                alt_rests = alternative.get("_p5_soprano_rests") or []
                if len(alt_rests) != len(alt_score.slots):
                    raise RuntimeError("Alternative soprano rest map does not match the returned chord slots.")
                for slot, is_rest in zip(alt_score.slots, alt_rests):
                    slot.soprano_rest = bool(is_rest)
                alt_score.tempo = int(req.tempo or alt_score.tempo)
                alt_report = dict(report or {})
                alt_report["energy"] = alternative.get("energy")
                alt_report["quality_report"] = alternative.get("quality_report") or {}
                alt_report["rank"] = alternative.get("rank")
                run.alternatives.append(P5Run(request=req, score=alt_score, report=alt_report,
                                              warnings=list(alt_score.warnings), seconds=run.seconds,
                                              pipeline_name=name))
        sc.tempo = int(req.tempo or sc.tempo)
        run.score, run.report, run.warnings = sc, dict(report or {}), list(sc.warnings)
        if name == "p5_mock_pipeline":
            run.report["is_mock"] = True
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        run.traceback = p5_tb.format_exc()
    run.seconds = p5_time.time() - t0
    return run
