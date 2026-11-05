# %% P5-15 Local FastAPI bridge: the interactive UI can submit real team-pipeline jobs.
import json as p5_json
import urllib.error as p5_urllib_error
import urllib.request as p5_urllib_request

P5_API_BASE_URL = p5_os.environ.get("P5_API_URL", "http://127.0.0.1:8000").rstrip("/")


def p5_api_json(method, path, payload=None, timeout=20):
    """Small stdlib HTTP client so the notebook does not need another dependency."""
    body = None if payload is None else p5_json.dumps(payload).encode("utf-8")
    request = p5_urllib_request.Request(P5_API_BASE_URL + path, data=body, method=method,
                                        headers={"Content-Type": "application/json"})
    try:
        with p5_urllib_request.urlopen(request, timeout=timeout) as response:
            return p5_json.loads(response.read().decode("utf-8"))
    except p5_urllib_error.HTTPError as exc:
        try:
            detail = p5_json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = str(exc.reason)
        if isinstance(detail, dict):
            detail = detail.get("detail", detail)
        raise RuntimeError(f"Quantum API returned HTTP {exc.code}: {detail}") from None
    except (p5_urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach {P5_API_BASE_URL}. Start the local API in a separate terminal. ({exc})") from None


def p5_api_harmonize_pipeline(req, progress=None):
    """Submit one melody to the async API, poll it, and adapt its response to Part 5."""
    notes = req.melody.get("notes")
    if not notes:
        raise ValueError("This melody preset has no note list supported by the API yet. Choose Ode to Joy or enter a melody.")
    if req.solver not in ("qaoa", "xy_qaoa"):
        raise ValueError("The API supports QAOA solvers only. Select QAOA or XY-QAOA.")
    backend = req.options.get("backend", "aer")
    style = {"bach_chorale": "chorale", "strict_counterpoint": "strict", "jazz_loose": "jazz_loose"}.get(req.style)
    if style is None:
        raise ValueError(f"Unsupported style for the API: {req.style}")
    rest_flags = [midi is None for midi, _ in notes]
    carried = []
    previous = next((int(midi) for midi, _ in notes if midi is not None), None)
    for midi, _duration in notes:
        if midi is not None:
            previous = int(midi)
        if previous is None:
            raise ValueError("A melody cannot begin with a rest.")
        carried.append(previous)
    melody = {
        "title": req.melody.get("title") or "Untitled",
        "tonic": int(p5_key_tonic_pc(req.melody.get("key") or "C")),
        "mode": req.melody.get("mode", "major"),
        "time_signature": list(req.melody.get("time_sig") or (4, 4)),
        # Keep each melody event intact in the returned score (for example, a dotted quarter).
        # Melody-note boundaries still create chord slots; the four-beat pulse avoids splitting
        # an event at an unrelated quarter-beat boundary.
        "harmonic_rhythm": 4.0,
        "notes": [{"midi": midi, "duration": float(duration)} for midi, duration in zip(carried, (d for _, d in notes))],
    }
    payload = {"melody": melody, "tempo": int(req.tempo), "style": style,
               "run": {"backend": backend, "solver": req.solver, "shots": int(req.options.get("shots", 2048)),
                       "reps": int(req.options.get("layers_p", 2)), "window_size": int(req.options.get("window_size", 3)),
                       "max_qubits": 16, "max_iterations": 50, "seed": 42},
               # A hardware selection should always submit a fresh provider job.
               "use_cache": backend != "ibm"}
    job = p5_api_json("POST", "/jobs", payload, timeout=30)
    job_id = job.get("job_id")
    if not job_id:
        raise RuntimeError("API accepted the request but returned no job ID.")
    started = p5_time.monotonic()
    while p5_time.monotonic() - started < 315:
        state = p5_api_json("GET", f"/jobs/{job_id}", timeout=20)
        status = state.get("status")
        detail = state.get("progress") or {}
        done, total = detail.get("completed_windows"), detail.get("total_windows")
        frac = (float(done) / float(total)) if isinstance(done, (int, float)) and total else (0.05 if status == "queued" else 0.1)
        if progress:
            progress(min(0.98, frac), detail.get("stage") or f"Quantum API job {status}")
        if status == "failed":
            error = state.get("error") or {}
            raise RuntimeError(error.get("message") or "Quantum API job failed.")
        if status == "cancelled":
            raise RuntimeError("Quantum API job was cancelled.")
        if status == "completed":
            envelope = p5_api_json("GET", f"/jobs/{job_id}/result", timeout=30)
            result = envelope.get("result") or {}
            slots = result.get("slots") or []
            if not slots:
                raise RuntimeError("API job completed but returned no score slots.")
            if progress:
                progress(1.0, "Quantum run complete")
            return p5_api_adapt_result(result, envelope.get("provenance") or {}, req, job_id)
        p5_time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for API job {job_id}; check {P5_API_BASE_URL}/jobs/{job_id}.")


def p5_api_expand_slots(slots, input_notes):
    """Expand API slots against the melody while preserving soprano rests."""
    voicings = [slot["pitches"] for slot in slots]
    durations = [slot["duration"] for slot in slots]
    chords = [slot.get("chord", "") for slot in slots]
    melody_slots, rest_slots = [], []
    note_index, remaining = 0, (float(input_notes[0][1]) if input_notes else 0.0)
    for slot in slots:
        while note_index < len(input_notes) and remaining <= 1e-7:
            note_index += 1
            if note_index < len(input_notes):
                remaining = float(input_notes[note_index][1])
        if note_index >= len(input_notes):
            raise RuntimeError("API returned more chord duration than the input melody contains.")
        duration = float(slot["duration"])
        source_soprano = input_notes[note_index][0]
        is_rest = source_soprano is None
        soprano = (next((int(m) for m, _ in reversed(input_notes[:note_index]) if m is not None), None)
                   if is_rest else int(source_soprano))
        if soprano is None:
            raise RuntimeError("Melody cannot begin with a rest.")
        if int(slot["pitches"][3]) != soprano:
            raise RuntimeError(f"API changed the soprano melody at chord slot {len(melody_slots) + 1}.")
        if duration > remaining + 1e-7:
            raise RuntimeError("API returned a chord slot that crosses a melody note boundary.")
        melody_slots.append((soprano, duration))
        rest_slots.append(is_rest)
        remaining -= duration
    while note_index < len(input_notes) and remaining <= 1e-7:
        note_index += 1
        if note_index < len(input_notes):
            remaining = float(input_notes[note_index][1])
    if note_index != len(input_notes):
        raise RuntimeError("API chord slots do not cover the complete input melody.")
    return voicings, durations, chords, melody_slots, rest_slots


def p5_api_adapt_result(result, provenance, req, job_id):
    slots = result.get("slots") or []
    input_notes = req.melody.get("notes") or []
    voicings, durations, chords, melody_slots, rest_slots = p5_api_expand_slots(slots, input_notes)
    alternatives = []
    for alternative in (result.get("alternatives") or [])[:2]:
        alt_slots = alternative.get("slots") or []
        if not alt_slots:
            continue
        av, ad, ac, am, ar = p5_api_expand_slots(alt_slots, input_notes)
        alternatives.append({"voicings": av, "durations": ad, "chords": ac,
                            "_p5_input_notes": am, "_p5_soprano_rests": ar,
                            "energy": alternative.get("energy"),
                            "quality_report": alternative.get("quality_report") or {},
                            "window_energies": alternative.get("window_energies") or [],
                            "rank": alternative.get("rank")})
    execution = result.get("execution") or {}
    qreport = result.get("quantum_report") or {}
    quality = result.get("quality_report") or {}
    raw_windows = qreport.get("windows") or []
    windows = []
    resources = []
    for item in raw_windows:
        resource = item.get("resources") or {}
        if resource:
            resources.append(resource)
        windows.append({"start": item.get("start"), "end": item.get("stop"),
                        "qubits": resource.get("n_qubits"), "depth": resource.get("depth"),
                        "two_qubit_gates": resource.get("two_qubit_gates"), "energy": item.get("energy")})
    counts = quality.get("rule_violation_counts")
    violations = sum(counts.values()) if isinstance(counts, dict) else None
    report = {
        "solver": req.solver, "backend": execution.get("mode", req.options.get("backend", "aer")),
        "is_mock": bool(execution.get("is_mock", False)),
        "qubits": qreport.get("max_qubits_per_window"),
        "depth": max((r.get("depth", 0) for r in resources), default=None),
        "two_qubit_gates": max((r.get("two_qubit_gates", 0) for r in resources), default=None),
        "shots": req.options.get("shots"), "layers_p": req.options.get("layers_p"),
        "runtime_s": execution.get("elapsed_seconds"), "violations": violations,
        "energy": quality.get("energy"), "classical_energy": quality.get("optimal_energy"),
        "approx_ratio": None, "p_optimum": None,
        "feasible_fraction": min((w.get("feasible_fraction", 1.0) for w in raw_windows), default=None),
        "windows": windows, "verdict": quality.get("verdict"), "hardware": None,
        "notes": list(result.get("warnings") or []),
    }
    report["notes"].append("Top alternatives are ranked from sampled window solutions; window stitching does not prove a global top-three ranking.")
    report["top_solutions"] = [{"rank": 1, "energy": quality.get("energy")}] + [
        {"rank": item.get("rank"), "energy": item.get("energy"),
         "quality_report": item.get("quality_report"),
         "window_energies": item.get("window_energies")} for item in alternatives]
    report["notes"].append("Circuit diagrams are not included in the API response; reported resources are from transpiled windows.")
    if execution.get("mode") == "ibm":
        report["notes"].append(f"Local API job ID: {job_id}; IBM provider job ID is not exposed by this API.")
    if violations is None:
        report["notes"].append("No independent musical rule checker is configured; rule violations are unknown.")
    if provenance.get("cache_hit"):
        report["notes"].append("Returned from the API result cache.")
    return ({"voicings": voicings, "durations": durations, "chords": chords,
             "key": req.melody.get("key", "C"), "time_sig": req.melody.get("time_sig", (4, 4)),
             "tempo": req.tempo, "title": req.melody.get("title", ""),
             "_p5_input_notes": melody_slots, "_p5_soprano_rests": rest_slots,
             "_p5_alternatives": alternatives}, report)
