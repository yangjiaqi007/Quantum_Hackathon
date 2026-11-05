"""Dev tests for Part 5: (A) static name check, (B) pollution + clobber test, (C) behaviour tests."""
import ast, os, sys, symtable, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_loader

fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

# ---------- A. static: every module-level binding is p5_/P5-prefixed
allowed_prefix = ("p5_", "P5")
bad = []
n_names = 0
for path in p5_loader.source_files():
    for kind, title, src in p5_loader.split_cells(open(path, encoding="utf-8").read()):
        if kind != "code": continue
        top = symtable.symtable(src, "<cell>", "exec")
        for sym in top.get_symbols():
            if sym.is_assigned() or sym.is_imported() or sym.is_namespace():
                n_names += 1
                if not sym.get_name().startswith(allowed_prefix):
                    bad.append((os.path.basename(path), sym.get_name()))
check(not bad, f"A1 all {n_names} module-level names are p5_/P5-prefixed {bad[:5]}")

# i18n keys used exist
keys_used = set()
for path in p5_loader.source_files():
    keys_used |= set(re.findall(r'p5_t\(\s*"([a-z0-9_]+)"', open(path, encoding="utf-8").read()))
    keys_used |= set(re.findall(r'"(?:label_key)":\s*"([a-z0-9_]+)"', open(path, encoding="utf-8").read()))
    keys_used |= set(re.findall(r'\("(rep_[a-z0-9_]+)",', open(path, encoding="utf-8").read()))
    keys_used |= set(re.findall(r'\(\s*"(tab_[a-z]+)"', open(path, encoding="utf-8").read()))
ns0 = p5_loader.load()
missing = sorted(k for k in keys_used if k not in ns0["P5_STR"])
check(not missing, f"A2 all UI string keys exist {missing}")
zh_missing = [k for k, v in ns0["P5_STR"].items() if "zh" not in v]
check(not zh_missing, f"A3 every string has a Chinese version {zh_missing}")

# ---------- B. pollution: pre-define teammate-style names with junk; P5 must neither use nor overwrite them
junk_names = ("np plt display HTML Audio Image widgets music21 m21 io os re math json time base64 html wave struct "
              "h W candidates Voicing StyleWeights PRESETS harmonize solve main key melody result report fig ax buf data "
              "notes score qc circuit solver style Figure Path Circle Rectangle Ellipse dataclass field inspect tempfile").split()
ns = {"__name__": "__main__"}
for j in junk_names: ns[j] = f"JUNK<{j}>"
before = dict(ns)
p5_loader.load(ns)
clobbered = [k for k in before if ns[k] != before[k]]
newnames = [k for k in ns if k not in before and not k.startswith(("p5_", "P5", "__"))]
check(not clobbered, f"B1 no pre-existing teammate name was overwritten {clobbered}")
check(not newnames, f"B2 no non-prefixed name was introduced {newnames}")

# ---------- C. behaviour (using the polluted namespace on purpose)
m = ns["p5_preset_melody"]("ode_to_joy")
req = ns["P5Request"](melody=m, solver="xy_qaoa", options=dict(ns["P5_DEFAULT_OPTIONS"]), tempo=m["tempo"])
run = ns["p5_run_request"](req)
check(not run.error, f"C1 mock pipeline runs on Ode to Joy ({run.error})")
sc = run.score
check(len(sc.slots) == 30 and all(s.pitches[3] == n for s, (n, _) in zip(sc.slots, m["notes"])), "C2 soprano preserved, 30 slots")
check(run.warnings == [], f"C3 no sanity warnings {run.warnings}")
art = ns["p5_render_artifacts"](run)
check(art["png"][:4] == b"\x89PNG" and art["wav_full"][:4] == b"RIFF", "C4 PNG + WAV rendered")
import wave, io
w = wave.open(io.BytesIO(art["wav_full"])); dur = w.getnframes() / w.getframerate()
exp = sum(s.beats for s in sc.slots) * 60 / sc.tempo
check(exp < dur < exp + 1.2, f"C5 WAV duration {dur:.2f}s ~ expected {exp:.2f}s")
import numpy as np
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
check(np.abs(pcm).max() < 32767 and np.abs(pcm).max() > 8000, f"C6 audio not silent / not clipping (peak {np.abs(pcm).max()})")

# export round trip
import music21
mid = ns["p5_m21_bytes"](sc, "midi"); xml = ns["p5_m21_bytes"](sc, "musicxml")
with open("/tmp/rt.mid", "wb") as f: f.write(mid)
with open("/tmp/rt.musicxml", "wb") as f: f.write(xml)
for label, path in (("MIDI", "/tmp/rt.mid"), ("MusicXML", "/tmp/rt.musicxml")):
    s2 = music21.converter.parse(path)
    parts = list(s2.parts)
    sop = [n.pitch.midi for n in parts[0].flatten().notes]
    bass = [n.pitch.midi for n in parts[-1].flatten().notes]
    check(len(parts) == 4 and sop == [s.pitches[3] for s in sc.slots] and bass == [s.pitches[0] for s in sc.slots],
          f"C7 {label} round trip: 4 parts, soprano+bass pitches identical")
dur_rt = [n.quarterLength for n in music21.converter.parse("/tmp/rt.musicxml").parts[0].flatten().notes]
check([float(d) for d in dur_rt] == [s.beats for s in sc.slots], "C8 MusicXML durations identical (dotted notes)")

# pickup + 3/4 + flats export
P = ns["P5Slot"]
sc2 = ns["P5Score"](slots=[P((46,53,58,70),1.0,"I"), P((46,55,62,74),2.0,"I"), P((41,57,60,72),3.0,"IV"), P((43,55,64,76),4.0,"V")],
                    key="Bb", time_sig=(3,4), pickup=1.0, tempo=80, title="pickup")
xml2 = ns["p5_m21_bytes"](sc2, "musicxml")
s3 = music21.converter.parse(xml2 if False else (open("/tmp/p.musicxml","wb").write(xml2) and "/tmp/p.musicxml"))
nums = [m_.number for m_ in s3.parts[0].getElementsByClass("Measure")]
check(nums[0] == 0 and s3.parts[0].flatten().getElementsByClass("KeySignature")[0].sharps == -2, f"C9 pickup measure numbered 0, key Bb ({nums})")

# input handling
notes, errs = ns["p5_parse_melody_text"]("C4 D4 E4:1.5 Bb3 F#5 | G4:1/2 X9")
check(len(notes) == 6 and len(errs) == 1 and notes[2][1] == 1.5 and notes[3][0] == 58 and notes[4][0] == 78, f"C10 text parser {notes} {errs}")
check(ns["p5_guess_key"]([(67,1),(69,1),(71,1),(72,1),(74,1),(67,1)]) == "G", "C11 key guess G major")
e, _ = ns["p5_validate_melody"]([(60,1)] ); check(bool(e), "C12 validation: too few notes")
e, _ = ns["p5_validate_melody"]([(30,1),(60,1)]); check(bool(e), "C13 validation: out of range")

# adapter tolerant to signatures
calls = []
def fake_a(selected_melody, style_weight, solver_type):
    calls.append(("a", sorted(locals()))); return ([(40,48,55,60)]*2, {"violations": 0})
def fake_b(melody, style="x", solver="y", progress=None, **kw):
    calls.append(("b", sorted(kw))); return {"voicings": [(40,48,55,60)]*2}
def fake_c(melody):
    return ({"voicings": [[60,55,48,40]]*2}, {})   # reversed order S,A,T,B as list
for name, fn in (("a", fake_a), ("b", fake_b), ("c", fake_c)):
    ns["harmonize_pipeline"] = fn
    r = ns["p5_run_request"](ns["P5Request"](melody=ns["p5_make_melody"]([(60,1),(60,1)], key="C"), options=dict(ns["P5_DEFAULT_OPTIONS"])))
    check(not r.error and len(r.score.slots) == 2, f"C14{name} adapter handles signature {name} ({r.error})")
check(calls[1][1] == ["layers_p", "noisy", "shots", "window_size"] or "shots" in calls[1][1], f"C15 **kwargs receives options {calls[1]}")
r = ns["p5_run_request"](ns["P5Request"](melody=ns["p5_make_melody"]([(60,1),(60,1)], key="C")))
check(any("reversed" in w for w in r.warnings) , f"C16 reversed voicing order detected {r.warnings}")
ns["harmonize_pipeline"] = lambda melody: (_ for _ in ()).throw(ValueError("boom"))
r = ns["p5_run_request"](ns["P5Request"](melody=ns["p5_make_melody"]([(60,1),(60,1)], key="C")))
check("boom" in r.error and r.score is None, "C17 pipeline exception is captured, not raised")
del ns["harmonize_pipeline"]

# report html with full synthetic report
rep = {"solver": "xy_qaoa", "backend": "aer", "qubits": 12, "depth": 38, "two_qubit_gates": 24, "shots": 2048, "layers_p": 2,
       "violations": 1, "violation_details": [{"rule": "parallel fifths", "slot": 4, "voices": [0, 1]}],
       "energy": 12.5, "classical_energy": 11.0, "windows": [{"start": 0, "end": 3, "qubits": 12, "depth": 38, "gates": 24, "energy": 5.0, "classical_energy": 4.5}, {"start": 3, "end": 6, "qubits": 12, "depth": 40, "gates": 26, "energy": 7.5, "classical_energy": 6.5}],
       "hardware": {"backend": "ibm_x", "job_id": "abc"}, "verdict": "classical wins", "notes": ["n1 <b>"]}
html = ns["p5_report_html"](rep, sc, "harmonize_pipeline", 1.2)
check("&lt;b&gt;" in html and "12" in html and "Recorded hardware" in html and "MOCK" not in html, "C18 report html (escaped, hardware card, no mock banner)")
check(ns["p5_energy_chart_png"](rep)[:4] == b"\x89PNG", "C19 energy chart")
check(ns["p5_marks_from_report"](rep) == {4: {0, 1}} and ns["p5_windows_from_report"](rep) == [(0, 3), (3, 6)], "C20 marks/windows from report")
from qiskit import QuantumCircuit
qc = QuantumCircuit(3); qc.h(0); qc.cx(0, 1); qc.rzz(0.3, 1, 2)
check((ns["p5_circuit_png"](qc) or b"")[:4] == b"\x89PNG", "C21 circuit PNG via qiskit mpl drawer")

# UI construction + programmatic click
root = ns["p5_build_ui"]()
ui = ns["P5_UI"]
req2 = ns["p5_ui_collect"](ui)
check(req2.melody["notes"] and len(req2.melody["notes"]) == 30 and req2.solver == "xy_qaoa", "C22 UI collects preset request")
ui["melody"].value = "text"
req3 = ns["p5_ui_collect"](ui)
check(req3.melody["source"] == "text" and req3.melody["key"] == "C", f"C23 UI collects custom text, key auto -> {req3.melody['key']}")
ui["keys"][0].click(); check(ui["text"].value.endswith("C4"), "C24 palette button appends a note")
ui["text"].value = "C4 Q9"
try: ns["p5_ui_collect"](ui); ok = False
except ValueError: ok = True
check(ok, "C25 bad text raises a friendly ValueError")
ui["text"].value = "G4 A4 B4 C5 D5 E5 D5 C5"; ui["melody"].value = "text"
ui["run"].click()
check(ns["P5_STATE"]["status"] == "ok" and ns["P5_STATE"]["last_run"].score.key == "C", f"C26 run button end-to-end ({ns['P5_STATE']['status']})")
ui["melody"].value = "o_canada"; ui["run"].click()
check(ns["P5_STATE"]["status"].startswith("error") and "Part 1" in ns["P5_STATE"]["status"], "C27 o_canada with mock gives clear error")
ns["P5_LANG"] = "zh"; ns["p5_build_ui"](); ns["P5_LANG"] = "en"
check(ns["p5_selfcheck"](run_pipeline=True), "C28 p5_selfcheck passes")

print("\nFAILED:" if fails else "\nALL PASSED", fails if fails else "")
sys.exit(1 if fails else 0)
