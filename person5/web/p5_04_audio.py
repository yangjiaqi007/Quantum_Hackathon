# %% P5-04 Audio: offline additive synthesis -> WAV bytes -> <audio> player
# A .mid file cannot be played by a browser <audio> tag, so the notebook synthesizes the chorale itself
# with numpy (no soundfont / FluidSynth needed). MIDI is still exported for DAWs and MuseScore.
P5_HARMONICS = (1.0, 0.5, 0.33, 0.22, 0.12, 0.07)       # organ/choir-like spectrum
P5_VOICE_GAIN = {0: 1.0, 1: 0.8, 2: 0.8, 3: 0.95}


def p5_midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def p5_synth_wav(sc, voices=(0, 1, 2, 3), tempo=None, sr=None):
    """Synthesize the chosen voices (indices into pitches) and return mono 16-bit WAV bytes."""
    sr = sr or P5_SAMPLE_RATE
    tempo = tempo or sc.tempo or 90
    spb = 60.0 / tempo
    tail = 0.25
    total = sum(s.beats for s in sc.slots) * spb + tail + 0.3
    out = p5_np.zeros(int(total * sr) + 1)
    t0 = 0.0
    for s in sc.slots:
        dur = s.beats * spb
        n = int((dur + tail) * sr)
        tt = p5_np.arange(n) / sr
        env = p5_np.ones(n)
        att = max(1, int(0.03 * sr))
        env[:att] = p5_np.linspace(0.0, 1.0, att)
        rel0 = int(dur * 0.94 * sr)
        env[rel0:] = p5_np.linspace(1.0, 0.0, n - rel0) ** 2
        i0 = int(t0 * sr)
        for v in voices:
            if v == 3 and getattr(s, "soprano_rest", False):
                continue
            f = p5_midi_hz(s.pitches[v])
            wave = sum(a * p5_np.sin(2 * p5_math.pi * f * (k + 1) * tt)
                       for k, a in enumerate(P5_HARMONICS) if f * (k + 1) < sr * 0.45)
            seg = wave * env * P5_VOICE_GAIN[v]
            out[i0:i0 + n] += seg[: len(out) - i0]
        t0 += dur
    peak = float(p5_np.max(p5_np.abs(out))) or 1.0
    out = p5_np.tanh(1.4 * out / peak) / p5_np.tanh(1.4) * 0.85
    pcm = (out * 32767).astype("<i2")
    buf = p5_io.BytesIO()
    with p5_wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def p5_audio_html(wav_bytes, label=""):
    """Self-contained <audio> element (data URI), visible in Jupyter, VS Code and nbviewer."""
    b64 = p5_b64.b64encode(wav_bytes).decode("ascii")
    return ('<div style="font-family:system-ui,Arial,sans-serif;font-size:13px;margin:6px 0;color:#222">'
            f'<div style="margin-bottom:2px"><b>{p5_html.escape(label)}</b></div>'
            f'<audio controls preload="auto" src="data:audio/wav;base64,{b64}"></audio></div>')
