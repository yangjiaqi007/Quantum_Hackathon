# %% [markdown]
# # Part 5 · Interface, score, audio, export & quantum report
#
# Cells **P5-00 … P5-11** belong to Person 5. Every top-level name they define starts with
# `p5_` / `P5` (including import aliases), so they can be pasted into the team notebook
# without clashing with Parts 1-4. They only *call* the team pipeline:
#
# ```python
# harmonize_pipeline(melody, style=..., style_weight=..., solver=..., progress=...) -> (result, report)
# ```
#
# If `harmonize_pipeline` is not defined yet, cell **P5-10** supplies a clearly labelled *mock*
# so the interface can be tested. Delete that cell when the real pipeline is in the notebook.
# Run `p5_selfcheck()` (cell P5-11) after merging.

# %% P5-00 Setup: imports, configuration, UI strings
import base64 as p5_b64
import datetime as p5_dt
import html as p5_html
import inspect as p5_inspect
import io as p5_io
import math as p5_math
import os as p5_os
import re as p5_re
import tempfile as p5_tmp
import time as p5_time
import traceback as p5_tb
import wave as p5_wave
from dataclasses import dataclass as p5_dataclass, field as p5_field

import numpy as p5_np
import ipywidgets as p5_w
from IPython.display import display as p5_display, HTML as p5_HTML, Image as p5_Image
from matplotlib.figure import Figure as p5_Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as p5_Canvas
from matplotlib.patches import Ellipse as p5_Ellipse, Rectangle as p5_Rect, Circle as p5_Circle, PathPatch as p5_PathPatch
from matplotlib.path import Path as p5_Path

try:
    import music21 as p5_m21          # needed for MIDI / MusicXML export and MIDI upload
except Exception:
    p5_m21 = None

P5_LANG = "en"                        # "en" or "zh" - change before calling p5_launch()
P5_OUT_DIR = "p5_output"              # exported files are saved here
P5_MAX_NOTES = 40                     # cap for custom melodies typed into the UI
P5_SAMPLE_RATE = 16000                # audio sample rate (Hz)

P5_STR = {
    "title": {"en": "Quantum Four-Part Harmony Generator", "zh": "量子四部和声生成器"},
    "subtitle": {
        "en": "Enter a soprano melody → harmony rules are compiled into a Hamiltonian → a Qiskit solver returns alto, tenor and bass.",
        "zh": "输入女高音旋律 → 和声规则被编译为哈密顿量 → Qiskit 求解器返回中音、次中音与低音。"},
    "sec_melody": {"en": "1 · Melody", "zh": "1 · 旋律"},
    "sec_style": {"en": "2 · Style & solver", "zh": "2 · 风格与求解器"},
    "melody": {"en": "Melody", "zh": "选择旋律"},
    "opt_ode": {"en": "Ode to Joy (Beethoven)", "zh": "欢乐颂 (Beethoven)"},
    "opt_canada": {"en": "O Canada", "zh": "加拿大国歌 (O Canada)"},
    "opt_custom": {"en": "Custom notes (type or click)", "zh": "自定义音符（输入或点击）"},
    "opt_midi": {"en": "Upload a MIDI file", "zh": "上传 MIDI 文件"},
    "custom_hint": {
        "en": "Note names such as C4 F#4 Bb3; optional duration after ':' in beats (E4:1.5, D4:0.5); '|' is ignored.",
        "zh": "音名如 C4 F#4 Bb3；冒号后可写时值（拍），如 E4:1.5、D4:0.5；'|' 会被忽略。"},
    "pal_dur": {"en": "Duration (beats)", "zh": "时值（拍）"},
    "pal_acc": {"en": "Accidental", "zh": "升降号"},
    "btn_undo": {"en": "Undo", "zh": "撤销"},
    "btn_clear": {"en": "Clear", "zh": "清空"},
    "key_label": {"en": "Key (major)", "zh": "调性（大调）"},
    "key_auto": {"en": "Auto-detect", "zh": "自动识别"},
    "ts_label": {"en": "Time signature", "zh": "拍号"},
    "pickup_label": {"en": "Pickup beats", "zh": "弱起拍数"},
    "style_label": {"en": "Style preset", "zh": "风格预设"},
    "strict_label": {"en": "Rule strictness", "zh": "规则权重倍数"},
    "solver_label": {"en": "Solver", "zh": "求解器"},
    "s_viterbi": {"en": "viterbi - classical baseline (exact)", "zh": "viterbi - 经典基准（精确）"},
    "s_qaoa": {"en": "qaoa - standard X mixer", "zh": "qaoa - 标准 X 混合器"},
    "s_xy": {"en": "xy_qaoa - constraint-preserving XY mixer", "zh": "xy_qaoa - 约束保持 XY 混合器"},
    "adv_title": {"en": "Advanced quantum settings (passed to the pipeline if it accepts them)",
                  "zh": "高级量子设置（流程支持时才会传入）"},
    "adv_p": {"en": "QAOA layers p", "zh": "QAOA 层数 p"},
    "adv_shots": {"en": "Shots", "zh": "采样次数"},
    "adv_window": {"en": "Window size (slots)", "zh": "窗口大小（和弦数）"},
    "adv_noisy": {"en": "Noisy fake backend", "zh": "带噪声的假后端"},
    "tempo_label": {"en": "Tempo (BPM)", "zh": "速度 (BPM)"},
    "btn_preview": {"en": "Preview melody", "zh": "试听旋律"},
    "btn_run": {"en": "Run harmonization", "zh": "运行和声生成"},
    "status_ready": {"en": "Ready.", "zh": "就绪。"},
    "status_running": {"en": "Compiling rules and running the solver...", "zh": "正在编译规则并运行求解器..."},
    "status_done": {"en": "Done in {sec:.1f} s.", "zh": "完成，用时 {sec:.1f} 秒。"},
    "status_error": {"en": "Failed - see the first tab.", "zh": "失败 - 请查看第一个标签页。"},
    "tab_score": {"en": "Score & audio", "zh": "乐谱与音频"},
    "tab_report": {"en": "Quantum report", "zh": "量子报告"},
    "tab_export": {"en": "Export", "zh": "导出"},
    "preview_title": {"en": "Melody preview", "zh": "旋律预览"},
    "parsed": {"en": "Parsed {n} notes · key {key} major · {ts} · {beats:g} beats",
               "zh": "已解析 {n} 个音符 · {key} 大调 · {ts} 拍 · 共 {beats:g} 拍"},
    "no_notes_preset": {"en": "This preset's notes come from the team pipeline (Part 1); nothing to preview here.",
                        "zh": "该预设的音符由团队流程（Part 1）提供，此处无法预览。"},
    "mock_banner": {
        "en": "MOCK BACKEND - a placeholder classical harmonizer is answering. These are NOT quantum results.",
        "zh": "MOCK 后端 - 当前由占位的经典和声器回答，这些不是量子结果。"},
    "warn_title": {"en": "Integration warnings", "zh": "集成警告"},
    "err_title": {"en": "Something went wrong", "zh": "出错了"},
    "audio_full": {"en": "Full chorale (S+A+T+B)", "zh": "完整四部和声"},
    "audio_sop": {"en": "Soprano only (your input)", "zh": "仅女高音（你的输入）"},
    "score_title": {"en": "Four-part harmony", "zh": "四部和声"},
    "viol_marked": {"en": "Notes shown in red take part in a reported rule violation.",
                    "zh": "红色音符参与了被报告的规则违规。"},
    "export_saved": {"en": "Files saved in", "zh": "文件已保存到"},
    "dl_midi": {"en": "Download MIDI", "zh": "下载 MIDI"},
    "dl_xml": {"en": "Download MusicXML", "zh": "下载 MusicXML"},
    "dl_wav": {"en": "Download WAV", "zh": "下载 WAV"},
    "dl_png": {"en": "Download score (PNG)", "zh": "下载乐谱 (PNG)"},
    "no_m21": {"en": "music21 is not installed - MIDI/MusicXML export is unavailable (pip install music21).",
               "zh": "未安装 music21 - 无法导出 MIDI/MusicXML（pip install music21）。"},
    "rep_title": {"en": "Quantum resource & performance report", "zh": "量子资源与性能报告"},
    "rep_na": {"en": "-", "zh": "-"},
    "rep_pipeline": {"en": "Pipeline", "zh": "调用的流程"},
    "rep_solver": {"en": "Solver", "zh": "求解器"},
    "rep_backend": {"en": "Backend", "zh": "后端"},
    "rep_qubits": {"en": "Qubits (max per window)", "zh": "量子比特数（单窗口最大）"},
    "rep_depth": {"en": "Circuit depth", "zh": "电路深度"},
    "rep_2q": {"en": "Two-qubit gates", "zh": "两比特门数量"},
    "rep_shots": {"en": "Shots", "zh": "采样次数"},
    "rep_layers": {"en": "QAOA layers p", "zh": "QAOA 层数 p"},
    "rep_runtime": {"en": "Solver runtime (s)", "zh": "求解耗时 (s)"},
    "rep_slots": {"en": "Chord slots", "zh": "和弦数"},
    "rep_viol": {"en": "Rule violations", "zh": "规则违规数"},
    "rep_energy": {"en": "Energy found", "zh": "所得能量"},
    "rep_classical": {"en": "Classical optimum (Viterbi)", "zh": "经典最优（Viterbi）"},
    "rep_ratio": {"en": "Approximation ratio", "zh": "近似比"},
    "rep_popt": {"en": "P(optimum) in samples", "zh": "采样中最优解概率"},
    "rep_feas": {"en": "Feasible samples", "zh": "可行样本比例"},
    "rep_viol_none": {"en": "No rule violations reported.", "zh": "未报告规则违规。"},
    "rep_viol_head": {"en": "Reported rule violations", "zh": "已报告的规则违规"},
    "rep_windows": {"en": "Per-window resources", "zh": "各窗口资源"},
    "rep_win_no": {"en": "Window", "zh": "窗口"},
    "rep_win_slots": {"en": "Slots", "zh": "和弦"},
    "rep_win_gap": {"en": "Gap", "zh": "差距"},
    "rep_chart": {"en": "Energy per window (lower is better)", "zh": "各窗口能量（越低越好）"},
    "rep_q_series": {"en": "Quantum solution", "zh": "量子解"},
    "rep_c_series": {"en": "Classical optimum", "zh": "经典最优"},
    "rep_circuit": {"en": "Circuit of the first window", "zh": "第一个窗口的电路"},
    "rep_hardware": {"en": "Recorded hardware run", "zh": "已记录的真机运行"},
    "rep_verdict": {"en": "Verdict from the pipeline", "zh": "流程给出的结论"},
    "rep_honest_title": {"en": "Honest assessment (Why quantum?)", "zh": "诚实评估（为什么用量子？）"},
    "rep_honest": {
        "en": ("Four-part harmony is solved exactly and efficiently by classical dynamic programming (Viterbi); "
               "this tool ships that solver as the baseline and does not claim a quantum advantage. "
               "Its contribution is the bridge: musical rules → penalty Hamiltonian → a swappable Qiskit solver, "
               "with errors you can hear and a constraint-preserving (XY-mixer) QAOA to study."),
        "zh": ("四部和声可以用经典动态规划（Viterbi）精确且高效地求解；本工具把它作为基准一并提供，"
               "并不声称量子优势。本项目的价值在于这座桥：音乐规则 → 惩罚哈密顿量 → 可替换的 Qiskit 求解器，"
               "错误能被听出来，并可研究约束保持（XY 混合器）QAOA。")},
}


def p5_t(key, **kw):
    """Look up a UI string in the language chosen by P5_LANG (falls back to English / the key itself)."""
    entry = P5_STR.get(key)
    if entry is None:
        return key
    s = entry.get(P5_LANG) or entry["en"]
    return s.format(**kw) if kw else s
