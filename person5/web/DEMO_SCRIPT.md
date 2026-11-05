# Demo script — Quantum Four-Part Harmony (≈ 5 minutes)

*Spoken lines are in English (what the judges hear). Lines in 中文 are stage directions for the team.*
The judges score: technical soundness, creative thinking, clear explanation, team contribution. AI help is fine, **understanding is required** — every speaker must be able to explain their own part.

## Before you walk up (checklist)

- [ ] 提前运行整个 notebook 一次，确认 `p5_selfcheck()` 全部 ✔，且报告里**没有** MOCK 红色横幅
- [ ] 先在本机跑一遍 Ode to Joy 和 O Canada，保存好结果；录一段真机/缓存运行作为备份（不要现场依赖 IBM 排队）
- [ ] 把下面表格的数字从实验结果里填好（P3 / P4 提供）
- [ ] 电脑音量打开，浏览器允许播放音频；笔记本接电源；字体放大（Ctrl +）
- [ ] 打开顺序：幻灯片 → notebook（`p5_launch()` 已运行）→ 备份截图文件夹

| Number to say aloud | Value (fill in) | From |
|---|---|---|
| Qubits per window | ____ | Part 4 report |
| Circuit depth / two-qubit gates (one window) | ____ / ____ | Part 4 report |
| P(optimum): plain mixer vs XY mixer | ____ vs ____ | Part 3 experiment |
| Feasible samples: plain vs XY | ____ vs ____ | Part 3 experiment |
| Energy gap to Viterbi optimum | ____ | Part 4 report |

## Timeline

| Time | Speaker | Slide / screen | Say | Do |
|---|---|---|---|---|
| 0:00–0:30 | P1 | Slides 1–2 | "Four-part harmonization is a classic music-theory problem: given a soprano line, write alto, tenor and bass so that it follows rules such as no parallel fifths and resolve the leading tone. It has existed for centuries, and Bach's chorales are our ground truth. It has nothing to do with quantum computing — that is the point." | |
| 0:30–1:15 | P2 | Slide 3 | "Our contribution is the bridge. A melody becomes candidate voicings; each rule becomes a penalty term; together they form a QUBO, then an Ising Hamiltonian in Qiskit. A good harmonization is a low-energy state. The user never touches a qubit." | 指出 domain → encoding → algorithm → decoded answer 这条线 |
| 1:15–2:00 | P3 | Slide 4 | "Plain QAOA leaks amplitude into invalid states. Our XY mixer with a W-state start keeps exactly one voicing per chord slot, so every sample is valid. Full melodies don't fit in a simulator, so we solve sliding windows of 3–4 chords and pass the optimized angles to the next window." | |
| 2:00–3:30 | **P5** | notebook → panel | "Let me show it." | 见下方 *Live demo steps* |
| 3:30–4:15 | P4 | Report tab | "Every run produces an honest report: qubits, depth, two-qubit gates, energy per window versus the classical optimum, and which rules were violated." | 念出上表的数字；指向能量柱状图和电路图 |
| 4:15–5:00 | P1 or P3 | Slide 6–7 | "Four-part harmony is solvable exactly with Viterbi, and the tool ships that baseline. We are not claiming a quantum advantage. We built an interpretable, benchmarkable bridge — and we measured where quantum helps and where it doesn't." | 一句话收尾，转 Q&A |

## Live demo steps (P5, 90 seconds)

1. Panel is already open. Melody: **Ode to Joy**, solver **xy_qaoa**. Say: "Style is a preset plus a strictness slider that scales all rule weights."
2. Click **Run harmonization**. While it runs: "The progress bar follows the windows."
3. *Score & audio* tab: point to the soprano (input) and the three new voices. Press **Soprano only**, then **Full chorale**. "You can *hear* whether it is right — that is why music is a good benchmark."
4. If a note is red: "Red notes take part in a violated rule."
5. Change the solver to **qaoa** (plain), run again, compare the report: feasible-sample fraction and violations.
6. *Export* tab: "MIDI and MusicXML open in MuseScore or any DAW."
7. Type a custom melody (`G4 A4 B4 C5 D5 E5 D5 C5`) or click notes on the buttons, run once more. "Any monophonic melody works — up to 40 notes."

## If something goes wrong

| Problem | Fallback |
|---|---|
| Run is slow / queue / token expired | Switch to the cached run or the saved screenshots; say "this is the recorded run from earlier today" (be honest about it) |
| No sound | Open `p5_output/*.wav` directly, or play the exported MIDI in MuseScore |
| Widgets do not render | Scroll to the static showcase cell `p5_demo_showcase()` — its outputs are saved in the notebook |
| Red MOCK banner appears | Stop: the real `harmonize_pipeline` is not defined. Fix before presenting; never present mock output as quantum results |

## Likely judge questions — honest answers

**Why quantum for a problem Viterbi solves exactly?**
"For this instance, you shouldn't — and the tool tells you that and ships Viterbi as a baseline. The goal is to make quantum optimization usable and testable by people outside quantum, and to measure honestly where it helps."

**What is quantum about it?**
"The solver: QAOA uses superposition and interference over candidate harmonizations, and measurement samples low-energy voicings. The problem itself is classical; we say so."

**Does the quantum solver give multiple solutions while Viterbi gives one?**
"Not as a capability: k-best Viterbi and Boltzmann sampling give many solutions classically. QAOA's distribution is a different, circuit-controlled kind of diversity, and we compare them rather than claim an advantage."

**Why XY mixer and W-state?**
"One-hot slots must have exactly one active voicing. A standard X mixer leaks into invalid states, forcing a large penalty that squashes the energy landscape. The XY mixer stays inside the valid subspace."

**How do you fit a full melody on a simulator?**
"Sliding windows of 3–4 chords (12–16 qubits) with the boundary voicing fixed and warm-started angles."

**How do you know the result is correct?**
"An independent rule checker counts violations, tests that Hamiltonian energy equals weighted violations, and Viterbi gives the exact optimum to compare against."

**Why Qiskit?**
"QuadraticProgram/QUBO-to-Ising conversion, QAOA ansätze, V2 primitives that run on simulators, noisy fake backends and IBM hardware, plus the transpiler for the resource report."

**Did you use AI?**
Answer truthfully and say which parts each member can explain themselves.

## Part 5 — what to be able to explain (cheat sheet)

* **Why synthesize audio in numpy?** Browsers can't play `.mid`; additive synthesis (a few harmonics + attack/release envelope) works offline in any notebook.
* **Why draw the score with matplotlib?** `music21.show()` needs MuseScore/LilyPond installed; our renderer needs nothing and can colour violations and shade windows. MusicXML export is there for proper engraving tools.
* **Why `p5_` prefixes?** Five people paste code into one notebook; prefixes make name collisions impossible.
* **What does `p5_invoke_pipeline` do?** Looks at the pipeline's signature and passes only the parameters it accepts, so small naming differences between parts don't break the interface.
* **What does the sanity check do?** Confirms the soprano was not changed, voices are ordered bass ≤ tenor ≤ alto ≤ soprano and within normal ranges — it catches integration bugs, it does not judge harmony.
