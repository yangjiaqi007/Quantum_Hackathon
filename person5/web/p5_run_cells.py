# %% [markdown]
# ### Self-check — run once after merging the cells
#
# Prints a ✔/✘ checklist: dependencies, pipeline hook, melody parsing, score/audio/export, and a 4-note end-to-end run.

# %% P5-12 Self-check
p5_selfcheck()

# %% [markdown]
# ### Interactive panel
#
# Pick a melody (or type / click / upload one), then press **Run harmonization**. The panel uses standard QAOA.
# Tabs: *Score & audio* · *Quantum report* · *Export* (MIDI, MusicXML, WAV, PNG). Use `p5_launch("zh")` for Chinese.

# %% P5-13 Launch the panel
p5_launch()
