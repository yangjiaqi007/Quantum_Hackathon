# %% [markdown]
# ### Live solver demonstration
#
# Run this notebook from the project repository so its compiler, backend, and solver modules are available. No IBM hardware is needed.
# The setup checks the real API and runs both QAOA variants on Aer, then renders their scores and resource reports.

# %% P5-12 Self-check
p5_selfcheck()

# %% [markdown]
# ### Interactive panel
#
# Pick a melody (or type / click / upload one), choose QAOA or XY-QAOA, then press **Run harmonization**.
# Tabs: *Score & audio* · *Quantum report* · *Export* (MIDI, MusicXML, WAV, PNG). Use `p5_launch("zh")` for Chinese.

# %% P5-13 Launch the panel
p5_launch()
