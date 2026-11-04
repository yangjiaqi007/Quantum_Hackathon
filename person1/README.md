# Person 1: Music-to-Hamiltonian compiler

The compiler implementation lives in `person1/compiler/`. It parses text and MIDI melodies, provides built-in presets, generates voicing candidates, scores the harmony rules, normalizes costs, and checks selections with checker/Viterbi utilities. Its solver handoff is the `h` and `W` arrays in `Schema.CompiledProblem`.

`Schema.py` and the root `compiler/` compatibility facade are shared project files. Tests for the current compiler API are in `person1/tests/`; `legacy_tests/` holds earlier standalone checks that target an older interface. Example compiler output is in `person1/examples/`.
