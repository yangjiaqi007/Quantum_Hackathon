"""Shared test setup. Run from the repository root: ``pytest person2/tests``.

Imports use the namespaced package ``person2.quantum`` (never bare ``quantum``), so they
cannot collide with ``person3/quantum``. ``pytest.ini`` already has ``pythonpath = .``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:  # also lets tests run when pytest.ini's pythonpath is not picked up
    sys.path.insert(0, ROOT)
