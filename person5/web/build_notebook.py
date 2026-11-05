"""Assemble the Part-5 percent-format files into ONE notebook.
   python build_notebook.py                 -> project.ipynb (cells only, no outputs)
   python build_notebook.py --no-mock       -> same, without the mock cell P5-10 (use once the real pipeline exists)
   python build_notebook.py --execute       -> also executes it in a fresh kernel (QA) and keeps the outputs
"""
import argparse, glob, os, sys
import nbformat
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev"))
import p5_loader

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="project.ipynb")
ap.add_argument("--no-mock", action="store_true")
ap.add_argument("--execute", action="store_true")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, "p5_[0-9][0-9]_*.py")))
if args.no_mock:
    files = [f for f in files if "_mock" not in f]
files.append(os.path.join(here, "p5_run_cells.py"))

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
for path in files:
    for kind, title, src in p5_loader.split_cells(open(path, encoding="utf-8").read()):
        src = src.rstrip("\n")
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(f"# {title}\n{src}" if title else src))
if args.execute:
    from nbconvert.preprocessors import ExecutePreprocessor
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": here}})
nbformat.validate(nb)
nbformat.write(nb, os.path.join(here, args.out))
print("wrote", args.out, "cells:", len(nb.cells))
