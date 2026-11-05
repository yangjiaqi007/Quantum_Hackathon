"""Dev helper: execute the P5 percent-format files into ONE shared namespace (exactly what pasting cells into a notebook does)."""
import glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

def split_cells(text):
    """Percent format -> [(kind, title, source)]"""
    cells, kind, title, buf = [], None, "", []
    def flush():
        if kind is not None and "".join(buf).strip():
            cells.append((kind, title, "".join(buf).rstrip() + "\n"))
    for line in text.splitlines(keepends=True):
        m = re.match(r"^# %%(.*)$", line.rstrip("\n"))
        if m:
            flush()
            head = m.group(1).strip()
            kind = "markdown" if head.startswith("[markdown]") else "code"
            title = "" if kind == "markdown" else head
            buf = []
        else:
            buf.append(line)
    flush()
    out = []
    for k, t, src in cells:
        if k == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines()) + "\n"
        out.append((k, t, src))
    return out

def source_files():
    return sorted(glob.glob(os.path.join(SRC, "p5_[0-9][0-9]_*.py")))

def load(ns=None, skip=()):
    ns = {"__name__": "__main__"} if ns is None else ns
    for path in source_files():
        if any(s in os.path.basename(path) for s in skip):
            continue
        for kind, title, src in split_cells(open(path, encoding="utf-8").read()):
            if kind == "code":
                exec(compile(src, f"{os.path.basename(path)}::{title}", "exec"), ns)
    return ns
