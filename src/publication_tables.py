"""Export metrics to CSV and LaTeX tables for papers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def write_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write list of dict rows to CSV (union of all keys as columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k)) for k in keys})


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_metrics_latex(
    rows: list[dict[str, Any]],
    path: str | Path,
    columns: Sequence[str] | None = None,
    caption: str = "Results",
    label: str = "tab:results",
) -> None:
    """Write a simple booktabs-style LaTeX table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% empty table\n", encoding="utf-8")
        return
    cols = list(columns) if columns else sorted({k for r in rows for k in r.keys()})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(_latex_header(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(_latex_cell(r.get(c)) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_header(s: str) -> str:
    return s.replace("_", r"\_")


def _latex_cell(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.4f}"
    t = str(v).replace("_", r"\_").replace("%", r"\%")
    return t
