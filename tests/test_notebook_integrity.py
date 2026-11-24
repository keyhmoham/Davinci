"""Lightweight structural/syntax audit for committed Jupyter notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _pure_python_source(source: str) -> str | None:
    lines = source.splitlines()
    if any(line.lstrip().startswith(("%%", "%", "!")) for line in lines):
        return None  # IPython/shell syntax is runtime-specific; validate notebook JSON instead.
    return source


def test_all_notebooks_are_valid_nbformat_json():
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert paths, "No notebooks found"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("nbformat") == 4, f"{path.name}: expected nbformat 4"
        assert isinstance(data.get("cells"), list), f"{path.name}: cells must be a list"
        for index, cell in enumerate(data["cells"]):
            assert cell.get("cell_type") in {"code", "markdown", "raw"}, (
                f"{path.name}: cell {index} has invalid cell_type"
            )
            assert isinstance(cell.get("source"), (str, list)), (
                f"{path.name}: cell {index} source must be text or text lines"
            )


def test_pure_python_notebook_cells_compile():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(data["cells"]):
            if cell.get("cell_type") != "code":
                continue
            raw = cell.get("source", "")
            source = "".join(raw) if isinstance(raw, list) else raw
            source = _pure_python_source(source)
            if source is None or not source.strip():
                continue
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name}, code cell {index}: {exc}"
                ) from exc
