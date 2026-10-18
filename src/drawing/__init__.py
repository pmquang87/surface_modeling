"""STEP / STL to 2D engineering drawing (PDF, SVG, DXF + JSON quality report).

``python -m src.drawing part.step out/`` writes orthographic views with
hidden lines, automatic dimensions, one hatched section and an ISO 5457 /
ISO 7200 sheet. STEP goes through OCCT hidden-line removal; STL goes
through a faceted STEP conversion (prismatic meshes) or a numpy
depth-buffer projection (organic meshes and fallbacks). The run executes in
a sandboxed child process and keeps the ``RESULT {json}`` stdout contract
of ``src/convert.py``. Design: ``docs/superpowers/specs/
2026-10-17-drawing-module-design.md``.

Importing this package is cheap on purpose: no numpy, trimesh or OCP is
loaded until ``make_drawing`` runs (the sandbox parent sniffs the input and
applies the limits before any geometry library is imported).
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "make_drawing"]


def make_drawing(input_path, out_dir, options=None, log=None):
    """Run the drawing pipeline in-process (no sandbox).

    ``options`` is a ``DrawingOptions`` (defaults when None). Returns
    ``(exit_code, result_dict)`` from ``src.drawing.pipeline.run``:
    0 clean, 2 drawing written with quality warnings, 1 failed.
    """
    from .options import DrawingOptions
    from .pipeline import run

    if options is None:
        options = DrawingOptions()
    return run(input_path, out_dir, options, log=log)
