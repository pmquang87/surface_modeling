"""Regression tests for the automated bug hunt on commit 4e22413
(4e2241347de43175401df20c6722f38fa70820b4), bug report ``bug-report.md``.

Covered here:

* M10 (cand 23) - ``export_step_faceted`` counted every surface body as an
  open shell, so a clean closed cube exported with ``make_solids=False``
  reported ``open_shells == 1``.
* M21 (cand 26) - ``import_step`` failed under the nested ``OCP.OCP.*``
  package layout when cadquery is absent: ``TopAbs_REVERSED`` was imported
  from the flat ``OCP.TopAbs`` after the fallback block had already chosen
  the nested layout.
* M19 (cand 32) - ``FeatureTree.get_current_mesh`` / ``rebuild`` skipped a last
  feature that evaluated to ``None`` and reported the previous feature's mesh.
* S:nan_volume (report "unverified suspicions") - a NaN volume slipped through
  the volume gates of ``faceted_step.unify_verdict`` and
  ``NURBSSimplifier.simplify``, because every comparison with NaN is False.

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every gate gets an input it
must say "no" to and one it must say "yes" to, and counts are bracketed.
"""
import math
import os
import sys
import types

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# --------------------------------------------------------------------------
# M10: open_shells with make_solids=False
# --------------------------------------------------------------------------

def _box(sx=10.0, sy=20.0, sz=30.0):
    return trimesh.creation.box(extents=(sx, sy, sz))


@needs_ocp
def test_m10_closed_surface_body_is_not_counted_as_open_shell(tmp_path):
    """M10 (cand 23): a clean closed cube exported with ``make_solids=False``
    is one CLOSED surface body: ``open_shells`` must be 0 and ``solids`` 0.
    A box with a hole must still count as one open shell."""
    from src.io.faceted_step import export_step_faceted

    box = _box()
    holed = trimesh.Trimesh(vertices=box.vertices.copy(),
                            faces=np.delete(box.faces, 0, axis=0), process=False)
    # preconditions: the box is closed, the holed box is not
    assert box.is_watertight and box.is_winding_consistent
    assert not holed.is_watertight

    report = export_step_faceted(box, str(tmp_path / "surf.step"),
                                 make_solids=False, verify=False)
    assert report["ok"] is True, report.get("error")
    assert len(report["bodies"]) == 1
    assert report["bodies"][0]["closed"] is True
    assert report["bodies"][0]["solid"] is False
    assert report["closed_measured"] == [True]
    assert report["solids"] == 0
    assert report["open_shells"] == 0, report["bodies"]

    # negative control: the same switch on a holed box still reports the hole
    bad = export_step_faceted(holed, str(tmp_path / "holed.step"),
                              make_solids=False, verify=False)
    assert bad["ok"] is True, bad.get("error")
    assert bad["bodies"][0]["closed"] is False
    assert bad["open_shells"] == 1
    assert bad["solids"] == 0

    # control with solids on: closed box -> one solid, no open shell
    solid = export_step_faceted(box, str(tmp_path / "solid.step"), verify=False)
    assert solid["solids"] == 1 and solid["open_shells"] == 0

