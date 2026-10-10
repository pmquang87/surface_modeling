"""Same-pattern follow-ups to the bug report ``bug-report.md`` (automated bug
hunt on commit 4e2241347de43175401df20c6722f38fa70820b4).

The first round fixed each bug where the report found it. Reviewers of that
round then found the same defect again in files outside the fixed group. Each
test here names its follow-up item and the original bug it mirrors:

* F1 (mirrors M15, cand 7/8) - ``gui.panels.face_normal`` kept the absolute
  ``norm > 1e-8`` degeneracy test and no centring, so the Properties panel
  showed a zero normal for a valid triangle with 1e-5 legs.

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every checker gets an input
it must accept (positive control) and one it must reject, and expected values
are exact.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402


# --------------------------------------------------------------------------
# F1 (mirrors M15): the Properties panel normal of a small face
# --------------------------------------------------------------------------

# A nominally collinear triple far from the origin. Centred, its Newell vector
# is below the relative threshold (degenerate -> zero vector); uncentred, the
# rounding of P + Q leaves a noise vector above it (measured: |n| = 9.7e-19
# against a threshold of 2.0e-22), which normalises to a unit "normal".
_FAR_COLLINEAR = [
    [1182.8084516740714, 575.9326302015468, 105.62701902348016],
    [1182.808448309971, 575.9326348601165, 105.62701918277486],
    [1182.8084449458706, 575.9326395186863, 105.62701934206956],
]


def _single_face(points):
    mesh = HalfEdgeMesh.from_arrays(np.asarray(points, dtype=np.float64), [[0, 1, 2]])
    assert len(mesh.faces) == 1
    return mesh, mesh.faces[0]


def _core_normal(points):
    mesh, face = _single_face(points)
    mesh.compute_face_normals()
    return np.asarray(face.normal, dtype=np.float64)


@pytest.mark.parametrize("name,points,expected", [
    # positive control: a unit triangle, accepted by both the old and new rule
    ("unit", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.0, 0.0, 1.0]),
    # the F1 defect: a valid triangle with 1e-5 legs (a scan in metres)
    ("legs_1e-5", [[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0]], [0.0, 0.0, 1.0]),
    # must be rejected: an exactly collinear (zero-area) face
    ("collinear", [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0.0, 0.0, 0.0]),
    # must be rejected: a collinear face far from the origin (needs centring)
    ("far_collinear", _FAR_COLLINEAR, [0.0, 0.0, 0.0]),
])
def test_f1_panel_face_normal_matches_the_m15_core_rule(name, points, expected):
    """F1 (mirrors M15): ``gui.panels.face_normal`` says it mirrors
    ``HalfEdgeMesh.compute_face_normals``; it must give the same normal as the
    fixed core routine, including a unit normal for a 1e-5-leg triangle and
    the zero vector for a zero-area face."""
    from src.gui.panels import face_normal

    expected = np.asarray(expected, dtype=np.float64)
    # precondition: the M15-fixed core routine gives the expected value
    core = _core_normal(points)
    assert np.allclose(core, expected, atol=1e-12), (name, core)
    if name == "legs_1e-5":
        # precondition: the old absolute threshold really zeroes this face
        P = np.asarray(points, dtype=np.float64)
        assert np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])) < 1e-8

    mesh, face = _single_face(points)
    shown = face_normal(mesh, face)
    assert np.allclose(shown, expected, atol=1e-12), (name, shown)
    assert np.array_equal(shown, core), (name, shown, core)
    # the value is cached on the face, as before
    assert np.array_equal(np.asarray(face.normal), shown)
