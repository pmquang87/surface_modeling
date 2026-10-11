"""The G3 fitter's divergence clamp must not depend on the model's unit scale.

``G3Fitter.fit_surface`` resets a patch interior to its Coons initialization
when a fitted interior control point lies further than ``limit`` from the
corner centroid. The limit was ``max(1.25 * diag, diag + 0.5)``: the absolute
``+ 0.5`` (model units) dominated for quad diagonals below 2, so the same
part reset different patches when modelled in mm, in m or in um. The
SolidWorks criterion it guards (faces whose interiors span ~1.3x the quad
diagonal or more are dropped on import, see commit f5a8577) is relative, and
the clamp is now ``limit = 1.25 * diag``.

Fixture: three separate unit quads (no shared edge, so the continuity rows
play no part), z = 0 plane samples plus a checkerboard of amplitude a on
patch A, B and C. The fit is linear in the samples, so the runaway ratio
max|P_ij - centre| / diag of the least-squares optimum is set by a alone and
is computed here by an independent dense numpy oracle:

* A, a = 0.0012:  ratio ~0.87 - inside 1.25, never reset;
* B, a = 0.0018:  ratio ~1.29 - above 1.25, but below the old floor's
  1 + 0.5/sqrt(2) = 1.354 at scale 1: on 1f93473 kept at scale 1 and 1e-3,
  reset at scale 1e3;
* C, a = 0.25:    ratio ~178 - a genuine runaway; on 1f93473 reset at scale
  1 and 1e3 but NOT at 1e-3 (limit 0.5 against a 0.25-unit excursion).
"""
import os
import sys
from math import comb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.nurbs import g3_fitter as g3
from src.nurbs.g3_fitter import G3Fitter

SCALES = [1e-3, 1.0, 1e3]
AMPS = {'A': 0.0012, 'B': 0.0018, 'C': 0.25}
OLD_FLOOR_RATIO = 1.0 + 0.5 / np.sqrt(2.0)  # old limit / diag on a unit quad


def _bern(t):
    return np.array([[comb(5, i) * s ** i * (1 - s) ** (5 - i) for i in range(6)] for s in t])


def _fixture(scale):
    """Patches A, B, C at x = 0, 3, 6, every coordinate times ``scale``."""
    t = np.linspace(0.0, 1.0, 6)
    uu, vv = np.meshgrid(t, t, indexing='ij')
    rr, cc = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    checker = (-1.0) ** (rr + cc)
    quads = []
    for k, amp in enumerate(AMPS.values()):
        x0 = 3.0 * k
        corners = np.array([[x0, 0, 0], [x0 + 1, 0, 0], [x0 + 1, 1, 0], [x0, 1, 0]], float)
        dense = np.stack([x0 + uu, vv, amp * checker], axis=-1)
        quads.append({'corners': corners * scale, 'dense_points': dense * scale,
                      'neighbors': [-1, -1, -1, -1]})
    return quads


def _oracle_ratios(quads):
    """Runaway ratio of each patch's least-squares optimum (data rows only,
    dense numpy lstsq), independent of the fitter's solver."""
    Bm = _bern(np.linspace(0.0, 1.0, 6))
    out = []
    for q in quads:
        base = g3._generate_single_patch(q)
        rows, rhs = [], []
        for r in range(6):
            for c in range(6):
                outer = np.outer(Bm[r], Bm[c])
                fixed = outer.copy()
                fixed[1:5, 1:5] = 0.0
                rows.append(outer[1:5, 1:5].ravel())
                rhs.append(q['dense_points'][r, c] - np.einsum('ij,ijc->c', fixed, base))
        x = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        corners = np.array([base[0, 0], base[5, 0], base[5, 5], base[0, 5]])
        diag = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
        out.append(np.linalg.norm(x - corners.mean(axis=0), axis=1).max() / diag)
    return np.array(out)


def _reset_set(quads, patches):
    """Names of the patches whose interior came back as exactly the Coons
    initialization (the clamp's fallback)."""
    names = list(AMPS)
    reset = set()
    for name, q, p in zip(names, quads, patches):
        coons = g3._generate_single_patch(q)[1:5, 1:5]
        if np.array_equal(p[1:5, 1:5], coons):
            reset.add(name)
    return reset


def test_fixture_ratios_straddle_the_old_scale_dependent_band():
    """Preconditions, bracketed from both sides and at every scale: A is
    inside the relative limit, B sits in the band (1.25, 1.354) where the old
    absolute floor decided at scale 1, C is a genuine runaway."""
    for s in SCALES:
        ra, rb, rc = _oracle_ratios(_fixture(s))
        assert 0.7 < ra < 1.15, (s, ra)
        assert 1.26 < rb < OLD_FLOOR_RATIO - 0.03, (s, rb)
        assert rc > 50.0, (s, rc)


@pytest.mark.parametrize("scale", SCALES)
def test_clamp_resets_the_same_patches_at_every_scale(scale, capsys):
    """B and C exceed 1.25 x diag and must be reset at every scale, A must
    be fitted at every scale. On 1f93473 this fails at scale 1 (B kept: the
    floor allowed 1.354x) and at 1e-3 (nothing reset: the 0.5 floor is 350x a
    1.4e-3 diagonal)."""
    quads = _fixture(scale)
    patches = G3Fitter(continuity_weight=20.0).fit_surface(quads)
    out = capsys.readouterr().out

    reset = _reset_set(quads, patches)
    assert reset == {'B', 'C'}, f"scale {scale}: reset {sorted(reset)}"
    assert "reset 2 diverged patch interior" in out, out
    # the reset detector is not vacuous: A was solved and moved off Coons by
    # the full fitted excursion (~1.2 x diag in z), scaled with the model
    coons_a = g3._generate_single_patch(quads[0])[1:5, 1:5]
    moved = np.abs(patches[0][1:5, 1:5] - coons_a).max()
    assert moved > 1.0 * scale, (scale, moved)


@pytest.mark.parametrize("scale", SCALES)
def test_genuine_runaway_is_reset_at_every_scale(scale, capsys):
    """C alone (its optimum lies ~178 diagonals out): reset at every scale.
    On 1f93473 it survived at scale 1e-3: 178 x 1.4e-3 = 0.25 units of
    excursion against a limit of max(1.25 x 1.4e-3, 1.4e-3 + 0.5) = 0.5."""
    quads = _fixture(scale)[2:]
    patches = G3Fitter(continuity_weight=20.0).fit_surface(quads)
    out = capsys.readouterr().out
    coons = g3._generate_single_patch(quads[0])[1:5, 1:5]
    assert np.array_equal(patches[0][1:5, 1:5], coons), f"scale {scale}: runaway kept"
    assert "reset 1 diverged patch interior" in out, out
