"""Control-net fairing in the G3 fitter.

Since bug M1 the fit is an exact SuperLU solve. With 6x6 samples per patch
the data rows are badly conditioned (sigma_min ~1.5e-3: an alternating pole
pattern barely moves the surface), so sample noise blows up into oscillating
interiors that the divergence clamp then resets to Coons. lsqr's early stop
used to damp those modes implicitly. On the 2152-quad real part the clamp
reset 570 patches at G0 and 44 at G1, and SolidWorks 2025 dropped one face of
the G1 STEP.

``G3Fitter(fairing_weight=lam)`` adds, per patch, every second difference of
the control net along u and v that touches an interior pole,
``lam * (P[i-1,j] - 2 P[i,j] + P[i+1,j]) = 0`` (and along j), and
``SubDToNURBSConverter`` uses ``lam = 0.01`` for every continuity level.

This file also pins the honest meaning of the levels: G0 adds no cross-patch
row, G1, G2 and G3 add the same third-derivative rows (weights 5/20/50 that
the exact solve treats alike), so G1 == G2 == G3 to rounding while G0
differs. Changing that must be a deliberate decision that updates this test.
"""
import contextlib
import io
import os
import re
import sys
from math import comb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh
from src.nurbs import g3_fitter as g3
from src.nurbs.converter import SubDToNURBSConverter
from src.nurbs.g3_fitter import G3Fitter

R = 10.0
FAIR = 0.01


# ------------------------------------------------------------------ helpers

def _bern(t):
    t = np.asarray(t, dtype=float)
    return np.array([[comb(5, i) * s ** i * (1 - s) ** (5 - i) for i in range(6)] for s in t])


def _resets(text):
    m = re.search(r"reset (\d+) diverged", text)
    return int(m.group(1)) if m else 0


def _span(patches):
    """max |interior pole - corner centroid| / corner-bbox diagonal."""
    P = np.asarray(patches, dtype=float)
    c = np.stack([P[:, 0, 0], P[:, 5, 0], P[:, 5, 5], P[:, 0, 5]], axis=1)
    diag = np.linalg.norm(c.max(axis=1) - c.min(axis=1), axis=1)
    far = np.linalg.norm(P[:, 1:5, 1:5].reshape(len(P), 16, 3) - c.mean(axis=1)[:, None], axis=2)
    return far.max(axis=1) / diag


def _second_differences(P):
    """All second differences along u and v touching an interior pole."""
    out = []
    for a in range(1, 5):
        for c in range(1, 5):
            out.append(P[a - 1, c] - 2 * P[a, c] + P[a + 1, c])
            out.append(P[c, a - 1] - 2 * P[c, a] + P[c, a + 1])
    return np.array(out)


def _noisy_sphere_case(amp, seed=1):
    """Cube cage (3x3 quads per side, valence-3 corners) on a sphere of
    radius 10 and a reference icosphere (5120 triangles) whose vertices carry
    seeded radial noise of +-amp; the cage vertices are snapped onto it, as
    the shrink wrap does."""
    verts, faces = {}, []

    def vid(p):
        return verts.setdefault(tuple(np.round(p, 9)), len(verts))

    t = np.linspace(-1.0, 1.0, 4)
    for axis in range(3):
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1.0, 1.0):
            for i in range(3):
                for j in range(3):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[a1], p[a2] = sign, t[i + di], t[j + dj]
                        quad.append(vid(p))
                    faces.append(quad[::-1] if sign < 0 else quad)
    V = np.zeros((len(verts), 3))
    for key, k in verts.items():
        V[k] = key
    V = R * V / np.linalg.norm(V, axis=1, keepdims=True)
    ico = trimesh.creation.icosphere(subdivisions=4, radius=R)
    rng = np.random.default_rng(seed)
    noisy = ico.vertices * (1.0 + amp / R * rng.uniform(-1.0, 1.0, size=(len(ico.vertices), 1)))
    ref = trimesh.Trimesh(noisy, ico.faces, process=False)
    V = trimesh.proximity.closest_point(ref, V)[0]
    return V, faces, ref


def _convert(V, faces, ref, level):
    """Run the converter's reference fit; return patches, its stdout and the
    quad data it handed to the fitter."""
    seen = []
    orig = g3.G3Fitter.fit_surface

    def spy(self, quad_mesh):
        seen.append(quad_mesh)
        return orig(self, quad_mesh)

    g3.G3Fitter.fit_surface = spy
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            patches = SubDToNURBSConverter(continuity=level).generate_patches(
                HalfEdgeMesh.from_arrays(V, faces), reference_mesh=HalfEdgeMesh.from_trimesh(ref))
    finally:
        g3.G3Fitter.fit_surface = orig
    assert len(seen) == 1
    return np.array(patches), buf.getvalue(), seen[0]


def _deviation(patches, ref):
    t = np.linspace(0.0, 1.0, 9)
    Bm = _bern(t)
    pts = np.einsum('ai,bj,kijc->kabc', Bm, Bm, np.asarray(patches)).reshape(-1, 3)
    return float(np.asarray(trimesh.proximity.closest_point(ref, pts)[1]).mean())


def _data_only_ratios(quad_mesh):
    """Independent oracle (dense numpy lstsq of the data rows of each patch,
    no continuity, no fairing): runaway ratio of each patch's optimum."""
    out = []
    for q in quad_mesh:
        base = g3._generate_single_patch(q)
        Bm = _bern(np.linspace(0.0, 1.0, q['dense_points'].shape[0]))
        rows, rhs = [], []
        for r in range(Bm.shape[0]):
            for c in range(Bm.shape[0]):
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


# ------------------------------------------- the defect: noise runs away

@pytest.mark.parametrize("level, amp, bare_weight", [('G0', 0.05, 0.0), ('G1', 0.2, 5.0)])
def test_fairing_keeps_a_noisy_fit_bounded(level, amp, bare_weight):
    """Noisy reference (0.5 % / 2 % of the radius): the bare exact solve
    (no fairing, the 715f34b system) lets interiors run away and the clamp
    resets them - at G0 50 of 54 patches, at G1 4. The converter's faired
    fit resets none, keeps every interior within 0.75 x its quad diagonal
    and is at least as close to the reference. Fails on 715f34b: the
    converter reports the resets (50 at G0, 4 at G1)."""
    V, faces, ref = _noisy_sphere_case(amp)
    patches, out, quad_mesh = _convert(V, faces, ref, level)

    # precondition: the defect exists in the bare system on this very input
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bare = np.array(G3Fitter(continuity_weight=bare_weight).fit_surface(quad_mesh))
    assert _resets(buf.getvalue()) >= 4, buf.getvalue()
    if level == 'G0':
        # ... and is the least-squares optimum itself, not a clamp artefact
        assert (_data_only_ratios(quad_mesh) > 1.25).sum() >= 10

    assert _resets(out) == 0, out
    sp = _span(patches)
    assert sp.max() < 0.75, sp.max()
    assert _deviation(patches, ref) <= _deviation(bare, ref)
    # the faired interiors were solved, not left at Coons
    coons = np.array([g3._generate_single_patch(q)[1:5, 1:5] for q in quad_mesh])
    assert np.abs(patches[:, 1:5, 1:5] - coons).max() > 0.05


def test_converter_fairs_every_level():
    """fairing_weight 0.01 reaches the fitter at G0, G1, G2 and G3."""
    from src.subd.primitives import create_box
    seen = []

    class Recording(G3Fitter):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.append(self.fairing_weight)

    orig = g3.G3Fitter
    g3.G3Fitter = Recording
    try:
        for level in ('G0', 'G1', 'G2', 'G3'):
            with contextlib.redirect_stdout(io.StringIO()):
                SubDToNURBSConverter(continuity=level).generate_patches(
                    create_box(), reference_mesh=create_box())
    finally:
        g3.G3Fitter = orig
    assert seen == [FAIR] * 4, seen


# ------------------------------------------------ exactness and invariance

def _grid(nx, ny, offset=(0.0, 0.0, 0.0), scale=1.0, height=None, noise=0.0):
    """nx x ny unit quads in cyclic winding, 6x6 samples per quad lifted onto
    z = height(x, y) plus an optional seeded sample noise; everything shifted
    by ``offset`` after scaling by ``scale``."""
    h = height if height is not None else (lambda x, y: 0.0 * x)
    rng = np.random.default_rng(5)
    s = np.linspace(0.0, 1.0, 6)
    uu, vv = np.meshgrid(s, s, indexing='ij')

    def idx(i, j):
        return j * nx + i if (0 <= i < nx and 0 <= j < ny) else -1

    off = np.asarray(offset, dtype=float)
    quads = []
    for j in range(ny):
        for i in range(nx):
            xy = np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], dtype=float)
            corners = np.column_stack([xy, h(xy[:, 0], xy[:, 1])])
            gx = i + uu
            gy = j + vv
            dense = np.stack([gx, gy, h(gx, gy)], axis=-1)
            dense = dense + noise * rng.uniform(-1.0, 1.0, size=dense.shape)
            quads.append({'corners': corners * scale + off, 'dense_points': dense * scale + off,
                          'neighbors': [idx(i, j - 1), idx(i + 1, j), idx(i, j + 1), idx(i - 1, j)]})
    return quads


def _interiors(patches):
    return np.array([np.asarray(p)[1:5, 1:5] for p in patches])


def _tol(off):
    return 1e-9 * max(1.0, off / 1000.0)


@pytest.mark.parametrize("off", [0.0, 10.0, 500.0, 1e4])
def test_plane_stays_exact_with_fairing(off, capsys):
    """Two flat unit quads with samples taken exactly from the plane: the
    Coons start is the bilinear net, which reproduces the samples and has
    zero second differences, so it zeroes every data, continuity and fairing
    row - the faired fit must keep it to ~1e-9 at every offset."""
    qm = _grid(2, 1, offset=(off, off, 0.0))
    base = [g3._generate_single_patch(q) for q in qm]
    for b in base:
        assert np.abs(_second_differences(b)).max() < _tol(off)

    patches = G3Fitter(continuity_weight=20.0, fairing_weight=FAIR).fit_surface(qm)

    assert "diverged" not in capsys.readouterr().out
    moved = np.abs(_interiors(patches) - _interiors(base)).max()
    assert moved < _tol(off), f"offset {off}: plane interiors moved by {moved:.3e}"


def test_fairing_is_active_on_a_curved_net():
    """Control for the plane test: on the exact bump z = 0.3 u(1-u) v(1-v)
    (non-zero second differences) the faired fit leaves the exact net, the
    bare one reproduces it - so the plane stays exact because its second
    differences vanish, not because fairing is off."""
    def bump(x, y):
        u = x - np.floor(np.minimum(x, 1.999999))
        return 0.3 * u * (1 - u) * y * (1 - y)

    qm = _grid(2, 1, height=bump)
    c = np.array([i * (5 - i) / 20.0 for i in range(6)])
    with contextlib.redirect_stdout(io.StringIO()):
        bare = _interiors(G3Fitter(continuity_weight=20.0).fit_surface(qm))
        faired = _interiors(G3Fitter(continuity_weight=20.0, fairing_weight=FAIR).fit_surface(qm))
    exact_z = 0.3 * np.outer(c, c)[1:5, 1:5]
    assert np.abs(bare[..., 2] - exact_z).max() < 1e-9
    assert np.abs(faired[..., 2] - exact_z).max() > 1e-5


def _wavy(x, y):
    return 0.35 * np.sin(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.05 * x * y


def _faired(qm):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        patches = G3Fitter(continuity_weight=5.0, fairing_weight=FAIR).fit_surface(qm)
    assert "diverged" not in buf.getvalue()
    return _interiors(patches)


def test_faired_fit_is_translation_invariant():
    """5x4 curved grid with sample noise of +-0.01: the faired fit differs
    from the bare one (fairing matters here) and moves rigidly with the
    input at offsets 10, 500 and 1e4 (1e-8, relaxed linearly beyond 100 for
    the digits the absolute coordinates themselves lose)."""
    qm0 = _grid(5, 4, height=_wavy, noise=0.01)
    got0 = _faired(qm0)
    with contextlib.redirect_stdout(io.StringIO()):
        bare = _interiors(G3Fitter(continuity_weight=5.0).fit_surface(qm0))
    assert np.abs(got0 - bare).max() > 1e-3
    for off in (10.0, 500.0, 1e4):
        o = np.array([off, 0.7 * off, -0.3 * off])
        got = _faired(_grid(5, 4, offset=o, height=_wavy, noise=0.01)) - o
        drift = np.abs(got - got0).max()
        assert drift < 1e-8 * max(1.0, off / 100.0), f"offset {off}: moved {drift:.3e}"


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_faired_fit_scales_with_the_model(scale):
    """The same grid modelled in m (1e-3) or um (1e3): the faired interiors
    are the unit-scale ones times the scale (relative 1e-9). An absolute
    constant anywhere in the fairing rows or their weight would break this."""
    got1 = _faired(_grid(5, 4, height=_wavy, noise=0.01))
    got = _faired(_grid(5, 4, scale=scale, height=_wavy, noise=0.01)) / scale
    err = np.abs(got - got1).max()
    assert err < 1e-9, f"scale {scale}: off by {err:.3e} (unit-scale units)"


# --------------------------------------------- the levels, pinned honestly

def test_g1_g2_g3_give_the_same_surface_and_g0_does_not():
    """G1, G2 and G3 add the same C3 rows (weights 5, 20, 50): their faired
    fits on the noisy-sphere cage agree to 1e-5 x the quad diagonal
    (measured ~6e-7), while G0 (no cross-patch rows) differs by more than
    1e-2 x diagonal (measured ~6e-2). Deliberately changing what the levels
    do must update this test."""
    V, faces, ref = _noisy_sphere_case(0.02)
    fits = {lvl: _convert(V, faces, ref, lvl) for lvl in ('G0', 'G1', 'G2', 'G3')}
    for lvl, (_, out, _) in fits.items():
        assert _resets(out) == 0, (lvl, out)
    qm = fits['G1'][2]
    diag = np.median([np.linalg.norm(np.ptp(np.asarray(q['corners']), axis=0)) for q in qm])
    I = {lvl: f[0][:, 1:5, 1:5] for lvl, f in fits.items()}
    assert np.abs(I['G2'] - I['G1']).max() < 1e-5 * diag
    assert np.abs(I['G3'] - I['G1']).max() < 1e-5 * diag
    assert np.abs(I['G0'] - I['G1']).max() > 1e-2 * diag
