"""Regression tests for the G3 fitter's global least-squares solve.

Bug report: automated bug hunt on commit
4e2241347de43175401df20c6722f38fa70820b4 (bug-report.md).

Covers:
- M1 (cand 19): ``G3Fitter.fit_surface`` called ``scipy.sparse.linalg.lsqr``
  on its default tolerances for the absolute control-point coordinates. The
  stopping tests are relative to |b| and |x|, which grow with the part's
  distance from the origin, so the solve stopped ever earlier: an exactly
  representable plane came back bent at the origin, worse at offset 10, and
  "diverged" (reset to Coons) at offset 500.
- M18 (cand 18): without ``dense_points`` the system is rank deficient and
  ``lsqr`` returned the minimum-norm *solution*, pulling the patch interiors
  towards the coordinate origin. The documented intent is the smallest change
  from the Coons interior.

The fix solves for the change from the Coons interior: directly (factored
normal equations plus refinement) when every patch has dense samples, by
lsqr's minimum-norm change otherwise.

The expected values come from an independent dense oracle (``numpy.linalg.
lstsq`` on the same equations, written out from the fitter's documented
math) or from closed-form control nets, never from the fitter itself.
"""
import os
import sys
from math import comb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.nurbs import g3_fitter as g3
from src.nurbs.g3_fitter import G3Fitter

RESET_MSG = "diverged patch interior"

# degree-5 Bernstein control values of u(1-u): 0, .2, .3, .3, .2, 0
_C_U1MU = np.array([i * (5 - i) / 20.0 for i in range(6)])


def _bern_matrix(t):
    t = np.asarray(t, dtype=float)
    return np.array([[comb(5, i) * s ** i * (1 - s) ** (5 - i) for i in range(6)] for s in t])


def _eval(ctrl, t):
    """Evaluate a 6x6x3 Bezier control net on the tensor grid t x t."""
    Bm = _bern_matrix(t)
    return np.einsum('ai,bj,ijc->abc', Bm, Bm, ctrl)


def _bilinear(c0, c1, c2, c3, n=6):
    t = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(t, t, indexing='ij')
    return (((1 - uu) * (1 - vv))[..., None] * c0 + (uu * (1 - vv))[..., None] * c1
            + (uu * vv)[..., None] * c2 + ((1 - uu) * vv)[..., None] * c3)


def _grid_mesh(nx, ny, offset=(0.0, 0.0, 0.0), height=None, dense=True, normals=None):
    """nx x ny unit quads in cyclic winding, corners on z = height(x, y).

    ``height`` also gives the dense samples (6x6 per quad, bilinear in the
    parameter, lifted onto the surface). ``normals(x, y)`` gives corner
    normals (curved tangent-plane boundary curves)."""
    off = np.asarray(offset, dtype=float)
    h = height if height is not None else (lambda x, y: 0.0 * x)

    def idx(i, j):
        return j * nx + i if (0 <= i < nx and 0 <= j < ny) else -1

    quads = []
    for j in range(ny):
        for i in range(nx):
            xy = np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], dtype=float)
            corners = np.column_stack([xy, h(xy[:, 0], xy[:, 1])]) + off
            q = {'corners': corners,
                 'neighbors': [idx(i, j - 1), idx(i + 1, j), idx(i, j + 1), idx(i - 1, j)]}
            if dense:
                g = _bilinear(*np.column_stack([xy, np.zeros(4)]))
                g[..., 2] = h(g[..., 0], g[..., 1])
                q['dense_points'] = g + off
            if normals is not None:
                q['corner_normals'] = [np.asarray(normals(x, y), dtype=float) for x, y in xy]
            quads.append(q)
    return quads


def _interiors(patches):
    return np.array([p[1:5, 1:5] for p in patches])


_INWARD = {
    0: lambda idx, m: (m, idx),
    1: lambda idx, m: (5 - m, idx),
    2: lambda idx, m: (idx, m),
    3: lambda idx, m: (idx, 5 - m),
}


def _oracle_system(quad_mesh, weight):
    """Dense rows of the fitter's equations, written out independently.

    Unknowns: the 16 interior control points of every patch, local index
    (i-1)*4 + (j-1). Data rows: sum_ij B_i(u) B_j(v) P_ij = sample. Continuity
    rows (weight w) at every shared boundary column idx = 1..4:
        3 P1 - 3 P2 + P3 (patch k) + 3 P1 - 3 P2 + P3 (patch n) = 2 P0
    with P_m the m-th control point inward from the shared curve."""
    base = [g3._generate_single_patch(q) for q in quad_mesh]
    nv = 16 * len(base)
    rows, rhs = [], []

    def var(k, ij):
        i, j = ij
        assert 1 <= i <= 4 and 1 <= j <= 4
        return 16 * k + (i - 1) * 4 + (j - 1)

    for k, q in enumerate(quad_mesh):
        dense = q.get('dense_points')
        if dense is None:
            continue
        Bm = _bern_matrix(np.linspace(0.0, 1.0, dense.shape[0]))
        for r in range(dense.shape[0]):
            for c in range(dense.shape[1]):
                outer = np.outer(Bm[r], Bm[c])
                fixed = outer.copy()
                fixed[1:5, 1:5] = 0.0
                row = np.zeros(nv)
                row[16 * k:16 * (k + 1)] = outer[1:5, 1:5].ravel()
                rows.append(row)
                rhs.append(dense[r, c] - np.einsum('ij,ijc->c', fixed, base[k]))

    if weight > 0:
        done = set()
        for k, q in enumerate(quad_mesh):
            for nk in q.get('neighbors', []):
                if nk < 0 or tuple(sorted((k, nk))) in done:
                    continue
                cands = []
                for ek in range(4):
                    rk = np.array([base[k][_INWARD[ek](i, 0)] for i in range(6)])
                    for en in range(4):
                        rn = np.array([base[nk][_INWARD[en](i, 0)] for i in range(6)])
                        if np.abs(rk - rn).max() < 1e-7:
                            cands.append((ek, en, False))
                        elif np.abs(rk - rn[::-1]).max() < 1e-7:
                            cands.append((ek, en, True))
                if len(cands) != 1:
                    continue
                done.add(tuple(sorted((k, nk))))
                ek, en, rev = cands[0]
                for idx in range(1, 5):
                    idn = 5 - idx if rev else idx
                    row = np.zeros(nv)
                    for m, coef in ((1, 3.0), (2, -3.0), (3, 1.0)):
                        row[var(k, _INWARD[ek](idx, m))] += weight * coef
                        row[var(nk, _INWARD[en](idn, m))] += weight * coef
                    rows.append(row)
                    rhs.append(weight * 2.0 * base[k][_INWARD[ek](idx, 0)])
    x0 = np.concatenate([b[1:5, 1:5].reshape(16, 3) for b in base])
    return np.array(rows), np.array(rhs), x0, base


def _oracle_fit(quad_mesh, weight):
    """Minimum-norm change from the Coons interior that minimises the
    least-squares objective (numpy dense lstsq, SVD based)."""
    M, rhs, x0, _ = _oracle_system(quad_mesh, weight)
    dx = np.linalg.lstsq(M, rhs - M @ x0, rcond=None)[0]
    return (x0 + dx).reshape(-1, 4, 4, 3), M, rhs, x0


def _limit(p):
    """The fitter's divergence clamp: 1.25 x the corner bbox diagonal around
    the corner centroid (scale-free; tests/test_g3_clamp_scale.py)."""
    corners = np.array([p[0, 0], p[5, 0], p[5, 5], p[0, 5]])
    diag = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
    return corners.mean(axis=0), 1.25 * diag


# --------------------------------------------------------------------- M1

OFFSETS = [0.0, 10.0, 500.0, 1e4]


def _tol(off):
    """1e-9 on a unit patch, relaxed linearly beyond offset 1000: absolute
    coordinates of 1e4 carry only ~2e-12 of resolution, which the per-patch
    sample matrix (1/sigma_min^2 ~ 450) amplifies to ~1e-9 by itself."""
    return 1e-9 * max(1.0, off / 1000.0)


@pytest.mark.parametrize("off", OFFSETS)
def test_m1_plane_is_reproduced_at_any_offset(off, capsys):
    """M1 (cand 19): two flat unit quads with dense samples taken exactly
    from the plane, continuity_weight=20, shifted by (off, off, 0).

    The plane is exactly representable and the Coons interior already is
    that representation (zero residual in every equation), so the fit must
    keep it to ~1e-9 at every offset and must never report a divergence.
    Before the fix: interiors moved 0.015 at offset 0 and 0.13 at offset 10;
    at 500 and 1e4 both patches were flagged diverged and reset."""
    qm = _grid_mesh(2, 1, offset=(off, off, 0.0))
    M, rhs, x0, base = _oracle_system(qm, 20.0)
    # precondition: the plane is exactly representable, Coons is the solution
    assert np.abs(M @ x0 - rhs).max() < _tol(off)
    for b, q in zip(base, qm):
        assert np.abs(_eval(b, np.linspace(0, 1, 6)) - q['dense_points']).max() < _tol(off)

    patches = G3Fitter(continuity_weight=20.0).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out, \
        "an exactly representable plane was flagged as a diverged solve"
    moved = np.abs(_interiors(patches) - _interiors(base)).max()
    assert moved < _tol(off), f"offset {off}: plane interiors moved by {moved:.3e}"
    for p, q in zip(patches, qm):
        err = np.abs(_eval(p, np.linspace(0, 1, 6)) - q['dense_points']).max()
        assert err < _tol(off), f"offset {off}: surface off the plane by {err:.3e}"


@pytest.mark.parametrize("off", OFFSETS)
def test_m1_exact_bump_is_found_at_any_offset(off, capsys):
    """M1 (cand 19): the solve must actually REACH the optimum, not just
    leave a correct start alone. Dense samples of z = 0.3 u(1-u) v(1-v) on
    two unit quads (continuity_weight=20) have the closed-form exact
    solution P_ij.z = 0.3 c_i c_j with c = (0, .2, .3, .3, .2, 0), which also
    satisfies the continuity rows. Expected: that net to ~1e-9 at every
    offset (x, y and z shifted), i.e. a move of exactly 0.027 from Coons.
    Before the fix: 0.015 off at the origin, 0.13 at offset 10, reset to
    Coons at 500 and 1e4."""
    bump = 0.3
    o = np.array([off, -0.5 * off, 0.25 * off])

    def height(x, y):
        u, v = x - np.floor(np.minimum(x, 1.999999)), y
        return bump * u * (1 - u) * v * (1 - v)

    qm = _grid_mesh(2, 1, offset=o, height=height)
    expected = []
    for q in qm:
        net = _bilinear(*(q['corners'] - o)) + o
        net[..., 2] = o[2] + bump * np.outer(_C_U1MU, _C_U1MU)
        expected.append(net)
    M, rhs, x0, base = _oracle_system(qm, 20.0)
    xe = np.concatenate([e[1:5, 1:5].reshape(16, 3) for e in expected])
    # precondition: the closed-form net is an exact solution of every row,
    # and it is not the Coons start (the solve has to move)
    assert np.abs(M @ xe - rhs).max() < _tol(off)
    assert np.abs(M @ x0 - rhs).max() > 1e-3
    for e, q in zip(expected, qm):
        assert np.abs(_eval(e, np.linspace(0, 1, 6)) - q['dense_points']).max() < _tol(off)

    patches = G3Fitter(continuity_weight=20.0).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out
    err = np.abs(_interiors(patches) - np.array([e[1:5, 1:5] for e in expected])).max()
    assert err < _tol(off), f"offset {off}: interior off the exact net by {err:.3e}"
    moved = np.abs(_interiors(patches) - _interiors(base)).max()
    assert abs(moved - bump * 0.09) < _tol(off), f"moved {moved}, expected {bump * 0.09}"


def _wavy(x, y):
    return 0.35 * np.sin(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.05 * x * y


@pytest.mark.parametrize("weight", [5.0, 50.0])  # the converter's G1 and G3
def test_m1_curved_grid_matches_dense_lstsq_and_is_translation_invariant(weight, capsys):
    """M1 (cand 19): a 5x4 grid of quads sampled from a curved surface that
    a degree-5 net with straight boundary rows cannot represent (so the
    optimum has a real residual), continuity_weight 5 (G1) and 50 (G3).
    Expected: the fitted interiors equal the dense SVD least-squares optimum
    to ~1e-8, and the same grid shifted rigidly by 10, 500 or 1e4 gives the
    same shape (to 1e-8, relaxed linearly beyond offset 100 for the digits
    the absolute coordinates themselves lose). Before the fix the G3 fit at
    the origin was 0.22 off the optimum and at offset 500 every patch was
    reset to Coons."""
    qm0 = _grid_mesh(5, 4, height=_wavy)
    ref, M, rhs, x0 = _oracle_fit(qm0, weight)
    # preconditions: genuinely non-representable data, and an optimum that
    # is far from the Coons start
    assert np.linalg.norm(M @ ref.reshape(-1, 3) - rhs) > 1e-2
    assert np.abs(ref - x0.reshape(-1, 4, 4, 3)).max() > 1e-2
    # ... and the oracle is the optimum: normal-equation residual ~ 0, while
    # the same check rejects the Coons start
    g = M.T @ (M @ ref.reshape(-1, 3) - rhs)
    assert np.abs(g).max() < 1e-9
    assert np.abs(M.T @ (M @ x0 - rhs)).max() > 1e-2

    got0 = _interiors(G3Fitter(continuity_weight=weight).fit_surface(qm0))
    err = np.abs(got0 - ref).max()
    assert err < 1e-8, f"fit differs from the least-squares optimum by {err:.3e}"

    for off in (10.0, 500.0, 1e4):
        o = np.array([off, 0.7 * off, -0.3 * off])
        qm = _grid_mesh(5, 4, offset=o, height=_wavy)
        got = _interiors(G3Fitter(continuity_weight=weight).fit_surface(qm)) - o
        drift = np.abs(got - got0).max()
        assert drift < 1e-8 * max(1.0, off / 100.0), \
            f"offset {off}: the fit moved by {drift:.3e} under a rigid shift"
    assert RESET_MSG not in capsys.readouterr().out


def _checker(amp):
    rr, cc = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    return amp * (-1.0) ** (rr + cc)


def _clamp_fixture(noise):
    """Two separate quads (no shared edge) fitted in one call: patch 0 is a
    plane with optional checkerboard noise on its samples, patch 1 the exact
    bump z = 0.3 u(1-u) v(1-v), shifted to x in [3, 4]."""
    a = _grid_mesh(1, 1)[0]
    b = _grid_mesh(1, 1, offset=(3.0, 0.0, 0.0))[0]
    s = np.linspace(0.0, 1.0, 6)
    b['dense_points'][..., 2] = 0.3 * np.outer(s * (1 - s), s * (1 - s))
    a['dense_points'][..., 2] += _checker(noise)
    a['neighbors'] = [1, -1, -1, -1]   # listed, but there is no shared curve
    b['neighbors'] = [0, -1, -1, -1]
    return [a, b]


def test_clamp_still_resets_a_runaway_interior(capsys):
    """Divergence clamp (kept by the M1/M18 fix): checkerboard noise of
    +-0.25 on the samples of patch 0 is fitted by an interior that oscillates
    far outside 1.25 x the quad diagonal (the per-patch sample matrix has
    sigma_min ~ 1.5e-3). The optimum itself runs away, so the clamp must
    reset exactly that patch to its Coons interior and say so; the clean
    patch in the same call keeps its exact fitted interior (0.3 c_i c_j)."""
    weight = 20.0
    qm = _clamp_fixture(0.25)
    ref, M, _, _ = _oracle_fit(qm, weight)
    assert M.shape[0] == 72  # data rows only: no continuity row was matched
    base = [g3._generate_single_patch(q) for q in qm]
    ratio = []
    for k, b in enumerate(base):
        center, lim = _limit(b)
        ratio.append(np.linalg.norm(ref[k].reshape(-1, 3) - center, axis=1).max() / lim)
    # precondition: the optimum of patch 0 is far outside the clamp, patch 1
    # well inside it
    assert ratio[0] > 10.0 and ratio[1] < 0.9, ratio
    exact_b = base[1][1:5, 1:5].copy()
    exact_b[..., 2] = 0.3 * np.outer(_C_U1MU, _C_U1MU)[1:5, 1:5]
    assert np.abs(ref[1] - exact_b).max() < 1e-12

    patches = G3Fitter(continuity_weight=weight).fit_surface(qm)

    out = capsys.readouterr().out
    assert "reset 1 diverged patch interior" in out, out
    assert np.array_equal(patches[0][1:5, 1:5], base[0][1:5, 1:5])
    # the clean patch was solved, not reset
    assert np.abs(patches[1][1:5, 1:5] - exact_b).max() < 1e-9
    assert np.abs(patches[1][1:5, 1:5] - base[1][1:5, 1:5]).max() > 0.02


def test_clamp_checker_accepts_a_clean_fit(capsys):
    """Control for the clamp test (vacuous-test rule d): the same fixture
    with milder noise (optimum inside the limit) must NOT print a reset and
    must keep the fitted noisy interior."""
    qm = _clamp_fixture(0.0005)
    ref, _, _, _ = _oracle_fit(qm, 20.0)
    base = [g3._generate_single_patch(q) for q in qm]
    center, lim = _limit(base[0])
    assert np.linalg.norm(ref[0].reshape(-1, 3) - center, axis=1).max() < 0.9 * lim

    patches = G3Fitter(continuity_weight=20.0).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out
    assert np.abs(_interiors(patches) - ref).max() < 1e-9
    assert np.abs(patches[0][1:5, 1:5] - base[0][1:5, 1:5]).max() > 0.1


# -------------------------------------------------------------------- M18

@pytest.mark.parametrize("off", [0.0, 10.0])
def test_m18_no_dense_flat_quads_keep_coons_interior(off, capsys):
    """M18 (cand 18): two flat unit quads with neighbours but no
    dense_points, continuity_weight=20. The Coons interior already satisfies
    every continuity row, so the minimum change is zero: interiors must stay
    put (1e-12). Before the fix they moved by 1.56 towards the origin (offset
    0), or were reset as 'diverged' (offset 10)."""
    qm = _grid_mesh(2, 1, offset=(off, off, 0.0), dense=False)
    M, rhs, x0, base = _oracle_system(qm, 20.0)
    assert M.shape[0] == 4  # one shared edge, four continuity rows, no data
    assert np.abs(M @ x0 - rhs).max() < _tol(off)

    patches = G3Fitter(continuity_weight=20.0).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out
    moved = np.abs(_interiors(patches) - _interiors(base)).max()
    assert moved < 1e-12, f"offset {off}: interiors moved by {moved:.3e}"


def _tilted(x, y):
    # tilt the normals along the outer edge x = 0 only, so the left patch's
    # boundary curves get a cubic term the right patch does not have
    return (-0.5, 0.0, 1.0) if x < 0.5 else (0.0, 0.0, 1.0)


@pytest.mark.parametrize("off", [0.0, 10.0])
def test_m18_no_dense_moves_interior_by_the_minimum_norm_change(off, capsys):
    """M18 (cand 18): without dense_points but with curved boundary curves
    (tilted corner normals) the Coons interior violates the continuity rows.
    Expected: the interior moves by the minimum-norm change that satisfies
    them (dense SVD lstsq oracle), identical at offset 0 and 10."""
    o = np.array([off, off, 0.0])
    qm = _grid_mesh(2, 1, offset=o, dense=False, normals=_tilted)
    ref, M, rhs, x0 = _oracle_fit(qm, 20.0)
    # precondition: Coons violates the rows, the oracle satisfies them
    assert np.abs(M @ x0 - rhs).max() > 1e-2
    assert np.abs(M @ ref.reshape(-1, 3) - rhs).max() < 1e-9
    change = np.abs(ref - x0.reshape(-1, 4, 4, 3)).max()
    assert 1e-3 < change < 0.5, change

    patches = G3Fitter(continuity_weight=20.0).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out
    err = np.abs(_interiors(patches) - ref).max()
    assert err < 1e-9, f"offset {off}: not the minimum-norm change (off by {err:.3e})"


@pytest.mark.parametrize("weight", [5.0, 20.0])
@pytest.mark.parametrize("off", [0.0, 500.0])
def test_m18_patches_without_samples_next_to_fitted_ones(weight, off, capsys):
    """M18 (cand 18), mixed input: a 3x2 curved grid where two of the six
    quads carry no dense_points (rank 82 of 96 unknowns). This system takes
    the lsqr fallback, so it is held to lsqr's default 1e-6 tolerance, not
    1e-9: objective within 1e-3 of the minimum-norm optimum, interiors within
    0.05 of it (the optimum moves them up to 0.2 from Coons), and no patch
    flagged as diverged. Before the fix the sample-less interiors were
    pulled towards the origin and reset (objective 3 to 9e4 times too high)."""
    o = np.array([off, 0.5 * off, 0.2 * off])
    qm = _grid_mesh(3, 2, offset=o, height=_wavy)
    for k in (1, 4):
        del qm[k]['dense_points']
    ref, M, rhs, x0 = _oracle_fit(qm, weight)
    sv = np.linalg.svd(M, compute_uv=False)
    # preconditions: genuinely rank deficient, and the optimum is a real move
    assert (sv > sv[0] * 1e-12).sum() < M.shape[1]
    assert np.abs(ref - x0.reshape(-1, 4, 4, 3)).max() > 0.1
    j_opt = np.linalg.norm(M @ ref.reshape(-1, 3) - rhs) ** 2

    patches = G3Fitter(continuity_weight=weight).fit_surface(qm)

    assert RESET_MSG not in capsys.readouterr().out
    got = _interiors(patches)
    excess = np.linalg.norm(M @ got.reshape(-1, 3) - rhs) ** 2 / j_opt - 1.0
    assert excess < 1e-3, f"objective {excess:.2e} above the optimum"
    assert np.abs(got - ref).max() < 0.05
