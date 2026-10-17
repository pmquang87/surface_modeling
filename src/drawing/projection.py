"""View directions, world-to-view transform, auto orientation and view cells.

Pure numpy, no OCP import: ``ViewSpec.gp_ax2_args`` returns plain tuples that
the B-rep route turns into ``gp_Ax2(gp_Pnt(*o), gp_Dir(*n), gp_Dir(*x))``.

Measured facts this module relies on (HLR probe, OCP 7.9.3, REPORT-hlr.md):

* ``HLRAlgo_Projector(gp_Ax2(P, N, Vx))`` puts the viewer on the +N side: the
  faces whose normals point toward +N are visible. The 2D result is
  x = Vx.(p - P), y = (N x Vx).(p - P), Z = 0. Ten labelled feature centres
  matched this to 1e-3 in all seven views below.
* World convention: Z up, the front view looks at the -Y face (viewer at -Y
  looking toward +Y), X to the right. Verified table (N toward the viewer,
  Vx = 2D +x):

  ======  ==========  ==========  ======================
  name    N           Vx          2D =
  ======  ==========  ==========  ======================
  front   (0,-1,0)    (1,0,0)     (X, Z)
  back    (0,1,0)     (-1,0,0)    (-X, Z)
  top     (0,0,1)     (1,0,0)     (X, Y)
  bottom  (0,0,-1)    (1,0,0)     (X, -Y)
  right   (1,0,0)     (0,1,0)     (Y, Z)
  left    (-1,0,0)    (0,-1,0)    (-Y, Z)
  iso     (1,-1,1)    (1,1,0)     up = (-0.408, 0.408, 0.816)
  ======  ==========  ==========  ======================

* The 2D coordinates are not shifted: back, left and bottom views come out
  with negative ranges, so the layout translates every view by its own bbox.
* First- versus third-angle placement changes only the sheet cell, never the
  gp_Ax2 values.

The mesh route uses the same ``ViewSpec`` (x = Vx, y = N x Vx, depth = N.p,
larger = nearer the viewer) so both routes agree to the sign.

Arrangement cells (ISO 5456-2:1996 5.1/5.2, REPORT-standards.md), as
(col, row) relative to the front view, row +1 = above:

* first angle: top below (0,-1), bottom above (0,1), right-side view on the
  LEFT (-1,0), left-side view on the right (1,0), rear (2,0);
* third angle: mirrored through the front view.

The iso view goes to the free top-right quadrant: (1, 1) in both projections
(above the left-view cell in first angle, above the right-view cell in third
angle, as the spec section 3.3 says). The layout moves it if that cell is
taken.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)

Vec3 = Tuple[float, float, float]

_ORTHO_TOL = 1e-9


def _as_unit(v, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{what} must be a finite 3-vector, got {v!r}")
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError(f"{what} must be non-zero, got {v!r}")
    return a / norm


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class ViewSpec:
    """One view direction.

    ``n`` is the unit normal toward the viewer (= the OCCT gp_Ax2 main
    direction), ``x`` the unit view +x in world (= the gp_Ax2 X direction).
    Both are normalised on construction and stored read-only.

    Equality and hash are by value: (name, n, x) with the axis components
    rounded to 12 decimals, so a copy, a rebuilt spec or an unpickled one
    compares equal to ``VIEWS[name]``.
    """

    name: str
    n: np.ndarray
    x: np.ndarray

    def _key(self):
        return (self.name,
                tuple(round(float(c), 12) + 0.0 for c in self.n),
                tuple(round(float(c), 12) + 0.0 for c in self.x))

    def __eq__(self, other):
        if not isinstance(other, ViewSpec):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __post_init__(self):
        n = _as_unit(self.n, f"ViewSpec {self.name!r} n")
        x = _as_unit(self.x, f"ViewSpec {self.name!r} x")
        if abs(float(np.dot(n, x))) > _ORTHO_TOL:
            raise ValueError(f"ViewSpec {self.name!r}: n and x are not orthogonal "
                             f"(dot = {float(np.dot(n, x)):.3g})")
        object.__setattr__(self, "n", _readonly(n))
        object.__setattr__(self, "x", _readonly(x))

    @property
    def y(self) -> np.ndarray:
        """View +y in world: cross(n, x)."""
        return np.cross(self.n, self.x)

    def to_view(self, pts: np.ndarray) -> np.ndarray:
        """(n,3) world points -> (n,3) columns [x, y, depth].

        depth = dot(p, n): larger means nearer the viewer. A single (3,) point
        is accepted and returns shape (1, 3).
        """
        p = np.asarray(pts, dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"to_view expects (n,3) points, got shape {np.shape(pts)}")
        basis = np.stack([self.x, self.y, self.n], axis=1)   # columns x, y, n
        return p @ basis

    def gp_ax2_args(self) -> Tuple[Vec3, Vec3, Vec3]:
        """(origin, n, x) as plain float tuples for gp_Ax2(gp_Pnt, gp_Dir, gp_Dir)."""
        return ((0.0, 0.0, 0.0),
                tuple(float(c) for c in self.n),
                tuple(float(c) for c in self.x))


# Verified probe table: name -> (N toward the viewer, Vx).
_TABLE = {
    "front": ((0, -1, 0), (1, 0, 0)),
    "back": ((0, 1, 0), (-1, 0, 0)),
    "top": ((0, 0, 1), (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
    "right": ((1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, -1, 0)),
    "iso": ((1, -1, 1), (1, 1, 0)),
}

VIEWS: Dict[str, ViewSpec] = {name: ViewSpec(name, np.array(n, float), np.array(x, float))
                              for name, (n, x) in _TABLE.items()}
PRINCIPAL = ("front", "back", "top", "bottom", "right", "left")


# ------------------------------------------------------------------ orientation

_ORIENT_MODES = ("auto", "fixed", "y_up")
_UP_AXES = ("z", "y")

# y-up file -> z-up world: (x, y, z) -> (x, -z, y); det +1.
_Y_UP = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])


@dataclass(frozen=True, eq=False)
class Orientation:
    """A signed axis permutation applied to the model before projection.

    ``matrix`` is a 3x3 proper rotation (det +1) with world' = matrix @ world.
    Row i of the matrix is ``flips[i] * e[permutation[i]]``: new axis i takes
    old axis ``permutation[i]`` with sign ``flips[i]``.

    Equality and hash are by value on (permutation, flips, mode); the matrix
    follows from permutation and flips.
    """

    matrix: np.ndarray
    permutation: Tuple[int, int, int]
    flips: Tuple[int, int, int]
    mode: str

    def _key(self):
        return (tuple(int(i) for i in self.permutation),
                tuple(int(f) for f in self.flips), self.mode)

    def __eq__(self, other):
        if not isinstance(other, Orientation):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Rotate points: (n,3) -> (n,3), or (3,) -> (3,)."""
        p = np.asarray(pts, dtype=float)
        if p.shape[-1] != 3:
            raise ValueError(f"apply expects (...,3) points, got shape {p.shape}")
        return p @ self.matrix.T

    def to_dict(self) -> dict:
        """JSON-ready summary for the report (``geometry.orientation``)."""
        return {"permutation": [int(i) for i in self.permutation],
                "flips": [int(f) for f in self.flips],
                "mode": self.mode}


def _orientation_from_matrix(m: np.ndarray, mode: str) -> Orientation:
    m = np.array(np.rint(m), dtype=float)
    perm = tuple(int(np.argmax(np.abs(m[i]))) for i in range(3))
    flips = tuple(int(np.sign(m[i, perm[i]])) for i in range(3))
    if sorted(perm) != [0, 1, 2] or 0 in flips:
        raise ValueError(f"not a signed permutation matrix: {m.tolist()}")
    if abs(np.linalg.det(m) - 1.0) > 1e-9:
        raise ValueError(f"orientation matrix is not a proper rotation: {m.tolist()}")
    m.setflags(write=False)
    return Orientation(matrix=m, permutation=perm, flips=flips, mode=mode)


def _auto_permutation(ext: np.ndarray) -> np.ndarray:
    """Signed permutation sending longest -> X, shortest -> Y, middle -> Z.

    Ties keep the file axes: the permutations are tried with the identity
    first and the first one that satisfies ext[X] >= ext[Z] >= ext[Y] wins.
    An odd permutation (det -1, a mirror) gets its Y row negated, so only the
    depth direction flips (front and back swap), never the up direction.
    """
    tol = 1e-9 * max(float(np.max(ext)), 1e-300)
    chosen = None
    for perm in itertools.permutations((0, 1, 2)):
        ex, ey, ez = ext[perm[0]], ext[perm[1]], ext[perm[2]]
        if ex >= ez - tol and ez >= ey - tol:
            chosen = perm
            break
    assert chosen is not None  # sorting always yields a valid permutation
    m = np.zeros((3, 3))
    m[[0, 1, 2], list(chosen)] = 1.0
    if np.linalg.det(m) < 0:
        m[1] *= -1.0
    return m


def auto_orientation(extents: np.ndarray, mode: str = "auto", up: str = "z") -> Orientation:
    """Choose the model rotation applied before projection.

    ``extents`` are the bbox extents in the file's own axes. ``up="y"`` (or
    ``mode="y_up"``) first rotates a y-up file to z-up with
    (x, y, z) -> (x, -z, y). Then ``mode="auto"`` permutes the axes so that the
    longest extent is X (width), the shortest Y (depth) and the remaining one
    Z (height), with signs chosen so that det = +1; ``"fixed"`` and ``"y_up"``
    keep the (possibly up-rotated) axes.

    Note: in ``"auto"`` mode the extent order decides the result, so ``up``
    only matters when extents tie; it matters fully with ``"fixed"``.
    Negative extents raise ``ValueError`` (extents are max - min; a negative
    value means the caller passed something else, e.g. a bbox corner).
    """
    if mode not in _ORIENT_MODES:
        raise ValueError(f"orient mode must be one of {_ORIENT_MODES}, got {mode!r}")
    if up not in _UP_AXES:
        raise ValueError(f"up must be one of {_UP_AXES}, got {up!r}")
    ext = np.asarray(extents, dtype=float).reshape(-1)
    if ext.shape != (3,) or not np.all(np.isfinite(ext)):
        raise ValueError(f"extents must be three finite numbers, got {extents!r}")
    if np.any(ext < 0.0):
        raise ValueError(f"extents must be non-negative (max - min), got {extents!r}")

    base = _Y_UP if (up == "y" or mode == "y_up") else np.eye(3)
    m = base
    if mode == "auto":
        rotated_ext = np.abs(base @ ext)
        m = _auto_permutation(rotated_ext) @ base
    o = _orientation_from_matrix(m, mode)
    log.debug("orientation %s: permutation %s flips %s", mode, o.permutation, o.flips)
    return o


# ------------------------------------------------------------------ view cells

_FIRST_CELLS = {
    "front": (0, 0),
    "top": (0, -1),
    "bottom": (0, 1),
    "right": (-1, 0),
    "left": (1, 0),
    "back": (2, 0),
    "iso": (1, 1),
}
_THIRD_CELLS = {
    "front": (0, 0),
    "top": (0, 1),
    "bottom": (0, -1),
    "right": (1, 0),
    "left": (-1, 0),
    "back": (-2, 0),
    "iso": (1, 1),
}
_PROJECTIONS = {"first": _FIRST_CELLS, "first_angle": _FIRST_CELLS,
                "third": _THIRD_CELLS, "third_angle": _THIRD_CELLS}


def grid_cell(view: str, projection: str) -> Tuple[int, int]:
    """(col, row) of ``view`` relative to the front view at (0, 0); row +1 = above.

    ``projection`` is ``"first"`` / ``"third"`` (``"first_angle"`` and
    ``"third_angle"`` are accepted too). The iso cell is a default the layout
    may move when it is taken.
    """
    cells = _PROJECTIONS.get(projection)
    if cells is None:
        raise ValueError(f"projection must be 'first' or 'third', got {projection!r}")
    if view not in cells:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(cells)}")
    return cells[view]
