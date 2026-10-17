"""Cylindrical features (holes, bosses, fillets) of a B-rep shape.

Purpose: per cylindrical face read the ``gp_Cylinder`` axis and radius, the
U span and the V range, decide concave / convex, and group coaxial faces of
the same radius into one ``CylFeature``. The dimensioning step uses these for
hole callouts and positions; the section step picks its plane through the
largest hole.

Measured facts this module relies on (REPORT-brep-analysis.md, OCP 7.9.3.1):

* ``BRepLProp_SLProps`` on a ``BRepAdaptor_Surface`` ignores the face
  orientation. Hole walls are ``TopAbs_REVERSED`` faces; without the manual
  flip all five holes of the probe plate came out as bosses (negative
  control, reproduced by ``_flip_reversed = False`` in the tests).
* ``gp_Cylinder.Axis().Location()`` is the surface origin and can lie outside
  the face (z = -5 for holes cut by a tool starting at z = -5, V range
  5..20), so the feature centre and length come from the V range.
  ``BRepTools.UVBounds_s`` returns (umin, umax, vmin, vmax); a full hole spans
  U 0..2pi, a split half 180 degrees, a fillet 90 degrees.
* An OCCT STEP round trip does not split periodic faces, but real exporters
  do; ``ShapeUpgrade_ShapeDivideClosed`` with one split point gives 12
  half-cylinder faces on the plate (19 faces) and the grouping merges them
  back into 6 features with ``n_faces == 2``.
* Concave is not the same as hole: an inside fillet is a concave partial
  cylinder. ``kind``: concave and full circle -> "hole"; concave partial ->
  "fillet"; convex -> "boss" (consumers check ``full_circle`` for outside
  rounds, which are convex partial cylinders).
* Plate result: 4 holes r 3.25 + 1 hole r 10 (length 15, centre z 7.5) and
  one boss r 15 length 10 centre (78, 30, 20).
* On an inverted solid (``plate.Reversed()``, signed volume < 0) every face
  orientation is flipped and all five holes read as bosses. The concavity of
  the faces of a solid with negative signed volume is therefore flipped back
  (``_heal_inverted``); faces outside any solid are taken as oriented.
* ``full_circle`` needs 360 degrees of U at every height of the group, not
  just in total: two half-cylinders stacked along the axis are partial.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

import numpy as np

from .model import CylFeature

logger = logging.getLogger(__name__)

# Flip the sampled normal of TopAbs_REVERSED faces. Module constant so the test
# can switch it off as a negative control (holes must then turn into bosses).
_flip_reversed = True

# Flip the concavity of faces of a solid with negative signed volume (an
# inverted solid, e.g. shape.Reversed() or a bad export). Module constant so the
# test can switch it off as a negative control (holes then read as bosses).
_heal_inverted = True

_FULL_CIRCLE_DEG = 360.0 - 1e-3
_KIND_ORDER = {"hole": 0, "boss": 1, "fillet": 2}


def _vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def _signed_volume(solid: Any) -> float:
    """Signed volume by OCCT's fixed Gauss scheme (only the sign is used)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    return float(props.Mass())


def _faces_with_inversion(shape: Any) -> List[Any]:
    """``(face, inverted)`` for every unique cylindrical face of ``shape``.

    ``inverted`` is True when the face belongs to a solid whose signed volume
    is negative (an inverted solid: every face orientation is flipped, so a
    hole would read as convex). Faces outside any solid are never inverted.
    The volume is only integrated for solids that have a cylindrical face.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_MapOfShape

    seen = TopTools_MapOfShape()

    def cyl_faces(exp) -> List[Any]:
        found = []
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            exp.Next()
            try:
                if not seen.Add(face):        # a face shared by two parents: once
                    continue
                if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
                    found.append(face)
            except Exception as exc:
                logger.debug("face type failed: %s", exc)
        return found

    out: List[Any] = []
    sexp = TopExp_Explorer(shape, TopAbs_SOLID)
    while sexp.More():
        solid = sexp.Current()
        sexp.Next()
        faces = cyl_faces(TopExp_Explorer(solid, TopAbs_FACE))
        inverted = False
        if faces and _heal_inverted:
            try:
                inverted = _signed_volume(solid) < 0.0
            except Exception as exc:
                logger.debug("solid volume failed: %s", exc)
            if inverted:
                logger.info("inverted solid: hole/boss concavity flipped back")
        out.extend((f, inverted) for f in faces)
    # faces of shells / face compounds outside any solid
    out.extend((f, False) for f in cyl_faces(TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)))
    return out


def _cylinder_faces(shape: Any) -> List[Dict[str, Any]]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED

    out: List[Dict[str, Any]] = []
    for face, inverted in _faces_with_inversion(shape):
        try:
            ad = BRepAdaptor_Surface(face)
            cyl = ad.Cylinder()
            ax = cyl.Axis()
            loc = _vec(ax.Location())
            d = _vec(ax.Direction())
            d /= np.linalg.norm(d)
            umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
            props = BRepLProp_SLProps(ad, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
            if not props.IsNormalDefined():
                continue
            nrm = _vec(props.Normal())
            if _flip_reversed and face.Orientation() == TopAbs_REVERSED:
                nrm = -nrm          # SLProps ignores the face orientation
            if inverted:
                nrm = -nrm          # the solid is inside out: material is outside
            rel = _vec(props.Value()) - loc
            radial = rel - np.dot(rel, d) * d
            rn = np.linalg.norm(radial)
            if rn <= 0:
                continue
            radial /= rn
            out.append({"loc": loc, "dir": d, "radius": float(cyl.Radius()),
                        "v": (float(vmin), float(vmax)),
                        "concave": bool(np.dot(nrm, radial) < 0),
                        "u_span_deg": math.degrees(umax - umin)})
        except Exception as exc:
            logger.debug("cylinder face skipped: %s", exc)
    return out


def _group(cfaces: List[Dict[str, Any]], radius_tol: float, axis_tol: float
           ) -> List[Dict[str, Any]]:
    """Merge coaxial same-radius faces with overlapping or touching V ranges."""
    groups: List[Dict[str, Any]] = []
    for c in cfaces:
        d = c["dir"]
        placed = False
        for g in groups:
            gd = g["dir"]
            if abs(abs(float(np.dot(d, gd))) - 1.0) > axis_tol:
                continue
            if abs(c["radius"] - g["radius"]) > radius_tol:
                continue
            if c["concave"] != g["concave"]:
                continue
            rel = c["loc"] - g["loc"]
            if np.linalg.norm(rel - np.dot(rel, gd) * gd) > radius_tol:
                continue                     # parallel but not the same axis line
            sgn = 1.0 if np.dot(d, gd) > 0 else -1.0
            off = float(np.dot(rel, gd))
            a, b = off + sgn * c["v"][0], off + sgn * c["v"][1]
            lo, hi = min(a, b), max(a, b)
            if lo > g["hi"] + radius_tol or hi < g["lo"] - radius_tol:
                continue                     # same axis, separate stretch
            g["lo"], g["hi"] = min(g["lo"], lo), max(g["hi"], hi)
            g["spans"].append((lo, hi, c["u_span_deg"]))
            g["n_faces"] += 1
            placed = True
            break
        if not placed:
            lo, hi = min(c["v"]), max(c["v"])
            groups.append({"loc": c["loc"], "dir": d, "radius": c["radius"],
                           "concave": c["concave"], "lo": lo, "hi": hi,
                           "spans": [(lo, hi, c["u_span_deg"])], "n_faces": 1})
    return groups


def _covers_full_circle(spans: List[Any], min_len: float) -> bool:
    """True when every height of the group is covered by 360 degrees of U.

    ``spans`` are ``(lo, hi, u_span_deg)`` per face in the group's axial
    coordinate. The axial range is cut at every face end; each piece longer
    than ``min_len`` must have the U spans of the faces covering it add up to
    a full circle. Adding all U spans of the group instead would call two
    half-cylinders stacked along the axis (each 180 degrees, at different
    heights) a full circle.
    """
    cuts = sorted({v for lo, hi, _ in spans for v in (lo, hi)})
    checked = False
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a <= min_len:
            continue                          # a sliver between touching faces
        mid = 0.5 * (a + b)
        u = sum(s for lo, hi, s in spans if lo <= mid <= hi)
        if u < _FULL_CIRCLE_DEG:
            return False
        checked = True
    if not checked:                           # zero-length group: plain U sum
        return sum(s for _, _, s in spans) >= _FULL_CIRCLE_DEG
    return True


def cylinder_features(shape, radius_tol: float = 0.01, axis_tol: float = 1e-4
                      ) -> List[CylFeature]:
    """Cylindrical features of ``shape``, sorted hole / boss / fillet, then by
    decreasing radius, then by centre.

    ``radius_tol`` (mm) is the tolerance for equal radii, for the distance
    between two axis lines and for touching V ranges; ``axis_tol`` is the
    allowed ``1 - |cos|`` between two axis directions. Never raises; a null
    shape gives ``[]``.

    Concavity follows the face orientation, corrected for inverted solids
    (negative signed volume, see ``census["inverted"]``); open shells and
    loose faces are taken as oriented, so an inside-out shell reports its
    holes as bosses. ``full_circle`` is True only when every axial stretch of
    the group is covered by 360 degrees of U.
    """
    if shape is None:
        return []
    try:
        if shape.IsNull():
            return []
        cfaces = _cylinder_faces(shape)
    except Exception as exc:
        logger.warning("cylinder detection failed: %s", exc)
        return []
    feats: List[CylFeature] = []
    for g in _group(cfaces, radius_tol, axis_tol):
        mid = 0.5 * (g["lo"] + g["hi"])
        full = _covers_full_circle(g["spans"], radius_tol)
        if g["concave"]:
            kind = "hole" if full else "fillet"
        else:
            kind = "boss"
        feats.append(CylFeature(center=g["loc"] + mid * g["dir"], axis=g["dir"].copy(),
                                radius=float(g["radius"]), length=float(g["hi"] - g["lo"]),
                                kind=kind, full_circle=bool(full), n_faces=int(g["n_faces"])))
    feats.sort(key=lambda f: (_KIND_ORDER.get(f.kind, 9), -round(f.radius, 6),
                              tuple(np.round(f.center, 6))))
    return feats
