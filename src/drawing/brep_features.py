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

# Mark holes filled by a cylinder of another solid (assembly fasteners) as
# kind "filled_hole". Module constant so the test can switch it off.
_detect_filled = True
_FILL_MIN_RATIO = 0.8      # filler radius >= this x hole radius (clearance holes)
_FILL_MIN_OVERLAP = 0.5    # filler covers >= this share of the hole length
_FILL_AXIS_TOL = 1e-4      # 1 - |cos| between hole and filler axes

_FULL_CIRCLE_DEG = 360.0 - 1e-3
_KIND_ORDER = {"hole": 0, "boss": 1, "fillet": 2, "filled_hole": 3}


def _vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def _signed_volume(solid: Any) -> float:
    """Signed volume by OCCT's fixed Gauss scheme (only the sign is used)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    return float(props.Mass())


def _faces_with_inversion(shape: Any, cylinders_only: bool = True) -> List[Any]:
    """``(face, inverted, body)`` for every unique cylindrical face of
    ``shape`` (every face with ``cylinders_only=False``).

    ``inverted`` is True when the face belongs to a solid whose signed volume
    is negative (an inverted solid: every face orientation is flipped, so a
    hole would read as convex). Faces outside any solid are never inverted.
    The volume is only integrated for solids that have a selected face.
    ``body`` is the index of the solid the face belongs to (in explorer
    order), -1 for faces outside any solid.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_MapOfShape

    seen = TopTools_MapOfShape()

    def sel_faces(exp) -> List[Any]:
        found = []
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            exp.Next()
            try:
                if not seen.Add(face):        # a face shared by two parents: once
                    continue
                if not cylinders_only or BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
                    found.append(face)
            except Exception as exc:
                logger.debug("face type failed: %s", exc)
        return found

    out: List[Any] = []
    sexp = TopExp_Explorer(shape, TopAbs_SOLID)
    body = 0
    while sexp.More():
        solid = sexp.Current()
        sexp.Next()
        faces = sel_faces(TopExp_Explorer(solid, TopAbs_FACE))
        inverted = False
        if faces and _heal_inverted:
            try:
                inverted = _signed_volume(solid) < 0.0
            except Exception as exc:
                logger.debug("solid volume failed: %s", exc)
            if inverted:
                logger.info("inverted solid: hole/boss concavity flipped back")
        out.extend((f, inverted, body) for f in faces)
        body += 1
    # faces of shells / face compounds outside any solid
    out.extend((f, False, -1) for f in sel_faces(TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)))
    return out


def _cylinder_faces(shape: Any) -> List[Dict[str, Any]]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED

    out: List[Dict[str, Any]] = []
    for face, inverted, body in _faces_with_inversion(shape):
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
                        "u_span_deg": math.degrees(umax - umin), "body": int(body)})
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
            g["bodies"].add(c.get("body", -1))
            placed = True
            break
        if not placed:
            lo, hi = min(c["v"]), max(c["v"])
            groups.append({"loc": c["loc"], "dir": d, "radius": c["radius"],
                           "concave": c["concave"], "lo": lo, "hi": hi,
                           "spans": [(lo, hi, c["u_span_deg"])], "n_faces": 1,
                           "bodies": {c.get("body", -1)}})
    return groups


def _axial_range(g: Dict[str, Any], origin: np.ndarray, axis: np.ndarray) -> Any:
    """The group's axial stretch in the coordinate t = (p - origin) . axis."""
    a = float(np.dot(g["loc"] + g["lo"] * g["dir"] - origin, axis))
    b = float(np.dot(g["loc"] + g["hi"] * g["dir"] - origin, axis))
    return min(a, b), max(a, b)


def _mark_filled(groups: List[Dict[str, Any]], radius_tol: float, axis_tol: float) -> int:
    """Set ``g["filled"]`` on full concave groups (holes) that a convex
    cylinder of ANOTHER solid fills: a bolt shank, a pin or a rod end in an
    assembly. The filler must be coaxial within the clearance (axis lines at
    most ``radius_tol`` + (hole r - filler r) apart, axes parallel within
    ``_FILL_AXIS_TOL``), have a radius between ``_FILL_MIN_RATIO`` x and the
    hole radius (+ ``radius_tol``), and cover at least ``_FILL_MIN_OVERLAP``
    of the hole length. Returns the number of holes marked."""
    n = 0
    bosses = [g for g in groups if not g["concave"]]
    for h in groups:
        if not h["concave"] or not h.get("full", False):
            continue
        hb = {b for b in h["bodies"] if b >= 0}
        if not hb:
            continue
        hd = h["dir"]
        h0, h1 = _axial_range(h, h["loc"], hd)
        need = _FILL_MIN_OVERLAP * max(h1 - h0, 0.0)
        for b in bosses:
            bb = {x for x in b["bodies"] if x >= 0}
            if not bb or bb & hb:
                continue                       # same solid (or no solid): a real boss
            if abs(abs(float(np.dot(b["dir"], hd))) - 1.0) > max(axis_tol, _FILL_AXIS_TOL):
                continue
            if not (_FILL_MIN_RATIO * h["radius"] <= b["radius"] <= h["radius"] + radius_tol):
                continue
            rel = b["loc"] - h["loc"]
            off_axis = float(np.linalg.norm(rel - np.dot(rel, hd) * hd))
            if off_axis > radius_tol + (h["radius"] - b["radius"]):
                continue
            b0, b1 = _axial_range(b, h["loc"], hd)
            if min(h1, b1) - max(h0, b0) < need - radius_tol:
                continue
            h["filled"] = True
            n += 1
            break
    return n


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

    In a shape with more than one solid (an assembly) a hole that a convex
    cylinder of another solid fills (a bolt shank, a pin; see
    ``_mark_filled``) gets ``kind == "filled_hole"``: the dimensioning step
    gives callouts to ``"hole"`` only, so no callout points at a fastener.
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
    groups = _group(cfaces, radius_tol, axis_tol)
    for g in groups:
        g["full"] = _covers_full_circle(g["spans"], radius_tol)
    n_bodies = len({b for g in groups for b in g["bodies"] if b >= 0})
    if _detect_filled and n_bodies > 1:
        n_filled = _mark_filled(groups, radius_tol, axis_tol)
        if n_filled:
            logger.info("%d hole(s) filled by a cylinder of another solid: kind 'filled_hole'",
                        n_filled)
    for g in groups:
        mid = 0.5 * (g["lo"] + g["hi"])
        full = g["full"]
        if g["concave"]:
            kind = ("filled_hole" if g.get("filled") else "hole") if full else "fillet"
        else:
            kind = "boss"
        feats.append(CylFeature(center=g["loc"] + mid * g["dir"], axis=g["dir"].copy(),
                                radius=float(g["radius"]), length=float(g["hi"] - g["lo"]),
                                kind=kind, full_circle=bool(full), n_faces=int(g["n_faces"])))
    feats.sort(key=lambda f: (_KIND_ORDER.get(f.kind, 9), -round(f.radius, 6),
                              tuple(np.round(f.center, 6))))
    return feats


# ---------------------------------------------------------------------------
# depth sign: which side of the part should face the viewer
# ---------------------------------------------------------------------------

_FACING_COS = 0.5          # a face looks at the viewer when n_face . n > this
_MEAN_NORMAL_GRID = 6      # UV samples per direction for the mean normal of a curved face

# Use the area-weighted mean normal of curved faces (False: the old single
# normal at the UV-box centre). Module constant so the test can switch it off
# as a negative control (a through bore then counts for one side, depending on
# the reference direction of its cylinder).
_mean_normal = True


def _face_mean_normal(face: Any, ad: Any, uv: Any) -> Any:
    """Area-weighted mean of the (geometric, unflipped) surface normal of
    ``face``: sum of dS/du x dS/dv over a ``_MEAN_NORMAL_GRID`` squared grid
    of cell midpoints of the UV box, divided by the summed |dS/du x dS/dv|.
    Points the face classifier puts outside the trimmed face are skipped
    (all points are used when none is inside). A plane returns its unit
    normal from one sample; a full cylinder, sphere or torus returns about 0;
    a quarter cylinder (edge round) about 0.9 along its mid normal; a 90 deg
    countersink cone 0.71 along its axis. ``None`` when no sample has a
    tangent plane."""
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                             GeomAbs_Torus)
    from OCP.TopAbs import TopAbs_OUT
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

    umin, umax, vmin, vmax = uv
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()

    def cross_at(u: float, v: float) -> Any:
        ad.D1(u, v, p, du, dv)
        ax, ay, az = du.X(), du.Y(), du.Z()
        bx, by, bz = dv.X(), dv.Y(), dv.Z()
        return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    def unit_at(u: float, v: float) -> Any:
        c = np.array(cross_at(u, v))
        cn = float(np.linalg.norm(c))
        return c / cn if cn > 0 else None

    kind = ad.GetType()
    vmid = 0.5 * (vmin + vmax)
    if kind == GeomAbs_Plane:
        return unit_at(0.5 * (umin + umax), vmid)
    if kind in (GeomAbs_Cylinder, GeomAbs_Cone):
        # unit normal n(u) = a cos u + b sin u + c, independent of v, and the
        # area element does not depend on u: the mean is the u-mean of n(u),
        # in closed form from three samples (the analytic surface is defined
        # outside the face too). The U bounds come from the face's pcurves.
        n0, n1, n2 = unit_at(0.0, vmid), unit_at(0.5 * math.pi, vmid), unit_at(math.pi, vmid)
        if n0 is not None and n1 is not None and n2 is not None:
            c = 0.5 * (n0 + n2)
            a = 0.5 * (n0 - n2)
            b = n1 - c
            span = umax - umin
            if span >= 2.0 * math.pi - 1e-9 or span <= 0.0:
                return c if span > 0.0 else unit_at(0.5 * (umin + umax), vmid)
            s = (math.sin(umax) - math.sin(umin)) / span
            co = (math.cos(umin) - math.cos(umax)) / span
            return a * s + b * co + c
    k = _MEAN_NORMAL_GRID
    us = [umin + (i + 0.5) * (umax - umin) / k for i in range(k)]
    vs = [vmin + (i + 0.5) * (vmax - vmin) / k for i in range(k)]
    cls = None
    if kind not in (GeomAbs_Sphere, GeomAbs_Torus):    # free-form: trims are not UV-box aligned
        try:
            cls = BRepTopAdaptor_FClass2d(face, 1e-7)
        except Exception:
            cls = None
    s_in, a_in, s_all, a_all = [0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], 0.0
    for u in us:
        for v in vs:
            cx, cy, cz = cross_at(u, v)
            a = math.sqrt(cx * cx + cy * cy + cz * cz)
            s_all[0] += cx
            s_all[1] += cy
            s_all[2] += cz
            a_all += a
            inside = True
            if cls is not None:
                try:
                    inside = cls.Perform(gp_Pnt2d(u, v)) != TopAbs_OUT
                except Exception:
                    inside = True
            if inside:
                s_in[0] += cx
                s_in[1] += cy
                s_in[2] += cz
                a_in += a
    if a_in > 0:
        return np.array(s_in) / a_in
    return np.array(s_all) / a_all if a_all > 0 else None


def depth_sign_details(shape, n) -> Dict[str, Any]:
    """Per side of the view normal ``n``: how much of the part faces a viewer
    standing there.

    For every face the area-weighted mean outward normal is read
    (``_face_mean_normal``; flipped for ``TopAbs_REVERSED`` faces and for
    faces of an inverted solid). A face faces the ``+n`` viewer when
    ``n_mean . n > 0.5`` and the ``-n`` viewer when it is below -0.5.
    Features add faces on the side they open to: boss tops, pocket and
    blind-hole floors, countersink and chamfer cones, edge rounds; through
    holes and side walls count for neither side. The mean normal of a full
    cylinder is about 0, so a bore across the view normal counts for neither
    side whatever reference direction its cylinder has (a single normal at
    the UV-box centre depended on it: 1 vs 6 faces on a cross-bored slab).
    A split bore gives one half cylinder per side (mean normal 0.64 each).

    Returns ``faces_plus`` / ``faces_minus`` (counts), ``area_plus`` /
    ``area_minus`` (mm2 of those faces), ``area_total`` (their sum; the
    side walls are not integrated), ``score_plus`` /
    ``score_minus`` (count + facing area / area_total, so the area only
    breaks ties of the count) and ``faces`` (faces read). A null or empty
    shape gives zeros. Cost (measured): 0.22 to 0.42 s on a 663-face NIST
    part (most of it the face list with the signed volume per solid and
    the areas); 7.5 s on the 10665-face RC buggy assembly against 6.6 s
    with the centre normal, the mean normals themselves 1.8 s there
    (planes 1 sample, cylinders and cones 3 samples in closed form,
    B-splines a 6 x 6 grid with the face classifier: 1.1 s for 1504 faces).
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_REVERSED

    nv = np.asarray(n, dtype=float).reshape(3)
    nn = float(np.linalg.norm(nv))
    if nn < 1e-12 or not np.all(np.isfinite(nv)):
        raise ValueError(f"depth_sign_details: bad view normal {n!r}")
    nv = nv / nn
    out = {"faces_plus": 0, "faces_minus": 0, "area_plus": 0.0, "area_minus": 0.0,
           "area_total": 0.0, "score_plus": 0.0, "score_minus": 0.0, "faces": 0}
    if shape is None:
        return out
    try:
        if shape.IsNull():
            return out
        faces = _faces_with_inversion(shape, cylinders_only=False)
    except Exception as exc:
        logger.warning("depth sign scoring failed: %s", exc)
        return out
    for face, inverted, _ in faces:
        try:
            ad = BRepAdaptor_Surface(face)
            umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
            nrm = None
            if _mean_normal and all(math.isfinite(x) for x in (umin, umax, vmin, vmax)):
                nrm = _face_mean_normal(face, ad, (umin, umax, vmin, vmax))
            if nrm is None:
                props = BRepLProp_SLProps(ad, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
                if not props.IsNormalDefined():
                    continue
                nrm = _vec(props.Normal())
            if face.Orientation() == TopAbs_REVERSED:
                nrm = -nrm
            if inverted:
                nrm = -nrm
            c = float(np.dot(nrm, nv))
            area = 0.0
            if abs(c) > _FACING_COS:          # side walls need no area
                gp = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, gp)
                area = abs(float(gp.Mass()))
        except Exception as exc:
            logger.debug("depth sign: face skipped: %s", exc)
            continue
        out["faces"] += 1
        if c > _FACING_COS:
            out["faces_plus"] += 1
            out["area_plus"] += area
        elif c < -_FACING_COS:
            out["faces_minus"] += 1
            out["area_minus"] += area
    out["area_total"] = out["area_plus"] + out["area_minus"]
    tot = out["area_total"]
    out["score_plus"] = out["faces_plus"] + (out["area_plus"] / tot if tot > 0 else 0.0)
    out["score_minus"] = out["faces_minus"] + (out["area_minus"] / tot if tot > 0 else 0.0)
    return out


def score_depth_sign(shape, n) -> Any:
    """``(score_plus, score_minus)``: how much of ``shape`` faces a viewer on
    the ``+n`` side and on the ``-n`` side of the view normal ``n`` (see
    ``depth_sign_details``). The pipeline looks along ``-n`` from the ``+n``
    side (``VIEWS[name].n`` points to the viewer); when ``score_minus`` is
    clearly larger, a 180-degree turn about the vertical axis shows the
    featured side. Never raises on a null shape ((0.0, 0.0))."""
    d = depth_sign_details(shape, n)
    return float(d["score_plus"]), float(d["score_minus"])
