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
from typing import Any, Dict, List, Optional

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
_KIND_ORDER = {"hole": 0, "boss": 1, "fillet": 2, "filled_hole": 3, "round": 4}

# Report toroidal faces (edge rounds of circular edges) as kind "round"
# features (QA round 2 issue 16). Module constant so the test can switch it off.
_detect_rounds = True

# A filler of a filled hole counts as a fastener (``CylFeature.filler_bodies``;
# the section leaves it unsectioned, QA round 2 issue 28) when its solid is
# slender: volume <= this x pi r^2 L of its filling cylinder (bolt with head
# about 1.4 to 1.5, pin 1.0, a plate with a spigot >> 4) and it has no hole of
# at least half its radius (a bushing or sleeve is sectioned).
_FASTENER_VOL_RATIO = 4.0
_FASTENER_BORE_RATIO = 0.5


def _vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def _signed_volume(solid: Any) -> float:
    """Signed volume by OCCT's fixed Gauss scheme (only the sign is used)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    return float(props.Mass())


def _faces_with_inversion(shape: Any, cylinders_only: bool = True,
                          types: Any = None) -> List[Any]:
    """``(face, inverted, body)`` for every unique cylindrical face of
    ``shape`` (every face with ``cylinders_only=False``; the faces whose
    ``BRepAdaptor_Surface.GetType()`` is in ``types`` when given).

    ``inverted`` is True when the face belongs to a solid whose signed volume
    is negative (an inverted solid: every face orientation is flipped, so a
    hole would read as convex). Faces outside any solid are never inverted.
    The volume is only integrated for solids that have a selected face, and
    only once per unique solid (``TShape``; F7-perf item H): the instance's
    sign is the unique solid's sign times that of ``ScaleFactor ** 3`` of
    its location, flipped for a REVERSED instance (203 integrations, 4.35 s
    -> one per unique solid on the RC buggy assembly).
    ``body`` is the index of the solid the face belongs to (in explorer
    order, each solid instance once: ``_solid_list``), -1 for faces outside
    any solid.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_MapOfShape

    unique = TopTools_IndexedMapOfShape()
    unique_volume: dict = {}
    ident = TopLoc_Location()

    def inverted_solid(solid) -> bool:
        base = solid.Located(ident).Oriented(TopAbs_FORWARD)
        idx = unique.Add(base)
        if idx not in unique_volume:
            unique_volume[idx] = _signed_volume(base)
        f = float(solid.Location().Transformation().ScaleFactor()) ** 3
        if solid.Orientation() == TopAbs_REVERSED:
            f = -f
        return unique_volume[idx] * f < 0.0

    seen = TopTools_MapOfShape()

    def sel_faces(exp) -> List[Any]:
        found = []
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            exp.Next()
            try:
                if not seen.Add(face):        # a face shared by two parents: once
                    continue
                if types is not None:
                    if BRepAdaptor_Surface(face).GetType() in types:
                        found.append(face)
                elif not cylinders_only or BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
                    found.append(face)
            except Exception as exc:
                logger.debug("face type failed: %s", exc)
        return found

    out: List[Any] = []
    sexp = TopExp_Explorer(shape, TopAbs_SOLID)
    seen_solids = TopTools_MapOfShape()
    body = 0
    while sexp.More():
        solid = sexp.Current()
        sexp.Next()
        if not seen_solids.Add(solid):       # the same instance twice: once (as _solid_list)
            continue
        faces = sel_faces(TopExp_Explorer(solid, TopAbs_FACE))
        inverted = False
        if faces and _heal_inverted:
            try:
                inverted = inverted_solid(solid)
            except Exception as exc:
                logger.debug("solid volume failed: %s", exc)
            if inverted:
                logger.info("inverted solid: hole/boss concavity flipped back")
        out.extend((f, inverted, body) for f in faces)
        body += 1
    # faces of shells / face compounds outside any solid
    out.extend((f, False, -1) for f in sel_faces(TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)))
    return out


def _solid_list(shape: Any) -> List[Any]:
    """The solids of ``shape`` in explorer order, each instance (TShape +
    location) once: index i is the ``body`` of ``_faces_with_inversion`` and
    the solid index of ``brep_sections._parts`` (solids come first there)."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_MapOfShape
    seen = TopTools_MapOfShape()
    out: List[Any] = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        if seen.Add(ex.Current()):
            out.append(ex.Current())
        ex.Next()
    return out


def _cylinder_faces(shape: Any, faces: Any = None) -> List[Dict[str, Any]]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED

    out: List[Dict[str, Any]] = []
    for face, inverted, body in (_faces_with_inversion(shape) if faces is None else faces):
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
            h["filler"] = b
            n += 1
            break
    return n


def _torus_faces(faces: Any) -> List[Dict[str, Any]]:
    """Per toroidal face (``(face, inverted, body)`` items): axis, centre,
    major / minor radius, U span (around the main axis, degrees) and
    concavity (the normal points toward the tube's centre circle)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED

    out: List[Dict[str, Any]] = []
    for face, inverted, body in faces:
        try:
            ad = BRepAdaptor_Surface(face)
            tor = ad.Torus()
            ax = tor.Axis()
            loc = _vec(ax.Location())
            d = _vec(ax.Direction())
            d /= np.linalg.norm(d)
            umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
            props = BRepLProp_SLProps(ad, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
            if not props.IsNormalDefined():
                continue
            nrm = _vec(props.Normal())
            if _flip_reversed and face.Orientation() == TopAbs_REVERSED:
                nrm = -nrm
            if inverted:
                nrm = -nrm
            p = _vec(props.Value())
            rel = p - loc
            radial = rel - np.dot(rel, d) * d
            rn = float(np.linalg.norm(radial))
            big = float(tor.MajorRadius())
            tube = loc + (radial / rn * big if rn > 0 else np.zeros(3))
            out.append({"loc": loc, "dir": d, "major": big, "radius": float(tor.MinorRadius()),
                        "concave": bool(np.dot(nrm, p - tube) < 0),
                        "u_span_deg": math.degrees(umax - umin), "body": int(body)})
        except Exception as exc:
            logger.debug("torus face skipped: %s", exc)
    return out


def _round_features(tfaces: List[Dict[str, Any]], radius_tol: float, axis_tol: float
                    ) -> List[CylFeature]:
    """Group toroidal faces of the same circular edge round (same axis,
    centre, major and minor radius, concavity) into kind "round" features:
    ``radius`` = the round (minor) radius, ``major_radius`` = the torus major
    radius, ``center`` = the torus centre, ``length`` 0, ``full_circle`` when
    the U spans add up to 360 degrees."""
    groups: List[Dict[str, Any]] = []
    for t in tfaces:
        for g in groups:
            if abs(abs(float(np.dot(t["dir"], g["dir"]))) - 1.0) > axis_tol:
                continue
            if abs(t["radius"] - g["radius"]) > radius_tol or t["concave"] != g["concave"]:
                continue
            if abs(t["major"] - g["major"]) > radius_tol:
                continue
            if float(np.linalg.norm(t["loc"] - g["loc"])) > radius_tol:
                continue
            g["u"] += t["u_span_deg"]
            g["n"] += 1
            g["bodies"].add(t["body"])
            break
        else:
            groups.append({"loc": t["loc"], "dir": t["dir"], "radius": t["radius"],
                           "major": t["major"], "concave": t["concave"], "u": t["u_span_deg"],
                           "n": 1, "bodies": {t["body"]}})
    return [CylFeature(center=g["loc"].copy(), axis=g["dir"].copy(), radius=float(g["radius"]),
                       length=0.0, kind="round", full_circle=bool(g["u"] >= _FULL_CIRCLE_DEG),
                       n_faces=int(g["n"]), bodies=tuple(sorted(g["bodies"])),
                       major_radius=float(g["major"]))
            for g in groups]


def _fastener_bodies(h: Dict[str, Any], groups: List[Dict[str, Any]], solids: List[Any]
                     ) -> tuple:
    """The solids of the filler of the filled hole group ``h`` that look like
    fasteners (``_FASTENER_VOL_RATIO``, ``_FASTENER_BORE_RATIO``)."""
    b = h.get("filler")
    if b is None:
        return ()
    out = []
    cyl = math.pi * b["radius"] ** 2 * max(b["hi"] - b["lo"], 0.0)
    for idx in sorted(x for x in b["bodies"] if x >= 0):
        try:
            if idx >= len(solids) or cyl <= 0.0:
                continue
            vol = abs(_signed_volume(solids[idx]))
            if vol > _FASTENER_VOL_RATIO * cyl:
                continue                     # a part with a spigot, not a bolt or pin
            bored = any(g["concave"] and g.get("full") and idx in g["bodies"]
                        and g["radius"] >= _FASTENER_BORE_RATIO * b["radius"] for g in groups)
            if bored:
                continue                     # a bushing or sleeve: sectioned
            out.append(int(idx))
        except Exception as exc:
            logger.debug("fastener check failed: %s", exc)
    return tuple(out)


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
    Its ``filler_bodies`` lists the filler solids that look like fasteners
    (``_fastener_bodies``: slender, no bore; QA round 2 issue 28), which
    ``brep_sections.section_brep(features=...)`` leaves unsectioned.
    ``bodies`` of every feature: the solid indices of its faces
    (``_solid_list`` order, -1 = outside any solid).

    Toroidal faces (edge rounds of circular edges, e.g. a rounded boss top
    or the fillet at a boss foot) come out as kind ``"round"`` (QA round 2
    issue 16, ``_round_features``): ``radius`` = the round radius (torus
    minor radius), ``major_radius`` = the torus major radius, ``length`` 0;
    sorted after the filled holes.
    """
    if shape is None:
        return []
    tfaces: List[Dict[str, Any]] = []
    try:
        if shape.IsNull():
            return []
        if _detect_rounds:
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Torus
            both = _faces_with_inversion(shape, types=(GeomAbs_Cylinder, GeomAbs_Torus))
            cyl_items, tor_items = [], []
            for item in both:
                try:
                    is_torus = BRepAdaptor_Surface(item[0]).GetType() == GeomAbs_Torus
                except Exception:
                    continue
                (tor_items if is_torus else cyl_items).append(item)
            cfaces = _cylinder_faces(shape, cyl_items)
            tfaces = _torus_faces(tor_items)
        else:
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
    solids = None
    for g in groups:
        mid = 0.5 * (g["lo"] + g["hi"])
        full = g["full"]
        fillers: tuple = ()
        if g["concave"]:
            kind = ("filled_hole" if g.get("filled") else "hole") if full else "fillet"
            if kind == "filled_hole":
                if solids is None:
                    try:
                        solids = _solid_list(shape)
                    except Exception as exc:
                        logger.debug("solid list failed: %s", exc)
                        solids = []
                fillers = _fastener_bodies(g, groups, solids)
        else:
            kind = "boss"
        feats.append(CylFeature(center=g["loc"] + mid * g["dir"], axis=g["dir"].copy(),
                                radius=float(g["radius"]), length=float(g["hi"] - g["lo"]),
                                kind=kind, full_circle=bool(full), n_faces=int(g["n_faces"]),
                                bodies=tuple(sorted(g["bodies"])), filler_bodies=fillers))
    if tfaces:
        feats.extend(_round_features(tfaces, radius_tol, axis_tol))
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


def _extent_along(shape: Any, nv: np.ndarray, bbox: Any = None) -> Any:
    """``(h_lo, h_hi, bbox_volume)``: the range of ``p . nv`` over the
    axis-aligned bounding box of ``shape`` (or of ``bbox = (lo, hi)`` when
    given, e.g. the census's exact bbox) and the box volume. The box comes
    from ``brep_load.exact_bbox_bounded`` (the exact ``AddOptimal_s`` box by
    branch and bound: 0.9 to 1.1 s for a full ``AddOptimal_s`` on NIST FTC
    10, a few faces integrated here; the plain ``Add_s`` box is enlarged by
    every tolerance and by the control points of curved faces). ``None``
    when no box can be had."""
    lo = hi = None
    if bbox is not None:
        try:
            lo = np.asarray(bbox[0], dtype=float).reshape(3)
            hi = np.asarray(bbox[1], dtype=float).reshape(3)
        except Exception:
            lo = hi = None
    if lo is None:
        try:
            from .brep_load import exact_bbox_bounded
            box = exact_bbox_bounded(shape)
            if box is None:
                return None
            lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        except Exception as exc:
            logger.debug("depth sign: bbox failed: %s", exc)
            return None
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    h = corners @ nv
    return float(h.min()), float(h.max()), float(np.prod(np.maximum(hi - lo, 0.0)))


def depth_sign_details(shape, n, bbox: Any = None, cavity: bool = True) -> Dict[str, Any]:
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
    shape gives zeros.

    Cavity term (QA round 2 issue 4): ``cavity_plus`` = sum over the faces
    facing ``+n`` of ``area x |cos| x (h_hi - centroid . n)``, the projected
    area times how deep the face lies below the bounding plane on the
    ``+n`` side; ``cavity_minus`` likewise against ``h_lo``. It approximates
    the open volume a viewer on that side looks into: a pocket floor
    (large area, deep) dominates, shallow counterbores, notches and logos
    with many small faces do not (the face count turned NIST FTC 10 to its
    back: 105.5 vs 24.5). ``bbox_volume`` is the volume of the axis-aligned
    box (``bbox = (lo, hi)`` passes a box already known, e.g. the census's
    exact bbox; otherwise ``BRepBndLib.AddOptimal_s``), ``h_lo`` / ``h_hi``
    its range along ``n``. ``choose_depth_sign_brep`` decides with them.
    ``proj_area_plus`` / ``proj_area_minus`` are the projected facing areas
    (area x |cos|) and ``mean_recess_plus`` / ``mean_recess_minus`` the
    area-weighted mean recess as a fraction of the bbox depth along ``n``
    (cavity / (projected area x depth); 0 without a box): about 0.2 to 0.3
    for a pocketed part, 0.5 and more for an open housing or tray (review
    of issue 4). ``cavity=False`` skips the box and the cavity terms (they
    stay 0): ``score_depth_sign`` needs neither (0.25 s on NIST FTC 10).

    Cost (measured): 0.22 to 0.42 s on a 663-face NIST
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
           "area_total": 0.0, "score_plus": 0.0, "score_minus": 0.0, "faces": 0,
           "cavity_plus": 0.0, "cavity_minus": 0.0, "bbox_volume": 0.0,
           "h_lo": None, "h_hi": None, "proj_area_plus": 0.0, "proj_area_minus": 0.0,
           "mean_recess_plus": 0.0, "mean_recess_minus": 0.0}
    if shape is None:
        return out
    try:
        if shape.IsNull():
            return out
        faces = _faces_with_inversion(shape, cylinders_only=False)
    except Exception as exc:
        logger.warning("depth sign scoring failed: %s", exc)
        return out
    ext = _extent_along(shape, nv, bbox) if (faces and cavity) else None
    if ext is not None:
        out["h_lo"], out["h_hi"], out["bbox_volume"] = ext
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
            area, h = 0.0, None
            if abs(c) > _FACING_COS:          # side walls need no area
                gp = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, gp)
                area = abs(float(gp.Mass()))
                try:
                    h = float(np.dot(_vec(gp.CentreOfMass()), nv))
                    if not math.isfinite(h):
                        h = None
                except Exception:
                    h = None
        except Exception as exc:
            logger.debug("depth sign: face skipped: %s", exc)
            continue
        out["faces"] += 1
        if c > _FACING_COS:
            out["faces_plus"] += 1
            out["area_plus"] += area
            out["proj_area_plus"] += area * abs(c)
            if h is not None and ext is not None:
                out["cavity_plus"] += area * abs(c) * max(ext[1] - h, 0.0)
        elif c < -_FACING_COS:
            out["faces_minus"] += 1
            out["area_minus"] += area
            out["proj_area_minus"] += area * abs(c)
            if h is not None and ext is not None:
                out["cavity_minus"] += area * abs(c) * max(h - ext[0], 0.0)
    if ext is not None and ext[1] - ext[0] > 0.0:
        depth = ext[1] - ext[0]
        for s in ("plus", "minus"):
            pa = out["proj_area_" + s]
            out["mean_recess_" + s] = out["cavity_" + s] / (pa * depth) if pa > 0.0 else 0.0
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
    featured side. Never raises on a null shape ((0.0, 0.0)). The cavity
    terms are not computed (``cavity=False``: no bbox)."""
    d = depth_sign_details(shape, n, cavity=False)
    return float(d["score_plus"]), float(d["score_minus"])


DEPTH_CAVITY_SHARE = 0.02    # cavity decides when a side's cavity >= 2 % of the bbox volume
# Review of issue 4: the cavity rule leaves housings and trays to the face
# rule. Area-weighted mean recess / bbox depth of the deeper side, measured
# in the run2 pre-flip orientation: NIST CTC 02 housing 0.499 (the reference
# view shows the exterior), FTC 06 0.572 and FTC 07 0.743 (the face rule
# turns them as well), FTC 10 pocket 0.313 (cavity rule needed: the face rule
# turns it to its back), TO-220 0.256, the 150 x 30 x 75 pocket block 0.222.
DEPTH_CAVITY_MAX_MEAN_RECESS = 0.4
DEPTH_FLIP_RATIO = 1.25      # turn only when the far side scores > 1.25 x the kept side
DEPTH_FLIP_MIN_DIFF = 2.0    # face rule: and at least 2 (about two faces) more


def choose_depth_sign_brep(shape, n, bbox: Any = None, ratio: float = DEPTH_FLIP_RATIO,
                           min_diff: float = DEPTH_FLIP_MIN_DIFF,
                           cavity_share: float = DEPTH_CAVITY_SHARE,
                           max_mean_recess: Optional[float] = DEPTH_CAVITY_MAX_MEAN_RECESS
                           ) -> Dict[str, Any]:
    """Keep or turn the part for a viewer on the ``+n`` side (QA round 2 issue 4).

    ``n`` points from the part toward the current viewer (file coordinates,
    as for ``score_depth_sign``). With ``depth_sign_details``:

    * cavity rule (``method == "brep_cavity"``): when the larger of
      ``cavity_plus`` / ``cavity_minus`` is at least ``cavity_share`` of
      ``bbox_volume``, turn when ``cavity_minus > ratio x cavity_plus``,
      keep when ``cavity_plus > ratio x cavity_minus``; not when the larger
      cavity's ``mean_recess`` (area-weighted mean recess / bbox depth) is
      above ``max_mean_recess`` (None: no limit): that is the interior of an
      open housing or tray, drawn from its exterior as the NIST CTC 02
      reference does; ``"housing"`` in ``reason``;
    * otherwise (no open cavity worth the name, a housing, or the two
      cavities within the margin) the face rule (``method == "brep_faces"``): turn when
      ``score_minus > ratio x score_plus`` and ``score_minus - score_plus
      >= min_diff`` (the round-1 rule of ``pipeline._depth_sign_brep``).

    Returns ``{"flip", "method", "score_kept", "score_flipped" (the scores of
    the deciding rule), "cavity_kept", "cavity_flipped", "faces_kept",
    "faces_flipped" (the face-rule scores), "bbox_volume", "mean_recess_kept",
    "mean_recess_flipped", "housing" (True when the recess limit sent a
    decisive cavity to the face rule), "reason"}``.
    Never raises on a null shape (no flip); a bad ``n`` raises ValueError
    like ``depth_sign_details``.
    """
    d = depth_sign_details(shape, n, bbox=bbox)
    cp, cm = float(d["cavity_plus"]), float(d["cavity_minus"])
    sp, sm = float(d["score_plus"]), float(d["score_minus"])
    vol = float(d["bbox_volume"] or 0.0)
    out = {"flip": False, "method": "brep_faces", "score_kept": sp, "score_flipped": sm,
           "cavity_kept": cp, "cavity_flipped": cm, "faces_kept": sp, "faces_flipped": sm,
           "bbox_volume": vol, "mean_recess_kept": float(d["mean_recess_plus"]),
           "mean_recess_flipped": float(d["mean_recess_minus"]), "housing": False,
           "reason": ""}
    share = max(cp, cm) / vol if vol > 0 else 0.0
    cav = (f"open cavity {cp:.4g} mm3 on the kept side, {cm:.4g} mm3 on the far side "
           f"({100.0 * share:.3g} % of the bbox volume)")
    decisive = vol > 0 and share >= float(cavity_share) and (cm > ratio * cp or cp > ratio * cm)
    if decisive and max_mean_recess is not None:
        rec = float(d["mean_recess_minus"] if cm > cp else d["mean_recess_plus"])
        if rec > float(max_mean_recess):
            decisive = False
            out["housing"] = True
            cav += (f"; mean recess {rec:.3g} of the depth > {float(max_mean_recess):.3g}: "
                    f"an open housing, drawn from its exterior by the face rule")
    if decisive:
        out.update(method="brep_cavity", score_kept=cp, score_flipped=cm,
                   flip=bool(cm > ratio * cp))
        out["reason"] = cav + ("; turned 180 deg about Z" if out["flip"] else "")
        return out
    flip = bool(sm > ratio * sp and sm - sp >= float(min_diff))
    out["flip"] = flip
    out["reason"] = (f"faces facing the front viewer: {sp:.4g} on the kept side, {sm:.4g} on "
                     f"the far side ({cav})" + ("; turned 180 deg about Z" if flip else ""))
    return out
