"""Topological simplification of B-Rep shapes, with a measured post-condition.

``ShapeUpgrade_UnifySameDomain`` merges adjacent faces that lie on the same
geometric surface. That is exactly what a tessellation-derived B-Rep needs --
the two triangles of a box side are one plane -- but the operator has no
notion of "wrong". It can drop a face, swallow a small fillet, or rebuild a
surface into something that no longer bounds the same volume, and it reports
none of that: ``Shape()`` returns a perfectly valid shape either way.

So the result is MEASURED and thrown away when the measurement moved. Pattern
from BlinkingSun/stl2step (MIT), ``unifySameOnce`` in
``src/refit_prism_build.cpp``: run the unifier once, compare the signed volume
and the surface-type census before/after, and keep the ORIGINAL shape unless
both agree.

Three settings differ from the previous version of this module and each has a
reason:

* ``ConcatBSplines = False`` -- concatenation re-parameterises and rebuilds
  B-spline surfaces. On a fitted mesh that is a second approximation stacked
  on the first, and it is the setting most likely to move the volume.
* ``SetSafeInputMode(True)`` -- the unifier otherwise modifies the input shape
  in place, which makes "keep the original" impossible: the original is
  already gone by the time the measurement says no.
* angular tolerance ``radians(0.001)`` instead of ``0.1`` -- the argument is in
  RADIANS, so 0.1 meant 5.7 degrees. Faces up to 5.7 degrees apart were merged
  into one, which flattens shallow chamfers and small-radius blends. Callers
  that want the loose threshold can still pass it explicitly.
"""
import logging
import math
from typing import Any, Dict, Optional

try:
    from OCP.TopoDS import TopoDS_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.Precision import Precision
    OCP_AVAILABLE = True
    _CONFUSION = float(Precision.Confusion_s())
except ImportError:
    OCP_AVAILABLE = False
    TopoDS_Shape = None
    ShapeUpgrade_UnifySameDomain = None
    _CONFUSION = 1e-7

logger = logging.getLogger(__name__)

# Relative volume budget for accepting a unified shape. stl2step uses the same
# 1e-6: large enough for the last bits of a double, far below any real merge
# error (swallowing one 0.5 mm chamfer off a 100 mm part is ~1e-4).
VOLUME_REL_BUDGET = 1e-6

# Default angular tolerance, in RADIANS (0.001 degrees). Named so no caller has
# to guess the unit from the number.
DEFAULT_ANGULAR_TOLERANCE_RAD = math.radians(0.001)

# Default linear tolerance: Precision::Confusion, 1e-7 mm.
DEFAULT_LINEAR_TOLERANCE = _CONFUSION


def _face_census(shape: Any) -> Dict[str, int]:
    """``{GeomAbs surface type name: face count}`` for every face of ``shape``.

    Uses ``BRepAdaptor_Surface.GetType()``, i.e. what the geometry actually is,
    not what the face was built from.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    census: Dict[str, int] = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        try:
            kind = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()), False).GetType()
            key = getattr(kind, "name", None) or str(kind)
        except Exception:  # a face whose surface cannot be adapted still counts
            key = "unknown"
        census[key] = census.get(key, 0) + 1
        exp.Next()
    return census


def _planar_split(census: Dict[str, int]):
    """(planar faces, non-planar faces, total) from a census dict."""
    total = sum(census.values())
    planar = census.get("GeomAbs_Plane", 0)
    return planar, total - planar, total


class NURBSSimplifier:
    """Simplifies B-Rep shapes without letting the simplification change them.

    ``simplify`` never returns a shape it could not verify: on any doubt it
    returns the object it was given. The evidence for the decision is left in
    ``self.last_report``.
    """

    def __init__(self, linear_tolerance: float = DEFAULT_LINEAR_TOLERANCE,
                 angular_tolerance: float = DEFAULT_ANGULAR_TOLERANCE_RAD,
                 max_degree: int = 3):
        """``angular_tolerance`` is in RADIANS (default 0.001 deg = 1.7e-5 rad).

        ``linear_tolerance`` is in model units, default ``Precision::Confusion``
        (1e-7 mm). Explicit arguments are honoured unchanged, so a caller that
        deliberately wants a loose 0.1 rad merge still gets it.
        """
        self.linear_tol = linear_tolerance
        self.angular_tol = angular_tolerance
        self.max_degree = max_degree
        self.last_report: Dict[str, Any] = {
            "ran": False, "accepted": False, "reason": "not run yet",
        }

    def simplify(self, shape: Optional['TopoDS_Shape']) -> Optional['TopoDS_Shape']:
        """Unify same-domain faces, but only if the shape did not change."""
        report: Dict[str, Any] = {
            "ran": False,
            "accepted": False,
            "reason": None,
            "linear_tolerance": self.linear_tol,
            "angular_tolerance_rad": self.angular_tol,
            "volume_before": None, "volume_after": None,
            "volume_rel_delta": None,
            "faces_before": None, "faces_after": None,
            "planar_before": None, "planar_after": None,
            "non_planar_before": None, "non_planar_after": None,
            "census_before": None, "census_after": None,
        }
        self.last_report = report

        if not OCP_AVAILABLE or shape is None:
            report["reason"] = "OCP unavailable" if shape is not None else "no shape"
            return shape
        try:
            if shape.IsNull():
                report["reason"] = "null shape"
                return shape
        except Exception:
            pass

        # ---- measure the input -------------------------------------------
        # The import sits INSIDE the guard on purpose. The contract of this
        # method is that it never raises and always leaves evidence in
        # ``last_report``; an unimportable ``step_audit`` (partial or circular
        # import, a broken numpy) would otherwise escape past every gate with
        # the report still empty.
        try:
            from src.io.step_audit import brep_volume
            v0 = brep_volume(shape)
            census0 = _face_census(shape)
        except Exception as exc:
            report["reason"] = f"input could not be measured: {exc}"
            logger.warning("Simplification skipped: %s", report["reason"])
            return shape

        p0, np0, n0 = _planar_split(census0)
        report.update(volume_before=v0, census_before=census0,
                      faces_before=n0, planar_before=p0, non_planar_before=np0)

        # ---- run the unifier ---------------------------------------------
        # UnifyEdges, UnifyFaces, ConcatBSplines=False + safe input mode, so
        # the shape handed in stays intact and can still be returned.
        logger.info("Applying ShapeUpgrade_UnifySameDomain (linear=%g, angular=%g rad)",
                    self.linear_tol, self.angular_tol)
        try:
            unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
            unifier.SetLinearTolerance(self.linear_tol)
            unifier.SetAngularTolerance(self.angular_tol)
            unifier.SetSafeInputMode(True)
            unifier.Build()
            unified = unifier.Shape()
            report["ran"] = True
        except Exception as exc:
            report["reason"] = f"ShapeUpgrade_UnifySameDomain failed: {exc}"
            logger.warning("%s - keeping the original shape", report["reason"])
            return shape

        if unified is None:
            report["reason"] = "the unifier returned no shape"
            logger.warning("%s - keeping the original shape", report["reason"])
            return shape
        try:
            if unified.IsNull():
                report["reason"] = "the unifier returned a null shape"
                logger.warning("%s - keeping the original shape", report["reason"])
                return shape
        except Exception:
            pass

        # ---- measure the result ------------------------------------------
        try:
            v1 = brep_volume(unified)
            census1 = _face_census(unified)
        except Exception as exc:
            report["reason"] = f"unified shape could not be measured: {exc}"
            logger.warning("%s - keeping the original shape", report["reason"])
            return shape

        p1, np1, n1 = _planar_split(census1)
        report.update(volume_after=v1, census_after=census1,
                      faces_after=n1, planar_after=p1, non_planar_after=np1)

        # ---- the three gates ---------------------------------------------
        # stl2step's floor: an absolute budget scaled by the original volume,
        # so a zero-volume input rejects any change at all.
        floor = VOLUME_REL_BUDGET * abs(v0)
        dv = abs(v1 - v0)
        report["volume_rel_delta"] = (dv / abs(v0)) if v0 else None

        reason = None
        if dv > floor:
            reason = (f"volume moved by {dv:.6g} "
                      f"({report['volume_rel_delta']:.3g} relative, "
                      f"budget {VOLUME_REL_BUDGET:g})"
                      if v0 else f"volume moved by {dv:.6g} from zero")
        elif np1 != np0:
            reason = (f"non-planar face count changed {np0} -> {np1} "
                      f"(census {census0} -> {census1})")
        elif n1 > n0:
            reason = f"face count increased {n0} -> {n1}"

        if reason is not None:
            report["reason"] = reason
            logger.warning(
                "UnifySameDomain result discarded: %s; volume %.12g -> %.12g, "
                "faces %d -> %d. Keeping the original shape.",
                reason, v0, v1, n0, n1)
            return shape

        report["accepted"] = True
        report["reason"] = "volume and surface census unchanged"
        logger.info("UnifySameDomain accepted: faces %d -> %d, volume %.12g -> %.12g",
                    n0, n1, v0, v1)
        return unified
