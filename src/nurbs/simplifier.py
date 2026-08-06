import logging
from typing import Optional

try:
    from OCP.TopoDS import TopoDS_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.ShapeCustom import ShapeCustom
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    OCP_AVAILABLE = True
except ImportError:
    OCP_AVAILABLE = False
    TopoDS_Shape = None


class NURBSSimplifier:
    """Simplifies and reduces the size of B-Rep shapes (STEP data) without drastically altering geometry."""
    
    def __init__(self, linear_tolerance: float = 0.1, angular_tolerance: float = 0.1, max_degree: int = 3):
        self.linear_tol = linear_tolerance
        self.angular_tol = angular_tolerance
        self.max_degree = max_degree

    def simplify(self, shape: Optional['TopoDS_Shape']) -> Optional['TopoDS_Shape']:
        """Run simplification passes on the given shape."""
        if not OCP_AVAILABLE or shape is None:
            return shape

        logging.info("Starting NURBS Simplification...")
        simplified = shape

        # 1. Topological Merging: Unify same domain (merge adjacent coplanar/co-surface patches)
        try:
            logging.info("Applying ShapeUpgrade_UnifySameDomain...")
            unifier = ShapeUpgrade_UnifySameDomain(simplified, True, True, True)
            unifier.SetLinearTolerance(self.linear_tol)
            unifier.SetAngularTolerance(self.angular_tol)
            unifier.Build()
            simplified = unifier.Shape()
        except Exception as e:
            logging.warning(f"ShapeUpgrade_UnifySameDomain failed: {e}")

        # 2. Degree Reduction: BSpline restriction (DISABLED)
        # Applying rigorous math to drop polynomial degrees on 40,000 dense patches
        # simultaneously is too much for OCP's RAM limits and causes a silent segfault.
        # Keeping this disabled ensures stability on massive files.
        """
        try:
            logging.info(f"Applying BSplineRestriction (Max Degree: {self.max_degree})...")
            copied = BRepBuilderAPI_Copy(simplified).Shape()
            
            restricted = ShapeCustom.BSplineRestriction_s(
                copied, 
                self.linear_tol, 
                self.linear_tol * 0.1, 
                self.max_degree, 
                10000, 
                True,  # Degree3d
                False  # Degree2d
            )
            if restricted and not restricted.IsNull():
                simplified = restricted
        except Exception as e:
            logging.warning(f"BSplineRestriction failed: {e}")
        """

        logging.info("NURBS Simplification complete.")
        return simplified
