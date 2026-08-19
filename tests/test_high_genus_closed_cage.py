"""Regression tests: a closed watertight input must always give a CLOSED cage.

Pins the 2026-08-19 failure on the foxcore full-wing tool (marching-cubes
remesh of a topology-optimization void, genus 9, 461k triangles): the cage
came back with 42 boundary edges (open shell, free edges in the sewn STEP)
and only 1757 of the 4200 requested quads. Three root causes, each pinned by
a test here:

- ``collapse_short_edges`` re-derived its threshold from the CURRENT median
  every round; collapsing short edges raises the median, so the threshold
  ratcheted (0.26 -> 0.49 -> 0.78 -> 1.00 on the real part) and ate more than
  half of the decimated faces -- the quad-count shortfall. It also collapsed
  edges violating the link condition, pinching thin struts into non-manifold
  junk, boundary edges and loose scraps.
- ``flip_needle_triangles`` flipped an edge onto a vertex pair that was
  already connected elsewhere, turning that edge 4-incident (non-manifold).
- ``_repair_decimated``'s non-manifold loop ping-ponged with
  ``trimesh.repair.fill_holes`` (which only closes 3/4-edge holes and kept
  re-creating the same junk faces) and gave up after 3 rounds with
  non-manifold edges and inconsistent winding still in the mesh; every
  unpaired half-edge then surfaced as a cage boundary edge. Its
  keep-two-largest rule at over-shared edges could also keep two faces that
  cross the edge in the SAME direction -- an unresolvable pairing the
  directed hole filler "fixes" by adding a third face back, a repair cycle
  that never converges.

A watertight high-genus lattice (marching cubes, the same provenance as the
real parts) makes all of these fire at once.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper


def _edge_incidence(faces) -> dict:
    """Undirected edge -> number of incident faces, for a face index list."""
    inc = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            inc[key] = inc.get(key, 0) + 1
    return inc


def _jungle_gym() -> trimesh.Trimesh:
    """Closed genus-28 lattice: a 3x3x3 grid of thin fused square rods,
    surfaced by marching cubes, Taubin-smoothed and rotated off-axis -- the
    same provenance (smooth SDF/MC remesh of a strut lattice) as the real
    topology-optimization parts this failure was found on."""
    measure = pytest.importorskip("skimage.measure")
    n = 53
    idx = np.arange(n)
    x, y, z = idx[:, None, None], idx[None, :, None], idx[None, None, :]

    def rods(u, v):
        band = lambda w: np.any([np.abs(w - c) <= 2 for c in (12, 24, 36)],
                                axis=0)
        return band(u) & band(v)

    occ = rods(y, z) | rods(x, z) | rods(x, y)
    core = (idx >= 8) & (idx <= 40)
    occ &= core[:, None, None] & core[None, :, None] & core[None, None, :]
    vol = np.pad(occ, 2).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.merge_vertices()
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(0.41, [1, 0.7, 0.3]))

    # Preconditions: without these every closure assertion below is vacuous.
    assert mesh.is_watertight, "lattice construction failed: not watertight"
    assert mesh.is_winding_consistent
    assert len(mesh.split(only_watertight=False)) == 1
    genus = (2 - mesh.euler_number) // 2
    assert genus == 28, f"lattice construction failed: genus {genus}"
    return mesh


class TestCollapseShortEdgesManifold:
    def test_link_condition_violating_collapse_is_refused(self):
        """Bipyramid with a short equator edge: its endpoints share THREE
        common neighbours but the edge has only two apexes, so collapsing it
        folds two faces onto each other (duplicate triangle -> hole after
        dedup). The collapse must be refused, keeping the mesh closed."""
        from src.reverse_engineering.mesh_tools import collapse_short_edges
        verts = np.array([
            [0.0, 0.0, 0.0],    # a -+
            [0.05, 0.0, 0.0],   # b -+- the short edge
            [0.0, 1.0, 0.0],    # c
            [0.0, 0.4, 0.8],    # p apex
            [0.0, 0.4, -0.8],   # q apex
        ])
        faces = np.array([
            [0, 1, 3], [1, 2, 3], [2, 0, 3],   # upper fan around p
            [1, 0, 4], [0, 2, 4], [2, 1, 4],   # lower fan around q
        ])
        dirty = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        assert dirty.is_watertight and dirty.is_winding_consistent
        lens = np.linalg.norm(
            dirty.vertices[dirty.edges_unique[:, 0]] -
            dirty.vertices[dirty.edges_unique[:, 1]], axis=1)
        assert lens.min() < 0.15 * np.median(lens), "no short edge constructed"

        out = collapse_short_edges(dirty, rel_threshold=0.15)
        inc = _edge_incidence(np.asarray(out.faces))
        assert all(c == 2 for c in inc.values()), (
            "collapse broke the closed manifold: "
            f"incidences {sorted(set(inc.values()))}"
        )
        assert out.is_watertight and out.is_winding_consistent

    def test_threshold_does_not_ratchet_across_rounds(self):
        """The collapse threshold is a sliver criterion relative to the INPUT
        mesh scale. Re-deriving it per round from the shrinking mesh lets it
        grow every round (collapsing the short population raises the median)
        until it collapses real geometry -- measured 0.26 -> 1.00 on the
        foxcore part, eating half of all faces.

        Capsule with three edge scales: 30 densely packed rings (short axial
        edges, the sliver population), the ring cross-section (medium, the
        real geometry) and 40 widely spaced rings (long). Once the shorts are
        gone, the median jumps into the long population; a re-derived
        threshold (0.06 -> 0.45, measured) then eats every medium
        cross-section edge and crushes the tube, while a frozen threshold
        leaves the cross-sections alone."""
        from src.reverse_engineering.mesh_tools import collapse_short_edges
        K, R = 16, 1.0
        zs = ([i * 0.02 for i in range(30)] +
              [0.58 + i * 3.0 for i in range(1, 41)])
        ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
        verts = [[R * np.cos(a), R * np.sin(a), z] for z in zs for a in ang]
        bot = len(verts)
        verts.append([0.0, 0.0, zs[0] - 0.3])
        top = len(verts)
        verts.append([0.0, 0.0, zs[-1] + 0.3])
        faces = []
        for r in range(len(zs) - 1):
            for k in range(K):
                a, b = r * K + k, r * K + (k + 1) % K
                c, d = (r + 1) * K + (k + 1) % K, (r + 1) * K + k
                faces.extend([[a, b, c], [a, c, d]])
        base = (len(zs) - 1) * K
        for k in range(K):
            faces.append([bot, (k + 1) % K, k])
            faces.append([top, base + k, base + (k + 1) % K])
        dirty = trimesh.Trimesh(vertices=np.array(verts),
                                faces=np.array(faces), process=False)
        assert dirty.is_watertight and dirty.is_winding_consistent

        def edge_lens(m):
            return np.linalg.norm(m.vertices[m.edges_unique[:, 0]] -
                                  m.vertices[m.edges_unique[:, 1]], axis=1)

        lens = edge_lens(dirty)
        threshold0 = 0.15 * np.median(lens)
        assert (lens < threshold0).sum() >= 400, "no sliver population"
        mid_in = int(((lens > 0.3) & (lens < 0.5)).sum())
        assert mid_in >= 1000, "no medium population"

        out = collapse_short_edges(dirty, rel_threshold=0.15)
        assert len(out.faces) < len(dirty.faces), "no collapse happened"
        out_lens = edge_lens(out)
        mid_out = int(((out_lens > 0.3) & (out_lens < 0.5)).sum())
        # Frozen threshold: the dense band merges away but the tube
        # cross-sections survive (measured 896 of 1584). A ratcheting
        # threshold collapses every one of them (measured 0).
        assert mid_out >= 300, (
            f"threshold ratcheted: only {mid_out} of {mid_in} medium edges "
            f"survived"
        )
        assert out.is_watertight and out.is_winding_consistent


class TestFlipNeedleManifold:
    def test_flip_onto_existing_edge_is_refused(self):
        """Sliver tetrahedron: flipping the cap's longest edge would create
        the edge (apex, off-line vertex), which ALREADY exists -- the flip
        must be refused instead of making that edge 4-incident."""
        from src.reverse_engineering.mesh_tools import flip_needle_triangles
        verts = np.array([
            [0.0, 0.0, 1.0],    # a apex
            [-1.0, 0.0, 0.0],   # b
            [1.0, 0.0, 0.0],    # c
            [0.0, 0.02, 0.0],   # d: almost on segment b-c -> (b,d,c) is a cap
        ])
        faces = np.array([
            [1, 3, 2],   # base cap (b, d, c)
            [1, 2, 0],   # side (b, c, a)
            [2, 3, 0],   # side (c, d, a)
            [3, 1, 0],   # side (d, b, a)
        ])
        dirty = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        assert dirty.is_watertight and dirty.is_winding_consistent

        pts = dirty.vertices[dirty.faces]
        e = np.roll(pts, -1, axis=1) - pts
        areas = 0.5 * np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1)
        longest = np.linalg.norm(e, axis=2).max(axis=1)
        heights = 2.0 * areas / longest
        lens = np.linalg.norm(
            dirty.vertices[dirty.edges_unique[:, 0]] -
            dirty.vertices[dirty.edges_unique[:, 1]], axis=1)
        assert heights.min() < 0.05 * np.median(lens), "no needle constructed"

        out = flip_needle_triangles(dirty, rel_height=0.05)
        inc = _edge_incidence(np.asarray(out.faces))
        assert all(c == 2 for c in inc.values()), (
            "flip created a non-manifold edge: "
            f"incidences {sorted(set(inc.values()))}"
        )
        assert out.is_watertight and out.is_winding_consistent


class TestHighGenusClosedCage:
    TARGET = 700

    def _wrap(self):
        dense_tm = _jungle_gym()
        dense = HalfEdgeMesh.from_trimesh(dense_tm)
        cage = QuadWrapper(target_face_count=self.TARGET).wrap(dense)
        return dense_tm, cage

    def test_cage_is_closed_pure_quads_single_body(self):
        dense_tm, cage = self._wrap()
        assert len(cage.faces) > 0
        assert all(len(cage.get_face_vertices(f)) == 4 for f in cage.faces), (
            "cage is not pure quads"
        )
        n_boundary = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
        assert n_boundary == 0, (
            f"closed watertight input produced an OPEN cage: "
            f"{n_boundary} boundary edges"
        )
        # No edge may have more than two faces either (an unpaired third
        # half-edge also surfaces as a free edge in the sewn B-Rep).
        quad_faces = [[v.index for v in cage.get_face_vertices(f)]
                      for f in cage.faces]
        inc = _edge_incidence(quad_faces)
        assert all(c == 2 for c in inc.values()), (
            f"cage has non-manifold edges: incidences "
            f"{sorted(set(inc.values()))}"
        )
        comps = cage.to_trimesh().split(only_watertight=False)
        assert len(comps) == 1, f"cage fell apart into {len(comps)} bodies"

    def test_quad_count_reaches_target(self):
        """The sliver-collapse feedback loop ate half the decimation budget on
        lattice-like parts (1757 of 4200 requested). The wrap promises the
        request within roughly +-15%; allow 25% here for the adversarial
        input."""
        _, cage = self._wrap()
        n_quads = sum(
            1 for f in cage.faces if len(cage.get_face_vertices(f)) == 4)
        assert n_quads >= 0.75 * self.TARGET, (
            f"quad count collapsed: {n_quads} of {self.TARGET} requested"
        )
        assert n_quads <= 1.6 * self.TARGET, (
            f"quad count overshot: {n_quads} of {self.TARGET} requested"
        )

    def test_sewn_shells_are_closed(self):
        """End-to-end: the NURBS conversion of a high-genus cage must sew into
        closed shells (0 free edges) that promote to solids."""
        pytest.importorskip("OCP")
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_SOLID
        from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
        from src.nurbs.converter import SubDToNURBSConverter

        dense_tm, cage = self._wrap()
        n_boundary = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
        assert n_boundary == 0, "cage already open; sewn-shell test is moot"

        dense = HalfEdgeMesh.from_trimesh(dense_tm)
        result = SubDToNURBSConverter(continuity='G1', tolerance=1e-4).convert(
            cage, reference_mesh=dense)
        shape = result['shape']
        assert shape is not None

        def count(root, kind):
            n, exp = 0, TopExp_Explorer(root, kind)
            while exp.More():
                n += 1
                exp.Next()
            return n

        fb = ShapeAnalysis_FreeBounds(shape)
        free_edges = (count(fb.GetClosedWires(), TopAbs_EDGE) +
                      count(fb.GetOpenWires(), TopAbs_EDGE))
        assert free_edges == 0, f"sewn B-Rep has {free_edges} free edges"
        assert count(shape, TopAbs_SOLID) >= 1, (
            "closed cage did not promote to a solid"
        )
