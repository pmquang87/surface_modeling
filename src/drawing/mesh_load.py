"""Checked STL loading, mesh census and the prismatic/organic classifier.

The mesh route of the drawing module starts here (spec 3.2 item 3, 3.5).
Measured facts this module relies on (mesh probe, trimesh 5.0.0,
REPORT-mesh.md):

* ``trimesh.load`` NEVER raises on bad STL input: a 0-byte file, random
  bytes, STEP text saved as ``.stl`` and a truncated binary STL (header count
  larger than the data) all come back as an empty Trimesh with
  ``force="mesh"``. ``load_stl_checked`` therefore refuses 0 bytes and STEP
  text itself and refuses a result with 0 faces.
* ``process=True`` (the default) merges vertices so ``face_adjacency`` works
  (box: 36 -> 8 vertices, 18 adjacency pairs); it does NOT drop degenerate or
  duplicate faces, those are counted with ``nondegenerate_faces`` and
  ``unique_faces``.
* Components (bodies) on the vertex graph: ``face_adjacency`` excludes
  non-manifold edge pairs, so one body touching itself there split into
  several (QA round 1: bracket 3 -> 1). The face-adjacency count is kept as
  ``manifold_components`` (the shells the volume is summed over).
* An inside-out closed mesh has negative volume (the repo's genus-28 lattice
  fixture: -15356.87), so the census reports the absolute volume and flags
  ``inverted``. The volume is computed whenever there is no boundary edge
  (non-manifold edges allowed), with winding conflicts repaired (QA:
  multibody 0.01404 raw -> 0.007423 repaired, the faceted B-rep's value).
  It is summed over origin-independent pieces: a shell that is open because
  a non-manifold edge cut it (two cubes sharing a face: 4 open shells) has
  an origin-dependent signed volume, so the open shells of one body are
  summed together before any flip or void decision (review F4: the
  face-sharing cubes gave 2000 / 7333 / 6000 at x shifts 0 / +100 / -100).
* Classifier (14 meshes): ``planar_big`` = area of the coplanar facets that
  are each larger than 1 % of the total area, divided by the total area. CAD
  meshes gave >= 0.20 (worst: a 128-segment cylinder), organic meshes
  (spheres, torus, smoothed blob, genus-28 lattice) 0.000; threshold 0.10.
  Raw ``facets_area.sum() / area`` is useless (1.0 for any quad strip).
  ``sharp_turn`` = share of sum(edge length x dihedral angle) carried by edges
  above 30 deg; reported, not gating (the MC lattice has 0.625).

No OCP import: trimesh + numpy only. No output to stdout (logging only).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

log = logging.getLogger(__name__)

PLANAR_BIG_FACET_SHARE = 0.01     # a facet counts when larger than 1 % of the total area
SHARP_TURN_DEG = 30.0


class MeshLoadError(ValueError):
    """The file cannot be read as a non-empty triangle mesh."""


@dataclass
class MeshModel:
    mesh: "object"                   # trimesh.Trimesh
    census: dict
    classifier: dict
    warnings: List[str] = field(default_factory=list)


def load_stl_checked(path: str):
    """Load an STL as a processed ``trimesh.Trimesh`` or raise ``ValueError``.

    Refuses a 0-byte file, a file holding STEP text and any file from which
    trimesh reads no triangle (garbage, truncated binary). ``process=True``
    merges vertices so the face adjacency used by the views works.
    """
    import trimesh

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise MeshLoadError(f"{path}: cannot stat file ({exc})") from exc
    if size == 0:
        raise MeshLoadError(f"{path}: empty file")
    with open(path, "rb") as fh:
        head = fh.read(512)
    if head.lstrip().upper().startswith(b"ISO-10303-21"):
        raise MeshLoadError(f"{path}: file holds STEP text, not STL")
    try:
        mesh = trimesh.load(path, force="mesh", file_type="stl", process=True)
    except Exception as exc:  # trimesh raises assorted types on odd input
        raise MeshLoadError(f"{path}: STL could not be parsed ({type(exc).__name__}: {exc})") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshLoadError(f"{path}: no triangles could be read (corrupt or truncated STL)")
    return mesh


def _labels_from_pairs(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component label per node 0..n-1 for the undirected pairs (a, b)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return np.asarray(labels, dtype=np.int64)


def mesh_topology(mesh) -> dict:
    """Edge-level topology of a triangle mesh (numpy arrays, not JSON-ready).

    Every face edge is keyed by its sorted vertex pair. Returns:

    * ``body_label`` per face: components of the vertex graph (faces joined
      through every shared edge, non-manifold ones included, and through
      shared vertices), so a body touching itself along a non-manifold edge
      or at a non-manifold vertex stays one body. Measured on the QA corpus:
      topology-optimised bracket 1 body (face adjacency: 3), broken wheel
      tessellation 95 (face adjacency: 297, edge-only joining: 287).
    * ``shell_label`` per face: components over the manifold face adjacency
      only (``trimesh.graph.connected_components(face_adjacency)``, the old
      ``components`` count); these are the orientable shells used for the
      volume.
    * ``edge_count`` per unique edge, ``boundary_face`` (face of every
      boundary edge), ``conflict_face`` (one face per manifold edge whose two
      faces run it in the same direction: a winding conflict).
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = int(len(faces))
    nv = max(int(len(mesh.vertices)), 1)
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    face_of = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    lo = directed.min(axis=1)
    hi = directed.max(axis=1)
    key = lo * nv + hi
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    same = inv_s[1:] == inv_s[:-1]
    vlab = _labels_from_pairs(nv, uniq // nv, uniq % nv)
    _, body = np.unique(vlab[faces[:, 0]], return_inverse=True)
    body = np.asarray(body, dtype=np.int64).ravel()

    # manifold edges: exactly two faces; a conflict when both run a -> b
    two = counts[inv_s] == 2
    first = two & np.r_[True, ~same]                  # first entry of each 2-group
    idx1 = order[first]
    idx2 = order[np.flatnonzero(first) + 1]
    conflict = directed[idx1, 0] == directed[idx2, 0]
    a1, a2 = face_of[idx1], face_of[idx2]
    shell = _labels_from_pairs(n_faces, a1, a2)
    boundary_face = face_of[order[counts[inv_s] == 1]]
    return {"body_label": body, "shell_label": shell, "edge_count": counts,
            "boundary_face": boundary_face, "conflict_face": a1[conflict],
            "n_edges_unique": int(len(uniq)),
            # per directed face edge (3 per face, in face order): unique edge id
            # and +1 when it runs from the lower to the higher vertex index
            "edge_of_directed": inv,
            "edge_dir_sign": np.where(directed[:, 0] < directed[:, 1], 1, -1).astype(np.int8),
            # the manifold adjacency pairs and whether they are a winding conflict
            "pair_faces": np.stack([a1, a2], axis=1) if len(a1) else np.zeros((0, 2), np.int64),
            "pair_conflict": np.asarray(conflict, dtype=bool)}


def _face_signed_volumes(mesh) -> np.ndarray:
    tri = np.asarray(mesh.triangles, dtype=float)
    return np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0


_TIE_REL = 1e-9                   # |V| within this of the largest counts as a tie (sign: outward)
_VOID_GWN_PAIRS = 50_000_000      # point x triangle budget of the void test (~2-3 s)


def _orient_shells(n_faces: int, shell: np.ndarray, pair_faces: np.ndarray,
                   pair_conflict: np.ndarray, area: np.ndarray):
    """Per-face flip that winds every shell consistently, keeping its majority.

    Two-colouring of a signed graph (node ``(face, parity)``, a consistent
    manifold pair joins equal parities, a winding conflict opposite ones),
    vectorised with scipy instead of one ``trimesh.repair.fix_winding`` per
    shell. The orientation kept is the one most of the shell's ORIGINAL
    faces had (ties: the larger area). A non-orientable shell (both parities
    of a face connected) is left as it is. Returns ``(flip, nonorientable
    shell ids)``.
    """
    flip = np.zeros(n_faces, dtype=bool)
    if len(pair_faces) == 0 or not np.any(pair_conflict):
        return flip, np.zeros(0, dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = int(n_faces)
    a = pair_faces[:, 0].astype(np.int64)
    b = pair_faces[:, 1].astype(np.int64)
    c = np.asarray(pair_conflict, dtype=bool)
    src = np.concatenate([a, a + n])
    dst = np.concatenate([np.where(c, b + n, b), np.where(c, b, b + n)])
    g = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(2 * n, 2 * n))
    _, lab = connected_components(g, directed=False)
    lab = np.asarray(lab, dtype=np.int64)
    n_shell = int(shell.max()) + 1
    _, root = np.unique(shell, return_index=True)          # first face of each shell
    rel = lab[:n] != lab[root[shell]]                       # wound against the root face
    bad = np.unique(shell[lab[:n] == lab[n:]])
    if len(bad):
        rel[np.isin(shell, bad)] = False
    nf = np.bincount(shell, minlength=n_shell)
    nflip = np.bincount(shell, weights=rel.astype(float), minlength=n_shell)
    a_flip = np.bincount(shell, weights=area * rel, minlength=n_shell)
    a_all = np.bincount(shell, weights=area, minlength=n_shell)
    invert = (2.0 * nflip > nf) | ((2.0 * nflip == nf) & (2.0 * a_flip > a_all))
    return rel ^ invert[shell], bad


def _groups_balanced(edge_of_directed: np.ndarray, sign_dir: np.ndarray,
                     group_of_directed: np.ndarray, n_groups: int) -> np.ndarray:
    """Per group: True when every edge's directed uses inside the group cancel.

    A balanced (closed, consistently wound) set of faces has a signed volume
    that does not depend on the origin; an unbalanced one does.
    """
    key = edge_of_directed.astype(np.int64) * int(n_groups) + group_of_directed.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    net = np.bincount(np.asarray(inv).ravel(), weights=sign_dir.astype(float), minlength=len(uk))
    ok = np.ones(int(n_groups), dtype=bool)
    ok[np.unique(uk[np.abs(net) > 0.5] % int(n_groups))] = False
    return ok


def _volume_pieces(mesh, topo: dict) -> dict:
    """Signed volumes of the origin-independent pieces of a mesh with no boundary edge.

    Shells (manifold face-adjacency components) are first wound consistently
    (``_orient_shells``). A shell that is closed on its own (balanced) is a
    piece. A non-manifold edge splits a closed body into OPEN shells whose
    signed volumes depend on the origin (two cubes sharing a face: 4 open
    shells); the open shells of one body form one piece, since only their
    sum is closed. Each piece's volume is taken about its own area-weighted
    centroid: exact for a balanced piece, and still translation invariant
    for an unbalanced one (``balanced`` False, the value is approximate).

    Returns ``{"vols", "piece" (per face), "balanced", "lo", "hi",
    "flip" (per face), "nonorientable", "tri" (repaired corners),
    "order", "starts"}`` (faces sorted by piece: ``order[starts[k]:starts[k+1]]``).
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=float)
    n = int(len(faces))
    shell = np.asarray(topo["shell_label"], dtype=np.int64)
    body = np.asarray(topo["body_label"], dtype=np.int64)
    n_shell = int(shell.max()) + 1
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    flip, nonori = _orient_shells(n, shell, topo["pair_faces"], topo["pair_conflict"], area)
    fsign = np.where(flip, -1, 1).astype(np.int64)
    # Only the uses of non-manifold edges can leave an orientable shell
    # unbalanced: a 2-face edge joins two faces of the same shell, and they
    # are consistent after the rewind (no boundary edge exists here).
    eod_all = np.asarray(topo["edge_of_directed"], dtype=np.int64)
    nm = np.asarray(topo["edge_count"])[eod_all] > 2
    eod = eod_all[nm]
    face_of = np.repeat(np.arange(n, dtype=np.int64), 3)[nm]
    sdir = np.asarray(topo["edge_dir_sign"], dtype=np.int64)[nm] * fsign[face_of]
    shell_closed = _groups_balanced(eod, sdir, shell[face_of], n_shell)
    if len(nonori):
        shell_closed[nonori] = False
    if np.all(shell_closed):
        piece, n_piece = shell, n_shell
        balanced = np.ones(n_piece, dtype=bool)
    else:
        raw_group = np.where(shell_closed[shell], shell, n_shell + body)
        _, piece = np.unique(raw_group, return_inverse=True)
        piece = np.asarray(piece, dtype=np.int64).ravel()
        n_piece = int(piece.max()) + 1
        balanced = _groups_balanced(eod, sdir, piece[face_of], n_piece)
        if len(nonori):
            balanced[np.unique(piece[np.isin(shell, nonori)])] = False

    ctr = tri.mean(axis=1)
    a_sum = np.bincount(piece, weights=area, minlength=n_piece)
    cnt = np.bincount(piece, minlength=n_piece).astype(float)
    cen = np.empty((n_piece, 3))
    for d in range(3):
        wsum = np.bincount(piece, weights=area * ctr[:, d], minlength=n_piece)
        msum = np.bincount(piece, weights=ctr[:, d], minlength=n_piece)
        cen[:, d] = np.where(a_sum > 0.0, wsum / np.where(a_sum > 0.0, a_sum, 1.0),
                             msum / np.maximum(cnt, 1.0))
    rel = tri - cen[piece][:, None, :]
    fv = np.einsum("ij,ij->i", rel[:, 0], np.cross(rel[:, 1], rel[:, 2])) / 6.0 * fsign
    vols = np.bincount(piece, weights=fv, minlength=n_piece)

    order = np.argsort(piece, kind="stable")
    ps = piece[order]
    starts = np.r_[np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]]), n]
    lo = np.minimum.reduceat(tri.min(axis=1)[order], starts[:-1], axis=0)
    hi = np.maximum.reduceat(tri.max(axis=1)[order], starts[:-1], axis=0)
    tri_r = tri.copy()
    tri_r[flip] = tri_r[flip][:, [0, 2, 1]]                 # repaired winding
    return {"vols": vols, "piece": piece, "balanced": balanced, "lo": lo, "hi": hi,
            "flip": flip, "nonorientable": nonori, "tri": tri_r, "area": area,
            "order": order, "starts": starts}


def _winding_numbers(tri: np.ndarray, pts: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """Generalized winding number of the triangles ``tri`` (T, 3, 3) at ``pts`` (P, 3).

    Van Oosterom & Strackee solid angles (as ``src.operations.shell_thicken``):
    +1 inside an outward-wound closed surface, 0 outside, -1 inside an
    inside-out one.
    """
    w = np.zeros(len(pts))
    if len(tri) == 0:
        return w
    for i, p in enumerate(np.asarray(pts, dtype=float)):
        acc = 0.0
        for s in range(0, len(tri), chunk):
            k = tri[s:s + chunk] - p
            a, b, c = k[:, 0], k[:, 1], k[:, 2]
            la, lb, lc = (np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1),
                          np.linalg.norm(c, axis=1))
            det = np.einsum("ij,ij->i", a, np.cross(b, c))
            den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
                   + np.einsum("ij,ij->i", a, c) * lb + np.einsum("ij,ij->i", b, c) * la)
            acc += float(np.arctan2(det, den).sum())
        w[i] = acc / (2.0 * np.pi)
    return w


def _combine_piece_volumes(pc: dict):
    """Volume of a mesh with no boundary edge from its signed piece volumes.

    The dominant piece (largest |V|; balanced pieces preferred; an exact tie
    counts as outward) fixes the intended orientation. A piece of the
    opposite sign is a void when a point of it (centroid of its largest
    face) lies inside a piece of the dominant sign (generalized winding
    number > 0.5, tested only against pieces whose bbox contains its bbox):
    it is subtracted. Otherwise it is an independently inverted body and
    counts with its absolute value. Past ``_VOID_GWN_PAIRS`` point x
    triangle pairs the bbox containment alone decides. Returns
    ``(volume_abs, inverted, n_flipped)``.
    """
    vols = np.asarray(pc["vols"], dtype=float)
    if len(vols) == 0:
        return 0.0, False, 0
    absv = np.abs(vols)
    cand = np.flatnonzero(pc["balanced"]) if np.any(pc["balanced"]) else np.arange(len(vols))
    amax = float(absv[cand].max())
    if not amax > 0.0:
        return float(abs(vols.sum())), False, 0
    near = cand[absv[cand] >= amax * (1.0 - _TIE_REL)]
    sign = 1.0 if np.any(vols[near] > 0.0) else -1.0
    v = vols * sign                                       # dominant orientation positive
    eps = 1e-12 * float(absv.max())
    neg = np.flatnonzero(v < -eps)
    if len(neg) == 0:
        return float(abs(v.sum())), sign < 0.0, 0
    pos = np.flatnonzero(v > eps)
    lo, hi = pc["lo"], pc["hi"]
    tol = 1e-9 * max(float(np.linalg.norm(hi.max(axis=0) - lo.min(axis=0))), 1e-12)
    void = np.zeros(len(neg), dtype=bool)
    if len(pos):
        tri, area, order, starts = pc["tri"], pc["area"], pc["order"], pc["starts"]
        step = max(1, 4_000_000 // max(len(pos), 1))
        budget = _VOID_GWN_PAIRS
        lo_p, hi_p = lo[pos], hi[pos]
        for c0 in range(0, len(neg), step):
            chunk = neg[c0:c0 + step]
            box = (np.all(lo_p[None, :, :] <= lo[chunk][:, None, :] + tol, axis=2)
                   & np.all(hi_p[None, :, :] >= hi[chunk][:, None, :] - tol, axis=2))
            for i in np.flatnonzero(box.any(axis=1)):
                s = int(chunk[i])
                fs = order[starts[s]:starts[s + 1]]
                pt = tri[fs[int(np.argmax(area[fs]))]].mean(axis=0)
                inside = False
                for p in pos[box[i]]:
                    fp = order[starts[p]:starts[p + 1]]
                    if budget < len(fp):                   # out of budget: bbox decides
                        inside = True
                        break
                    budget -= len(fp)
                    if float(_winding_numbers(tri[fp], pt[None, :])[0]) * sign > 0.5:
                        inside = True
                        break
                void[c0 + i] = inside
        if budget <= 0:
            log.debug("void test: point-in-solid budget spent, bbox containment decided the rest")
    keep = v >= -eps
    total = float(v[keep].sum() + v[neg[void]].sum() - v[neg[~void]].sum())
    return float(abs(total)), sign < 0.0, int((~void).sum())


def mesh_census(mesh) -> dict:
    """Quality-report census of a triangle mesh (JSON-ready values).

    Keys: triangles vertices watertight winding_consistent components
    (bodies of the vertex graph: joined across every shared edge and vertex,
    non-manifold ones included) manifold_components (shells over the
    manifold face adjacency, the pre-QA ``components`` count)
    closed_components open_components volume_mm3 inverted volume_repaired
    volume_approximate volume_pieces volume_unbalanced_pieces volume_raw_mm3
    volume_source area_mm2 bbox_min bbox_max extents euler degenerate_faces
    duplicate_faces boundary_edges nonmanifold_edges winding_conflicts.

    Volume: computed whenever the mesh has no boundary edge, even when
    trimesh calls it not watertight because of non-manifold edges
    (``volume_approximate`` True then). Shells with winding conflicts are
    rewound to their majority orientation; the volume is then summed over
    origin-independent pieces (``_volume_pieces``: a shell closed on its
    own, or all open shells of one body together, as left by a non-manifold
    edge), flipping pieces that are inverted on their own and subtracting
    nested voids (point-in-solid test). It does not change when the mesh is
    translated. ``volume_repaired`` says whether a rewind or a flip
    happened; ``volume_raw_mm3`` keeps ``abs`` of the raw signed face sum
    (= ``abs(mesh.volume)``). ``volume_unbalanced_pieces``: pieces that are
    still not closed after the rewind (their volume is approximate).
    ``inverted``: the dominant piece is inside out. None when the mesh has
    boundary edges.
    """
    n_faces = int(len(mesh.faces))
    out: dict = {"triangles": n_faces, "vertices": int(len(mesh.vertices))}
    if n_faces == 0:
        out.update(watertight=False, winding_consistent=False, components=0,
                   manifold_components=0, closed_components=0, open_components=0,
                   volume_mm3=None, inverted=False, volume_repaired=False,
                   volume_approximate=False, volume_pieces=0, volume_unbalanced_pieces=0,
                   volume_raw_mm3=None, volume_source=None,
                   area_mm2=0.0, bbox_min=None, bbox_max=None, extents=None,
                   euler=0, degenerate_faces=0, duplicate_faces=0, boundary_edges=0,
                   nonmanifold_edges=0, winding_conflicts=0)
        return out
    watertight = bool(mesh.is_watertight)
    out["watertight"] = watertight
    out["winding_consistent"] = bool(mesh.is_winding_consistent)
    topo = mesh_topology(mesh)
    body, shell = topo["body_label"], topo["shell_label"]
    n_body = int(body.max()) + 1
    out["components"] = n_body
    out["manifold_components"] = int(shell.max()) + 1
    open_bodies = np.unique(body[topo["boundary_face"]]) if len(topo["boundary_face"]) else []
    out["open_components"] = int(len(open_bodies))
    out["closed_components"] = int(n_body - len(open_bodies))
    counts = topo["edge_count"]
    n_boundary = int((counts == 1).sum())
    out["boundary_edges"] = n_boundary
    out["nonmanifold_edges"] = int((counts > 2).sum())
    out["winding_conflicts"] = int(len(topo["conflict_face"]))

    vol_abs, inverted, repaired, raw = None, False, False, None
    n_pieces, n_unbalanced = 0, 0
    if n_boundary == 0:
        # raw from the face sum, not mesh.volume: trimesh's mass properties
        # warn on stderr (divide by zero) for a zero-volume closed mesh
        raw_signed = float(_face_signed_volumes(mesh).sum())
        raw = abs(raw_signed)
        try:
            pc = _volume_pieces(mesh, topo)
            vol_abs, inverted, n_flipped = _combine_piece_volumes(pc)
            repaired = bool(np.any(pc["flip"])) or n_flipped > 0
            n_pieces = int(len(pc["vols"]))
            n_unbalanced = int((~pc["balanced"]).sum())
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("piece volume failed: %s: %s", type(exc).__name__, exc)
            vol_abs, inverted = raw, raw_signed < 0.0
    out["volume_mm3"] = vol_abs
    out["inverted"] = bool(inverted)
    out["volume_repaired"] = bool(repaired)
    out["volume_approximate"] = bool(vol_abs is not None and (not watertight or n_unbalanced > 0))
    out["volume_pieces"] = n_pieces
    out["volume_unbalanced_pieces"] = n_unbalanced
    out["volume_raw_mm3"] = raw
    out["volume_source"] = (None if vol_abs is None else
                            ("winding_repaired" if repaired else "signed"))
    out["area_mm2"] = float(mesh.area)
    bounds = np.asarray(mesh.bounds, dtype=float)
    out["bbox_min"] = [float(v) for v in bounds[0]]
    out["bbox_max"] = [float(v) for v in bounds[1]]
    out["extents"] = [float(v) for v in bounds[1] - bounds[0]]
    out["euler"] = int(mesh.euler_number)
    out["degenerate_faces"] = int((~mesh.nondegenerate_faces(height=1e-8)).sum())
    out["duplicate_faces"] = int(n_faces - int(np.asarray(mesh.unique_faces()).sum()))
    return out


def mesh_volume_repaired(mesh) -> "tuple":
    """``(volume_abs or None, repaired: bool)`` of a mesh, as the census computes it.

    The value the faceted-route audit should compare the drawn B-rep volume
    against (issue 12): piece volumes after winding repair and after
    flipping independently inverted pieces, voids subtracted; translation
    invariant. None when the mesh has boundary edges.
    """
    c = mesh_census(mesh)
    return c["volume_mm3"], bool(c["volume_repaired"])


def watertight_detail(census: dict) -> str:
    """``"N boundary edge(s), M non-manifold edge(s)"`` for flags and warnings."""
    return (f"{int(census.get('boundary_edges') or 0)} boundary edge(s), "
            f"{int(census.get('nonmanifold_edges') or 0)} non-manifold edge(s)")


def not_solid_reason(census: dict):
    """Why a mesh is no solid at all (triangle soup, open sheets), or None.

    ``not_solid`` when the mesh has more than one body and none of them is
    closed, or when it has at least as many boundary edges as triangles (a
    closed triangle mesh has none; a soup of separate triangles has three
    per triangle). A single open body (a scan with a few holes) is NOT
    ``not_solid``: it still has a usable outline and section.
    """
    tri = int(census.get("triangles") or 0)
    if tri == 0:
        return "no triangles"
    comps = int(census.get("components") or 0)
    closed = int(census.get("closed_components") or 0)
    be = int(census.get("boundary_edges") or 0)
    if comps > 1 and closed == 0:
        return f"{comps} disconnected open components, 0 closed"
    if be >= tri:
        return f"{be} boundary edges for {tri} triangles"
    return None


def classify(mesh, threshold: float = 0.10, census: "dict | None" = None) -> dict:
    """Prismatic/organic/not_solid verdict (see module docstring).

    Returns ``{"verdict", "surface_verdict", "planar_big", "sharp_turn",
    "threshold", "not_solid_reason"}``. ``surface_verdict`` is
    ``"prismatic"`` when ``planar_big >= threshold`` else ``"organic"``;
    ``verdict`` equals it unless ``not_solid_reason(census)`` gives a
    reason, then it is ``"not_solid"`` (``census`` is computed when not
    passed, which doubles the cost: pass the census you already have, as
    ``load_mesh_model`` does). Callers that branch on ``verdict ==
    "organic"`` must treat ``"not_solid"`` as at least as restrictive (mesh
    route, no section, no conversion).
    """
    threshold = float(threshold)
    if census is None:
        try:
            census = mesh_census(mesh)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("classify: census failed: %s", exc)
            census = {}
    reason = not_solid_reason(census) if census else None
    out = _classify_surface(mesh, threshold)
    out["surface_verdict"] = out["verdict"]
    out["not_solid_reason"] = reason
    if reason is not None:
        out["verdict"] = "not_solid"
    return out


def _classify_surface(mesh, threshold: float) -> dict:
    area = float(mesh.area) if len(mesh.faces) else 0.0
    if not area > 0.0:
        return {"verdict": "organic", "planar_big": 0.0, "sharp_turn": 0.0, "threshold": threshold}
    fa = np.asarray(mesh.facets_area, dtype=float)
    planar_big = float(fa[fa > PLANAR_BIG_FACET_SHARE * area].sum() / area) if len(fa) else 0.0
    ang = np.degrees(np.asarray(mesh.face_adjacency_angles, dtype=float))
    sharp_turn = 0.0
    if len(ang):
        v = np.asarray(mesh.vertices, dtype=float)
        e = np.asarray(mesh.face_adjacency_edges)
        el = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
        turn = float((el * ang).sum())
        if turn > 0.0:
            sharp_turn = float((el * ang)[ang > SHARP_TURN_DEG].sum() / turn)
    verdict = "prismatic" if planar_big >= threshold else "organic"
    return {"verdict": verdict, "planar_big": planar_big, "sharp_turn": sharp_turn,
            "threshold": threshold}


# ------------------------------------------------------ unit plausibility --

UNIT_MIN_MM = 2.0                 # largest extent below this: suspicious
UNIT_MAX_MM = 5000.0              # largest extent above this: suspicious
_TYPICAL_PART_MM = 100.0          # alternatives are ranked by log distance to this size
_INCH_FRACTION = 1.0 / 16.0


def _near_multiple(x: float, step: float) -> bool:
    q = x / step
    return abs(q - round(q)) * step <= max(1e-4, 1e-6 * abs(x))


def unit_plausibility(extents_mm, declared_unit: "str | None" = None):
    """``(verdict, detail, alternative)`` for a part size read in mm (QA issue 5).

    ``verdict`` is ``"ok"`` or ``"suspicious"``. Suspicious when the largest
    extent is below ``UNIT_MIN_MM`` (2 mm) or above ``UNIT_MAX_MM``
    (5000 mm), or, for a file without a declared unit (STL), when every
    extent is a multiple of 1/16 and at least one is not a multiple of 1/4
    (inch fractions such as 1.375 = 1 3/8; whole or half millimetre sizes do
    not trigger) with a largest extent of at most 100. ``declared_unit``:
    the STEP unit ("inch"/"in" makes a > 5000 mm size suggest that the raw
    numbers were mm). ``alternative`` is None for "ok", else
    ``{"unit", "factor", "extents_mm"}``: the reading whose largest extent
    is closest (log scale) to a 100 mm part, ``factor`` multiplies the
    current mm values. ``detail`` is one human-readable sentence.

    Known gap: an inch-declared file whose numbers were really mm is only
    caught when the inch reading exceeds 5000 mm (raw value > 196.85). A
    150 mm part read as 3810 mm stays "ok"; the triage's other heuristics
    (raw values mostly integers, hole diameters that convert to standard
    metric sizes) need the B-rep and are not implemented here.
    """
    e = [abs(float(v)) for v in (extents_mm or [])]
    if not e or not max(e) > 0.0:
        return "suspicious", "the part has no extent", None
    mx = max(e)
    unit = (declared_unit or "").strip().lower()
    inch_declared = unit in ("in", "inch", "inches")
    reason, cands = None, []
    if mx < UNIT_MIN_MM:
        reason = f"largest extent {mx:.4g} mm is below {UNIT_MIN_MM:g} mm"
        cands = [("m", 1000.0), ("cm", 10.0), ("in", 25.4)]
    elif mx > UNIT_MAX_MM:
        reason = f"largest extent {mx:.6g} mm is above {UNIT_MAX_MM:g} mm"
        cands = [("mm", 1.0 / 25.4)] if inch_declared else [("um", 0.001), ("0.1 mm", 0.1)]
    elif not unit and mx <= 100.0 and all(_near_multiple(v, _INCH_FRACTION) for v in e) \
            and any(not _near_multiple(v, 0.25) for v in e):
        reason = ("every extent is an inch fraction (multiple of 1/16, not all of 1/4): "
                  + " x ".join(f"{v:g}" for v in e))
        cands = [("in", 25.4)]
    if reason is None:
        return "ok", f"largest extent {mx:.6g} mm", None
    import math
    best = min(cands, key=lambda c: abs(math.log10(mx * c[1] / _TYPICAL_PART_MM)))
    alt_ext = [v * best[1] for v in e]
    alt = {"unit": best[0], "factor": float(best[1]), "extents_mm": [float(v) for v in alt_ext]}
    detail = (f"{reason}; read as {best[0]} the part would be "
              + " x ".join(f"{v:.4g}" for v in alt_ext) + " mm")
    return "suspicious", detail, alt


# ------------------------------------------------- depth sign (QA issue 14) --

def score_depth_sign_mesh(mesh, n, feature_angle_deg: float = SHARP_TURN_DEG) -> float:
    """How much feature detail a viewer looking from direction ``n`` faces.

    ``n`` points from the part TOWARD the viewer, in the mesh's coordinates
    (``VIEWS[name].n`` mapped back: ``R.T @ VIEWS["front"].n`` for a mesh
    drawn with orientation matrix ``R``). Score = total length of the
    feature edges (dihedral angle above ``feature_angle_deg``) with at least
    one adjacent face whose normal points toward the viewer. Bosses, pocket
    rims and steps on the viewer's side count; the plain back face of a
    plate contributes only its outline.

    The area of the faces facing the viewer (the triage proposal) cannot
    decide this: for a closed mesh the projected areas facing +n and -n are
    equal (divergence theorem), and for a plate with prismatic bosses the
    unweighted facing areas are equal too (boss tops replace the footprints
    they cover). Measured on that fixture (100 x 10 x 60 plate, six 10 mm
    cube bosses): projected area +n == -n to 1e-12, flat facing area 6000
    == 6000, feature-edge score 800 (+n) vs 320 (-n).
    """
    v = np.asarray(n, dtype=float).reshape(3)
    v = v / max(float(np.linalg.norm(v)), 1e-300)
    if len(mesh.faces) == 0:
        return 0.0
    ang = np.degrees(np.asarray(mesh.face_adjacency_angles, dtype=float))
    if not len(ang):
        return 0.0
    sharp = ang > float(feature_angle_deg)
    if not sharp.any():
        return 0.0
    adj = np.asarray(mesh.face_adjacency)[sharp]
    fn = np.asarray(mesh.face_normals, dtype=float)
    facing = (fn[adj[:, 0]] @ v > 1e-6) | (fn[adj[:, 1]] @ v > 1e-6)
    verts = np.asarray(mesh.vertices, dtype=float)
    e = np.asarray(mesh.face_adjacency_edges)[sharp]
    length = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    return float(length[facing].sum())


def choose_depth_sign_mesh(mesh, matrix, margin: float = 1.2) -> dict:
    """Keep or flip the front view's depth sign for a mesh drawn with ``matrix``.

    ``matrix`` is the orientation (``v' = matrix @ v``) from
    ``projection.auto_orientation``; the front viewer stands at -Y of the
    drawing. Flipping = rotating 180 deg about the drawing Z axis
    (``diag(-1, -1, 1) @ matrix``, det stays +1), which moves the viewer to
    the other side. Flips only when the other side scores more than
    ``margin`` x the current side (symmetric parts keep the permutation).

    Returns ``{"flip", "matrix" (3x3 list to use), "score_kept",
    "score_flipped", "reason"}``.
    """
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    front = np.array([0.0, -1.0, 0.0])                    # projection.VIEWS["front"].n
    n_mesh = m.T @ front
    s_keep = score_depth_sign_mesh(mesh, n_mesh)
    s_flip = score_depth_sign_mesh(mesh, -n_mesh)
    flip = bool(s_flip > float(margin) * s_keep and s_flip > 0.0)
    out_m = (np.diag([-1.0, -1.0, 1.0]) @ m) if flip else m
    reason = (f"feature edges facing the front viewer: {s_keep:.6g} kept side, "
              f"{s_flip:.6g} other side" + ("; flipped (180 deg about Z)" if flip else ""))
    return {"flip": flip, "matrix": out_m.tolist(), "score_kept": s_keep,
            "score_flipped": s_flip, "reason": reason}


def transform_mesh(mesh, matrix):
    """Return a rotated copy: ``v' = matrix @ v`` (3x3, det +1 expected).

    The input mesh is not modified. A matrix with det -1 is applied too
    (trimesh then flips the face winding to keep normals outward) but logged.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"transform_mesh expects a 3x3 matrix, got shape {m.shape}")
    det = float(np.linalg.det(m))
    if abs(det - 1.0) > 1e-6:
        log.warning("transform_mesh: matrix is not a proper rotation (det %.6g)", det)
    t = np.eye(4)
    t[:3, :3] = m
    out = mesh.copy()
    out.apply_transform(t)
    return out


def load_mesh_model(path: str, threshold: float = 0.10) -> MeshModel:
    """``load_stl_checked`` + census + classifier + human-readable warnings."""
    mesh = load_stl_checked(path)
    census = mesh_census(mesh)
    classifier = classify(mesh, threshold, census=census)
    warnings: List[str] = []
    if not census["watertight"]:
        # covers the non-manifold count too: trimesh's is_watertight is False
        # for a non-manifold edge even with 0 boundary edges
        warnings.append(f"mesh is not watertight: {watertight_detail(census)}")
    if census["inverted"]:
        warnings.append("mesh is inside out (negative volume); the absolute volume is reported")
    if census["volume_repaired"]:
        warnings.append(f"volume computed on a winding-repaired copy ({census['volume_mm3']:.6g}; "
                        f"raw signed sum {census['volume_raw_mm3']:.6g})")
    if classifier.get("verdict") == "not_solid":
        warnings.append(f"mesh is not a solid: {classifier.get('not_solid_reason')}")
    if census["degenerate_faces"]:
        warnings.append(f"{census['degenerate_faces']} degenerate (zero-area) faces")
    if census["duplicate_faces"]:
        warnings.append(f"{census['duplicate_faces']} duplicate faces")
    for w in warnings:
        log.info("mesh %s: %s", os.path.basename(path), w)
    return MeshModel(mesh=mesh, census=census, classifier=classifier, warnings=warnings)


__all__ = ["MeshLoadError", "MeshModel", "load_stl_checked", "mesh_census", "classify",
           "transform_mesh", "load_mesh_model", "mesh_topology", "mesh_volume_repaired",
           "watertight_detail", "not_solid_reason", "unit_plausibility",
           "score_depth_sign_mesh", "choose_depth_sign_mesh"]
