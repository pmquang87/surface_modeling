import warnings
import numpy as np
import trimesh
from typing import Optional, List, Dict, Any, Tuple
from src.core.halfedge_mesh import HalfEdgeMesh


def _warn_if_not_triangles(mesh: HalfEdgeMesh, func_name: str) -> None:
    """Warn that a quad/ngon cage will come back triangulated.

    Every helper here round-trips through ``HalfEdgeMesh.to_trimesh()``, which
    fans any face with more than three corners into triangles. That is a silent
    loss of the quad cage, so say so.
    """
    n_ngon = sum(1 for f in mesh.faces if len(mesh.get_face_vertices(f)) != 3)
    if n_ngon:
        warnings.warn(
            f"{func_name}: input has {n_ngon} non-triangle face(s); the result "
            f"is triangulated (the quad/ngon cage is NOT preserved).",
            UserWarning,
            stacklevel=3,
        )


def _boundary_loops(faces: List[List[int]]) -> List[List[int]]:
    """Return the boundary loops of a triangle soup, wound for hole filling.

    A directed edge (a, b) that has no opposing (b, a) sits on a boundary. New
    faces closing that hole must contain (b, a), so each loop is returned in the
    reverse of the traversal direction implied by the existing faces: feeding it
    straight to a triangulator yields faces consistent with the surrounding
    winding.
    """
    directed = set()
    for f in faces:
        n = len(f)
        for i in range(n):
            directed.add((f[i], f[(i + 1) % n]))

    outgoing: Dict[int, List[int]] = {}
    for a, b in directed:
        if (b, a) not in directed:
            outgoing.setdefault(a, []).append(b)

    loops = []
    for start in list(outgoing.keys()):
        while outgoing.get(start):
            loop = [start]
            at = {start: 0}          # vertex -> position in `loop`, O(1) lookup
            cur = start
            while True:
                nxt_list = outgoing.get(cur)
                if not nxt_list:
                    loop = None
                    break
                nxt = nxt_list.pop()
                if not nxt_list:
                    outgoing.pop(cur, None)
                if nxt == start:
                    break
                if nxt in at:
                    # self-touching boundary: cut the sub-loop off and keep it
                    k = at[nxt]
                    loops.append(list(reversed(loop[k:])))
                    for v in loop[k:]:
                        at.pop(v, None)
                    loop = loop[:k]
                at[nxt] = len(loop)
                loop.append(nxt)
                cur = nxt
            if loop and len(loop) >= 3:
                loops.append(list(reversed(loop)))
    return [lp for lp in loops if len(lp) >= 3]


def _polygon_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newell normal plus an orthonormal in-plane basis (e1, e2).

    Projected on (e1, e2) the polygon is counter-clockwise, so an ear clipper
    can assume CCW input and the triangles it emits keep the polygon's winding.
    """
    q = np.roll(points, -1, axis=0)
    normal = np.cross(points, q).sum(axis=0)
    nl = np.linalg.norm(normal)
    if nl < 1e-14:
        # Fully degenerate (collinear) loop: fall back to any frame.
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / nl
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - normal * np.dot(ref, normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return normal, e1, e2


def _ear_clip(poly2d: np.ndarray,
              forbidden: Optional[set] = None) -> Optional[List[Tuple[int, int, int]]]:
    """Ear-clip a simple CCW polygon. Returns local index triples, or None.

    ``forbidden`` holds local index pairs (i, j), i < j, whose diagonal must not
    be created -- used for chords that already exist as edges of the surrounding
    mesh, since re-creating one would give that edge a third face.
    """
    n = len(poly2d)
    if n < 3:
        return []
    forbidden = forbidden or set()
    scale = float(np.ptp(poly2d, axis=0).max())
    if scale <= 0:
        return None
    eps = 1e-10 * scale * scale

    def in_tri(p, a, b, c):
        d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        return d1 >= -eps and d2 >= -eps and d3 >= -eps

    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > n * n + 16:
            return None
        found = False
        for k in range(len(idx)):
            i0 = idx[k - 1]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = poly2d[i0], poly2d[i1], poly2d[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= eps:
                continue  # reflex or degenerate corner
            if (min(i0, i2), max(i0, i2)) in forbidden:
                continue  # the diagonal already exists in the surrounding mesh
            if any(in_tri(poly2d[m], a, b, c) for m in idx if m not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:
            return None
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def fill_holes(mesh: HalfEdgeMesh, max_hole_edges: int = 20) -> HalfEdgeMesh:
    """Fill boundary holes with new triangles.

    Boundary loops are detected from the (triangulated) face set and each loop
    up to ``max_hole_edges`` edges long is closed by ear-clipping it in its
    best-fit plane; loops the ear clipper cannot handle (self-intersecting when
    projected) fall back to a fan from a new centroid vertex.

    Loops LONGER than ``max_hole_edges`` are deliberately left open and reported
    via a warning -- unlike ``trimesh.repair.fill_holes``, which silently closes
    only 3- and 4-edge holes and whose boolean result the old implementation
    threw away.

    Note: the mesh round-trips through ``to_trimesh()``, so a quad cage comes
    back triangulated (a warning says so).
    """
    _warn_if_not_triangles(mesh, "fill_holes")
    t_mesh = mesh.to_trimesh()
    if len(t_mesh.faces) == 0:
        return mesh.copy()

    try:
        verts = [np.asarray(p, dtype=np.float64) for p in np.asarray(t_mesh.vertices)]
        faces = [[int(i) for i in f] for f in np.asarray(t_mesh.faces)]

        # Chords already present in the mesh: an ear whose diagonal duplicates
        # one would put a third face on that edge.
        existing = set()
        for f in faces:
            m = len(f)
            for i in range(m):
                a, b = f[i], f[(i + 1) % m]
                existing.add((a, b) if a < b else (b, a))

        loops = _boundary_loops(faces)
        skipped = []
        for loop in loops:
            if len(loop) > max_hole_edges:
                skipped.append(len(loop))
                continue
            pts = np.array([verts[i] for i in loop], dtype=np.float64)
            _, e1, e2 = _polygon_frame(pts)
            poly2d = np.column_stack([pts @ e1, pts @ e2])
            n_loop = len(loop)
            forbidden = set()
            for i in range(n_loop):
                for j in range(i + 2, n_loop):
                    if i == 0 and j == n_loop - 1:
                        continue  # the closing edge of the loop itself
                    a, b = loop[i], loop[j]
                    if ((a, b) if a < b else (b, a)) in existing:
                        forbidden.add((i, j))
            tris = _ear_clip(poly2d, forbidden=forbidden)
            if tris is None:
                center = len(verts)
                verts.append(pts.mean(axis=0))
                n = len(loop)
                for i in range(n):
                    faces.append([loop[i], loop[(i + 1) % n], center])
            else:
                for a, b, c in tris:
                    tri = [loop[a], loop[b], loop[c]]
                    faces.append(tri)
                    # keep `existing` current so a later loop sharing these
                    # vertices cannot duplicate one of the new chords
                    for i in range(3):
                        p, q = tri[i], tri[(i + 1) % 3]
                        existing.add((p, q) if p < q else (q, p))

        if skipped:
            warnings.warn(
                f"fill_holes: {len(skipped)} hole(s) left open, boundary loops of "
                f"{sorted(skipped)} edges exceed max_hole_edges={max_hole_edges}.",
                UserWarning,
                stacklevel=2,
            )

        return HalfEdgeMesh.from_arrays(np.array(verts, dtype=np.float64), faces)
    except Exception as e:
        print(f"Error filling holes: {e}")
        return mesh.copy()

def smooth_mesh(mesh: HalfEdgeMesh, iterations: int = 3, 
                method: str = 'taubin', lambda_factor: float = 0.5,
                mu_factor: float = -0.53) -> HalfEdgeMesh:
    """Smooth a mesh using Laplacian or Taubin smoothing.
    
    'laplacian': Simple Laplacian (shrinks mesh)
    'taubin': Taubin smoothing (volume-preserving, alternates lambda and mu)
    """
    result = mesh.copy()
    
    for _ in range(iterations):
        if method == 'laplacian':
            _apply_laplacian(result, lambda_factor)
        elif method == 'taubin':
            _apply_laplacian(result, lambda_factor)
            _apply_laplacian(result, mu_factor)
            
    result.compute_vertex_normals()
    return result

def _apply_laplacian(mesh: HalfEdgeMesh, factor: float):
    new_pos = []
    for v in mesh.vertices:
        neighbors = mesh.get_vertex_neighbors(v)
        if not neighbors or mesh.is_boundary_vertex(v):
            new_pos.append(v.position.copy())
        else:
            avg_pos = np.mean([n.position for n in neighbors], axis=0)
            new_pos.append(v.position + factor * (avg_pos - v.position))
            
    for i, v in enumerate(mesh.vertices):
        v.position = new_pos[i]

def offset_mesh(mesh: HalfEdgeMesh, distance: float = 0.1) -> HalfEdgeMesh:
    """Offset mesh along vertex normals by the given distance."""
    result = mesh.copy()
    result.compute_vertex_normals()
    
    for v in result.vertices:
        v.position = v.position + v.normal * distance
        
    return result

def _decimate_with_frozen(verts: np.ndarray, faces: np.ndarray,
                          frozen_set: set, target_faces: int):
    """Edge-collapse decimation that never moves or removes a frozen vertex.

    Only true mesh edges between two currently-adjacent, unfrozen vertices are
    collapsed, and the vertex/vertex and vertex/face adjacency is updated after
    every collapse. The previous implementation kept a union-find alias table
    built from the ORIGINAL edge list, so once a vertex had been merged its
    stale edges were reinterpreted as edges between the merge representatives --
    which are usually not adjacent at all, and on a sphere end up antipodal.

    Collapses are additionally guarded by:
      * the link condition (the common neighbourhood of the two endpoints must
        be exactly the vertices opposite the shared faces) which is what keeps
        the result manifold and fold-free, and
      * a floor of three incident faces on every frozen vertex, so a frozen
        vertex can never be stranded without a face and dropped later as
        unreferenced.
    """
    import heapq

    n_v = len(verts)
    vertex_faces = [set() for _ in range(n_v)]
    adj = [set() for _ in range(n_v)]
    for fi, f in enumerate(faces):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for v in (a, b, c):
            vertex_faces[v].add(fi)
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    for v in range(n_v):
        adj[v].discard(v)

    alive = np.ones(n_v, dtype=bool)
    face_active = np.ones(len(faces), dtype=bool)
    current_faces = int(len(faces))

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    heap = []
    seen = set()
    for v in range(n_v):
        if v in frozen_set:
            continue
        for w in adj[v]:
            if w in frozen_set:
                continue
            k = edge_key(v, w)
            if k in seen:
                continue
            seen.add(k)
            heapq.heappush(heap, (float(np.linalg.norm(verts[k[0]] - verts[k[1]])), k[0], k[1]))

    while heap and current_faces > target_faces:
        _, v1, v2 = heapq.heappop(heap)
        # lazy validation: both endpoints must still exist and still be adjacent
        if not (alive[v1] and alive[v2]) or v2 not in adj[v1]:
            continue

        shared = vertex_faces[v1] & vertex_faces[v2]
        shared = {fi for fi in shared if face_active[fi]}
        if not shared or len(shared) > 2:
            continue

        # link condition: neighbours common to both endpoints must be exactly
        # the apex vertices of the shared faces, otherwise the collapse folds
        # the surface onto itself.
        apex = set()
        for fi in shared:
            apex.update(int(x) for x in faces[fi])
        apex -= {v1, v2}
        if (adj[v1] & adj[v2]) != apex:
            continue

        # frozen vertices must keep at least three live faces
        bad = False
        for fi in shared:
            for x in faces[fi]:
                x = int(x)
                if x in frozen_set:
                    live = sum(1 for g in vertex_faces[x] if face_active[g])
                    if live - 1 < 3:
                        bad = True
                        break
            if bad:
                break
        if bad:
            continue

        # perform the collapse: v2 -> v1
        for fi in shared:
            face_active[fi] = False
            current_faces -= 1
        for fi in list(vertex_faces[v2]):
            if face_active[fi]:
                vertex_faces[v1].add(fi)
                faces[fi] = [v1 if int(x) == v2 else int(x) for x in faces[fi]]
        vertex_faces[v2] = set()
        vertex_faces[v1] = {fi for fi in vertex_faces[v1] if face_active[fi]}

        for w in adj[v2]:
            adj[w].discard(v2)
            if w != v1:
                adj[w].add(v1)
                adj[v1].add(w)
        adj[v2] = set()
        adj[v1].discard(v1)
        alive[v2] = False

        if v1 not in frozen_set:
            for w in adj[v1]:
                if w in frozen_set:
                    continue
                heapq.heappush(
                    heap,
                    (float(np.linalg.norm(verts[v1] - verts[w])), min(v1, w), max(v1, w)),
                )

    new_faces = []
    for fi, active in enumerate(face_active):
        if not active:
            continue
        a, b, c = (int(x) for x in faces[fi])
        if a != b and b != c and c != a:
            new_faces.append([a, b, c])
    return new_faces


def decimate_mesh(mesh: HalfEdgeMesh, target_faces: int = None,
                  ratio: float = 0.5, frozen_vertices: Optional[List[int]] = None) -> HalfEdgeMesh:
    """Reduce face count using edge collapse decimation.

    Uses trimesh's simplify_quadric_decimation, or -- when ``frozen_vertices``
    is given -- a local edge-collapse pass that never touches those vertices.

    Note: the mesh round-trips through ``to_trimesh()``, so a quad/ngon cage
    comes back triangulated (a warning says so).
    """
    _warn_if_not_triangles(mesh, "decimate_mesh")
    t_mesh = mesh.to_trimesh()

    if target_faces is None:
        target_faces = int(len(mesh.faces) * ratio)

    if frozen_vertices:
        verts = np.array(t_mesh.vertices, dtype=np.float64)
        faces = np.array(t_mesh.faces, dtype=np.int64).tolist()
        frozen_set = {int(i) for i in frozen_vertices}

        new_faces = _decimate_with_frozen(verts, faces, frozen_set, int(target_faces))

        try:
            import trimesh
            # process=False: merging vertices here would silently relocate or
            # fuse the very vertices the caller asked us to freeze.
            new_t_mesh = trimesh.Trimesh(vertices=verts, faces=new_faces, process=False)
            new_t_mesh.update_faces(new_t_mesh.nondegenerate_faces())
            new_t_mesh.remove_unreferenced_vertices()
            return HalfEdgeMesh.from_trimesh(new_t_mesh)
        except Exception as e:
            warnings.warn(
                f"decimate_mesh: rebuilding the trimesh after the frozen-vertex "
                f"pass failed ({e}); returning an UNDECIMATED copy of the input "
                f"-- the face count was NOT reduced.",
                RuntimeWarning,
                stacklevel=2,
            )
            return mesh.copy()

    try:
        # trimesh >= 5 renamed the args to (percent, face_count, aggression);
        # passing the count positionally lands in `percent` and raises.
        decimated = t_mesh.simplify_quadric_decimation(face_count=target_faces)
        return HalfEdgeMesh.from_trimesh(decimated)
    except TypeError:
        try:
            # older trimesh: face_count was the first positional argument
            decimated = t_mesh.simplify_quadric_decimation(target_faces)
            return HalfEdgeMesh.from_trimesh(decimated)
        except Exception as e:
            warnings.warn(
                f"decimate_mesh: quadric decimation failed ({e}); returning an "
                f"UNDECIMATED copy of the input -- the face count was NOT "
                f"reduced.",
                RuntimeWarning,
                stacklevel=2,
            )
            return mesh.copy()
    except Exception as e:
        warnings.warn(
            f"decimate_mesh: quadric decimation failed ({e}); returning an "
            f"UNDECIMATED copy of the input -- the face count was NOT reduced.",
            RuntimeWarning,
            stacklevel=2,
        )
        return mesh.copy()

def collapse_short_edges(t_mesh: 'trimesh.Trimesh',
                         rel_threshold: float = 0.15) -> 'trimesh.Trimesh':
    """Collapse triangle-mesh edges much shorter than the median edge length.

    Each short edge's endpoints merge at their midpoint (disjoint pairs per
    round, up to 4 rounds). Degenerate and duplicate faces are dropped after
    every round.
    """
    for _ in range(4):
        verts = np.asarray(t_mesh.vertices).copy()
        faces = np.asarray(t_mesh.faces)
        if len(faces) == 0:
            break
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        edges = np.unique(edges, axis=0)
        lens = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        threshold = rel_threshold * np.median(lens)
        short = edges[lens < threshold]
        if len(short) == 0:
            break
        used = set()
        remap = np.arange(len(verts))
        n_collapsed = 0
        for a, b in short:
            if a in used or b in used:
                continue
            used.update((int(a), int(b)))
            verts[a] = (verts[a] + verts[b]) / 2.0
            remap[b] = a
            n_collapsed += 1
        if n_collapsed == 0:
            break
        new_faces = remap[faces]
        keep = ((new_faces[:, 0] != new_faces[:, 1]) &
                (new_faces[:, 1] != new_faces[:, 2]) &
                (new_faces[:, 2] != new_faces[:, 0]))
        print(f"Sliver removal: collapsed {n_collapsed} short edge(s) "
              f"(< {threshold:.3f})")
        t_mesh = trimesh.Trimesh(vertices=verts, faces=new_faces[keep],
                                 process=False)
        t_mesh.merge_vertices()
        t_mesh.update_faces(t_mesh.nondegenerate_faces())
        t_mesh.update_faces(t_mesh.unique_faces())
        t_mesh.remove_unreferenced_vertices()
    return t_mesh


def flip_needle_triangles(t_mesh: 'trimesh.Trimesh',
                          rel_height: float = 0.05) -> 'trimesh.Trimesh':
    """Remove needle/cap triangles by flipping their longest edge.

    A needle's edges can all be long while its height is near zero. Flipping
    the longest edge against the neighbouring triangle removes the needle
    without moving any vertex; a flip only happens when it strictly improves
    the worse of the two triangles.
    """
    for _ in range(5):
        verts = np.asarray(t_mesh.vertices)
        faces = np.asarray(t_mesh.faces).copy()
        if len(faces) == 0:
            break
        tri_pts = verts[faces]
        edge_vecs = np.roll(tri_pts, -1, axis=1) - tri_pts
        edge_lens = np.linalg.norm(edge_vecs, axis=2)
        areas = 0.5 * np.linalg.norm(
            np.cross(edge_vecs[:, 0], -edge_vecs[:, 2]), axis=1)
        longest = edge_lens.max(axis=1)
        heights = np.divide(2.0 * areas, longest,
                            out=np.zeros_like(areas), where=longest > 1e-12)
        median_edge = np.median(edge_lens)
        h_min = rel_height * median_edge
        sliver_ids = np.where(heights < h_min)[0]
        if len(sliver_ids) == 0:
            break

        edge_face = {}
        for fi, f in enumerate(faces):
            for k in range(3):
                edge_face[(f[k], f[(k + 1) % 3])] = fi

        def height_of(tri):
            p = verts[tri]
            e = np.roll(p, -1, axis=0) - p
            area = 0.5 * np.linalg.norm(np.cross(e[0], -e[2]))
            longest_e = np.linalg.norm(e, axis=1).max()
            return 2.0 * area / longest_e if longest_e > 1e-12 else 0.0

        flipped = set()
        n_flips = 0
        for fi in sliver_ids:
            if fi in flipped:
                continue
            f = faces[fi]
            k = int(np.argmax(edge_lens[fi]))
            a, b = f[k], f[(k + 1) % 3]
            c = f[(k + 2) % 3]
            nb = edge_face.get((b, a))
            if nb is None or nb in flipped or nb == fi:
                continue
            g = faces[nb]
            d = [v for v in g if v != a and v != b]
            if len(d) != 1:
                continue
            d = d[0]
            # flip AB -> CD: (A,B,C),(B,A,D) => (A,D,C),(D,B,C)
            new1 = np.array([a, d, c])
            new2 = np.array([d, b, c])
            if min(height_of(new1), height_of(new2)) <= heights[fi]:
                continue
            faces[fi] = new1
            faces[nb] = new2
            flipped.update((fi, nb))
            n_flips += 1
        if n_flips == 0:
            break
        print(f"Sliver removal: flipped {n_flips} needle triangle(s) "
              f"(height < {h_min:.3f})")
        t_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        t_mesh.merge_vertices()
        t_mesh.update_faces(t_mesh.nondegenerate_faces())
        t_mesh.update_faces(t_mesh.unique_faces())
        t_mesh.remove_unreferenced_vertices()
    return t_mesh


def remove_sliver_edges(mesh: HalfEdgeMesh,
                        rel_threshold: float = 0.15,
                        rel_height: float = 0.05) -> HalfEdgeMesh:
    """Remove sliver geometry: collapse very short edges and flip needle
    triangles.

    Sliver elements survive smoothing and projection, and their faces are
    silently dropped by strict CAD importers (SolidWorks rejects any solid
    face with a sub-tolerance edge — the body then arrives as an unknittable
    surface and the part looks empty). Thresholds are relative to the median
    edge length.

    Note: the mesh round-trips through ``to_trimesh()``, so a quad cage comes
    back triangulated (a warning says so).
    """
    _warn_if_not_triangles(mesh, "remove_sliver_edges")
    t_mesh = mesh.to_trimesh()
    if len(t_mesh.faces) == 0:
        return mesh.copy()
    t_mesh = collapse_short_edges(t_mesh, rel_threshold=rel_threshold)
    t_mesh = flip_needle_triangles(t_mesh, rel_height=rel_height)
    return HalfEdgeMesh.from_trimesh(t_mesh)


def remove_duplicate_vertices(mesh: HalfEdgeMesh, tolerance: float = 1e-6) -> HalfEdgeMesh:
    """Merge vertices that are within tolerance distance of each other.

    trimesh >= 5 quantizes to a power-of-ten grid (``digits_vertex`` decimal
    places) rather than to an arbitrary tolerance, so the tolerance can only be
    honoured approximately. The digit count is rounded UP (``ceil``), making the
    effective grid step 10**-digits <= tolerance: e.g. tolerance 5e-5 gives
    digits 5 (grid 1e-5), never digits 4 (grid 1e-4) which would merge vertices
    up to twice as far apart as requested. Erring towards a finer grid keeps
    the operation conservative -- it may leave a near-duplicate behind, but it
    will not fuse geometry the caller wanted kept apart.

    Note: the mesh round-trips through ``to_trimesh()``, so a quad/ngon cage
    comes back triangulated (a warning says so).
    """
    _warn_if_not_triangles(mesh, "remove_duplicate_vertices")
    try:
        t_mesh = mesh.to_trimesh()
        digits = max(0, int(np.ceil(-np.log10(max(tolerance, 1e-12)))))
        try:
            t_mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_vertex=digits)
        except TypeError:
            t_mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_or_tol=tolerance)
        return HalfEdgeMesh.from_trimesh(t_mesh)
    except Exception as e:
        print(f"Error merging vertices: {e}")
        return mesh.copy()

def compute_mesh_quality(mesh: HalfEdgeMesh) -> dict:
    """Compute mesh quality metrics.

    Returns dict with:
        'face_count', 'vertex_count', 'edge_count',
        'min_angle', 'max_angle', 'avg_angle'  (corner angles in DEGREES,
            measured on the triangulated mesh -- a quad cage is fanned into
            triangles by ``to_trimesh()`` first),
        'min_area', 'max_area',
        'watertight': bool,
        'manifold': bool,
        'boundary_edges': int,
        'non_manifold_edges': int  (undirected edges shared by >2 faces)
    """
    t_mesh = mesh.to_trimesh()

    stats = {
        'face_count': len(mesh.faces),
        'vertex_count': len(mesh.vertices),
        'edge_count': len(mesh.edges),
        'watertight': t_mesh.is_watertight,
        'manifold': t_mesh.is_winding_consistent,
    }

    if len(mesh.faces) > 0 and len(t_mesh.faces) > 0:
        areas = t_mesh.area_faces
        stats['min_area'] = float(np.min(areas))
        stats['max_area'] = float(np.max(areas))

        # Corner angles come from trimesh (radians) -> report degrees.
        angles = np.degrees(np.asarray(t_mesh.face_angles).ravel())
        angles = angles[np.isfinite(angles)]
        if len(angles):
            stats['min_angle'] = float(np.min(angles))
            stats['max_angle'] = float(np.max(angles))
            stats['avg_angle'] = float(np.mean(angles))

    # Boundary edges
    boundary_edges = sum(1 for e in mesh.edges if mesh.is_boundary_edge(e))
    stats['boundary_edges'] = boundary_edges

    # Non-manifold edges: count undirected edges with face incidence > 2.
    # The half-edge structure does NOT enforce manifoldness -- add_face happily
    # accepts a third face on an existing edge, it just leaves the extra
    # half-edge untwinned -- so this has to be measured on the face set.
    incidence: Dict[tuple, int] = {}
    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        n = len(fv)
        for i in range(n):
            a, b = fv[i], fv[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            incidence[key] = incidence.get(key, 0) + 1
    stats['non_manifold_edges'] = int(sum(1 for c in incidence.values() if c > 2))

    return stats
