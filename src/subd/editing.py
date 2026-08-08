import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh


def _unit(vec) -> np.ndarray | None:
    """Normalise `vec`, or return None if it carries no direction.

    Dividing by an unguarded length turns a degenerate (zero-area) face's normal
    into NaN and silently poisons every vertex derived from it.
    """
    v = np.array(vec, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length < 1e-12:
        return None
    return v / length


def extrude_faces(mesh: HalfEdgeMesh, face_indices: list[int], distance: float = 0.1, direction: np.ndarray = None) -> HalfEdgeMesh:
    """Extrude selected faces along their normals or a given direction.

    A face with no usable normal (degenerate/zero area) and no explicit
    `direction` is skipped rather than extruded to NaN.
    """
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]

    mesh.compute_face_normals()

    for f_idx in face_indices:
        if f_idx >= len(mesh.faces): continue
        f = mesh.faces[f_idx]
        f_verts = [v.index for v in mesh.get_face_vertices(f)]

        extrude_dir = _unit(direction if direction is not None else f.normal)
        if extrude_dir is None:
            continue

        new_f_verts = []
        for v_idx in f_verts:
            new_pos = verts[v_idx] + extrude_dir * distance
            verts.append(new_pos)
            new_f_verts.append(len(verts) - 1)
            
        faces[f_idx] = new_f_verts
        
        n = len(f_verts)
        for i in range(n):
            v1 = f_verts[i]
            v2 = f_verts[(i + 1) % n]
            nv1 = new_f_verts[i]
            nv2 = new_f_verts[(i + 1) % n]
            faces.append([v1, v2, nv2, nv1])
            
    return HalfEdgeMesh.from_arrays(verts, faces)


def extrude_edges(mesh: HalfEdgeMesh, edge_indices: list[int], distance: float = 0.1, direction: np.ndarray = None) -> HalfEdgeMesh:
    """Extrude edges, creating new faces.

    `direction` is a direction only: the offset is always `distance` long,
    independent of the vector's magnitude.
    """
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]

    mesh.compute_vertex_normals()

    for e_idx in edge_indices:
        if e_idx >= len(mesh.edges): continue
        e = mesh.edges[e_idx]
        if e.half_edge is None or e.half_edge.prev is None: continue
        # e.half_edge runs v1 -> v2 inside its own face, so that face already
        # owns the directed edge (v1, v2).
        v1_idx = e.half_edge.prev.vertex.index
        v2_idx = e.half_edge.vertex.index

        dir1 = _unit(direction if direction is not None else mesh.vertices[v1_idx].normal)
        dir2 = _unit(direction if direction is not None else mesh.vertices[v2_idx].normal)
        if dir1 is None or dir2 is None:
            continue

        nv1 = verts[v1_idx] + dir1 * distance
        nv2 = verts[v2_idx] + dir2 * distance

        verts.append(nv1)
        nv1_idx = len(verts) - 1
        verts.append(nv2)
        nv2_idx = len(verts) - 1

        # Wind the strip against the source face: it must traverse the shared
        # edge as (v2, v1). Reusing (v1, v2) — as the old code did — means both
        # faces walk the edge the same way, so they can never be twinned and the
        # result can never close up.
        faces.append([v2_idx, v1_idx, nv1_idx, nv2_idx])

    return HalfEdgeMesh.from_arrays(verts, faces)


def inset_faces(mesh: HalfEdgeMesh, face_indices: list[int], inset_amount: float = 0.1) -> HalfEdgeMesh:
    """Create an inset ring within each selected face."""
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    for f_idx in face_indices:
        if f_idx >= len(mesh.faces): continue
        f = mesh.faces[f_idx]
        f_verts = [v.index for v in mesh.get_face_vertices(f)]
        
        centroid = np.mean([verts[v] for v in f_verts], axis=0)
        
        new_f_verts = []
        for v_idx in f_verts:
            dir_to_center = centroid - verts[v_idx]
            length = np.linalg.norm(dir_to_center)
            if length > 0:
                dir_to_center /= length
            
            actual_inset = min(inset_amount, length)
            new_pos = verts[v_idx] + dir_to_center * actual_inset
            verts.append(new_pos)
            new_f_verts.append(len(verts) - 1)
            
        faces[f_idx] = new_f_verts
        
        n = len(f_verts)
        for i in range(n):
            v1 = f_verts[i]
            v2 = f_verts[(i + 1) % n]
            nv1 = new_f_verts[i]
            nv2 = new_f_verts[(i + 1) % n]
            faces.append([v1, v2, nv2, nv1])
            
    return HalfEdgeMesh.from_arrays(verts, faces)


def _orient_edge_ring(mesh: HalfEdgeMesh, ring_edges: list, ring_edge_indices: set) -> dict:
    """
    Choose a consistent (start, end) vertex pair for every edge in a ring.

    Each edge's stored half-edge points whichever way that edge happened to be
    created, so interpolating `position` along it independently makes the loop
    zigzag from one end of the ring to the other. Walking the ring and flipping
    the orientation across each quad keeps every new point on the same side.

    In a quad a -> b -> c -> d the ring edges are (a,b) and (c,d); a's partner
    across the quad is d, so an edge oriented a->b forces the opposite one to
    d->c, i.e. the reverse of that face's own traversal.
    """
    from collections import deque

    orient: dict[int, tuple[int, int]] = {}
    queue: deque = deque()

    for seed in ring_edges:
        if seed.index in orient:
            continue
        he = seed.half_edge
        if he is None or he.prev is None or he.vertex is None or he.prev.vertex is None:
            continue
        orient[seed.index] = (he.prev.vertex.index, he.vertex.index)
        queue.append(seed.index)

        while queue:
            ei = queue.popleft()
            e = mesh.edges[ei]
            for h in (e.half_edge, e.half_edge.twin if e.half_edge else None):
                if h is None or h.prev is None or h.next is None:
                    continue
                opp = h.next.next
                # Only propagate across a genuine quad.
                if opp is None or opp.next is None or opp.next.next is not h:
                    continue
                if opp.edge is None or opp.edge.index not in ring_edge_indices:
                    continue
                if opp.edge.index in orient:
                    continue
                a, b = h.prev.vertex.index, h.vertex.index
                c, d = opp.prev.vertex.index, opp.vertex.index
                orient[opp.edge.index] = (d, c) if orient[ei] == (a, b) else (c, d)
                queue.append(opp.edge.index)

    return orient


def insert_edge_loop(mesh: HalfEdgeMesh, edge_index: int, position: float = 0.5) -> HalfEdgeMesh:
    """Insert a new edge loop cutting through a ring of connected quads."""
    if edge_index >= len(mesh.edges):
        return mesh.copy()

    # get_edge_loop in halfedge_mesh traverses opposite edges, which effectively gets an edge ring
    raw_edges = mesh.get_edge_loop(mesh.edges[edge_index])
    ring_edges = list({e.index: e for e in raw_edges}.values())
    ring_edge_indices = set(e.index for e in ring_edges)

    verts = [v.position.copy() for v in mesh.vertices]
    faces = []

    orient = _orient_edge_ring(mesh, ring_edges, ring_edge_indices)

    edge_to_new_vert = {}
    for e in ring_edges:
        if e.half_edge is None or e.half_edge.prev is None: continue
        start, end = orient.get(
            e.index, (e.half_edge.prev.vertex.index, e.half_edge.vertex.index))
        v1 = mesh.vertices[start].position
        v2 = mesh.vertices[end].position
        new_pos = v1 * (1.0 - position) + v2 * position
        verts.append(new_pos)
        edge_to_new_vert[e.index] = len(verts) - 1

    for f in mesh.faces:
        f_edges = []
        he = f.half_edge
        start_he = he
        if he is None: continue
        visited = set()
        while True:
            if he is None or he.index in visited: break
            visited.add(he.index)
            f_edges.append(he)
            he = he.next
            if he == start_he: break
            
        split_hes = [he for he in f_edges if he.edge.index in ring_edge_indices]
        if len(split_hes) == 2 and len(f_edges) == 4:
            idx0 = f_edges.index(split_hes[0])
            he0, he1, he2, he3 = f_edges[idx0], f_edges[(idx0+1)%4], f_edges[(idx0+2)%4], f_edges[(idx0+3)%4]
            nv0 = edge_to_new_vert[he0.edge.index]
            nv2 = edge_to_new_vert[he2.edge.index]
            v_a = he3.vertex.index
            v_b = he0.vertex.index
            v_c = he1.vertex.index
            v_d = he2.vertex.index
            faces.append([v_a, nv0, nv2, v_d])
            faces.append([nv0, v_b, v_c, nv2])
        else:
            faces.append([he.vertex.index for he in f_edges])
            
    return HalfEdgeMesh.from_arrays(verts, faces)


def bridge_faces(mesh: HalfEdgeMesh, face_indices_a: list[int], face_indices_b: list[int]) -> HalfEdgeMesh:
    """Delete both face groups and connect their boundary edges with new quad faces.

    If the two groups do not present matching boundary loops there is nothing to
    bridge, and the mesh is returned untouched. Deleting the faces first — as the
    old code did — punched two permanent holes in the model whenever the bridge
    could not be built.
    """
    verts = [v.position.copy() for v in mesh.vertices]

    def get_boundary_loop(group_indices):
        boundary_edges = []
        for f_idx in group_indices:
            f = mesh.faces[f_idx]
            he = f.half_edge
            start_he = he
            if he is None: continue
            visited = set()
            while True:
                if he is None or he.index in visited: break
                visited.add(he.index)
                twin_face_idx = he.twin.face.index if he.twin and he.twin.face else -1
                if twin_face_idx not in group_indices:
                    boundary_edges.append(he)
                he = he.next
                if he == start_he: break
        
        if not boundary_edges: return []
        loop = [boundary_edges.pop(0)]
        while boundary_edges:
            last_v = loop[-1].vertex
            found = False
            for i, he in enumerate(boundary_edges):
                if he.prev and he.prev.vertex == last_v:
                    loop.append(boundary_edges.pop(i))
                    found = True
                    break
            if not found: break 
        return [he.prev.vertex.index for he in loop if he.prev]

    loop_a = get_boundary_loop(set(face_indices_a))
    loop_b = get_boundary_loop(set(face_indices_b))

    # Decide before destroying anything: an unbridgeable request is a no-op.
    if len(loop_a) != len(loop_b) or len(loop_a) < 3:
        return mesh.copy()

    to_delete = set(face_indices_a + face_indices_b)
    faces = [[v.index for v in mesh.get_face_vertices(f)]
             for f in mesh.faces if f.index not in to_delete]

    loop_b.reverse()
    n = len(loop_a)
    best_shift = 0
    min_dist = float('inf')
    for shift in range(n):
        dist = sum(np.linalg.norm(verts[loop_a[i]] - verts[loop_b[(i + shift) % n]]) for i in range(n))
        if dist < min_dist:
            min_dist = dist
            best_shift = shift

    loop_b_aligned = loop_b[best_shift:] + loop_b[:best_shift]

    for i in range(n):
        v1 = loop_a[i]
        v2 = loop_a[(i + 1) % n]
        v3 = loop_b_aligned[(i + 1) % n]
        v4 = loop_b_aligned[i]
        faces.append([v1, v2, v3, v4])

    return HalfEdgeMesh.from_arrays(verts, faces)


def mirror_mesh(mesh: HalfEdgeMesh, axis: str = 'x', merge_distance: float = 0.001) -> HalfEdgeMesh:
    """Mirror the mesh across the given axis plane, merging vertices within tolerance.

    A face whose vertices all lie in the mirror plane maps onto itself, so its
    reflection is dropped instead of being stacked on top of the original as a
    zero-thickness double wall.

    The same applies to a body that is ALREADY symmetric about the plane: every
    mirrored vertex merges back onto an existing one, so every reflected face
    lands on a face that is already there. Reflections whose vertex set repeats
    a face already in the result are dropped, so mirroring a symmetric mesh is
    an identity instead of a doubled, zero-thickness shell.
    """
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    n_orig = len(mesh.vertices)
    if n_orig == 0:
        return mesh.copy()

    positions = np.array([v.position for v in mesh.vertices], dtype=np.float64)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)

    mirrored = positions.copy()
    mirrored[:, axis_idx] *= -1.0

    # One spatial index instead of rebuilding an (N,3) array per vertex, which
    # made the merge O(N^2).
    remap = np.empty(n_orig, dtype=np.int64)
    all_verts = [p.copy() for p in positions]
    try:
        from scipy.spatial import cKDTree
        dist, nearest = cKDTree(positions).query(
            mirrored, distance_upper_bound=merge_distance)
        for i in range(n_orig):
            if nearest[i] < n_orig and dist[i] <= merge_distance:
                remap[i] = nearest[i]
            else:
                remap[i] = len(all_verts)
                all_verts.append(mirrored[i])
    except ImportError:
        # Grid hash fallback: still linear, no extra dependency.
        cell = max(merge_distance, 1e-12)
        buckets: dict[tuple, list[int]] = {}
        for i, p in enumerate(positions):
            buckets.setdefault(tuple(np.floor(p / cell).astype(np.int64)), []).append(i)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        for i, mp in enumerate(mirrored):
            base = np.floor(mp / cell).astype(np.int64)
            hit = -1
            best = merge_distance
            for off in offsets:
                for j in buckets.get((base[0] + off[0], base[1] + off[1], base[2] + off[2]), ()):
                    d = float(np.linalg.norm(positions[j] - mp))
                    if d <= best:
                        best, hit = d, j
            if hit >= 0:
                remap[i] = hit
            else:
                remap[i] = len(all_verts)
                all_verts.append(mp)

    on_plane = np.abs(positions[:, axis_idx]) <= merge_distance

    new_faces = list(faces)
    # Vertex sets already present, so a reflection that lands on an existing
    # face can be recognised regardless of winding or starting corner.
    seen = {frozenset(int(i) for i in f) for f in faces if f}
    for f in faces:
        if f and all(on_plane[idx] for idx in f):
            # Its own reflection: adding it again would double-wall the surface.
            continue
        reflected = [int(remap[idx]) for idx in f[::-1]]
        key = frozenset(reflected)
        if key in seen:
            # Already-symmetric input: the reflection coincides with a face
            # that is already there. Appending it would build a second, reverse
            # -wound shell on the very same vertices.
            continue
        seen.add(key)
        new_faces.append(reflected)

    return HalfEdgeMesh.from_arrays(all_verts, new_faces)


def soft_selection_move(mesh: HalfEdgeMesh, vertex_index: int, offset: np.ndarray, radius: float, falloff: str = 'smooth') -> HalfEdgeMesh:
    """Move a vertex by offset, with surrounding vertices influenced by distance-based falloff."""
    new_mesh = mesh.copy()
    if vertex_index >= len(new_mesh.vertices):
        return new_mesh
        
    center_pos = new_mesh.vertices[vertex_index].position.copy()
    offset = np.array(offset, dtype=np.float64)
    
    for v in new_mesh.vertices:
        dist = np.linalg.norm(v.position - center_pos)
        if dist < radius:
            t = dist / radius
            if falloff == 'linear':
                weight = 1.0 - t
            elif falloff == 'smooth':
                weight = 0.5 * (1.0 + np.cos(np.pi * t))
            elif falloff == 'sharp':
                weight = (1.0 - t) ** 2
            else:
                weight = 1.0 - t
            v.position += offset * weight
            
    return new_mesh


def set_edge_weight(mesh: HalfEdgeMesh, edge_indices: list[int], weight: float) -> HalfEdgeMesh:
    """Set crease weight on selected edges."""
    new_mesh = mesh.copy()
    for e_idx in edge_indices:
        if e_idx < len(new_mesh.edges):
            new_mesh.edges[e_idx].crease_weight = weight
    return new_mesh


def edge_slide(mesh: HalfEdgeMesh, vertex_indices: list[int], amount: float) -> HalfEdgeMesh:
    """Slide selected vertices along their adjacent edges."""
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    for v_idx in vertex_indices:
        if v_idx >= len(mesh.vertices): continue
        v = mesh.vertices[v_idx]
        neighbors = mesh.get_vertex_neighbors(v)
        if not neighbors: continue
        
        # Calculate average edge vector away from vertex
        avg_vec = np.zeros(3)
        for n in neighbors:
            vec = n.position - v.position
            length = np.linalg.norm(vec)
            if length > 1e-6:
                avg_vec += (vec / length)
                
        avg_vec /= len(neighbors)
        avg_len = np.linalg.norm(avg_vec)
        if avg_len > 1e-6:
            verts[v_idx] += (avg_vec / avg_len) * amount
            
    return HalfEdgeMesh.from_arrays(verts, faces)


def bevel_edges(mesh: HalfEdgeMesh, edge_indices: list[int], distance: float = 0.1) -> HalfEdgeMesh:
    """Bevel selected edges by scaling adjacent faces (simplified for Sub-D)."""
    if not edge_indices:
        return mesh.copy()
        
    face_indices = set()
    for e_idx in edge_indices:
        if e_idx >= len(mesh.edges): continue
        e = mesh.edges[e_idx]
        if e.half_edge:
            if e.half_edge.face:
                face_indices.add(e.half_edge.face.index)
            if e.half_edge.twin and e.half_edge.twin.face:
                face_indices.add(e.half_edge.twin.face.index)
            
    return inset_faces(mesh, list(face_indices), inset_amount=distance)


def _newell_normal(points: np.ndarray) -> np.ndarray:
    """Polygon normal, robust for non-planar and non-convex outlines."""
    Q = np.roll(points, -1, axis=0)
    d = points - Q
    s = points + Q
    return np.array([
        float(np.dot(d[:, 1], s[:, 2])),
        float(np.dot(d[:, 2], s[:, 0])),
        float(np.dot(d[:, 0], s[:, 1])),
    ])


def knife_cut(mesh: HalfEdgeMesh, face_idx: int, p1: np.ndarray, p2: np.ndarray) -> HalfEdgeMesh:
    """Split one face along the line through p1 and p2.

    The line is swept perpendicular to the face, so `p1`/`p2` need not lie in the
    face plane; the infinite line is used, not just the segment between them.
    The face is replaced by the two pieces either side of the cut, and any point
    inserted on a shared edge is also inserted into the neighbouring face so the
    result keeps its T-junction-free, watertight topology.

    Returns the mesh unchanged if the line does not cross the face cleanly in
    exactly two places.
    """
    if face_idx < 0 or face_idx >= len(mesh.faces):
        return mesh.copy()

    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]

    poly = faces[face_idx]
    n = len(poly)
    if n < 3:
        return mesh.copy()

    P = np.array([verts[i] for i in poly], dtype=np.float64)

    face_n = _newell_normal(P)
    a = np.asarray(p1, dtype=np.float64).reshape(3)
    b = np.asarray(p2, dtype=np.float64).reshape(3)
    line_d = b - a

    # Cutting plane: contains the line and is perpendicular to the face.
    cut_n = np.cross(line_d, face_n)
    scale = float(np.linalg.norm(face_n)) * float(np.linalg.norm(line_d))
    if scale < 1e-20 or float(np.linalg.norm(cut_n)) < 1e-9 * scale:
        return mesh.copy()
    cut_n /= np.linalg.norm(cut_n)

    s = P @ cut_n - float(np.dot(cut_n, a))
    tol = 1e-12 * max(1.0, float(np.abs(P).max()))

    # Walk the outline, inserting a point wherever the cut plane crosses it.
    augmented: list[int] = []
    cut_slots: list[int] = []
    split_edges: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        if abs(s[i]) <= tol:
            cut_slots.append(len(augmented))
            augmented.append(poly[i])
            continue
        augmented.append(poly[i])
        if abs(s[j]) > tol and (s[i] > 0) != (s[j] > 0):
            t = s[i] / (s[i] - s[j])
            point = P[i] + t * (P[j] - P[i])
            new_idx = len(verts)
            verts.append(point)
            cut_slots.append(len(augmented))
            augmented.append(new_idx)
            split_edges.append((poly[i], poly[j], new_idx))

    if len(cut_slots) != 2:
        return mesh.copy()

    k1, k2 = cut_slots
    half_a = augmented[k1:k2 + 1]
    half_b = augmented[k2:] + augmented[:k1 + 1]
    if len(half_a) < 3 or len(half_b) < 3:
        return mesh.copy()

    # Carry each inserted point into the face on the other side of that edge,
    # otherwise the neighbour keeps a straight edge across a new vertex and the
    # mesh acquires a T-junction (a hole, as far as the half-edge structure goes).
    for (va, vb, new_idx) in split_edges:
        for fi, fv in enumerate(faces):
            if fi == face_idx:
                continue
            m = len(fv)
            for k in range(m):
                if fv[k] == vb and fv[(k + 1) % m] == va:
                    faces[fi] = fv[:k + 1] + [new_idx] + fv[k + 1:]
                    break
            else:
                continue
            break

    faces[face_idx] = half_a
    faces.append(half_b)

    return HalfEdgeMesh.from_arrays(verts, faces)
