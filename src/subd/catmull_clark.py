import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh, Vertex, HalfEdge, Edge, Face

def subdivide(mesh: HalfEdgeMesh, levels: int = 1, smooth: bool = True) -> HalfEdgeMesh:
    """
    Subdivide a mesh using the Catmull-Clark algorithm.

    smooth=False performs linear (midpoint) subdivision: same topology split,
    but existing vertices stay in place and new points sit on face centroids /
    edge midpoints, so the shape is preserved exactly.
    """
    if levels <= 0:
        return mesh.copy()

    current_mesh = mesh
    for _ in range(levels):
        current_mesh = _subdivide_once(current_mesh, smooth=smooth)
    return current_mesh

def _build_incident_edge_map(mesh: HalfEdgeMesh) -> dict[int, list[Edge]]:
    """
    Map every vertex index to the edges touching it, in ascending edge order.

    Built in a single sweep over the edge list so that callers never have to
    rescan all edges per vertex.
    """
    incident: dict[int, list[Edge]] = {v.index: [] for v in mesh.vertices}
    for e in mesh.edges:
        he = e.half_edge
        if he is None or he.prev is None or he.vertex is None or he.prev.vertex is None:
            continue
        tgt = he.vertex.index
        src = he.prev.vertex.index
        if tgt in incident:
            incident[tgt].append(e)
        # A self-loop must not be counted twice for the same vertex.
        if src != tgt and src in incident:
            incident[src].append(e)
    return incident


def _subdivide_once(mesh: HalfEdgeMesh, smooth: bool = True) -> HalfEdgeMesh:
    # 1. Face points
    face_points = {}
    for f in mesh.faces:
        verts = mesh.get_face_vertices(f)
        if not verts:
            continue
        pos = np.mean([v.position for v in verts], axis=0)
        face_points[f.index] = pos

    # 2. Edge points & midpoints
    edge_points = {}
    edge_midpoints = {}
    for e in mesh.edges:
        he = e.half_edge
        v_tgt = he.vertex.position
        v_src = he.prev.vertex.position
        
        midpoint = (v_src + v_tgt) / 2.0
        edge_midpoints[e.index] = midpoint
        
        f1, f2 = mesh.get_edge_faces(e)
        if not smooth or f2 is None:
            # Linear subdivision or boundary edge
            ep = midpoint
        else:
            fp1 = face_points[f1.index]
            fp2 = face_points[f2.index]
            smooth_ep = (v_src + v_tgt + fp1 + fp2) / 4.0
            
            w = np.clip(e.crease_weight, 0.0, 1.0)
            ep = smooth_ep * (1.0 - w) + midpoint * w
            
        edge_points[e.index] = ep
        
    # 3. Vertex points
    # One pass over the edges builds every vertex's incident-edge list. Asking
    # each vertex to rescan all edges instead is O(V*E), which on a real mesh
    # (100k+ edges) dominates everything else in the algorithm.
    incident_edges = _build_incident_edge_map(mesh)

    vertex_points = {}
    for v in mesh.vertices:
        P = v.position
        if not smooth:
            # Linear subdivision: existing vertices do not move
            vertex_points[v.index] = P
            continue
        faces = mesh.get_vertex_faces(v)

        inc_edges = incident_edges[v.index]

        is_boundary = mesh.is_boundary_vertex(v)
        n = len(inc_edges)
        
        if is_boundary:
            bound_edges = [e for e in inc_edges if mesh.is_boundary_edge(e)]
            avg_bound_mid = np.mean([edge_midpoints[e.index] for e in bound_edges], axis=0) if bound_edges else P
            vp = (avg_bound_mid + P) / 2.0
        else:
            F_avg = np.mean([face_points[f.index] for f in faces], axis=0) if faces else P
            R_avg = np.mean([edge_midpoints[e.index] for e in inc_edges], axis=0) if inc_edges else P
            if n > 0:
                vp = (F_avg + 2 * R_avg + (n - 3) * P) / n
            else:
                vp = P
            
        max_w = max([e.crease_weight for e in inc_edges] + [0.0])
        max_w = np.clip(max_w, 0.0, 1.0)
        if max_w > 0:
            crease_edges = [e for e in inc_edges if e.crease_weight > 0]
            if len(crease_edges) == 2:
                avg_crease_mid = np.mean([edge_midpoints[e.index] for e in crease_edges], axis=0)
                crease_vp = (avg_crease_mid + P) / 2.0
            elif len(crease_edges) == 1 and not is_boundary:
                # Dart: a crease that ends inside the surface. The standard
                # dart rule (Hoppe et al. 1994; DeRose, Kass & Truong 1998)
                # is the smooth vertex rule, so the crease fades out here
                # instead of pinning a cone point. A boundary vertex already
                # has two sharp boundary edges, so one crease is a corner.
                crease_vp = vp
            else:
                crease_vp = P
            vp = vp * (1.0 - max_w) + crease_vp * max_w
            
        vertex_points[v.index] = vp
        
    # 4. Build new mesh
    new_mesh = HalfEdgeMesh()
    
    v_idx_map = {}
    for v in mesh.vertices:
        new_v = new_mesh.add_vertex(vertex_points[v.index])
        v_idx_map[v.index] = new_v.index
        
    e_idx_map = {}
    for e in mesh.edges:
        new_v = new_mesh.add_vertex(edge_points[e.index])
        e_idx_map[e.index] = new_v.index
        
    f_idx_map = {}
    for f in mesh.faces:
        new_v = new_mesh.add_vertex(face_points[f.index])
        f_idx_map[f.index] = new_v.index
        
    for f in mesh.faces:
        curr = f.half_edge
        hes = []
        if curr is None:
            continue
        visited = set()
        while True:
            if curr is None or curr.index in visited:
                break
            visited.add(curr.index)
            hes.append(curr)
            curr = curr.next
            if curr == f.half_edge:
                break
                
        for he in hes:
            v_idx = v_idx_map[he.vertex.index]
            e_out_idx = e_idx_map[he.next.edge.index]
            f_pt_idx = f_idx_map[f.index]
            e_in_idx = e_idx_map[he.edge.index]
            
            new_mesh.add_face([v_idx, e_out_idx, f_pt_idx, e_in_idx])
            
    v_is_orig = {idx: orig_idx for orig_idx, idx in v_idx_map.items()}
    v_is_edge = {idx: orig_idx for orig_idx, idx in e_idx_map.items()}
    
    for new_e in new_mesh.edges:
        idx1 = new_e.half_edge.prev.vertex.index
        idx2 = new_e.half_edge.vertex.index
        
        v_orig = None
        e_orig = None
        if idx1 in v_is_orig and idx2 in v_is_edge:
            v_orig = v_is_orig[idx1]
            e_orig = v_is_edge[idx2]
        elif idx2 in v_is_orig and idx1 in v_is_edge:
            v_orig = v_is_orig[idx2]
            e_orig = v_is_edge[idx1]
            
        if v_orig is not None and e_orig is not None:
            orig_edge = mesh.edges[e_orig]
            orig_v1 = orig_edge.half_edge.vertex.index
            orig_v2 = orig_edge.half_edge.prev.vertex.index
            if orig_v1 == v_orig or orig_v2 == v_orig:
                new_e.crease_weight = orig_edge.crease_weight
                
    return new_mesh


def evaluate_limit_surface(mesh: HalfEdgeMesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Project every cage vertex onto the Catmull-Clark limit surface.

    Interior positions are exact: they are where repeated `_subdivide_once`
    passes take each vertex. For an interior vertex of valence n whose edges
    all have crease weight 0 the limit point is

        limit = ((n - 3) * P + 4 * Rbar + 4 * Fbar) / (n + 5)

    with P the vertex, Rbar the mean of the incident edge midpoints and Fbar the
    mean of the incident face centroids. That is the eigenvector of the local
    subdivision matrix for eigenvalue 1, derived from the very rules applied in
    `_subdivide_once`; on a regular (valence 4, all-quad) neighbourhood it
    reproduces the exact bicubic B-spline limit (16*P + 4*sum(neighbours) +
    sum(diagonals)) / 36. The weights sum to n + 5, so the map is affine
    invariant.

    An interior vertex with a creased edge goes through its own local
    subdivision matrix instead (`_creased_interior_limit`). `_subdivide_once`
    clips every weight to [0, 1] and hands it on unchanged to both halves of
    its edge, so the rules around the vertex repeat identically at every level,
    for fractional weights as much as for sharp ones, and the eigenvector gives
    the exact limit for any weights. The fully sharp cases reduce to the
    familiar closed forms: P for three or more sharp edges (corner), and
    (a + 4P + b) / 6 for two sharp edges to a and b (a cubic B-spline crease).
    A dart (one creased edge) uses the smooth vertex rule, but its edge stays
    sharp, so its limit is not the smooth stencil.

    Boundary vertices are left where they are, which keeps the outline of an
    open patch on the cage. That is NOT their subdivision limit: the boundary
    rule of `_subdivide_once` is a cubic B-spline along the boundary, whose
    limit is (a + 4P + b) / 6. A vertex whose fan is not a closed manifold ring
    keeps the smooth formula above, creases or not.

    Normals are an APPROXIMATION, not the analytic limit normal. They are the
    angle-independent area-weighted normals of the polygon mesh formed by the
    limit positions, which converges to the true limit normal as the cage is
    refined but is not equal to it on a coarse cage. Use the limit tangent masks
    if you need exact normals at extraordinary vertices.

    The input mesh is not modified.
    """
    limit_positions = np.zeros((len(mesh.vertices), 3))

    face_points = {}
    for f in mesh.faces:
        verts = mesh.get_face_vertices(f)
        if verts:
            face_points[f.index] = np.mean([v.position for v in verts], axis=0)

    edge_midpoints = {}
    for e in mesh.edges:
        he = e.half_edge
        if he is None or he.prev is None or he.vertex is None or he.prev.vertex is None:
            continue
        edge_midpoints[e.index] = (he.prev.vertex.position + he.vertex.position) / 2.0

    # The local limit weights of a creased vertex depend only on its weights
    # and ring layout, which repeat all along a crease.
    crease_cache = {}

    for v in mesh.vertices:
        P = v.position
        if mesh.is_boundary_vertex(v):
            limit_positions[v.index] = P
            continue

        fan = mesh.get_vertex_fan(v)
        inc_edges = [he.edge for he in fan
                     if he.edge is not None and he.edge.index in edge_midpoints]
        faces = [he.face for he in fan
                 if he.face is not None and he.face.index in face_points]

        n = len(inc_edges)
        if n == 0 or not faces:
            limit_positions[v.index] = P
            continue

        if any(e.crease_weight > 0 for e in inc_edges):
            creased = _creased_interior_limit(P, fan, face_points, edge_midpoints,
                                              crease_cache)
            if creased is not None:
                limit_positions[v.index] = creased
                continue

        R_bar = np.mean([edge_midpoints[e.index] for e in inc_edges], axis=0)
        F_bar = np.mean([face_points[f.index] for f in faces], axis=0)
        limit_positions[v.index] = ((n - 3) * P + 4.0 * R_bar + 4.0 * F_bar) / (n + 5.0)

    limit_normals = _normals_from_positions(mesh, limit_positions)
    return limit_positions, limit_normals


def _creased_interior_limit(P, fan, face_points, edge_midpoints, cache):
    """
    Exact limit of an interior vertex with creased edges, by `_subdivide_once`.

    After one pass the vertex, its n edge points and its n face points form a
    ring of n quads, and since every crease weight is passed on unchanged,
    each later pass maps that ring onto the next by the same
    (2n+1) x (2n+1) matrix S. The limit is l . X1, with X1 the ring after the
    first pass and l the left eigenvector of S for eigenvalue 1, scaled to
    sum 1. `step` below restates the interior vertex and edge rules of
    `_subdivide_once`; it runs on real positions for the first pass and on
    unit coefficient rows to build S.

    Returns None when the fan is not a closed manifold ring.
    """
    n = len(fan)
    slot = {}
    for j, he in enumerate(fan):
        if he.edge is None or he.face is None or he.face.index not in face_points:
            return None
        slot[he.edge.index] = j
    # Face j holds spoke j and the spoke that runs into he.prev.
    ring = []
    for j, he in enumerate(fan):
        k = slot.get(he.prev.edge.index) if he.prev is not None and he.prev.edge is not None else None
        if k is None or k == j:
            return None
        ring.append(k)
    if len(slot) != n or sorted(ring) != list(range(n)):
        return None
    # Spoke j borders face j and face other[j].
    other = [0] * n
    for f, k in enumerate(ring):
        other[k] = f

    raw = [float(he.edge.crease_weight) for he in fan]
    w = np.clip(raw, 0.0, 1.0)[:, None]
    max_w = float(w.max())
    sharp = [j for j in range(n) if raw[j] > 0]

    def step(P, mids, fps):
        vp = (fps.mean(axis=0) + 2.0 * mids.mean(axis=0) + (n - 3) * P) / n
        if max_w > 0:
            if len(sharp) == 2:
                crease_vp = (mids[sharp].mean(axis=0) + P) / 2.0
            elif len(sharp) == 1:
                crease_vp = vp          # dart: smooth rule
            else:
                crease_vp = P           # corner
            vp = vp * (1.0 - max_w) + crease_vp * max_w
        smooth_ep = (2.0 * mids + fps + fps[other]) / 4.0
        return vp, smooth_ep * (1.0 - w) + mids * w

    fps0 = np.array([face_points[he.face.index] for he in fan])
    mids0 = np.array([edge_midpoints[he.edge.index] for he in fan])
    p1, e1 = step(P, mids0, fps0)
    ring1 = np.vstack([p1, e1, fps0])

    key = (tuple(raw), tuple(ring))
    l = cache.get(key)
    if l is None:
        eye = np.eye(2 * n + 1)
        Pb, Eb, Cb = eye[0], eye[1:n + 1], eye[n + 1:]
        fps = (Pb + Eb + Cb + Eb[ring]) / 4.0       # quad j: P, E_j, C_j, E_ring[j]
        pn, en = step(Pb, (Pb + Eb) / 2.0, fps)
        S = np.vstack([pn, en, fps])
        A = np.vstack([S.T - eye, np.ones((1, 2 * n + 1))])
        b = np.zeros(2 * n + 2)
        b[-1] = 1.0
        l = np.linalg.lstsq(A, b, rcond=None)[0]
        cache[key] = l
    return l @ ring1


def _normals_from_positions(mesh: HalfEdgeMesh, positions: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals of `mesh`'s topology evaluated at `positions`.

    Computed without touching the mesh's own cached normals, so calling it does
    not disturb whatever the caller had stored there.
    """
    normals = np.zeros((len(mesh.vertices), 3))
    # sum of the squared edge lengths of the faces around each vertex
    scale = np.zeros(len(mesh.vertices))

    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        if len(fv) < 3:
            continue
        # Newell's method: robust for non-planar and non-convex polygons, and
        # its magnitude is twice the projected area, giving the area weighting.
        # Centring first keeps a small face far from the origin from drowning
        # in the rounding of P + Q.
        Pp = positions[fv]
        Pp = Pp - Pp.mean(axis=0)
        Q = np.roll(Pp, -1, axis=0)
        d = Pp - Q
        s = Pp + Q
        fn = np.array([
            np.dot(d[:, 1], s[:, 2]),
            np.dot(d[:, 2], s[:, 0]),
            np.dot(d[:, 0], s[:, 1]),
        ])
        sq = np.einsum('ij,ij->', d, d)
        for idx in fv:
            normals[idx] += fn
            scale[idx] += sq

    # Degenerate means numerically zero area, judged against the squared edge
    # lengths (an absolute threshold left the normals of a cage below ~1e-6
    # unnormalised). Those vertices keep the zero vector.
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12 * scale
    normals[good] /= lengths[good][:, None]
    return normals

def identify_regular_regions(mesh: HalfEdgeMesh) -> tuple[list[Vertex], list[Vertex]]:
    """
    Identify regular and irregular vertices in a quad mesh.
    Regular vertices have valence 4 (or 3 for boundary) and are surrounded by quads.
    Treats regular regions as bicubic B-spline patches conceptually.
    """
    regular = []
    irregular = []
    
    for v in mesh.vertices:
        is_reg = True
        faces = mesh.get_vertex_faces(v)
        if not faces:
            # No faces is not "surrounded by quads". Without this an isolated
            # vertex passed as regular: it counts as boundary, valence 0 <= 3.
            irregular.append(v)
            continue

        for f in faces:
            if len(mesh.get_face_vertices(f)) != 4:
                is_reg = False
                break
                
        if not is_reg:
            irregular.append(v)
            continue
            
        valence = len(mesh.get_vertex_neighbors(v))
        if mesh.is_boundary_vertex(v):
            is_reg = (valence <= 3)
        else:
            is_reg = (valence == 4)
            
        if is_reg:
            regular.append(v)
        else:
            irregular.append(v)
            
    return regular, irregular

