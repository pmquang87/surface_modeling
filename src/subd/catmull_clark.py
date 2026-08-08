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

    Positions are exact. For an interior vertex of valence n the limit point is

        limit = ((n - 3) * P + 4 * Rbar + 4 * Fbar) / (n + 5)

    with P the vertex, Rbar the mean of the incident edge midpoints and Fbar the
    mean of the incident face centroids. That is the eigenvector of the local
    subdivision matrix for eigenvalue 1, derived from the very rules applied in
    `_subdivide_once`; on a regular (valence 4, all-quad) neighbourhood it
    reproduces the exact bicubic B-spline limit (16*P + 4*sum(neighbours) +
    sum(diagonals)) / 36. The weights sum to n + 5, so the map is affine
    invariant.

    Boundary vertices are left where they are: the boundary limit rule depends on
    the crease/boundary curve rather than the interior mask, and this module does
    not model open boundaries.

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

        R_bar = np.mean([edge_midpoints[e.index] for e in inc_edges], axis=0)
        F_bar = np.mean([face_points[f.index] for f in faces], axis=0)
        limit_positions[v.index] = ((n - 3) * P + 4.0 * R_bar + 4.0 * F_bar) / (n + 5.0)

    limit_normals = _normals_from_positions(mesh, limit_positions)
    return limit_positions, limit_normals


def _normals_from_positions(mesh: HalfEdgeMesh, positions: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals of `mesh`'s topology evaluated at `positions`.

    Computed without touching the mesh's own cached normals, so calling it does
    not disturb whatever the caller had stored there.
    """
    normals = np.zeros((len(mesh.vertices), 3))

    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        if len(fv) < 3:
            continue
        # Newell's method: robust for non-planar and non-convex polygons, and
        # its magnitude is twice the projected area, giving the area weighting.
        Pp = positions[fv]
        Q = np.roll(Pp, -1, axis=0)
        d = Pp - Q
        s = Pp + Q
        fn = np.array([
            np.dot(d[:, 1], s[:, 2]),
            np.dot(d[:, 2], s[:, 0]),
            np.dot(d[:, 0], s[:, 1]),
        ])
        for idx in fv:
            normals[idx] += fn

    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
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

