import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh, Vertex, HalfEdge, Edge, Face

def subdivide(mesh: HalfEdgeMesh, levels: int = 1) -> HalfEdgeMesh:
    """
    Subdivide a mesh using the Catmull-Clark algorithm.
    """
    if levels <= 0:
        return mesh.copy()
    
    current_mesh = mesh
    for _ in range(levels):
        current_mesh = _subdivide_once(current_mesh)
    return current_mesh

def _subdivide_once(mesh: HalfEdgeMesh) -> HalfEdgeMesh:
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
        if f2 is None:
            # Boundary edge
            ep = midpoint
        else:
            fp1 = face_points[f1.index]
            fp2 = face_points[f2.index]
            smooth_ep = (v_src + v_tgt + fp1 + fp2) / 4.0
            
            w = np.clip(e.crease_weight, 0.0, 1.0)
            ep = smooth_ep * (1.0 - w) + midpoint * w
            
        edge_points[e.index] = ep
        
    # 3. Vertex points
    vertex_points = {}
    for v in mesh.vertices:
        P = v.position
        faces = mesh.get_vertex_faces(v)
        
        inc_edges = []
        for e in mesh.edges:
            if e.half_edge.vertex == v or e.half_edge.prev.vertex == v:
                inc_edges.append(e)
                
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
        while True:
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
    Evaluate the exact limit surface positions and normals for a Catmull-Clark mesh.
    """
    limit_positions = np.zeros((len(mesh.vertices), 3))
    limit_normals = np.zeros((len(mesh.vertices), 3))
    
    mesh.compute_vertex_normals()
    
    face_points = {}
    for f in mesh.faces:
        verts = mesh.get_face_vertices(f)
        if verts:
            face_points[f.index] = np.mean([v.position for v in verts], axis=0)
        
    edge_midpoints = {}
    for e in mesh.edges:
        v_tgt = e.half_edge.vertex.position
        v_src = e.half_edge.prev.vertex.position
        edge_midpoints[e.index] = (v_src + v_tgt) / 2.0
        
    for v in mesh.vertices:
        if mesh.is_boundary_vertex(v):
            limit_positions[v.index] = v.position
            limit_normals[v.index] = v.normal
            continue
            
        faces = mesh.get_vertex_faces(v)
        inc_edges = []
        for e in mesh.edges:
            if e.half_edge.vertex == v or e.half_edge.prev.vertex == v:
                inc_edges.append(e)
                
        n = len(inc_edges)
        P = v.position
        
        if n > 0:
            sum_R = np.sum([edge_midpoints[e.index] for e in inc_edges], axis=0)
            sum_F = np.sum([face_points[f.index] for f in faces], axis=0) if faces else np.zeros(3)
            limit_pos = ((n - 2) / n) * P + (1 / (n * n)) * sum_R + (1 / (n * n)) * sum_F
        else:
            limit_pos = P
            
        limit_positions[v.index] = limit_pos
        limit_normals[v.index] = v.normal

    return limit_positions, limit_normals
