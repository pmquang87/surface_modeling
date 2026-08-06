import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh

def extrude_faces(mesh: HalfEdgeMesh, face_indices: list[int], distance: float = 0.1, direction: np.ndarray = None) -> HalfEdgeMesh:
    """Extrude selected faces along their normals or a given direction."""
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    mesh.compute_face_normals()
    
    for f_idx in face_indices:
        if f_idx >= len(mesh.faces): continue
        f = mesh.faces[f_idx]
        f_verts = [v.index for v in mesh.get_face_vertices(f)]
        
        extrude_dir = direction if direction is not None else f.normal
        extrude_dir = np.array(extrude_dir, dtype=np.float64)
        extrude_dir /= np.linalg.norm(extrude_dir)
        
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
    """Extrude edges, creating new faces."""
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    mesh.compute_vertex_normals()
    
    for e_idx in edge_indices:
        if e_idx >= len(mesh.edges): continue
        e = mesh.edges[e_idx]
        v1_idx = e.half_edge.prev.vertex.index
        v2_idx = e.half_edge.vertex.index
        
        v1_norm = mesh.vertices[v1_idx].normal
        v2_norm = mesh.vertices[v2_idx].normal
        
        dir1 = direction if direction is not None else v1_norm
        dir2 = direction if direction is not None else v2_norm
        
        nv1 = verts[v1_idx] + np.array(dir1) * distance
        nv2 = verts[v2_idx] + np.array(dir2) * distance
        
        verts.append(nv1)
        nv1_idx = len(verts) - 1
        verts.append(nv2)
        nv2_idx = len(verts) - 1
        
        faces.append([v1_idx, v2_idx, nv2_idx, nv1_idx])
        
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
    
    edge_to_new_vert = {}
    for e in ring_edges:
        v1 = e.half_edge.prev.vertex.position
        v2 = e.half_edge.vertex.position
        new_pos = v1 * (1.0 - position) + v2 * position
        verts.append(new_pos)
        edge_to_new_vert[e.index] = len(verts) - 1
        
    for f in mesh.faces:
        f_edges = []
        he = f.half_edge
        start_he = he
        while True:
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
    """Delete both face groups and connect their boundary edges with new quad faces."""
    import numpy as np
    verts = [v.position.copy() for v in mesh.vertices]
    faces = []
    
    to_delete = set(face_indices_a + face_indices_b)
    
    for f in mesh.faces:
        if f.index not in to_delete:
            faces.append([v.index for v in mesh.get_face_vertices(f)])
            
    def get_boundary_loop(group_indices):
        boundary_edges = []
        for f_idx in group_indices:
            f = mesh.faces[f_idx]
            he = f.half_edge
            start_he = he
            while True:
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
                if he.prev.vertex == last_v:
                    loop.append(boundary_edges.pop(i))
                    found = True
                    break
            if not found: break 
        return [he.prev.vertex.index for he in loop]

    loop_a = get_boundary_loop(set(face_indices_a))
    loop_b = get_boundary_loop(set(face_indices_b))
    
    if len(loop_a) == len(loop_b) and len(loop_a) > 2:
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
    """Mirror the mesh across the given axis plane, merging vertices within tolerance."""
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    mirror_verts = []
    for p in verts:
        mp = p.copy()
        if axis == 'x': mp[0] *= -1
        elif axis == 'y': mp[1] *= -1
        elif axis == 'z': mp[2] *= -1
        mirror_verts.append(mp)
        
    mirror_faces = [f[::-1] for f in faces]
    
    all_verts = list(verts)
    new_faces = list(faces)
    
    n_orig = len(verts)
    remap = {}
    
    for i, mv in enumerate(mirror_verts):
        dists = np.linalg.norm(np.array(verts) - mv, axis=1)
        if len(dists) > 0:
            min_idx = np.argmin(dists)
            if dists[min_idx] < merge_distance:
                remap[i + n_orig] = min_idx
                continue
                
        remap[i + n_orig] = len(all_verts)
        all_verts.append(mv)
            
    for f in mirror_faces:
        new_f = [remap[idx + n_orig] for idx in f]
        new_faces.append(new_f)
        
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
        if e.half_edge and e.half_edge.face:
            face_indices.add(e.half_edge.face.index)
        if e.half_edge.twin and e.half_edge.twin.face:
            face_indices.add(e.half_edge.twin.face.index)
            
    return inset_faces(mesh, list(face_indices), inset_amount=distance)


def knife_cut(mesh: HalfEdgeMesh, face_idx: int, p1: np.ndarray, p2: np.ndarray) -> HalfEdgeMesh:
    """Subdivide a face along a line segment (simplified split)."""
    if face_idx >= len(mesh.faces):
        return mesh.copy()
        
    verts = [v.position.copy() for v in mesh.vertices]
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    
    f_verts = faces[face_idx]
    
    # Simple split of a quad into two triangles
    if len(f_verts) == 4:
        faces[face_idx] = [f_verts[0], f_verts[1], f_verts[2]]
        faces.append([f_verts[0], f_verts[2], f_verts[3]])
        
    return HalfEdgeMesh.from_arrays(verts, faces)
