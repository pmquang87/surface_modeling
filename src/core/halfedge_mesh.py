import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

class Vertex:
    __slots__ = ['position', 'normal', 'half_edge', 'index', 'selected']
    def __init__(self, position: np.ndarray, index: int):
        self.position: np.ndarray = np.array(position, dtype=np.float64)
        self.normal: np.ndarray = np.zeros(3, dtype=np.float64)
        self.half_edge: Optional['HalfEdge'] = None
        self.index: int = index
        self.selected: bool = False

class Face:
    __slots__ = ['half_edge', 'normal', 'index', 'material_id', 'selected']
    def __init__(self, index: int):
        self.half_edge: Optional['HalfEdge'] = None
        self.normal: np.ndarray = np.zeros(3, dtype=np.float64)
        self.index: int = index
        self.material_id: int = 0
        self.selected: bool = False

class Edge:
    __slots__ = ['half_edge', 'crease_weight', 'index', 'selected']
    def __init__(self, index: int):
        self.half_edge: Optional['HalfEdge'] = None
        self.crease_weight: float = 0.0
        self.index: int = index
        self.selected: bool = False

class HalfEdge:
    __slots__ = ['vertex', 'face', 'next', 'prev', 'twin', 'edge', 'index']
    def __init__(self, index: int):
        self.vertex: Optional[Vertex] = None
        self.face: Optional[Face] = None
        self.next: Optional['HalfEdge'] = None
        self.prev: Optional['HalfEdge'] = None
        self.twin: Optional['HalfEdge'] = None
        self.edge: Optional[Edge] = None
        self.index: int = index


class HalfEdgeMesh:
    """
    Half-Edge data structure for manifold meshes, supporting arbitrary polygons (mixed topology).
    """
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.half_edges: List[HalfEdge] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []
        self._he_dict: Dict[Tuple[int, int], HalfEdge] = {}

    def add_vertex(self, position: Any) -> Vertex:
        v = Vertex(position, len(self.vertices))
        self.vertices.append(v)
        return v

    def add_face(self, vertex_indices: List[int]) -> Face:
        face = Face(len(self.faces))
        self.faces.append(face)
        
        n = len(vertex_indices)
        face_hes = []
        for i in range(n):
            he = HalfEdge(len(self.half_edges))
            self.half_edges.append(he)
            face_hes.append(he)
            
        for i in range(n):
            v1_idx = vertex_indices[i]
            v2_idx = vertex_indices[(i+1)%n]
            
            he = face_hes[i]
            he.vertex = self.vertices[v2_idx]
            he.face = face
            he.next = face_hes[(i+1)%n]
            he.prev = face_hes[(i-1)%n]
            
            if self.vertices[v1_idx].half_edge is None:
                self.vertices[v1_idx].half_edge = he
                
            if face.half_edge is None:
                face.half_edge = he
                
            self._he_dict[(v1_idx, v2_idx)] = he
            
            twin = self._he_dict.get((v2_idx, v1_idx))
            if twin is not None:
                he.twin = twin
                twin.twin = he
                he.edge = twin.edge
            else:
                edge = Edge(len(self.edges))
                edge.half_edge = he
                self.edges.append(edge)
                he.edge = edge
                
        return face

    def get_face_vertices(self, face: Face) -> List[Vertex]:
        if face.half_edge is None:
            return []
        vertices = []
        curr = face.half_edge
        while True:
            # curr.vertex is the vertex the halfedge points to.
            # to get vertices in order, we can collect prev.vertex
            vertices.append(curr.prev.vertex)
            curr = curr.next
            if curr == face.half_edge:
                break
        return vertices

    def get_vertex_faces(self, vertex: Vertex) -> List[Face]:
        if vertex.half_edge is None:
            return []
        faces = []
        curr = vertex.half_edge
        while True:
            if curr.face is not None:
                faces.append(curr.face)
            if curr.twin is None:
                break
            curr = curr.twin.next
            if curr == vertex.half_edge:
                break
        return faces

    def get_vertex_neighbors(self, vertex: Vertex) -> List[Vertex]:
        if vertex.half_edge is None:
            return []
        neighbors = []
        curr = vertex.half_edge
        while True:
            neighbors.append(curr.vertex)
            if curr.twin is None:
                break
            curr = curr.twin.next
            if curr == vertex.half_edge:
                break
        return neighbors

    def get_edge_faces(self, edge: Edge) -> Tuple[Face, Optional[Face]]:
        he = edge.half_edge
        f1 = he.face
        f2 = he.twin.face if he.twin else None
        return (f1, f2)

    def get_face_edges(self, face: Face) -> List[Edge]:
        if face.half_edge is None:
            return []
        edges = []
        curr = face.half_edge
        while True:
            edges.append(curr.edge)
            curr = curr.next
            if curr == face.half_edge:
                break
        return edges

    def vertex_valence(self, vertex: Vertex) -> int:
        return len(self.get_vertex_neighbors(vertex))

    def is_boundary_vertex(self, vertex: Vertex) -> bool:
        if vertex.half_edge is None:
            return True
        curr = vertex.half_edge
        while True:
            if curr.twin is None:
                return True
            curr = curr.twin.next
            if curr == vertex.half_edge:
                break
        return False

    def is_boundary_edge(self, edge: Edge) -> bool:
        return edge.half_edge.twin is None

    def get_edge_loop(self, start_edge: Edge) -> List[Edge]:
        # Simple loop traversal across quad topologies
        loop = [start_edge]
        # Implementation depends on specific Sub-D logic, basic traversal for quads
        for direction in [0, 1]:
            curr = start_edge.half_edge if direction == 0 else (start_edge.half_edge.twin if start_edge.half_edge.twin else None)
            if curr is None: continue
            
            while curr is not None:
                # Get opposite edge in quad
                face_edges = []
                temp = curr.next
                while temp != curr:
                    face_edges.append(temp.edge)
                    temp = temp.next
                if len(face_edges) == 3: # Quad face (3 remaining edges)
                    opp_edge = face_edges[1]
                    if opp_edge == start_edge: break
                    if direction == 0:
                        loop.append(opp_edge)
                    else:
                        loop.insert(0, opp_edge)
                        
                    # cross to next face
                    next_he = opp_edge.half_edge if opp_edge.half_edge.face != curr.face else opp_edge.half_edge.twin
                    curr = next_he
                else:
                    break
        return loop

    def get_edge_ring(self, start_edge: Edge) -> List[Edge]:
        # Walk along adjacent edges in quad structure
        return [start_edge] # simplified stub for edge ring

    def select_vertices(self, indices: List[int]):
        for v in self.vertices:
            v.selected = v.index in indices

    def select_edges(self, indices: List[int]):
        for e in self.edges:
            e.selected = e.index in indices

    def select_faces(self, indices: List[int]):
        for f in self.faces:
            f.selected = f.index in indices

    def expand_selection_by_angle(self, start_face_ids: List[int], max_angle_degrees: float) -> List[int]:
        if not self.faces:
            return []
            
        if np.allclose(self.faces[0].normal, np.zeros(3)):
            self.compute_face_normals()
            
        selected = set(start_face_ids)
        queue = list(start_face_ids)
        max_angle_rad = np.radians(max_angle_degrees)
        
        idx = 0
        while idx < len(queue):
            curr_id = queue[idx]
            idx += 1
            
            if curr_id < 0 or curr_id >= len(self.faces):
                continue
                
            curr_face = self.faces[curr_id]
            curr_normal = curr_face.normal
            
            he = curr_face.half_edge
            if he is None:
                continue
                
            curr_he = he
            while True:
                if curr_he.twin is not None and curr_he.twin.face is not None:
                    neighbor_face = curr_he.twin.face
                    neighbor_id = neighbor_face.index
                    
                    if neighbor_id not in selected:
                        neighbor_normal = neighbor_face.normal
                        dot = np.dot(curr_normal, neighbor_normal)
                        dot = np.clip(dot, -1.0, 1.0)
                        angle_rad = np.arccos(dot)
                        
                        if angle_rad <= max_angle_rad + 1e-6:
                            selected.add(neighbor_id)
                            queue.append(neighbor_id)
                            
                curr_he = curr_he.next
                if curr_he == he:
                    break
                    
        return list(selected)

    def to_arrays(self) -> Dict[str, Any]:
        verts = np.array([v.position for v in self.vertices], dtype=np.float64)
        faces = [ [v.index for v in self.get_face_vertices(f)] for f in self.faces ]
        return {'vertices': verts, 'faces': faces}

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, faces: List[List[int]]) -> 'HalfEdgeMesh':
        mesh = cls()
        for v in vertices:
            mesh.add_vertex(v)
        for f in faces:
            mesh.add_face(f)
        return mesh

    def to_trimesh(self) -> Any:
        import trimesh
        verts = np.array([v.position for v in self.vertices], dtype=np.float64)
        tri_faces = []
        for f in self.faces:
            fv = [v.index for v in self.get_face_vertices(f)]
            if len(fv) >= 3:
                for i in range(1, len(fv) - 1):
                    tri_faces.append([fv[0], fv[i], fv[i+1]])
        return trimesh.Trimesh(vertices=verts, faces=tri_faces, process=False)

    @classmethod
    def from_trimesh(cls, mesh: Any) -> 'HalfEdgeMesh':
        return cls.from_arrays(mesh.vertices, mesh.faces.tolist())

    def to_pyvista(self) -> Any:
        import pyvista as pv
        verts = np.array([v.position for v in self.vertices], dtype=np.float64)
        pv_faces = []
        for f in self.faces:
            fv = [v.index for v in self.get_face_vertices(f)]
            pv_faces.append(len(fv))
            pv_faces.extend(fv)
        return pv.PolyData(verts, np.array(pv_faces))

    def compute_face_normals(self):
        for f in self.faces:
            verts = [v.position for v in self.get_face_vertices(f)]
            if len(verts) >= 3:
                v0 = verts[0]
                v1 = verts[1]
                v2 = verts[-1]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                f.normal = n / norm if norm > 1e-8 else np.zeros(3)

    def compute_vertex_normals(self):
        self.compute_face_normals()
        for v in self.vertices:
            faces = self.get_vertex_faces(v)
            if not faces: continue
            n = np.sum([f.normal for f in faces], axis=0)
            norm = np.linalg.norm(n)
            v.normal = n / norm if norm > 1e-8 else np.zeros(3)

    def copy(self) -> 'HalfEdgeMesh':
        m = HalfEdgeMesh()
        m.vertices = [Vertex(v.position.copy(), v.index) for v in self.vertices]
        for i, v in enumerate(self.vertices):
            m.vertices[i].normal = v.normal.copy()
            m.vertices[i].selected = v.selected
            
        m.faces = [Face(f.index) for f in self.faces]
        for i, f in enumerate(self.faces):
            m.faces[i].normal = f.normal.copy()
            m.faces[i].material_id = f.material_id
            m.faces[i].selected = f.selected
            
        m.edges = [Edge(e.index) for e in self.edges]
        for i, e in enumerate(self.edges):
            m.edges[i].crease_weight = e.crease_weight
            m.edges[i].selected = e.selected
            
        m.half_edges = [HalfEdge(he.index) for he in self.half_edges]
        
        for i, he in enumerate(self.half_edges):
            nhe = m.half_edges[i]
            if he.vertex: nhe.vertex = m.vertices[he.vertex.index]
            if he.face: nhe.face = m.faces[he.face.index]
            if he.edge: nhe.edge = m.edges[he.edge.index]
            if he.next: nhe.next = m.half_edges[he.next.index]
            if he.prev: nhe.prev = m.half_edges[he.prev.index]
            if he.twin: nhe.twin = m.half_edges[he.twin.index]
            
        for i, v in enumerate(self.vertices):
            if v.half_edge: m.vertices[i].half_edge = m.half_edges[v.half_edge.index]
            
        for i, f in enumerate(self.faces):
            if f.half_edge: m.faces[i].half_edge = m.half_edges[f.half_edge.index]
            
        for i, e in enumerate(self.edges):
            if e.half_edge: m.edges[i].half_edge = m.half_edges[e.half_edge.index]
            
        m._he_dict = {(k[0], k[1]): m.half_edges[v.index] for k, v in self._he_dict.items()}
        return m

    def get_adjacent_faces(self, face_ids: List[int]) -> List[int]:
        adjacent = set()
        for f_id in face_ids:
            if f_id < 0 or f_id >= len(self.faces): continue
            face = self.faces[f_id]
            he = face.half_edge
            if not he: continue
            curr = he
            while True:
                if curr.twin and curr.twin.face:
                    adjacent.add(curr.twin.face.index)
                curr = curr.next
                if curr == he: break
        # Do not include the originally selected faces in the "adjacent" boundary
        return list(adjacent - set(face_ids))

    def get_adjacent_vertices(self, vertex_ids: List[int]) -> List[int]:
        adjacent = set()
        for v_id in vertex_ids:
            if v_id < 0 or v_id >= len(self.vertices): continue
            v = self.vertices[v_id]
            neighbors = self.get_vertex_neighbors(v)
            for n in neighbors:
                adjacent.add(n.index)
        return list(adjacent - set(vertex_ids))

    def get_adjacent_edges(self, edge_ids: List[int]) -> List[int]:
        adjacent = set()
        for e_id in edge_ids:
            if e_id < 0 or e_id >= len(self.edges): continue
            e = self.edges[e_id]
            he1 = e.half_edge
            he2 = e.half_edge.twin
            for he in [he1, he2]:
                if not he: continue
                if he.next and he.next.edge: adjacent.add(he.next.edge.index)
                if he.prev and he.prev.edge: adjacent.add(he.prev.edge.index)
        return list(adjacent - set(edge_ids))

    def get_connected_faces(self, start_face_ids: List[int]) -> List[int]:
        if not start_face_ids: return []
        visited = set(start_face_ids)
        queue = list(start_face_ids)
        
        idx = 0
        while idx < len(queue):
            curr_id = queue[idx]
            idx += 1
            if curr_id < 0 or curr_id >= len(self.faces): continue
            
            face = self.faces[curr_id]
            he = face.half_edge
            if not he: continue
            
            curr_he = he
            while True:
                if curr_he.twin and curr_he.twin.face:
                    n_id = curr_he.twin.face.index
                    if n_id not in visited:
                        visited.add(n_id)
                        queue.append(n_id)
                curr_he = curr_he.next
                if curr_he == he: break
                
        return list(visited)

    def get_connected_vertices(self, start_vertex_ids: List[int]) -> List[int]:
        if not start_vertex_ids: return []
        visited = set(start_vertex_ids)
        queue = list(start_vertex_ids)
        idx = 0
        while idx < len(queue):
            curr_id = queue[idx]
            idx += 1
            if curr_id < 0 or curr_id >= len(self.vertices): continue
            v = self.vertices[curr_id]
            for n in self.get_vertex_neighbors(v):
                if n.index not in visited:
                    visited.add(n.index)
                    queue.append(n.index)
        return list(visited)

    def get_connected_edges(self, start_edge_ids: List[int]) -> List[int]:
        if not start_edge_ids: return []
        visited = set(start_edge_ids)
        queue = list(start_edge_ids)
        idx = 0
        while idx < len(queue):
            curr_id = queue[idx]
            idx += 1
            if curr_id < 0 or curr_id >= len(self.edges): continue
            e = self.edges[curr_id]
            he1 = e.half_edge
            he2 = e.half_edge.twin
            for he in [he1, he2]:
                if not he: continue
                if he.next and he.next.edge and he.next.edge.index not in visited:
                    visited.add(he.next.edge.index)
                    queue.append(he.next.edge.index)
                if he.prev and he.prev.edge and he.prev.edge.index not in visited:
                    visited.add(he.prev.edge.index)
                    queue.append(he.prev.edge.index)
        return list(visited)
