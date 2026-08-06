import numpy as np
from typing import Tuple, Optional, Union

class MeshSOA:
    """
    Structure of Arrays (SoA) data structure for triangle meshes.
    Designed for fast, vectorized operations on large meshes (>1M faces).
    """
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize from vertices and faces.
        
        Args:
            vertices: (V, 3) float64 array of vertex positions
            faces: (F, 3) int64 array of face vertex indices
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        
        if self.faces.shape[1] != 3:
            raise ValueError("MeshSOA currently only supports triangle meshes (F, 3)")
            
        self.V = len(self.vertices)
        self.F = len(self.faces)
        self.H = 3 * self.F
        
        self._build_topology()
        
        # Caches for adjacency
        self._vf_offsets: Optional[np.ndarray] = None
        self._vf_faces: Optional[np.ndarray] = None
        self._vv_offsets: Optional[np.ndarray] = None
        self._vv_vertices: Optional[np.ndarray] = None
        
    @classmethod
    def from_halfedge_mesh(cls, mesh) -> 'MeshSOA':
        """
        Creates a MeshSOA from a HalfEdgeMesh. 
        Triangulates polygons using simple fan triangulation if necessary.
        """
        vertices = np.array([v.position for v in mesh.vertices], dtype=np.float64)
        
        faces_list = []
        for f in mesh.faces:
            v_idx = [v.index for v in mesh.get_face_vertices(f)]
            if len(v_idx) == 3:
                faces_list.append(v_idx)
            elif len(v_idx) > 3:
                # Basic fan triangulation
                v0 = v_idx[0]
                for i in range(1, len(v_idx) - 1):
                    faces_list.append([v0, v_idx[i], v_idx[i+1]])
                    
        faces = np.array(faces_list, dtype=np.int64) if faces_list else np.empty((0, 3), dtype=np.int64)
        return cls(vertices, faces)
        
    def to_halfedge_mesh(self):
        """
        Converts the SoA structure back to a HalfEdgeMesh.
        """
        from src.core.halfedge_mesh import HalfEdgeMesh
        mesh = HalfEdgeMesh()
        for v in self.vertices:
            mesh.add_vertex(v.tolist())
        for f in self.faces:
            mesh.add_face(f.tolist())
        return mesh

    def _build_topology(self):
        """Builds the vectorized half-edge topology arrays."""
        # v0 and v1 are the source and target vertices for each half-edge
        v0 = self.faces.ravel()
        v1 = self.faces[:, [1, 2, 0]].ravel()
        
        # Fast unique edge identification using 64-bit integer packing
        # Note: Assumes vertex indices fit in 32 bits (< 4.2 billion vertices)
        v_min = np.minimum(v0, v1).astype(np.uint64)
        v_max = np.maximum(v0, v1).astype(np.uint64)
        edges_64 = np.bitwise_or(np.left_shift(v_min, 32), v_max)
        
        unique_edges_64, edge_indices = np.unique(edges_64, return_inverse=True)
        self.E = len(unique_edges_64)
        
        # Half-edge to edge mapping
        self.he_edge = edge_indices
        
        # Find half-edge twins
        self.he_twin = np.full(self.H, -1, dtype=np.int64)
        order = np.argsort(edge_indices)
        sorted_edges = edge_indices[order]
        
        is_duplicate = sorted_edges[:-1] == sorted_edges[1:]
        dup_indices = np.where(is_duplicate)[0]
        
        h1 = order[dup_indices]
        h2 = order[dup_indices + 1]
        
        self.he_twin[h1] = h2
        self.he_twin[h2] = h1
        
        # Half-edge to face mapping
        self.he_face = np.repeat(np.arange(self.F, dtype=np.int64), 3)
        
        # Next and prev half-edges
        f_idx = np.arange(self.F, dtype=np.int64) * 3
        self.he_next = np.empty(self.H, dtype=np.int64)
        self.he_next[f_idx] = f_idx + 1
        self.he_next[f_idx + 1] = f_idx + 2
        self.he_next[f_idx + 2] = f_idx
        
        self.he_prev = np.empty(self.H, dtype=np.int64)
        self.he_prev[f_idx] = f_idx + 2
        self.he_prev[f_idx + 1] = f_idx
        self.he_prev[f_idx + 2] = f_idx + 1
        
        # Half-edge to vertex (points to this vertex)
        self.he_vertex = v1
        
        # Vertex to half-edge (one outgoing half-edge per vertex)
        self.vertex_he = np.full(self.V, -1, dtype=np.int64)
        self.vertex_he[v0] = np.arange(self.H, dtype=np.int64)
        
        # Face to half-edge
        self.face_he = f_idx
        
        # Edge to half-edge (stores one arbitrary half-edge per edge)
        self.edge_he = np.full(self.E, -1, dtype=np.int64)
        self.edge_he[edge_indices] = np.arange(self.H, dtype=np.int64)

    def get_face_face_adjacency(self) -> np.ndarray:
        """
        Returns an (F, 3) array where each row contains the adjacent face indices.
        -1 indicates a boundary edge with no adjacent face.
        """
        ff = np.full(self.H, -1, dtype=np.int64)
        has_twin = self.he_twin != -1
        ff[has_twin] = self.he_face[self.he_twin[has_twin]]
        return ff.reshape(self.F, 3)

    def _build_vertex_face_adjacency(self):
        v_indices = self.faces.ravel()
        f_indices = np.repeat(np.arange(self.F, dtype=np.int64), 3)
        
        order = np.argsort(v_indices)
        self._vf_faces = f_indices[order]
        
        degrees = np.bincount(v_indices, minlength=self.V)
        self._vf_offsets = np.empty(self.V + 1, dtype=np.int64)
        self._vf_offsets[0] = 0
        np.cumsum(degrees, out=self._vf_offsets[1:])

    def get_vertex_faces(self, v_index: Union[int, np.ndarray]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get adjacent faces for one or more vertices.
        
        Args:
            v_index: int or array of ints
            
        Returns:
            If v_index is an int, returns a 1D array of face indices.
            If v_index is an array, returns (offsets, faces) for the queried vertices,
            where the faces for the i-th vertex are located in faces[offsets[i]:offsets[i+1]].
        """
        if self._vf_offsets is None:
            self._build_vertex_face_adjacency()
            
        if np.isscalar(v_index):
            start = self._vf_offsets[v_index]
            end = self._vf_offsets[v_index + 1]
            return self._vf_faces[start:end]
        else:
            v_index = np.asarray(v_index, dtype=np.int64)
            starts = self._vf_offsets[v_index]
            ends = self._vf_offsets[v_index + 1]
            lengths = ends - starts
            
            total_len = lengths.sum()
            out_faces = np.empty(total_len, dtype=np.int64)
            out_offsets = np.empty(len(v_index) + 1, dtype=np.int64)
            out_offsets[0] = 0
            np.cumsum(lengths, out=out_offsets[1:])
            
            if total_len > 0:
                bases = np.repeat(starts - out_offsets[:-1], lengths)
                ranges = np.arange(total_len)
                read_indices = bases + ranges
                out_faces = self._vf_faces[read_indices]
                
            return out_offsets, out_faces

    def _build_vertex_vertex_adjacency(self):
        # We use unique edges to ensure correct handling of boundaries 
        # and to avoid duplicate neighbors.
        edges = np.empty((self.E, 2), dtype=np.int64)
        he = self.edge_he
        edges[:, 0] = self.faces.ravel()[he]  # start vertex
        edges[:, 1] = self.he_vertex[he]      # end vertex
        
        # Create symmetric directed edges for full neighborhood
        v0 = np.concatenate([edges[:, 0], edges[:, 1]])
        v1 = np.concatenate([edges[:, 1], edges[:, 0]])
        
        order = np.argsort(v0)
        self._vv_vertices = v1[order]
        
        degrees = np.bincount(v0, minlength=self.V)
        self._vv_offsets = np.empty(self.V + 1, dtype=np.int64)
        self._vv_offsets[0] = 0
        np.cumsum(degrees, out=self._vv_offsets[1:])

    def get_vertex_neighbors(self, v_index: Union[int, np.ndarray]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get adjacent vertices for one or more vertices.
        
        Args:
            v_index: int or array of ints
            
        Returns:
            If v_index is an int, returns a 1D array of vertex indices.
            If v_index is an array, returns (offsets, vertices) for the queried vertices,
            where the neighbors for the i-th vertex are in vertices[offsets[i]:offsets[i+1]].
        """
        if self._vv_offsets is None:
            self._build_vertex_vertex_adjacency()
            
        if np.isscalar(v_index):
            start = self._vv_offsets[v_index]
            end = self._vv_offsets[v_index + 1]
            return self._vv_vertices[start:end]
        else:
            v_index = np.asarray(v_index, dtype=np.int64)
            starts = self._vv_offsets[v_index]
            ends = self._vv_offsets[v_index + 1]
            lengths = ends - starts
            
            total_len = lengths.sum()
            out_neighbors = np.empty(total_len, dtype=np.int64)
            out_offsets = np.empty(len(v_index) + 1, dtype=np.int64)
            out_offsets[0] = 0
            np.cumsum(lengths, out=out_offsets[1:])
            
            if total_len > 0:
                bases = np.repeat(starts - out_offsets[:-1], lengths)
                ranges = np.arange(total_len)
                read_indices = bases + ranges
                out_neighbors = self._vv_vertices[read_indices]
                
            return out_offsets, out_neighbors

    def compute_face_normals(self) -> np.ndarray:
        """
        Computes and returns face normals as a (F, 3) array.
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Avoid division by zero for degenerate triangles
        norms = np.where(norms < 1e-8, 1.0, norms)
        return normals / norms

    def compute_vertex_normals(self) -> np.ndarray:
        """
        Computes and returns area-weighted vertex normals as a (V, 3) array.
        """
        vertex_normals = np.zeros((self.V, 3), dtype=np.float64)
        
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # Area-weighted normals are proportional to unnormalized cross products
        cross = np.cross(v1 - v0, v2 - v0)
        
        np.add.at(vertex_normals, self.faces[:, 0], cross)
        np.add.at(vertex_normals, self.faces[:, 1], cross)
        np.add.at(vertex_normals, self.faces[:, 2], cross)
        
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return vertex_normals / norms
        
    def is_boundary_vertex(self) -> np.ndarray:
        """
        Returns a boolean array (V,) indicating if each vertex is on a boundary.
        """
        is_boundary_he = self.he_twin == -1
        boundary_vertices = np.zeros(self.V, dtype=bool)
        
        # A half-edge without a twin forms a boundary.
        # Both its start and end vertices are boundary vertices.
        v0 = self.faces.ravel()[is_boundary_he]
        v1 = self.he_vertex[is_boundary_he]
        
        boundary_vertices[v0] = True
        boundary_vertices[v1] = True
        return boundary_vertices

    def is_boundary_edge(self) -> np.ndarray:
        """
        Returns a boolean array (E,) indicating if each edge is on a boundary.
        """
        # An edge is a boundary edge if the half-edge recorded for it has no twin.
        # Note: Since the half-edge is arbitrarily one of the edge's half-edges,
        # it is either on the boundary itself, or its twin is missing.
        # Wait, if an edge has only one half-edge, its recorded half-edge MUST have no twin!
        he = self.edge_he
        return self.he_twin[he] == -1
