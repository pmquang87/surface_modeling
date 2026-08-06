"""
T-Splines Core Data Structures
==============================
Handles partial edge splits (T-junctions) and extraordinary vertices
without forcing global subdivision.
"""

from typing import List, Optional, Tuple, Dict
import math

class TVertex:
    """
    Represents a vertex in a T-Mesh control grid.
    Can be a regular vertex, a T-junction, or an extraordinary vertex.
    """
    def __init__(self, id: int, x: float, y: float, z: float, w: float = 1.0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        
        # Connections to incident edges in up to 4 topological directions
        # For non-quad meshes, this might need to be a list of half-edges, 
        # but for standard T-Splines (which are essentially quad-based with T-junctions):
        self.edges: Dict[str, Optional['TEdge']] = {
            'left': None,
            'right': None,
            'up': None,
            'down': None
        }
        
    @property
    def valence(self) -> int:
        """Number of incident edges."""
        return sum(1 for e in self.edges.values() if e is not None)
    
    def is_t_junction(self) -> bool:
        """A T-junction typically has valence 3 in a quad-based T-mesh."""
        return self.valence == 3
        
    def is_extraordinary(self) -> bool:
        """An extraordinary vertex has valence != 4 (excluding T-junctions on boundaries)."""
        # True extraordinary vertices are interior vertices with valence != 4
        return self.valence != 4 and not self.is_t_junction()


class TEdge:
    """
    An edge in the T-Mesh connecting two TVertices.
    Contains a knot interval used to infer local knot vectors.
    """
    def __init__(self, id: int, v1: TVertex, v2: TVertex, knot_interval: float = 1.0):
        self.id = id
        self.v1 = v1
        self.v2 = v2
        self.knot_interval = knot_interval
        
    def get_other_vertex(self, v: TVertex) -> Optional[TVertex]:
        if self.v1 == v:
            return self.v2
        if self.v2 == v:
            return self.v1
        return None


class TFace:
    """
    A face in the T-Mesh, typically a topological quad but can have 
    more than 4 vertices/edges due to T-junctions on its boundary.
    """
    def __init__(self, id: int, edges: List[TEdge]):
        self.id = id
        self.edges = edges


class TMesh:
    """
    The T-Mesh structure containing vertices, edges, and faces.
    Supports local refinement by splitting edges (creating T-junctions).
    """
    def __init__(self):
        self.vertices: Dict[int, TVertex] = {}
        self.edges: Dict[int, TEdge] = {}
        self.faces: Dict[int, TFace] = {}
        
        self._next_v_id = 0
        self._next_e_id = 0
        self._next_f_id = 0
        
    def add_vertex(self, x: float, y: float, z: float, w: float = 1.0) -> TVertex:
        v = TVertex(self._next_v_id, x, y, z, w)
        self.vertices[self._next_v_id] = v
        self._next_v_id += 1
        return v
        
    def add_edge(self, v1_id: int, v2_id: int, dir_v1: str, dir_v2: str, knot_interval: float = 1.0) -> TEdge:
        """
        Add an edge between two vertices.
        dir_v1: the direction of the edge relative to v1 (e.g., 'right')
        dir_v2: the direction of the edge relative to v2 (e.g., 'left')
        """
        if v1_id not in self.vertices or v2_id not in self.vertices:
            raise ValueError("Vertices must be added to the mesh first.")
            
        v1 = self.vertices[v1_id]
        v2 = self.vertices[v2_id]
        
        e = TEdge(self._next_e_id, v1, v2, knot_interval)
        self.edges[self._next_e_id] = e
        self._next_e_id += 1
        
        v1.edges[dir_v1] = e
        v2.edges[dir_v2] = e
        return e

    def extract_local_knot_vector(self, vertex_id: int, direction: str, degree: int = 3) -> List[float]:
        """
        Extract the local knot vector for a given vertex in a specific topological direction.
        direction: 's' (horizontal: left/right) or 't' (vertical: up/down)
        Returns a list of knots. For a cubic T-spline (degree=3), this returns 5 knots.
        """
        v = self.vertices[vertex_id]
        
        # We need (degree + 2) knots. For degree 3, we need 5 knots.
        # k_{-2}, k_{-1}, k_0, k_1, k_2
        # k_0 is the knot at the vertex itself, typically 0 for local evaluation.
        
        dir_neg, dir_pos = ('left', 'right') if direction == 's' else ('down', 'up')
        
        knots_neg = []
        curr_v = v
        for _ in range(degree // 2 + 1):
            edge = curr_v.edges.get(dir_neg)
            if edge is None:
                # Boundary condition: repeat the last knot interval (0)
                knots_neg.append(0.0)
            else:
                knots_neg.append(edge.knot_interval)
                curr_v = edge.get_other_vertex(curr_v)
                
        knots_pos = []
        curr_v = v
        for _ in range(degree // 2 + 1):
            edge = curr_v.edges.get(dir_pos)
            if edge is None:
                knots_pos.append(0.0)
            else:
                knots_pos.append(edge.knot_interval)
                curr_v = edge.get_other_vertex(curr_v)
                
        # Construct the final knot vector by accumulating intervals
        knot_vector = [0.0] * (degree + 2)
        center_idx = (degree + 2) // 2
        
        current_k = 0.0
        for i in range(len(knots_pos)):
            current_k += knots_pos[i]
            if center_idx + i + 1 < len(knot_vector):
                knot_vector[center_idx + i + 1] = current_k
                
        current_k = 0.0
        for i in range(len(knots_neg)):
            current_k -= knots_neg[i]
            if center_idx - i - 1 >= 0:
                knot_vector[center_idx - i - 1] = current_k
                
        return knot_vector

    def split_edge(self, edge_id: int, alpha: float = 0.5) -> TVertex:
        """
        Perform a local refinement by splitting an edge.
        This introduces a T-junction if not propagated.
        alpha: ratio of the split (0 to 1).
        """
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} does not exist.")
            
        edge = self.edges[edge_id]
        v1 = edge.v1
        v2 = edge.v2
        
        # Calculate new vertex position (linear interpolation for control net)
        nx = v1.x * (1 - alpha) + v2.x * alpha
        ny = v1.y * (1 - alpha) + v2.y * alpha
        nz = v1.z * (1 - alpha) + v2.z * alpha
        nw = v1.w * (1 - alpha) + v2.w * alpha
        
        new_v = self.add_vertex(nx, ny, nz, nw)
        
        # Find which direction edge was relative to v1 and v2
        dir_v1 = next((d for d, e in v1.edges.items() if e == edge), None)
        dir_v2 = next((d for d, e in v2.edges.items() if e == edge), None)
        
        # Update knot intervals
        old_interval = edge.knot_interval
        int1 = old_interval * alpha
        int2 = old_interval * (1 - alpha)
        
        # Remove old edge
        del self.edges[edge_id]
        v1.edges[dir_v1] = None
        v2.edges[dir_v2] = None
        
        # Add two new edges
        self.add_edge(v1.id, new_v.id, dir_v1, dir_v2, int1)
        self.add_edge(new_v.id, v2.id, dir_v1, dir_v2, int2)
        
        return new_v
