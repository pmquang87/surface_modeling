import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.primitives import create_box, create_cylinder
from src.subd.editing import insert_edge_loop, bridge_faces

def test_insert_edge_loop():
    box = create_box()
    res = insert_edge_loop(box, 0, 0.5)
    print("insert_edge_loop faces:", len(res.faces), "expected:", 10)
    print("insert_edge_loop edges:", len(res.edges), "expected:", 20)
    print("insert_edge_loop verts:", len(res.vertices), "expected:", 12)
    assert len(res.faces) == 10
    assert len(res.edges) == 20
    assert len(res.vertices) == 12
    # Ensure it is still closed
    boundaries = [e for e in res.edges if res.is_boundary_edge(e)]
    print("insert_edge_loop boundary edges:", len(boundaries), "expected:", 0)
    assert len(boundaries) == 0

def test_bridge_faces():
    verts = []
    faces = []
    box1 = create_box()
    for v in box1.vertices:
        verts.append(v.position)
    for f in box1.faces:
        faces.append([v.index for v in box1.get_face_vertices(f)])
        
    offset = len(verts)
    box2 = create_box()
    for v in box2.vertices:
        verts.append(v.position + np.array([3.0, 0.0, 0.0]))
    for f in box2.faces:
        faces.append([v.index + offset for v in box2.get_face_vertices(f)])
        
    mesh = HalfEdgeMesh.from_arrays(verts, faces)
    
    res = bridge_faces(mesh, [0], [6])
    print("bridge_faces faces:", len(res.faces), "expected:", 14)
    assert len(res.faces) == 14
    # The two merged boxes should form a single closed mesh
    boundaries = [e for e in res.edges if res.is_boundary_edge(e)]
    print("bridge_faces boundary edges:", len(boundaries), "expected:", 0)
    assert len(boundaries) == 0

test_insert_edge_loop()
test_bridge_faces()
print("All passed!")
