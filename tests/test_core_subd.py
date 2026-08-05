import sys
import os
import json
import numpy as np

sys.path.insert(0, r'C:\Users\pmqua\PycharmProjects\surface_modeling')

try:
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.core.feature_tree import FeatureTree, Feature
    from src.subd.catmull_clark import subdivide, evaluate_limit_surface
    from src.subd.primitives import create_box, create_cylinder, create_torus, create_cone, create_plane, create_sphere
    from src.subd.editing import extrude_faces, inset_faces, mirror_mesh, soft_selection_move, set_edge_weight
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

results = {'pass': 0, 'fail': 0}

def test(name, condition, details=""):
    if condition:
        print(f"PASS: {name}")
        results['pass'] += 1
    else:
        print(f"FAIL: {name} - {details}")
        results['fail'] += 1

def run_halfedge_mesh_tests():
    print("\n--- HalfEdgeMesh Tests ---")
    try:
        # Create from arrays
        vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0.5, 0.5, 1]])
        faces = [[0,1,2,3], [0,1,4], [1,2,4], [2,3,4], [3,0,4]]
        mesh = HalfEdgeMesh.from_arrays(vertices, faces)
        test("Create mesh from arrays", len(mesh.vertices) == 5 and len(mesh.faces) == 5, f"v: {len(mesh.vertices)}, f: {len(mesh.faces)}")
        
        # Test adjacency
        fv = mesh.get_face_vertices(mesh.faces[0])
        test("get_face_vertices", len(fv) == 4, f"fv: {len(fv)}")
        
        vf = mesh.get_vertex_faces(mesh.vertices[4])
        test("get_vertex_faces", len(vf) == 4, f"vf: {len(vf)}")
        
        vn = mesh.get_vertex_neighbors(mesh.vertices[4])
        test("get_vertex_neighbors", len(vn) == 4, f"vn: {len(vn)}")
        
        test("vertex_valence", mesh.vertex_valence(mesh.vertices[4]) == 4, f"valence: {mesh.vertex_valence(mesh.vertices[4])}")
        
        # Open mesh for boundary test
        faces_open = [[0,1,2,3]]
        mesh_open = HalfEdgeMesh.from_arrays(vertices[:4], faces_open)
        test("is_boundary_vertex", mesh_open.is_boundary_vertex(mesh_open.vertices[0]), "Vertex 0 should be boundary")
        
        # Edge loop / ring
        mesh_box = create_box()
        try:
            loop = mesh_box.get_edge_loop(mesh_box.edges[0])
            test("edge loop", isinstance(loop, list))
        except Exception as e:
            test("edge loop", False, str(e))
        
        mesh.edges[0].crease_weight = 1.0
        test("crease_weight", mesh.edges[0].crease_weight == 1.0)
        
        mesh_copy = mesh.copy()
        mesh_copy.vertices[0].position[0] = 100
        test("copy produces independent deep copy", mesh.vertices[0].position[0] == 0)
        
        try:
            pv = mesh.to_pyvista()
            test("to_pyvista", pv is not None)
        except Exception as e:
            test("to_pyvista", False, str(e))
            
        try:
            tm = mesh.to_trimesh()
            test("to_trimesh", tm is not None)
            
            mesh_from_tm = HalfEdgeMesh.from_trimesh(tm)
            test("from_trimesh roundtrip", len(mesh_from_tm.vertices) > 0)
        except Exception as e:
            test("to_trimesh/from_trimesh", False, str(e))
            
        try:
            mesh.compute_face_normals()
            mesh.compute_vertex_normals()
            # check that normal was populated
            test("compute_normals", np.linalg.norm(mesh.vertices[0].normal) > 0)
        except Exception as e:
            test("compute_normals", False, str(e))
            
        empty_mesh = HalfEdgeMesh()
        test("empty mesh handling", len(empty_mesh.vertices) == 0)
        
    except Exception as e:
        print(f"Error in HalfEdgeMesh tests: {e}")

def run_feature_tree_tests():
    print("\n--- FeatureTree Tests ---")
    try:
        tree = FeatureTree()
        node = Feature("Box", "primitive", {"size": 1})
        tree.add_feature(node)
        test("Create and add feature", len(tree.features) == 1)
        
        tree.undo()
        test("undo", len(tree.features) == 0)
        tree.redo()
        test("redo", len(tree.features) == 1)
        
        node.enabled = False
        test("toggle feature", not tree.features[0].enabled)
        
        node2 = Feature("Cyl", "primitive", {})
        tree.add_feature(node2)
        tree.move_feature(1, 0)
        test("move_feature", tree.features[0].name == "Cyl")
        
        tree.remove_feature(0)
        test("remove_feature", len(tree.features) == 1 and tree.features[0].name == "Box")
        
        # Serialization
        try:
            d = tree.to_dict()
            tree2 = FeatureTree.from_dict(d)
            test("serialization roundtrip", len(tree2.features) == 1 and tree2.features[0].name == "Box")
        except Exception as e:
            test("serialization roundtrip", False, str(e))
            
    except Exception as e:
        print(f"Error in FeatureTree tests: {e}")

def run_catmull_clark_tests():
    print("\n--- Catmull-Clark Tests ---")
    try:
        box = create_box()
        sub1 = subdivide(box, 1)
        test("Subdivide box 1 level", len(sub1.vertices) == 26 and len(sub1.faces) == 24, f"v: {len(sub1.vertices)}, f: {len(sub1.faces)}")
        
        sub2 = subdivide(box, 2)
        test("Subdivide box 2 levels", len(sub2.faces) > 24)
        
        # Test creases
        box.edges[0].crease_weight = 1.0
        sub_crease = subdivide(box, 1)
        test("Subdivide with crease weights", len(sub_crease.vertices) > 0)
        
        try:
            lim_pos, lim_norm = evaluate_limit_surface(box)
            test("evaluate_limit_surface", len(lim_pos) == len(box.vertices))
        except Exception as e:
            test("evaluate_limit_surface", False, str(e))
            
    except Exception as e:
        print(f"Error in Catmull-Clark tests: {e}")

def run_primitives_tests():
    print("\n--- Primitives Tests ---")
    try:
        prims = [
            ("box", create_box()),
            ("cylinder", create_cylinder()),
            ("torus", create_torus()),
            ("cone", create_cone()),
            ("plane", create_plane()),
            ("sphere", create_sphere())
        ]
        for name, mesh in prims:
            test(f"create_{name}", len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"v: {len(mesh.vertices)}, f: {len(mesh.faces)}")
            # basic valid face check
            valid = all(len(mesh.get_face_vertices(f)) >= 3 for f in mesh.faces)
            test(f"{name} valid faces", valid)
            
    except Exception as e:
        print(f"Error in primitives tests: {e}")

def run_editing_tests():
    print("\n--- Editing Tests ---")
    try:
        box = create_box()
        try:
            b2 = extrude_faces(box, [0], 1.0)
            test("extrude_faces", len(b2.faces) > len(box.faces))
        except Exception as e:
            test("extrude_faces", False, str(e))
            
        try:
            b3 = inset_faces(box, [0], 0.2)
            test("inset_faces", len(b3.faces) > len(box.faces))
        except Exception as e:
            test("inset_faces", False, str(e))
            
        try:
            b4 = mirror_mesh(box, axis='x')
            test("mirror_mesh", len(b4.faces) >= len(box.faces))
        except Exception as e:
            test("mirror_mesh", False, str(e))
            
        try:
            b5 = soft_selection_move(box, 0, np.array([1,1,1]), radius=2.0)
            test("soft_selection_move", np.any(b5.vertices[0].position != box.vertices[0].position))
        except Exception as e:
            test("soft_selection_move", False, str(e))
            
        try:
            b6 = set_edge_weight(box, [0], 0.5)
            test("set_edge_weight", b6.edges[0].crease_weight == 0.5)
        except Exception as e:
            test("set_edge_weight", False, str(e))
            
    except Exception as e:
        print(f"Error in editing tests: {e}")

if __name__ == '__main__':
    run_halfedge_mesh_tests()
    run_feature_tree_tests()
    run_catmull_clark_tests()
    run_primitives_tests()
    run_editing_tests()
    
    print("\n=== SUMMARY ===")
    print(f"Total PASS: {results['pass']}")
    print(f"Total FAIL: {results['fail']}")
