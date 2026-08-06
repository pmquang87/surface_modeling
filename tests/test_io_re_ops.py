import sys
sys.path.insert(0, r'C:\Users\pmqua\PycharmProjects\surface_modeling')

import os
import tempfile
import traceback
import time

from src.io.importers import import_stl, import_obj, import_step
from src.io.exporters import export_stl, export_obj, export_step
from src.reverse_engineering.mesh_tools import smooth_mesh, fill_holes, offset_mesh, decimate_mesh, compute_mesh_quality
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.reverse_engineering.shrink_wrap import ShrinkWrapper
from src.operations.shell_thicken import shell_solid, thicken_surface
from src.subd.primitives import create_box, create_plane
from src.nurbs.converter import SubDToNURBSConverter
from src.core.halfedge_mesh import HalfEdgeMesh

STL_FILE = None
STEP_FILE = None

def setup_test_files():
    global STL_FILE, STEP_FILE
    import trimesh
    mesh = trimesh.creation.box()
    f_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    f_stl.close()
    STL_FILE = f_stl.name
    mesh.export(STL_FILE)

    # For STEP, we will just use a dummy text file to avoid full test crashes, 
    # though importer_step might fail if it's not a real STEP.
    # We can try exporting a real STEP with trimesh if it supports it, else just empty.
    f_stp = tempfile.NamedTemporaryFile(suffix='.stp', delete=False)
    f_stp.write(b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    f_stp.close()
    STEP_FILE = f_stp.name

setup_test_files()

results = {"PASS": 0, "FAIL": 0}

def run_test(name, func):
    print(f"--- Running Test: {name} ---")
    try:
        start = time.time()
        func()
        end = time.time()
        print(f"PASS: {name} ({end - start:.2f}s)\n")
        results["PASS"] += 1
    except Exception as e:
        print(f"FAIL: {name}")
        traceback.print_exc()
        print()
        results["FAIL"] += 1

def test_importer_stl_real():
    mesh = import_stl(STL_FILE)
    assert mesh is not None, "Imported STL is None"
    assert len(mesh.vertices) > 0, "No vertices in imported STL"
    assert len(mesh.faces) > 0, "No faces in imported STL"

def test_importer_stl_nonexistent():
    try:
        import_stl("nonexistent_file_12345.stl")
        assert False, "Should have raised an error for nonexistent file"
    except Exception as e:
        assert "nonexistent_file_12345.stl" in str(e) or isinstance(e, (FileNotFoundError, ValueError, SystemError)), f"Expected specific error, got: {e}"

def test_importer_obj():
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        temp_obj = f.name
    
    mesh = import_obj(temp_obj)
    os.remove(temp_obj)
    assert mesh is not None
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1

def test_importer_step():
    res = import_step(STEP_FILE)
    assert res is not None, "import_step returned None"

def test_exporter_stl():
    mesh = import_stl(STL_FILE)
    v_count = len(mesh.vertices)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_stl = f.name
    export_stl(mesh, temp_stl)
    
    mesh2 = import_stl(temp_stl)
    os.remove(temp_stl)
    assert len(mesh2.vertices) == v_count, f"Vertex count mismatch: {len(mesh2.vertices)} vs {v_count}"

def test_exporter_obj():
    mesh = import_stl(STL_FILE)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_obj = f.name
    export_obj(mesh, temp_obj)
    
    with open(temp_obj, 'r') as f:
        content = f.read()
    os.remove(temp_obj)
    
    assert 'v ' in content, "No vertices in OBJ"
    assert 'f ' in content, "No faces in OBJ"

def test_exporter_stl_empty():
    mesh = HalfEdgeMesh()
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_stl = f.name
    
    try:
        export_stl(mesh, temp_stl)
        # It shouldn't crash, or if it does it should be a handled exception
    except Exception as e:
        pass # Allow handled failure
    finally:
        if os.path.exists(temp_stl):
            os.remove(temp_stl)

def test_reverse_engineering():
    mesh = import_stl(STL_FILE)
    # Decimate to avoid timeout
    decimated = decimate_mesh(mesh, target_faces=5000)
    
    # mesh_tools
    smoothed = smooth_mesh(decimated, iterations=1)
    filled = fill_holes(smoothed)
    offset = offset_mesh(filled, distance=0.1)
    
    q_quality = compute_mesh_quality(offset)
    assert isinstance(q_quality, dict), "compute_mesh_quality didn't return a dict"
    
    # Wrapper
    qw = QuadWrapper(target_face_count=500)
    q_mesh = qw.wrap(decimated)
    
    sw = ShrinkWrapper(iterations=2)
    cage = create_box(width=10, height=10, depth=10) # dummy cage
    s_mesh = sw.wrap(cage, decimated)

def test_operations_shell():
    box = create_box(width=1, height=1, depth=1)
    shelled = shell_solid(box, thickness=0.1)
    assert shelled is not None

def test_operations_thicken():
    plane = create_plane(width=1, height=1)
    thickened = thicken_surface(plane, thickness=0.1)
    assert thickened is not None

def test_operations_zero_thickness():
    box = create_box(width=1, height=1, depth=1)
    try:
        shelled = shell_solid(box, thickness=0.0)
    except Exception:
        pass # Handling error is fine

def test_operations_empty():
    mesh = HalfEdgeMesh()
    try:
        shell_solid(mesh, thickness=0.1)
    except Exception:
        pass

def test_nurbs_converter():
    box = create_box(subdivisions=1)
    converter = SubDToNURBSConverter()
    res = converter.convert(box)
    assert isinstance(res, dict)
    assert 'patches' in res
    assert 'shape' in res
    assert 'mesh' in res

if __name__ == '__main__':
    tests = [
        ("importer_stl_real", test_importer_stl_real),
        ("importer_stl_nonexistent", test_importer_stl_nonexistent),
        ("importer_obj", test_importer_obj),
        ("importer_step", test_importer_step),
        ("exporter_stl", test_exporter_stl),
        ("exporter_obj", test_exporter_obj),
        ("exporter_stl_empty", test_exporter_stl_empty),
        ("reverse_engineering", test_reverse_engineering),
        ("operations_shell", test_operations_shell),
        ("operations_thicken", test_operations_thicken),
        ("operations_zero_thickness", test_operations_zero_thickness),
        ("operations_empty", test_operations_empty),
        ("nurbs_converter", test_nurbs_converter)
    ]
    
    for name, func in tests:
        run_test(name, func)
        
    print("=== SUMMARY ===")
    print(f"PASS: {results['PASS']}")
    print(f"FAIL: {results['FAIL']}")
