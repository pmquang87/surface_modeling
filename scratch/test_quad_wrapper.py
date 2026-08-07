import sys
import os
import pyvista as pv

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.importers import import_stl
from src.reverse_engineering.quad_wrap import QuadWrapper

def main():
    file_path = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\7_LLzugdruck_maxstress_smooth_iso0p3_inv.STL"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading {file_path}...")
    mesh = import_stl(file_path)
    print(f"Original Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("\nRunning Quad Wrapper (Target: 5000 faces, Smooth: 0.1)...")
    print("This will use the new MIQ Parametrization and Ray-Casting Tangential Relaxation.")
    
    wrapper = QuadWrapper(target_face_count=5000, smoothing_weight=0.1)
    quad_mesh = wrapper.wrap(mesh)

    print("\nQuad Wrap Complete!")
    print(f"Resulting Mesh: {len(quad_mesh.vertices)} vertices, {len(quad_mesh.faces)} quads")
    
    # Check for watertightness (no boundary edges)
    boundary_edges = [e for e in quad_mesh.edges if quad_mesh.is_boundary_edge(e)]
    print(f"Boundary Edges (Holes): {len(boundary_edges)}")
    
    if len(boundary_edges) == 0:
        print("SUCCESS: The mesh is perfectly watertight!")
    else:
        print("WARNING: The mesh is not watertight.")

    # Render a screenshot
    print("\nRendering screenshot...")
    plotter = pv.Plotter(off_screen=True)
    
    # Convert HalfEdgeMesh to PyVista PolyData for rendering
    vertices = []
    for v in quad_mesh.vertices:
        vertices.append(v.position)
        
    faces_list = []
    for f in quad_mesh.faces:
        verts = quad_mesh.get_face_vertices(f)
        faces_list.append(len(verts))
        for v in verts:
            faces_list.append(v.index)
            
    poly_data = pv.PolyData(vertices, faces_list)
    
    plotter.add_mesh(poly_data, show_edges=True, color='lightblue', edge_color='black')
    plotter.camera_position = 'iso'
    
    screenshot_path = os.path.join(os.path.dirname(__file__), "quad_wrap_result.png")
    plotter.screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

if __name__ == "__main__":
    main()
