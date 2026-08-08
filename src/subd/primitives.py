import numpy as np
from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.catmull_clark import subdivide

def create_box(width: float = 1.0, height: float = 1.0, depth: float = 1.0, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a box primitive."""
    mesh = HalfEdgeMesh()
    
    w, h, d = width / 2, height / 2, depth / 2
    verts = [
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]
    ]
    for v in verts:
        mesh.add_vertex(v)
        
    faces = [
        [0, 3, 2, 1], # back
        [4, 5, 6, 7], # front
        [0, 1, 5, 4], # bottom
        [2, 3, 7, 6], # top
        [0, 4, 7, 3], # left
        [1, 2, 6, 5]  # right
    ]
    for f in faces:
        mesh.add_face(f)
        
    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh


def create_cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 8, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a cylinder primitive."""
    mesh = HalfEdgeMesh()
    h = height / 2
    
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        mesh.add_vertex([radius * np.cos(theta), -h, radius * np.sin(theta)])
        
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        mesh.add_vertex([radius * np.cos(theta), h, radius * np.sin(theta)])
        
    for i in range(segments):
        v1 = i
        v2 = (i + 1) % segments
        v3 = v2 + segments
        v4 = v1 + segments
        mesh.add_face([v1, v4, v3, v2])

    # The walls traverse the bottom ring backwards (i+1 -> i) and the top ring
    # forwards, so each cap has to run the opposite way round to twin with them.
    # Winding the caps like the walls leaves every cap half-edge without a twin
    # and the winding globally inconsistent.
    mesh.add_face(list(range(segments)))
    mesh.add_face(list(range(segments, 2 * segments))[::-1])

    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh


def create_torus(major_radius: float = 1.0, minor_radius: float = 0.3, major_segments: int = 16, minor_segments: int = 8, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a torus primitive."""
    mesh = HalfEdgeMesh()
    
    for i in range(major_segments):
        theta = 2.0 * np.pi * i / major_segments
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        
        for j in range(minor_segments):
            phi = 2.0 * np.pi * j / minor_segments
            cos_ph, sin_ph = np.cos(phi), np.sin(phi)
            
            x = (major_radius + minor_radius * cos_ph) * cos_th
            y = minor_radius * sin_ph
            z = (major_radius + minor_radius * cos_ph) * sin_th
            
            mesh.add_vertex([x, y, z])
            
    for i in range(major_segments):
        for j in range(minor_segments):
            v1 = i * minor_segments + j
            v2 = i * minor_segments + (j + 1) % minor_segments
            v3 = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            v4 = ((i + 1) % major_segments) * minor_segments + j
            
            mesh.add_face([v1, v2, v3, v4])
            
    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh


def create_cone(radius: float = 0.5, height: float = 1.0, segments: int = 8, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a cone primitive."""
    mesh = HalfEdgeMesh()
    h = height / 2
    
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        mesh.add_vertex([radius * np.cos(theta), -h, radius * np.sin(theta)])
        
    tip_idx = segments
    mesh.add_vertex([0, h, 0])
    
    # Outward winding: [v2, v1, tip] puts the side normals away from the axis,
    # and the base then has to run forwards to twin with them. The reverse of
    # both (the old code) is a consistent but inside-out solid: negative volume.
    for i in range(segments):
        v1 = i
        v2 = (i + 1) % segments
        mesh.add_face([v2, v1, tip_idx])

    mesh.add_face(list(range(segments)))

    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh


def create_plane(width: float = 1.0, height: float = 1.0, subdivisions_x: int = 1, subdivisions_y: int = 1, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a plane primitive."""
    mesh = HalfEdgeMesh()
    
    w, h = width / 2, height / 2
    for j in range(subdivisions_y + 1):
        y = -h + (j / subdivisions_y) * height
        for i in range(subdivisions_x + 1):
            x = -w + (i / subdivisions_x) * width
            mesh.add_vertex([x, 0, y])
            
    for j in range(subdivisions_y):
        for i in range(subdivisions_x):
            v1 = j * (subdivisions_x + 1) + i
            v2 = v1 + 1
            v3 = v2 + (subdivisions_x + 1)
            v4 = v1 + (subdivisions_x + 1)
            mesh.add_face([v1, v4, v3, v2])
            
    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh


def create_sphere(radius: float = 0.5, segments: int = 8, rings: int = 6, subdivisions: int = 0) -> HalfEdgeMesh:
    """Create a UV sphere primitive."""
    mesh = HalfEdgeMesh()
    
    mesh.add_vertex([0, -radius, 0])
    
    for i in range(1, rings):
        phi = np.pi * (i / rings) - np.pi / 2
        y = radius * np.sin(phi)
        r = radius * np.cos(phi)
        
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            mesh.add_vertex([x, y, z])
            
    mesh.add_vertex([0, radius, 0])
    
    # Every ring below was wound the other way round, giving a consistent but
    # inside-out sphere (negative volume), unlike create_box/create_torus.
    for j in range(segments):
        v1 = 0
        v2 = 1 + j
        v3 = 1 + (j + 1) % segments
        mesh.add_face([v1, v2, v3])

    for i in range(rings - 2):
        row1 = 1 + i * segments
        row2 = 1 + (i + 1) * segments
        for j in range(segments):
            v1 = row1 + j
            v2 = row1 + (j + 1) % segments
            v3 = row2 + (j + 1) % segments
            v4 = row2 + j
            mesh.add_face([v4, v3, v2, v1])

    n_pole = 1 + (rings - 1) * segments
    row = 1 + (rings - 2) * segments
    for j in range(segments):
        v1 = row + j
        v2 = n_pole
        v3 = row + (j + 1) % segments
        mesh.add_face([v1, v2, v3])

    if subdivisions > 0:
        mesh = subdivide(mesh, subdivisions)
    return mesh
