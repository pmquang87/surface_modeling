import os
from typing import Any
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh

def export_stl(mesh: 'HalfEdgeMesh', filepath: str, binary: bool = True) -> None:
    """Export HalfEdgeMesh as STL file.
    
    Converts to trimesh first to handle tessellation of arbitrary polygons if needed,
    and then exports.
    """
    try:
        t_mesh = mesh.to_trimesh()
        t_mesh.export(filepath, file_type='stl' + ('' if binary else '_ascii'))
        print(f"Exported STL to {filepath}")
    except Exception as e:
        print(f"Failed to export STL: {e}")

def export_obj(mesh: 'HalfEdgeMesh', filepath: str) -> None:
    """Export HalfEdgeMesh as OBJ file.
    
    Writes vertices and faces manually to preserve quad topologies,
    since trimesh might triangulate them.
    """
    try:
        with open(filepath, 'w') as f:
            for v in mesh.vertices:
                f.write(f"v {v.position[0]} {v.position[1]} {v.position[2]}\n")
            
            for face in mesh.faces:
                f.write("f")
                for v in mesh.get_face_vertices(face):
                    f.write(f" {v.index + 1}")
                f.write("\n")
        print(f"Exported OBJ to {filepath}")
    except Exception as e:
        print(f"Failed to export OBJ: {e}")

def export_step(brep_shape: Any, filepath: str) -> None:
    """Export an OCC shape as STEP file.
    
    Args:
        brep_shape: OCP TopoDS_Shape or similar
        filepath: output path
    """
    if brep_shape is None:
        print("No B-Rep shape provided for STEP export.")
        return
        
    try:
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
        
        writer = STEPControl_Writer()
        Interface_Static.SetCVal_s("write.step.schema", "AP214")
        
        status = writer.Transfer(brep_shape, STEPControl_AsIs)
        if status == 1:
            writer.Write(filepath)
            print(f"Exported STEP to {filepath}")
        else:
            print("Failed to transfer shape for STEP export.")
            
    except ImportError:
        try:
            import cadquery as cq
            # cadquery exporter
            cq.exporters.export(brep_shape, filepath, "STEP")
            print(f"Exported STEP to {filepath} using CadQuery.")
        except ImportError:
            print("Neither OCP nor CadQuery are available. Cannot export STEP.")
    except Exception as e:
        print(f"Failed to export STEP: {e}")
