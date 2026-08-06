import os
from typing import Any
import trimesh
import logging

from src.core.halfedge_mesh import HalfEdgeMesh

logger = logging.getLogger(__name__)

def export_stl(mesh: 'HalfEdgeMesh', filepath: str, binary: bool = True) -> None:
    """Export HalfEdgeMesh as STL file.
    
    Converts to trimesh first to handle tessellation of arbitrary polygons if needed,
    and then exports.
    """
    if len(mesh.faces) == 0:
        logger.warning(f"Mesh is empty, cannot export STL to {filepath}")
        return
        
    try:
        t_mesh = mesh.to_trimesh()
        t_mesh.export(filepath, file_type='stl' + ('' if binary else '_ascii'))
        logger.info(f"Exported STL to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export STL: {e}")

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
        logger.info(f"Exported OBJ to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export OBJ: {e}")

def export_step(brep_shape: Any, filepath: str) -> None:
    """Export an OCC shape as STEP file.
    
    Args:
        brep_shape: OCP TopoDS_Shape or similar
        filepath: output path
    """
    if brep_shape is None:
        logger.error("No B-Rep shape provided for STEP export.")
        return
        
    try:
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
        
        writer = STEPControl_Writer()
        Interface_Static.SetCVal_s("write.step.schema", "AP214")
        
        status = writer.Transfer(brep_shape, STEPControl_AsIs)
        if status == 1:
            write_status = writer.Write(filepath)
            from OCP.IFSelect import IFSelect_RetDone
            if write_status == IFSelect_RetDone:
                logger.info(f"Exported STEP to {filepath}")
            else:
                logger.error(f"Failed to write STEP file. Return status: {write_status}")
        else:
            logger.error("Failed to transfer shape for STEP export.")
            
    except ImportError:
        try:
            import cadquery as cq
            # cadquery exporter
            cq.exporters.export(brep_shape, filepath, "STEP")
            logger.info(f"Exported STEP to {filepath} using CadQuery.")
        except ImportError:
            logger.error("Neither OCP nor CadQuery are available. Cannot export STEP.")
    except Exception as e:
        logger.error(f"Failed to export STEP: {e}")
