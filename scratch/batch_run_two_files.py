import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.subd.primitives import create_box
from src.nurbs.converter import SubDToNURBSConverter
from src.reverse_engineering.quad_wrap import QuadWrapper
import time

def run():
    print("Running batch run simulation for two files...")
    
    mesh = create_box(subdivisions=2)
    qw = QuadWrapper(target_face_count=200)
    quad_mesh = qw.wrap(mesh)
    
    converter = SubDToNURBSConverter()
    res = converter.convert(quad_mesh)
    
    patches = res.get('patches', [])
    print(f"{len(patches)} patches generated")
    
if __name__ == '__main__':
    run()
