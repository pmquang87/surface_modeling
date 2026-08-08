"""Rebuild the shell STEP as ONE solid with internal void cavities."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
from OCP.TopoDS import TopoDS, TopoDS_Solid
from OCP.BRep import BRep_Builder
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.Interface import Interface_Static

SRC = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\for_claude\7_LLzugdruck_maxstress_smooth_iso0p3_inv_claude_shell3mm.step"
OUT = SRC  # rewrite in place

r = STEPControl_Reader()
assert r.ReadFile(SRC) == 1
r.TransferRoots()
shape = r.OneShape()

solids = []
exp = TopExp_Explorer(shape, TopAbs_SOLID)
while exp.More():
    s = TopoDS.Solid_s(exp.Current())
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, props)
    solids.append((props.Mass(), s))
    exp.Next()
solids.sort(key=lambda t: -abs(t[0]))
print("input solids volumes:", [round(v) for v, _ in solids])

def first_shell(solid):
    e = TopExp_Explorer(solid, TopAbs_SHELL)
    return TopoDS.Shell_s(e.Current())

builder = BRep_Builder()
voided = TopoDS_Solid()
builder.MakeSolid(voided)
builder.Add(voided, first_shell(solids[0][1]))          # outer wall
for vol, s in solids[1:]:
    sh = first_shell(s)
    sh.Reverse()                                        # cavity: material outside
    builder.Add(voided, sh)

props = GProp_GProps()
BRepGProp.VolumeProperties_s(voided, props)
expected = solids[0][0] - sum(abs(v) for v, _ in solids[1:])
print(f"voided solid volume: {props.Mass():.0f} (expected ~{expected:.0f})")
valid = BRepCheck_Analyzer(voided).IsValid()
print("BRepCheck valid:", valid)

nf = 0
e = TopExp_Explorer(voided, TopAbs_FACE)
while e.More():
    nf += 1; e.Next()
print("faces:", nf)

assert valid and props.Mass() > 0 and abs(props.Mass() - expected) < 0.02 * expected

Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
w = STEPControl_Writer()
w.Transfer(voided, STEPControl_AsIs)
w.Write(OUT)
print("wrote", OUT)
