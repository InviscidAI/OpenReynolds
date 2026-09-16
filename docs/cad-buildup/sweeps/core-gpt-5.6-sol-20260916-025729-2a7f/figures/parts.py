"""The part inside the flow: named wall patches off the DELIVERED polyMesh.

Reads `constant/polyMesh` through pyvista's OpenFOAM reader rather than any `VTK/`
directory already on disk -- several runs left one behind from an earlier mesh
iteration of their own, and reusing it renders geometry the case did not deliver.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv
pv.OFF_SCREEN = True

SD = Path("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/957c0f71-2b79-452c-b452-7f7bbeb66bda/scratchpad")
SWEEP = json.load(open("/home/qiuzi/PycharmProjects/OpenReynolds/docs/cad-buildup/sweeps/"
                       "core-gpt-5.6-sol-20260916-025729-2a7f/sweep.json"))
WANT = {
 "T22": ["vaneSurfaces"], "T11": ["honeycombWalls"], "T18": ["fins", "barrel", "head"],
 "T15": ["pinion", "wheel"], "T12": ["blades", "hub"],
 "T10": ["bladeSuction", "bladePressure"], "T17": ["runner_walls"],
 "T16": ["outer_cowl", "core_cowl"], "T23": ["ball"],
 "T13": ["neck_flank", "lid_flank"], "T26": ["treads", "column"],
 "T19": ["keywayFlanks", "keywayFloor"], "T9": ["spinner_wetted", "outer_ring_wetted"],
 "T24": ["ductWall1", "ductWall2", "ductWall3", "ductWall4", "cylinderWall"],
 "T2": ["sphere"], "T14": ["wing"], "T21": ["standoffOuterSurfaces", "blindHoleSurfaces"],
 "T3": ["walls"], "T1": ["walls"], "T20": ["gasket_inner_face", "flange1_bore_wall"],
}
COLOURS = ["#c0392b", "#27ae60", "#2c6fbb", "#d4a017", "#7d3c98", "#16a085"]

def render(case, view=None):
    d = Path([r["case_dir"] for r in SWEEP["runs"] if r["case"] == case][0])
    (d / "case.foam").touch()
    reader = pv.OpenFOAMReader(str(d / "case.foam"))
    reader.set_active_time_value(reader.time_values[0])
    reader.enable_all_patch_arrays()
    boundary = reader.read()["boundary"]
    have = list(boundary.keys())
    found = {n: boundary[n] for n in WANT[case] if n in have and boundary[n].n_cells}
    if not found:
        return f"{case}: none of {WANT[case]} in {have}"
    p = pv.Plotter(off_screen=True, window_size=(1000, 800))
    for i, (name, mesh) in enumerate(found.items()):
        p.add_mesh(mesh, color=COLOURS[i % len(COLOURS)], show_edges=False)
    p.background_color = "white"
    p.camera_position = view or "iso"
    out = SD / "renders" / f"{case}-part.png"
    p.screenshot(str(out))
    p.close()
    return f"{case}: " + ", ".join(f"{k} {v.n_cells:,}f" for k, v in found.items())

if __name__ == "__main__":
    for c in sys.argv[1:]:
        try: print(render(c))
        except Exception as e: print(f"{c}: FAILED {type(e).__name__}: {e}")
