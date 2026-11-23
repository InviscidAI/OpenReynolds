# Renders

Every animation in this folder plays, and every one of them is scaled so that what moves
in the flow is what moves on the screen. Both halves of that sentence had to be fixed,
and both were found by **looking at the frames** rather than by counting them.

**The files were stills.** `figures.ensure_rgb` flattens an image to RGB so that an RGBA
PNG does not render as a black panel in a thumbnail index, and it did the same to GIFs --
keeping the first frame and discarding the rest. Three demo files held exactly one frame
each while `PROGRESS_LIVE.md` claimed they animated. `figures.animate` now refuses fewer
than two frames, all-identical frames, and frames that collapse on encoding, and
`ensure_rgb` returns early on anything with more than one frame.

**Then the files that played were mis-scaled.** `figures.animate` took its colour scale
from the first five frames. That is right for a decaying flow, whose peak is in frame
one, and wrong for every flow that starts from rest: the dam break's surge front, the
falling sphere's wake and the quadrotor's downwash all came out as saturated blocks of
one colour, with the clipping boundary as the only thing that moved. Scaling from the
largest value anywhere in the sequence then went too far the other way -- the quadrotor
came back as pale pastel, because a few cells at the rim of an actuator disk carry
several times the velocity of the downwash the picture is of. The scale is now the
largest **99.5th percentile** of any frame: one number for the whole sequence, with half
a percent of the field allowed to clip. Every clip here was re-rendered against it.

| file | what it is | frames | mean frame difference | source run |
| --- | --- | --- | --- | --- |
| `ground-effect-disk-z0.75.mp4` | one actuator disk three quarters of a radius off the floor: the jet forms, reaches the ground and turns outward into a wall jet, with the fountain rising at both sides | 200 | 0.140 | `gif-ground-effect`, H100, 5/5 graded checks |
| `ground-effect-free-air.mp4` | the same disk at six radii with the floor opened to an outflow, which is the denominator the ground-effect ratio is measured against | 200 | 0.276 | same run |
| `underwater-sphere-f1.mp4` | the capsule of zero length, which is a sphere: the wake forming behind it, and the wake the graded Schiller-Naumann drag is then computed from | 200 | 0.109 (median pair 0.024) | `gif-underwater`, H100, 1/1 graded check |
| `underwater-hull-f4.mp4` | the fineness-4 hull on the same code path, the same voxelisation and the same force sum, whose drag is reported and not graded | 200 | 0.174 (median pair 0.050) | same run |
| `quadrotor-ground-effect-z1.mp4` | a quadrotor one rotor radius off the ground; the downwash hits and turns into a wall jet, and the floor vortices under the airframe are the fountain the controller has to reject | 200 | 0.833 | `gif-quadrotor`, L40S, passed |
| `quadrotor-ground-effect-z2.5.mp4` | the same craft at 2.5 radii, where the ground effect has mostly gone and the plumes stay separate | 200 | 0.933 | same run |
| `falling-sphere.mp4` | a sphere released in water, two-way coupled to Newton, with the acoustic rings of the release still visible | 200 | 0.389 | `gif-falling-sphere`, L40S, passed |
| `cylinder-wake-n44.mp4` | the Schafer-Turek 2D-2 cylinder at Re 100, 44 cells per diameter: the Karman street at the resolution the convergence study ends on | 77 | 20.376 | `gif-cylinder`, L40S, **1/3 graded checks** |
| `dam-break-finest.mp4` | the dam break at 1,300 cells, the case behind the convergence study | 60 | 0.539 | `gif-dam-break`, H100, **4/5 graded checks** |
| `isaacsim-dam-break.mp4` | **RTX-rendered inside Isaac Sim.** A 144x64x64 dam break at 167 MLUPS, water as a heightfield mesh rebuilt each frame | 70 | not measured; see below | a live Kit process |
| `isaacsim-submarine.mp4` | **RTX-rendered inside Isaac Sim.** A capsule hull swims 30 -> 99 mm, then takes a dive pulse to 45 degrees nose-down on thrust vectoring alone. The wake is drawn from the decimated velocity field | 88 | not measured; see below | a live Kit process |

## How the numbers were measured, and why both of them

The frame count comes from PIL, on the **source GIF** each MP4 was encoded from. The
frame difference is the mean, over consecutive pairs, of the mean absolute difference
between the two frames on 0-255 RGB. A frame count on its own proves nothing moves,
which is exactly how the earlier false claim survived a year of tooling that agreed the
files were GIFs.

Both numbers are what `benchmarks/live/gallery.py --check` prints -- `frames`, and
`mean` beside the median it calls `motion` -- for the ten of these the gallery carries a
copy of, and the copies are the same bytes. That script is the measurement and this
table is a transcription of it. The two it does not carry are the quadrotor's second
height and the Isaac Sim dam break; the quadrotor's was measured the same way, on
`benchmarks/live/results/gif-quadrotor/live-quadrotor-z2.5.gif`.

The two Isaac Sim clips are RTX renders delivered as MP4 rather than as a lattice slice,
so there is no source GIF to measure them from in the same way. Their frame counts above
are `ffprobe`'s, of the MP4 itself. The gallery holds GIF copies of the same two scenes
(`isaacsim-freesurface.gif`, 40 frames, 2.958, and `isaacsim-submarine.gif`, 44 frames,
1.407), measured the same way as everything else.

## What the numbers do and do not say

**A small frame difference is not a dead file here.** The ground-effect box is sixteen
radii wide and most of it is still air for the whole run, so 0.140 is a true measure of
a picture in which a small, real thing moves. The cylinder's 20.376 is the opposite
case: a wake that fills the frame. Compare a clip against its own still (`live-*.png`
beside it in `benchmarks/live/results/`), not against another clip.

**The two underwater clips really are nearly still after the first fifth**, and their
median pair difference -- 0.024 and 0.050, a long way under their means -- is where that
shows. Re 100 is below the sphere's shedding threshold, so once the wake has formed there
is nothing left to shed: the animation is the wake arriving, and then a steady wake. That
is the physics and not a dead encoder. It is also why these two are the only clips here
whose frames start at step 0 rather than at the sampling window -- the force average
still begins at three quarters of the run, exactly where it did before, and the measured
Cd 1.1137 is the recorded run's number to five figures.

**Two of these come from runs that fail a graded check, and that is said on the row
rather than left to be discovered.** The cylinder misses Schafer and Turek's drag by
about 12% (Cd 3.6137 against the paper's 3.22-3.24) and its observed order of convergence
is 0.30 rather than the 2 a second-order wall should give. The dam break's depth at the
dam reads 25.72 mm against Ritter's 22.22 mm, outside the 15% bound. Neither is new and
neither is the picture's fault: the same two numbers come back to five figures from the
graded runs on record (Cd 3.61376, depth 25.724), which is itself the useful part -- the
re-render changed nothing but the colours.

**The edge spikes in the Isaac Sim dam break's late frames are a meshing artifact**, not
the solver. They are left in rather than filtered out.

## What is NOT here, and why

**No robot-manipulation clip.** The cup of water carried by an end effector -- the demo
that shows a free surface responding to commanded robot motion -- is being built in
`openreynolds/live/carry.py` as this is written, and `benchmarks/live/results/isaacsim/`
already holds a `carry.gif` (48 frames, 7.898) and a `boat.gif` (44 frames, 6.501) from
it. They are not converted here because that work has not landed; whoever lands it owns
the clip.

**Three M2-and-later cases still have no animation path at all**: `translating_sphere`
and `momentum` (both M2), and the free-surface cases other than the dam break
(`hydrostatic`, `sloshing`, `archimedes`) along with the whole SPH list. The two M2 ones
are bookkeeping checks rather than demos -- Galilean invariance, and the impulse against
the momentum the fluid gained -- and nobody has asked them for a picture. They would each
take what `ground_effect` and `underwater` just took: frames collected in the sampling
callback the case already runs, gated on the frame count.

**The graded runs' own GIFs are still one-frame files.** `benchmarks/live/results/<case>/`
holds ten of them, written before `ensure_rgb` was fixed. The animations in this folder
all come from separate `gif-*` picture runs, which is deliberate -- a figure and a graded
number should not have to come from the same incidental run -- but it does mean a reader
who opens `results/quadrotor/live-quadrotor-z1.gif` still gets a still. Re-running the
graded cases would cost GPU hours and change nothing anyone reads.
