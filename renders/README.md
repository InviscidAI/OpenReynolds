# Renders

Every animation in this folder plays. That is worth stating because it was not true
before: `figures.ensure_rgb` flattens an image to RGB so an RGBA PNG does not render as
a black panel in a thumbnail index, and it did the same to GIFs -- keeping the first
frame and discarding the rest. Three demo files held exactly one frame each while
`PROGRESS_LIVE.md` claimed they animated. `figures.animate` now refuses fewer than two
frames, all-identical frames, and frames that collapse on encoding.

| file | what it is | frames | verified |
| --- | --- | --- | --- |
| `quadrotor-ground-effect-z1.mp4` | a quadrotor one rotor diameter off the ground; the downwash hits and turns into a wall jet | 200 | mean frame difference 0.473 |
| `quadrotor-ground-effect-z2.5.mp4` | the same craft at 2.5 diameters, where the ground effect has mostly gone | 200 | 0.579 |
| `dam-break-finest.mp4` | the dam break at its finest grid, the case behind the convergence study | 60 | 1.530 |
| `isaacsim-submarine.mp4` | **RTX-rendered inside Isaac Sim.** A capsule hull swims 30 -> 99 mm, then takes a dive pulse to 45 degrees nose-down on thrust vectoring alone. The wake is drawn from the decimated velocity field | - | rendered in a live Kit process |
| `isaacsim-dam-break.mp4` | **RTX-rendered inside Isaac Sim.** A 144x64x64 dam break at 167 MLUPS, water as a heightfield mesh rebuilt each frame | - | same |

The frame counts and the mean absolute difference between consecutive frames were both
measured with PIL. A frame count on its own does not prove anything moves, which is how
the earlier false claim survived.

## What is NOT here, and why

**No robot-manipulation demo.** The cup of water carried by an end effector -- the demo
that shows a free surface responding to commanded robot motion, and the one the project
exists for -- is not built. It needs the SPH backend's held-body path, and that path had
a bug: `_float_rig`'s `hold()` wrote Newton's public `body_q` while MuJoCo integrated its
own `qpos` and never re-read it, so a held hull drifted 0.060 -> 0.0055 m over 50 steps
while the pinned array read 0.060000 throughout. That is fixed by settling the fluid
without stepping Newton at all, and the fix is not yet verified end to end.

**The edge spikes in the Isaac Sim dam break's late frames are a meshing artifact**, not
the solver. They are left in rather than filtered out.

**Two demos have no animation path at all**: ground effect and the underwater hull were
written to emit PNG frames, so there is nothing to convert for them.
