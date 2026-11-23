# `openreynolds[live]` v0.1 -- release notes

**v0.1 of the live-sync extension**, cut from `feature/live-sync` onto `release/v0.1`.
This is the extension's version, not the package's: it ships inside `openreynolds`
0.4.2 as the optional `[live]` extra.

Every number below names where it came from. Where a number is in the engineering record
but not in a machine-readable result, this file says so rather than smoothing over it;
where a result on disk disagrees with a number quoted here, this file says which is
newer and why.

---

## 1. Scope

**v0.1 is the lattice Boltzmann backend for single-phase flow.** D3Q19 on NVIDIA Warp,
two-way coupled to Newton's MuJoCo-Warp solver. Two fluids are in scope:

* **air** -- rotors, ground effect, a quadrotor in its own downwash;
* **submerged water** -- a body fully under the surface, where no free surface is in the
  picture.

**The SPH backend is out of scope for v0.1.** Not beta, not preview, not shipping under a
caveat: out of scope. `openreynolds/live/sph.py` and `sph_cases.py` are in the tree and
import, and that is all this release claims about them. Their eight validation cases do
not have passing numbers, they are not graded for this release, and nothing in v0.1 is
supported on `backend="sph"` or `free_surface=True`. `PROGRESS_LIVE.md` carries their
measurements, including the ones that fail.

**The lattice free surface is frozen, not broken-and-unfixed.** `freesurface.py` stays
behind an `experimental=True` gate with its tests green. `DECISIONS_LIVE.md` D15 is the
decision and the measurement behind it.

So: a boat, a cup, a tank that pours, a hull at a waterline -- none of these is a v0.1
capability, on either backend.

---

## 2. M1 -- laminar validation, case by case

Newest run of each case. The verdict column is the verdict of the check the case grades,
against the reference the case grades against today.

| case | check | target | measured | verdict |
|---|---|---|---|---|
| **Poiseuille**, 64 cells | max \|u - u_exact\| / u_max | <= 2.0e-3 | **7.844e-4** | **PASS** |
| | u_max against the closed form | 0.050000 +/- 1% | **0.0499486** (-0.10%) | **PASS** |
| **cavity Re 100**, 128 cells | u on x = 0.5, RMS vs Ghia Table I (15 interior points) | <= 2% of the lid speed | **0.194%** | **PASS** |
| | v on y = 0.5, RMS vs Ghia Table II | <= 2% of the lid speed | **0.439%** | **PASS** |
| **cavity Re 1000**, 192 cells | u on x = 0.5, RMS vs Ghia Table I | <= 2% of the lid speed | **0.238%** | **PASS** |
| | v on y = 0.5, RMS vs Ghia Table II | <= 2% of the lid speed | **0.642%** | **PASS** |
| **cylinder**, Schafer-Turek 2D-2, 44 cells/D | Cd_max | 3.2300 +/- 5% (interval 3.22-3.24) | **3.29976** (+2.16%) | **PASS** |
| | Strouhal | 0.3000 +/- 3% (interval 0.295-0.305) | **0.302461** (+0.82%) | **PASS** |
| **cross-check** vs this repo's OpenFOAM run | mean Cd | 1.355514 +/- 5% | **1.34469** (-0.80%) | **PASS** |
| | Strouhal | 0.166141 +/- 3% | **0.166053** (-0.053%) | **PASS** |
| | Strouhal vs Williamson (1996), unconfined | 0.164 +/- 3% | **0.166053** (+1.25%) | **PASS** |
| | the OpenFOAM reference has not drifted | < 1e-5 on both | St 3.06e-07, Cd 1.52e-07 | **PASS** |
| **cross-check 2D-1**, Schafer-Turek, steady Re 20 | Cd against the paper's 5.5795 | within 2% | **5.67801** (+1.77%) | **PASS** |
| | Cl inside 0.0104-0.0110 (paper's interval) | an absolute interval | **not recorded** -- see note | **open** |
| | the wake was steady when the forces were read | Cd's own sigma <= 1% of its mean | **not recorded** -- see note | **open** |
| | Cd against the Felix backend | reported, not graded (D17) | 5.67801 vs Felix's 5.522 | -- |
| **sphere** Re 10, 28 cells/D | Cd against Clift, Grace & Weber (D16) | 4.25839 +/- 5% | **4.38591** (+2.99%) | **PASS** |
| **sphere** Re 50 | Cd against Schiller-Naumann | 1.5381 +/- 5% | **1.60141** (+4.12%) | **PASS** |
| **sphere** Re 100 | Cd against Schiller-Naumann | 1.09173 +/- 5% | **1.11662** (+2.28%) | **PASS** |
| **sphere** Re 200 | Cd against Schiller-Naumann | 0.80561 +/- 5% | **0.79894** (-0.83%) | **PASS** |

**Five of the seven M1 cases pass every check they grade.** The two that do not are 2D-1
and the sphere, and both are in section 7 with their numbers.

### What the table is sourced from, and the two gaps in it

* Poiseuille, both cavities and the sphere are read out of
  `benchmarks/live/results/<case>/result.json`, H100 80GB, run at commit `2e60227` with a
  clean tree. `python benchmarks/live/summarise.py` regenerates those rows.
* **The cylinder and the unsteady cross-check rows are not in
  `benchmarks/live/results/`.** Those two cases were re-run after the outlet fix
  (section 3) and the results were never written back, so the directories
  `results/cylinder/` and `results/crosscheck/` still hold the pre-fix runs -- Cd 3.61376
  and 1.70196, both FAIL -- and `summarise.py` still prints the pre-fix table, headed
  "56 of 64 graded checks pass". The post-fix numbers quoted above are from
  `PROGRESS_LIVE.md` and from commits `2091a99` and the cylinder work before it. **They
  are not independently auditable from this tree**, and re-running both cases with the
  results committed is the first thing v0.1.1 owes.
* **2D-1 has no recorded result at all.** Its Cd comes from `DECISIONS_LIVE.md` D17. Its
  lift and steadiness checks have no post-fix number anywhere in the tree, so they are
  marked open rather than given a verdict. The brief for this release records 2D-1 as
  2 of 4 with the lift check as the failure; that is consistent with Cd and steadiness
  passing and lift failing, but this file will not print a verdict it cannot source.
* For the same reasons **no aggregate "n of m checks" figure for M1 is given here.** The
  count that the tree can regenerate is the pre-fix one and it is wrong; the count in the
  release brief (16 of 21) does not reproduce from the per-case numbers in that same
  brief. A headline that cannot be regenerated is worth less than the table above.

---

## 3. The outlet boundary condition, which is the engineering result of this release

Three separate M1 failures -- the cylinder, the unsteady cross-check and 2D-1 -- traced
to one boundary condition, and in each of the three the error had first been attributed
to something else: compressibility for the cylinder, a transverse acoustic mode for the
cross-check, a wrong converged value for 2D-1.

The old `outlet` rule extrapolated populations from one cell inside -- the "do nothing"
outflow. With a velocity inlet at the other end, **nothing in the domain anchors its
mass.** The measurement that settles it needs no cylinder in it at all. An **empty**
channel, 2D-2's geometry with the body removed, 32 cells per diameter, 704,000 steps on
an A100 (`benchmarks/live/results/channel-704k/result.json`):

| outlet | mass change over 704,000 steps | inlet-to-outlet flux imbalance | dp/dx vs analytic Poiseuille |
|---|---|---|---|
| extrapolating (`outlet_rho = 0`) | **+28.113%** | **-0.538%** | 1.172 |
| density-anchored (`outlet_rho = 1.0`) | **+0.032%** | +0.119% | 0.978 |

The flux imbalance is the mechanism: mass flux out sits well under half a per cent below
flux in, every step, forever, and it never corrects. Density rises, pressure rises, and a
force that is a streamwise surface integral rises with it. Lift is a top-minus-bottom
difference, so a rising background cancels out of it -- which is why 2D-1's lift landed
inside the paper's interval while its drag climbed to Cd 17.75.

With the density-anchored outlet the cylinder reads **+2.16%** where it had read
**+11.73%**, and the cross-check reads **-0.80%** on drag and **-0.053%** on Strouhal
where it had read **+25.6%** and **+636%**.

Two estimates of the cylinder's residual error, both made before this fix existed,
predicted it: the Mach sweep extrapolated to **+2.6%** at zero Mach and four extra
diameters of inlet development length gave **+2.4%**. Measured after the fix: **+2.16%**.
So the error decomposed into about nine points of mass-conservation defect and about two
of compressibility residual, and the second had been characterised correctly long before
the first was found.

### What a reader reproduces today, which is not the table in section 2

**The anchored outlet is opt-in and off by default.** In this tree:

* `world.FluidWorld(..., outlet_rho=...)` defaults to **0.0**, the extrapolating outlet.
* `run_cylinder` and `crosscheck` both default `outlet_rho=0.0`, deliberately, so the
  numbers already on record stay reproducible (`cases.py` says so at the call site).
* Only `crosscheck_2d1` defaults to `outlet_rho=1.0`.
* Neither `cylinder` nor `crosscheck` carries an `outlet_rho` default in the case
  registry.

So `openreynolds live run cylinder` with no arguments **reproduces the +11.9% failure**,
at three resolutions, which also brings back the observed-order-of-convergence check that
the single-resolution passing run does not have. The passing numbers in section 2 were
measured with `--extra '{"outlet_rho": 1.0}'`. Deciding whether the anchored outlet
becomes the default for these two cases is open, and there is no decision recorded for
it; until there is, the gap between "the cylinder passes" and "the shipped case passes"
is this flag.

---

## 4. The other milestones

| | state | source |
|---|---|---|
| **M0** smoke | **4 of 4**, A10G, commit `fcf2417` | `results/smoke/` |
| **M2** moving boundaries and two-way coupling | **16 of 16 graded checks, all 4 cases** | `results/{falling-sphere,falling-sphere-windowed,momentum,translating-sphere}/` |
| **M3** robot demos | **10 of 10 graded checks, all 3 cases** | `results/{ground-effect,quadrotor,underwater}/` |
| **M4** performance | **1478.7 MLUPS** on an H100 80GB at 384^3, against a 1000 target; **1095.2** at 128^3 | `PROGRESS_LIVE.md`, `lattice.py`'s layout table |

M0, M2 and M3 were re-tallied from the `result.json` files for this release and come out
at exactly those numbers. M4's H100 figure is recorded in `PROGRESS_LIVE.md` and in
`openreynolds/live/lattice.py`; **the H100 layouts run itself is not in
`benchmarks/live/results/`** -- `layouts.json` there is the A10G run (510.8 soa / 503.1
aos) -- so the headline rate has the same provenance gap as the cylinder.

The target is met at 128^3 as well as at 384^3, which is the useful half: it is reached
at a lattice a real-time loop can afford, not only at one that fills the card.

---

## 5. Isaac Sim

This is the point of the extension, and it was established by running it rather than by
reading version numbers. `docs/isaac-sim-notes.md` is the full write-up.

* **It runs inside a live Kit process on Kit's own Warp.** Isaac Sim 6.2.0.0 bundles Warp
  **1.17.0** and Newton **1.6.1** -- two minor versions below what `benchmarks/live/`
  pins -- and `import warp` inside Kit resolves to `omni.warp.core-1.17.0` rather than the
  pip wheel. Against that: **63 passed, 0 failed, no shim.** No kernel failed to compile
  and none gave a different answer. `hydrostatic` passed 7/7 and `sloshing` 4/4 inside
  Kit.
* **The coupling is two-way there**, confirmed by two numbers that arrive on independent
  routes -- the body's kinematics in Newton, and the populations crossing boundary links
  in the Warp kernels. A one-way coupling flies the submarine and fails both:

  | | measured |
  |---|---|
  | fluid drag against thrust at cruise, \|Fx\| / T | **0.815** |
  | momentum balance, \|F_fluid\| / (T - m a) | **0.879** |
  | samples | **347** wrench samples, sigma 9.33e-6 N, over 20 frames |
  | lattice | 480 x 160 x 200 = 15.36 M nodes at 514 MLUPS on an L40S |

* **`openreynolds.live.isaac.HoverFluidEnv` is a quadrotor hover task in Isaac Lab's
  `DirectRLEnv` interface** -- method for method, every signature checked against the real
  package with `inspect.signature` rather than remembered. It returns the gymnasium >= 1.0
  five-tuple with SAME_STEP autoreset, `is_vector_env` is True, and it drops into Isaac
  Lab's own RL wrappers.

  **It is not a `DirectRLEnv` subclass, and it cannot be.** `DirectRLEnvCfg` requires a
  `scene: InteractiveSceneCfg`, and an `InteractiveScene` cannot be constructed without
  Isaac Sim's USD stage. A class inheriting `DirectRLEnv` here would import and then fail
  on construction. So: no USD scene, nothing viewable in Omniverse, no URDF articulation,
  no Isaac Sim cameras or contact sensors, and no batched physics view across
  environments, therefore no multi-GPU scale-out. `isaac.py`'s own header states all of
  that, including what adapting it with Isaac Sim present would take.

* **Not established**: whether `isaacsim.physics.newton` can drive
  `couple.NewtonCoupling` directly. The in-Kit demos read PhysX bodies and render the
  fluid; the coupled run above uses Newton directly.

---

## 6. Decisions that bear on this release

From `DECISIONS_LIVE.md`, which is the full record with the evidence for each.

| | decision | why it matters to v0.1 |
|---|---|---|
| **D1** | write the LBM rather than depend on Autodesk XLB | the backend is this repository's code, evaluated against the alternative by reading its wheel |
| **D3** | two collision operators, default chosen per use | the graded cases run `trt` with no subgrid model; that is the configuration every number here was measured in |
| **D4** | Bouzidi interpolated bounce-back, halfway as the fallback | the forces this extension exists to produce are surface integrals, and a staircased wall puts a 1/(2 dx) error into every one |
| **D8** | D3Q19 only; D3Q27 is not shipped | the brief's option, not taken |
| **D9** | the fluid carries no gravity; the body gets weight minus buoyancy | a user must not add `rho V g` as well -- buoyancy comes out of the pressure field |
| **D12** | surface tension is not shipped | the knob refuses a non-zero value rather than ignoring it |
| **D13** | a thin direction is periodic, decided once in the world builder | how the quasi-2D cases compute the 2D answer with 3D kernels |
| **D15** | the lattice free surface is frozen; free-surface liquids move to SPH | **the scope line of this release.** LBM stays supported for single-phase flow; the free surface is missing a moving-solid formulation rather than broken, and supplying it is a research task |
| **D16** | the sphere is graded against Clift, Grace & Weber (1978) | changes the sphere's Re 10 verdict. See section 7 |
| **D17** | 2D-1 is graded against Schafer-Turek (1996) only; Felix is dropped | Felix's lift is a factor of ten low and its drag sits outside the paper's own interval, so the lattice passed against the paper and failed against Felix. Lift is now graded against the paper's interval instead of against Felix's 0.0010 |

**A decision this release needed and could not find: the outlet.** Nothing in
`DECISIONS_LIVE.md` records the density-anchored outlet -- not the change, not why
`outlet_rho` defaults to 0, and not whether it becomes the default. It is the
load-bearing change of v0.1 (section 3), three M1 cases depend on it, and
`docs/live-sync.md`'s face-rule table still describes `outlet` as first-order
extrapolation with no mention of the anchor. That gap is the one thing in this release
with no decision behind it.

**And the scope statement in section 1 has no decision behind it either.** D15 moves
free-surface liquids to SPH; it does not say SPH is out of v0.1. The call that SPH is out
of scope for this release is recorded here and nowhere else.

---

## 7. Known limitations

Each with its number, and none of them deferred to a later document.

**The sphere's positive drag bias at Re 10 is real.** Measured Cd **4.38591**. Graded
against Clift, Grace & Weber (D16) that is **+2.99%** and inside the 5% bound; against
Schiller-Naumann it is **+5.66%** and outside it. D16's own caveats hold and this release
does not erase them:

* The five genuinely distinct correlations at Re 10 span **3.13% of their mean** --
  Brown & Lawler 4.12751, Clift-Gauvin 4.13474, Schiller-Naumann 4.15107, Cheng 4.25548,
  CGW 4.25839 -- against a discrepancy of 5.66%. The reference uncertainty and the bound
  are the same size, so the choice of correlation decides the verdict. That is not a
  property a benchmark should have.
* **The measurement sits above every one of the five**, by +2.99% to +6.26%. The positive
  bias is real and is not scatter. What D16 changes is that its size is no longer stated
  as 5.65%, which was an artefact of the correlation chosen.
* CGW's primary text (*Bubbles, Drops, and Particles*, 1978) could not be reached. The
  branch used was read from two independent code transcriptions that both cite it, and the
  case's own target string carries that caveat.
* Resolution, confinement, tau and the Mach number were each eliminated with a number
  (0.07% over a 50% refinement, 0.28%, 0.29%), and a sixth elimination -- "the reference
  is uncertain" at 0.00% -- was **withdrawn as circular**, because `clift_gauvin_cd` calls
  `schiller_naumann_cd`.

**2D-1's lift check is the open one**, and its post-fix Cl is not recorded in this tree
(section 2). Lift here is 0.01 against a drag of 5.6 -- a small difference of two large
surface integrals -- which is why D17 grades it against the paper's absolute interval
rather than a relative tolerance.

**The lattice free surface is frozen by decision** (D15), not broken-and-unfixed. It is
correct for still water: `run_hydrostatic` passes 7 of 7, with `d(rho)/dz` within 0.5% of
`-rho g / c_s^2`, and a fully submerged body in it feels +1.000 x `rho V g`. It fails for
a *moving* rigid body, measured as interface cells per wet column on a released floating
box: **0.957 pinned** (one layer, as the mass exchange requires), **1.391 after ten
steps**, 1.584 after a hundred, and the run diverges later. Korner's formulation has no
moving solids in it; this is missing rather than broken.

**Long closed recirculating runs erode in float32.** A closed recirculating flow loses
about **0.86% of accuracy per million steps**, straight over five points spanning a factor
of nineteen in run length, so the Re 1000 cavity crosses its 2% tolerance at about 2.45
million steps. M1's and M3's runs are clear of that by an order of magnitude, but it is
why a long run needs its length justified rather than maximised.

**The GIFs, counted rather than claimed.** Measured for this release with PIL, frame
count and mean frame-to-frame difference, over every `.gif` in the tree:

| | |
|---|---|
| GIF files on disk | **23** |
| animating (more than one frame, with motion between frames) | **12** |
| holding exactly one frame | **11** |

Of the twelve that animate, four are the Isaac Sim RTX renders (`boat` 44, `carry` 48,
`freesurface` 40, `submarine` 44 frames) and eight are lattice demos: the cylinder at 77
frames, the falling sphere and the quadrotor at two heights at 200 each, the dam break at
three resolutions at 60 each, and the 2D-1 cross-check at 180. The eleven single-frame
files are all superseded duplicates in older result directories or diagnostic runs, with
one exception: `results/crosscheck/live-crosscheck-cylinder.gif` is the current file for
that case and it holds one frame.

Two corrections to the record, both in the direction of the project being better off than
it says:

* `PROGRESS_LIVE.md` states "five animate and eight are stills". That was true when it was
  written and the re-runs have since landed; it is now eight lattice demos animating.
* `PROGRESS_LIVE.md` states that `run_ground_effect` and `run_underwater` "have **no**
  `animate` parameter at all". **Both have one in this tree**, defaulting to True, with a
  `figures.animate` call each. What they do not have is a GIF on disk, because neither has
  been re-run since the `ensure_rgb` fix. Nothing blocks them.

The underlying bug is worth keeping on the page: `figures.ensure_rgb` flattens an image to
RGB so an RGBA PNG does not render as a black panel, and it did that to GIFs as well,
keeping the first frame and discarding the rest. The cases had been collecting frames
correctly the whole time. `animate` now refuses fewer than two frames, all-identical
frames, and frames that collapse on encoding.

---

## 8. Verification performed for this release

Four checks, each run rather than assumed, on this machine (Windows, one CPU core, no
CUDA device, warp 1.19.0).

**1. Core imports without the `[live]` extra. PASS.**
`pytest tests/test_live_optional.py` -- **12 passed**. Independently, with `warp`,
`newton` and `mujoco_warp` blocked at the import system: `import openreynolds`,
`openreynolds.cli` and `openreynolds.live` all succeed, `openreynolds.live.FluidWorld`
raises `ImportError` through the lazy `__getattr__` rather than at import time, and
`openreynolds --help` exits 0.

**2. The documented quickstart runs unmodified. PASS, both blocks.**
Both Python blocks were extracted programmatically from `docs/live-sync.md`'s Quickstart
section -- not retyped -- and run as extracted.

* The lattice block: **exit 0**, 8,000 steps of a 24-cell Re 100 cavity in **5.5 s** at
  2.5 MLUPS, printing `u minimum on x = 0.5: -0.20848` against Ghia, Ghia and Shin's
  `-0.21090`, which is **1.15% low at 24 cells**. The documented value is reproduced
  exactly.
* The SPH block: **exit 0**, 100 steps of 2,244 particles (500 fluid, 1,744 boundary) in
  3.8 s, reporting **0.559% max compression** and 0.117% mean -- the 0.56% the page
  documents. (The block runs; this is not a claim that the SPH backend is in scope.)

The page says the lattice block takes 8 to 10 s on one CPU core; it took 5.5 s here.

**3. Every `[live]` dependency is permissively licensed. PASS.**
Read off the installed distributions' own metadata, not from memory. The extra declares
exactly five:

| dependency | installed | licence | read from |
|---|---|---|---|
| `warp-lang` | 1.19.0 | **Apache-2.0** | `License` |
| `newton` | 1.8.0 | **Apache-2.0** | `License-Expression` |
| `mujoco-warp` | 4.0.1 | **Apache-2.0** | `License-Expression` |
| `numpy` | 2.3.2 | **BSD** (3-Clause) | OSI classifier, and the notice text in `License` |
| `matplotlib` | 3.10.5 | **PSF** | OSI classifier, and the agreement text in `License` |

All five are permissive and none is copyleft. Note that "Apache-2.0 only", as
`PROGRESS_LIVE.md`'s M0 table puts it, is not accurate: three of the five are Apache-2.0,
numpy is BSD and matplotlib is PSF. OpenReynolds itself is **MIT** (`LICENSE`,
`pyproject.toml`), which is compatible with all five.

**4. Nothing is owed in a `LICENSES` file. PASS -- no such file is added.**
Verified by reading `openreynolds/live/sph.py`'s header rather than by taking it on
trust. It states that the scheme was written from Bender and Koschier (SCA 2015; IEEE
TVCG 23(3), 2017), the rigid coupling from Akinci et al. (SIGGRAPH 2012), the kernel from
Monaghan's cubic spline and the viscosity from Monaghan (1992); that **no SPlisHSPlasH
code is in it** and SPlisHSPlasH's source was not available in the environment; and that
a notice for code nobody copied would itself be a false claim about provenance. The one
borrowing it does record is the *shape* of three `wp.HashGrid` calls and the
`for index in query` loop, which is the documented API pattern from Warp's own
Apache-2.0 example, with no expression reproduced. Nothing in the five dependencies'
licences requires a notice file either.

---

## 9. Install

    pip install 'openreynolds[live]'

Core installs and imports without it, and that is an acceptance criterion rather than a
nicety -- the extra pulls a CUDA toolchain and a robotics simulator that most people
installing OpenReynolds will never want.

Tested dependency range, which `pyproject.toml` and `units.py::WARP_TESTED` both carry
and `tests/test_live_versions.py` asserts cannot drift apart: `warp-lang>=1.17,<1.20`,
`newton>=1.6,<1.9`, `mujoco-warp>=3.12`. Measured at both ends (1.19.0/1.8.0 and
1.17.0/1.6.1) and interpolated across the middle; warp 1.18 and newton 1.7 were never
installed, and `docs/live-version-matrix.md` says so in those words. The lower ends are
not arbitrary: Isaac Lab 3.0.0 pins `warp-lang==1.17.0` and `newton[sim]==1.6.1`.

`docs/live-sync.md` is the page to read first.
