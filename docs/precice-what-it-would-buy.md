# preCICE: what it would actually buy, and what already covers the rest

The 120-item roadmap (internal, kept outside this repository) lists "preCICE coupling" as
**out**, with the reason "preCICE is not in the image and adding it is a 1.2 GB rebuild
plus a coupling library this repository does not own".

That is true and it is also too short to act on, because it does not say what preCICE is
*for* here. This note does, so that the decision to spend the image budget is made against
a known gain rather than against a feature name.

## The rigid case already ships, and is graded

**Fluid-structure interaction is on main** and does not use preCICE.
`openreynolds/toolbox/templates/viv_cylinder.py` writes two complete `pimpleFoam` cases
sharing one O-grid: one with the body bolted still, and one on
`dynamicMotionSolverFvMesh` with `sixDoFRigidBodyMotion`, the cylinder on a linear spring
with zero structural damping. OpenFOAM solves that coupling internally -- the structure is
six degrees of freedom and a spring, so there is no second solver to talk to.

It is graded, in `benchmarks/moving_mesh/`:

- the Strouhal number against Williamson (1989) *JFM* **206** eq. (6), measured **+1.10%**,
  with the honest note that Roshko and Williamson & Brown both give 0.1671, against which
  the same result is **-0.55%** -- the published correlations disagree by more than the
  measurement does;
- net fluid work per cycle, which is **exactly zero** for zero structural damping and needs
  no paper at all. That audit **failed at -10.1%** on the first attempt and found a limit
  cycle 13% too wide, which is why the released case now ships tightly coupled
  (`moveMeshOuterCorrectors yes`, `accelerationRelaxation 1.0`);
- `A/D` is **reported with no verdict**, because no published value for this exact
  configuration was found.

So "FSI" as most people mean it -- a body moving in a flow, with a measured answer -- is
done, and the energy audit that caught the loose coupling is the part worth keeping in
mind when judging what follows.

## What preCICE adds that nothing here can do

One thing: **a structure that deforms**, coupled to a solver that owns the solid.

`sixDoFRigidBodyMotion` moves a rigid body. A flexible beam, a membrane, a composite
blade twisting under load -- those need a structural solver with a real constitutive model,
and the partitioned way to do it is preCICE between OpenFOAM and CalculiX, deal.II or
similar. The image's own `solidDisplacementFoam` is not that partner: it is small-strain
thermo-elastic and `benchmarks/thermal_stress/` uses it for exactly that, a constrained bar
at **-220.0000 MPa** and **-314.2857 MPa** on the hot face. It solves a solid; it does not
exchange an interface with a fluid solver time step by time step.

So the gain is specific and it is not small: deformable FSI is a class of problem the
product currently cannot touch, and saying "FSI is on main" without this distinction would
be the kind of claim this repository spends its time catching.

## What it would cost, and the part that is not the megabytes

**The image.** preCICE, its OpenFOAM adapter, and a solid solver with its own adapter. The
roadmap says 1.2 GB and that is the easy number: it is one `docker/workspace/Dockerfile`
change, a digest bump, and a production `build-solver` dispatch, and the hosted image
derives from the public one by digest so nothing hosted moves until somebody bumps it.

**The thing that is actually hard** is that a partitioned coupling has its own failure
modes, and every one of them looks like a physics result:

- **the coupling scheme.** Explicit (serial) coupling is cheap and unstable for
  low solid-to-fluid density ratios -- the added-mass instability. Implicit with Aitken or
  IQN-ILS underrelaxation is stable and costs several fluid solves per time step. Choosing
  wrong gives a diverging interface that reads as a transient;
- **two time steps and one of them wins.** preCICE subcycles, and a solid step that is a
  multiple of the fluid's is not the same calculation as one that is not;
- **interface mapping.** Non-matching meshes need consistent or conservative mapping, and
  picking consistent for a force field silently loses load;
- **three logs.** A failure is in the fluid, the solid, or the coupling, and this
  repository's `job_status`, `log_digest.py` and `locate.py` all assume one solver writing
  one log.

That last point is the real integration cost. A `preCICE` run is not a `job_start` of one
command: it is two processes that must both be alive, and the backend protocol has no
vocabulary for that today.

## What could be verified here, and what could not

Worth saying precisely, because the Slurm refusal turned on exactly this
(`docs/slurm-backend-plan.md`).

**Verifiable with nothing installed:** the `precice-config.xml` writer as text, the way
`templates/` are already checked -- participants, meshes, data, mapping, coupling scheme,
convergence measures. That is a dictionary writer and it is this repository's usual shape.

**Verifiable in the image, once it has the libraries:** that both participants start, that
the interface exchanges data, and that a rigid-body case run through preCICE reproduces
what `viv_cylinder.py` already gets internally. **That last one is the acceptance test**,
and it is a good one: the same Strouhal number and the same zero net work, by a completely
different route. If preCICE cannot reproduce the case OpenFOAM solves on its own, the
coupling is wrong and nothing downstream is worth reading.

**Not verifiable without a reference:** a deformable case's actual answer. The obvious
candidate is the Turek & Hron FSI2/FSI3 benchmark, which is widely reproduced and has
published displacement amplitudes and frequencies -- but this repository has not read the
primary source, and the pattern for that is already set: `benchmarks/sph/`'s dam break
names Martin & Moyce and declines to grade against a paper it could not obtain. So an FSI2
row would start life with the same caveat, and the honest sequence is: write the config,
reproduce the rigid case, *then* get the paper.

## So the recommendation

Keep it **out**, and keep the reason, but replace the one-liner with this: preCICE buys
**deformable** FSI, the rigid case is already shipped and graded, and the first deliverable
is not a flexible beam but **preCICE reproducing `viv_cylinder.py`'s own answer**. Nobody
should spend 1.2 GB until somebody wants a deforming structure, and when they do, that
reproduction is the test that says the coupling is sound before any new physics is
believed.
