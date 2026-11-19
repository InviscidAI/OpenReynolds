# Free-surface field notes (VOF, `interFoam`)

What a two-phase run costs to learn, written down so it is read instead of
rediscovered. Like everything in the toolbox this is a description, not an
instruction: none of it is a sequence, and nothing here checks whether it was
followed. `templates/free_surface.py` is one case built this way, which is a worked
example and not the only shape a VOF case can have.

A free-surface case is unusual in one respect worth stating up front. Most of the ways
it goes wrong produce a run that finishes, writes every time step it was asked for, logs
no bounding warning, and renders a plausible video. The failures below are failures of
the answer, not of the solver, so the log is the wrong place to look for them.

## The interface has its own Courant limit

`adjustTimeStep` with `maxCo` alone is the most common quiet failure in the set.
`maxCo` is the bulk Courant number: the fastest cell anywhere in the domain, water or
air. In a VOF case the air is thin and fast and sets that number, while the interface
moves at the speed of the water. A step inside `maxCo 0.5` routinely moves the surface
a large fraction of a cell, and a surface advanced most of a cell per step diffuses
across five or ten of them over a few hundred steps.

`maxAlphaCo` is the separate limit on that, computed only over cells containing an
interface. A case with both looks identical in the log to a case with one of them. The
difference shows in a picture of alpha after a second of simulated time, and in nothing
else.

`MULESCorr yes` is what makes `maxAlphaCo 0.5` tolerable at all. Explicit MULES wants
something nearer 0.1; the semi-implicit limiter plus `nLimiterIter` buys back the
factor of five. A case carrying `maxAlphaCo 0.5` with `MULESCorr no` is running outside
the limiter's range and will either stiffen into a staircase or lose boundedness in
alpha, and the second of those at least shows in the log.

## `p_rgh` is the unknown, `p` is a result

`interFoam` solves for `p_rgh` = p - rho\*g\*h. The hydrostatic part of the pressure is
subtracted out before the solve, which is the whole reason the solver does not spend
every time step rediscovering gravity. Two consequences follow and neither is obvious
from the field names:

- A hydrostatic column at rest has `p_rgh` **flat and zero at every depth**, not
  increasing downward. A `p_rgh` field initialised with a hydrostatic profile is a case
  that starts with a pressure anomaly the size of the tank depth, and it will solve.
- `p` is derived and written by the solver. A `0/p` handed to it is an initial condition
  for a field that is about to be overwritten from `p_rgh`, and the only thing it
  reliably achieves is making a reader believe `p` is what the case controls.

The wall condition that goes with `p_rgh` is `fixedFluxPressure`, not `zeroGradient`.
It sets the wall-normal gradient from the flux the momentum equation needs, which under
a buoyancy-driven field is not zero. A `zeroGradient` wall in a gravity field generates
or absorbs a small flow at the boundary, and on a long run that is the sort of error
that shows up as a slowly changing total mass.

## An open boundary has to let flow back in

This is the condition most often set to something that cannot work. A collapsing water
column does not simply push air out of the way: behind the surge front the air has to
come back in, and a boundary that only permits outflow makes the domain a sealed box
with a pressure condition painted on the lid.

The pair that works on an open top is `totalPressure` on `p_rgh`,
`pressureInletOutletVelocity` on `U`, and `inletOutlet` on `alpha.water`. Each half
matters:

- `totalPressure` with `p0 0` is a static pressure that drops by the dynamic head where
  there is inflow, so air is admitted without the boundary doing work on the solution.
  A plain `fixedValue 0` there holds the free surface flat as it crosses the boundary --
  which, in a case whose measurement is a surface position, is the measurement being
  overwritten by a boundary condition.
- `pressureInletOutletVelocity` is `zeroGradient` where there is outflow and takes its
  direction from the patch normal where there is inflow. A `fixedValue (0 0 0)` lid is
  the sealed box again.
- `inletOutlet` on alpha, with `inletValue 0`, makes the returning fluid air. The same
  patch on `zeroGradient` copies the internal alpha outward and then back in, so once a
  splash reaches the ceiling the boundary begins feeding water in through it.

A case with `fixedValue` on an open boundary rarely diverges. It stalls the front, or
reports a surge that arrives late, and nothing in the residuals distinguishes that from
physics.

## A uniform `alpha` start is not an initial condition

`internalField uniform 0.5` is the one to recognise on sight. There is no free surface
in it: every cell is half water, which the solver reads as a single fluid of mean
density in every cell, and the result is a fog that settles rather than a column that
collapses. It runs to completion.

`internalField uniform 0` with nothing else is the subtler version: a perfectly valid
case full of air, with the water forgotten. The surge front is then at the origin
forever and the residuals are tiny, which reads as convergence.

What makes the initial field real is `setFields` and a `setFieldsDict` that states the
water as **geometry** -- a `boxToCell`, a `sphereToCell`, a `cylinderToCell`, in metres.
The reason to prefer a region over a cell list is not convenience: a cell list is a
statement about one particular mesh, so a refinement study silently changes the initial
condition along with the grid, and the two effects are then impossible to separate.
`boxToCell` selects on cell centres, which is worth knowing when a box face is meant to
sit exactly on the one-cell thickness of a 2D case -- a face coincident with the cell
centre plane is decided by floating point.

`setFields` overwrites the field it sets, so a case that keeps an `alpha.water.orig`
beside it can be set again from a known state; one that does not is a case where
re-running the setup reads an already-set field as its background.

## Telling a smeared interface from a resolved one in a picture

The question a plot of alpha answers is how many cells wide the transition is, and the
plot has to be drawn so that it can answer it:

- A resolved interface is **one to three cells** from alpha 0.9 to alpha 0.1. Drawn with
  the mesh edges on, it is a line with cells on one side full and cells on the other
  side empty, and at most a couple in between.
- A smeared interface is a band five, ten or twenty cells wide, and the giveaway is that
  the band is wider later in the run than it was at the start. Numerical diffusion is
  monotonic in time: an interface never re-sharpens on its own, so two frames at
  different times settle it where one frame cannot.
- A continuous colour map hides all of this. It draws a smeared band as a pretty
  gradient and a sharp one as a hard edge, and both look intentional. A two-colour map
  split at alpha 0.5, or a single contour at 0.5 with the mesh drawn under it, shows the
  thickness directly. `render.py` and `mesh_look.py` both draw cell edges, which is what
  makes the count possible at all.
- The alpha extrema in the log are a weaker instrument than the picture here, because a
  thoroughly smeared interface can still be bounded in [0, 1] to eight figures. Bounded
  alpha says the limiter is working, not that there is a surface.

A staircase is the opposite failure and reads differently: alpha jumping between 0 and 1
on a cell-by-cell pattern along a surface that ought to be smooth, usually from `cAlpha`
pushed well above 1, or from an interface aligned diagonally across a mesh too coarse to
carry it.

## What the dam break is for

The collapsing water column is the first case worth building because Martin & Moyce
(1952) published the measurement: surge front position against time for a column of
width *a* and height *na*, in dimensionless form, with the scatter of the experiment
shown. A free-surface run with no published reference produces a video, and a video is
compatible with a great many wrong cases.

The comparison that discriminates is the front position at several early times rather
than a final frame: the front leaves the wall, accelerates, and the acceleration is set
by gravity and the column's aspect ratio, so a case with a boundary condition holding
the surface back is late at every time rather than wrong at one.

A density ratio entered as water 1000 against air 1 -- the round numbers the tutorials
carry -- is about 20 percent off in the lighter phase against air at 20 degrees C
(1.204 kg/m^3). It barely moves the surge front, which is driven by gravity on the
heavy phase, and it is worth getting right anyway, because the same numbers are what a
later case with a thin film or an entrained bubble will turn out to depend on.
