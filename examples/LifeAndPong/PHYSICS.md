# Life Arcade collision model

Authority: owner steering `DEC-life-arcade-physics-006`. This replaces the old
2% noise, minimum-angle clamp, automatic return steering and per-hit speed reset.

## What “realistic” means here

This is an idealized planar rigid-body model, not a full aerodynamic simulation.
The ball follows a straight trajectory between contacts. No gravity, drag or
Magnus force is applied. The ball has unit mass and solid-disc inertia
`I = m r² / 2`; walls/cells are fixed, and the player paddle is kinematic.
Normal restitution is 1 (elastic). Paddle friction is 0.08; walls/cells have
zero friction. Particle effects cannot modify any physical state.

These are conventional rigid-body concepts: the official [Box2D simulation
guide](https://box2d.org/documentation/md_simulation.html) describes kinematic
bodies, Coulomb friction, restitution and continuous collision detection.
This project implements its own small C# solver; it does not depend on Box2D.

## Geometry and time

- Circle ball: radius 10 world units.
- Capsule paddle: 18 by 144, corner radius 9.
- Rounded cells: 22 by 22, corner radius 2, on a 24-unit pitch.
- The solid renderer uses the same dimensions. Glow is not collision geometry.

Each shape is expanded by the ball radius. Straight face crossings are solved
linearly; rounded corners use the smaller nonnegative ray/circle quadratic
root. A swept box is only the broad-phase filter: its empty corner cannot
produce a false hit. Paddle queries use the ball's velocity relative to the
paddle. The controller updates at 120 Hz; the paddle follows piecewise-linear
motion within each step and stops at its arena bounds.

The session chooses the earliest collision, phase crossing, paddle-bound stop
or goal. It advances to that time, resolves the event and continues through the
remaining time. Cells remain frozen throughout the left 70%; a crossing cannot
accidentally spend its whole frame on the wrong Life clock.

## Contact response

For outward unit normal `n`, incoming ball velocity `v` and surface velocity `u`,
the isolated frictionless response is:

`v' = v - 2 dot(v - u, n) n`

Thus a stationary flat wall reflects the normal component and preserves the
tangent component and speed. A moving surface can add energy: this is work
done by the surface, not an arbitrary game-speed multiplier.

Simultaneous contacts use nonnegative normal impulses satisfying elastic
outgoing-normal constraints. In 2D, single and paired independent active
constraints suffice for the feasible set. Degenerate overlap recovery projects
out of closing directions rather than inventing a steering angle.

For an isolated paddle contact, let `t = (-n.y, n.x)` and angular speed be `w`:

`slip = dot(v' - u, t) - w r`

`jt = clamp(-slip / 3, -0.08 jn, +0.08 jn)`

`v'' = v' + jt t;  w' = w - 2 jt / r`

The denominator includes rotational inertia. Friction accounts for spin and
can dissipate mechanical energy; it is not a fixed vertical velocity bonus.
Multi-contact manifolds resolve normal constraints without the isolated-contact
friction approximation. A ball mark shows rotation; spin alone does not curve
free flight in this no-aerodynamics model.

## Requested 5% noise

After one resolved contact manifold (wall, cell or paddle), draw `q` uniformly
from `[-0.05, +0.05]`. For the resulting physical velocity `v`, form:

`candidate = v + q (-v.y, v.x)`

Normalize the candidate to `|v|`. The speed is the computed post-impulse speed,
not a sector target. This gives at most `atan(0.05)`, about 2.86 degrees, of
angular perturbation. If it points back into a contact, reduce the magnitude
of the same draw by deterministic bisection. Never resample or exceed the bound.
No noise is added in free flight. A noise-disabled internal test mode verifies
the ideal trajectories independently.

## Safety and acceptance

A 0.001-unit separation tolerance handles roundoff. Positive crossing times,
however small, are consumed rather than skipped. An unresolved loop over 128
events within one controller step pauses with an external-HUD explanation;
it does not silently drop the rest of the frame or teleport the ball.

Regression cases cover analytic face/corner times, missed empty bounding-box
corners, simultaneous wall/cell contacts, remaining frame travel, moving-paddle
work, Coulomb/spin energy, zero-speed and grazing noise safety, unchanged
shallow trajectories over six seconds, and a 100,000-unit/second stress shot.
The stress speed tests the solver; it is not a normal launch speed or a claim
that extreme shots have been play-balanced.
