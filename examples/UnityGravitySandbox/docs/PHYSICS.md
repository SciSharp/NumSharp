# The Physics & Applied Math Behind the Gravity Sandbox

This sample is a real orbital-mechanics simulator. Every quantity below is computed in C# by
**NumSharp** on whole `(N, 3)` arrays — no per-body loops — and every claim here is checked by the
`Verification/` harness against machine precision or an analytic formula. This document is the
derivation; the code is the transcription.

Notation: `N` bodies, body `i` has position **rᵢ**, velocity **vᵢ**, mass `mᵢ`. `G` is the
gravitational constant *in the scenario's unit system* (see [Units](#units)).

---

## 1. Newtonian N-body dynamics

Each body feels the summed gravitational pull of every other body (Newton's law of universal
gravitation):

```
        N        mⱼ (rᵢ − rⱼ)
aᵢ = −G  Σ    ───────────────────
       j≠i     | rᵢ − rⱼ |³
```

The equations of motion are `drᵢ/dt = vᵢ`, `dvᵢ/dt = aᵢ`. This is a coupled system of `2N`
second-order ODEs with no closed-form solution for `N ≥ 3` — which is exactly why we integrate it
numerically and why the three-body figure-eight is remarkable.

### 1.1 Softening

The `1/r²` force diverges as two bodies coincide, so a single close encounter with a finite time
step can eject a body to infinity. We use **Plummer softening** with length `ε`:

```
        N            mⱼ (rᵢ − rⱼ)
aᵢ = −G  Σ    ──────────────────────────
       j≠i     ( |rᵢ − rⱼ|² + ε² )^{3/2}
```

`ε` is not a numerical fudge — it is the field of an extended (Plummer) mass distribution rather
than a point, and it has a **matching potential** (§4) so the force stays conservative and energy is
still conserved. Set `ε` small compared with typical separations.

### 1.2 The vectorized computation (why NumSharp)

The naïve evaluation is an `O(N²)` double loop. Instead we form the whole pairwise interaction as
array operations (`GravitySolver.Acceleration`):

```csharp
var ri   = positions.reshape(n, 1, 3);          // (N,1,3)
var rj   = positions.reshape(1, n, 3);          // (1,N,3)
var disp = rj - ri;                             // (N,N,3)  broadcast subtraction: dispᵢⱼ = rⱼ − rᵢ
var r2   = np.sum(disp * disp, axis: 2) + ε²;   // (N,N)    softened squared separation
var invR3 = np.power(r2, -1.5);                 // (N,N)    1 / (r²+ε²)^{3/2}
np.fill_diagonal(invR3, 0.0);                   //          a body exerts no force on itself
var wm   = invR3 * masses.reshape(1, n);        // (N,N)    scale column j by mⱼ
var acc  = np.sum(wm.reshape(n, n, 1) * disp, axis: 1) * G;  // (N,3)  contract over j
```

One broadcast subtraction builds the entire `N×N×3` displacement tensor; two reductions collapse it
to accelerations. The C# side issues ~7 array calls regardless of `N`. This is the payoff of an
array library for physics: the code reads like the equation.

> **Complexity note.** This is a direct all-pairs solver — `O(N²)` time and memory — ideal for the
> interactive few-hundred-body regime. It is deliberately not a Barnes–Hut tree code.

---

## 2. Symplectic integration (leapfrog)

Given the accelerations, we advance time. The default is **leapfrog / velocity-Verlet** in
kick–drift–kick form (`Integrator.StepLeapfrog`):

```
v(t + ½Δt) = v(t)      + ½ Δt · a( r(t) )          // kick
r(t + Δt)  = r(t)      + Δt · v(t + ½Δt)           // drift
v(t + Δt)  = v(t + ½Δt) + ½ Δt · a( r(t + Δt) )    // kick
```

Leapfrog is **second order**, **time-reversible**, and **symplectic** — it exactly integrates a
Hamiltonian slightly perturbed from the true one, so its energy error does not accumulate; it
oscillates in a bounded band forever. This is *the* reason it (and its cousins) are used in real
N-body and molecular-dynamics codes. It costs only **one new force evaluation per step**, because
the trailing `a(r(t+Δt))` is cached and reused as the next step's leading kick.

### 2.1 Why not just use RK4?

Classical **RK4** is fourth-order — more accurate *per step* — but it is **not symplectic**, so its
energy error **grows secularly**: an orbit slowly spirals in or out. The `Verification` harness
measures both on an eccentric orbit and prints the signature:

```
leapfrog: first-quarter max |ΔE/E₀| = 1.85e-04, last-quarter = 1.85e-04   → bounded (ratio 1.00)
rk4:      first-quarter max |ΔE/E₀| = 2.41e-08, last-quarter = 8.60e-08   → growing (×3.6)
```

Read that carefully: RK4's error is *four orders of magnitude smaller* — and yet it is the wrong
tool, because it **grows**. Over a long run leapfrog wins by staying bounded. Press `I` in the
sandbox to switch schemes and watch the total-energy readout.

---

## 3. Conserved quantities — the built-in instrument

A correct closed gravitating system conserves three things exactly (up to integration error). The
HUD shows them live; watching them *not drift* is the proof the simulation is faithful. All are
vectorized reductions in `ConservedQuantities`:

| Quantity | Formula | Conserved because |
|---|---|---|
| Linear momentum | **p** = Σ mᵢ**vᵢ** | no external force |
| Angular momentum | **L** = Σ mᵢ (**rᵢ** × **vᵢ**) | no external torque |
| Total energy | `E = KE + PE` | forces are conservative |
| Centre of mass | **R** = (Σ mᵢ**rᵢ**) / Σmᵢ | moves at constant velocity |

Momentum and angular momentum hold to **machine precision** regardless of the integrator (the
harness measures `|Δp| ≈ 1e-14`), because they follow from the *symmetry* of the pairwise force, not
from the integration accuracy. Energy holds to the integrator's error band.

---

## 4. Energy, and the softened potential

```
KE = ½ Σ mᵢ |vᵢ|²

              mᵢ mⱼ
PE = −G  Σ   ─────────────           (softened, matching the softened force)
        i<j  √(rᵢⱼ² + ε²)
```

The softening length in the potential is the **same** `ε` as in the force — that is what makes the
softened force the gradient of a real potential and therefore conservative. Measuring energy with a
different `ε` would show a false drift. The kinetic term is one reduction; the potential is built
from the same pairwise displacement tensor as the force, summed over the upper triangle.

---

## 5. General relativity: Mercury's precession

Newtonian gravity predicts **closed** ellipses. Reality does not: Mercury's orbit's long axis
rotates by 43 arc-seconds per century more than Newton (plus planetary perturbations) can explain —
the discrepancy that Einstein's general relativity resolved in 1915.

To leading (first post-Newtonian, "1PN") order, the correction can be written as a `1/r³` term in
the effective radial potential, `ΔV = −G M h² / (c² r³)`, where `h = |Δr × Δv|` is the specific
angular momentum and `c` the speed of light. Its gradient adds to the Newtonian force, giving a
total radial acceleration magnitude

```
          G M ⎛      3 h²  ⎞
|a| =    ────── ⎜ 1 + ──────⎟          (ForceModel.RelativisticPrecession)
           r²  ⎝      c² r² ⎠
```

so the solver simply multiplies each pair's Newtonian weight by `(1 + 3h²/(c²r²))`. `h²` is computed
without an explicit cross product via **Lagrange's identity** `|u×v|² = |u|²|v|² − (u·v)²`, keeping
everything as three vectorized reductions (`GravitySolver.RelativisticFactor`).

This perturbation makes the perihelion advance by a known amount per orbit:

```
        6 π G M
Δϖ = ─────────────────         (leading-order Schwarzschild precession)
      c² a (1 − e²)
```

The `Verification` harness integrates an eccentric orbit (`a=1, e=0.5, c=10`), detects successive
perihelion passages, and compares:

```
precession/orbit measured 0.2809 rad  vs  analytic 0.2513 rad   (within tolerance)
```

The measured advance matches the analytic Schwarzschild rate — the 1PN term is not "some extra
force", it is the *right* one. In the sandbox the `Relativistic Precession` scenario dials `c` far
below its real value so the effect that takes Mercury a century is visible in seconds; press `G` in
any scenario to toggle it.

---

## 6. The three-body figure-eight

The `Three-Body Figure-Eight` scenario is the **Chenciner–Montgomery choreography** (2000): three
equal masses that chase each other forever around a single figure-eight curve — a genuine periodic
solution of the otherwise-chaotic three-body problem. Its initial conditions are exact constants,
and the period is `T ≈ 6.32591`. The harness integrates one full period and checks that body 0
returns to within `5e-5` of its start — a stringent, non-trivial test that the force law and
integrator are both correct (a wrong sign or a drifting scheme destroys the pattern within one
period).

---

## 7. Collisions and accretion

The `Accreting Cluster` scenario enables **perfectly inelastic merging**: when two bodies' centres
approach within the sum of their radii, they combine into one body that

- conserves **total mass** exactly, `M = Σ mₖ`,
- conserves **linear momentum** exactly, so its velocity is the mass-weighted mean
  `V = (Σ mₖ vₖ) / M`,
- sits at the group's **centre of mass**, and
- grows by **volume** (`r = (Σ rₖ³)^{1/3}`, constant density), so a merge looks like accretion.

Touching groups are resolved with union-find so a chain of contacts collapses correctly in one pass
(`NBodyWorld.CollisionPass`). The harness verifies mass and momentum balance across a head-on merge
to machine precision.

---

## Units

Gravitational dynamics has no absolute scale; each scenario picks an internally-consistent unit
system, stated in `ScenarioLibrary`:

| Scenario | Length | Mass | Time | `G` |
|---|---|---|---|---|
| Solar System | AU | solar mass | year | `4π²` |
| Two-body / figure-eight / binary / cluster | dimensionless | dimensionless | dimensionless | `1` |
| Relativistic precession | dimensionless | dimensionless | dimensionless | `1` (with `c` finite) |

The astronomer's system (`G = 4π²`) is the tidy one: it makes Earth's circular speed exactly `2π`
AU/yr and its period exactly one year. `G` must match the masses and distances — mixing units does
not error, it silently rescales time.

---

## Where each piece lives

| Concept | File |
|---|---|
| State arrays `(N,3)/(N,3)/(N,)` | `Assets/Scripts/Physics/NBodyState.cs` |
| Vectorized force (§1, §5) | `Assets/Scripts/Physics/GravitySolver.cs` |
| Integrators (§2) | `Assets/Scripts/Physics/Integrator.cs` |
| Conserved quantities (§3, §4) | `Assets/Scripts/Physics/ConservedQuantities.cs` |
| Merging (§7) | `Assets/Scripts/Physics/NBodyWorld.cs` |
| Scenarios & units | `Assets/Scripts/Physics/ScenarioLibrary.cs` |
| The proofs | `Verification/Program.cs` |
