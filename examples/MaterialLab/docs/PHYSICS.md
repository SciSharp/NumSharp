# Material Lab — physics and numerics

This is the reference for what the simulator computes and why each choice was made. File names point into
[`../Simulation/`](../Simulation/). Every number quoted as "measured" comes from the verification gate
([`../Verification/Program.cs`](../Verification/Program.cs)) or from the profiling done while building it.

---

## 1. Units and scale

- The tank is **1 unit tall** and 16/9 units wide. The grid has `Ny` rows (72 / 90 / 108 / 126 by quality) and
  `Nx = round(Ny·16/9)` columns; the cell size is **Δx = 1/Ny**. Row `j` is height `j·Δx`, y up.
- Density is relative to water (water = 1). Moduli are in density × (units/s)², so a wave speed is
  √(modulus/density) units per second.
- Gravity is **g = 110 units/s²** and each rendered frame advances **2.8 ms** of simulated time. Played at
  60 FPS, a body falls one tank height in ~0.8 s of wall time — the dynamics of a tank about **3 m tall seen in
  real time**.
- Every particle represents a quarter cell: **V₀ = Δx²/4**, mass m = ρ·V₀. The rest packing is one particle per
  quarter-cell *slot*, 2×2 per cell, for every material.

---

## 2. The MLS-MPM step

The solver is the moving-least-squares material point method (Hu et al. 2018) with affine (APIC) transfers
(Jiang et al. 2015). Particles carry the state; a background grid is rebuilt every substep to compute forces.

**Weights** (quadratic B-spline). For a particle at x/Δx, `base = ⌊x/Δx − ½⌋` and `f = x/Δx − base ∈ [½, 3/2)`:

```
w₀ = ½(3/2 − f)²      w₁ = ¾ − (f − 1)²      w₂ = ½(f − ½)²
```

in each axis; a node (i, j) of the 3×3 stencil gets `w = wₓ(i)·w_y(j)`.

**Particle → grid** (`Transfers.P2GFused`, `Transfers.P2GReference`). Each node accumulates

```
m_i  += w·m_p
p_i  += w·(m_p·v_p + A_p·(x_i − x_p))          A_p = −Δt·V₀·(4/Δx²)·τ_p + m_p·C_p = −Δt·τ_p + m_p·C_p
V_i  += w·V₀                                    (volume — occupancy and mixture density)
c_i  += w·m_p   if the particle is water        (coolant — lava quenching)
```

with τ the particle's Kirchhoff stress and C its APIC velocity gradient. The factor 4/Δx² is the inverse of the
quadratic kernel's inertia tensor, and V₀·4/Δx² = 1 for V₀ = Δx²/4. Since Σ_i w·(x_i − x_p) = 0 for this kernel,
the affine term moves momentum around but adds none: **P2G conserves mass and linear momentum exactly** —
measured relative error ≤ 2·10⁻⁹ (float round-off) for both implementations.

The node array is `(Ny, Nx, 8)` float32, interleaved — `[mass, px, py, volume, coolant, wall-volume, –, –]` —
so the fused kernel moves one node with one 256-bit load, fused multiply-add and store.

**Grid update** (`GridUpdate`). `v_i = p_i/m_i + Δt·g` (g in any direction — the tank tilts), blended toward
the mouse's drag velocity inside the stir disk, then walls (§7).

**Grid → particle** (`Transfers.G2PFused`). `v_p = Σ w·v_i` and `C_p = (4/Δx²)·Σ w·v_i·(x_i − x_p)ᵀ`; the same
gather returns the local mass, volume, wall volume and coolant, from which the particle gets its occupancy (§5),
the mixture density (§6) and its water fraction (§8). Then `x_p += Δt·(v_p + drift)`, a push-out from walls, and
lava cooling.

**Time step** (`MpmWorld.StableDt`). An explicit elastic update is stable only if a wave does not cross a cell
per substep, so

```
Δt = min( CFL·Δx / c_max ,  ½Δx / |v|_max )        CFL = 0.85
```

where `c_max` is the fastest wave speed among the materials PRESENT and `|v|_max` the fastest particle of the
previous frame. The frame is split into `⌈2.8 ms / Δt⌉` substeps (8–21 in the scenes, at most 36). Wave speeds:

| Material | Wave speed (units/s) | | Material | Wave speed |
|---|---:|---|---|---:|
| water `√(K/ρ)` | 38.7 | | sand `√((λ+2μ)/ρ)` | 54.3 |
| oil | 39.1 | | snow (at 2× hardening) | 55.3 |
| honey | 37.8 | | jelly | 46.5 |
| lava | 36.5 | | rubber / clay / rock | 58.0 / 56.0 / 57.7 |

The half-cell rule is the safety net for user input: a particle flung by the mouse can never skip a wall.

---

## 3. The closed-form 2×2 SVD ("σ-space")

Plasticity needs the singular values of the trial deformation gradient `T = (I + Δt·C)·F`. In 2-D they have a
closed form without trigonometry (`Constitutive.BuildSigmaSpace`). With `S = T·Tᵀ = [[p, q], [q, r]]`:

```
D  = √((p − r)² + 4q²)          σ₁ = √((p + r + D)/2)          σ₂ = det T / σ₁   (signed: inversion is kept)
cos 2φ = (p − r)/D              sin 2φ = 2q/D                  (φ = angle of the left rotation U)
```

and any `U·diag(k₁, k₂)·Uᵀ = h·I + g·[[cos 2φ, sin 2φ], [sin 2φ, −cos 2φ]]` with `h = (k₁+k₂)/2`,
`g = (k₁−k₂)/2`. That single identity gives both the plastic return `F = U·diag(σ′/σ)·Uᵀ·T` and the stress
`τ = U·diag(t₁, t₂)·Uᵀ` as plain arithmetic — which is what lets every material run as fused array expressions.

---

## 4. Constitutive models (`Constitutive.cs`)

| Material | Model | Parameters |
|---|---|---|
| Water | fluid | ρ 1.0, K 1500, μ 0, tension 0 |
| Oil | fluid | ρ 0.72, K 1100, μ 0.02, tension 0 |
| Honey | fluid | ρ 1.4, K 2000, μ 1.0 (capped, below), tension 0.002 |
| Lava | fluid, thermal | ρ 1.8, K 2400, μ 0.01·(1 + 200(1−T)²) (capped), tension 0.002 |
| Sand | Drucker–Prager | ρ 1.6, E 3500, ν 0.3, friction angle 35° |
| Snow | snow plasticity | ρ 0.4, E 550, ν 0.2, θc 0.025, θs 0.0075, ξ 10, hardening ≤ 2 |
| Jelly | fixed corotated | ρ 1.0, E 1800, ν 0.25 |
| Rubber | fixed corotated | ρ 1.2, E 3000, ν 0.3 |
| Clay | von Mises | ρ 1.5, E 3500, ν 0.3, yield strain 0.015 |
| Rock | fixed corotated | ρ 1.8, E 5000, ν 0.25 |

The moduli are "film-real": stiff enough that liquids stay within a few percent of their volume and solids
read as solid, soft enough that the whole tank runs in real time (a literal granite would need thousands of
substeps per frame).

**Fluids.** `J ← clamp(J·(1 + Δt·tr C), 0.6, 1 + t)` (J from §5), then

```
τ = K·J·(J − 1)·I + μ·J·(C + Cᵀ)
```

— the equation-of-state pressure plus Newtonian viscosity. The explicit update carries viscosity only up to the
diffusion limit `ν·Δt/Δx² ≤ 0.2`, so every viscosity is clamped to `0.2·ρ·Δx²/Δt` at the current step (a 0-d
parameter array per material, updated each frame — honey never destabilizes a coarse grid). Lava's viscosity
multiplies by `1 + 200·(1 − T)²` as it cools, capped the same way: molten lava runs, crusting lava crawls.

**Fixed corotated** (jelly, rubber, rock). `τ = 2μ·(T − R)·Tᵀ + λ·J·(J − 1)·I` with R the polar rotation of T,
which in 2-D is the normalized vector `(T₀₀ + T₁₁, T₁₀ − T₀₁)` — no SVD. No plasticity: the trial gradient becomes
the new F.

**Snow** (Stomakhin et al. 2013). The singular values are clamped to `[1 − θc, 1 + θs]`; the clipped part is
plastic flow, recorded in `Jp ← clamp(Jp·σ₁σ₂/(σ₁ᶜσ₂ᶜ), 0.3, 3)`. The elastic part is fixed-corotated, stiffened
by `exp(ξ·(1 − Jp))` (clamped to [0.1, 2]): compacted snow is harder, stretched snow crumbles. Measured: a
snowball hitting the floor compacts 96 % of its particles, down to Jp = 0.3.

**Sand** (Klár et al. 2016). On the Hencky strain ε = ln σ with St. Venant–Kirchhoff–Hencky stress
`τᵢ = 2μ·εᵢ + λ·(ε₁ + ε₂)`:

- In tension (`tr ε ≥ 0`) grains separate: project to ε = 0, no stress.
- In compression, the allowed shear grows with pressure (friction): `δγ = ‖ε̂‖ + β·tr ε` with
  `β = α·(λ + μ)/μ` and `α = √(2/3)·2 sin φ / (3 − sin φ)`; if `δγ > 0` the deviatoric part shrinks by δγ.

**Clay** (von Mises). The deviatoric strain magnitude is capped at the yield strain — pressure-independent, so
clay dents and keeps the dent. Measured: dropped from the same height, a jelly block ends at height/width 0.99,
a clay block at 0.09.

**Implementation.** Each model is a list of 5–25 fused `NDExpr` stages over the particle group's buffers,
compiled once (`expr.Compile()`) and evaluated in place (`Evaluate(out)`) every substep. Δt, the stress scale
and the viscosity caps are 0-d NumSharp arrays — hoisted parameters, not constants — so an adaptive Δt never
recompiles. Two measured rules shape the stage lists:

- A fused tree has no common-subexpression elimination, and `Log`/`Exp` cost ~1.1–1.4 ns per element against
  ~0.2 ns for plain SIMD arithmetic. The Drucker–Prager projection written inline evaluates ln σ about ten times
  (14.3 ns/element). So every transcendental is written once into a scratch lane and read back as an operand —
  the granular update got **3.5× faster**.
- Magnitudes are `Sqrt(a² + b²)`, not `Hypot`: the overflow-safe `Hypot` measured **~19× slower** on values that
  cannot overflow.

---

## 5. Liquid volume from occupancy

**The problem.** The textbook MPM liquid tracks its volume ratio by integrating the velocity divergence,
`J ← J·(1 + Δt·tr C)`. Two things break it in a real-time mixing tank:

1. *Clamping erases memory.* J must be clamped (an unbounded J in a splash produces runaway tension), and every
   clamp forgets how stretched the particle was. When it recompresses, the forgotten stretch reads as
   over-compression, the pressure pushes outward, and the liquid slowly **gains volume** — a pool that domes up
   and turns cloudy.
2. *The drift flux (§6) moves particles without a velocity gradient*, so J never sees the compression it causes.

**The fix: J = 1/φ.** In G2P each liquid particle gathers the local occupancy

```
φ = Σ_i w·(V_i + V_wall,i) / Δx²            (1 = rest packing)
```

and sets `J = 1/φ`. There is nothing to drift: every material's volume counts (sand displaces water — which is
exactly buoyancy), and a free surface naturally reads φ < 1. The constitutive stage then advances this J over
the substep (`J·(1 + Δt·tr C)`) before computing the pressure.

**Walls must read as full.** A particle next to a wall has part of its kernel inside the wall, where there are no
particles, so it reads φ < 1 and feels no pressure until the liquid has packed itself denser against the wall.
Measured, that alone cost a resting pool ~10 % of its mean height. So each node carries a **wall volume**
(`MpmGrid.LaneWall`), constant between wall edits:

```
V_wall,i = Δx²·C(½ − φ_i)          C(t) = ∫₋∞ᵗ N(s) ds,  the quadratic B-spline's cumulative weight
```

where φ_i is the node's distance to the wall surface in cells. Particles are kept at least ½ cell from a wall, so
at rest packing a node at distance φ from a flat wall collects `C(φ − ½)·Δx²` of particle volume, and the wall
lane supplies exactly the complement. `C` is piecewise cubic: `(t + 3/2)³/6` on [−3/2, −1/2],
`½ + ¾t − t³/3` on [−1/2, 1/2], `1 − (3/2 − t)³/6` on [1/2, 3/2]. The grid clear restores this lane from a
template (one block copy, the cost of a clear), and nothing else writes it.

**Measured:** a pool 0.351 deep, left to settle, is compressed **1.6 %** in mean height; the equation of state
predicts `ρ·g·h/(2K)` = **1.3 %**. Its surface is flat to 0.19 cells and its fastest particle moves 0.23 units/s.

**Tension.** Because every free-surface particle reads φ ≈ 0.6 (J ≈ 1.7), the upper clamp `1 + t` applies to all
of them at once and pulls the surface layer together with `K·(1 + t)·t` — the tensile instability of particle
methods (Swegle et al. 1995): the surface clumps back to full density, grows a crust of bumps, and the
over-packed clumps kick droplets out. Liquids a tank in size have negligible surface tension anyway, so water and
oil use t = 0 and honey/lava a rope-like 0.002.

**Fills never over-pack.** Placing material on top of material is now a compressed spring (a region packed twice
reads J = ½), so every fill — scene builders, the brush, faucets — adds a particle only to an empty quarter-cell
slot, counted from the particles' current positions (`MpmWorld.Fill`). Scene builders place the enclosed material
first (the oil blob, then the water around it). Measured: filling a rectangle twice adds 0 the second time; a
half-overlapping rectangle adds 1.02× the free half.

---

## 6. Mixtures: buoyancy at two scales

MPM gives all material at a node one velocity. That is exactly right for a **resolved** blob — a lighter region
accelerates upward on its own, because the pressure gradient is set by the surrounding liquid; this is where the
mushroom-shaped oil plumes of F6 come from. It cannot separate what is mixed **below** the grid scale (a stirred
emulsion, sand in suspension): once mixed, it would stay mixed.

So the particle update adds a **drift flux** (after Ishii's drift-flux mixture model):

```
v_d = τ·(ρ_p − ρ_mix)/ρ_p · g · s(φ)          ρ_mix = Σ w·m / Σ w·V     (gathered from the grid)
```

- A pure region (`ρ_p = ρ_mix`) does not drift. Per node, `Σ m·v_d = τ·g·Σ V·(ρ_p − ρ_mix) = 0`: the drift moves
  material without creating momentum. It is applied to positions only.
- `s(φ) = clamp((φ − 0.6)/0.4, 0, 1)` fades it out toward a free surface. Without it, a light particle at the top
  of a thin layer — which still "sees" the heavy liquid under it through the 3-cell kernel — is driven straight
  out of the liquid, over and over (beads hopping on an oil slick).
- τ is **small** (water/oil 0.006 s, honey/lava 0.004, sand/snow 0.008). Every blob's boundary layer also reads
  a mixed density, so a large τ strips that layer off faster than the blob moves: with τ = 0.03 a rising oil blob
  dissolved into an orange cloud instead of rising as a plume.

Measured on F6: oil released at the bottom ends as a layer on top (mean height 0.437, all of it at the surface),
honey dropped in ends on the bottom (mean height 0.059), water between (0.231).

---

## 7. Walls (`MpmGrid.cs`)

- **Analytic shapes.** Disks, capsules (thick segments, used for painted strokes, ramps and ledges) and
  rounded boxes, each with an exact signed distance. Adding a solid is `φ ← min(φ, sdf)`, carving is
  `φ ← max(φ, −sdf)` (then the tank walls are re-imposed). φ is clamped to ±8 cells, so an edit touches only its
  shape's bounding box plus a band — painting with the mouse stays cheap. A ramp is a true straight ramp, not a
  staircase of cells, so liquid slides along it smoothly.
- **Two resolutions.** φ at the grid nodes drives the physics; a 4× finer copy drives the renderer's crisp stone.
- **Grid boundary.** At nodes with φ < 1 cell, velocity INTO the wall (`vₙ < 0`) is removed — material may leave a
  wall freely, so splashes lift off the floor — and Coulomb friction (μ = 0.5) removes tangential speed in
  proportion to the normal speed cancelled: `v_t ← v_t·max(0, 1 + μ·vₙ/|v_t|)`. That keeps sand piles standing
  while water runs down a ramp.
- **Particle push-out.** After advection, a particle closer than ½ cell to a surface is moved out along the
  normal and loses its inward velocity. A nearest-node φ above 1.5 cells proves no push-out is needed, so most
  particles skip the bilinear sample.

Measured: a shelf painted across the tank holds 5,883 water particles with none leaking below its face.

---

## 8. Lava

Lava particles carry a temperature T ∈ [0, 1]. Each substep

```
T ← max(0, T − Δt·(0.35 + 60·c))            c = water mass fraction around the particle (gathered)
```

— air is a poor coolant (a flow stays molten ~2.5 s and runs down a slope, crusting as its viscosity climbs),
water a violent one. Below T = 0.12 a lava particle becomes **rock** (same position and velocity, now an elastic
solid). The renderer maps T through a blackbody-like ramp (dull red → orange → yellow-orange) under a cracking
crust that darkens as it cools.

---

## 9. The two transfer implementations

- **Reference** (`TransferMode.Reference`, key **T**): pure NumSharp. The stencil of every particle is built as
  broadcast `(3, 3, N)` weight and index arrays by fused expressions; P2G is `np.bincount(index, weights, G)`
  per lane; G2P is `np.take` of the 9 nodes into `(9, N)` arrays reduced by fused axis-0 sums. Readable line by
  line — the specification.
- **Fused** (default): the same algorithm in one pass per particle with `Vector256` over the 8 node lanes,
  directly on the NumSharp buffers. G2P reduces the stencil row by row — per row, with t = wₓ₁·n₁,
  `s = wₓ₀·n₀ + wₓ₂·n₂ + t` and the first moment `m = 2wₓ₂·n₂ + t` — so the value and both moments cost nine loads
  and about twenty fused multiply-adds (the direct form spilled AVX2's 16 registers and measured 13 % slower).
  Both kernels are compiled fully optimized from the first call (`AggressiveOptimization`) so the app does not
  run through tiered-JIT stepping stones during its first second.

Verified to agree to float rounding: after 60 substeps of every material interacting, positions differ by at
most 1.7·10⁻⁵ cells and J by 8·10⁻⁶. The fused transfers are **9.5–28× faster** than the reference ones at High
quality (it never materializes the 9N-element stencil, and the reference P2G builds grid-sized bincounts per
material group): whole simulation steps measure Dam Break 7.2 vs 56 ms, Oil/Water/Honey 13.6 vs 128 ms,
Hourglass 6.1 vs 43 ms, Material Zoo 9.2 vs 173 ms. In the default mode NumSharp operations (constitutive update,
grid update, sort) are 12–46 % of the step; in the reference mode about 98 % — see the README's
"What runs on NumSharp" section for the full split.

---

## 10. Validation against the laboratory

- **Granular column collapse.** Lube, Huppert, Sparks & Freundt (2005) measured the collapse of 2-D granular
  columns: for aspect ratio a = H₀/L₀ > 2.8 the run-out obeys `(L∞ − L₀)/L₀ ≈ 1.9·a^(2/3)` and the final height
  `H∞/L₀ ≈ a^(2/5)`. The lab's sand, with no rule written for collapse, lands at **×1.20** of the run-out law and
  **×0.79** of the height law at a = 5.6 on a 12-cell-wide column.
- **Angle of repose.** Sand poured onto one spot builds a cone with flanks at **26°** for a 35° friction angle —
  MPM Drucker–Prager at this resolution flows a little too easily; so does the collapse above (a longer run-out,
  a lower pile).
- **Hydrostatics.** §5: 1.6 % measured vs 1.3 % predicted compression.

The full list of 27 checks and their latest values is in the [README](../README.md#verification).

---

## 11. Limitations

- **Explicit time integration.** Viscosity is capped at the explicit diffusion limit, ~0.1–0.15 at High quality:
  honey flows in seconds (real honey is 10⁴× more viscous than water). Stiffness is capped by the CFL condition,
  so rock is a stiff rubber, not granite.
- **Weakly compressible liquids.** K = 1500 gives ~1.3 % hydrostatic compression and density swings of up to
  ~10 % in the most violent surges (Mach ≈ 0.3).
- **No surface tension**, no heat conduction beyond lava's cooling (lava does not melt snow; no steam).
- **Sand is slightly too fluid** (26° cones for 35°; run-out 20 % long).
- **Drift flux is a mixture model**, not a multiphase solver: resolved phases share one velocity per node.
- **2-D, single-threaded** — NumSharp's kernels are single-threaded by design, and so is this app.

---

## References

- Y. Hu, Y. Fang, Z. Ge, Z. Qu, Y. Zhu, A. Pradhana, C. Jiang. *A moving least squares material point method
  with displacement discontinuity and two-way rigid body coupling.* ACM Trans. Graph. 37(4), 2018.
- C. Jiang, C. Schroeder, A. Selle, J. Teran, A. Stomakhin. *The affine particle-in-cell method.* ACM Trans.
  Graph. 34(4), 2015.
- A. Stomakhin, C. Schroeder, L. Chai, J. Teran, A. Selle. *A material point method for snow simulation.* ACM
  Trans. Graph. 32(4), 2013.
- G. Klár, T. Gast, A. Pradhana, C. Fu, C. Schroeder, C. Jiang, J. Teran. *Drucker–Prager elastoplasticity for
  sand animation.* ACM Trans. Graph. 35(4), 2016.
- G. Lube, H. E. Huppert, R. S. J. Sparks, A. Freundt. *Collapses of two-dimensional granular columns.* Phys.
  Rev. E 72, 041301, 2005.
- M. Ishii. *Thermo-fluid dynamic theory of two-phase flow.* Eyrolles, 1975 (the drift-flux model).
- J. W. Swegle, D. L. Hicks, S. W. Attaway. *Smoothed particle hydrodynamics stability analysis.* J. Comput.
  Phys. 116, 1995 (the tensile instability).
