# NumSharp Physics Games — Rundown

Two complete, self-contained Unity example games were built in `examples/`, each with its **entire
physics engine computed by NumSharp** (not Unity). Both are structured so the physics is
Unity-independent and **machine-verified without a Unity license**, and both ship a standalone way to
run them outside Unity.

| Example | What it is | Physics |
|---|---|---|
| [`UnityGravitySandbox/`](UnityGravitySandbox/) | Interactive N-body orbital-mechanics sandbox | Newtonian gravity + symplectic integration + a relativistic (1PN) correction |
| [`UnityFallingSand/`](UnityFallingSand/) | Falling-sand / powder game (sand, water, oil, smoke, walls) | Mass-conserving cellular automaton with density stratification |

---

## 1. The shared architecture

Both examples use the same three-layer split, which is what makes the physics testable and reusable:

```
UnityProject/Assets/Scripts/
├── Physics/ or Simulation/   ← pure C# + NumSharp, NO UnityEngine dependency  (the "backend to reality")
└── Game/                     ← the Unity view (UnityEngine): MonoBehaviours, rendering, input
Verification/                 ← standalone console app: references NumSharp.Core, compiles the physics
                                 folder, and asserts the science. A green run == the physics is correct.
Player/                       ← (falling-sand only) a Unity-free way to actually run/render the game
```

- The **view is a pure function of the physics state** — one self-bootstrapping MonoBehaviour per game
  builds its own camera, lights, materials and bodies from code, so there are **no scenes, prefabs, or
  material assets** to set up. Drop the component on an empty GameObject and press Play.
- The **Verification** project pulls the physics in with
  `<Compile Include="..\UnityProject\Assets\Scripts\{Physics|Simulation}\**\*.cs" />`, so it exercises the
  exact same source Unity runs.
- The Unity C# cannot be compiled without the editor (there is none in the dev environment here), so it
  was **type-checked against a faithful `UnityEngine` shim**; the physics it drives is what's
  machine-verified.

---

## 2. UnityGravitySandbox (N-body gravity)

An interactive orbital sandbox: six scenarios, a fly-around camera, slingshot-spawn new bodies, switch
integrators, and a live conserved-quantity HUD that proves the simulation is faithful.

**The NumSharp showpiece** — the gravitational acceleration on every body is one broadcast subtraction
plus two reductions, no per-body loop:

```csharp
var disp  = positions.reshape(n,1,3) - positions.reshape(1,n,3);   // (N,N,3) all pairwise displacements
var invR3 = np.power(np.sum(disp*disp, axis:2) + eps², -1.5);      // (N,N) softened 1/r³
np.fill_diagonal(invR3, 0.0);                                      // no self-gravity
var acc   = np.sum((invR3*masses.reshape(1,n)).reshape(n,n,1)*disp, axis:1) * G;  // (N,3)
```

**Scenarios:** two-body circular orbit · the Chenciner–Montgomery three-body **figure-eight** · the
**solar system** (real relative masses, AU / M☉ / year units, G = 4π²) · a binary star + circumbinary
planet · an accreting cluster · **relativistic perihelion precession** (Mercury's effect, exaggerated).

**Verification** (`cd UnityGravitySandbox/Verification && dotnet run -c Release`) — **all 25 checks pass**:
- Leapfrog energy conservation: two-body max |ΔE/E₀| = **1.0e-9**; figure-eight = **1.5e-7**.
- Linear & angular momentum conserved to machine precision (**~1e-14**).
- The figure-eight returns to its start within **5.4e-5** after one period T≈6.3259.
- Spawn + head-on merge conserve mass exactly and momentum to **4.5e-15**.
- Relativistic precession measured **0.281 rad/orbit** vs the analytic Schwarzschild
  `6πGM/(c²a(1-e²))` = **0.251** — the 1PN term reproduces the right effect.
- Integrator contrast: symplectic leapfrog's energy error stays **bounded** (ratio 1.00) while RK4's
  **grows** (2.4e-8 → 8.6e-8) despite better per-step accuracy — why leapfrog is the default.

**Docs:** [`UnityGravitySandbox/docs/PHYSICS.md`](UnityGravitySandbox/docs/PHYSICS.md).

---

## 3. UnityFallingSand (powder game)

Paint sand, water, oil, smoke and walls with the mouse and watch them fall, pile, pool, float, sink and
rise — the whole world is a cellular automaton stepped through vectorized NumSharp array ops.

**The model** (full derivation in [`UnityFallingSand/docs/PHYSICS.md`](UnityFallingSand/docs/PHYSICS.md)):
- **One master rule — density.** A cell sinks past a strictly-lighter cell below it (Archimedes). The
  ordering `Smoke(-1) < Empty(0) < Oil(1) < Water(2) < Sand(3)` makes sand sink, oil float on water, and
  smoke **rise** (negative density → the air above it is heavier) — all from one comparison.
- **Granular vs fluid.** Sand slides diagonally and **piles** (~45°); water/oil/smoke spread horizontally
  and **level**. Same swap primitive, different selection mask.
- **Vectorized AND mass-conserving.** The naive "copy every cell down" duplicates/loses material on
  collisions. `GridOps.ApplySwap` uses a conflict-free swap — `allowed = move AND NOT move(neighbour)` —
  which is provably a permutation of the grid, so nothing is created or destroyed.
- **Water leveling needs randomized flow.** A lone surface cell with a fixed left/right preference
  oscillates forever and freezes water into a sand-like ramp; a **random per-cell flow direction** turns
  that into a random walk that finds a drop and levels out. (This bug was hit and fixed during
  development — see PHYSICS.md §4.)

**Verification** (`cd UnityFallingSand/Verification && dotnet run -c Release`) — **all checks pass**:
- **Mass conservation**: all 6 material counts invariant over 300 steps on a busy random grid.
- Sand settles to the floor; **density stratification** smoke < oil < water < sand.
- Water levels: a 12-tall column pools into 2 rows (its flat depth).
- Disk brush paints ~πr² cells centred; emitter (faucet) adds mass and its stream reaches the floor.
- Determinism: same seed → bit-identical grid. Boundary walls contain loose material.

### 3a. Resolution & the high-pixel render

The game runs at **1000×500** (500,000 cells) by default. A terminal can't show that as text, so the
Unity view (GPU) and the standalone Player render it at full resolution:

- **Measured cost: ~31 ms/step at 1000×500** (~3.4 ms at 300×160). Unity defaults to 1 substep/frame ≈
  **32 fps**; lower the resolution or raise substeps to taste. Offline PNG rendering is full-res regardless.
- A real **1000×500 PNG** (settled density layers) is committed at
  [`UnityFallingSand/docs/settled-1000x500.png`](UnityFallingSand/docs/settled-1000x500.png).
- Browser-viewable, zoomable high-res gallery of the four stages:
  **https://claude.ai/code/artifact/d6962c84-11fa-4d50-941a-00e73ffb8177**

### 3b. Play / render it now (no Unity)

```bash
cd UnityFallingSand/Player
dotnet run -c Release -- --png out.png                 # native 1000×500 PNGs (the high-pixel output)
dotnet run -c Release -- --png big.png --scale 2       # 2000×1000 pixels
dotnet run -c Release -- --play                        # interactive: WASD move · Space pen · 1-6 material · E faucet · Q quit
dotnet run -c Release -- --demo                        # self-driving colour terminal showcase (downscaled preview)
dotnet run -c Release -- --ascii                       # plain-text showcase
dotnet run -c Release -- --bench --width 1000 --height 500   # time the step cost
```

`--width`/`--height` set the sim size; terminal modes downscale a big grid to a readable preview.

---

## 4. Getting NumSharp into Unity (the one real requirement)

Unity **is** installed here (`2022.3.62f3` and `6000.4.3f1`). To run either game in the editor:

1. Build the backend: `dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0`, and
   copy `NumSharp.dll` into `<example>/UnityProject/Assets/Plugins/NumSharp/` (already staged there).
2. Set **Player ▸ Scripting Backend** to **.NET / CoreCLR** (Unity 6.2+). NumSharp targets .NET 8 and
   emits IL kernels at runtime (`System.Reflection.Emit`), so it needs a JIT-capable modern-.NET backend:
   **Mono** can't load the net8 DLL and **IL2CPP** (AOT) can't emit the kernels.
3. Ensure **Active Input Handling** includes the legacy Input Manager.
4. Add the game component (`GravitySandbox` or `FallingSandGame`) to an empty GameObject and press Play.

---

## 5. Commits (this work, on `master`)

| Commit | Summary |
|---|---|
| `77e419b0` | Unity gravity-sandbox game with a NumSharp-backed N-body physics engine |
| `2b67f296` | Falling-sand powder game on a NumSharp mass-conserving cellular automaton |
| `6847cd7f` | Standalone terminal Player to launch the falling-sand game without Unity |
| `25e85362` | Falling-sand high-resolution 1000×500 — Unity default + native PNG renderer |

Each example's `Verification` project is in `SciSharp.NumSharp.sln` (under the `examples` folder), plus
the falling-sand `Player`.

---

## 6. Known issues / notes

- **Stray process (dev-environment only).** An earlier interactive `FallingSand.Player` run (PID 10976)
  is holding `NumSharp.dll`, which blocks rebuilding the *in-repo* Player `.exe`. The code is verified via
  an identical-source out-of-repo build; close that process (`taskkill /PID 10976 /F` or Task Manager) and
  the in-repo build works. (Not committed to code; environment state only.)
- **Performance is CPU-bound**, by design: both engines are pure-vectorized NumSharp and allocate several
  full-grid arrays per step. That's the showcase — array math, not hand-loops — and it's ideal for the
  interactive scales here (a few hundred bodies; a 1000×500 grid).
- **Precision.** Both engines run in `double`; the view down-casts to `float`/bytes only for rendering.
- **Unity scripts are not compiled here** (no editor); they were type-checked against a `UnityEngine`
  shim. The physics is the machine-verified part.
