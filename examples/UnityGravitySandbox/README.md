# Gravity Sandbox — a Unity game powered by NumSharp physics

An interactive **N-body gravitational simulator** built as a Unity game whose entire physics engine
— pairwise gravity, symplectic time integration, energy/momentum diagnostics, collisions, even a
general-relativistic correction — is computed by **[NumSharp](../../README.md)** on `(N, 3)` arrays.
Unity does the rendering and input; NumSharp does the science.

It ships with six scenarios: a textbook two-body orbit, the three-body **figure-eight
choreography**, the **solar system** in real astronomer's units, a **binary star** with a
circumbinary planet, an **accreting cluster**, and **relativistic perihelion precession** (Mercury's
43″/century, dialed up so you can watch it). Fly the camera around, slingshot new bodies in, switch
integrators, and watch the conserved-quantity readout hold steady — the on-screen proof that the
NumSharp math is faithful.

> **The physics is verified, with or without Unity.** Because the engine is plain NumSharp (no
> `UnityEngine` dependency), the `Verification/` console project compiles the *exact same physics
> source* Unity runs and asserts the science to machine precision. You can confirm the simulation is
> correct in ten seconds without a Unity license — see [Verify the physics](#verify-the-physics).

---

## Why this is a good NumSharp showcase

Orbital mechanics is the ideal advertisement for an array library. The gravitational acceleration on
every body is one broadcast subtraction plus two reductions:

```csharp
var disp  = positions.reshape(n,1,3) - positions.reshape(1,n,3);  // (N,N,3) all pairwise displacements
var invR3 = np.power(np.sum(disp*disp, axis:2) + ε², -1.5);       // (N,N)   softened 1/r³
np.fill_diagonal(invR3, 0.0);                                     //         no self-gravity
var acc   = np.sum((invR3*masses.reshape(1,n)).reshape(n,n,1) * disp, axis:1) * G;  // (N,3)
```

That is the whole `O(N²)` interaction, with no per-body loop, reading like the physics equation it
is. The full derivation of this and everything else is in **[docs/PHYSICS.md](docs/PHYSICS.md)**.

---

## Verify the physics

No Unity required. From this folder:

```bash
cd Verification
dotnet run -c Release
```

Expected output (abridged):

```
Energy conservation (leapfrog):
  [PASS] two-body max |ΔE/E₀| = 1.002E-009 (< 1e-6)
  [PASS] figure-eight max |ΔE/E₀| = 1.473E-007 (< 1e-4)
Momentum conservation (figure-eight):
  [PASS] |Δp| = 1.666E-014 (< 1e-10)
  [PASS] |ΔL| = 1.971E-015 (< 1e-10)
Figure-eight periodicity:
  [PASS] body 0 returns to within 5.435E-005 of its start after one period (< 5e-3)
Spawn & merge (accretion):
  [PASS] momentum conserved across merge: |Δp| = 4.524E-015 (< 1e-9)
Relativistic perihelion precession:
  [PASS] precession/orbit measured 0.2809 rad vs analytic 0.2513 rad (within 15%)
...
ALL CHECKS PASSED.
```

The harness (`Verification/Verification.csproj`) references `NumSharp.Core` and pulls in the engine
via `<Compile Include="..\UnityProject\Assets\Scripts\Physics\**\*.cs" />`, so it exercises the same
files the Unity game does. If a physics change ever breaks conservation, this goes red.

---

## Architecture

Three layers, cleanly separated so the physics is Unity-independent and testable:

```
UnityProject/Assets/Scripts/
├── Physics/          ← pure C# + NumSharp, NO UnityEngine dependency  ← the "backend to reality"
│   ├── NBodyState.cs         positions/velocities/masses as (N,3)/(N,3)/(N,) NDArrays
│   ├── GravitySolver.cs      vectorized pairwise gravity (+ optional relativistic term)
│   ├── Integrator.cs         symplectic leapfrog / symplectic Euler / RK4
│   ├── ConservedQuantities.cs  energy, momentum, angular momentum, centre of mass
│   ├── NBodyWorld.cs         the facade Unity talks to: step, spawn, merge, reset
│   ├── ScenarioLibrary.cs    the six scenarios, in stated unit systems
│   ├── Scenario.cs / Vector3d.cs   data + a tiny Unity-free vector type
│
└── Game/             ← the Unity view layer (UnityEngine)
    ├── GravitySandbox.cs     ONE self-bootstrapping MonoBehaviour — the whole game
    ├── CameraRig.cs          orbit / pan / zoom camera
    ├── BodyView.cs           a body's sphere + trail + star light
    ├── SandboxHud.cs         the live diagnostics + controls overlay
    └── SandboxRenderKit.cs   runtime materials that work in Built-in AND URP

Verification/         ← standalone console harness (references NumSharp.Core, compiles Physics/)
```

The view is a pure function of the physics state: `GravitySandbox` steps the `NBodyWorld` and mirrors
its bodies into `BodyView`s each frame, rebuilding them when the body set changes (a `Version`
counter, bumped on spawn/merge/reset). There are **no prefabs, scenes, or material assets** — the one
MonoBehaviour creates its camera, lights and bodies from code.

---

## Getting NumSharp into Unity

### 1. Build and drop in the DLL

```bash
dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0
# copy bin/Release/net8.0/NumSharp.dll  ->  UnityProject/Assets/Plugins/NumSharp/NumSharp.dll
```

`NumSharp.Core` is 100 % managed with no external runtime dependencies, so that single DLL is all
you need. (See `Assets/Plugins/NumSharp/PLACE_NUMSHARP_DLL_HERE.md`.)

### 2. Use a JIT-capable, modern-.NET scripting backend

This is the one real constraint. `NumSharp.Core` targets **.NET 8** and generates some compute
kernels as IL at runtime (`System.Reflection.Emit`), so it needs a runtime that is both modern-.NET
*and* JIT:

| Unity scripting backend | Works? | Why |
|---|---|---|
| **`.NET` / CoreCLR** (Unity 6.2+ preview) | ✅ **recommended, this sample's target** | runs `net8.0` assemblies and JITs the IL kernels |
| **Mono** | ⚠️ not with the stock build | Unity's Mono is `.NET Standard 2.1`-era; the `net8.0` DLL references `Vector512`, generic-math, etc. that it lacks |
| **IL2CPP** | ❌ | ahead-of-time; `Reflection.Emit` is unavailable so the kernels can't be generated |

Set it under **Edit ▸ Project Settings ▸ Player ▸ Configuration ▸ Scripting Backend** (choose the
`.NET`/CoreCLR backend on Unity 6.2+). Also ensure **Active Input Handling** includes the **legacy
Input Manager** (the camera and slingshot use `UnityEngine.Input`).

### 3. Play

1. Open `UnityProject/` in Unity 6.2+.
2. Create an empty GameObject (**GameObject ▸ Create Empty**).
3. Add the **Gravity Sandbox** component to it.
4. Press **Play**.

That's it — the script builds the rest of the scene itself.

---

## Playing it

| Key | Action |
|---|---|
| `1`–`6` | load a scenario |
| `Space` | pause / resume |
| `R` | reset the current scenario |
| `[` / `]` | slow down / speed up time |
| `I` | cycle integrator (leapfrog → RK4 → symplectic Euler) |
| `G` | toggle the relativistic force model |
| `C` | toggle collisions / merging |
| **Left-drag** | **slingshot** a new body: click a start point, drag to aim and set speed, release to launch |
| Right-drag | orbit camera · Middle-drag pan · Scroll zoom |

The HUD's **Conserved quantities** panel is the instrument: with leapfrog, total-energy *drift* sits
near zero and momentum barely moves — that steadiness is the NumSharp physics being faithful. Switch
to RK4 and speed up time to watch energy slowly drift instead.

### The scenarios

1. **Two-Body Circular Orbit** — the baseline; the orbit closes, energy holds to `1e-9`.
2. **Three-Body Figure-Eight** — the Chenciner–Montgomery choreography; three equal masses on one
   figure-eight curve.
3. **Solar System** — Sun + six planets, real relative masses, AU / M☉ / year units (`G = 4π²`).
4. **Binary Star + Planet** — a close binary with a distant circumbinary ("Tatooine") planet.
5. **Accreting Cluster** — a rotating cloud that gravitationally clumps and *merges* into larger
   bodies (collisions on).
6. **Relativistic Precession** — an eccentric orbit whose perihelion visibly rotates: Mercury's
   effect, exaggerated. Press `G` off to see it stop precessing under pure Newton.

---

## Notes & limitations

- **Performance.** The solver is direct `O(N²)` and allocates fresh NDArrays each substep, which is
  perfect for the interactive few-hundred-body regime but not a galaxy. Keep counts modest; the
  `MaxSubstepsPerFrame` cap protects the frame budget.
- **Precision.** The engine runs in `double`; the view down-casts to `float` only to place
  GameObjects. Running the physics in float would visibly drift within seconds.
- **The Unity C# is written against the Unity 6 API but is not compiled here** (no editor in this
  repo). The *physics* it drives is what's machine-verified — see [above](#verify-the-physics).

## Learn more

- **[docs/PHYSICS.md](docs/PHYSICS.md)** — the full derivations: vectorized gravity, softening,
  symplectic integration, conserved quantities, the 1PN relativistic correction, the figure-eight,
  and the unit systems.
- **[../UnityFallingSand/](../UnityFallingSand/)** — the companion falling-sand powder game: a
  mass-conserving cellular automaton with sand, water, oil and smoke (density stratification, piling,
  leveling), also NumSharp-powered.
- **[../../README.md](../../README.md)** — NumSharp itself.
