# Life Arcade implementation guide

This document maps the current Life Arcade behavior to its implementation and
contributor operations. It describes source baseline
`5f6b551aabf01e2539554eedd3bc40ebad759c09`; the hash identifies reviewed source
and is not a public-release claim.

Behavioral authority is split deliberately:

- [ARCADE_DESIGN.md](ARCADE_DESIGN.md) owns the adopted game rules, scoring,
  pacing, effects, accessibility, and acceptance targets.
- [PHYSICS.md](PHYSICS.md) owns collision geometry, continuous-contact
  resolution, paddle friction and spin, and the bounded 5% contact noise.
- This guide owns the implementation map, lifecycle, persistence, platform
  boundary, and contributor/release journey.

## Product model

Life Arcade is one 1600 × 900 logical arena, one ball, one right-hand player
paddle, and one Conway colony. The phase boundary is at x=1120: Life occupies
70% of the width and the player occupies 30%. There is no left or AI paddle and
no separate Life editor during a run.

The session has four states:

- **Ready** — the ball is attached to the paddle and can be launched.
- **Playing** — ball, paddle, phase clocks, scoring, and effects advance.
- **Paused** — simulation and visual-effect clocks stop until resumed.
- **GameOver** — reached after the third miss; the completed run is recorded.

A new run starts with three lives, score zero, sector 1, a seeded 42 × 32
bounded B3/S23 colony replenished to 160 cells, and ruleset version
`life-arcade-3`.

## Grow and shatter phases

`ArcadeSession` changes phase when the ball center crosses the 70% boundary.
The crossing is a time-of-impact event within the current simulation step, not
a wall or a render-only test.

- In **GROW**, while the ball is on the player side, Life advances at the
  current sector rate. If population is below 64, a 250 ms active-GROW cue is
  followed by replenishment to 160 cells and a 750 ms cooldown. Surviving cells
  are preserved.
- In **SHATTER**, while the ball is on the colony side, Life is frozen. The ball
  collides with the frozen live-cell geometry; each distinct contacted live
  cell is destroyed once.
- Pausing stops both phases. Crossing back into GROW resumes evolution from the
  damaged colony rather than restoring an earlier board.

`LifeSimulation` stores the current and next generations in two reusable
NumSharp byte NDArrays. The arcade constructs it with non-wrapping edges;
outside cells are dead.

## Scoring, lives, and progression

The successive awards for cells destroyed during one shot are:

```text
+1, +2, +4, +6, +8, +10, ...
```

The first three cells therefore add 7 total points. A physical paddle contact
resets the shot count and next award to `+1`; a phase crossing does not. A miss
also resets the chain, consumes one life, and preserves the accumulated score.
Life births, ordinary Life deaths, walls, and paddle hits award no points.

The implementation saturates awards and total score instead of wrapping.
Presentation effects escalate at 20, 50, and 100 cells destroyed during one
shot. These effects do not mutate simulation state. Sector progression is based
on total cells destroyed, with the pending sector increasing every 40 cells and
being adopted at a launch or paddle contact.

## Physics boundary

`ArcadeSession` advances at fixed 120 Hz steps and processes the earliest
collision, phase crossing, paddle stop, or goal before consuming the remainder
of the step. `CollisionMath` performs swept circle-versus-rounded-box queries,
elastic manifold response, and isolated moving-paddle friction/spin response.

After a resolved physical contact manifold, the session adds a uniformly drawn
perpendicular perturbation bounded by 5%, renormalized to the computed physical
speed and reduced if necessary to avoid pointing back into the contact. Noise
is not applied during free flight. The exact equations, tolerances, and safety
behavior are in [PHYSICS.md](PHYSICS.md).

## Input, presentation, and accessibility

`GameSurface` owns the accessible native menu controls around the custom-drawn
arena, keyboard and pointer input, the fixed-step accumulator, event feedback,
and presentation timers.

- Mouse motion over the right 30%, `W`/`S`, and Up/Down control the paddle.
- `Space` launches, pauses, or resumes. `Escape` pauses. `Enter` retries from
  game over. Held key aliases are tracked independently.
- Window deactivation, focus loss from the surface, detachment, and disposal
  release transient input. Window deactivation also pauses play.
- Restart requires confirmation while a run is paused. Sound, reduced motion,
  and high contrast are menu options.
- `ArenaView` renders the colony, ball, paddle, ball mark/spin, impacts,
  milestones, and phase background. Text remains outside the playfield.
- `MainWindow` supplies the classic desktop window, initial focus, 1440 × 900
  default size, and 1120 × 700 minimum size.

## Persistence and audio

`PlayerProfile.OpenLocal()` reads and writes:

```text
%LOCALAPPDATA%\NumSharp\LifeArcade\profile.json
```

The profile stores sound, reduced-motion, and high-contrast preferences plus
run results. Results include score, best shot chain, total destroyed cells,
sector, seed, and ruleset version. The UI calculates its best score only from
`life-arcade-3` records. Older-version records can be retained without being
compared with the current scoring system. Read/write failures are shown to the
player and do not prevent play.

The shared project depends only on `IGameAudio`. The desktop host supplies
`WindowsGameAudio`, which generates its PCM cues in memory and plays them off
the UI thread through `winmm.dll`. It reports audio unavailable on non-Windows
systems; no downloaded sound assets are required.

## Source ownership

| Area | Owner in source |
| --- | --- |
| Rules, state, clocks, progression, collision orchestration | `Models/ArcadeSession.cs` |
| NumSharp-backed Conway state and replenishment | `Models/LifeSimulation.cs` |
| Swept geometry and physical response | `Models/CollisionMath.cs` |
| Profile schema and audio abstraction | `Models/PlayerProfile.cs` |
| Input, menus, HUD, event dispatch | `Views/GameSurface.cs` |
| Arena drawing and effects | `Views/ArenaView.cs` |
| Desktop lifecycle | `Views/MainWindow.cs` and the Desktop `Program.cs` |
| Windows PCM implementation | `NumSharp.LifeAndPong.Desktop/WindowsGameAudio.cs` |
| Model, physics, profile, and headless UI regressions | `NumSharp.LifeAndPong.Tests` |

The shared game project references `src/NumSharp.Core`. The focused solution
also includes NumSharp.Core and its build/analyzer dependency. Consequently,
`examples/LifeAndPong` must remain inside a complete NumSharp checkout; copying
this folder alone is not a supported build layout.

## Build, test, and run

The documented contributor baseline is a .NET 10 SDK selected by the local
`global.json`. From `examples\LifeAndPong`:

```powershell
.\build.ps1
dotnet run --project .\NumSharp.LifeAndPong.Desktop\NumSharp.LifeAndPong.Desktop.csproj -c Release --no-build
```

The equivalent explicit commands are:

```powershell
dotnet restore .\NumSharp.LifeAndPong.sln --nologo
dotnet build .\NumSharp.LifeAndPong.sln -c Release --no-restore --nologo
dotnet test .\NumSharp.LifeAndPong.Tests\NumSharp.LifeAndPong.Tests.csproj -c Release --no-build --no-restore --nologo
```

For IDE use, open `NumSharp.LifeAndPong.sln` and set
`NumSharp.LifeAndPong.Desktop` as the startup project.

## Windows packaging and automation

`publish-windows.ps1` runs the Release tests and creates a self-contained,
untrimmed `win-x64` portable application. Each invocation uses a new
`artifacts\release-<GUID>` directory. The directory contains the unpacked app,
`NumSharp-LifeAndPong-win-x64.zip`, and `SHA256SUMS` for that ZIP.

The packaged app includes the runtime, every root-level Markdown file in this
folder, `LICENSE`, `preview.png`, and `preview.ready.png`. Links among the
packaged guides therefore use package-local filenames; repository resources use
absolute upstream links.

The automation definitions live at the NumSharp repository root, not under the
example folder:

- [Life Arcade Windows workflow](https://github.com/SciSharp/NumSharp/actions/workflows/life-arcade.yml)
- [Life Arcade bug form](https://github.com/SciSharp/NumSharp/issues/new?template=life-arcade.yml)

The checked-in workflow is the automation contract. This guide does not claim
that a hosted workflow run or public binary release has occurred.

## Platform and support limits

- The package target and native audio implementation are Windows x64.
- The source uses Avalonia's desktop boundary, but other desktop operating
  systems have not been validated by this project's current acceptance record.
- Android and iOS heads are not present.
- The package is a portable ZIP, not an installer.
- This example does not define a detached-source distribution or a separate
  support/version lifecycle from the NumSharp repository.

## Historical design trail

The following identifiers are preserved for traceability only. They describe
the superseded, separate Life-editor and two-paddle Pong product and are not
evidence for current Life Arcade behavior or delivery status:

- Product baseline `SPEC-numsharp-life-pong-001`
- Delivered unit `DU-life-pong-finish-002`
- Adopted decision `DEC-life-pong-finish-002`
- Historical implementation `996dba5d`, reviewed in
  `E-01a0711f-cf00-7883-9a8a-1d109cda19cb`

The subsequent unified-arcade design lineage is recorded in
[ARCADE_DESIGN.md](ARCADE_DESIGN.md): `WORK-legacy`,
`JOB-life-arcade-design-003`, `SPEC-life-arcade-003`, and
`DEC-life-arcade-003`, later amended by `DEC-life-arcade-steering-005` and
`DEC-life-arcade-physics-006`. These identifiers document design history; the
current source and the three authority documents named at the top of this guide
define the implementation contributors should follow.
