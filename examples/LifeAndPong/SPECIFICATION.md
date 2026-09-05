# Life + Pong delivered product and design specification

- Product baseline: `SPEC-numsharp-life-pong-001`
- Delivered unit: `DU-life-pong-finish-002`
- Adopted decision: `DEC-life-pong-finish-002`
- Implementation: `996dba5d` (code QA approved in `E-01a0711f-cf00-7883-9a8a-1d109cda19cb`)

## Delivered product

The application is a self-contained NumSharp showcase in one custom-rendered Avalonia surface. An editable Conway simulation occupies the left panel while a player-versus-AI Pong match runs in the right panel. The two systems share one visual language and one input surface.

The desktop window defaults to 1440 × 900 and has an enforced minimum of 1120 × 700. The source checkout retains approved previews as `preview.png` at the default size and `preview.minimum.png` at the minimum size; the player ZIP includes the primary preview.

## Windows player delivery

The player artifact is `NumSharp-LifeAndPong-win-x64.zip`. It is an untrimmed, self-contained Windows x64 publish, so the extracted application carries its .NET runtime and does not require a separate .NET installation. A player extracts the archive and launches `NumSharp.LifeAndPong.Desktop.exe` from the contained `NumSharp-LifeAndPong-win-x64` folder.

This delivery is a portable folder and ZIP, not an installer. Windows x64 is the packaged and tested runtime. No Android or iOS binary is delivered, and other desktop operating systems were not run or validated for this release.

## Life behavior

- A 48 × 40 toroidal Conway field follows B3/S23 rules.
- The model owns two reusable NumSharp `NDArray` buffers with byte dtype (`NPTypeCode.Byte`): one current field and one next-generation field. Each step fills the next buffer and swaps the two references; it does not allocate another field buffer.
- The startup field and subsequent Seed sequence are reproducible for the model's fixed random seed; repeated Seed actions advance that sequence and need not produce identical fields. Clear empties the field.
- Glider and Pulsar load centered deterministic presets. The Pulsar is a period-three oscillator.
- Run/pause, single-step, clear, reseed, preset selection, and generation rates of 3, 6, 12, 24, and 40 generations per second are available.
- Step, Clear, Glider, and Pulsar explicitly pause Life and reset its accumulated generation time. Seed preserves the current run/pause state.
- A pointer press toggles the first cell and establishes paint or erase mode. Captured pointer movement interpolates every cell between reported positions, avoiding gaps during fast drags. Evolution waits while a stroke is active.
- Generation and live-cell counts remain visible.

## Pong behavior

- A new match begins in Ready state. Start or `Space` begins the serve flow; New Match or `R` clears scores, rally state, paddle state, and returns to Ready.
- The left paddle accepts mouse targeting over the arena and keyboard movement through either `W`/`S` or `Up`/`Down`. Aliases are held independently, and keyboard intent overrides an earlier pointer target.
- `Space` starts, pauses, or resumes; `R` starts a new match. Both actions are limited to the first key-down of each press so operating-system key repeat cannot toggle them repeatedly.
- Physics advances at a fixed 120 Hz. Adaptive micro-steps account for the combined ball and maximum paddle travel to keep high-speed and moving-paddle collisions from tunneling.
- Paddles accelerate and decelerate rather than teleport. AI periodically predicts the ball's wall-reflected arrival point.
- Paddle contact is a circle-versus-rectangle closest-point test. The ball reflects about the physical contact normal in the moving paddle's frame, returns to the world frame, and receives a bounded tangential contribution from paddle motion.
- A paddle hit raises rally speed by 4.5% up to a 920-world-unit-per-second cap. It then adds a uniformly sampled signed perpendicular component bounded to 2% of speed and renormalizes the result to preserve the capped speed.
- The ball reflects from the upper and lower walls. Points reset rally count and enter a short serve countdown. The first side to seven wins; current and best rally counts remain visible.

## Input and lifecycle behavior

- The surface receives initial focus when the desktop window opens.
- `Tab` and `Shift`+`Tab` cycle through custom controls; `Enter` or `Space` activates the selected control. Pointer clicks use the same activation paths.
- `Escape` and desktop-window deactivation clear held keys, pointer targeting, pointer capture, and an active Life stroke, then pause Pong. Detaching or disposing the surface also releases transient input. Life keeps its existing run/pause setting.
- Losing surface focus releases transient input without inventing a missed key-up or continuing a captured stroke.

## Visual system

One custom Avalonia control renders the interface with vector primitives: a dark ink field, technical grid, mint Life accents, warm coral Pong accents, restrained bloom layers, rounded panels, focus/hover outlines, live statistics, rally state, visible control hints, and responsive scaling. The game requires no raster artwork or stock game widgets; the PNG files are documentation and QA previews.

## Architecture and platform boundary

The implementation targets .NET 10 and Avalonia 12.1.2. The shared `NumSharp.LifeAndPong` project owns application resources, both simulation models, rendering, and input. It supports the classic desktop lifetime and a single-view lifetime. The `NumSharp.LifeAndPong.Desktop` project contains only the executable desktop head and boots the classic desktop lifetime.

This separation is mobile-ready at the project boundary, but mobile remains later scope: adding Android or iOS heads and validating touch layout/input are still required. The current binary and QA evidence cover Windows x64 only; cross-platform Avalonia code is not evidence that another desktop operating system has been tested.

## Verification and release operations

From the repository root, developers can launch and test with:

```powershell
dotnet run --project examples/LifeAndPong/NumSharp.LifeAndPong.Desktop
dotnet test examples/LifeAndPong/NumSharp.LifeAndPong.Tests/NumSharp.LifeAndPong.Tests.csproj -c Release --nologo
```

Create a player ZIP with:

```powershell
.\examples\LifeAndPong\publish-windows.ps1
```

`publish-windows.ps1` resolves paths from `$PSScriptRoot`, so it also works when invoked by its path from another working directory. It runs the Release tests before publishing, uses `--self-contained true` with trimming disabled, copies README, specification, primary preview, and NumSharp license into the player folder, and compresses the result. Each invocation writes to a unique `examples/LifeAndPong/artifacts/release-<GUID>/` directory and prints the ZIP SHA-256 hash plus Play and Share paths.
