# NumSharp Life + Pong

An interactive, custom-rendered Avalonia showcase for NumSharp. Conway's Game of Life runs on the left while a single-player Pong match runs on the right.

![NumSharp Life and Pong at 1440 × 900](preview.png)

The source checkout also records the responsive minimum-size layout in `preview.minimum.png` at 1120 × 700. The player ZIP includes the primary preview shown above.

## Play the Windows release

The `NumSharp-LifeAndPong-win-x64.zip` release is an unpack-and-play, self-contained Windows x64 build. It includes the .NET runtime; players do not need to install .NET separately.

1. Extract the ZIP to a writable folder.
2. Open the extracted `NumSharp-LifeAndPong-win-x64` folder.
3. Run `NumSharp.LifeAndPong.Desktop.exe`.

The ZIP is a portable application folder, not an installer.

## Run from source

Developers need the .NET 10 SDK. From the NumSharp repository root, run:

```powershell
dotnet run --project examples/LifeAndPong/NumSharp.LifeAndPong.Desktop
```

Run the dedicated Release test suite with:

```powershell
dotnet test examples/LifeAndPong/NumSharp.LifeAndPong.Tests/NumSharp.LifeAndPong.Tests.csproj -c Release --nologo
```

## Build the Windows ZIP

From the repository root:

```powershell
.\examples\LifeAndPong\publish-windows.ps1
```

The script also works from any current working directory when invoked by its path because it resolves inputs and outputs relative to its own location. Each run:

- runs the Release test suite and stops on failure;
- publishes an untrimmed, self-contained `win-x64` application;
- creates a new `examples/LifeAndPong/artifacts/release-<GUID>/` directory, preserving earlier output;
- produces both an unpacked `NumSharp-LifeAndPong-win-x64` folder and `NumSharp-LifeAndPong-win-x64.zip`; and
- prints the ZIP SHA-256 hash plus the executable and archive paths.

## Controls and lifecycle

Press `Tab` or `Shift`+`Tab` to move through the custom controls. Press `Enter` or `Space` to activate the selected control. Mouse clicks activate the same controls.

### Life

- Click a cell to toggle it. Drag from a dead cell to paint or from a live cell to erase; skipped cells are filled by an interpolated stroke.
- Use Run/Pause, Step, Seed, Clear, Glider, Pulsar, and the `-`/`+` generation-rate controls.
- Step, Clear, Glider, and Pulsar leave Life paused so the result can be inspected. Seed replaces the field without changing its current run/pause setting.
- Evolution pauses temporarily during a drag and resumes afterward if Life was running.

### Pong

- Choose Start or press `Space` from the initial Ready screen. `Space` then pauses or resumes; New Match or `R` resets the score and returns to Ready.
- Move the mouse over the arena to target the left paddle, or use `W`/`S` and `Up`/`Down`. The letter and arrow aliases are tracked independently, so releasing one alias does not cancel another that remains held.
- `Space` and `R` act once per physical press; keyboard repeat messages do not retrigger them.
- Press `Escape`, or deactivate the window, to pause Pong and clear transient keyboard, pointer, and paint input. Life retains its existing run/pause setting.
- The first side to seven points wins.

## Implementation notes

Life owns a 48 × 40 toroidal field in two reusable NumSharp `NDArray` buffers created with byte dtype (`NPTypeCode.Byte`). The Conway B3/S23 kernel reads one buffer, writes the other, and swaps them after each generation without allocating another field buffer.

Pong advances through a fixed 120 Hz accumulator with adaptive micro-steps based on ball and paddle travel. Paddle collision uses the circle/rectangle closest-point normal: velocity is reflected in the moving paddle's frame, paddle motion contributes tangential transfer, and the result is renormalized to a rally speed capped at 920 world units per second. Each paddle impact then adds a uniformly sampled perpendicular component bounded to 2% and renormalizes without changing that speed.

The shared `NumSharp.LifeAndPong` project contains the simulations, application resources, input, and custom vector-rendered view. `NumSharp.LifeAndPong.Desktop` is a thin classic-desktop host.

## Supported scope and limitations

- This delivery is packaged and tested for Windows x64. Other desktop operating systems have not been run or validated for this release.
- Android and iOS heads remain future work; no mobile binary is included.
- The supported window range starts at 1120 × 700; the default is 1440 × 900.
- The release is a portable ZIP rather than an installer.
