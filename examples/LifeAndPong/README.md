# NumSharp Life + Pong

An interactive, custom-rendered Avalonia showcase for NumSharp. Conway's Game of Life runs on the left while a single-player Pong match runs on the right.

![NumSharp Life and Pong desktop preview](preview.png)

## Run on desktop

```powershell
dotnet run --project examples/LifeAndPong/NumSharp.LifeAndPong.Desktop
```

The first target is Windows desktop. The simulation and view live in the shared `NumSharp.LifeAndPong` project, while the desktop executable is a thin host. Future Android and iOS heads can reuse the same project.

## Controls

- **Life:** click or drag across the grid to paint cells. Use Run/Pause, Step, Seed, Clear, and the speed controls.
- **Pong:** move the pointer over the arena or use `W`/`S` and the arrow keys. Press `Space` to pause or resume and `R` to start a new match.
- The first player to seven points wins.

The Life grid is stored in a NumSharp `NDArray<byte>` and advanced with an allocation-conscious Conway kernel. Pong runs at a fixed 120 Hz simulation step. Paddle impacts transfer motion into the ball, increase rally speed, and apply a bounded 2% perpendicular directional jitter.
