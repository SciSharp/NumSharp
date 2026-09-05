# Life + Pong product and design baseline

Status: adopted for `SPEC-numsharp-life-pong-001`
Delivery: `DU-numsharp-life-pong-001`

## Product

The application is a polished, self-contained NumSharp showcase. A single desktop window presents two live systems at once: an editable Conway simulation on the left and a player-versus-AI Pong match on the right. It must feel like one designed instrument rather than two stock controls placed beside each other.

The desktop layout is designed for a 1440 × 900 window and remains usable down to 1120 × 700. Game state, drawing, and input are portable. Windows is the first supported runtime; Android and iOS heads are a later delivery.

## Life behavior

- A 48 × 40 toroidal grid is owned by a NumSharp `NDArray<byte>`.
- Conway B3/S23 rules advance the grid.
- The initial field is a deterministic, attractive seeded pattern.
- Run/pause, single-step, clear, reseed, and adjustable generation rate are available.
- Pointer press and drag paint cells directly.
- Generation and live-cell counts remain visible.

## Pong behavior

- The player controls the left paddle by pointer, `W`/`S`, or arrow keys; an AI controls the right paddle.
- Physics advances at a fixed 120 Hz and uses adaptive micro-steps to prevent tunneling at maximum ball speed.
- Paddles accelerate and decelerate rather than teleport. Paddle velocity and impact position influence rebound angle.
- Every paddle impact perturbs the velocity by a uniformly sampled perpendicular component bounded to 2%, then renormalizes it. Rally speed increases gradually to a cap.
- The ball reflects from the upper and lower walls, points reset through a short serve countdown, and the first side to seven wins.
- Pause/resume and new-match controls are available.

## Visual system

The interface is rendered by one custom Avalonia control using vector primitives. It uses a dark ink field, subtle technical grid, warm coral for Pong, mint for Life, restrained bloom layers, rounded panels, custom buttons, readable live statistics, and responsive scaling. No raster artwork or stock game widgets are required.

## Architecture decision

Avalonia 12.1.2 and .NET 10 are used because the NumSharp branch targets .NET 10 and current Avalonia mobile targets require .NET 10. The shared project contains application resources, simulation models, and the custom view. The desktop project only boots the classic desktop lifetime. A future mobile delivery adds Android/iOS heads without moving the game code.
