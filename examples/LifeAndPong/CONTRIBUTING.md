# Contributing to Life Arcade

Thank you for helping improve Life Arcade. This example is developed inside the
[NumSharp repository](https://github.com/SciSharp/NumSharp), and contributions
use that repository's issues, pull requests, license, and conduct policy.

## Before starting

Read the documents that own the area you intend to change:

- [README.md](README.md) for the player and contributor journey
- [SPECIFICATION.md](SPECIFICATION.md) for implementation ownership and limits
- [ARCADE_DESIGN.md](ARCADE_DESIGN.md) for adopted gameplay behavior
- [PHYSICS.md](PHYSICS.md) for collision behavior and acceptance cases
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md) for
  community and reporting expectations

Gameplay and physics changes require explicit design agreement; a code change
must not silently redefine those documents. For a bug or scoped improvement,
use the repository-root [Life Arcade bug form](https://github.com/SciSharp/NumSharp/issues/new?template=life-arcade.yml).
Discuss broad changes before investing in an implementation.

## Set up the repository

Clone or fork the complete NumSharp repository. Do not copy only
`examples/LifeAndPong`: the shared game project references
`src/NumSharp.Core`, and the focused solution includes repository build tools.

```powershell
git clone https://github.com/SciSharp/NumSharp.git
cd .\NumSharp\examples\LifeAndPong
dotnet --version
```

Install Git, PowerShell, and a .NET 10 SDK. From this folder, `global.json`
requests SDK `10.0.100`, permits later .NET 10 feature bands, and rejects
prerelease SDKs.

## Build and test

The normal local gate is:

```powershell
.\build.ps1
```

The script restores and builds `NumSharp.LifeAndPong.sln` and then runs the game
test project. It resolves paths from its own location, so this is also valid
from the repository root:

```powershell
.\examples\LifeAndPong\build.ps1
```

To run each stage directly from `examples\LifeAndPong`:

```powershell
dotnet restore .\NumSharp.LifeAndPong.sln --nologo
dotnet build .\NumSharp.LifeAndPong.sln -c Release --no-restore --nologo
dotnet test .\NumSharp.LifeAndPong.Tests\NumSharp.LifeAndPong.Tests.csproj -c Release --no-build --no-restore --nologo
```

Run the desktop host after the Release build:

```powershell
dotnet run --project .\NumSharp.LifeAndPong.Desktop\NumSharp.LifeAndPong.Desktop.csproj -c Release --no-build
```

For Visual Studio, open `NumSharp.LifeAndPong.sln` and set
`NumSharp.LifeAndPong.Desktop` as the startup project.

The repository-root [Life Arcade workflow](https://github.com/SciSharp/NumSharp/actions/workflows/life-arcade.yml)
defines the scoped Windows automation. Do not describe a local pass as a hosted
CI pass; inspect the actual workflow run when one is required.

## Source map

| Change area | Start here | Expected tests/docs |
| --- | --- | --- |
| State, scoring, phases, lives, sectors | `Models/ArcadeSession.cs` | `ArcadeSessionTests.cs`, `ARCADE_DESIGN.md` |
| Life rules and storage | `Models/LifeSimulation.cs` | `LifeSimulationTests.cs` |
| Contact geometry and response | `Models/CollisionMath.cs` | `CollisionPhysicsTests.cs`, `PHYSICS.md` |
| Profile schema and ruleset handling | `Models/PlayerProfile.cs` | `PlayerProfileTests.cs` |
| Input, menus, lifecycle, accessibility | `Views/GameSurface.cs` | `VisualSmokeTests.cs`, `SPECIFICATION.md` |
| Drawing and effects | `Views/ArenaView.cs` | headless visual smoke tests and relevant previews |
| Window and Windows audio | Desktop host and `Views/MainWindow.cs` | platform-focused tests/manual evidence |
| Build or packaging | `.sln`, `build.ps1`, `publish-windows.ps1` | full build, tests, package and checksum verification |

The tests use MSTest and Avalonia's headless test host. Preview regeneration is
test-controlled through `LIFE_ARCADE_PREVIEW_DIR`; do not replace checked-in
previews unless the visual change is intentional and reviewed.

## Coding and documentation conventions

- Follow the repository conventions and the local `.editorconfig`.
- Keep the shared project independent of a concrete audio implementation; the
  desktop host supplies Windows audio through `IGameAudio`.
- Preserve deterministic seeds in model and physics regression tests.
- Add focused regression coverage for behavior changes, including boundary and
  lifecycle cases.
- Update the owning guide when behavior, setup, packaging, or platform limits
  change. Mark intention or unverified behavior explicitly.
- Keep root-level document links usable both in the source folder and in the
  packaged application. Use absolute NumSharp GitHub links for repository-root
  resources; package readers cannot follow `..\..` source links.
- Do not commit `bin`, `obj`, local profile data, or generated `artifacts`.

## Package check

On Windows, create a fresh local package:

```powershell
.\publish-windows.ps1

$releaseDir = Get-ChildItem .\artifacts -Directory -Filter 'release-*' |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
$checksumFile = Join-Path $releaseDir.FullName 'SHA256SUMS'
$archive = Join-Path $releaseDir.FullName 'NumSharp-LifeAndPong-win-x64.zip'
$expected = ((Get-Content -LiteralPath $checksumFile -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$actual -eq $expected
```

Require `True` from the comparison. Inspect the unpacked app and ZIP for the
executable, runtime, NumSharp assemblies, `LICENSE`, all root Markdown guides,
`preview.png`, and `preview.ready.png`. `SHA256SUMS` remains next to the archive.

## Pull-request checklist

- Keep the change focused and explain its player or contributor impact.
- State the exact commands run and their results.
- Add or update tests for changed behavior.
- Keep [ARCADE_DESIGN.md](ARCADE_DESIGN.md), [PHYSICS.md](PHYSICS.md), and
  [SPECIFICATION.md](SPECIFICATION.md) consistent with the implementation.
- For packaging changes, verify a fresh ZIP and its `SHA256SUMS` entry.
- Remove private paths, personal information, credentials, and generated local
  profile data from logs and screenshots.
- Confirm that no unrelated worktree changes were included.

Contributions are licensed under the project's [Apache License 2.0](LICENSE)
and must follow the [NumSharp Code of Conduct](CODE_OF_CONDUCT.md).
