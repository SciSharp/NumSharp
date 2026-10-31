---
paths:
  - ".github/**"
  - "Directory.Build.props"
  - "**/*.csproj"
  - "src/NumSharp.Core/Assembly/Properties.cs"
  - "src/NumSharp.Interop.OpenBLAS/AssemblyInfo.cs"
  - "docs/releases/**"
---

# CI, packaging and strong naming

## Workflows

- `.github/workflows/build-and-release.yml`: tests on three OSes (Windows, Ubuntu, macOS), builds the NuGet packages on a
  tag push, creates the GitHub Release and publishes to nuget.org. The `FuzzMatrix` gate runs here (it replays the
  committed corpora; no Python). The `interop-test` job builds and tests the interop packages, the unreleased ones
  included (`interop-packages.md`), with its "Package consumer:" steps last (`openblas.md`).
- The packages embed NO README: the "No embedded package README" guard in `Directory.Build.props` fails `dotnet pack` if
  a project sets `PackageReadmeFile`, because an embedded readme would lock each version's nuget.org README tab. After a
  release publishes, the owner pastes the GitHub release markdown into every package version's README by hand
  (nuget.org → Manage → Readme); the step is in the `changelog` skill.
- `.github/workflows/fuzz-soak.yml`: the nightly soak sweeps one fixed seed (a deterministic canary whose corpus
  `source_sha256` should repeat night to night) plus nine fresh random seeds through `test/oracle/fuzz_random.py`
  (200K cases each), replays them, and uploads any failing corpus; copy a shrunk repro into `Fuzz/corpus/regressions/` to
  pin it on every CI run. It also runs the `collections-stress` job (the authored stress budgets) and the
  `random-api-lp64` / `random-api-soak` jobs.
- `.github/workflows/benchmark.yml`: a post-release benchmark (triggered by a published GitHub Release or a manual
  dispatch; a separate workflow so that a slow or failed benchmark never gates a release). It runs the whole
  `benchmark/run_benchmark.py` comparison (the op/dtype/N matrix, the NDIter benchmark and cards, the layout / operand /
  cast / fusion subsystems, and the isolated Managed and OpenBLAS/LAPACK backend profiles), publishes the snapshot
  (report, subsystem sheets, cards) to the orphan `data` branch (`tools/dashboard_data/publish.py`), and triggers a docs
  redeploy, the site reading it through the `refs/data` submodule. Nothing is committed to master. On shared runners
  absolute times are not comparable run to run; each NPY/NS ratio is measured on one runner. The NDIter harness reports a
  section that crashes on every retry (its known intermittent AccessViolation) as NA.
- `.github/workflows/example-pythonnet.yml`: the pythonnet example pipeline (`interop-packages.md`).

## Strong naming (identity, not authenticity)

- Every assembly in the repo is strong-named with `Open.snk` at the repo root, configured once in `Directory.Build.props`
  (`SignAssembly` defaults to true; a project opts out with `<SignAssembly>false</SignAssembly>`). Public key token:
  `cc7b13ffcd2ddd51`.
- The key is Microsoft's PUBLISHED open-source key, byte-identical to `dotnet/arcade`'s
  `src/Microsoft.DotNet.Arcade.Sdk/tools/snk/Open.snk`: the identity of `netstandard`, `System.Memory` and
  `System.Buffers`, and the key TensorFlow.NET signs with (a NumSharp-owned key would break the cross-repo
  `InternalsVisibleTo("TensorFlowNET.UnitTest")`). Its private half is public by design, so this is assembly IDENTITY
  (version binding, telling this `NumSharp` from another), not a security property; `InternalsVisibleTo` is not an access
  control. Authenticity would be Authenticode plus NuGet author signing, which needs a code-signing certificate and does
  not exist here.
- Every `InternalsVisibleTo` must carry `PublicKey=` (a keyless one is `CS1726`, a hard error). The friend declarations
  live in `src/NumSharp.Core/Assembly/Properties.cs` and `src/NumSharp.Interop.OpenBLAS/AssemblyInfo.cs`, with the key
  literal written once per assembly in the `NumSharpFriendKey` / `BlasFriendKey` consts.
- `dotnet run` scripts outside the repo root need the signing properties to see internals (the root CLAUDE.md's script
  template).
- The identity is asserted at run time, not trusted to the build: `test/NumSharp.Tests/Assembly/StrongNameTests.cs`
  checks the token, the full public key, that friend access resolves, and that no friend declaration is keyless.
