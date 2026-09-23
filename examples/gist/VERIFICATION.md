# Actual verification — 2026-09-09

The source-demonstration audit is complete for the **ten selected numerical ports**. All ten actual demos ran, and the expanded tests passed on both target frameworks. Tests call the compiled example implementations, not a second C# copy.

| Framework | Managed Gist cases | Live Python/NumPy cases | Failed | Skipped |
|---|---:|---:|---:|---:|
| .NET 8 | 88 | 55 | 0 | 0 |
| .NET 10 | 88 | 55 | 0 | 0 |

**143 cases per framework, 286 passing executions.** Three live cases test the parity assertion itself; the other 52 exercise numerical examples. These are the Gist suites, not every NumSharp test. Release builds succeeded for both projects and frameworks with warnings, no errors.

[Source-to-test coverage](DEMONSTRATION_COVERAGE.md) · [Machine-readable evidence](demonstration-verification.json) · [Actual demo transcript](demo-output.txt) · [Original source pins](sources.json)

## What is now checked

- All ten actual `Demo()` outputs are regression-tested, including every printed value. The recorded transcript is not its own numerical oracle: separate managed/source-reference and live NumPy tests establish the calculations.
- Ranking: all 25 original doctest cases are represented, including the invalid-k exception. Forecasting: all 30 metrics, six helpers, registry and both evaluators are covered; its source contains no worked dataset or numeric output to reproduce.
- NES: the 15 original printed progress checkpoints match their printed precision. All **301 states**, comprising 903 weights and 301 rewards, match live NumPy exactly—not just the final answer.
- Catch: the original 10×10 grid, 1,000-episode count, capacity-500 replay and batch size 50 are exercised. All 9,000 action/state/reward rows and sampled replay inputs/Bellman targets match NumPy byte-for-byte. The fixed-Q fixture wins 330 games; that is **not** a trained-network success claim.
- Tensor workflows: all 200 full 1×4×64×64 latent interpolation frames; 512×512 RGB conversion; both original 1,000-sample time-series datasets and their complete windows/splits; 224×224 GoogLeNet preprocessing; both full LRN stages; all four pooling crops; and third-head selection from 1,000-class scores.
- Signal examples: original interpolation markers, the three-point polynomial companion example and every peak scatter coordinate, plus explicitly labeled synthetic accuracy/limitation experiments.
- Histogram: all bucket limits, counts and summary fields at the source's default **1,000 bins**, on float32/float64/int32/int64 reversed-and-strided fixtures.

Each claim is bounded by its recorded inputs and scope. The coverage report distinguishes literal source examples, source-parameter workflows using synthetic inputs, port adaptations, regression checks and unverified historical/external claims.

## Numerical agreement and fixes found by the stronger tests

The oracle receives the same input buffers through zero-copy NumSharp → NumPy exports. Shape/dtype and canonical C-order value bytes are compared; canonicalization changes layout, not values. Interop cleanup also checks conversion lifetimes.

Exact bytes remain the default. The existing composed-metric suite retains its explicitly selected allowances: only forecast `smape`/`mrae` allow one ULP, ranking method-1 DCG one and NDCG two. The dedicated 150-scalar comparison reports 134 exact, ten one-ULP and six two-ULP results per framework. Other helper assertions are additional. There is no failed-exact-to-`allclose` fallback. Zero-norm Slerp and undefined legacy-HPS interpolation have separately identified NaN-classification checks.

The larger fixtures exposed three reduction-order differences:

1. **NES:** the old implementation reached the same final weights but differed in 118 intermediate weight/reward values (maximum 92 ULP in weights). Explicit-axis mean and NumPy's two-pass population-variance composition now match the entire trajectory exactly.
2. **Histogram:** the dense float32 square buffer's flat sum differed by three float32 ULP. Flattening that result in memory order and using an explicit-axis sum selects the matching pairwise fold.
3. **Zero crossings:** crossing positions and period differences were already exact; the flat period mean caused a one-ULP estimate difference on two larger fixtures. Explicit-axis mean matches NumPy.

These changes use public NumSharp operations in the examples. **No Core numerical implementation was changed and none of these exact gates was relaxed.** The [preserved reduction probe](../../outputs/gist-demonstrations-2026-09-09/reduction-probe/README.md) records the pre-fix assemblies, stage measurements and replay.

Earlier work also fixed a widened float32 Slerp scalar, detached Catch/replay-owned arrays from temporary scopes, preserved histogram input-dtype arithmetic, and explicitly enabled/restored the native backend for the polynomial companion.

The ranking source's separate Python-list precision-area expression is a historical last-bit exception: Python 3.12 evaluates it as 0.7833333333333334, while its older printed output says 0.7833333333333333. The tests record that one-ULP relationship rather than forge the old interpreter result; the average-precision routine itself retains its expected result.

These are fixture- and host-specific observations, not universal numerical or performance guarantees. In particular, matching synthetic signal estimates to NumPy does not establish the historical author's unspecified audio measurements, perceptual accuracy or timing claims.

## Environment and reproduction

- Windows x64; .NET SDK 10.0.101, runtimes 8.0.29 and 10.0.1; Release.
- CPython 3.12.12, NumPy 2.4.2, SciPy 1.16.3 under `C:\Users\ELI\.claude\python`.
- Local NumSharp 0.70.0 project references; explicit OpenBLAS backend, one thread. The native assets were already staged.
- No original framework/IO top-level script, training model, checkpoint or external audio was executed. The reviewed numerical references run independently in actual NumPy/SciPy.

From the repository root, build and test both projects for each framework. Use a fresh output directory to retain earlier evidence:

```powershell
$env:PYTHONNET_PYDLL='C:\Users\ELI\.claude\python\python312.dll'
dotnet build test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --nologo -v quiet -clp:ErrorsOnly
dotnet build test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --nologo -v quiet -clp:ErrorsOnly
dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --no-build --filter 'FullyQualifiedName~.Gist' --logger 'trx;LogFileName=source-demo-unit-net10-final-v3.trx' --results-directory outputs/gist-demonstrations-2026-09-09
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --no-build --filter 'FullyQualifiedName~.Gist' --logger 'trx;LogFileName=source-demo-live-net10-final-v3.trx' --results-directory outputs/gist-demonstrations-2026-09-09
dotnet run --file examples/gist/run.cs -c Release -- all
```

Repeat with `net8.0` and `net8` in the log names. The four `final-v3` TRX files are the final passing records. Earlier failed/diagnostic logs remain separate and are not substituted for final results. The parent live runs explicitly set `PYTHONNET_PYDLL`; child diagnostics with different discovery settings are identified separately.

The reusable [coverage verifier](verify-demonstrations.ps1) joins the audit's test references to actual TRX definitions/outcomes, requires both frameworks, includes every data row, checks source/log hashes, and retains exclusions without counting them as passed experiments.

## Historical evidence and limits

[verification.json](verification.json) remains the **initial** run record, not the current version's source pin. Its original 23 pinned files are preserved byte-for-byte under `outputs/gist-demonstrations-2026-09-09/initial-source-snapshot/`, at their original repository-relative paths. Initial logs remain under `outputs/gist-ports-2026-09-09/`.

A counting correction: the initial `ClassName~Gist` filter also matched **13 unrelated RandomLogistic tests**. Thus its recorded 49 managed cases contained 36 actual Gist cases. Current runs use `FullyQualifiedName~.Gist`, selecting exactly the Gist classes; the 88 managed / 55 live totals above do not include those unrelated tests.

The agreed scope remains functional **numerical routines**, not full Keras/Torch/TensorFlow applications. Missing original audio/timing inputs and external theoretical/perceptual claims are explicitly unverified. See the coverage report for each exclusion. Source attribution does not create a license grant; keep [SOURCES.md](SOURCES.md) with redistribution.

Unrelated concurrent repository changes and the original research data were left untouched. The evidence records source hashes and observed repository state, not a pristine-checkout claim.
