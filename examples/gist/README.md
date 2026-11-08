# Ten Gist numerical ports, using NumSharp

Ported Gist implementation filenames use `GitHubHandle.GistName.cs`, preserving the author's GitHub-handle casing—for example, `karpathy.NaturalEvolutionStrategies.cs` and `EderSantana.CatchReinforcementLearning.cs`. Runner/reference/test infrastructure is not a Gist and keeps its own name. Historical verification records retain their original observed paths; [filename-renames.json](filename-renames.json) maps those paths to the byte-identical renamed implementations.

Also available: [Karpathy's RNN, Pong policy gradients, batched LSTM and microGPT](karpathy/README.md), with a six-example runner including the existing NES and Stable Diffusion walk routines. Their sources/tests/evidence are separate from this original top-ten selection.

These are functional C# implementations of the numerical routines from the **top ten entries in our saved stars × NumPy-call-site Gist ranking**. Every implementation starts with its original Gist link and pinned revision. Ten main Python files and four companions are represented in ten C# implementation files; [SOURCES.md](SOURCES.md) and [sources.json](sources.json) record exact selection, hashes, scope and attribution.

As requested, this ports the **NumPy-dependent routines**, not entire Keras, TensorFlow or Stable Diffusion applications. No model, audio file, survey data, cloud service or Python interpreter is needed to run the demos. They calculate real outputs; none delegates its C# computation back to Python. Python is used only as the independent test oracle.

## Run it

From the repository root, with the .NET 10 SDK:

```powershell
dotnet run --file examples/gist/run.cs -c Release -- all
dotnet run --file examples/gist/run.cs -c Release -- 6
```

The second command runs the 300-iteration natural-evolution-strategy optimizer. Choose any number from 1 to 10, or `all`. The small [file-based runner](run.cs) calls the **same compiled implementations that the tests exercise**, through [NumSharp.GistExamples.csproj](NumSharp.GistExamples.csproj). That library targets both .NET 8 and .NET 10. This is a local-source build of NumSharp 0.70.0, not a claim of installing a released package.

The example project references NumSharp's OpenBLAS package and the runner enables it with one worker thread. It is necessary for the least-squares interpolation companion (`np.polyfit` → LAPACK); it also supplies the matrix-product implementation used by the live parity suite. The repository's native assets were already staged on the verification host. On a source checkout missing those ignored assets, the existing downloader is:

```powershell
python src/NumSharp.Interop.OpenBLAS/tools/fetch_openblas.py
```

That command downloads and verifies the manifest-pinned native artifacts. It was **not** needed or run to generate this example's results. NumSharp.Core itself remains managed C#; the native dependency belongs to the explicit OpenBLAS reference.

## What each file actually does

| Rank | C# implementation | Functional numerical scope | Deliberately outside this port |
|---:|---|---|---|
| 1 | [bshishov.ForecastingMetrics.cs](bshishov.ForecastingMetrics.cs) | All 30 forecast metrics; error, seasonal/explicit benchmark helpers and evaluation dispatch | Plotting or a forecasting model |
| 2 | [bwhite.RankingMetrics.cs](bwhite.RankingMetrics.cs) | All seven retrieval metrics, including DCG/NDCG and mean average precision | Search-engine infrastructure |
| 3 | [endolith.FrequencyEstimation.cs](endolith.FrequencyEstimation.cs) | Zero crossings, windowed FFT, autocorrelation, HPS, quadratic and least-squares interpolation | Audio decoding and plots |
| 4 | [karpathy.StableDiffusionWalk.cs](karpathy.StableDiffusionWalk.cs) | Spherical/linear interpolation, classifier-free guidance, decoded NCHW-to-HWC byte conversion | UNet, VAE, diffusion scheduler, checkpoint loading, image generation |
| 5 | [EderSantana.CatchReinforcementLearning.cs](EderSantana.CatchReinforcementLearning.cs) | Playable Catch state transitions, observations/rewards, bounded replay memory and Bellman targets using a prediction delegate | Keras network training, model files and graphical playback |
| 6 | [karpathy.NaturalEvolutionStrategies.cs](karpathy.NaturalEvolutionStrategies.cs) | Seeded Gaussian perturbations, objective rewards, standardized updates and the full quadratic optimization | An external environment or neural-network objective |
| 7 | [endolith.PeakDetection.cs](endolith.PeakDetection.cs) | Delta-hysteresis maxima/minima, optional sample positions and empty-result behavior | Plotting |
| 8 | [gyglim.TensorboardHistogram.cs](gyglim.TensorboardHistogram.cs) | Equal-width bucket counts/limits and all numerical HistogramProto fields; float32 statistics retain float32 reductions | Protobuf/event-file writing, image encoding, TensorFlow |
| 9 | [jkleint.TimeseriesCnn.cs](jkleint.TimeseriesCnn.cs) | Window/target/query construction, time-major normalization and chronological train/test slicing | CNN construction, fitting and prediction |
| 10 | [joelouismarino.GoogLeNet.cs](joelouismarino.GoogLeNet.cs) | RGB-to-BGR/NCHW preprocessing, local response normalization, pooling crop and argmax | Image decoding/resizing, neural-network graph, pretrained weights and inference |

## Ownership, types and deliberate adaptations

- Scalar extraction uses `ndarray.item<T>()`. APIs already returning a C# scalar, such as axis-free `np.argmax`, are used directly.
- Forecasting/ranking metrics and the NES algorithm use float64 vectors. Signal estimators normalize real input to float64. Slerp, guidance and LRN preserve float32/float64; image preprocessing produces float32 and decoded-image conversion produces uint8. Time-series windowing preserves the input dtype.
- Histogram inputs support NumPy-compatible integer types, float32 and float64, with finite representable ranges. Empty input, nonfinite ranges, invalid bin counts and collapsed bin edges are rejected. The numerical payload excludes density and custom-edge modes, which the original call did not use. The implementation composes NumSharp operations because Core has no `np.histogram` entry point; float32 edge arithmetic follows NumPy's typed `linspace` construction.
- Returned arrays belong to the caller. Time-series targets, normalized views, split arrays and the pooling crop can share input storage. Forecast `NaiveForecasting` returns a view. Windows/query arrays and replay snapshots are independent copies. Dispose returned containers/arrays after use; temporary calculations use `NDScope` or `using`.
- Catch's persistent state and replay snapshots are detached from temporary scopes and disposed by their owning objects. A regression test retains them across a per-step `NDScope`; replay history cannot become invalid when that scope closes.
- The original Catch reset mixes scalars and size-one arrays in `np.asarray`, which modern NumPy rejects. The port explicitly extracts the two sampled integers. It also rejects invalid actions and acting after a terminal state rather than indexing outside the grid.
- The original Slerp NumPy-input branch reads an uninitialized Torch-detection variable. The C# API accepts NDArray directly. Its float32 scalar intermediates are kept float32 explicitly; widening them to C# double caused a measured one-ULP discrepancy and was fixed, not excused.
- Original HPS prints diagnostics but returns no estimate and repeatedly downsamples an already-shrunk spectrum. `LegacyHarmonicProductPasses` preserves those six diagnostics, marking undefined interpolation as NaN. `FromHarmonicProductSpectrum` is a separately named, documented repair using the original spectrum's decimations. It is tested against that repaired mathematical reference, not misrepresented as an unchanged source result.
- Ranking's removed `np.asfarray` is represented by explicit float64 arrays in the Python oracle. Original metric conventions are preserved, including raw-relevance DCG gain, last-relevant-index R-precision, signed-error `std_ae`, and next-prediction-versus-previous-actual MDA. Invalid positive-size parameters are rejected explicitly; meaningful IEEE NaN/infinity results remain visible.
- Time-series `testSize=0` retains the original Python `-0` slicing behavior: empty training data and all samples as test data. Use a positive test size for a normal split.
- NES rejects zero/nonfinite reward standard deviation rather than generating an undefined update. It uses an independent seeded `RandomState`, leaving the caller's global RNG untouched. Its optional `progress` observer reports every pre-update state and the final state; the weights passed to it are borrowed and must be copied before retention.

These are focused numerical examples, not newly validated scientific applications or performance benchmarks. Source-level defects and adaptations are not silently labeled NumPy parity.

## Tests and live Python parity

The managed tests reference the actual example project; they do not maintain separate C# copies of its algorithms:

```powershell
dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --filter 'FullyQualifiedName~.Gist'
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --filter 'FullyQualifiedName~.Gist'
```

Replace `net10.0` with `net8.0` for the second target. The live suite needs CPython + NumPy; the signal-reference tests also need SciPy. The existing Python discovery mechanism is used. For the interpreter used here, the explicit PowerShell setting is:

```powershell
$env:PYTHONNET_PYDLL='C:\Users\ELI\.claude\python\python312.dll'
```

Live tests export the **same NumSharp input buffers zero-copy** into Python, evaluate independent NumPy/SciPy reference formulas, and compare dtype, shape and canonical C-order bytes. They do not import or run the original scripts' framework/IO top-level code. Error cases, strided/reversed views, ownership and special-value behavior are tested too. The existing interop cleanup checks for leaked exports/imports after each test.

Exact-byte equality is the default. A few composed metrics have explicit, measured 1–2 ULP allowances, selected by metric/convention before comparison; there is no fallback that turns a failed exact assertion into a loose `allclose`. Undefined estimates and zero-norm Slerp have separately labeled NaN-classification checks. [VERIFICATION.md](VERIFICATION.md) records actual results, runtime versions, limits, commands and artifacts.

The expanded suite passes **88 managed + 55 live cases on each framework** (286 executions, zero failures/skips). It asserts all ten actual demo transcripts, original doctests and printed checkpoints, complete source-sized tensor workflows, and every NES intermediate state. [DEMONSTRATION_COVERAGE.md](DEMONSTRATION_COVERAGE.md) maps the pinned sources to tests and distinguishes original examples from synthetic fixtures and historical claims that cannot be reproduced from the supplied data.

Keep [SOURCES.md](SOURCES.md) with any redistribution. Attribution is not an invented license grant; seven pinned gists contain no explicit license declaration. The C# numerical implementations are independently expressed and do not copy original docstrings or bulk source.
