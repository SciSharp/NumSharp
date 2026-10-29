# Demonstrations and measurements: coverage of the ten Gist ports

Every located reproducible numerical demonstration in the ten pinned Gists has an assertion-bearing test. The audit also records non-reproducible measurements, framework-only work and source proposals so they cannot disappear behind a “covered” label.

This concerns the **top ten selected numerical ports**, not every Gist in the research corpus. The original selection, files, revisions, source hashes and license findings remain in [SOURCES.md](SOURCES.md) and [sources.json](sources.json).

## What the tests actually exercise

| Rank / numerical port | Demonstrations and measurements asserted | Important boundary |
|---|---|---|
| 1 — Forecast metrics | All 30 metrics, six helpers, registry and both evaluators; contiguous, strided and reversed inputs | Original source has no worked dataset, doctest or printed numerical measurement. These are explicit test fixtures. |
| 2 — Ranking metrics | All 25 original doctests: 24 numerical cases and one exception; both DCG conventions | Removed `np.asfarray` uses explicit float64 arrays. One historical Python-list sum differs by one ULP on Python 3.12 and is documented separately. |
| 3 — Frequency estimation | Original parabolic docstring/main markers and README polynomial fit; FFT, crossings, autocorrelation and HPS outputs; labeled synthetic accuracy/limitation experiments | Original audio and timing environment are absent. Synthetic checks do not reproduce those historical numbers or establish universal accuracy. |
| 4 — Stable Diffusion walk | All 200 full 1×4×64×64 interpolation frames; guidance at 7.5; full 512×512 RGB byte conversion | Shared synthetic tensors replace model outputs. No Torch-RNG equivalence, diffusion model, scheduler or generated-image claim. |
| 5 — Catch and replay | Full 1,000-episode numerical loop: 9,000 actions/states/rewards; epsilon-greedy selection; capacity-500 replay; 50 sampled input/target rows | Fixed-Q predictor is a disclosed fixture, not a trained Keras model. Its 330 wins are fixture-specific. |
| 6 — Natural evolution strategies | All 15 original printed checkpoints; all 301 states and rewards over 300 updates | Original seed, population, step sizes and quadratic objective; no claim about a different optimization problem. |
| 7 — Peak detection | Every maxima/minima plot coordinate from the original example, custom positions and validation branches | Plot rendering is excluded. The normalized float64 signal and MATLAB/Python coordinate convention are explicit adaptations. |
| 8 — TensorBoard histogram | All numeric payload fields, all 1,000 limits/counts, four dtypes and a reversed/strided input | The source supplies the default bin count, not a worked input dataset. No protobuf, image encoding or event-file writer. |
| 9 — Time-series CNN preparation | Original three-step example and both full 1,000-sample datasets; every window, target, query and train/test split | CNN fitting, predictions and learned metrics are excluded. |
| 10 — GoogLeNet helpers | Full 224×224 preprocessing; both LRN layer shapes; all four pooling crops; third 1,000-class head selection | Synthetic image/activations/scores, not pretrained classification, decoding/resizing or external class-name files. |

The actual C# `Demo()` for **each of the ten** also runs in a data-driven test comparing its entire printed transcript. This is output regression coverage, not a replacement for the independent NumPy oracle.

## Evidence, not inferred pass status

The four detailed inventories carry source lines, pins, fixture adaptations, test names and limits:

- [Metrics inventory](coverage-metrics.json): forecasting surface and original ranking doctests.
- [Signal inventory](coverage-signal.json): original examples, synthetic experiments, accuracy statements and unreproduced claims.
- [Tensor inventory](coverage-tensor.json): original tensor dimensions, full datasets, derivations and framework exclusions.
- [Learning/histogram/demo inventory](coverage-learning.json): full numerical workflows, 15 printed NES checkpoints and ten actual demo outputs.

Inventory prose such as “asserted” describes what a test is intended to check. **Only executed test outcomes establish a pass.** [demonstration-verification.json](demonstration-verification.json) joins these references to the four final TRX files. Multiple source records can point to the same test; inventory-entry counts are not counts of independent experiments.

[verify-demonstrations.ps1](verify-demonstrations.ps1) performs that join reproducibly. It checks the Gist-only selection, resolves test references, requires passing results on both frameworks, includes parameterized rows and retains exclusions/limited claims without relabeling them as reproduced. Re-run it after fresh tests:

```powershell
./examples/gist/verify-demonstrations.ps1
```

The current result is **88 managed + 55 live cases per framework, 286 passing executions, zero failures/skips**, on .NET 8 and .NET 10. See [VERIFICATION.md](VERIFICATION.md) for exact test commands, host versions, numerical budgets, findings and preserved historical evidence.

## Claims deliberately not certified

The original frequency material includes 1000.185, 1000.000129 and 1000.000004 Hz examples without the actual recording/sample settings, plus elapsed-time and speed claims without a reproducible environment. Those remain **unverified**. New explicit sine/harmonic/record-length experiments illustrate behavior, but cannot certify an unspecified original recording, all waveforms, human pitch perception or a historical benchmark.

The numerical-only scope excludes framework integration, training/inference results, audio/image decoding, charts, TensorBoard files and external alternative algorithms. Future proposals and external theory links are inventoried, not silently implemented or treated as measured facts.

Independent review found no missing reproducible literal numerical example in the pinned files. That finding means **all located demonstrations are accounted for**, not that every original external/scientific/performance claim has been reproduced.
