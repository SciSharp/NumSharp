# Two parity levels for Karpathy's Gists

Both requested levels now run through **NumSharp.Interop.pythonnet**, embedding real CPython/NumPy in the test process. The first executes short, complete numerical workflows using the original pinned Gist definitions; the second requires exact NumPy dtype, shape and value bytes at each checkpoint.

## Run level 1, then level 2

From the repository root, with a Python containing NumPy discoverable by pythonnet:

```powershell
# This is the tested host's Python DLL; use your installation's equivalent.
$env:PYTHONNET_PYDLL='C:\Users\ELI\.claude\python\python312.dll'

# Level 1: original numerical workflows through the real Python bridge.
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --filter 'TestCategory=KarpathyShortRun'

# Level 2: no tolerance or allclose fallback—NumPy bytes must match.
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --filter 'TestCategory=KarpathyByteParity'

# Both levels plus original-source integrity gates, without duplicate execution.
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --filter 'FullyQualifiedName~.Karpathy'
```

Use `net8.0` for the other verified framework. The legacy Python-2 Gists currently use **CPython 3.12's `lib2to3`** for syntax compatibility; a runtime missing that converter fails explicitly instead of quietly skipping source tests. These tests do not install Python, download a dataset, start Gym or execute an unbounded original main loop.

## Actual results

| Framework | Live Karpathy cases | Managed regression cases | Failed | Skipped |
|---|---:|---:|---:|---:|
| .NET 8 | 40 | 38 | 0 | 0 |
| .NET 10 | 40 | 38 | 0 | 0 |

**156 passing executions**, including 80 live Python/NumPy cases. Live coverage increased from 22 to 40 cases per framework. The four final logs are `outputs/karpathy-parity-2026-09-09/karpathy-{unit,live}-{net8,net10}-final.trx`.

The two requested commands were also executed separately, in order, on .NET 10: **8 short-run cases** passed, then **29 byte-parity cases** passed. There are **6 source-integrity cases**. Three NES/walk cases belong to both numerical levels, so 8+29+6 is not a count of distinct tests; their union is 40. The extra tier replays are not added to the 156 total above. These are correctness gates, not performance benchmarks.

[Machine-readable evidence](parity-verification.json) · [Verifier](verify.ps1) · [Original-source fixture manifest](../../../test/NumSharp.Tests.Interop/Fixtures/Karpathy/sources.json)

## What each short run proves

| Gist | Original-source / short-run level | Byte-to-byte NumPy level |
|---|---|---|
| Character RNN | Original `lossFun` and `sample`, four complete H=100/T=25 updates with clipping, reset boundaries, smoothing and 48 sampled tokens | At every update: all hidden/probability values, carried state, loss/smoothing, raw and clipped five gradients, all five parameters, Adagrad memories and sampled tokens. Existing 40-step learning parity remains. |
| Pong policy gradient | Five original numerical functions process 40 NumSharp frames across ten synthetic episodes, update the model and return trained state through pythonnet | Eleven episodes / 44 frames, checking image mutation, differences, probabilities/actions, stacked rollout, rewards/advantages, both gradients, accumulated buffers, RMSProp caches, weights and counters. **655 array comparisons / 39,398,640 compared bytes, zero mismatches.** The eleventh episode uses updated weights. Existing full H=200/D=6400 checks remain. |
| Batched LSTM | Actual original class and both original check functions: four printed `True` results and **414/414 original gradient checks reporting OK** | Five recurrent chunks and six genuine SGD updates: all six forward intermediates, all four derivatives, weights, losses and carried cell/hidden states. SGD is an additional test workflow—the original LSTM has no optimizer. |
| microGPT | Actual original `Value` autograd and transformer functions: three complete backward/Adam updates and twenty generated samples | Six alternating-document updates with **228 dtype/shape/byte checkpoints**, including every loss, logit, parameter gradient, updated weight and both Adam moments. Existing source-sized architecture, 1,000-step managed training and independent autoregressive-inference checks remain. |
| NES | Original objective, original population/step sizes, twelve complete optimization updates | All thirteen weight/reward states—52 float64 values—match exactly. The prior full 301-state regression remains separate. |
| Stable Diffusion walk | Original Slerp numerical body on five full 1×4×64×64 latent frames in each spherical / near-parallel branch | Every float32 value of every frame matches. No diffusion model, Torch RNG or generated-video claim. |

The byte level evolves **independent Python parameter copies after the initial shared inputs**. It does not feed C#'s updated weights back into the oracle, which could conceal accumulating errors. Exports use NumSharp's buffers; owning imports test the reverse crossing where applicable. The inherited interop lifetime gate checks for leaked exports/imports after each test.

## Original code, not just a second rewrite

Six byte-identical original source fixtures are embedded from `test/NumSharp.Tests.Interop/Fixtures/Karpathy/`, named `karpathy.<gist>.py`. Tests therefore do not depend on a temporary output directory or network at build/run time. The original acquisition sources and receipts remain untouched.

`KarpathyOriginalSource` verifies a fixed SHA256 before loading and uses Python AST to select only named class/function definitions. It does **not** execute the source's top-level imports, assignments, downloads, application loops or main block. The test supplies a bounded driver and explicit data/model globals. Compatibility edits are recorded, not concealed:

- RNN/Pong/LSTM: Python-2 print, `xrange` and backtick syntax conversion.
- LSTM: integer hidden-width division `/4` becomes `//4`.
- Pong: removed `np.float` alias becomes `np.float64`, without changing the NumPy module globally.
- Walk: initialize the missing `inputs_are_torch=False` local for the source's otherwise broken NumPy-input branch; numerical expressions are unchanged.
- microGPT and NES: no mathematical or legacy-syntax edits to the selected original definitions.

Tests restore NumPy's process-global RNG when original functions require it. Original Python dataset ordering/random streams are not silently claimed equivalent to an explicitly substituted local fixture or NumPy RandomState.

## microGPT: distinguish numerical equivalence from byte identity

microGPT's original is **scalar Python autograd**, not NumPy. Its scalar evaluation/gradient accumulation order differs from the vectorized NumSharp/NumPy implementation. The original-source short run uses a declared numerical gate (`rtol=1e-9`, `atol=1e-11`) for every loss/logit/gradient/weight/moment; the largest observed absolute state difference was **2.220446049250313e-16**. Its twenty sampled strings also matched under the explicitly shared uniform-draw convention.

That is not scalar-source bit identity. The **NumPy matrix-level tests remain exact-byte tests**, with no tolerance fallback. Earlier finite-difference and scalar-value checks are retained as additional independent evidence. No exception to the byte contract is hidden behind the first level.

## Evidence and scope

This follow-up changed **tests, fixtures and documentation only**. All fourteen port implementation files still match the byte hashes recorded in [filename-renames.json](../filename-renames.json). One initial test failed because a 3-D RGB frame was read with two indices; the test now verifies shape/dtype and reads `(row,column,0)`. No numerical discrepancy was excused or kernel changed to make the new byte tests pass.

Verification host: Windows x64, SDK 10.0.101; .NET runtimes 8.0.29/10.0.1; CPython 3.12.12, NumPy 2.4.2. Matrix products use the explicit OpenBLAS backend at one thread. Exactness is a result for these fixtures and the tested numerical build/CPU, not a guarantee across arbitrary BLAS builds or inputs. Release builds succeeded with warnings and no final errors.

`verify.ps1` records current file/fixture/helper hashes, source and resource mappings, source-declared test/DataRow counts, actual TRX outcomes and overlapping tier membership. Current hashes are not by themselves proof of historical compilation. The earlier [verification.json](verification.json) remains the prior-stage record; new results go to `parity-verification.json`. Original licensing observations, missing external datasets and excluded ML applications remain explicit in the source/coverage manifests.
