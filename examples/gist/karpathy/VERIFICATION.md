# Karpathy ports: executed verification

This is the initial port-verification stage. The later [two-level parity report](PARITY.md) expands live coverage from 22 to 40 cases per framework and adds original-source execution; its new machine-readable record is [parity-verification.json](parity-verification.json).

All four new numerical implementations and the six-example runner work. Final Release tests passed on **.NET 8 and .NET 10**, with no failures or skipped cases.

| Test group | Managed cases per framework | Live cases per framework |
|---|---:|---:|
| Minimal character RNN | 9 | 5 |
| Pong policy gradients | 9 | 7 |
| Batched LSTM | 6 | 4 |
| microGPT | 8 | 6 |
| Actual six-demo output assertions | 6 | — |
| **Karpathy total** | **38** | **22** |

That is **120 passing Karpathy executions** across both frameworks. The previous top-ten Gist suite also passed on both: 88 managed + 55 live each, **286 regression executions**. Together: **406 executed cases**, zero failures/skips. These are selected example suites, not a rerun of all NumSharp tests.

[Machine-readable evidence](verification.json) · [Actual runner transcript](demo-output.txt) · [Source pins](sources.json) · [Verifier](verify.ps1)

## Numerical evidence

- **RNN:** complete hidden/probability traces, accumulated loss, all five BPTT gradient arrays, seeded samples, Adagrad parameters/memory and bounded training trace match the live NumPy formulation exactly. The original H=100/T=25 dimensions are tested. A separate 40-parameter finite-difference check validates calculus. The actual demo's 40 updates reduce batch loss from **54.929665653107975 to 0.69944287395149884** on its explicit repeated-text corpus; its exact sampled strings are asserted.
- **Pong:** full source H=200/D=6400 initialization/forward/backward, the entire frame-preprocessing mutation, return discounting/standardization, accumulated gradients, RMSProp state and batch updates pass strict live comparisons. All 15 parameters in a smaller network pass a finite-difference check. The bounded full-size synthetic demo performs one real update over two episodes; weight-change norm is about **0.0408186**, last up-action probability **0.493647**. These are synthetic-fixture results, not Atari performance.
- **LSTM:** all six forward intermediate arrays and all four derivative arrays are byte-identical to NumPy at original dimensions, with default/supplied states, terminal-gradient carries and reversed inputs. Sequential/batched forward and all four backward results match exactly. **All 414 coordinates** pass the original central-difference test; maximum relative error on this host is approximately **2.073e-8**, below the source's 1e-2 warning threshold. Nonfinite gradients fail before the tiny-gradient exemption.
- **microGPT:** full logits, loss and every parameter gradient match an independent NumPy matrix formulation exactly on single- and two-layer configurations. Four Adam updates compare every parameter and both optimizer moments exactly. Twenty autoregressive samples after training match an independently evaluated NumPy inference loop. The small model's **228 parameters** pass exhaustive finite differences, maximum normalized error about **3.204e-9**. The source-sized model completes all **1,000 default updates** and callbacks; corpus loss on ten local names falls **2.649477742973952 → 0.518578513883086**. Its final per-document training loss is **0.48776857505308924**, which is a different observable from corpus-average loss.

**microGPT's original is scalar Python, not NumPy.** Its scalar-list causal equations form a separate value oracle. Maximum observed absolute logit difference is **2.220446049250313e-16**, under the declared 2e-14 value gate. This is explicitly **not scalar-source byte equality**. Matrix summation/gradient accumulation and NumPy's RNG differ from the original scalar autograd/Python RNG. The exact byte claim concerns the independently expressed NumPy formulation over shared inputs. Finite differences add an independent derivative check; they are not byte-parity assertions.

The live exact gates have no tolerance fallback. Canonical C-order byte comparison normalizes output layout, not numerical values; dtype and shape are also checked. The inherited interop cleanup checks leaked imports/exports after every live test. All observations are fixture/host-specific, not guarantees for every input, CPU, BLAS build or library version.

## Review findings addressed

1. microGPT's owning gradient container initially returned arrays still attached to a caller's temporary scope. It now detaches all owned logits/gradient buffers. A regression closes that scope, then applies Adam successfully. Other model/cache/optimizer ownership cases are tested too.
2. Finite-difference tests now explicitly reject nonfinite derivatives before checking tiny-gradient exceptions; NaN cannot silently count as a successful coordinate.
3. The file-based runner's up-to-date check initially reused a **stale referenced assembly** after a library-only edit, still reporting microGPT's earlier 200 steps. Actual verification reran with **`--no-cache`**, confirmed 1,000 steps, and recorded fresh output. The documented runner command deliberately includes this flag.
4. Independent reviews checked RNN, LSTM and microGPT against the pinned originals and their coverage maps. No unaddressed numerical implementation defect or omitted reproducible literal source demonstration was found. Missing datasets/framework applications remain explicit scope limitations.

No NumSharp.Core numerical implementation was changed for these ports. Normalized-return and transformer reductions use explicit axes where required to preserve the tested NumPy operation order. Public C# scalar extraction uses `item<T>()`.

## Exact commands and environment

- Windows x64; .NET SDK **10.0.101**; runtimes **8.0.29 / 10.0.1**; Release.
- CPython **3.12.12**, NumPy **2.4.2**, SciPy **1.16.3**. SciPy is needed by the earlier signal regressions, not the four new algorithms.
- Local NumSharp **0.70.0** project references and the explicit OpenBLAS backend, one thread. Native assets were already staged; no backend download was needed.
- All new neural arithmetic is float64; image preprocessing uses uint8. No external model, original training corpus, Atari ROM or framework application was downloaded or executed.

Run from the repository root. These are the final .NET 10 commands; substitute `net8.0` and `net8` in the corresponding paths for the other framework:

```powershell
dotnet build test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --nologo -v quiet -clp:ErrorsOnly
dotnet build test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --nologo -v quiet -clp:ErrorsOnly
$env:PYTHONNET_PYDLL='C:\Users\ELI\.claude\python\python312.dll'
dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --no-build --filter 'FullyQualifiedName~.Karpathy' --logger 'trx;LogFileName=karpathy-unit-net10-final-v2.trx' --results-directory outputs/karpathy-gists-2026-09-09
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --no-build --filter 'FullyQualifiedName~.Karpathy' --logger 'trx;LogFileName=karpathy-live-net10-final-v2.trx' --results-directory outputs/karpathy-gists-2026-09-09
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- all
```

Both projects built successfully for both target frameworks. Existing warnings were reported; there were no final build errors. Earlier successful partial runs are retained separately from the final four `*-final-v2.trx` files. Old-suite regression logs are `prior-gist-{unit,live}-{net8,net10}-regression.trx` in the same output directory, using `FullyQualifiedName~.Gist`. Use new log names/directories for future runs rather than overwriting this evidence.

`runner-final.json` and `runner-final.txt` in that output directory record the successful freshly rebuilt six-example run. Tests execute each actual `Demo()` and assert the full corresponding transcript, including sample characters. This is regression coverage, not a circular substitute for independent references.

The [verifier](verify.ps1) derives test/DataRow counts from current test sources, resolves each coverage-map reference against actual TRX definitions/outcomes, requires both frameworks, checks pins, retains excluded cases and produces [verification.json](verification.json). It does not fabricate evidence of compilation from source timestamps; current file hashes and historical test execution records are identified separately. Original top-ten verification files remain historical snapshots and were not overwritten.

Source acquisition and its failures are documented in [sources.json](sources.json). Immutable raw-file revisions are distinguished from verified Gist history revisions. The original article/repository/Gist usage-research counts were not recalculated or rewritten, and unrelated concurrent worktree changes were preserved.
