# Karpathy's learning algorithms, running on NumSharp

Latest testing: [two explicit parity levels](PARITY.md)—short runs of the hash-pinned original definitions through pythonnet, followed by strict NumPy byte checks at every update. **40 live cases plus 38 managed regressions pass on each of .NET 8/10.**

Gist implementation filenames follow `GitHubHandle.GistName.cs`, such as `karpathy.MicroGpt.cs`. Historical verification paths can be resolved through [the filename rename map](../filename-renames.json); the rename did not change implementation bytes.

Four new functional ports join the existing NES and Stable Diffusion walk numerical examples. These are executable algorithms with forward passes, derivatives, optimizer state and bounded demonstrations—not interfaces that call Python for their computation. Python/NumPy is the independent test oracle.

## Run

From the repository root, using the .NET 10 SDK:

```powershell
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- all
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- rnn
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- pong
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- lstm
dotnet run --file examples/gist/karpathy/run.cs -c Release --no-cache -- microgpt
```

`--no-cache` is intentional: .NET 10's file-based up-to-date check reused a stale referenced example assembly after a library-only edit on this host. The verification runs force a rebuild. `nes` and `walk` run the two existing ports. [run.cs](run.cs) is a file-based C# application calling the same compiled classes the tests exercise. The shared example library targets .NET 8 and .NET 10. It references local NumSharp 0.70.0 source and the optional OpenBLAS package, enabled with one thread; no claim of an installed released NuGet package is made. Missing native assets can be staged with the downloader documented in the [parent README](../README.md). No new ML framework, checkpoint, Atari ROM or downloaded dataset is needed.

## What is implemented

| Example | Working numerical implementation | What its demonstration means |
|---|---|---|
| [karpathy.MinimalCharacterRnn.cs](karpathy.MinimalCharacterRnn.cs) | One-hot recurrent forward pass, softmax loss, full backpropagation through time, five clipped parameter gradients, persistent Adagrad, hidden-state resets, smoothed loss and categorical sampling | Learns a stated local character corpus in 40 updates. Original hidden size 100 and sequence length 25 are also exercised in tests; the small demo uses hidden size 16. |
| [karpathy.PongPolicyGradient.cs](karpathy.PongPolicyGradient.cs) | Original 210×160 RGB preprocessing and frame differencing, H=200/D=6400 policy network, action probabilities, point-reset reward discounting, standardized returns, policy gradients, accumulation and RMSProp ascent | Two explicit synthetic episodes produce a real parameter update. This is not an Atari environment or evidence of learning to win Pong. The original ten-episode accumulation interval is separately tested. |
| [karpathy.BatchedLstm.cs](karpathy.BatchedLstm.cs) | Packed IFOG gates, batched recurrent forward pass, every intermediate, complete backward pass and initial/final cell/hidden gradient carries | Original sequential-versus-batched assertions and all 414-coordinate finite-difference checks are executable tests. Source dimensions are time=5, batch=3, input=10, hidden=4. |
| [karpathy.MicroGpt.cs](karpathy.MicroGpt.cs) | Token/position embeddings, causal multi-head attention, residual RMSNorm/ReLU transformer, complete manual reverse-mode derivatives, Adam with bias correction/linear decay, BOS-aware temperature sampling | Learns ten supplied names with the original E=16/H=4/L=1/context=16 architecture and 1,000-update schedule; emits 20 samples. No downloaded name corpus or claim of language-model quality. |
| [karpathy.NaturalEvolutionStrategies.cs](../karpathy.NaturalEvolutionStrategies.cs) | Original seeded quadratic optimizer and all 300 updates | Existing port, with all 301 states/rewards and 15 printed source checkpoints tested. |
| [karpathy.StableDiffusionWalk.cs](../karpathy.StableDiffusionWalk.cs) | Latent Slerp, classifier-free guidance and image-array conversion | Existing numerical-only port; not a diffusion model or image generator. |

The RNN demo reduces its per-batch loss from approximately **54.9297 to 0.699443** on its tiny repeated corpus. This demonstrates actual learning, not generalization. The [retained transcript](demo-output.txt) includes the sampled characters and every other demo output. [VERIFICATION.md](VERIFICATION.md) records executed tests and evidence.

## Fidelity and deliberate adaptations

- Scalars are extracted with `ndarray.item<T>()`. Neural arithmetic is float64; Pong frames are uint8. These examples do not claim arbitrary-dtype model training.
- RNN, Pong and LSTM are NumPy originals. Live tests compare complete states/gradients/updates against independently expressed NumPy references, starting with the same exported buffers. Source Python 2 syntax and removed dtype aliases are adapted explicitly.
- **microGPT is originally scalar Python, not NumPy.** This port vectorizes its equations using NumSharp and derives matrix-level backward passes. Exact bytes are checked against an independent NumPy formulation; agreement with the original scalar operation order is a separate, measured value comparison. The two claims must not be conflated.
- microGPT prefix recomputation replaces the original incremental key/value cache. Causality and autoregressive sampling are tested. NumPy RandomState replaces Python's `random` stream, and the caller controls the local document order; identical seed numbers do not imply identical original parameters or names.
- The RNN vocabulary is sorted explicitly instead of relying on the source's nondeterministic set order. Its unspecified `input.txt` becomes a supplied corpus; its infinite loop becomes a caller-selected number of updates.
- Pong preprocessing intentionally retains the source's mutation of selected red-channel pixels. Copy a frame first if it must remain unchanged. Undefined zero-variance advantage normalization is rejected, not allowed to contaminate weights with NaNs.
- LSTM's code initializes a **positive +3 forget bias** despite an inconsistent source comment. The port follows the executable code. Empty sequences are rejected explicitly. Forward caches own independent weight/state snapshots rather than borrowing mutable input arrays.
- Models, caches, gradient containers and optimizer buffers have explicit ownership/disposal. Arrays exposed by a model/container are borrowed from that owner unless documented otherwise; standalone returned arrays belong to the caller.

## Tests and reproducibility

```powershell
dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 --filter 'FullyQualifiedName~.Karpathy'
$env:PYTHONNET_PYDLL='C:\Users\ELI\.claude\python\python312.dll'
dotnet test test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj -c Release -f net10.0 --filter 'FullyQualifiedName~.Karpathy'
```

Replace `net10.0` with `net8.0` for the second library target. Python/NumPy is needed only by live tests, not by the C# demos. The explicit DLL path above is the verified host's interpreter, not a portable installation requirement.

Every actual runner demo has a complete stdout regression assertion, including its sampled strings. Those golden outputs do not substitute for the independent numerical, gradient, training and live-parity checks. The coverage maps preserve this distinction:

- [RNN](coverage-rnn.json), [Pong](coverage-pong.json), [LSTM](coverage-lstm.json), [microGPT](coverage-microgpt.json), [all demo outputs](coverage-demo.json).
- [Source pins and acquisition record](sources.json), [executed verification](VERIFICATION.md), [machine-readable results](verification.json), [reusable verifier](verify.ps1).

## Source provenance and reuse

The author catalogue returned 13 public Gists in the observed page. These four are an editorial selection of substantial standalone learning code, not an updated popularity ranking. Original usage-research results remain unchanged.

Sources were retrieved using the repository's Claude curl/curl-research workflow, saved with HTTP receipts and SHA256 hashes. RNN and microGPT's large metadata endpoints returned HTTP 502; their complete source files were successfully retrieved at the **immutable raw-file URLs already supplied by the author catalogue**. Their raw-file revision is not mislabeled a verified Gist HEAD. Pong/LSTM metadata also supplies a verified Gist history revision. See `sources.json` for exact paths and failed receipts.

RNN's header says “BSD License” without specifying a variant or supplying full text. The other three pinned files declare no explicit license. Attribution is not a license grant. These C# files independently express the numerical algorithms; they do not copy the original docstrings, prose or full source. Keep the provenance and observed license limitations with redistribution.
