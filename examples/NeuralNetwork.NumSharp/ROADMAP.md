# NeuralNetwork.NumSharp — Feature Roadmap

Gap analysis of the NN framework against the Python ecosystem it mirrors, and a
phased plan for closing it. Reference libraries: **Keras 3** (the API model this
project follows), **PyTorch 2.x** (autograd / optim / DataLoader concepts),
**scikit-learn** (metrics), with notes from numpy-native micro-frameworks
(tinygrad, micrograd) for the autograd tier.

Every "Status" cell below was verified against the actual source on this
checkout — not the docs. Every "Core dependency" cell was verified against
`src/NumSharp.Core` (e.g. batched matmul confirmed in
`Backends/Default/Math/BLAS/Default.MatMul.cs`, which broadcasts ≥3-D stack
dims; `np.random.bernoulli`, byte-exact `.npz`, and `np.evaluate` confirmed
present; `take_along_axis`, `sliding_window_view`, scatter-add, `einsum`,
`linalg.qr` confirmed absent).

---

## 1. Current inventory (what exists today)

| Area | Have | Notes |
|---|---|---|
| Model | `NeuralNet` (Sequential), `MlpTrainer` | Train/Predict, `EpochEnd` event; MlpTrainer adds periodic test eval |
| Layers | `FullyConnected`, `FullyConnectedFused`, `Dropout`, `BatchNormalization`, `LayerNormalization`, `Embedding`, `Flatten`, `Reshape` | Fused variant folds bias+ReLU into one NDIter kernel; P4 added the regularization/normalization set behind a `Training` flag |
| Regularizers | ✅ P4 | `L1`, `L2`, `L1L2`, attached per parameter key and applied by the trainer |
| Activations | `ReLU`, `Sigmoid`, `Softmax` | `BaseActivation.Get()` resolves only `"relu"`/`"sigmoid"` — Softmax class exists but is **unreachable by name** |
| Losses | `MeanSquaredError`, `BinaryCrossEntropy`, `CategoricalCrossentropy`, `SoftmaxCrossEntropy` | SCE is the combined stable form, lives in `MnistMlp/` |
| Metrics | `Accuacy` (sic), `BinaryAccuacy` (sic), `MeanAbsoluteError` | Typo'd class names kept for compat |
| Optimizers | `SGD` (momentum, decay), `Adam` | `BaseOptimizer.Get()` resolves `"sgd"`/`"adam"` |
| Init | He-normal (ReLU), Xavier-normal (else) | Hardcoded in the Dense ctors, not a pluggable module |
| Data | `MnistLoader` (IDX + synthetic), `OneHot` helper | No general Dataset/DataLoader abstraction |
| Serialization | ✅ P1 | `ModelWeights` (`.npz`, numpy-readable) + `ModelArchitecture` (JSON + factory registry) |
| Callbacks | ✅ P1 | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger` on a Keras-shaped `BaseCallback` |
| Training loop | ✅ P1 | shuffle, `validationSplit`/`validationData`, callbacks, partial final batch, `verbose` 0/1/2, gradient clipping |

**Known defects to fix before building on top (P0)** — ✅ ALL FIXED
(verified by `tests/verify_p0_p2.cs`, 86 checks):

1. ~~`BaseActivation.Get("softmax")` returns `null`~~ — registered; resolver
   is now case-insensitive, `""`/`linear`/`none` → null, unknown names throw.
2. ~~String-vs-enum activation surfaces~~ — `FullyConnectedFused` gained the
   string ctor (only fused-capable names resolve; others throw with a pointer
   at `FullyConnected`).
3. ~~`SGD`/`Adam` `DecayRate` compounds~~ — both compute
   `lr_t = lr0/(1+decay·t)` fresh from the never-mutated base rate.
4. ~~`MlpTrainer.Evaluate` floor-divides~~ — scores every sample; the final
   batch may be partial.
5. ~~Stale perf docs~~ — CLAUDE.md rewritten from 2026-07-25 Release
   measurements; the probe now reports three paths (hand-rolled NDIter,
   np.evaluate, naive) at two sizes. **Finding: the inversion is real at every
   size** — even at 4.2M elements the unfused whole-array SIMD kernels beat
   the NDIter fused path ~4× for this 2-op expression; np.evaluate's
   documented wins need long chains. Core follow-up candidate: NDIter
   broadcast-operand inner-loop throughput vs Direct kernels.
6. ~~`np.allclose` mutates operand dtypes~~ — fixed **in core**:
   `Cast(copy:false)` now follows NumPy semantics (self only when no
   conversion; a conversion allocates and never touches the input). Repairs
   `np.allclose`/`np.isclose`/`np.where` operand corruption; pinned by the
   rewritten `NDArray.astype.Test.cs` (12,017-test core suite green).

---

## 2. Core-capability map (what NumSharp enables or blocks)

NN features are only as feasible as the `np.*` surface underneath them.

**Ready today (no core work needed):**

| Core capability | Unblocks |
|---|---|
| `np.random.bernoulli` | Dropout masks |
| `np.random.permutation` / `shuffle` + fancy indexing | Per-epoch shuffling |
| `mean`/`var`/`sum` with `axis`+`keepdims`, `sqrt` | BatchNorm, LayerNorm |
| Byte-exact `.npz` writer (`np.savez`/`np.load`) | Weight serialization interoperable with Keras/PyTorch numpy dumps |
| `np.evaluate` (fused NDExpr) | Zero-temp optimizer updates (Adam allocates ~14 temps/param/step today) |
| Batched `np.matmul` (≥3-D, broadcast stack dims) | Attention, batched RNN cells |
| Stride-aware GEMM (`np.dot` on transposed views) | All dense backward paths (already used) |
| `tanh`/`exp`/`log`/`maximum`/`clip`/`where`/`abs`/`power` | Every activation and loss in P2 |
| `Half` dtype + SIMD casts | Mixed-precision groundwork |
| `np.pad` | Conv padding |

**Missing in core (each blocks NN features AND is a real NumPy-parity item —
these belong on the main library roadmap regardless):**

| Missing `np.*` API | NN features blocked | Workaround until then |
|---|---|---|
| `take_along_axis` | SparseCategoricalCrossentropy gather, TopK metrics | manual unsafe loop / OneHot materialization |
| `sliding_window_view` (public `as_strided`) | MaxPool/AvgPool general case, fast im2col | slicing-loop im2col (correct, slower) |
| scatter-add (`np.add.at` / `put` with accumulate) | Embedding backward | manual unsafe accumulate loop |
| `einsum` | elegant attention/conv notation | compose `matmul`+`transpose` (fully workable) |
| `linalg.qr` / `svd` | Orthogonal initializer | modified Gram-Schmidt in the example |
| 2-D correlate/convolve kernel | direct-conv fast path | im2col + GEMM (the standard numpy approach anyway) |

---

## 3. Phased plan

Phases are ordered by (usability gained) / (effort), and each phase is
shippable on its own. "Ref" = which Python library defines the semantics we
match.

### P0 — Hygiene (fix what exists) — ✅ DONE

All six defects in §1 fixed (see the checklist there). The fusion probe was
re-framed rather than re-tuned: it measures three paths at two sizes and the
finding is that the unfused SIMD kernels win at both — recorded honestly in
CLAUDE.md instead of chasing a size where fusion looks good.

### P1 — Training-loop parity (pure C#, no new math) — ✅ DONE

Shipped in full: every row of the table below. `Callbacks/` (BaseCallback +
TrainingContext + EarlyStopping / ModelCheckpoint / ReduceLROnPlateau /
CSVLogger), `Serialization/` (ModelWeights `.npz`, LayerConfig,
ModelArchitecture JSON + factory registry), gradient clipping on
`BaseOptimizer` (`ClipNorm` / `GlobalClipNorm` / `ClipValue`), and a
generalized `MlpTrainer.Train` (per-epoch shuffle, `validationSplit` /
`validationData`, callback list, partial final batch, `verbose` 0/1/2).
`NeuralNet` kept working and gained callbacks + `SaveWeights`/`LoadWeights`/
`ToJson`/`FromJson`.

Verified by `tests/verify_p1.cs` (**131 checks**, 0 failed); the P0/P2 gates
stayed green (86 + 94). Demo convergence held exactly — final test accuracy
99.90%, with epoch-1 loss moving 1.1247 → 1.1004 because the epoch now
shuffles and runs 47 batches instead of 46 (the partial final batch used to be
dropped).

**Findings worth keeping:**

1. **Checkpoint keys must be POSITIONAL.** `BaseLayer.Name` comes from
   `Util.GetNext()`, a process-global counter that never resets, so a
   name-keyed archive cannot load into a model rebuilt in the same process —
   which is the entire point of a checkpoint. Keys are `layer{i}/param/{name}`
   and `layer{i}/state/{name}`.
2. **Validate everything before writing anything.** The first implementation
   shape-checked during assignment, so a mismatch in a late layer left the model
   with layer 0 from the checkpoint and layer 1 from the initializer — no
   exception surfaced to the caller's data, and it trains to garbage. The gate
   caught it; `Load`/`Restore` are now two-pass.
3. **`BaseLayer.NonTrainable`** was added in P1 rather than P4 so the archive
   format would not need a break when BatchNorm's running statistics arrive.
   Optimizers demand a `Grads` entry for every `Parameters` key, so
   gradient-less state cannot live there.
4. **The float32 split point is the Keras-parity one.**
   `(int)(n * (1 - split))` in float32 gives 8 for `n=10, split=0.2f`; in double
   it gives 7. Keras (Python doubles, literal `0.2`) gives 8. The C# `float`
   literal's rounding happens to land on Python's answer where the double
   promotion does not.
5. **`NDArray` overloads `==`/`!=` element-wise**, so `x != null` is not a null
   check — it returns an `NDArray<bool>`. Use `is null` / `is not null`.
6. **`dotnet run <script>.cs` caches on the script hash, not the referenced
   project.** A gate re-run after a source-only change silently replays the old
   build. Clear `%LOCALAPPDATA%\Temp\dotnet\runfile\<name>-*` first.

Interop was checked rather than assumed: real `numpy.load()` opens the
checkpoints and returns correctly-shaped float32 arrays for every key.

Original planning table kept below for reference:

| Feature | Ref | Semantics |
|---|---|---|
| Per-epoch shuffle | Keras `fit(shuffle=True)` | `np.random.permutation(n)` + fancy-index both x and y |
| `validation_split` / `validation_data` | Keras | tail-split before training; report `val_loss`/`val_acc` per epoch |
| Callback list | Keras `Callback` | replace the single `EpochEnd` event with `on_epoch_{begin,end}`, `on_batch_end`, `on_train_{begin,end}` |
| `EarlyStopping` | Keras | monitor + patience + min_delta + restore-best-weights |
| `ModelCheckpoint` | Keras | save-best-only via P1 serialization |
| `ReduceLROnPlateau` | Keras | factor/patience/cooldown |
| `CSVLogger` | Keras | per-epoch metrics to file |
| Weight save/load | Keras `save_weights` / PyTorch `state_dict` | one `.npz` per model: `{layerName}/{paramName}` → array. Core's byte-exact npz means the file is loadable by real numpy/Keras tooling |
| Architecture save/load | Keras `to_json` | `System.Text.Json` layer-config list; `NeuralNet.FromJson` rebuilds |
| Gradient clipping | Keras `clipnorm`/`clipvalue`, PyTorch `clip_grad_norm_` | global-norm and per-value variants on `BaseOptimizer` |
| Partial final batch | Keras | stop flooring `n/batchSize` in trainer and evaluator |
| Progress verbosity | Keras `verbose=` | 0/1/2 modes on the trainer |

### P2 — Activations, losses, metrics, initializers — ✅ DONE

Shipped (all rows below): activations Tanh/LeakyReLU/ELU/GELU(tanh-approx)/
SiLU/Softplus/SELU + full resolver registration; losses
SparseCategoricalCrossentropy/Huber/KLDivergence/Hinge/LogCosh; metrics
Precision/Recall/F1Score(binary+macro)/TopKCategoricalAccuracy/
RootMeanSquaredError/R2Score + Accuracy/BinaryAccuracy rename with
`[Obsolete]` typo shims; the full `Initializers/` module (VarianceScaling with
Keras-exact truncated normals, Glorot/He/LeCun ×{uniform,normal}, Orthogonal
via Gram-Schmidt, Zeros/Ones/Constant/RandomNormal/RandomUniform, `Get(name)`
resolver) wired into both dense layers as opt-in `kernelInitializer`/
`biasInitializer` params (null keeps the historical seeded defaults
bit-for-bit). Verified by `tests/verify_p0_p2.cs`: NumPy 2.4.2 reference
constants, finite-difference gradient grids, initializer statistics.
Original planning table kept below for reference:

| Kind | Add | Ref | Notes |
|---|---|---|---|
| Activation | `Tanh` | everyone | `np.tanh` exists; grad `1−y²` |
| Activation | `LeakyReLU(alpha)` | Keras/PyTorch | `np.where(x>0, x, αx)` |
| Activation | `ELU(alpha)` | Keras | `α(eˣ−1)` for x<0 |
| Activation | `GELU` | PyTorch/transformers | tanh approximation form |
| Activation | `SiLU`/`Swish` | PyTorch | `x·σ(x)` |
| Activation | `Softplus`, `SELU` | Keras | |
| Activation | register `Softmax` + all new ones in `Get()` | — | fixes the P0 resolver gap for good |
| Loss | `SparseCategoricalCrossentropy` | Keras | int labels, no OneHot materialization — memory win on big vocabularies; gather via manual loop until `take_along_axis` lands |
| Loss | `Huber(delta)` | Keras/PyTorch | robust regression |
| Loss | `KLDivergence`, `Hinge`, `LogCosh` | Keras | |
| Metric | `Precision`, `Recall`, `F1` | sklearn | binary + macro-averaged multi-class |
| Metric | `TopKCategoricalAccuracy` | Keras | needs per-row top-k — argsort exists |
| Metric | `RMSE`, `R2Score` | sklearn/Keras | |
| Metric | rename `Accuacy`→`Accuracy` | — | keep typo'd subclass as `[Obsolete]` shim |
| Init | pluggable `IInitializer` module: `GlorotUniform/Normal`, `HeUniform/Normal`, `LecunNormal`, `Zeros/Ones/Constant`, `RandomUniform/Normal`, `Orthogonal` | Keras `initializers` | Dense ctors take an initializer instead of hardcoding; `Orthogonal` via Gram-Schmidt until `linalg.qr` |

### P3 — Optimizers & LR schedules

| Feature | Ref | Notes |
|---|---|---|
| `RMSprop` | Keras | the classic third optimizer; trivial given Adam's structure |
| `AdamW` | PyTorch | decoupled weight decay — the modern default |
| `Adagrad`, `Adadelta`, `Nadam`, `Adamax` | Keras | complete the family |
| First-class LR schedulers: `StepDecay`, `ExponentialDecay`, `CosineAnnealing`, `LinearWarmup` | Keras `schedules`, PyTorch `lr_scheduler` | replaces the buggy in-place `DecayRate` mutation; scheduler owns `lr(t)`, optimizer just reads it |
| **Fused optimizer updates** | (perf) | rewrite Adam's ~14-temp update as one/two `np.evaluate` expressions per param — this is the flagship application of core's own fusion engine, and the honest replacement for the demo's inverted fusion-probe story |

### P4 — Regularization & normalization layers — ✅ DONE

Shipped every row: the `Training` flag on `BaseLayer` (the contract change),
`Dropout`, `BatchNormalization`, `LayerNormalization`, `Embedding`,
`Flatten`/`Reshape`, and `Regularizers/` (L1/L2/L1L2). All registered with
`ModelArchitecture`, all serializing through `ModelWeights` — BatchNorm's
running statistics included, via the `NonTrainable` slot P1 added for them.

Verified two independent ways. The Keras oracle grew from 93 to **111 cases**
(`verify_edge_cases.cs`: **160 checks**), with P4 layer gradients taken from
`jax.grad` over `keras.Layer.stateless_call` — differentiating the ACTUAL Keras
implementation rather than a re-derivation. `tests/verify_p4.cs` (**102
checks**) covers what a single-layer oracle cannot: flag plumbing, running-stat
state machine, seed determinism, serialization round-trips, and independent
finite-difference gradient checks. Total gate coverage is now **479 checks**
across four scripts; the demo's headline run is unchanged (99.90%).

**Findings worth keeping:**

1. **Keras's defaults are not the common ones — probe, don't assume.**
   BatchNorm and LayerNorm both use `epsilon = 1e-3`, not the 1e-5/1e-6
   everyone else uses, and the variance is the **population** (ddof=0) one for
   both the normalization AND the running-variance update. On a batch of 3 the
   sample variance changes the output ~20%; on a batch of 256 it is invisible.
   `Embedding`'s default initializer is `RandomUniform(±0.05)`, not Glorot.
2. **The oracle caught a real bug: L1's subgradient at `w = 0`.**
   `np.sign(0)` is 0, but `jax.grad(jnp.abs)` at ±0.0 is **+1** — Keras/JAX's
   `x >= 0` branch, the same convention P2 found in LeakyReLU. `w = 0` is
   exactly where an L1 penalty spends its time, so the wrong value stalls every
   weight the penalty has already driven there.
3. **Running statistics cannot live in `Parameters`.** Optimizers iterate it and
   demand a `Grads` entry per key, so a running mean parked there either crashes
   the step or gets "optimized". Hence `NonTrainable`, added in P1.
4. **A flag beat a signature change.** `Forward(x, training)` would have broken
   every existing layer, activation and verification script for no behavioral
   gain. The cost of the flag is that it must be SET — so the gate spies on
   forward calls per mode rather than trusting the trainer.
5. **The backward passes use the folded form**
   `dx = invStd·(dxhat - mean(dxhat) - xhat·mean(dxhat·xhat))`, algebraically
   identical to the textbook `dvar`/`dmean` chain once `Σ(x-μ) = 0` is
   substituted, and exact with a non-zero epsilon since epsilon only enters
   through `σ`. Four temporaries instead of nine. Both forms are written out in
   the source so the equivalence is checkable.
6. **`jax.grad` refuses integer inputs**, which is the same fact `Embedding`
   encodes by leaving `InputGrad` null — the oracle generator skips `dx` there
   rather than working around it.

Original planning table kept below for reference:

| Feature | Ref | Core deps | Notes |
|---|---|---|---|
| `Dropout(rate)` | everyone | `np.random.bernoulli` ✅ | inverted dropout (scale by `1/(1−p)` at train time). **Requires a train/eval mode flag** — add `Training` to the forward contract (Keras `training=`, PyTorch `model.eval()`); BatchNorm needs the same flag |
| `BatchNormalization` | everyone | axis mean/var ✅ | learnable γ/β, running mean/var buffers (momentum 0.99), the full manual backward |
| `LayerNormalization` | transformers | axis mean/var ✅ | per-sample, no running stats — simpler than BN |
| `Flatten`, `Reshape` | Keras | ✅ | trivial view layers, needed the moment conv exists |
| `L1`/`L2`/`L1L2` weight regularizers | Keras | ✅ | loss term + additive grad term, attached per-layer |
| `Embedding` | Keras/PyTorch | gather ✅ (`np.take`); **scatter-add ❌** | backward accumulates duplicate-index grads — manual unsafe loop until `np.add.at` |

### P5 — Convolutional stack

The im2col + GEMM formulation runs on today's core (`np.pad`, reshape,
transpose, stride-aware `np.dot` all ✅) — correctness first, dedicated
kernels later.

| Feature | Ref | Notes |
|---|---|---|
| `Conv2D` | everyone | im2col + GEMM forward; backward = GEMM against the same patches + col2im. NCHW layout (contiguous-friendly) |
| `MaxPooling2D` / `AveragePooling2D` | everyone | reshape trick when `stride==kernel`; general case wants `sliding_window_view` (core backlog) |
| `GlobalAveragePooling2D` | Keras | axis mean ✅ — trivial |
| `Conv1D` | Keras | same im2col machinery, one spatial dim |
| Real-MNIST CNN demo | — | LeNet-style, target ≥99% test — the acceptance gate for the whole phase |
| (later) direct-conv / winograd kernel in core | — | only after profiling shows im2col-GEMM is the bottleneck |

### P6 — Sequence & attention

| Feature | Ref | Core deps | Notes |
|---|---|---|---|
| `SimpleRNN`, `LSTM`, `GRU` | Keras/PyTorch | GEMM ✅ | time-step loop of dense ops; BPTT backward |
| `MultiHeadAttention` | transformers | **batched matmul ✅ (verified)** | scaled dot-product; softmax over last axis ✅ |
| Positional encoding, `Bidirectional` wrapper | transformers/Keras | ✅ | |

### P7 — Architectural (long horizon)

| Feature | Ref | Notes |
|---|---|---|
| Functional / graph API | Keras functional | non-sequential topologies + merge layers (`Add`, `Concatenate`, `Multiply`) |
| Tape-based autograd | PyTorch, micrograd | records op graph on NDArray wrappers, derives every `Backward` automatically — replaces all hand-written layer backwards. Transformative but a rewrite of the layer contract; evaluate after P5 |
| Mixed precision | PyTorch AMP | core `Half` + SIMD casts exist; needs loss-scaling |
| ONNX weight export | — | interop escape hatch; `.npz` covers the numpy world already |

---

## 4. Cross-cutting: testing & data

- **A real MSTest project** (`test/NeuralNetwork.NumSharp.Tests`). Today the
  gates are dotnet-run scripts. Minimum bar: finite-difference gradient
  check for every layer/activation/loss `Backward` (the Softmax one already
  proved its worth), optimizer trajectory pins vs hand-computed values, and a
  seeded 2-epoch convergence pin.
- ✅ **Oracle-style parity — SHIPPED** (`tests/gen_keras_oracle.py` →
  `tests/corpus/keras_edge_oracle.json` → `tests/verify_edge_cases.cs`):
  real Keras 3 (JAX backend) values + jax.grad gradients, Keras metric
  classes, sklearn metrics — 93 committed edge cases (±inf/NaN/saturation/
  kinks/clip boundaries/ties/zero denominators), replayed with no Python;
  94 checks green, 2 excused-documented divergences (CCE renormalization).
  First run caught 3 real bugs: relu(-inf)=NaN, LeakyReLU grad at 0,
  SoftmaxCrossEntropy loss capped at -log(eps) for extreme logits.
- **Dataset/DataLoader abstraction** (PyTorch semantics: `Dataset` +
  batching/shuffling iterator) so MNIST stops being a special case; CIFAR-10
  loader when P5 lands.

---

## 5. Priority order (effort × impact)

| Rank | Item | Why first |
|---|---|---|
| 1 | ~~P0 hygiene~~ ✅ DONE | everything else builds on it; two of the bugs silently corrupt training configs |
| 2 | ~~P1 shuffle + validation + checkpoint + save/load~~ ✅ DONE | turns the demo into a usable framework; zero new math |
| 3 | ~~P2 activations/losses (esp. `Tanh`, `LeakyReLU`, `SparseCCE`)~~ ✅ DONE | hours of work each, immediate expressiveness |
| 4 | P3 `AdamW` + schedulers + fused Adam | modern training defaults; the fusion story redeemed |
| 5 | ~~P4 `Dropout` + `BatchNorm` (+ the train/eval flag)~~ ✅ DONE | the two layers that gate real-dataset accuracy |
| 6 | P5 conv stack + real-MNIST CNN | the headline capability jump |
| 7 | P6 attention/RNN | already unblocked by batched matmul |
| 8 | P7 autograd/functional | decide after the library has users |

**Core backlog to file as issues (serves NumPy parity AND this project):**
`np.take_along_axis`, `np.sliding_window_view`, scatter-add
(`np.add.at`/`np.put` accumulate mode), `np.einsum`, `np.linalg.qr`.
