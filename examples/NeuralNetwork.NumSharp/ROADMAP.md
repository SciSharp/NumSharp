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
| Layers | `FullyConnected`, `FullyConnectedFused` | Dense only. Fused variant folds bias+ReLU into one NDIter kernel |
| Activations | `ReLU`, `Sigmoid`, `Softmax` | `BaseActivation.Get()` resolves only `"relu"`/`"sigmoid"` — Softmax class exists but is **unreachable by name** |
| Losses | `MeanSquaredError`, `BinaryCrossEntropy`, `CategoricalCrossentropy`, `SoftmaxCrossEntropy` | SCE is the combined stable form, lives in `MnistMlp/` |
| Metrics | `Accuacy` (sic), `BinaryAccuacy` (sic), `MeanAbsoluteError` | Typo'd class names kept for compat |
| Optimizers | `SGD` (momentum, decay), `Adam` | `BaseOptimizer.Get()` resolves `"sgd"`/`"adam"` |
| Init | He-normal (ReLU), Xavier-normal (else) | Hardcoded in the Dense ctors, not a pluggable module |
| Data | `MnistLoader` (IDX + synthetic), `OneHot` helper | No general Dataset/DataLoader abstraction |
| Serialization | — | None. Weights cannot be saved/loaded |
| Training loop | ordered batches, whole-batch floor | No shuffling, no validation split, no callbacks, no gradient clipping |

**Known defects to fix before building on top (P0):**

1. `BaseActivation.Get("softmax")` returns `null` — the class exists but isn't registered.
2. `FullyConnected` selects activation by string, `FullyConnectedFused` by enum — inconsistent surface.
3. `SGD`/`Adam` `DecayRate` **compounds** onto the already-decayed rate
   (`lr *= 1/(1+decay·t)`) instead of Keras's `lr_t = lr0/(1+decay·t)`.
   Inert at the default `decay=0`, wrong the moment anyone enables it.
4. `MlpTrainer.Evaluate` floor-divides into whole batches — with 1000 test
   samples at batch 128 only 896 are scored.
5. Stale perf docs: the fusion probe now measures **0.73×** (fused *slower*
   than naive) in Release on current core — the naive `np.add`+`np.maximum`
   path got ~6× faster since the CLAUDE.md was written, while the hand-rolled
   NDIter path didn't. Training is 12.8 s, not ~70 s; kernel-cache delta is 9,
   not 6. The demo predates `np.evaluate` and should migrate to it.
6. Core bug leaking into this project: `np.allclose` mutates operand dtypes
   (documented workaround: manual max-abs-diff).

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

### P0 — Hygiene (fix what exists)

All six defects in §1. Plus: re-tune or re-frame the fusion probe (larger
tensors and/or `np.evaluate`) so the demo's premise holds again, and refresh
the project CLAUDE.md perf section. No new features until the foundation is
honest.

### P1 — Training-loop parity (pure C#, no new math)

The gap between "demo" and "usable framework" is almost entirely here.

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

### P2 — Activations, losses, metrics, initializers (elementwise; all core-ready)

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

### P4 — Regularization & normalization layers

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

- **A real MSTest project** (`test/NeuralNetwork.NumSharp.UnitTest`). Today the
  smoke test is an ad-hoc stdin script. Minimum bar: finite-difference gradient
  check for every layer/activation/loss `Backward` (the Softmax one already
  proved its worth), optimizer trajectory pins vs hand-computed values, and a
  seeded 2-epoch convergence pin.
- **Oracle-style parity where a Python twin exists**: losses/metrics/optimizer
  steps can be pinned against Keras/PyTorch outputs the same way core pins
  against NumPy — small committed fixtures, no Python at test time.
- **Dataset/DataLoader abstraction** (PyTorch semantics: `Dataset` +
  batching/shuffling iterator) so MNIST stops being a special case; CIFAR-10
  loader when P5 lands.

---

## 5. Priority order (effort × impact)

| Rank | Item | Why first |
|---|---|---|
| 1 | P0 hygiene | everything else builds on it; two of the bugs silently corrupt training configs |
| 2 | P1 shuffle + validation + checkpoint + save/load | turns the demo into a usable framework; zero new math |
| 3 | P2 activations/losses (esp. `Tanh`, `LeakyReLU`, `SparseCCE`) | hours of work each, immediate expressiveness |
| 4 | P3 `AdamW` + schedulers + fused Adam | modern training defaults; the fusion story redeemed |
| 5 | P4 `Dropout` + `BatchNorm` (+ the train/eval flag) | the two layers that gate real-dataset accuracy |
| 6 | P5 conv stack + real-MNIST CNN | the headline capability jump |
| 7 | P6 attention/RNN | already unblocked by batched matmul |
| 8 | P7 autograd/functional | decide after the library has users |

**Core backlog to file as issues (serves NumPy parity AND this project):**
`np.take_along_axis`, `np.sliding_window_view`, scatter-add
(`np.add.at`/`np.put` accumulate mode), `np.einsum`, `np.linalg.qr`.
