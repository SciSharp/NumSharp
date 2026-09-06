# NeuralNetwork.NumSharp Example Project

A small Keras-style neural-network framework built on top of NumSharp, plus an
end-to-end MNIST 2-layer MLP demo that fuses the post-matmul element-wise work
into a single NDIter per layer via NDExpr.

Dual purpose:
1. **Library scaffolding** — `BaseLayer`, `BaseActivation`, `BaseCost`,
   `BaseOptimizer`, `BaseMetric`, `NeuralNet` (sequential model runner).
2. **Runnable MLP demo** — `MnistMlp/Program.cs` trains a 784 → 128 ReLU → 10
   classifier on real MNIST (if IDX files present) or learnable synthetic
   data (fallback).

---

## Build / Run

```bash
cd examples/NeuralNetwork.NumSharp
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
dotnet run --no-build --framework net8.0      # or --framework net10.0
```

The csproj is an **Exe** (not a library) with `OutputType=Exe`,
`AllowUnsafeBlocks=true`, multi-targets `net8.0;net10.0`. It builds against
**NumSharp's public API only** (no `InternalsVisibleTo` grant): `NDIterRef`,
`NDExpr`, `GeneratedDelegates.InnerLoopCount` (kernel-cache observability),
`DelegateSlots.RegisteredCount`, and raw buffer writes via `NDArray.Unsafe.Address`
are all public surface.

Current demo defaults (in `MnistMlp/Program.cs`):
- `Epochs = 100`, `BatchSize = 128`
- Adam lr=1e-3
- Synthetic-data noise sigma = 2.5 (in `MnistMlp/MnistLoader.cs`)
- Test evaluation every `min(5, epochs)` epochs

Place real MNIST at `examples/NeuralNetwork.NumSharp/data/`:
- `train-images.idx3-ubyte`, `train-labels.idx1-ubyte` (60k train)
- `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte` (10k test)

---

## Directory Map

```
examples/NeuralNetwork.NumSharp/
├── NeuralNet.cs               Sequential model (forward / backward / Train /
│                               Predict). Uses BaseLayer list + BaseCost +
│                               BaseOptimizer. Train now slices correctly.
├── Util.cs                    int counter for layer-name uniqueness.
│
├── Layers/
│   ├── BaseLayer.cs           Abstract: Input, Output, Parameters["w"/"b"],
│   │                           Grads[...], InputGrad, NonTrainable[...],
│   │                           Training flag, Regularizers[...]. Subclasses
│   │                           override Forward/Backward (+ GetConfig).
│   ├── FullyConnected.cs      Dense layer with bias + He/Xavier init (float32).
│   │                           Optional kernel/bias BaseInitializer overrides
│   │                           (null keeps the historical seeded defaults).
│   ├── Dropout.cs             Inverted dropout: train scales survivors by
│   │                           1/(1-rate), inference is the identity. Caches
│   │                           the SCALED mask so backward reuses it exactly.
│   ├── BatchNormalization.cs  2-D (N,C). Keras defaults momentum=0.99,
│   │                           epsilon=1e-3 (NOT 1e-5), POPULATION variance.
│   │                           gamma/beta in Parameters; moving_mean and
│   │                           moving_variance in NonTrainable.
│   ├── LayerNormalization.cs  Per-SAMPLE over the features; no running state,
│   │                           ignores Training, works at batch size 1.
│   ├── Embedding.cs           Gather forward; backward is a scatter-ADD
│   │                           (duplicate indices accumulate). InputGrad stays
│   │                           null — indices aren't differentiable.
│   └── Reshape.cs             Flatten + Reshape (target shape excludes batch).
│
├── Regularizers/
│   └── BaseRegularizer.cs     L1 / L2 / L1L2 + Get(name). Keras scaling has NO
│                               1/2, so d(l2·Σw²)/dw = 2·l2·w. L1's subgradient
│                               at 0 is +1 (Keras/JAX `w >= 0`), NOT np.sign's 0.
│   └── Activations/
│       ├── BaseActivation.cs  Get(name), case-insensitive: relu, sigmoid,
│       │                       softmax, tanh, leaky_relu, elu, gelu,
│       │                       silu/swish, softplus, selu; ""/linear/none →
│       │                       null; unknown name THROWS.
│       ├── ReLU.cs            np.maximum(x, 0) — NOT (x>0)*x, which made
│       │                       relu(-inf) = 0*-inf = NaN (Keras: 0).
│       ├── Sigmoid.cs         1/(1+exp(-x)); Backward uses cached Output.
│       ├── Softmax.cs         Numerically-stable row-wise softmax;
│       │                       Backward = Output * (grad - Σ(grad*Output, axis=1, keepdims)).
│       ├── Tanh.cs            np.tanh; Backward = grad * (1 - y²).
│       ├── LeakyReLU.cs       alpha=0.3 (Keras layer default; PyTorch uses
│       │                       0.01); branch on x >= 0 like Keras/JAX, so the
│       │                       gradient at exactly 0 is 1, not alpha.
│       ├── ELU.cs             alpha=1; neg-branch grad reuses y: αeˣ = y + α.
│       ├── GELU.cs            tanh APPROXIMATION (no np.erf in core yet);
│       │                       caches tanh(u) for the exact-derivative backward.
│       ├── SiLU.cs            x·σ(x) (Swish); caches σ.
│       ├── Softplus.cs        max(x,0)+log1p(exp(-|x|)) — overflow-safe form.
│       └── SELU.cs            fixed λ/α self-normalizing constants; pair with
│                               LecunNormal init.
│
├── Cost/
│   ├── BaseCost.cs            Abstract: Forward, Backward, float Epsilon.
│   ├── CategoricalCrossentropy.cs  L = -Σ(y*log(clip(p))) / batch;
│   │                                dL/dp = -y / clip(p) / batch.
│   ├── SparseCategoricalCrossentropy.cs  Same loss, INTEGER labels
│   │                                (Byte/Int32/Int64) — no one-hot matrix.
│   │                                Gather is an explicit loop (core lacks
│   │                                take_along_axis).
│   ├── BinaryCrossEntropy.cs       mean(-y*log(clip(p)) - (1-y)*log(1-clip(p)));
│   │                                dL/dp = (p - y) / (p*(1-p)) / N.
│   ├── MeanSquaredError.cs    mean((preds - labels)²); ∇ = 2*(preds-labels)/batch.
│   ├── Huber.cs               delta=1; quadratic≤δ, linear beyond; ∇ scaled 1/size.
│   ├── KLDivergence.cs        mean over batch of Σ yt·log(yt/yp), both clipped [eps,1].
│   ├── Hinge.cs               max(1-yt·yp, 0); {0,1} labels auto-converted to ±1
│   │                           (Keras _maybe_convert_labels).
│   └── LogCosh.cs             |e|+log1p(exp(-2|e|))-ln2 (safe); ∇ = tanh(e)/size.
│
├── Metrics/
│   ├── BaseMetric.cs          Abstract: Calculate(preds, labels) → NDArray.
│   ├── Accuracy.cs            Accuracy (+ [Obsolete] Accuacy shim). argmax
│   │                           (preds,1) == argmax(labels,1), mean.
│   ├── BinaryAccuacy.cs       BinaryAccuracy (+ [Obsolete] BinaryAccuacy shim).
│   ├── MeanAbsoluteError.cs   mean(|preds - labels|).
│   ├── Precision.cs           binary TP/(TP+FP), threshold param (strict >),
│   │                           0 when nothing predicted positive (Keras).
│   ├── Recall.cs              binary TP/(TP+FN), same conventions.
│   ├── F1Score.cs             harmonic P/R; F1Average.Binary (thresholded) or
│   │                           .Macro (argmax per-class, unweighted mean).
│   ├── TopKCategoricalAccuracy.cs  tf.in_top_k tie semantics: correct when
│   │                           #{p_j > p_true} < K.
│   ├── RootMeanSquaredError.cs
│   └── R2Score.cs             sklearn semantics incl. constant-labels edge.
│
├── Initializers/              Keras keras.initializers port; all draws flow
│   │                           through np.random → deterministic under seed.
│   ├── BaseInitializer.cs     Initialize(Shape) → float32; ComputeFans
│   │                           (Keras _compute_fans); Get(name) resolver
│   │                           (unknown throws, "" → null = layer default).
│   ├── SimpleInitializers.cs  Zeros, Ones, Constant, RandomNormal(0,.05),
│   │                           RandomUniform(±.05).
│   ├── VarianceScaling.cs     The workhorse + thin subclasses GlorotUniform/
│   │                           Normal, HeUniform/Normal, LecunUniform/Normal.
│   │                           *_normal are TRUNCATED at 2σ with Keras's
│   │                           /0.87962566103423978 std correction (rejection
│   │                           resampling, still seed-deterministic).
│   └── Orthogonal.cs          Modified Gram-Schmidt in f64 (core lacks
│   │                           linalg.qr); positive R diag ⇒ Haar, same as
│   │                           numpy's q *= sign(diag(r)). Rank>2 flattens
│   │                           Keras-style; rows<cols orthonormalizes rowsᵀ.
│
├── Optimizers/
│   ├── BaseOptimizer.cs       Abstract. Get("sgd") / Get("adam") resolvers.
│   │                           Gradient clipping lives here: ClipNorm (Keras
│   │                           PER-PARAMETER), GlobalClipNorm (whole-model,
│   │                           = PyTorch clip_grad_norm_), ClipValue. Keras
│   │                           precedence (clipnorm > global > value) and its
│   │                           clip_by_norm formula v*c/max(|v|,c) verbatim;
│   │                           clipnorm+global together throws.
│   ├── SGD.cs                 Vanilla SGD; classical momentum. Inverse-time
│   │                           decay lr_t = lr0/(1+decay·t) computed FRESH
│   │                           from the base rate (never mutated).
│   └── Adam.cs                First/second moments with proper np.zeros init.
│                               Step counter must be monotonic across run.
│                               Same non-mutating decay as SGD.
│
├── Callbacks/                 Keras keras.callbacks port.
│   ├── BaseCallback.cs        Hooks OnTrain{Begin,End}, OnEpoch{Begin,End},
│   │                           OnBatchEnd + TrainingContext (Layers, Optimizer,
│   │                           StopTraining flag = Keras model.stop_training).
│   │                           Epoch/batch indices are 0-BASED like Keras.
│   ├── EarlyStopping.cs       monitor/patience/min_delta/mode/baseline/
│   │                           restore_best_weights/start_from_epoch.
│   ├── ModelCheckpoint.cs     save_best_only + {epoch:D3}/{val_loss:F4} path
│   │                           placeholders ({epoch} is 1-based, as in Keras).
│   ├── ReduceLROnPlateau.cs   factor/patience/cooldown/min_lr; writes the
│   │                           optimizer's BASE LearningRate.
│   └── CSVLogger.cs           columns frozen at epoch 0 (sorted keys), the
│                               `epoch` column is 0-based (Keras writes it raw).
│
├── Serialization/             Weight + architecture persistence.
│   ├── ModelWeights.cs        Save/Load one .npz; Capture/Restore in-memory
│   │                           snapshots. Keys are `layer{i}/param/{name}` and
│   │                           `layer{i}/state/{name}` — POSITIONAL, see below.
│   ├── LayerConfig.cs         {class_name, config} descriptor + typed getters.
│   └── ModelArchitecture.cs   ToJson/FromJson + the layer-factory registry.
│
├── MnistMlp/                  The runnable experiment. Files described below.
│
├── tests/VerifyComponentsTests.cs   86-check gate (dotnet-run file-based app,
│      excluded from the demo build via csproj Compile Remove). Activations,
│      losses, metrics, initializers, optimizer decay. See Testing below.
├── tests/VerifyKerasOracleTests.cs  160-check Keras 3 edge-case oracle replay
│      (activations/losses/metrics/inits + the layers); no Python at test time.
├── tests/VerifyTrainingLoopTests.cs 131-check gate (serialization, gradient
│      clipping, the four callbacks, trainer behavior).
├── tests/VerifyLayersTests.cs       102-check gate (Training-flag plumbing,
│      layer state machines, FD gradients, serialization).
├── Open.snk                   Strong-name key shared with NumSharp.Core.
└── NeuralNetwork.NumSharp.csproj   Exe, net8.0+net10.0, AllowUnsafeBlocks.
```

---

## MnistMlp — fused forward + backward

All fusion happens in `FullyConnectedFused`. The idea: every post-matmul
element-wise chunk (bias-add + ReLU, bias-add only, ReLU gradient mask)
collapses into **one NDIter kernel**, compiled once per process and
cache-hit on every subsequent forward/backward pass.

| Stage | NDExpr tree | Inputs → Output |
|---|---|---|
| Forward ReLU | `Max(Input(0) + Input(1), Const(0f))` | (preact, bias) → y |
| Forward linear | `Input(0) + Input(1)` | (preact, bias) → y |
| Backward ReLU | `Input(0) * Greater(Input(1), Const(0f))` | (gradOut, y) → gradPreact |
| Backward linear | — (pass-through) | gradOut → gradPreact |

**`MnistMlp/` files:**

| File | What it does |
|---|---|
| `Program.cs` | Entry point. Loads data, builds 2-FC model, runs fusion probe, trains via MlpTrainer, runs the **serialization + callbacks showcase** (weights→.npz + architecture→JSON, rebuild-from-JSON and reload — random 8.9% → restored 99.90%, then a 12-epoch fit driven by EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + CSVLogger with `validationSplit: 0.1` and `ClipNorm = 1.0`) and the **regularization + normalization showcase** (Dense→BatchNorm→Dropout→Dense + L2, 8 epochs; then the same input forwarded with `Training` false twice and true once, showing eval is deterministic and train differs), reports IL-kernel cache + delegate-slot counts. Both showcases are deliberately SEPARATE short runs so the headline 100-epoch numbers stay comparable across commits. |
| `MnistLoader.cs` | IDX parser (big-endian) + learnable synthetic fallback (shared class templates across train/test, sigma=2.5 noise). |
| `FullyConnectedFused.cs` | FC with bias + optional fused activation. Three NDIter kernels (two forward, one backward), cache keys are stable strings. |
| `SoftmaxCrossEntropy.cs` | Combined loss computed in LOG space (log-softmax via log-sum-exp) so extreme logits give the true loss — a clipped softmax-then-log caps at -log(eps) ≈ 16.1 where e.g. logits (-1000, 0) should give 1000 (Keras from_logits=True parity, oracle-pinned). Caches softmax; (softmax-labels)/batch backward. Also ships `OneHot` helper. |
| `MlpTrainer.cs` | Explicit train loop (`NeuralNet.Train` replacement). Per-epoch shuffle, `validationSplit`/`validationData`, callback list, partial final batch, verbose 0/1/2, periodic test eval (`min(5, epochs)` cadence). Returns per-epoch loss/train_acc/val_loss/val_acc, (epoch, test_acc) pairs, `EpochsRun` and `StoppedEarly`. `EvaluateFull` scores loss AND accuracy in one pass. |
| `FusedMlp.cs`, `NaiveMlp.cs` | Side-by-side forward implementations for the correctness probe at Program startup. |

---

## Layer / Cost / Optimizer contract

Every BaseLayer subclass MUST populate on Forward:
- `this.Input = x` (via `base.Forward(x)`)
- `this.Output = result`

And on Backward:
- `this.Grads[key] = ∂L/∂param` for every entry in `this.Parameters`
- `this.InputGrad = ∂L/∂x` (consumed by the previous layer)

Optimizers iterate `layer.Parameters.ToList()` and expect `layer.Grads[paramKey]`
to be populated by Backward. Param-name convention is `"w"` / `"b"`.

**`BaseLayer.NonTrainable`** holds tensors that belong to the layer but must
never reach an optimizer — BatchNorm's running mean/variance being the case it
exists for. Anything in `Parameters` needs a matching `Grads` entry or the
optimizer throws, so gradient-less state goes here instead. Serialization walks
BOTH dictionaries, so a checkpoint carries running statistics.

**`BaseLayer.GetConfig()`** returns the `{class_name, config}` descriptor used by
`ModelArchitecture`. The default reports only the type name, which serializes as
"unsupported"; a layer that wants JSON round-tripping overrides it AND registers
a factory via `ModelArchitecture.Register`.

## Training-loop contract

`MlpTrainer.Train` is the full loop. Beyond the original positional arguments it
takes `shuffle` (default **true**, Keras's default), `validationSplit`,
`validationData`, `callbacks` and `verbose` (0 silent / 1 per-epoch / 2 per-batch).

- **Log keys are Keras-named**: `loss`, `acc`, `val_loss`, `val_acc`,
  `learning_rate`. Callbacks match these by string.
- **`validationSplit` takes the LAST fraction and does so BEFORE shuffling**, as
  Keras does — so the split is deterministic and data ordered by class must be
  shuffled by the caller first. `validationData` overrides it entirely.
  The split point is `(int)(n * (1 - split))` computed in **float32**, which is
  what reproduces Keras's Python-float answer for the usual literals; doing it in
  double does not (`n=10, split=0.2f` → 8 in float, 7 in double, 8 in Keras).
- **Every sample trains, including a partial final batch**, and epoch loss and
  accuracy are averaged over SAMPLES — so a short last batch is weighted
  correctly rather than counting as a whole batch. (`SoftmaxCrossEntropy.Backward`
  already divides by `preds.shape[0]`, so ragged batches were always gradient-correct.)
- **Shuffling gathers per batch** from one `np.random.permutation(n)` per epoch
  rather than materializing a shuffled copy of the whole set — peak memory stays
  at one batch. All randomness flows through `np.random` (MT19937), so a seeded
  run is reproducible end to end.
- `TrainResult` gained `EpochValLoss`, `EpochValAcc`, `EpochsRun` and
  `StoppedEarly`. `Epochs` remains the number REQUESTED; `EpochsRun` is what
  actually ran.

## Train/eval mode contract

`BaseLayer.Training` is the Keras `training=` / PyTorch `model.eval()` analog,
as a **flag** rather than a `Forward` parameter — the signature change would
break every existing layer, activation and verification script for no
behavioral gain.

- **Default is `false`**, so a bare `Forward` is always inference.
- `MlpTrainer` sets it **true** around the training forward pass and **false**
  in `Evaluate`/`EvaluateFull`; `NeuralNet.Train` sets true, `NeuralNet.Predict`
  sets false. **A hand-rolled loop must do the same** — otherwise Dropout does
  not drop and BatchNorm normalizes with stale running statistics, and both
  failures are silent (the model trains, just worse).
- Only `Dropout` and `BatchNormalization` read it. `LayerNormalization`
  deliberately does not — it has no running state and behaves identically in
  both modes.

**Regularizers are applied by the TRAINER, not by each layer's `Backward`.**
After the backward sweep the trainer calls `layer.ApplyRegularizerGradients()`
on every layer and adds `layer.RegularizationPenalty()` to the reported loss
(as Keras does). Centralizing it means a layer author cannot forget to honour a
regularizer someone attached to their layer. Do NOT gate the gradient call on
`penalty != 0` — that conflates "no regularizer" with "a regularizer that
happens to score zero right now".

**Checkpoint keys are POSITIONAL, not by layer `Name`.** `Name` comes from
`Util.GetNext()`, a process-global counter that never resets — build the same
architecture twice in one process and the second copy is `fc_fused2`/`fc_fused3`.
Name-keyed checkpoints would therefore fail to load into a freshly-built model,
which is the entire point of a checkpoint. `ModelWeights` keys by layer INDEX
(`layer0/param/w`), so the caller must rebuild the same architecture in the same
order. Every mismatch is a hard error naming the slot; validation of ALL tensors
completes before ANY is written, so a mismatch in a late layer cannot leave the
model half-overwritten (a model with layer 0 from the checkpoint and layer 1 from
the initializer trains to garbage without raising anything — this was a real bug
the training-loop gate caught).

BaseCost contract:
- `Forward(preds, labels)` → scalar NDArray (the loss)
- `Backward(preds, labels)` → NDArray shape-matched to preds (the first
  incoming gradient for the network's output layer)

BaseMetric contract:
- `Calculate(preds, labels)` → scalar NDArray in [0, 1]

---

## Sharp edges that bit us

### 1. np.dot + strided operands (historical)
Before the stride-aware GEMM shipped in `f5c05a7f`, `np.dot(x.T, grad)` with
non-contiguous operands was **~100x slower** than contiguous (240 ms vs 2.5 ms
on the layer-1 backward shapes). Workaround was `.transpose().copy()` before
the dot. Now removed — the stride-aware kernel handles transposed views
directly and is ~1.4x slower than fully-contig (normal stride overhead).
Don't add `.copy()` back.

### 2. `x[i, j]` is 2-index element selection, NOT a slice
`NeuralNet.Train` originally did `x[currentIndex, currentIndex + batchSize]`
which read a single element, not a batch. Correct form:
`x[$"{start}:{end}"]` — string-slicing the outer dim returns a view.

### 3. `np.argmax(x)` without axis returns a scalar
For batched predictions you need `axis: 1`. The metrics previously returned
scalars that matched two scalar argmaxes — broken for batches.

### 4. `np.allclose` used to mutate its arguments — FIXED in core
Historical: `np.allclose`/`np.isclose`/`np.where` called
`astype(Double, copy:false)` on their operands, and core's `Cast(copy:false)`
swapped the input's storage in-place — silently flipping caller arrays from
Single to Double. Fixed in core (`Default.Cast.cs`): `astype(copy:false)` now
follows NumPy semantics — returns the input itself only when no conversion is
needed, otherwise allocates a new array and NEVER touches the input (pinned by
`NDArray.astype.Test.cs` and `tests/VerifyComponentsTests.cs`). The manual max-abs-diff
loop in `Program.MaxAbsDiff` predates the fix and is now just a plain
correctness check, not a workaround.

### 5. `np.argmax(preds, axis:1)` returns Int64
When comparing against `labels.GetByte(i)` use `predIdx.GetInt64(i)` —
calling `GetInt32` on Int64 storage throws `Memory corruption expected`.

### 5b. `NDArray` overloads `==`/`!=` ELEMENT-WISE — `x != null` is not a null check
`np.array(...) != null` compiles and returns an `NDArray<bool>`, so
`if (someNDArray != null)` is a type error at best and a silent wrong answer at
worst. Use the pattern form: `x is null` / `x is not null`, which ignores
operator overloads. Cost the training-loop trainer 18 compile errors on first build.

### 5c. `dotnet run <script>.cs` caches on the SCRIPT, not the referenced project
A file-based app is cached under
`%LOCALAPPDATA%\Temp\dotnet\runfile\<name>-<hash of the .cs>`. Edit
`NeuralNetwork.NumSharp` source, leave the script alone, re-run the gate — and it
replays the OLD build, reporting a failure you have already fixed (or, worse,
green on code you just broke). Clear it before trusting a gate run after a
source-only change:
```bash
rm -rf "$LOCALAPPDATA/Temp/dotnet/runfile/Verify"*
```

### 6. Adam step counter MUST be monotonic across the full run
Don't reset per epoch. Adam's `1 - β^t` bias correction needs `t` to increase
monotonically across the whole training run, otherwise the first batch of
each epoch gets the same broken divisor (`1 - β^1` with β^1 close to β →
large correction factor).

### 7. FullyConnected weight init was `normal(0.5, 1, ...)` (wrong)
Float64 dtype, mean=0.5. Now He-normal for ReLU, Xavier/Glorot otherwise,
all float32. If you see the class still using that init, you're looking at
a pre-fix checkout.

### 8. Slice view dtype
`images[$"0:{BatchSize}"]` preserves dtype. Feeding the slice directly to
`np.dot` works. But the `np.dot` result dtype depends on input dtypes —
float32 × float32 → float32, as expected. Use `.astype(NPTypeCode.Single)`
after `np.random.normal(...)` which returns float64 by default.

---

## Perf characteristics

Measured 2026-07-25 on core `badd9c37`, **Release**, net8.0 (Debug taints
kernels ~2x — never quote Debug numbers).

**100-epoch training on 6000 synthetic / 1000 test (batch=128, Adam, sigma=2.5,
`np.random.seed(1337)`, shuffled):**
- **47** batches/epoch — was 46; the training-loop work stopped flooring `n/batchSize`, so the
  6000th sample is no longer dropped every epoch
- Epoch 1: loss **1.1004**, train_acc 73.03% (random init → partial fit)
- Epoch 2: loss 0.0110, train_acc 99.83%
- Epoch 100: loss ≈ 0, **test_acc 99.90%**
- Total training time: ~13–15 s (was ~70 s when this doc was first written —
  core's elementwise/GEMM work since then did most of that)

The training-loop work moved the epoch-1 loss from the previously-documented 1.1247 to 1.1004: the
epoch now shuffles and covers 47 batches instead of 46, so it is a different
(and slightly better-conditioned) epoch. Final test accuracy is unchanged at
99.90% — convergence is the pin that matters, and it held.

**Fusion probe (bias+ReLU post-matmul; three paths × two sizes):**

| Size | fused NDIter | np.evaluate | naive add+maximum | naive/fused |
|---|---|---|---|---|
| 128×128 (16K) | ~0.09 ms | ~0.10 ms | ~0.06 ms | **0.60x — naive wins** |
| 2048×2048 (4.2M) | ~18.7 ms | ~18.4 ms | ~4.8 ms | **0.26x — naive wins** |

**The fusion premise is currently INVERTED for this expression.** The original
2.5x fused-wins measurement predates core's DirectILKernelGenerator SIMD
elementwise/broadcast kernels; those unfused whole-array passes now run at
full memory bandwidth and beat the NDIter per-chunk path at every size for a
2-op expression. np.evaluate's documented 3.2–6.1x wins are on LONG chains
(many intermediates eliminated) — bias+ReLU is too short a chain to amortize
the iterator. The demo keeps the fused layers as an architecture/correctness
showcase and the probe now reports both sizes honestly. Core follow-up worth
filing: NDIter broadcast-operand inner-loop throughput vs the Direct kernels.

**Instrumentation (after a 100-epoch run):**
- IL kernel cache entries: delta of 10 (unique fused expressions + probe paths)
- NDExpr delegate slots: 0 (pure DSL, no captured lambdas)

---

## Testing

No dedicated MSTest project yet (roadmap P-cross-cutting). FOUR committed
gates, all dotnet-run file-based apps under `tests/` — **479 checks total**
(86 + 160 + 131 + 102). Run them all after any source change — and see sharp
edge 5c: **clear the runfile cache first**, or a gate replays a stale build.

```bash
rm -rf "$LOCALAPPDATA/Temp/dotnet/runfile/Verify"*
cd examples/NeuralNetwork.NumSharp/tests
for f in VerifyComponentsTests VerifyKerasOracleTests VerifyTrainingLoopTests VerifyLayersTests; do dotnet run $f.cs; done
```

### Gate 4: `tests/VerifyLayersTests.cs` — regularization & normalization (102 checks)

The VALUES and GRADIENTS of the layers are pinned against real Keras by
Gate 2; this gate covers what an oracle over single layers cannot see:

- **The Training flag is actually plumbed** — a spy layer counts forwards per
  mode and proves `MlpTrainer` trains with it true and scores with it false,
  that `Evaluate` is pure inference, and that `NeuralNet.Predict`/`Train` agree.
- **Running statistics update in training and NOT in inference**, and are absent
  from `Parameters` so an optimizer step over a BatchNorm layer does not throw.
- **Dropout** seed-determinism through `np.random`, `rate=0` as the identity in
  training, and E[x] preservation over 160 000 draws.
- **Independent finite-difference gradient checks** for BatchNorm (training AND
  inference branches) and LayerNorm — computed with no Keras involved, so the
  hand-derived backward passes have a second opinion.
- **Embedding**: duplicate-index accumulation checked by hand (7 and 70 from
  three occurrences of one row), index dtypes, 2-D sequence input, and a real
  3-layer stack that trains with the null `InputGrad` at its head.
- **Regularizer wiring**: the penalty reaching the reported loss, the gradient
  reaching the optimizer (a zero-input run where the ONLY gradient is the
  penalty's, so the weights must decay).
- **Serialization**: the new layers through `.npz` + architecture JSON, with a
  reloaded model asserted to EVALUATE identically — which is the reason
  BatchNorm's running statistics are in the archive at all.

```bash
cd examples/NeuralNetwork.NumSharp/tests && dotnet run VerifyLayersTests.cs
# → RESULT: 102 passed, 0 failed
```

### Gate 2: `tests/VerifyKerasOracleTests.cs` — the Keras edge-case oracle (160 checks)

House oracle philosophy scaled to this project: `tests/gen_keras_oracle.py`
runs **real Keras 3 (JAX backend, float32)** — values AND `jax.grad`
gradients through the actual Keras losses and LAYERS — plus Keras metric
classes and scikit-learn, and writes the committed corpus
`tests/corpus/keras_edge_oracle.json` (111 cases). The replay runs with **no
Python**. Coverage: activation values over ±inf/NaN/±1e30/saturation/kink
grids + gradient grids; softmax -inf lanes and huge-logit ties; loss values
and gradients at clip boundaries, |e|==delta Huber boundary, zero margins,
label-conversion edges, ±300 log-cosh tails, from-logits parity for
SoftmaxCrossEntropy at ±1000 logits; metric threshold-exact/tie/
zero-denominator conventions; initializer std targets sampled from Keras's
own initializers (5-seed means, conv-rank fans); and the **normalization / embedding layers**.

**The layer gradients come from `jax.grad` over `keras.Layer.stateless_call`**,
so the oracle differentiates the ACTUAL Keras implementation rather than a
re-derivation of it. Every layer case carries a non-uniform upstream cotangent
— a vector of ones hides an axis mix-up, and LayerNorm's gamma/beta gradients
reduce over a different axis than its input gradient. Cases: BatchNorm
train+inference forward, `dx`/`dgamma`/`dbeta`, and the running-stat update
(plus a batch-of-1 case where the variance is exactly 0 so epsilon alone
divides — the case that separates 1e-3 from 1e-5); LayerNorm the same plus a
batch-independence pair; Embedding with duplicate / all-same / no-repeat
indices and 2-D sequence input; Dropout's inverted-scaling contract; and the
regularizer penalties with their gradients.

Two generator details that are load-bearing: `training=` is forwarded only to
layers whose `call()` declares it (LayerNormalization and Embedding raise
`TypeError` otherwise), and an **integer input is skipped for `dx`** because
`jax.grad` refuses to differentiate it — which is the same fact our `Embedding`
encodes by leaving `InputGrad` null.

Excused divergences are explicit in the corpus (`expected_ns` + reason,
printed at replay — MisalignedRegistry spirit, never silent). Currently two:
Keras renormalizes CCE probability rows (we document "expects post-softmax"
and don't), and consequently Keras's CCE gradient carries a renormalization
projection while ours is the exact gradient of OUR forward (FD-verified).

```bash
cd examples/NeuralNetwork.NumSharp/tests && dotnet run VerifyKerasOracleTests.cs
# → RESULT: 160 passed, 0 failed, 2 excused-documented divergences
# regenerate corpus (needs keras>=3.15 + jax + scikit-learn):
KERAS_BACKEND=jax python gen_keras_oracle.py
```

Bugs this oracle caught on first run. In the activations & losses: `relu(-inf)` returned NaN (the
`(x>0)*x` form), LeakyReLU's gradient at exactly 0 (Keras/JAX use `x >= 0`),
and SoftmaxCrossEntropy capping extreme-logit losses at -log(eps) ≈ 16.1
instead of the true value (now log-sum-exp). In the layers: **L1's subgradient at
`w = 0`** — `np.sign(0)` is 0, but `jax.grad(abs)` at ±0.0 is **+1**, the same
`x >= 0` convention as LeakyReLU. That is exactly where an L1 penalty spends
its time, so the wrong value stalls every weight the penalty has already
driven to zero. Fixed via `L1.SubGradient`; do not "simplify" it back to
`np.sign`.

### Gate 1: `tests/VerifyComponentsTests.cs` — behavior + formula checks (86 checks)

- Behavior pins: astype(copy:false) non-mutation, allclose/where operand
  dtypes, activation resolver (softmax registered, unknown throws), fused
  string ctor, SGD/Adam non-compounding decay from the base rate,
  Evaluate scoring the partial final batch
- All 7 new activations: forward + backward vs NumPy 2.4.2 reference
  constants AND central-difference gradient checks (kink points at x=0 for
  leaky_relu/selu are excluded from FD — central differences measure the
  average one-sided slope there)
- All 5 new losses: values + gradients vs NumPy constants + FD grids
- Metrics vs hand/NumPy-computed constants (precision/recall/F1 binary+macro,
  top-k tie semantics, RMSE, R², typo-shim compat)
- Initializers: seeded determinism, mean/std within 3%, 2σ truncation bound,
  uniform bounds, orthogonality (QᵀQ=I tall / QQᵀ=I wide, gain²)

```bash
cd examples/NeuralNetwork.NumSharp/tests && dotnet run VerifyComponentsTests.cs
# → RESULT: 86 passed, 0 failed
```
(Must run from tests/ — a csproj in the CWD makes `dotnet run` pick the
project instead of the file. The csproj excludes tests/** from compilation.)

### Gate 3: `tests/VerifyTrainingLoopTests.cs` — training-loop parity (131 checks)

Pins the training-loop behavior, in five sections:

- **Serialization** — npz key layout and entry count, dtype preserved,
  round-trip into a differently-initialized model, load-is-a-copy (not an
  alias), snapshot Capture/Restore independence and reusability, and the three
  failure modes (missing key / shape mismatch / wrong layer count) each naming
  the offending slot. Includes the regression that motivated the two-pass
  design: **a failed load must leave layer 0 untouched.**
- **Gradient clipping** — `clipnorm` scaling `[3,4]→[1.5,2]`, under-threshold
  clipping being EXACTLY the identity, `clipvalue` element clamps,
  `global_clipnorm` sharing one norm across two layers (the case where the
  Keras and PyTorch conventions visibly disagree), and the rejected pair in
  both assignment orders.
- **EarlyStopping** — driven against synthetic metric sequences, because the
  point is the state machine and a real loss curve reaches these branches only
  by luck: patience arithmetic, the `epoch > 0` guard, the SIGN of `min_delta`
  (a matched pair — with and without — where an unsigned delta inverts the
  test), `auto` mode resolving `val_acc` to maximize, explicit-mode override,
  `start_from_epoch`, restore-vs-keep weights, and the baseline's documented
  asymmetry (an improvement that misses the baseline updates `best` but does
  NOT reset `wait` — again as a matched pair).
- **ReduceLROnPlateau / ModelCheckpoint / CSVLogger** — cooldown blocking then
  releasing, `min_lr` flooring the cut and then stopping further cuts,
  save-best-only counts, 1-based `{epoch}` in filenames, frozen CSV columns and
  the 0-based `epoch` column.
- **Trainer** — a spy layer records which sample indices each batch contained,
  which pins `validation_split` taking the TAIL, the partial final batch being
  trained on rather than dropped, shuffling being a genuine permutation that is
  seed-deterministic and seed-sensitive, sample-weighted epoch accuracy
  (`4/5`, not the batch-mean `0.5`), hook counts and ordering, and verbose 0/2.

```bash
cd examples/NeuralNetwork.NumSharp/tests && dotnet run VerifyTrainingLoopTests.cs
# → RESULT: 131 passed, 0 failed
```

Ad-hoc sanity checks still work as stdin scripts:
```bash
cat /tmp/script.cs | dotnet_run
```
where the script references the project via `#:project`.

---

## Q&A

**Why do we have both `FullyConnected` and `FullyConnectedFused`?**
`FullyConnected` is the vanilla version that goes through `np.dot + (x + b) +
activation` as separate ops. `FullyConnectedFused` collapses bias+activation
into a single NDIter — the fusion demo's point. Both share the BaseLayer
contract and are interchangeable in a NeuralNet pipeline.

**Why do the metric classes have typos in their names?**
`Accuacy`, `BinaryAccuacy` — misspelled in the original scaffolding. The
correctly-named `Accuracy` / `BinaryAccuracy` are now the real classes; the
typo'd names survive as `[Obsolete]` shims subclassing them, so external
callers keep compiling (with a warning) and can migrate at leisure.

**Why is SoftmaxCrossEntropy in `MnistMlp/` instead of `Cost/`?**
It's the combined-form loss — assumes softmax is applied internally, not by
a separate Softmax layer. The standalone `Softmax` + `CategoricalCrossentropy`
chain still works and is numerically fine for most cases; SCE is faster and
slightly more stable for the MLP demo's specific pipeline.

**Is `NeuralNet.Train` usable now?**
Yes — the slicing bug is fixed (uses `$"{start}:{end}"` string-slice), the
optimizer step counter is monotonic, and it accepts a callback list and applies
global-norm clipping. But `MnistMlp/MlpTrainer.cs` is still the richer path
(shuffling, validation split, partial final batch, verbosity, periodic test
eval). Use `NeuralNet` for simple cases, `MlpTrainer` for real training.

**Can I load a NumSharp checkpoint in Python?**
Yes, with plain `numpy` and no .NET involved — `ModelWeights.Save` goes through
core's byte-exact `np.savez`, so the file IS a NumPy archive. Verified:
```python
z = np.load("mlp.npz")
z.files          # ['layer0/param/w', 'layer0/param/b', 'layer1/param/w', 'layer1/param/b']
z['layer0/param/w'].shape, z['layer0/param/w'].dtype   # ((784, 128), float32)
```
The architecture JSON is ordinary JSON. Note this is NOT Keras's own
`.weights.h5` format — interop is with the numpy world, not with `load_model`.

**Can we train on real MNIST?**
Yes — drop the four IDX files into `examples/NeuralNetwork.NumSharp/data/`.
The loader auto-detects and switches off synthetic. Real-MNIST accuracy
with this 2-layer MLP should land ~97-98% after 10-20 epochs.

---

## Known limitations

- **Adam re-allocates per step.** Each Adam update allocates ~14 temp
  NDArrays per parameter. For a 2-layer FC this is ~200 ms/epoch of GC
  pressure. Fixable by fusing Adam's update into NDIter like the rest,
  but out of scope for the current demo (roadmap P3).
- **Architecture JSON covers only registered layer types.** `ModelArchitecture`
  refuses to serialize a layer with no factory rather than emitting something
  that cannot be read back. Register user layers explicitly.
- **BatchNorm/LayerNorm are 2-D only.** `(N, C)` inputs; the `(N, C, H, W)`
  reduction over spatial axes arrives with the P5 conv stack. Both throw a
  named `NotSupportedException` rather than silently normalizing the wrong axis.
- **Embedding's gradient is DENSE.** A `(vocab, dim)` zero array is allocated per
  backward and scattered into. Real frameworks keep a sparse gradient for large
  vocabularies, which needs optimizer support this project does not have.
- **`Reshape` needs an explicit shape** — no inferred `-1` dimension.
- **Regularizers attach per parameter key**, and only the trainer applies them.
  A hand-rolled training loop must call `ApplyRegularizerGradients()` and add
  `RegularizationPenalty()` itself.
- **`NeuralNet.Train` is the thin loop.** It gained callbacks and global-norm
  clipping, but shuffling, validation and partial final batches live in
  `MlpTrainer.Train`. Use `MlpTrainer` for real training.
- **Only relu (or none) fuses.** `FullyConnectedFused` accepts the same
  string-activation surface as `FullyConnected` now, but only ReLU has a
  fused kernel — anything else throws with a pointer at `FullyConnected`.
- **GELU is the tanh approximation.** Core has no `np.erf`; the exact-erf
  form (Keras `approximate=False`, PyTorch default) differs by ~1e-3.
