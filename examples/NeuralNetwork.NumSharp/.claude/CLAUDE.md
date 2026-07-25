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
│   │                           Grads[...], InputGrad. Subclasses override
│   │                           Forward/Backward.
│   ├── FullyConnected.cs      Dense layer with bias + He/Xavier init (float32).
│   │                           Optional kernel/bias BaseInitializer overrides
│   │                           (null keeps the historical seeded defaults).
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
│   ├── SGD.cs                 Vanilla SGD; classical momentum. Inverse-time
│   │                           decay lr_t = lr0/(1+decay·t) computed FRESH
│   │                           from the base rate (never mutated).
│   └── Adam.cs                First/second moments with proper np.zeros init.
│                               Step counter must be monotonic across run.
│                               Same non-mutating decay as SGD.
│
├── MnistMlp/                  The runnable experiment. Files described below.
│
├── tests/verify_p0_p2.cs      86-check verification script (dotnet-run file-
│                               based app; excluded from the demo build via
│                               csproj Compile Remove). See Testing below.
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
| `Program.cs` | Entry point. Loads data, builds 2-FC model, runs fusion probe, trains via MlpTrainer, reports IL-kernel cache + delegate-slot counts. |
| `MnistLoader.cs` | IDX parser (big-endian) + learnable synthetic fallback (shared class templates across train/test, sigma=2.5 noise). |
| `FullyConnectedFused.cs` | FC with bias + optional fused activation. Three NDIter kernels (two forward, one backward), cache keys are stable strings. |
| `SoftmaxCrossEntropy.cs` | Combined loss computed in LOG space (log-softmax via log-sum-exp) so extreme logits give the true loss — a clipped softmax-then-log caps at -log(eps) ≈ 16.1 where e.g. logits (-1000, 0) should give 1000 (Keras from_logits=True parity, oracle-pinned). Caches softmax; (softmax-labels)/batch backward. Also ships `OneHot` helper. |
| `MlpTrainer.cs` | Explicit train loop (`NeuralNet.Train` replacement). Periodic test eval (`min(5, epochs)` cadence). Returns per-epoch loss/train_acc + list of (epoch, test_acc) pairs. |
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
`NDArray.astype.Test.cs` and `tests/verify_p0_p2.cs`). The manual max-abs-diff
loop in `Program.MaxAbsDiff` predates the fix and is now just a plain
correctness check, not a workaround.

### 5. `np.argmax(preds, axis:1)` returns Int64
When comparing against `labels.GetByte(i)` use `predIdx.GetInt64(i)` —
calling `GetInt32` on Int64 storage throws `Memory corruption expected`.

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

**100-epoch training on 6000 synthetic / 1000 test (batch=128, Adam, sigma=2.5):**
- Epoch 1: loss ≈ 1.12, train_acc ≈ 73% (random init → partial fit)
- Epoch 2: loss ≈ 0.009, train_acc ≈ 99.9%
- Epoch 100: loss ≈ 0, test_acc ≈ 99.9%
- Total training time: ~13–16 s (was ~70 s when this doc was first written —
  core's elementwise/GEMM work since then did most of that)

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

No dedicated MSTest project yet (roadmap P-cross-cutting). TWO committed
gates, both dotnet-run file-based apps under `tests/`:

### Gate 2: `tests/verify_edge_cases.cs` — the Keras edge-case oracle (94 checks)

House oracle philosophy scaled to this project: `tests/gen_keras_oracle.py`
runs **real Keras 3 (JAX backend, float32)** — values AND `jax.grad`
gradients through the actual Keras losses — plus Keras metric classes and
scikit-learn, and writes the committed corpus
`tests/corpus/keras_edge_oracle.json` (93 cases). The replay runs with **no
Python**. Coverage: activation values over ±inf/NaN/±1e30/saturation/kink
grids + gradient grids; softmax -inf lanes and huge-logit ties; loss values
and gradients at clip boundaries, |e|==delta Huber boundary, zero margins,
label-conversion edges, ±300 log-cosh tails, from-logits parity for
SoftmaxCrossEntropy at ±1000 logits; metric threshold-exact/tie/
zero-denominator conventions; initializer std targets sampled from Keras's
own initializers (5-seed means, conv-rank fans).

Excused divergences are explicit in the corpus (`expected_ns` + reason,
printed at replay — MisalignedRegistry spirit, never silent). Currently two:
Keras renormalizes CCE probability rows (we document "expects post-softmax"
and don't), and consequently Keras's CCE gradient carries a renormalization
projection while ours is the exact gradient of OUR forward (FD-verified).

```bash
cd examples/NeuralNetwork.NumSharp/tests && dotnet run verify_edge_cases.cs
# → RESULT: 94 passed, 0 failed, 2 excused-documented divergences
# regenerate corpus (needs keras>=3.15 + jax + scikit-learn):
python gen_keras_oracle.py
```

Bugs this oracle caught on first run: `relu(-inf)` returned NaN (the
`(x>0)*x` form), LeakyReLU's gradient at exactly 0 (Keras/JAX use `x >= 0`),
and SoftmaxCrossEntropy capping extreme-logit losses at -log(eps) ≈ 16.1
instead of the true value (now log-sum-exp).

### Gate 1: `tests/verify_p0_p2.cs` — behavior + formula checks (86 checks)

- P0 behavior pins: astype(copy:false) non-mutation, allclose/where operand
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
cd examples/NeuralNetwork.NumSharp/tests && dotnet run verify_p0_p2.cs
# → RESULT: 86 passed, 0 failed
```
(Must run from tests/ — a csproj in the CWD makes `dotnet run` pick the
project instead of the file. The csproj excludes tests/** from compilation.)

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
Yes — the slicing bug is fixed (uses `$"{start}:{end}"` string-slice) and
the optimizer step counter is monotonic. But `MnistMlp/MlpTrainer.cs` is
still the richer path (periodic test eval, per-epoch timing output). Use
`NeuralNet` for simple cases, `MlpTrainer` when you want instrumentation.

**Can we train on real MNIST?**
Yes — drop the four IDX files into `examples/NeuralNetwork.NumSharp/data/`.
The loader auto-detects and switches off synthetic. Real-MNIST accuracy
with this 2-layer MLP should land ~97-98% after 10-20 epochs.

---

## Known limitations

- **No data shuffling.** `MlpTrainer` iterates batches in order. Works fine
  for synthetic data and MNIST (which is pre-shuffled) but would hurt
  generalization on ordered datasets.
- **No validation split.** Train / test is a fixed split; no held-out
  validation for early stopping.
- **Adam re-allocates per step.** Each Adam update allocates ~14 temp
  NDArrays per parameter. For a 2-layer FC this is ~200 ms/epoch of GC
  pressure. Fixable by fusing Adam's update into NDIter like the rest,
  but out of scope for the current demo.
- **No model serialization.** Parameters can't be saved / loaded yet.
- **Only relu (or none) fuses.** `FullyConnectedFused` accepts the same
  string-activation surface as `FullyConnected` now, but only ReLU has a
  fused kernel — anything else throws with a pointer at `FullyConnected`.
- **GELU is the tanh approximation.** Core has no `np.erf`; the exact-erf
  form (Keras `approximate=False`, PyTorch default) differs by ~1e-3.
