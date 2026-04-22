# NeuralNetwork.NumSharp Example Project

A small Keras-style neural-network framework built on top of NumSharp, plus an
end-to-end MNIST 2-layer MLP demo that fuses the post-matmul element-wise work
into a single NpyIter per layer via NpyExpr.

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
`AllowUnsafeBlocks=true`, multi-targets `net8.0;net10.0`. It has
`InternalsVisibleTo("NeuralNetwork.NumSharp")` in `src/NumSharp.Core/Assembly/
Properties.cs`, so `NpyIterRef`, `NpyExpr`, `ILKernelGenerator.InnerLoopCachedCount`,
and `DelegateSlots.RegisteredCount` are all accessible.

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
│   │                           Composes an optional BaseActivation by name.
│   └── Activations/
│       ├── BaseActivation.cs  Get(name): resolves "relu"/"sigmoid" by name.
│       ├── ReLU.cs            (NDArray > 0) * NDArray formulation (works).
│       ├── Sigmoid.cs         1/(1+exp(-x)); Backward uses cached Output.
│       └── Softmax.cs         Numerically-stable row-wise softmax;
│                               Backward = Output * (grad - Σ(grad*Output, axis=1, keepdims)).
│
├── Cost/
│   ├── BaseCost.cs            Abstract: Forward, Backward, float Epsilon.
│   ├── CategoricalCrossentropy.cs  L = -Σ(y*log(clip(p))) / batch;
│   │                                dL/dp = -y / clip(p) / batch.
│   ├── BinaryCrossEntropy.cs       mean(-y*log(clip(p)) - (1-y)*log(1-clip(p)));
│   │                                dL/dp = (p - y) / (p*(1-p)) / N.
│   └── MeanSquaredError.cs    mean((preds - labels)²); ∇ = 2*(preds-labels)/batch.
│
├── Metrics/
│   ├── BaseMetric.cs          Abstract: Calculate(preds, labels) → NDArray.
│   ├── Accuracy.cs            class Accuacy (typo preserved). argmax(preds,1)
│   │                           == argmax(labels,1), mean.
│   ├── BinaryAccuacy.cs       round(clip(preds, 0, 1)) == labels, mean.
│   └── MeanAbsoluteError.cs   mean(|preds - labels|).
│
├── Optimizers/
│   ├── BaseOptimizer.cs       Abstract. Get("sgd") / Get("adam") resolvers.
│   ├── SGD.cs                 Vanilla SGD; classical momentum; inverse-time
│   │                           LR decay.
│   └── Adam.cs                First/second moments with proper np.zeros init.
│                               Step counter must be monotonic across run.
│
├── MnistMlp/                  The runnable experiment. Files described below.
│
├── Open.snk                   Strong-name key shared with NumSharp.Core.
└── NeuralNetwork.NumSharp.csproj   Exe, net8.0+net10.0, AllowUnsafeBlocks.
```

---

## MnistMlp — fused forward + backward

All fusion happens in `FullyConnectedFused`. The idea: every post-matmul
element-wise chunk (bias-add + ReLU, bias-add only, ReLU gradient mask)
collapses into **one NpyIter kernel**, compiled once per process and
cache-hit on every subsequent forward/backward pass.

| Stage | NpyExpr tree | Inputs → Output |
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
| `FullyConnectedFused.cs` | FC with bias + optional fused activation. Three NpyIter kernels (two forward, one backward), cache keys are stable strings. |
| `SoftmaxCrossEntropy.cs` | Combined loss — numerically stable softmax forward, cached softmax, (softmax-labels)/batch backward. Also ships `OneHot` helper. |
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

### 4. `np.allclose` mutates its arguments
`np.allclose` calls `astype(Double, copy:false)` on both operands, which
in-place flips their dtype from Single to Double. Use a manual max-abs-diff
loop if you need the operands untouched. (This is a NumSharp core library
bug — not fixed here.)

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

**100-epoch training on 6000 synthetic / 1000 test (batch=128, Adam, sigma=2.5):**
- Epoch 1: loss ≈ 1.12, train_acc ≈ 73% (random init → partial fit)
- Epoch 2: loss ≈ 0.009, train_acc ≈ 99.9%
- Epoch 100: loss ≈ 0, test_acc ≈ 99.89%
- Total training time: ~70 s (net8.0)

**Fusion probe on post-matmul bias+ReLU, batch (128, 128) fp32:**
- Fused (1 NpyIter): ~0.14 ms
- Naive (np.add + np.maximum): ~0.36 ms
- Speedup: ~2.5x

**Instrumentation (after a 100-epoch run):**
- IL kernel cache entries: delta of 6 (all unique fused expressions)
- NpyExpr delegate slots: 0 (pure DSL, no captured lambdas)

---

## Testing

No dedicated MSTest project. The **smoke test** for the NN scaffolding lives
in-line as a `dotnet run` stdin script — 29 checks covering:
- Softmax forward + backward (finite-difference gradient check)
- Sigmoid (saturation limits)
- CCE / BCE (loss values + backward components)
- Accuracy / BinaryAccuacy (argmax + round)
- FullyConnected with bias (shape checks)
- SGD vanilla + momentum (hand-computed trajectories)
- `BaseOptimizer.Get("sgd")` / `Get("adam")`

Run pattern for ad-hoc sanity checks:
```bash
cat /tmp/script.cs | dotnet_run
```
where the script references the two projects via `#:project`.

---

## Q&A

**Why do we have both `FullyConnected` and `FullyConnectedFused`?**
`FullyConnected` is the vanilla version that goes through `np.dot + (x + b) +
activation` as separate ops. `FullyConnectedFused` collapses bias+activation
into a single NpyIter — the fusion demo's point. Both share the BaseLayer
contract and are interchangeable in a NeuralNet pipeline.

**Why do the metric classes have typos in their names?**
`Accuacy`, `BinaryAccuacy` — misspelled in the original scaffolding, kept
for backward compat with any external caller. Fixing the implementation
without renaming the class is the lower-risk path.

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
  pressure. Fixable by fusing Adam's update into NpyIter like the rest,
  but out of scope for the current demo.
- **No model serialization.** Parameters can't be saved / loaded yet.
- **Activation resolution by string only.** `FullyConnected` takes `act =
  "relu"` etc. `FullyConnectedFused` uses an enum (`FusedActivation`) —
  the two are slightly inconsistent.
