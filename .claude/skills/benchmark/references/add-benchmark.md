# Adding a benchmark for a new op

Goal: the official op matrix measures `np.<foo>` every run, with the C# result JOINED to its NumPy twin so the
report shows an NPY/NS ratio. Two edits (C# class + NumPy twin), then a smoke check. No `run_benchmark.py` change
if you place the C# class in an existing suite namespace.

## 1. C# side — a `[Benchmark]` method

Pick the class in `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/<Category>/` that fits your op, or add a new
class in that **namespace** (`NumSharp.Benchmark.CSharp.Benchmarks.<Category>`) so the suite filter
`*Benchmarks.<Category>.*` auto-includes it — no orchestrator change.

Two base classes:
- **`BenchmarkBase`** — single dtype (float64 via `np.random.rand`). Use for dtype-agnostic ops (manipulation,
  slicing). The merge defaults such rows to `dtype=float64`.
- **`TypedBenchmarkBase`** — sweeps `[ParamsSource(nameof(Types))] NPTypeCode DType`. Use for arithmetic/unary/etc.

Worked example (`Benchmarks/Manipulation/FlipRotBenchmarks.cs`, the flip/rot90/transpose-alias/trim_zeros twin):

```csharp
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

[BenchmarkCategory("Manipulation", "FlipRot")]
public class FlipRotBenchmarks : BenchmarkBase
{
    private NDArray _arr2D = null!;

    [Params(ArraySizeSource.Medium, ArraySizeSource.Large)]   // 100K, 10M — match the sibling manip classes
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        var rows = (int)Math.Sqrt(N);
        _arr2D = np.random.rand(rows, N / rows) * 100;
    }

    [GlobalCleanup]
    public void Cleanup() { _arr2D = null!; GC.Collect(); }

    [Benchmark(Description = "np.flip(a)")]                    // Description is the JOIN KEY (normalized)
    [BenchmarkCategory("Flip")]
    public NDArray Flip() => np.flip(_arr2D);

    [Benchmark(Description = "np.trim_zeros(a)")]
    [BenchmarkCategory("TrimZeros")]
    public NDArray TrimZeros() => np.trim_zeros(_arr2D, "fb");
    // ...fliplr / flipud / rot90 / permute_dims / matrix_transpose the same way
}
```

## 2. NumPy twin — a row in `numpy_benchmark.py`

Append to the matching `run_<suite>_benchmarks(...)` (e.g. `run_manipulation_benchmarks`). Each benchmark: define a
closure, time it with `benchmark(fn, n, iterations=iterations)`, set the four fields. The suite function is already
wired into `main()`'s dispatch and `run_benchmark.py`, so appending is all it takes.

```python
def np_flip(): return np.flip(arr_2d)
r = benchmark(np_flip, n, iterations=iterations)
r.name, r.category, r.suite, r.dtype = "np.flip", "Flip", "Manipulation", dtype_name
results.append(r)

def np_trim_zeros(): return np.trim_zeros(arr_2d, 'fb')
r = benchmark(np_trim_zeros, n, iterations=iterations)
r.name, r.category, r.suite, r.dtype = "np.trim_zeros", "TrimZeros", "Manipulation", dtype_name
results.append(r)
```

## 3. The join — make the names normalize identically

The merge key is `(normalize_op_name(name), dtype, N)`. `normalize_op_name` (in `merge-results.py`) applied to BOTH
sides: strips a trailing dtype tag `(float64)`, strips `[annotations]`, folds `(a, axis=k) -> axis=k`, and strips
**identifier-only** arg lists (`(a)`, `(a, b)`) but KEEPS numeric args (`(a, 50)`). So:

- C# `"np.flip(a)"` → strip `(a)` → `np.flip`.  NumPy `"np.flip"` → `np.flip`.  **Join.** ✓
- C# `"np.percentile(a, 50)"` → `np.percentile(a, 50)` (numeric arg kept). NumPy must match that literally.

Rule of thumb: name the C# Description `"np.<op>(a)"` and the NumPy `.name` `"np.<op>"`, and they join. If a row
shows as "C# not run" or "NumPy only" in the report, the names didn't normalize to the same string — that's the
first thing to check.

Dtype/N join: `BenchmarkBase` (no `DType`) → merge defaults dtype to `float64`, so the NumPy twin must set
`dtype_name = 'float64'`. `[Params(Medium, Large)]` joins the NumPy 100K/10M rows; NumPy's 1K row shows C#-missing
(fine — the sibling manip classes do the same).

## 4. Smoke test (the usual scope)

A full measured run is expensive and normally the post-release `benchmark.yml` job. To confirm the wiring:

```bash
cd benchmark/NumSharp.Benchmark.CSharp
dotnet build -c Release -v q --nologo
dotnet run -c Release --no-build -f net10.0 -- --list flat | grep FlipRot     # BDN discovers the 7 methods
cd ../NumSharp.Benchmark.Python
python numpy_benchmark.py --suite manipulation --quick --size medium | grep -iE "flip|trim_zeros"   # NumPy rows emit
```

`--list flat` is reflection-only (no toolchain), so it works despite the out-of-process-toolchain limitation.
A direct-call `dotnet run -c Release` script (mirroring each benchmark body) confirms the C# calls execute.

## 5. Full run (optional)

`python run_benchmark.py --suites manipulation` measures just your suite; `python run_benchmark.py` runs everything
and writes a `benchmark/history/<date>_<sha>/` snapshot. See `references/run-and-report.md`.

## Adding a whole new SUITE (rare — most ops fit an existing one)

Only when your ops form a new category with no existing home:

1. **C#** — create `Benchmarks/<NewCategory>/<Foo>Benchmarks.cs` in namespace
   `NumSharp.Benchmark.CSharp.Benchmarks.<NewCategory>`.
2. **NumPy** — add `def run_<newsuite>_benchmarks(n, dtype_name, iterations): ...` in `numpy_benchmark.py` and
   dispatch it in `main()` (`if suite in ["<newsuite>", "all"]: results_all.extend(run_<newsuite>_benchmarks(...))`).
3. **Orchestrator** — add `"<newsuite>": "*Benchmarks.<NewCategory>.*"` to the `SUITES` map in `run_benchmark.py`.
4. (legacy) optionally add the filter to `run-benchmarks.ps1`'s `ValidateSet` and Program.cs menu.

Then the new suite runs in `python run_benchmark.py --suites <newsuite>` (and in the full run), joined and reported
exactly like the built-in 14. Keep the C# `[Benchmark(Description)]` labels normalizing onto the NumPy `.name`s.

## Choosing base class & size (quick reference)

- **Dtype-agnostic op** (shape/view/copy) → `BenchmarkBase`, NumPy twin `dtype_name='float64'` (merge default).
- **Numeric op you want swept over dtypes** → `TypedBenchmarkBase` (override `Types` for the relevant set), and give
  the NumPy twin a `dtype_name` loop in `main()` so both sides sweep the same dtypes.
- **Size**: `[Params(Medium, Large)]` (100K/10M) is the manipulation/reshape norm; add `Small` (1K) for ops where
  per-element overhead matters. The NumPy side sweeps all three via `--cache-sizes`; unmatched sizes just show as
  C#-missing cells.
