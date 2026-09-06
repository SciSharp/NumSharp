# NumSharp-NumPy Hybrid Test Framework Design

**Status**: FINAL DESIGN
**Goal**: 100% behavioral consistency between NumSharp and NumPy 2.x

## Overview

A C#-driven test framework that:
1. **Defines test cases** via `yield return np.function(...)` expressions (semantic, not execution)
2. **Expands markers** into combinatorial grid variations automatically
3. **Python executes first** — creates arrays, runs operations, saves artifacts to disk
4. **NumSharp executes second** — loads artifacts, runs same operations, compares results
5. **Reports alignment** with statistics and detailed mismatches

### Key Principles

- **Yield is semantic** — `yield return np.sum(...)` is a *description*, not execution
- **Python-first** — Python is the source of truth; NumSharp validates against it
- **Grid expansion** — `np.dot(arrays, arrays)` = N² combinations (full cross product)
- **Every step verified** — Chained operations store and compare each intermediate result
- **Executor manages state** — Re-seeding, sequencing, parallelization handled outside contracts

---

## Core Idea

One line generates thousands of test cases:

```csharp
yield return np.sum(arrays, axis: axes, keepdims: bools);
```

Expands to:
```python
np.sum(np.arange(6), axis=None, keepdims=False)
np.sum(np.arange(6), axis=None, keepdims=True)
np.sum(np.arange(6), axis=0, keepdims=False)
np.sum(np.zeros((3,4)), axis=0, keepdims=False)
np.sum(np.zeros((3,4)), axis=1, keepdims=True)
np.sum(np.zeros((3,4)), axis=2, keepdims=False)  # Invalid - should throw
... (hundreds more)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   NumSharp.Tests.Battletesting                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │   Contracts     │     │    Expander     │                   │
│  │   (semantic)    │────▶│                 │                   │
│  │                 │     │ Marker → Values │                   │
│  │ yield return    │     │ Grid expansion  │                   │
│  │ np.sum(arrays,  │     │ Context-aware   │                   │
│  │   axis: axes)   │     │                 │                   │
│  └─────────────────┘     └────────┬────────┘                   │
│                                   │                            │
│                                   ▼                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Expanded Calls                        │   │
│  │  np.sum(arange(6), axis=None, keepdims=False)           │   │
│  │  np.sum(arange(6), axis=0, keepdims=True)               │   │
│  │  np.dot(arange(6), zeros(3,4))  ← grid: arrays × arrays │   │
│  │  ...                                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PHASE 1: Python Executor (first)            │   │
│  │                                                          │   │
│  │  For each expanded call:                                 │   │
│  │    1. Execute array builders → save intermediates        │   │
│  │    2. Execute operation → save result                    │   │
│  │    3. Catch exceptions → save error info                 │   │
│  │                                                          │   │
│  │  Artifacts: inputs/*.npy, outputs/*.npy, results.json   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PHASE 2: NumSharp Executor (second)         │   │
│  │                                                          │   │
│  │  For each expanded call:                                 │   │
│  │    1. Execute array builders → compare with saved        │   │
│  │    2. Execute operation → compare with saved             │   │
│  │    3. Catch exceptions → compare error messages          │   │
│  │                                                          │   │
│  │  Every intermediate step is verified against Python      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Report Generator                        │   │
│  │  - Alignment % per function                             │   │
│  │  - Mismatch details (values, shapes, dtypes, errors)    │   │
│  │  - Intermediate step failures                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Execution Model

### Yield is Semantic

The `yield return` statement is a **description**, not execution:

```csharp
yield return np.sum(arange(24).reshape(2,3,4), axis: 1);
```

This describes: "Test `np.sum` on a reshaped arange array with axis=1."

No code runs during yield. The executor interprets this description later.

### Python-First Execution

Python is the source of truth. The executor:

1. **Python Phase**: Execute operation, save all artifacts to disk
2. **NumSharp Phase**: Load artifacts, execute same operation, compare

```
Contract                    Python Executor              NumSharp Executor
    │                             │                             │
    │ np.sum(arange(24)          │                             │
    │   .reshape(2,3,4),         │                             │
    │   axis=1)                  │                             │
    │────────────────────────────▶                             │
    │                             │                             │
    │                        Execute & Save:                   │
    │                        step1 = np.arange(24)             │
    │                        save("step1.npy")                 │
    │                        step2 = step1.reshape(2,3,4)      │
    │                        save("step2.npy")                 │
    │                        result = np.sum(step2, axis=1)    │
    │                        save("result.npy")                │
    │                             │                             │
    │                             │────────────────────────────▶│
    │                             │                             │
    │                             │                   Execute & Compare:
    │                             │                   step1 = np.arange(24)
    │                             │                   compare(step1, "step1.npy")
    │                             │                   step2 = step1.reshape(2,3,4)
    │                             │                   compare(step2, "step2.npy")
    │                             │                   result = np.sum(step2, axis: 1)
    │                             │                   compare(result, "result.npy")
```

### Chained Operations

Every intermediate step is stored and compared:

```csharp
yield return np.sum(arange(24).reshape(2,3,4), axis: 1);
```

Becomes 3 verification points:
1. `arange(24)` — NumSharp's arange must match NumPy's
2. `.reshape(2,3,4)` — NumSharp's reshape must match NumPy's
3. `np.sum(..., axis=1)` — Final result must match

If `reshape` differs, we catch it before `sum` runs — pinpointing the actual bug.

### Grid Expansion

When the same marker appears multiple times, it's a **full cross product**:

```csharp
yield return np.dot(arrays, arrays);
```

If `arrays` = `[arange(6), zeros(3,4), ones(2,3)]`, this generates 9 test cases:

| Left | Right | Expected |
|------|-------|----------|
| arange(6) | arange(6) | Inner product (works) |
| arange(6) | zeros(3,4) | Error: shapes not aligned |
| arange(6) | ones(2,3) | Error: shapes not aligned |
| zeros(3,4) | arange(6) | Error: shapes not aligned |
| zeros(3,4) | zeros(3,4) | Error: (3,4)·(3,4) not aligned |
| ... | ... | ... |

Most combinations are errors — and that's valid testing. NumSharp must throw the same errors as NumPy.

### Tuple Returns

Functions returning tuples use ITuple-compliant C# tuples:

```csharp
// NumPy returns: (unique_values, indices, inverse_indices)
// NumSharp returns: (NDArray, NDArray, NDArray)

yield return np.unique(arrays, return_index: true, return_inverse: true);
```

The framework compares tuple element-by-element. No special declaration needed.

---

## Contract Definition

### The Contract Base Class

```csharp
public abstract class Contract
{
    // Dynamic np proxy - captures calls instead of executing
    protected static dynamic np => new NpCapture();

    // ===== Array Builders =====
    protected static ArraySpec arange(int stop) => new("arange", stop) { Ndim = 1 };
    protected static ArraySpec arange(int start, int stop, int step = 1) => new("arange", start, stop, step) { Ndim = 1 };
    protected static ArraySpec zeros(params int[] shape) => new("zeros", shape) { Ndim = shape.Length };
    protected static ArraySpec ones(params int[] shape) => new("ones", shape) { Ndim = shape.Length };
    protected static ArraySpec eye(int n) => new("eye", n) { Ndim = 2 };
    protected static ArraySpec array(params object[] values) => new("array", values) { Ndim = 1 };
    protected static ArraySpec linspace(double start, double stop, int num) => new("linspace", start, stop, num) { Ndim = 1 };
    protected static ArraySpec empty(params int[] shape) => new("empty", shape) { Ndim = shape.Length };
    protected static ArraySpec full(int[] shape, object fill) => new("full", shape, fill) { Ndim = shape.Length };

    // ===== Standard Markers (expand to predefined variations) =====
    protected static Marker arrays => new("arrays");           // Standard array variations
    protected static Marker arrays_float => new("arrays_float"); // Float arrays only (for sqrt, log, etc.)
    protected static Marker arrays_sorted => new("sorted_arrays"); // Pre-sorted arrays
    protected static Marker matrices => new("matrices");       // 2D arrays only
    protected static Marker scalars => new("scalars");         // Numeric scalars
    protected static Marker axes => new("axes");               // Context-aware axis values
    protected static Marker bools => new("bools");             // true, false
    protected static Marker dtypes => new("dtypes");           // All NPTypeCodes
    protected static Marker dtypes_float => new("dtypes_float"); // Float dtypes
    protected static Marker dtypes_int => new("dtypes_int");   // Integer dtypes
    protected static Marker shapes => new("shapes");           // Output shapes
    protected static Marker sides => new("sides");             // "left", "right"
    protected static Marker orders => new("orders");           // "C", "F"

    // ===== Inline Variation (custom values) =====
    protected static Vary Vary(params object[] values) => new(values);

    // ===== Special Values =====
    protected static double nan => double.NaN;
    protected static double inf => double.PositiveInfinity;
    protected static double neginf => double.NegativeInfinity;

    // ===== The test case generator - override this =====
    public abstract IEnumerable<dynamic> TestCases();
}
```

### Writing Contracts

```csharp
public class NpMathContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        // ===== Unary operations =====
        yield return np.abs(arrays);
        yield return np.negative(arrays);
        yield return np.sqrt(arrays_float);      // Float only (negative ints would fail)
        yield return np.exp(arrays);
        yield return np.log(arrays_float);
        yield return np.sin(arrays);
        yield return np.cos(arrays);
        yield return np.floor(arrays);
        yield return np.ceil(arrays);
        yield return np.sign(arrays);

        // ===== Reductions =====
        yield return np.sum(arrays, axis: axes, keepdims: bools);
        yield return np.prod(arrays, axis: axes, keepdims: bools);
        yield return np.mean(arrays, axis: axes, keepdims: bools);
        yield return np.std(arrays, axis: axes, keepdims: bools);
        yield return np.var(arrays, axis: axes, keepdims: bools);
        yield return np.min(arrays, axis: axes, keepdims: bools);
        yield return np.max(arrays, axis: axes, keepdims: bools);
        yield return np.argmin(arrays, axis: axes);
        yield return np.argmax(arrays, axis: axes);
        yield return np.all(arrays, axis: axes);
        yield return np.any(arrays, axis: axes);

        // ===== Binary operations =====
        yield return np.add(arrays, arrays);
        yield return np.subtract(arrays, arrays);
        yield return np.multiply(arrays, arrays);
        yield return np.divide(arrays, arrays);
        yield return np.power(arrays, Vary(0, 1, 2, 3, -1));
        yield return np.dot(arrays, arrays);
        yield return np.matmul(matrices, matrices);

        // ===== Edge cases - explicit =====
        yield return np.sum(array(nan, 1.0, 2.0), axis: 0);
        yield return np.sum(array(inf, neginf, 0.0));
        yield return np.divide(array(1.0, 2.0), array(0.0, 0.0));  // Division by zero
    }
}

public class NpCreationContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        yield return np.zeros(shapes, dtype: dtypes);
        yield return np.ones(shapes, dtype: dtypes);
        yield return np.empty(shapes, dtype: dtypes);
        yield return np.full(shapes, scalars, dtype: dtypes);
        yield return np.arange(Vary(0, 1, 10), Vary(10, 100), Vary(1, 2, 5));
        yield return np.linspace(scalars, scalars, Vary(0, 1, 10, 50));
        yield return np.eye(Vary(1, 3, 5, 10), dtype: dtypes_float);
    }
}

public class NpManipulationContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        yield return np.reshape(arrays, Vary(new[]{-1}, new[]{2,-1}, new[]{3,4}));
        yield return np.transpose(matrices);
        yield return np.squeeze(arrays);
        yield return np.expand_dims(arrays, axis: axes);
        yield return np.concatenate(array_tuples, axis: Vary(0, 1, -1));
        yield return np.stack(array_tuples, axis: Vary(0, 1, -1));
        yield return np.broadcast_to(arrays, broadcast_shapes);
        yield return np.unique(arrays, return_index: bools, return_inverse: bools, return_counts: bools);
        yield return np.nonzero(arrays);
    }

    // Custom markers for this contract
    Marker array_tuples => new("array_tuples");
    Marker broadcast_shapes => new("broadcast_shapes");
}

[Sequential("np.random")]  // All cases run sequentially in this group
public class NpRandomContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        // Seed first for reproducibility
        yield return np.random.seed(42);

        yield return np.random.rand(shapes);
        yield return np.random.randn(shapes);
        yield return np.random.randint(Vary(0, 1, 10), Vary(10, 100, 256), size: shapes);
        yield return np.random.uniform(scalars, scalars, size: shapes);
        yield return np.random.normal(Vary(0.0, 1.0), Vary(1.0, 0.5), size: shapes);
        yield return np.random.choice(Vary(5, 10), size: shapes, replace: bools, p: probs);
        yield return np.random.permutation(Vary(5, 10));
        yield return np.random.shuffle(arange(10));
    }

    Marker probs => new("probs");  // Context-aware probability arrays
}
```

### Mixing Markers with Concrete Values

You have full control over test coverage:

```csharp
public class NpSumDetailedContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        // 1. Full combinatorial (broad coverage)
        yield return np.sum(arrays, axis: axes, keepdims: bools);

        // 2. Custom axis values with fixed keepdims (edge cases)
        yield return np.sum(arrays, axis: Vary(-1, -2, 100, -100), keepdims: true);

        // 3. Specific array with varied params
        yield return np.sum(arange(24).reshape(2,3,4), axis: Vary(0, 1, 2, null));

        // 4. Explicit edge cases
        yield return np.sum(array(nan, 1.0, 2.0), axis: 0);
        yield return np.sum(array(inf, neginf, 1.0), axis: null);
        yield return np.sum(array(), axis: null);  // Empty array

        // 5. Dtype-specific tests
        yield return np.sum(arrays, dtype: Vary(NPTypeCode.Float32, NPTypeCode.Float64, NPTypeCode.Int64));
    }
}
```

| Pattern | Example | Effect |
|---------|---------|--------|
| Standard marker | `arrays` | Expands to all predefined array variations |
| Inline Vary() | `Vary(-1, -2, 100)` | Expands to exactly those values |
| Concrete value | `keepdims: true` | Fixed, no expansion |
| Explicit array | `array(nan, 1.0, 2.0)` | One specific test case |
| Named custom marker | `edge_axes` | Reusable custom variations |

---

## Markers and Expansion

### Standard Markers

| Marker | Expands To | Context-Aware |
|--------|------------|---------------|
| `arrays` | arange(6), zeros(3,4), ones(2,3,4), eye(3), ... | No |
| `arrays_float` | Arrays with float dtypes only | No |
| `arrays_sorted` | Pre-sorted arrays | No |
| `matrices` | 2D arrays only | No |
| `scalars` | 0, 1, -1, 0.5, 1e10, nan, inf, ... | No |
| `bools` | false, true | No |
| `dtypes` | All NPTypeCodes | No |
| `dtypes_float` | Float32, Float64 | No |
| `dtypes_int` | Int8...Int64, UInt8...UInt64 | No |
| `shapes` | (), (1,), (5,), (3,4), (2,3,4), ... | No |
| `axes` | null, 0, 1, ..., ndim-1, ndim (invalid) | **Yes** - adapts to input array |
| `probs` | null, uniform, skewed, invalid | **Yes** - adapts to size param |
| `broadcast_shapes` | Valid broadcast targets | **Yes** - adapts to input shape |

### Context-Aware Expansion

The `axes` marker generates different values based on the input array's ndim:

```csharp
yield return np.sum(arrays, axis: axes);

// For arange(6) [1D]:     axis = null, 0, 1(invalid)
// For zeros(3,4) [2D]:    axis = null, 0, 1, 2(invalid)
// For ones(2,3,4) [3D]:   axis = null, 0, 1, 2, 3(invalid)
```

The `probs` marker generates probability arrays that match the `a` parameter:

```csharp
yield return np.random.choice(Vary(5, 10), p: probs);

// For a=5:  p = null, [0.2,0.2,0.2,0.2,0.2], [0.9,0.025,...]
// For a=10: p = null, [0.1,0.1,...], [0.9,0.01,...]
```

### Inline Vary()

For custom values that don't need a named marker:

```csharp
// Test specific axis values
yield return np.sum(arrays, axis: Vary(-1, -2, 100));

// Test specific dtype conversions
yield return np.sum(arrays, dtype: Vary(null, NPTypeCode.Float32, NPTypeCode.Int64));

// Test specific shapes (array literals)
yield return np.zeros(Vary(new[]{0}, new[]{1}, new[]{3,4}, new[]{2,3,4}));

// Array values in Vary
yield return np.concatenate(
    Vary(arange(6), zeros(3,4)),           // First array options
    Vary(new[]{1,2,3}, new[]{3,2,1}),      // Second array options (literals)
    axis: Vary(0, 1)
);
```

### Expansion Combinations

Multiple `Vary()` or markers expand as a **grid** (cross product):

```csharp
yield return np.foo(Vary(a1, a2), Vary(b1, b2), axis: Vary(0, 1));
// Generates: 2 × 2 × 2 = 8 test cases
// (a1, b1, 0), (a1, b1, 1), (a1, b2, 0), (a1, b2, 1),
// (a2, b1, 0), (a2, b1, 1), (a2, b2, 0), (a2, b2, 1)
```

### Testing Error Cases

For functions with many error cases, use explicit yields or loops:

```csharp
public override IEnumerable<dynamic> TestCases()
{
    // Happy paths with markers
    yield return np.sum(arrays, axis: axes);

    // Explicit error cases
    yield return np.sum(arange(6), axis: 5);   // Out of bounds
    yield return np.sum(arange(6), axis: -10); // Out of bounds negative

    // Loop for systematic edge case coverage
    foreach (var invalidAxis in new[] { 5, -5, 100, -100 })
    {
        yield return np.sum(zeros(3, 4), axis: invalidAxis);
    }

    // Or use Vary for invalid values
    yield return np.sum(arrays, axis: Vary(10, -10, 50, -50));
}
```

The framework doesn't distinguish "expected error" from "expected success" — it just compares behavior. If both NumPy and NumSharp throw, that's a passing test.

---

## The Call Object

When you write `np.sum(arrays, axis: axes)`, it returns a `Call` object:

```csharp
public class Call
{
    public string Path { get; }           // "np.sum"
    public object[] Args { get; }         // [ArraySpec, Marker, ...]
    public string[] ArgNames { get; }     // ["axis", "keepdims"]

    // Generate Python code
    public string ToPython();
    // "np.sum(np.arange(6), axis=0, keepdims=False)"

    // Generate NumSharp invocation
    public object ExecuteNumSharp();
    // Calls np.sum(ndarray, axis: 0, keepdims: false)

    // Auto-generated case ID
    public string CaseId { get; }
    // "np.sum.case042:arange6_axis0_keepdimsF"
}
```

---

## Python Runner

### Single Process, File-Based Communication

```
C# Test Runner                         Python Runner
      │                                      │
      │  1. Write manifest.json              │
      │─────────────────────────────────────▶│
      │                                      │
      │  2. Write input arrays (.npy)        │
      │─────────────────────────────────────▶│
      │                                      │
      │  3. Launch: python runner.py         │
      │─────────────────────────────────────▶│
      │                                      │  Execute all cases
      │                                      │  Save outputs (.npy)
      │  4. Wait for exit code 0             │
      │◀─────────────────────────────────────│
      │                                      │
      │  5. Read results.json                │
      │◀─────────────────────────────────────│
      │                                      │
      │  6. Load output arrays (.npy)        │
      │◀─────────────────────────────────────│
```

### Work Directory Structure

```
./testresults/<run-id>/
├── manifest.json           # All test cases to execute
├── inputs/                 # Input arrays
│   ├── np.sum.case001_a.npy
│   └── ...
├── outputs/                # NumPy outputs
│   ├── np.sum.case001.npy
│   └── ...
├── results.json            # Execution results + metadata
└── report.md               # Final alignment report
```

### Manifest Format

```json
{
  "version": "1.0",
  "cases": [
    {
      "case_id": "np.sum.case001:arange6_axis0_keepdimsF",
      "python_code": "np.sum(np.arange(6), axis=0, keepdims=False)",
      "inputs": {
        "a": "inputs/np.sum.case001_a.npy"
      },
      "output_path": "outputs/np.sum.case001.npy",
      "sequential_group": null
    },
    {
      "case_id": "np.random.rand.case001:shape3x4",
      "python_code": "np.random.rand(3, 4)",
      "inputs": {},
      "output_path": "outputs/np.random.rand.case001.npy",
      "sequential_group": "np.random"
    }
  ]
}
```

### Results Format

```json
{
  "version": "1.0",
  "numpy_version": "2.4.2",
  "results": {
    "np.sum.case001:arange6_axis0_keepdimsF": {
      "success": true,
      "return_type": "scalar",
      "dtype": "int64",
      "shape": [],
      "output_path": "outputs/np.sum.case001.npy"
    },
    "np.sum.case099:arange6_axis5_keepdimsF": {
      "success": false,
      "error_type": "AxisError",
      "error_message": "axis 5 is out of bounds for array of dimension 1"
    }
  }
}
```

---

## Result Comparison

### Comparison Rules

1. **Both succeeded** → Compare return type, dtype, shape, values
2. **Both failed** → Compare exception behavior (see below)
3. **One succeeded, one failed** → **MISMATCH**

### Exception Comparison

When both sides throw, the framework checks:

1. **Both threw** — This is already valuable (behavior matches)
2. **Exception type similarity** — Optional, logged for review
3. **Message comparison** — Configurable per test

The default is lenient: if both threw, it's a **PASS** (same behavior).

For stricter matching, use explicit assertions:

```csharp
// Explicit error message validation
yield return np.sum(arange(6), axis: 5)
    .ExpectError(msg => msg.Contains("out of bounds"));

// Or just let both sides throw and compare (default behavior)
yield return np.sum(arange(6), axis: 5);
```

**Why lenient by default?**

NumPy: `"axis 5 is out of bounds for array of dimension 1"`
NumSharp: `"Axis must be in range [-1, 1) for 1-dimensional array"`

These are semantically equivalent but textually different. Both correctly reject axis=5. Requiring exact message containment would flag false negatives.

The report still shows both messages for manual review.

### Type Matching

Return types must match exactly:

| NumPy Returns | NumSharp Must Return |
|---------------|---------------------|
| Python scalar (float, int) | C# scalar (double, int) |
| 0-d ndarray (shape=()) | NDArray with shape=() |
| tuple of arrays | ValueTuple or array of NDArray |

**A scalar is NOT the same as a 0-d array.**

### Value Comparison

```csharp
public class ResultComparator
{
    public ComparisonResult Compare(NumpyResult numpy, NumSharpResult numsharp, ToleranceConfig tol)
    {
        // Both succeeded
        if (numpy.Success && numsharp.Success)
        {
            if (numpy.ReturnType != numsharp.ReturnType)
                return Mismatch("Return type differs");

            if (!numpy.Shape.SequenceEqual(numsharp.Shape))
                return Mismatch("Shape differs");

            if (numpy.Dtype != numsharp.Dtype)
                return Mismatch("Dtype differs");

            // Value comparison with tolerance
            if (!np.allclose(numpy.Data, numsharp.Data, tol.Rtol, tol.Atol, equal_nan: true))
                return Mismatch("Values differ");

            return Match();
        }

        // Both failed - check exception message
        if (!numpy.Success && !numsharp.Success)
        {
            if (!numsharp.ErrorMessage.Contains(numpy.ErrorMessage))
                return Mismatch($"Exception message mismatch: NumPy='{numpy.ErrorMessage}', NumSharp='{numsharp.ErrorMessage}'");

            return Match();
        }

        // One succeeded, one failed
        return Mismatch($"Behavior mismatch: NumPy {(numpy.Success ? "succeeded" : "failed")}, NumSharp {(numsharp.Success ? "succeeded" : "failed")}");
    }
}
```

### Tolerance Configuration

```csharp
[Tolerance(Rtol = 1e-7, Atol = 1e-8)]           // Default for most
[Tolerance(Rtol = 1e-5, Atol = 1e-6)]           // For linalg operations
[Tolerance(ExactMatch = true)]                   // For RNG (seeded)
[Tolerance(EqualNaN = true, EqualInf = true)]   // Default: NaN==NaN, Inf==Inf
```

---

## Report Format

### Console Output

```
================================================================================
                    NumSharp Alignment Report
                    NumPy 2.4.2 | NumSharp 0.41.0
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total Functions:     127
Total Test Cases:    45,320
Passed:              44,120 (97.4%)
Failed:              1,200 (2.6%)

PER-FUNCTION RESULTS
--------------------------------------------------------------------------------
Function                    Cases    Pass    Fail    Alignment
--------------------------------------------------------------------------------
np.abs                        156     156       0      100.0%
np.sum                       1560    1554       6       99.6%
np.mean                      1560    1560       0      100.0%
np.dot                        324     320       4       98.8%
np.random.choice              240     228      12       95.0%
...

FAILURES (first 10)
--------------------------------------------------------------------------------
[FAIL] np.sum.case042:zeros3x4_axis5_keepdimsF
  NumPy:    AxisError: axis 5 is out of bounds for array of dimension 2
  NumSharp: ArgumentOutOfRangeException: Axis must be within array dimensions
  Issue:    Exception message mismatch

[FAIL] np.dot.case089:arange6_zeros3x4
  NumPy:    ValueError: shapes (6,) and (3,4) not aligned
  NumSharp: IncorrectShapeException: Shapes are not aligned for dot product
  Issue:    Exception message mismatch
```

---

## Sequential Execution

For functions with shared state (like `np.random.*`), use `[Sequential]`:

```csharp
[Sequential("np.random")]  // Group name
public class NpRandomContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        yield return np.random.seed(42);
        yield return np.random.rand(Vary((3,4), (5,), (2,3,4)));
        yield return np.random.randn(Vary((2,3), (4,)));
    }
}
```

### Executor Handles Re-seeding

The contract is declarative — it doesn't manage state. The **executor** handles re-seeding:

```
For np.random.rand with shapes [(3,4), (5,), (2,3,4)]:

  seed(42) → rand(3,4) → compare with Python
  seed(42) → rand(5,)  → compare with Python    ← re-seeded!
  seed(42) → rand(2,3,4) → compare with Python  ← re-seeded!
```

Each expanded test case starts fresh from the seed. The executor:
1. Detects `seed()` calls in sequential groups
2. Re-runs the seed before each subsequent expanded case
3. Ensures deterministic, independent test cases

### Execution Order

- Cases **within same group** run sequentially (in yield order)
- **Different groups** run in parallel with each other
- **Non-sequential contracts** run fully in parallel

### Why This Matters

Without re-seeding:
```
seed(42) → rand(3,4) → rand(5,) → rand(2,3,4)
                ↑           ↑          ↑
           state A     state B    state C
```

The `rand(5,)` result depends on `rand(3,4)` running first — fragile.

With re-seeding:
```
seed(42) → rand(3,4)    # Independent
seed(42) → rand(5,)     # Independent
seed(42) → rand(2,3,4)  # Independent
```

Each test is self-contained and reproducible.

---

## Project Structure

```
test/NumSharp.Tests.Battletesting/
├── NumSharp.Tests.Battletesting.csproj
│
├── Core/
│   ├── Contract.cs              # Base class with np, markers, Vary()
│   ├── Call.cs                  # Captured call object
│   ├── NpCapture.cs             # DynamicObject for np proxy
│   ├── ArraySpec.cs             # Array builder specs
│   ├── Marker.cs                # Variation markers
│   ├── Vary.cs                  # Inline variations
│   └── Expander.cs              # Marker → concrete values
│
├── Contracts/
│   ├── NpMathContracts.cs       # abs, sum, mean, dot, ...
│   ├── NpCreationContracts.cs   # zeros, ones, arange, ...
│   ├── NpManipulationContracts.cs # reshape, transpose, ...
│   ├── NpRandomContracts.cs     # rand, randn, choice, ...
│   └── NpLinalgContracts.cs     # dot, matmul, svd, ...
│
├── Runners/
│   ├── PythonRunner.cs          # Launches Python, reads results
│   ├── NumSharpRunner.cs        # Executes C# calls
│   └── runner.py                # Embedded Python script
│
├── Comparison/
│   ├── ResultComparator.cs      # Compare NumPy vs NumSharp
│   └── ToleranceConfig.cs       # Rtol, Atol, ExactMatch
│
├── Reports/
│   ├── ConsoleReporter.cs
│   ├── MarkdownReporter.cs
│   └── JsonReporter.cs
│
└── Program.cs                   # CLI entry point
```

---

## CLI Usage

```bash
# Run all tests
dotnet run --project test/NumSharp.Tests.Battletesting

# Run specific contract
dotnet run -- --contract NpMathContracts

# Run specific function pattern
dotnet run -- --filter "np.sum*"

# Limit cases per function
dotnet run -- --max-cases 100

# Output formats
dotnet run -- --output console
dotnet run -- --output markdown --output-file report.md

# Show only failures
dotnet run -- --failures-only
```

---

## Summary

### The Framework is Minimal

| Step | What Happens |
|------|--------------|
| 1. **Write contracts** | `yield return np.function(markers)` — semantic, not execution |
| 2. **Markers expand** | Grid expansion (cross product), context-aware |
| 3. **Python runs first** | Creates arrays, executes operations, saves artifacts |
| 4. **NumSharp runs second** | Loads artifacts, executes same operations, compares |
| 5. **Every step verified** | Chained operations compare each intermediate |
| 6. **Report alignment** | Per-function statistics, mismatch details |

### Key Design Principles

| Principle | Meaning |
|-----------|---------|
| **Semantic yield** | `yield return` describes what to test, doesn't execute |
| **Python-first** | Python is source of truth; NumSharp validates against it |
| **Grid expansion** | `np.dot(arrays, arrays)` = N² combinations |
| **Chained verification** | `arange(24).reshape(2,3,4)` verifies both steps |
| **Executor manages state** | Re-seeding, sequencing, parallelization outside contracts |
| **Errors are valid tests** | Both throwing = PASS (same behavior) |

### Coverage

~100 lines of contract definitions → thousands of test cases covering the entire NumPy API.
