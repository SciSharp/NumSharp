# NumSharp Python Benchmarks

NumPy baseline benchmarks for comparing against NumSharp performance.

## Requirements

```bash
pip install numpy tabulate
```

## Usage

```bash
# Full benchmark suite
python numpy_benchmark.py

# Quick run (fewer iterations)
python numpy_benchmark.py --quick

# Specific suite
python numpy_benchmark.py --suite arithmetic
python numpy_benchmark.py --suite reduction

# Specific type
python numpy_benchmark.py --type float64

# JSON output
python numpy_benchmark.py --output results.json

# Combine options
python numpy_benchmark.py --quick --suite arithmetic --type int32 --output results.json
```

## Available Suites

| Suite | Operations |
|-------|------------|
| `dispatch` | Basic add operations with different patterns |
| `fusion` | Multi-operation expressions (a*a+2*b, variance, etc.) |
| `arithmetic` | +, -, *, /, % with scalars and arrays |
| `unary` | sqrt, abs, exp, log, sin, cos, etc. |
| `reduction` | sum, mean, var, std, min, max, argmin, argmax |
| `broadcast` | Scalar, row, column broadcasting |
| `creation` | zeros, ones, empty, full, copy |
| `manipulation` | reshape, transpose, ravel, flatten, stack |
| `slicing` | Contiguous, strided, reversed slices |

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "name": "a + b (int32)",
  "category": "Add",
  "suite": "Arithmetic",
  "dtype": "int32",
  "n": 10000000,
  "mean_ms": 10.5,
  "stddev_ms": 0.5,
  "min_ms": 9.8,
  "max_ms": 11.2,
  "iterations": 50,
  "ops_per_sec": 95.2
}
```

## Integration

This script is typically run via the parent `run-benchmarks.ps1` script which:
1. Runs Python benchmarks
2. Runs C# benchmarks
3. Merges results into a comparison report

See `../README.md` for the latest benchmark results.
