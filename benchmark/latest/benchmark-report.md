# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell; the winner names the implementation that actually executed, not merely the process profile.

**Timing basis:** best window (min) of each side's per-case sweep — min-based harnesses run each case to a ~200 ms time budget (a >20 ms/call op runs exactly 100 times), while the C# op-matrix keeps BenchmarkDotNet's 50 iterations; interference only adds time, so the fastest window estimates intrinsic cost; per-case means are retained in the JSON (`numpy_mean_ms`/`numsharp_mean_ms`) as tail diagnostics.

- Managed effective cells: **3,647**
- OpenBLAS effective cells: **98**
- Missing backend records: **48**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.06600 ms | openblas | 0.679x |
| `np.polyfit(x, y, 3)` | float64 | 100,000 | missing_backend | 1.1174 ms | openblas | 0.803x |
| `np.polyfit(x, y, 3)` | float64 | 10,000,000 | missing_backend | 7.0263 ms | openblas | 1.220x |
| `np.roots(p)` | float64 | 1,000 | missing_backend | 0.07360 ms | openblas | 1.193x |
| `np.roots(p)` | float64 | 100,000 | missing_backend | 0.1891 ms | openblas | 1.076x |
| `np.roots(p)` | float64 | 10,000,000 | missing_backend | 0.7742 ms | openblas | 0.995x |
| `np.array2string(a) (float64)` | float64 | 1 | 0.000502 ms | not_measured | managed | 33.602x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 0.9499 ms | not_measured | managed | 4.934x |
| `np.array_repr(a) (float64)` | float64 | 1 | 0.000546 ms | not_measured | managed | 34.057x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 0.9065 ms | not_measured | managed | 5.263x |
| `np.array_str(a) (float64)` | float64 | 1 | 0.000501 ms | not_measured | managed | 34.603x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 0.9412 ms | not_measured | managed | 2.108x |
| `np.can_cast(from, to) (float64)` | float64 | 1 | 0.000013 ms | not_measured | managed | 16.231x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.000025 ms | not_measured | managed | 12.880x |
| `np.common_type(a, b) (float64)` | float64 | 1 | 0.000015 ms | not_measured | managed | 54.267x |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.000025 ms | not_measured | managed | 31.960x |
| `np.copyto(dst, src) (float64)` | float64 | 1 | 0.000169 ms | not_measured | managed | 1.698x |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.000743 ms | not_measured | managed | 1.750x |
| `np.format_float_positional(x) (float64)` | float64 | 1 | 0.000401 ms | not_measured | managed | 1.212x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.000419 ms | not_measured | managed | 1.790x |
| `np.format_float_scientific(x) (float64)` | float64 | 1 | 0.000354 ms | not_measured | managed | 2.102x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.000384 ms | not_measured | managed | 1.935x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.03985 ms | not_measured | managed | 1.918x |
| `np.get_printoptions() (float64)` | float64 | 1 | 0.000224 ms | not_measured | managed | 2.076x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.000224 ms | not_measured | managed | 2.076x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1 | 0.000028 ms | not_measured | managed | 24.000x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000028 ms | not_measured | managed | 33.714x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1 | 0.000022 ms | not_measured | managed | 7.818x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000022 ms | not_measured | managed | 12.909x |
| `np.iterable(a) (float64)` | float64 | 1 | 0.000000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.04740 ms | not_measured | managed | 3.122x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.5357 ms | not_measured | managed | 0.931x |
| `np.min_scalar_type(value) (float64)` | float64 | 1 | 0.000002 ms | not_measured | managed | 132.000x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.000002 ms | not_measured | managed | 381.500x |
| `np.mintypecode(typechars) (float64)` | float64 | 1 | 0.000155 ms | not_measured | managed | 4.568x |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.000148 ms | not_measured | managed | 9.966x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1 | 0.00158 ms | not_measured | managed | 0.299x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.00163 ms | not_measured | managed | 0.507x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.1296 ms | not_measured | managed | 0.500x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.00190 ms | not_measured | managed | 2.519x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.00485 ms | not_measured | managed | 1.482x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.1556 ms | not_measured | managed | 2.083x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.00772 ms | not_measured | managed | 1.118x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.01005 ms | not_measured | managed | 1.466x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.00203 ms | not_measured | managed | 2.366x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1019 ms | not_measured | managed | 0.559x |
| `np.printoptions() (float64)` | float64 | 1 | 0.000104 ms | not_measured | managed | 61.048x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.000114 ms | not_measured | managed | 57.035x |
| `np.promote_types(a, b) (float64)` | float64 | 1 | 0.000012 ms | not_measured | managed | 10.333x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.000012 ms | not_measured | managed | 16.667x |
| `np.result_type(a, b) (float64)` | float64 | 1 | 0.000005 ms | not_measured | managed | 29.200x |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.000002 ms | not_measured | managed | 105.000x |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.2077 ms | not_measured | managed | 0.787x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 0.9594 ms | not_measured | managed | 3.692x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 1.9077 ms | not_measured | managed | 0.202x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.4502 ms | not_measured | managed | 0.706x |
| `np.set_printoptions() (float64)` | float64 | 1 | 0.000056 ms | not_measured | managed | 59.125x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.000042 ms | not_measured | managed | 80.048x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.01765 ms | not_measured | managed | 0.787x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.8260 ms | not_measured | managed | 0.886x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 177.0321 ms | not_measured | managed | 0.953x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.01045 ms | not_measured | managed | 1.088x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.6120 ms | not_measured | managed | 0.863x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 140.7377 ms | not_measured | managed | 1.069x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.00235 ms | not_measured | managed | 0.754x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.6402 ms | not_measured | managed | 0.605x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 50.8428 ms | not_measured | managed | 0.872x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.00415 ms | not_measured | managed | 0.849x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.6463 ms | not_measured | managed | 0.603x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 69.8825 ms | not_measured | managed | 0.701x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.01114 ms | not_measured | managed | 1.024x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 0.9699 ms | not_measured | managed | 1.515x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 152.4525 ms | not_measured | managed | 1.026x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.00882 ms | not_measured | managed | 1.134x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 1.3926 ms | not_measured | managed | 0.896x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 146.7164 ms | not_measured | managed | 0.932x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.00150 ms | not_measured | managed | 1.002x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.5231 ms | not_measured | managed | 0.685x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 59.8805 ms | not_measured | managed | 0.702x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.00260 ms | not_measured | managed | 1.186x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.5885 ms | not_measured | managed | 0.612x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 65.1822 ms | not_measured | managed | 0.722x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.00411 ms | not_measured | managed | 0.217x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.08073 ms | not_measured | managed | 3.371x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 33.4829 ms | not_measured | managed | 0.920x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.00924 ms | not_measured | managed | 0.369x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.8924 ms | not_measured | managed | 0.307x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 47.7110 ms | not_measured | managed | 0.629x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.00231 ms | not_measured | managed | 0.289x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.02448 ms | not_measured | managed | 0.251x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 7.8071 ms | not_measured | managed | 0.950x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.00279 ms | not_measured | managed | 0.270x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.04459 ms | not_measured | managed | 0.251x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 16.0587 ms | not_measured | managed | 1.030x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.00180 ms | not_measured | managed | 0.468x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.01318 ms | not_measured | managed | 1.632x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 4.0001 ms | not_measured | managed | 1.031x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.00166 ms | not_measured | managed | 0.511x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.01458 ms | not_measured | managed | 1.479x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 8.2981 ms | not_measured | managed | 0.883x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.00309 ms | not_measured | managed | 0.266x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.05062 ms | not_measured | managed | 0.426x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 20.2444 ms | not_measured | managed | 0.732x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.00154 ms | not_measured | managed | 0.514x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.00739 ms | not_measured | managed | 2.907x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 1.4678 ms | not_measured | managed | 2.012x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.00193 ms | not_measured | managed | 0.437x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.01274 ms | not_measured | managed | 1.690x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 4.0370 ms | not_measured | managed | 1.047x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.00252 ms | not_measured | managed | 0.336x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.02378 ms | not_measured | managed | 0.909x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 5.1610 ms | not_measured | managed | 1.408x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.00304 ms | not_measured | managed | 0.276x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.05315 ms | not_measured | managed | 0.406x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 12.7058 ms | not_measured | managed | 1.126x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.00171 ms | not_measured | managed | 0.464x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.01166 ms | not_measured | managed | 1.845x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 1.4531 ms | not_measured | managed | 2.120x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.00207 ms | not_measured | managed | 0.316x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.1429 ms | not_measured | managed | 1.869x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 25.3390 ms | not_measured | managed | 1.819x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.00932 ms | not_measured | managed | 0.334x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.6984 ms | not_measured | managed | 0.394x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 46.9751 ms | not_measured | managed | 0.639x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.000965 ms | not_measured | managed | 0.409x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.01976 ms | not_measured | managed | 0.298x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 9.0679 ms | not_measured | managed | 0.825x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.00115 ms | not_measured | managed | 0.400x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.03970 ms | not_measured | managed | 0.278x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 19.7716 ms | not_measured | managed | 0.793x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.000799 ms | not_measured | managed | 0.791x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.01019 ms | not_measured | managed | 2.669x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 4.1413 ms | not_measured | managed | 1.136x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.000649 ms | not_measured | managed | 0.978x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.02093 ms | not_measured | managed | 1.319x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 9.0931 ms | not_measured | managed | 0.851x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.00128 ms | not_measured | managed | 0.496x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.05132 ms | not_measured | managed | 0.535x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 20.5913 ms | not_measured | managed | 0.695x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.000672 ms | not_measured | managed | 0.884x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.01199 ms | not_measured | managed | 2.266x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 0.9485 ms | not_measured | managed | 3.804x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.000574 ms | not_measured | managed | 1.098x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.00791 ms | not_measured | managed | 3.443x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 4.1290 ms | not_measured | managed | 1.102x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.000995 ms | not_measured | managed | 0.637x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.02015 ms | not_measured | managed | 1.364x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 8.0365 ms | not_measured | managed | 0.926x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.00128 ms | not_measured | managed | 0.503x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.05053 ms | not_measured | managed | 0.546x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 19.9650 ms | not_measured | managed | 0.727x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.000588 ms | not_measured | managed | 1.007x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.01207 ms | not_measured | managed | 2.249x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 0.8752 ms | not_measured | managed | 4.090x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.00132 ms | not_measured | managed | 0.540x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.1625 ms | not_measured | managed | 1.695x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 54.6724 ms | not_measured | managed | 0.596x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.00934 ms | not_measured | managed | 0.334x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.8987 ms | not_measured | managed | 0.305x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 47.4989 ms | not_measured | managed | 0.633x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.000975 ms | not_measured | managed | 0.403x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.02182 ms | not_measured | managed | 0.307x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 3.8564 ms | not_measured | managed | 2.089x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.00130 ms | not_measured | managed | 0.369x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.05036 ms | not_measured | managed | 0.449x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 24.7745 ms | not_measured | managed | 0.675x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.000815 ms | not_measured | managed | 0.777x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.01191 ms | not_measured | managed | 2.285x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 5.8352 ms | not_measured | managed | 0.823x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.00100 ms | not_measured | managed | 0.639x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.02261 ms | not_measured | managed | 1.224x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 10.6306 ms | not_measured | managed | 0.792x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.00134 ms | not_measured | managed | 0.480x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.05250 ms | not_measured | managed | 0.571x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 28.0430 ms | not_measured | managed | 0.578x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.000633 ms | not_measured | managed | 0.945x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.01203 ms | not_measured | managed | 2.262x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 2.1103 ms | not_measured | managed | 1.713x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.000578 ms | not_measured | managed | 1.102x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.01476 ms | not_measured | managed | 1.844x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 2.7882 ms | not_measured | managed | 1.742x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.000836 ms | not_measured | managed | 0.766x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.02486 ms | not_measured | managed | 1.106x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.5750 ms | not_measured | managed | 0.784x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.00105 ms | not_measured | managed | 0.613x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.03653 ms | not_measured | managed | 0.850x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 26.6576 ms | not_measured | managed | 0.600x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.000669 ms | not_measured | managed | 0.891x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.01201 ms | not_measured | managed | 2.264x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 2.2242 ms | not_measured | managed | 1.627x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.00115 ms | not_measured | managed | 0.641x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.09357 ms | not_measured | managed | 2.897x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 40.4289 ms | not_measured | managed | 0.935x |
| `a * scalar (float16)` | float16 | 1,000 | 0.00787 ms | not_measured | managed | 0.415x |
| `a * scalar (float16)` | float16 | 100,000 | 0.6890 ms | not_measured | managed | 0.398x |
| `a * scalar (float16)` | float16 | 10,000,000 | 86.5691 ms | not_measured | managed | 0.349x |
| `a * scalar (float32)` | float32 | 1,000 | 0.000998 ms | not_measured | managed | 0.524x |
| `a * scalar (float32)` | float32 | 100,000 | 0.01912 ms | not_measured | managed | 0.303x |
| `a * scalar (float32)` | float32 | 10,000,000 | 8.5938 ms | not_measured | managed | 0.865x |
| `a * scalar (float64)` | float64 | 1,000 | 0.000745 ms | not_measured | managed | 0.819x |
| `a * scalar (float64)` | float64 | 100,000 | 0.03913 ms | not_measured | managed | 0.291x |
| `a * scalar (float64)` | float64 | 10,000,000 | 13.0329 ms | not_measured | managed | 1.192x |
| `a * scalar (int16)` | int16 | 1,000 | 0.000841 ms | not_measured | managed | 0.853x |
| `a * scalar (int16)` | int16 | 100,000 | 0.01065 ms | not_measured | managed | 2.007x |
| `a * scalar (int16)` | int16 | 10,000,000 | 4.1542 ms | not_measured | managed | 0.998x |
| `a * scalar (int32)` | int32 | 1,000 | 0.000974 ms | not_measured | managed | 0.747x |
| `a * scalar (int32)` | int32 | 100,000 | 0.02078 ms | not_measured | managed | 1.032x |
| `a * scalar (int32)` | int32 | 10,000,000 | 8.5824 ms | not_measured | managed | 0.863x |
| `a * scalar (int64)` | int64 | 1,000 | 0.000957 ms | not_measured | managed | 0.763x |
| `a * scalar (int64)` | int64 | 100,000 | 0.05028 ms | not_measured | managed | 0.427x |
| `a * scalar (int64)` | int64 | 10,000,000 | 20.1884 ms | not_measured | managed | 0.734x |
| `a * scalar (int8)` | int8 | 1,000 | 0.000394 ms | not_measured | managed | 1.766x |
| `a * scalar (int8)` | int8 | 100,000 | 0.00614 ms | not_measured | managed | 3.474x |
| `a * scalar (int8)` | int8 | 10,000,000 | 1.4580 ms | not_measured | managed | 2.037x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.000795 ms | not_measured | managed | 0.933x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.00710 ms | not_measured | managed | 3.013x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 2.6109 ms | not_measured | managed | 1.575x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.000998 ms | not_measured | managed | 0.728x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.01897 ms | not_measured | managed | 1.131x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 7.5419 ms | not_measured | managed | 0.972x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.00113 ms | not_measured | managed | 0.653x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.04655 ms | not_measured | managed | 0.460x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 15.7007 ms | not_measured | managed | 0.935x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.000586 ms | not_measured | managed | 1.172x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.00978 ms | not_measured | managed | 2.185x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 0.8741 ms | not_measured | managed | 3.559x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.00394 ms | not_measured | managed | 0.223x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.07669 ms | not_measured | managed | 3.566x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 33.7302 ms | not_measured | managed | 1.021x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.00721 ms | not_measured | managed | 0.472x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.8935 ms | not_measured | managed | 0.307x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 89.0967 ms | not_measured | managed | 0.336x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.00203 ms | not_measured | managed | 0.327x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.02287 ms | not_measured | managed | 0.275x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 8.7239 ms | not_measured | managed | 0.863x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.00299 ms | not_measured | managed | 0.251x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.05331 ms | not_measured | managed | 0.211x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 18.0864 ms | not_measured | managed | 0.856x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.00149 ms | not_measured | managed | 0.580x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.01363 ms | not_measured | managed | 1.746x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 4.0588 ms | not_measured | managed | 1.068x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.00154 ms | not_measured | managed | 0.561x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.02196 ms | not_measured | managed | 1.079x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 8.5751 ms | not_measured | managed | 0.884x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.00308 ms | not_measured | managed | 0.272x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.02935 ms | not_measured | managed | 0.794x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 18.9702 ms | not_measured | managed | 0.759x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.00154 ms | not_measured | managed | 0.533x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.00807 ms | not_measured | managed | 2.928x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 1.2911 ms | not_measured | managed | 2.545x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.00115 ms | not_measured | managed | 0.760x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.01296 ms | not_measured | managed | 1.838x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 3.9975 ms | not_measured | managed | 1.122x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.00160 ms | not_measured | managed | 0.542x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.02313 ms | not_measured | managed | 1.005x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 7.5789 ms | not_measured | managed | 0.972x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.00310 ms | not_measured | managed | 0.280x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.04706 ms | not_measured | managed | 0.495x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 19.4564 ms | not_measured | managed | 0.737x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.00165 ms | not_measured | managed | 0.497x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.00496 ms | not_measured | managed | 4.766x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 0.9434 ms | not_measured | managed | 3.443x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.00121 ms | not_measured | managed | 0.536x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.09772 ms | not_measured | managed | 2.858x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 55.7208 ms | not_measured | managed | 0.603x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.00477 ms | not_measured | managed | 0.653x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.8984 ms | not_measured | managed | 0.306x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 79.7439 ms | not_measured | managed | 0.375x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.000816 ms | not_measured | managed | 0.488x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.02087 ms | not_measured | managed | 0.320x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 10.7335 ms | not_measured | managed | 0.758x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.00135 ms | not_measured | managed | 0.352x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.04332 ms | not_measured | managed | 0.527x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 14.0488 ms | not_measured | managed | 1.144x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.000810 ms | not_measured | managed | 0.794x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.01135 ms | not_measured | managed | 2.492x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 5.7790 ms | not_measured | managed | 0.845x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.00101 ms | not_measured | managed | 0.640x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.02150 ms | not_measured | managed | 1.321x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 3.9639 ms | not_measured | managed | 2.009x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.00141 ms | not_measured | managed | 0.464x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.04430 ms | not_measured | managed | 0.688x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 25.1668 ms | not_measured | managed | 0.641x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.000659 ms | not_measured | managed | 0.923x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.00392 ms | not_measured | managed | 7.230x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 1.8811 ms | not_measured | managed | 1.997x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.000729 ms | not_measured | managed | 0.908x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.01289 ms | not_measured | managed | 2.198x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 2.7537 ms | not_measured | managed | 1.774x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.000952 ms | not_measured | managed | 0.681x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.02154 ms | not_measured | managed | 1.315x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.6328 ms | not_measured | managed | 0.764x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.000791 ms | not_measured | managed | 0.827x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.04472 ms | not_measured | managed | 0.721x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 13.6523 ms | not_measured | managed | 1.202x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.000664 ms | not_measured | managed | 0.919x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.00624 ms | not_measured | managed | 4.543x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.9933 ms | not_measured | managed | 1.874x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.00128 ms | not_measured | managed | 0.566x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.1231 ms | not_measured | managed | 2.243x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 38.5555 ms | not_measured | managed | 0.812x |
| `a + scalar (float16)` | float16 | 1,000 | 0.00928 ms | not_measured | managed | 0.351x |
| `a + scalar (float16)` | float16 | 100,000 | 0.8900 ms | not_measured | managed | 0.308x |
| `a + scalar (float16)` | float16 | 10,000,000 | 79.7548 ms | not_measured | managed | 0.376x |
| `a + scalar (float32)` | float32 | 1,000 | 0.000961 ms | not_measured | managed | 0.542x |
| `a + scalar (float32)` | float32 | 100,000 | 0.01295 ms | not_measured | managed | 0.474x |
| `a + scalar (float32)` | float32 | 10,000,000 | 8.4829 ms | not_measured | managed | 0.871x |
| `a + scalar (float64)` | float64 | 1,000 | 0.000811 ms | not_measured | managed | 0.744x |
| `a + scalar (float64)` | float64 | 100,000 | 0.04105 ms | not_measured | managed | 0.276x |
| `a + scalar (float64)` | float64 | 10,000,000 | 18.9922 ms | not_measured | managed | 0.812x |
| `a + scalar (int16)` | int16 | 1,000 | 0.000845 ms | not_measured | managed | 0.877x |
| `a + scalar (int16)` | int16 | 100,000 | 0.01058 ms | not_measured | managed | 2.241x |
| `a + scalar (int16)` | int16 | 10,000,000 | 3.3020 ms | not_measured | managed | 1.294x |
| `a + scalar (int32)` | int32 | 1,000 | 0.000966 ms | not_measured | managed | 0.780x |
| `a + scalar (int32)` | int32 | 100,000 | 0.02019 ms | not_measured | managed | 1.156x |
| `a + scalar (int32)` | int32 | 10,000,000 | 8.5689 ms | not_measured | managed | 0.876x |
| `a + scalar (int64)` | int64 | 1,000 | 0.00135 ms | not_measured | managed | 0.558x |
| `a + scalar (int64)` | int64 | 100,000 | 0.02592 ms | not_measured | managed | 0.899x |
| `a + scalar (int64)` | int64 | 10,000,000 | 18.9620 ms | not_measured | managed | 0.772x |
| `a + scalar (int8)` | int8 | 1,000 | 0.000555 ms | not_measured | managed | 1.295x |
| `a + scalar (int8)` | int8 | 100,000 | 0.00376 ms | not_measured | managed | 6.250x |
| `a + scalar (int8)` | int8 | 10,000,000 | 1.3020 ms | not_measured | managed | 2.535x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.000813 ms | not_measured | managed | 0.932x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.01106 ms | not_measured | managed | 2.142x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 3.9456 ms | not_measured | managed | 1.125x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.000629 ms | not_measured | managed | 1.191x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.01345 ms | not_measured | managed | 1.717x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 7.5251 ms | not_measured | managed | 0.979x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.00131 ms | not_measured | managed | 0.581x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.04023 ms | not_measured | managed | 0.575x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 19.0081 ms | not_measured | managed | 0.772x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.000412 ms | not_measured | managed | 1.772x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.00568 ms | not_measured | managed | 4.136x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 1.3246 ms | not_measured | managed | 2.439x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.00171 ms | not_measured | managed | 0.381x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.09995 ms | not_measured | managed | 2.781x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 55.4683 ms | not_measured | managed | 0.632x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.00688 ms | not_measured | managed | 0.471x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.8616 ms | not_measured | managed | 0.318x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 90.4369 ms | not_measured | managed | 0.331x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.00105 ms | not_measured | managed | 0.378x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.02275 ms | not_measured | managed | 0.294x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 5.5119 ms | not_measured | managed | 1.446x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.00147 ms | not_measured | managed | 0.325x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.04270 ms | not_measured | managed | 0.535x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 13.7704 ms | not_measured | managed | 1.170x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.000698 ms | not_measured | managed | 0.924x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.00806 ms | not_measured | managed | 3.510x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 3.0036 ms | not_measured | managed | 1.639x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.000838 ms | not_measured | managed | 0.764x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.02264 ms | not_measured | managed | 1.260x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 7.0050 ms | not_measured | managed | 1.175x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.00149 ms | not_measured | managed | 0.439x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.04770 ms | not_measured | managed | 0.639x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 14.6999 ms | not_measured | managed | 1.077x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.000647 ms | not_measured | managed | 0.937x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.00636 ms | not_measured | managed | 4.460x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 1.5752 ms | not_measured | managed | 2.369x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.000608 ms | not_measured | managed | 1.053x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.01196 ms | not_measured | managed | 2.372x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.7467 ms | not_measured | managed | 1.352x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.000845 ms | not_measured | managed | 0.761x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.02220 ms | not_measured | managed | 1.279x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.6071 ms | not_measured | managed | 0.788x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.00135 ms | not_measured | managed | 0.475x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.04562 ms | not_measured | managed | 0.711x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 13.9404 ms | not_measured | managed | 1.135x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.000690 ms | not_measured | managed | 0.880x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.00577 ms | not_measured | managed | 4.914x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.2518 ms | not_measured | managed | 2.976x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.00184 ms | not_measured | managed | 0.391x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.07580 ms | not_measured | managed | 3.642x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 26.7889 ms | not_measured | managed | 1.162x |
| `a - scalar (float16)` | float16 | 1,000 | 0.00935 ms | not_measured | managed | 0.349x |
| `a - scalar (float16)` | float16 | 100,000 | 0.4692 ms | not_measured | managed | 0.585x |
| `a - scalar (float16)` | float16 | 10,000,000 | 45.9254 ms | not_measured | managed | 0.651x |
| `a - scalar (float32)` | float32 | 1,000 | 0.000958 ms | not_measured | managed | 0.547x |
| `a - scalar (float32)` | float32 | 100,000 | 0.02267 ms | not_measured | managed | 0.257x |
| `a - scalar (float32)` | float32 | 10,000,000 | 5.0963 ms | not_measured | managed | 1.499x |
| `a - scalar (float64)` | float64 | 1,000 | 0.00131 ms | not_measured | managed | 0.458x |
| `a - scalar (float64)` | float64 | 100,000 | 0.02690 ms | not_measured | managed | 0.421x |
| `a - scalar (float64)` | float64 | 10,000,000 | 12.9035 ms | not_measured | managed | 1.216x |
| `a - scalar (int16)` | int16 | 1,000 | 0.000850 ms | not_measured | managed | 0.873x |
| `a - scalar (int16)` | int16 | 100,000 | 0.01150 ms | not_measured | managed | 2.007x |
| `a - scalar (int16)` | int16 | 10,000,000 | 2.5316 ms | not_measured | managed | 1.697x |
| `a - scalar (int32)` | int32 | 1,000 | 0.00105 ms | not_measured | managed | 0.711x |
| `a - scalar (int32)` | int32 | 100,000 | 0.01282 ms | not_measured | managed | 1.815x |
| `a - scalar (int32)` | int32 | 10,000,000 | 5.1305 ms | not_measured | managed | 1.481x |
| `a - scalar (int64)` | int64 | 1,000 | 0.000843 ms | not_measured | managed | 0.885x |
| `a - scalar (int64)` | int64 | 100,000 | 0.02643 ms | not_measured | managed | 0.915x |
| `a - scalar (int64)` | int64 | 10,000,000 | 13.0160 ms | not_measured | managed | 1.112x |
| `a - scalar (int8)` | int8 | 1,000 | 0.000583 ms | not_measured | managed | 1.226x |
| `a - scalar (int8)` | int8 | 100,000 | 0.00387 ms | not_measured | managed | 5.969x |
| `a - scalar (int8)` | int8 | 10,000,000 | 1.0600 ms | not_measured | managed | 3.078x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.000697 ms | not_measured | managed | 1.088x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.00695 ms | not_measured | managed | 3.326x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 4.4018 ms | not_measured | managed | 0.980x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.00102 ms | not_measured | managed | 0.736x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.01397 ms | not_measured | managed | 1.663x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 3.3087 ms | not_measured | managed | 2.217x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.00101 ms | not_measured | managed | 0.748x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.04320 ms | not_measured | managed | 0.552x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 12.9254 ms | not_measured | managed | 1.125x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.000677 ms | not_measured | managed | 1.053x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.00657 ms | not_measured | managed | 3.515x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 0.9483 ms | not_measured | managed | 3.478x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.00131 ms | not_measured | managed | 0.347x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.06186 ms | not_measured | managed | 0.195x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 3.9044 ms | not_measured | managed | 2.039x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.00271 ms | not_measured | managed | 0.264x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.1947 ms | not_measured | managed | 0.192x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 14.3026 ms | not_measured | managed | 1.110x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.00403 ms | not_measured | managed | 0.407x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.08098 ms | not_measured | managed | 1.014x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 13.5025 ms | not_measured | managed | 1.439x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.00410 ms | not_measured | managed | 0.386x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.08082 ms | not_measured | managed | 0.937x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 15.6059 ms | not_measured | managed | 1.429x |
| `a / scalar (float32)` | float32 | 1,000 | 0.00131 ms | not_measured | managed | 0.448x |
| `a / scalar (float32)` | float32 | 100,000 | 0.06207 ms | not_measured | managed | 0.193x |
| `a / scalar (float32)` | float32 | 10,000,000 | 8.7331 ms | not_measured | managed | 0.879x |
| `a / scalar (float64)` | float64 | 1,000 | 0.00264 ms | not_measured | managed | 0.318x |
| `a / scalar (float64)` | float64 | 100,000 | 0.1944 ms | not_measured | managed | 0.191x |
| `a / scalar (float64)` | float64 | 10,000,000 | 23.6382 ms | not_measured | managed | 0.646x |
| `a / scalar (int32)` | int32 | 1,000 | 0.00439 ms | not_measured | managed | 0.341x |
| `a / scalar (int32)` | int32 | 100,000 | 0.1990 ms | not_measured | managed | 0.304x |
| `a / scalar (int32)` | int32 | 10,000,000 | 23.6747 ms | not_measured | managed | 0.686x |
| `a / scalar (int64)` | int64 | 1,000 | 0.00411 ms | not_measured | managed | 0.366x |
| `a / scalar (int64)` | int64 | 100,000 | 0.1997 ms | not_measured | managed | 0.285x |
| `a / scalar (int64)` | int64 | 10,000,000 | 14.2049 ms | not_measured | managed | 1.259x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.00153 ms | not_measured | managed | 0.429x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.1480 ms | not_measured | managed | 1.899x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 56.1526 ms | not_measured | managed | 0.610x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.00934 ms | not_measured | managed | 0.334x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.8964 ms | not_measured | managed | 0.307x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 89.9405 ms | not_measured | managed | 0.333x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.000586 ms | not_measured | managed | 0.688x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.02421 ms | not_measured | managed | 0.275x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 10.6598 ms | not_measured | managed | 0.763x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.00134 ms | not_measured | managed | 0.358x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.04250 ms | not_measured | managed | 0.537x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 13.7121 ms | not_measured | managed | 1.176x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.000822 ms | not_measured | managed | 0.794x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.00750 ms | not_measured | managed | 3.770x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 3.8714 ms | not_measured | managed | 1.248x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.000974 ms | not_measured | managed | 0.672x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.02262 ms | not_measured | managed | 1.257x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 4.6542 ms | not_measured | managed | 1.740x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.00135 ms | not_measured | managed | 0.489x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.04478 ms | not_measured | managed | 0.683x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 24.6353 ms | not_measured | managed | 0.648x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.000609 ms | not_measured | managed | 1.026x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.00609 ms | not_measured | managed | 4.658x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 1.8525 ms | not_measured | managed | 2.044x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.000799 ms | not_measured | managed | 0.826x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.01183 ms | not_measured | managed | 2.396x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 4.9108 ms | not_measured | managed | 1.002x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.00100 ms | not_measured | managed | 0.655x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.01442 ms | not_measured | managed | 1.969x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 3.8491 ms | not_measured | managed | 2.089x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.00136 ms | not_measured | managed | 0.484x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.04363 ms | not_measured | managed | 0.743x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 13.9017 ms | not_measured | managed | 1.205x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.000650 ms | not_measured | managed | 0.972x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.00404 ms | not_measured | managed | 7.034x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 1.9819 ms | not_measured | managed | 1.882x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.00131 ms | not_measured | managed | 0.354x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.06205 ms | not_measured | managed | 0.191x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 3.9773 ms | not_measured | managed | 2.027x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.00276 ms | not_measured | managed | 0.260x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.1944 ms | not_measured | managed | 0.193x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 13.9937 ms | not_measured | managed | 1.135x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.00393 ms | not_measured | managed | 0.417x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.1967 ms | not_measured | managed | 0.418x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 23.9032 ms | not_measured | managed | 0.815x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.00406 ms | not_measured | managed | 0.388x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.1983 ms | not_measured | managed | 0.382x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 18.8953 ms | not_measured | managed | 1.179x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.01095 ms | not_measured | managed | 1.076x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 1.5373 ms | not_measured | managed | 0.963x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 115.4708 ms | not_measured | managed | 1.368x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.00617 ms | not_measured | managed | 1.600x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 1.0549 ms | not_measured | managed | 1.180x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 141.2398 ms | not_measured | managed | 0.973x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.00230 ms | not_measured | managed | 0.640x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.6037 ms | not_measured | managed | 0.597x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 61.4808 ms | not_measured | managed | 0.677x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.00231 ms | not_measured | managed | 0.941x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.6008 ms | not_measured | managed | 0.593x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 67.3333 ms | not_measured | managed | 0.695x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.00569 ms | not_measured | managed | 2.037x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 1.5485 ms | not_measured | managed | 0.950x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 130.1594 ms | not_measured | managed | 1.198x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.00908 ms | not_measured | managed | 1.107x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 0.9033 ms | not_measured | managed | 1.378x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 127.0469 ms | not_measured | managed | 1.079x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.00229 ms | not_measured | managed | 0.693x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.5817 ms | not_measured | managed | 0.616x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 51.7219 ms | not_measured | managed | 0.812x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.00234 ms | not_measured | managed | 1.331x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.5037 ms | not_measured | managed | 0.713x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 65.6287 ms | not_measured | managed | 0.718x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.00212 ms | not_measured | managed | 0.340x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.1496 ms | not_measured | managed | 1.842x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 54.4428 ms | not_measured | managed | 0.612x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.00934 ms | not_measured | managed | 0.334x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.8964 ms | not_measured | managed | 0.306x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 79.3706 ms | not_measured | managed | 0.379x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.000625 ms | not_measured | managed | 0.642x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.02138 ms | not_measured | managed | 0.314x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 10.6156 ms | not_measured | managed | 0.791x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.00126 ms | not_measured | managed | 0.385x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.04704 ms | not_measured | managed | 0.482x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 16.6765 ms | not_measured | managed | 0.973x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.000569 ms | not_measured | managed | 1.127x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.01196 ms | not_measured | managed | 2.276x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 5.8190 ms | not_measured | managed | 0.822x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.000977 ms | not_measured | managed | 0.662x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.02286 ms | not_measured | managed | 1.217x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 3.9227 ms | not_measured | managed | 2.153x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.00111 ms | not_measured | managed | 0.586x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.05461 ms | not_measured | managed | 0.549x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 28.0939 ms | not_measured | managed | 0.578x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.000648 ms | not_measured | managed | 0.954x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.00564 ms | not_measured | managed | 4.825x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 1.9758 ms | not_measured | managed | 1.828x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.000617 ms | not_measured | managed | 1.057x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.01357 ms | not_measured | managed | 2.005x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 5.7998 ms | not_measured | managed | 0.832x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.000654 ms | not_measured | managed | 0.992x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.02396 ms | not_measured | managed | 1.152x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 10.7089 ms | not_measured | managed | 0.754x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.00131 ms | not_measured | managed | 0.498x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.05864 ms | not_measured | managed | 0.528x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 26.7197 ms | not_measured | managed | 0.629x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.000515 ms | not_measured | managed | 1.196x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.01205 ms | not_measured | managed | 2.257x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 2.1741 ms | not_measured | managed | 1.665x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.00203 ms | not_measured | managed | 0.320x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.1470 ms | not_measured | managed | 1.891x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 56.6760 ms | not_measured | managed | 0.568x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.00940 ms | not_measured | managed | 0.342x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.7848 ms | not_measured | managed | 0.350x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 84.7039 ms | not_measured | managed | 0.355x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.00102 ms | not_measured | managed | 0.393x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.02528 ms | not_measured | managed | 0.264x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 5.8430 ms | not_measured | managed | 1.403x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.00129 ms | not_measured | managed | 0.372x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.04746 ms | not_measured | managed | 0.480x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 20.7675 ms | not_measured | managed | 0.768x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.000838 ms | not_measured | managed | 0.773x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.00730 ms | not_measured | managed | 3.880x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 5.1701 ms | not_measured | managed | 0.955x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.00100 ms | not_measured | managed | 0.659x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.02408 ms | not_measured | managed | 1.183x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 9.5617 ms | not_measured | managed | 0.838x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.000923 ms | not_measured | managed | 0.714x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.04488 ms | not_measured | managed | 0.682x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 24.7261 ms | not_measured | managed | 0.638x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.000563 ms | not_measured | managed | 1.115x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.00643 ms | not_measured | managed | 4.416x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 2.0441 ms | not_measured | managed | 1.828x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.000807 ms | not_measured | managed | 0.803x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.01296 ms | not_measured | managed | 2.190x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 4.9013 ms | not_measured | managed | 1.017x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.00101 ms | not_measured | managed | 0.649x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.02267 ms | not_measured | managed | 1.252x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 8.8618 ms | not_measured | managed | 0.955x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.00125 ms | not_measured | managed | 0.524x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.04788 ms | not_measured | managed | 0.677x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 14.6532 ms | not_measured | managed | 1.120x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.000603 ms | not_measured | managed | 1.041x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.00672 ms | not_measured | managed | 4.226x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 2.0704 ms | not_measured | managed | 1.810x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.00132 ms | not_measured | managed | 0.350x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.06185 ms | not_measured | managed | 0.192x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 10.7128 ms | not_measured | managed | 0.756x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.00274 ms | not_measured | managed | 0.263x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.1947 ms | not_measured | managed | 0.192x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 25.5065 ms | not_measured | managed | 0.627x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.00393 ms | not_measured | managed | 0.417x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.1978 ms | not_measured | managed | 0.414x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 14.3391 ms | not_measured | managed | 1.349x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.00270 ms | not_measured | managed | 0.584x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.1982 ms | not_measured | managed | 0.382x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 28.5705 ms | not_measured | managed | 0.793x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.00182 ms | not_measured | managed | 0.402x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.07766 ms | not_measured | managed | 3.475x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 26.1033 ms | not_measured | managed | 1.604x |
| `scalar - a (float16)` | float16 | 1,000 | 0.00620 ms | not_measured | managed | 0.528x |
| `scalar - a (float16)` | float16 | 100,000 | 0.4529 ms | not_measured | managed | 0.607x |
| `scalar - a (float16)` | float16 | 10,000,000 | 46.2916 ms | not_measured | managed | 0.650x |
| `scalar - a (float32)` | float32 | 1,000 | 0.00113 ms | not_measured | managed | 0.480x |
| `scalar - a (float32)` | float32 | 100,000 | 0.01307 ms | not_measured | managed | 0.449x |
| `scalar - a (float32)` | float32 | 10,000,000 | 5.1467 ms | not_measured | managed | 1.448x |
| `scalar - a (float64)` | float64 | 1,000 | 0.00132 ms | not_measured | managed | 0.471x |
| `scalar - a (float64)` | float64 | 100,000 | 0.02656 ms | not_measured | managed | 0.429x |
| `scalar - a (float64)` | float64 | 10,000,000 | 12.8703 ms | not_measured | managed | 1.203x |
| `scalar - a (int16)` | int16 | 1,000 | 0.000824 ms | not_measured | managed | 0.937x |
| `scalar - a (int16)` | int16 | 100,000 | 0.00676 ms | not_measured | managed | 3.507x |
| `scalar - a (int16)` | int16 | 10,000,000 | 3.0286 ms | not_measured | managed | 1.441x |
| `scalar - a (int32)` | int32 | 1,000 | 0.00100 ms | not_measured | managed | 0.768x |
| `scalar - a (int32)` | int32 | 100,000 | 0.01299 ms | not_measured | managed | 1.812x |
| `scalar - a (int32)` | int32 | 10,000,000 | 4.9895 ms | not_measured | managed | 1.448x |
| `scalar - a (int64)` | int64 | 1,000 | 0.000768 ms | not_measured | managed | 1.008x |
| `scalar - a (int64)` | int64 | 100,000 | 0.02665 ms | not_measured | managed | 0.887x |
| `scalar - a (int64)` | int64 | 10,000,000 | 12.9520 ms | not_measured | managed | 1.137x |
| `scalar - a (int8)` | int8 | 1,000 | 0.000655 ms | not_measured | managed | 1.118x |
| `scalar - a (int8)` | int8 | 100,000 | 0.00713 ms | not_measured | managed | 3.295x |
| `scalar - a (int8)` | int8 | 10,000,000 | 0.9412 ms | not_measured | managed | 3.393x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.000682 ms | not_measured | managed | 1.119x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.00697 ms | not_measured | managed | 3.410x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 3.0552 ms | not_measured | managed | 1.428x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.000694 ms | not_measured | managed | 1.101x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.01351 ms | not_measured | managed | 1.739x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 3.7698 ms | not_measured | managed | 1.963x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.00103 ms | not_measured | managed | 0.749x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.04053 ms | not_measured | managed | 0.580x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 13.6568 ms | not_measured | managed | 1.051x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.000592 ms | not_measured | managed | 1.230x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.00495 ms | not_measured | managed | 4.754x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 0.9680 ms | not_measured | managed | 3.304x |
| `scalar / a (float32)` | float32 | 1,000 | 0.000654 ms | not_measured | managed | 0.911x |
| `scalar / a (float32)` | float32 | 100,000 | 0.01591 ms | not_measured | managed | 0.754x |
| `scalar / a (float32)` | float32 | 10,000,000 | 4.7438 ms | not_measured | managed | 1.596x |
| `scalar / a (float64)` | float64 | 1,000 | 0.00272 ms | not_measured | managed | 0.310x |
| `scalar / a (float64)` | float64 | 100,000 | 0.1362 ms | not_measured | managed | 0.272x |
| `scalar / a (float64)` | float64 | 10,000,000 | 23.6845 ms | not_measured | managed | 0.635x |
| `scalar / a (int32)` | int32 | 1,000 | 0.00288 ms | not_measured | managed | 0.526x |
| `scalar / a (int32)` | int32 | 100,000 | 0.1602 ms | not_measured | managed | 0.379x |
| `scalar / a (int32)` | int32 | 10,000,000 | 23.7133 ms | not_measured | managed | 0.696x |
| `scalar / a (int64)` | int64 | 1,000 | 0.00443 ms | not_measured | managed | 0.343x |
| `scalar / a (int64)` | int64 | 100,000 | 0.1996 ms | not_measured | managed | 0.285x |
| `scalar / a (int64)` | int64 | 10,000,000 | 24.1785 ms | not_measured | managed | 0.736x |
| `a & b (bool)` | bool | 1,000 | 0.000413 ms | not_measured | managed | 0.801x |
| `a & b (bool)` | bool | 100,000 | 0.00398 ms | not_measured | managed | 1.303x |
| `a & b (bool)` | bool | 10,000,000 | 1.9255 ms | not_measured | managed | 1.253x |
| `a & b (int16)` | int16 | 1,000 | 0.000833 ms | not_measured | managed | 1.780x |
| `a & b (int16)` | int16 | 100,000 | 0.01298 ms | not_measured | managed | 3.868x |
| `a & b (int16)` | int16 | 10,000,000 | 5.6775 ms | not_measured | managed | 1.387x |
| `a & b (int32)` | int32 | 1,000 | 0.00100 ms | not_measured | managed | 1.536x |
| `a & b (int32)` | int32 | 100,000 | 0.02278 ms | not_measured | managed | 2.300x |
| `a & b (int32)` | int32 | 10,000,000 | 10.6458 ms | not_measured | managed | 1.462x |
| `a & b (int64)` | int64 | 1,000 | 0.00127 ms | not_measured | managed | 1.229x |
| `a & b (int64)` | int64 | 100,000 | 0.03440 ms | not_measured | managed | 1.654x |
| `a & b (int64)` | int64 | 10,000,000 | 24.7785 ms | not_measured | managed | 0.922x |
| `a & b (int8)` | int8 | 1,000 | 0.000674 ms | not_measured | managed | 1.788x |
| `a & b (int8)` | int8 | 100,000 | 0.00682 ms | not_measured | managed | 7.195x |
| `a & b (int8)` | int8 | 10,000,000 | 1.9328 ms | not_measured | managed | 3.074x |
| `a & b (uint16)` | uint16 | 1,000 | 0.000829 ms | not_measured | managed | 0.774x |
| `a & b (uint16)` | uint16 | 100,000 | 0.01359 ms | not_measured | managed | 3.698x |
| `a & b (uint16)` | uint16 | 10,000,000 | 4.2893 ms | not_measured | managed | 1.779x |
| `a & b (uint32)` | uint32 | 1,000 | 0.000964 ms | not_measured | managed | 1.559x |
| `a & b (uint32)` | uint32 | 100,000 | 0.01357 ms | not_measured | managed | 3.863x |
| `a & b (uint32)` | uint32 | 10,000,000 | 10.6741 ms | not_measured | managed | 1.473x |
| `a & b (uint64)` | uint64 | 1,000 | 0.00128 ms | not_measured | managed | 1.234x |
| `a & b (uint64)` | uint64 | 100,000 | 0.03664 ms | not_measured | managed | 1.554x |
| `a & b (uint64)` | uint64 | 10,000,000 | 22.4760 ms | not_measured | managed | 1.093x |
| `a & b (uint8)` | uint8 | 1,000 | 0.000691 ms | not_measured | managed | 0.881x |
| `a & b (uint8)` | uint8 | 100,000 | 0.00646 ms | not_measured | managed | 7.599x |
| `a & b (uint8)` | uint8 | 10,000,000 | 1.9399 ms | not_measured | managed | 3.018x |
| `a ^ b (bool)` | bool | 1,000 | 0.000629 ms | not_measured | managed | 0.526x |
| `a ^ b (bool)` | bool | 100,000 | 0.00625 ms | not_measured | managed | 0.427x |
| `a ^ b (bool)` | bool | 10,000,000 | 1.1249 ms | not_measured | managed | 2.174x |
| `a ^ b (int16)` | int16 | 1,000 | 0.000549 ms | not_measured | managed | 2.745x |
| `a ^ b (int16)` | int16 | 100,000 | 0.01334 ms | not_measured | managed | 3.763x |
| `a ^ b (int16)` | int16 | 10,000,000 | 2.7308 ms | not_measured | managed | 3.067x |
| `a ^ b (int32)` | int32 | 1,000 | 0.000877 ms | not_measured | managed | 1.719x |
| `a ^ b (int32)` | int32 | 100,000 | 0.02211 ms | not_measured | managed | 2.370x |
| `a ^ b (int32)` | int32 | 10,000,000 | 3.8716 ms | not_measured | managed | 4.048x |
| `a ^ b (int64)` | int64 | 1,000 | 0.00124 ms | not_measured | managed | 1.273x |
| `a ^ b (int64)` | int64 | 100,000 | 0.03431 ms | not_measured | managed | 1.658x |
| `a ^ b (int64)` | int64 | 10,000,000 | 25.1260 ms | not_measured | managed | 1.235x |
| `a ^ b (int8)` | int8 | 1,000 | 0.000659 ms | not_measured | managed | 1.806x |
| `a ^ b (int8)` | int8 | 100,000 | 0.00573 ms | not_measured | managed | 8.569x |
| `a ^ b (int8)` | int8 | 10,000,000 | 1.9804 ms | not_measured | managed | 2.954x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.000554 ms | not_measured | managed | 2.679x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.01338 ms | not_measured | managed | 3.750x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 5.7993 ms | not_measured | managed | 1.440x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.000797 ms | not_measured | managed | 1.898x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.02173 ms | not_measured | managed | 2.412x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 10.6233 ms | not_measured | managed | 1.501x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.00127 ms | not_measured | managed | 1.227x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.04612 ms | not_measured | managed | 1.233x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 24.6831 ms | not_measured | managed | 0.667x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.000679 ms | not_measured | managed | 0.903x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.00626 ms | not_measured | managed | 7.836x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 0.9857 ms | not_measured | managed | 6.005x |
| `a | b (bool)` | bool | 1,000 | 0.000625 ms | not_measured | managed | 0.528x |
| `a | b (bool)` | bool | 100,000 | 0.00635 ms | not_measured | managed | 0.853x |
| `a | b (bool)` | bool | 10,000,000 | 1.9253 ms | not_measured | managed | 1.280x |
| `a | b (int16)` | int16 | 1,000 | 0.000681 ms | not_measured | managed | 0.916x |
| `a | b (int16)` | int16 | 100,000 | 0.01238 ms | not_measured | managed | 3.875x |
| `a | b (int16)` | int16 | 10,000,000 | 5.7536 ms | not_measured | managed | 1.400x |
| `a | b (int32)` | int32 | 1,000 | 0.000667 ms | not_measured | managed | 2.289x |
| `a | b (int32)` | int32 | 100,000 | 0.02499 ms | not_measured | managed | 1.923x |
| `a | b (int32)` | int32 | 10,000,000 | 10.6608 ms | not_measured | managed | 1.431x |
| `a | b (int64)` | int64 | 1,000 | 0.00125 ms | not_measured | managed | 1.255x |
| `a | b (int64)` | int64 | 100,000 | 0.04342 ms | not_measured | managed | 1.310x |
| `a | b (int64)` | int64 | 10,000,000 | 25.0184 ms | not_measured | managed | 1.222x |
| `a | b (int8)` | int8 | 1,000 | 0.000457 ms | not_measured | managed | 2.683x |
| `a | b (int8)` | int8 | 100,000 | 0.00687 ms | not_measured | managed | 7.149x |
| `a | b (int8)` | int8 | 10,000,000 | 1.0187 ms | not_measured | managed | 5.745x |
| `a | b (uint16)` | uint16 | 1,000 | 0.000719 ms | not_measured | managed | 2.043x |
| `a | b (uint16)` | uint16 | 100,000 | 0.01349 ms | not_measured | managed | 3.557x |
| `a | b (uint16)` | uint16 | 10,000,000 | 5.7940 ms | not_measured | managed | 1.437x |
| `a | b (uint32)` | uint32 | 1,000 | 0.000962 ms | not_measured | managed | 1.564x |
| `a | b (uint32)` | uint32 | 100,000 | 0.01389 ms | not_measured | managed | 3.458x |
| `a | b (uint32)` | uint32 | 10,000,000 | 10.6137 ms | not_measured | managed | 1.501x |
| `a | b (uint64)` | uint64 | 1,000 | 0.00137 ms | not_measured | managed | 1.142x |
| `a | b (uint64)` | uint64 | 100,000 | 0.04707 ms | not_measured | managed | 1.209x |
| `a | b (uint64)` | uint64 | 10,000,000 | 24.8189 ms | not_measured | managed | 1.225x |
| `a | b (uint8)` | uint8 | 1,000 | 0.000636 ms | not_measured | managed | 0.961x |
| `a | b (uint8)` | uint8 | 100,000 | 0.00649 ms | not_measured | managed | 7.571x |
| `a | b (uint8)` | uint8 | 10,000,000 | 1.9332 ms | not_measured | managed | 3.051x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.000603 ms | not_measured | managed | 0.582x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.00657 ms | not_measured | managed | 0.824x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 1.8361 ms | not_measured | managed | 1.329x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.000534 ms | not_measured | managed | 1.232x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.00778 ms | not_measured | managed | 6.484x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 2.7184 ms | not_measured | managed | 3.024x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.000979 ms | not_measured | managed | 1.714x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.01597 ms | not_measured | managed | 3.294x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 10.6409 ms | not_measured | managed | 1.456x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.000757 ms | not_measured | managed | 2.288x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.03985 ms | not_measured | managed | 1.432x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 13.9786 ms | not_measured | managed | 2.215x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.000647 ms | not_measured | managed | 2.122x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.00708 ms | not_measured | managed | 6.963x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 1.6813 ms | not_measured | managed | 3.552x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.000802 ms | not_measured | managed | 1.996x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.00740 ms | not_measured | managed | 6.802x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 5.8995 ms | not_measured | managed | 1.373x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.000675 ms | not_measured | managed | 2.473x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.02221 ms | not_measured | managed | 2.367x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 10.3606 ms | not_measured | managed | 1.552x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.00129 ms | not_measured | managed | 1.334x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.04955 ms | not_measured | managed | 1.152x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 25.1602 ms | not_measured | managed | 1.204x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.000545 ms | not_measured | managed | 2.503x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.00737 ms | not_measured | managed | 6.690x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 1.9348 ms | not_measured | managed | 3.051x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.000562 ms | not_measured | managed | 0.580x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.00343 ms | not_measured | managed | 1.329x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 1.3284 ms | not_measured | managed | 1.565x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.000766 ms | not_measured | managed | 1.909x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.01165 ms | not_measured | managed | 3.629x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 2.4818 ms | not_measured | managed | 2.536x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.000944 ms | not_measured | managed | 1.656x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.02004 ms | not_measured | managed | 2.547x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 8.5698 ms | not_measured | managed | 1.470x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.000746 ms | not_measured | managed | 1.960x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.04391 ms | not_measured | managed | 0.964x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 18.7996 ms | not_measured | managed | 1.206x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.000638 ms | not_measured | managed | 1.948x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.00428 ms | not_measured | managed | 9.868x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 0.8817 ms | not_measured | managed | 5.718x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.000821 ms | not_measured | managed | 1.778x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.01171 ms | not_measured | managed | 3.606x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 4.1545 ms | not_measured | managed | 1.487x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.000621 ms | not_measured | managed | 2.514x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.01216 ms | not_measured | managed | 4.193x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 3.1229 ms | not_measured | managed | 3.806x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.00116 ms | not_measured | managed | 1.278x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.02643 ms | not_measured | managed | 1.600x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 18.3554 ms | not_measured | managed | 1.235x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.000637 ms | not_measured | managed | 1.945x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.00632 ms | not_measured | managed | 6.683x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 1.3137 ms | not_measured | managed | 2.620x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.000484 ms | not_measured | managed | 0.721x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.00598 ms | not_measured | managed | 0.932x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 1.9470 ms | not_measured | managed | 1.227x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.000812 ms | not_measured | managed | 2.011x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.00758 ms | not_measured | managed | 6.354x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 5.8062 ms | not_measured | managed | 1.429x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.000977 ms | not_measured | managed | 1.703x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.02874 ms | not_measured | managed | 1.678x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 7.4478 ms | not_measured | managed | 2.104x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.00130 ms | not_measured | managed | 1.324x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.03588 ms | not_measured | managed | 1.591x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 13.8036 ms | not_measured | managed | 1.999x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.000414 ms | not_measured | managed | 3.413x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.00429 ms | not_measured | managed | 11.490x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 1.9056 ms | not_measured | managed | 3.086x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.000849 ms | not_measured | managed | 1.900x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.01206 ms | not_measured | managed | 3.991x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 5.8510 ms | not_measured | managed | 1.421x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.000990 ms | not_measured | managed | 1.668x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.01923 ms | not_measured | managed | 2.502x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 3.9058 ms | not_measured | managed | 3.965x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.00128 ms | not_measured | managed | 1.352x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.03356 ms | not_measured | managed | 1.702x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 22.9391 ms | not_measured | managed | 0.908x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.000627 ms | not_measured | managed | 2.201x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.00424 ms | not_measured | managed | 11.631x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 0.9690 ms | not_measured | managed | 6.131x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.000605 ms | not_measured | managed | 0.579x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.00395 ms | not_measured | managed | 1.349x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 1.9598 ms | not_measured | managed | 1.243x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.000837 ms | not_measured | managed | 1.973x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.01266 ms | not_measured | managed | 3.986x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 2.9465 ms | not_measured | managed | 2.817x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.000995 ms | not_measured | managed | 1.674x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.02189 ms | not_measured | managed | 1.301x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 10.7440 ms | not_measured | managed | 1.456x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.00130 ms | not_measured | managed | 1.318x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.03661 ms | not_measured | managed | 1.559x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 24.9531 ms | not_measured | managed | 1.225x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.000421 ms | not_measured | managed | 3.309x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.00698 ms | not_measured | managed | 7.052x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 1.8580 ms | not_measured | managed | 3.163x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.000854 ms | not_measured | managed | 1.915x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.01461 ms | not_measured | managed | 3.444x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 5.8014 ms | not_measured | managed | 1.336x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.000963 ms | not_measured | managed | 1.720x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.02231 ms | not_measured | managed | 2.354x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 3.7896 ms | not_measured | managed | 4.208x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.00131 ms | not_measured | managed | 1.310x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.03480 ms | not_measured | managed | 1.640x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 14.0470 ms | not_measured | managed | 2.075x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.000626 ms | not_measured | managed | 2.211x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.00738 ms | not_measured | managed | 6.675x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 1.9117 ms | not_measured | managed | 3.103x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.000454 ms | not_measured | managed | 0.720x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.00382 ms | not_measured | managed | 1.218x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 1.3381 ms | not_measured | managed | 1.563x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.000699 ms | not_measured | managed | 2.093x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.01111 ms | not_measured | managed | 3.802x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 2.5625 ms | not_measured | managed | 1.746x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.000951 ms | not_measured | managed | 1.653x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.02049 ms | not_measured | managed | 2.491x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 5.2777 ms | not_measured | managed | 2.330x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.000718 ms | not_measured | managed | 2.088x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.02551 ms | not_measured | managed | 1.659x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 18.8412 ms | not_measured | managed | 1.252x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.000631 ms | not_measured | managed | 1.987x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.00600 ms | not_measured | managed | 7.044x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 1.3329 ms | not_measured | managed | 3.765x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.000792 ms | not_measured | managed | 1.847x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.00679 ms | not_measured | managed | 6.215x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 4.0295 ms | not_measured | managed | 1.548x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.000627 ms | not_measured | managed | 2.509x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.02104 ms | not_measured | managed | 2.423x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 7.6101 ms | not_measured | managed | 1.606x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.000800 ms | not_measured | managed | 1.856x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.03973 ms | not_measured | managed | 1.065x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 18.9245 ms | not_measured | managed | 1.207x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.000625 ms | not_measured | managed | 0.933x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.00584 ms | not_measured | managed | 7.238x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 1.3003 ms | not_measured | managed | 3.861x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.00327 ms | not_measured | managed | 0.385x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.04022 ms | not_measured | managed | 2.201x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 15.7040 ms | not_measured | managed | 1.383x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.000896 ms | not_measured | managed | 2.481x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.01274 ms | not_measured | managed | 3.433x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 2.6198 ms | not_measured | managed | 2.433x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.00156 ms | not_measured | managed | 1.515x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.01391 ms | not_measured | managed | 3.734x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.5126 ms | not_measured | managed | 1.361x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.00137 ms | not_measured | managed | 0.570x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.05005 ms | not_measured | managed | 0.743x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 18.9785 ms | not_measured | managed | 1.129x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.00117 ms | not_measured | managed | 1.787x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.00440 ms | not_measured | managed | 9.917x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.3080 ms | not_measured | managed | 4.092x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.000886 ms | not_measured | managed | 2.572x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.01326 ms | not_measured | managed | 2.121x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 2.9277 ms | not_measured | managed | 2.408x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00171 ms | not_measured | managed | 1.382x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.02034 ms | not_measured | managed | 2.551x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 4.8205 ms | not_measured | managed | 1.758x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00201 ms | not_measured | managed | 1.086x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.02753 ms | not_measured | managed | 1.356x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 18.4840 ms | not_measured | managed | 1.271x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.000769 ms | not_measured | managed | 1.126x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.00766 ms | not_measured | managed | 5.709x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 0.8890 ms | not_measured | managed | 6.076x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.00161 ms | not_measured | managed | 0.846x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.00164 ms | not_measured | managed | 67.889x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.00135 ms | not_measured | managed | 1.820x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.00805 ms | not_measured | managed | 8.242x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.1323 ms | not_measured | managed | 2.015x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.00164 ms | not_measured | managed | 1.497x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.02126 ms | not_measured | managed | 2.854x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.5065 ms | not_measured | managed | 1.479x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.00139 ms | not_measured | managed | 1.571x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.04994 ms | not_measured | managed | 1.210x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 18.9699 ms | not_measured | managed | 1.218x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.00104 ms | not_measured | managed | 0.932x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.00447 ms | not_measured | managed | 16.327x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.1460 ms | not_measured | managed | 7.194x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.00136 ms | not_measured | managed | 1.693x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.00727 ms | not_measured | managed | 6.726x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 2.4861 ms | not_measured | managed | 2.694x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00140 ms | not_measured | managed | 1.585x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.02172 ms | not_measured | managed | 1.724x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.4936 ms | not_measured | managed | 1.409x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00209 ms | not_measured | managed | 1.103x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.04305 ms | not_measured | managed | 0.443x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 18.9321 ms | not_measured | managed | 1.170x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.00101 ms | not_measured | managed | 2.075x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.00438 ms | not_measured | managed | 11.179x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.2773 ms | not_measured | managed | 4.437x |
| `matrix * 2.0` | float64 | 1,000 | 0.00175 ms | not_measured | managed | 0.345x |
| `matrix * 2.0` | float64 | 100,000 | 0.03880 ms | not_measured | managed | 0.324x |
| `matrix * 2.0` | float64 | 10,000,000 | 12.9252 ms | not_measured | managed | 1.741x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.000906 ms | not_measured | managed | 1.158x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.04787 ms | not_measured | managed | 0.587x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 18.7964 ms | not_measured | managed | 1.314x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.00163 ms | not_measured | managed | 0.594x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.04619 ms | not_measured | managed | 0.574x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 19.5444 ms | not_measured | managed | 1.271x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.00156 ms | not_measured | managed | 0.653x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.03823 ms | not_measured | managed | 0.735x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 19.3167 ms | not_measured | managed | 0.813x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.00147 ms | not_measured | managed | 0.660x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.02931 ms | not_measured | managed | 0.904x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 12.6137 ms | not_measured | managed | 1.816x |
| `matrix + scalar` | float64 | 1,000 | 0.00117 ms | not_measured | managed | 0.395x |
| `matrix + scalar` | float64 | 100,000 | 0.03997 ms | not_measured | managed | 0.309x |
| `matrix + scalar` | float64 | 10,000,000 | 18.9265 ms | not_measured | managed | 0.800x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.000315 ms | not_measured | managed | 5.543x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.000561 ms | not_measured | managed | 7.529x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.000358 ms | not_measured | managed | 11.743x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.000225 ms | not_measured | managed | 7.676x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.000533 ms | not_measured | managed | 7.908x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.000539 ms | not_measured | managed | 7.798x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.000949 ms | not_measured | managed | 0.911x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.05687 ms | not_measured | managed | 0.454x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 20.3377 ms | not_measured | managed | 1.122x |
| `a != b (float32)` | float32 | 1,000 | 0.000522 ms | not_measured | managed | 1.755x |
| `a != b (float32)` | float32 | 100,000 | 0.01123 ms | not_measured | managed | 0.390x |
| `a != b (float32)` | float32 | 10,000,000 | 8.4489 ms | not_measured | managed | 1.038x |
| `a != b (float64)` | float64 | 1,000 | 0.000473 ms | not_measured | managed | 0.841x |
| `a != b (float64)` | float64 | 100,000 | 0.02007 ms | not_measured | managed | 1.064x |
| `a != b (float64)` | float64 | 10,000,000 | 16.2160 ms | not_measured | managed | 0.393x |
| `a != b (int32)` | int32 | 1,000 | 0.000675 ms | not_measured | managed | 0.581x |
| `a != b (int32)` | int32 | 100,000 | 0.00705 ms | not_measured | managed | 1.553x |
| `a != b (int32)` | int32 | 10,000,000 | 2.8502 ms | not_measured | managed | 3.440x |
| `a != b (int64)` | int64 | 1,000 | 0.000418 ms | not_measured | managed | 1.022x |
| `a != b (int64)` | int64 | 100,000 | 0.01876 ms | not_measured | managed | 1.026x |
| `a != b (int64)` | int64 | 10,000,000 | 5.2865 ms | not_measured | managed | 3.260x |
| `a < b (float32)` | float32 | 1,000 | 0.000645 ms | not_measured | managed | 1.350x |
| `a < b (float32)` | float32 | 100,000 | 0.01193 ms | not_measured | managed | 1.101x |
| `a < b (float32)` | float32 | 10,000,000 | 8.4235 ms | not_measured | managed | 1.123x |
| `a < b (float64)` | float64 | 1,000 | 0.000689 ms | not_measured | managed | 1.444x |
| `a < b (float64)` | float64 | 100,000 | 0.02066 ms | not_measured | managed | 1.049x |
| `a < b (float64)` | float64 | 10,000,000 | 5.2302 ms | not_measured | managed | 3.443x |
| `a < b (int32)` | int32 | 1,000 | 0.000636 ms | not_measured | managed | 0.596x |
| `a < b (int32)` | int32 | 100,000 | 0.01220 ms | not_measured | managed | 0.567x |
| `a < b (int32)` | int32 | 10,000,000 | 8.4336 ms | not_measured | managed | 1.151x |
| `a < b (int64)` | int64 | 1,000 | 0.000438 ms | not_measured | managed | 1.110x |
| `a < b (int64)` | int64 | 100,000 | 0.01051 ms | not_measured | managed | 1.671x |
| `a < b (int64)` | int64 | 10,000,000 | 5.1222 ms | not_measured | managed | 2.895x |
| `a <= b (float32)` | float32 | 1,000 | 0.000626 ms | not_measured | managed | 1.369x |
| `a <= b (float32)` | float32 | 100,000 | 0.01178 ms | not_measured | managed | 1.023x |
| `a <= b (float32)` | float32 | 10,000,000 | 7.4822 ms | not_measured | managed | 1.241x |
| `a <= b (float64)` | float64 | 1,000 | 0.000549 ms | not_measured | managed | 1.809x |
| `a <= b (float64)` | float64 | 100,000 | 0.01805 ms | not_measured | managed | 1.243x |
| `a <= b (float64)` | float64 | 10,000,000 | 16.1714 ms | not_measured | managed | 0.399x |
| `a <= b (int32)` | int32 | 1,000 | 0.000725 ms | not_measured | managed | 0.530x |
| `a <= b (int32)` | int32 | 100,000 | 0.01216 ms | not_measured | managed | 0.983x |
| `a <= b (int32)` | int32 | 10,000,000 | 6.0533 ms | not_measured | managed | 1.615x |
| `a <= b (int64)` | int64 | 1,000 | 0.000784 ms | not_measured | managed | 1.355x |
| `a <= b (int64)` | int64 | 100,000 | 0.02392 ms | not_measured | managed | 0.866x |
| `a <= b (int64)` | int64 | 10,000,000 | 15.7667 ms | not_measured | managed | 1.110x |
| `a == b (float32)` | float32 | 1,000 | 0.000419 ms | not_measured | managed | 2.136x |
| `a == b (float32)` | float32 | 100,000 | 0.01175 ms | not_measured | managed | 1.070x |
| `a == b (float32)` | float32 | 10,000,000 | 8.2037 ms | not_measured | managed | 1.137x |
| `a == b (float64)` | float64 | 1,000 | 0.000695 ms | not_measured | managed | 0.564x |
| `a == b (float64)` | float64 | 100,000 | 0.01459 ms | not_measured | managed | 0.665x |
| `a == b (float64)` | float64 | 10,000,000 | 16.1012 ms | not_measured | managed | 0.757x |
| `a == b (int32)` | int32 | 1,000 | 0.000412 ms | not_measured | managed | 0.908x |
| `a == b (int32)` | int32 | 100,000 | 0.01136 ms | not_measured | managed | 0.927x |
| `a == b (int32)` | int32 | 10,000,000 | 2.5954 ms | not_measured | managed | 3.718x |
| `a == b (int64)` | int64 | 1,000 | 0.000713 ms | not_measured | managed | 0.571x |
| `a == b (int64)` | int64 | 100,000 | 0.01918 ms | not_measured | managed | 0.926x |
| `a == b (int64)` | int64 | 10,000,000 | 15.6021 ms | not_measured | managed | 1.066x |
| `a > b (float32)` | float32 | 1,000 | 0.000662 ms | not_measured | managed | 1.331x |
| `a > b (float32)` | float32 | 100,000 | 0.01179 ms | not_measured | managed | 1.061x |
| `a > b (float32)` | float32 | 10,000,000 | 3.0147 ms | not_measured | managed | 2.942x |
| `a > b (float64)` | float64 | 1,000 | 0.000439 ms | not_measured | managed | 2.267x |
| `a > b (float64)` | float64 | 100,000 | 0.01418 ms | not_measured | managed | 1.539x |
| `a > b (float64)` | float64 | 10,000,000 | 5.1564 ms | not_measured | managed | 3.265x |
| `a > b (int32)` | int32 | 1,000 | 0.000487 ms | not_measured | managed | 0.780x |
| `a > b (int32)` | int32 | 100,000 | 0.01120 ms | not_measured | managed | 1.069x |
| `a > b (int32)` | int32 | 10,000,000 | 8.3355 ms | not_measured | managed | 1.164x |
| `a > b (int64)` | int64 | 1,000 | 0.000545 ms | not_measured | managed | 1.831x |
| `a > b (int64)` | int64 | 100,000 | 0.01959 ms | not_measured | managed | 0.910x |
| `a > b (int64)` | int64 | 10,000,000 | 15.9810 ms | not_measured | managed | 1.019x |
| `a >= b (float32)` | float32 | 1,000 | 0.000612 ms | not_measured | managed | 1.451x |
| `a >= b (float32)` | float32 | 100,000 | 0.01201 ms | not_measured | managed | 1.024x |
| `a >= b (float32)` | float32 | 10,000,000 | 8.4365 ms | not_measured | managed | 1.110x |
| `a >= b (float64)` | float64 | 1,000 | 0.000410 ms | not_measured | managed | 2.351x |
| `a >= b (float64)` | float64 | 100,000 | 0.01227 ms | not_measured | managed | 1.769x |
| `a >= b (float64)` | float64 | 10,000,000 | 16.1283 ms | not_measured | managed | 1.066x |
| `a >= b (int32)` | int32 | 1,000 | 0.000633 ms | not_measured | managed | 0.607x |
| `a >= b (int32)` | int32 | 100,000 | 0.00840 ms | not_measured | managed | 0.803x |
| `a >= b (int32)` | int32 | 10,000,000 | 8.5959 ms | not_measured | managed | 1.133x |
| `a >= b (int64)` | int64 | 1,000 | 0.000441 ms | not_measured | managed | 2.424x |
| `a >= b (int64)` | int64 | 100,000 | 0.01273 ms | not_measured | managed | 1.628x |
| `a >= b (int64)` | int64 | 10,000,000 | 5.3828 ms | not_measured | managed | 3.258x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.000463 ms | not_measured | managed | 0.844x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.01129 ms | not_measured | managed | 0.396x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 8.0752 ms | not_measured | managed | 1.156x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.000420 ms | not_measured | managed | 2.743x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.02039 ms | not_measured | managed | 1.063x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 5.2503 ms | not_measured | managed | 1.302x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.000661 ms | not_measured | managed | 0.589x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.01130 ms | not_measured | managed | 0.946x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 8.1137 ms | not_measured | managed | 1.057x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.000682 ms | not_measured | managed | 1.691x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.01066 ms | not_measured | managed | 1.685x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 14.7915 ms | not_measured | managed | 1.187x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.000422 ms | not_measured | managed | 0.938x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.01178 ms | not_measured | managed | 1.075x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 2.5835 ms | not_measured | managed | 1.583x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.000695 ms | not_measured | managed | 1.643x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.02061 ms | not_measured | managed | 1.070x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 15.9610 ms | not_measured | managed | 1.061x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.000448 ms | not_measured | managed | 0.866x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.00650 ms | not_measured | managed | 1.053x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 8.5534 ms | not_measured | managed | 1.055x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.000732 ms | not_measured | managed | 1.567x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.01985 ms | not_measured | managed | 0.906x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 13.5637 ms | not_measured | managed | 1.280x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.000593 ms | not_measured | managed | 1.755x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.01148 ms | not_measured | managed | 0.383x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 2.8383 ms | not_measured | managed | 3.100x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.000668 ms | not_measured | managed | 0.606x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.02021 ms | not_measured | managed | 1.076x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 15.8188 ms | not_measured | managed | 1.096x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.000601 ms | not_measured | managed | 0.672x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.01209 ms | not_measured | managed | 1.000x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 8.7683 ms | not_measured | managed | 1.133x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.000763 ms | not_measured | managed | 1.561x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.02385 ms | not_measured | managed | 0.875x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 15.9419 ms | not_measured | managed | 1.087x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.000425 ms | not_measured | managed | 2.449x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.01173 ms | not_measured | managed | 1.121x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 8.4709 ms | not_measured | managed | 1.101x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.000452 ms | not_measured | managed | 2.454x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.01257 ms | not_measured | managed | 1.738x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 16.2240 ms | not_measured | managed | 1.112x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.000647 ms | not_measured | managed | 0.601x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.00594 ms | not_measured | managed | 2.036x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 2.9188 ms | not_measured | managed | 3.244x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.000638 ms | not_measured | managed | 1.776x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.02009 ms | not_measured | managed | 0.894x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 5.1700 ms | not_measured | managed | 1.368x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.000610 ms | not_measured | managed | 1.693x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.01175 ms | not_measured | managed | 1.055x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 8.5259 ms | not_measured | managed | 1.010x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.000485 ms | not_measured | managed | 2.338x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.01489 ms | not_measured | managed | 1.506x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 16.2097 ms | not_measured | managed | 1.056x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.000539 ms | not_measured | managed | 0.742x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.00606 ms | not_measured | managed | 2.003x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 8.2920 ms | not_measured | managed | 1.168x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.000768 ms | not_measured | managed | 1.553x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.02363 ms | not_measured | managed | 0.883x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 15.5358 ms | not_measured | managed | 1.113x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.000601 ms | not_measured | managed | 0.654x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.00862 ms | not_measured | managed | 1.587x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 7.7941 ms | not_measured | managed | 1.204x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.000739 ms | not_measured | managed | 1.571x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.02004 ms | not_measured | managed | 1.077x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 5.5649 ms | not_measured | managed | 1.196x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.000506 ms | not_measured | managed | 0.806x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.00722 ms | not_measured | managed | 1.537x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 8.3971 ms | not_measured | managed | 1.131x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.000690 ms | not_measured | managed | 1.697x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.02088 ms | not_measured | managed | 0.931x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 5.3915 ms | not_measured | managed | 3.235x |
| `a.copy() (float32)` | float32 | 1,000 | 0.000505 ms | not_measured | managed | 1.095x |
| `a.copy() (float32)` | float32 | 100,000 | 0.01010 ms | not_measured | managed | 1.020x |
| `a.copy() (float32)` | float32 | 10,000,000 | 1.8545 ms | not_measured | managed | 4.147x |
| `a.copy() (float64)` | float64 | 1,000 | 0.000518 ms | not_measured | managed | 1.195x |
| `a.copy() (float64)` | float64 | 100,000 | 0.01902 ms | not_measured | managed | 1.934x |
| `a.copy() (float64)` | float64 | 10,000,000 | 16.1315 ms | not_measured | managed | 0.989x |
| `a.copy() (int32)` | int32 | 1,000 | 0.000497 ms | not_measured | managed | 0.527x |
| `a.copy() (int32)` | int32 | 100,000 | 0.00604 ms | not_measured | managed | 1.702x |
| `a.copy() (int32)` | int32 | 10,000,000 | 4.1660 ms | not_measured | managed | 1.862x |
| `a.copy() (int64)` | int64 | 1,000 | 0.000577 ms | not_measured | managed | 1.090x |
| `a.copy() (int64)` | int64 | 100,000 | 0.01657 ms | not_measured | managed | 1.222x |
| `a.copy() (int64)` | int64 | 10,000,000 | 12.1888 ms | not_measured | managed | 1.442x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.000613 ms | not_measured | managed | 2.542x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.04604 ms | not_measured | managed | 1.079x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 5.5919 ms | not_measured | managed | 1.653x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.000475 ms | not_measured | managed | 3.320x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.04720 ms | not_measured | managed | 1.080x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 15.8968 ms | not_measured | managed | 1.005x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.000532 ms | not_measured | managed | 1.143x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.02213 ms | not_measured | managed | 1.968x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 5.5301 ms | not_measured | managed | 1.584x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.000913 ms | not_measured | managed | 1.561x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.04925 ms | not_measured | managed | 0.754x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 16.1892 ms | not_measured | managed | 1.109x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.000692 ms | not_measured | managed | 1.124x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.04143 ms | not_measured | managed | 0.477x |
| `np.array(a) (float64)` | float64 | 10,000,000 | 0.4296 ms | not_measured | managed | 2.772x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.000003 ms | not_measured | managed | 31.000x |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.000005 ms | not_measured | managed | 17.800x |
| `np.asanyarray(a) (float64)` | float64 | 10,000,000 | 0.000003 ms | not_measured | managed | 29.333x |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 89.000x |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 88.000x |
| `np.asarray(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 89.000x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.000052 ms | not_measured | managed | 80.192x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.00653 ms | not_measured | managed | 3.270x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 10,000,000 | 0.1679 ms | not_measured | managed | 1.525x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.00131 ms | not_measured | managed | 0.511x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.01880 ms | not_measured | managed | 1.527x |
| `np.ascontiguousarray(a) (float64)` | float64 | 10,000,000 | 0.2750 ms | not_measured | managed | 3.205x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.00182 ms | not_measured | managed | 0.762x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.1437 ms | not_measured | managed | 0.396x |
| `np.asfortranarray(a) (float64)` | float64 | 10,000,000 | 1.2900 ms | not_measured | managed | 1.483x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.000239 ms | not_measured | managed | 7.828x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.000244 ms | not_measured | managed | 17.459x |
| `np.asmatrix(a) (float64)` | float64 | 10,000,000 | 0.000335 ms | not_measured | managed | 12.693x |
| `np.copy (float32)` | float32 | 1,000 | 0.000530 ms | not_measured | managed | 2.309x |
| `np.copy (float32)` | float32 | 100,000 | 0.01016 ms | not_measured | managed | 1.095x |
| `np.copy (float32)` | float32 | 10,000,000 | 2.3814 ms | not_measured | managed | 3.376x |
| `np.copy (float64)` | float64 | 1,000 | 0.000552 ms | not_measured | managed | 2.543x |
| `np.copy (float64)` | float64 | 100,000 | 0.02132 ms | not_measured | managed | 1.775x |
| `np.copy (float64)` | float64 | 10,000,000 | 15.9503 ms | not_measured | managed | 1.084x |
| `np.copy (int32)` | int32 | 1,000 | 0.000525 ms | not_measured | managed | 0.882x |
| `np.copy (int32)` | int32 | 100,000 | 0.00902 ms | not_measured | managed | 1.233x |
| `np.copy (int32)` | int32 | 10,000,000 | 4.1861 ms | not_measured | managed | 1.945x |
| `np.copy (int64)` | int64 | 1,000 | 0.000594 ms | not_measured | managed | 2.350x |
| `np.copy (int64)` | int64 | 100,000 | 0.02140 ms | not_measured | managed | 0.989x |
| `np.copy (int64)` | int64 | 10,000,000 | 12.4490 ms | not_measured | managed | 1.404x |
| `np.empty (float32)` | float32 | 1,000 | 0.000372 ms | not_measured | managed | 1.118x |
| `np.empty (float32)` | float32 | 100,000 | 0.000240 ms | not_measured | managed | 3.154x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.03169 ms | not_measured | managed | 0.611x |
| `np.empty (float64)` | float64 | 1,000 | 0.000340 ms | not_measured | managed | 1.221x |
| `np.empty (float64)` | float64 | 100,000 | 0.000343 ms | not_measured | managed | 2.210x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.06948 ms | not_measured | managed | 0.213x |
| `np.empty (int32)` | int32 | 1,000 | 0.000225 ms | not_measured | managed | 0.916x |
| `np.empty (int32)` | int32 | 100,000 | 0.000311 ms | not_measured | managed | 2.328x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.03578 ms | not_measured | managed | 0.547x |
| `np.empty (int64)` | int64 | 1,000 | 0.000228 ms | not_measured | managed | 1.803x |
| `np.empty (int64)` | int64 | 100,000 | 0.000366 ms | not_measured | managed | 1.893x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.07305 ms | not_measured | managed | 0.275x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.000422 ms | not_measured | managed | 1.204x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.000446 ms | not_measured | managed | 1.901x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.05411 ms | not_measured | managed | 0.360x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.000252 ms | not_measured | managed | 2.056x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.000397 ms | not_measured | managed | 2.111x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.06766 ms | not_measured | managed | 0.225x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.000435 ms | not_measured | managed | 0.570x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.000333 ms | not_measured | managed | 2.556x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.03638 ms | not_measured | managed | 0.541x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.000452 ms | not_measured | managed | 1.102x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.000266 ms | not_measured | managed | 3.203x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.03932 ms | not_measured | managed | 0.513x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.00104 ms | not_measured | managed | 2.965x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.07446 ms | not_measured | managed | 0.358x |
| `np.eye(n) (float64)` | float64 | 10,000,000 | 1.2395 ms | not_measured | managed | 0.871x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.000399 ms | not_measured | managed | 1.782x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.000466 ms | not_measured | managed | 1.521x |
| `np.frombuffer(buffer) (float64)` | float64 | 10,000,000 | 0.000804 ms | not_measured | managed | 0.878x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.03242 ms | not_measured | managed | 5.195x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 5.8404 ms | not_measured | managed | 3.131x |
| `np.fromstring(text) (float64)` | float64 | 10,000,000 | 55.9657 ms | not_measured | managed | 7.183x |
| `np.full (float32)` | float32 | 1,000 | 0.000429 ms | not_measured | managed | 5.676x |
| `np.full (float32)` | float32 | 100,000 | 0.01597 ms | not_measured | managed | 0.815x |
| `np.full (float32)` | float32 | 10,000,000 | 3.5293 ms | not_measured | managed | 2.167x |
| `np.full (float64)` | float64 | 1,000 | 0.000477 ms | not_measured | managed | 5.312x |
| `np.full (float64)` | float64 | 100,000 | 0.01988 ms | not_measured | managed | 1.186x |
| `np.full (float64)` | float64 | 10,000,000 | 16.1395 ms | not_measured | managed | 0.985x |
| `np.full (int32)` | int32 | 1,000 | 0.000510 ms | not_measured | managed | 1.480x |
| `np.full (int32)` | int32 | 100,000 | 0.01066 ms | not_measured | managed | 1.201x |
| `np.full (int32)` | int32 | 10,000,000 | 3.5137 ms | not_measured | managed | 2.398x |
| `np.full (int64)` | int64 | 1,000 | 0.000539 ms | not_measured | managed | 1.364x |
| `np.full (int64)` | int64 | 100,000 | 0.01993 ms | not_measured | managed | 1.172x |
| `np.full (int64)` | int64 | 10,000,000 | 11.9210 ms | not_measured | managed | 1.326x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.000397 ms | not_measured | managed | 6.854x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.01798 ms | not_measured | managed | 0.743x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 4.0752 ms | not_measured | managed | 1.998x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.000386 ms | not_measured | managed | 7.098x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.01986 ms | not_measured | managed | 1.201x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 13.4129 ms | not_measured | managed | 1.197x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.000389 ms | not_measured | managed | 2.401x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.01349 ms | not_measured | managed | 0.971x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 3.3098 ms | not_measured | managed | 2.533x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.000511 ms | not_measured | managed | 4.843x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.01184 ms | not_measured | managed | 1.983x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 13.4848 ms | not_measured | managed | 1.127x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.00160 ms | not_measured | managed | 2.460x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.07379 ms | not_measured | managed | 0.376x |
| `np.identity(n) (float64)` | float64 | 10,000,000 | 0.9617 ms | not_measured | managed | 1.187x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.00162 ms | not_measured | managed | 8.612x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.1759 ms | not_measured | managed | 1.351x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 17.7189 ms | not_measured | managed | 2.836x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.00214 ms | not_measured | managed | 5.793x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.1214 ms | not_measured | managed | 0.851x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 24.2478 ms | not_measured | managed | 1.394x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.00335 ms | not_measured | managed | 1.452x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.1741 ms | not_measured | managed | 1.461x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 30.3487 ms | not_measured | managed | 1.099x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.00186 ms | not_measured | managed | 8.024x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.1636 ms | not_measured | managed | 1.611x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 37.3758 ms | not_measured | managed | 1.784x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.00520 ms | not_measured | managed | 4.247x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.1121 ms | not_measured | managed | 3.835x |
| `np.meshgrid(x, y) (float64)` | float64 | 10,000,000 | 1.8010 ms | not_measured | managed | 1.718x |
| `np.ones (float32)` | float32 | 1,000 | 0.000508 ms | not_measured | managed | 4.746x |
| `np.ones (float32)` | float32 | 100,000 | 0.01789 ms | not_measured | managed | 0.726x |
| `np.ones (float32)` | float32 | 10,000,000 | 3.6022 ms | not_measured | managed | 2.321x |
| `np.ones (float64)` | float64 | 1,000 | 0.000473 ms | not_measured | managed | 5.368x |
| `np.ones (float64)` | float64 | 100,000 | 0.01981 ms | not_measured | managed | 1.189x |
| `np.ones (float64)` | float64 | 10,000,000 | 15.8904 ms | not_measured | managed | 1.066x |
| `np.ones (int32)` | int32 | 1,000 | 0.000345 ms | not_measured | managed | 2.183x |
| `np.ones (int32)` | int32 | 100,000 | 0.01790 ms | not_measured | managed | 0.717x |
| `np.ones (int32)` | int32 | 10,000,000 | 3.4616 ms | not_measured | managed | 2.372x |
| `np.ones (int64)` | int64 | 1,000 | 0.000538 ms | not_measured | managed | 1.359x |
| `np.ones (int64)` | int64 | 100,000 | 0.01971 ms | not_measured | managed | 1.183x |
| `np.ones (int64)` | int64 | 10,000,000 | 14.6201 ms | not_measured | managed | 1.168x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.000497 ms | not_measured | managed | 5.410x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.01693 ms | not_measured | managed | 0.787x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 1.8452 ms | not_measured | managed | 4.151x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.000339 ms | not_measured | managed | 8.195x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.01192 ms | not_measured | managed | 2.008x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 16.0328 ms | not_measured | managed | 0.996x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.000564 ms | not_measured | managed | 1.596x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.01087 ms | not_measured | managed | 1.205x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 4.1057 ms | not_measured | managed | 2.033x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.000585 ms | not_measured | managed | 4.193x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.01195 ms | not_measured | managed | 1.967x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 16.2660 ms | not_measured | managed | 0.947x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.00144 ms | not_measured | managed | 1.307x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.03370 ms | not_measured | managed | 0.892x |
| `np.require(a, requirements=C) (float64)` | float64 | 10,000,000 | 0.3196 ms | not_measured | managed | 2.789x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.00603 ms | not_measured | managed | 1.548x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.1279 ms | not_measured | managed | 1.259x |
| `np.vander(x) (float64)` | float64 | 10,000,000 | 2.6654 ms | not_measured | managed | 2.324x |
| `np.zeros (float32)` | float32 | 1,000 | 0.000468 ms | not_measured | managed | 0.583x |
| `np.zeros (float32)` | float32 | 100,000 | 0.00335 ms | not_measured | managed | 3.350x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.00645 ms | not_measured | managed | 3.048x |
| `np.zeros (float64)` | float64 | 1,000 | 0.000532 ms | not_measured | managed | 1.699x |
| `np.zeros (float64)` | float64 | 100,000 | 0.00535 ms | not_measured | managed | 4.045x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.00579 ms | not_measured | managed | 2.597x |
| `np.zeros (int32)` | int32 | 1,000 | 0.000249 ms | not_measured | managed | 1.100x |
| `np.zeros (int32)` | int32 | 100,000 | 0.00155 ms | not_measured | managed | 7.211x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.00730 ms | not_measured | managed | 2.704x |
| `np.zeros (int64)` | int64 | 1,000 | 0.000565 ms | not_measured | managed | 0.503x |
| `np.zeros (int64)` | int64 | 100,000 | 0.00304 ms | not_measured | managed | 7.159x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.00622 ms | not_measured | managed | 3.235x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.000282 ms | not_measured | managed | 8.351x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.00181 ms | not_measured | managed | 7.177x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.01097 ms | not_measured | managed | 727.175x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.000425 ms | not_measured | managed | 5.908x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.00383 ms | not_measured | managed | 6.151x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.01136 ms | not_measured | managed | 1512.348x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.000509 ms | not_measured | managed | 1.692x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.00255 ms | not_measured | managed | 5.046x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.01002 ms | not_measured | managed | 857.143x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.000306 ms | not_measured | managed | 7.974x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.00389 ms | not_measured | managed | 6.018x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.00836 ms | not_measured | managed | 1906.653x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.00833 ms | not_measured | managed | 0.952x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 1.2532 ms | not_measured | managed | 1.892x |
| `np.fft.fft(a) (complex128)` | complex128 | 10,000,000 | 23.3213 ms | not_measured | managed | 0.949x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.06629 ms | not_measured | managed | 0.561x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 10.4893 ms | not_measured | managed | 0.750x |
| `np.fft.fft2(a) (complex128)` | complex128 | 10,000,000 | 39.0573 ms | not_measured | managed | 0.807x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.00415 ms | not_measured | managed | 1.909x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.1732 ms | not_measured | managed | 2.797x |
| `np.fft.fftfreq(n) (float64)` | float64 | 10,000,000 | 1.6947 ms | not_measured | managed | 3.199x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.03279 ms | not_measured | managed | 1.065x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 10.0989 ms | not_measured | managed | 0.776x |
| `np.fft.fftn(a) (complex128)` | complex128 | 10,000,000 | 36.5321 ms | not_measured | managed | 0.788x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.00276 ms | not_measured | managed | 1.498x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.1011 ms | not_measured | managed | 3.878x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 10,000,000 | 1.3143 ms | not_measured | managed | 1.875x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.00555 ms | not_measured | managed | 1.150x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 1.0212 ms | not_measured | managed | 1.436x |
| `np.fft.hfft(a) (float64)` | float64 | 10,000,000 | 14.3011 ms | not_measured | managed | 1.158x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.00864 ms | not_measured | managed | 1.005x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 1.2540 ms | not_measured | managed | 1.842x |
| `np.fft.ifft(a) (complex128)` | complex128 | 10,000,000 | 30.3592 ms | not_measured | managed | 0.991x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.06034 ms | not_measured | managed | 0.641x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 10.4815 ms | not_measured | managed | 0.756x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 10,000,000 | 20.3703 ms | not_measured | managed | 1.114x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.06067 ms | not_measured | managed | 0.599x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 9.9701 ms | not_measured | managed | 0.793x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 10,000,000 | 29.9061 ms | not_measured | managed | 1.087x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.00363 ms | not_measured | managed | 1.130x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.07193 ms | not_measured | managed | 5.497x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 10,000,000 | 1.1963 ms | not_measured | managed | 2.094x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.00544 ms | not_measured | managed | 1.346x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 0.9726 ms | not_measured | managed | 1.137x |
| `np.fft.ihfft(a) (float64)` | float64 | 10,000,000 | 8.3254 ms | not_measured | managed | 1.777x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.00501 ms | not_measured | managed | 1.269x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 1.0439 ms | not_measured | managed | 1.080x |
| `np.fft.irfft(a) (float64)` | float64 | 10,000,000 | 13.1541 ms | not_measured | managed | 0.673x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.03294 ms | not_measured | managed | 0.758x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 2.8800 ms | not_measured | managed | 1.432x |
| `np.fft.irfft2(a) (float64)` | float64 | 10,000,000 | 16.9274 ms | not_measured | managed | 1.017x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.01920 ms | not_measured | managed | 3.049x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 5.0047 ms | not_measured | managed | 0.781x |
| `np.fft.irfftn(a) (float64)` | float64 | 10,000,000 | 16.4659 ms | not_measured | managed | 0.910x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.00669 ms | not_measured | managed | 0.895x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 0.9075 ms | not_measured | managed | 1.141x |
| `np.fft.rfft(a) (float64)` | float64 | 10,000,000 | 10.8872 ms | not_measured | managed | 1.329x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.03188 ms | not_measured | managed | 0.823x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 4.9400 ms | not_measured | managed | 0.816x |
| `np.fft.rfft2(a) (float64)` | float64 | 10,000,000 | 14.9178 ms | not_measured | managed | 1.087x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.00225 ms | not_measured | managed | 1.894x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.07363 ms | not_measured | managed | 0.812x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 10,000,000 | 0.4598 ms | not_measured | managed | 4.104x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.03217 ms | not_measured | managed | 0.734x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 4.9466 ms | not_measured | managed | 0.771x |
| `np.fft.rfftn(a) (float64)` | float64 | 10,000,000 | 14.3878 ms | not_measured | managed | 1.092x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.000201 ms | 0.000157 ms | managed | 3.134x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.000156 ms | 0.000198 ms | managed | 6.551x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.000160 ms | 0.000137 ms | managed | 7.161x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.000759 ms | 0.000636 ms | openblas | 0.810x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.02206 ms | 0.00683 ms | openblas | 1.978x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 7.6026 ms | 7.6582 ms | managed | 1.113x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 1,000 | 0.00390 ms | 0.00210 ms | openblas | 2.905x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 100,000 | 3.8778 ms | 2.2524 ms | openblas | 2.298x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 10,000,000 | 6.4587 ms | 4.0000 ms | openblas | 2.415x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.00274 ms | 0.00182 ms | openblas | 0.749x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 0.01196 ms | 0.01494 ms | managed | 3.282x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 6.6112 ms | 8.0582 ms | managed | 1.370x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.00287 ms | 0.00273 ms | managed | 3.204x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.00259 ms | 0.00257 ms | managed | 8.454x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.00262 ms | 0.00185 ms | managed | 11.664x |
| `np.linalg.cholesky(a)` | float64 | 1,000 | missing_backend | 0.00340 ms | openblas | 1.265x |
| `np.linalg.cholesky(a)` | float64 | 100,000 | missing_backend | 0.03260 ms | openblas | 0.534x |
| `np.linalg.cholesky(a)` | float64 | 10,000,000 | missing_backend | 0.06800 ms | openblas | 1.526x |
| `np.linalg.cond(a)` | float64 | 1,000 | missing_backend | 0.05330 ms | openblas | 0.655x |
| `np.linalg.cond(a)` | float64 | 100,000 | missing_backend | 0.4661 ms | openblas | 1.021x |
| `np.linalg.cond(a)` | float64 | 10,000,000 | missing_backend | 0.9824 ms | openblas | 1.007x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.01158 ms | 0.01148 ms | managed | 1.115x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 0.8483 ms | 0.8561 ms | managed | 0.841x |
| `np.linalg.cross(a, b) (float64)` | float64 | 10,000,000 | 146.6854 ms | 155.8550 ms | managed | 1.293x |
| `np.linalg.det(a)` | float64 | 1,000 | 0.00430 ms | 0.00800 ms | managed | 2.930x |
| `np.linalg.det(a)` | float64 | 100,000 | 0.07280 ms | 0.05430 ms | openblas | 1.081x |
| `np.linalg.det(a)` | float64 | 10,000,000 | 0.1875 ms | 0.1129 ms | openblas | 1.051x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.000175 ms | 0.000232 ms | managed | 3.491x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.000200 ms | 0.000148 ms | managed | 8.230x |
| `np.linalg.diagonal(a) (float64)` | float64 | 10,000,000 | 0.000169 ms | 0.000202 ms | managed | 7.225x |
| `np.linalg.eig(a)` | float64 | 1,000 | missing_backend | 0.1941 ms | openblas | 1.047x |
| `np.linalg.eig(a)` | float64 | 100,000 | missing_backend | 2.6725 ms | openblas | 1.010x |
| `np.linalg.eig(a)` | float64 | 10,000,000 | missing_backend | 5.8937 ms | openblas | 0.990x |
| `np.linalg.eigh(a)` | float64 | 1,000 | missing_backend | 0.08470 ms | openblas | 0.547x |
| `np.linalg.eigh(a)` | float64 | 100,000 | missing_backend | 0.8180 ms | openblas | 0.489x |
| `np.linalg.eigh(a)` | float64 | 10,000,000 | missing_backend | 1.5346 ms | openblas | 1.007x |
| `np.linalg.eigvals(a)` | float64 | 1,000 | missing_backend | 0.1109 ms | openblas | 1.153x |
| `np.linalg.eigvals(a)` | float64 | 100,000 | missing_backend | 1.5514 ms | openblas | 0.995x |
| `np.linalg.eigvals(a)` | float64 | 10,000,000 | missing_backend | 3.2028 ms | openblas | 0.950x |
| `np.linalg.eigvalsh(a)` | float64 | 1,000 | missing_backend | 0.02940 ms | openblas | 1.245x |
| `np.linalg.eigvalsh(a)` | float64 | 100,000 | missing_backend | 0.2839 ms | openblas | 1.019x |
| `np.linalg.eigvalsh(a)` | float64 | 10,000,000 | missing_backend | 0.5183 ms | openblas | 1.015x |
| `np.linalg.inv(a)` | float64 | 1,000 | 0.01250 ms | 0.01570 ms | managed | 1.912x |
| `np.linalg.inv(a)` | float64 | 100,000 | 0.2485 ms | 0.1604 ms | openblas | 1.137x |
| `np.linalg.inv(a)` | float64 | 10,000,000 | 0.5972 ms | 0.3403 ms | openblas | 1.081x |
| `np.linalg.lstsq(a, b)` | float64 | 1,000 | missing_backend | 0.1518 ms | openblas | 1.078x |
| `np.linalg.lstsq(a, b)` | float64 | 100,000 | missing_backend | 1.6220 ms | openblas | 1.004x |
| `np.linalg.lstsq(a, b)` | float64 | 10,000,000 | missing_backend | 2.9606 ms | openblas | 1.001x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.00862 ms | 0.00355 ms | openblas | 0.692x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 3.8018 ms | 2.2820 ms | openblas | 0.988x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 10,000,000 | 6.3805 ms | 3.2369 ms | openblas | 1.319x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.00445 ms | 0.00559 ms | managed | 0.516x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.01166 ms | 0.01025 ms | managed | 1.360x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 10,000,000 | 0.02142 ms | 0.01844 ms | managed | 0.765x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 1,000 | missing_backend | 0.05320 ms | openblas | 1.306x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 100,000 | missing_backend | 0.4653 ms | openblas | 1.034x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 10,000,000 | missing_backend | 0.9814 ms | openblas | 1.010x |
| `np.linalg.matrix_power(a, -1)` | float64 | 1,000 | 0.01330 ms | 0.01650 ms | managed | 1.932x |
| `np.linalg.matrix_power(a, -1)` | float64 | 100,000 | 0.2505 ms | 0.1635 ms | openblas | 1.127x |
| `np.linalg.matrix_power(a, -1)` | float64 | 10,000,000 | 0.6002 ms | 0.3473 ms | openblas | 1.070x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.01709 ms | 0.00942 ms | openblas | 1.531x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.7269 ms | 0.2635 ms | openblas | 1.228x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 10,000,000 | 0.7257 ms | 0.3242 ms | openblas | 0.999x |
| `np.linalg.matrix_rank(a)` | float64 | 1,000 | missing_backend | 0.05590 ms | openblas | 1.283x |
| `np.linalg.matrix_rank(a)` | float64 | 100,000 | missing_backend | 0.4716 ms | openblas | 1.022x |
| `np.linalg.matrix_rank(a)` | float64 | 10,000,000 | missing_backend | 0.9822 ms | openblas | 1.013x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.000246 ms | 0.000402 ms | managed | 3.951x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.000360 ms | 0.000294 ms | managed | 3.265x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 10,000,000 | 0.000320 ms | 0.000302 ms | managed | 3.212x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.00891 ms | 0.00975 ms | managed | 1.530x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.6097 ms | 0.3269 ms | openblas | 1.007x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 10,000,000 | 0.4144 ms | 0.1638 ms | openblas | 2.011x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.000902 ms | 0.00162 ms | managed | 2.948x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.00965 ms | 0.02012 ms | managed | 1.514x |
| `np.linalg.norm(a) (float64)` | float64 | 10,000,000 | 8.2056 ms | 8.2493 ms | managed | 1.004x |
| `np.linalg.norm(a, ord=2)` | float64 | 1,000 | missing_backend | 0.03190 ms | openblas | 2.182x |
| `np.linalg.norm(a, ord=2)` | float64 | 100,000 | missing_backend | 0.4662 ms | openblas | 1.032x |
| `np.linalg.norm(a, ord=2)` | float64 | 10,000,000 | missing_backend | 0.9787 ms | openblas | 1.015x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.00473 ms | 0.00486 ms | managed | 1.123x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.02954 ms | 0.02911 ms | managed | 2.568x |
| `np.linalg.outer(a, b) (float64)` | float64 | 10,000,000 | 12.4446 ms | 11.3506 ms | managed | 1.476x |
| `np.linalg.pinv(a)` | float64 | 1,000 | missing_backend | 0.1601 ms | openblas | 0.528x |
| `np.linalg.pinv(a)` | float64 | 100,000 | missing_backend | 1.5796 ms | openblas | 1.000x |
| `np.linalg.pinv(a)` | float64 | 10,000,000 | missing_backend | 3.0850 ms | openblas | 1.069x |
| `np.linalg.qr(a)` | float64 | 1,000 | missing_backend | 0.03330 ms | openblas | 0.676x |
| `np.linalg.qr(a)` | float64 | 100,000 | missing_backend | 0.3141 ms | openblas | 1.222x |
| `np.linalg.qr(a)` | float64 | 10,000,000 | missing_backend | 0.6888 ms | openblas | 1.155x |
| `np.linalg.slogdet(a)` | float64 | 1,000 | 0.00460 ms | 0.00800 ms | managed | 3.130x |
| `np.linalg.slogdet(a)` | float64 | 100,000 | 0.03220 ms | 0.05480 ms | managed | 1.876x |
| `np.linalg.slogdet(a)` | float64 | 10,000,000 | 0.1889 ms | 0.1133 ms | openblas | 1.055x |
| `np.linalg.solve(a, b)` | float64 | 1,000 | 0.00710 ms | 0.00830 ms | managed | 1.000x |
| `np.linalg.solve(a, b)` | float64 | 100,000 | 0.04640 ms | 0.05790 ms | managed | 1.379x |
| `np.linalg.solve(a, b)` | float64 | 10,000,000 | 0.2406 ms | 0.1187 ms | openblas | 1.051x |
| `np.linalg.svd(a)` | float64 | 1,000 | missing_backend | 0.1357 ms | openblas | 1.098x |
| `np.linalg.svd(a)` | float64 | 100,000 | missing_backend | 1.4284 ms | openblas | 1.028x |
| `np.linalg.svd(a)` | float64 | 10,000,000 | missing_backend | 2.7944 ms | openblas | 1.068x |
| `np.linalg.svdvals(a)` | float64 | 1,000 | missing_backend | 0.04740 ms | openblas | 1.177x |
| `np.linalg.svdvals(a)` | float64 | 100,000 | missing_backend | 0.4632 ms | openblas | 1.008x |
| `np.linalg.svdvals(a)` | float64 | 10,000,000 | missing_backend | 0.9755 ms | openblas | 1.002x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.01067 ms | 0.00705 ms | openblas | 2.005x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.3683 ms | 0.1693 ms | openblas | 1.016x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 10,000,000 | 0.3679 ms | 0.1677 ms | openblas | 1.027x |
| `np.linalg.tensorinv(a)` | float64 | 1,000 | 0.01280 ms | 0.01580 ms | managed | 2.023x |
| `np.linalg.tensorinv(a)` | float64 | 100,000 | 0.2490 ms | 0.1609 ms | openblas | 1.147x |
| `np.linalg.tensorinv(a)` | float64 | 10,000,000 | 0.5967 ms | 0.3431 ms | openblas | 1.084x |
| `np.linalg.tensorsolve(a, b)` | float64 | 1,000 | 0.00760 ms | 0.00890 ms | managed | 2.421x |
| `np.linalg.tensorsolve(a, b)` | float64 | 100,000 | 0.1032 ms | 0.05870 ms | openblas | 1.136x |
| `np.linalg.tensorsolve(a, b)` | float64 | 10,000,000 | 0.2408 ms | 0.1198 ms | openblas | 1.065x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.000503 ms | 0.000541 ms | managed | 7.956x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.000632 ms | 0.000382 ms | managed | 10.668x |
| `np.linalg.trace(a) (float64)` | float64 | 10,000,000 | 0.000639 ms | 0.000481 ms | managed | 8.501x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00272 ms | 0.000654 ms | openblas | 3.644x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.05132 ms | 0.00686 ms | openblas | 2.086x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 10,000,000 | 30.3331 ms | 8.2557 ms | openblas | 1.010x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.00252 ms | 0.00452 ms | managed | 3.017x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.07105 ms | 0.05104 ms | managed | 0.963x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 10,000,000 | 32.7745 ms | 52.3547 ms | managed | 1.114x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.00608 ms | 0.00501 ms | openblas | 0.456x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 3.8260 ms | 1.1103 ms | openblas | 2.026x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 2.4981 ms | 2.7946 ms | managed | 1.715x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.00128 ms | 0.000511 ms | openblas | 1.374x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.01627 ms | 0.01016 ms | openblas | 1.132x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.01083 ms | 0.01486 ms | managed | 1.479x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.00481 ms | 0.00497 ms | managed | 0.376x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.03603 ms | 0.03854 ms | managed | 2.039x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 11.4872 ms | 11.1998 ms | managed | 1.482x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.00258 ms | 0.00267 ms | managed | 1.422x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 0.01508 ms | 0.00857 ms | openblas | 2.413x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 3.2693 ms | 3.3442 ms | managed | 2.583x |
| `np.tensordot(a, b, axes=1)` | float64 | 1,000 | 0.00410 ms | 0.00480 ms | managed | 1.317x |
| `np.tensordot(a, b, axes=1)` | float64 | 100,000 | 3.8851 ms | 2.2485 ms | openblas | 1.014x |
| `np.tensordot(a, b, axes=1)` | float64 | 10,000,000 | 6.4310 ms | 3.9968 ms | openblas | 1.074x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.00103 ms | 0.000731 ms | openblas | 0.767x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.01387 ms | 0.01143 ms | openblas | 2.017x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 6.4934 ms | 3.0356 ms | openblas | 2.703x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00282 ms | 0.000857 ms | openblas | 0.711x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.04122 ms | 0.00913 ms | openblas | 1.501x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 19.6137 ms | 8.2440 ms | openblas | 1.014x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.00162 ms | 0.000869 ms | openblas | 0.829x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.01542 ms | 0.01205 ms | openblas | 1.114x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.01902 ms | 0.00729 ms | openblas | 2.660x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.04004 ms | not_measured | managed | 0.737x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 3.1556 ms | not_measured | managed | 1.264x |
| `np.allclose(a, b) (float16)` | float16 | 10,000,000 | 31.9262 ms | not_measured | managed | 1.321x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.01072 ms | not_measured | managed | 3.382x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.1244 ms | not_measured | managed | 1.059x |
| `np.allclose(a, b) (float32)` | float32 | 10,000,000 | 3.5737 ms | not_measured | managed | 1.610x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.01027 ms | not_measured | managed | 3.533x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.2759 ms | not_measured | managed | 2.169x |
| `np.allclose(a, b) (float64)` | float64 | 10,000,000 | 10.8347 ms | not_measured | managed | 1.174x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.00157 ms | not_measured | managed | 1.542x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.1234 ms | not_measured | managed | 1.571x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 12.4039 ms | not_measured | managed | 1.633x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.000310 ms | not_measured | managed | 14.877x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.01023 ms | not_measured | managed | 1.498x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 8.5522 ms | not_measured | managed | 1.093x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.000554 ms | not_measured | managed | 8.664x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.01920 ms | not_measured | managed | 1.348x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 5.1587 ms | not_measured | managed | 3.295x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.00240 ms | not_measured | managed | 1.064x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.7134 ms | not_measured | managed | 1.264x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 79.3551 ms | not_measured | managed | 1.209x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.00108 ms | not_measured | managed | 1.232x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.03482 ms | not_measured | managed | 0.479x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 10.9939 ms | not_measured | managed | 1.378x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.000748 ms | not_measured | managed | 2.008x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.06921 ms | not_measured | managed | 0.461x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 15.2170 ms | not_measured | managed | 1.989x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.00349 ms | not_measured | managed | 0.761x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.6495 ms | not_measured | managed | 1.373x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 80.8982 ms | not_measured | managed | 1.173x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 1.288x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.01369 ms | not_measured | managed | 1.219x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 11.3146 ms | not_measured | managed | 1.359x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.000747 ms | not_measured | managed | 2.078x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.06931 ms | not_measured | managed | 0.461x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 26.6829 ms | not_measured | managed | 1.112x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.04030 ms | not_measured | managed | 0.662x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 1.8584 ms | not_measured | managed | 2.134x |
| `np.isclose(a, b) (float16)` | float16 | 10,000,000 | 31.9391 ms | not_measured | managed | 1.228x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.01129 ms | not_measured | managed | 2.775x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.1290 ms | not_measured | managed | 0.993x |
| `np.isclose(a, b) (float32)` | float32 | 10,000,000 | 2.8720 ms | not_measured | managed | 2.052x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.00724 ms | not_measured | managed | 4.344x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.2602 ms | not_measured | managed | 2.282x |
| `np.isclose(a, b) (float64)` | float64 | 10,000,000 | 11.0364 ms | not_measured | managed | 1.146x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.000234 ms | not_measured | managed | 1.885x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.00415 ms | not_measured | managed | 1.055x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | 0.00221 ms | not_measured | managed | 9.593x |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.000361 ms | not_measured | managed | 3.061x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.00400 ms | not_measured | managed | 1.105x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | 0.00465 ms | not_measured | managed | 4.550x |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.000358 ms | not_measured | managed | 3.064x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.00407 ms | not_measured | managed | 1.071x |
| `np.iscomplex(a) (float64)` | float64 | 10,000,000 | 0.00275 ms | not_measured | managed | 7.752x |
| `np.iscomplexobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.000578 ms | not_measured | managed | 1.396x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.04142 ms | not_measured | managed | 2.146x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 5.0183 ms | not_measured | managed | 1.968x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.000694 ms | not_measured | managed | 1.401x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.00546 ms | not_measured | managed | 1.674x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 4.4311 ms | not_measured | managed | 1.248x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.000804 ms | not_measured | managed | 1.331x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.01820 ms | not_measured | managed | 0.819x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 9.4374 ms | not_measured | managed | 0.960x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.000777 ms | not_measured | managed | 1.076x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.05339 ms | not_measured | managed | 1.795x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 3.9220 ms | not_measured | managed | 2.768x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.000571 ms | not_measured | managed | 1.716x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.00897 ms | not_measured | managed | 1.023x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 4.4211 ms | not_measured | managed | 1.253x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.000774 ms | not_measured | managed | 1.359x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.00856 ms | not_measured | managed | 1.746x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 9.2472 ms | not_measured | managed | 1.103x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.000569 ms | not_measured | managed | 1.784x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.05242 ms | not_measured | managed | 2.363x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 5.2680 ms | not_measured | managed | 2.547x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.000733 ms | not_measured | managed | 1.291x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.00911 ms | not_measured | managed | 0.867x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 4.4796 ms | not_measured | managed | 1.155x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.000781 ms | not_measured | managed | 1.327x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.01584 ms | not_measured | managed | 0.780x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 9.0206 ms | not_measured | managed | 1.057x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.000351 ms | not_measured | managed | 5.060x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.00229 ms | not_measured | managed | 75.567x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 0.2088 ms | not_measured | managed | 96.061x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.000264 ms | not_measured | managed | 11.962x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.00251 ms | not_measured | managed | 8.547x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 0.3473 ms | not_measured | managed | 29.688x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.000266 ms | not_measured | managed | 12.737x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.00252 ms | not_measured | managed | 14.688x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 0.3519 ms | not_measured | managed | 54.042x |
| `np.isrealobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 1,000 | 0.000001 ms | not_measured | managed | 475.000x |
| `np.isscalar(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 100,000 | 0.000001 ms | not_measured | managed | 603.000x |
| `np.isscalar(a) (float32)` | float32 | 10,000,000 | 0.000001 ms | not_measured | managed | 603.000x |
| `np.isscalar(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 608.000x |
| `np.isscalar(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 603.000x |
| `np.isscalar(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 602.000x |
| `np.logical_and(a, b) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.logical_and(a, b) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.logical_and(a, b) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.logical_not(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.logical_not(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.logical_not(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.logical_or(a, b) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.logical_or(a, b) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.logical_or(a, b) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.logical_xor(a, b) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.logical_xor(a, b) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.logical_xor(a, b) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.00365 ms | not_measured | managed | 0.748x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.6370 ms | not_measured | managed | 1.425x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 81.8608 ms | not_measured | managed | 1.175x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.00101 ms | not_measured | managed | 1.331x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.02272 ms | not_measured | managed | 0.720x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 3.7508 ms | not_measured | managed | 4.022x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.00121 ms | not_measured | managed | 1.288x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.04562 ms | not_measured | managed | 0.683x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 27.1062 ms | not_measured | managed | 0.858x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.00340 ms | not_measured | managed | 0.858x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.6382 ms | not_measured | managed | 1.423x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 88.3463 ms | not_measured | managed | 1.025x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.000609 ms | not_measured | managed | 2.187x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.01422 ms | not_measured | managed | 1.149x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 7.8511 ms | not_measured | managed | 1.727x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.000884 ms | not_measured | managed | 0.531x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.05035 ms | not_measured | managed | 0.620x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 19.6115 ms | not_measured | managed | 0.819x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.000355 ms | not_measured | managed | 0.242x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.000364 ms | not_measured | managed | 0.363x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.000232 ms | not_measured | managed | 0.578x |
| `a.flatten` | float64 | 1,000 | 0.00111 ms | not_measured | managed | 0.397x |
| `a.flatten` | float64 | 100,000 | 0.02133 ms | not_measured | managed | 1.014x |
| `a.flatten` | float64 | 10,000,000 | 12.4560 ms | not_measured | managed | 1.362x |
| `a.ravel() (view)` | float64 | 1,000 | 0.000276 ms | not_measured | managed | 0.297x |
| `a.ravel() (view)` | float64 | 100,000 | 0.000408 ms | not_measured | managed | 0.299x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.000353 ms | not_measured | managed | 0.343x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.00197 ms | not_measured | managed | 1.728x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.09959 ms | not_measured | managed | 3.101x |
| `np.append(a, b) (float64)` | float64 | 10,000,000 | 0.9482 ms | not_measured | managed | 2.546x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.00117 ms | not_measured | managed | 13.360x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.00141 ms | not_measured | managed | 11.104x |
| `np.array_split(a, sections) (float64)` | float64 | 10,000,000 | 0.00119 ms | not_measured | managed | 13.154x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 297.000x |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 297.000x |
| `np.atleast_1d(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 297.000x |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.000296 ms | not_measured | managed | 2.270x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.000293 ms | not_measured | managed | 2.331x |
| `np.atleast_2d(a) (float64)` | float64 | 10,000,000 | 0.000233 ms | not_measured | managed | 2.940x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.000701 ms | not_measured | managed | 1.030x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.000700 ms | not_measured | managed | 1.031x |
| `np.atleast_3d(a) (float64)` | float64 | 10,000,000 | 0.000714 ms | not_measured | managed | 1.015x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.00216 ms | not_measured | managed | 3.380x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.09652 ms | not_measured | managed | 3.985x |
| `np.block([a, b]) (float64)` | float64 | 10,000,000 | 1.4324 ms | not_measured | managed | 1.793x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.000556 ms | not_measured | managed | 14.374x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.000521 ms | not_measured | managed | 5.749x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 10,000,000 | 0.000368 ms | not_measured | managed | 22.245x |
| `np.byteswap` | float64 | 1,000 | 0.000658 ms | not_measured | managed | 2.377x |
| `np.byteswap` | float64 | 100,000 | 0.03805 ms | not_measured | managed | 1.878x |
| `np.byteswap` | float64 | 10,000,000 | 13.3482 ms | not_measured | managed | 1.854x |
| `np.c_` | float64 | 1,000 | 0.00384 ms | not_measured | managed | 2.537x |
| `np.c_` | float64 | 100,000 | 0.1773 ms | not_measured | managed | 2.216x |
| `np.c_` | float64 | 10,000,000 | 32.7765 ms | not_measured | managed | 1.891x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.00191 ms | not_measured | managed | 2.577x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.09355 ms | not_measured | managed | 4.420x |
| `np.column_stack((a, b)) (float64)` | float64 | 10,000,000 | 2.8967 ms | not_measured | managed | 1.685x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.00143 ms | not_measured | managed | 1.499x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.1035 ms | not_measured | managed | 3.573x |
| `np.concat((a, b)) (float64)` | float64 | 10,000,000 | 0.9789 ms | not_measured | managed | 2.534x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.00202 ms | not_measured | managed | 1.484x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.1790 ms | not_measured | managed | 2.943x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 10,000,000 | 3.7023 ms | not_measured | managed | 1.088x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.00206 ms | not_measured | managed | 1.055x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.09566 ms | not_measured | managed | 3.888x |
| `np.concatenate([a, b]) (float64)` | float64 | 10,000,000 | 1.2333 ms | not_measured | managed | 2.069x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.00152 ms | not_measured | managed | 1.631x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.1023 ms | not_measured | managed | 3.447x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 10,000,000 | 1.2096 ms | not_measured | managed | 1.992x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00194 ms | not_measured | managed | 1.367x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.09089 ms | not_measured | managed | 3.695x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 10,000,000 | 2.7403 ms | not_measured | managed | 2.647x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.00239 ms | not_measured | managed | 2.007x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.04543 ms | not_measured | managed | 0.561x |
| `np.delete(a, index) (float64)` | float64 | 10,000,000 | 0.6096 ms | not_measured | managed | 2.007x |
| `np.diag` | float64 | 1,000 | 0.00132 ms | not_measured | managed | 1.035x |
| `np.diag` | float64 | 100,000 | 0.06767 ms | not_measured | managed | 0.021x |
| `np.diag` | float64 | 10,000,000 | 3.9335 ms | not_measured | managed | 0.000x |
| `np.diagflat` | float64 | 1,000 | 0.00156 ms | not_measured | managed | 4.762x |
| `np.diagflat` | float64 | 100,000 | 0.06934 ms | not_measured | managed | 0.460x |
| `np.diagflat` | float64 | 10,000,000 | 2.5660 ms | not_measured | managed | 1.588x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.00174 ms | not_measured | managed | 12.334x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.00191 ms | not_measured | managed | 11.322x |
| `np.dsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00182 ms | not_measured | managed | 11.954x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.00372 ms | not_measured | managed | 1.438x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.1877 ms | not_measured | managed | 2.178x |
| `np.dstack([a, b]) (float64)` | float64 | 10,000,000 | 1.9895 ms | not_measured | managed | 2.098x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.000306 ms | not_measured | managed | 3.873x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.000195 ms | not_measured | managed | 13.349x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.000257 ms | not_measured | managed | 9.907x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.000317 ms | not_measured | managed | 3.666x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.000226 ms | not_measured | managed | 11.571x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.000266 ms | not_measured | managed | 9.564x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.000310 ms | not_measured | managed | 3.835x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.000236 ms | not_measured | managed | 11.233x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.000310 ms | not_measured | managed | 8.542x |
| `np.fill_diagonal` | float64 | 1,000 | 0.000607 ms | not_measured | managed | 1.290x |
| `np.fill_diagonal` | float64 | 100,000 | 0.000897 ms | not_measured | managed | 4.951x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.03658 ms | not_measured | managed | 1.058x |
| `np.flip` | float64 | 1,000 | 0.000237 ms | not_measured | managed | 2.405x |
| `np.flip` | float64 | 100,000 | 0.000199 ms | not_measured | managed | 2.799x |
| `np.flip` | float64 | 10,000,000 | 0.000189 ms | not_measured | managed | 2.905x |
| `np.fliplr` | float64 | 1,000 | 0.000237 ms | not_measured | managed | 2.068x |
| `np.fliplr` | float64 | 100,000 | 0.000254 ms | not_measured | managed | 1.882x |
| `np.fliplr` | float64 | 10,000,000 | 0.000196 ms | not_measured | managed | 2.526x |
| `np.flipud` | float64 | 1,000 | 0.000204 ms | not_measured | managed | 2.020x |
| `np.flipud` | float64 | 100,000 | 0.000201 ms | not_measured | managed | 2.100x |
| `np.flipud` | float64 | 10,000,000 | 0.000196 ms | not_measured | managed | 1.444x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.00179 ms | not_measured | managed | 11.891x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.00183 ms | not_measured | managed | 11.795x |
| `np.hsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00180 ms | not_measured | managed | 12.119x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.00212 ms | not_measured | managed | 1.807x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.1007 ms | not_measured | managed | 3.566x |
| `np.hstack([a, b]) (float64)` | float64 | 10,000,000 | 1.3268 ms | not_measured | managed | 2.010x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.00349 ms | not_measured | managed | 3.580x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.04231 ms | not_measured | managed | 0.822x |
| `np.insert(a, index, value) (float64)` | float64 | 10,000,000 | 0.4713 ms | not_measured | managed | 2.589x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.03960 ms | not_measured | managed | 2.028x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 9.7425 ms | not_measured | managed | 1.119x |
| `np.intersect1d(a, b) (int32)` | int32 | 10,000,000 | 43.9221 ms | not_measured | managed | 4.745x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.02178 ms | not_measured | managed | 2.052x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 0.9557 ms | not_measured | managed | 0.746x |
| `np.isin(a, b) (int32)` | int32 | 10,000,000 | 9.6958 ms | not_measured | managed | 1.219x |
| `np.ix_` | float64 | 1,000 | 0.00106 ms | not_measured | managed | 3.460x |
| `np.ix_` | float64 | 100,000 | 0.000799 ms | not_measured | managed | 4.700x |
| `np.ix_` | float64 | 10,000,000 | 0.000743 ms | not_measured | managed | 4.997x |
| `np.matrix_transpose` | float64 | 1,000 | 0.000229 ms | not_measured | managed | 3.511x |
| `np.matrix_transpose` | float64 | 100,000 | 0.000214 ms | not_measured | managed | 3.729x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.000303 ms | not_measured | managed | 2.644x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.000912 ms | not_measured | managed | 4.361x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.000853 ms | not_measured | managed | 4.620x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.000663 ms | not_measured | managed | 6.012x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.00669 ms | not_measured | managed | 3.554x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.04693 ms | not_measured | managed | 0.927x |
| `np.pad(a, width) (float64)` | float64 | 10,000,000 | 0.6250 ms | not_measured | managed | 2.003x |
| `np.permute_dims` | float64 | 1,000 | 0.000334 ms | not_measured | managed | 1.428x |
| `np.permute_dims` | float64 | 100,000 | 0.000295 ms | not_measured | managed | 1.603x |
| `np.permute_dims` | float64 | 10,000,000 | 0.000310 ms | not_measured | managed | 1.500x |
| `np.r_` | float64 | 1,000 | 0.00214 ms | not_measured | managed | 2.792x |
| `np.r_` | float64 | 100,000 | 0.07489 ms | not_measured | managed | 5.095x |
| `np.r_` | float64 | 10,000,000 | 28.1185 ms | not_measured | managed | 1.157x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.00283 ms | not_measured | managed | 6.275x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.1391 ms | not_measured | managed | 3.305x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 23.7767 ms | not_measured | managed | 2.103x |
| `np.r_(0:N)` | float64 | 1,000 | 0.00273 ms | not_measured | managed | 2.182x |
| `np.r_(0:N)` | float64 | 100,000 | 0.09495 ms | not_measured | managed | 4.167x |
| `np.r_(0:N)` | float64 | 10,000,000 | 25.8535 ms | not_measured | managed | 1.243x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.00383 ms | not_measured | managed | 2.240x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.09107 ms | not_measured | managed | 4.194x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 27.6757 ms | not_measured | managed | 1.156x |
| `np.ravel` | float64 | 1,000 | 0.000354 ms | not_measured | managed | 0.864x |
| `np.ravel` | float64 | 100,000 | 0.000284 ms | not_measured | managed | 1.447x |
| `np.ravel` | float64 | 10,000,000 | 0.000545 ms | not_measured | managed | 0.749x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.00261 ms | not_measured | managed | 3.916x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.3149 ms | not_measured | managed | 3.025x |
| `np.repeat(a, repeats) (float64)` | float64 | 10,000,000 | 2.2971 ms | not_measured | managed | 4.234x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.000544 ms | not_measured | managed | 0.989x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.000500 ms | not_measured | managed | 2.656x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.000489 ms | not_measured | managed | 2.712x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.00210 ms | not_measured | managed | 2.824x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.1011 ms | not_measured | managed | 3.483x |
| `np.resize(a, shape) (float64)` | float64 | 10,000,000 | 1.9232 ms | not_measured | managed | 1.262x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.00490 ms | not_measured | managed | 2.314x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.04496 ms | not_measured | managed | 0.723x |
| `np.roll(a, shift) (float64)` | float64 | 10,000,000 | 0.5770 ms | not_measured | managed | 2.172x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.000419 ms | not_measured | managed | 2.482x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.000364 ms | not_measured | managed | 2.780x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.000222 ms | not_measured | managed | 4.689x |
| `np.rot90` | float64 | 1,000 | 0.000535 ms | not_measured | managed | 16.095x |
| `np.rot90` | float64 | 100,000 | 0.000583 ms | not_measured | managed | 14.832x |
| `np.rot90` | float64 | 10,000,000 | 0.000531 ms | not_measured | managed | 16.375x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.03988 ms | not_measured | managed | 2.817x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 9.4613 ms | not_measured | managed | 1.149x |
| `np.setdiff1d(a, b) (int32)` | int32 | 10,000,000 | 40.2557 ms | not_measured | managed | 5.279x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.04853 ms | not_measured | managed | 1.712x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 9.8504 ms | not_measured | managed | 1.102x |
| `np.setxor1d(a, b) (int32)` | int32 | 10,000,000 | 35.0097 ms | not_measured | managed | 5.806x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.00168 ms | not_measured | managed | 12.206x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.00164 ms | not_measured | managed | 12.660x |
| `np.split(a, sections) (float64)` | float64 | 10,000,000 | 0.00272 ms | not_measured | managed | 7.695x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.000230 ms | not_measured | managed | 1.222x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.000215 ms | not_measured | managed | 2.158x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.000227 ms | not_measured | managed | 2.062x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.000168 ms | not_measured | managed | 1.821x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.000200 ms | not_measured | managed | 2.945x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.000234 ms | not_measured | managed | 2.487x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.00302 ms | not_measured | managed | 1.773x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.1021 ms | not_measured | managed | 3.093x |
| `np.stack([a, b]) (float64)` | float64 | 10,000,000 | 1.3255 ms | not_measured | managed | 1.943x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00371 ms | not_measured | managed | 1.633x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.1879 ms | not_measured | managed | 2.213x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 10,000,000 | 2.8676 ms | not_measured | managed | 1.859x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.000295 ms | not_measured | managed | 1.756x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.000281 ms | not_measured | managed | 1.858x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.000283 ms | not_measured | managed | 1.827x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.000302 ms | not_measured | managed | 1.748x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.000451 ms | not_measured | managed | 1.180x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.000268 ms | not_measured | managed | 1.996x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.00401 ms | not_measured | managed | 1.590x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.07426 ms | not_measured | managed | 4.911x |
| `np.tile(a, reps) (float64)` | float64 | 10,000,000 | 1.8517 ms | not_measured | managed | 1.268x |
| `np.transpose(a)` | float64 | 1,000 | 0.000312 ms | not_measured | managed | 1.035x |
| `np.transpose(a)` | float64 | 100,000 | 0.000369 ms | not_measured | managed | 1.268x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.000373 ms | not_measured | managed | 1.276x |
| `np.tri` | float64 | 1,000 | 0.00171 ms | not_measured | managed | 4.842x |
| `np.tri` | float64 | 100,000 | 0.02545 ms | not_measured | managed | 3.384x |
| `np.tri` | float64 | 10,000,000 | 17.5280 ms | not_measured | managed | 1.023x |
| `np.tril` | float64 | 1,000 | 0.00141 ms | not_measured | managed | 7.650x |
| `np.tril` | float64 | 100,000 | 0.03467 ms | not_measured | managed | 2.760x |
| `np.tril` | float64 | 10,000,000 | 19.9772 ms | not_measured | managed | 1.129x |
| `np.tril_indices` | float64 | 1,000 | 0.00313 ms | not_measured | managed | 9.431x |
| `np.tril_indices` | float64 | 100,000 | 0.02884 ms | not_measured | managed | 5.058x |
| `np.tril_indices` | float64 | 10,000,000 | 11.4232 ms | not_measured | managed | 2.173x |
| `np.trim_zeros` | float64 | 1,000 | 0.00414 ms | not_measured | managed | 6.872x |
| `np.trim_zeros` | float64 | 100,000 | 0.01419 ms | not_measured | managed | 95.094x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.03229 ms | not_measured | managed | 5740.969x |
| `np.triu` | float64 | 1,000 | 0.000878 ms | not_measured | managed | 12.236x |
| `np.triu` | float64 | 100,000 | 0.04153 ms | not_measured | managed | 2.302x |
| `np.triu` | float64 | 10,000,000 | 19.8933 ms | not_measured | managed | 1.154x |
| `np.triu_indices` | float64 | 1,000 | 0.00350 ms | not_measured | managed | 8.684x |
| `np.triu_indices` | float64 | 100,000 | 0.04263 ms | not_measured | managed | 3.409x |
| `np.triu_indices` | float64 | 10,000,000 | 11.3976 ms | not_measured | managed | 2.331x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.01588 ms | not_measured | managed | 2.817x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 5.0695 ms | not_measured | managed | 1.380x |
| `np.union1d(a, b) (int32)` | int32 | 10,000,000 | 30.3665 ms | not_measured | managed | 5.335x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.00923 ms | not_measured | managed | 3.640x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.4608 ms | not_measured | managed | 0.722x |
| `np.unique(a) (int32)` | int32 | 10,000,000 | 16.5467 ms | not_measured | managed | 5.244x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.02110 ms | not_measured | managed | 3.199x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 3.3166 ms | not_measured | managed | 2.724x |
| `np.unique_all(a) (int32)` | int32 | 10,000,000 | 36.7695 ms | not_measured | managed | 3.729x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.00844 ms | not_measured | managed | 2.913x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 4.8499 ms | not_measured | managed | 0.232x |
| `np.unique_counts(a) (int32)` | int32 | 10,000,000 | 14.4817 ms | not_measured | managed | 1.038x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.01397 ms | not_measured | managed | 4.208x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 2.6334 ms | not_measured | managed | 2.077x |
| `np.unique_inverse(a) (int32)` | int32 | 10,000,000 | 29.9619 ms | not_measured | managed | 2.215x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.00940 ms | not_measured | managed | 3.405x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 3.4028 ms | not_measured | managed | 1.328x |
| `np.unique_values(a) (int32)` | int32 | 10,000,000 | 18.6665 ms | not_measured | managed | 3.714x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.00168 ms | not_measured | managed | 3.796x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.00183 ms | not_measured | managed | 3.489x |
| `np.unstack(a) (float64)` | float64 | 10,000,000 | 0.00130 ms | not_measured | managed | 4.937x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.00171 ms | not_measured | managed | 12.361x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.00183 ms | not_measured | managed | 11.635x |
| `np.vsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00169 ms | not_measured | managed | 12.566x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.00305 ms | not_measured | managed | 1.544x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.09494 ms | not_measured | managed | 4.013x |
| `np.vstack([a, b]) (float64)` | float64 | 10,000,000 | 1.3291 ms | not_measured | managed | 1.822x |
| `reshape 1D->2D` | float64 | 1,000 | 0.000487 ms | not_measured | managed | 0.302x |
| `reshape 1D->2D` | float64 | 100,000 | 0.000521 ms | not_measured | managed | 0.390x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.000501 ms | not_measured | managed | 0.407x |
| `reshape 1D->3D` | float64 | 1,000 | 0.000676 ms | not_measured | managed | 0.232x |
| `reshape 1D->3D` | float64 | 100,000 | 0.000465 ms | not_measured | managed | 0.523x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.000372 ms | not_measured | managed | 0.651x |
| `reshape 2D->1D` | float64 | 1,000 | 0.000535 ms | not_measured | managed | 0.262x |
| `reshape 2D->1D` | float64 | 100,000 | 0.000327 ms | not_measured | managed | 0.578x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.000531 ms | not_measured | managed | 0.358x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.000734 ms | not_measured | managed | 1.698x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.00755 ms | not_measured | managed | 10.590x |
| `a.all() [ndarray] (float64)` | float64 | 10,000,000 | 0.2148 ms | not_measured | managed | 3.660x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.000842 ms | not_measured | managed | 1.473x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.00117 ms | not_measured | managed | 66.028x |
| `a.any() [ndarray] (float64)` | float64 | 10,000,000 | 0.00106 ms | not_measured | managed | 709.661x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.000754 ms | not_measured | managed | 0.512x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.01909 ms | not_measured | managed | 3.350x |
| `a.argmax() [ndarray] (float64)` | float64 | 10,000,000 | 0.4254 ms | not_measured | managed | 1.499x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.000967 ms | not_measured | managed | 0.393x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.04897 ms | not_measured | managed | 1.306x |
| `a.argmin() [ndarray] (float64)` | float64 | 10,000,000 | 0.2062 ms | not_measured | managed | 3.091x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.00709 ms | not_measured | managed | 1.712x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.5488 ms | not_measured | managed | 0.606x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 10,000,000 | 8.7297 ms | not_measured | managed | 0.435x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.02443 ms | not_measured | managed | 1.651x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 3.2725 ms | not_measured | managed | 1.530x |
| `a.argsort() [ndarray] (float64)` | float64 | 10,000,000 | 29.1289 ms | not_measured | managed | 2.059x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.00290 ms | not_measured | managed | 1.514x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.1307 ms | not_measured | managed | 8.694x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 10,000,000 | 3.6988 ms | not_measured | managed | 3.029x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.000809 ms | not_measured | managed | 1.232x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.07724 ms | not_measured | managed | 0.312x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 10,000,000 | 1.1076 ms | not_measured | managed | 1.515x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.000804 ms | not_measured | managed | 1.735x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.06964 ms | not_measured | managed | 3.414x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 10,000,000 | 0.6986 ms | not_measured | managed | 4.771x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 60.000x |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 87.000x |
| `a.conj() [ndarray] (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 87.000x |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 60.000x |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 86.000x |
| `a.conjugate() [ndarray] (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 86.000x |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.00188 ms | not_measured | managed | 1.488x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.1751 ms | not_measured | managed | 1.622x |
| `a.cumprod() [ndarray] (float64)` | float64 | 10,000,000 | 1.6144 ms | not_measured | managed | 2.298x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.00260 ms | not_measured | managed | 0.819x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.1432 ms | not_measured | managed | 1.973x |
| `a.cumsum() [ndarray] (float64)` | float64 | 10,000,000 | 1.4411 ms | not_measured | managed | 2.569x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.000146 ms | not_measured | managed | 0.705x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.000213 ms | not_measured | managed | 0.709x |
| `a.diagonal() [ndarray] (float64)` | float64 | 10,000,000 | 0.000194 ms | not_measured | managed | 0.778x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.000394 ms | 0.000600 ms | managed | 2.698x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.01701 ms | 0.01220 ms | openblas | 1.283x |
| `a.dot(b) [ndarray] (float64)` | float64 | 10,000,000 | 0.3974 ms | 8.3942 ms | managed | 0.838x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.000085 ms | not_measured | managed | 4.282x |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.02046 ms | not_measured | managed | 1.017x |
| `a.fill(value) [ndarray] (float64)` | float64 | 10,000,000 | 0.2503 ms | not_measured | managed | 0.999x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.000380 ms | not_measured | managed | 0.518x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.000303 ms | not_measured | managed | 0.498x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 10,000,000 | 0.000283 ms | not_measured | managed | 0.696x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.000006 ms | not_measured | managed | 20.667x |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.000007 ms | not_measured | managed | 17.714x |
| `a.item(index) [ndarray] (float64)` | float64 | 10,000,000 | 0.000007 ms | not_measured | managed | 17.714x |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.000542 ms | not_measured | managed | 1.659x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.01662 ms | not_measured | managed | 2.252x |
| `a.max() [ndarray] (float64)` | float64 | 10,000,000 | 0.2164 ms | not_measured | managed | 1.661x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.000431 ms | not_measured | managed | 2.093x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.01706 ms | not_measured | managed | 2.188x |
| `a.min() [ndarray] (float64)` | float64 | 10,000,000 | 0.2026 ms | not_measured | managed | 1.773x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.00218 ms | not_measured | managed | 1.564x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.1028 ms | not_measured | managed | 2.818x |
| `a.nonzero() [ndarray] (float64)` | float64 | 10,000,000 | 1.3746 ms | not_measured | managed | 2.683x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.01039 ms | not_measured | managed | 0.652x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.6156 ms | not_measured | managed | 0.622x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 10,000,000 | 7.3366 ms | not_measured | managed | 0.671x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.000387 ms | not_measured | managed | 5.305x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.00794 ms | not_measured | managed | 12.276x |
| `a.prod() [ndarray] (float64)` | float64 | 10,000,000 | 0.1748 ms | not_measured | managed | 5.398x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.00198 ms | not_measured | managed | 0.708x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.09932 ms | not_measured | managed | 1.866x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 10,000,000 | 1.3055 ms | not_measured | managed | 2.042x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.00331 ms | not_measured | managed | 0.674x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.2632 ms | not_measured | managed | 3.592x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 10,000,000 | 3.0874 ms | not_measured | managed | 3.113x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.000322 ms | not_measured | managed | 0.661x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.000284 ms | not_measured | managed | 0.785x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 10,000,000 | 0.000516 ms | not_measured | managed | 0.417x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.00203 ms | not_measured | managed | 0.321x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.06454 ms | not_measured | managed | 0.360x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 10,000,000 | 1.5505 ms | not_measured | managed | 0.778x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.00103 ms | not_measured | managed | 1.270x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.02942 ms | not_measured | managed | 0.743x |
| `a.round() [ndarray] (float64)` | float64 | 10,000,000 | 0.9590 ms | not_measured | managed | 1.729x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.01129 ms | not_measured | managed | 1.026x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 2.7768 ms | not_measured | managed | 0.919x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 10,000,000 | 29.9858 ms | not_measured | managed | 0.947x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.00213 ms | not_measured | managed | 0.176x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.05097 ms | not_measured | managed | 0.429x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 10,000,000 | 0.5329 ms | not_measured | managed | 0.472x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.000016 ms | not_measured | managed | 8.375x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.000014 ms | not_measured | managed | 13.429x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 10,000,000 | 0.000012 ms | not_measured | managed | 15.833x |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.02082 ms | not_measured | managed | 0.405x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 1.7291 ms | not_measured | managed | 0.682x |
| `a.sort() [ndarray] (float64)` | float64 | 10,000,000 | 19.8318 ms | not_measured | managed | 0.805x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.000170 ms | not_measured | managed | 1.112x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.000201 ms | not_measured | managed | 0.945x |
| `a.squeeze() [ndarray] (float64)` | float64 | 10,000,000 | 0.000181 ms | not_measured | managed | 1.044x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.000317 ms | not_measured | managed | 0.603x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.000378 ms | not_measured | managed | 0.508x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 10,000,000 | 0.000313 ms | not_measured | managed | 0.617x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.000825 ms | not_measured | managed | 3.078x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.04085 ms | not_measured | managed | 4.424x |
| `a.take(indices) [ndarray] (float64)` | float64 | 10,000,000 | 1.2936 ms | not_measured | managed | 1.741x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.000381 ms | not_measured | managed | 0.717x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.07586 ms | not_measured | managed | 0.285x |
| `a.tobytes() [ndarray] (float64)` | float64 | 10,000,000 | 0.6371 ms | not_measured | managed | 1.901x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.1353 ms | not_measured | managed | 1.335x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 1.6425 ms | not_measured | managed | 0.287x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 10,000,000 | 3.1015 ms | not_measured | managed | 0.443x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.00565 ms | not_measured | managed | 1.366x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 1.6028 ms | not_measured | managed | 0.882x |
| `a.tolist() [ndarray] (float64)` | float64 | 10,000,000 | 34.3612 ms | not_measured | managed | 0.470x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.000484 ms | not_measured | managed | 5.393x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.000482 ms | not_measured | managed | 5.421x |
| `a.trace() [ndarray] (float64)` | float64 | 10,000,000 | 0.000368 ms | not_measured | managed | 7.128x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.000294 ms | not_measured | managed | 0.493x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.000301 ms | not_measured | managed | 0.478x |
| `a.transpose() [ndarray] (float64)` | float64 | 10,000,000 | 0.000224 ms | not_measured | managed | 0.638x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.000184 ms | not_measured | managed | 0.663x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.000202 ms | not_measured | managed | 0.584x |
| `a.view() [ndarray] (float64)` | float64 | 10,000,000 | 0.000200 ms | not_measured | managed | 0.590x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.04100 ms | not_measured | managed | 0.973x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 7.9524 ms | not_measured | managed | 0.895x |
| `np.random.beta(a, b) (float64)` | float64 | 10,000,000 | 66.1312 ms | not_measured | managed | 0.683x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.1010 ms | not_measured | managed | 0.336x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 13.0397 ms | not_measured | managed | 0.251x |
| `np.random.binomial(n, p) (int64)` | int64 | 10,000,000 | 137.5616 ms | not_measured | managed | 0.241x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.02167 ms | not_measured | managed | 0.938x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 4.1596 ms | not_measured | managed | 0.849x |
| `np.random.chisquare(df) (float64)` | float64 | 10,000,000 | 28.5676 ms | not_measured | managed | 0.734x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.00652 ms | not_measured | managed | 3.837x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 1.4315 ms | not_measured | managed | 0.781x |
| `np.random.choice(a, size) (float64)` | float64 | 10,000,000 | 7.2102 ms | not_measured | managed | 1.147x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.05420 ms | not_measured | managed | 1.747x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 5.5788 ms | not_measured | managed | 1.637x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 10,000,000 | 54.8261 ms | not_measured | managed | 1.665x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.02005 ms | not_measured | managed | 0.440x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 0.9713 ms | not_measured | managed | 1.355x |
| `np.random.exponential(scale) (float64)` | float64 | 10,000,000 | 9.8346 ms | not_measured | managed | 1.444x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.04426 ms | not_measured | managed | 0.905x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 8.1500 ms | not_measured | managed | 0.874x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 10,000,000 | 83.8360 ms | not_measured | managed | 0.878x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.02144 ms | not_measured | managed | 0.956x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 2.1097 ms | not_measured | managed | 1.669x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 10,000,000 | 38.6304 ms | not_measured | managed | 0.673x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.02155 ms | not_measured | managed | 1.184x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 2.1810 ms | not_measured | managed | 1.053x |
| `np.random.geometric(p) (int64)` | int64 | 10,000,000 | 21.9392 ms | not_measured | managed | 1.021x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.02832 ms | not_measured | managed | 0.503x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 1.5226 ms | not_measured | managed | 1.619x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 10,000,000 | 19.6637 ms | not_measured | managed | 1.319x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1520 ms | not_measured | managed | 0.632x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 15.1865 ms | not_measured | managed | 0.616x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 10,000,000 | 154.0227 ms | not_measured | managed | 0.616x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.01629 ms | not_measured | managed | 0.848x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 1.5452 ms | not_measured | managed | 1.221x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 10,000,000 | 24.0706 ms | not_measured | managed | 0.826x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.01105 ms | not_measured | managed | 0.833x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 1.1038 ms | not_measured | managed | 1.290x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 10,000,000 | 10.7332 ms | not_measured | managed | 1.422x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.01439 ms | not_measured | managed | 1.247x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 1.4761 ms | not_measured | managed | 1.866x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 10,000,000 | 18.2934 ms | not_measured | managed | 1.290x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.02782 ms | not_measured | managed | 1.451x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 2.5744 ms | not_measured | managed | 1.485x |
| `np.random.logseries(p) (int64)` | int64 | 10,000,000 | 38.8331 ms | not_measured | managed | 0.986x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.2138 ms | not_measured | managed | 0.522x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 21.6308 ms | not_measured | managed | 0.519x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 10,000,000 | 216.0906 ms | not_measured | managed | 0.520x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.08507 ms | not_measured | managed | 0.709x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 4.3350 ms | not_measured | managed | 1.597x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 10,000,000 | 65.1665 ms | not_measured | managed | 0.799x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.1359 ms | not_measured | managed | 0.790x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 13.5743 ms | not_measured | managed | 0.789x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 10,000,000 | 102.3849 ms | not_measured | managed | 1.035x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.06436 ms | not_measured | managed | 0.509x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 6.4288 ms | not_measured | managed | 0.917x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 10,000,000 | 41.6304 ms | not_measured | managed | 1.482x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.09894 ms | not_measured | managed | 0.530x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 9.8464 ms | not_measured | managed | 0.967x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 10,000,000 | 98.5614 ms | not_measured | managed | 0.994x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.02123 ms | not_measured | managed | 0.531x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 1.5279 ms | not_measured | managed | 1.185x |
| `np.random.normal(loc, scale) (float64)` | float64 | 10,000,000 | 11.4726 ms | not_measured | managed | 1.686x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.02639 ms | not_measured | managed | 0.566x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 1.8444 ms | not_measured | managed | 1.272x |
| `np.random.pareto(a) (float64)` | float64 | 10,000,000 | 26.0722 ms | not_measured | managed | 0.952x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.01705 ms | not_measured | managed | 0.820x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 1.8096 ms | not_measured | managed | 1.060x |
| `np.random.permutation(a) (float64)` | float64 | 10,000,000 | 24.4646 ms | not_measured | managed | 1.061x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.05292 ms | not_measured | managed | 0.878x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 5.2639 ms | not_measured | managed | 0.835x |
| `np.random.poisson(lam) (int64)` | int64 | 10,000,000 | 52.5315 ms | not_measured | managed | 0.844x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.01487 ms | not_measured | managed | 1.918x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 1.5235 ms | not_measured | managed | 3.248x |
| `np.random.power(a) (float64)` | float64 | 10,000,000 | 26.4957 ms | not_measured | managed | 1.941x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.00606 ms | not_measured | managed | 0.669x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 1.2363 ms | not_measured | managed | 0.430x |
| `np.random.rand(n) (float64)` | float64 | 10,000,000 | 12.2685 ms | not_measured | managed | 0.496x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.00857 ms | not_measured | managed | 1.842x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.8673 ms | not_measured | managed | 0.680x |
| `np.random.randint(low, high) (int64)` | int64 | 10,000,000 | 5.7044 ms | not_measured | managed | 1.232x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.02175 ms | not_measured | managed | 0.853x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 1.2451 ms | not_measured | managed | 1.358x |
| `np.random.randn(n) (float64)` | float64 | 10,000,000 | 21.2504 ms | not_measured | managed | 0.847x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.01255 ms | not_measured | managed | 0.543x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 0.7690 ms | not_measured | managed | 0.692x |
| `np.random.random(n) (float64)` | float64 | 10,000,000 | 5.8730 ms | not_measured | managed | 1.039x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.01307 ms | not_measured | managed | 0.511x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 0.5877 ms | not_measured | managed | 0.904x |
| `np.random.random_sample(n) (float64)` | float64 | 10,000,000 | 12.2571 ms | not_measured | managed | 0.498x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.00985 ms | not_measured | managed | 1.852x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 1.5869 ms | not_measured | managed | 1.033x |
| `np.random.rayleigh(scale) (float64)` | float64 | 10,000,000 | 9.9726 ms | not_measured | managed | 1.780x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.01167 ms | not_measured | managed | 1.658x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 1.0880 ms | not_measured | managed | 1.728x |
| `np.random.shuffle(a) (float64)` | float64 | 10,000,000 | 19.2075 ms | not_measured | managed | 0.708x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.01709 ms | not_measured | managed | 2.138x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 1.5031 ms | not_measured | managed | 2.371x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 10,000,000 | 25.0220 ms | not_measured | managed | 1.505x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.01849 ms | not_measured | managed | 0.762x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 1.0412 ms | not_measured | managed | 0.785x |
| `np.random.standard_exponential(n) (float64)` | float64 | 10,000,000 | 18.0175 ms | not_measured | managed | 0.754x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.02086 ms | not_measured | managed | 1.740x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 2.1343 ms | not_measured | managed | 1.621x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 10,000,000 | 20.9457 ms | not_measured | managed | 1.758x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.01257 ms | not_measured | managed | 1.457x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 1.2392 ms | not_measured | managed | 1.359x |
| `np.random.standard_normal(n) (float64)` | float64 | 10,000,000 | 21.2515 ms | not_measured | managed | 0.866x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.03286 ms | not_measured | managed | 1.785x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 6.0123 ms | not_measured | managed | 0.955x |
| `np.random.standard_t(df) (float64)` | float64 | 10,000,000 | 60.1078 ms | not_measured | managed | 0.999x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.01227 ms | not_measured | managed | 1.202x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 0.9165 ms | not_measured | managed | 1.325x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 10,000,000 | 14.1186 ms | not_measured | managed | 0.950x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.01071 ms | not_measured | managed | 0.926x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 0.9665 ms | not_measured | managed | 0.656x |
| `np.random.uniform(low, high) (float64)` | float64 | 10,000,000 | 5.1115 ms | not_measured | managed | 1.389x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.04839 ms | not_measured | managed | 1.718x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 8.4982 ms | not_measured | managed | 0.598x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 10,000,000 | 79.7199 ms | not_measured | managed | 1.065x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.04678 ms | not_measured | managed | 0.839x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 4.6890 ms | not_measured | managed | 0.801x |
| `np.random.wald(mean, scale) (float64)` | float64 | 10,000,000 | 27.6820 ms | not_measured | managed | 1.338x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.03596 ms | not_measured | managed | 1.002x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 2.1596 ms | not_measured | managed | 1.569x |
| `np.random.weibull(a) (float64)` | float64 | 10,000,000 | 35.5585 ms | not_measured | managed | 1.001x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.08103 ms | not_measured | managed | 0.754x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 4.7903 ms | not_measured | managed | 1.236x |
| `np.random.zipf(a) (int64)` | int64 | 10,000,000 | 80.8535 ms | not_measured | managed | 0.742x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.00215 ms | not_measured | managed | 1.953x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1130 ms | not_measured | managed | 1.470x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 13.8809 ms | not_measured | managed | 1.145x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.00210 ms | not_measured | managed | 2.827x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4389 ms | not_measured | managed | 1.381x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 32.5504 ms | not_measured | managed | 1.972x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.000644 ms | not_measured | managed | 4.054x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.00864 ms | not_measured | managed | 2.324x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 3.3688 ms | not_measured | managed | 0.923x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.000432 ms | not_measured | managed | 6.521x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.01679 ms | not_measured | managed | 2.254x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 7.8555 ms | not_measured | managed | 0.949x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.000540 ms | not_measured | managed | 4.546x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.00251 ms | not_measured | managed | 2.252x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.5443 ms | not_measured | managed | 0.908x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.000588 ms | not_measured | managed | 1.446x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.00409 ms | not_measured | managed | 2.200x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 1.1918 ms | not_measured | managed | 2.159x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.000597 ms | not_measured | managed | 4.628x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.02658 ms | not_measured | managed | 1.247x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 8.0184 ms | not_measured | managed | 0.997x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.000364 ms | not_measured | managed | 6.832x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.00160 ms | not_measured | managed | 2.643x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.1638 ms | not_measured | managed | 1.477x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.000699 ms | not_measured | managed | 3.498x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.00251 ms | not_measured | managed | 2.236x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 0.3282 ms | not_measured | managed | 1.499x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.000609 ms | not_measured | managed | 4.094x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.00326 ms | not_measured | managed | 2.847x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 1.1354 ms | not_measured | managed | 2.406x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.000458 ms | not_measured | managed | 6.238x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.03712 ms | not_measured | managed | 1.171x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 7.1076 ms | not_measured | managed | 1.187x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.000554 ms | not_measured | managed | 1.487x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.00161 ms | not_measured | managed | 2.588x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.1458 ms | not_measured | managed | 1.564x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.00216 ms | not_measured | managed | 1.912x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1430 ms | not_measured | managed | 1.161x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 16.2822 ms | not_measured | managed | 1.143x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.00178 ms | not_measured | managed | 3.189x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.4031 ms | not_measured | managed | 1.482x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 38.8458 ms | not_measured | managed | 1.339x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.000618 ms | not_measured | managed | 4.201x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.00717 ms | not_measured | managed | 2.803x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 1.3697 ms | not_measured | managed | 2.293x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.000686 ms | not_measured | managed | 4.058x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.01718 ms | not_measured | managed | 2.205x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 7.9229 ms | not_measured | managed | 0.985x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.000785 ms | not_measured | managed | 3.163x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.00253 ms | not_measured | managed | 2.235x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 0.3251 ms | not_measured | managed | 1.527x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.000401 ms | not_measured | managed | 6.209x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.00415 ms | not_measured | managed | 2.229x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 2.7045 ms | not_measured | managed | 0.946x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.000863 ms | not_measured | managed | 3.189x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.02717 ms | not_measured | managed | 1.211x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 8.2472 ms | not_measured | managed | 1.101x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.000451 ms | not_measured | managed | 5.448x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.00156 ms | not_measured | managed | 2.669x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.2270 ms | not_measured | managed | 1.055x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.000597 ms | not_measured | managed | 4.106x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.00243 ms | not_measured | managed | 2.373x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 0.3404 ms | not_measured | managed | 1.458x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.000437 ms | not_measured | managed | 5.689x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.00367 ms | not_measured | managed | 2.439x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 2.9033 ms | not_measured | managed | 0.896x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.000963 ms | not_measured | managed | 2.962x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.03648 ms | not_measured | managed | 1.194x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 7.7218 ms | not_measured | managed | 1.041x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.000604 ms | not_measured | managed | 1.349x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.000977 ms | not_measured | managed | 4.263x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.2250 ms | not_measured | managed | 1.011x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.000501 ms | not_measured | managed | 13.948x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.01207 ms | not_measured | managed | 3.866x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 16.0023 ms | not_measured | managed | 1.088x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.00124 ms | not_measured | managed | 9.837x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.1643 ms | not_measured | managed | 1.033x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 14.6068 ms | not_measured | managed | 1.139x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.000622 ms | not_measured | managed | 15.429x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.00313 ms | not_measured | managed | 9.890x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 1.2963 ms | not_measured | managed | 6.370x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.000635 ms | not_measured | managed | 9.704x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.00602 ms | not_measured | managed | 4.784x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 2.9052 ms | not_measured | managed | 4.868x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.000387 ms | not_measured | managed | 20.331x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.04398 ms | not_measured | managed | 1.938x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 4.4380 ms | not_measured | managed | 1.774x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.000385 ms | not_measured | managed | 20.418x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.04342 ms | not_measured | managed | 0.866x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 2.8577 ms | not_measured | managed | 2.863x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.000370 ms | not_measured | managed | 21.211x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.00627 ms | not_measured | managed | 13.110x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 6.5796 ms | not_measured | managed | 1.623x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.000408 ms | not_measured | managed | 19.118x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.01857 ms | not_measured | managed | 4.554x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 1.8440 ms | not_measured | managed | 4.341x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.000544 ms | not_measured | managed | 14.362x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.03905 ms | not_measured | managed | 2.184x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 4.1913 ms | not_measured | managed | 1.907x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.000993 ms | not_measured | managed | 7.894x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.01900 ms | not_measured | managed | 4.075x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 5.0240 ms | not_measured | managed | 1.520x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.000400 ms | not_measured | managed | 20.172x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.00963 ms | not_measured | managed | 10.406x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 6.9513 ms | not_measured | managed | 1.689x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.000575 ms | not_measured | managed | 4.247x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.03988 ms | not_measured | managed | 2.119x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 1.8492 ms | not_measured | managed | 4.272x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.00426 ms | not_measured | managed | 8.893x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.1794 ms | not_measured | managed | 8.491x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 17.8496 ms | not_measured | managed | 8.850x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.000561 ms | not_measured | managed | 41.007x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.01815 ms | not_measured | managed | 4.732x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 2.8443 ms | not_measured | managed | 10.614x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.000956 ms | not_measured | managed | 19.785x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.03385 ms | not_measured | managed | 3.135x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 14.6520 ms | not_measured | managed | 3.030x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.000479 ms | not_measured | managed | 5.762x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.01086 ms | not_measured | managed | 3.931x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 6.4543 ms | not_measured | managed | 2.495x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.000989 ms | not_measured | managed | 4.633x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.06717 ms | not_measured | managed | 2.838x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 6.8837 ms | not_measured | managed | 2.790x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.000592 ms | not_measured | managed | 4.191x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.00572 ms | not_measured | managed | 4.158x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 2.7596 ms | not_measured | managed | 3.358x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.000622 ms | not_measured | managed | 3.986x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.00771 ms | not_measured | managed | 3.256x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 2.7658 ms | not_measured | managed | 5.243x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.000456 ms | not_measured | managed | 7.344x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.02551 ms | not_measured | managed | 2.573x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 4.4370 ms | not_measured | managed | 1.436x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.000794 ms | not_measured | managed | 4.451x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.04464 ms | not_measured | managed | 1.826x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 5.1351 ms | not_measured | managed | 1.679x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.000406 ms | not_measured | managed | 6.352x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.00692 ms | not_measured | managed | 4.173x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 7.3965 ms | not_measured | managed | 1.198x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.000804 ms | not_measured | managed | 4.338x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.04480 ms | not_measured | managed | 1.775x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 4.6279 ms | not_measured | managed | 1.661x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.000561 ms | not_measured | managed | 6.257x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.03916 ms | not_measured | managed | 2.048x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 4.1858 ms | not_measured | managed | 1.900x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.00100 ms | not_measured | managed | 1.163x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.03938 ms | not_measured | managed | 1.673x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 4.8706 ms | not_measured | managed | 0.772x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.000532 ms | not_measured | managed | 4.868x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.00777 ms | not_measured | managed | 3.704x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 2.7811 ms | not_measured | managed | 1.308x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.000996 ms | not_measured | managed | 1.248x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.04055 ms | not_measured | managed | 1.959x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 3.4633 ms | not_measured | managed | 2.240x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.00203 ms | not_measured | managed | 17.902x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.3862 ms | not_measured | managed | 3.991x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 27.6389 ms | not_measured | managed | 5.344x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.000808 ms | not_measured | managed | 26.797x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.01351 ms | not_measured | managed | 6.296x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 6.4004 ms | not_measured | managed | 2.379x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.000745 ms | not_measured | managed | 24.149x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.03577 ms | not_measured | managed | 2.934x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 15.5878 ms | not_measured | managed | 2.056x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.00213 ms | not_measured | managed | 1.409x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1424 ms | not_measured | managed | 1.179x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 18.9426 ms | not_measured | managed | 1.081x |
| `np.amax (float16)` | float16 | 1,000 | 0.00208 ms | not_measured | managed | 3.652x |
| `np.amax (float16)` | float16 | 100,000 | 0.4387 ms | not_measured | managed | 1.428x |
| `np.amax (float16)` | float16 | 10,000,000 | 46.4466 ms | not_measured | managed | 1.085x |
| `np.amax (float32)` | float32 | 1,000 | 0.000469 ms | not_measured | managed | 9.049x |
| `np.amax (float32)` | float32 | 100,000 | 0.00863 ms | not_measured | managed | 2.523x |
| `np.amax (float32)` | float32 | 10,000,000 | 3.3783 ms | not_measured | managed | 0.918x |
| `np.amax (float64)` | float64 | 1,000 | 0.000390 ms | not_measured | managed | 11.336x |
| `np.amax (float64)` | float64 | 100,000 | 0.00658 ms | not_measured | managed | 5.997x |
| `np.amax (float64)` | float64 | 10,000,000 | 7.8927 ms | not_measured | managed | 0.976x |
| `np.amax (int16)` | int16 | 1,000 | 0.000575 ms | not_measured | managed | 7.049x |
| `np.amax (int16)` | int16 | 100,000 | 0.00252 ms | not_measured | managed | 2.901x |
| `np.amax (int16)` | int16 | 10,000,000 | 0.4052 ms | not_measured | managed | 1.237x |
| `np.amax (int32)` | int32 | 1,000 | 0.000591 ms | not_measured | managed | 2.535x |
| `np.amax (int32)` | int32 | 100,000 | 0.00269 ms | not_measured | managed | 3.951x |
| `np.amax (int32)` | int32 | 10,000,000 | 2.7300 ms | not_measured | managed | 0.951x |
| `np.amax (int64)` | int64 | 1,000 | 0.000862 ms | not_measured | managed | 5.103x |
| `np.amax (int64)` | int64 | 100,000 | 0.00780 ms | not_measured | managed | 4.464x |
| `np.amax (int64)` | int64 | 10,000,000 | 8.0198 ms | not_measured | managed | 0.995x |
| `np.amax (int8)` | int8 | 1,000 | 0.000581 ms | not_measured | managed | 6.924x |
| `np.amax (int8)` | int8 | 100,000 | 0.00147 ms | not_measured | managed | 3.958x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2231 ms | not_measured | managed | 1.100x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.000574 ms | not_measured | managed | 7.044x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.00245 ms | not_measured | managed | 2.986x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 0.5159 ms | not_measured | managed | 0.974x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.000367 ms | not_measured | managed | 11.185x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.00481 ms | not_measured | managed | 2.272x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 2.7462 ms | not_measured | managed | 0.963x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.000949 ms | not_measured | managed | 4.715x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.03657 ms | not_measured | managed | 1.234x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 7.9748 ms | not_measured | managed | 1.019x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.000759 ms | not_measured | managed | 1.947x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.00159 ms | not_measured | managed | 3.665x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.1959 ms | not_measured | managed | 1.232x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.00239 ms | not_measured | managed | 2.484x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1825 ms | not_measured | managed | 0.808x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 15.1289 ms | not_measured | managed | 1.263x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.00195 ms | not_measured | managed | 3.846x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6561 ms | not_measured | managed | 0.939x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 122.2388 ms | not_measured | managed | 0.402x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.000737 ms | not_measured | managed | 6.433x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.02804 ms | not_measured | managed | 0.874x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 3.9537 ms | not_measured | managed | 0.851x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.000594 ms | not_measured | managed | 8.327x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.05513 ms | not_measured | managed | 0.702x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 8.3698 ms | not_measured | managed | 1.016x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.000610 ms | not_measured | managed | 7.431x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.00749 ms | not_measured | managed | 1.851x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 0.7047 ms | not_measured | managed | 0.959x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.000682 ms | not_measured | managed | 6.872x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.00528 ms | not_measured | managed | 3.085x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 2.1201 ms | not_measured | managed | 1.521x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.000784 ms | not_measured | managed | 6.193x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.02792 ms | not_measured | managed | 1.285x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 7.9371 ms | not_measured | managed | 1.079x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.000660 ms | not_measured | managed | 6.921x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.00760 ms | not_measured | managed | 1.895x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3319 ms | not_measured | managed | 1.051x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.000411 ms | not_measured | managed | 11.187x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.00752 ms | not_measured | managed | 1.847x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 0.7678 ms | not_measured | managed | 0.896x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.000544 ms | not_measured | managed | 8.540x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.01018 ms | not_measured | managed | 1.585x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 3.4149 ms | not_measured | managed | 0.916x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.000616 ms | not_measured | managed | 7.971x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.01507 ms | not_measured | managed | 2.748x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 4.6017 ms | not_measured | managed | 1.766x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.000669 ms | not_measured | managed | 6.779x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.00691 ms | not_measured | managed | 2.080x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.2199 ms | not_measured | managed | 1.580x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.00215 ms | not_measured | managed | 2.706x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1131 ms | not_measured | managed | 1.484x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 18.7325 ms | not_measured | managed | 1.086x |
| `np.amin (float16)` | float16 | 1,000 | 0.00145 ms | not_measured | managed | 5.061x |
| `np.amin (float16)` | float16 | 100,000 | 0.3006 ms | not_measured | managed | 2.016x |
| `np.amin (float16)` | float16 | 10,000,000 | 36.9135 ms | not_measured | managed | 1.725x |
| `np.amin (float32)` | float32 | 1,000 | 0.000398 ms | not_measured | managed | 10.731x |
| `np.amin (float32)` | float32 | 100,000 | 0.00441 ms | not_measured | managed | 4.955x |
| `np.amin (float32)` | float32 | 10,000,000 | 3.3457 ms | not_measured | managed | 0.960x |
| `np.amin (float64)` | float64 | 1,000 | 0.000681 ms | not_measured | managed | 6.443x |
| `np.amin (float64)` | float64 | 100,000 | 0.01757 ms | not_measured | managed | 2.248x |
| `np.amin (float64)` | float64 | 10,000,000 | 3.3523 ms | not_measured | managed | 2.316x |
| `np.amin (int16)` | int16 | 1,000 | 0.000788 ms | not_measured | managed | 5.166x |
| `np.amin (int16)` | int16 | 100,000 | 0.00198 ms | not_measured | managed | 3.674x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.5188 ms | not_measured | managed | 0.978x |
| `np.amin (int32)` | int32 | 1,000 | 0.000564 ms | not_measured | managed | 7.257x |
| `np.amin (int32)` | int32 | 100,000 | 0.00423 ms | not_measured | managed | 0.959x |
| `np.amin (int32)` | int32 | 10,000,000 | 2.8239 ms | not_measured | managed | 0.369x |
| `np.amin (int64)` | int64 | 1,000 | 0.000571 ms | not_measured | managed | 7.697x |
| `np.amin (int64)` | int64 | 100,000 | 0.02750 ms | not_measured | managed | 1.257x |
| `np.amin (int64)` | int64 | 10,000,000 | 5.3386 ms | not_measured | managed | 1.713x |
| `np.amin (int8)` | int8 | 1,000 | 0.000552 ms | not_measured | managed | 7.310x |
| `np.amin (int8)` | int8 | 100,000 | 0.00160 ms | not_measured | managed | 3.659x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.2244 ms | not_measured | managed | 1.085x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.000758 ms | not_measured | managed | 5.294x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.00257 ms | not_measured | managed | 2.882x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 0.5297 ms | not_measured | managed | 0.957x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.000775 ms | not_measured | managed | 5.274x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.00479 ms | not_measured | managed | 2.224x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 2.9519 ms | not_measured | managed | 0.914x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.000427 ms | not_measured | managed | 10.539x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.01010 ms | not_measured | managed | 4.473x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 7.9647 ms | not_measured | managed | 1.023x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.000774 ms | not_measured | managed | 1.956x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.00164 ms | not_measured | managed | 3.528x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.2248 ms | not_measured | managed | 1.047x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.00427 ms | not_measured | managed | 1.389x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.2249 ms | not_measured | managed | 0.656x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 15.2225 ms | not_measured | managed | 1.239x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.00315 ms | not_measured | managed | 2.353x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.6575 ms | not_measured | managed | 0.921x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 139.2373 ms | not_measured | managed | 0.435x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.000723 ms | not_measured | managed | 6.584x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.02806 ms | not_measured | managed | 0.873x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 3.9491 ms | not_measured | managed | 0.831x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.00109 ms | not_measured | managed | 4.643x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.01855 ms | not_measured | managed | 2.084x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 6.1868 ms | not_measured | managed | 1.343x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.000470 ms | not_measured | managed | 9.830x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.00462 ms | not_measured | managed | 3.020x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 0.7438 ms | not_measured | managed | 0.904x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.000568 ms | not_measured | managed | 8.120x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.01025 ms | not_measured | managed | 1.578x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 3.2391 ms | not_measured | managed | 0.997x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.000781 ms | not_measured | managed | 6.225x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.01271 ms | not_measured | managed | 2.704x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 7.9789 ms | not_measured | managed | 1.037x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.000428 ms | not_measured | managed | 10.678x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.00768 ms | not_measured | managed | 1.870x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.2970 ms | not_measured | managed | 1.175x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.000482 ms | not_measured | managed | 9.575x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.00743 ms | not_measured | managed | 1.870x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 0.7339 ms | not_measured | managed | 0.941x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.000714 ms | not_measured | managed | 6.536x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.00971 ms | not_measured | managed | 1.669x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 3.4727 ms | not_measured | managed | 0.870x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.000836 ms | not_measured | managed | 5.888x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.01943 ms | not_measured | managed | 2.123x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 8.0158 ms | not_measured | managed | 1.030x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.000666 ms | not_measured | managed | 6.956x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.00768 ms | not_measured | managed | 1.871x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.3266 ms | not_measured | managed | 1.064x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.00152 ms | not_measured | managed | 2.057x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1145 ms | not_measured | managed | 0.889x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 18.6858 ms | not_measured | managed | 0.911x |
| `np.argmax (float16)` | float16 | 1,000 | 0.00276 ms | not_measured | managed | 1.797x |
| `np.argmax (float16)` | float16 | 100,000 | 0.1426 ms | not_measured | managed | 3.807x |
| `np.argmax (float16)` | float16 | 10,000,000 | 18.6102 ms | not_measured | managed | 2.961x |
| `np.argmax (float32)` | float32 | 1,000 | 0.000767 ms | not_measured | managed | 3.003x |
| `np.argmax (float32)` | float32 | 100,000 | 0.02468 ms | not_measured | managed | 1.394x |
| `np.argmax (float32)` | float32 | 10,000,000 | 3.8778 ms | not_measured | managed | 1.030x |
| `np.argmax (float64)` | float64 | 1,000 | 0.000341 ms | not_measured | managed | 7.871x |
| `np.argmax (float64)` | float64 | 100,000 | 0.04902 ms | not_measured | managed | 1.344x |
| `np.argmax (float64)` | float64 | 10,000,000 | 6.3733 ms | not_measured | managed | 1.443x |
| `np.argmax (int16)` | int16 | 1,000 | 0.000361 ms | not_measured | managed | 6.097x |
| `np.argmax (int16)` | int16 | 100,000 | 0.00191 ms | not_measured | managed | 6.544x |
| `np.argmax (int16)` | int16 | 10,000,000 | 0.5715 ms | not_measured | managed | 1.830x |
| `np.argmax (int32)` | int32 | 1,000 | 0.000783 ms | not_measured | managed | 2.794x |
| `np.argmax (int32)` | int32 | 100,000 | 0.00406 ms | not_measured | managed | 5.035x |
| `np.argmax (int32)` | int32 | 10,000,000 | 2.9775 ms | not_measured | managed | 1.079x |
| `np.argmax (int64)` | int64 | 1,000 | 0.000801 ms | not_measured | managed | 3.235x |
| `np.argmax (int64)` | int64 | 100,000 | 0.05087 ms | not_measured | managed | 1.164x |
| `np.argmax (int64)` | int64 | 10,000,000 | 8.1706 ms | not_measured | managed | 1.026x |
| `np.argmax (int8)` | int8 | 1,000 | 0.000638 ms | not_measured | managed | 3.481x |
| `np.argmax (int8)` | int8 | 100,000 | 0.00178 ms | not_measured | managed | 4.231x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.2403 ms | not_measured | managed | 2.182x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.000382 ms | not_measured | managed | 5.801x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.00187 ms | not_measured | managed | 10.332x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 0.5623 ms | not_measured | managed | 3.081x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.000504 ms | not_measured | managed | 4.653x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.00355 ms | not_measured | managed | 9.656x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 1.4036 ms | not_measured | managed | 2.793x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.000532 ms | not_measured | managed | 5.055x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.05976 ms | not_measured | managed | 1.131x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 8.5116 ms | not_measured | managed | 1.049x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.000681 ms | not_measured | managed | 1.266x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.00114 ms | not_measured | managed | 2.599x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.2388 ms | not_measured | managed | 3.632x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.00208 ms | not_measured | managed | 1.522x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1424 ms | not_measured | managed | 0.795x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 18.8410 ms | not_measured | managed | 0.931x |
| `np.argmin (float16)` | float16 | 1,000 | 0.00275 ms | not_measured | managed | 1.731x |
| `np.argmin (float16)` | float16 | 100,000 | 0.2095 ms | not_measured | managed | 2.504x |
| `np.argmin (float16)` | float16 | 10,000,000 | 21.4657 ms | not_measured | managed | 2.367x |
| `np.argmin (float32)` | float32 | 1,000 | 0.000789 ms | not_measured | managed | 2.990x |
| `np.argmin (float32)` | float32 | 100,000 | 0.02470 ms | not_measured | managed | 1.394x |
| `np.argmin (float32)` | float32 | 10,000,000 | 2.5434 ms | not_measured | managed | 1.578x |
| `np.argmin (float64)` | float64 | 1,000 | 0.000663 ms | not_measured | managed | 1.389x |
| `np.argmin (float64)` | float64 | 100,000 | 0.04906 ms | not_measured | managed | 1.341x |
| `np.argmin (float64)` | float64 | 10,000,000 | 8.7433 ms | not_measured | managed | 1.042x |
| `np.argmin (int16)` | int16 | 1,000 | 0.000633 ms | not_measured | managed | 3.449x |
| `np.argmin (int16)` | int16 | 100,000 | 0.00277 ms | not_measured | managed | 4.504x |
| `np.argmin (int16)` | int16 | 10,000,000 | 0.5832 ms | not_measured | managed | 1.790x |
| `np.argmin (int32)` | int32 | 1,000 | 0.000808 ms | not_measured | managed | 2.786x |
| `np.argmin (int32)` | int32 | 100,000 | 0.00261 ms | not_measured | managed | 7.843x |
| `np.argmin (int32)` | int32 | 10,000,000 | 2.9159 ms | not_measured | managed | 1.081x |
| `np.argmin (int64)` | int64 | 1,000 | 0.00104 ms | not_measured | managed | 2.466x |
| `np.argmin (int64)` | int64 | 100,000 | 0.04176 ms | not_measured | managed | 1.364x |
| `np.argmin (int64)` | int64 | 10,000,000 | 4.5729 ms | not_measured | managed | 1.810x |
| `np.argmin (int8)` | int8 | 1,000 | 0.000642 ms | not_measured | managed | 3.433x |
| `np.argmin (int8)` | int8 | 100,000 | 0.00115 ms | not_measured | managed | 6.511x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.2424 ms | not_measured | managed | 2.166x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.000633 ms | not_measured | managed | 3.528x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.00252 ms | not_measured | managed | 7.701x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 0.5606 ms | not_measured | managed | 3.079x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.000846 ms | not_measured | managed | 2.751x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.00552 ms | not_measured | managed | 6.212x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 2.2964 ms | not_measured | managed | 1.721x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.000920 ms | not_measured | managed | 2.920x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.05945 ms | not_measured | managed | 1.137x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 5.4267 ms | not_measured | managed | 1.604x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.000664 ms | not_measured | managed | 1.255x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.00178 ms | not_measured | managed | 6.093x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.2421 ms | not_measured | managed | 3.581x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.00959 ms | not_measured | managed | 1.433x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 0.9667 ms | not_measured | managed | 0.920x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 125.3013 ms | not_measured | managed | 0.577x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.01762 ms | not_measured | managed | 2.942x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 2.4314 ms | not_measured | managed | 2.762x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 434.2098 ms | not_measured | managed | 1.456x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.00176 ms | not_measured | managed | 3.523x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 3.9467 ms | not_measured | managed | 1.699x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 622.3045 ms | not_measured | managed | 0.595x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.00246 ms | not_measured | managed | 2.575x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.3097 ms | not_measured | managed | 1.615x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 31.2953 ms | not_measured | managed | 1.657x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.01670 ms | not_measured | managed | 0.652x |
| `np.cumsum (float16)` | float16 | 100,000 | 1.6412 ms | not_measured | managed | 0.467x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 128.7429 ms | not_measured | managed | 0.379x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.00163 ms | not_measured | managed | 3.801x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.07832 ms | not_measured | managed | 3.625x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 7.6144 ms | not_measured | managed | 2.622x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.00136 ms | not_measured | managed | 4.550x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.1235 ms | not_measured | managed | 2.298x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 13.3994 ms | not_measured | managed | 2.895x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.00174 ms | not_measured | managed | 3.222x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.1549 ms | not_measured | managed | 2.875x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 14.3945 ms | not_measured | managed | 2.831x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.00108 ms | not_measured | managed | 5.351x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1183 ms | not_measured | managed | 3.700x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 15.6218 ms | not_measured | managed | 2.812x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.000999 ms | not_measured | managed | 3.833x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1623 ms | not_measured | managed | 0.312x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 19.6917 ms | not_measured | managed | 1.214x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.00172 ms | not_measured | managed | 3.337x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.1559 ms | not_measured | managed | 2.839x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 12.1314 ms | not_measured | managed | 2.218x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.00171 ms | not_measured | managed | 3.383x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.1128 ms | not_measured | managed | 3.891x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 14.3267 ms | not_measured | managed | 1.938x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.00170 ms | not_measured | managed | 3.408x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1602 ms | not_measured | managed | 2.791x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 16.4033 ms | not_measured | managed | 2.617x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.00168 ms | not_measured | managed | 2.332x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1629 ms | not_measured | managed | 0.310x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 16.7828 ms | not_measured | managed | 1.434x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.00162 ms | not_measured | managed | 1.375x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.1544 ms | not_measured | managed | 2.873x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 12.8963 ms | not_measured | managed | 3.182x |
| `np.max (complex128)` | complex128 | 1,000 | 0.00215 ms | not_measured | managed | 2.706x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1422 ms | not_measured | managed | 1.181x |
| `np.max (complex128)` | complex128 | 10,000,000 | 17.8028 ms | not_measured | managed | 0.889x |
| `np.max (float16)` | float16 | 1,000 | 0.00173 ms | not_measured | managed | 4.407x |
| `np.max (float16)` | float16 | 100,000 | 0.4394 ms | not_measured | managed | 1.404x |
| `np.max (float16)` | float16 | 10,000,000 | 46.4132 ms | not_measured | managed | 1.305x |
| `np.max (float32)` | float32 | 1,000 | 0.000454 ms | not_measured | managed | 9.344x |
| `np.max (float32)` | float32 | 100,000 | 0.00363 ms | not_measured | managed | 5.998x |
| `np.max (float32)` | float32 | 10,000,000 | 3.3938 ms | not_measured | managed | 0.921x |
| `np.max (float64)` | float64 | 1,000 | 0.000724 ms | not_measured | managed | 2.206x |
| `np.max (float64)` | float64 | 100,000 | 0.01673 ms | not_measured | managed | 2.367x |
| `np.max (float64)` | float64 | 10,000,000 | 5.2142 ms | not_measured | managed | 1.434x |
| `np.max (int16)` | int16 | 1,000 | 0.000565 ms | not_measured | managed | 7.258x |
| `np.max (int16)` | int16 | 100,000 | 0.00222 ms | not_measured | managed | 3.293x |
| `np.max (int16)` | int16 | 10,000,000 | 0.3226 ms | not_measured | managed | 1.556x |
| `np.max (int32)` | int32 | 1,000 | 0.000626 ms | not_measured | managed | 6.626x |
| `np.max (int32)` | int32 | 100,000 | 0.00410 ms | not_measured | managed | 2.599x |
| `np.max (int32)` | int32 | 10,000,000 | 2.6982 ms | not_measured | managed | 0.972x |
| `np.max (int64)` | int64 | 1,000 | 0.000680 ms | not_measured | managed | 6.597x |
| `np.max (int64)` | int64 | 100,000 | 0.02701 ms | not_measured | managed | 1.294x |
| `np.max (int64)` | int64 | 10,000,000 | 7.9635 ms | not_measured | managed | 0.996x |
| `np.max (int8)` | int8 | 1,000 | 0.000649 ms | not_measured | managed | 6.282x |
| `np.max (int8)` | int8 | 100,000 | 0.00166 ms | not_measured | managed | 3.500x |
| `np.max (int8)` | int8 | 10,000,000 | 0.1357 ms | not_measured | managed | 1.804x |
| `np.max (uint16)` | uint16 | 1,000 | 0.000641 ms | not_measured | managed | 6.385x |
| `np.max (uint16)` | uint16 | 100,000 | 0.00253 ms | not_measured | managed | 2.888x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.3337 ms | not_measured | managed | 1.499x |
| `np.max (uint32)` | uint32 | 1,000 | 0.000611 ms | not_measured | managed | 6.704x |
| `np.max (uint32)` | uint32 | 100,000 | 0.00482 ms | not_measured | managed | 2.281x |
| `np.max (uint32)` | uint32 | 10,000,000 | 2.7073 ms | not_measured | managed | 0.986x |
| `np.max (uint64)` | uint64 | 1,000 | 0.000977 ms | not_measured | managed | 4.556x |
| `np.max (uint64)` | uint64 | 100,000 | 0.01176 ms | not_measured | managed | 3.843x |
| `np.max (uint64)` | uint64 | 10,000,000 | 7.7959 ms | not_measured | managed | 0.427x |
| `np.max (uint8)` | uint8 | 1,000 | 0.000624 ms | not_measured | managed | 2.425x |
| `np.max (uint8)` | uint8 | 100,000 | 0.00161 ms | not_measured | managed | 3.602x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.2249 ms | not_measured | managed | 0.976x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.00103 ms | not_measured | managed | 7.810x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.02102 ms | not_measured | managed | 2.234x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 15.0820 ms | not_measured | managed | 1.122x |
| `np.mean (float16)` | float16 | 1,000 | 0.00199 ms | not_measured | managed | 6.673x |
| `np.mean (float16)` | float16 | 100,000 | 0.08021 ms | not_measured | managed | 2.131x |
| `np.mean (float16)` | float16 | 10,000,000 | 16.1319 ms | not_measured | managed | 1.035x |
| `np.mean (float32)` | float32 | 1,000 | 0.000392 ms | not_measured | managed | 27.380x |
| `np.mean (float32)` | float32 | 100,000 | 0.00317 ms | not_measured | managed | 10.078x |
| `np.mean (float32)` | float32 | 10,000,000 | 2.9963 ms | not_measured | managed | 2.901x |
| `np.mean (float64)` | float64 | 1,000 | 0.000539 ms | not_measured | managed | 13.432x |
| `np.mean (float64)` | float64 | 100,000 | 0.00883 ms | not_measured | managed | 3.372x |
| `np.mean (float64)` | float64 | 10,000,000 | 4.3546 ms | not_measured | managed | 3.330x |
| `np.mean (int16)` | int16 | 1,000 | 0.000395 ms | not_measured | managed | 22.689x |
| `np.mean (int16)` | int16 | 100,000 | 0.01922 ms | not_measured | managed | 4.487x |
| `np.mean (int16)` | int16 | 10,000,000 | 1.9852 ms | not_measured | managed | 4.055x |
| `np.mean (int32)` | int32 | 1,000 | 0.000391 ms | not_measured | managed | 23.069x |
| `np.mean (int32)` | int32 | 100,000 | 0.04347 ms | not_measured | managed | 0.922x |
| `np.mean (int32)` | int32 | 10,000,000 | 2.7582 ms | not_measured | managed | 1.473x |
| `np.mean (int64)` | int64 | 1,000 | 0.000394 ms | not_measured | managed | 22.665x |
| `np.mean (int64)` | int64 | 100,000 | 0.00905 ms | not_measured | managed | 9.252x |
| `np.mean (int64)` | int64 | 10,000,000 | 6.5686 ms | not_measured | managed | 1.647x |
| `np.mean (int8)` | int8 | 1,000 | 0.000386 ms | not_measured | managed | 23.187x |
| `np.mean (int8)` | int8 | 100,000 | 0.04467 ms | not_measured | managed | 1.922x |
| `np.mean (int8)` | int8 | 10,000,000 | 1.8474 ms | not_measured | managed | 4.258x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.000543 ms | not_measured | managed | 16.549x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.01882 ms | not_measured | managed | 4.590x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 1.9715 ms | not_measured | managed | 4.035x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.000986 ms | not_measured | managed | 8.956x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.03937 ms | not_measured | managed | 1.994x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 2.8985 ms | not_measured | managed | 2.650x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.000652 ms | not_measured | managed | 14.103x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.00717 ms | not_measured | managed | 14.201x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 2.8794 ms | not_measured | managed | 4.062x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.000718 ms | not_measured | managed | 3.943x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.04064 ms | not_measured | managed | 2.106x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 3.0568 ms | not_measured | managed | 2.603x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.00310 ms | not_measured | managed | 2.932x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.02411 ms | not_measured | managed | 1.886x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 7.2161 ms | not_measured | managed | 2.390x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.00676 ms | not_measured | managed | 2.075x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.2817 ms | not_measured | managed | 0.593x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 14.3275 ms | not_measured | managed | 1.091x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.00242 ms | not_measured | managed | 4.166x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.01148 ms | not_measured | managed | 1.951x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 1.6152 ms | not_measured | managed | 1.999x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.00272 ms | not_measured | managed | 3.189x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.01663 ms | not_measured | managed | 1.674x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 3.5217 ms | not_measured | managed | 2.312x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.000695 ms | not_measured | managed | 14.984x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.00893 ms | not_measured | managed | 9.515x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 1.9882 ms | not_measured | managed | 3.969x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.000692 ms | not_measured | managed | 15.000x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.01868 ms | not_measured | managed | 4.383x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 3.6955 ms | not_measured | managed | 2.173x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.000802 ms | not_measured | managed | 13.191x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.03045 ms | not_measured | managed | 2.720x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 6.7579 ms | not_measured | managed | 1.688x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.000490 ms | not_measured | managed | 21.278x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.01457 ms | not_measured | managed | 5.827x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 0.8243 ms | not_measured | managed | 9.805x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.000698 ms | not_measured | managed | 14.749x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.01764 ms | not_measured | managed | 4.812x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 1.9801 ms | not_measured | managed | 4.105x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.000705 ms | not_measured | managed | 14.462x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.01675 ms | not_measured | managed | 4.617x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 4.1412 ms | not_measured | managed | 1.874x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.000644 ms | not_measured | managed | 16.398x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.01247 ms | not_measured | managed | 8.054x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 4.0178 ms | not_measured | managed | 2.907x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.000686 ms | not_measured | managed | 14.949x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.01656 ms | not_measured | managed | 5.102x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 1.6330 ms | not_measured | managed | 4.936x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.00273 ms | not_measured | managed | 3.331x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.03331 ms | not_measured | managed | 1.642x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 8.4602 ms | not_measured | managed | 0.864x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.00455 ms | not_measured | managed | 3.082x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.2088 ms | not_measured | managed | 0.857x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 12.5330 ms | not_measured | managed | 1.304x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.00259 ms | not_measured | managed | 3.979x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.02280 ms | not_measured | managed | 1.512x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 3.2138 ms | not_measured | managed | 2.321x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.00293 ms | not_measured | managed | 2.932x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.01479 ms | not_measured | managed | 2.369x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 5.0153 ms | not_measured | managed | 2.437x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.000569 ms | not_measured | managed | 18.132x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.01152 ms | not_measured | managed | 7.969x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 0.9692 ms | not_measured | managed | 8.566x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.000713 ms | not_measured | managed | 14.550x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.01938 ms | not_measured | managed | 4.611x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 3.8630 ms | not_measured | managed | 2.165x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.00129 ms | not_measured | managed | 8.034x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.06856 ms | not_measured | managed | 1.318x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 7.2408 ms | not_measured | managed | 1.541x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.000725 ms | not_measured | managed | 14.353x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.01935 ms | not_measured | managed | 4.787x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 0.7302 ms | not_measured | managed | 11.039x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.000735 ms | not_measured | managed | 13.984x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.00882 ms | not_measured | managed | 10.385x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 0.9824 ms | not_measured | managed | 8.319x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.000760 ms | not_measured | managed | 13.467x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.02524 ms | not_measured | managed | 3.369x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 2.4470 ms | not_measured | managed | 3.284x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.00219 ms | not_measured | managed | 4.879x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.1011 ms | not_measured | managed | 1.071x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 13.7692 ms | not_measured | managed | 0.878x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.000688 ms | not_measured | managed | 14.833x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.01935 ms | not_measured | managed | 4.785x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 0.8348 ms | not_measured | managed | 9.596x |
| `np.min (complex128)` | complex128 | 1,000 | 0.00156 ms | not_measured | managed | 3.718x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1143 ms | not_measured | managed | 1.467x |
| `np.min (complex128)` | complex128 | 10,000,000 | 18.2532 ms | not_measured | managed | 1.120x |
| `np.min (float16)` | float16 | 1,000 | 0.00164 ms | not_measured | managed | 4.469x |
| `np.min (float16)` | float16 | 100,000 | 0.3614 ms | not_measured | managed | 1.660x |
| `np.min (float16)` | float16 | 10,000,000 | 31.7059 ms | not_measured | managed | 1.975x |
| `np.min (float32)` | float32 | 1,000 | 0.000614 ms | not_measured | managed | 2.489x |
| `np.min (float32)` | float32 | 100,000 | 0.00901 ms | not_measured | managed | 2.427x |
| `np.min (float32)` | float32 | 10,000,000 | 1.6178 ms | not_measured | managed | 1.945x |
| `np.min (float64)` | float64 | 1,000 | 0.000601 ms | not_measured | managed | 7.338x |
| `np.min (float64)` | float64 | 100,000 | 0.01718 ms | not_measured | managed | 2.297x |
| `np.min (float64)` | float64 | 10,000,000 | 7.9470 ms | not_measured | managed | 0.985x |
| `np.min (int16)` | int16 | 1,000 | 0.000560 ms | not_measured | managed | 7.196x |
| `np.min (int16)` | int16 | 100,000 | 0.00254 ms | not_measured | managed | 2.885x |
| `np.min (int16)` | int16 | 10,000,000 | 0.5237 ms | not_measured | managed | 0.955x |
| `np.min (int32)` | int32 | 1,000 | 0.000592 ms | not_measured | managed | 6.836x |
| `np.min (int32)` | int32 | 100,000 | 0.00373 ms | not_measured | managed | 1.088x |
| `np.min (int32)` | int32 | 10,000,000 | 2.7782 ms | not_measured | managed | 0.960x |
| `np.min (int64)` | int64 | 1,000 | 0.000874 ms | not_measured | managed | 5.100x |
| `np.min (int64)` | int64 | 100,000 | 0.02712 ms | not_measured | managed | 1.277x |
| `np.min (int64)` | int64 | 10,000,000 | 8.1770 ms | not_measured | managed | 1.119x |
| `np.min (int8)` | int8 | 1,000 | 0.000593 ms | not_measured | managed | 6.764x |
| `np.min (int8)` | int8 | 100,000 | 0.00141 ms | not_measured | managed | 4.091x |
| `np.min (int8)` | int8 | 10,000,000 | 0.1373 ms | not_measured | managed | 1.789x |
| `np.min (uint16)` | uint16 | 1,000 | 0.000585 ms | not_measured | managed | 6.853x |
| `np.min (uint16)` | uint16 | 100,000 | 0.00258 ms | not_measured | managed | 2.865x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.3293 ms | not_measured | managed | 1.516x |
| `np.min (uint32)` | uint32 | 1,000 | 0.000361 ms | not_measured | managed | 11.172x |
| `np.min (uint32)` | uint32 | 100,000 | 0.00445 ms | not_measured | managed | 2.388x |
| `np.min (uint32)` | uint32 | 10,000,000 | 2.9735 ms | not_measured | managed | 0.887x |
| `np.min (uint64)` | uint64 | 1,000 | 0.000446 ms | not_measured | managed | 10.004x |
| `np.min (uint64)` | uint64 | 100,000 | 0.01017 ms | not_measured | managed | 4.441x |
| `np.min (uint64)` | uint64 | 10,000,000 | 7.9931 ms | not_measured | managed | 1.003x |
| `np.min (uint8)` | uint8 | 1,000 | 0.000587 ms | not_measured | managed | 2.494x |
| `np.min (uint8)` | uint8 | 100,000 | 0.00164 ms | not_measured | managed | 3.512x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.1365 ms | not_measured | managed | 1.701x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.00301 ms | not_measured | managed | 7.426x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.2710 ms | not_measured | managed | 2.573x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 24.6352 ms | not_measured | managed | 3.080x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.000981 ms | not_measured | managed | 18.771x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.03605 ms | not_measured | managed | 2.078x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 9.1712 ms | not_measured | managed | 2.179x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.00109 ms | not_measured | managed | 16.751x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.06730 ms | not_measured | managed | 1.803x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 18.9885 ms | not_measured | managed | 1.402x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.00485 ms | not_measured | managed | 4.570x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.2710 ms | not_measured | managed | 1.785x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 18.2619 ms | not_measured | managed | 3.668x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.00207 ms | not_measured | managed | 8.780x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.03198 ms | not_measured | managed | 2.342x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 9.5082 ms | not_measured | managed | 2.129x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.00115 ms | not_measured | managed | 15.877x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.05979 ms | not_measured | managed | 2.028x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 18.9501 ms | not_measured | managed | 1.152x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.00146 ms | not_measured | managed | 8.188x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.4149 ms | not_measured | managed | 1.476x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 30.8654 ms | not_measured | managed | 1.729x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.000782 ms | not_measured | managed | 3.648x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.00475 ms | not_measured | managed | 5.934x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 1.0923 ms | not_measured | managed | 3.095x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.000747 ms | not_measured | managed | 12.295x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.00388 ms | not_measured | managed | 12.244x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 7.2449 ms | not_measured | managed | 1.078x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.00283 ms | not_measured | managed | 4.304x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.1052 ms | not_measured | managed | 4.498x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 20.1585 ms | not_measured | managed | 2.507x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.00139 ms | not_measured | managed | 6.663x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.07124 ms | not_measured | managed | 2.176x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 7.2962 ms | not_measured | managed | 4.688x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.00135 ms | not_measured | managed | 19.053x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.07147 ms | not_measured | managed | 2.361x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 6.4357 ms | not_measured | managed | 8.208x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.00747 ms | not_measured | managed | 2.194x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.2914 ms | not_measured | managed | 0.965x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 125.6728 ms | not_measured | managed | 0.913x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.00452 ms | not_measured | managed | 8.402x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 0.3565 ms | not_measured | managed | 1.832x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 40.3272 ms | not_measured | managed | 1.907x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.00282 ms | not_measured | managed | 12.086x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.3588 ms | not_measured | managed | 1.902x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 58.4312 ms | not_measured | managed | 1.903x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.00184 ms | not_measured | managed | 6.390x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.4243 ms | not_measured | managed | 1.382x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 33.4489 ms | not_measured | managed | 1.578x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.000796 ms | not_measured | managed | 11.097x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.00474 ms | not_measured | managed | 5.940x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 2.9827 ms | not_measured | managed | 1.109x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.000601 ms | not_measured | managed | 15.161x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.01087 ms | not_measured | managed | 4.337x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 7.3698 ms | not_measured | managed | 1.079x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.01146 ms | not_measured | managed | 7.686x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.6835 ms | not_measured | managed | 1.423x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 111.0336 ms | not_measured | managed | 1.089x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.00362 ms | not_measured | managed | 21.417x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.3484 ms | not_measured | managed | 2.703x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 32.5439 ms | not_measured | managed | 1.546x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.00270 ms | not_measured | managed | 28.628x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.2331 ms | not_measured | managed | 4.263x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 59.2902 ms | not_measured | managed | 1.216x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.00220 ms | not_measured | managed | 28.471x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.2084 ms | not_measured | managed | 2.369x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 9.4823 ms | not_measured | managed | 2.151x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.000510 ms | not_measured | managed | 39.714x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 1.9431 ms | not_measured | managed | 3.438x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 136.3894 ms | not_measured | managed | 4.319x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.000598 ms | not_measured | managed | 19.010x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 3.0522 ms | not_measured | managed | 2.155x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 281.8562 ms | not_measured | managed | 0.930x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.00666 ms | not_measured | managed | 12.983x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6180 ms | not_measured | managed | 1.480x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 125.6704 ms | not_measured | managed | 1.167x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00453 ms | not_measured | managed | 16.600x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2772 ms | not_measured | managed | 3.388x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 48.1602 ms | not_measured | managed | 1.252x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00279 ms | not_measured | managed | 26.878x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.2818 ms | not_measured | managed | 3.512x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 58.9489 ms | not_measured | managed | 1.054x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.00524 ms | not_measured | managed | 14.997x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.4017 ms | not_measured | managed | 5.253x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 40.0969 ms | not_measured | managed | 3.418x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.00260 ms | not_measured | managed | 7.286x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.1479 ms | not_measured | managed | 2.118x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 9.2691 ms | not_measured | managed | 6.382x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.00128 ms | not_measured | managed | 13.421x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.1353 ms | not_measured | managed | 2.574x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 18.3646 ms | not_measured | managed | 5.162x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.00110 ms | not_measured | managed | 5.415x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.1877 ms | not_measured | managed | 1.994x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 18.7178 ms | not_measured | managed | 2.236x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.000626 ms | not_measured | managed | 16.922x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.00587 ms | not_measured | managed | 9.554x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 1.7385 ms | not_measured | managed | 13.732x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.000705 ms | not_measured | managed | 4.837x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.01809 ms | not_measured | managed | 4.098x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 8.4085 ms | not_measured | managed | 4.168x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.00524 ms | not_measured | managed | 14.462x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.2154 ms | not_measured | managed | 9.702x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 40.1327 ms | not_measured | managed | 3.116x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.00268 ms | not_measured | managed | 21.579x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.1483 ms | not_measured | managed | 2.094x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 10.5816 ms | not_measured | managed | 2.865x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.00128 ms | not_measured | managed | 41.179x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.1425 ms | not_measured | managed | 2.423x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 14.6060 ms | not_measured | managed | 3.203x |
| `np.prod (float64)` | float64 | 1,000 | 0.000627 ms | not_measured | managed | 7.681x |
| `np.prod (float64)` | float64 | 100,000 | 1.1161 ms | not_measured | managed | 2.055x |
| `np.prod (float64)` | float64 | 10,000,000 | 135.7276 ms | not_measured | managed | 2.191x |
| `np.prod (int64)` | int64 | 1,000 | 0.000995 ms | not_measured | managed | 1.944x |
| `np.prod (int64)` | int64 | 100,000 | 0.01945 ms | not_measured | managed | 6.215x |
| `np.prod (int64)` | int64 | 10,000,000 | 3.9156 ms | not_measured | managed | 3.223x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.000648 ms | not_measured | managed | 6.835x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.01851 ms | not_measured | managed | 0.586x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 120.7512 ms | not_measured | managed | 0.161x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.000497 ms | not_measured | managed | 9.751x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.04064 ms | not_measured | managed | 1.424x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 4.7779 ms | not_measured | managed | 1.691x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.000674 ms | not_measured | managed | 6.963x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.01173 ms | not_measured | managed | 4.854x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 7.9933 ms | not_measured | managed | 22.090x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.00131 ms | not_measured | managed | 3.670x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.04362 ms | not_measured | managed | 2.642x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 7.9259 ms | not_measured | managed | 1.570x |
| `np.std (float16)` | float16 | 1,000 | 0.00426 ms | not_measured | managed | 9.234x |
| `np.std (float16)` | float16 | 100,000 | 0.1801 ms | not_measured | managed | 8.432x |
| `np.std (float16)` | float16 | 10,000,000 | 38.6467 ms | not_measured | managed | 2.436x |
| `np.std (float32)` | float32 | 1,000 | 0.000813 ms | not_measured | managed | 30.154x |
| `np.std (float32)` | float32 | 100,000 | 0.01819 ms | not_measured | managed | 4.824x |
| `np.std (float32)` | float32 | 10,000,000 | 6.3983 ms | not_measured | managed | 4.789x |
| `np.std (float64)` | float64 | 1,000 | 0.000951 ms | not_measured | managed | 21.545x |
| `np.std (float64)` | float64 | 100,000 | 0.02450 ms | not_measured | managed | 4.393x |
| `np.std (float64)` | float64 | 10,000,000 | 16.2628 ms | not_measured | managed | 3.344x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.01048 ms | not_measured | managed | 4.520x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.3653 ms | not_measured | managed | 2.881x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 82.9783 ms | not_measured | managed | 1.312x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.000994 ms | not_measured | managed | 24.735x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.06127 ms | not_measured | managed | 1.366x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 4.5510 ms | not_measured | managed | 5.459x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.000556 ms | not_measured | managed | 39.304x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.03654 ms | not_measured | managed | 3.525x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 8.8838 ms | not_measured | managed | 5.870x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.000905 ms | not_measured | managed | 5.025x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.02121 ms | not_measured | managed | 2.105x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 15.9346 ms | not_measured | managed | 1.102x |
| `np.sum (float16)` | float16 | 1,000 | 0.000982 ms | not_measured | managed | 6.464x |
| `np.sum (float16)` | float16 | 100,000 | 0.05763 ms | not_measured | managed | 3.459x |
| `np.sum (float16)` | float16 | 10,000,000 | 6.8867 ms | not_measured | managed | 3.420x |
| `np.sum (float32)` | float32 | 1,000 | 0.000590 ms | not_measured | managed | 7.227x |
| `np.sum (float32)` | float32 | 100,000 | 0.00580 ms | not_measured | managed | 4.399x |
| `np.sum (float32)` | float32 | 10,000,000 | 2.8338 ms | not_measured | managed | 2.889x |
| `np.sum (float64)` | float64 | 1,000 | 0.00487 ms | not_measured | managed | 0.892x |
| `np.sum (float64)` | float64 | 100,000 | 0.1873 ms | not_measured | managed | 0.144x |
| `np.sum (float64)` | float64 | 10,000,000 | 10.4568 ms | not_measured | managed | 1.398x |
| `np.sum (int16)` | int16 | 1,000 | 0.000394 ms | not_measured | managed | 13.307x |
| `np.sum (int16)` | int16 | 100,000 | 0.01897 ms | not_measured | managed | 3.564x |
| `np.sum (int16)` | int16 | 10,000,000 | 4.4423 ms | not_measured | managed | 0.700x |
| `np.sum (int32)` | int32 | 1,000 | 0.000404 ms | not_measured | managed | 13.245x |
| `np.sum (int32)` | int32 | 100,000 | 0.04448 ms | not_measured | managed | 1.876x |
| `np.sum (int32)` | int32 | 10,000,000 | 5.1016 ms | not_measured | managed | 1.686x |
| `np.sum (int64)` | int64 | 1,000 | 0.000630 ms | not_measured | managed | 6.949x |
| `np.sum (int64)` | int64 | 100,000 | 0.00776 ms | not_measured | managed | 3.942x |
| `np.sum (int64)` | int64 | 10,000,000 | 7.4052 ms | not_measured | managed | 1.227x |
| `np.sum (int8)` | int8 | 1,000 | 0.00104 ms | not_measured | managed | 5.127x |
| `np.sum (int8)` | int8 | 100,000 | 0.01856 ms | not_measured | managed | 4.384x |
| `np.sum (int8)` | int8 | 10,000,000 | 1.9359 ms | not_measured | managed | 4.009x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.000716 ms | not_measured | managed | 7.439x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.03916 ms | not_measured | managed | 2.095x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 4.1516 ms | not_measured | managed | 0.758x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.000971 ms | not_measured | managed | 5.342x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.03924 ms | not_measured | managed | 1.726x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 4.8746 ms | not_measured | managed | 1.561x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.000638 ms | not_measured | managed | 6.807x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.00449 ms | not_measured | managed | 6.796x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 7.4542 ms | not_measured | managed | 1.207x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.000761 ms | not_measured | managed | 2.598x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.03354 ms | not_measured | managed | 2.424x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 3.7265 ms | not_measured | managed | 2.083x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.00247 ms | not_measured | managed | 2.026x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.02021 ms | not_measured | managed | 1.971x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 14.2991 ms | not_measured | managed | 1.160x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.00452 ms | not_measured | managed | 2.395x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1524 ms | not_measured | managed | 4.089x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 13.6077 ms | not_measured | managed | 4.897x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.00149 ms | not_measured | managed | 3.103x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.00777 ms | not_measured | managed | 1.954x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 3.4190 ms | not_measured | managed | 0.905x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.00125 ms | not_measured | managed | 3.757x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.01323 ms | not_measured | managed | 1.744x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 8.3057 ms | not_measured | managed | 0.973x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.000793 ms | not_measured | managed | 7.488x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.00981 ms | not_measured | managed | 9.278x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 0.9238 ms | not_measured | managed | 9.291x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.000682 ms | not_measured | managed | 8.783x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.01416 ms | not_measured | managed | 7.561x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 3.4395 ms | not_measured | managed | 3.226x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.000673 ms | not_measured | managed | 7.339x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.01826 ms | not_measured | managed | 1.493x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 7.8093 ms | not_measured | managed | 1.018x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.00117 ms | not_measured | managed | 5.141x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.00928 ms | not_measured | managed | 11.326x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.7403 ms | not_measured | managed | 13.560x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.00118 ms | not_measured | managed | 5.195x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.00990 ms | not_measured | managed | 10.665x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 0.9635 ms | not_measured | managed | 10.654x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.000388 ms | not_measured | managed | 5.564x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.01420 ms | not_measured | managed | 6.428x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 3.5127 ms | not_measured | managed | 1.432x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.000671 ms | not_measured | managed | 7.505x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.01871 ms | not_measured | managed | 2.890x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 7.8527 ms | not_measured | managed | 1.030x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.00117 ms | not_measured | managed | 1.945x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.00791 ms | not_measured | managed | 13.284x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.7396 ms | not_measured | managed | 13.402x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.00254 ms | not_measured | managed | 1.965x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.04466 ms | not_measured | managed | 1.091x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 16.9045 ms | not_measured | managed | 1.037x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.00209 ms | not_measured | managed | 3.315x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.07007 ms | not_measured | managed | 3.383x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 3.3356 ms | not_measured | managed | 5.875x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.00121 ms | not_measured | managed | 3.777x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.01935 ms | not_measured | managed | 1.415x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 3.4191 ms | not_measured | managed | 2.055x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.00210 ms | not_measured | managed | 2.225x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.02168 ms | not_measured | managed | 1.400x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 7.5333 ms | not_measured | managed | 1.615x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.000536 ms | not_measured | managed | 10.489x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.00398 ms | not_measured | managed | 17.862x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 0.8094 ms | not_measured | managed | 7.897x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.000621 ms | not_measured | managed | 9.277x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.00542 ms | not_measured | managed | 6.901x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 3.5327 ms | not_measured | managed | 1.076x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.000450 ms | not_measured | managed | 10.553x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.01139 ms | not_measured | managed | 3.017x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 7.8497 ms | not_measured | managed | 1.101x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.000666 ms | not_measured | managed | 8.637x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.00363 ms | not_measured | managed | 23.499x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.3024 ms | not_measured | managed | 26.321x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.000677 ms | not_measured | managed | 8.480x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.00407 ms | not_measured | managed | 21.015x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 0.8058 ms | not_measured | managed | 9.829x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.000427 ms | not_measured | managed | 13.073x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.00953 ms | not_measured | managed | 7.498x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 2.0660 ms | not_measured | managed | 3.641x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.000730 ms | not_measured | managed | 6.407x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.00856 ms | not_measured | managed | 4.013x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 3.0920 ms | not_measured | managed | 2.932x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.000647 ms | not_measured | managed | 3.391x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.00666 ms | not_measured | managed | 12.766x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.3069 ms | not_measured | managed | 25.327x |
| `np.var (float16)` | float16 | 1,000 | 0.00425 ms | not_measured | managed | 8.913x |
| `np.var (float16)` | float16 | 100,000 | 0.1795 ms | not_measured | managed | 8.445x |
| `np.var (float16)` | float16 | 10,000,000 | 38.6628 ms | not_measured | managed | 2.928x |
| `np.var (float32)` | float32 | 1,000 | 0.000481 ms | not_measured | managed | 47.609x |
| `np.var (float32)` | float32 | 100,000 | 0.01784 ms | not_measured | managed | 4.853x |
| `np.var (float32)` | float32 | 10,000,000 | 6.5042 ms | not_measured | managed | 4.555x |
| `np.var (float64)` | float64 | 1,000 | 0.000586 ms | not_measured | managed | 32.973x |
| `np.var (float64)` | float64 | 100,000 | 0.01891 ms | not_measured | managed | 3.087x |
| `np.var (float64)` | float64 | 10,000,000 | 16.4923 ms | not_measured | managed | 2.926x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.01052 ms | not_measured | managed | 4.413x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.8060 ms | not_measured | managed | 3.021x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 136.9422 ms | not_measured | managed | 1.598x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.00100 ms | not_measured | managed | 23.487x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.02116 ms | not_measured | managed | 3.875x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 5.5370 ms | not_measured | managed | 4.541x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.000944 ms | not_measured | managed | 22.237x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.02211 ms | not_measured | managed | 5.739x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 15.3635 ms | not_measured | managed | 3.388x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.00262 ms | not_measured | managed | 1.976x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.1447 ms | not_measured | managed | 7.902x |
| `np.choose(selector, choices) (float64)` | float64 | 10,000,000 | 1.9390 ms | not_measured | managed | 5.964x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.000764 ms | not_measured | managed | 2.470x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.08472 ms | not_measured | managed | 2.824x |
| `np.compress(mask, a) (float64)` | float64 | 10,000,000 | 0.9301 ms | not_measured | managed | 3.497x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.000861 ms | not_measured | managed | 0.472x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.000923 ms | not_measured | managed | 1.437x |
| `np.diag_indices(n) (float64)` | float64 | 10,000,000 | 0.00104 ms | not_measured | managed | 1.494x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.000981 ms | not_measured | managed | 3.943x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.00134 ms | not_measured | managed | 9.880x |
| `np.diag_indices_from(a) (float64)` | float64 | 10,000,000 | 0.00170 ms | not_measured | managed | 7.928x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.000696 ms | not_measured | managed | 4.082x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.08504 ms | not_measured | managed | 2.843x |
| `np.extract(mask, a) (float64)` | float64 | 10,000,000 | 0.6694 ms | not_measured | managed | 4.868x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.00139 ms | not_measured | managed | 5.380x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.1544 ms | not_measured | managed | 2.314x |
| `np.indices(shape) (float64)` | float64 | 10,000,000 | 1.3308 ms | not_measured | managed | 2.358x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.00726 ms | not_measured | managed | 1.105x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.4547 ms | not_measured | managed | 1.686x |
| `np.mask_indices(n, triu) (float64)` | float64 | 10,000,000 | 3.2937 ms | not_measured | managed | 2.195x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.000547 ms | not_measured | managed | 2.285x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.2793 ms | not_measured | managed | 1.400x |
| `np.place(a, mask, values) (float64)` | float64 | 10,000,000 | 3.3659 ms | not_measured | managed | 1.170x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.00375 ms | not_measured | managed | 0.481x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.06286 ms | not_measured | managed | 2.994x |
| `np.put(a, indices, values) (float64)` | float64 | 10,000,000 | 1.1706 ms | not_measured | managed | 2.293x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.00193 ms | not_measured | managed | 1.623x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.1627 ms | not_measured | managed | 0.450x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 10,000,000 | 1.4423 ms | not_measured | managed | 1.251x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.00154 ms | not_measured | managed | 7.362x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.08102 ms | not_measured | managed | 11.530x |
| `np.select(condlist, choicelist) (float64)` | float64 | 10,000,000 | 1.5030 ms | not_measured | managed | 7.424x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.00120 ms | not_measured | managed | 1.108x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.09716 ms | not_measured | managed | 1.889x |
| `np.take(a, indices) (float64)` | float64 | 10,000,000 | 1.3462 ms | not_measured | managed | 1.670x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.000712 ms | not_measured | managed | 2.343x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.02168 ms | not_measured | managed | 2.400x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 10,000,000 | 0.6758 ms | not_measured | managed | 1.414x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.00358 ms | not_measured | managed | 2.769x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.04081 ms | not_measured | managed | 3.579x |
| `np.tril_indices_from(a) (float64)` | float64 | 10,000,000 | 0.7099 ms | not_measured | managed | 3.175x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.00349 ms | not_measured | managed | 2.967x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.04122 ms | not_measured | managed | 3.530x |
| `np.triu_indices_from(a) (float64)` | float64 | 10,000,000 | 0.5196 ms | not_measured | managed | 5.215x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.00152 ms | not_measured | managed | 1.779x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1124 ms | not_measured | managed | 0.958x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 10,000,000 | 1.0301 ms | not_measured | managed | 1.354x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.00114 ms | not_measured | managed | 0.656x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.05078 ms | not_measured | managed | 1.130x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 8.1562 ms | not_measured | managed | 1.264x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.00145 ms | not_measured | managed | 0.733x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.05508 ms | not_measured | managed | 1.220x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 28.3598 ms | not_measured | managed | 1.131x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.00178 ms | not_measured | managed | 0.059x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.00181 ms | not_measured | managed | 0.059x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.00177 ms | not_measured | managed | 0.086x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.00278 ms | not_measured | managed | 0.056x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.00266 ms | not_measured | managed | 0.084x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.00278 ms | not_measured | managed | 0.080x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.00263 ms | not_measured | managed | 0.059x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.00200 ms | not_measured | managed | 0.111x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.00259 ms | not_measured | managed | 0.087x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.00164 ms | not_measured | managed | 0.063x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.00164 ms | not_measured | managed | 0.092x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.00139 ms | not_measured | managed | 0.109x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.00154 ms | not_measured | managed | 0.068x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.00164 ms | not_measured | managed | 0.064x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.00163 ms | not_measured | managed | 0.092x |
| `a[::2] * 2` | float64 | 1,000 | 0.00743 ms | not_measured | managed | 0.111x |
| `a[::2] * 2` | float64 | 100,000 | 0.03695 ms | not_measured | managed | 0.822x |
| `a[::2] * 2` | float64 | 10,000,000 | 18.8604 ms | not_measured | managed | 0.882x |
| `a[N/10:9N/10] * 2` | float64 | 1,000 | 0.00451 ms | not_measured | managed | 0.182x |
| `a[N/10:9N/10] * 2` | float64 | 100,000 | 0.03964 ms | not_measured | managed | 0.456x |
| `a[N/10:9N/10] * 2` | float64 | 10,000,000 | 12.1641 ms | not_measured | managed | 1.475x |
| `a[N/10:9N/10].copy()` | float64 | 1,000 | 0.00196 ms | not_measured | managed | 0.195x |
| `a[N/10:9N/10].copy()` | float64 | 100,000 | 0.02330 ms | not_measured | managed | 0.693x |
| `a[N/10:9N/10].copy()` | float64 | 10,000,000 | 8.4979 ms | not_measured | managed | 1.611x |
| `np.copy(a[N/10:9N/10])` | float64 | 1,000 | 0.00311 ms | not_measured | managed | 0.188x |
| `np.copy(a[N/10:9N/10])` | float64 | 100,000 | 0.03524 ms | not_measured | managed | 0.484x |
| `np.copy(a[N/10:9N/10])` | float64 | 10,000,000 | 6.7731 ms | not_measured | managed | 1.964x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.00970 ms | not_measured | managed | 1.193x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 0.5263 ms | not_measured | managed | 0.565x |
| `np.argpartition(a, kth) (float32)` | float32 | 10,000,000 | 6.2714 ms | not_measured | managed | 0.537x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.01034 ms | not_measured | managed | 1.344x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.4771 ms | not_measured | managed | 0.688x |
| `np.argpartition(a, kth) (float64)` | float64 | 10,000,000 | 5.4378 ms | not_measured | managed | 0.696x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.00532 ms | not_measured | managed | 0.553x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 0.4336 ms | not_measured | managed | 0.558x |
| `np.argpartition(a, kth) (int32)` | int32 | 10,000,000 | 6.3361 ms | not_measured | managed | 0.515x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.00937 ms | not_measured | managed | 1.079x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 0.4601 ms | not_measured | managed | 0.736x |
| `np.argpartition(a, kth) (int64)` | int64 | 10,000,000 | 7.3021 ms | not_measured | managed | 0.566x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.01581 ms | not_measured | managed | 2.025x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.4300 ms | not_measured | managed | 2.732x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 230.1706 ms | not_measured | managed | 4.928x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.02445 ms | not_measured | managed | 1.715x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 2.2957 ms | not_measured | managed | 2.182x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 415.5380 ms | not_measured | managed | 3.620x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.01587 ms | not_measured | managed | 0.683x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 0.8714 ms | not_measured | managed | 0.901x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 196.0981 ms | not_measured | managed | 1.349x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.02692 ms | not_measured | managed | 0.475x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 1.6603 ms | not_measured | managed | 0.678x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 329.0815 ms | not_measured | managed | 1.271x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.000904 ms | not_measured | managed | 9.006x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.09421 ms | not_measured | managed | 6.662x |
| `np.argwhere(a) (float32)` | float32 | 10,000,000 | 1.1934 ms | not_measured | managed | 4.063x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.00107 ms | not_measured | managed | 7.580x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.1004 ms | not_measured | managed | 3.698x |
| `np.argwhere(a) (float64)` | float64 | 10,000,000 | 0.9456 ms | not_measured | managed | 5.203x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.000856 ms | not_measured | managed | 3.381x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.09427 ms | not_measured | managed | 5.538x |
| `np.argwhere(a) (int32)` | int32 | 10,000,000 | 1.1973 ms | not_measured | managed | 3.049x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.00181 ms | not_measured | managed | 3.988x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.05039 ms | not_measured | managed | 9.843x |
| `np.argwhere(a) (int64)` | int64 | 10,000,000 | 1.3322 ms | not_measured | managed | 2.905x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.00198 ms | not_measured | managed | 2.679x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.1608 ms | not_measured | managed | 1.804x |
| `np.flatnonzero(a) (float32)` | float32 | 10,000,000 | 1.5516 ms | not_measured | managed | 2.352x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.00204 ms | not_measured | managed | 2.619x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.1648 ms | not_measured | managed | 1.769x |
| `np.flatnonzero(a) (float64)` | float64 | 10,000,000 | 1.9027 ms | not_measured | managed | 1.928x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.00202 ms | not_measured | managed | 0.977x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.1805 ms | not_measured | managed | 0.970x |
| `np.flatnonzero(a) (int32)` | int32 | 10,000,000 | 1.2839 ms | not_measured | managed | 1.946x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.00136 ms | not_measured | managed | 3.158x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.1730 ms | not_measured | managed | 1.037x |
| `np.flatnonzero(a) (int64)` | int64 | 10,000,000 | 1.8626 ms | not_measured | managed | 1.378x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.03581 ms | not_measured | managed | 1.786x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 2.2091 ms | not_measured | managed | 7.807x |
| `np.lexsort((b, a)) (float32)` | float32 | 10,000,000 | 44.9611 ms | not_measured | managed | 5.349x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.05282 ms | not_measured | managed | 1.092x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 7.0893 ms | not_measured | managed | 2.535x |
| `np.lexsort((b, a)) (float64)` | float64 | 10,000,000 | 82.2428 ms | not_measured | managed | 2.995x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.03518 ms | not_measured | managed | 1.098x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 1.8420 ms | not_measured | managed | 8.278x |
| `np.lexsort((b, a)) (int32)` | int32 | 10,000,000 | 38.3356 ms | not_measured | managed | 6.578x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.03369 ms | not_measured | managed | 1.401x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 3.4952 ms | not_measured | managed | 4.790x |
| `np.lexsort((b, a)) (int64)` | int64 | 10,000,000 | 68.6975 ms | not_measured | managed | 4.319x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.00211 ms | not_measured | managed | 2.027x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.09469 ms | not_measured | managed | 3.055x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 15.5874 ms | not_measured | managed | 2.381x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.00220 ms | not_measured | managed | 1.954x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.1034 ms | not_measured | managed | 2.812x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 29.9693 ms | not_measured | managed | 1.274x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.00211 ms | not_measured | managed | 0.724x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.04178 ms | not_measured | managed | 4.166x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 20.2763 ms | not_measured | managed | 0.991x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.00217 ms | not_measured | managed | 0.721x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.1011 ms | not_measured | managed | 1.763x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 16.9931 ms | not_measured | managed | 1.345x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.00829 ms | not_measured | managed | 0.256x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.3131 ms | not_measured | managed | 1.068x |
| `np.partition(a, kth) (float32)` | float32 | 10,000,000 | 5.1059 ms | not_measured | managed | 0.761x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.00986 ms | not_measured | managed | 0.802x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.3871 ms | not_measured | managed | 0.992x |
| `np.partition(a, kth) (float64)` | float64 | 10,000,000 | 7.1380 ms | not_measured | managed | 0.404x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.00543 ms | not_measured | managed | 0.269x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.2832 ms | not_measured | managed | 0.236x |
| `np.partition(a, kth) (int32)` | int32 | 10,000,000 | 2.1995 ms | not_measured | managed | 0.560x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.00685 ms | not_measured | managed | 0.847x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.3240 ms | not_measured | managed | 0.605x |
| `np.partition(a, kth) (int64)` | int64 | 10,000,000 | 4.1696 ms | not_measured | managed | 0.470x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.01152 ms | not_measured | managed | 1.154x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 2.8718 ms | not_measured | managed | 0.886x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 320.1911 ms | not_measured | managed | 0.898x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.01205 ms | not_measured | managed | 1.098x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 2.5753 ms | not_measured | managed | 0.994x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 325.7368 ms | not_measured | managed | 0.913x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.00948 ms | not_measured | managed | 1.933x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 2.3226 ms | not_measured | managed | 1.767x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 302.0983 ms | not_measured | managed | 1.833x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.00601 ms | not_measured | managed | 3.045x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 2.1843 ms | not_measured | managed | 1.814x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 286.0294 ms | not_measured | managed | 1.860x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.01077 ms | not_measured | managed | 0.521x |
| `np.sort(a) (float32)` | float32 | 100,000 | 0.5893 ms | not_measured | managed | 0.954x |
| `np.sort(a) (float32)` | float32 | 10,000,000 | 8.9273 ms | not_measured | managed | 0.852x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.02156 ms | not_measured | managed | 0.434x |
| `np.sort(a) (float64)` | float64 | 100,000 | 1.1988 ms | not_measured | managed | 0.986x |
| `np.sort(a) (float64)` | float64 | 10,000,000 | 19.7992 ms | not_measured | managed | 0.800x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.00758 ms | not_measured | managed | 0.263x |
| `np.sort(a) (int32)` | int32 | 100,000 | 0.5406 ms | not_measured | managed | 0.403x |
| `np.sort(a) (int32)` | int32 | 10,000,000 | 7.7599 ms | not_measured | managed | 0.351x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.02544 ms | not_measured | managed | 0.211x |
| `np.sort(a) (int64)` | int64 | 100,000 | 1.0366 ms | not_measured | managed | 0.632x |
| `np.sort(a) (int64)` | int64 | 10,000,000 | 18.3609 ms | not_measured | managed | 0.421x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.01528 ms | not_measured | managed | 0.487x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 0.6751 ms | not_measured | managed | 1.324x |
| `np.sort_complex(a) (float32)` | float32 | 10,000,000 | 7.7330 ms | not_measured | managed | 1.415x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.02449 ms | not_measured | managed | 0.452x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 1.3183 ms | not_measured | managed | 1.155x |
| `np.sort_complex(a) (float64)` | float64 | 10,000,000 | 22.1663 ms | not_measured | managed | 0.863x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.00960 ms | not_measured | managed | 0.279x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 0.5979 ms | not_measured | managed | 0.875x |
| `np.sort_complex(a) (int32)` | int32 | 10,000,000 | 7.5716 ms | not_measured | managed | 0.785x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.02793 ms | not_measured | managed | 0.218x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 1.0595 ms | not_measured | managed | 0.916x |
| `np.sort_complex(a) (int64)` | int64 | 10,000,000 | 13.1490 ms | not_measured | managed | 0.852x |
| `np.average(a) (float16)` | float16 | 1,000 | 0.00103 ms | not_measured | managed | 4.741x |
| `np.average(a) (float16)` | float16 | 100,000 | 0.1644 ms | not_measured | managed | 1.064x |
| `np.average(a) (float16)` | float16 | 10,000,000 | 8.5769 ms | not_measured | managed | 1.928x |
| `np.average(a) (float32)` | float32 | 1,000 | 0.000856 ms | not_measured | managed | 4.533x |
| `np.average(a) (float32)` | float32 | 100,000 | 0.00356 ms | not_measured | managed | 4.905x |
| `np.average(a) (float32)` | float32 | 10,000,000 | 2.6607 ms | not_measured | managed | 3.024x |
| `np.average(a) (float64)` | float64 | 1,000 | 0.000938 ms | not_measured | managed | 8.912x |
| `np.average(a) (float64)` | float64 | 100,000 | 0.00402 ms | not_measured | managed | 7.759x |
| `np.average(a) (float64)` | float64 | 10,000,000 | 2.6235 ms | not_measured | managed | 5.272x |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.00153 ms | not_measured | managed | 2.114x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.09029 ms | not_measured | managed | 2.460x |
| `np.bincount(a) (int32)` | int32 | 10,000,000 | 2.2219 ms | not_measured | managed | 1.732x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | 0.00934 ms | 0.04240 ms | managed | 2.493x |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | 0.1252 ms | 1.9148 ms | managed | 16.606x |
| `np.convolve(a, kernel) (float64)` | float64 | 10,000,000 | 1.2349 ms | 206.6472 ms | managed | 17.809x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | 0.02386 ms | not_measured | managed | 2.385x |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | 0.3439 ms | not_measured | managed | 2.306x |
| `np.corrcoef(a, b) (float64)` | float64 | 10,000,000 | 7.3565 ms | not_measured | managed | 1.461x |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | 0.00490 ms | 0.03320 ms | managed | 4.552x |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | 0.3823 ms | 1.9150 ms | managed | 5.419x |
| `np.correlate(a, kernel) (float64)` | float64 | 10,000,000 | 3.8500 ms | 166.2216 ms | managed | 4.922x |
| `np.count_nonzero(a) (float16)` | float16 | 1,000 | 0.00101 ms | not_measured | managed | 1.711x |
| `np.count_nonzero(a) (float16)` | float16 | 100,000 | 0.07826 ms | not_measured | managed | 2.996x |
| `np.count_nonzero(a) (float16)` | float16 | 10,000,000 | 8.2089 ms | not_measured | managed | 2.932x |
| `np.count_nonzero(a) (float32)` | float32 | 1,000 | 0.000306 ms | not_measured | managed | 1.905x |
| `np.count_nonzero(a) (float32)` | float32 | 100,000 | 0.00505 ms | not_measured | managed | 14.227x |
| `np.count_nonzero(a) (float32)` | float32 | 10,000,000 | 1.7610 ms | not_measured | managed | 4.237x |
| `np.count_nonzero(a) (float64)` | float64 | 1,000 | 0.000389 ms | not_measured | managed | 2.648x |
| `np.count_nonzero(a) (float64)` | float64 | 100,000 | 0.00892 ms | not_measured | managed | 8.216x |
| `np.count_nonzero(a) (float64)` | float64 | 10,000,000 | 6.9868 ms | not_measured | managed | 1.332x |
| `np.cov(a, b) (float64)` | float64 | 1,000 | 0.01341 ms | not_measured | managed | 3.175x |
| `np.cov(a, b) (float64)` | float64 | 100,000 | 0.3273 ms | not_measured | managed | 2.389x |
| `np.cov(a, b) (float64)` | float64 | 10,000,000 | 9.4482 ms | not_measured | managed | 1.123x |
| `np.cross(a, b) (float64)` | float64 | 1,000 | 0.04521 ms | not_measured | managed | 0.632x |
| `np.cross(a, b) (float64)` | float64 | 100,000 | 0.9692 ms | not_measured | managed | 0.862x |
| `np.cross(a, b) (float64)` | float64 | 10,000,000 | 13.4662 ms | not_measured | managed | 0.859x |
| `np.diff(a) (float64)` | float64 | 1,000 | 0.00119 ms | not_measured | managed | 3.611x |
| `np.diff(a) (float64)` | float64 | 100,000 | 0.04286 ms | not_measured | managed | 0.546x |
| `np.diff(a) (float64)` | float64 | 10,000,000 | 0.9062 ms | not_measured | managed | 1.839x |
| `np.digitize(a, bins) (float64)` | float64 | 1,000 | 0.01286 ms | not_measured | managed | 2.176x |
| `np.digitize(a, bins) (float64)` | float64 | 100,000 | 4.5322 ms | not_measured | managed | 1.015x |
| `np.digitize(a, bins) (float64)` | float64 | 10,000,000 | 34.6270 ms | not_measured | managed | 1.336x |
| `np.ediff1d(a) (float64)` | float64 | 1,000 | 0.000963 ms | not_measured | managed | 2.814x |
| `np.ediff1d(a) (float64)` | float64 | 100,000 | 0.03976 ms | not_measured | managed | 0.544x |
| `np.ediff1d(a) (float64)` | float64 | 10,000,000 | 0.9789 ms | not_measured | managed | 1.651x |
| `np.inner(a, b) (float64)` | float64 | 1,000 | 0.000752 ms | 0.000700 ms | openblas | 2.446x |
| `np.inner(a, b) (float64)` | float64 | 100,000 | 0.01155 ms | 0.01220 ms | managed | 1.569x |
| `np.inner(a, b) (float64)` | float64 | 10,000,000 | 0.3765 ms | 7.6714 ms | managed | 0.888x |
| `np.kron(a, b) (float64)` | float64 | 1,000 | 0.00480 ms | not_measured | managed | 4.563x |
| `np.kron(a, b) (float64)` | float64 | 100,000 | 0.04645 ms | not_measured | managed | 1.911x |
| `np.kron(a, b) (float64)` | float64 | 10,000,000 | 0.7266 ms | not_measured | managed | 2.752x |
| `np.median(a) (float16)` | float16 | 1,000 | 0.00636 ms | not_measured | managed | 2.052x |
| `np.median(a) (float16)` | float16 | 100,000 | 1.5469 ms | not_measured | managed | 0.716x |
| `np.median(a) (float16)` | float16 | 10,000,000 | 117.4406 ms | not_measured | managed | 1.187x |
| `np.median(a) (float32)` | float32 | 1,000 | 0.00384 ms | not_measured | managed | 2.623x |
| `np.median(a) (float32)` | float32 | 100,000 | 0.2871 ms | not_measured | managed | 2.202x |
| `np.median(a) (float32)` | float32 | 10,000,000 | 27.6873 ms | not_measured | managed | 3.291x |
| `np.median(a) (float64)` | float64 | 1,000 | 0.00247 ms | not_measured | managed | 3.594x |
| `np.median(a) (float64)` | float64 | 100,000 | 0.1960 ms | not_measured | managed | 3.344x |
| `np.median(a) (float64)` | float64 | 10,000,000 | 58.6913 ms | not_measured | managed | 1.860x |
| `np.percentile(a, 50) (float16)` | float16 | 1,000 | 0.01142 ms | not_measured | managed | 2.386x |
| `np.percentile(a, 50) (float16)` | float16 | 100,000 | 1.5221 ms | not_measured | managed | 1.497x |
| `np.percentile(a, 50) (float16)` | float16 | 10,000,000 | 122.2218 ms | not_measured | managed | 1.186x |
| `np.percentile(a, 50) (float32)` | float32 | 1,000 | 0.00377 ms | not_measured | managed | 6.103x |
| `np.percentile(a, 50) (float32)` | float32 | 100,000 | 0.2873 ms | not_measured | managed | 3.200x |
| `np.percentile(a, 50) (float32)` | float32 | 10,000,000 | 47.0458 ms | not_measured | managed | 1.322x |
| `np.percentile(a, 50) (float64)` | float64 | 1,000 | 0.00396 ms | not_measured | managed | 5.636x |
| `np.percentile(a, 50) (float64)` | float64 | 100,000 | 0.3074 ms | not_measured | managed | 3.142x |
| `np.percentile(a, 50) (float64)` | float64 | 10,000,000 | 58.4421 ms | not_measured | managed | 1.336x |
| `np.ptp(a) (float16)` | float16 | 1,000 | 0.00248 ms | not_measured | managed | 2.583x |
| `np.ptp(a) (float16)` | float16 | 100,000 | 0.8519 ms | not_measured | managed | 1.425x |
| `np.ptp(a) (float16)` | float16 | 10,000,000 | 90.0910 ms | not_measured | managed | 1.404x |
| `np.ptp(a) (float32)` | float32 | 1,000 | 0.00178 ms | not_measured | managed | 1.385x |
| `np.ptp(a) (float32)` | float32 | 100,000 | 0.00805 ms | not_measured | managed | 5.230x |
| `np.ptp(a) (float32)` | float32 | 10,000,000 | 6.3487 ms | not_measured | managed | 1.011x |
| `np.ptp(a) (float64)` | float64 | 1,000 | 0.00200 ms | not_measured | managed | 3.757x |
| `np.ptp(a) (float64)` | float64 | 100,000 | 0.03435 ms | not_measured | managed | 2.248x |
| `np.ptp(a) (float64)` | float64 | 10,000,000 | 15.9196 ms | not_measured | managed | 0.979x |
| `np.quantile(a, 0.5) (float16)` | float16 | 1,000 | 0.00637 ms | not_measured | managed | 4.154x |
| `np.quantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6518 ms | not_measured | managed | 1.374x |
| `np.quantile(a, 0.5) (float16)` | float16 | 10,000,000 | 122.9661 ms | not_measured | managed | 1.206x |
| `np.quantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00387 ms | not_measured | managed | 5.725x |
| `np.quantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2875 ms | not_measured | managed | 3.203x |
| `np.quantile(a, 0.5) (float32)` | float32 | 10,000,000 | 46.7303 ms | not_measured | managed | 1.321x |
| `np.quantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00396 ms | not_measured | managed | 5.507x |
| `np.quantile(a, 0.5) (float64)` | float64 | 100,000 | 0.2975 ms | not_measured | managed | 3.232x |
| `np.quantile(a, 0.5) (float64)` | float64 | 10,000,000 | 58.0829 ms | not_measured | managed | 1.339x |
| `np.trace(a) (float64)` | float64 | 1,000 | 0.000393 ms | not_measured | managed | 9.097x |
| `np.trace(a) (float64)` | float64 | 100,000 | 0.000824 ms | not_measured | managed | 4.888x |
| `np.trace(a) (float64)` | float64 | 10,000,000 | 0.00121 ms | not_measured | managed | 5.372x |
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.00487 ms | not_measured | managed | 1.542x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.5867 ms | not_measured | managed | 0.469x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 67.0815 ms | not_measured | managed | 0.777x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.000644 ms | not_measured | managed | 0.630x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.01361 ms | not_measured | managed | 0.419x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 5.0125 ms | not_measured | managed | 1.472x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.000902 ms | not_measured | managed | 0.513x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.04176 ms | not_measured | managed | 0.304x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 12.9198 ms | not_measured | managed | 1.854x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.01000 ms | not_measured | managed | 1.010x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 1.2958 ms | not_measured | managed | 0.668x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 187.0538 ms | not_measured | managed | 0.885x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.00134 ms | not_measured | managed | 0.573x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.02844 ms | not_measured | managed | 0.433x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 7.4648 ms | not_measured | managed | 3.671x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.00276 ms | not_measured | managed | 0.323x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.07639 ms | not_measured | managed | 4.061x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 34.2241 ms | not_measured | managed | 1.528x |
| `np.abs (float16)` | float16 | 1,000 | 0.000634 ms | not_measured | managed | 0.959x |
| `np.abs (float16)` | float16 | 100,000 | 0.02126 ms | not_measured | managed | 1.207x |
| `np.abs (float16)` | float16 | 10,000,000 | 3.7305 ms | not_measured | managed | 1.210x |
| `np.abs (float32)` | float32 | 1,000 | 0.000592 ms | not_measured | managed | 0.669x |
| `np.abs (float32)` | float32 | 100,000 | 0.01684 ms | not_measured | managed | 0.333x |
| `np.abs (float32)` | float32 | 10,000,000 | 5.1101 ms | not_measured | managed | 2.155x |
| `np.abs (float64)` | float64 | 1,000 | 0.000735 ms | not_measured | managed | 0.616x |
| `np.abs (float64)` | float64 | 100,000 | 0.03002 ms | not_measured | managed | 0.364x |
| `np.abs (float64)` | float64 | 10,000,000 | 13.9184 ms | not_measured | managed | 1.623x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.00105 ms | not_measured | managed | 0.577x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.02080 ms | not_measured | managed | 1.233x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 2.9720 ms | not_measured | managed | 2.348x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.000598 ms | not_measured | managed | 0.659x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.01229 ms | not_measured | managed | 0.454x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 7.5633 ms | not_measured | managed | 1.569x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.000796 ms | not_measured | managed | 0.548x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.04083 ms | not_measured | managed | 0.320x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 18.7445 ms | not_measured | managed | 1.217x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.01994 ms | not_measured | managed | 0.817x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 1.5189 ms | not_measured | managed | 0.537x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 153.6552 ms | not_measured | managed | 0.956x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.00681 ms | not_measured | managed | 0.809x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 1.0307 ms | not_measured | managed | 0.499x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 55.0523 ms | not_measured | managed | 1.063x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.01336 ms | not_measured | managed | 0.492x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 1.3013 ms | not_measured | managed | 0.477x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 113.2387 ms | not_measured | managed | 0.763x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.01779 ms | not_measured | managed | 1.001x |
| `np.angle(a) (float16)` | float16 | 100,000 | 1.4744 ms | not_measured | managed | 0.712x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 104.4782 ms | not_measured | managed | 1.370x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.00786 ms | not_measured | managed | 0.497x |
| `np.angle(a) (float32)` | float32 | 100,000 | 1.0527 ms | not_measured | managed | 0.586x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 135.2042 ms | not_measured | managed | 0.743x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.00560 ms | not_measured | managed | 0.754x |
| `np.angle(a) (float64)` | float64 | 100,000 | 1.0178 ms | not_measured | managed | 0.567x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 98.0834 ms | not_measured | managed | 0.935x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.01068 ms | not_measured | managed | 1.395x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 2.0687 ms | not_measured | managed | 0.522x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 178.2738 ms | not_measured | managed | 1.050x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.00397 ms | not_measured | managed | 0.937x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 0.7840 ms | not_measured | managed | 0.966x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 83.6425 ms | not_measured | managed | 1.079x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.00433 ms | not_measured | managed | 0.850x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 0.8414 ms | not_measured | managed | 0.949x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 107.4022 ms | not_measured | managed | 0.876x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.01569 ms | not_measured | managed | 1.039x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 1.8607 ms | not_measured | managed | 0.439x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 139.5282 ms | not_measured | managed | 1.082x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.00556 ms | not_measured | managed | 0.991x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 0.5418 ms | not_measured | managed | 0.949x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 64.2704 ms | not_measured | managed | 1.383x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.01337 ms | not_measured | managed | 0.491x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 0.6641 ms | not_measured | managed | 0.934x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 83.8267 ms | not_measured | managed | 1.538x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.00801 ms | not_measured | managed | 1.986x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 2.1351 ms | not_measured | managed | 0.513x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 144.0764 ms | not_measured | managed | 1.231x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.00723 ms | not_measured | managed | 0.558x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 0.8165 ms | not_measured | managed | 0.973x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 128.1767 ms | not_measured | managed | 0.698x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.00450 ms | not_measured | managed | 0.864x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 1.1055 ms | not_measured | managed | 0.717x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 143.0813 ms | not_measured | managed | 0.746x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.02495 ms | not_measured | managed | 0.426x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 2.5242 ms | not_measured | managed | 0.457x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 245.7459 ms | not_measured | managed | 0.878x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.01454 ms | not_measured | managed | 0.459x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 0.8845 ms | not_measured | managed | 0.970x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 119.8292 ms | not_measured | managed | 1.396x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.00836 ms | not_measured | managed | 0.959x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 2.0142 ms | not_measured | managed | 0.461x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 196.9930 ms | not_measured | managed | 1.096x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.01625 ms | not_measured | managed | 0.803x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 1.4310 ms | not_measured | managed | 0.871x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 169.4713 ms | not_measured | managed | 1.162x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.00717 ms | not_measured | managed | 0.558x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 0.9879 ms | not_measured | managed | 0.908x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 98.5911 ms | not_measured | managed | 1.197x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.00865 ms | not_measured | managed | 0.474x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 1.0341 ms | not_measured | managed | 0.929x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 153.9861 ms | not_measured | managed | 1.088x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.01207 ms | not_measured | managed | 1.746x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 3.6898 ms | not_measured | managed | 0.448x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 366.8175 ms | not_measured | managed | 0.754x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.01732 ms | not_measured | managed | 0.251x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 2.2107 ms | not_measured | managed | 0.583x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 264.5525 ms | not_measured | managed | 0.690x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.01047 ms | not_measured | managed | 0.543x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 1.3933 ms | not_measured | managed | 0.984x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 258.3147 ms | not_measured | managed | 0.969x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.01028 ms | not_measured | managed | 1.821x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 2.1246 ms | not_measured | managed | 0.478x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 151.2654 ms | not_measured | managed | 1.124x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.00506 ms | not_measured | managed | 0.971x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 1.2797 ms | not_measured | managed | 0.595x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 107.5354 ms | not_measured | managed | 1.324x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.00596 ms | not_measured | managed | 0.969x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 1.3145 ms | not_measured | managed | 0.600x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 135.0112 ms | not_measured | managed | 0.996x |
| `np.around (float16)` | float16 | 1,000 | 0.00411 ms | not_measured | managed | 1.157x |
| `np.around (float16)` | float16 | 100,000 | 0.3878 ms | not_measured | managed | 0.975x |
| `np.around (float16)` | float16 | 10,000,000 | 39.6165 ms | not_measured | managed | 1.047x |
| `np.around (float32)` | float32 | 1,000 | 0.000569 ms | not_measured | managed | 1.714x |
| `np.around (float32)` | float32 | 100,000 | 0.01114 ms | not_measured | managed | 0.555x |
| `np.around (float32)` | float32 | 10,000,000 | 4.8988 ms | not_measured | managed | 2.298x |
| `np.around (float64)` | float64 | 1,000 | 0.000690 ms | not_measured | managed | 1.371x |
| `np.around (float64)` | float64 | 100,000 | 0.02484 ms | not_measured | managed | 0.479x |
| `np.around (float64)` | float64 | 10,000,000 | 13.7790 ms | not_measured | managed | 1.614x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.02495 ms | not_measured | managed | 0.876x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 1.4573 ms | not_measured | managed | 0.791x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 150.7550 ms | not_measured | managed | 1.359x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.00661 ms | not_measured | managed | 1.005x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 1.5467 ms | not_measured | managed | 0.554x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 155.4739 ms | not_measured | managed | 0.813x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.01912 ms | not_measured | managed | 0.421x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 1.0818 ms | not_measured | managed | 0.857x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 171.7219 ms | not_measured | managed | 0.840x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.02067 ms | not_measured | managed | 0.904x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 2.1223 ms | not_measured | managed | 0.480x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 139.6679 ms | not_measured | managed | 1.218x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.01020 ms | not_measured | managed | 0.479x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 1.1810 ms | not_measured | managed | 0.644x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 109.7355 ms | not_measured | managed | 0.776x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.00604 ms | not_measured | managed | 0.959x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 1.4444 ms | not_measured | managed | 0.548x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 127.8457 ms | not_measured | managed | 0.877x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.02299 ms | not_measured | managed | 0.399x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 1.3547 ms | not_measured | managed | 0.839x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 149.6464 ms | not_measured | managed | 0.790x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.01304 ms | not_measured | managed | 0.427x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 0.8523 ms | not_measured | managed | 0.987x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 125.5387 ms | not_measured | managed | 0.879x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.01512 ms | not_measured | managed | 0.576x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 1.0457 ms | not_measured | managed | 0.962x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 174.7803 ms | not_measured | managed | 0.647x |
| `np.ceil (float16)` | float16 | 1,000 | 0.00348 ms | not_measured | managed | 1.181x |
| `np.ceil (float16)` | float16 | 100,000 | 0.3219 ms | not_measured | managed | 1.157x |
| `np.ceil (float16)` | float16 | 10,000,000 | 32.6788 ms | not_measured | managed | 1.272x |
| `np.ceil (float32)` | float32 | 1,000 | 0.000595 ms | not_measured | managed | 0.684x |
| `np.ceil (float32)` | float32 | 100,000 | 0.01141 ms | not_measured | managed | 0.490x |
| `np.ceil (float32)` | float32 | 10,000,000 | 4.8747 ms | not_measured | managed | 2.310x |
| `np.ceil (float64)` | float64 | 1,000 | 0.000695 ms | not_measured | managed | 0.658x |
| `np.ceil (float64)` | float64 | 100,000 | 0.02535 ms | not_measured | managed | 0.433x |
| `np.ceil (float64)` | float64 | 10,000,000 | 13.3549 ms | not_measured | managed | 1.703x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.00555 ms | not_measured | managed | 2.569x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 0.7486 ms | not_measured | managed | 1.185x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 85.9537 ms | not_measured | managed | 1.420x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.00208 ms | not_measured | managed | 0.932x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.01572 ms | not_measured | managed | 0.490x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 4.8351 ms | not_measured | managed | 2.422x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.00205 ms | not_measured | managed | 0.841x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.02935 ms | not_measured | managed | 0.458x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 15.7261 ms | not_measured | managed | 1.063x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.00119 ms | not_measured | managed | 0.465x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.03194 ms | not_measured | managed | 0.650x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 4.4590 ms | not_measured | managed | 1.301x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.000707 ms | not_measured | managed | 0.769x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.05550 ms | not_measured | managed | 0.341x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 8.5787 ms | not_measured | managed | 1.411x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.00148 ms | not_measured | managed | 0.364x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.06637 ms | not_measured | managed | 0.290x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 12.1990 ms | not_measured | managed | 1.151x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.000733 ms | not_measured | managed | 1.902x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.04269 ms | not_measured | managed | 0.486x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 5.2948 ms | not_measured | managed | 1.081x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.000676 ms | not_measured | managed | 0.803x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.05501 ms | not_measured | managed | 0.344x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 8.0679 ms | not_measured | managed | 1.550x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.000821 ms | not_measured | managed | 0.655x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.06506 ms | not_measured | managed | 0.295x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 18.9365 ms | not_measured | managed | 1.163x |
| `np.cos (float16)` | float16 | 1,000 | 0.01637 ms | not_measured | managed | 0.429x |
| `np.cos (float16)` | float16 | 100,000 | 1.1002 ms | not_measured | managed | 0.864x |
| `np.cos (float16)` | float16 | 10,000,000 | 179.0415 ms | not_measured | managed | 0.750x |
| `np.cos (float32)` | float32 | 1,000 | 0.00182 ms | not_measured | managed | 0.591x |
| `np.cos (float32)` | float32 | 100,000 | 0.06740 ms | not_measured | managed | 1.070x |
| `np.cos (float32)` | float32 | 10,000,000 | 26.5868 ms | not_measured | managed | 0.508x |
| `np.cos (float64)` | float64 | 1,000 | 0.01126 ms | not_measured | managed | 0.432x |
| `np.cos (float64)` | float64 | 100,000 | 1.1912 ms | not_measured | managed | 0.602x |
| `np.cos (float64)` | float64 | 10,000,000 | 81.5039 ms | not_measured | managed | 1.117x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.02108 ms | not_measured | managed | 0.362x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 1.4649 ms | not_measured | managed | 0.620x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 192.8899 ms | not_measured | managed | 0.828x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.00390 ms | not_measured | managed | 0.974x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 1.0486 ms | not_measured | managed | 0.583x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 102.9707 ms | not_measured | managed | 0.662x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.00385 ms | not_measured | managed | 0.935x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 0.9998 ms | not_measured | managed | 0.565x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 95.4693 ms | not_measured | managed | 1.386x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.00594 ms | not_measured | managed | 0.942x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.6647 ms | not_measured | managed | 0.498x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 62.3080 ms | not_measured | managed | 0.709x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.000624 ms | not_measured | managed | 1.731x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.01950 ms | not_measured | managed | 3.764x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 7.7043 ms | not_measured | managed | 2.048x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.000779 ms | not_measured | managed | 1.611x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.02619 ms | not_measured | managed | 3.506x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 13.2237 ms | not_measured | managed | 1.838x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.00698 ms | not_measured | managed | 0.779x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.6651 ms | not_measured | managed | 0.494x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 66.4652 ms | not_measured | managed | 0.709x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.000663 ms | not_measured | managed | 1.630x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.02082 ms | not_measured | managed | 3.524x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 7.4982 ms | not_measured | managed | 1.996x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.00120 ms | not_measured | managed | 1.050x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.04007 ms | not_measured | managed | 2.292x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 14.1193 ms | not_measured | managed | 1.654x |
| `np.exp (float16)` | float16 | 1,000 | 0.00592 ms | not_measured | managed | 0.732x |
| `np.exp (float16)` | float16 | 100,000 | 0.7248 ms | not_measured | managed | 0.555x |
| `np.exp (float16)` | float16 | 10,000,000 | 83.8261 ms | not_measured | managed | 0.514x |
| `np.exp (float32)` | float32 | 1,000 | 0.000873 ms | not_measured | managed | 1.034x |
| `np.exp (float32)` | float32 | 100,000 | 0.05298 ms | not_measured | managed | 1.043x |
| `np.exp (float32)` | float32 | 10,000,000 | 15.2873 ms | not_measured | managed | 2.056x |
| `np.exp (float64)` | float64 | 1,000 | 0.00317 ms | not_measured | managed | 0.892x |
| `np.exp (float64)` | float64 | 100,000 | 0.2551 ms | not_measured | managed | 0.975x |
| `np.exp (float64)` | float64 | 10,000,000 | 47.0292 ms | not_measured | managed | 1.111x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.00950 ms | not_measured | managed | 0.461x |
| `np.exp2 (float16)` | float16 | 100,000 | 1.7794 ms | not_measured | managed | 0.226x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 145.9332 ms | not_measured | managed | 0.493x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.00120 ms | not_measured | managed | 1.671x |
| `np.exp2 (float32)` | float32 | 100,000 | 0.09173 ms | not_measured | managed | 1.793x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 30.2970 ms | not_measured | managed | 0.729x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.00833 ms | not_measured | managed | 0.259x |
| `np.exp2 (float64)` | float64 | 100,000 | 0.8126 ms | not_measured | managed | 0.220x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 134.2793 ms | not_measured | managed | 0.191x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.00871 ms | not_measured | managed | 0.619x |
| `np.expm1 (float16)` | float16 | 100,000 | 1.3252 ms | not_measured | managed | 0.373x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 106.8290 ms | not_measured | managed | 0.822x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.00208 ms | not_measured | managed | 1.481x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.1803 ms | not_measured | managed | 1.424x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 27.6401 ms | not_measured | managed | 1.780x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.00340 ms | not_measured | managed | 1.085x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.2654 ms | not_measured | managed | 1.258x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 36.5600 ms | not_measured | managed | 1.863x |
| `np.floor (float16)` | float16 | 1,000 | 0.00348 ms | not_measured | managed | 1.176x |
| `np.floor (float16)` | float16 | 100,000 | 0.3219 ms | not_measured | managed | 1.161x |
| `np.floor (float16)` | float16 | 10,000,000 | 45.0019 ms | not_measured | managed | 0.897x |
| `np.floor (float32)` | float32 | 1,000 | 0.000641 ms | not_measured | managed | 0.630x |
| `np.floor (float32)` | float32 | 100,000 | 0.01143 ms | not_measured | managed | 0.490x |
| `np.floor (float32)` | float32 | 10,000,000 | 4.8510 ms | not_measured | managed | 2.283x |
| `np.floor (float64)` | float64 | 1,000 | 0.000693 ms | not_measured | managed | 0.671x |
| `np.floor (float64)` | float64 | 100,000 | 0.02529 ms | not_measured | managed | 0.434x |
| `np.floor (float64)` | float64 | 10,000,000 | 12.9801 ms | not_measured | managed | 1.060x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.000467 ms | not_measured | managed | 1.604x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.00148 ms | not_measured | managed | 1.762x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | 0.00154 ms | not_measured | managed | 3.903x |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.000272 ms | not_measured | managed | 1.235x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.00171 ms | not_measured | managed | 2.827x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | 0.00168 ms | not_measured | managed | 11.080x |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.000588 ms | not_measured | managed | 0.587x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.00405 ms | not_measured | managed | 2.298x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | 0.01168 ms | not_measured | managed | 1.622x |
| `np.log (float16)` | float16 | 1,000 | 0.00631 ms | not_measured | managed | 0.714x |
| `np.log (float16)` | float16 | 100,000 | 0.6125 ms | not_measured | managed | 0.679x |
| `np.log (float16)` | float16 | 10,000,000 | 61.4179 ms | not_measured | managed | 0.727x |
| `np.log (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 1.157x |
| `np.log (float32)` | float32 | 100,000 | 0.06167 ms | not_measured | managed | 1.371x |
| `np.log (float32)` | float32 | 10,000,000 | 14.1323 ms | not_measured | managed | 3.145x |
| `np.log (float64)` | float64 | 1,000 | 0.00574 ms | not_measured | managed | 0.460x |
| `np.log (float64)` | float64 | 100,000 | 0.2463 ms | not_measured | managed | 0.932x |
| `np.log (float64)` | float64 | 10,000,000 | 29.7432 ms | not_measured | managed | 1.915x |
| `np.log10 (float16)` | float16 | 1,000 | 0.00855 ms | not_measured | managed | 0.544x |
| `np.log10 (float16)` | float16 | 100,000 | 0.6271 ms | not_measured | managed | 0.697x |
| `np.log10 (float16)` | float16 | 10,000,000 | 62.7468 ms | not_measured | managed | 0.736x |
| `np.log10 (float32)` | float32 | 1,000 | 0.00244 ms | not_measured | managed | 0.917x |
| `np.log10 (float32)` | float32 | 100,000 | 0.2050 ms | not_measured | managed | 0.924x |
| `np.log10 (float32)` | float32 | 10,000,000 | 20.3570 ms | not_measured | managed | 2.271x |
| `np.log10 (float64)` | float64 | 1,000 | 0.00291 ms | not_measured | managed | 0.949x |
| `np.log10 (float64)` | float64 | 100,000 | 0.2585 ms | not_measured | managed | 0.931x |
| `np.log10 (float64)` | float64 | 10,000,000 | 31.0352 ms | not_measured | managed | 1.734x |
| `np.log1p (float16)` | float16 | 1,000 | 0.00796 ms | not_measured | managed | 1.379x |
| `np.log1p (float16)` | float16 | 100,000 | 0.7773 ms | not_measured | managed | 0.680x |
| `np.log1p (float16)` | float16 | 10,000,000 | 77.3480 ms | not_measured | managed | 1.075x |
| `np.log1p (float32)` | float32 | 1,000 | 0.00256 ms | not_measured | managed | 1.215x |
| `np.log1p (float32)` | float32 | 100,000 | 0.2237 ms | not_measured | managed | 1.226x |
| `np.log1p (float32)` | float32 | 10,000,000 | 38.8612 ms | not_measured | managed | 0.829x |
| `np.log1p (float64)` | float64 | 1,000 | 0.00308 ms | not_measured | managed | 1.149x |
| `np.log1p (float64)` | float64 | 100,000 | 0.4860 ms | not_measured | managed | 0.646x |
| `np.log1p (float64)` | float64 | 10,000,000 | 35.0580 ms | not_measured | managed | 1.370x |
| `np.log2 (float16)` | float16 | 1,000 | 0.00777 ms | not_measured | managed | 0.621x |
| `np.log2 (float16)` | float16 | 100,000 | 0.6327 ms | not_measured | managed | 0.705x |
| `np.log2 (float16)` | float16 | 10,000,000 | 62.4123 ms | not_measured | managed | 1.204x |
| `np.log2 (float32)` | float32 | 1,000 | 0.00478 ms | not_measured | managed | 0.465x |
| `np.log2 (float32)` | float32 | 100,000 | 0.1943 ms | not_measured | managed | 0.959x |
| `np.log2 (float32)` | float32 | 10,000,000 | 22.7875 ms | not_measured | managed | 0.990x |
| `np.log2 (float64)` | float64 | 1,000 | 0.00428 ms | not_measured | managed | 0.957x |
| `np.log2 (float64)` | float64 | 100,000 | 0.3923 ms | not_measured | managed | 0.958x |
| `np.log2 (float64)` | float64 | 10,000,000 | 43.9210 ms | not_measured | managed | 1.876x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.00160 ms | not_measured | managed | 1.452x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.07322 ms | not_measured | managed | 3.768x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 8.5345 ms | not_measured | managed | 4.455x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.00410 ms | not_measured | managed | 0.566x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.2882 ms | not_measured | managed | 1.001x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 30.3063 ms | not_measured | managed | 2.101x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.00110 ms | not_measured | managed | 0.551x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.02062 ms | not_measured | managed | 1.256x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 4.3172 ms | not_measured | managed | 1.637x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.000907 ms | not_measured | managed | 0.428x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.01985 ms | not_measured | managed | 0.326x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 8.3978 ms | not_measured | managed | 1.361x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.000801 ms | not_measured | managed | 0.573x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.02724 ms | not_measured | managed | 0.407x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 17.9210 ms | not_measured | managed | 0.811x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.000530 ms | not_measured | managed | 1.047x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.00649 ms | not_measured | managed | 3.196x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 0.7302 ms | not_measured | managed | 8.173x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.000751 ms | not_measured | managed | 0.724x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.01992 ms | not_measured | managed | 0.951x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 2.8289 ms | not_measured | managed | 4.394x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.00119 ms | not_measured | managed | 0.453x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.03774 ms | not_measured | managed | 0.503x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 14.0597 ms | not_measured | managed | 1.656x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.00737 ms | not_measured | managed | 2.521x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.6562 ms | not_measured | managed | 1.201x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 65.5460 ms | not_measured | managed | 2.708x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.00129 ms | not_measured | managed | 1.396x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.07408 ms | not_measured | managed | 1.610x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 8.9499 ms | not_measured | managed | 2.117x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.00421 ms | not_measured | managed | 0.402x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.05918 ms | not_measured | managed | 2.021x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 32.9246 ms | not_measured | managed | 2.073x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.00511 ms | not_measured | managed | 1.956x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.8921 ms | not_measured | managed | 1.146x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 49.2365 ms | not_measured | managed | 4.202x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.00185 ms | not_measured | managed | 1.151x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.01363 ms | not_measured | managed | 10.963x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 9.1350 ms | not_measured | managed | 2.053x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.00156 ms | not_measured | managed | 1.360x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.04142 ms | not_measured | managed | 3.594x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 13.7492 ms | not_measured | managed | 2.008x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.01879 ms | not_measured | managed | 1.493x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 3.8965 ms | not_measured | managed | 0.373x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 379.4934 ms | not_measured | managed | 0.645x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.00623 ms | not_measured | managed | 0.872x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 0.6581 ms | not_measured | managed | 0.995x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 115.3744 ms | not_measured | managed | 0.781x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.01011 ms | not_measured | managed | 0.914x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 1.9779 ms | not_measured | managed | 0.523x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 199.7576 ms | not_measured | managed | 0.900x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.00702 ms | not_measured | managed | 0.775x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.6655 ms | not_measured | managed | 0.492x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 37.1878 ms | not_measured | managed | 1.256x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.000707 ms | not_measured | managed | 1.525x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.01912 ms | not_measured | managed | 3.837x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 7.2281 ms | not_measured | managed | 1.974x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.00127 ms | not_measured | managed | 1.001x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.04606 ms | not_measured | managed | 1.994x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 13.6580 ms | not_measured | managed | 1.832x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.00587 ms | not_measured | managed | 0.727x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.3683 ms | not_measured | managed | 0.913x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 45.8013 ms | not_measured | managed | 1.059x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.000743 ms | not_measured | managed | 1.450x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.01906 ms | not_measured | managed | 3.848x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 7.3224 ms | not_measured | managed | 2.048x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.00109 ms | not_measured | managed | 1.153x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.04730 ms | not_measured | managed | 1.943x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 13.5001 ms | not_measured | managed | 1.287x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 1,000 | 0.000001 ms | not_measured | managed | 142.000x |
| `np.real(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 209.000x |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.00766 ms | not_measured | managed | 0.311x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.5453 ms | not_measured | managed | 0.370x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 66.5162 ms | not_measured | managed | 0.334x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.000627 ms | not_measured | managed | 0.766x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.06227 ms | not_measured | managed | 0.227x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 7.5816 ms | not_measured | managed | 1.463x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.00266 ms | not_measured | managed | 0.265x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.1937 ms | not_measured | managed | 0.191x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 12.8857 ms | not_measured | managed | 2.068x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.00789 ms | not_measured | managed | 0.682x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.5066 ms | not_measured | managed | 0.892x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 56.9657 ms | not_measured | managed | 1.171x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.000834 ms | not_measured | managed | 0.484x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.01239 ms | not_measured | managed | 0.450x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 7.5998 ms | not_measured | managed | 1.409x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.00116 ms | not_measured | managed | 0.383x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.04071 ms | not_measured | managed | 0.329x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 21.0898 ms | not_measured | managed | 0.655x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.00802 ms | not_measured | managed | 1.093x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.7878 ms | not_measured | managed | 0.578x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 65.6221 ms | not_measured | managed | 1.046x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.000907 ms | not_measured | managed | 0.993x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.01216 ms | not_measured | managed | 0.507x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 7.6658 ms | not_measured | managed | 1.468x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.00115 ms | not_measured | managed | 0.853x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.03998 ms | not_measured | managed | 0.362x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 20.8635 ms | not_measured | managed | 0.804x |
| `np.sign (float16)` | float16 | 1,000 | 0.00368 ms | not_measured | managed | 0.327x |
| `np.sign (float16)` | float16 | 100,000 | 0.6316 ms | not_measured | managed | 0.133x |
| `np.sign (float16)` | float16 | 10,000,000 | 70.8486 ms | not_measured | managed | 0.144x |
| `np.sign (float32)` | float32 | 1,000 | 0.00182 ms | not_measured | managed | 0.538x |
| `np.sign (float32)` | float32 | 100,000 | 0.3788 ms | not_measured | managed | 0.762x |
| `np.sign (float32)` | float32 | 10,000,000 | 39.3311 ms | not_measured | managed | 0.974x |
| `np.sign (float64)` | float64 | 1,000 | 0.00192 ms | not_measured | managed | 0.496x |
| `np.sign (float64)` | float64 | 100,000 | 0.3872 ms | not_measured | managed | 0.731x |
| `np.sign (float64)` | float64 | 10,000,000 | 64.0649 ms | not_measured | managed | 0.792x |
| `np.sin (float16)` | float16 | 1,000 | 0.00798 ms | not_measured | managed | 0.820x |
| `np.sin (float16)` | float16 | 100,000 | 1.1082 ms | not_measured | managed | 0.883x |
| `np.sin (float16)` | float16 | 10,000,000 | 112.1327 ms | not_measured | managed | 0.913x |
| `np.sin (float32)` | float32 | 1,000 | 0.000960 ms | not_measured | managed | 1.079x |
| `np.sin (float32)` | float32 | 100,000 | 0.06945 ms | not_measured | managed | 1.008x |
| `np.sin (float32)` | float32 | 10,000,000 | 26.5999 ms | not_measured | managed | 0.424x |
| `np.sin (float64)` | float64 | 1,000 | 0.01122 ms | not_measured | managed | 0.383x |
| `np.sin (float64)` | float64 | 100,000 | 1.1831 ms | not_measured | managed | 0.583x |
| `np.sin (float64)` | float64 | 10,000,000 | 103.0694 ms | not_measured | managed | 0.786x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.01062 ms | not_measured | managed | 1.529x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 2.1727 ms | not_measured | managed | 0.437x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 118.1813 ms | not_measured | managed | 1.326x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.01053 ms | not_measured | managed | 0.341x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 0.6377 ms | not_measured | managed | 0.968x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 110.4924 ms | not_measured | managed | 0.657x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.01206 ms | not_measured | managed | 0.358x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 1.2836 ms | not_measured | managed | 0.478x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 115.5146 ms | not_measured | managed | 1.198x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.00350 ms | not_measured | managed | 1.039x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.3956 ms | not_measured | managed | 0.877x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 36.4867 ms | not_measured | managed | 1.018x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.00103 ms | not_measured | managed | 0.461x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.01727 ms | not_measured | managed | 0.817x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 5.0760 ms | not_measured | managed | 2.356x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.00107 ms | not_measured | managed | 0.817x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.05963 ms | not_measured | managed | 0.924x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 13.0795 ms | not_measured | managed | 2.838x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.00865 ms | not_measured | managed | 0.275x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.4416 ms | not_measured | managed | 0.456x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 43.8654 ms | not_measured | managed | 1.033x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.000689 ms | not_measured | managed | 0.575x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.01325 ms | not_measured | managed | 0.422x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 7.5996 ms | not_measured | managed | 1.474x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.00119 ms | not_measured | managed | 0.369x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.04625 ms | not_measured | managed | 0.235x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 18.7357 ms | not_measured | managed | 1.192x |
| `np.tan (float16)` | float16 | 1,000 | 0.00839 ms | not_measured | managed | 0.812x |
| `np.tan (float16)` | float16 | 100,000 | 1.8860 ms | not_measured | managed | 0.503x |
| `np.tan (float16)` | float16 | 10,000,000 | 164.4258 ms | not_measured | managed | 0.926x |
| `np.tan (float32)` | float32 | 1,000 | 0.00700 ms | not_measured | managed | 0.462x |
| `np.tan (float32)` | float32 | 100,000 | 1.1105 ms | not_measured | managed | 0.584x |
| `np.tan (float32)` | float32 | 10,000,000 | 109.5687 ms | not_measured | managed | 0.665x |
| `np.tan (float64)` | float64 | 1,000 | 0.00461 ms | not_measured | managed | 0.961x |
| `np.tan (float64)` | float64 | 100,000 | 0.8440 ms | not_measured | managed | 0.983x |
| `np.tan (float64)` | float64 | 10,000,000 | 93.2385 ms | not_measured | managed | 1.004x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.01999 ms | not_measured | managed | 0.898x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 2.0453 ms | not_measured | managed | 0.483x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 139.6387 ms | not_measured | managed | 1.254x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.00110 ms | not_measured | managed | 1.435x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.07095 ms | not_measured | managed | 1.721x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 36.7742 ms | not_measured | managed | 0.442x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.00350 ms | not_measured | managed | 1.767x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.6628 ms | not_measured | managed | 0.861x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 85.4104 ms | not_measured | managed | 1.769x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.00632 ms | not_measured | managed | 0.636x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.3258 ms | not_measured | managed | 1.184x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 60.0314 ms | not_measured | managed | 0.773x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.000728 ms | not_measured | managed | 0.556x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.01995 ms | not_measured | managed | 0.281x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 5.1825 ms | not_measured | managed | 2.102x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.00114 ms | not_measured | managed | 0.401x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.04045 ms | not_measured | managed | 0.269x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 13.0702 ms | not_measured | managed | 1.055x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-29 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.46× geomean · 69%🕐 of NumPy's time · 108 win / 57 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████████▋ ....   1.46×    68%🕐  ( 20 win / 13 lose)
1K         ████████████████▌ ..   1.65×    60%🕐  ( 26 win /  7 lose)
100K       ██████████████▎ ....   1.43×    70%🕐  ( 19 win / 14 lose)
1M         ███████████████▍ ...   1.55×    65%🕐  ( 21 win / 12 lose)
10M        ████████████▎ ......   1.23×    82%🕐  ( 22 win / 11 lose)
ALL        ██████████████▌ ....   1.46×    69%🕐  (108 win / 57 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████▋ .......   1.16×    86%🕐  ( 28 win / 12 lose)
reductions ███████████████████▶   2.64×    38%🕐  ( 32 win /  8 lose)
selection  ████████████████▍ ..   1.65×    61%🕐  ( 22 win / 13 lose)
copy/cast  █████████▉ .........   0.99×   101%🕐  ( 16 win /  9 lose)  ◄ PARITY
index-math ██████████▎ ........   1.04×    97%🕐  (  6 win /  4 lose)
dtypes     █████████▋ .........   0.96×   104%🕐  (  4 win / 11 lose)  ◄ SLOWER

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.38×    1.54×    1.28×    1.11×    0.71×
reductions      6.51×    3.42×    1.95×    1.55×    1.91×
selection       0.81×    1.86×    1.83×    2.60×    1.70×
copy/cast       0.80×    0.88×    0.70×    1.66×    1.19×
index-math      0.85×    1.12×    1.08×    1.23×    0.94×
dtypes          0.50×    0.81×    1.82×    1.18×    0.95×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          0.98×    1.53×    0.91×    1.13×    0.33×     0.87×
  sqrt         1.08×    1.16×    5.11×    1.00×    0.23×     1.08×
  copy         1.57×    2.35×    1.27×    1.15×    1.72×     1.56×
  strided      1.65×    1.33×    1.00×    0.97×    1.01×     1.17×
  bcast        1.62×    1.22×    0.98×    0.67×    0.92×     1.04×
  reversed     1.63×    1.26×    0.64×    0.65×    1.00×     0.97×
  castbuf      1.31×    1.34×    1.55×    2.59×    1.04×     1.49×
  mixbuf       1.37×    2.79×    1.25×    1.56×    0.50×     1.30×
-- reductions
  sum          3.15×    4.66×    3.54×    2.08×    1.93×     2.91×
  sum ax0      8.95×    1.13×    1.05×    0.95×    0.98×     1.58×
  sum ax1      9.46×    1.60×    0.86×    1.51×    1.62×     2.00×
  sum dt=      2.90×    1.96×    0.56×    1.05×    1.10×     1.30×
  amin         8.97×    1.23×    0.81×    0.73×    0.93×     1.43×
  cumsum       4.43×    1.68×    1.88×    2.50×    1.75×     2.28×
  any(F)       8.97×   23.27×    3.37×    0.68×    1.38×     3.66×
  any(hit)    11.71×   23.13×   23.18×    8.57×   23.53×    16.61×
-- selection
  where        0.94×    1.62×    1.67×    1.93×    1.55×     1.50×
  a[mask]      0.25×    2.91×    3.31×    4.82×    2.12×     1.90×
  a[mask]=     0.22×    4.53×    8.34×   16.46×    5.90×     3.82×
  count_nz     0.67×    2.64×    4.04×    3.60×    1.23×     2.00×
  argwhere     8.39×    6.28×    1.67×    3.90×    3.13×     4.04×
  a[idx]       0.65×    0.46×    0.46×    0.68×    0.89×     0.61×
  a[idx]=      1.18×    0.47×    0.48×    0.55×    0.62×     0.62×
-- copy/cast
  flatten      1.07×    1.00×    0.60×    3.24×    1.19×     1.20×
  astype       0.32×    1.14×    1.54×    3.28×    1.48×     1.22×
  ravel.T      2.16×    1.37×    0.89×    1.93×    1.25×     1.45×
  in-place     0.62×    0.70×    1.14×    1.02×    1.05×     0.88×
  less->b      0.71×    0.49×    0.18×    0.60×    1.02×     0.52×
-- index-math
  unravel      0.56×    0.86×    1.08×    1.00×    1.06×     0.89×
  ravel_mi     1.29×    1.45×    1.09×    1.51×    0.84×     1.21×
-- dtypes
  complex      0.33×    0.63×    0.91×    0.73×    0.53×     0.59×
  float16      0.61×    0.60×    0.54×    0.34×    0.34×     0.47×
  int8         0.63×    1.43×   12.16×    6.62×    4.76×     3.22×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████▊ .......   1.18×    84%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.09×    24%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   3.30×    30%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.47×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.05×    49%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.71×    17%🕐  (  1 win /  0 lose)
4d           ██████████████████▍    1.84×    54%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.24×    45%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.18×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.73×    37%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████▊ ..............   0.48×   207%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████▏ ........   1.02×    98%🕐  (  1 win /  0 lose)  ◄ PARITY
w=64         ██████████ .........   1.00×   100%🕐  (  1 win /  0 lose)  ◄ PARITY
w=256        ████████████▎ ......   1.23×    81%🕐  (  1 win /  0 lose)
w=1024       █████████████▉ .....   1.40×    72%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    936.06×   (936.1× faster, faster)
  allocate          1.04×   (1.0× faster, faster)
  overlap_copy      1.83×   (1.8× faster, faster)
  forder_out        1.24×   (1.2× faster, faster)
  zerodim           0.62×   (1.6× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.16×    3.89×    1.44×    1.66×    1.93×   vs chained 6× add
reuse            6.16×    6.41×    1.03×    1.01×    1.02×   vs rebuild each call
par8                 -    0.74×    3.36×    6.03×    2.94×   vs single-thread

biggest NumSharp wins: anyeh@10M 23.53× · anyff@1K 23.27× · anyeh@100K 23.18× · anyeh@1K 23.13× · bassign@1M 16.46×
most behind:           lessbool@100K 0.18× · bassign@1 0.22× · sqrt@10M 0.23× · bread@1 0.25× · astype@1 0.32×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.10 ✅ | 1.11 ✅ | 1.10 ✅ | 0.60 🟡 | 1.03 ✅ | 0.73 🟡 | 0.91 🟡 | 0.83 🟡 |
| 1M | 1.05 ✅ | 1.01 ✅ | 0.98 🟡 | 0.63 🟡 | 1.02 ✅ | 0.72 🟡 | 0.81 🟡 | 0.75 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.11 ✅ | 1.30 ✅ | 0.97 🟡 | 0.12 🔴 | 1.05 ✅ | 1.50 ✅ | 1.14 ✅ |
| 1M | 1.02 ✅ | 1.28 ✅ | 0.96 🟡 | 0.11 🔴 | 1.07 ✅ | 1.31 ✅ | 1.07 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.95 🟡 | 0.74 🟡 | 0.76 🟡 | 1.34 ✅ |
| 1M | 0.95 🟡 | 0.67 🟡 | 0.68 🟡 | 1.32 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.3695 | 0.1408 | 0.02 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.1086 | 0.1125 | 0.03 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7283 | 0.0204 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.5294 | 0.2240 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.5579 | 0.2255 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7326 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7286 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7284 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7282 | 0.0230 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.4363 | 0.2417 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.2451 | 0.2436 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7190 | 0.0248 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.5333 | 0.2184 | 0.04 🔴 |
| 100K\|dec\|negrow\|sum\|ax1 | 0.4159 | 0.0184 | 0.04 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 1.1684 | 0.0603 | 0.05 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 3.78 ✅ | 2.96 ✅ | 3.47 ✅ | 1.55 ✅ | 3.76 ✅ | 3.95 ✅ | 3.58 ✅ | 4.26 ✅ |
| 1M | 6.83 ✅ | 6.29 ✅ | 6.51 ✅ | 3.32 ✅ | 5.41 ✅ | 6.04 ✅ | 6.47 ✅ | 9.24 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 5.43 ✅ | 4.13 ✅ | 4.31 ✅ | 4.58 ✅ | 2.80 ✅ | 2.76 ✅ | 1.87 ✅ | 1.51 ✅ | 4.72 ✅ | 4.89 ✅ | 1.73 ✅ | 1.28 ✅ | 10.68 ✅ |
| 1M | 10.08 ✅ | 10.08 ✅ | 8.28 ✅ | 8.35 ✅ | 4.46 ✅ | 4.98 ✅ | 3.65 ✅ | 4.39 ✅ | 7.96 ✅ | 7.65 ✅ | 4.17 ✅ | 3.93 ✅ | 5.93 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|pos | 0.0285 | 0.0159 | 0.56 🟡 |
| 100K\|f64\|strided\|pos | 0.0312 | 0.0180 | 0.58 🟡 |
| 100K\|i32\|strided\|pos | 0.0286 | 0.0197 | 0.69 🟡 |
| 100K\|u32\|strided\|pos | 0.0285 | 0.0197 | 0.69 🟡 |
| 100K\|f64\|T\|pos | 0.0310 | 0.0277 | 0.89 🟡 |
| 100K\|i64\|strided\|pos | 0.0317 | 0.0299 | 0.94 🟡 |
| 100K\|u64\|strided\|pos | 0.0312 | 0.0298 | 0.96 🟡 |
| 100K\|f64\|C\|pos | 0.0281 | 0.0277 | 0.98 🟡 |
| 100K\|i64\|F\|pos | 0.0370 | 0.0364 | 0.98 🟡 |
| 100K\|f64\|F\|pos | 0.0305 | 0.0329 | 1.08 ✅ |
| 100K\|u64\|F\|pos | 0.0317 | 0.0364 | 1.15 ✅ |
| 100K\|u64\|T\|pos | 0.0310 | 0.0365 | 1.18 ✅ |
| 100K\|u64\|C\|pos | 0.0284 | 0.0364 | 1.28 ✅ |
| 1M\|i32\|strided\|pos | 0.2711 | 0.3486 | 1.29 ✅ |
| 100K\|i8\|strided\|pos | 0.0089 | 0.0117 | 1.31 ✅ |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.26 ✅ | 1.19 ✅ | 1.15 ✅ | 0.54 🟡 | 1.59 ✅ | 1.55 ✅ | 1.13 ✅ | 1.46 ✅ |
| 1M | 2.15 ✅ | 1.96 ✅ | 1.97 ✅ | 0.99 🟡 | 2.38 ✅ | 2.40 ✅ | 1.49 ✅ | 2.89 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 1.07 ✅ | 0.86 🟡 | 2.61 ✅ | 0.91 🟡 | 1.27 ✅ | 0.98 🟡 |
| 1M | 2.14 ✅ | 2.34 ✅ | 1.93 ✅ | 1.12 ✅ | 2.40 ✅ | 2.05 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 1.44 ✅ | 1.39 ✅ | 1.30 ✅ | 1.20 ✅ | 0.93 🟡 | 0.87 🟡 | 1.26 ✅ |
| 1M | 2.18 ✅ | 2.17 ✅ | 2.80 ✅ | 2.44 ✅ | 1.27 ✅ | 0.88 🟡 | 2.88 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|add | 0.1828 | 0.0256 | 0.14 🔴 |
| 100K\|f64\|strided\|add | 0.1900 | 0.0287 | 0.15 🔴 |
| 100K\|f64\|strided\|mul | 0.1833 | 0.0301 | 0.16 🔴 |
| 100K\|i64\|strided\|mul | 0.1831 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|neg | 0.0943 | 0.0163 | 0.17 🔴 |
| 100K\|f64\|strided\|abs | 0.0931 | 0.0165 | 0.18 🔴 |
| 100K\|f64\|T\|sqrt | 0.2797 | 0.0552 | 0.20 🔴 |
| 100K\|f32\|strided\|abs | 0.0492 | 0.0098 | 0.20 🟠 |
| 100K\|i64\|strided\|sqrt | 0.1752 | 0.0408 | 0.23 🟠 |
| 100K\|i32\|bcast\|sqrt | 0.3200 | 0.0804 | 0.25 🟠 |
| 100K\|f32\|negcol\|add | 0.1845 | 0.0494 | 0.27 🟠 |
| 100K\|i32\|F\|sqrt | 0.2797 | 0.0778 | 0.28 🟠 |
| 100K\|i32\|strided\|add | 0.0932 | 0.0279 | 0.30 🟠 |
| 100K\|i32\|strided\|mul | 0.0930 | 0.0279 | 0.30 🟠 |
| 100K\|f32\|strided\|add | 0.0933 | 0.0286 | 0.31 🟠 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 4.48 ✅ | 5.87 ✅ | 0.75 🟡 | 4.42 ✅ | 3.33 ✅ | 1.66 ✅ | 2.80 ✅ |
| 1-D strided a[::2] | 3.80 ✅ | 4.13 ✅ | 1.29 ✅ | 0.57 🟡 | 0.57 🟡 | 2.04 ✅ | 1.54 ✅ |
| 1-D reversed a[::-1] | 4.67 ✅ | 4.38 ✅ | 0.69 🟡 | 0.53 🟡 | 0.48 🟠 | 1.65 ✅ | 1.35 ✅ |
| array + scalar | 5.52 ✅ | 4.46 ✅ | 0.73 🟡 | 4.60 ✅ | 3.04 ✅ | 1.79 ✅ | 2.77 ✅ |
| scalar + array | 5.55 ✅ | 8.52 ✅ | 0.74 🟡 | 4.63 ✅ | 3.50 ✅ | 1.69 ✅ | 3.14 ✅ |
| mixed C + F | 2.86 ✅ | 3.89 ✅ | 0.77 🟡 | 0.93 🟡 | 0.74 🟡 | 1.77 ✅ | 1.48 ✅ |
| mixed C + T | 3.18 ✅ | 0.89 🟡 | 0.77 🟡 | 0.91 🟡 | 0.75 🟡 | 2.40 ✅ | 1.23 ✅ |
| binary broadcast +row(1,C) | 5.75 ✅ | 3.01 ✅ | 0.74 🟡 | 4.97 ✅ | 3.84 ✅ | 1.71 ✅ | 2.73 ✅ |
| binary broadcast +col(R,1) | 5.55 ✅ | 3.06 ✅ | 0.69 🟡 | 5.42 ✅ | 6.77 ✅ | 2.60 ✅ | 3.23 ✅ |
| col-broadcast unary (inner stride-0) | 5.12 ✅ | 1.98 ✅ | 0.95 🟡 | 2.27 ✅ | 3.42 ✅ | 5.45 ✅ | 2.72 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_rev|i64 | 3.6240 | 1.7425 | 0.48 🟠 |
| 1d_rev|i32 | 1.8063 | 0.9642 | 0.53 🟡 |
| 1d_strided|i64 | 1.8131 | 1.0274 | 0.57 🟡 |
| 1d_strided|i32 | 0.9047 | 0.5148 | 0.57 🟡 |
| 1d_rev|f16 | 9.7203 | 6.6619 | 0.69 🟡 |
| bcast_col|f16 | 9.5953 | 6.6540 | 0.69 🟡 |
| scalar_rhs|f16 | 9.0203 | 6.5493 | 0.73 🟡 |
| bcast_row|f16 | 9.1483 | 6.7400 | 0.74 🟡 |
| scalar_lhs|f16 | 8.9600 | 6.6276 | 0.74 🟡 |
| mix_C_F|i64 | 3.9903 | 2.9637 | 0.74 🟡 |
| 1d_C|f16 | 8.9710 | 6.6916 | 0.75 🟡 |
| mix_C_T|i64 | 3.7170 | 2.7903 | 0.75 🟡 |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **381 / 1568** comparable cells lag (<1.0); **1187** win (≥1.0).
- **🔴 <0.2** — 26 cells. Top: 14× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 5× int → sub-word (narrow); 3× bool → f16
- **🟠 0.2–0.5** — 156 cells. Top: 67× int → sub-word (narrow); 24× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 15× * → bool
- **🟡 0.5–1.0** — 199 cells. Top: 32× int → sub-word (narrow); 27× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 8× f16 → u64

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **1.62**, `f64`→narrow **1.16**, `f16`→narrow **3.67**, `c128`→narrow **0.88**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 3.27 ✅ | 3.18 ✅ | 3.27 ✅ | 2.88 ✅ | 2.98 ✅ | 1.76 ✅ | 1.53 ✅ | 0.43 🟠 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.98 ✅ | 3.30 ✅ | 3.17 ✅ | 3.43 ✅ | 3.49 ✅ | 1.97 ✅ | 1.99 ✅ | 1.36 ✅ | 1.31 ✅ | 3.01 ✅ | 2.38 ✅ | 1.56 ✅ | 1.32 ✅ | nan ? | 1.05 ✅ |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.94 ✅ | 1.68 ✅ | 1.68 ✅ | 2.27 ✅ | 2.33 ✅ | 2.41 ✅ | 1.85 ✅ | 2.44 ✅ | 2.01 ✅ | 2.27 ✅ | 1.59 ✅ | 1.96 ✅ | 2.39 ✅ | nan ? | 2.51 ✅ |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.61🟡 | 7.31✅ | 7.34✅ | 8.93✅ | 9.52✅ | 7.84✅ | 7.83✅ | 7.55✅ | 7.11✅ | 7.15✅ | 0.25🟠 | 8.07✅ | 7.43✅ | — | 5.93✅ |
| **u8** | 5.00✅ | 0.74🟡 | 10.54✅ | 9.68✅ | 10.47✅ | 5.89✅ | 6.23✅ | 4.24✅ | 6.99✅ | 5.11✅ | 2.86✅ | 1.95✅ | 3.00✅ | — | 3.57✅ |
| **i8** | 11.65✅ | 8.73✅ | 1.64✅ | 6.04✅ | 7.70✅ | 4.82✅ | 4.67✅ | 8.25✅ | 3.93✅ | 7.74✅ | 2.88✅ | 1.95✅ | 3.01✅ | — | 3.45✅ |
| **i16** | 8.38✅ | 7.80✅ | 10.52✅ | 3.85✅ | 6.95✅ | 5.38✅ | 5.15✅ | 4.83✅ | 3.81✅ | 7.30✅ | 2.89✅ | 6.11✅ | 2.96✅ | — | 3.20✅ |
| **u16** | 8.43✅ | 10.58✅ | 9.20✅ | 4.36✅ | 3.71✅ | 9.34✅ | 4.58✅ | 4.74✅ | 5.03✅ | 3.92✅ | 2.88✅ | 5.91✅ | 2.96✅ | — | 3.42✅ |
| **i32** | 3.33✅ | 1.01✅ | 0.90🟡 | 4.16✅ | 5.39✅ | 3.42✅ | 5.44✅ | 4.23✅ | 4.31✅ | 4.91✅ | 3.00✅ | 2.43✅ | 4.11✅ | — | 2.51✅ |
| **u32** | 1.23✅ | 1.45✅ | 0.90🟡 | 4.33✅ | 5.37✅ | 5.22✅ | 3.59✅ | 4.23✅ | 4.34✅ | 5.08✅ | 2.73✅ | 1.87✅ | 2.88✅ | — | 2.90✅ |
| **i64** | 1.21✅ | 1.34✅ | 0.89🟡 | 1.79✅ | 1.86✅ | 3.07✅ | 6.35✅ | 2.63✅ | 4.49✅ | 2.23✅ | 1.43✅ | 1.66✅ | 2.97✅ | — | 2.45✅ |
| **u64** | 1.21✅ | 1.03✅ | 1.33✅ | 1.48✅ | 1.75✅ | 3.24✅ | 2.93✅ | 4.45✅ | 2.55✅ | 1.59✅ | 0.96🟡 | 1.50✅ | 1.77✅ | — | 2.30✅ |
| **char** | 8.37✅ | 10.77✅ | 10.72✅ | 7.39✅ | 4.61✅ | 2.32✅ | 2.13✅ | 2.21✅ | 2.29✅ | 3.58✅ | 10.67✅ | 3.28✅ | 2.30✅ | — | 5.38✅ |
| **f16** | 17.29✅ | 4.34✅ | 4.65✅ | 5.44✅ | 5.87✅ | 6.92✅ | 4.42✅ | 3.29✅ | 0.72🟡 | 5.71✅ | 3.47✅ | 3.16✅ | 0.88🟡 | — | 1.23✅ |
| **f32** | 4.93✅ | 2.75✅ | 2.71✅ | 5.25✅ | 6.18✅ | 4.65✅ | 0.71🟡 | 0.89🟡 | 0.57🟡 | 5.03✅ | 3.06✅ | 3.51✅ | 4.51✅ | — | 3.04✅ |
| **f64** | 2.26✅ | 2.62✅ | 2.60✅ | 3.28✅ | 2.91✅ | 3.26✅ | 0.97🟡 | 1.07✅ | 0.61🟡 | 2.53✅ | 4.95✅ | 5.40✅ | 3.06✅ | — | 1.88✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.86🟡 | 1.51✅ | 1.51✅ | 1.89✅ | 1.97✅ | 2.14✅ | 1.12✅ | 1.12✅ | 0.62🟡 | 2.05✅ | 1.52✅ | 1.66✅ | 1.77✅ | — | 6.39✅ |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 4.39✅ | 15.91✅ | 8.64✅ | 8.94✅ | 15.51✅ | 7.49✅ | 6.23✅ | 5.36✅ | 5.38✅ | 9.29✅ | 0.29🟠 | 6.61✅ | 5.07✅ | — | 3.59✅ |
| **u8** | 10.68✅ | 1.43✅ | 12.29✅ | 10.07✅ | 8.19✅ | 5.91✅ | 5.77✅ | 4.41✅ | 4.46✅ | 9.37✅ | 2.87✅ | 1.88✅ | 2.50✅ | — | 2.23✅ |
| **i8** | 11.47✅ | 8.15✅ | 1.69✅ | 9.64✅ | 6.67✅ | 5.80✅ | 5.78✅ | 3.95✅ | 3.66✅ | 8.03✅ | 2.90✅ | 1.90✅ | 2.96✅ | — | 3.53✅ |
| **i16** | 8.39✅ | 3.52✅ | 3.61✅ | 3.40✅ | 6.73✅ | 5.65✅ | 5.11✅ | 4.70✅ | 5.08✅ | 6.12✅ | 2.81✅ | 6.46✅ | 2.22✅ | — | 3.41✅ |
| **u16** | 8.43✅ | 3.61✅ | 10.58✅ | 4.51✅ | 3.75✅ | 5.31✅ | 5.25✅ | 4.82✅ | 4.85✅ | 3.41✅ | 2.83✅ | 5.70✅ | 3.01✅ | — | 3.36✅ |
| **i32** | 3.37✅ | 1.03✅ | 0.90🟡 | 4.01✅ | 5.65✅ | 3.53✅ | 5.44✅ | 4.36✅ | 4.30✅ | 5.11✅ | 3.02✅ | 2.44✅ | 3.95✅ | — | 3.00✅ |
| **u32** | 3.33✅ | 1.31✅ | 0.90🟡 | 4.09✅ | 5.12✅ | 5.29✅ | 6.34✅ | 6.40✅ | 4.42✅ | 5.11✅ | 2.71✅ | 1.86✅ | 5.51✅ | — | 2.96✅ |
| **i64** | 1.21✅ | 1.35✅ | 0.47🟠 | 1.80✅ | 1.70✅ | 3.02✅ | 3.17✅ | 3.26✅ | 4.73✅ | 2.32✅ | 1.44✅ | 2.38✅ | 2.96✅ | — | 2.24✅ |
| **u64** | 1.21✅ | 1.03✅ | 1.30✅ | 1.46✅ | 1.15✅ | 2.05✅ | 2.16✅ | 4.56✅ | 3.30✅ | 2.34✅ | 1.80✅ | 1.52✅ | 1.79✅ | — | 2.12✅ |
| **char** | 3.41✅ | 10.38✅ | 11.62✅ | 5.90✅ | 6.66✅ | 2.37✅ | 2.29✅ | 3.37✅ | 3.44✅ | 3.37✅ | 2.84✅ | 1.90✅ | 2.78✅ | — | 1.40✅ |
| **f16** | 17.23✅ | 2.26✅ | 2.26✅ | 5.36✅ | 5.75✅ | 7.08✅ | 1.46✅ | 3.32✅ | 0.72🟡 | 5.72✅ | 3.65✅ | 3.19✅ | 0.93🟡 | — | 1.22✅ |
| **f32** | 4.93✅ | 2.74✅ | 2.71✅ | 4.76✅ | 4.99✅ | 4.64✅ | 0.72🟡 | 0.90🟡 | 0.58🟡 | 5.01✅ | 3.04✅ | 3.37✅ | 4.23✅ | — | 2.10✅ |
| **f64** | 2.23✅ | 2.63✅ | 2.62✅ | 3.28✅ | 2.65✅ | 3.19✅ | 0.96🟡 | 1.04✅ | 0.60🟡 | 2.61✅ | 4.64✅ | 2.87✅ | 5.74✅ | — | 1.69✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.72🟡 | 1.11✅ | 1.51✅ | 1.96✅ | 1.93✅ | 3.88✅ | 1.13✅ | 1.17✅ | 0.66🟡 | 1.93✅ | 1.50✅ | 1.78✅ | 2.06✅ | — | 3.15✅ |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.23✅ | 9.99✅ | 11.36✅ | 9.60✅ | 7.85✅ | 6.26✅ | 6.66✅ | 5.41✅ | 9.87✅ | 8.71✅ | 0.30🟠 | 6.89✅ | 5.08✅ | — | 2.85✅ |
| **u8** | 10.79✅ | 2.28✅ | 11.33✅ | 9.68✅ | 8.06✅ | 5.78✅ | 5.86✅ | 4.55✅ | 4.26✅ | 13.74✅ | 10.38✅ | 1.90✅ | 2.97✅ | — | 3.66✅ |
| **i8** | 11.61✅ | 8.43✅ | 1.72✅ | 9.62✅ | 8.01✅ | 5.66✅ | 5.40✅ | 3.84✅ | 3.91✅ | 8.00✅ | 2.87✅ | 1.89✅ | 2.71✅ | — | 3.34✅ |
| **i16** | 8.32✅ | 6.75✅ | 9.36✅ | 3.54✅ | 7.24✅ | 5.83✅ | 5.59✅ | 4.80✅ | 4.88✅ | 6.31✅ | 2.88✅ | 5.78✅ | 2.91✅ | — | 3.23✅ |
| **u16** | 8.38✅ | 10.53✅ | 9.94✅ | 6.40✅ | 4.32✅ | 5.26✅ | 3.92✅ | 6.60✅ | 5.07✅ | 3.43✅ | 2.91✅ | 5.75✅ | 2.98✅ | — | 3.51✅ |
| **i32** | 3.30✅ | 0.95🟡 | 0.86🟡 | 4.18✅ | 5.56✅ | 3.66✅ | 5.45✅ | 4.19✅ | 4.29✅ | 5.09✅ | 1.57✅ | 2.53✅ | 4.16✅ | — | 3.09✅ |
| **u32** | 3.27✅ | 1.44✅ | 0.90🟡 | 4.22✅ | 10.09✅ | 5.72✅ | 3.82✅ | 3.28✅ | 3.57✅ | 4.80✅ | 2.70✅ | 1.87✅ | 2.92✅ | — | 2.91✅ |
| **i64** | 1.21✅ | 1.36✅ | 0.87🟡 | 1.76✅ | 1.81✅ | 2.72✅ | 2.73✅ | 3.17✅ | 4.53✅ | 2.14✅ | 1.46✅ | 2.24✅ | 2.56✅ | — | 3.00✅ |
| **u64** | 1.21✅ | 1.03✅ | 1.27✅ | 1.00🟡 | 1.15✅ | 3.27✅ | 3.00✅ | 4.49✅ | 3.17✅ | 2.28✅ | 1.83✅ | 1.51✅ | 1.79✅ | — | 2.22✅ |
| **char** | 8.36✅ | 10.71✅ | 9.34✅ | 6.52✅ | 4.13✅ | 2.44✅ | 2.35✅ | 6.94✅ | 3.54✅ | 3.52✅ | 2.89✅ | 1.86✅ | 2.64✅ | — | 2.17✅ |
| **f16** | 17.18✅ | 4.33✅ | 4.66✅ | 5.42✅ | 5.86✅ | 7.12✅ | 1.48✅ | 3.31✅ | 0.71🟡 | 5.72✅ | 3.90✅ | 3.18✅ | 0.95🟡 | — | 1.22✅ |
| **f32** | 2.46✅ | 2.69✅ | 2.72✅ | 5.31✅ | 9.91✅ | 4.80✅ | 0.70🟡 | 0.90🟡 | 0.57🟡 | 4.98✅ | 3.04✅ | 3.43✅ | 4.12✅ | — | 2.18✅ |
| **f64** | 2.19✅ | 2.75✅ | 2.45✅ | 3.34✅ | 2.87✅ | 3.14✅ | 0.99🟡 | 1.05✅ | 0.61🟡 | 2.70✅ | 1.70✅ | 3.19✅ | 3.20✅ | — | 2.29✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.88🟡 | 1.45✅ | 1.41✅ | 1.97✅ | 1.92✅ | 2.13✅ | 1.11✅ | 1.09✅ | 0.63🟡 | 1.91✅ | 1.51✅ | 1.69✅ | 1.99✅ | — | 3.07✅ |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 1.19✅ | 0.90🟡 | 1.05✅ | 1.26✅ | 1.43✅ | 1.74✅ | 1.72✅ | 4.18✅ | 2.96✅ | 1.29✅ | 0.16🔴 | 1.37✅ | 2.89✅ | — | 2.49✅ |
| **u8** | 8.17✅ | 1.35✅ | 11.60✅ | 9.18✅ | 7.42✅ | 5.47✅ | 5.77✅ | 2.96✅ | 2.97✅ | 9.10✅ | 2.76✅ | 1.89✅ | 2.97✅ | — | 3.85✅ |
| **i8** | 8.55✅ | 7.70✅ | 0.67🟡 | 9.53✅ | 8.06✅ | 5.78✅ | 6.02✅ | 3.02✅ | 3.04✅ | 7.57✅ | 2.81✅ | 1.90✅ | 2.95✅ | — | 3.65✅ |
| **i16** | 7.97✅ | 7.08✅ | 8.65✅ | 5.15✅ | 7.31✅ | 5.66✅ | 5.21✅ | 4.56✅ | 5.01✅ | 6.79✅ | 2.80✅ | 5.88✅ | 3.50✅ | — | 3.67✅ |
| **u16** | 8.03✅ | 8.86✅ | 9.19✅ | 6.80✅ | 4.52✅ | 4.76✅ | 4.91✅ | 4.65✅ | 4.50✅ | 4.87✅ | 1.74✅ | 5.91✅ | 3.76✅ | — | 3.36✅ |
| **i32** | 3.00✅ | 3.00✅ | 3.05✅ | 2.74✅ | 2.90✅ | 3.72✅ | 3.99✅ | 4.23✅ | 4.30✅ | 4.92✅ | 2.89✅ | 1.67✅ | 4.38✅ | — | 3.06✅ |
| **u32** | 3.16✅ | 4.32✅ | 3.05✅ | 3.93✅ | 4.99✅ | 4.43✅ | 4.22✅ | 4.08✅ | 3.09✅ | 4.88✅ | 2.66✅ | 2.31✅ | 3.67✅ | — | 3.23✅ |
| **i64** | 1.11✅ | 1.72✅ | 1.22✅ | 2.57✅ | 2.17✅ | 2.32✅ | 2.75✅ | 2.06✅ | 1.87✅ | 1.58✅ | 1.42✅ | 3.28✅ | 2.70✅ | — | 2.26✅ |
| **u64** | 1.18✅ | 1.22✅ | 1.76✅ | 2.27✅ | 2.30✅ | 2.98✅ | 2.73✅ | 2.79✅ | 2.83✅ | 2.18✅ | 0.96🟡 | 1.38✅ | 1.20✅ | — | 1.57✅ |
| **char** | 8.01✅ | 8.66✅ | 8.53✅ | 6.80✅ | 5.70✅ | 2.31✅ | 2.32✅ | 4.08✅ | 4.21✅ | 4.92✅ | 2.80✅ | 1.85✅ | 2.96✅ | — | 3.57✅ |
| **f16** | 16.46✅ | 4.15✅ | 4.37✅ | 5.06✅ | 5.45✅ | 6.17✅ | 1.52✅ | 3.11✅ | 0.72🟡 | 5.34✅ | 5.64✅ | 3.01✅ | 0.90🟡 | — | 1.15✅ |
| **f32** | 4.31✅ | 2.33✅ | 2.32✅ | 4.27✅ | 4.42✅ | 4.32✅ | 0.70🟡 | 0.89🟡 | 0.57🟡 | 4.25✅ | 3.03✅ | 3.82✅ | 4.21✅ | — | 2.84✅ |
| **f64** | 2.07✅ | 2.16✅ | 2.18✅ | 1.75✅ | 1.68✅ | 2.97✅ | 0.94🟡 | 1.02✅ | 0.62🟡 | 2.43✅ | 1.73✅ | 3.12✅ | 2.92✅ | — | 2.32✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.91🟡 | 1.38✅ | 1.40✅ | 1.86✅ | 1.80✅ | 1.89✅ | 1.01✅ | 1.10✅ | 0.63🟡 | 1.79✅ | 1.50✅ | 1.55✅ | 1.13✅ | — | 1.59✅ |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 1.41✅ | 0.90🟡 | 1.04✅ | 1.34✅ | 1.06✅ | 1.26✅ | 1.67✅ | 2.88✅ | 2.81✅ | 1.29✅ | 0.15🔴 | 1.99✅ | 2.82✅ | — | 2.79✅ |
| **u8** | 8.62✅ | 1.53✅ | 17.94✅ | 9.60✅ | 10.05✅ | 5.65✅ | 8.86✅ | 2.96✅ | 7.78✅ | 9.41✅ | 2.75✅ | 1.91✅ | 2.92✅ | — | 2.28✅ |
| **i8** | 8.44✅ | 12.17✅ | 1.14✅ | 9.86✅ | 7.99✅ | 5.92✅ | 5.91✅ | 2.97✅ | 3.01✅ | 8.34✅ | 2.83✅ | 1.90✅ | 2.96✅ | — | 3.54✅ |
| **i16** | 7.46✅ | 6.76✅ | 9.40✅ | 5.20✅ | 4.25✅ | 5.39✅ | 5.28✅ | 4.41✅ | 4.22✅ | 7.03✅ | 2.80✅ | 5.55✅ | 3.31✅ | — | 3.54✅ |
| **u16** | 7.48✅ | 10.39✅ | 9.33✅ | 7.19✅ | 4.79✅ | 5.29✅ | 5.12✅ | 4.86✅ | 4.80✅ | 5.07✅ | 2.82✅ | 6.02✅ | 3.50✅ | — | 3.81✅ |
| **i32** | 2.85✅ | 2.90✅ | 2.88✅ | 4.15✅ | 8.52✅ | 3.74✅ | 3.95✅ | 4.15✅ | 4.33✅ | 5.01✅ | 2.89✅ | 1.71✅ | 4.10✅ | — | 2.88✅ |
| **u32** | 2.86✅ | 3.90✅ | 2.90✅ | 4.15✅ | 5.38✅ | 4.15✅ | 2.87✅ | 4.12✅ | 4.16✅ | 5.06✅ | 2.59✅ | 2.20✅ | 3.41✅ | — | 3.27✅ |
| **i64** | 1.15✅ | 1.70✅ | 1.25✅ | 2.79✅ | 2.39✅ | 3.00✅ | 2.88✅ | 2.43✅ | 2.67✅ | 2.09✅ | 1.41✅ | 1.71✅ | 2.35✅ | — | 1.85✅ |
| **u64** | 1.15✅ | 1.23✅ | 1.67✅ | 2.26✅ | 2.26✅ | 3.04✅ | 2.97✅ | 3.13✅ | 3.01✅ | 2.21✅ | 0.94🟡 | 0.77🟡 | 2.90✅ | — | 2.20✅ |
| **char** | 3.15✅ | 10.35✅ | 10.36✅ | 7.37✅ | 4.55✅ | 2.27✅ | 2.30✅ | 4.07✅ | 6.93✅ | 5.53✅ | 2.80✅ | 1.88✅ | 2.74✅ | — | 3.28✅ |
| **f16** | 15.60✅ | 4.11✅ | 4.27✅ | 4.99✅ | 5.21✅ | 6.69✅ | 1.52✅ | 3.24✅ | 0.72🟡 | 5.08✅ | 5.45✅ | 3.09✅ | 0.88🟡 | — | 1.22✅ |
| **f32** | 4.19✅ | 2.19✅ | 2.21✅ | 4.56✅ | 4.35✅ | 4.32✅ | 0.65🟡 | 0.83🟡 | 1.40✅ | 4.28✅ | 2.92✅ | 3.68✅ | 3.83✅ | — | 2.93✅ |
| **f64** | 2.00✅ | 2.22✅ | 2.23✅ | 3.20✅ | 2.80✅ | 3.34✅ | 0.95🟡 | 0.99🟡 | 0.62🟡 | 2.66✅ | 1.66✅ | 3.17✅ | 2.83✅ | — | 2.35✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.85🟡 | 1.35✅ | 1.34✅ | 1.80✅ | 1.93✅ | 1.96✅ | 1.06✅ | 1.06✅ | 0.64🟡 | 1.68✅ | 1.45✅ | 1.59✅ | 1.89✅ | — | 1.85✅ |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.86🟡 | 0.90🟡 | 0.89🟡 | 1.31✅ | 1.34✅ | 1.67✅ | 1.67✅ | 2.90✅ | 2.84✅ | 1.26✅ | 0.30🟠 | 3.50✅ | 5.13✅ | — | 3.47✅ |
| **u8** | 0.68🟡 | 8.93✅ | 7.66✅ | 9.53✅ | 10.56✅ | 2.17✅ | 2.12✅ | 3.99✅ | 3.99✅ | 9.81✅ | 1.07✅ | 2.07✅ | 3.32✅ | — | 3.61✅ |
| **i8** | 1.08✅ | 11.79✅ | 9.13✅ | 7.96✅ | 7.28✅ | 2.00✅ | 2.16✅ | 3.59✅ | 3.62✅ | 9.62✅ | 1.04✅ | 1.43✅ | 3.28✅ | — | 3.51✅ |
| **i16** | 3.86✅ | 9.02✅ | 3.82✅ | 8.90✅ | 9.48✅ | 2.27✅ | 2.20✅ | 2.63✅ | 2.74✅ | 5.67✅ | 2.14✅ | 1.95✅ | 3.21✅ | — | 2.49✅ |
| **u16** | 3.83✅ | 5.18✅ | 5.21✅ | 7.24✅ | 8.55✅ | 2.59✅ | 1.95✅ | 4.22✅ | 3.83✅ | 8.73✅ | 2.15✅ | 2.00✅ | 3.23✅ | — | 3.62✅ |
| **i32** | 0.48🟠 | 0.55🟡 | 0.23🟠 | 0.46🟠 | 0.72🟡 | 1.79✅ | 1.77✅ | 3.61✅ | 3.61✅ | 0.64🟡 | 2.21✅ | 3.56✅ | 2.91✅ | — | 3.24✅ |
| **u32** | 0.48🟠 | 0.41🟠 | 0.23🟠 | 0.64🟡 | 0.50🟡 | 1.85✅ | 1.78✅ | 3.74✅ | 3.87✅ | 0.81🟡 | 2.11✅ | 1.91✅ | 2.33✅ | — | 2.87✅ |
| **i64** | 0.39🟠 | 0.40🟠 | 0.39🟠 | 0.56🟡 | 0.49🟠 | 0.69🟡 | 0.72🟡 | 2.33✅ | 2.35✅ | 0.45🟠 | 1.23✅ | 1.80✅ | 2.94✅ | — | 2.24✅ |
| **u64** | 0.39🟠 | 0.39🟠 | 0.39🟠 | 0.54🟡 | 4.29✅ | 1.04✅ | 0.72🟡 | 2.00✅ | 1.99✅ | 0.55🟡 | 1.49✅ | 1.16✅ | 1.86✅ | — | 2.21✅ |
| **char** | 3.86✅ | 5.25✅ | 5.24✅ | 7.23✅ | 9.19✅ | 2.46✅ | 2.46✅ | 2.94✅ | 3.89✅ | 8.81✅ | 2.08✅ | 1.88✅ | 2.67✅ | — | 2.39✅ |
| **f16** | 9.56✅ | 2.28✅ | 3.87✅ | 2.76✅ | 2.72✅ | 3.40✅ | 1.26✅ | 2.28✅ | 0.73🟡 | 2.66✅ | 5.81✅ | 3.49✅ | 0.82🟡 | — | 1.23✅ |
| **f32** | 0.82🟡 | 0.40🟠 | 0.54🟡 | 0.69🟡 | 0.66🟡 | 0.98🟡 | 0.41🟠 | 0.86🟡 | 0.48🟠 | 0.66🟡 | 2.17✅ | 1.74✅ | 2.27✅ | — | 2.84✅ |
| **f64** | 0.57🟡 | 0.30🟠 | 0.30🟠 | 0.58🟡 | 0.51🟡 | 2.93✅ | 0.57🟡 | 0.98🟡 | 0.58🟡 | 0.49🟠 | 1.36✅ | 1.37✅ | 2.55✅ | — | 2.30✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 1.00🟡 | 0.39🟠 | 0.40🟠 | 0.64🟡 | 0.62🟡 | 0.84🟡 | 0.69🟡 | 1.07✅ | 0.63🟡 | 0.60🟡 | 1.24✅ | 1.64✅ | 2.10✅ | — | 2.05✅ |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.83🟡 | 0.88🟡 | 0.63🟡 | 0.58🟡 | 0.87🟡 | 1.73✅ | 1.91✅ | 2.96✅ | 2.84✅ | 0.87🟡 | 0.14🔴 | 2.02✅ | 2.92✅ | — | 4.93✅ |
| **u8** | 0.98🟡 | 6.38✅ | 5.43✅ | 5.78✅ | 5.12✅ | 1.85✅ | 2.57✅ | 4.44✅ | 3.15✅ | 2.79✅ | 0.89🟡 | 2.05✅ | 3.43✅ | — | 4.89✅ |
| **i8** | 0.98🟡 | 7.47✅ | 6.25✅ | 4.26✅ | 5.16✅ | 2.07✅ | 2.26✅ | 3.69✅ | 3.86✅ | 5.83✅ | 1.87✅ | 2.15✅ | 3.52✅ | — | 4.92✅ |
| **i16** | 2.92✅ | 3.87✅ | 2.79✅ | 5.38✅ | 4.66✅ | 2.17✅ | 2.47✅ | 3.85✅ | 2.70✅ | 5.36✅ | 1.92✅ | 2.03✅ | 3.58✅ | — | 4.75✅ |
| **u16** | 2.92✅ | 3.85✅ | 3.85✅ | 3.81✅ | 1.96✅ | 1.76✅ | 2.91✅ | 4.26✅ | 4.23✅ | 5.33✅ | 0.92🟡 | 2.14✅ | 3.63✅ | — | 4.65✅ |
| **i32** | 0.46🟠 | 0.21🟠 | 0.56🟡 | 0.57🟡 | 0.39🟠 | 1.83✅ | 1.32✅ | 3.49✅ | 3.50✅ | 0.39🟠 | 0.97🟡 | 1.58✅ | 2.85✅ | — | 3.66✅ |
| **u32** | 0.33🟠 | 0.39🟠 | 0.56🟡 | 0.23🟠 | 0.23🟠 | 1.82✅ | 2.87✅ | 3.71✅ | 3.99✅ | 0.59🟡 | 1.85✅ | 2.10✅ | 3.37✅ | — | 4.08✅ |
| **i64** | 0.41🟠 | 0.41🟠 | 0.19🔴 | 0.41🟠 | 0.32🟠 | 0.78🟡 | 0.83🟡 | 2.67✅ | 2.59✅ | 0.32🟠 | 0.54🟡 | 1.38✅ | 2.62✅ | — | 2.90✅ |
| **u64** | 0.42🟠 | 0.42🟠 | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.85🟡 | 0.81🟡 | 2.64✅ | 2.73✅ | 0.41🟠 | 1.33✅ | 1.20✅ | 2.05✅ | — | 2.66✅ |
| **char** | 2.91✅ | 3.83✅ | 7.51✅ | 3.75✅ | 4.73✅ | 2.70✅ | 2.81✅ | 4.33✅ | 4.21✅ | 5.29✅ | 1.93✅ | 1.89✅ | 2.98✅ | — | 4.29✅ |
| **f16** | 7.74✅ | 1.84✅ | 1.98✅ | 2.43✅ | 2.25✅ | 3.32✅ | 1.31✅ | 2.26✅ | 0.73🟡 | 2.25✅ | 5.31✅ | 1.99✅ | 0.86🟡 | — | 1.22✅ |
| **f32** | 0.82🟡 | 0.38🟠 | 0.54🟡 | 0.38🟠 | 0.38🟠 | 1.07✅ | 0.49🟠 | 0.85🟡 | 0.45🟠 | 0.38🟠 | 2.00✅ | 3.46✅ | 2.52✅ | — | 3.65✅ |
| **f64** | 0.60🟡 | 0.39🟠 | 0.37🟠 | 0.42🟠 | 0.38🟠 | 0.61🟡 | 0.64🟡 | 1.13✅ | 0.41🟠 | 0.19🔴 | 1.25✅ | 1.64✅ | 2.80✅ | — | 3.49✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.35🟠 | 0.61🟡 | 0.62🟡 | 0.57🟡 | 0.57🟡 | 1.04✅ | 0.71🟡 | 1.17✅ | 0.55🟡 | 0.57🟡 | 0.42🟠 | 1.08✅ | 1.96✅ | — | 2.53✅ |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.03🔴 | 0.28🟠 | 0.27🟠 | 0.27🟠 | 0.27🟠 | 0.49🟠 | 0.50🟠 | 0.97🟡 | 0.88🟡 | 0.36🟠 | 0.30🟠 | 0.51🟡 | 0.97🟡 | — | 1.42✅ |
| **u8** | 0.29🟠 | 0.03🔴 | 0.25🟠 | 0.39🟠 | 0.39🟠 | 0.47🟠 | 0.50🟠 | 0.98🟡 | 0.99🟡 | 0.37🟠 | 0.59🟡 | 0.54🟡 | 0.92🟡 | — | 1.39✅ |
| **i8** | 0.29🟠 | 0.34🟠 | 0.03🔴 | 0.33🟠 | 0.38🟠 | 0.38🟠 | 0.50🟡 | 0.91🟡 | 0.92🟡 | 0.38🟠 | 0.58🟡 | 0.48🟠 | 0.98🟡 | — | 1.35✅ |
| **i16** | 0.27🟠 | 0.31🟠 | 0.23🟠 | 0.23🟠 | 0.25🟠 | 0.51🟡 | 0.49🟠 | 0.98🟡 | 0.98🟡 | 0.39🟠 | 0.57🟡 | 0.47🟠 | 0.91🟡 | — | 1.28✅ |
| **u16** | 0.56🟡 | 0.31🟠 | 0.31🟠 | 0.32🟠 | 0.25🟠 | 0.48🟠 | 0.51🟡 | 0.97🟡 | 0.99🟡 | 0.23🟠 | 0.59🟡 | 1.05✅ | 1.30✅ | — | 1.29✅ |
| **i32** | 0.25🟠 | 0.30🟠 | 0.29🟠 | 0.34🟠 | 0.30🟠 | 0.31🟠 | 0.32🟠 | 0.92🟡 | 0.96🟡 | 0.30🟠 | 0.62🟡 | 0.41🟠 | 0.98🟡 | — | 1.31✅ |
| **u32** | 0.19🔴 | 0.13🔴 | 0.30🟠 | 0.24🟠 | 0.41🟠 | 0.40🟠 | 0.41🟠 | 0.74🟡 | 1.00🟡 | 0.38🟠 | 0.59🟡 | 0.41🟠 | 0.98🟡 | — | 1.24✅ |
| **i64** | 0.29🟠 | 0.26🟠 | 0.26🟠 | 0.32🟠 | 0.25🟠 | 0.39🟠 | 0.46🟠 | 0.98🟡 | 0.99🟡 | 0.26🟠 | 0.54🟡 | 0.36🟠 | 0.98🟡 | — | 1.24✅ |
| **u64** | 0.29🟠 | 0.11🔴 | 0.11🔴 | 0.34🟠 | 0.34🟠 | 0.44🟠 | 0.47🟠 | 0.95🟡 | 1.00🟡 | 0.32🟠 | 0.54🟡 | 0.50🟡 | 1.12✅ | — | 1.12✅ |
| **char** | 0.26🟠 | 0.13🔴 | 0.31🟠 | 0.33🟠 | 0.25🟠 | 0.47🟠 | 0.47🟠 | 0.93🟡 | 0.80🟡 | 0.24🟠 | 0.58🟡 | 0.49🟠 | 0.97🟡 | — | 1.19✅ |
| **f16** | 0.77🟡 | 0.40🟠 | 0.41🟠 | 0.85🟡 | 0.45🟠 | 0.49🟠 | 0.51🟡 | 0.55🟡 | 0.67🟡 | 0.44🟠 | 0.26🟠 | 0.49🟠 | 0.77🟡 | — | 0.94🟡 |
| **f32** | 0.44🟠 | 0.11🔴 | 0.15🔴 | 0.19🔴 | 0.19🔴 | 0.33🟠 | 0.33🟠 | 0.75🟡 | 0.59🟡 | 0.18🔴 | 0.74🟡 | 0.37🟠 | 0.95🟡 | — | 1.30✅ |
| **f64** | 0.44🟠 | 0.12🔴 | 0.13🔴 | 0.24🟠 | 0.21🟠 | 0.37🟠 | 0.41🟠 | 0.85🟡 | 0.66🟡 | 0.19🔴 | 0.66🟡 | 0.36🟠 | 0.99🟡 | — | 1.27✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.36🟠 | 0.12🔴 | 0.08🔴 | 0.14🔴 | 0.17🔴 | 0.26🟠 | 0.27🟠 | 0.87🟡 | 0.40🟠 | 0.16🔴 | 0.66🟡 | 0.29🟠 | 0.97🟡 | — | 1.34✅ |

## Lagging cells (<1.0) — the worklist  (381 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| bool\|bcast\|bool | 1.4941 | 0.0471 | 0.03 🔴 |
| u8\|bcast\|u8 | 1.4360 | 0.0487 | 0.03 🔴 |
| i8\|bcast\|i8 | 1.4392 | 0.0488 | 0.03 🔴 |
| c128\|bcast\|i8 | 3.1701 | 0.2452 | 0.08 🔴 |
| u64\|bcast\|u8 | 1.8771 | 0.1978 | 0.11 🔴 |
| u64\|bcast\|i8 | 1.8710 | 0.1982 | 0.11 🔴 |
| f32\|bcast\|u8 | 3.1493 | 0.3548 | 0.11 🔴 |
| c128\|bcast\|u8 | 3.0778 | 0.3567 | 0.12 🔴 |
| f64\|bcast\|u8 | 2.8647 | 0.3550 | 0.12 🔴 |
| f64\|bcast\|i8 | 2.8382 | 0.3550 | 0.13 🔴 |
| char\|bcast\|u8 | 1.5715 | 0.2004 | 0.13 🔴 |
| u32\|bcast\|u8 | 1.6123 | 0.2092 | 0.13 🔴 |
| bool\|strided\|f16 | 5.3589 | 0.7299 | 0.14 🔴 |
| c128\|bcast\|i16 | 3.2484 | 0.4548 | 0.14 🔴 |
| bool\|negrow\|f16 | 11.2341 | 1.6957 | 0.15 🔴 |
| f32\|bcast\|i8 | 3.1514 | 0.4795 | 0.15 🔴 |
| bool\|sliced\|f16 | 10.7603 | 1.6681 | 0.16 🔴 |
| c128\|bcast\|char | 3.5035 | 0.5628 | 0.16 🔴 |
| c128\|bcast\|u16 | 3.6305 | 0.6166 | 0.17 🔴 |
| f32\|bcast\|char | 3.1652 | 0.5588 | 0.18 🔴 |
| u32\|bcast\|bool | 1.6259 | 0.3021 | 0.19 🔴 |
| i64\|strided\|i8 | 0.6632 | 0.1242 | 0.19 🔴 |
| f64\|bcast\|char | 2.9148 | 0.5615 | 0.19 🔴 |
| f32\|bcast\|i16 | 3.1536 | 0.6089 | 0.19 🔴 |
| f64\|strided\|char | 0.6744 | 0.1303 | 0.19 🔴 |
| f32\|bcast\|u16 | 3.1549 | 0.6132 | 0.19 🔴 |
| f64\|bcast\|u16 | 2.8638 | 0.5966 | 0.21 🟠 |
| i32\|strided\|u8 | 0.4569 | 0.0975 | 0.21 🟠 |
| u32\|strided\|i16 | 0.4546 | 0.1028 | 0.23 🟠 |
| u32\|strided\|u16 | 0.4547 | 0.1029 | 0.23 🟠 |
| i32\|negcol\|i8 | 0.9034 | 0.2052 | 0.23 🟠 |
| u32\|negcol\|i8 | 0.9032 | 0.2058 | 0.23 🟠 |
| u16\|bcast\|char | 1.7865 | 0.4086 | 0.23 🟠 |
| i16\|bcast\|i8 | 1.5473 | 0.3549 | 0.23 🟠 |
| i16\|bcast\|i16 | 1.7875 | 0.4120 | 0.23 🟠 |
| f64\|bcast\|i16 | 2.8660 | 0.6834 | 0.24 🟠 |
| u32\|bcast\|i16 | 1.8290 | 0.4414 | 0.24 🟠 |
| char\|bcast\|char | 1.7404 | 0.4233 | 0.24 🟠 |
| char\|bcast\|u16 | 1.7612 | 0.4327 | 0.25 🟠 |
| bool\|C\|f16 | 6.7123 | 1.6517 | 0.25 🟠 |
| u8\|bcast\|i8 | 1.4393 | 0.3546 | 0.25 🟠 |
| i64\|bcast\|u16 | 2.2411 | 0.5613 | 0.25 🟠 |
| u16\|bcast\|u16 | 1.7871 | 0.4508 | 0.25 🟠 |
| i32\|bcast\|bool | 1.6313 | 0.4128 | 0.25 🟠 |
| i16\|bcast\|u16 | 1.7805 | 0.4534 | 0.25 🟠 |
| i64\|bcast\|u8 | 1.8659 | 0.4796 | 0.26 🟠 |
| c128\|bcast\|i32 | 3.5397 | 0.9135 | 0.26 🟠 |
| f16\|bcast\|f16 | 1.7593 | 0.4543 | 0.26 🟠 |
| i64\|bcast\|i8 | 1.8528 | 0.4794 | 0.26 🟠 |
| i64\|bcast\|char | 2.1941 | 0.5704 | 0.26 🟠 |
| char\|bcast\|bool | 1.5623 | 0.4127 | 0.26 🟠 |
| c128\|bcast\|u32 | 3.2429 | 0.8621 | 0.27 🟠 |
| i16\|bcast\|bool | 1.5287 | 0.4127 | 0.27 🟠 |
| bool\|bcast\|i16 | 1.8070 | 0.4909 | 0.27 🟠 |
| bool\|bcast\|i8 | 1.5112 | 0.4129 | 0.27 🟠 |
| bool\|bcast\|u16 | 1.8610 | 0.5089 | 0.27 🟠 |
| bool\|bcast\|u8 | 1.5005 | 0.4129 | 0.28 🟠 |
| bool\|F\|f16 | 10.9523 | 3.1945 | 0.29 🟠 |
| u8\|bcast\|bool | 1.6394 | 0.4790 | 0.29 🟠 |
| u64\|bcast\|bool | 1.6370 | 0.4792 | 0.29 🟠 |
| i64\|bcast\|bool | 1.6368 | 0.4793 | 0.29 🟠 |
| i8\|bcast\|bool | 1.6353 | 0.4792 | 0.29 🟠 |
| i32\|bcast\|i8 | 1.6298 | 0.4797 | 0.29 🟠 |
| c128\|bcast\|f32 | 2.9914 | 0.8814 | 0.29 🟠 |
| i32\|bcast\|u16 | 2.1051 | 0.6210 | 0.30 🟠 |
| i32\|bcast\|u8 | 1.6245 | 0.4794 | 0.30 🟠 |
| bool\|bcast\|f16 | 10.7720 | 3.1954 | 0.30 🟠 |
| f64\|negcol\|i8 | 1.3202 | 0.3919 | 0.30 🟠 |
| u32\|bcast\|i8 | 1.6031 | 0.4786 | 0.30 🟠 |
| bool\|negcol\|f16 | 10.7650 | 3.2144 | 0.30 🟠 |
| bool\|T\|f16 | 10.8092 | 3.2278 | 0.30 🟠 |
| f64\|negcol\|u8 | 1.3199 | 0.3972 | 0.30 🟠 |
| i32\|bcast\|char | 1.8444 | 0.5605 | 0.30 🟠 |
| i32\|bcast\|i32 | 2.7475 | 0.8479 | 0.31 🟠 |
| char\|bcast\|i8 | 1.5396 | 0.4797 | 0.31 🟠 |
| u16\|bcast\|i8 | 1.5372 | 0.4798 | 0.31 🟠 |
| u16\|bcast\|u8 | 1.5318 | 0.4785 | 0.31 🟠 |
| i16\|bcast\|u8 | 1.5279 | 0.4797 | 0.31 🟠 |
| u16\|bcast\|i16 | 1.7396 | 0.5516 | 0.32 🟠 |
| i64\|strided\|u16 | 0.6776 | 0.2150 | 0.32 🟠 |
| i64\|strided\|char | 0.6775 | 0.2158 | 0.32 🟠 |
| i64\|bcast\|i16 | 2.1528 | 0.6862 | 0.32 🟠 |
| u64\|bcast\|char | 2.1580 | 0.6934 | 0.32 🟠 |
| i32\|bcast\|u32 | 2.7680 | 0.8899 | 0.32 🟠 |
| i8\|bcast\|i16 | 1.7830 | 0.5826 | 0.33 🟠 |
| f32\|bcast\|i32 | 2.5413 | 0.8367 | 0.33 🟠 |
| u32\|strided\|bool | 0.4525 | 0.1494 | 0.33 🟠 |
| char\|bcast\|i16 | 1.7562 | 0.5854 | 0.33 🟠 |
| f32\|bcast\|u32 | 2.5479 | 0.8505 | 0.33 🟠 |
| i8\|bcast\|u8 | 1.4281 | 0.4784 | 0.34 🟠 |
| u64\|bcast\|i16 | 2.1845 | 0.7408 | 0.34 🟠 |
| i32\|bcast\|i16 | 2.0800 | 0.7092 | 0.34 🟠 |
| u64\|bcast\|u16 | 2.1781 | 0.7474 | 0.34 🟠 |
| c128\|strided\|bool | 1.3592 | 0.4746 | 0.35 🟠 |
| bool\|bcast\|char | 1.8310 | 0.6576 | 0.36 🟠 |
| c128\|bcast\|bool | 1.9800 | 0.7128 | 0.36 🟠 |
| i64\|bcast\|f32 | 2.4956 | 0.9095 | 0.36 🟠 |
| f64\|bcast\|f32 | 2.3416 | 0.8545 | 0.36 🟠 |
| f32\|bcast\|f32 | 2.1590 | 0.7983 | 0.37 🟠 |
| f64\|bcast\|i32 | 2.1736 | 0.8047 | 0.37 🟠 |
| f64\|strided\|i8 | 0.6602 | 0.2449 | 0.37 🟠 |
| u8\|bcast\|char | 1.8215 | 0.6793 | 0.37 🟠 |
| f32\|strided\|u8 | 0.4733 | 0.1775 | 0.38 🟠 |
| f64\|strided\|u16 | 0.6748 | 0.2536 | 0.38 🟠 |
| u32\|bcast\|char | 1.8217 | 0.6860 | 0.38 🟠 |
| i8\|bcast\|u16 | 1.8106 | 0.6838 | 0.38 🟠 |
| i8\|bcast\|i32 | 1.8346 | 0.6994 | 0.38 🟠 |
| f32\|strided\|i16 | 0.4639 | 0.1779 | 0.38 🟠 |
| f32\|strided\|char | 0.4639 | 0.1780 | 0.38 🟠 |
| i8\|bcast\|char | 1.7790 | 0.6844 | 0.38 🟠 |
| f32\|strided\|u16 | 0.4639 | 0.1786 | 0.38 🟠 |
| i16\|bcast\|char | 1.7640 | 0.6795 | 0.39 🟠 |
| u8\|bcast\|i16 | 1.7906 | 0.6901 | 0.39 🟠 |
| u8\|bcast\|u16 | 1.8441 | 0.7146 | 0.39 🟠 |
| c128\|negcol\|u8 | 1.3321 | 0.5168 | 0.39 🟠 |
| f64\|strided\|u8 | 0.6603 | 0.2580 | 0.39 🟠 |
| u64\|negcol\|i8 | 1.3261 | 0.5191 | 0.39 🟠 |
| i32\|strided\|u16 | 0.4548 | 0.1786 | 0.39 🟠 |
| i32\|strided\|char | 0.4551 | 0.1789 | 0.39 🟠 |
| u64\|negcol\|u8 | 1.3242 | 0.5212 | 0.39 🟠 |
| i64\|negcol\|bool | 1.3547 | 0.5341 | 0.39 🟠 |
| i64\|negcol\|i8 | 1.3272 | 0.5235 | 0.39 🟠 |
| u64\|negcol\|bool | 1.3559 | 0.5349 | 0.39 🟠 |
| u32\|strided\|u8 | 0.4564 | 0.1802 | 0.39 🟠 |
| i64\|bcast\|i32 | 2.1195 | 0.8372 | 0.39 🟠 |
| i64\|negcol\|u8 | 1.3262 | 0.5239 | 0.40 🟠 |
| c128\|negcol\|i8 | 1.3337 | 0.5278 | 0.40 🟠 |
| f16\|bcast\|u8 | 4.5833 | 1.8157 | 0.40 🟠 |
| c128\|bcast\|u64 | 4.0657 | 1.6189 | 0.40 🟠 |
| u32\|bcast\|i32 | 2.0450 | 0.8154 | 0.40 🟠 |
| f32\|negcol\|u8 | 0.9269 | 0.3698 | 0.40 🟠 |
| f64\|bcast\|u32 | 2.1256 | 0.8662 | 0.41 🟠 |
| u32\|bcast\|u16 | 1.8552 | 0.7565 | 0.41 🟠 |
| u32\|bcast\|u32 | 2.0508 | 0.8370 | 0.41 🟠 |
| u32\|negcol\|u8 | 0.9033 | 0.3687 | 0.41 🟠 |
| i64\|strided\|i16 | 0.6773 | 0.2768 | 0.41 🟠 |
| i64\|strided\|bool | 0.6739 | 0.2757 | 0.41 🟠 |
| f64\|strided\|u64 | 1.6374 | 0.6715 | 0.41 🟠 |
| u64\|strided\|i16 | 0.6774 | 0.2786 | 0.41 🟠 |
| i64\|strided\|u8 | 0.6642 | 0.2734 | 0.41 🟠 |
| f32\|negcol\|u32 | 2.2645 | 0.9331 | 0.41 🟠 |
| f16\|bcast\|i8 | 4.5715 | 1.8849 | 0.41 🟠 |
| u64\|strided\|char | 0.6777 | 0.2800 | 0.41 🟠 |
| u64\|strided\|u16 | 0.6772 | 0.2801 | 0.41 🟠 |
| u32\|bcast\|f32 | 2.1745 | 0.8995 | 0.41 🟠 |
| i32\|bcast\|f32 | 2.1462 | 0.8881 | 0.41 🟠 |
| u64\|strided\|i8 | 0.6635 | 0.2751 | 0.41 🟠 |
| u64\|strided\|u8 | 0.6635 | 0.2760 | 0.42 🟠 |
| u64\|strided\|bool | 0.6739 | 0.2816 | 0.42 🟠 |
| c128\|strided\|f16 | 1.6986 | 0.7116 | 0.42 🟠 |
| f64\|strided\|i16 | 0.6746 | 0.2858 | 0.42 🟠 |
| f64\|bcast\|bool | 1.6365 | 0.7126 | 0.44 🟠 |
| f32\|bcast\|bool | 1.6281 | 0.7124 | 0.44 🟠 |
| u64\|bcast\|i32 | 1.9583 | 0.8683 | 0.44 🟠 |
| f16\|bcast\|char | 4.5725 | 2.0314 | 0.44 🟠 |
| f32\|strided\|u64 | 2.0005 | 0.8941 | 0.45 🟠 |
| f16\|bcast\|u16 | 4.6025 | 2.0777 | 0.45 🟠 |
| i64\|negcol\|char | 1.3523 | 0.6147 | 0.45 🟠 |
| i32\|strided\|bool | 0.4522 | 0.2075 | 0.46 🟠 |
| i64\|bcast\|u32 | 2.0154 | 0.9318 | 0.46 🟠 |
| i32\|negcol\|i16 | 0.9188 | 0.4249 | 0.46 🟠 |
| u8\|bcast\|i32 | 1.8453 | 0.8648 | 0.47 🟠 |
| char\|bcast\|i32 | 1.9353 | 0.9074 | 0.47 🟠 |
| u64\|bcast\|u32 | 1.9509 | 0.9219 | 0.47 🟠 |
| i64\|F\|i8 | 0.4027 | 0.1905 | 0.47 🟠 |
| i16\|bcast\|f32 | 2.0267 | 0.9604 | 0.47 🟠 |
| char\|bcast\|u32 | 1.9708 | 0.9349 | 0.47 🟠 |
| u32\|negcol\|bool | 0.9065 | 0.4309 | 0.48 🟠 |
| i8\|bcast\|f32 | 2.0448 | 0.9735 | 0.48 🟠 |
| i32\|negcol\|bool | 0.9054 | 0.4323 | 0.48 🟠 |
| u16\|bcast\|i32 | 1.8803 | 0.8989 | 0.48 🟠 |
| f32\|negcol\|u64 | 3.5822 | 1.7257 | 0.48 🟠 |
| f16\|bcast\|i32 | 4.7029 | 2.2863 | 0.49 🟠 |
| f64\|negcol\|char | 1.3556 | 0.6593 | 0.49 🟠 |
| i16\|bcast\|u32 | 1.9314 | 0.9394 | 0.49 🟠 |
| f16\|bcast\|f32 | 3.8049 | 1.8519 | 0.49 🟠 |
| char\|bcast\|f32 | 1.9226 | 0.9368 | 0.49 🟠 |
| bool\|bcast\|i32 | 1.8616 | 0.9108 | 0.49 🟠 |
| i64\|negcol\|u16 | 1.3539 | 0.6627 | 0.49 🟠 |
| f32\|strided\|u32 | 1.1334 | 0.5557 | 0.49 🟠 |
| bool\|bcast\|u32 | 1.8413 | 0.9134 | 0.50 🟠 |
| u8\|bcast\|u32 | 1.7389 | 0.8658 | 0.50 🟠 |
| i8\|bcast\|u32 | 1.8755 | 0.9395 | 0.50 🟡 |
| u32\|negcol\|u16 | 0.9183 | 0.4601 | 0.50 🟡 |
| u64\|bcast\|f32 | 2.2217 | 1.1154 | 0.50 🟡 |
| i16\|bcast\|i32 | 1.8713 | 0.9480 | 0.51 🟡 |
| bool\|bcast\|f32 | 1.9075 | 0.9676 | 0.51 🟡 |
| f16\|bcast\|u32 | 4.5027 | 2.2950 | 0.51 🟡 |
| f64\|negcol\|u16 | 1.3549 | 0.6929 | 0.51 🟡 |
| u16\|bcast\|u32 | 1.8897 | 0.9725 | 0.51 🟡 |
| f32\|strided\|i8 | 0.4734 | 0.2537 | 0.54 🟡 |
| u64\|bcast\|f16 | 6.2618 | 3.3806 | 0.54 🟡 |
| i64\|strided\|f16 | 1.2818 | 0.6929 | 0.54 🟡 |
| f32\|negcol\|i8 | 0.9277 | 0.5027 | 0.54 🟡 |
| i64\|bcast\|f16 | 5.6754 | 3.0787 | 0.54 🟡 |
| u8\|bcast\|f32 | 1.7622 | 0.9567 | 0.54 🟡 |
| u64\|negcol\|i16 | 1.3524 | 0.7343 | 0.54 🟡 |
| c128\|strided\|u64 | 2.2039 | 1.2023 | 0.55 🟡 |
| i32\|negcol\|u8 | 0.9032 | 0.4992 | 0.55 🟡 |
| u64\|negcol\|char | 1.3517 | 0.7476 | 0.55 🟡 |
| f16\|bcast\|i64 | 4.8163 | 2.6670 | 0.55 🟡 |
| u32\|strided\|i8 | 0.4563 | 0.2543 | 0.56 🟡 |
| i64\|negcol\|i16 | 1.3536 | 0.7559 | 0.56 🟡 |
| u16\|bcast\|bool | 0.7389 | 0.4127 | 0.56 🟡 |
| i32\|strided\|i8 | 0.4567 | 0.2551 | 0.56 🟡 |
| f64\|negcol\|bool | 1.3556 | 0.7664 | 0.57 🟡 |
| i16\|bcast\|f16 | 5.4506 | 3.0827 | 0.57 🟡 |
| f32\|sliced\|u64 | 3.0199 | 1.7097 | 0.57 🟡 |
| f32\|T\|u64 | 3.0152 | 1.7090 | 0.57 🟡 |
| f32\|C\|u64 | 3.0135 | 1.7103 | 0.57 🟡 |
| c128\|strided\|u16 | 0.7347 | 0.4170 | 0.57 🟡 |
| i32\|strided\|i16 | 0.4548 | 0.2586 | 0.57 🟡 |
| f64\|negcol\|u32 | 1.7397 | 0.9948 | 0.57 🟡 |
| c128\|strided\|char | 0.7280 | 0.4178 | 0.57 🟡 |
| c128\|strided\|i16 | 0.7276 | 0.4178 | 0.57 🟡 |
| i8\|bcast\|f16 | 5.4668 | 3.1439 | 0.58 🟡 |
| char\|bcast\|f16 | 5.3379 | 3.0822 | 0.58 🟡 |
| bool\|strided\|i16 | 0.2695 | 0.1561 | 0.58 🟡 |
| f64\|negcol\|u64 | 3.1539 | 1.8355 | 0.58 🟡 |
| f32\|F\|u64 | 3.0336 | 1.7665 | 0.58 🟡 |
| f64\|negcol\|i16 | 1.3547 | 0.7917 | 0.58 🟡 |
| u8\|bcast\|f16 | 5.2612 | 3.0841 | 0.59 🟡 |
| u32\|bcast\|f16 | 5.3163 | 3.1225 | 0.59 🟡 |
| u32\|strided\|char | 0.4546 | 0.2674 | 0.59 🟡 |
| u16\|bcast\|f16 | 5.3294 | 3.1352 | 0.59 🟡 |
| f32\|bcast\|u64 | 2.7264 | 1.6141 | 0.59 🟡 |
| c128\|negcol\|char | 1.4568 | 0.8674 | 0.60 🟡 |
| f64\|strided\|bool | 0.6747 | 0.4057 | 0.60 🟡 |
| f64\|F\|u64 | 2.9212 | 1.7660 | 0.60 🟡 |
| f64\|T\|u64 | 2.9211 | 1.7783 | 0.61 🟡 |
| f64\|strided\|i32 | 0.9748 | 0.5941 | 0.61 🟡 |
| c128\|strided\|u8 | 0.6754 | 0.4130 | 0.61 🟡 |
| f64\|C\|u64 | 2.9233 | 1.7945 | 0.61 🟡 |
| bool\|C\|bool | 0.0236 | 0.0145 | 0.61 🟡 |
| c128\|negcol\|u16 | 1.4550 | 0.8954 | 0.62 🟡 |
| i32\|bcast\|f16 | 4.9837 | 3.0758 | 0.62 🟡 |
| f64\|negrow\|u64 | 2.9567 | 1.8273 | 0.62 🟡 |
| c128\|strided\|i8 | 0.6741 | 0.4167 | 0.62 🟡 |
| c128\|C\|u64 | 3.2339 | 2.0020 | 0.62 🟡 |
| f64\|sliced\|u64 | 2.9367 | 1.8185 | 0.62 🟡 |
| c128\|T\|u64 | 3.2350 | 2.0368 | 0.63 🟡 |
| c128\|sliced\|u64 | 3.2747 | 2.0622 | 0.63 🟡 |
| bool\|strided\|i8 | 0.2349 | 0.1487 | 0.63 🟡 |
| c128\|negcol\|u64 | 3.3406 | 2.1150 | 0.63 🟡 |
| c128\|negcol\|i16 | 1.4549 | 0.9240 | 0.64 🟡 |
| f64\|strided\|u32 | 0.8517 | 0.5437 | 0.64 🟡 |
| u32\|negcol\|i16 | 0.9193 | 0.5895 | 0.64 🟡 |
| i32\|negcol\|char | 0.9179 | 0.5891 | 0.64 🟡 |
| c128\|negrow\|u64 | 3.3556 | 2.1623 | 0.64 🟡 |
| f32\|negrow\|u32 | 1.3383 | 0.8680 | 0.65 🟡 |
| f64\|bcast\|f16 | 5.4475 | 3.5734 | 0.66 🟡 |
| f64\|bcast\|u64 | 2.4326 | 1.5958 | 0.66 🟡 |
| c128\|bcast\|f16 | 5.7581 | 3.7818 | 0.66 🟡 |
| c128\|F\|u64 | 3.2345 | 2.1316 | 0.66 🟡 |
| f32\|negcol\|u16 | 0.9102 | 0.6006 | 0.66 🟡 |
| f32\|negcol\|char | 0.9092 | 0.6016 | 0.66 🟡 |
| i8\|sliced\|i8 | 0.0456 | 0.0306 | 0.67 🟡 |
| f16\|bcast\|u64 | 4.6871 | 3.1504 | 0.67 🟡 |
| u8\|negcol\|bool | 0.4497 | 0.3037 | 0.68 🟡 |
| i64\|negcol\|i32 | 1.3612 | 0.9355 | 0.69 🟡 |
| f32\|negcol\|i16 | 0.9095 | 0.6264 | 0.69 🟡 |
| c128\|negcol\|u32 | 1.7738 | 1.2236 | 0.69 🟡 |
| f32\|T\|u32 | 1.3157 | 0.9193 | 0.70 🟡 |
| f32\|sliced\|u32 | 1.3321 | 0.9339 | 0.70 🟡 |
| f16\|T\|u64 | 4.0011 | 2.8231 | 0.71 🟡 |
| c128\|strided\|u32 | 0.9967 | 0.7079 | 0.71 🟡 |
| f32\|C\|u32 | 1.3165 | 0.9400 | 0.71 🟡 |
| f16\|negrow\|u64 | 4.0182 | 2.8741 | 0.72 🟡 |
| f32\|F\|u32 | 1.3157 | 0.9411 | 0.72 🟡 |
| f16\|C\|u64 | 3.9940 | 2.8596 | 0.72 🟡 |
| f16\|F\|u64 | 4.0070 | 2.8742 | 0.72 🟡 |
| i32\|negcol\|u16 | 0.9187 | 0.6591 | 0.72 🟡 |
| f16\|sliced\|u64 | 3.9947 | 2.8694 | 0.72 🟡 |
| i64\|negcol\|u32 | 1.3587 | 0.9775 | 0.72 🟡 |
| c128\|F\|bool | 0.8137 | 0.5865 | 0.72 🟡 |
| u64\|negcol\|u32 | 1.3581 | 0.9813 | 0.72 🟡 |
| f16\|strided\|u64 | 2.1976 | 1.6060 | 0.73 🟡 |
| f16\|negcol\|u64 | 4.3789 | 3.2042 | 0.73 🟡 |
| f32\|bcast\|f16 | 4.0202 | 2.9592 | 0.74 🟡 |
| u32\|bcast\|i64 | 1.5893 | 1.1785 | 0.74 🟡 |
| u8\|C\|u8 | 0.0419 | 0.0311 | 0.74 🟡 |
| f32\|bcast\|i64 | 2.1225 | 1.5817 | 0.75 🟡 |
| f16\|bcast\|bool | 1.5366 | 1.1786 | 0.77 🟡 |
| u64\|negrow\|f32 | 1.0318 | 0.7941 | 0.77 🟡 |
| f16\|bcast\|f64 | 2.6438 | 2.0363 | 0.77 🟡 |
| i64\|strided\|i32 | 0.6772 | 0.5280 | 0.78 🟡 |
| char\|bcast\|u64 | 1.9775 | 1.5897 | 0.80 🟡 |
| u64\|strided\|u32 | 0.6767 | 0.5474 | 0.81 🟡 |
| u32\|negcol\|char | 0.9199 | 0.7447 | 0.81 🟡 |
| f16\|negcol\|f64 | 2.5426 | 2.0858 | 0.82 🟡 |
| f32\|negcol\|bool | 0.8970 | 0.7375 | 0.82 🟡 |
| f32\|strided\|bool | 0.4502 | 0.3705 | 0.82 🟡 |
| f32\|negrow\|i64 | 1.9434 | 1.6091 | 0.83 🟡 |
| i64\|strided\|u32 | 0.6769 | 0.5626 | 0.83 🟡 |
| bool\|strided\|bool | 0.2465 | 0.2057 | 0.83 🟡 |
| c128\|negcol\|i32 | 1.4319 | 1.2056 | 0.84 🟡 |
| c128\|negrow\|bool | 0.8742 | 0.7391 | 0.85 🟡 |
| u64\|strided\|i32 | 0.6775 | 0.5767 | 0.85 🟡 |
| f64\|bcast\|i64 | 1.8789 | 1.6027 | 0.85 🟡 |
| f32\|strided\|i64 | 0.9461 | 0.8073 | 0.85 🟡 |
| f16\|bcast\|i16 | 2.4768 | 2.1173 | 0.85 🟡 |
| f16\|strided\|f64 | 1.2617 | 1.0841 | 0.86 🟡 |
| f32\|negcol\|i64 | 1.9531 | 1.6792 | 0.86 🟡 |
| bool\|negcol\|bool | 0.4863 | 0.4193 | 0.86 🟡 |
| c128\|C\|bool | 0.8218 | 0.7095 | 0.86 🟡 |
| i32\|T\|i8 | 0.4088 | 0.3535 | 0.86 🟡 |
| c128\|bcast\|i64 | 1.8134 | 1.5776 | 0.87 🟡 |
| bool\|strided\|u16 | 0.2694 | 0.2351 | 0.87 🟡 |
| i64\|T\|i8 | 0.4090 | 0.3570 | 0.87 🟡 |
| bool\|strided\|char | 0.2692 | 0.2350 | 0.87 🟡 |
| c128\|T\|bool | 0.8127 | 0.7115 | 0.88 🟡 |
| bool\|strided\|u8 | 0.2350 | 0.2058 | 0.88 🟡 |
| f16\|negrow\|f64 | 2.5575 | 2.2402 | 0.88 🟡 |
| f16\|C\|f64 | 2.5232 | 2.2257 | 0.88 🟡 |
| bool\|bcast\|u64 | 1.6358 | 1.4462 | 0.88 🟡 |
| i64\|C\|i8 | 0.4026 | 0.3574 | 0.89 🟡 |
| f32\|sliced\|i64 | 1.8989 | 1.6893 | 0.89 🟡 |
| f32\|C\|i64 | 1.8728 | 1.6669 | 0.89 🟡 |
| bool\|negcol\|i8 | 0.4657 | 0.4158 | 0.89 🟡 |
| u8\|strided\|f16 | 0.7678 | 0.6859 | 0.89 🟡 |
| f32\|T\|i64 | 1.8737 | 1.6776 | 0.90 🟡 |
| i32\|F\|i8 | 0.3947 | 0.3533 | 0.90 🟡 |
| u32\|C\|i8 | 0.3943 | 0.3531 | 0.90 🟡 |
| bool\|negcol\|u8 | 0.4642 | 0.4158 | 0.90 🟡 |
| bool\|negrow\|u8 | 0.4646 | 0.4162 | 0.90 🟡 |
| f32\|F\|i64 | 1.8767 | 1.6815 | 0.90 🟡 |
| i32\|C\|i8 | 0.3943 | 0.3533 | 0.90 🟡 |
| u32\|T\|i8 | 0.3944 | 0.3535 | 0.90 🟡 |
| u32\|F\|i8 | 0.3942 | 0.3536 | 0.90 🟡 |
| bool\|sliced\|u8 | 0.4582 | 0.4112 | 0.90 🟡 |
| f16\|sliced\|f64 | 2.5141 | 2.2656 | 0.90 🟡 |
| i8\|bcast\|i64 | 1.6156 | 1.4696 | 0.91 🟡 |
| c128\|sliced\|bool | 0.7833 | 0.7135 | 0.91 🟡 |
| i16\|bcast\|f64 | 1.6389 | 1.4968 | 0.91 🟡 |
| i32\|bcast\|i64 | 1.5835 | 1.4494 | 0.92 🟡 |
| u8\|bcast\|f64 | 1.7224 | 1.5850 | 0.92 🟡 |
| i8\|bcast\|u64 | 1.5858 | 1.4630 | 0.92 🟡 |
| u16\|strided\|f16 | 0.7439 | 0.6875 | 0.92 🟡 |
| f16\|F\|f64 | 2.4132 | 2.2400 | 0.93 🟡 |
| char\|bcast\|i64 | 1.6684 | 1.5591 | 0.93 🟡 |
| u64\|negrow\|f16 | 1.9400 | 1.8196 | 0.94 🟡 |
| f64\|sliced\|u32 | 1.0641 | 1.0005 | 0.94 🟡 |
| f16\|bcast\|c128 | 3.5628 | 3.3579 | 0.94 🟡 |
| i32\|T\|u8 | 0.3735 | 0.3536 | 0.95 🟡 |
| u64\|bcast\|i64 | 1.6390 | 1.5547 | 0.95 🟡 |
| f32\|bcast\|f64 | 1.6309 | 1.5473 | 0.95 🟡 |
| f16\|T\|f64 | 2.3716 | 2.2520 | 0.95 🟡 |
| f64\|negrow\|u32 | 1.0785 | 1.0264 | 0.95 🟡 |
| i32\|bcast\|u64 | 1.6317 | 1.5597 | 0.96 🟡 |
| u64\|sliced\|f16 | 1.8987 | 1.8166 | 0.96 🟡 |
| f64\|F\|u32 | 1.0542 | 1.0102 | 0.96 🟡 |
| u64\|C\|f16 | 1.8664 | 1.7950 | 0.96 🟡 |
| f64\|C\|u32 | 1.0543 | 1.0174 | 0.97 🟡 |
| bool\|bcast\|i64 | 1.6615 | 1.6041 | 0.97 🟡 |
| bool\|bcast\|f64 | 1.5764 | 1.5300 | 0.97 🟡 |
| i32\|strided\|f16 | 0.7054 | 0.6857 | 0.97 🟡 |
| c128\|bcast\|f64 | 1.6403 | 1.5960 | 0.97 🟡 |
| u16\|bcast\|i64 | 1.6091 | 1.5658 | 0.97 🟡 |
| char\|bcast\|f64 | 1.6014 | 1.5610 | 0.97 🟡 |
| u8\|strided\|bool | 0.2461 | 0.2405 | 0.98 🟡 |
| i16\|bcast\|u64 | 1.6098 | 1.5743 | 0.98 🟡 |
| i8\|strided\|bool | 0.2458 | 0.2405 | 0.98 🟡 |
| i64\|bcast\|f64 | 1.5806 | 1.5487 | 0.98 🟡 |
| i8\|bcast\|f64 | 1.5900 | 1.5581 | 0.98 🟡 |
| i32\|bcast\|f64 | 1.5886 | 1.5573 | 0.98 🟡 |
| f32\|negcol\|i32 | 0.9468 | 0.9285 | 0.98 🟡 |
| u8\|bcast\|i64 | 1.6199 | 1.5892 | 0.98 🟡 |
| f64\|negcol\|i64 | 1.7569 | 1.7243 | 0.98 🟡 |
| u32\|bcast\|f64 | 1.6055 | 1.5768 | 0.98 🟡 |
| i64\|bcast\|i64 | 1.6563 | 1.6281 | 0.98 🟡 |
| i16\|bcast\|i64 | 1.6036 | 1.5782 | 0.98 🟡 |
| i64\|bcast\|u64 | 1.6060 | 1.5873 | 0.99 🟡 |
| f64\|bcast\|f64 | 1.6164 | 1.5989 | 0.99 🟡 |
| u8\|bcast\|u64 | 1.5978 | 1.5839 | 0.99 🟡 |
| u16\|bcast\|u64 | 1.5788 | 1.5677 | 0.99 🟡 |
| f64\|T\|u32 | 1.0535 | 1.0469 | 0.99 🟡 |
| f64\|negrow\|i64 | 1.7433 | 1.7328 | 0.99 🟡 |
| c128\|negcol\|bool | 0.8538 | 0.8498 | 1.00 🟡 |
| u64\|bcast\|u64 | 1.6033 | 1.5959 | 1.00 🟡 |
| u64\|T\|i16 | 0.4115 | 0.4099 | 1.00 🟡 |
| u32\|bcast\|u64 | 1.5699 | 1.5676 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 381 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best window (min over ~200 ms):
  a*b+c       fused    3.42 ms   unfused    5.68 ms   (1.66x)
  (a-b)/(a+b) fused    2.83 ms   unfused   11.66 ms   (4.12x)
  sum(a*b)    fused    2.17 ms   unfused    3.67 ms   (1.69x)
  sum(af*bf)  fused    1.22 ms   unfused    1.64 ms   (1.34x)  [f32]
  a*b+c out=  fused    3.33 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.73 ms   unfused    3.91 ms   (1.43x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.44 ms   unfused    5.90 ms   (1.72x)
    [F      ] fused   11.76 ms   unfused   17.82 ms   (1.52x)
    [T      ] fused   11.82 ms   unfused   17.38 ms   (1.47x)
    [strided] fused   11.84 ms   unfused   16.55 ms   (1.40x)
    [bcast  ] fused    2.23 ms   unfused    9.12 ms   (4.10x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best window (min over ~200 ms):
  a*b+c         12.68 ms
  (a-b)/(a+b)   19.13 ms
  sum(a*b)       8.42 ms
  sum(af*bf)     4.23 ms  [f32]
  a*b+c out=     5.08 ms  [two-pass with out=]
  i4*2+f8        9.93 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.83 ms
    [F      ]   12.50 ms
    [T      ]   12.59 ms
    [strided]    7.87 ms
    [bcast  ]   12.19 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-c3b80689652f6d539fa08543e6c9e4277204935d5b149c15f879e1ce51c9d9b5\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b) [ndarray]` | 1,000 | 0.0004 ms | 0.0006 ms | 0.0004 | 1.000x |
| `a.dot(b) [ndarray]` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0068 | 0.557x |
| `a.dot(b) [ndarray]` | 10,000,000 | 7.5925 ms | 8.3942 ms | 8.3031 | 1.094x |
| `np.convolve(a, kernel)` | 1,000 | 0.0131 ms | 0.0424 ms | 0.0235 | 1.794x |
| `np.convolve(a, kernel)` | 100,000 | 0.3791 ms | 1.9148 ms | 2.2177 | 5.850x |
| `np.convolve(a, kernel)` | 10,000,000 | 46.4639 ms | 206.6472 ms | 199.3455 | 4.290x |
| `np.correlate(a, kernel)` | 1,000 | 0.0043 ms | 0.0332 ms | 0.0224 | 5.209x |
| `np.correlate(a, kernel)` | 100,000 | 0.3706 ms | 1.9150 ms | 2.1965 | 5.927x |
| `np.correlate(a, kernel)` | 10,000,000 | 46.2195 ms | 166.2216 ms | 206.0500 | 4.458x |
| `np.dot(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0005 | 1.250x |
| `np.dot(a, b)` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0069 | 0.566x |
| `np.dot(a, b)` | 10,000,000 | 7.9269 ms | 4.4499 ms | 7.4223 | 1.668x |
| `np.einsum('ij,jk->ik', a, b)` | 1,000 | 0.0039 ms | 0.0021 ms | 0.0061 | 2.905x |
| `np.einsum('ij,jk->ik', a, b)` | 100,000 | 3.8778 ms | 2.2524 ms | 5.1769 | 2.298x |
| `np.einsum('ij,jk->ik', a, b)` | 10,000,000 | 6.4587 ms | 4.0000 ms | 9.6582 | 2.415x |
| `np.einsum(subscripts, operands)` | 1,000 | 0.0024 ms | 0.0011 ms | 0.0014 | 1.273x |
| `np.einsum(subscripts, operands)` | 100,000 | 0.0134 ms | 0.0134 ms | 0.0394 | 2.940x |
| `np.einsum(subscripts, operands)` | 10,000,000 | 8.4578 ms | 8.4570 ms | 9.2857 | 1.098x |
| `np.inner(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0006 | 1.500x |
| `np.inner(a, b)` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0070 | 0.574x |
| `np.inner(a, b)` | 10,000,000 | 7.9907 ms | 7.6714 ms | 8.3426 | 1.087x |
| `np.linalg.cholesky(a)` | 1,000 | missing_backend | 0.0034 ms | 0.0043 | 1.265x |
| `np.linalg.cholesky(a)` | 100,000 | missing_backend | 0.0326 ms | 0.0174 | 0.534x |
| `np.linalg.cholesky(a)` | 10,000,000 | missing_backend | 0.0680 ms | 0.1038 | 1.526x |
| `np.linalg.cond(a)` | 1,000 | missing_backend | 0.0533 ms | 0.0349 | 0.655x |
| `np.linalg.cond(a)` | 100,000 | missing_backend | 0.4661 ms | 0.4757 | 1.021x |
| `np.linalg.cond(a)` | 10,000,000 | missing_backend | 0.9824 ms | 0.9891 | 1.007x |
| `np.linalg.det(a)` | 1,000 | 0.0043 ms | 0.0080 ms | 0.0126 | 2.930x |
| `np.linalg.det(a)` | 100,000 | 0.0728 ms | 0.0543 ms | 0.0587 | 1.081x |
| `np.linalg.det(a)` | 10,000,000 | 0.1875 ms | 0.1129 ms | 0.1187 | 1.051x |
| `np.linalg.eig(a)` | 1,000 | missing_backend | 0.1941 ms | 0.2033 | 1.047x |
| `np.linalg.eig(a)` | 100,000 | missing_backend | 2.6725 ms | 2.6986 | 1.010x |
| `np.linalg.eig(a)` | 10,000,000 | missing_backend | 5.8937 ms | 5.8331 | 0.990x |
| `np.linalg.eigh(a)` | 1,000 | missing_backend | 0.0847 ms | 0.0463 | 0.547x |
| `np.linalg.eigh(a)` | 100,000 | missing_backend | 0.8180 ms | 0.4004 | 0.489x |
| `np.linalg.eigh(a)` | 10,000,000 | missing_backend | 1.5346 ms | 1.5458 | 1.007x |
| `np.linalg.eigvals(a)` | 1,000 | missing_backend | 0.1109 ms | 0.1279 | 1.153x |
| `np.linalg.eigvals(a)` | 100,000 | missing_backend | 1.5514 ms | 1.5435 | 0.995x |
| `np.linalg.eigvals(a)` | 10,000,000 | missing_backend | 3.2028 ms | 3.0426 | 0.950x |
| `np.linalg.eigvalsh(a)` | 1,000 | missing_backend | 0.0294 ms | 0.0366 | 1.245x |
| `np.linalg.eigvalsh(a)` | 100,000 | missing_backend | 0.2839 ms | 0.2893 | 1.019x |
| `np.linalg.eigvalsh(a)` | 10,000,000 | missing_backend | 0.5183 ms | 0.5261 | 1.015x |
| `np.linalg.inv(a)` | 1,000 | 0.0125 ms | 0.0157 ms | 0.0239 | 1.912x |
| `np.linalg.inv(a)` | 100,000 | 0.2485 ms | 0.1604 ms | 0.1823 | 1.137x |
| `np.linalg.inv(a)` | 10,000,000 | 0.5972 ms | 0.3403 ms | 0.3679 | 1.081x |
| `np.linalg.lstsq(a, b)` | 1,000 | missing_backend | 0.1518 ms | 0.1637 | 1.078x |
| `np.linalg.lstsq(a, b)` | 100,000 | missing_backend | 1.6220 ms | 1.6285 | 1.004x |
| `np.linalg.lstsq(a, b)` | 10,000,000 | missing_backend | 2.9606 ms | 2.9627 | 1.001x |
| `np.linalg.matmul(a, b)` | 1,000 | 0.0036 ms | 0.0016 ms | 0.0023 | 1.437x |
| `np.linalg.matmul(a, b)` | 100,000 | 3.8721 ms | 2.2462 ms | 2.2518 | 1.002x |
| `np.linalg.matmul(a, b)` | 10,000,000 | 6.4165 ms | 3.9929 ms | 4.2537 | 1.065x |
| `np.linalg.matrix_norm(a, ord=2)` | 1,000 | missing_backend | 0.0532 ms | 0.0695 | 1.306x |
| `np.linalg.matrix_norm(a, ord=2)` | 100,000 | missing_backend | 0.4653 ms | 0.4811 | 1.034x |
| `np.linalg.matrix_norm(a, ord=2)` | 10,000,000 | missing_backend | 0.9814 ms | 0.9909 | 1.010x |
| `np.linalg.matrix_power(a, -1)` | 1,000 | 0.0133 ms | 0.0165 ms | 0.0257 | 1.932x |
| `np.linalg.matrix_power(a, -1)` | 100,000 | 0.2505 ms | 0.1635 ms | 0.1843 | 1.127x |
| `np.linalg.matrix_power(a, -1)` | 10,000,000 | 0.6002 ms | 0.3473 ms | 0.3715 | 1.070x |
| `np.linalg.matrix_power(a, 3)` | 1,000 | 0.0155 ms | 0.0078 ms | 0.0052 | 0.667x |
| `np.linalg.matrix_power(a, 3)` | 100,000 | 0.7173 ms | 0.3164 ms | 0.3235 | 1.022x |
| `np.linalg.matrix_power(a, 3)` | 10,000,000 | 0.7173 ms | 0.3163 ms | 0.3239 | 1.024x |
| `np.linalg.matrix_rank(a)` | 1,000 | missing_backend | 0.0559 ms | 0.0717 | 1.283x |
| `np.linalg.matrix_rank(a)` | 100,000 | missing_backend | 0.4716 ms | 0.4822 | 1.022x |
| `np.linalg.matrix_rank(a)` | 10,000,000 | missing_backend | 0.9822 ms | 0.9950 | 1.013x |
| `np.linalg.multi_dot(arrays)` | 1,000 | 0.0157 ms | 0.0078 ms | 0.0051 | 0.654x |
| `np.linalg.multi_dot(arrays)` | 100,000 | 0.7175 ms | 0.3163 ms | 0.3291 | 1.040x |
| `np.linalg.multi_dot(arrays)` | 10,000,000 | 0.7174 ms | 0.3166 ms | 0.3290 | 1.039x |
| `np.linalg.norm(a, ord=2)` | 1,000 | missing_backend | 0.0319 ms | 0.0696 | 2.182x |
| `np.linalg.norm(a, ord=2)` | 100,000 | missing_backend | 0.4662 ms | 0.4809 | 1.032x |
| `np.linalg.norm(a, ord=2)` | 10,000,000 | missing_backend | 0.9787 ms | 0.9930 | 1.015x |
| `np.linalg.pinv(a)` | 1,000 | missing_backend | 0.1601 ms | 0.0845 | 0.528x |
| `np.linalg.pinv(a)` | 100,000 | missing_backend | 1.5796 ms | 1.5793 | 1.000x |
| `np.linalg.pinv(a)` | 10,000,000 | missing_backend | 3.0850 ms | 3.2965 | 1.069x |
| `np.linalg.qr(a)` | 1,000 | missing_backend | 0.0333 ms | 0.0225 | 0.676x |
| `np.linalg.qr(a)` | 100,000 | missing_backend | 0.3141 ms | 0.3839 | 1.222x |
| `np.linalg.qr(a)` | 10,000,000 | missing_backend | 0.6888 ms | 0.7958 | 1.155x |
| `np.linalg.slogdet(a)` | 1,000 | 0.0046 ms | 0.0080 ms | 0.0144 | 3.130x |
| `np.linalg.slogdet(a)` | 100,000 | 0.0322 ms | 0.0548 ms | 0.0604 | 1.876x |
| `np.linalg.slogdet(a)` | 10,000,000 | 0.1889 ms | 0.1133 ms | 0.1195 | 1.055x |
| `np.linalg.solve(a, b)` | 1,000 | 0.0071 ms | 0.0083 ms | 0.0071 | 1.000x |
| `np.linalg.solve(a, b)` | 100,000 | 0.0464 ms | 0.0579 ms | 0.0640 | 1.379x |
| `np.linalg.solve(a, b)` | 10,000,000 | 0.2406 ms | 0.1187 ms | 0.1247 | 1.051x |
| `np.linalg.svd(a)` | 1,000 | missing_backend | 0.1357 ms | 0.1490 | 1.098x |
| `np.linalg.svd(a)` | 100,000 | missing_backend | 1.4284 ms | 1.4689 | 1.028x |
| `np.linalg.svd(a)` | 10,000,000 | missing_backend | 2.7944 ms | 2.9843 | 1.068x |
| `np.linalg.svdvals(a)` | 1,000 | missing_backend | 0.0474 ms | 0.0558 | 1.177x |
| `np.linalg.svdvals(a)` | 100,000 | missing_backend | 0.4632 ms | 0.4667 | 1.008x |
| `np.linalg.svdvals(a)` | 10,000,000 | missing_backend | 0.9755 ms | 0.9773 | 1.002x |
| `np.linalg.tensordot(a, b)` | 1,000 | 0.0041 ms | 0.0048 ms | 0.0056 | 1.366x |
| `np.linalg.tensordot(a, b)` | 100,000 | 0.3594 ms | 0.1591 ms | 0.1721 | 1.082x |
| `np.linalg.tensordot(a, b)` | 10,000,000 | 0.3621 ms | 0.1591 ms | 0.1721 | 1.082x |
| `np.linalg.tensorinv(a)` | 1,000 | 0.0128 ms | 0.0158 ms | 0.0259 | 2.023x |
| `np.linalg.tensorinv(a)` | 100,000 | 0.2490 ms | 0.1609 ms | 0.1845 | 1.147x |
| `np.linalg.tensorinv(a)` | 10,000,000 | 0.5967 ms | 0.3431 ms | 0.3720 | 1.084x |
| `np.linalg.tensorsolve(a, b)` | 1,000 | 0.0076 ms | 0.0089 ms | 0.0184 | 2.421x |
| `np.linalg.tensorsolve(a, b)` | 100,000 | 0.1032 ms | 0.0587 ms | 0.0667 | 1.136x |
| `np.linalg.tensorsolve(a, b)` | 10,000,000 | 0.2408 ms | 0.1198 ms | 0.1276 | 1.065x |
| `np.linalg.vecdot(a, b)` | 1,000 | 0.0018 ms | 0.0008 ms | 0.0008 | 1.000x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0341 ms | 0.0123 ms | 0.0073 | 0.593x |
| `np.linalg.vecdot(a, b)` | 10,000,000 | 29.2990 ms | 8.4696 ms | 8.4273 | 0.995x |
| `np.matmul(a, b)` | 1,000 | 0.0036 ms | 0.0016 ms | 0.0022 | 1.375x |
| `np.matmul(a, b)` | 100,000 | 3.8830 ms | 2.2470 ms | 2.2609 | 1.006x |
| `np.matmul(a, b)` | 10,000,000 | 6.5035 ms | 3.9891 ms | 4.2523 | 1.066x |
| `np.matvec(a, b)` | 1,000 | 0.0010 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.matvec(a, b)` | 100,000 | 0.0155 ms | 0.0098 ms | 0.0116 | 1.184x |
| `np.matvec(a, b)` | 10,000,000 | 0.0216 ms | 0.0142 ms | 0.0162 | 1.141x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.0660 ms | 0.0448 | 0.679x |
| `np.polyfit(x, y, 3)` | 100,000 | missing_backend | 1.1174 ms | 0.8973 | 0.803x |
| `np.polyfit(x, y, 3)` | 10,000,000 | missing_backend | 7.0263 ms | 8.5729 | 1.220x |
| `np.roots(p)` | 1,000 | missing_backend | 0.0736 ms | 0.0878 | 1.193x |
| `np.roots(p)` | 100,000 | missing_backend | 0.1891 ms | 0.2035 | 1.076x |
| `np.roots(p)` | 10,000,000 | missing_backend | 0.7742 ms | 0.7704 | 0.995x |
| `np.tensordot(a, b)` | 1,000 | 0.0013 ms | 0.0013 ms | 0.0036 | 2.769x |
| `np.tensordot(a, b)` | 100,000 | 0.0134 ms | 0.0134 ms | 0.0209 | 1.560x |
| `np.tensordot(a, b)` | 10,000,000 | 7.2145 ms | 7.7939 ms | 8.5836 | 1.190x |
| `np.tensordot(a, b, axes=1)` | 1,000 | 0.0041 ms | 0.0048 ms | 0.0054 | 1.317x |
| `np.tensordot(a, b, axes=1)` | 100,000 | 3.8851 ms | 2.2485 ms | 2.2798 | 1.014x |
| `np.tensordot(a, b, axes=1)` | 10,000,000 | 6.4310 ms | 3.9968 ms | 4.2931 | 1.074x |
| `np.vdot(a, b)` | 1,000 | 0.0010 ms | 0.0007 ms | 0.0005 | 0.714x |
| `np.vdot(a, b)` | 100,000 | 0.0125 ms | 0.0122 ms | 0.0070 | 0.574x |
| `np.vdot(a, b)` | 10,000,000 | 7.9176 ms | 8.2873 ms | 8.3329 | 1.052x |
| `np.vecdot(a, b)` | 1,000 | 0.0026 ms | 0.0008 ms | 0.0006 | 0.750x |
| `np.vecdot(a, b)` | 100,000 | 0.0341 ms | 0.0123 ms | 0.0070 | 0.569x |
| `np.vecdot(a, b)` | 10,000,000 | 29.1383 ms | 8.4108 ms | 8.2750 | 0.984x |
| `np.vecmat(a, b)` | 1,000 | 0.0009 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.vecmat(a, b)` | 100,000 | 0.0098 ms | 0.0117 ms | 0.0136 | 1.388x |
| `np.vecmat(a, b)` | 10,000,000 | 0.0272 ms | 0.0176 ms | 0.0194 | 1.102x |
