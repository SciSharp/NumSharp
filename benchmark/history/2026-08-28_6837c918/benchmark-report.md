# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell; the winner names the implementation that actually executed, not merely the process profile.

**Timing basis:** best window (min) of each side's per-case sweep — min-based harnesses run each case to a ~200 ms time budget (a >20 ms/call op runs exactly 100 times), while the C# op-matrix keeps BenchmarkDotNet's 50 iterations; interference only adds time, so the fastest window estimates intrinsic cost; per-case means are retained in the JSON (`numpy_mean_ms`/`numsharp_mean_ms`) as tail diagnostics.

- Managed effective cells: **3,640**
- OpenBLAS effective cells: **105**
- Missing backend records: **69**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.06630 ms | openblas | 0.677x |
| `np.polyfit(x, y, 3)` | float64 | 100,000 | missing_backend | 0.6043 ms | openblas | 0.767x |
| `np.polyfit(x, y, 3)` | float64 | 10,000,000 | missing_backend | 4.2502 ms | openblas | 1.215x |
| `np.roots(p)` | float64 | 1,000 | missing_backend | 0.03410 ms | openblas | 1.073x |
| `np.roots(p)` | float64 | 100,000 | missing_backend | 0.09470 ms | openblas | 1.032x |
| `np.roots(p)` | float64 | 10,000,000 | missing_backend | 0.4145 ms | openblas | 0.988x |
| `np.array2string(a) (float64)` | float64 | 1 | 0.000406 ms | not_measured | managed | 40.239x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 0.9947 ms | not_measured | managed | 1.998x |
| `np.array_repr(a) (float64)` | float64 | 1 | 0.000575 ms | not_measured | managed | 30.927x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 1.0535 ms | not_measured | managed | 1.891x |
| `np.array_str(a) (float64)` | float64 | 1 | 0.000430 ms | not_measured | managed | 39.242x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 0.9131 ms | not_measured | managed | 2.171x |
| `np.can_cast(from, to) (float64)` | float64 | 1 | 0.000014 ms | not_measured | managed | 15.071x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.000024 ms | not_measured | managed | 8.750x |
| `np.common_type(a, b) (float64)` | float64 | 1 | 0.000017 ms | not_measured | managed | 48.059x |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.000030 ms | not_measured | managed | 27.167x |
| `np.copyto(dst, src) (float64)` | float64 | 1 | 0.000204 ms | not_measured | managed | 1.363x |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.000386 ms | not_measured | managed | 1.407x |
| `np.format_float_positional(x) (float64)` | float64 | 1 | 0.000293 ms | not_measured | managed | 1.618x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.000433 ms | not_measured | managed | 1.106x |
| `np.format_float_scientific(x) (float64)` | float64 | 1 | 0.000426 ms | not_measured | managed | 1.117x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.000407 ms | not_measured | managed | 1.157x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.04549 ms | not_measured | managed | 0.693x |
| `np.get_printoptions() (float64)` | float64 | 1 | 0.000242 ms | not_measured | managed | 1.343x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.000285 ms | not_measured | managed | 1.123x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1 | 0.000038 ms | not_measured | managed | 18.105x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000036 ms | not_measured | managed | 18.639x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1 | 0.000032 ms | not_measured | managed | 5.656x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000033 ms | not_measured | managed | 5.364x |
| `np.iterable(a) (float64)` | float64 | 1 | 0.000000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.05800 ms | not_measured | managed | 1.064x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.6182 ms | not_measured | managed | 0.451x |
| `np.min_scalar_type(value) (float64)` | float64 | 1 | 0.000003 ms | not_measured | managed | 90.000x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.000003 ms | not_measured | managed | 90.333x |
| `np.mintypecode(typechars) (float64)` | float64 | 1 | 0.000174 ms | not_measured | managed | 4.086x |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.000193 ms | not_measured | managed | 3.715x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1 | 0.00177 ms | not_measured | managed | 0.269x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.00199 ms | not_measured | managed | 0.238x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.1466 ms | not_measured | managed | 0.153x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.000934 ms | not_measured | managed | 1.792x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.00191 ms | not_measured | managed | 1.320x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.2047 ms | not_measured | managed | 0.511x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.00451 ms | not_measured | managed | 0.655x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.01013 ms | not_measured | managed | 1.427x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.00215 ms | not_measured | managed | 0.808x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1051 ms | not_measured | managed | 0.191x |
| `np.printoptions() (float64)` | float64 | 1 | 0.000127 ms | not_measured | managed | 25.882x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.000089 ms | not_measured | managed | 38.022x |
| `np.promote_types(a, b) (float64)` | float64 | 1 | 0.000012 ms | not_measured | managed | 11.083x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.000012 ms | not_measured | managed | 11.083x |
| `np.result_type(a, b) (float64)` | float64 | 1 | 0.000005 ms | not_measured | managed | 29.400x |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.000006 ms | not_measured | managed | 24.667x |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.3620 ms | not_measured | managed | 0.368x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 1.1584 ms | not_measured | managed | 1.385x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 1.3628 ms | not_measured | managed | 0.141x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.3378 ms | not_measured | managed | 0.930x |
| `np.set_printoptions() (float64)` | float64 | 1 | 0.000050 ms | not_measured | managed | 39.620x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.000051 ms | not_measured | managed | 38.078x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.00776 ms | not_measured | managed | 1.787x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.1173 ms | not_measured | managed | 1.449x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 112.0555 ms | not_measured | managed | 1.510x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.00604 ms | not_measured | managed | 1.902x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.0860 ms | not_measured | managed | 1.295x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 116.2286 ms | not_measured | managed | 1.295x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.00194 ms | not_measured | managed | 0.954x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.4505 ms | not_measured | managed | 0.860x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 46.6106 ms | not_measured | managed | 0.949x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.00312 ms | not_measured | managed | 1.142x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.5213 ms | not_measured | managed | 0.747x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 58.9948 ms | not_measured | managed | 0.833x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.00569 ms | not_measured | managed | 2.019x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 0.9668 ms | not_measured | managed | 1.518x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 99.4526 ms | not_measured | managed | 1.575x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.00478 ms | not_measured | managed | 2.131x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 0.8844 ms | not_measured | managed | 1.410x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 96.8340 ms | not_measured | managed | 1.423x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.00150 ms | not_measured | managed | 1.013x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.4103 ms | not_measured | managed | 0.874x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 44.0635 ms | not_measured | managed | 0.962x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.00232 ms | not_measured | managed | 1.355x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.5008 ms | not_measured | managed | 0.718x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 57.0377 ms | not_measured | managed | 0.833x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.00225 ms | not_measured | managed | 0.392x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.07812 ms | not_measured | managed | 3.455x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 25.0328 ms | not_measured | managed | 1.201x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.00553 ms | not_measured | managed | 0.617x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.4625 ms | not_measured | managed | 0.594x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 46.4492 ms | not_measured | managed | 0.649x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.00143 ms | not_measured | managed | 0.478x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.01334 ms | not_measured | managed | 0.447x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 5.0193 ms | not_measured | managed | 1.560x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.00193 ms | not_measured | managed | 0.385x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.02748 ms | not_measured | managed | 0.417x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 12.7846 ms | not_measured | managed | 1.231x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.00121 ms | not_measured | managed | 0.702x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.00730 ms | not_measured | managed | 2.956x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 2.9667 ms | not_measured | managed | 1.409x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.00125 ms | not_measured | managed | 0.680x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.01250 ms | not_measured | managed | 1.724x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 4.9130 ms | not_measured | managed | 1.542x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.00180 ms | not_measured | managed | 0.458x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.02678 ms | not_measured | managed | 0.802x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 12.2583 ms | not_measured | managed | 1.216x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.00107 ms | not_measured | managed | 0.746x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.00530 ms | not_measured | managed | 4.054x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 0.9076 ms | not_measured | managed | 3.309x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.00122 ms | not_measured | managed | 0.706x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.00730 ms | not_measured | managed | 2.946x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 2.7717 ms | not_measured | managed | 1.506x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.00156 ms | not_measured | managed | 0.546x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.01333 ms | not_measured | managed | 1.616x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 4.7521 ms | not_measured | managed | 1.764x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.00179 ms | not_measured | managed | 0.481x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.02796 ms | not_measured | managed | 0.771x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 12.7819 ms | not_measured | managed | 1.159x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.00128 ms | not_measured | managed | 0.622x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.00505 ms | not_measured | managed | 4.260x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 0.9066 ms | not_measured | managed | 3.382x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.00117 ms | not_measured | managed | 0.566x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.08038 ms | not_measured | managed | 3.341x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 25.1761 ms | not_measured | managed | 1.150x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.00489 ms | not_measured | managed | 0.640x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.4634 ms | not_measured | managed | 0.593x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 46.9236 ms | not_measured | managed | 0.636x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.000620 ms | not_measured | managed | 0.640x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.01232 ms | not_measured | managed | 0.462x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 4.8975 ms | not_measured | managed | 1.553x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.000717 ms | not_measured | managed | 0.636x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.02631 ms | not_measured | managed | 0.413x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 12.8866 ms | not_measured | managed | 1.171x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.000588 ms | not_measured | managed | 1.094x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.00658 ms | not_measured | managed | 4.142x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 2.7264 ms | not_measured | managed | 1.693x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.000601 ms | not_measured | managed | 1.065x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.01230 ms | not_measured | managed | 2.234x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 4.8601 ms | not_measured | managed | 1.553x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.000719 ms | not_measured | managed | 0.896x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.02600 ms | not_measured | managed | 1.059x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 12.6832 ms | not_measured | managed | 1.202x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.000462 ms | not_measured | managed | 1.290x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.00553 ms | not_measured | managed | 4.914x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 0.9627 ms | not_measured | managed | 3.772x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.000586 ms | not_measured | managed | 1.092x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.00695 ms | not_measured | managed | 3.913x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 2.6970 ms | not_measured | managed | 1.784x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.000615 ms | not_measured | managed | 1.039x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.01228 ms | not_measured | managed | 2.235x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 4.8956 ms | not_measured | managed | 1.816x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.000714 ms | not_measured | managed | 0.902x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.02574 ms | not_measured | managed | 1.077x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 13.4442 ms | not_measured | managed | 1.052x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.000479 ms | not_measured | managed | 1.259x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.00606 ms | not_measured | managed | 4.490x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 0.9447 ms | not_measured | managed | 3.795x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.00124 ms | not_measured | managed | 0.570x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.09523 ms | not_measured | managed | 2.913x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 27.6590 ms | not_measured | managed | 1.195x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.00488 ms | not_measured | managed | 0.641x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.4597 ms | not_measured | managed | 0.596x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 46.9280 ms | not_measured | managed | 0.641x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.000606 ms | not_measured | managed | 0.668x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.01408 ms | not_measured | managed | 0.484x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 6.8134 ms | not_measured | managed | 1.249x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.000734 ms | not_measured | managed | 0.646x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.03503 ms | not_measured | managed | 0.663x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 14.4652 ms | not_measured | managed | 1.187x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.000582 ms | not_measured | managed | 1.096x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.00763 ms | not_measured | managed | 3.577x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 3.1337 ms | not_measured | managed | 1.539x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.000612 ms | not_measured | managed | 1.052x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.01473 ms | not_measured | managed | 1.868x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 5.7445 ms | not_measured | managed | 1.424x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.000719 ms | not_measured | managed | 0.903x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.03549 ms | not_measured | managed | 0.836x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 14.6911 ms | not_measured | managed | 1.134x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.000410 ms | not_measured | managed | 1.476x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.00551 ms | not_measured | managed | 4.942x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 1.1675 ms | not_measured | managed | 3.095x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.000594 ms | not_measured | managed | 1.066x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.00718 ms | not_measured | managed | 3.784x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.1694 ms | not_measured | managed | 1.547x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.000612 ms | not_measured | managed | 1.051x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.01359 ms | not_measured | managed | 2.031x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 5.6854 ms | not_measured | managed | 2.146x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.000750 ms | not_measured | managed | 0.863x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.03532 ms | not_measured | managed | 0.887x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 15.3227 ms | not_measured | managed | 1.072x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.000495 ms | not_measured | managed | 1.214x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.00567 ms | not_measured | managed | 4.800x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.2646 ms | not_measured | managed | 2.869x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.00114 ms | not_measured | managed | 0.658x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.07732 ms | not_measured | managed | 3.500x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 24.6741 ms | not_measured | managed | 1.229x |
| `a * scalar (float16)` | float16 | 1,000 | 0.00499 ms | not_measured | managed | 0.655x |
| `a * scalar (float16)` | float16 | 100,000 | 0.4589 ms | not_measured | managed | 0.598x |
| `a * scalar (float16)` | float16 | 10,000,000 | 46.4201 ms | not_measured | managed | 0.647x |
| `a * scalar (float32)` | float32 | 1,000 | 0.000630 ms | not_measured | managed | 0.851x |
| `a * scalar (float32)` | float32 | 100,000 | 0.01201 ms | not_measured | managed | 0.485x |
| `a * scalar (float32)` | float32 | 10,000,000 | 4.8826 ms | not_measured | managed | 1.614x |
| `a * scalar (float64)` | float64 | 1,000 | 0.000730 ms | not_measured | managed | 0.825x |
| `a * scalar (float64)` | float64 | 100,000 | 0.02595 ms | not_measured | managed | 0.436x |
| `a * scalar (float64)` | float64 | 10,000,000 | 15.1138 ms | not_measured | managed | 1.030x |
| `a * scalar (int16)` | int16 | 1,000 | 0.000561 ms | not_measured | managed | 1.294x |
| `a * scalar (int16)` | int16 | 100,000 | 0.00610 ms | not_measured | managed | 3.511x |
| `a * scalar (int16)` | int16 | 10,000,000 | 2.8730 ms | not_measured | managed | 1.463x |
| `a * scalar (int32)` | int32 | 1,000 | 0.000610 ms | not_measured | managed | 1.197x |
| `a * scalar (int32)` | int32 | 100,000 | 0.01190 ms | not_measured | managed | 1.801x |
| `a * scalar (int32)` | int32 | 10,000,000 | 4.9378 ms | not_measured | managed | 1.544x |
| `a * scalar (int64)` | int64 | 1,000 | 0.000730 ms | not_measured | managed | 1.007x |
| `a * scalar (int64)` | int64 | 100,000 | 0.02535 ms | not_measured | managed | 0.844x |
| `a * scalar (int64)` | int64 | 10,000,000 | 12.1059 ms | not_measured | managed | 1.240x |
| `a * scalar (int8)` | int8 | 1,000 | 0.000491 ms | not_measured | managed | 1.403x |
| `a * scalar (int8)` | int8 | 100,000 | 0.00411 ms | not_measured | managed | 5.191x |
| `a * scalar (int8)` | int8 | 10,000,000 | 0.9231 ms | not_measured | managed | 3.245x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.000590 ms | not_measured | managed | 1.246x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.00617 ms | not_measured | managed | 3.467x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 2.7633 ms | not_measured | managed | 1.578x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.000619 ms | not_measured | managed | 1.192x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.01174 ms | not_measured | managed | 1.825x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 4.8933 ms | not_measured | managed | 2.054x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.000731 ms | not_measured | managed | 1.010x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.02559 ms | not_measured | managed | 0.837x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 12.6139 ms | not_measured | managed | 1.151x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.000487 ms | not_measured | managed | 1.427x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.00398 ms | not_measured | managed | 5.375x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 0.9018 ms | not_measured | managed | 3.421x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.00211 ms | not_measured | managed | 0.416x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.08955 ms | not_measured | managed | 3.054x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 24.5024 ms | not_measured | managed | 1.295x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.00547 ms | not_measured | managed | 0.622x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.4748 ms | not_measured | managed | 0.578x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 46.2623 ms | not_measured | managed | 0.653x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.00155 ms | not_measured | managed | 0.438x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.01614 ms | not_measured | managed | 0.370x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 4.9387 ms | not_measured | managed | 1.606x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.00179 ms | not_measured | managed | 0.414x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.02827 ms | not_measured | managed | 0.406x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 12.9385 ms | not_measured | managed | 1.270x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.00131 ms | not_measured | managed | 0.665x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.00742 ms | not_measured | managed | 3.217x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 2.9692 ms | not_measured | managed | 1.433x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.00117 ms | not_measured | managed | 0.748x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.01270 ms | not_measured | managed | 1.829x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 5.0081 ms | not_measured | managed | 1.448x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.00178 ms | not_measured | managed | 0.473x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.02805 ms | not_measured | managed | 0.826x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 13.3324 ms | not_measured | managed | 1.413x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.00103 ms | not_measured | managed | 0.795x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.00456 ms | not_measured | managed | 5.176x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 1.3182 ms | not_measured | managed | 2.489x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.00124 ms | not_measured | managed | 0.706x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.00746 ms | not_measured | managed | 3.193x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 2.9214 ms | not_measured | managed | 1.489x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.00159 ms | not_measured | managed | 0.549x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.01354 ms | not_measured | managed | 1.715x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 5.1389 ms | not_measured | managed | 2.413x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.00176 ms | not_measured | managed | 0.498x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.03132 ms | not_measured | managed | 0.743x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 13.2276 ms | not_measured | managed | 1.140x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.00113 ms | not_measured | managed | 0.725x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.00473 ms | not_measured | managed | 5.055x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 1.0187 ms | not_measured | managed | 3.235x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.00114 ms | not_measured | managed | 0.575x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.09341 ms | not_measured | managed | 2.984x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 28.8461 ms | not_measured | managed | 1.108x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.00485 ms | not_measured | managed | 0.643x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.4666 ms | not_measured | managed | 0.589x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 46.7085 ms | not_measured | managed | 0.651x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.000627 ms | not_measured | managed | 0.646x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.01329 ms | not_measured | managed | 0.504x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 5.5894 ms | not_measured | managed | 1.493x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.000735 ms | not_measured | managed | 0.629x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.03576 ms | not_measured | managed | 0.659x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 14.2762 ms | not_measured | managed | 1.195x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.000558 ms | not_measured | managed | 1.158x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.00677 ms | not_measured | managed | 4.199x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 3.1325 ms | not_measured | managed | 1.621x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.000622 ms | not_measured | managed | 1.068x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.01283 ms | not_measured | managed | 2.209x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 5.6725 ms | not_measured | managed | 1.442x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.000744 ms | not_measured | managed | 0.884x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.03511 ms | not_measured | managed | 0.872x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 14.4070 ms | not_measured | managed | 1.422x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.000422 ms | not_measured | managed | 1.453x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.00365 ms | not_measured | managed | 7.769x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 1.1780 ms | not_measured | managed | 3.210x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.000577 ms | not_measured | managed | 1.142x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.00674 ms | not_measured | managed | 4.215x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.2574 ms | not_measured | managed | 1.510x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.000613 ms | not_measured | managed | 1.062x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.01305 ms | not_measured | managed | 2.177x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 5.5922 ms | not_measured | managed | 1.555x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.000728 ms | not_measured | managed | 0.896x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.03789 ms | not_measured | managed | 0.851x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 14.6971 ms | not_measured | managed | 1.124x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.000479 ms | not_measured | managed | 1.282x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.00368 ms | not_measured | managed | 7.736x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.0906 ms | not_measured | managed | 3.448x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.00101 ms | not_measured | managed | 0.731x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.09048 ms | not_measured | managed | 3.014x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 24.6911 ms | not_measured | managed | 1.271x |
| `a + scalar (float16)` | float16 | 1,000 | 0.00493 ms | not_measured | managed | 0.662x |
| `a + scalar (float16)` | float16 | 100,000 | 0.4726 ms | not_measured | managed | 0.580x |
| `a + scalar (float16)` | float16 | 10,000,000 | 46.7371 ms | not_measured | managed | 0.645x |
| `a + scalar (float32)` | float32 | 1,000 | 0.000626 ms | not_measured | managed | 0.853x |
| `a + scalar (float32)` | float32 | 100,000 | 0.01313 ms | not_measured | managed | 0.444x |
| `a + scalar (float32)` | float32 | 10,000,000 | 4.9613 ms | not_measured | managed | 1.630x |
| `a + scalar (float64)` | float64 | 1,000 | 0.000725 ms | not_measured | managed | 0.806x |
| `a + scalar (float64)` | float64 | 100,000 | 0.02931 ms | not_measured | managed | 0.378x |
| `a + scalar (float64)` | float64 | 10,000,000 | 13.0143 ms | not_measured | managed | 1.247x |
| `a + scalar (int16)` | int16 | 1,000 | 0.000593 ms | not_measured | managed | 1.258x |
| `a + scalar (int16)` | int16 | 100,000 | 0.00622 ms | not_measured | managed | 3.813x |
| `a + scalar (int16)` | int16 | 10,000,000 | 3.1185 ms | not_measured | managed | 1.382x |
| `a + scalar (int32)` | int32 | 1,000 | 0.000613 ms | not_measured | managed | 1.258x |
| `a + scalar (int32)` | int32 | 100,000 | 0.01196 ms | not_measured | managed | 1.932x |
| `a + scalar (int32)` | int32 | 10,000,000 | 5.2137 ms | not_measured | managed | 1.423x |
| `a + scalar (int64)` | int64 | 1,000 | 0.000729 ms | not_measured | managed | 1.034x |
| `a + scalar (int64)` | int64 | 100,000 | 0.02636 ms | not_measured | managed | 0.876x |
| `a + scalar (int64)` | int64 | 10,000,000 | 13.1188 ms | not_measured | managed | 1.284x |
| `a + scalar (int8)` | int8 | 1,000 | 0.000491 ms | not_measured | managed | 1.485x |
| `a + scalar (int8)` | int8 | 100,000 | 0.00334 ms | not_measured | managed | 7.038x |
| `a + scalar (int8)` | int8 | 10,000,000 | 1.2424 ms | not_measured | managed | 2.617x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.000576 ms | not_measured | managed | 1.307x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.00626 ms | not_measured | managed | 3.784x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 3.0245 ms | not_measured | managed | 1.453x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.000636 ms | not_measured | managed | 1.190x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.01189 ms | not_measured | managed | 1.943x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 5.0492 ms | not_measured | managed | 2.023x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.000745 ms | not_measured | managed | 1.012x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.02684 ms | not_measured | managed | 0.862x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 13.2960 ms | not_measured | managed | 1.122x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.000506 ms | not_measured | managed | 1.437x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.00344 ms | not_measured | managed | 6.838x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 1.0849 ms | not_measured | managed | 2.996x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.00125 ms | not_measured | managed | 0.523x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.09763 ms | not_measured | managed | 2.854x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 27.9158 ms | not_measured | managed | 1.189x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.00490 ms | not_measured | managed | 0.666x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.4581 ms | not_measured | managed | 0.599x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 46.6976 ms | not_measured | managed | 0.648x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.000548 ms | not_measured | managed | 0.743x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.01359 ms | not_measured | managed | 0.499x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 5.6014 ms | not_measured | managed | 1.470x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.000742 ms | not_measured | managed | 0.632x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.03466 ms | not_measured | managed | 0.672x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 14.1978 ms | not_measured | managed | 1.269x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.000555 ms | not_measured | managed | 1.169x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.00696 ms | not_measured | managed | 4.085x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 3.1939 ms | not_measured | managed | 1.533x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.000597 ms | not_measured | managed | 1.089x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.01352 ms | not_measured | managed | 2.100x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 5.7108 ms | not_measured | managed | 1.419x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.000752 ms | not_measured | managed | 0.875x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.03746 ms | not_measured | managed | 0.803x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 14.0412 ms | not_measured | managed | 1.180x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.000479 ms | not_measured | managed | 1.267x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.00438 ms | not_measured | managed | 6.471x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 1.1331 ms | not_measured | managed | 3.295x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.000559 ms | not_measured | managed | 1.152x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.00694 ms | not_measured | managed | 4.092x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.2219 ms | not_measured | managed | 1.562x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.000620 ms | not_measured | managed | 1.052x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.01372 ms | not_measured | managed | 2.070x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 5.5646 ms | not_measured | managed | 2.273x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.000753 ms | not_measured | managed | 0.869x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.03484 ms | not_measured | managed | 0.917x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 14.1663 ms | not_measured | managed | 1.193x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.000479 ms | not_measured | managed | 1.271x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.00387 ms | not_measured | managed | 7.330x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.1626 ms | not_measured | managed | 3.207x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.00101 ms | not_measured | managed | 0.707x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.07744 ms | not_measured | managed | 3.568x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 24.6800 ms | not_measured | managed | 1.254x |
| `a - scalar (float16)` | float16 | 1,000 | 0.00483 ms | not_measured | managed | 0.677x |
| `a - scalar (float16)` | float16 | 100,000 | 0.4550 ms | not_measured | managed | 0.603x |
| `a - scalar (float16)` | float16 | 10,000,000 | 47.2417 ms | not_measured | managed | 0.639x |
| `a - scalar (float32)` | float32 | 1,000 | 0.000674 ms | not_measured | managed | 0.800x |
| `a - scalar (float32)` | float32 | 100,000 | 0.01201 ms | not_measured | managed | 0.488x |
| `a - scalar (float32)` | float32 | 10,000,000 | 4.7990 ms | not_measured | managed | 1.638x |
| `a - scalar (float64)` | float64 | 1,000 | 0.000734 ms | not_measured | managed | 0.812x |
| `a - scalar (float64)` | float64 | 100,000 | 0.02559 ms | not_measured | managed | 0.442x |
| `a - scalar (float64)` | float64 | 10,000,000 | 12.8320 ms | not_measured | managed | 1.251x |
| `a - scalar (int16)` | int16 | 1,000 | 0.000589 ms | not_measured | managed | 1.284x |
| `a - scalar (int16)` | int16 | 100,000 | 0.00616 ms | not_measured | managed | 3.756x |
| `a - scalar (int16)` | int16 | 10,000,000 | 2.7882 ms | not_measured | managed | 1.541x |
| `a - scalar (int32)` | int32 | 1,000 | 0.000619 ms | not_measured | managed | 1.234x |
| `a - scalar (int32)` | int32 | 100,000 | 0.01215 ms | not_measured | managed | 1.908x |
| `a - scalar (int32)` | int32 | 10,000,000 | 4.8515 ms | not_measured | managed | 1.663x |
| `a - scalar (int64)` | int64 | 1,000 | 0.000738 ms | not_measured | managed | 1.022x |
| `a - scalar (int64)` | int64 | 100,000 | 0.02710 ms | not_measured | managed | 0.880x |
| `a - scalar (int64)` | int64 | 10,000,000 | 13.0122 ms | not_measured | managed | 1.121x |
| `a - scalar (int8)` | int8 | 1,000 | 0.000456 ms | not_measured | managed | 1.583x |
| `a - scalar (int8)` | int8 | 100,000 | 0.00349 ms | not_measured | managed | 6.610x |
| `a - scalar (int8)` | int8 | 10,000,000 | 0.9593 ms | not_measured | managed | 3.381x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.000557 ms | not_measured | managed | 1.377x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.00619 ms | not_measured | managed | 3.727x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 2.7806 ms | not_measured | managed | 1.561x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.000629 ms | not_measured | managed | 1.203x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.01200 ms | not_measured | managed | 1.932x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 4.8685 ms | not_measured | managed | 2.355x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.000745 ms | not_measured | managed | 1.011x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.02666 ms | not_measured | managed | 0.893x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 12.7789 ms | not_measured | managed | 1.112x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.000490 ms | not_measured | managed | 1.467x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.00337 ms | not_measured | managed | 6.849x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 0.9574 ms | not_measured | managed | 3.383x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.000613 ms | not_measured | managed | 0.746x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.01396 ms | not_measured | managed | 0.847x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 5.6774 ms | not_measured | managed | 1.467x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.000852 ms | not_measured | managed | 0.857x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.04145 ms | not_measured | managed | 0.900x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 14.2124 ms | not_measured | managed | 1.151x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.00165 ms | not_measured | managed | 0.999x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.07784 ms | not_measured | managed | 1.049x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 13.2996 ms | not_measured | managed | 1.460x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.00169 ms | not_measured | managed | 0.944x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.07877 ms | not_measured | managed | 0.963x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 14.7332 ms | not_measured | managed | 1.605x |
| `a / scalar (float32)` | float32 | 1,000 | 0.000729 ms | not_measured | managed | 0.818x |
| `a / scalar (float32)` | float32 | 100,000 | 0.01384 ms | not_measured | managed | 0.865x |
| `a / scalar (float32)` | float32 | 10,000,000 | 4.8632 ms | not_measured | managed | 1.611x |
| `a / scalar (float64)` | float64 | 1,000 | 0.000981 ms | not_measured | managed | 0.878x |
| `a / scalar (float64)` | float64 | 100,000 | 0.03988 ms | not_measured | managed | 0.928x |
| `a / scalar (float64)` | float64 | 10,000,000 | 12.5495 ms | not_measured | managed | 1.282x |
| `a / scalar (int32)` | int32 | 1,000 | 0.00205 ms | not_measured | managed | 0.753x |
| `a / scalar (int32)` | int32 | 100,000 | 0.07939 ms | not_measured | managed | 0.764x |
| `a / scalar (int32)` | int32 | 10,000,000 | 13.3042 ms | not_measured | managed | 1.426x |
| `a / scalar (int64)` | int64 | 1,000 | 0.00209 ms | not_measured | managed | 0.727x |
| `a / scalar (int64)` | int64 | 100,000 | 0.08241 ms | not_measured | managed | 0.688x |
| `a / scalar (int64)` | int64 | 10,000,000 | 14.1257 ms | not_measured | managed | 1.329x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.00114 ms | not_measured | managed | 0.584x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.09590 ms | not_measured | managed | 2.909x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 27.7875 ms | not_measured | managed | 1.192x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.00484 ms | not_measured | managed | 0.645x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.4583 ms | not_measured | managed | 0.601x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 46.5594 ms | not_measured | managed | 0.648x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.000599 ms | not_measured | managed | 0.686x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.01331 ms | not_measured | managed | 0.502x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 5.6010 ms | not_measured | managed | 1.560x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.000738 ms | not_measured | managed | 0.644x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.03404 ms | not_measured | managed | 0.689x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 14.0718 ms | not_measured | managed | 1.180x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.000546 ms | not_measured | managed | 1.212x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.00676 ms | not_measured | managed | 4.218x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 3.1604 ms | not_measured | managed | 1.595x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.000642 ms | not_measured | managed | 1.042x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.01281 ms | not_measured | managed | 2.217x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 5.6475 ms | not_measured | managed | 1.447x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.000735 ms | not_measured | managed | 0.901x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.03589 ms | not_measured | managed | 0.839x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 14.0984 ms | not_measured | managed | 1.254x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.000460 ms | not_measured | managed | 1.365x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.00367 ms | not_measured | managed | 7.735x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 1.1450 ms | not_measured | managed | 3.264x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.000566 ms | not_measured | managed | 1.157x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.00670 ms | not_measured | managed | 4.241x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 3.1674 ms | not_measured | managed | 1.558x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.000605 ms | not_measured | managed | 1.091x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.01298 ms | not_measured | managed | 2.187x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 5.8062 ms | not_measured | managed | 1.826x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.000734 ms | not_measured | managed | 0.903x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.03764 ms | not_measured | managed | 0.853x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 14.2831 ms | not_measured | managed | 1.202x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.000459 ms | not_measured | managed | 1.377x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.00368 ms | not_measured | managed | 7.740x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 1.1691 ms | not_measured | managed | 3.202x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.000601 ms | not_measured | managed | 0.770x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.01741 ms | not_measured | managed | 0.680x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 6.3055 ms | not_measured | managed | 1.335x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.000873 ms | not_measured | managed | 0.843x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.04345 ms | not_measured | managed | 0.860x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 15.3354 ms | not_measured | managed | 1.052x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.00167 ms | not_measured | managed | 0.992x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.07985 ms | not_measured | managed | 1.023x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 13.7927 ms | not_measured | managed | 1.530x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.00170 ms | not_measured | managed | 0.927x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.08279 ms | not_measured | managed | 0.917x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 15.4863 ms | not_measured | managed | 1.476x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.00570 ms | not_measured | managed | 2.113x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 0.9708 ms | not_measured | managed | 1.527x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 99.3900 ms | not_measured | managed | 1.586x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.00512 ms | not_measured | managed | 2.032x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 0.9251 ms | not_measured | managed | 1.340x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 98.0363 ms | not_measured | managed | 1.418x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.00152 ms | not_measured | managed | 0.977x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.4661 ms | not_measured | managed | 0.768x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 47.9979 ms | not_measured | managed | 0.871x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.00232 ms | not_measured | managed | 0.938x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.5318 ms | not_measured | managed | 0.669x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 58.6033 ms | not_measured | managed | 0.799x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.00573 ms | not_measured | managed | 2.034x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 0.9684 ms | not_measured | managed | 1.519x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 100.2752 ms | not_measured | managed | 1.558x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.00480 ms | not_measured | managed | 2.111x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 0.8997 ms | not_measured | managed | 1.384x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 97.3743 ms | not_measured | managed | 1.422x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.00152 ms | not_measured | managed | 1.013x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.4133 ms | not_measured | managed | 0.871x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 44.0410 ms | not_measured | managed | 0.957x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.00230 ms | not_measured | managed | 1.343x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.4973 ms | not_measured | managed | 0.725x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 56.9796 ms | not_measured | managed | 0.832x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.00127 ms | not_measured | managed | 0.563x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.09849 ms | not_measured | managed | 2.804x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 29.4816 ms | not_measured | managed | 1.133x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.00494 ms | not_measured | managed | 0.632x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.4718 ms | not_measured | managed | 0.581x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 48.8491 ms | not_measured | managed | 0.616x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.000614 ms | not_measured | managed | 0.668x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.01467 ms | not_measured | managed | 0.457x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 5.7664 ms | not_measured | managed | 1.443x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.000857 ms | not_measured | managed | 0.558x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.03505 ms | not_measured | managed | 0.663x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 14.4808 ms | not_measured | managed | 1.231x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.000621 ms | not_measured | managed | 1.034x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.00703 ms | not_measured | managed | 3.891x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 3.2923 ms | not_measured | managed | 1.489x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.000668 ms | not_measured | managed | 0.972x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.01425 ms | not_measured | managed | 1.958x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 5.7974 ms | not_measured | managed | 1.435x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.000763 ms | not_measured | managed | 0.860x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.03634 ms | not_measured | managed | 0.819x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 14.8412 ms | not_measured | managed | 1.130x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.000452 ms | not_measured | managed | 1.365x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.00556 ms | not_measured | managed | 4.895x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 1.1133 ms | not_measured | managed | 3.286x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.000556 ms | not_measured | managed | 1.173x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.00716 ms | not_measured | managed | 3.798x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 3.1697 ms | not_measured | managed | 1.534x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.000677 ms | not_measured | managed | 0.962x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.01402 ms | not_measured | managed | 1.994x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 5.8092 ms | not_measured | managed | 2.083x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.000874 ms | not_measured | managed | 0.751x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.03954 ms | not_measured | managed | 0.795x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 14.6278 ms | not_measured | managed | 1.117x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.000492 ms | not_measured | managed | 1.256x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.00557 ms | not_measured | managed | 4.883x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 1.1750 ms | not_measured | managed | 3.122x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.00120 ms | not_measured | managed | 0.548x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.09978 ms | not_measured | managed | 2.747x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 28.2539 ms | not_measured | managed | 1.162x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.00493 ms | not_measured | managed | 0.660x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.4576 ms | not_measured | managed | 0.600x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 47.0094 ms | not_measured | managed | 0.642x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.000615 ms | not_measured | managed | 0.672x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.01358 ms | not_measured | managed | 0.503x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 5.4818 ms | not_measured | managed | 1.502x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.000771 ms | not_measured | managed | 0.617x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.03484 ms | not_measured | managed | 0.671x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 13.9861 ms | not_measured | managed | 1.212x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.000568 ms | not_measured | managed | 1.146x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.00693 ms | not_measured | managed | 4.116x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 3.1329 ms | not_measured | managed | 1.572x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.000616 ms | not_measured | managed | 1.091x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.01364 ms | not_measured | managed | 2.082x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 5.6696 ms | not_measured | managed | 1.497x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.000761 ms | not_measured | managed | 0.871x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.03442 ms | not_measured | managed | 0.873x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 14.1946 ms | not_measured | managed | 1.141x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.000460 ms | not_measured | managed | 1.363x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.00384 ms | not_measured | managed | 7.381x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 1.1504 ms | not_measured | managed | 3.244x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.000565 ms | not_measured | managed | 1.150x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.00686 ms | not_measured | managed | 4.144x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 3.0988 ms | not_measured | managed | 1.612x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.000612 ms | not_measured | managed | 1.078x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.01351 ms | not_measured | managed | 2.103x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 5.5580 ms | not_measured | managed | 2.308x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.000764 ms | not_measured | managed | 0.863x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.03368 ms | not_measured | managed | 0.949x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 14.4779 ms | not_measured | managed | 1.181x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.000471 ms | not_measured | managed | 1.335x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.00386 ms | not_measured | managed | 7.352x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 1.1395 ms | not_measured | managed | 3.271x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.000613 ms | not_measured | managed | 0.755x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.01401 ms | not_measured | managed | 0.845x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 5.8718 ms | not_measured | managed | 1.433x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.000913 ms | not_measured | managed | 0.821x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.04416 ms | not_measured | managed | 0.847x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 15.5305 ms | not_measured | managed | 1.085x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.00167 ms | not_measured | managed | 0.989x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.07815 ms | not_measured | managed | 1.045x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 13.7081 ms | not_measured | managed | 1.515x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.00169 ms | not_measured | managed | 0.934x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.08066 ms | not_measured | managed | 0.940x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 15.5244 ms | not_measured | managed | 1.485x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.00104 ms | not_measured | managed | 0.706x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.07434 ms | not_measured | managed | 3.643x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 24.6819 ms | not_measured | managed | 1.273x |
| `scalar - a (float16)` | float16 | 1,000 | 0.00476 ms | not_measured | managed | 0.696x |
| `scalar - a (float16)` | float16 | 100,000 | 0.4509 ms | not_measured | managed | 0.610x |
| `scalar - a (float16)` | float16 | 10,000,000 | 45.9596 ms | not_measured | managed | 0.656x |
| `scalar - a (float32)` | float32 | 1,000 | 0.000632 ms | not_measured | managed | 0.888x |
| `scalar - a (float32)` | float32 | 100,000 | 0.01215 ms | not_measured | managed | 0.481x |
| `scalar - a (float32)` | float32 | 10,000,000 | 4.8596 ms | not_measured | managed | 1.567x |
| `scalar - a (float64)` | float64 | 1,000 | 0.000731 ms | not_measured | managed | 0.850x |
| `scalar - a (float64)` | float64 | 100,000 | 0.02645 ms | not_measured | managed | 0.418x |
| `scalar - a (float64)` | float64 | 10,000,000 | 12.6387 ms | not_measured | managed | 1.236x |
| `scalar - a (int16)` | int16 | 1,000 | 0.000575 ms | not_measured | managed | 1.395x |
| `scalar - a (int16)` | int16 | 100,000 | 0.00616 ms | not_measured | managed | 3.858x |
| `scalar - a (int16)` | int16 | 10,000,000 | 2.7897 ms | not_measured | managed | 1.540x |
| `scalar - a (int32)` | int32 | 1,000 | 0.000618 ms | not_measured | managed | 1.259x |
| `scalar - a (int32)` | int32 | 100,000 | 0.01226 ms | not_measured | managed | 1.919x |
| `scalar - a (int32)` | int32 | 10,000,000 | 4.7965 ms | not_measured | managed | 1.608x |
| `scalar - a (int64)` | int64 | 1,000 | 0.000730 ms | not_measured | managed | 1.067x |
| `scalar - a (int64)` | int64 | 100,000 | 0.02649 ms | not_measured | managed | 0.888x |
| `scalar - a (int64)` | int64 | 10,000,000 | 12.6695 ms | not_measured | managed | 1.163x |
| `scalar - a (int8)` | int8 | 1,000 | 0.000442 ms | not_measured | managed | 1.670x |
| `scalar - a (int8)` | int8 | 100,000 | 0.00336 ms | not_measured | managed | 6.992x |
| `scalar - a (int8)` | int8 | 10,000,000 | 0.9320 ms | not_measured | managed | 3.409x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.000583 ms | not_measured | managed | 1.364x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.00619 ms | not_measured | managed | 3.831x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 2.7261 ms | not_measured | managed | 1.619x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.000613 ms | not_measured | managed | 1.266x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.01227 ms | not_measured | managed | 1.916x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 4.8316 ms | not_measured | managed | 2.265x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.000737 ms | not_measured | managed | 1.035x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.02728 ms | not_measured | managed | 0.862x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 12.9978 ms | not_measured | managed | 1.163x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.000483 ms | not_measured | managed | 1.536x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.00339 ms | not_measured | managed | 6.936x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 0.9617 ms | not_measured | managed | 3.296x |
| `scalar / a (float32)` | float32 | 1,000 | 0.000659 ms | not_measured | managed | 0.929x |
| `scalar / a (float32)` | float32 | 100,000 | 0.01364 ms | not_measured | managed | 0.879x |
| `scalar / a (float32)` | float32 | 10,000,000 | 4.8925 ms | not_measured | managed | 1.554x |
| `scalar / a (float64)` | float64 | 1,000 | 0.000919 ms | not_measured | managed | 0.964x |
| `scalar / a (float64)` | float64 | 100,000 | 0.04016 ms | not_measured | managed | 0.922x |
| `scalar / a (float64)` | float64 | 10,000,000 | 12.7721 ms | not_measured | managed | 1.232x |
| `scalar / a (int32)` | int32 | 1,000 | 0.00210 ms | not_measured | managed | 0.742x |
| `scalar / a (int32)` | int32 | 100,000 | 0.08095 ms | not_measured | managed | 0.751x |
| `scalar / a (int32)` | int32 | 10,000,000 | 13.2574 ms | not_measured | managed | 1.425x |
| `scalar / a (int64)` | int64 | 1,000 | 0.00213 ms | not_measured | managed | 0.721x |
| `scalar / a (int64)` | int64 | 100,000 | 0.08235 ms | not_measured | managed | 0.688x |
| `scalar / a (int64)` | int64 | 10,000,000 | 14.2684 ms | not_measured | managed | 1.296x |
| `a & b (bool)` | bool | 1,000 | 0.000422 ms | not_measured | managed | 0.789x |
| `a & b (bool)` | bool | 100,000 | 0.00452 ms | not_measured | managed | 0.501x |
| `a & b (bool)` | bool | 10,000,000 | 2.1651 ms | not_measured | managed | 0.850x |
| `a & b (int16)` | int16 | 1,000 | 0.000890 ms | not_measured | managed | 0.725x |
| `a & b (int16)` | int16 | 100,000 | 0.01182 ms | not_measured | managed | 2.393x |
| `a & b (int16)` | int16 | 10,000,000 | 4.8761 ms | not_measured | managed | 1.017x |
| `a & b (int32)` | int32 | 1,000 | 0.00105 ms | not_measured | managed | 0.617x |
| `a & b (int32)` | int32 | 100,000 | 0.02238 ms | not_measured | managed | 1.262x |
| `a & b (int32)` | int32 | 10,000,000 | 10.7437 ms | not_measured | managed | 0.755x |
| `a & b (int64)` | int64 | 1,000 | 0.00123 ms | not_measured | managed | 0.535x |
| `a & b (int64)` | int64 | 100,000 | 0.04923 ms | not_measured | managed | 0.634x |
| `a & b (int64)` | int64 | 10,000,000 | 14.5947 ms | not_measured | managed | 1.114x |
| `a & b (int8)` | int8 | 1,000 | 0.000512 ms | not_measured | managed | 1.191x |
| `a & b (int8)` | int8 | 100,000 | 0.00674 ms | not_measured | managed | 4.204x |
| `a & b (int8)` | int8 | 10,000,000 | 2.5616 ms | not_measured | managed | 1.450x |
| `a & b (uint16)` | uint16 | 1,000 | 0.000614 ms | not_measured | managed | 1.055x |
| `a & b (uint16)` | uint16 | 100,000 | 0.01270 ms | not_measured | managed | 2.227x |
| `a & b (uint16)` | uint16 | 10,000,000 | 3.6345 ms | not_measured | managed | 1.396x |
| `a & b (uint32)` | uint32 | 1,000 | 0.00115 ms | not_measured | managed | 0.573x |
| `a & b (uint32)` | uint32 | 100,000 | 0.01953 ms | not_measured | managed | 1.447x |
| `a & b (uint32)` | uint32 | 10,000,000 | 10.8582 ms | not_measured | managed | 0.740x |
| `a & b (uint64)` | uint64 | 1,000 | 0.000804 ms | not_measured | managed | 0.815x |
| `a & b (uint64)` | uint64 | 100,000 | 0.04996 ms | not_measured | managed | 0.618x |
| `a & b (uint64)` | uint64 | 10,000,000 | 22.8756 ms | not_measured | managed | 0.699x |
| `a & b (uint8)` | uint8 | 1,000 | 0.000534 ms | not_measured | managed | 1.150x |
| `a & b (uint8)` | uint8 | 100,000 | 0.00659 ms | not_measured | managed | 4.297x |
| `a & b (uint8)` | uint8 | 10,000,000 | 1.8329 ms | not_measured | managed | 2.032x |
| `a ^ b (bool)` | bool | 1,000 | 0.000614 ms | not_measured | managed | 0.542x |
| `a ^ b (bool)` | bool | 100,000 | 0.00595 ms | not_measured | managed | 0.386x |
| `a ^ b (bool)` | bool | 10,000,000 | 2.7550 ms | not_measured | managed | 0.702x |
| `a ^ b (int16)` | int16 | 1,000 | 0.000923 ms | not_measured | managed | 0.688x |
| `a ^ b (int16)` | int16 | 100,000 | 0.01221 ms | not_measured | managed | 2.321x |
| `a ^ b (int16)` | int16 | 10,000,000 | 3.8699 ms | not_measured | managed | 1.273x |
| `a ^ b (int32)` | int32 | 1,000 | 0.00110 ms | not_measured | managed | 0.590x |
| `a ^ b (int32)` | int32 | 100,000 | 0.02458 ms | not_measured | managed | 1.155x |
| `a ^ b (int32)` | int32 | 10,000,000 | 4.3381 ms | not_measured | managed | 1.869x |
| `a ^ b (int64)` | int64 | 1,000 | 0.00133 ms | not_measured | managed | 0.509x |
| `a ^ b (int64)` | int64 | 100,000 | 0.05552 ms | not_measured | managed | 0.550x |
| `a ^ b (int64)` | int64 | 10,000,000 | 15.1424 ms | not_measured | managed | 1.096x |
| `a ^ b (int8)` | int8 | 1,000 | 0.000571 ms | not_measured | managed | 1.074x |
| `a ^ b (int8)` | int8 | 100,000 | 0.00632 ms | not_measured | managed | 4.482x |
| `a ^ b (int8)` | int8 | 10,000,000 | 2.1385 ms | not_measured | managed | 1.751x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.000630 ms | not_measured | managed | 1.029x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.01218 ms | not_measured | managed | 2.331x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 5.5297 ms | not_measured | managed | 0.888x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.00112 ms | not_measured | managed | 0.596x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.02280 ms | not_measured | managed | 1.246x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 9.8019 ms | not_measured | managed | 0.821x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.000841 ms | not_measured | managed | 0.784x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.05308 ms | not_measured | managed | 0.578x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 25.5498 ms | not_measured | managed | 0.630x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.000500 ms | not_measured | managed | 1.226x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.00509 ms | not_measured | managed | 5.583x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 2.3917 ms | not_measured | managed | 1.570x |
| `a | b (bool)` | bool | 1,000 | 0.000456 ms | not_measured | managed | 0.726x |
| `a | b (bool)` | bool | 100,000 | 0.00517 ms | not_measured | managed | 0.431x |
| `a | b (bool)` | bool | 10,000,000 | 2.6292 ms | not_measured | managed | 0.681x |
| `a | b (int16)` | int16 | 1,000 | 0.000935 ms | not_measured | managed | 0.688x |
| `a | b (int16)` | int16 | 100,000 | 0.00958 ms | not_measured | managed | 2.957x |
| `a | b (int16)` | int16 | 10,000,000 | 5.9229 ms | not_measured | managed | 0.833x |
| `a | b (int32)` | int32 | 1,000 | 0.00108 ms | not_measured | managed | 0.591x |
| `a | b (int32)` | int32 | 100,000 | 0.02247 ms | not_measured | managed | 1.263x |
| `a | b (int32)` | int32 | 10,000,000 | 7.6341 ms | not_measured | managed | 1.074x |
| `a | b (int64)` | int64 | 1,000 | 0.00122 ms | not_measured | managed | 0.531x |
| `a | b (int64)` | int64 | 100,000 | 0.04644 ms | not_measured | managed | 0.657x |
| `a | b (int64)` | int64 | 10,000,000 | 14.5606 ms | not_measured | managed | 1.143x |
| `a | b (int8)` | int8 | 1,000 | 0.000436 ms | not_measured | managed | 1.404x |
| `a | b (int8)` | int8 | 100,000 | 0.00516 ms | not_measured | managed | 5.486x |
| `a | b (int8)` | int8 | 10,000,000 | 1.8281 ms | not_measured | managed | 2.069x |
| `a | b (uint16)` | uint16 | 1,000 | 0.000884 ms | not_measured | managed | 0.732x |
| `a | b (uint16)` | uint16 | 100,000 | 0.01337 ms | not_measured | managed | 2.121x |
| `a | b (uint16)` | uint16 | 10,000,000 | 4.7585 ms | not_measured | managed | 1.037x |
| `a | b (uint32)` | uint32 | 1,000 | 0.00111 ms | not_measured | managed | 0.592x |
| `a | b (uint32)` | uint32 | 100,000 | 0.02411 ms | not_measured | managed | 1.178x |
| `a | b (uint32)` | uint32 | 10,000,000 | 11.1564 ms | not_measured | managed | 0.757x |
| `a | b (uint64)` | uint64 | 1,000 | 0.000904 ms | not_measured | managed | 0.726x |
| `a | b (uint64)` | uint64 | 100,000 | 0.04714 ms | not_measured | managed | 0.648x |
| `a | b (uint64)` | uint64 | 10,000,000 | 24.8275 ms | not_measured | managed | 0.647x |
| `a | b (uint8)` | uint8 | 1,000 | 0.000522 ms | not_measured | managed | 1.172x |
| `a | b (uint8)` | uint8 | 100,000 | 0.00594 ms | not_measured | managed | 4.761x |
| `a | b (uint8)` | uint8 | 10,000,000 | 2.5071 ms | not_measured | managed | 1.513x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.000500 ms | not_measured | managed | 0.702x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.00562 ms | not_measured | managed | 0.405x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 2.4072 ms | not_measured | managed | 0.721x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.000864 ms | not_measured | managed | 0.758x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.01301 ms | not_measured | managed | 2.173x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 4.7359 ms | not_measured | managed | 1.042x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.00108 ms | not_measured | managed | 0.606x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.02368 ms | not_measured | managed | 1.193x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 6.4988 ms | not_measured | managed | 1.231x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.00151 ms | not_measured | managed | 0.443x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.05275 ms | not_measured | managed | 0.580x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 24.3263 ms | not_measured | managed | 0.655x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.000438 ms | not_measured | managed | 1.438x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.00695 ms | not_measured | managed | 4.089x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 1.9424 ms | not_measured | managed | 1.927x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.000821 ms | not_measured | managed | 0.811x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.01349 ms | not_measured | managed | 2.094x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 4.2329 ms | not_measured | managed | 1.171x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.00105 ms | not_measured | managed | 0.629x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.02498 ms | not_measured | managed | 1.134x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 8.2270 ms | not_measured | managed | 0.991x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.00133 ms | not_measured | managed | 0.503x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.04363 ms | not_measured | managed | 0.703x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 25.0222 ms | not_measured | managed | 0.627x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.000624 ms | not_measured | managed | 1.014x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.00631 ms | not_measured | managed | 4.502x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 2.1210 ms | not_measured | managed | 1.753x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.000425 ms | not_measured | managed | 0.769x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.00472 ms | not_measured | managed | 0.390x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 1.5672 ms | not_measured | managed | 1.012x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.000777 ms | not_measured | managed | 0.771x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.01170 ms | not_measured | managed | 2.191x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 3.6757 ms | not_measured | managed | 1.248x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.00113 ms | not_measured | managed | 0.547x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.02160 ms | not_measured | managed | 1.188x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 8.5147 ms | not_measured | managed | 0.861x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.00149 ms | not_measured | managed | 0.410x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.03918 ms | not_measured | managed | 0.656x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 18.7865 ms | not_measured | managed | 0.784x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.000521 ms | not_measured | managed | 1.107x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.00351 ms | not_measured | managed | 7.304x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 1.3017 ms | not_measured | managed | 2.660x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.000808 ms | not_measured | managed | 0.756x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.01167 ms | not_measured | managed | 2.195x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 2.5103 ms | not_measured | managed | 1.802x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.000764 ms | not_measured | managed | 0.800x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.02103 ms | not_measured | managed | 1.220x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 7.5749 ms | not_measured | managed | 0.987x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.00111 ms | not_measured | managed | 0.551x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.04042 ms | not_measured | managed | 0.636x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 16.2743 ms | not_measured | managed | 0.893x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.000636 ms | not_measured | managed | 0.914x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.00595 ms | not_measured | managed | 4.305x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 1.2980 ms | not_measured | managed | 2.634x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.000591 ms | not_measured | managed | 0.589x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.00675 ms | not_measured | managed | 0.331x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 2.3334 ms | not_measured | managed | 0.752x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.000783 ms | not_measured | managed | 0.838x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.01264 ms | not_measured | managed | 2.246x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 2.9673 ms | not_measured | managed | 1.685x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.00117 ms | not_measured | managed | 0.562x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.02354 ms | not_measured | managed | 1.206x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 10.9913 ms | not_measured | managed | 0.742x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.00132 ms | not_measured | managed | 0.498x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.05011 ms | not_measured | managed | 0.612x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 21.4143 ms | not_measured | managed | 0.748x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.000420 ms | not_measured | managed | 1.495x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.00666 ms | not_measured | managed | 4.247x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 1.7513 ms | not_measured | managed | 2.169x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.000872 ms | not_measured | managed | 0.752x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.01218 ms | not_measured | managed | 2.330x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 3.2519 ms | not_measured | managed | 1.535x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.00102 ms | not_measured | managed | 0.646x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.02394 ms | not_measured | managed | 1.185x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 7.1598 ms | not_measured | managed | 1.124x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.000835 ms | not_measured | managed | 0.801x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.04858 ms | not_measured | managed | 0.631x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 24.6962 ms | not_measured | managed | 0.653x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.000681 ms | not_measured | managed | 0.928x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.00421 ms | not_measured | managed | 6.724x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 2.0783 ms | not_measured | managed | 1.817x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.000469 ms | not_measured | managed | 0.748x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.00612 ms | not_measured | managed | 0.371x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 2.5400 ms | not_measured | managed | 0.674x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.000765 ms | not_measured | managed | 0.856x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.01300 ms | not_measured | managed | 2.181x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 5.6777 ms | not_measured | managed | 0.873x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.00118 ms | not_measured | managed | 0.567x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.02218 ms | not_measured | managed | 1.280x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 10.3071 ms | not_measured | managed | 0.791x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.00123 ms | not_measured | managed | 0.541x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.05056 ms | not_measured | managed | 0.605x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 24.5408 ms | not_measured | managed | 0.661x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.000507 ms | not_measured | managed | 1.243x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.00419 ms | not_measured | managed | 6.773x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 1.9960 ms | not_measured | managed | 1.878x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.000900 ms | not_measured | managed | 0.732x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.00954 ms | not_measured | managed | 2.976x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 4.3571 ms | not_measured | managed | 1.119x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.000721 ms | not_measured | managed | 0.913x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.02389 ms | not_measured | managed | 1.188x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 11.0269 ms | not_measured | managed | 0.713x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.00108 ms | not_measured | managed | 0.617x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.04743 ms | not_measured | managed | 0.646x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 25.2699 ms | not_measured | managed | 0.646x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.000710 ms | not_measured | managed | 0.889x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.00661 ms | not_measured | managed | 4.284x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 1.8169 ms | not_measured | managed | 2.050x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.000633 ms | not_measured | managed | 0.515x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.00351 ms | not_measured | managed | 0.523x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 1.5680 ms | not_measured | managed | 0.989x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.000836 ms | not_measured | managed | 0.720x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.01220 ms | not_measured | managed | 2.100x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 4.4579 ms | not_measured | managed | 1.024x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.00104 ms | not_measured | managed | 0.579x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.02189 ms | not_measured | managed | 1.172x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 8.2711 ms | not_measured | managed | 0.901x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.00124 ms | not_measured | managed | 0.501x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.04297 ms | not_measured | managed | 0.598x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 13.8893 ms | not_measured | managed | 1.019x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.000522 ms | not_measured | managed | 1.105x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.00456 ms | not_measured | managed | 5.616x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 0.8781 ms | not_measured | managed | 3.939x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.000731 ms | not_measured | managed | 0.824x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.01153 ms | not_measured | managed | 2.223x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 4.1211 ms | not_measured | managed | 1.107x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.000853 ms | not_measured | managed | 0.741x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.02055 ms | not_measured | managed | 1.248x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 6.2666 ms | not_measured | managed | 1.180x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.00118 ms | not_measured | managed | 0.519x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.04399 ms | not_measured | managed | 0.584x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 19.2295 ms | not_measured | managed | 0.746x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.000618 ms | not_measured | managed | 0.942x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.00464 ms | not_measured | managed | 5.523x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 1.0347 ms | not_measured | managed | 3.303x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.00221 ms | not_measured | managed | 0.572x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.06599 ms | not_measured | managed | 0.643x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 15.3128 ms | not_measured | managed | 0.943x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.00146 ms | not_measured | managed | 0.624x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.01254 ms | not_measured | managed | 2.238x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.0343 ms | not_measured | managed | 1.164x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.00202 ms | not_measured | managed | 0.413x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.02285 ms | not_measured | managed | 0.831x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 7.5336 ms | not_measured | managed | 0.981x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.00249 ms | not_measured | managed | 0.321x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.04509 ms | not_measured | managed | 0.422x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 18.5679 ms | not_measured | managed | 0.763x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.000688 ms | not_measured | managed | 1.270x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.00696 ms | not_measured | managed | 4.035x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.0334 ms | not_measured | managed | 3.545x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.00151 ms | not_measured | managed | 0.597x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.01256 ms | not_measured | managed | 2.236x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.1188 ms | not_measured | managed | 1.139x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00181 ms | not_measured | managed | 0.468x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.02270 ms | not_measured | managed | 0.837x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.6626 ms | not_measured | managed | 0.846x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00211 ms | not_measured | managed | 0.389x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.04431 ms | not_measured | managed | 0.430x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 18.0362 ms | not_measured | managed | 0.791x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.00101 ms | not_measured | managed | 0.861x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.00679 ms | not_measured | managed | 4.135x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.4967 ms | not_measured | managed | 2.415x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.00259 ms | not_measured | managed | 0.532x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.00271 ms | not_measured | managed | 19.348x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.00138 ms | not_measured | managed | 0.723x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.01237 ms | not_measured | managed | 3.006x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 3.2365 ms | not_measured | managed | 1.723x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.00190 ms | not_measured | managed | 0.491x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.02205 ms | not_measured | managed | 1.273x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 6.4061 ms | not_measured | managed | 1.149x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.00230 ms | not_measured | managed | 0.387x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.04818 ms | not_measured | managed | 0.582x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 19.9223 ms | not_measured | managed | 0.744x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.000858 ms | not_measured | managed | 1.125x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.00799 ms | not_measured | managed | 4.654x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.4184 ms | not_measured | managed | 3.227x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.00132 ms | not_measured | managed | 0.687x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.01313 ms | not_measured | managed | 2.136x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 3.6223 ms | not_measured | managed | 1.283x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00184 ms | not_measured | managed | 0.463x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.01549 ms | not_measured | managed | 1.226x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 4.5022 ms | not_measured | managed | 1.672x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00154 ms | not_measured | managed | 0.531x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.04956 ms | not_measured | managed | 0.385x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 15.9370 ms | not_measured | managed | 0.923x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.000931 ms | not_measured | managed | 0.932x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.00735 ms | not_measured | managed | 3.816x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.3032 ms | not_measured | managed | 2.775x |
| `matrix * 2.0` | float64 | 1,000 | 0.00196 ms | not_measured | managed | 0.311x |
| `matrix * 2.0` | float64 | 100,000 | 0.04143 ms | not_measured | managed | 0.313x |
| `matrix * 2.0` | float64 | 10,000,000 | 19.6703 ms | not_measured | managed | 0.775x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.00143 ms | not_measured | managed | 0.736x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.04983 ms | not_measured | managed | 0.561x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 19.3589 ms | not_measured | managed | 0.830x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.00167 ms | not_measured | managed | 0.599x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.05064 ms | not_measured | managed | 0.513x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 19.3958 ms | not_measured | managed | 0.829x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.00106 ms | not_measured | managed | 0.963x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.05402 ms | not_measured | managed | 0.518x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 17.8606 ms | not_measured | managed | 0.885x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.00137 ms | not_measured | managed | 0.725x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.05467 ms | not_measured | managed | 0.475x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 13.3433 ms | not_measured | managed | 1.181x |
| `matrix + scalar` | float64 | 1,000 | 0.000769 ms | not_measured | managed | 0.607x |
| `matrix + scalar` | float64 | 100,000 | 0.04703 ms | not_measured | managed | 0.271x |
| `matrix + scalar` | float64 | 10,000,000 | 18.7302 ms | not_measured | managed | 0.826x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.000352 ms | not_measured | managed | 4.989x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.000368 ms | not_measured | managed | 4.753x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.000353 ms | not_measured | managed | 4.960x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.000408 ms | not_measured | managed | 4.287x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.000299 ms | not_measured | managed | 5.829x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.000267 ms | not_measured | managed | 6.543x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.000987 ms | not_measured | managed | 0.899x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.03908 ms | not_measured | managed | 0.670x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 19.3504 ms | not_measured | managed | 0.768x |
| `a != b (float32)` | float32 | 1,000 | 0.000607 ms | not_measured | managed | 0.621x |
| `a != b (float32)` | float32 | 100,000 | 0.00862 ms | not_measured | managed | 0.515x |
| `a != b (float32)` | float32 | 10,000,000 | 2.7540 ms | not_measured | managed | 1.286x |
| `a != b (float64)` | float64 | 1,000 | 0.000736 ms | not_measured | managed | 0.554x |
| `a != b (float64)` | float64 | 100,000 | 0.01844 ms | not_measured | managed | 0.524x |
| `a != b (float64)` | float64 | 10,000,000 | 7.4673 ms | not_measured | managed | 0.819x |
| `a != b (int32)` | int32 | 1,000 | 0.000485 ms | not_measured | managed | 0.806x |
| `a != b (int32)` | int32 | 100,000 | 0.00767 ms | not_measured | managed | 0.888x |
| `a != b (int32)` | int32 | 10,000,000 | 2.8224 ms | not_measured | managed | 1.427x |
| `a != b (int64)` | int64 | 1,000 | 0.000658 ms | not_measured | managed | 0.646x |
| `a != b (int64)` | int64 | 100,000 | 0.01640 ms | not_measured | managed | 0.738x |
| `a != b (int64)` | int64 | 10,000,000 | 5.8581 ms | not_measured | managed | 1.144x |
| `a < b (float32)` | float32 | 1,000 | 0.000518 ms | not_measured | managed | 0.720x |
| `a < b (float32)` | float32 | 100,000 | 0.00754 ms | not_measured | managed | 0.592x |
| `a < b (float32)` | float32 | 10,000,000 | 2.7250 ms | not_measured | managed | 1.329x |
| `a < b (float64)` | float64 | 1,000 | 0.000644 ms | not_measured | managed | 0.621x |
| `a < b (float64)` | float64 | 100,000 | 0.01927 ms | not_measured | managed | 0.502x |
| `a < b (float64)` | float64 | 10,000,000 | 14.9115 ms | not_measured | managed | 0.423x |
| `a < b (int32)` | int32 | 1,000 | 0.000586 ms | not_measured | managed | 0.648x |
| `a < b (int32)` | int32 | 100,000 | 0.00632 ms | not_measured | managed | 1.079x |
| `a < b (int32)` | int32 | 10,000,000 | 2.7754 ms | not_measured | managed | 1.361x |
| `a < b (int64)` | int64 | 1,000 | 0.000668 ms | not_measured | managed | 0.731x |
| `a < b (int64)` | int64 | 100,000 | 0.02025 ms | not_measured | managed | 0.868x |
| `a < b (int64)` | int64 | 10,000,000 | 16.6090 ms | not_measured | managed | 0.396x |
| `a <= b (float32)` | float32 | 1,000 | 0.000439 ms | not_measured | managed | 0.863x |
| `a <= b (float32)` | float32 | 100,000 | 0.01195 ms | not_measured | managed | 0.374x |
| `a <= b (float32)` | float32 | 10,000,000 | 2.7535 ms | not_measured | managed | 1.302x |
| `a <= b (float64)` | float64 | 1,000 | 0.000660 ms | not_measured | managed | 0.620x |
| `a <= b (float64)` | float64 | 100,000 | 0.01255 ms | not_measured | managed | 0.772x |
| `a <= b (float64)` | float64 | 10,000,000 | 5.6134 ms | not_measured | managed | 1.100x |
| `a <= b (int32)` | int32 | 1,000 | 0.000443 ms | not_measured | managed | 0.867x |
| `a <= b (int32)` | int32 | 100,000 | 0.00682 ms | not_measured | managed | 1.002x |
| `a <= b (int32)` | int32 | 10,000,000 | 6.3136 ms | not_measured | managed | 0.633x |
| `a <= b (int64)` | int64 | 1,000 | 0.000640 ms | not_measured | managed | 0.783x |
| `a <= b (int64)` | int64 | 100,000 | 0.01877 ms | not_measured | managed | 0.935x |
| `a <= b (int64)` | int64 | 10,000,000 | 5.6590 ms | not_measured | managed | 1.196x |
| `a == b (float32)` | float32 | 1,000 | 0.000508 ms | not_measured | managed | 0.748x |
| `a == b (float32)` | float32 | 100,000 | 0.01167 ms | not_measured | managed | 0.381x |
| `a == b (float32)` | float32 | 10,000,000 | 2.7447 ms | not_measured | managed | 1.296x |
| `a == b (float64)` | float64 | 1,000 | 0.000420 ms | not_measured | managed | 0.940x |
| `a == b (float64)` | float64 | 100,000 | 0.01676 ms | not_measured | managed | 0.578x |
| `a == b (float64)` | float64 | 10,000,000 | 5.3607 ms | not_measured | managed | 1.147x |
| `a == b (int32)` | int32 | 1,000 | 0.000412 ms | not_measured | managed | 0.908x |
| `a == b (int32)` | int32 | 100,000 | 0.00629 ms | not_measured | managed | 1.082x |
| `a == b (int32)` | int32 | 10,000,000 | 3.9698 ms | not_measured | managed | 0.988x |
| `a == b (int64)` | int64 | 1,000 | 0.000625 ms | not_measured | managed | 0.653x |
| `a == b (int64)` | int64 | 100,000 | 0.02062 ms | not_measured | managed | 0.586x |
| `a == b (int64)` | int64 | 10,000,000 | 5.2468 ms | not_measured | managed | 1.260x |
| `a > b (float32)` | float32 | 1,000 | 0.000432 ms | not_measured | managed | 0.856x |
| `a > b (float32)` | float32 | 100,000 | 0.00784 ms | not_measured | managed | 0.562x |
| `a > b (float32)` | float32 | 10,000,000 | 2.8153 ms | not_measured | managed | 1.315x |
| `a > b (float64)` | float64 | 1,000 | 0.000486 ms | not_measured | managed | 0.821x |
| `a > b (float64)` | float64 | 100,000 | 0.01419 ms | not_measured | managed | 0.681x |
| `a > b (float64)` | float64 | 10,000,000 | 5.1016 ms | not_measured | managed | 1.218x |
| `a > b (int32)` | int32 | 1,000 | 0.000631 ms | not_measured | managed | 0.605x |
| `a > b (int32)` | int32 | 100,000 | 0.00644 ms | not_measured | managed | 1.063x |
| `a > b (int32)` | int32 | 10,000,000 | 2.7460 ms | not_measured | managed | 1.397x |
| `a > b (int64)` | int64 | 1,000 | 0.000529 ms | not_measured | managed | 0.921x |
| `a > b (int64)` | int64 | 100,000 | 0.01589 ms | not_measured | managed | 1.105x |
| `a > b (int64)` | int64 | 10,000,000 | 5.1660 ms | not_measured | managed | 1.253x |
| `a >= b (float32)` | float32 | 1,000 | 0.000601 ms | not_measured | managed | 0.629x |
| `a >= b (float32)` | float32 | 100,000 | 0.01190 ms | not_measured | managed | 0.371x |
| `a >= b (float32)` | float32 | 10,000,000 | 3.1246 ms | not_measured | managed | 1.142x |
| `a >= b (float64)` | float64 | 1,000 | 0.000439 ms | not_measured | managed | 0.907x |
| `a >= b (float64)` | float64 | 100,000 | 0.01870 ms | not_measured | managed | 0.517x |
| `a >= b (float64)` | float64 | 10,000,000 | 5.2353 ms | not_measured | managed | 1.198x |
| `a >= b (int32)` | int32 | 1,000 | 0.000463 ms | not_measured | managed | 0.834x |
| `a >= b (int32)` | int32 | 100,000 | 0.01213 ms | not_measured | managed | 0.560x |
| `a >= b (int32)` | int32 | 10,000,000 | 2.8137 ms | not_measured | managed | 1.415x |
| `a >= b (int64)` | int64 | 1,000 | 0.000786 ms | not_measured | managed | 0.639x |
| `a >= b (int64)` | int64 | 100,000 | 0.02201 ms | not_measured | managed | 0.798x |
| `a >= b (int64)` | int64 | 10,000,000 | 5.5860 ms | not_measured | managed | 1.222x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.000491 ms | not_measured | managed | 0.802x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.00839 ms | not_measured | managed | 0.532x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 8.2526 ms | not_measured | managed | 0.437x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.000705 ms | not_measured | managed | 0.582x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.01875 ms | not_measured | managed | 0.517x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 7.3875 ms | not_measured | managed | 0.848x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.000619 ms | not_measured | managed | 0.628x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.00641 ms | not_measured | managed | 1.064x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 3.2723 ms | not_measured | managed | 1.165x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.000492 ms | not_measured | managed | 0.866x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.01574 ms | not_measured | managed | 0.770x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 9.3009 ms | not_measured | managed | 0.706x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.000629 ms | not_measured | managed | 0.617x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.01292 ms | not_measured | managed | 0.340x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 3.8756 ms | not_measured | managed | 0.923x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.000407 ms | not_measured | managed | 1.022x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.01371 ms | not_measured | managed | 0.706x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 5.3861 ms | not_measured | managed | 1.151x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.000479 ms | not_measured | managed | 0.816x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.00632 ms | not_measured | managed | 1.084x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 2.7316 ms | not_measured | managed | 1.428x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.000530 ms | not_measured | managed | 0.951x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.01604 ms | not_measured | managed | 1.096x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 16.3117 ms | not_measured | managed | 0.403x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.000556 ms | not_measured | managed | 0.709x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.00789 ms | not_measured | managed | 0.559x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 5.3568 ms | not_measured | managed | 0.668x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.000527 ms | not_measured | managed | 0.789x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.01367 ms | not_measured | managed | 0.708x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 6.1436 ms | not_measured | managed | 1.002x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.000483 ms | not_measured | managed | 0.834x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.00728 ms | not_measured | managed | 0.932x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 2.8493 ms | not_measured | managed | 1.386x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.000673 ms | not_measured | managed | 0.767x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.01986 ms | not_measured | managed | 0.883x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 5.5781 ms | not_measured | managed | 1.223x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.000485 ms | not_measured | managed | 0.804x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.00783 ms | not_measured | managed | 0.569x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 2.9927 ms | not_measured | managed | 1.181x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.000511 ms | not_measured | managed | 0.814x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.01358 ms | not_measured | managed | 0.714x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 5.2626 ms | not_measured | managed | 1.179x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.000643 ms | not_measured | managed | 0.607x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.00766 ms | not_measured | managed | 0.891x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 2.7920 ms | not_measured | managed | 1.384x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.000424 ms | not_measured | managed | 1.191x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.01621 ms | not_measured | managed | 1.084x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 5.2385 ms | not_measured | managed | 1.263x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.000473 ms | not_measured | managed | 0.844x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.01279 ms | not_measured | managed | 0.348x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 4.2631 ms | not_measured | managed | 0.846x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.000420 ms | not_measured | managed | 0.993x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.01360 ms | not_measured | managed | 0.713x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 7.1151 ms | not_measured | managed | 0.871x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.000402 ms | not_measured | managed | 1.012x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.00677 ms | not_measured | managed | 1.009x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 2.8672 ms | not_measured | managed | 1.395x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.000431 ms | not_measured | managed | 1.197x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.01935 ms | not_measured | managed | 0.907x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 5.5299 ms | not_measured | managed | 1.246x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.000626 ms | not_measured | managed | 0.639x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.00853 ms | not_measured | managed | 0.523x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 8.5519 ms | not_measured | managed | 0.415x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.000690 ms | not_measured | managed | 0.613x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.01541 ms | not_measured | managed | 0.629x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 5.8048 ms | not_measured | managed | 1.083x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.000558 ms | not_measured | managed | 0.728x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.00716 ms | not_measured | managed | 0.950x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 6.8414 ms | not_measured | managed | 0.579x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.000523 ms | not_measured | managed | 0.843x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.01992 ms | not_measured | managed | 0.609x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 5.2982 ms | not_measured | managed | 1.266x |
| `a.copy() (float32)` | float32 | 1,000 | 0.000484 ms | not_measured | managed | 0.539x |
| `a.copy() (float32)` | float32 | 100,000 | 0.00840 ms | not_measured | managed | 0.681x |
| `a.copy() (float32)` | float32 | 10,000,000 | 4.8552 ms | not_measured | managed | 1.277x |
| `a.copy() (float64)` | float64 | 1,000 | 0.000534 ms | not_measured | managed | 0.586x |
| `a.copy() (float64)` | float64 | 100,000 | 0.02128 ms | not_measured | managed | 0.522x |
| `a.copy() (float64)` | float64 | 10,000,000 | 13.2826 ms | not_measured | managed | 0.962x |
| `a.copy() (int32)` | int32 | 1,000 | 0.000426 ms | not_measured | managed | 0.617x |
| `a.copy() (int32)` | int32 | 100,000 | 0.01011 ms | not_measured | managed | 0.563x |
| `a.copy() (int32)` | int32 | 10,000,000 | 4.7949 ms | not_measured | managed | 1.314x |
| `a.copy() (int64)` | int64 | 1,000 | 0.000592 ms | not_measured | managed | 0.500x |
| `a.copy() (int64)` | int64 | 100,000 | 0.01098 ms | not_measured | managed | 1.010x |
| `a.copy() (int64)` | int64 | 10,000,000 | 12.7545 ms | not_measured | managed | 0.979x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.000820 ms | not_measured | managed | 0.873x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.04852 ms | not_measured | managed | 0.796x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 5.6669 ms | not_measured | managed | 1.368x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.000503 ms | not_measured | managed | 1.225x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.02734 ms | not_measured | managed | 1.001x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 15.0500 ms | not_measured | managed | 0.846x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.000607 ms | not_measured | managed | 0.995x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.02269 ms | not_measured | managed | 1.271x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 2.7555 ms | not_measured | managed | 2.441x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.000698 ms | not_measured | managed | 0.731x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.03846 ms | not_measured | managed | 0.485x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 15.1712 ms | not_measured | managed | 0.817x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.000800 ms | not_measured | managed | 0.383x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.03601 ms | not_measured | managed | 0.321x |
| `np.array(a) (float64)` | float64 | 10,000,000 | 0.4253 ms | not_measured | managed | 2.269x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.000004 ms | not_measured | managed | 15.250x |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.000006 ms | not_measured | managed | 10.000x |
| `np.asanyarray(a) (float64)` | float64 | 10,000,000 | 0.000005 ms | not_measured | managed | 12.400x |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 59.000x |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.000086 ms | not_measured | managed | 15.744x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.00984 ms | not_measured | managed | 1.176x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 10,000,000 | 0.1846 ms | not_measured | managed | 0.699x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.000764 ms | not_measured | managed | 0.442x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.02760 ms | not_measured | managed | 0.345x |
| `np.ascontiguousarray(a) (float64)` | float64 | 10,000,000 | 0.7308 ms | not_measured | managed | 0.887x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.00144 ms | not_measured | managed | 0.490x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.1206 ms | not_measured | managed | 0.210x |
| `np.asfortranarray(a) (float64)` | float64 | 10,000,000 | 2.6303 ms | not_measured | managed | 0.562x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.000236 ms | not_measured | managed | 7.962x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.000401 ms | not_measured | managed | 4.731x |
| `np.asmatrix(a) (float64)` | float64 | 10,000,000 | 0.000364 ms | not_measured | managed | 5.176x |
| `np.copy (float32)` | float32 | 1,000 | 0.000558 ms | not_measured | managed | 0.830x |
| `np.copy (float32)` | float32 | 100,000 | 0.01119 ms | not_measured | managed | 0.529x |
| `np.copy (float32)` | float32 | 10,000,000 | 4.8748 ms | not_measured | managed | 1.276x |
| `np.copy (float64)` | float64 | 1,000 | 0.000573 ms | not_measured | managed | 0.885x |
| `np.copy (float64)` | float64 | 100,000 | 0.02041 ms | not_measured | managed | 0.555x |
| `np.copy (float64)` | float64 | 10,000,000 | 21.0648 ms | not_measured | managed | 0.605x |
| `np.copy (int32)` | int32 | 1,000 | 0.000592 ms | not_measured | managed | 0.784x |
| `np.copy (int32)` | int32 | 100,000 | 0.00620 ms | not_measured | managed | 0.947x |
| `np.copy (int32)` | int32 | 10,000,000 | 3.8604 ms | not_measured | managed | 1.594x |
| `np.copy (int64)` | int64 | 1,000 | 0.000657 ms | not_measured | managed | 0.769x |
| `np.copy (int64)` | int64 | 100,000 | 0.02239 ms | not_measured | managed | 0.506x |
| `np.copy (int64)` | int64 | 10,000,000 | 27.4462 ms | not_measured | managed | 0.452x |
| `np.empty (float32)` | float32 | 1,000 | 0.000228 ms | not_measured | managed | 0.890x |
| `np.empty (float32)` | float32 | 100,000 | 0.000323 ms | not_measured | managed | 0.889x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.05658 ms | not_measured | managed | 0.152x |
| `np.empty (float64)` | float64 | 1,000 | 0.000211 ms | not_measured | managed | 0.981x |
| `np.empty (float64)` | float64 | 100,000 | 0.000241 ms | not_measured | managed | 1.178x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.05430 ms | not_measured | managed | 0.205x |
| `np.empty (int32)` | int32 | 1,000 | 0.000350 ms | not_measured | managed | 0.591x |
| `np.empty (int32)` | int32 | 100,000 | 0.000373 ms | not_measured | managed | 0.718x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.03407 ms | not_measured | managed | 0.137x |
| `np.empty (int64)` | int64 | 1,000 | 0.000230 ms | not_measured | managed | 0.891x |
| `np.empty (int64)` | int64 | 100,000 | 0.000276 ms | not_measured | managed | 1.007x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.06619 ms | not_measured | managed | 0.166x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.000464 ms | not_measured | managed | 0.532x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.000462 ms | not_measured | managed | 0.716x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.03776 ms | not_measured | managed | 0.230x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.000414 ms | not_measured | managed | 0.594x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.000363 ms | not_measured | managed | 0.871x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.03888 ms | not_measured | managed | 0.288x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.000421 ms | not_measured | managed | 0.587x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.000471 ms | not_measured | managed | 0.705x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.05770 ms | not_measured | managed | 0.084x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.000499 ms | not_measured | managed | 0.495x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.000437 ms | not_measured | managed | 0.751x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.07519 ms | not_measured | managed | 0.148x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.00203 ms | not_measured | managed | 0.487x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.07574 ms | not_measured | managed | 0.152x |
| `np.eye(n) (float64)` | float64 | 10,000,000 | 1.1219 ms | not_measured | managed | 0.674x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.000454 ms | not_measured | managed | 0.648x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.000759 ms | not_measured | managed | 0.387x |
| `np.frombuffer(buffer) (float64)` | float64 | 10,000,000 | 0.000436 ms | not_measured | managed | 0.674x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.06261 ms | not_measured | managed | 1.483x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 5.6461 ms | not_measured | managed | 1.725x |
| `np.fromstring(text) (float64)` | float64 | 10,000,000 | 71.5167 ms | not_measured | managed | 3.473x |
| `np.full (float32)` | float32 | 1,000 | 0.000559 ms | not_measured | managed | 1.365x |
| `np.full (float32)` | float32 | 100,000 | 0.01720 ms | not_measured | managed | 0.303x |
| `np.full (float32)` | float32 | 10,000,000 | 4.5377 ms | not_measured | managed | 1.575x |
| `np.full (float64)` | float64 | 1,000 | 0.000461 ms | not_measured | managed | 1.718x |
| `np.full (float64)` | float64 | 100,000 | 0.01806 ms | not_measured | managed | 0.536x |
| `np.full (float64)` | float64 | 10,000,000 | 15.7856 ms | not_measured | managed | 0.915x |
| `np.full (int32)` | int32 | 1,000 | 0.000474 ms | not_measured | managed | 1.597x |
| `np.full (int32)` | int32 | 100,000 | 0.01067 ms | not_measured | managed | 0.486x |
| `np.full (int32)` | int32 | 10,000,000 | 4.0608 ms | not_measured | managed | 1.757x |
| `np.full (int64)` | int64 | 1,000 | 0.000648 ms | not_measured | managed | 1.142x |
| `np.full (int64)` | int64 | 100,000 | 0.01932 ms | not_measured | managed | 0.499x |
| `np.full (int64)` | int64 | 10,000,000 | 27.7660 ms | not_measured | managed | 0.529x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.000554 ms | not_measured | managed | 1.711x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.01468 ms | not_measured | managed | 0.368x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 2.6999 ms | not_measured | managed | 2.577x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.000522 ms | not_measured | managed | 1.856x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.01990 ms | not_measured | managed | 0.496x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 12.8438 ms | not_measured | managed | 1.162x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.000612 ms | not_measured | managed | 1.520x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.01401 ms | not_measured | managed | 0.384x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 4.5415 ms | not_measured | managed | 1.557x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.000446 ms | not_measured | managed | 2.052x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.02014 ms | not_measured | managed | 0.489x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 15.1652 ms | not_measured | managed | 0.935x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.00197 ms | not_measured | managed | 0.666x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.07775 ms | not_measured | managed | 0.153x |
| `np.identity(n) (float64)` | float64 | 10,000,000 | 1.0796 ms | not_measured | managed | 0.703x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.00182 ms | not_measured | managed | 2.464x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.1632 ms | not_measured | managed | 0.576x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 14.3939 ms | not_measured | managed | 2.118x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.00169 ms | not_measured | managed | 2.358x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.1136 ms | not_measured | managed | 0.442x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 18.3570 ms | not_measured | managed | 1.104x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.00345 ms | not_measured | managed | 1.416x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.2782 ms | not_measured | managed | 0.950x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 22.1948 ms | not_measured | managed | 1.564x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.00188 ms | not_measured | managed | 2.586x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.1637 ms | not_measured | managed | 0.960x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 24.6590 ms | not_measured | managed | 1.674x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.00664 ms | not_measured | managed | 1.172x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.1179 ms | not_measured | managed | 2.773x |
| `np.meshgrid(x, y) (float64)` | float64 | 10,000,000 | 1.8830 ms | not_measured | managed | 1.931x |
| `np.ones (float32)` | float32 | 1,000 | 0.000572 ms | not_measured | managed | 1.323x |
| `np.ones (float32)` | float32 | 100,000 | 0.01657 ms | not_measured | managed | 0.314x |
| `np.ones (float32)` | float32 | 10,000,000 | 3.9317 ms | not_measured | managed | 1.760x |
| `np.ones (float64)` | float64 | 1,000 | 0.000502 ms | not_measured | managed | 1.574x |
| `np.ones (float64)` | float64 | 100,000 | 0.01883 ms | not_measured | managed | 0.514x |
| `np.ones (float64)` | float64 | 10,000,000 | 23.6491 ms | not_measured | managed | 0.598x |
| `np.ones (int32)` | int32 | 1,000 | 0.000512 ms | not_measured | managed | 1.477x |
| `np.ones (int32)` | int32 | 100,000 | 0.01091 ms | not_measured | managed | 0.474x |
| `np.ones (int32)` | int32 | 10,000,000 | 3.0623 ms | not_measured | managed | 2.369x |
| `np.ones (int64)` | int64 | 1,000 | 0.000600 ms | not_measured | managed | 1.232x |
| `np.ones (int64)` | int64 | 100,000 | 0.01975 ms | not_measured | managed | 0.488x |
| `np.ones (int64)` | int64 | 10,000,000 | 25.3151 ms | not_measured | managed | 0.544x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.000577 ms | not_measured | managed | 1.610x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.01369 ms | not_measured | managed | 0.394x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 2.2846 ms | not_measured | managed | 3.128x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.000401 ms | not_measured | managed | 2.406x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.01591 ms | not_measured | managed | 0.621x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 14.2423 ms | not_measured | managed | 1.010x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.000584 ms | not_measured | managed | 1.587x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.01543 ms | not_measured | managed | 0.349x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 3.7792 ms | not_measured | managed | 1.908x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.000628 ms | not_measured | managed | 1.451x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.02007 ms | not_measured | managed | 0.490x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 15.8558 ms | not_measured | managed | 0.901x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.000924 ms | not_measured | managed | 0.787x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.03456 ms | not_measured | managed | 0.304x |
| `np.require(a, requirements=C) (float64)` | float64 | 10,000,000 | 0.3743 ms | not_measured | managed | 1.747x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.00716 ms | not_measured | managed | 0.480x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.2048 ms | not_measured | managed | 0.784x |
| `np.vander(x) (float64)` | float64 | 10,000,000 | 2.7677 ms | not_measured | managed | 1.311x |
| `np.zeros (float32)` | float32 | 1,000 | 0.000495 ms | not_measured | managed | 0.549x |
| `np.zeros (float32)` | float32 | 100,000 | 0.00215 ms | not_measured | managed | 2.218x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.00733 ms | not_measured | managed | 1.190x |
| `np.zeros (float64)` | float64 | 1,000 | 0.000543 ms | not_measured | managed | 0.527x |
| `np.zeros (float64)` | float64 | 100,000 | 0.00426 ms | not_measured | managed | 2.170x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.01304 ms | not_measured | managed | 0.842x |
| `np.zeros (int32)` | int32 | 1,000 | 0.000463 ms | not_measured | managed | 0.594x |
| `np.zeros (int32)` | int32 | 100,000 | 0.00358 ms | not_measured | managed | 1.326x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.01128 ms | not_measured | managed | 0.426x |
| `np.zeros (int64)` | int64 | 1,000 | 0.000496 ms | not_measured | managed | 0.577x |
| `np.zeros (int64)` | int64 | 100,000 | 0.00364 ms | not_measured | managed | 2.542x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.00945 ms | not_measured | managed | 1.178x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.000444 ms | not_measured | managed | 2.002x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.00378 ms | not_measured | managed | 1.415x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.00713 ms | not_measured | managed | 997.180x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.000296 ms | not_measured | managed | 3.139x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.00465 ms | not_measured | managed | 2.115x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.00690 ms | not_measured | managed | 2077.535x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.000390 ms | not_measured | managed | 2.269x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.00364 ms | not_measured | managed | 1.469x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.01019 ms | not_measured | managed | 724.269x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.000535 ms | not_measured | managed | 1.628x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.00430 ms | not_measured | managed | 2.275x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.01091 ms | not_measured | managed | 1309.284x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.00869 ms | not_measured | managed | 0.931x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 1.2489 ms | not_measured | managed | 1.150x |
| `np.fft.fft(a) (complex128)` | complex128 | 10,000,000 | 14.2372 ms | not_measured | managed | 1.124x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.06702 ms | not_measured | managed | 0.558x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 5.8737 ms | not_measured | managed | 0.750x |
| `np.fft.fft2(a) (complex128)` | complex128 | 10,000,000 | 25.8080 ms | not_measured | managed | 0.863x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.00620 ms | not_measured | managed | 0.420x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.1118 ms | not_measured | managed | 3.080x |
| `np.fft.fftfreq(n) (float64)` | float64 | 10,000,000 | 1.6496 ms | not_measured | managed | 2.509x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.03102 ms | not_measured | managed | 1.130x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 5.9200 ms | not_measured | managed | 0.733x |
| `np.fft.fftn(a) (complex128)` | complex128 | 10,000,000 | 20.6930 ms | not_measured | managed | 1.060x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.00304 ms | not_measured | managed | 1.365x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.07711 ms | not_measured | managed | 4.033x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 10,000,000 | 1.0609 ms | not_measured | managed | 1.924x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.00554 ms | not_measured | managed | 1.148x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 1.0718 ms | not_measured | managed | 0.860x |
| `np.fft.hfft(a) (float64)` | float64 | 10,000,000 | 12.0548 ms | not_measured | managed | 0.832x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.00899 ms | not_measured | managed | 0.983x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 1.2886 ms | not_measured | managed | 1.115x |
| `np.fft.ifft(a) (complex128)` | complex128 | 10,000,000 | 15.5395 ms | not_measured | managed | 1.081x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.03224 ms | not_measured | managed | 1.205x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 5.8770 ms | not_measured | managed | 0.765x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 10,000,000 | 21.0880 ms | not_measured | managed | 1.055x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.03416 ms | not_measured | managed | 1.065x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 5.9390 ms | not_measured | managed | 0.756x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 10,000,000 | 21.0843 ms | not_measured | managed | 1.042x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.00249 ms | not_measured | managed | 1.662x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.07590 ms | not_measured | managed | 4.078x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 10,000,000 | 1.4530 ms | not_measured | managed | 1.385x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.00548 ms | not_measured | managed | 1.361x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 1.0569 ms | not_measured | managed | 0.654x |
| `np.fft.ihfft(a) (float64)` | float64 | 10,000,000 | 13.7118 ms | not_measured | managed | 0.667x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.00780 ms | not_measured | managed | 0.828x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 0.8408 ms | not_measured | managed | 0.793x |
| `np.fft.irfft(a) (float64)` | float64 | 10,000,000 | 11.9730 ms | not_measured | managed | 0.720x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.01854 ms | not_measured | managed | 1.365x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 2.9735 ms | not_measured | managed | 0.776x |
| `np.fft.irfft2(a) (float64)` | float64 | 10,000,000 | 13.0036 ms | not_measured | managed | 0.698x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.01773 ms | not_measured | managed | 1.458x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 4.4160 ms | not_measured | managed | 0.532x |
| `np.fft.irfftn(a) (float64)` | float64 | 10,000,000 | 10.7914 ms | not_measured | managed | 0.899x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.00538 ms | not_measured | managed | 1.127x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 0.6104 ms | not_measured | managed | 1.083x |
| `np.fft.rfft(a) (float64)` | float64 | 10,000,000 | 8.1527 ms | not_measured | managed | 1.097x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.01758 ms | not_measured | managed | 1.492x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 4.0248 ms | not_measured | managed | 0.550x |
| `np.fft.rfft2(a) (float64)` | float64 | 10,000,000 | 15.3278 ms | not_measured | managed | 0.558x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.00358 ms | not_measured | managed | 0.408x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.04375 ms | not_measured | managed | 0.583x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 10,000,000 | 0.5555 ms | not_measured | managed | 2.544x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.01740 ms | not_measured | managed | 1.369x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 3.1011 ms | not_measured | managed | 0.726x |
| `np.fft.rfftn(a) (float64)` | float64 | 10,000,000 | 9.9131 ms | not_measured | managed | 0.897x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.000174 ms | 0.000225 ms | managed | 2.966x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.000174 ms | 0.000173 ms | managed | 2.827x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.000162 ms | 0.000192 ms | managed | 3.068x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.000461 ms | 0.000519 ms | managed | 1.108x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.00680 ms | 0.01406 ms | managed | 1.027x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 5.8514 ms | 3.4411 ms | openblas | 0.865x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 1,000 | 0.00400 ms | 0.00300 ms | openblas | 2.067x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 100,000 | 1.8205 ms | 0.7969 ms | openblas | 4.433x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 10,000,000 | 2.3765 ms | 1.4162 ms | openblas | 4.519x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.00178 ms | 0.00180 ms | managed | 0.780x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 0.00802 ms | 0.01148 ms | managed | 2.458x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 4.1303 ms | 3.1737 ms | openblas | 1.199x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.00148 ms | 0.00166 ms | managed | 6.053x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.00143 ms | 0.00147 ms | managed | 6.211x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.00146 ms | 0.00154 ms | managed | 6.033x |
| `np.linalg.cholesky(a)` | float64 | 1,000 | missing_backend | 0.00170 ms | openblas | 2.706x |
| `np.linalg.cholesky(a)` | float64 | 100,000 | missing_backend | 0.01440 ms | openblas | 1.306x |
| `np.linalg.cholesky(a)` | float64 | 10,000,000 | missing_backend | 0.03150 ms | openblas | 1.816x |
| `np.linalg.cond(a)` | float64 | 1,000 | missing_backend | 0.03180 ms | openblas | 1.104x |
| `np.linalg.cond(a)` | float64 | 100,000 | missing_backend | 0.2608 ms | openblas | 0.993x |
| `np.linalg.cond(a)` | float64 | 10,000,000 | missing_backend | 0.5530 ms | openblas | 0.997x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.01504 ms | 0.01381 ms | managed | 0.966x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 0.2039 ms | 1.0319 ms | managed | 2.763x |
| `np.linalg.cross(a, b) (float64)` | float64 | 10,000,000 | 65.2549 ms | 127.7356 ms | managed | 1.423x |
| `np.linalg.det(a)` | float64 | 1,000 | missing_backend | 0.00440 ms | openblas | 1.318x |
| `np.linalg.det(a)` | float64 | 100,000 | missing_backend | 0.02560 ms | openblas | 1.047x |
| `np.linalg.det(a)` | float64 | 10,000,000 | missing_backend | 0.05310 ms | openblas | 1.023x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.000181 ms | 0.000275 ms | managed | 3.762x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.000170 ms | 0.000258 ms | managed | 3.882x |
| `np.linalg.diagonal(a) (float64)` | float64 | 10,000,000 | 0.000173 ms | 0.000181 ms | managed | 3.532x |
| `np.linalg.eig(a)` | float64 | 1,000 | missing_backend | 0.1034 ms | openblas | 0.970x |
| `np.linalg.eig(a)` | float64 | 100,000 | missing_backend | 1.3699 ms | openblas | 1.006x |
| `np.linalg.eig(a)` | float64 | 10,000,000 | missing_backend | 2.8455 ms | openblas | 1.078x |
| `np.linalg.eigh(a)` | float64 | 1,000 | missing_backend | 0.04510 ms | openblas | 1.060x |
| `np.linalg.eigh(a)` | float64 | 100,000 | missing_backend | 0.3887 ms | openblas | 1.010x |
| `np.linalg.eigh(a)` | float64 | 10,000,000 | missing_backend | 0.7142 ms | openblas | 1.024x |
| `np.linalg.eigvals(a)` | float64 | 1,000 | missing_backend | 0.05990 ms | openblas | 1.067x |
| `np.linalg.eigvals(a)` | float64 | 100,000 | missing_backend | 0.8128 ms | openblas | 0.966x |
| `np.linalg.eigvals(a)` | float64 | 10,000,000 | missing_backend | 1.5956 ms | openblas | 0.960x |
| `np.linalg.eigvalsh(a)` | float64 | 1,000 | missing_backend | 0.01890 ms | openblas | 1.122x |
| `np.linalg.eigvalsh(a)` | float64 | 100,000 | missing_backend | 0.1581 ms | openblas | 1.002x |
| `np.linalg.eigvalsh(a)` | float64 | 10,000,000 | missing_backend | 0.2950 ms | openblas | 0.995x |
| `np.linalg.inv(a)` | float64 | 1,000 | missing_backend | 0.00830 ms | openblas | 1.325x |
| `np.linalg.inv(a)` | float64 | 100,000 | missing_backend | 0.07560 ms | openblas | 1.040x |
| `np.linalg.inv(a)` | float64 | 10,000,000 | missing_backend | 0.1608 ms | openblas | 1.132x |
| `np.linalg.lstsq(a, b)` | float64 | 1,000 | missing_backend | 0.07720 ms | openblas | 1.027x |
| `np.linalg.lstsq(a, b)` | float64 | 100,000 | missing_backend | 0.7965 ms | openblas | 0.993x |
| `np.linalg.lstsq(a, b)` | float64 | 10,000,000 | missing_backend | 1.4325 ms | openblas | 0.997x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.00423 ms | 0.00468 ms | managed | 0.617x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 1.4334 ms | 2.4712 ms | managed | 0.556x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 10,000,000 | 3.2309 ms | 1.6091 ms | openblas | 1.005x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.00334 ms | 0.00345 ms | managed | 0.705x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.01044 ms | 0.01330 ms | managed | 0.616x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 10,000,000 | 0.01025 ms | 0.01713 ms | managed | 0.635x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 1,000 | missing_backend | 0.03130 ms | openblas | 1.169x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 100,000 | missing_backend | 0.2576 ms | openblas | 1.010x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 10,000,000 | missing_backend | 0.5521 ms | openblas | 0.996x |
| `np.linalg.matrix_power(a, -1)` | float64 | 1,000 | missing_backend | 0.00880 ms | openblas | 1.284x |
| `np.linalg.matrix_power(a, -1)` | float64 | 100,000 | missing_backend | 0.07740 ms | openblas | 1.017x |
| `np.linalg.matrix_power(a, -1)` | float64 | 10,000,000 | missing_backend | 0.1628 ms | openblas | 1.121x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.00821 ms | 0.00667 ms | openblas | 0.840x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.3767 ms | 0.1278 ms | openblas | 0.924x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 10,000,000 | 0.3840 ms | 0.1375 ms | openblas | 0.858x |
| `np.linalg.matrix_rank(a)` | float64 | 1,000 | missing_backend | 0.03230 ms | openblas | 1.139x |
| `np.linalg.matrix_rank(a)` | float64 | 100,000 | missing_backend | 0.2600 ms | openblas | 1.007x |
| `np.linalg.matrix_rank(a)` | float64 | 10,000,000 | missing_backend | 0.5533 ms | openblas | 0.995x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.000247 ms | 0.000279 ms | managed | 2.721x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.000242 ms | 0.000360 ms | managed | 2.727x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 10,000,000 | 0.000269 ms | 0.000418 ms | managed | 2.457x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.00835 ms | 0.00560 ms | openblas | 0.970x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.3917 ms | 0.3375 ms | openblas | 0.358x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 10,000,000 | 0.3848 ms | 0.2550 ms | openblas | 0.474x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.000956 ms | 0.00211 ms | managed | 0.987x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.00717 ms | 0.01854 ms | managed | 1.034x |
| `np.linalg.norm(a) (float64)` | float64 | 10,000,000 | 3.0409 ms | 7.5339 ms | managed | 0.926x |
| `np.linalg.norm(a, ord=2)` | float64 | 1,000 | missing_backend | 0.03170 ms | openblas | 1.158x |
| `np.linalg.norm(a, ord=2)` | float64 | 100,000 | missing_backend | 0.2578 ms | openblas | 1.016x |
| `np.linalg.norm(a, ord=2)` | float64 | 10,000,000 | missing_backend | 0.5519 ms | openblas | 0.999x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.00256 ms | 0.00563 ms | managed | 0.890x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.02809 ms | 0.05137 ms | managed | 1.379x |
| `np.linalg.outer(a, b) (float64)` | float64 | 10,000,000 | 11.2218 ms | 13.7038 ms | managed | 1.156x |
| `np.linalg.pinv(a)` | float64 | 1,000 | missing_backend | 0.08040 ms | openblas | 1.057x |
| `np.linalg.pinv(a)` | float64 | 100,000 | missing_backend | 0.7248 ms | openblas | 1.010x |
| `np.linalg.pinv(a)` | float64 | 10,000,000 | missing_backend | 1.4073 ms | openblas | 1.139x |
| `np.linalg.qr(a)` | float64 | 1,000 | missing_backend | 0.01860 ms | openblas | 1.231x |
| `np.linalg.qr(a)` | float64 | 100,000 | missing_backend | 0.1474 ms | openblas | 1.092x |
| `np.linalg.qr(a)` | float64 | 10,000,000 | missing_backend | 0.3365 ms | openblas | 1.185x |
| `np.linalg.slogdet(a)` | float64 | 1,000 | missing_backend | 0.00440 ms | openblas | 1.500x |
| `np.linalg.slogdet(a)` | float64 | 100,000 | missing_backend | 0.02580 ms | openblas | 1.074x |
| `np.linalg.slogdet(a)` | float64 | 10,000,000 | missing_backend | 0.05330 ms | openblas | 1.036x |
| `np.linalg.solve(a, b)` | float64 | 1,000 | missing_backend | 0.00460 ms | openblas | 1.630x |
| `np.linalg.solve(a, b)` | float64 | 100,000 | missing_backend | 0.02730 ms | openblas | 1.081x |
| `np.linalg.solve(a, b)` | float64 | 10,000,000 | missing_backend | 0.05520 ms | openblas | 1.043x |
| `np.linalg.svd(a)` | float64 | 1,000 | missing_backend | 0.07160 ms | openblas | 1.046x |
| `np.linalg.svd(a)` | float64 | 100,000 | missing_backend | 0.6798 ms | openblas | 1.001x |
| `np.linalg.svd(a)` | float64 | 10,000,000 | missing_backend | 1.2957 ms | openblas | 1.144x |
| `np.linalg.svdvals(a)` | float64 | 1,000 | missing_backend | 0.02940 ms | openblas | 1.085x |
| `np.linalg.svdvals(a)` | float64 | 100,000 | missing_backend | 0.2572 ms | openblas | 0.999x |
| `np.linalg.svdvals(a)` | float64 | 10,000,000 | missing_backend | 0.5506 ms | openblas | 0.989x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.00543 ms | 0.00783 ms | managed | 1.105x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.2119 ms | 0.1075 ms | openblas | 0.593x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 10,000,000 | 0.2036 ms | 0.1017 ms | openblas | 0.626x |
| `np.linalg.tensorinv(a)` | float64 | 1,000 | missing_backend | 0.00850 ms | openblas | 1.353x |
| `np.linalg.tensorinv(a)` | float64 | 100,000 | missing_backend | 0.07570 ms | openblas | 1.044x |
| `np.linalg.tensorinv(a)` | float64 | 10,000,000 | missing_backend | 0.1606 ms | openblas | 1.139x |
| `np.linalg.tensorsolve(a, b)` | float64 | 1,000 | missing_backend | 0.00500 ms | openblas | 1.660x |
| `np.linalg.tensorsolve(a, b)` | float64 | 100,000 | missing_backend | 0.02770 ms | openblas | 1.094x |
| `np.linalg.tensorsolve(a, b)` | float64 | 10,000,000 | missing_backend | 0.05570 ms | openblas | 1.052x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.000321 ms | 0.000571 ms | managed | 4.555x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.000427 ms | 0.000489 ms | managed | 3.061x |
| `np.linalg.trace(a) (float64)` | float64 | 10,000,000 | 0.000350 ms | 0.000614 ms | managed | 3.706x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00137 ms | 0.000754 ms | openblas | 1.164x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.02575 ms | 0.00997 ms | openblas | 0.739x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 10,000,000 | 18.2591 ms | 3.2902 ms | openblas | 0.870x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.00241 ms | 0.00485 ms | managed | 1.029x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.05122 ms | 0.05809 ms | managed | 0.571x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 10,000,000 | 33.4813 ms | 49.3639 ms | managed | 0.585x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.00422 ms | 0.00424 ms | managed | 0.562x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 1.8755 ms | 0.8552 ms | openblas | 0.940x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 4.7218 ms | 1.4691 ms | openblas | 1.106x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.000991 ms | 0.000436 ms | openblas | 1.638x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.00794 ms | 0.00487 ms | openblas | 1.040x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.00919 ms | 0.00744 ms | openblas | 0.965x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.00262 ms | 0.00342 ms | managed | 0.699x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.02678 ms | 0.03058 ms | managed | 1.441x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 11.4648 ms | 11.7713 ms | managed | 1.107x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.00164 ms | 0.00194 ms | managed | 2.330x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 0.01279 ms | 0.01132 ms | openblas | 0.901x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 4.6787 ms | 3.1494 ms | openblas | 0.972x |
| `np.tensordot(a, b, axes=1)` | float64 | 1,000 | 0.00430 ms | 0.00230 ms | openblas | 2.348x |
| `np.tensordot(a, b, axes=1)` | float64 | 100,000 | 3.4449 ms | 0.7964 ms | openblas | 1.022x |
| `np.tensordot(a, b, axes=1)` | float64 | 10,000,000 | 2.3840 ms | 1.4137 ms | openblas | 1.146x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.000785 ms | 0.000699 ms | openblas | 0.828x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.00701 ms | 0.00956 ms | managed | 1.000x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 3.2287 ms | 5.3735 ms | managed | 0.930x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00139 ms | 0.000687 ms | openblas | 0.900x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.04303 ms | 0.00956 ms | openblas | 0.739x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 23.9367 ms | 3.2659 ms | openblas | 0.923x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.000888 ms | 0.000412 ms | openblas | 1.743x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.01037 ms | 0.00741 ms | openblas | 0.704x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.01424 ms | 0.00707 ms | openblas | 1.009x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.01788 ms | not_measured | managed | 1.649x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 0.8200 ms | not_measured | managed | 2.096x |
| `np.allclose(a, b) (float16)` | float16 | 10,000,000 | 15.0952 ms | not_measured | managed | 1.249x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.00780 ms | not_measured | managed | 1.590x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.3460 ms | not_measured | managed | 0.188x |
| `np.allclose(a, b) (float32)` | float32 | 10,000,000 | 16.1942 ms | not_measured | managed | 0.285x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.01036 ms | not_measured | managed | 1.220x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.3449 ms | not_measured | managed | 1.768x |
| `np.allclose(a, b) (float64)` | float64 | 10,000,000 | 11.1959 ms | not_measured | managed | 0.809x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.00152 ms | not_measured | managed | 1.592x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.09267 ms | not_measured | managed | 0.908x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 12.9019 ms | not_measured | managed | 0.719x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.000453 ms | not_measured | managed | 3.311x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.00487 ms | not_measured | managed | 1.344x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 9.0887 ms | not_measured | managed | 0.403x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.000302 ms | not_measured | managed | 5.076x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.02094 ms | not_measured | managed | 0.580x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 16.0988 ms | not_measured | managed | 0.393x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.00381 ms | not_measured | managed | 0.671x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.8649 ms | not_measured | managed | 0.855x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 87.0893 ms | not_measured | managed | 0.906x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.00117 ms | not_measured | managed | 0.364x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.03721 ms | not_measured | managed | 0.216x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 12.4489 ms | not_measured | managed | 0.632x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.00156 ms | not_measured | managed | 0.308x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.05360 ms | not_measured | managed | 0.434x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 29.2650 ms | not_measured | managed | 0.541x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.00368 ms | not_measured | managed | 0.714x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.8664 ms | not_measured | managed | 0.845x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 80.4308 ms | not_measured | managed | 0.983x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.00120 ms | not_measured | managed | 0.352x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.02213 ms | not_measured | managed | 0.356x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 12.7519 ms | not_measured | managed | 0.605x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.00160 ms | not_measured | managed | 0.303x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.06275 ms | not_measured | managed | 0.374x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 28.6278 ms | not_measured | managed | 0.565x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.01851 ms | not_measured | managed | 1.501x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 0.8341 ms | not_measured | managed | 2.038x |
| `np.isclose(a, b) (float16)` | float16 | 10,000,000 | 14.4769 ms | not_measured | managed | 1.308x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.01355 ms | not_measured | managed | 0.796x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.3552 ms | not_measured | managed | 0.178x |
| `np.isclose(a, b) (float32)` | float32 | 10,000,000 | 13.1691 ms | not_measured | managed | 0.345x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.00993 ms | not_measured | managed | 1.104x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.3010 ms | not_measured | managed | 1.970x |
| `np.isclose(a, b) (float64)` | float64 | 10,000,000 | 12.9080 ms | not_measured | managed | 0.690x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.000284 ms | not_measured | managed | 1.539x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.00369 ms | not_measured | managed | 0.438x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | 0.00790 ms | not_measured | managed | 0.972x |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.000397 ms | not_measured | managed | 1.103x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.00260 ms | not_measured | managed | 0.623x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | 0.01337 ms | not_measured | managed | 0.663x |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.000268 ms | not_measured | managed | 1.634x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.00285 ms | not_measured | managed | 0.568x |
| `np.iscomplex(a) (float64)` | float64 | 10,000,000 | 0.01477 ms | not_measured | managed | 0.508x |
| `np.iscomplexobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.000837 ms | not_measured | managed | 0.956x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.03968 ms | not_measured | managed | 1.186x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 3.9304 ms | not_measured | managed | 1.428x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.000695 ms | not_measured | managed | 0.540x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.00733 ms | not_measured | managed | 0.683x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 4.7012 ms | not_measured | managed | 0.605x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.000779 ms | not_measured | managed | 0.533x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.01856 ms | not_measured | managed | 0.515x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 6.0561 ms | not_measured | managed | 0.779x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.000984 ms | not_measured | managed | 0.847x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.04197 ms | not_measured | managed | 1.176x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 5.3269 ms | not_measured | managed | 1.101x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.000707 ms | not_measured | managed | 0.533x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.00678 ms | not_measured | managed | 0.766x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 4.8251 ms | not_measured | managed | 0.622x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.000742 ms | not_measured | managed | 0.557x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.01795 ms | not_measured | managed | 0.550x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 9.4846 ms | not_measured | managed | 0.516x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.00113 ms | not_measured | managed | 0.897x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.05417 ms | not_measured | managed | 1.242x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 5.0918 ms | not_measured | managed | 1.497x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.000677 ms | not_measured | managed | 0.539x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.00577 ms | not_measured | managed | 0.689x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 4.4291 ms | not_measured | managed | 0.610x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.000757 ms | not_measured | managed | 0.539x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.01504 ms | not_measured | managed | 0.560x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 9.5320 ms | not_measured | managed | 0.482x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.000337 ms | not_measured | managed | 5.439x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.00242 ms | not_measured | managed | 31.591x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 0.3623 ms | not_measured | managed | 29.777x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.000346 ms | not_measured | managed | 3.234x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.00257 ms | not_measured | managed | 3.485x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 0.3663 ms | not_measured | managed | 20.240x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.000328 ms | not_measured | managed | 3.497x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.00257 ms | not_measured | managed | 6.287x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 0.2520 ms | not_measured | managed | 54.012x |
| `np.isrealobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 1,000 | 0.000001 ms | not_measured | managed | 476.000x |
| `np.isscalar(a) (float16)` | float16 | 100,000 | 0.000001 ms | not_measured | managed | 468.000x |
| `np.isscalar(a) (float16)` | float16 | 10,000,000 | 0.000001 ms | not_measured | managed | 476.000x |
| `np.isscalar(a) (float32)` | float32 | 1,000 | 0.000001 ms | not_measured | managed | 471.000x |
| `np.isscalar(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 473.000x |
| `np.isscalar(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
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
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.00309 ms | not_measured | managed | 0.875x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.8285 ms | not_measured | managed | 0.909x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 83.1281 ms | not_measured | managed | 0.963x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 0.412x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.01479 ms | not_measured | managed | 0.527x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 11.8270 ms | not_measured | managed | 0.643x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.00140 ms | not_measured | managed | 0.349x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.05882 ms | not_measured | managed | 0.396x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 29.1653 ms | not_measured | managed | 0.543x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.00294 ms | not_measured | managed | 0.987x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.9215 ms | not_measured | managed | 0.804x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 92.3156 ms | not_measured | managed | 0.867x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 0.408x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.01967 ms | not_measured | managed | 0.398x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 9.8702 ms | not_measured | managed | 0.795x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.00137 ms | not_measured | managed | 0.347x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.05424 ms | not_measured | managed | 0.431x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 29.2314 ms | not_measured | managed | 0.563x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.000226 ms | not_measured | managed | 0.381x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.000254 ms | not_measured | managed | 0.343x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.000152 ms | not_measured | managed | 0.572x |
| `a.flatten` | float64 | 1,000 | 0.00108 ms | not_measured | managed | 0.411x |
| `a.flatten` | float64 | 100,000 | 0.02689 ms | not_measured | managed | 0.434x |
| `a.flatten` | float64 | 10,000,000 | 12.1791 ms | not_measured | managed | 0.985x |
| `a.ravel() (view)` | float64 | 1,000 | 0.000492 ms | not_measured | managed | 0.169x |
| `a.ravel() (view)` | float64 | 100,000 | 0.000338 ms | not_measured | managed | 0.254x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.000510 ms | not_measured | managed | 0.163x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.00129 ms | not_measured | managed | 1.111x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.1149 ms | not_measured | managed | 2.611x |
| `np.append(a, b) (float64)` | float64 | 10,000,000 | 1.8357 ms | not_measured | managed | 1.055x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.00102 ms | not_measured | managed | 6.189x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.00133 ms | not_measured | managed | 4.742x |
| `np.array_split(a, sections) (float64)` | float64 | 10,000,000 | 0.00132 ms | not_measured | managed | 4.742x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 202.000x |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 195.000x |
| `np.atleast_1d(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.000206 ms | not_measured | managed | 1.981x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.000252 ms | not_measured | managed | 1.611x |
| `np.atleast_2d(a) (float64)` | float64 | 10,000,000 | 0.000185 ms | not_measured | managed | 2.200x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.000496 ms | not_measured | managed | 0.855x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.000527 ms | not_measured | managed | 0.805x |
| `np.atleast_3d(a) (float64)` | float64 | 10,000,000 | 0.000382 ms | not_measured | managed | 1.107x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.00148 ms | not_measured | managed | 2.074x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.1099 ms | not_measured | managed | 2.758x |
| `np.block([a, b]) (float64)` | float64 | 10,000,000 | 1.3440 ms | not_measured | managed | 1.477x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.000561 ms | not_measured | managed | 5.390x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.000583 ms | not_measured | managed | 5.271x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 10,000,000 | 0.000605 ms | not_measured | managed | 5.000x |
| `np.byteswap` | float64 | 1,000 | 0.000677 ms | not_measured | managed | 1.028x |
| `np.byteswap` | float64 | 100,000 | 0.04410 ms | not_measured | managed | 0.777x |
| `np.byteswap` | float64 | 10,000,000 | 10.4459 ms | not_measured | managed | 1.685x |
| `np.c_` | float64 | 1,000 | 0.00412 ms | not_measured | managed | 1.017x |
| `np.c_` | float64 | 100,000 | 0.1619 ms | not_measured | managed | 1.769x |
| `np.c_` | float64 | 10,000,000 | 35.3245 ms | not_measured | managed | 1.015x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.00331 ms | not_measured | managed | 0.611x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.1343 ms | not_measured | managed | 2.195x |
| `np.column_stack((a, b)) (float64)` | float64 | 10,000,000 | 2.9084 ms | not_measured | managed | 1.144x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.00171 ms | not_measured | managed | 0.477x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.1168 ms | not_measured | managed | 2.578x |
| `np.concat((a, b)) (float64)` | float64 | 10,000,000 | 1.2689 ms | not_measured | managed | 1.574x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.00179 ms | not_measured | managed | 0.648x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.1894 ms | not_measured | managed | 2.966x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 10,000,000 | 3.1133 ms | not_measured | managed | 0.952x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.00146 ms | not_measured | managed | 0.560x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.09984 ms | not_measured | managed | 3.929x |
| `np.concatenate([a, b]) (float64)` | float64 | 10,000,000 | 1.3753 ms | not_measured | managed | 1.445x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.00215 ms | not_measured | managed | 0.420x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.09886 ms | not_measured | managed | 3.746x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 10,000,000 | 1.3797 ms | not_measured | managed | 1.426x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00189 ms | not_measured | managed | 0.588x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.09448 ms | not_measured | managed | 3.875x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 10,000,000 | 1.5440 ms | not_measured | managed | 2.928x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.00177 ms | not_measured | managed | 1.075x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.04198 ms | not_measured | managed | 0.307x |
| `np.delete(a, index) (float64)` | float64 | 10,000,000 | 0.4487 ms | not_measured | managed | 2.179x |
| `np.diag` | float64 | 1,000 | 0.000938 ms | not_measured | managed | 0.777x |
| `np.diag` | float64 | 100,000 | 0.06999 ms | not_measured | managed | 0.011x |
| `np.diag` | float64 | 10,000,000 | 3.6458 ms | not_measured | managed | 0.000x |
| `np.diagflat` | float64 | 1,000 | 0.00100 ms | not_measured | managed | 2.696x |
| `np.diagflat` | float64 | 100,000 | 0.07207 ms | not_measured | managed | 0.187x |
| `np.diagflat` | float64 | 10,000,000 | 2.4444 ms | not_measured | managed | 1.062x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.00197 ms | not_measured | managed | 4.527x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.00193 ms | not_measured | managed | 4.722x |
| `np.dsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00154 ms | not_measured | managed | 5.676x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.00233 ms | not_measured | managed | 1.018x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.1869 ms | not_measured | managed | 2.008x |
| `np.dstack([a, b]) (float64)` | float64 | 10,000,000 | 1.5404 ms | not_measured | managed | 2.220x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.000194 ms | not_measured | managed | 6.149x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.000203 ms | not_measured | managed | 6.182x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.000254 ms | not_measured | managed | 4.472x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.000213 ms | not_measured | managed | 5.498x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.000346 ms | not_measured | managed | 3.422x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.000331 ms | not_measured | managed | 3.405x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.000254 ms | not_measured | managed | 4.819x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.000200 ms | not_measured | managed | 6.625x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.000289 ms | not_measured | managed | 4.118x |
| `np.fill_diagonal` | float64 | 1,000 | 0.000378 ms | not_measured | managed | 2.029x |
| `np.fill_diagonal` | float64 | 100,000 | 0.000495 ms | not_measured | managed | 4.840x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.01033 ms | not_measured | managed | 1.701x |
| `np.flip` | float64 | 1,000 | 0.000237 ms | not_measured | managed | 1.616x |
| `np.flip` | float64 | 100,000 | 0.000156 ms | not_measured | managed | 2.532x |
| `np.flip` | float64 | 10,000,000 | 0.000198 ms | not_measured | managed | 1.914x |
| `np.fliplr` | float64 | 1,000 | 0.000145 ms | not_measured | managed | 2.166x |
| `np.fliplr` | float64 | 100,000 | 0.000139 ms | not_measured | managed | 2.424x |
| `np.fliplr` | float64 | 10,000,000 | 0.000252 ms | not_measured | managed | 1.246x |
| `np.flipud` | float64 | 1,000 | 0.000187 ms | not_measured | managed | 1.519x |
| `np.flipud` | float64 | 100,000 | 0.000179 ms | not_measured | managed | 1.620x |
| `np.flipud` | float64 | 10,000,000 | 0.000153 ms | not_measured | managed | 1.843x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.00204 ms | not_measured | managed | 4.347x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.00191 ms | not_measured | managed | 4.707x |
| `np.hsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00161 ms | not_measured | managed | 5.622x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.00221 ms | not_measured | managed | 0.700x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.07687 ms | not_measured | managed | 5.131x |
| `np.hstack([a, b]) (float64)` | float64 | 10,000,000 | 1.0505 ms | not_measured | managed | 1.896x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.00380 ms | not_measured | managed | 1.287x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.03144 ms | not_measured | managed | 0.517x |
| `np.insert(a, index, value) (float64)` | float64 | 10,000,000 | 0.4604 ms | not_measured | managed | 2.130x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.04317 ms | not_measured | managed | 0.890x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 9.6357 ms | not_measured | managed | 0.805x |
| `np.intersect1d(a, b) (int32)` | int32 | 10,000,000 | 45.9168 ms | not_measured | managed | 2.810x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.01434 ms | not_measured | managed | 1.127x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 0.9734 ms | not_measured | managed | 0.417x |
| `np.isin(a, b) (int32)` | int32 | 10,000,000 | 7.0406 ms | not_measured | managed | 0.936x |
| `np.ix_` | float64 | 1,000 | 0.00121 ms | not_measured | managed | 1.389x |
| `np.ix_` | float64 | 100,000 | 0.000624 ms | not_measured | managed | 2.723x |
| `np.ix_` | float64 | 10,000,000 | 0.000607 ms | not_measured | managed | 2.707x |
| `np.matrix_transpose` | float64 | 1,000 | 0.000248 ms | not_measured | managed | 2.190x |
| `np.matrix_transpose` | float64 | 100,000 | 0.000233 ms | not_measured | managed | 2.429x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.000342 ms | not_measured | managed | 1.585x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.000932 ms | not_measured | managed | 2.108x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.000732 ms | not_measured | managed | 2.734x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.000937 ms | not_measured | managed | 2.088x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.00675 ms | not_measured | managed | 1.150x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.03062 ms | not_measured | managed | 0.626x |
| `np.pad(a, width) (float64)` | float64 | 10,000,000 | 0.4101 ms | not_measured | managed | 2.460x |
| `np.permute_dims` | float64 | 1,000 | 0.000231 ms | not_measured | managed | 1.437x |
| `np.permute_dims` | float64 | 100,000 | 0.000233 ms | not_measured | managed | 1.455x |
| `np.permute_dims` | float64 | 10,000,000 | 0.000210 ms | not_measured | managed | 1.571x |
| `np.r_` | float64 | 1,000 | 0.00137 ms | not_measured | managed | 1.994x |
| `np.r_` | float64 | 100,000 | 0.09360 ms | not_measured | managed | 3.234x |
| `np.r_` | float64 | 10,000,000 | 22.8472 ms | not_measured | managed | 1.075x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.00400 ms | not_measured | managed | 1.400x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.07955 ms | not_measured | managed | 3.996x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 34.1003 ms | not_measured | managed | 0.948x |
| `np.r_(0:N)` | float64 | 1,000 | 0.00209 ms | not_measured | managed | 0.985x |
| `np.r_(0:N)` | float64 | 100,000 | 0.04711 ms | not_measured | managed | 6.154x |
| `np.r_(0:N)` | float64 | 10,000,000 | 30.7386 ms | not_measured | managed | 0.783x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.00421 ms | not_measured | managed | 0.968x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.1007 ms | not_measured | managed | 2.967x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 22.3200 ms | not_measured | managed | 1.107x |
| `np.ravel` | float64 | 1,000 | 0.000518 ms | not_measured | managed | 0.589x |
| `np.ravel` | float64 | 100,000 | 0.000536 ms | not_measured | managed | 0.591x |
| `np.ravel` | float64 | 10,000,000 | 0.000521 ms | not_measured | managed | 0.572x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.00237 ms | not_measured | managed | 1.096x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.3591 ms | not_measured | managed | 0.988x |
| `np.repeat(a, repeats) (float64)` | float64 | 10,000,000 | 2.7296 ms | not_measured | managed | 1.265x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.000315 ms | not_measured | managed | 1.765x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.000329 ms | not_measured | managed | 1.653x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.000395 ms | not_measured | managed | 1.385x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.00198 ms | not_measured | managed | 1.098x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.08356 ms | not_measured | managed | 3.534x |
| `np.resize(a, shape) (float64)` | float64 | 10,000,000 | 1.2510 ms | not_measured | managed | 1.558x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.00283 ms | not_measured | managed | 1.486x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.02965 ms | not_measured | managed | 0.532x |
| `np.roll(a, shift) (float64)` | float64 | 10,000,000 | 0.6118 ms | not_measured | managed | 1.626x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.000343 ms | not_measured | managed | 1.665x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.000218 ms | not_measured | managed | 2.716x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.000250 ms | not_measured | managed | 2.212x |
| `np.rot90` | float64 | 1,000 | 0.000490 ms | not_measured | managed | 6.271x |
| `np.rot90` | float64 | 100,000 | 0.000407 ms | not_measured | managed | 7.934x |
| `np.rot90` | float64 | 10,000,000 | 0.000534 ms | not_measured | managed | 5.436x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.02838 ms | not_measured | managed | 1.737x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 7.3148 ms | not_measured | managed | 0.951x |
| `np.setdiff1d(a, b) (int32)` | int32 | 10,000,000 | 30.1996 ms | not_measured | managed | 4.185x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.05017 ms | not_measured | managed | 0.786x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 9.6690 ms | not_measured | managed | 0.738x |
| `np.setxor1d(a, b) (int32)` | int32 | 10,000,000 | 40.1791 ms | not_measured | managed | 3.127x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.00127 ms | not_measured | managed | 6.789x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.00145 ms | not_measured | managed | 6.022x |
| `np.split(a, sections) (float64)` | float64 | 10,000,000 | 0.00145 ms | not_measured | managed | 5.771x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.000150 ms | not_measured | managed | 1.913x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.000145 ms | not_measured | managed | 1.993x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.000160 ms | not_measured | managed | 1.781x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.000238 ms | not_measured | managed | 1.311x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.000136 ms | not_measured | managed | 2.449x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.000144 ms | not_measured | managed | 2.146x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.00200 ms | not_measured | managed | 0.963x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.1007 ms | not_measured | managed | 4.006x |
| `np.stack([a, b]) (float64)` | float64 | 10,000,000 | 1.0417 ms | not_measured | managed | 1.965x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00265 ms | not_measured | managed | 0.866x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.1217 ms | not_measured | managed | 3.320x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 10,000,000 | 1.4312 ms | not_measured | managed | 2.398x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.000410 ms | not_measured | managed | 0.922x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.000246 ms | not_measured | managed | 1.585x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.000231 ms | not_measured | managed | 1.619x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.000249 ms | not_measured | managed | 1.522x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.000214 ms | not_measured | managed | 1.799x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.000196 ms | not_measured | managed | 1.929x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.00248 ms | not_measured | managed | 0.962x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.08634 ms | not_measured | managed | 3.346x |
| `np.tile(a, reps) (float64)` | float64 | 10,000,000 | 1.4756 ms | not_measured | managed | 1.323x |
| `np.transpose(a)` | float64 | 1,000 | 0.000375 ms | not_measured | managed | 0.877x |
| `np.transpose(a)` | float64 | 100,000 | 0.000262 ms | not_measured | managed | 1.286x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.000370 ms | not_measured | managed | 0.876x |
| `np.tri` | float64 | 1,000 | 0.00140 ms | not_measured | managed | 2.079x |
| `np.tri` | float64 | 100,000 | 0.02327 ms | not_measured | managed | 2.138x |
| `np.tri` | float64 | 10,000,000 | 15.6812 ms | not_measured | managed | 0.898x |
| `np.tril` | float64 | 1,000 | 0.000876 ms | not_measured | managed | 4.320x |
| `np.tril` | float64 | 100,000 | 0.03318 ms | not_measured | managed | 1.467x |
| `np.tril` | float64 | 10,000,000 | 17.0575 ms | not_measured | managed | 0.918x |
| `np.tril_indices` | float64 | 1,000 | 0.00323 ms | not_measured | managed | 3.061x |
| `np.tril_indices` | float64 | 100,000 | 0.04295 ms | not_measured | managed | 1.619x |
| `np.tril_indices` | float64 | 10,000,000 | 10.2361 ms | not_measured | managed | 1.841x |
| `np.trim_zeros` | float64 | 1,000 | 0.00399 ms | not_measured | managed | 2.845x |
| `np.trim_zeros` | float64 | 100,000 | 0.00685 ms | not_measured | managed | 165.114x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.03347 ms | not_measured | managed | 3142.583x |
| `np.triu` | float64 | 1,000 | 0.00128 ms | not_measured | managed | 3.009x |
| `np.triu` | float64 | 100,000 | 0.02721 ms | not_measured | managed | 1.706x |
| `np.triu` | float64 | 10,000,000 | 12.2508 ms | not_measured | managed | 1.307x |
| `np.triu_indices` | float64 | 1,000 | 0.00301 ms | not_measured | managed | 3.428x |
| `np.triu_indices` | float64 | 100,000 | 0.03512 ms | not_measured | managed | 1.983x |
| `np.triu_indices` | float64 | 10,000,000 | 11.2704 ms | not_measured | managed | 1.796x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.01075 ms | not_measured | managed | 1.981x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 5.2560 ms | not_measured | managed | 0.874x |
| `np.union1d(a, b) (int32)` | int32 | 10,000,000 | 27.1307 ms | not_measured | managed | 3.292x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.01030 ms | not_measured | managed | 1.552x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.4124 ms | not_measured | managed | 0.691x |
| `np.unique(a) (int32)` | int32 | 10,000,000 | 13.9834 ms | not_measured | managed | 4.155x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.01920 ms | not_measured | managed | 1.822x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 2.1803 ms | not_measured | managed | 2.959x |
| `np.unique_all(a) (int32)` | int32 | 10,000,000 | 28.0818 ms | not_measured | managed | 3.693x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.01294 ms | not_measured | managed | 0.733x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 4.6144 ms | not_measured | managed | 0.135x |
| `np.unique_counts(a) (int32)` | int32 | 10,000,000 | 14.5719 ms | not_measured | managed | 0.628x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.01823 ms | not_measured | managed | 1.230x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 1.9057 ms | not_measured | managed | 1.285x |
| `np.unique_inverse(a) (int32)` | int32 | 10,000,000 | 22.6724 ms | not_measured | managed | 1.600x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.01006 ms | not_measured | managed | 1.559x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 3.2943 ms | not_measured | managed | 0.868x |
| `np.unique_values(a) (int32)` | int32 | 10,000,000 | 13.2577 ms | not_measured | managed | 4.281x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.00158 ms | not_measured | managed | 1.960x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.00120 ms | not_measured | managed | 2.602x |
| `np.unstack(a) (float64)` | float64 | 10,000,000 | 0.00143 ms | not_measured | managed | 2.126x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.00138 ms | not_measured | managed | 6.447x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.00200 ms | not_measured | managed | 4.412x |
| `np.vsplit(a, sections) (float64)` | float64 | 10,000,000 | 0.00143 ms | not_measured | managed | 5.927x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.00337 ms | not_measured | managed | 0.561x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.08002 ms | not_measured | managed | 4.761x |
| `np.vstack([a, b]) (float64)` | float64 | 10,000,000 | 1.3637 ms | not_measured | managed | 1.439x |
| `reshape 1D->2D` | float64 | 1,000 | 0.000464 ms | not_measured | managed | 0.319x |
| `reshape 1D->2D` | float64 | 100,000 | 0.000360 ms | not_measured | managed | 0.411x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.000565 ms | not_measured | managed | 0.260x |
| `reshape 1D->3D` | float64 | 1,000 | 0.000505 ms | not_measured | managed | 0.315x |
| `reshape 1D->3D` | float64 | 100,000 | 0.000432 ms | not_measured | managed | 0.380x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.000385 ms | not_measured | managed | 0.416x |
| `reshape 2D->1D` | float64 | 1,000 | 0.000358 ms | not_measured | managed | 0.397x |
| `reshape 2D->1D` | float64 | 100,000 | 0.000550 ms | not_measured | managed | 0.262x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.000396 ms | not_measured | managed | 0.356x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.000728 ms | not_measured | managed | 1.716x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.00773 ms | not_measured | managed | 5.089x |
| `a.all() [ndarray] (float64)` | float64 | 10,000,000 | 0.1249 ms | not_measured | managed | 3.072x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.000580 ms | not_measured | managed | 2.141x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.000684 ms | not_measured | managed | 55.339x |
| `a.any() [ndarray] (float64)` | float64 | 10,000,000 | 0.000661 ms | not_measured | managed | 562.431x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.000435 ms | not_measured | managed | 0.887x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.01608 ms | not_measured | managed | 0.986x |
| `a.argmax() [ndarray] (float64)` | float64 | 10,000,000 | 0.1628 ms | not_measured | managed | 0.954x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.000435 ms | not_measured | managed | 0.880x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.01589 ms | not_measured | managed | 0.997x |
| `a.argmin() [ndarray] (float64)` | float64 | 10,000,000 | 0.1628 ms | not_measured | managed | 0.950x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.00666 ms | not_measured | managed | 0.517x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.4919 ms | not_measured | managed | 0.289x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 10,000,000 | 9.1927 ms | not_measured | managed | 0.321x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.01645 ms | not_measured | managed | 0.552x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 2.4579 ms | not_measured | managed | 0.547x |
| `a.argsort() [ndarray] (float64)` | float64 | 10,000,000 | 31.8632 ms | not_measured | managed | 0.723x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.00314 ms | not_measured | managed | 1.412x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.08695 ms | not_measured | managed | 6.302x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 10,000,000 | 1.7886 ms | not_measured | managed | 3.466x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.000886 ms | not_measured | managed | 1.121x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.02870 ms | not_measured | managed | 0.527x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 10,000,000 | 0.6261 ms | not_measured | managed | 2.508x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.000785 ms | not_measured | managed | 1.785x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.04615 ms | not_measured | managed | 1.734x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 10,000,000 | 0.5632 ms | not_measured | managed | 3.541x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `a.conj() [ndarray] (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 60.000x |
| `a.conjugate() [ndarray] (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.00199 ms | not_measured | managed | 1.405x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.1828 ms | not_measured | managed | 0.909x |
| `a.cumprod() [ndarray] (float64)` | float64 | 10,000,000 | 1.5084 ms | not_measured | managed | 1.838x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.00160 ms | not_measured | managed | 1.348x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.1516 ms | not_measured | managed | 1.078x |
| `a.cumsum() [ndarray] (float64)` | float64 | 10,000,000 | 1.7916 ms | not_measured | managed | 1.519x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.000176 ms | not_measured | managed | 0.591x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.000211 ms | not_measured | managed | 0.488x |
| `a.diagonal() [ndarray] (float64)` | float64 | 10,000,000 | 0.000157 ms | not_measured | managed | 0.656x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.000472 ms | 0.000600 ms | managed | 0.867x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.01247 ms | 0.00900 ms | openblas | 1.113x |
| `a.dot(b) [ndarray] (float64)` | float64 | 10,000,000 | 0.2207 ms | 2.9441 ms | managed | 0.771x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.000091 ms | not_measured | managed | 1.934x |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.01239 ms | not_measured | managed | 0.736x |
| `a.fill(value) [ndarray] (float64)` | float64 | 10,000,000 | 0.1532 ms | not_measured | managed | 0.953x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.000179 ms | not_measured | managed | 0.827x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.000186 ms | not_measured | managed | 0.801x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 10,000,000 | 0.000322 ms | not_measured | managed | 0.466x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.000005 ms | not_measured | managed | 18.200x |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.000004 ms | not_measured | managed | 23.000x |
| `a.item(index) [ndarray] (float64)` | float64 | 10,000,000 | 0.000005 ms | not_measured | managed | 18.200x |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.000492 ms | not_measured | managed | 1.896x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.00721 ms | not_measured | managed | 1.276x |
| `a.max() [ndarray] (float64)` | float64 | 10,000,000 | 0.1022 ms | not_measured | managed | 0.932x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.000371 ms | not_measured | managed | 2.523x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.00729 ms | not_measured | managed | 1.261x |
| `a.min() [ndarray] (float64)` | float64 | 10,000,000 | 0.1007 ms | not_measured | managed | 0.945x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.00133 ms | not_measured | managed | 1.535x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.04973 ms | not_measured | managed | 3.438x |
| `a.nonzero() [ndarray] (float64)` | float64 | 10,000,000 | 0.7570 ms | not_measured | managed | 3.761x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.00628 ms | not_measured | managed | 0.405x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.6189 ms | not_measured | managed | 0.222x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 10,000,000 | 4.6193 ms | not_measured | managed | 0.595x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.000342 ms | not_measured | managed | 5.655x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.00781 ms | not_measured | managed | 9.513x |
| `a.prod() [ndarray] (float64)` | float64 | 10,000,000 | 0.09541 ms | not_measured | managed | 7.679x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.783x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.05098 ms | not_measured | managed | 2.078x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 10,000,000 | 0.6875 ms | not_measured | managed | 2.638x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.00197 ms | not_measured | managed | 1.132x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.2598 ms | not_measured | managed | 1.395x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 10,000,000 | 2.7886 ms | not_measured | managed | 1.536x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.000450 ms | not_measured | managed | 0.342x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.000448 ms | not_measured | managed | 0.366x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 10,000,000 | 0.000317 ms | not_measured | managed | 0.511x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.00176 ms | not_measured | managed | 0.376x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.04541 ms | not_measured | managed | 0.274x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 10,000,000 | 1.0140 ms | not_measured | managed | 1.286x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.000755 ms | not_measured | managed | 0.623x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.02496 ms | not_measured | managed | 0.527x |
| `a.round() [ndarray] (float64)` | float64 | 10,000,000 | 0.6861 ms | not_measured | managed | 1.828x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.01056 ms | not_measured | managed | 0.544x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 2.6343 ms | not_measured | managed | 0.777x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 10,000,000 | 30.6055 ms | not_measured | managed | 0.770x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.00101 ms | not_measured | managed | 0.375x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.02268 ms | not_measured | managed | 0.409x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 10,000,000 | 0.2348 ms | not_measured | managed | 0.624x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.000007 ms | not_measured | managed | 18.857x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.000008 ms | not_measured | managed | 16.500x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 10,000,000 | 0.000007 ms | not_measured | managed | 18.857x |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.01392 ms | not_measured | managed | 0.207x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 1.3402 ms | not_measured | managed | 0.353x |
| `a.sort() [ndarray] (float64)` | float64 | 10,000,000 | 21.4747 ms | not_measured | managed | 0.361x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.000184 ms | not_measured | managed | 0.685x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.000152 ms | not_measured | managed | 0.822x |
| `a.squeeze() [ndarray] (float64)` | float64 | 10,000,000 | 0.000212 ms | not_measured | managed | 0.580x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.000408 ms | not_measured | managed | 0.328x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.000222 ms | not_measured | managed | 0.604x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 10,000,000 | 0.000269 ms | not_measured | managed | 0.498x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.000843 ms | not_measured | managed | 0.961x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.04048 ms | not_measured | managed | 1.253x |
| `a.take(indices) [ndarray] (float64)` | float64 | 10,000,000 | 0.6261 ms | not_measured | managed | 1.563x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 82.000x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 81.000x |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.000235 ms | not_measured | managed | 0.672x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.05131 ms | not_measured | managed | 0.226x |
| `a.tobytes() [ndarray] (float64)` | float64 | 10,000,000 | 0.7087 ms | not_measured | managed | 2.119x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.3209 ms | not_measured | managed | 0.392x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 2.0126 ms | not_measured | managed | 0.127x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 10,000,000 | 2.9952 ms | not_measured | managed | 0.466x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.00583 ms | not_measured | managed | 1.260x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 1.4777 ms | not_measured | managed | 0.523x |
| `a.tolist() [ndarray] (float64)` | float64 | 10,000,000 | 23.6342 ms | not_measured | managed | 0.601x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.000326 ms | not_measured | managed | 2.718x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.000471 ms | not_measured | managed | 1.885x |
| `a.trace() [ndarray] (float64)` | float64 | 10,000,000 | 0.000372 ms | not_measured | managed | 2.422x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.000294 ms | not_measured | managed | 0.323x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.000182 ms | not_measured | managed | 0.522x |
| `a.transpose() [ndarray] (float64)` | float64 | 10,000,000 | 0.000318 ms | not_measured | managed | 0.302x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.000138 ms | not_measured | managed | 0.580x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.000182 ms | not_measured | managed | 0.440x |
| `a.view() [ndarray] (float64)` | float64 | 10,000,000 | 0.000270 ms | not_measured | managed | 0.296x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.04411 ms | not_measured | managed | 0.917x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 4.3088 ms | not_measured | managed | 0.918x |
| `np.random.beta(a, b) (float64)` | float64 | 10,000,000 | 80.7691 ms | not_measured | managed | 0.526x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.09001 ms | not_measured | managed | 0.255x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 9.1448 ms | not_measured | managed | 0.245x |
| `np.random.binomial(n, p) (int64)` | int64 | 10,000,000 | 108.9320 ms | not_measured | managed | 0.210x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.02269 ms | not_measured | managed | 0.902x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 2.2744 ms | not_measured | managed | 0.869x |
| `np.random.chisquare(df) (float64)` | float64 | 10,000,000 | 33.8522 ms | not_measured | managed | 0.620x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.01395 ms | not_measured | managed | 0.718x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 1.1435 ms | not_measured | managed | 0.759x |
| `np.random.choice(a, size) (float64)` | float64 | 10,000,000 | 13.3708 ms | not_measured | managed | 0.499x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.09871 ms | not_measured | managed | 0.545x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 7.0916 ms | not_measured | managed | 0.810x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 10,000,000 | 98.4400 ms | not_measured | managed | 0.560x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.02159 ms | not_measured | managed | 0.413x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 0.9686 ms | not_measured | managed | 0.847x |
| `np.random.exponential(scale) (float64)` | float64 | 10,000,000 | 10.0589 ms | not_measured | managed | 0.942x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.08463 ms | not_measured | managed | 0.478x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 4.4307 ms | not_measured | managed | 0.895x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 10,000,000 | 56.1036 ms | not_measured | managed | 0.729x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.04067 ms | not_measured | managed | 0.515x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 2.1016 ms | not_measured | managed | 0.940x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 10,000,000 | 20.6259 ms | not_measured | managed | 1.002x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.01580 ms | not_measured | managed | 1.124x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 1.7095 ms | not_measured | managed | 1.049x |
| `np.random.geometric(p) (int64)` | int64 | 10,000,000 | 15.8639 ms | not_measured | managed | 1.113x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.03136 ms | not_measured | managed | 0.460x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 1.5241 ms | not_measured | managed | 0.890x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 10,000,000 | 18.5802 ms | not_measured | managed | 0.780x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1049 ms | not_measured | managed | 0.599x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 13.9238 ms | not_measured | managed | 0.458x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 10,000,000 | 146.4697 ms | not_measured | managed | 0.429x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.02612 ms | not_measured | managed | 0.534x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 1.5334 ms | not_measured | managed | 0.852x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 10,000,000 | 15.2649 ms | not_measured | managed | 0.923x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.01174 ms | not_measured | managed | 0.792x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 1.0904 ms | not_measured | managed | 0.779x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 10,000,000 | 10.6602 ms | not_measured | managed | 0.875x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.02108 ms | not_measured | managed | 0.857x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 1.4596 ms | not_measured | managed | 1.186x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 10,000,000 | 20.9737 ms | not_measured | managed | 0.883x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.02648 ms | not_measured | managed | 0.997x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 4.0411 ms | not_measured | managed | 0.658x |
| `np.random.logseries(p) (int64)` | int64 | 10,000,000 | 29.4820 ms | not_measured | managed | 0.911x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.1382 ms | not_measured | managed | 0.511x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 13.8631 ms | not_measured | managed | 0.561x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 10,000,000 | 200.5783 ms | not_measured | managed | 0.363x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.08988 ms | not_measured | managed | 0.665x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 7.1528 ms | not_measured | managed | 1.074x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 10,000,000 | 42.2343 ms | not_measured | managed | 1.049x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.08493 ms | not_measured | managed | 0.830x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 9.9850 ms | not_measured | managed | 0.711x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 10,000,000 | 126.0805 ms | not_measured | managed | 0.563x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.03707 ms | not_measured | managed | 0.888x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 3.2829 ms | not_measured | managed | 0.981x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 10,000,000 | 45.5488 ms | not_measured | managed | 0.730x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.05391 ms | not_measured | managed | 0.978x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 5.1640 ms | not_measured | managed | 1.006x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 10,000,000 | 51.6681 ms | not_measured | managed | 1.026x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.02258 ms | not_measured | managed | 0.503x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 1.1845 ms | not_measured | managed | 0.896x |
| `np.random.normal(loc, scale) (float64)` | float64 | 10,000,000 | 11.4575 ms | not_measured | managed | 1.016x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.01692 ms | not_measured | managed | 0.887x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 1.5058 ms | not_measured | managed | 0.954x |
| `np.random.pareto(a) (float64)` | float64 | 10,000,000 | 14.9428 ms | not_measured | managed | 1.064x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.01732 ms | not_measured | managed | 0.802x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 1.8356 ms | not_measured | managed | 0.912x |
| `np.random.permutation(a) (float64)` | float64 | 10,000,000 | 16.8370 ms | not_measured | managed | 1.074x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.03368 ms | not_measured | managed | 0.954x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 3.2902 ms | not_measured | managed | 0.983x |
| `np.random.poisson(lam) (int64)` | int64 | 10,000,000 | 40.2091 ms | not_measured | managed | 0.802x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.01594 ms | not_measured | managed | 1.793x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 2.3861 ms | not_measured | managed | 1.170x |
| `np.random.power(a) (float64)` | float64 | 10,000,000 | 15.6366 ms | not_measured | managed | 1.851x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.00634 ms | not_measured | managed | 0.643x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 0.5967 ms | not_measured | managed | 0.588x |
| `np.random.rand(n) (float64)` | float64 | 10,000,000 | 6.7351 ms | not_measured | managed | 0.629x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.00577 ms | not_measured | managed | 1.311x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.5967 ms | not_measured | managed | 0.747x |
| `np.random.randint(low, high) (int64)` | int64 | 10,000,000 | 5.6257 ms | not_measured | managed | 0.912x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.01279 ms | not_measured | managed | 0.834x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 1.2637 ms | not_measured | managed | 0.798x |
| `np.random.randn(n) (float64)` | float64 | 10,000,000 | 12.1108 ms | not_measured | managed | 0.910x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.00721 ms | not_measured | managed | 0.558x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 0.5905 ms | not_measured | managed | 0.594x |
| `np.random.random(n) (float64)` | float64 | 10,000,000 | 5.8362 ms | not_measured | managed | 0.726x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.00606 ms | not_measured | managed | 0.657x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 0.5883 ms | not_measured | managed | 0.597x |
| `np.random.random_sample(n) (float64)` | float64 | 10,000,000 | 6.0339 ms | not_measured | managed | 0.704x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.00996 ms | not_measured | managed | 1.016x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 1.0226 ms | not_measured | managed | 0.929x |
| `np.random.rayleigh(scale) (float64)` | float64 | 10,000,000 | 10.0105 ms | not_measured | managed | 1.032x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.01092 ms | not_measured | managed | 1.203x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 1.8874 ms | not_measured | managed | 0.753x |
| `np.random.shuffle(a) (float64)` | float64 | 10,000,000 | 15.6589 ms | not_measured | managed | 0.909x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.01529 ms | not_measured | managed | 1.391x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 1.4979 ms | not_measured | managed | 1.390x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 10,000,000 | 14.8211 ms | not_measured | managed | 1.465x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.01472 ms | not_measured | managed | 0.585x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 1.0479 ms | not_measured | managed | 0.775x |
| `np.random.standard_exponential(n) (float64)` | float64 | 10,000,000 | 13.7944 ms | not_measured | managed | 0.659x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.02109 ms | not_measured | managed | 0.952x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 2.1795 ms | not_measured | managed | 0.890x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 10,000,000 | 36.6095 ms | not_measured | managed | 0.556x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.01232 ms | not_measured | managed | 0.854x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 1.2655 ms | not_measured | managed | 0.791x |
| `np.random.standard_normal(n) (float64)` | float64 | 10,000,000 | 12.3985 ms | not_measured | managed | 0.888x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.04302 ms | not_measured | managed | 0.765x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 3.2884 ms | not_measured | managed | 0.992x |
| `np.random.standard_t(df) (float64)` | float64 | 10,000,000 | 32.9334 ms | not_measured | managed | 1.023x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.00902 ms | not_measured | managed | 1.053x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 1.4332 ms | not_measured | managed | 0.578x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 10,000,000 | 9.1554 ms | not_measured | managed | 1.004x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.00630 ms | not_measured | managed | 0.794x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 0.5400 ms | not_measured | managed | 0.661x |
| `np.random.uniform(low, high) (float64)` | float64 | 10,000,000 | 5.1699 ms | not_measured | managed | 0.866x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.04983 ms | not_measured | managed | 1.005x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 4.9472 ms | not_measured | managed | 1.001x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 10,000,000 | 49.1238 ms | not_measured | managed | 1.038x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.04786 ms | not_measured | managed | 0.511x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 2.7787 ms | not_measured | managed | 0.856x |
| `np.random.wald(mean, scale) (float64)` | float64 | 10,000,000 | 27.8582 ms | not_measured | managed | 0.890x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.03661 ms | not_measured | managed | 0.583x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 2.1346 ms | not_measured | managed | 0.972x |
| `np.random.weibull(a) (float64)` | float64 | 10,000,000 | 21.2828 ms | not_measured | managed | 1.014x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.05221 ms | not_measured | managed | 0.643x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 5.4219 ms | not_measured | managed | 0.663x |
| `np.random.zipf(a) (int64)` | int64 | 10,000,000 | 72.1573 ms | not_measured | managed | 0.468x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.00171 ms | not_measured | managed | 1.334x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1308 ms | not_measured | managed | 1.133x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 14.1279 ms | not_measured | managed | 1.161x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.00103 ms | not_measured | managed | 2.697x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4024 ms | not_measured | managed | 1.209x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 49.1851 ms | not_measured | managed | 1.016x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.000707 ms | not_measured | managed | 1.273x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.00826 ms | not_measured | managed | 0.610x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 3.7706 ms | not_measured | managed | 0.442x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.000660 ms | not_measured | managed | 1.398x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.00670 ms | not_measured | managed | 1.372x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 5.4278 ms | not_measured | managed | 0.567x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.000406 ms | not_measured | managed | 2.084x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.00172 ms | not_measured | managed | 1.320x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.7345 ms | not_measured | managed | 0.327x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.000364 ms | not_measured | managed | 2.365x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.00222 ms | not_measured | managed | 1.534x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 4.4089 ms | not_measured | managed | 0.215x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.000810 ms | not_measured | managed | 1.138x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.02732 ms | not_measured | managed | 0.297x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 5.4392 ms | not_measured | managed | 0.570x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.000371 ms | not_measured | managed | 2.299x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.000744 ms | not_measured | managed | 2.122x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.1414 ms | not_measured | managed | 0.764x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.000361 ms | not_measured | managed | 2.363x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.00174 ms | not_measured | managed | 1.302x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 0.7746 ms | not_measured | managed | 0.334x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.000365 ms | not_measured | managed | 2.345x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.00452 ms | not_measured | managed | 0.753x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 1.9634 ms | not_measured | managed | 0.448x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.000454 ms | not_measured | managed | 2.093x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.00999 ms | not_measured | managed | 1.121x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 8.6446 ms | not_measured | managed | 0.378x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.000391 ms | not_measured | managed | 2.174x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.000756 ms | not_measured | managed | 2.075x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.2219 ms | not_measured | managed | 0.486x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.00157 ms | not_measured | managed | 1.448x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1499 ms | not_measured | managed | 0.989x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 13.9550 ms | not_measured | managed | 1.170x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.000968 ms | not_measured | managed | 2.870x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.3078 ms | not_measured | managed | 1.551x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 31.3461 ms | not_measured | managed | 1.639x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.000706 ms | not_measured | managed | 1.276x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.00379 ms | not_measured | managed | 1.330x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 2.9065 ms | not_measured | managed | 0.420x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.000416 ms | not_measured | managed | 2.240x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.01844 ms | not_measured | managed | 0.498x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 5.0341 ms | not_measured | managed | 0.594x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.000559 ms | not_measured | managed | 1.517x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.00259 ms | not_measured | managed | 0.875x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 0.3982 ms | not_measured | managed | 0.640x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.000378 ms | not_measured | managed | 2.267x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.00229 ms | not_measured | managed | 1.486x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 2.9941 ms | not_measured | managed | 0.298x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.000446 ms | not_measured | managed | 2.099x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.02729 ms | not_measured | managed | 0.320x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 8.3641 ms | not_measured | managed | 0.369x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.000583 ms | not_measured | managed | 1.465x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.000845 ms | not_measured | managed | 1.870x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.1465 ms | not_measured | managed | 0.736x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.000892 ms | not_measured | managed | 0.956x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.00249 ms | not_measured | managed | 0.911x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 0.9166 ms | not_measured | managed | 0.287x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.000383 ms | not_measured | managed | 2.261x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.00403 ms | not_measured | managed | 0.846x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 2.7702 ms | not_measured | managed | 0.488x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.000444 ms | not_measured | managed | 2.142x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.01874 ms | not_measured | managed | 0.601x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 7.6728 ms | not_measured | managed | 0.427x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.000371 ms | not_measured | managed | 2.283x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.00114 ms | not_measured | managed | 1.373x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.1338 ms | not_measured | managed | 0.802x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.00108 ms | not_measured | managed | 1.996x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.01804 ms | not_measured | managed | 1.621x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 6.5053 ms | not_measured | managed | 1.373x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.00141 ms | not_measured | managed | 2.853x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.1222 ms | not_measured | managed | 0.804x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 7.9645 ms | not_measured | managed | 1.208x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.000595 ms | not_measured | managed | 5.170x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.00315 ms | not_measured | managed | 5.245x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 1.1070 ms | not_measured | managed | 2.355x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.000412 ms | not_measured | managed | 4.619x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.00686 ms | not_measured | managed | 2.412x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 2.9527 ms | not_measured | managed | 1.469x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.000779 ms | not_measured | managed | 3.142x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.03798 ms | not_measured | managed | 1.342x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 2.0262 ms | not_measured | managed | 2.451x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.000790 ms | not_measured | managed | 2.946x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.03324 ms | not_measured | managed | 1.128x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 4.8201 ms | not_measured | managed | 0.865x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.000596 ms | not_measured | managed | 3.852x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.00978 ms | not_measured | managed | 3.405x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 7.3573 ms | not_measured | managed | 0.775x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.000787 ms | not_measured | managed | 3.104x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.03060 ms | not_measured | managed | 1.658x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 3.9741 ms | not_measured | managed | 1.233x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.000754 ms | not_measured | managed | 3.243x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.02141 ms | not_measured | managed | 2.389x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 1.9929 ms | not_measured | managed | 2.493x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.000538 ms | not_measured | managed | 4.333x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.03927 ms | not_measured | managed | 0.998x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 2.8037 ms | not_measured | managed | 1.745x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.000494 ms | not_measured | managed | 4.913x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.00958 ms | not_measured | managed | 5.050x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 2.8531 ms | not_measured | managed | 2.378x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.000860 ms | not_measured | managed | 2.857x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.01854 ms | not_measured | managed | 2.858x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 1.8522 ms | not_measured | managed | 2.675x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.00421 ms | not_measured | managed | 3.838x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.1809 ms | not_measured | managed | 4.783x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 34.2788 ms | not_measured | managed | 2.605x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.000480 ms | not_measured | managed | 15.617x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.01380 ms | not_measured | managed | 3.193x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 7.7601 ms | not_measured | managed | 2.011x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.000578 ms | not_measured | managed | 10.441x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.03441 ms | not_measured | managed | 1.699x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 16.1406 ms | not_measured | managed | 1.817x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.000461 ms | not_measured | managed | 2.247x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.02134 ms | not_measured | managed | 1.295x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 11.5895 ms | not_measured | managed | 0.668x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.000402 ms | not_measured | managed | 6.701x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.02736 ms | not_measured | managed | 6.876x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 2.5396 ms | not_measured | managed | 7.500x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.000605 ms | not_measured | managed | 1.494x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.00389 ms | not_measured | managed | 3.672x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 3.6859 ms | not_measured | managed | 0.833x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.000662 ms | not_measured | managed | 1.367x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.00785 ms | not_measured | managed | 1.977x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 6.5075 ms | not_measured | managed | 0.684x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.000787 ms | not_measured | managed | 1.535x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.01888 ms | not_measured | managed | 1.688x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 1.9762 ms | not_measured | managed | 1.600x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.000668 ms | not_measured | managed | 1.957x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.02974 ms | not_measured | managed | 1.087x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 3.4697 ms | not_measured | managed | 1.105x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.000667 ms | not_measured | managed | 1.450x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.01039 ms | not_measured | managed | 1.363x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 7.1470 ms | not_measured | managed | 0.547x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.000387 ms | not_measured | managed | 3.147x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.01948 ms | not_measured | managed | 1.679x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 4.7325 ms | not_measured | managed | 0.654x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.00100 ms | not_measured | managed | 1.219x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.01961 ms | not_measured | managed | 1.617x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 2.6481 ms | not_measured | managed | 1.191x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.000599 ms | not_measured | managed | 2.015x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.03950 ms | not_measured | managed | 0.803x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 4.9308 ms | not_measured | managed | 0.890x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.000655 ms | not_measured | managed | 1.475x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.01025 ms | not_measured | managed | 1.317x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 3.2014 ms | not_measured | managed | 1.225x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.000699 ms | not_measured | managed | 1.874x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.01872 ms | not_measured | managed | 1.753x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 2.5974 ms | not_measured | managed | 1.190x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.00229 ms | not_measured | managed | 6.810x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.1864 ms | not_measured | managed | 4.632x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 34.2767 ms | not_measured | managed | 2.599x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.000966 ms | not_measured | managed | 7.258x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.01808 ms | not_measured | managed | 2.416x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 7.1087 ms | not_measured | managed | 2.337x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.000991 ms | not_measured | managed | 5.697x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.03526 ms | not_measured | managed | 1.643x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 16.7861 ms | not_measured | managed | 1.736x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.00159 ms | not_measured | managed | 1.838x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1429 ms | not_measured | managed | 1.043x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 18.4357 ms | not_measured | managed | 0.891x |
| `np.amax (float16)` | float16 | 1,000 | 0.00188 ms | not_measured | managed | 1.794x |
| `np.amax (float16)` | float16 | 100,000 | 0.4127 ms | not_measured | managed | 1.181x |
| `np.amax (float16)` | float16 | 10,000,000 | 48.7957 ms | not_measured | managed | 1.024x |
| `np.amax (float32)` | float32 | 1,000 | 0.000726 ms | not_measured | managed | 2.070x |
| `np.amax (float32)` | float32 | 100,000 | 0.00354 ms | not_measured | managed | 1.620x |
| `np.amax (float32)` | float32 | 10,000,000 | 3.9115 ms | not_measured | managed | 0.369x |
| `np.amax (float64)` | float64 | 1,000 | 0.000695 ms | not_measured | managed | 2.312x |
| `np.amax (float64)` | float64 | 100,000 | 0.00666 ms | not_measured | managed | 1.484x |
| `np.amax (float64)` | float64 | 10,000,000 | 4.4746 ms | not_measured | managed | 0.678x |
| `np.amax (int16)` | int16 | 1,000 | 0.000492 ms | not_measured | managed | 3.075x |
| `np.amax (int16)` | int16 | 100,000 | 0.00258 ms | not_measured | managed | 1.149x |
| `np.amax (int16)` | int16 | 10,000,000 | 0.4378 ms | not_measured | managed | 0.558x |
| `np.amax (int32)` | int32 | 1,000 | 0.000529 ms | not_measured | managed | 2.892x |
| `np.amax (int32)` | int32 | 100,000 | 0.00224 ms | not_measured | managed | 1.816x |
| `np.amax (int32)` | int32 | 10,000,000 | 4.4009 ms | not_measured | managed | 0.269x |
| `np.amax (int64)` | int64 | 1,000 | 0.000855 ms | not_measured | managed | 1.860x |
| `np.amax (int64)` | int64 | 100,000 | 0.02714 ms | not_measured | managed | 0.325x |
| `np.amax (int64)` | int64 | 10,000,000 | 8.1774 ms | not_measured | managed | 0.384x |
| `np.amax (int8)` | int8 | 1,000 | 0.000365 ms | not_measured | managed | 4.164x |
| `np.amax (int8)` | int8 | 100,000 | 0.000778 ms | not_measured | managed | 2.909x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2257 ms | not_measured | managed | 0.484x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.000664 ms | not_measured | managed | 2.279x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.00175 ms | not_measured | managed | 1.659x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 0.6933 ms | not_measured | managed | 0.400x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.000602 ms | not_measured | managed | 2.565x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.00388 ms | not_measured | managed | 1.054x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 3.6372 ms | not_measured | managed | 0.232x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.000729 ms | not_measured | managed | 2.210x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.01010 ms | not_measured | managed | 1.178x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 8.6240 ms | not_measured | managed | 0.387x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.000361 ms | not_measured | managed | 4.208x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.000967 ms | not_measured | managed | 2.329x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.1344 ms | not_measured | managed | 0.815x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.00374 ms | not_measured | managed | 0.770x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1656 ms | not_measured | managed | 0.827x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 17.0548 ms | not_measured | managed | 0.902x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.00178 ms | not_measured | managed | 2.038x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6528 ms | not_measured | managed | 0.747x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 136.0526 ms | not_measured | managed | 0.359x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.000492 ms | not_measured | managed | 4.000x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.01060 ms | not_measured | managed | 0.862x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 4.5382 ms | not_measured | managed | 0.350x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.000627 ms | not_measured | managed | 3.190x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.01767 ms | not_measured | managed | 0.804x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 7.7550 ms | not_measured | managed | 0.451x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.000602 ms | not_measured | managed | 3.110x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.00344 ms | not_measured | managed | 1.746x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 0.9945 ms | not_measured | managed | 0.334x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.000506 ms | not_measured | managed | 3.733x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.00936 ms | not_measured | managed | 0.891x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 4.3197 ms | not_measured | managed | 0.412x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.000972 ms | not_measured | managed | 1.968x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.01208 ms | not_measured | managed | 1.161x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 8.3986 ms | not_measured | managed | 0.433x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.000450 ms | not_measured | managed | 4.151x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.00356 ms | not_measured | managed | 2.058x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3516 ms | not_measured | managed | 0.519x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.000395 ms | not_measured | managed | 4.628x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.00406 ms | not_measured | managed | 1.513x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 1.5323 ms | not_measured | managed | 0.254x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.000464 ms | not_measured | managed | 4.084x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.00542 ms | not_measured | managed | 1.544x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 3.9071 ms | not_measured | managed | 0.354x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.000435 ms | not_measured | managed | 4.538x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.01388 ms | not_measured | managed | 1.189x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 9.5461 ms | not_measured | managed | 0.417x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.000480 ms | not_measured | managed | 3.881x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.00669 ms | not_measured | managed | 1.157x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.3489 ms | not_measured | managed | 0.523x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.00156 ms | not_measured | managed | 1.874x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1161 ms | not_measured | managed | 1.267x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 13.9433 ms | not_measured | managed | 1.184x |
| `np.amin (float16)` | float16 | 1,000 | 0.00114 ms | not_measured | managed | 3.157x |
| `np.amin (float16)` | float16 | 100,000 | 0.2992 ms | not_measured | managed | 1.606x |
| `np.amin (float16)` | float16 | 10,000,000 | 31.3699 ms | not_measured | managed | 1.641x |
| `np.amin (float32)` | float32 | 1,000 | 0.000694 ms | not_measured | managed | 2.269x |
| `np.amin (float32)` | float32 | 100,000 | 0.00791 ms | not_measured | managed | 0.719x |
| `np.amin (float32)` | float32 | 10,000,000 | 1.8966 ms | not_measured | managed | 0.649x |
| `np.amin (float64)` | float64 | 1,000 | 0.000376 ms | not_measured | managed | 4.282x |
| `np.amin (float64)` | float64 | 100,000 | 0.01318 ms | not_measured | managed | 0.750x |
| `np.amin (float64)` | float64 | 10,000,000 | 8.4627 ms | not_measured | managed | 0.360x |
| `np.amin (int16)` | int16 | 1,000 | 0.000527 ms | not_measured | managed | 2.886x |
| `np.amin (int16)` | int16 | 100,000 | 0.00261 ms | not_measured | managed | 1.138x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.3705 ms | not_measured | managed | 0.735x |
| `np.amin (int32)` | int32 | 1,000 | 0.000557 ms | not_measured | managed | 2.752x |
| `np.amin (int32)` | int32 | 100,000 | 0.00237 ms | not_measured | managed | 1.729x |
| `np.amin (int32)` | int32 | 10,000,000 | 3.8497 ms | not_measured | managed | 0.222x |
| `np.amin (int64)` | int64 | 1,000 | 0.000435 ms | not_measured | managed | 3.589x |
| `np.amin (int64)` | int64 | 100,000 | 0.02752 ms | not_measured | managed | 0.338x |
| `np.amin (int64)` | int64 | 10,000,000 | 3.7873 ms | not_measured | managed | 0.833x |
| `np.amin (int8)` | int8 | 1,000 | 0.000603 ms | not_measured | managed | 2.517x |
| `np.amin (int8)` | int8 | 100,000 | 0.000738 ms | not_measured | managed | 3.069x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.2276 ms | not_measured | managed | 0.474x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.000417 ms | not_measured | managed | 3.640x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.00248 ms | not_measured | managed | 1.198x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 1.0291 ms | not_measured | managed | 0.262x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.000655 ms | not_measured | managed | 2.333x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.00370 ms | not_measured | managed | 1.106x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 3.7504 ms | not_measured | managed | 0.415x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.000511 ms | not_measured | managed | 3.170x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.01996 ms | not_measured | managed | 0.597x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 8.3657 ms | not_measured | managed | 0.389x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.000946 ms | not_measured | managed | 1.605x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.00140 ms | not_measured | managed | 1.744x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.1348 ms | not_measured | managed | 0.814x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.00340 ms | not_measured | managed | 0.846x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.2078 ms | not_measured | managed | 0.661x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 22.7295 ms | not_measured | managed | 0.673x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.00178 ms | not_measured | managed | 2.148x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.5200 ms | not_measured | managed | 0.888x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 97.4029 ms | not_measured | managed | 0.475x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.000498 ms | not_measured | managed | 3.938x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.01049 ms | not_measured | managed | 0.874x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 2.9930 ms | not_measured | managed | 0.544x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.000438 ms | not_measured | managed | 4.575x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.01697 ms | not_measured | managed | 0.834x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 7.1635 ms | not_measured | managed | 0.487x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.000386 ms | not_measured | managed | 4.886x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.00681 ms | not_measured | managed | 0.930x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 1.0910 ms | not_measured | managed | 0.305x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.000500 ms | not_measured | managed | 3.774x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.00702 ms | not_measured | managed | 1.180x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 3.9021 ms | not_measured | managed | 0.439x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.000791 ms | not_measured | managed | 2.473x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.02774 ms | not_measured | managed | 0.544x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 8.4142 ms | not_measured | managed | 0.444x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.000559 ms | not_measured | managed | 3.363x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.00366 ms | not_measured | managed | 1.885x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.1773 ms | not_measured | managed | 1.016x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.000470 ms | not_measured | managed | 4.000x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.00525 ms | not_measured | managed | 1.162x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 0.8682 ms | not_measured | managed | 0.448x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.000371 ms | not_measured | managed | 5.105x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.00850 ms | not_measured | managed | 0.982x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 4.2958 ms | not_measured | managed | 0.293x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.000406 ms | not_measured | managed | 4.828x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.03724 ms | not_measured | managed | 0.445x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 5.7679 ms | not_measured | managed | 0.696x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.000446 ms | not_measured | managed | 4.188x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.00794 ms | not_measured | managed | 0.852x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.1690 ms | not_measured | managed | 1.068x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.00149 ms | not_measured | managed | 1.253x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1141 ms | not_measured | managed | 0.968x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 13.6593 ms | not_measured | managed | 1.017x |
| `np.argmax (float16)` | float16 | 1,000 | 0.00178 ms | not_measured | managed | 1.389x |
| `np.argmax (float16)` | float16 | 100,000 | 0.1406 ms | not_measured | managed | 3.011x |
| `np.argmax (float16)` | float16 | 10,000,000 | 14.0835 ms | not_measured | managed | 3.070x |
| `np.argmax (float32)` | float32 | 1,000 | 0.000509 ms | not_measured | managed | 1.607x |
| `np.argmax (float32)` | float32 | 100,000 | 0.00799 ms | not_measured | managed | 1.049x |
| `np.argmax (float32)` | float32 | 10,000,000 | 2.0397 ms | not_measured | managed | 0.902x |
| `np.argmax (float64)` | float64 | 1,000 | 0.000458 ms | not_measured | managed | 1.989x |
| `np.argmax (float64)` | float64 | 100,000 | 0.01572 ms | not_measured | managed | 1.023x |
| `np.argmax (float64)` | float64 | 10,000,000 | 4.7438 ms | not_measured | managed | 0.863x |
| `np.argmax (int16)` | int16 | 1,000 | 0.000459 ms | not_measured | managed | 1.732x |
| `np.argmax (int16)` | int16 | 100,000 | 0.00186 ms | not_measured | managed | 1.742x |
| `np.argmax (int16)` | int16 | 10,000,000 | 0.3657 ms | not_measured | managed | 0.908x |
| `np.argmax (int32)` | int32 | 1,000 | 0.000489 ms | not_measured | managed | 1.562x |
| `np.argmax (int32)` | int32 | 100,000 | 0.00261 ms | not_measured | managed | 2.040x |
| `np.argmax (int32)` | int32 | 10,000,000 | 1.3198 ms | not_measured | managed | 0.982x |
| `np.argmax (int64)` | int64 | 1,000 | 0.000613 ms | not_measured | managed | 1.437x |
| `np.argmax (int64)` | int64 | 100,000 | 0.02818 ms | not_measured | managed | 0.495x |
| `np.argmax (int64)` | int64 | 10,000,000 | 4.6457 ms | not_measured | managed | 0.840x |
| `np.argmax (int8)` | int8 | 1,000 | 0.000462 ms | not_measured | managed | 1.786x |
| `np.argmax (int8)` | int8 | 100,000 | 0.000788 ms | not_measured | managed | 2.657x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.1462 ms | not_measured | managed | 1.000x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.000471 ms | not_measured | managed | 1.735x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.00185 ms | not_measured | managed | 2.622x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 0.3727 ms | not_measured | managed | 1.867x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.000493 ms | not_measured | managed | 1.680x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.00349 ms | not_measured | managed | 2.463x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 1.3017 ms | not_measured | managed | 1.298x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.000657 ms | not_measured | managed | 1.388x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.03304 ms | not_measured | managed | 0.507x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 5.0310 ms | not_measured | managed | 0.778x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.000475 ms | not_measured | managed | 1.800x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.000785 ms | not_measured | managed | 3.704x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.1412 ms | not_measured | managed | 1.505x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.00210 ms | not_measured | managed | 0.878x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1151 ms | not_measured | managed | 0.965x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 14.2740 ms | not_measured | managed | 0.978x |
| `np.argmin (float16)` | float16 | 1,000 | 0.00277 ms | not_measured | managed | 0.923x |
| `np.argmin (float16)` | float16 | 100,000 | 0.1411 ms | not_measured | managed | 2.824x |
| `np.argmin (float16)` | float16 | 10,000,000 | 14.2493 ms | not_measured | managed | 2.935x |
| `np.argmin (float32)` | float32 | 1,000 | 0.000433 ms | not_measured | managed | 1.905x |
| `np.argmin (float32)` | float32 | 100,000 | 0.00815 ms | not_measured | managed | 1.028x |
| `np.argmin (float32)` | float32 | 10,000,000 | 2.2334 ms | not_measured | managed | 0.804x |
| `np.argmin (float64)` | float64 | 1,000 | 0.000564 ms | not_measured | managed | 1.621x |
| `np.argmin (float64)` | float64 | 100,000 | 0.01660 ms | not_measured | managed | 0.969x |
| `np.argmin (float64)` | float64 | 10,000,000 | 4.2877 ms | not_measured | managed | 0.980x |
| `np.argmin (int16)` | int16 | 1,000 | 0.000483 ms | not_measured | managed | 1.656x |
| `np.argmin (int16)` | int16 | 100,000 | 0.00187 ms | not_measured | managed | 1.735x |
| `np.argmin (int16)` | int16 | 10,000,000 | 0.4119 ms | not_measured | managed | 0.815x |
| `np.argmin (int32)` | int32 | 1,000 | 0.000373 ms | not_measured | managed | 2.142x |
| `np.argmin (int32)` | int32 | 100,000 | 0.00259 ms | not_measured | managed | 2.035x |
| `np.argmin (int32)` | int32 | 10,000,000 | 1.7289 ms | not_measured | managed | 0.782x |
| `np.argmin (int64)` | int64 | 1,000 | 0.000689 ms | not_measured | managed | 1.274x |
| `np.argmin (int64)` | int64 | 100,000 | 0.02859 ms | not_measured | managed | 0.486x |
| `np.argmin (int64)` | int64 | 10,000,000 | 4.6821 ms | not_measured | managed | 0.787x |
| `np.argmin (int8)` | int8 | 1,000 | 0.000318 ms | not_measured | managed | 2.610x |
| `np.argmin (int8)` | int8 | 100,000 | 0.000836 ms | not_measured | managed | 2.511x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.1492 ms | not_measured | managed | 0.994x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.000357 ms | not_measured | managed | 2.283x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.00185 ms | not_measured | managed | 2.623x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 0.3821 ms | not_measured | managed | 3.408x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.000521 ms | not_measured | managed | 1.583x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.00350 ms | not_measured | managed | 2.452x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 1.4185 ms | not_measured | managed | 1.465x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.000531 ms | not_measured | managed | 1.704x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.05974 ms | not_measured | managed | 0.280x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 5.2184 ms | not_measured | managed | 0.747x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.000378 ms | not_measured | managed | 2.183x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.000803 ms | not_measured | managed | 3.667x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.1354 ms | not_measured | managed | 1.573x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.01209 ms | not_measured | managed | 1.162x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 0.9540 ms | not_measured | managed | 0.390x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 144.0882 ms | not_measured | managed | 0.261x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.01818 ms | not_measured | managed | 1.041x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 2.3800 ms | not_measured | managed | 1.033x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 417.3613 ms | not_measured | managed | 0.596x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.00266 ms | not_measured | managed | 1.030x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 3.6987 ms | not_measured | managed | 0.639x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 598.5392 ms | not_measured | managed | 0.424x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.00248 ms | not_measured | managed | 1.127x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.2798 ms | not_measured | managed | 1.295x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 32.4457 ms | not_measured | managed | 1.058x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.01527 ms | not_measured | managed | 0.377x |
| `np.cumsum (float16)` | float16 | 100,000 | 0.9813 ms | not_measured | managed | 0.469x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 151.6401 ms | not_measured | managed | 0.322x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.00178 ms | not_measured | managed | 1.467x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.08649 ms | not_measured | managed | 1.824x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 7.1094 ms | not_measured | managed | 2.764x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.00225 ms | not_measured | managed | 1.217x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.1481 ms | not_measured | managed | 1.098x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 19.2071 ms | not_measured | managed | 1.250x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.00202 ms | not_measured | managed | 1.072x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.1399 ms | not_measured | managed | 2.286x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 13.8658 ms | not_measured | managed | 1.947x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.00166 ms | not_measured | managed | 1.310x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1333 ms | not_measured | managed | 2.275x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 13.7240 ms | not_measured | managed | 2.033x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.00183 ms | not_measured | managed | 0.842x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1578 ms | not_measured | managed | 0.191x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 18.2308 ms | not_measured | managed | 0.805x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.00172 ms | not_measured | managed | 1.264x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.1347 ms | not_measured | managed | 2.395x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 12.3552 ms | not_measured | managed | 2.145x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.00200 ms | not_measured | managed | 1.080x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.1402 ms | not_measured | managed | 2.187x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 13.7339 ms | not_measured | managed | 2.050x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.00191 ms | not_measured | managed | 1.123x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1237 ms | not_measured | managed | 2.523x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 15.6286 ms | not_measured | managed | 1.799x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.00184 ms | not_measured | managed | 0.840x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1465 ms | not_measured | managed | 0.207x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 15.0252 ms | not_measured | managed | 0.981x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.00135 ms | not_measured | managed | 1.591x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.1232 ms | not_measured | managed | 2.490x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 11.0485 ms | not_measured | managed | 2.422x |
| `np.max (complex128)` | complex128 | 1,000 | 0.00159 ms | not_measured | managed | 1.849x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1449 ms | not_measured | managed | 1.011x |
| `np.max (complex128)` | complex128 | 10,000,000 | 14.2775 ms | not_measured | managed | 1.178x |
| `np.max (float16)` | float16 | 1,000 | 0.00104 ms | not_measured | managed | 3.283x |
| `np.max (float16)` | float16 | 100,000 | 0.3221 ms | not_measured | managed | 1.511x |
| `np.max (float16)` | float16 | 10,000,000 | 39.8807 ms | not_measured | managed | 1.255x |
| `np.max (float32)` | float32 | 1,000 | 0.000410 ms | not_measured | managed | 3.793x |
| `np.max (float32)` | float32 | 100,000 | 0.00358 ms | not_measured | managed | 1.598x |
| `np.max (float32)` | float32 | 10,000,000 | 1.9272 ms | not_measured | managed | 0.768x |
| `np.max (float64)` | float64 | 1,000 | 0.000419 ms | not_measured | managed | 3.840x |
| `np.max (float64)` | float64 | 100,000 | 0.00778 ms | not_measured | managed | 1.271x |
| `np.max (float64)` | float64 | 10,000,000 | 3.2534 ms | not_measured | managed | 0.981x |
| `np.max (int16)` | int16 | 1,000 | 0.000364 ms | not_measured | managed | 4.159x |
| `np.max (int16)` | int16 | 100,000 | 0.00178 ms | not_measured | managed | 1.662x |
| `np.max (int16)` | int16 | 10,000,000 | 0.6539 ms | not_measured | managed | 0.377x |
| `np.max (int32)` | int32 | 1,000 | 0.000371 ms | not_measured | managed | 4.105x |
| `np.max (int32)` | int32 | 100,000 | 0.00224 ms | not_measured | managed | 1.821x |
| `np.max (int32)` | int32 | 10,000,000 | 1.1639 ms | not_measured | managed | 0.754x |
| `np.max (int64)` | int64 | 1,000 | 0.000450 ms | not_measured | managed | 3.531x |
| `np.max (int64)` | int64 | 100,000 | 0.00751 ms | not_measured | managed | 1.177x |
| `np.max (int64)` | int64 | 10,000,000 | 6.1263 ms | not_measured | managed | 0.499x |
| `np.max (int8)` | int8 | 1,000 | 0.000397 ms | not_measured | managed | 3.821x |
| `np.max (int8)` | int8 | 100,000 | 0.000766 ms | not_measured | managed | 2.963x |
| `np.max (int8)` | int8 | 10,000,000 | 0.1506 ms | not_measured | managed | 0.724x |
| `np.max (uint16)` | uint16 | 1,000 | 0.000377 ms | not_measured | managed | 4.000x |
| `np.max (uint16)` | uint16 | 100,000 | 0.00233 ms | not_measured | managed | 1.268x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.3480 ms | not_measured | managed | 1.942x |
| `np.max (uint32)` | uint32 | 1,000 | 0.000393 ms | not_measured | managed | 3.880x |
| `np.max (uint32)` | uint32 | 100,000 | 0.00326 ms | not_measured | managed | 1.256x |
| `np.max (uint32)` | uint32 | 10,000,000 | 1.3878 ms | not_measured | managed | 0.679x |
| `np.max (uint64)` | uint64 | 1,000 | 0.000453 ms | not_measured | managed | 3.563x |
| `np.max (uint64)` | uint64 | 100,000 | 0.00982 ms | not_measured | managed | 1.209x |
| `np.max (uint64)` | uint64 | 10,000,000 | 3.9754 ms | not_measured | managed | 0.809x |
| `np.max (uint8)` | uint8 | 1,000 | 0.000609 ms | not_measured | managed | 2.481x |
| `np.max (uint8)` | uint8 | 100,000 | 0.000769 ms | not_measured | managed | 2.935x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.1358 ms | not_measured | managed | 0.804x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.000986 ms | not_measured | managed | 2.605x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.01051 ms | not_measured | managed | 2.871x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 6.5361 ms | not_measured | managed | 1.362x |
| `np.mean (float16)` | float16 | 1,000 | 0.00171 ms | not_measured | managed | 2.587x |
| `np.mean (float16)` | float16 | 100,000 | 0.09912 ms | not_measured | managed | 0.965x |
| `np.mean (float16)` | float16 | 10,000,000 | 8.0019 ms | not_measured | managed | 1.203x |
| `np.mean (float32)` | float32 | 1,000 | 0.000580 ms | not_measured | managed | 6.026x |
| `np.mean (float32)` | float32 | 100,000 | 0.00446 ms | not_measured | managed | 3.803x |
| `np.mean (float32)` | float32 | 10,000,000 | 1.2552 ms | not_measured | managed | 2.111x |
| `np.mean (float64)` | float64 | 1,000 | 0.000590 ms | not_measured | managed | 3.846x |
| `np.mean (float64)` | float64 | 100,000 | 0.00873 ms | not_measured | managed | 1.941x |
| `np.mean (float64)` | float64 | 10,000,000 | 3.0180 ms | not_measured | managed | 1.442x |
| `np.mean (int16)` | int16 | 1,000 | 0.000563 ms | not_measured | managed | 5.052x |
| `np.mean (int16)` | int16 | 100,000 | 0.03873 ms | not_measured | managed | 1.328x |
| `np.mean (int16)` | int16 | 10,000,000 | 1.9887 ms | not_measured | managed | 2.490x |
| `np.mean (int32)` | int32 | 1,000 | 0.000492 ms | not_measured | managed | 5.494x |
| `np.mean (int32)` | int32 | 100,000 | 0.03208 ms | not_measured | managed | 1.195x |
| `np.mean (int32)` | int32 | 10,000,000 | 5.1451 ms | not_measured | managed | 0.853x |
| `np.mean (int64)` | int64 | 1,000 | 0.000507 ms | not_measured | managed | 5.292x |
| `np.mean (int64)` | int64 | 100,000 | 0.00898 ms | not_measured | managed | 3.762x |
| `np.mean (int64)` | int64 | 10,000,000 | 2.7771 ms | not_measured | managed | 2.052x |
| `np.mean (int8)` | int8 | 1,000 | 0.00105 ms | not_measured | managed | 2.700x |
| `np.mean (int8)` | int8 | 100,000 | 0.02815 ms | not_measured | managed | 1.824x |
| `np.mean (int8)` | int8 | 10,000,000 | 2.9668 ms | not_measured | managed | 1.651x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.000427 ms | not_measured | managed | 6.649x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.02721 ms | not_measured | managed | 1.892x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 1.9766 ms | not_measured | managed | 2.526x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.000558 ms | not_measured | managed | 4.892x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.03417 ms | not_measured | managed | 1.163x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 4.5407 ms | not_measured | managed | 1.098x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.000494 ms | not_measured | managed | 5.700x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.01080 ms | not_measured | managed | 4.513x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 3.0277 ms | not_measured | managed | 2.253x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.000877 ms | not_measured | managed | 3.249x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.01855 ms | not_measured | managed | 2.771x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 1.8771 ms | not_measured | managed | 2.637x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.00309 ms | not_measured | managed | 0.973x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.02415 ms | not_measured | managed | 0.863x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 16.4463 ms | not_measured | managed | 0.417x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.00642 ms | not_measured | managed | 0.738x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.1488 ms | not_measured | managed | 0.503x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 25.0443 ms | not_measured | managed | 0.284x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.00228 ms | not_measured | managed | 1.490x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.00899 ms | not_measured | managed | 1.001x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 2.6752 ms | not_measured | managed | 0.468x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.00282 ms | not_measured | managed | 1.016x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.01551 ms | not_measured | managed | 0.810x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 5.5327 ms | not_measured | managed | 0.553x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.000479 ms | not_measured | managed | 7.050x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.00829 ms | not_measured | managed | 5.980x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 1.2082 ms | not_measured | managed | 4.192x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.000428 ms | not_measured | managed | 7.612x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.01084 ms | not_measured | managed | 3.243x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 2.1044 ms | not_measured | managed | 2.136x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.000381 ms | not_measured | managed | 8.491x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.01212 ms | not_measured | managed | 2.591x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 4.6947 ms | not_measured | managed | 1.226x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.000430 ms | not_measured | managed | 7.858x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.01880 ms | not_measured | managed | 2.628x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 0.7824 ms | not_measured | managed | 6.555x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.000453 ms | not_measured | managed | 7.481x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.01483 ms | not_measured | managed | 3.333x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 1.1197 ms | not_measured | managed | 4.616x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.000362 ms | not_measured | managed | 9.099x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.00935 ms | not_measured | managed | 4.030x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 3.7969 ms | not_measured | managed | 1.164x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.000462 ms | not_measured | managed | 7.288x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.01197 ms | not_measured | managed | 3.830x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 7.5861 ms | not_measured | managed | 0.900x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.000620 ms | not_measured | managed | 5.442x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.01887 ms | not_measured | managed | 2.627x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 0.7914 ms | not_measured | managed | 6.452x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.00242 ms | not_measured | managed | 1.231x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.04667 ms | not_measured | managed | 0.693x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 16.8366 ms | not_measured | managed | 0.470x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.00443 ms | not_measured | managed | 1.060x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.1383 ms | not_measured | managed | 0.597x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 25.8742 ms | not_measured | managed | 0.303x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.00250 ms | not_measured | managed | 1.343x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.01252 ms | not_measured | managed | 1.412x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 2.1537 ms | not_measured | managed | 1.347x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.00282 ms | not_measured | managed | 1.005x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.01493 ms | not_measured | managed | 1.236x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 5.6107 ms | not_measured | managed | 0.878x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.000364 ms | not_measured | managed | 9.187x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.01493 ms | not_measured | managed | 3.724x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 1.1044 ms | not_measured | managed | 4.608x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.000314 ms | not_measured | managed | 10.268x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.00843 ms | not_measured | managed | 4.891x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 1.5806 ms | not_measured | managed | 2.852x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.00112 ms | not_measured | managed | 2.849x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.05996 ms | not_measured | managed | 0.628x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 8.8012 ms | not_measured | managed | 0.676x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.000478 ms | not_measured | managed | 7.023x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.00794 ms | not_measured | managed | 7.001x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 0.7207 ms | not_measured | managed | 7.108x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.000364 ms | not_measured | managed | 9.181x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.01811 ms | not_measured | managed | 3.069x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 1.1551 ms | not_measured | managed | 4.458x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.000511 ms | not_measured | managed | 6.403x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.00897 ms | not_measured | managed | 4.887x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 2.0312 ms | not_measured | managed | 2.250x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.00226 ms | not_measured | managed | 1.469x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.1412 ms | not_measured | managed | 0.367x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 14.9540 ms | not_measured | managed | 0.464x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.000693 ms | not_measured | managed | 4.837x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.00777 ms | not_measured | managed | 7.169x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 0.7298 ms | not_measured | managed | 6.843x |
| `np.min (complex128)` | complex128 | 1,000 | 0.00180 ms | not_measured | managed | 1.661x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1449 ms | not_measured | managed | 1.029x |
| `np.min (complex128)` | complex128 | 10,000,000 | 14.1701 ms | not_measured | managed | 1.138x |
| `np.min (float16)` | float16 | 1,000 | 0.000974 ms | not_measured | managed | 3.682x |
| `np.min (float16)` | float16 | 100,000 | 0.4046 ms | not_measured | managed | 1.187x |
| `np.min (float16)` | float16 | 10,000,000 | 31.4790 ms | not_measured | managed | 1.634x |
| `np.min (float32)` | float32 | 1,000 | 0.000666 ms | not_measured | managed | 2.353x |
| `np.min (float32)` | float32 | 100,000 | 0.00599 ms | not_measured | managed | 0.958x |
| `np.min (float32)` | float32 | 10,000,000 | 1.4520 ms | not_measured | managed | 0.849x |
| `np.min (float64)` | float64 | 1,000 | 0.000793 ms | not_measured | managed | 2.033x |
| `np.min (float64)` | float64 | 100,000 | 0.01900 ms | not_measured | managed | 0.520x |
| `np.min (float64)` | float64 | 10,000,000 | 3.3060 ms | not_measured | managed | 0.916x |
| `np.min (int16)` | int16 | 1,000 | 0.000450 ms | not_measured | managed | 3.367x |
| `np.min (int16)` | int16 | 100,000 | 0.00319 ms | not_measured | managed | 0.929x |
| `np.min (int16)` | int16 | 10,000,000 | 0.3710 ms | not_measured | managed | 0.678x |
| `np.min (int32)` | int32 | 1,000 | 0.000692 ms | not_measured | managed | 2.214x |
| `np.min (int32)` | int32 | 100,000 | 0.00221 ms | not_measured | managed | 1.852x |
| `np.min (int32)` | int32 | 10,000,000 | 3.1489 ms | not_measured | managed | 0.404x |
| `np.min (int64)` | int64 | 1,000 | 0.000986 ms | not_measured | managed | 1.626x |
| `np.min (int64)` | int64 | 100,000 | 0.00910 ms | not_measured | managed | 1.002x |
| `np.min (int64)` | int64 | 10,000,000 | 4.6536 ms | not_measured | managed | 0.650x |
| `np.min (int8)` | int8 | 1,000 | 0.000860 ms | not_measured | managed | 1.762x |
| `np.min (int8)` | int8 | 100,000 | 0.000757 ms | not_measured | managed | 2.980x |
| `np.min (int8)` | int8 | 10,000,000 | 0.2163 ms | not_measured | managed | 0.502x |
| `np.min (uint16)` | uint16 | 1,000 | 0.000610 ms | not_measured | managed | 2.482x |
| `np.min (uint16)` | uint16 | 100,000 | 0.00172 ms | not_measured | managed | 1.717x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.6313 ms | not_measured | managed | 0.410x |
| `np.min (uint32)` | uint32 | 1,000 | 0.000621 ms | not_measured | managed | 2.385x |
| `np.min (uint32)` | uint32 | 100,000 | 0.00324 ms | not_measured | managed | 1.265x |
| `np.min (uint32)` | uint32 | 10,000,000 | 1.5940 ms | not_measured | managed | 0.614x |
| `np.min (uint64)` | uint64 | 1,000 | 0.000727 ms | not_measured | managed | 2.215x |
| `np.min (uint64)` | uint64 | 100,000 | 0.03776 ms | not_measured | managed | 0.318x |
| `np.min (uint64)` | uint64 | 10,000,000 | 5.1811 ms | not_measured | managed | 0.636x |
| `np.min (uint8)` | uint8 | 1,000 | 0.000745 ms | not_measured | managed | 1.966x |
| `np.min (uint8)` | uint8 | 100,000 | 0.00152 ms | not_measured | managed | 1.506x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.1401 ms | not_measured | managed | 0.778x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.00265 ms | not_measured | managed | 3.198x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.1667 ms | not_measured | managed | 3.024x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 20.3242 ms | not_measured | managed | 2.701x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.00102 ms | not_measured | managed | 5.498x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.03628 ms | not_measured | managed | 0.771x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 6.2630 ms | not_measured | managed | 1.893x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.00105 ms | not_measured | managed | 5.394x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.02635 ms | not_measured | managed | 1.746x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 9.0224 ms | not_measured | managed | 2.487x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.00271 ms | not_measured | managed | 3.198x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.1671 ms | not_measured | managed | 2.886x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 28.0301 ms | not_measured | managed | 1.901x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.000939 ms | not_measured | managed | 6.023x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.01468 ms | not_measured | managed | 1.956x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 9.1722 ms | not_measured | managed | 1.327x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.00107 ms | not_measured | managed | 5.577x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.05616 ms | not_measured | managed | 0.818x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 22.0142 ms | not_measured | managed | 1.018x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.000936 ms | not_measured | managed | 5.120x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.4324 ms | not_measured | managed | 1.133x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 41.8284 ms | not_measured | managed | 1.192x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.000453 ms | not_measured | managed | 6.241x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.00314 ms | not_measured | managed | 2.212x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 1.2223 ms | not_measured | managed | 1.059x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.000379 ms | not_measured | managed | 7.533x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.00719 ms | not_measured | managed | 1.569x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 2.9812 ms | not_measured | managed | 1.061x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.00145 ms | not_measured | managed | 8.332x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.1197 ms | not_measured | managed | 2.633x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 17.9085 ms | not_measured | managed | 2.003x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.00133 ms | not_measured | managed | 7.139x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.03842 ms | not_measured | managed | 1.892x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 4.2267 ms | not_measured | managed | 4.495x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.000737 ms | not_measured | managed | 11.342x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.03814 ms | not_measured | managed | 2.254x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 5.6905 ms | not_measured | managed | 4.978x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.00651 ms | not_measured | managed | 2.457x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.1595 ms | not_measured | managed | 0.799x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 128.3326 ms | not_measured | managed | 0.885x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.00265 ms | not_measured | managed | 4.711x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 0.3541 ms | not_measured | managed | 1.329x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 33.6419 ms | not_measured | managed | 2.266x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.00273 ms | not_measured | managed | 4.134x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.3479 ms | not_measured | managed | 1.363x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 38.0239 ms | not_measured | managed | 2.334x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.000926 ms | not_measured | managed | 4.951x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.3029 ms | not_measured | managed | 1.603x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 39.2082 ms | not_measured | managed | 1.264x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.000434 ms | not_measured | managed | 6.256x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.00317 ms | not_measured | managed | 2.192x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 1.1826 ms | not_measured | managed | 1.139x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.000374 ms | not_measured | managed | 7.666x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.00395 ms | not_measured | managed | 2.851x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 7.4272 ms | not_measured | managed | 0.432x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.00648 ms | not_measured | managed | 4.680x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.1621 ms | not_measured | managed | 1.538x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 126.5264 ms | not_measured | managed | 0.944x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.00265 ms | not_measured | managed | 9.580x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.3330 ms | not_measured | managed | 2.121x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 46.6887 ms | not_measured | managed | 1.078x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.00273 ms | not_measured | managed | 9.272x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.2454 ms | not_measured | managed | 2.858x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 54.1758 ms | not_measured | managed | 1.144x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.00112 ms | not_measured | managed | 19.105x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.09634 ms | not_measured | managed | 2.210x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 19.1235 ms | not_measured | managed | 1.027x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.000837 ms | not_measured | managed | 24.381x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 0.3018 ms | not_measured | managed | 7.842x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 30.0553 ms | not_measured | managed | 8.194x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.000588 ms | not_measured | managed | 7.027x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 0.5547 ms | not_measured | managed | 4.192x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 104.6642 ms | not_measured | managed | 2.443x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.00629 ms | not_measured | managed | 4.718x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.1607 ms | not_measured | managed | 1.553x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 118.0380 ms | not_measured | managed | 1.011x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00289 ms | not_measured | managed | 8.260x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2158 ms | not_measured | managed | 3.273x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 30.4668 ms | not_measured | managed | 1.654x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00308 ms | not_measured | managed | 7.935x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3570 ms | not_measured | managed | 1.948x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 44.6134 ms | not_measured | managed | 1.390x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.00272 ms | not_measured | managed | 10.789x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.2150 ms | not_measured | managed | 5.177x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 40.2767 ms | not_measured | managed | 2.924x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.00263 ms | not_measured | managed | 7.359x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.09080 ms | not_measured | managed | 1.785x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 9.3218 ms | not_measured | managed | 3.318x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.00130 ms | not_measured | managed | 13.667x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.08212 ms | not_measured | managed | 2.222x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 18.6469 ms | not_measured | managed | 2.506x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.00112 ms | not_measured | managed | 5.334x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.1770 ms | not_measured | managed | 1.512x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 18.9144 ms | not_measured | managed | 1.614x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.000415 ms | not_measured | managed | 8.451x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.00494 ms | not_measured | managed | 6.178x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 1.4828 ms | not_measured | managed | 8.797x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.000403 ms | not_measured | managed | 8.635x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.01834 ms | not_measured | managed | 2.311x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 4.4631 ms | not_measured | managed | 5.091x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.00272 ms | not_measured | managed | 10.560x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.2153 ms | not_measured | managed | 5.147x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 26.5198 ms | not_measured | managed | 4.422x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.00265 ms | not_measured | managed | 7.018x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.08626 ms | not_measured | managed | 1.857x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 9.2876 ms | not_measured | managed | 3.300x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.00130 ms | not_measured | managed | 13.399x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.1121 ms | not_measured | managed | 1.619x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 20.2835 ms | not_measured | managed | 2.306x |
| `np.prod (float64)` | float64 | 1,000 | 0.000649 ms | not_measured | managed | 3.359x |
| `np.prod (float64)` | float64 | 100,000 | 0.1762 ms | not_measured | managed | 12.985x |
| `np.prod (float64)` | float64 | 10,000,000 | 324.2812 ms | not_measured | managed | 0.730x |
| `np.prod (int64)` | int64 | 1,000 | 0.000357 ms | not_measured | managed | 5.415x |
| `np.prod (int64)` | int64 | 100,000 | 0.01777 ms | not_measured | managed | 3.163x |
| `np.prod (int64)` | int64 | 10,000,000 | 6.3692 ms | not_measured | managed | 0.948x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.000661 ms | not_measured | managed | 2.694x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.01573 ms | not_measured | managed | 0.710x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 85.6798 ms | not_measured | managed | 0.229x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.000853 ms | not_measured | managed | 2.205x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.02405 ms | not_measured | managed | 1.127x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 8.3503 ms | not_measured | managed | 0.559x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.000709 ms | not_measured | managed | 2.591x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.01145 ms | not_measured | managed | 4.949x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 3.0019 ms | not_measured | managed | 23.207x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.000725 ms | not_measured | managed | 2.512x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.04393 ms | not_measured | managed | 1.036x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 8.5784 ms | not_measured | managed | 0.702x |
| `np.std (float16)` | float16 | 1,000 | 0.00290 ms | not_measured | managed | 5.757x |
| `np.std (float16)` | float16 | 100,000 | 0.1796 ms | not_measured | managed | 4.810x |
| `np.std (float16)` | float16 | 10,000,000 | 27.9225 ms | not_measured | managed | 3.211x |
| `np.std (float32)` | float32 | 1,000 | 0.000619 ms | not_measured | managed | 12.864x |
| `np.std (float32)` | float32 | 100,000 | 0.01821 ms | not_measured | managed | 2.454x |
| `np.std (float32)` | float32 | 10,000,000 | 4.8230 ms | not_measured | managed | 3.483x |
| `np.std (float64)` | float64 | 1,000 | 0.000830 ms | not_measured | managed | 7.871x |
| `np.std (float64)` | float64 | 100,000 | 0.03441 ms | not_measured | managed | 1.721x |
| `np.std (float64)` | float64 | 10,000,000 | 7.2392 ms | not_measured | managed | 4.047x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.00883 ms | not_measured | managed | 2.185x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.3603 ms | not_measured | managed | 2.929x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 173.9578 ms | not_measured | managed | 0.619x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 7.847x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.03079 ms | not_measured | managed | 1.186x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 4.8606 ms | not_measured | managed | 2.825x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.000513 ms | not_measured | managed | 14.115x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.03781 ms | not_measured | managed | 1.800x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 12.9557 ms | not_measured | managed | 2.148x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.000482 ms | not_measured | managed | 3.612x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.02138 ms | not_measured | managed | 1.328x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 12.0361 ms | not_measured | managed | 0.678x |
| `np.sum (float16)` | float16 | 1,000 | 0.000420 ms | not_measured | managed | 8.231x |
| `np.sum (float16)` | float16 | 100,000 | 0.05346 ms | not_measured | managed | 3.597x |
| `np.sum (float16)` | float16 | 10,000,000 | 6.3547 ms | not_measured | managed | 3.099x |
| `np.sum (float32)` | float32 | 1,000 | 0.000405 ms | not_measured | managed | 3.985x |
| `np.sum (float32)` | float32 | 100,000 | 0.00491 ms | not_measured | managed | 3.065x |
| `np.sum (float32)` | float32 | 10,000,000 | 2.1329 ms | not_measured | managed | 1.452x |
| `np.sum (float64)` | float64 | 1,000 | 0.00486 ms | not_measured | managed | 0.331x |
| `np.sum (float64)` | float64 | 100,000 | 0.1872 ms | not_measured | managed | 0.087x |
| `np.sum (float64)` | float64 | 10,000,000 | 10.5367 ms | not_measured | managed | 0.416x |
| `np.sum (int16)` | int16 | 1,000 | 0.000584 ms | not_measured | managed | 3.267x |
| `np.sum (int16)` | int16 | 100,000 | 0.01889 ms | not_measured | managed | 1.728x |
| `np.sum (int16)` | int16 | 10,000,000 | 3.5696 ms | not_measured | managed | 0.901x |
| `np.sum (int32)` | int32 | 1,000 | 0.000398 ms | not_measured | managed | 4.927x |
| `np.sum (int32)` | int32 | 100,000 | 0.02798 ms | not_measured | managed | 1.213x |
| `np.sum (int32)` | int32 | 10,000,000 | 5.2095 ms | not_measured | managed | 0.766x |
| `np.sum (int64)` | int64 | 1,000 | 0.000529 ms | not_measured | managed | 3.183x |
| `np.sum (int64)` | int64 | 100,000 | 0.01012 ms | not_measured | managed | 1.467x |
| `np.sum (int64)` | int64 | 10,000,000 | 4.1859 ms | not_measured | managed | 0.926x |
| `np.sum (int8)` | int8 | 1,000 | 0.000379 ms | not_measured | managed | 5.063x |
| `np.sum (int8)` | int8 | 100,000 | 0.01860 ms | not_measured | managed | 1.753x |
| `np.sum (int8)` | int8 | 10,000,000 | 4.7133 ms | not_measured | managed | 0.657x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.000784 ms | not_measured | managed | 2.457x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.01886 ms | not_measured | managed | 1.721x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 4.2303 ms | not_measured | managed | 0.745x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.000763 ms | not_measured | managed | 2.518x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.03502 ms | not_measured | managed | 0.928x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 4.9372 ms | not_measured | managed | 0.826x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.000375 ms | not_measured | managed | 4.475x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.00956 ms | not_measured | managed | 1.551x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 6.2942 ms | not_measured | managed | 0.610x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.000423 ms | not_measured | managed | 4.612x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.01856 ms | not_measured | managed | 1.826x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 3.2037 ms | not_measured | managed | 0.967x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.00250 ms | not_measured | managed | 0.786x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.04161 ms | not_measured | managed | 0.392x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 16.4545 ms | not_measured | managed | 0.431x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.00371 ms | not_measured | managed | 1.295x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1558 ms | not_measured | managed | 1.801x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 13.7615 ms | not_measured | managed | 2.068x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.00198 ms | not_measured | managed | 0.931x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.01354 ms | not_measured | managed | 0.520x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 4.0178 ms | not_measured | managed | 0.365x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.00145 ms | not_measured | managed | 1.281x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.01559 ms | not_measured | managed | 0.723x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 4.4297 ms | not_measured | managed | 0.726x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.00121 ms | not_measured | managed | 1.832x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.00949 ms | not_measured | managed | 4.864x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 1.1991 ms | not_measured | managed | 3.793x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.000652 ms | not_measured | managed | 3.459x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.01055 ms | not_measured | managed | 4.406x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 4.1568 ms | not_measured | managed | 1.266x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.000639 ms | not_measured | managed | 2.989x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.01769 ms | not_measured | managed | 1.525x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 8.3953 ms | not_measured | managed | 0.572x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.000954 ms | not_measured | managed | 2.282x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.00636 ms | not_measured | managed | 7.377x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.5717 ms | not_measured | managed | 7.596x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.00122 ms | not_measured | managed | 1.831x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.00607 ms | not_measured | managed | 7.608x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 1.2824 ms | not_measured | managed | 3.500x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.000604 ms | not_measured | managed | 3.677x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.01087 ms | not_measured | managed | 4.239x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 2.8518 ms | not_measured | managed | 1.973x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.000630 ms | not_measured | managed | 3.040x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.01797 ms | not_measured | managed | 1.503x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 8.1436 ms | not_measured | managed | 0.593x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.00116 ms | not_measured | managed | 1.959x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.00934 ms | not_measured | managed | 4.979x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.5469 ms | not_measured | managed | 7.869x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.00262 ms | not_measured | managed | 0.751x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.04724 ms | not_measured | managed | 0.645x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 10.1751 ms | not_measured | managed | 0.863x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.00423 ms | not_measured | managed | 0.872x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.08236 ms | not_measured | managed | 2.395x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 5.6260 ms | not_measured | managed | 3.474x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.00239 ms | not_measured | managed | 0.752x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.01463 ms | not_measured | managed | 1.077x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 3.9653 ms | not_measured | managed | 0.734x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.00223 ms | not_measured | managed | 0.814x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.02210 ms | not_measured | managed | 0.781x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 8.4883 ms | not_measured | managed | 0.588x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.000723 ms | not_measured | managed | 2.992x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.00682 ms | not_measured | managed | 5.290x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 1.1132 ms | not_measured | managed | 2.879x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.000592 ms | not_measured | managed | 3.694x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.00959 ms | not_measured | managed | 3.815x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 4.4966 ms | not_measured | managed | 0.880x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.000691 ms | not_measured | managed | 2.671x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.00749 ms | not_measured | managed | 2.200x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 5.2264 ms | not_measured | managed | 0.740x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.000578 ms | not_measured | managed | 3.730x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.00661 ms | not_measured | managed | 5.438x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.2749 ms | not_measured | managed | 11.236x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.000739 ms | not_measured | managed | 2.930x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.00357 ms | not_measured | managed | 10.071x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 1.2395 ms | not_measured | managed | 2.536x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.000654 ms | not_measured | managed | 3.294x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.00509 ms | not_measured | managed | 7.063x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 4.3827 ms | not_measured | managed | 1.048x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.000492 ms | not_measured | managed | 3.638x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.01122 ms | not_measured | managed | 1.463x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 8.3491 ms | not_measured | managed | 0.478x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.000732 ms | not_measured | managed | 3.014x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.00627 ms | not_measured | managed | 5.757x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.5206 ms | not_measured | managed | 5.922x |
| `np.var (float16)` | float16 | 1,000 | 0.00246 ms | not_measured | managed | 6.592x |
| `np.var (float16)` | float16 | 100,000 | 0.3117 ms | not_measured | managed | 2.777x |
| `np.var (float16)` | float16 | 10,000,000 | 35.1905 ms | not_measured | managed | 2.547x |
| `np.var (float32)` | float32 | 1,000 | 0.000841 ms | not_measured | managed | 8.969x |
| `np.var (float32)` | float32 | 100,000 | 0.01492 ms | not_measured | managed | 2.966x |
| `np.var (float32)` | float32 | 10,000,000 | 8.0053 ms | not_measured | managed | 2.097x |
| `np.var (float64)` | float64 | 1,000 | 0.000791 ms | not_measured | managed | 7.789x |
| `np.var (float64)` | float64 | 100,000 | 0.03086 ms | not_measured | managed | 1.907x |
| `np.var (float64)` | float64 | 10,000,000 | 10.6119 ms | not_measured | managed | 2.763x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.00482 ms | not_measured | managed | 3.914x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.4051 ms | not_measured | managed | 2.593x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 134.3930 ms | not_measured | managed | 0.799x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.00231 ms | not_measured | managed | 3.424x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.04599 ms | not_measured | managed | 0.791x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 5.4515 ms | not_measured | managed | 2.481x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.000548 ms | not_measured | managed | 12.668x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.03770 ms | not_measured | managed | 1.787x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 8.0835 ms | not_measured | managed | 3.390x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.00280 ms | not_measured | managed | 1.899x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.09930 ms | not_measured | managed | 5.819x |
| `np.choose(selector, choices) (float64)` | float64 | 10,000,000 | 1.8169 ms | not_measured | managed | 2.990x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.000756 ms | not_measured | managed | 2.520x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.06616 ms | not_measured | managed | 1.238x |
| `np.compress(mask, a) (float64)` | float64 | 10,000,000 | 0.5669 ms | not_measured | managed | 2.929x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.000407 ms | not_measured | managed | 0.993x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.00131 ms | not_measured | managed | 0.387x |
| `np.diag_indices(n) (float64)` | float64 | 10,000,000 | 0.00112 ms | not_measured | managed | 0.552x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.000788 ms | not_measured | managed | 4.942x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.00121 ms | not_measured | managed | 3.473x |
| `np.diag_indices_from(a) (float64)` | float64 | 10,000,000 | 0.00116 ms | not_measured | managed | 3.743x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.000741 ms | not_measured | managed | 3.854x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.04316 ms | not_measured | managed | 1.909x |
| `np.extract(mask, a) (float64)` | float64 | 10,000,000 | 0.5313 ms | not_measured | managed | 3.124x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.00157 ms | not_measured | managed | 1.478x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.07757 ms | not_measured | managed | 3.647x |
| `np.indices(shape) (float64)` | float64 | 10,000,000 | 1.3809 ms | not_measured | managed | 1.919x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.00400 ms | not_measured | managed | 1.976x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.2955 ms | not_measured | managed | 1.936x |
| `np.mask_indices(n, triu) (float64)` | float64 | 10,000,000 | 3.6667 ms | not_measured | managed | 1.475x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.000628 ms | not_measured | managed | 1.997x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.2717 ms | not_measured | managed | 1.171x |
| `np.place(a, mask, values) (float64)` | float64 | 10,000,000 | 2.9426 ms | not_measured | managed | 1.077x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.00177 ms | not_measured | managed | 1.093x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.09640 ms | not_measured | managed | 1.151x |
| `np.put(a, indices, values) (float64)` | float64 | 10,000,000 | 0.8346 ms | not_measured | managed | 2.034x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.00231 ms | not_measured | managed | 0.553x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.1249 ms | not_measured | managed | 0.596x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 10,000,000 | 0.9888 ms | not_measured | managed | 1.078x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.00169 ms | not_measured | managed | 6.754x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.08276 ms | not_measured | managed | 8.681x |
| `np.select(condlist, choicelist) (float64)` | float64 | 10,000,000 | 0.8665 ms | not_measured | managed | 10.213x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.00104 ms | not_measured | managed | 1.256x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.04073 ms | not_measured | managed | 1.298x |
| `np.take(a, indices) (float64)` | float64 | 10,000,000 | 0.5954 ms | not_measured | managed | 1.530x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.000746 ms | not_measured | managed | 2.247x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.04847 ms | not_measured | managed | 0.534x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 10,000,000 | 0.3457 ms | not_measured | managed | 2.143x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.00326 ms | not_measured | managed | 3.078x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.02531 ms | not_measured | managed | 2.793x |
| `np.tril_indices_from(a) (float64)` | float64 | 10,000,000 | 0.5267 ms | not_measured | managed | 2.969x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.00264 ms | not_measured | managed | 3.924x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.02588 ms | not_measured | managed | 2.637x |
| `np.triu_indices_from(a) (float64)` | float64 | 10,000,000 | 0.5586 ms | not_measured | managed | 3.463x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.000954 ms | not_measured | managed | 1.538x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1134 ms | not_measured | managed | 0.837x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 10,000,000 | 0.9375 ms | not_measured | managed | 1.370x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.00107 ms | not_measured | managed | 0.699x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.07429 ms | not_measured | managed | 0.417x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 4.1961 ms | not_measured | managed | 1.745x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.00169 ms | not_measured | managed | 0.634x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.05336 ms | not_measured | managed | 0.687x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 21.5482 ms | not_measured | managed | 0.830x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.00166 ms | not_measured | managed | 0.063x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.00120 ms | not_measured | managed | 0.089x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.00178 ms | not_measured | managed | 0.059x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.00173 ms | not_measured | managed | 0.089x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.00201 ms | not_measured | managed | 0.077x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.00272 ms | not_measured | managed | 0.059x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.00253 ms | not_measured | managed | 0.061x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.00164 ms | not_measured | managed | 0.094x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.00171 ms | not_measured | managed | 0.090x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.00144 ms | not_measured | managed | 0.073x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.00162 ms | not_measured | managed | 0.065x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.00108 ms | not_measured | managed | 0.097x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.00133 ms | not_measured | managed | 0.079x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.00106 ms | not_measured | managed | 0.100x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.00112 ms | not_measured | managed | 0.095x |
| `a[::2] * 2` | float64 | 1,000 | 0.00278 ms | not_measured | managed | 0.298x |
| `a[::2] * 2` | float64 | 100,000 | 0.01911 ms | not_measured | managed | 0.746x |
| `a[::2] * 2` | float64 | 10,000,000 | 5.1071 ms | not_measured | managed | 1.724x |
| `a[N/10:9N/10] * 2` | float64 | 1,000 | 0.00493 ms | not_measured | managed | 0.167x |
| `a[N/10:9N/10] * 2` | float64 | 100,000 | 0.02246 ms | not_measured | managed | 0.432x |
| `a[N/10:9N/10] * 2` | float64 | 10,000,000 | 5.0507 ms | not_measured | managed | 2.459x |
| `a[N/10:9N/10].copy()` | float64 | 1,000 | 0.00212 ms | not_measured | managed | 0.180x |
| `a[N/10:9N/10].copy()` | float64 | 100,000 | 0.03916 ms | not_measured | managed | 0.231x |
| `a[N/10:9N/10].copy()` | float64 | 10,000,000 | 7.4242 ms | not_measured | managed | 1.306x |
| `np.copy(a[N/10:9N/10])` | float64 | 1,000 | 0.00223 ms | not_measured | managed | 0.262x |
| `np.copy(a[N/10:9N/10])` | float64 | 100,000 | 0.03238 ms | not_measured | managed | 0.285x |
| `np.copy(a[N/10:9N/10])` | float64 | 10,000,000 | 6.5110 ms | not_measured | managed | 1.514x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.00608 ms | not_measured | managed | 0.709x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 0.4412 ms | not_measured | managed | 0.313x |
| `np.argpartition(a, kth) (float32)` | float32 | 10,000,000 | 7.4939 ms | not_measured | managed | 0.299x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.01026 ms | not_measured | managed | 0.383x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.4498 ms | not_measured | managed | 0.313x |
| `np.argpartition(a, kth) (float64)` | float64 | 10,000,000 | 7.9925 ms | not_measured | managed | 0.316x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.00647 ms | not_measured | managed | 0.457x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 0.5923 ms | not_measured | managed | 0.185x |
| `np.argpartition(a, kth) (int32)` | int32 | 10,000,000 | 4.8678 ms | not_measured | managed | 0.463x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.00932 ms | not_measured | managed | 0.348x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 0.5524 ms | not_measured | managed | 0.239x |
| `np.argpartition(a, kth) (int64)` | int64 | 10,000,000 | 5.1150 ms | not_measured | managed | 0.468x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.01044 ms | not_measured | managed | 1.088x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.7203 ms | not_measured | managed | 0.870x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 175.5030 ms | not_measured | managed | 2.743x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.01714 ms | not_measured | managed | 0.566x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 2.4739 ms | not_measured | managed | 0.545x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 247.3270 ms | not_measured | managed | 2.421x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.01653 ms | not_measured | managed | 0.665x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 1.0517 ms | not_measured | managed | 0.387x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 158.3200 ms | not_measured | managed | 0.993x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.01561 ms | not_measured | managed | 0.830x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 1.9095 ms | not_measured | managed | 0.242x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 402.8677 ms | not_measured | managed | 0.724x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.000892 ms | not_measured | managed | 4.474x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.09392 ms | not_measured | managed | 1.981x |
| `np.argwhere(a) (float32)` | float32 | 10,000,000 | 0.8265 ms | not_measured | managed | 4.299x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.00199 ms | not_measured | managed | 2.011x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.06685 ms | not_measured | managed | 2.857x |
| `np.argwhere(a) (float64)` | float64 | 10,000,000 | 1.1284 ms | not_measured | managed | 4.777x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.00136 ms | not_measured | managed | 2.181x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.08892 ms | not_measured | managed | 1.247x |
| `np.argwhere(a) (int32)` | int32 | 10,000,000 | 1.1761 ms | not_measured | managed | 2.478x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.00182 ms | not_measured | managed | 1.641x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.07938 ms | not_measured | managed | 4.745x |
| `np.argwhere(a) (int64)` | int64 | 10,000,000 | 0.7432 ms | not_measured | managed | 3.996x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.00114 ms | not_measured | managed | 2.657x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.1793 ms | not_measured | managed | 0.969x |
| `np.flatnonzero(a) (float32)` | float32 | 10,000,000 | 1.3147 ms | not_measured | managed | 1.808x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.00238 ms | not_measured | managed | 1.258x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.1796 ms | not_measured | managed | 0.973x |
| `np.flatnonzero(a) (float64)` | float64 | 10,000,000 | 1.8728 ms | not_measured | managed | 1.660x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.00138 ms | not_measured | managed | 1.465x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.1873 ms | not_measured | managed | 0.545x |
| `np.flatnonzero(a) (int32)` | int32 | 10,000,000 | 1.2714 ms | not_measured | managed | 1.395x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.00116 ms | not_measured | managed | 1.775x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.1414 ms | not_measured | managed | 0.715x |
| `np.flatnonzero(a) (int64)` | int64 | 10,000,000 | 1.6727 ms | not_measured | managed | 1.054x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.03768 ms | not_measured | managed | 1.099x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 2.8374 ms | not_measured | managed | 4.493x |
| `np.lexsort((b, a)) (float32)` | float32 | 10,000,000 | 30.8604 ms | not_measured | managed | 5.693x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.05369 ms | not_measured | managed | 0.698x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 5.8614 ms | not_measured | managed | 2.184x |
| `np.lexsort((b, a)) (float64)` | float64 | 10,000,000 | 79.8606 ms | not_measured | managed | 2.327x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.03329 ms | not_measured | managed | 1.157x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 1.9216 ms | not_measured | managed | 5.219x |
| `np.lexsort((b, a)) (int32)` | int32 | 10,000,000 | 43.8396 ms | not_measured | managed | 4.392x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.03066 ms | not_measured | managed | 1.237x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 3.5684 ms | not_measured | managed | 3.046x |
| `np.lexsort((b, a)) (int64)` | int64 | 10,000,000 | 54.0510 ms | not_measured | managed | 5.400x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.00116 ms | not_measured | managed | 2.011x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.03965 ms | not_measured | managed | 4.365x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 13.3948 ms | not_measured | managed | 1.866x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.00215 ms | not_measured | managed | 0.992x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.1037 ms | not_measured | managed | 1.590x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 16.8662 ms | not_measured | managed | 1.598x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.00130 ms | not_measured | managed | 1.180x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.03733 ms | not_measured | managed | 2.748x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 14.0340 ms | not_measured | managed | 1.366x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.00154 ms | not_measured | managed | 1.027x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.1024 ms | not_measured | managed | 1.032x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 20.7844 ms | not_measured | managed | 1.098x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.00513 ms | not_measured | managed | 0.420x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.3288 ms | not_measured | managed | 0.289x |
| `np.partition(a, kth) (float32)` | float32 | 10,000,000 | 4.0650 ms | not_measured | managed | 0.365x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.00607 ms | not_measured | managed | 0.457x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.3921 ms | not_measured | managed | 0.348x |
| `np.partition(a, kth) (float64)` | float64 | 10,000,000 | 5.5643 ms | not_measured | managed | 0.497x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.00356 ms | not_measured | managed | 0.428x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.2912 ms | not_measured | managed | 0.091x |
| `np.partition(a, kth) (int32)` | int32 | 10,000,000 | 3.2874 ms | not_measured | managed | 0.251x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.00703 ms | not_measured | managed | 0.284x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.5564 ms | not_measured | managed | 0.150x |
| `np.partition(a, kth) (int64)` | int64 | 10,000,000 | 3.2398 ms | not_measured | managed | 0.566x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.01089 ms | not_measured | managed | 0.638x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 2.5471 ms | not_measured | managed | 0.786x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 328.5732 ms | not_measured | managed | 0.725x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.01145 ms | not_measured | managed | 0.558x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 2.6839 ms | not_measured | managed | 0.761x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 307.5072 ms | not_measured | managed | 0.797x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.00629 ms | not_measured | managed | 2.915x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 2.2411 ms | not_measured | managed | 1.256x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 288.8701 ms | not_measured | managed | 1.322x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.00598 ms | not_measured | managed | 3.060x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 2.3702 ms | not_measured | managed | 1.197x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 308.2325 ms | not_measured | managed | 1.245x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.00799 ms | not_measured | managed | 0.247x |
| `np.sort(a) (float32)` | float32 | 100,000 | 0.7146 ms | not_measured | managed | 0.310x |
| `np.sort(a) (float32)` | float32 | 10,000,000 | 9.6656 ms | not_measured | managed | 0.375x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.01446 ms | not_measured | managed | 0.224x |
| `np.sort(a) (float64)` | float64 | 100,000 | 1.7432 ms | not_measured | managed | 0.268x |
| `np.sort(a) (float64)` | float64 | 10,000,000 | 15.9117 ms | not_measured | managed | 0.458x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.00750 ms | not_measured | managed | 0.272x |
| `np.sort(a) (int32)` | int32 | 100,000 | 0.5926 ms | not_measured | managed | 0.154x |
| `np.sort(a) (int32)` | int32 | 10,000,000 | 7.5312 ms | not_measured | managed | 0.202x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.01221 ms | not_measured | managed | 0.446x |
| `np.sort(a) (int64)` | int64 | 100,000 | 1.6808 ms | not_measured | managed | 0.155x |
| `np.sort(a) (int64)` | int64 | 10,000,000 | 17.7446 ms | not_measured | managed | 0.206x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.00962 ms | not_measured | managed | 0.276x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 1.0847 ms | not_measured | managed | 0.439x |
| `np.sort_complex(a) (float32)` | float32 | 10,000,000 | 7.9637 ms | not_measured | managed | 0.761x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.02477 ms | not_measured | managed | 0.159x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 1.8859 ms | not_measured | managed | 0.390x |
| `np.sort_complex(a) (float64)` | float64 | 10,000,000 | 19.8594 ms | not_measured | managed | 0.512x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.00887 ms | not_measured | managed | 0.305x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 0.6157 ms | not_measured | managed | 0.577x |
| `np.sort_complex(a) (int32)` | int32 | 10,000,000 | 10.5823 ms | not_measured | managed | 0.374x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.01588 ms | not_measured | managed | 0.386x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 2.1686 ms | not_measured | managed | 0.248x |
| `np.sort_complex(a) (int64)` | int64 | 10,000,000 | 20.8201 ms | not_measured | managed | 0.303x |
| `np.average(a) (float16)` | float16 | 1,000 | 0.00207 ms | not_measured | managed | 2.358x |
| `np.average(a) (float16)` | float16 | 100,000 | 0.08131 ms | not_measured | managed | 1.209x |
| `np.average(a) (float16)` | float16 | 10,000,000 | 8.0310 ms | not_measured | managed | 1.200x |
| `np.average(a) (float32)` | float32 | 1,000 | 0.000559 ms | not_measured | managed | 6.784x |
| `np.average(a) (float32)` | float32 | 100,000 | 0.00559 ms | not_measured | managed | 3.117x |
| `np.average(a) (float32)` | float32 | 10,000,000 | 2.9208 ms | not_measured | managed | 0.885x |
| `np.average(a) (float64)` | float64 | 1,000 | 0.000689 ms | not_measured | managed | 3.807x |
| `np.average(a) (float64)` | float64 | 100,000 | 0.00811 ms | not_measured | managed | 2.077x |
| `np.average(a) (float64)` | float64 | 10,000,000 | 3.2529 ms | not_measured | managed | 1.493x |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.00160 ms | not_measured | managed | 0.833x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.1896 ms | not_measured | managed | 0.501x |
| `np.bincount(a) (int32)` | int32 | 10,000,000 | 3.1049 ms | not_measured | managed | 0.841x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | 0.00763 ms | 0.01560 ms | managed | 1.463x |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | 0.3996 ms | 0.9192 ms | managed | 2.465x |
| `np.convolve(a, kernel) (float64)` | float64 | 10,000,000 | 2.1347 ms | 100.8334 ms | managed | 5.232x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | 0.02292 ms | not_measured | managed | 0.786x |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | 0.3542 ms | not_measured | managed | 1.559x |
| `np.corrcoef(a, b) (float64)` | float64 | 10,000,000 | 10.9953 ms | not_measured | managed | 0.697x |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | 0.00499 ms | 0.01320 ms | managed | 2.162x |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | 0.3853 ms | 0.9014 ms | managed | 2.522x |
| `np.correlate(a, kernel) (float64)` | float64 | 10,000,000 | 1.2169 ms | 100.8807 ms | managed | 9.126x |
| `np.count_nonzero(a) (float16)` | float16 | 1,000 | 0.00105 ms | not_measured | managed | 1.651x |
| `np.count_nonzero(a) (float16)` | float16 | 100,000 | 0.04304 ms | not_measured | managed | 3.274x |
| `np.count_nonzero(a) (float16)` | float16 | 10,000,000 | 6.3785 ms | not_measured | managed | 2.265x |
| `np.count_nonzero(a) (float32)` | float32 | 1,000 | 0.000313 ms | not_measured | managed | 1.866x |
| `np.count_nonzero(a) (float32)` | float32 | 100,000 | 0.00774 ms | not_measured | managed | 4.741x |
| `np.count_nonzero(a) (float32)` | float32 | 10,000,000 | 1.8197 ms | not_measured | managed | 2.182x |
| `np.count_nonzero(a) (float64)` | float64 | 1,000 | 0.000405 ms | not_measured | managed | 1.442x |
| `np.count_nonzero(a) (float64)` | float64 | 100,000 | 0.00972 ms | not_measured | managed | 3.775x |
| `np.count_nonzero(a) (float64)` | float64 | 10,000,000 | 4.0770 ms | not_measured | managed | 1.468x |
| `np.cov(a, b) (float64)` | float64 | 1,000 | 0.01412 ms | not_measured | managed | 0.926x |
| `np.cov(a, b) (float64)` | float64 | 100,000 | 0.3361 ms | not_measured | managed | 1.597x |
| `np.cov(a, b) (float64)` | float64 | 10,000,000 | 9.6925 ms | not_measured | managed | 0.755x |
| `np.cross(a, b) (float64)` | float64 | 1,000 | 0.04747 ms | not_measured | managed | 0.261x |
| `np.cross(a, b) (float64)` | float64 | 100,000 | 1.0171 ms | not_measured | managed | 0.737x |
| `np.cross(a, b) (float64)` | float64 | 10,000,000 | 10.0843 ms | not_measured | managed | 0.977x |
| `np.diff(a) (float64)` | float64 | 1,000 | 0.00129 ms | not_measured | managed | 1.187x |
| `np.diff(a) (float64)` | float64 | 100,000 | 0.04368 ms | not_measured | managed | 0.283x |
| `np.diff(a) (float64)` | float64 | 10,000,000 | 1.0717 ms | not_measured | managed | 1.272x |
| `np.digitize(a, bins) (float64)` | float64 | 1,000 | 0.01422 ms | not_measured | managed | 0.860x |
| `np.digitize(a, bins) (float64)` | float64 | 100,000 | 3.5960 ms | not_measured | managed | 0.906x |
| `np.digitize(a, bins) (float64)` | float64 | 10,000,000 | 45.4933 ms | not_measured | managed | 0.742x |
| `np.ediff1d(a) (float64)` | float64 | 1,000 | 0.00171 ms | not_measured | managed | 0.568x |
| `np.ediff1d(a) (float64)` | float64 | 100,000 | 0.05192 ms | not_measured | managed | 0.223x |
| `np.ediff1d(a) (float64)` | float64 | 10,000,000 | 0.9780 ms | not_measured | managed | 1.383x |
| `np.inner(a, b) (float64)` | float64 | 1,000 | 0.000535 ms | 0.000700 ms | managed | 1.140x |
| `np.inner(a, b) (float64)` | float64 | 100,000 | 0.01790 ms | 0.00900 ms | openblas | 1.114x |
| `np.inner(a, b) (float64)` | float64 | 10,000,000 | 0.4062 ms | 2.9246 ms | managed | 0.502x |
| `np.kron(a, b) (float64)` | float64 | 1,000 | 0.00495 ms | not_measured | managed | 1.668x |
| `np.kron(a, b) (float64)` | float64 | 100,000 | 0.03361 ms | not_measured | managed | 1.261x |
| `np.kron(a, b) (float64)` | float64 | 10,000,000 | 0.7320 ms | not_measured | managed | 2.606x |
| `np.median(a) (float16)` | float16 | 1,000 | 0.00621 ms | not_measured | managed | 2.096x |
| `np.median(a) (float16)` | float16 | 100,000 | 1.1698 ms | not_measured | managed | 0.736x |
| `np.median(a) (float16)` | float16 | 10,000,000 | 106.5602 ms | not_measured | managed | 0.979x |
| `np.median(a) (float32)` | float32 | 1,000 | 0.00386 ms | not_measured | managed | 2.627x |
| `np.median(a) (float32)` | float32 | 100,000 | 0.2892 ms | not_measured | managed | 1.554x |
| `np.median(a) (float32)` | float32 | 10,000,000 | 46.0247 ms | not_measured | managed | 1.565x |
| `np.median(a) (float64)` | float64 | 1,000 | 0.00421 ms | not_measured | managed | 2.145x |
| `np.median(a) (float64)` | float64 | 100,000 | 0.3196 ms | not_measured | managed | 1.420x |
| `np.median(a) (float64)` | float64 | 10,000,000 | 42.5415 ms | not_measured | managed | 1.957x |
| `np.percentile(a, 50) (float16)` | float16 | 1,000 | 0.00854 ms | not_measured | managed | 3.156x |
| `np.percentile(a, 50) (float16)` | float16 | 100,000 | 1.4008 ms | not_measured | managed | 1.225x |
| `np.percentile(a, 50) (float16)` | float16 | 10,000,000 | 104.7791 ms | not_measured | managed | 1.056x |
| `np.percentile(a, 50) (float32)` | float32 | 1,000 | 0.00393 ms | not_measured | managed | 5.786x |
| `np.percentile(a, 50) (float32)` | float32 | 100,000 | 0.2452 ms | not_measured | managed | 2.743x |
| `np.percentile(a, 50) (float32)` | float32 | 10,000,000 | 29.8292 ms | not_measured | managed | 1.567x |
| `np.percentile(a, 50) (float64)` | float64 | 1,000 | 0.00414 ms | not_measured | managed | 5.299x |
| `np.percentile(a, 50) (float64)` | float64 | 100,000 | 0.3135 ms | not_measured | managed | 2.155x |
| `np.percentile(a, 50) (float64)` | float64 | 10,000,000 | 41.7629 ms | not_measured | managed | 1.371x |
| `np.ptp(a) (float16)` | float16 | 1,000 | 0.00442 ms | not_measured | managed | 1.525x |
| `np.ptp(a) (float16)` | float16 | 100,000 | 0.8416 ms | not_measured | managed | 1.159x |
| `np.ptp(a) (float16)` | float16 | 10,000,000 | 84.0569 ms | not_measured | managed | 1.207x |
| `np.ptp(a) (float32)` | float32 | 1,000 | 0.00178 ms | not_measured | managed | 1.366x |
| `np.ptp(a) (float32)` | float32 | 100,000 | 0.01828 ms | not_measured | managed | 0.606x |
| `np.ptp(a) (float32)` | float32 | 10,000,000 | 4.9427 ms | not_measured | managed | 0.454x |
| `np.ptp(a) (float64)` | float64 | 1,000 | 0.00236 ms | not_measured | managed | 1.075x |
| `np.ptp(a) (float64)` | float64 | 100,000 | 0.03494 ms | not_measured | managed | 0.558x |
| `np.ptp(a) (float64)` | float64 | 10,000,000 | 6.6855 ms | not_measured | managed | 1.117x |
| `np.quantile(a, 0.5) (float16)` | float16 | 1,000 | 0.00620 ms | not_measured | managed | 4.264x |
| `np.quantile(a, 0.5) (float16)` | float16 | 100,000 | 1.5912 ms | not_measured | managed | 1.082x |
| `np.quantile(a, 0.5) (float16)` | float16 | 10,000,000 | 119.8276 ms | not_measured | managed | 0.927x |
| `np.quantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00404 ms | not_measured | managed | 5.315x |
| `np.quantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2944 ms | not_measured | managed | 2.287x |
| `np.quantile(a, 0.5) (float32)` | float32 | 10,000,000 | 45.1417 ms | not_measured | managed | 1.031x |
| `np.quantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00403 ms | not_measured | managed | 5.299x |
| `np.quantile(a, 0.5) (float64)` | float64 | 100,000 | 0.2970 ms | not_measured | managed | 2.292x |
| `np.quantile(a, 0.5) (float64)` | float64 | 10,000,000 | 58.9939 ms | not_measured | managed | 0.970x |
| `np.trace(a) (float64)` | float64 | 1,000 | 0.000571 ms | not_measured | managed | 1.998x |
| `np.trace(a) (float64)` | float64 | 100,000 | 0.000437 ms | not_measured | managed | 2.773x |
| `np.trace(a) (float64)` | float64 | 10,000,000 | 0.000740 ms | not_measured | managed | 2.492x |
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.00760 ms | not_measured | managed | 0.413x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.9424 ms | not_measured | managed | 0.292x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 92.3848 ms | not_measured | managed | 0.326x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.000584 ms | not_measured | managed | 0.675x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.02006 ms | not_measured | managed | 0.293x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 9.0011 ms | not_measured | managed | 0.806x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.000837 ms | not_measured | managed | 0.560x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.04110 ms | not_measured | managed | 0.305x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 19.9595 ms | not_measured | managed | 0.776x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.01403 ms | not_measured | managed | 0.718x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 2.2475 ms | not_measured | managed | 0.384x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 230.4135 ms | not_measured | managed | 0.404x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.00119 ms | not_measured | managed | 0.644x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.05061 ms | not_measured | managed | 0.248x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 18.8508 ms | not_measured | managed | 0.847x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.00171 ms | not_measured | managed | 0.536x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.1110 ms | not_measured | managed | 2.758x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 47.2193 ms | not_measured | managed | 0.645x |
| `np.abs (float16)` | float16 | 1,000 | 0.000625 ms | not_measured | managed | 0.976x |
| `np.abs (float16)` | float16 | 100,000 | 0.02093 ms | not_measured | managed | 1.226x |
| `np.abs (float16)` | float16 | 10,000,000 | 3.3315 ms | not_measured | managed | 1.373x |
| `np.abs (float32)` | float32 | 1,000 | 0.000577 ms | not_measured | managed | 0.692x |
| `np.abs (float32)` | float32 | 100,000 | 0.01126 ms | not_measured | managed | 0.497x |
| `np.abs (float32)` | float32 | 10,000,000 | 5.1197 ms | not_measured | managed | 1.374x |
| `np.abs (float64)` | float64 | 1,000 | 0.000704 ms | not_measured | managed | 0.641x |
| `np.abs (float64)` | float64 | 100,000 | 0.02708 ms | not_measured | managed | 0.397x |
| `np.abs (float64)` | float64 | 10,000,000 | 13.2315 ms | not_measured | managed | 1.077x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.000735 ms | not_measured | managed | 0.849x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.02050 ms | not_measured | managed | 1.252x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 2.9082 ms | not_measured | managed | 1.654x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.000934 ms | not_measured | managed | 0.429x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.02093 ms | not_measured | managed | 0.266x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 5.3690 ms | not_measured | managed | 1.317x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.000977 ms | not_measured | managed | 0.432x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.03310 ms | not_measured | managed | 0.324x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 13.5603 ms | not_measured | managed | 1.196x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.01082 ms | not_measured | managed | 0.791x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 1.8085 ms | not_measured | managed | 0.451x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 106.8006 ms | not_measured | managed | 0.796x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.00558 ms | not_measured | managed | 0.985x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 0.5351 ms | not_measured | managed | 0.961x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 55.0622 ms | not_measured | managed | 1.015x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.00972 ms | not_measured | managed | 0.676x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 0.6552 ms | not_measured | managed | 0.947x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 105.4458 ms | not_measured | managed | 0.666x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.00940 ms | not_measured | managed | 1.202x |
| `np.angle(a) (float16)` | float16 | 100,000 | 0.8090 ms | not_measured | managed | 1.302x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 107.0731 ms | not_measured | managed | 1.038x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.01218 ms | not_measured | managed | 0.319x |
| `np.angle(a) (float32)` | float32 | 100,000 | 0.6657 ms | not_measured | managed | 0.932x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 100.6711 ms | not_measured | managed | 0.671x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.00538 ms | not_measured | managed | 0.773x |
| `np.angle(a) (float64)` | float64 | 100,000 | 0.8649 ms | not_measured | managed | 0.666x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 81.7615 ms | not_measured | managed | 0.816x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.00767 ms | not_measured | managed | 0.895x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 1.2371 ms | not_measured | managed | 0.879x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 142.4846 ms | not_measured | managed | 0.789x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.00406 ms | not_measured | managed | 0.916x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 1.1123 ms | not_measured | managed | 0.681x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 80.5445 ms | not_measured | managed | 1.030x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.00909 ms | not_measured | managed | 0.427x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 0.8430 ms | not_measured | managed | 0.955x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 91.8313 ms | not_measured | managed | 1.009x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.01899 ms | not_measured | managed | 0.448x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 1.0600 ms | not_measured | managed | 0.771x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 108.2083 ms | not_measured | managed | 0.789x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.00565 ms | not_measured | managed | 0.975x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 0.5352 ms | not_measured | managed | 0.961x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 55.3034 ms | not_measured | managed | 1.010x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.01335 ms | not_measured | managed | 0.492x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 0.6492 ms | not_measured | managed | 0.956x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 74.2665 ms | not_measured | managed | 0.945x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.00803 ms | not_measured | managed | 0.885x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 1.2314 ms | not_measured | managed | 0.891x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 187.0184 ms | not_measured | managed | 0.618x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.00874 ms | not_measured | managed | 0.468x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 1.2771 ms | not_measured | managed | 0.619x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 91.4770 ms | not_measured | managed | 0.944x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.00459 ms | not_measured | managed | 0.862x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 0.8275 ms | not_measured | managed | 0.950x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 108.2761 ms | not_measured | managed | 0.846x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.02498 ms | not_measured | managed | 0.427x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 1.4315 ms | not_measured | managed | 0.807x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 176.8516 ms | not_measured | managed | 0.675x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.00676 ms | not_measured | managed | 0.981x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 0.8700 ms | not_measured | managed | 0.984x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 119.5134 ms | not_measured | managed | 0.766x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.01055 ms | not_measured | managed | 0.761x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 1.7100 ms | not_measured | managed | 0.542x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 137.8066 ms | not_measured | managed | 0.744x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.01621 ms | not_measured | managed | 0.420x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 1.9331 ms | not_measured | managed | 0.644x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 169.4150 ms | not_measured | managed | 0.760x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.00409 ms | not_measured | managed | 0.968x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 1.3289 ms | not_measured | managed | 0.676x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 99.5468 ms | not_measured | managed | 0.979x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.00859 ms | not_measured | managed | 0.477x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 1.1186 ms | not_measured | managed | 0.855x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 106.0930 ms | not_measured | managed | 1.006x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.01290 ms | not_measured | managed | 0.748x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 1.8457 ms | not_measured | managed | 0.894x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 301.8319 ms | not_measured | managed | 0.566x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.01740 ms | not_measured | managed | 0.250x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 3.3188 ms | not_measured | managed | 0.387x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 271.6354 ms | not_measured | managed | 0.508x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.00583 ms | not_measured | managed | 0.978x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 1.3931 ms | not_measured | managed | 0.983x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 206.3993 ms | not_measured | managed | 0.723x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.01526 ms | not_measured | managed | 0.592x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 1.2376 ms | not_measured | managed | 0.821x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 192.0743 ms | not_measured | managed | 0.556x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.01171 ms | not_measured | managed | 0.418x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 0.7892 ms | not_measured | managed | 0.964x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 81.8820 ms | not_measured | managed | 1.008x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.01234 ms | not_measured | managed | 0.469x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 0.8420 ms | not_measured | managed | 0.942x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 91.0756 ms | not_measured | managed | 0.993x |
| `np.around (float16)` | float16 | 1,000 | 0.00419 ms | not_measured | managed | 1.144x |
| `np.around (float16)` | float16 | 100,000 | 0.4055 ms | not_measured | managed | 0.925x |
| `np.around (float16)` | float16 | 10,000,000 | 39.4013 ms | not_measured | managed | 1.048x |
| `np.around (float32)` | float32 | 1,000 | 0.000601 ms | not_measured | managed | 1.677x |
| `np.around (float32)` | float32 | 100,000 | 0.01177 ms | not_measured | managed | 0.524x |
| `np.around (float32)` | float32 | 10,000,000 | 5.0231 ms | not_measured | managed | 1.430x |
| `np.around (float64)` | float64 | 1,000 | 0.000698 ms | not_measured | managed | 1.451x |
| `np.around (float64)` | float64 | 100,000 | 0.02666 ms | not_measured | managed | 0.430x |
| `np.around (float64)` | float64 | 10,000,000 | 13.5429 ms | not_measured | managed | 1.072x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.01214 ms | not_measured | managed | 0.876x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 2.2210 ms | not_measured | managed | 0.520x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 181.8418 ms | not_measured | managed | 0.654x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.00685 ms | not_measured | managed | 0.971x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 0.8787 ms | not_measured | managed | 0.977x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 108.5509 ms | not_measured | managed | 0.847x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.01178 ms | not_measured | managed | 0.682x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 1.6776 ms | not_measured | managed | 0.553x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 141.3374 ms | not_measured | managed | 0.726x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.01278 ms | not_measured | managed | 0.725x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 1.2389 ms | not_measured | managed | 0.822x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 192.7494 ms | not_measured | managed | 0.548x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.00566 ms | not_measured | managed | 0.867x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 1.2812 ms | not_measured | managed | 0.595x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 80.9905 ms | not_measured | managed | 1.020x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.00948 ms | not_measured | managed | 0.619x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 1.0521 ms | not_measured | managed | 0.754x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 91.8726 ms | not_measured | managed | 0.984x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.01750 ms | not_measured | managed | 0.533x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 2.3136 ms | not_measured | managed | 0.492x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 223.9790 ms | not_measured | managed | 0.526x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.01163 ms | not_measured | managed | 0.481x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 1.2581 ms | not_measured | managed | 0.673x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 138.6896 ms | not_measured | managed | 0.656x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.01798 ms | not_measured | managed | 0.462x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 2.0055 ms | not_measured | managed | 0.504x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 206.1748 ms | not_measured | managed | 0.542x |
| `np.ceil (float16)` | float16 | 1,000 | 0.00370 ms | not_measured | managed | 1.112x |
| `np.ceil (float16)` | float16 | 100,000 | 0.3348 ms | not_measured | managed | 1.115x |
| `np.ceil (float16)` | float16 | 10,000,000 | 32.4185 ms | not_measured | managed | 1.282x |
| `np.ceil (float32)` | float32 | 1,000 | 0.000569 ms | not_measured | managed | 0.745x |
| `np.ceil (float32)` | float32 | 100,000 | 0.01232 ms | not_measured | managed | 0.454x |
| `np.ceil (float32)` | float32 | 10,000,000 | 4.9731 ms | not_measured | managed | 1.367x |
| `np.ceil (float64)` | float64 | 1,000 | 0.000734 ms | not_measured | managed | 0.629x |
| `np.ceil (float64)` | float64 | 100,000 | 0.02722 ms | not_measured | managed | 0.395x |
| `np.ceil (float64)` | float64 | 10,000,000 | 13.2750 ms | not_measured | managed | 1.073x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.00477 ms | not_measured | managed | 1.746x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 0.7093 ms | not_measured | managed | 1.242x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 70.6008 ms | not_measured | managed | 1.306x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.00234 ms | not_measured | managed | 0.816x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.01684 ms | not_measured | managed | 0.445x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 4.8503 ms | not_measured | managed | 1.456x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.00209 ms | not_measured | managed | 0.829x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.02881 ms | not_measured | managed | 0.438x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 12.5886 ms | not_measured | managed | 1.193x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.000951 ms | not_measured | managed | 0.597x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.04152 ms | not_measured | managed | 0.500x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 3.0045 ms | not_measured | managed | 1.616x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.000739 ms | not_measured | managed | 0.724x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.02386 ms | not_measured | managed | 0.792x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 3.7904 ms | not_measured | managed | 1.886x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.00116 ms | not_measured | managed | 0.464x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.05310 ms | not_measured | managed | 0.357x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 12.3935 ms | not_measured | managed | 1.218x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.000802 ms | not_measured | managed | 0.712x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.02163 ms | not_measured | managed | 0.960x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 2.9474 ms | not_measured | managed | 1.563x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.00119 ms | not_measured | managed | 0.449x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.02182 ms | not_measured | managed | 0.867x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 3.6472 ms | not_measured | managed | 1.977x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.000922 ms | not_measured | managed | 0.584x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.02799 ms | not_measured | managed | 0.676x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 12.5751 ms | not_measured | managed | 1.132x |
| `np.cos (float16)` | float16 | 1,000 | 0.01633 ms | not_measured | managed | 0.427x |
| `np.cos (float16)` | float16 | 100,000 | 1.2458 ms | not_measured | managed | 0.768x |
| `np.cos (float16)` | float16 | 10,000,000 | 162.0875 ms | not_measured | managed | 0.618x |
| `np.cos (float32)` | float32 | 1,000 | 0.00198 ms | not_measured | managed | 0.550x |
| `np.cos (float32)` | float32 | 100,000 | 0.2058 ms | not_measured | managed | 0.350x |
| `np.cos (float32)` | float32 | 10,000,000 | 18.4178 ms | not_measured | managed | 0.619x |
| `np.cos (float64)` | float64 | 1,000 | 0.00465 ms | not_measured | managed | 1.043x |
| `np.cos (float64)` | float64 | 100,000 | 0.7485 ms | not_measured | managed | 0.956x |
| `np.cos (float64)` | float64 | 10,000,000 | 116.6414 ms | not_measured | managed | 0.701x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.02115 ms | not_measured | managed | 0.368x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 1.1639 ms | not_measured | managed | 0.784x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 152.1322 ms | not_measured | managed | 0.622x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.00392 ms | not_measured | managed | 0.968x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 0.6200 ms | not_measured | managed | 0.988x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 63.8423 ms | not_measured | managed | 1.056x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.01181 ms | not_measured | managed | 0.304x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 0.5750 ms | not_measured | managed | 0.982x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 102.6105 ms | not_measured | managed | 0.645x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.00466 ms | not_measured | managed | 0.802x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.3646 ms | not_measured | managed | 0.906x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 65.3639 ms | not_measured | managed | 0.570x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.000638 ms | not_measured | managed | 1.682x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.02033 ms | not_measured | managed | 3.610x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 6.9599 ms | not_measured | managed | 1.614x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.000984 ms | not_measured | managed | 1.276x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.02800 ms | not_measured | managed | 3.273x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 14.1471 ms | not_measured | managed | 1.255x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.00488 ms | not_measured | managed | 0.750x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.5363 ms | not_measured | managed | 0.607x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 60.2525 ms | not_measured | managed | 0.650x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.000674 ms | not_measured | managed | 1.593x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.01276 ms | not_measured | managed | 5.751x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 3.1392 ms | not_measured | managed | 3.594x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.00127 ms | not_measured | managed | 0.998x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.02489 ms | not_measured | managed | 3.681x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 13.4107 ms | not_measured | managed | 1.323x |
| `np.exp (float16)` | float16 | 1,000 | 0.00592 ms | not_measured | managed | 0.733x |
| `np.exp (float16)` | float16 | 100,000 | 0.5769 ms | not_measured | managed | 0.700x |
| `np.exp (float16)` | float16 | 10,000,000 | 58.2314 ms | not_measured | managed | 0.741x |
| `np.exp (float32)` | float32 | 1,000 | 0.000874 ms | not_measured | managed | 1.047x |
| `np.exp (float32)` | float32 | 100,000 | 0.05139 ms | not_measured | managed | 1.076x |
| `np.exp (float32)` | float32 | 10,000,000 | 5.2071 ms | not_measured | managed | 1.932x |
| `np.exp (float64)` | float64 | 1,000 | 0.00302 ms | not_measured | managed | 0.937x |
| `np.exp (float64)` | float64 | 100,000 | 0.2658 ms | not_measured | managed | 0.935x |
| `np.exp (float64)` | float64 | 10,000,000 | 53.3518 ms | not_measured | managed | 0.609x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.01470 ms | not_measured | managed | 0.303x |
| `np.exp2 (float16)` | float16 | 100,000 | 0.9492 ms | not_measured | managed | 0.427x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 148.5465 ms | not_measured | managed | 0.312x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.00130 ms | not_measured | managed | 1.567x |
| `np.exp2 (float32)` | float32 | 100,000 | 0.08786 ms | not_measured | managed | 1.872x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 26.2697 ms | not_measured | managed | 0.810x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.00836 ms | not_measured | managed | 0.260x |
| `np.exp2 (float64)` | float64 | 100,000 | 1.2045 ms | not_measured | managed | 0.148x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 153.5348 ms | not_measured | managed | 0.169x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.01383 ms | not_measured | managed | 0.405x |
| `np.expm1 (float16)` | float16 | 100,000 | 0.9696 ms | not_measured | managed | 0.523x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 110.0935 ms | not_measured | managed | 0.488x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.00222 ms | not_measured | managed | 1.398x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.2110 ms | not_measured | managed | 1.218x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 29.0007 ms | not_measured | managed | 1.044x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.00310 ms | not_measured | managed | 1.195x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.2681 ms | not_measured | managed | 1.246x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 52.4683 ms | not_measured | managed | 0.790x |
| `np.floor (float16)` | float16 | 1,000 | 0.00359 ms | not_measured | managed | 1.144x |
| `np.floor (float16)` | float16 | 100,000 | 0.3241 ms | not_measured | managed | 1.152x |
| `np.floor (float16)` | float16 | 10,000,000 | 32.3634 ms | not_measured | managed | 1.303x |
| `np.floor (float32)` | float32 | 1,000 | 0.000562 ms | not_measured | managed | 0.731x |
| `np.floor (float32)` | float32 | 100,000 | 0.01194 ms | not_measured | managed | 0.469x |
| `np.floor (float32)` | float32 | 10,000,000 | 4.9872 ms | not_measured | managed | 1.386x |
| `np.floor (float64)` | float64 | 1,000 | 0.000761 ms | not_measured | managed | 0.611x |
| `np.floor (float64)` | float64 | 100,000 | 0.02736 ms | not_measured | managed | 0.394x |
| `np.floor (float64)` | float64 | 10,000,000 | 13.2173 ms | not_measured | managed | 1.078x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.000448 ms | not_measured | managed | 0.752x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.00192 ms | not_measured | managed | 1.354x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | 0.00156 ms | not_measured | managed | 4.972x |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.000430 ms | not_measured | managed | 0.784x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.00169 ms | not_measured | managed | 2.859x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | 0.00485 ms | not_measured | managed | 2.154x |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.000288 ms | not_measured | managed | 1.212x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.00437 ms | not_measured | managed | 2.128x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | 0.00796 ms | not_measured | managed | 0.716x |
| `np.log (float16)` | float16 | 1,000 | 0.00664 ms | not_measured | managed | 0.679x |
| `np.log (float16)` | float16 | 100,000 | 0.6183 ms | not_measured | managed | 0.676x |
| `np.log (float16)` | float16 | 10,000,000 | 61.4499 ms | not_measured | managed | 0.728x |
| `np.log (float32)` | float32 | 1,000 | 0.00194 ms | not_measured | managed | 0.627x |
| `np.log (float32)` | float32 | 100,000 | 0.06067 ms | not_measured | managed | 1.398x |
| `np.log (float32)` | float32 | 10,000,000 | 6.3135 ms | not_measured | managed | 2.038x |
| `np.log (float64)` | float64 | 1,000 | 0.00279 ms | not_measured | managed | 0.945x |
| `np.log (float64)` | float64 | 100,000 | 0.2503 ms | not_measured | managed | 0.918x |
| `np.log (float64)` | float64 | 10,000,000 | 30.3537 ms | not_measured | managed | 1.004x |
| `np.log10 (float16)` | float16 | 1,000 | 0.00647 ms | not_measured | managed | 0.723x |
| `np.log10 (float16)` | float16 | 100,000 | 0.6310 ms | not_measured | managed | 0.694x |
| `np.log10 (float16)` | float16 | 10,000,000 | 62.8310 ms | not_measured | managed | 0.744x |
| `np.log10 (float32)` | float32 | 1,000 | 0.00407 ms | not_measured | managed | 0.567x |
| `np.log10 (float32)` | float32 | 100,000 | 0.4581 ms | not_measured | managed | 0.413x |
| `np.log10 (float32)` | float32 | 10,000,000 | 26.1176 ms | not_measured | managed | 0.866x |
| `np.log10 (float64)` | float64 | 1,000 | 0.00290 ms | not_measured | managed | 0.956x |
| `np.log10 (float64)` | float64 | 100,000 | 0.2761 ms | not_measured | managed | 0.872x |
| `np.log10 (float64)` | float64 | 10,000,000 | 30.5373 ms | not_measured | managed | 1.046x |
| `np.log1p (float16)` | float16 | 1,000 | 0.00788 ms | not_measured | managed | 0.727x |
| `np.log1p (float16)` | float16 | 100,000 | 0.8440 ms | not_measured | managed | 0.613x |
| `np.log1p (float16)` | float16 | 10,000,000 | 77.3898 ms | not_measured | managed | 0.772x |
| `np.log1p (float32)` | float32 | 1,000 | 0.00488 ms | not_measured | managed | 0.641x |
| `np.log1p (float32)` | float32 | 100,000 | 0.4312 ms | not_measured | managed | 0.636x |
| `np.log1p (float32)` | float32 | 10,000,000 | 22.1146 ms | not_measured | managed | 1.429x |
| `np.log1p (float64)` | float64 | 1,000 | 0.00311 ms | not_measured | managed | 1.139x |
| `np.log1p (float64)` | float64 | 100,000 | 0.2698 ms | not_measured | managed | 1.167x |
| `np.log1p (float64)` | float64 | 10,000,000 | 33.5946 ms | not_measured | managed | 1.209x |
| `np.log2 (float16)` | float16 | 1,000 | 0.00640 ms | not_measured | managed | 0.773x |
| `np.log2 (float16)` | float16 | 100,000 | 0.6310 ms | not_measured | managed | 0.707x |
| `np.log2 (float16)` | float16 | 10,000,000 | 62.4133 ms | not_measured | managed | 0.768x |
| `np.log2 (float32)` | float32 | 1,000 | 0.00231 ms | not_measured | managed | 0.960x |
| `np.log2 (float32)` | float32 | 100,000 | 0.4662 ms | not_measured | managed | 0.400x |
| `np.log2 (float32)` | float32 | 10,000,000 | 21.0786 ms | not_measured | managed | 1.059x |
| `np.log2 (float64)` | float64 | 1,000 | 0.00434 ms | not_measured | managed | 0.950x |
| `np.log2 (float64)` | float64 | 100,000 | 0.3843 ms | not_measured | managed | 0.978x |
| `np.log2 (float64)` | float64 | 10,000,000 | 46.3495 ms | not_measured | managed | 0.970x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.00145 ms | not_measured | managed | 1.752x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.09117 ms | not_measured | managed | 3.029x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 10.9921 ms | not_measured | managed | 3.344x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.00193 ms | not_measured | managed | 1.213x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.2457 ms | not_measured | managed | 1.944x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 28.3181 ms | not_measured | managed | 1.643x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.000616 ms | not_measured | managed | 0.987x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.05276 ms | not_measured | managed | 0.491x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 5.8724 ms | not_measured | managed | 0.784x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.000676 ms | not_measured | managed | 0.587x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.02267 ms | not_measured | managed | 0.283x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 5.0621 ms | not_measured | managed | 1.503x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.00114 ms | not_measured | managed | 0.403x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.04104 ms | not_measured | managed | 0.266x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 19.5469 ms | not_measured | managed | 0.774x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.000736 ms | not_measured | managed | 0.757x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.01103 ms | not_measured | managed | 1.884x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 1.5234 ms | not_measured | managed | 2.727x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.000832 ms | not_measured | managed | 0.644x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.02223 ms | not_measured | managed | 0.852x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 4.7927 ms | not_measured | managed | 1.579x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.00120 ms | not_measured | managed | 0.448x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.03990 ms | not_measured | managed | 0.475x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 13.1740 ms | not_measured | managed | 1.108x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.00478 ms | not_measured | managed | 1.786x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.3730 ms | not_measured | managed | 2.122x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 33.2954 ms | not_measured | managed | 2.488x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.00211 ms | not_measured | managed | 0.857x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.01921 ms | not_measured | managed | 6.209x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 8.3428 ms | not_measured | managed | 1.888x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.00250 ms | not_measured | managed | 0.678x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.06031 ms | not_measured | managed | 1.981x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 20.9569 ms | not_measured | managed | 0.928x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.00626 ms | not_measured | managed | 1.601x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.4681 ms | not_measured | managed | 2.200x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 70.1896 ms | not_measured | managed | 1.534x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.00190 ms | not_measured | managed | 1.120x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.02184 ms | not_measured | managed | 6.837x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 4.9489 ms | not_measured | managed | 3.795x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.00266 ms | not_measured | managed | 0.797x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.04230 ms | not_measured | managed | 3.515x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 20.5795 ms | not_measured | managed | 1.088x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.01826 ms | not_measured | managed | 0.627x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 2.7693 ms | not_measured | managed | 0.524x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 348.0324 ms | not_measured | managed | 0.436x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.01352 ms | not_measured | managed | 0.406x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 0.6545 ms | not_measured | managed | 1.002x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 93.6525 ms | not_measured | managed | 0.758x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.02271 ms | not_measured | managed | 0.408x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 1.0693 ms | not_measured | managed | 0.968x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 190.9802 ms | not_measured | managed | 0.591x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.00502 ms | not_measured | managed | 0.730x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.3616 ms | not_measured | managed | 0.901x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 59.5498 ms | not_measured | managed | 0.628x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.000720 ms | not_measured | managed | 1.493x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.01287 ms | not_measured | managed | 5.710x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 3.1423 ms | not_measured | managed | 3.546x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.00113 ms | not_measured | managed | 1.111x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.02591 ms | not_measured | managed | 3.538x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 16.4597 ms | not_measured | managed | 1.058x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.00697 ms | not_measured | managed | 0.536x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.3898 ms | not_measured | managed | 0.849x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 49.9933 ms | not_measured | managed | 0.737x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.000947 ms | not_measured | managed | 1.132x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.01840 ms | not_measured | managed | 3.990x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 3.8200 ms | not_measured | managed | 2.936x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.00115 ms | not_measured | managed | 1.093x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.02624 ms | not_measured | managed | 3.493x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 14.1630 ms | not_measured | managed | 1.256x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 100,000 | 0.000001 ms | not_measured | managed | 144.000x |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 100,000 | 0.000001 ms | not_measured | managed | 143.000x |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.000001 ms | not_measured | managed | 149.000x |
| `np.real(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 145.000x |
| `np.real(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 143.000x |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 144.000x |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.00695 ms | not_measured | managed | 0.343x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.5701 ms | not_measured | managed | 0.354x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 43.4468 ms | not_measured | managed | 0.511x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.000810 ms | not_measured | managed | 0.586x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.06195 ms | not_measured | managed | 0.228x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 8.4471 ms | not_measured | managed | 0.828x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.00193 ms | not_measured | managed | 0.364x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.1814 ms | not_measured | managed | 0.204x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 23.5760 ms | not_measured | managed | 0.600x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.00407 ms | not_measured | managed | 1.161x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.4927 ms | not_measured | managed | 0.916x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 74.1100 ms | not_measured | managed | 0.655x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.000893 ms | not_measured | managed | 0.446x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.01330 ms | not_measured | managed | 0.419x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 3.1186 ms | not_measured | managed | 2.264x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.000819 ms | not_measured | managed | 0.534x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.03981 ms | not_measured | managed | 0.270x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 13.6558 ms | not_measured | managed | 1.041x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.00417 ms | not_measured | managed | 1.313x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.4872 ms | not_measured | managed | 0.935x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 61.6779 ms | not_measured | managed | 0.786x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.000908 ms | not_measured | managed | 1.047x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.02150 ms | not_measured | managed | 0.287x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 6.0918 ms | not_measured | managed | 1.189x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.00123 ms | not_measured | managed | 0.807x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.02845 ms | not_measured | managed | 0.403x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 12.7899 ms | not_measured | managed | 1.106x |
| `np.sign (float16)` | float16 | 1,000 | 0.00367 ms | not_measured | managed | 0.328x |
| `np.sign (float16)` | float16 | 100,000 | 0.6314 ms | not_measured | managed | 0.133x |
| `np.sign (float16)` | float16 | 10,000,000 | 64.8190 ms | not_measured | managed | 0.158x |
| `np.sign (float32)` | float32 | 1,000 | 0.00176 ms | not_measured | managed | 0.576x |
| `np.sign (float32)` | float32 | 100,000 | 0.3746 ms | not_measured | managed | 0.758x |
| `np.sign (float32)` | float32 | 10,000,000 | 38.8994 ms | not_measured | managed | 0.918x |
| `np.sign (float64)` | float64 | 1,000 | 0.00183 ms | not_measured | managed | 0.523x |
| `np.sign (float64)` | float64 | 100,000 | 0.3742 ms | not_measured | managed | 0.766x |
| `np.sign (float64)` | float64 | 10,000,000 | 44.7712 ms | not_measured | managed | 0.885x |
| `np.sin (float16)` | float16 | 1,000 | 0.01043 ms | not_measured | managed | 0.624x |
| `np.sin (float16)` | float16 | 100,000 | 1.2157 ms | not_measured | managed | 0.808x |
| `np.sin (float16)` | float16 | 10,000,000 | 182.4185 ms | not_measured | managed | 0.566x |
| `np.sin (float32)` | float32 | 1,000 | 0.00106 ms | not_measured | managed | 1.015x |
| `np.sin (float32)` | float32 | 100,000 | 0.2722 ms | not_measured | managed | 0.257x |
| `np.sin (float32)` | float32 | 10,000,000 | 6.7534 ms | not_measured | managed | 1.668x |
| `np.sin (float64)` | float64 | 1,000 | 0.00415 ms | not_measured | managed | 1.031x |
| `np.sin (float64)` | float64 | 100,000 | 0.9992 ms | not_measured | managed | 0.693x |
| `np.sin (float64)` | float64 | 10,000,000 | 108.3373 ms | not_measured | managed | 0.732x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.01058 ms | not_measured | managed | 0.724x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 1.1781 ms | not_measured | managed | 0.808x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 144.9162 ms | not_measured | managed | 0.680x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.00421 ms | not_measured | managed | 0.866x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 0.6415 ms | not_measured | managed | 0.962x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 75.2938 ms | not_measured | managed | 0.907x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.00447 ms | not_measured | managed | 0.975x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 0.6402 ms | not_measured | managed | 0.953x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 68.5876 ms | not_measured | managed | 1.022x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.00349 ms | not_measured | managed | 1.043x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.3242 ms | not_measured | managed | 1.038x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 32.5379 ms | not_measured | managed | 1.182x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.000582 ms | not_measured | managed | 0.806x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.01559 ms | not_measured | managed | 0.904x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 5.0234 ms | not_measured | managed | 1.347x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.000992 ms | not_measured | managed | 0.888x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.05759 ms | not_measured | managed | 0.959x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 12.7684 ms | not_measured | managed | 1.168x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.00863 ms | not_measured | managed | 0.275x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.8311 ms | not_measured | managed | 0.243x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 69.3044 ms | not_measured | managed | 0.321x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.000810 ms | not_measured | managed | 0.489x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.02025 ms | not_measured | managed | 0.276x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 8.7167 ms | not_measured | managed | 0.785x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.00116 ms | not_measured | managed | 0.378x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.04349 ms | not_measured | managed | 0.249x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 19.3877 ms | not_measured | managed | 0.730x |
| `np.tan (float16)` | float16 | 1,000 | 0.01627 ms | not_measured | managed | 0.422x |
| `np.tan (float16)` | float16 | 100,000 | 1.1039 ms | not_measured | managed | 0.863x |
| `np.tan (float16)` | float16 | 10,000,000 | 184.6271 ms | not_measured | managed | 0.539x |
| `np.tan (float32)` | float32 | 1,000 | 0.00335 ms | not_measured | managed | 0.975x |
| `np.tan (float32)` | float32 | 100,000 | 1.1120 ms | not_measured | managed | 0.585x |
| `np.tan (float32)` | float32 | 10,000,000 | 94.8243 ms | not_measured | managed | 0.752x |
| `np.tan (float64)` | float64 | 1,000 | 0.00497 ms | not_measured | managed | 0.920x |
| `np.tan (float64)` | float64 | 100,000 | 0.8573 ms | not_measured | managed | 0.961x |
| `np.tan (float64)` | float64 | 10,000,000 | 131.1067 ms | not_measured | managed | 0.710x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.01084 ms | not_measured | managed | 0.904x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 1.0669 ms | not_measured | managed | 0.924x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 122.1519 ms | not_measured | managed | 0.833x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.00105 ms | not_measured | managed | 1.502x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.07109 ms | not_measured | managed | 1.716x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 6.7684 ms | not_measured | managed | 2.393x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.00559 ms | not_measured | managed | 1.098x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.3136 ms | not_measured | managed | 1.845x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 35.9644 ms | not_measured | managed | 1.840x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.00630 ms | not_measured | managed | 0.652x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.5686 ms | not_measured | managed | 0.648x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 51.2869 ms | not_measured | managed | 0.803x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.000914 ms | not_measured | managed | 0.439x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.02217 ms | not_measured | managed | 0.253x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 7.9759 ms | not_measured | managed | 0.919x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.00116 ms | not_measured | managed | 0.389x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.02529 ms | not_measured | managed | 0.427x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 19.2412 ms | not_measured | managed | 0.734x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-28 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.49× geomean · 67%🕐 of NumPy's time · 114 win / 51 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████████▏ ....   1.42×    70%🕐  ( 19 win / 14 lose)
1K         ███████████████▌ ...   1.56×    64%🕐  ( 24 win /  9 lose)
100K       ███████████████▌ ...   1.56×    64%🕐  ( 24 win /  9 lose)
1M         ████████████████▏ ..   1.62×    62%🕐  ( 23 win / 10 lose)
10M        █████████████▏ .....   1.32×    76%🕐  ( 24 win /  9 lose)
ALL        ██████████████▉ ....   1.49×    67%🕐  (114 win / 51 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise████████████▏ ......   1.22×    82%🕐  ( 35 win /  5 lose)
reductions ███████████████████▶   2.78×    36%🕐  ( 35 win /  5 lose)
selection  ███████████████▎ ...   1.53×    65%🕐  ( 20 win / 15 lose)
copy/cast  ██████████▎ ........   1.03×    97%🕐  ( 13 win / 12 lose)  ◄ PARITY
index-math ██████████▋ ........   1.06×    94%🕐  (  7 win /  3 lose)
dtypes     ██████████▋ ........   1.07×    93%🕐  (  4 win / 11 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.33×    1.51×    1.29×    1.10×    0.95×
reductions      7.14×    2.83×    2.31×    1.86×    1.92×
selection       0.64×    1.99×    2.17×    2.36×    1.30×
copy/cast       0.86×    0.80×    0.70×    1.79×    1.32×
index-math      0.87×    1.10×    1.07×    1.09×    1.21×
dtypes          0.48×    0.76×    1.99×    1.48×    1.32×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          0.99×    1.27×    1.16×    0.94×    1.01×     1.07×
  sqrt         1.01×    1.15×    1.00×    1.00×    0.23×     0.77×
  copy         1.50×    2.23×    1.25×    1.09×    1.77×     1.52×
  strided      1.56×    1.27×    1.07×    0.96×    1.02×     1.16×
  bcast        1.58×    1.39×    1.01×    1.02×    1.04×     1.18×
  reversed     1.54×    1.23×    1.55×    0.97×    1.37×     1.32×
  castbuf      1.29×    1.34×    1.89×    1.57×    1.07×     1.40×
  mixbuf       1.33×    2.84×    1.71×    1.44×    1.08×     1.59×
-- reductions
  sum          4.43×    3.53×    3.49×    2.08×    1.92×     2.94×
  sum ax0      8.84×    0.50×    1.06×    1.07×    0.97×     1.37×
  sum ax1      8.94×    1.17×    1.47×    1.59×    1.74×     2.12×
  sum dt=      4.75×    1.66×    1.10×    1.05×    1.09×     1.58×
  amin         8.83×    1.31×    0.80×    0.73×    0.92×     1.44×
  cumsum       3.11×    1.61×    1.96×    1.14×    1.60×     1.78×
  any(F)      12.15×   23.88×    3.47×    1.86×    1.44×     4.86×
  any(hit)    12.18×   23.89×   24.27×   24.25×   24.07×    21.04×
-- selection
  where        0.96×    1.25×    1.64×    1.93×    0.82×     1.25×
  a[mask]      0.26×    2.90×    4.52×    5.48×    1.85×     2.02×
  a[mask]=     0.22×    4.50×    7.78×    7.71×    3.89×     2.97×
  count_nz     0.67×    3.11×    4.05×    3.41×    1.03×     1.97×
  argwhere     3.36×    6.13×    2.09×    3.81×    2.85×     3.42×
  a[idx]       0.41×    0.76×    0.94×    0.65×    0.72×     0.67×
  a[idx]=      0.88×    0.52×    0.50×    0.59×    0.49×     0.58×
-- copy/cast
  flatten      1.11×    1.03×    0.58×    3.18×    0.86×     1.12×
  astype       0.32×    0.60×    1.60×    2.95×    1.54×     1.07×
  ravel.T      2.18×    1.65×    0.84×    1.81×    1.22×     1.46×
  in-place     0.77×    0.62×    0.87×    1.72×    1.03×     0.94×
  less->b      0.78×    0.52×    0.25×    0.63×    2.42×     0.69×
-- index-math
  unravel      0.55×    0.86×    1.03×    0.91×    1.06×     0.86×
  ravel_mi     1.37×    1.42×    1.11×    1.30×    1.39×     1.31×
-- dtypes
  complex      0.34×    0.52×    0.90×    0.84×    0.86×     0.65×
  float16      0.57×    0.61×    0.64×    0.64×    0.58×     0.61×
  int8         0.56×    1.41×   13.86×    6.11×    4.62×     3.15×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████▋ .......   1.17×    86%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.30×    23%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   3.35×    30%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.46×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.07×    48%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.31×    19%🕐  (  1 win /  0 lose)
4d           ██████████████████▊    1.89×    53%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.25×    44%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.24×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.74×    36%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████▍ ..............   0.44×   228%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████ .........   1.01×    99%🕐  (  1 win /  0 lose)  ◄ PARITY
w=64         ███████████▎ .......   1.13×    88%🕐  (  1 win /  0 lose)
w=256        ████████████▎ ......   1.23×    81%🕐  (  1 win /  0 lose)
w=1024       █████████████▊ .....   1.38×    72%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    839.13×   (839.1× faster, faster)
  allocate          1.09×   (1.1× faster, faster)
  overlap_copy      1.91×   (1.9× faster, faster)
  forder_out        1.14×   (1.1× faster, faster)
  zerodim           0.62×   (1.6× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           14.84×    3.94×    1.41×    1.69×    2.02×   vs chained 6× add
reuse            5.68×    4.91×    1.04×    1.07×    1.03×   vs rebuild each call
par8                 -    0.78×    3.19×    4.32×    2.94×   vs single-thread

biggest NumSharp wins: anyeh@100K 24.27× · anyeh@1M 24.25× · anyeh@10M 24.07× · anyeh@1K 23.89× · anyff@1K 23.88×
most behind:           bassign@1 0.22× · sqrt@10M 0.23× · lessbool@100K 0.25× · bread@1 0.26× · astype@1 0.32×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.02 ✅ | 1.16 ✅ | 1.10 ✅ | 0.65 🟡 | 1.05 ✅ | 0.77 🟡 | 0.92 🟡 | 0.84 🟡 |
| 1M | 1.03 ✅ | 1.03 ✅ | 1.12 ✅ | 0.58 🟡 | 0.94 🟡 | 0.70 🟡 | 0.84 🟡 | 0.69 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.08 ✅ | 1.30 ✅ | 0.98 🟡 | 0.13 🔴 | 1.08 ✅ | 1.46 ✅ | 1.25 ✅ |
| 1M | 1.01 ✅ | 1.24 ✅ | 0.96 🟡 | 0.11 🔴 | 1.05 ✅ | 1.30 ✅ | 1.07 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.99 🟡 | 0.76 🟡 | 0.78 🟡 | 1.31 ✅ |
| 1M | 0.93 🟡 | 0.65 🟡 | 0.70 🟡 | 1.30 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.3704 | 0.0627 | 0.01 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7278 | 0.0205 | 0.03 🔴 |
| 1M\|dec\|bcast\|min\|ax0 | 3.1558 | 0.0892 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.5109 | 0.2257 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.5254 | 0.2269 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.4183 | 0.2266 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7378 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7283 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7281 | 0.0228 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.4767 | 0.2384 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.3708 | 0.2435 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7189 | 0.0248 | 0.03 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.1325 | 0.1709 | 0.04 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.5507 | 0.0229 | 0.04 🔴 |
| 100K\|dec\|C\|max\|ax0 | 0.3266 | 0.0147 | 0.05 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 3.79 ✅ | 3.09 ✅ | 3.14 ✅ | 1.50 ✅ | 3.83 ✅ | 3.75 ✅ | 3.40 ✅ | 4.01 ✅ |
| 1M | 8.07 ✅ | 7.88 ✅ | 8.29 ✅ | 3.88 ✅ | 6.44 ✅ | 6.31 ✅ | 7.22 ✅ | 9.62 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 5.21 ✅ | 4.10 ✅ | 4.91 ✅ | 4.51 ✅ | 2.75 ✅ | 2.73 ✅ | 1.91 ✅ | 1.48 ✅ | 4.92 ✅ | 4.32 ✅ | 1.39 ✅ | 1.16 ✅ | 10.38 ✅ |
| 1M | 9.96 ✅ | 10.11 ✅ | 9.29 ✅ | 9.12 ✅ | 5.52 ✅ | 5.67 ✅ | 4.86 ✅ | 5.15 ✅ | 9.15 ✅ | 8.98 ✅ | 5.12 ✅ | 5.10 ✅ | 6.74 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|pos | 0.0287 | 0.0158 | 0.55 🟡 |
| 100K\|f64\|strided\|pos | 0.0317 | 0.0181 | 0.57 🟡 |
| 100K\|i32\|strided\|pos | 0.0288 | 0.0199 | 0.69 🟡 |
| 100K\|u32\|strided\|pos | 0.0288 | 0.0199 | 0.69 🟡 |
| 100K\|f64\|T\|pos | 0.0302 | 0.0278 | 0.92 🟡 |
| 100K\|i64\|strided\|pos | 0.0318 | 0.0299 | 0.94 🟡 |
| 100K\|u64\|strided\|pos | 0.0314 | 0.0300 | 0.96 🟡 |
| 100K\|f64\|C\|pos | 0.0291 | 0.0278 | 0.96 🟡 |
| 100K\|f64\|F\|pos | 0.0286 | 0.0278 | 0.97 🟡 |
| 100K\|u64\|T\|pos | 0.0334 | 0.0365 | 1.09 ✅ |
| 100K\|u64\|C\|pos | 0.0318 | 0.0365 | 1.15 ✅ |
| 100K\|u64\|F\|pos | 0.0304 | 0.0365 | 1.20 ✅ |
| 100K\|f32\|F\|pos | 0.0157 | 0.0189 | 1.20 ✅ |
| 100K\|i64\|F\|pos | 0.0292 | 0.0366 | 1.25 ✅ |
| 100K\|f32\|negcol\|pos | 0.0538 | 0.0734 | 1.36 ✅ |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.46 ✅ | 1.24 ✅ | 0.56 🟡 | 1.65 ✅ | 1.48 ✅ | 1.14 ✅ | 1.50 ✅ |
| 1M | 2.47 ✅ | 2.22 ✅ | 2.24 ✅ | 1.25 ✅ | 2.61 ✅ | 2.59 ✅ | 1.65 ✅ | 3.04 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 0.92 🟡 | 2.92 ✅ | 0.88 🟡 | 1.28 ✅ | 0.95 🟡 |
| 1M | 2.39 ✅ | 2.82 ✅ | 2.17 ✅ | 1.24 ✅ | 2.57 ✅ | 2.35 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 1.59 ✅ | 1.42 ✅ | 1.27 ✅ | 1.16 ✅ | 0.98 🟡 | 0.89 🟡 | 1.26 ✅ |
| 1M | 2.43 ✅ | 2.41 ✅ | 3.27 ✅ | 2.80 ✅ | 1.45 ✅ | 0.85 🟡 | 3.62 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|add | 0.1834 | 0.0257 | 0.14 🔴 |
| 100K\|i64\|strided\|mul | 0.1832 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|mul | 0.1832 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|neg | 0.0938 | 0.0166 | 0.18 🔴 |
| 100K\|f64\|strided\|abs | 0.0931 | 0.0168 | 0.18 🔴 |
| 100K\|f32\|strided\|abs | 0.0492 | 0.0099 | 0.20 🟠 |
| 100K\|i32\|strided\|sqrt | 0.1827 | 0.0419 | 0.23 🟠 |
| 100K\|i64\|strided\|sqrt | 0.1759 | 0.0408 | 0.23 🟠 |
| 100K\|i64\|C\|sqrt | 0.2797 | 0.0744 | 0.27 🟠 |
| 100K\|i32\|strided\|add | 0.0932 | 0.0279 | 0.30 🟠 |
| 100K\|i32\|strided\|mul | 0.0931 | 0.0279 | 0.30 🟠 |
| 100K\|f32\|strided\|add | 0.0930 | 0.0286 | 0.31 🟠 |
| 100K\|f32\|strided\|mul | 0.0931 | 0.0287 | 0.31 🟠 |
| 100K\|i32\|strided\|less | 0.0486 | 0.0168 | 0.35 🟠 |
| 100K\|c128\|C\|less | 0.1901 | 0.0685 | 0.36 🟠 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 5.06 ✅ | 5.24 ✅ | 0.72 🟡 | 4.23 ✅ | 2.85 ✅ | 1.58 ✅ | 2.67 ✅ |
| 1-D strided a[::2] | 3.54 ✅ | 3.96 ✅ | 0.66 🟡 | 0.56 🟡 | 0.50 🟠 | 1.78 ✅ | 1.29 ✅ |
| 1-D reversed a[::-1] | 4.14 ✅ | 4.07 ✅ | 0.68 🟡 | 0.51 🟡 | 0.46 🟠 | 1.65 ✅ | 1.28 ✅ |
| array + scalar | 4.76 ✅ | 5.55 ✅ | 0.72 🟡 | 3.89 ✅ | 2.66 ✅ | 1.59 ✅ | 2.61 ✅ |
| scalar + array | 4.75 ✅ | 7.03 ✅ | 0.73 🟡 | 4.34 ✅ | 2.83 ✅ | 1.60 ✅ | 2.80 ✅ |
| mixed C + F | 2.59 ✅ | 0.85 🟡 | 0.76 🟡 | 0.88 🟡 | 0.72 🟡 | 1.77 ✅ | 1.11 ✅ |
| mixed C + T | 3.16 ✅ | 0.84 🟡 | 0.75 🟡 | 0.85 🟡 | 0.72 🟡 | 2.25 ✅ | 1.18 ✅ |
| binary broadcast +row(1,C) | 5.02 ✅ | 3.79 ✅ | 0.73 🟡 | 4.36 ✅ | 3.31 ✅ | 1.78 ✅ | 2.66 ✅ |
| binary broadcast +col(R,1) | 5.22 ✅ | 4.28 ✅ | 0.48 🟠 | 4.65 ✅ | 3.37 ✅ | 2.80 ✅ | 2.79 ✅ |
| col-broadcast unary (inner stride-0) | 4.63 ✅ | 1.81 ✅ | 0.94 🟡 | 2.01 ✅ | 3.24 ✅ | 6.79 ✅ | 2.65 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_rev|i64 | 3.6472 | 1.6790 | 0.46 🟠 |
| bcast_col|f16 | 9.5752 | 4.6351 | 0.48 🟠 |
| 1d_strided|i64 | 1.8140 | 0.9057 | 0.50 🟠 |
| 1d_rev|i32 | 1.8061 | 0.9294 | 0.51 🟡 |
| 1d_strided|i32 | 0.9038 | 0.5030 | 0.56 🟡 |
| 1d_strided|f16 | 4.7716 | 3.1634 | 0.66 🟡 |
| 1d_rev|f16 | 9.6988 | 6.5489 | 0.68 🟡 |
| mix_C_F|i64 | 4.0741 | 2.9182 | 0.72 🟡 |
| scalar_rhs|f16 | 8.9542 | 6.4298 | 0.72 🟡 |
| 1d_C|f16 | 9.1140 | 6.5510 | 0.72 🟡 |
| mix_C_T|i64 | 3.7560 | 2.7030 | 0.72 🟡 |
| scalar_lhs|f16 | 8.9616 | 6.5114 | 0.73 🟡 |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **327 / 1568** comparable cells lag (<1.0); **1241** win (≥1.0).
- **🔴 <0.2** — 12 cells. Top: 7× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 2× same-type diagonal (copy); 2× int → sub-word (narrow)
- **🟠 0.2–0.5** — 146 cells. Top: 62× int → sub-word (narrow); 31× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 18× * → bool
- **🟡 0.5–1.0** — 169 cells. Top: 40× int → sub-word (narrow); 24× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 8× f16 → u64

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **1.77**, `f64`→narrow **1.21**, `f16`→narrow **3.72**, `c128`→narrow **0.96**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 3.55 ✅ | 3.50 ✅ | 3.47 ✅ | 3.25 ✅ | 3.21 ✅ | 1.88 ✅ | 1.75 ✅ | 0.48 🟠 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.13 ✅ | 3.46 ✅ | 3.42 ✅ | 3.84 ✅ | 3.86 ✅ | 2.14 ✅ | 2.28 ✅ | 1.52 ✅ | 1.49 ✅ | 3.31 ✅ | 2.48 ✅ | 1.71 ✅ | 1.41 ✅ | nan ? | 1.17 ✅ |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.02 ✅ | 1.79 ✅ | 1.78 ✅ | 2.47 ✅ | 2.46 ✅ | 2.64 ✅ | 2.07 ✅ | 2.71 ✅ | 2.30 ✅ | 2.57 ✅ | 1.77 ✅ | 2.19 ✅ | 2.73 ✅ | nan ? | 2.64 ✅ |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.60🟡 | 7.24✅ | 6.31✅ | 8.92✅ | 8.38✅ | 7.96✅ | 8.25✅ | 7.69✅ | 7.71✅ | 9.15✅ | 0.25🟠 | 8.82✅ | 8.02✅ | — | 6.05✅ |
| **u8** | 11.28✅ | 1.31✅ | 11.06✅ | 9.48✅ | 7.65✅ | 5.78✅ | 6.45✅ | 4.33✅ | 4.97✅ | 10.29✅ | 10.66✅ | 2.20✅ | 3.07✅ | — | 3.06✅ |
| **i8** | 11.21✅ | 7.64✅ | 1.25✅ | 9.40✅ | 8.54✅ | 5.90✅ | 6.16✅ | 3.85✅ | 4.08✅ | 8.46✅ | 10.62✅ | 4.90✅ | 2.96✅ | — | 7.47✅ |
| **i16** | 8.35✅ | 6.60✅ | 9.05✅ | 4.68✅ | 7.28✅ | 6.44✅ | 5.23✅ | 5.00✅ | 5.47✅ | 7.93✅ | 2.91✅ | 5.70✅ | 3.07✅ | — | 2.42✅ |
| **u16** | 8.39✅ | 9.39✅ | 9.63✅ | 7.38✅ | 4.54✅ | 5.68✅ | 6.10✅ | 5.02✅ | 5.60✅ | 5.06✅ | 2.91✅ | 6.83✅ | 3.37✅ | — | 3.59✅ |
| **i32** | 1.23✅ | 0.98🟡 | 0.88🟡 | 4.73✅ | 3.00✅ | 3.88✅ | 6.02✅ | 4.36✅ | 4.01✅ | 5.43✅ | 3.05✅ | 2.91✅ | 4.26✅ | — | 3.07✅ |
| **u32** | 6.05✅ | 1.47✅ | 0.90🟡 | 4.75✅ | 5.91✅ | 6.48✅ | 4.51✅ | 4.37✅ | 4.91✅ | 5.73✅ | 2.75✅ | 2.13✅ | 3.08✅ | — | 3.39✅ |
| **i64** | 1.19✅ | 1.41✅ | 0.85🟡 | 3.96✅ | 3.68✅ | 3.30✅ | 3.67✅ | 3.36✅ | 4.71✅ | 2.54✅ | 1.47✅ | 2.85✅ | 3.36✅ | — | 2.51✅ |
| **u64** | 1.19✅ | 1.01✅ | 1.32✅ | 1.65✅ | 1.85✅ | 3.31✅ | 3.41✅ | 4.94✅ | 4.06✅ | 2.55✅ | 1.83✅ | 1.67✅ | 2.12✅ | — | 2.20✅ |
| **char** | 8.35✅ | 9.80✅ | 9.97✅ | 8.21✅ | 3.85✅ | 2.72✅ | 2.72✅ | 4.89✅ | 4.61✅ | 4.31✅ | 2.92✅ | 2.15✅ | 3.24✅ | — | 2.34✅ |
| **f16** | 17.20✅ | 6.31✅ | 4.65✅ | 5.45✅ | 5.96✅ | 7.21✅ | 4.67✅ | 3.52✅ | 0.79🟡 | 5.96✅ | 4.48✅ | 3.43✅ | 1.00✅ | — | 1.22✅ |
| **f32** | 4.94✅ | 2.75✅ | 2.71✅ | 5.53✅ | 5.67✅ | 5.22✅ | 0.77🟡 | 1.04✅ | 0.67🟡 | 5.71✅ | 3.13✅ | 4.32✅ | 5.05✅ | — | 3.26✅ |
| **f64** | 2.25✅ | 2.54✅ | 2.58✅ | 3.53✅ | 2.90✅ | 3.38✅ | 1.07✅ | 1.09✅ | 0.72🟡 | 3.06✅ | 1.75✅ | 3.59✅ | 3.91✅ | — | 2.49✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.88🟡 | 1.50✅ | 1.50✅ | 2.06✅ | 2.16✅ | 2.43✅ | 1.26✅ | 1.33✅ | 0.76🟡 | 2.11✅ | 1.54✅ | 1.90✅ | 2.36✅ | — | 3.11✅ |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 4.54✅ | 18.39✅ | 17.95✅ | 9.58✅ | 8.51✅ | 6.39✅ | 7.31✅ | 5.33✅ | 4.53✅ | 10.30✅ | 0.31🟠 | 7.13✅ | 5.56✅ | — | 3.16✅ |
| **u8** | 11.14✅ | 0.71🟡 | 11.14✅ | 9.71✅ | 10.75✅ | 6.12✅ | 6.24✅ | 4.19✅ | 4.49✅ | 10.72✅ | 2.91✅ | 2.21✅ | 3.57✅ | — | 3.36✅ |
| **i8** | 10.66✅ | 13.40✅ | 1.86✅ | 10.36✅ | 6.51✅ | 6.66✅ | 6.39✅ | 3.30✅ | 4.33✅ | 8.67✅ | 2.91✅ | 2.20✅ | 3.48✅ | — | 3.12✅ |
| **i16** | 8.16✅ | 6.62✅ | 8.87✅ | 3.63✅ | 7.40✅ | 5.94✅ | 5.75✅ | 4.83✅ | 5.23✅ | 7.31✅ | 2.90✅ | 6.56✅ | 3.48✅ | — | 2.97✅ |
| **u16** | 8.40✅ | 10.59✅ | 9.52✅ | 6.84✅ | 4.69✅ | 5.67✅ | 6.17✅ | 5.24✅ | 5.70✅ | 4.44✅ | 2.91✅ | 6.87✅ | 3.54✅ | — | 3.57✅ |
| **i32** | 3.28✅ | 2.75✅ | 0.90🟡 | 4.83✅ | 5.90✅ | 3.95✅ | 5.95✅ | 4.09✅ | 3.88✅ | 5.20✅ | 3.07✅ | 2.45✅ | 4.23✅ | — | 2.28✅ |
| **u32** | 3.34✅ | 1.47✅ | 0.90🟡 | 4.83✅ | 5.97✅ | 6.41✅ | 4.48✅ | 8.85✅ | 9.01✅ | 5.86✅ | 2.75✅ | 2.14✅ | 3.42✅ | — | 3.12✅ |
| **i64** | 1.19✅ | 1.39✅ | 0.89🟡 | 1.86✅ | 1.83✅ | 3.22✅ | 3.63✅ | 3.24✅ | 4.41✅ | 2.57✅ | 1.46✅ | 2.83✅ | 3.62✅ | — | 2.72✅ |
| **u64** | 1.19✅ | 1.03✅ | 1.31✅ | 1.60✅ | 1.86✅ | 3.36✅ | 3.46✅ | 4.73✅ | 3.79✅ | 2.62✅ | 1.83✅ | 1.67✅ | 2.12✅ | — | 5.73✅ |
| **char** | 7.96✅ | 9.11✅ | 9.36✅ | 6.77✅ | 4.62✅ | 2.52✅ | 1.93✅ | 4.59✅ | 4.39✅ | 4.37✅ | 2.91✅ | 2.22✅ | 3.52✅ | — | 3.92✅ |
| **f16** | 17.24✅ | 4.33✅ | 4.65✅ | 5.58✅ | 5.95✅ | 7.18✅ | 1.58✅ | 3.50✅ | 0.79🟡 | 5.95✅ | 4.36✅ | 3.23✅ | 1.04✅ | — | 1.21✅ |
| **f32** | 4.94✅ | 2.69✅ | 2.72✅ | 3.25✅ | 3.23✅ | 4.61✅ | 0.80🟡 | 0.93🟡 | 0.58🟡 | 4.99✅ | 3.14✅ | 3.33✅ | 4.80✅ | — | 2.98✅ |
| **f64** | 2.23✅ | 2.54✅ | 2.53✅ | 3.60✅ | 3.13✅ | 3.86✅ | 1.12✅ | 1.13✅ | 0.71🟡 | 3.10✅ | 1.74✅ | 3.76✅ | 3.83✅ | — | 2.51✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.87🟡 | 1.52✅ | 1.44✅ | 2.18✅ | 2.09✅ | 2.33✅ | 1.17✅ | 1.23✅ | 0.74🟡 | 2.05✅ | 1.53✅ | 1.89✅ | 2.35✅ | — | 3.03✅ |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.10✅ | 10.00✅ | 11.07✅ | 9.48✅ | 10.36✅ | 6.63✅ | 7.31✅ | 5.78✅ | 6.13✅ | 10.69✅ | 0.31🟠 | 7.62✅ | 5.92✅ | — | 3.24✅ |
| **u8** | 13.87✅ | 1.11✅ | 11.10✅ | 10.27✅ | 8.25✅ | 6.12✅ | 6.72✅ | 4.96✅ | 4.18✅ | 10.48✅ | 2.90✅ | 2.20✅ | 3.59✅ | — | 3.21✅ |
| **i8** | 11.04✅ | 9.46✅ | 1.77✅ | 10.23✅ | 6.91✅ | 4.63✅ | 6.58✅ | 3.92✅ | 3.77✅ | 8.80✅ | 2.90✅ | 2.20✅ | 3.30✅ | — | 2.86✅ |
| **i16** | 8.41✅ | 6.71✅ | 9.55✅ | 3.92✅ | 7.44✅ | 5.85✅ | 5.96✅ | 5.67✅ | 5.08✅ | 7.23✅ | 2.91✅ | 6.72✅ | 3.56✅ | — | 2.80✅ |
| **u16** | 8.41✅ | 10.56✅ | 10.60✅ | 6.42✅ | 4.96✅ | 5.69✅ | 6.15✅ | 5.14✅ | 5.49✅ | 4.34✅ | 2.91✅ | 6.74✅ | 3.42✅ | — | 3.18✅ |
| **i32** | 3.25✅ | 1.03✅ | 0.90🟡 | 4.15✅ | 5.73✅ | 3.78✅ | 5.97✅ | 4.10✅ | 4.04✅ | 5.68✅ | 3.06✅ | 2.92✅ | 4.28✅ | — | 3.31✅ |
| **u32** | 3.28✅ | 1.41✅ | 0.90🟡 | 4.28✅ | 5.85✅ | 6.04✅ | 3.59✅ | 4.44✅ | 4.77✅ | 5.83✅ | 2.74✅ | 2.14✅ | 3.23✅ | — | 2.82✅ |
| **i64** | 1.20✅ | 1.43✅ | 0.87🟡 | 1.99✅ | 1.85✅ | 3.49✅ | 3.72✅ | 3.20✅ | 4.78✅ | 2.56✅ | 1.47✅ | 2.80✅ | 3.26✅ | — | 2.37✅ |
| **u64** | 1.18✅ | 1.03✅ | 1.26✅ | 1.60✅ | 1.88✅ | 3.49✅ | 3.46✅ | 5.16✅ | 3.63✅ | 2.58✅ | 1.83✅ | 1.66✅ | 2.10✅ | — | 2.20✅ |
| **char** | 8.35✅ | 9.59✅ | 10.70✅ | 6.26✅ | 4.40✅ | 2.78✅ | 2.80✅ | 4.66✅ | 5.19✅ | 4.51✅ | 2.90✅ | 2.20✅ | 3.54✅ | — | 3.67✅ |
| **f16** | 17.22✅ | 4.33✅ | 4.65✅ | 5.36✅ | 5.96✅ | 7.37✅ | 1.59✅ | 3.52✅ | 0.79🟡 | 5.76✅ | 4.68✅ | 3.43✅ | 1.06✅ | — | 1.25✅ |
| **f32** | 4.88✅ | 2.71✅ | 2.72✅ | 5.56✅ | 5.51✅ | 4.67✅ | 0.80🟡 | 0.97🟡 | 0.67🟡 | 5.39✅ | 3.13✅ | 4.00✅ | 4.85✅ | — | 3.17✅ |
| **f64** | 2.20✅ | 2.53✅ | 2.50✅ | 3.36✅ | 3.08✅ | 3.69✅ | 1.10✅ | 1.23✅ | 0.66🟡 | 3.08✅ | 1.71✅ | 3.56✅ | 3.86✅ | — | 2.39✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.89🟡 | 1.53✅ | 1.51✅ | 2.17✅ | 2.08✅ | 2.29✅ | 1.24✅ | 1.23✅ | 0.75🟡 | 2.11✅ | 1.52✅ | 1.91✅ | 2.31✅ | — | 3.11✅ |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 1.44✅ | 0.90🟡 | 1.05✅ | 1.28✅ | 1.44✅ | 1.38✅ | 2.01✅ | 3.17✅ | 3.29✅ | 1.47✅ | 0.30🟠 | 2.24✅ | 3.48✅ | — | 3.70✅ |
| **u8** | 8.27✅ | 1.57✅ | 11.29✅ | 10.18✅ | 9.71✅ | 6.06✅ | 6.38✅ | 3.44✅ | 3.54✅ | 10.23✅ | 2.84✅ | 2.20✅ | 3.53✅ | — | 2.96✅ |
| **i8** | 8.88✅ | 8.06✅ | 1.17✅ | 9.58✅ | 6.81✅ | 6.03✅ | 6.54✅ | 3.30✅ | 3.54✅ | 8.33✅ | 10.15✅ | 2.20✅ | 3.53✅ | — | 2.83✅ |
| **i16** | 7.95✅ | 6.24✅ | 11.60✅ | 4.36✅ | 7.64✅ | 6.13✅ | 5.34✅ | 4.80✅ | 5.33✅ | 10.73✅ | 2.83✅ | 6.19✅ | 3.71✅ | — | 2.77✅ |
| **u16** | 7.99✅ | 8.91✅ | 8.88✅ | 6.97✅ | 6.23✅ | 5.58✅ | 6.03✅ | 5.47✅ | 5.87✅ | 6.08✅ | 2.84✅ | 6.62✅ | 4.62✅ | — | 3.35✅ |
| **i32** | 3.00✅ | 3.05✅ | 3.08✅ | 4.57✅ | 5.52✅ | 4.57✅ | 4.62✅ | 4.86✅ | 4.98✅ | 5.37✅ | 2.97✅ | 1.99✅ | 4.57✅ | — | 2.78✅ |
| **u32** | 3.15✅ | 4.22✅ | 3.03✅ | 4.54✅ | 5.61✅ | 4.74✅ | 4.55✅ | 4.61✅ | 5.18✅ | 5.54✅ | 2.68✅ | 2.67✅ | 4.40✅ | — | 3.27✅ |
| **i64** | 1.14✅ | 1.76✅ | 1.22✅ | 2.74✅ | 2.47✅ | 3.15✅ | 3.47✅ | 3.45✅ | 3.71✅ | 2.47✅ | 1.45✅ | 2.07✅ | 3.16✅ | — | 2.69✅ |
| **u64** | 1.15✅ | 1.21✅ | 1.75✅ | 2.25✅ | 2.47✅ | 3.13✅ | 3.35✅ | 3.89✅ | 3.95✅ | 2.43✅ | 1.78✅ | 1.55✅ | 1.94✅ | — | 2.26✅ |
| **char** | 7.99✅ | 9.86✅ | 9.91✅ | 7.75✅ | 4.43✅ | 2.69✅ | 2.75✅ | 7.18✅ | 4.99✅ | 6.05✅ | 2.83✅ | 2.17✅ | 3.50✅ | — | 2.65✅ |
| **f16** | 16.63✅ | 4.14✅ | 4.36✅ | 2.92✅ | 5.51✅ | 6.44✅ | 1.60✅ | 3.31✅ | 0.79🟡 | 5.46✅ | 4.36✅ | 3.28✅ | 1.01✅ | — | 1.26✅ |
| **f32** | 9.82✅ | 2.38✅ | 2.35✅ | 4.89✅ | 4.86✅ | 4.60✅ | 0.79🟡 | 0.94🟡 | 0.67🟡 | 4.76✅ | 3.06✅ | 4.55✅ | 4.92✅ | — | 3.11✅ |
| **f64** | 2.04✅ | 2.10✅ | 2.08✅ | 3.09✅ | 2.70✅ | 3.37✅ | 1.10✅ | 1.11✅ | 0.64🟡 | 2.59✅ | 1.71✅ | 3.41✅ | 3.69✅ | — | 2.49✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.87🟡 | 1.43✅ | 1.36✅ | 2.06✅ | 2.00✅ | 2.42✅ | 1.20✅ | 1.27✅ | 0.75🟡 | 1.99✅ | 1.51✅ | 1.94✅ | 2.48✅ | — | 1.93✅ |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 1.39✅ | 0.90🟡 | 1.04✅ | 1.43✅ | 1.45✅ | 1.94✅ | 1.96✅ | 3.34✅ | 3.40✅ | 1.46✅ | 0.31🟠 | 2.23✅ | 3.48✅ | — | 3.32✅ |
| **u8** | 8.50✅ | 1.56✅ | 10.98✅ | 5.89✅ | 10.41✅ | 6.61✅ | 6.69✅ | 3.08✅ | 3.01✅ | 10.27✅ | 2.82✅ | 2.19✅ | 3.48✅ | — | 3.26✅ |
| **i8** | 8.52✅ | 7.75✅ | 1.42✅ | 10.65✅ | 6.65✅ | 6.43✅ | 6.28✅ | 2.34✅ | 3.50✅ | 8.30✅ | 2.82✅ | 2.19✅ | 3.41✅ | — | 2.77✅ |
| **i16** | 7.43✅ | 7.31✅ | 10.22✅ | 5.49✅ | 7.72✅ | 5.56✅ | 5.80✅ | 4.52✅ | 10.08✅ | 7.79✅ | 9.60✅ | 5.95✅ | 3.60✅ | — | 3.74✅ |
| **u16** | 7.48✅ | 9.31✅ | 9.38✅ | 7.23✅ | 6.25✅ | 5.61✅ | 5.99✅ | 4.92✅ | 5.43✅ | 6.31✅ | 2.81✅ | 6.73✅ | 3.99✅ | — | 3.55✅ |
| **i32** | 2.85✅ | 2.87✅ | 2.88✅ | 3.96✅ | 5.56✅ | 4.14✅ | 4.72✅ | 3.90✅ | 4.04✅ | 5.59✅ | 2.92✅ | 1.99✅ | 3.93✅ | — | 3.14✅ |
| **u32** | 2.85✅ | 3.93✅ | 2.93✅ | 4.19✅ | 5.60✅ | 4.50✅ | 4.37✅ | 8.31✅ | 4.77✅ | 5.45✅ | 2.59✅ | 2.44✅ | 3.73✅ | — | 3.09✅ |
| **i64** | 1.15✅ | 1.69✅ | 1.24✅ | 2.91✅ | 2.48✅ | 3.45✅ | 3.50✅ | 3.27✅ | 3.40✅ | 2.25✅ | 1.45✅ | 1.89✅ | 2.67✅ | — | 2.63✅ |
| **u64** | 1.15✅ | 1.23✅ | 1.67✅ | 2.50✅ | 2.46✅ | 3.54✅ | 3.34✅ | 3.59✅ | 3.72✅ | 2.48✅ | 1.77✅ | 1.51✅ | 1.88✅ | — | 2.11✅ |
| **char** | 7.45✅ | 9.18✅ | 9.36✅ | 7.11✅ | 4.78✅ | 2.55✅ | 2.69✅ | 4.32✅ | 4.67✅ | 6.12✅ | 2.82✅ | 2.19✅ | 3.50✅ | — | 3.37✅ |
| **f16** | 15.61✅ | 2.10✅ | 4.26✅ | 5.16✅ | 5.37✅ | 7.09✅ | 1.63✅ | 3.60✅ | 0.80🟡 | 5.18✅ | 4.64✅ | 3.22✅ | 0.95🟡 | — | 1.22✅ |
| **f32** | 4.21✅ | 2.23✅ | 2.27✅ | 4.67✅ | 4.72✅ | 4.93✅ | 0.79🟡 | 1.02✅ | 0.67🟡 | 4.40✅ | 3.02✅ | 4.44✅ | 4.53✅ | — | 3.15✅ |
| **f64** | 2.03✅ | 2.11✅ | 2.21✅ | 3.45✅ | 2.81✅ | 3.58✅ | 1.09✅ | 1.15✅ | 0.71🟡 | 2.82✅ | 1.66✅ | 3.04✅ | 3.22✅ | — | 2.59✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.88🟡 | 1.37✅ | 1.40✅ | 2.06✅ | 3.83✅ | 2.44✅ | 1.23✅ | 1.28✅ | 0.73🟡 | 2.02✅ | 1.50✅ | 1.90✅ | 2.31✅ | — | 1.84✅ |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.40🟠 | 0.66🟡 | 0.90🟡 | 1.46✅ | 1.43✅ | 1.96✅ | 1.93✅ | 3.27✅ | 3.33✅ | 1.42✅ | 0.31🟠 | 2.25✅ | 3.51✅ | — | 3.04✅ |
| **u8** | 1.07✅ | 8.57✅ | 7.30✅ | 10.38✅ | 7.94✅ | 1.95✅ | 2.58✅ | 4.56✅ | 4.19✅ | 10.36✅ | 2.07✅ | 2.98✅ | 3.85✅ | — | 2.98✅ |
| **i8** | 1.07✅ | 10.22✅ | 8.69✅ | 8.58✅ | 7.91✅ | 2.29✅ | 2.48✅ | 3.45✅ | 4.10✅ | 10.30✅ | 2.04✅ | 3.90✅ | 3.68✅ | — | 2.99✅ |
| **i16** | 3.80✅ | 10.13✅ | 3.64✅ | 9.68✅ | 9.25✅ | 2.48✅ | 2.48✅ | 4.08✅ | 4.32✅ | 10.05✅ | 2.14✅ | 2.34✅ | 3.82✅ | — | 3.17✅ |
| **u16** | 3.85✅ | 5.18✅ | 5.21✅ | 8.43✅ | 9.56✅ | 2.88✅ | 2.89✅ | 4.56✅ | 4.91✅ | 9.79✅ | 1.11✅ | 2.21✅ | 3.59✅ | — | 3.25✅ |
| **i32** | 0.47🟠 | 0.55🟡 | 0.55🟡 | 0.87🟡 | 0.74🟡 | 1.93✅ | 2.09✅ | 3.38✅ | 3.19✅ | 0.73🟡 | 2.20✅ | 2.56✅ | 2.97✅ | — | 3.04✅ |
| **u32** | 0.48🟠 | 0.41🟠 | 0.55🟡 | 0.74🟡 | 0.87🟡 | 1.92✅ | 2.03✅ | 4.15✅ | 4.62✅ | 0.87🟡 | 1.22✅ | 2.24✅ | 3.78✅ | — | 3.09✅ |
| **i64** | 0.40🟠 | 0.39🟠 | 0.17🔴 | 0.60🟡 | 0.52🟡 | 0.82🟡 | 0.84🟡 | 2.87✅ | 2.87✅ | 0.51🟡 | 1.24✅ | 2.53✅ | 3.07✅ | — | 2.49✅ |
| **u64** | 0.39🟠 | 0.39🟠 | 0.39🟠 | 0.58🟡 | 0.60🟡 | 0.76🟡 | 0.84🟡 | 3.22✅ | 2.84✅ | 0.61🟡 | 1.49✅ | 1.31✅ | 2.14✅ | — | 2.23✅ |
| **char** | 3.84✅ | 5.20✅ | 5.21✅ | 7.37✅ | 7.78✅ | 2.67✅ | 2.90✅ | 4.32✅ | 4.86✅ | 9.88✅ | 2.14✅ | 2.19✅ | 3.49✅ | — | 2.97✅ |
| **f16** | 21.84✅ | 2.28✅ | 2.36✅ | 2.86✅ | 2.77✅ | 3.67✅ | 1.35✅ | 2.53✅ | 0.80🟡 | 2.77✅ | 8.66✅ | 2.14✅ | 0.94🟡 | — | 1.18✅ |
| **f32** | 0.43🟠 | 0.41🟠 | 0.54🟡 | 0.68🟡 | 0.74🟡 | 1.06✅ | 0.47🟠 | 0.92🟡 | 0.54🟡 | 0.73🟡 | 2.24✅ | 1.90✅ | 2.50✅ | — | 3.12✅ |
| **f64** | 0.57🟡 | 0.30🟠 | 0.30🟠 | 0.62🟡 | 0.53🟡 | 2.26✅ | 0.62🟡 | 1.08✅ | 0.67🟡 | 0.52🟡 | 3.14✅ | 1.47✅ | 2.84✅ | — | 2.67✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 1.01✅ | 0.41🟠 | 0.41🟠 | 0.61🟡 | 0.65🟡 | 0.91🟡 | 0.76🟡 | 1.18✅ | 0.75🟡 | 0.67🟡 | 1.27✅ | 1.89✅ | 2.32✅ | — | 2.24✅ |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.83🟡 | 0.88🟡 | 0.88🟡 | 0.90🟡 | 0.87🟡 | 1.70✅ | 1.99✅ | 3.25✅ | 3.45✅ | 0.87🟡 | 0.28🟠 | 1.94✅ | 3.24✅ | — | 5.44✅ |
| **u8** | 0.98🟡 | 6.34✅ | 5.43✅ | 5.67✅ | 4.82✅ | 2.38✅ | 2.73✅ | 4.63✅ | 4.93✅ | 5.62✅ | 1.86✅ | 2.39✅ | 3.93✅ | — | 4.88✅ |
| **i8** | 0.98🟡 | 7.43✅ | 6.32✅ | 4.08✅ | 5.03✅ | 1.66✅ | 2.52✅ | 4.31✅ | 4.36✅ | 5.75✅ | 1.86✅ | 2.37✅ | 3.97✅ | — | 4.80✅ |
| **i16** | 2.91✅ | 3.85✅ | 2.79✅ | 5.30✅ | 5.31✅ | 2.37✅ | 2.55✅ | 3.33✅ | 4.50✅ | 5.30✅ | 1.92✅ | 2.37✅ | 4.02✅ | — | 5.18✅ |
| **u16** | 2.92✅ | 3.84✅ | 3.86✅ | 3.83✅ | 5.36✅ | 2.97✅ | 2.93✅ | 4.77✅ | 5.13✅ | 5.32✅ | 1.92✅ | 2.29✅ | 4.00✅ | — | 4.91✅ |
| **i32** | 0.46🟠 | 0.56🟡 | 0.56🟡 | 0.57🟡 | 0.39🟠 | 1.79✅ | 2.14✅ | 3.92✅ | 4.23✅ | 0.39🟠 | 1.99✅ | 2.42✅ | 3.28✅ | — | 3.97✅ |
| **u32** | 0.46🟠 | 0.39🟠 | 0.56🟡 | 0.39🟠 | 0.57🟡 | 2.08✅ | 2.09✅ | 4.11✅ | 4.51✅ | 0.57🟡 | 1.84✅ | 2.46✅ | 7.04✅ | — | 4.55✅ |
| **i64** | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.18🔴 | 0.91🟡 | 0.92🟡 | 2.95✅ | 2.99✅ | 0.31🟠 | 1.10✅ | 2.26✅ | 3.17✅ | — | 3.23✅ |
| **u64** | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.41🟠 | 0.91🟡 | 0.92🟡 | 3.04✅ | 3.05✅ | 0.41🟠 | 1.33✅ | 1.42✅ | 2.27✅ | — | 3.04✅ |
| **char** | 2.91✅ | 3.85✅ | 3.84✅ | 3.81✅ | 5.35✅ | 2.69✅ | 2.93✅ | 4.68✅ | 5.05✅ | 5.29✅ | 1.92✅ | 2.17✅ | 3.53✅ | — | 5.03✅ |
| **f16** | 7.73✅ | 1.84✅ | 1.97✅ | 2.42✅ | 2.25✅ | 3.30✅ | 1.34✅ | 2.38✅ | 0.80🟡 | 2.25✅ | 5.31✅ | 2.09✅ | 0.95🟡 | — | 1.36✅ |
| **f32** | 0.82🟡 | 2.50✅ | 3.51✅ | 0.38🟠 | 0.38🟠 | 1.22✅ | 0.51🟡 | 1.07✅ | 0.49🟠 | 0.38🟠 | 2.00✅ | 2.06✅ | 2.90✅ | — | 4.67✅ |
| **f64** | 0.58🟡 | 0.37🟠 | 0.37🟠 | 0.41🟠 | 0.37🟠 | 0.69🟡 | 0.76🟡 | 1.27✅ | 0.65🟡 | 0.38🟠 | 1.25✅ | 1.75✅ | 3.02✅ | — | 3.65✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.35🟠 | 0.62🟡 | 0.58🟡 | 0.59🟡 | 0.59🟡 | 1.13✅ | 0.83🟡 | 1.28✅ | 0.61🟡 | 0.59🟡 | 1.11✅ | 1.62✅ | 2.25✅ | — | 2.82✅ |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.03🔴 | 0.28🟠 | 0.27🟠 | 0.42🟠 | 0.41🟠 | 0.53🟡 | 0.53🟡 | 1.15✅ | 1.13✅ | 0.40🟠 | 0.30🟠 | 0.58🟡 | 1.16✅ | — | 1.39✅ |
| **u8** | 0.29🟠 | 0.04🔴 | 0.24🟠 | 0.38🟠 | 0.39🟠 | 0.40🟠 | 0.42🟠 | 1.04✅ | 1.04✅ | 0.24🟠 | 0.59🟡 | 0.54🟡 | 1.12✅ | — | 1.21✅ |
| **i8** | 0.29🟠 | 0.33🟠 | 0.04🔴 | 0.34🟠 | 0.39🟠 | 0.46🟠 | 0.52🟡 | 0.99🟡 | 1.10✅ | 0.42🟠 | 0.30🟠 | 0.38🟠 | 1.05✅ | — | 1.28✅ |
| **i16** | 0.27🟠 | 0.31🟠 | 0.23🟠 | 0.28🟠 | 0.44🟠 | 0.54🟡 | 0.57🟡 | 1.17✅ | 1.18✅ | 0.44🟠 | 0.58🟡 | 0.60🟡 | 1.14✅ | — | 1.27✅ |
| **u16** | 0.27🟠 | 0.31🟠 | 0.31🟠 | 0.37🟠 | 0.29🟠 | 1.20✅ | 0.55🟡 | 1.00🟡 | 1.14✅ | 0.29🟠 | 0.58🟡 | 0.54🟡 | 1.02✅ | — | 1.27✅ |
| **i32** | 0.25🟠 | 0.30🟠 | 0.30🟠 | 0.25🟠 | 0.35🟠 | 0.48🟠 | 0.52🟡 | 0.93🟡 | 1.01✅ | 0.35🟠 | 0.63🟡 | 0.53🟡 | 1.01✅ | — | 1.30✅ |
| **u32** | 0.25🟠 | 0.22🟠 | 0.30🟠 | 0.35🟠 | 0.43🟠 | 0.49🟠 | 0.51🟡 | 1.14✅ | 1.20✅ | 0.42🟠 | 0.59🟡 | 0.45🟠 | 0.96🟡 | — | 1.40✅ |
| **i64** | 0.29🟠 | 0.26🟠 | 0.26🟠 | 0.36🟠 | 0.30🟠 | 0.47🟠 | 0.52🟡 | 1.04✅ | 1.02✅ | 0.30🟠 | 0.55🟡 | 0.33🟠 | 1.02✅ | — | 1.40✅ |
| **u64** | 0.29🟠 | 0.26🟠 | 0.26🟠 | 0.35🟠 | 0.35🟠 | 0.45🟠 | 0.51🟡 | 1.09✅ | 1.20✅ | 0.36🟠 | 0.54🟡 | 0.58🟡 | 1.31✅ | — | 1.41✅ |
| **char** | 0.27🟠 | 0.31🟠 | 0.31🟠 | 0.37🟠 | 0.28🟠 | 0.54🟡 | 0.54🟡 | 1.19✅ | 1.19✅ | 0.28🟠 | 0.60🟡 | 0.54🟡 | 1.19✅ | — | 1.28✅ |
| **f16** | 0.77🟡 | 0.39🟠 | 0.41🟠 | 0.48🟠 | 0.46🟠 | 0.49🟠 | 0.54🟡 | 0.60🟡 | 0.77🟡 | 0.44🟠 | 0.28🟠 | 0.50🟡 | 0.85🟡 | — | 1.04✅ |
| **f32** | 0.44🟠 | 0.11🔴 | 0.15🔴 | 0.20🟠 | 0.20🟠 | 0.38🟠 | 0.38🟠 | 0.94🟡 | 0.70🟡 | 0.20🟠 | 0.74🟡 | 0.47🟠 | 0.85🟡 | — | 1.38✅ |
| **f64** | 0.38🟠 | 0.09🔴 | 0.12🔴 | 0.27🟠 | 0.22🟠 | 0.77🟡 | 0.37🟠 | 0.95🟡 | 0.79🟡 | 0.23🟠 | 0.66🟡 | 0.41🟠 | 1.16✅ | — | 1.17✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.37🟠 | 0.11🔴 | 0.12🔴 | 0.25🟠 | 0.19🔴 | 0.33🟠 | 0.31🟠 | 1.07✅ | 0.50🟠 | 0.21🟠 | 0.67🟡 | 0.33🟠 | 0.99🟡 | — | 1.28✅ |

## Lagging cells (<1.0) — the worklist  (327 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| bool\|bcast\|bool | 1.4752 | 0.0491 | 0.03 🔴 |
| i8\|bcast\|i8 | 1.4399 | 0.0538 | 0.04 🔴 |
| u8\|bcast\|u8 | 1.4484 | 0.0552 | 0.04 🔴 |
| f64\|bcast\|u8 | 2.8553 | 0.2468 | 0.09 🔴 |
| f32\|bcast\|u8 | 3.1535 | 0.3549 | 0.11 🔴 |
| c128\|bcast\|u8 | 3.1422 | 0.3568 | 0.11 🔴 |
| c128\|bcast\|i8 | 3.0743 | 0.3566 | 0.12 🔴 |
| f64\|bcast\|i8 | 2.8773 | 0.3552 | 0.12 🔴 |
| f32\|bcast\|i8 | 3.1105 | 0.4795 | 0.15 🔴 |
| i64\|negcol\|i8 | 1.3242 | 0.2278 | 0.17 🔴 |
| i64\|strided\|u16 | 0.6775 | 0.1216 | 0.18 🔴 |
| c128\|bcast\|u16 | 3.4093 | 0.6462 | 0.19 🔴 |
| f32\|bcast\|i16 | 3.1785 | 0.6411 | 0.20 🟠 |
| f32\|bcast\|char | 3.1570 | 0.6422 | 0.20 🟠 |
| f32\|bcast\|u16 | 3.1469 | 0.6413 | 0.20 🟠 |
| c128\|bcast\|char | 3.0892 | 0.6471 | 0.21 🟠 |
| f64\|bcast\|u16 | 2.8911 | 0.6437 | 0.22 🟠 |
| u32\|bcast\|u8 | 1.5795 | 0.3548 | 0.22 🟠 |
| f64\|bcast\|char | 2.8506 | 0.6451 | 0.23 🟠 |
| i16\|bcast\|i8 | 1.5369 | 0.3546 | 0.23 🟠 |
| u8\|bcast\|char | 1.8857 | 0.4559 | 0.24 🟠 |
| u8\|bcast\|i8 | 1.4582 | 0.3545 | 0.24 🟠 |
| c128\|bcast\|i16 | 3.1564 | 0.7754 | 0.25 🟠 |
| i32\|bcast\|bool | 1.6294 | 0.4127 | 0.25 🟠 |
| i32\|bcast\|i16 | 1.8264 | 0.4642 | 0.25 🟠 |
| bool\|C\|f16 | 6.6646 | 1.6946 | 0.25 🟠 |
| u32\|bcast\|bool | 1.6228 | 0.4127 | 0.25 🟠 |
| i64\|bcast\|i8 | 1.8562 | 0.4793 | 0.26 🟠 |
| i64\|bcast\|u8 | 1.8487 | 0.4794 | 0.26 🟠 |
| u64\|bcast\|i8 | 1.8495 | 0.4797 | 0.26 🟠 |
| u64\|bcast\|u8 | 1.8421 | 0.4796 | 0.26 🟠 |
| f64\|bcast\|i16 | 2.8806 | 0.7722 | 0.27 🟠 |
| u16\|bcast\|bool | 1.5388 | 0.4126 | 0.27 🟠 |
| i16\|bcast\|bool | 1.5370 | 0.4127 | 0.27 🟠 |
| char\|bcast\|bool | 1.5263 | 0.4126 | 0.27 🟠 |
| bool\|bcast\|i8 | 1.5022 | 0.4129 | 0.27 🟠 |
| f16\|bcast\|f16 | 1.7927 | 0.4962 | 0.28 🟠 |
| bool\|bcast\|u8 | 1.4908 | 0.4129 | 0.28 🟠 |
| bool\|strided\|f16 | 5.3325 | 1.4889 | 0.28 🟠 |
| char\|bcast\|char | 1.7432 | 0.4960 | 0.28 🟠 |
| char\|bcast\|u16 | 1.7417 | 0.4957 | 0.28 🟠 |
| i16\|bcast\|i16 | 1.7409 | 0.4958 | 0.28 🟠 |
| u16\|bcast\|char | 1.7475 | 0.4991 | 0.29 🟠 |
| u16\|bcast\|u16 | 1.7483 | 0.5007 | 0.29 🟠 |
| i64\|bcast\|bool | 1.6384 | 0.4791 | 0.29 🟠 |
| u8\|bcast\|bool | 1.6372 | 0.4790 | 0.29 🟠 |
| i8\|bcast\|bool | 1.6355 | 0.4789 | 0.29 🟠 |
| u64\|bcast\|bool | 1.6354 | 0.4791 | 0.29 🟠 |
| i64\|bcast\|char | 2.1636 | 0.6393 | 0.30 🟠 |
| f64\|negcol\|i8 | 1.3212 | 0.3910 | 0.30 🟠 |
| f64\|negcol\|u8 | 1.3208 | 0.3916 | 0.30 🟠 |
| i64\|bcast\|u16 | 2.1476 | 0.6410 | 0.30 🟠 |
| u32\|bcast\|i8 | 1.6001 | 0.4787 | 0.30 🟠 |
| i32\|bcast\|u8 | 1.5977 | 0.4793 | 0.30 🟠 |
| i8\|bcast\|f16 | 5.4377 | 1.6356 | 0.30 🟠 |
| i32\|bcast\|i8 | 1.5917 | 0.4797 | 0.30 🟠 |
| bool\|sliced\|f16 | 10.8282 | 3.2671 | 0.30 🟠 |
| bool\|bcast\|f16 | 10.7732 | 3.2842 | 0.30 🟠 |
| bool\|F\|f16 | 10.7465 | 3.2798 | 0.31 🟠 |
| bool\|T\|f16 | 10.7418 | 3.2795 | 0.31 🟠 |
| bool\|negcol\|f16 | 10.7798 | 3.2948 | 0.31 🟠 |
| bool\|negrow\|f16 | 10.7634 | 3.3048 | 0.31 🟠 |
| i64\|strided\|char | 0.6776 | 0.2107 | 0.31 🟠 |
| char\|bcast\|u8 | 1.5379 | 0.4784 | 0.31 🟠 |
| char\|bcast\|i8 | 1.5388 | 0.4797 | 0.31 🟠 |
| u16\|bcast\|u8 | 1.5332 | 0.4785 | 0.31 🟠 |
| u16\|bcast\|i8 | 1.5314 | 0.4795 | 0.31 🟠 |
| c128\|bcast\|u32 | 3.1052 | 0.9736 | 0.31 🟠 |
| i16\|bcast\|u8 | 1.5273 | 0.4791 | 0.31 🟠 |
| c128\|bcast\|f32 | 2.8231 | 0.9328 | 0.33 🟠 |
| i8\|bcast\|u8 | 1.4410 | 0.4784 | 0.33 🟠 |
| i64\|bcast\|f32 | 2.2868 | 0.7608 | 0.33 🟠 |
| c128\|bcast\|i32 | 3.1684 | 1.0576 | 0.33 🟠 |
| i8\|bcast\|i16 | 1.8802 | 0.6439 | 0.34 🟠 |
| u64\|bcast\|u16 | 2.1524 | 0.7456 | 0.35 🟠 |
| i32\|bcast\|char | 1.8489 | 0.6449 | 0.35 🟠 |
| u32\|bcast\|i16 | 1.8348 | 0.6400 | 0.35 🟠 |
| u64\|bcast\|i16 | 2.1958 | 0.7716 | 0.35 🟠 |
| c128\|strided\|bool | 1.3620 | 0.4824 | 0.35 🟠 |
| i32\|bcast\|u16 | 1.8099 | 0.6420 | 0.35 🟠 |
| i64\|bcast\|i16 | 2.1529 | 0.7695 | 0.36 🟠 |
| u64\|bcast\|char | 2.1630 | 0.7741 | 0.36 🟠 |
| u16\|bcast\|i16 | 1.7447 | 0.6419 | 0.37 🟠 |
| char\|bcast\|i16 | 1.7403 | 0.6417 | 0.37 🟠 |
| f64\|bcast\|u32 | 2.1099 | 0.7815 | 0.37 🟠 |
| f64\|strided\|i8 | 0.6607 | 0.2448 | 0.37 🟠 |
| f64\|strided\|u8 | 0.6601 | 0.2462 | 0.37 🟠 |
| c128\|bcast\|bool | 1.9086 | 0.7126 | 0.37 🟠 |
| f64\|strided\|u16 | 0.6742 | 0.2522 | 0.37 🟠 |
| f64\|strided\|char | 0.6743 | 0.2539 | 0.38 🟠 |
| f32\|bcast\|u32 | 2.5513 | 0.9688 | 0.38 🟠 |
| f32\|bcast\|i32 | 2.5412 | 0.9679 | 0.38 🟠 |
| f64\|bcast\|bool | 1.8684 | 0.7128 | 0.38 🟠 |
| u8\|bcast\|i16 | 1.8298 | 0.7003 | 0.38 🟠 |
| i8\|bcast\|f32 | 1.9612 | 0.7531 | 0.38 🟠 |
| f32\|strided\|char | 0.4638 | 0.1784 | 0.38 🟠 |
| f32\|strided\|i16 | 0.4637 | 0.1784 | 0.38 🟠 |
| f32\|strided\|u16 | 0.4637 | 0.1784 | 0.38 🟠 |
| u32\|strided\|u8 | 0.4561 | 0.1777 | 0.39 🟠 |
| u64\|negcol\|i8 | 1.3249 | 0.5178 | 0.39 🟠 |
| u64\|negcol\|u8 | 1.3245 | 0.5188 | 0.39 🟠 |
| u64\|negcol\|bool | 1.3555 | 0.5312 | 0.39 🟠 |
| i32\|strided\|u16 | 0.4546 | 0.1783 | 0.39 🟠 |
| u32\|strided\|i16 | 0.4547 | 0.1784 | 0.39 🟠 |
| i64\|negcol\|u8 | 1.3248 | 0.5199 | 0.39 🟠 |
| i32\|strided\|char | 0.4547 | 0.1784 | 0.39 🟠 |
| u8\|bcast\|u16 | 1.9588 | 0.7714 | 0.39 🟠 |
| i8\|bcast\|u16 | 1.9566 | 0.7706 | 0.39 🟠 |
| f16\|bcast\|u8 | 4.5955 | 1.8137 | 0.39 🟠 |
| bool\|negcol\|bool | 0.4850 | 0.1918 | 0.40 🟠 |
| u8\|bcast\|i32 | 2.2904 | 0.9066 | 0.40 🟠 |
| i64\|negcol\|bool | 1.3538 | 0.5434 | 0.40 🟠 |
| bool\|bcast\|char | 1.8109 | 0.7295 | 0.40 🟠 |
| f32\|negcol\|u8 | 0.9244 | 0.3749 | 0.41 🟠 |
| i64\|strided\|bool | 0.6733 | 0.2743 | 0.41 🟠 |
| i64\|strided\|i16 | 0.6776 | 0.2762 | 0.41 🟠 |
| u64\|strided\|u16 | 0.6773 | 0.2763 | 0.41 🟠 |
| u64\|strided\|i16 | 0.6775 | 0.2764 | 0.41 🟠 |
| u64\|strided\|bool | 0.6736 | 0.2749 | 0.41 🟠 |
| u64\|strided\|char | 0.6773 | 0.2768 | 0.41 🟠 |
| f16\|bcast\|i8 | 4.5903 | 1.8797 | 0.41 🟠 |
| c128\|negcol\|i8 | 1.3342 | 0.5469 | 0.41 🟠 |
| bool\|bcast\|u16 | 1.8467 | 0.7572 | 0.41 🟠 |
| f64\|bcast\|f32 | 2.4530 | 1.0063 | 0.41 🟠 |
| i64\|strided\|u8 | 0.6637 | 0.2724 | 0.41 🟠 |
| u64\|strided\|i8 | 0.6635 | 0.2724 | 0.41 🟠 |
| i64\|strided\|i8 | 0.6633 | 0.2723 | 0.41 🟠 |
| u64\|strided\|u8 | 0.6635 | 0.2724 | 0.41 🟠 |
| u32\|negcol\|u8 | 0.9033 | 0.3710 | 0.41 🟠 |
| c128\|negcol\|u8 | 1.3374 | 0.5494 | 0.41 🟠 |
| f64\|strided\|i16 | 0.6744 | 0.2792 | 0.41 🟠 |
| i8\|bcast\|char | 1.8459 | 0.7706 | 0.42 🟠 |
| bool\|bcast\|i16 | 1.8457 | 0.7730 | 0.42 🟠 |
| u8\|bcast\|u32 | 2.3006 | 0.9733 | 0.42 🟠 |
| u32\|bcast\|char | 1.8162 | 0.7696 | 0.42 🟠 |
| u32\|bcast\|u16 | 1.7994 | 0.7707 | 0.43 🟠 |
| f32\|negcol\|bool | 0.8989 | 0.3897 | 0.43 🟠 |
| i16\|bcast\|char | 1.7683 | 0.7736 | 0.44 🟠 |
| f32\|bcast\|bool | 1.6283 | 0.7124 | 0.44 🟠 |
| f16\|bcast\|char | 4.5855 | 2.0363 | 0.44 🟠 |
| i16\|bcast\|u16 | 1.7366 | 0.7722 | 0.44 🟠 |
| u32\|bcast\|f32 | 2.2449 | 1.0002 | 0.45 🟠 |
| u64\|bcast\|i32 | 2.0093 | 0.9067 | 0.45 🟠 |
| u32\|strided\|bool | 0.4522 | 0.2069 | 0.46 🟠 |
| i32\|strided\|bool | 0.4520 | 0.2073 | 0.46 🟠 |
| i8\|bcast\|i32 | 2.1109 | 0.9701 | 0.46 🟠 |
| f16\|bcast\|u16 | 4.5715 | 2.1032 | 0.46 🟠 |
| f32\|negcol\|u32 | 2.2634 | 1.0625 | 0.47 🟠 |
| i64\|bcast\|i32 | 2.0548 | 0.9662 | 0.47 🟠 |
| f32\|bcast\|f32 | 2.0403 | 0.9643 | 0.47 🟠 |
| i32\|negcol\|bool | 0.9059 | 0.4301 | 0.47 🟠 |
| u32\|negcol\|bool | 0.9060 | 0.4310 | 0.48 🟠 |
| f16\|bcast\|i16 | 4.5777 | 2.1819 | 0.48 🟠 |
| i32\|bcast\|i32 | 1.9986 | 0.9639 | 0.48 🟠 |
| u32\|bcast\|i32 | 1.9871 | 0.9688 | 0.49 🟠 |
| f32\|strided\|u64 | 1.9999 | 0.9805 | 0.49 🟠 |
| f16\|bcast\|i32 | 4.7169 | 2.3171 | 0.49 🟠 |
| c128\|bcast\|u64 | 3.8807 | 1.9313 | 0.50 🟠 |
| f16\|bcast\|f32 | 3.8098 | 1.9168 | 0.50 🟡 |
| u32\|bcast\|u32 | 1.9056 | 0.9637 | 0.51 🟡 |
| f32\|strided\|u32 | 1.1333 | 0.5743 | 0.51 🟡 |
| u64\|bcast\|u32 | 2.0401 | 1.0447 | 0.51 🟡 |
| i64\|negcol\|char | 1.3530 | 0.6953 | 0.51 🟡 |
| i64\|negcol\|u16 | 1.3521 | 0.6973 | 0.52 🟡 |
| i64\|bcast\|u32 | 2.0245 | 1.0494 | 0.52 🟡 |
| i8\|bcast\|u32 | 2.0195 | 1.0512 | 0.52 🟡 |
| f64\|negcol\|char | 1.3563 | 0.7076 | 0.52 🟡 |
| i32\|bcast\|u32 | 2.0120 | 1.0534 | 0.52 🟡 |
| i32\|bcast\|f32 | 2.0078 | 1.0553 | 0.53 🟡 |
| bool\|bcast\|u32 | 1.9669 | 1.0381 | 0.53 🟡 |
| f64\|negcol\|u16 | 1.3550 | 0.7204 | 0.53 🟡 |
| bool\|bcast\|i32 | 1.9613 | 1.0486 | 0.53 🟡 |
| f32\|negcol\|u64 | 3.5829 | 1.9183 | 0.54 🟡 |
| i16\|bcast\|i32 | 1.8598 | 1.0024 | 0.54 🟡 |
| f16\|bcast\|u32 | 4.5311 | 2.4526 | 0.54 🟡 |
| char\|bcast\|i32 | 1.9538 | 1.0580 | 0.54 🟡 |
| char\|bcast\|f32 | 2.0372 | 1.1059 | 0.54 🟡 |
| u8\|bcast\|f32 | 2.0353 | 1.1053 | 0.54 🟡 |
| f32\|negcol\|i8 | 0.9274 | 0.5044 | 0.54 🟡 |
| u64\|bcast\|f16 | 6.2427 | 3.3977 | 0.54 🟡 |
| u16\|bcast\|f32 | 1.8861 | 1.0272 | 0.54 🟡 |
| char\|bcast\|u32 | 1.9463 | 1.0605 | 0.54 🟡 |
| i64\|bcast\|f16 | 5.6662 | 3.1089 | 0.55 🟡 |
| u16\|bcast\|u32 | 1.9230 | 1.0583 | 0.55 🟡 |
| u32\|negcol\|i8 | 0.9037 | 0.4980 | 0.55 🟡 |
| i32\|negcol\|i8 | 0.9042 | 0.4992 | 0.55 🟡 |
| i32\|negcol\|u8 | 0.9033 | 0.4989 | 0.55 🟡 |
| u32\|strided\|i8 | 0.4562 | 0.2536 | 0.56 🟡 |
| i32\|strided\|i8 | 0.4562 | 0.2537 | 0.56 🟡 |
| i32\|strided\|u8 | 0.4561 | 0.2538 | 0.56 🟡 |
| i32\|strided\|i16 | 0.4547 | 0.2577 | 0.57 🟡 |
| u32\|strided\|u16 | 0.4545 | 0.2578 | 0.57 🟡 |
| u32\|strided\|char | 0.4546 | 0.2580 | 0.57 🟡 |
| i16\|bcast\|u32 | 1.8419 | 1.0561 | 0.57 🟡 |
| f64\|negcol\|bool | 1.3588 | 0.7796 | 0.57 🟡 |
| c128\|strided\|i8 | 0.6815 | 0.3930 | 0.58 🟡 |
| i16\|bcast\|f16 | 5.4727 | 3.1636 | 0.58 🟡 |
| f64\|strided\|bool | 0.6741 | 0.3909 | 0.58 🟡 |
| bool\|bcast\|f32 | 1.9698 | 1.1433 | 0.58 🟡 |
| u64\|bcast\|f32 | 2.1628 | 1.2559 | 0.58 🟡 |
| f32\|F\|u64 | 3.0428 | 1.7740 | 0.58 🟡 |
| u64\|negcol\|i16 | 1.3538 | 0.7900 | 0.58 🟡 |
| u16\|bcast\|f16 | 5.3203 | 3.1084 | 0.58 🟡 |
| c128\|strided\|u16 | 0.7335 | 0.4326 | 0.59 🟡 |
| c128\|strided\|i16 | 0.7330 | 0.4331 | 0.59 🟡 |
| c128\|strided\|char | 0.7335 | 0.4343 | 0.59 🟡 |
| u32\|bcast\|f16 | 5.3241 | 3.1597 | 0.59 🟡 |
| u8\|bcast\|f16 | 5.3190 | 3.1640 | 0.59 🟡 |
| char\|bcast\|f16 | 5.3011 | 3.1579 | 0.60 🟡 |
| i64\|negcol\|i16 | 1.3528 | 0.8098 | 0.60 🟡 |
| bool\|C\|bool | 0.0239 | 0.0143 | 0.60 🟡 |
| i16\|bcast\|f32 | 1.8312 | 1.0999 | 0.60 🟡 |
| f16\|bcast\|i64 | 4.7578 | 2.8632 | 0.60 🟡 |
| u64\|negcol\|u16 | 1.3536 | 0.8181 | 0.60 🟡 |
| c128\|negcol\|i16 | 1.4599 | 0.8835 | 0.61 🟡 |
| c128\|strided\|u64 | 2.2017 | 1.3346 | 0.61 🟡 |
| u64\|negcol\|char | 1.3520 | 0.8271 | 0.61 🟡 |
| f64\|negcol\|u32 | 1.7383 | 1.0706 | 0.62 🟡 |
| f64\|negcol\|i16 | 1.3559 | 0.8385 | 0.62 🟡 |
| c128\|strided\|u8 | 0.6802 | 0.4246 | 0.62 🟡 |
| i32\|bcast\|f16 | 4.9619 | 3.1064 | 0.63 🟡 |
| f64\|sliced\|u64 | 2.9306 | 1.8658 | 0.64 🟡 |
| c128\|negcol\|u16 | 1.4593 | 0.9475 | 0.65 🟡 |
| f64\|strided\|u64 | 1.6330 | 1.0630 | 0.65 🟡 |
| f64\|bcast\|f16 | 5.4514 | 3.5881 | 0.66 🟡 |
| f64\|T\|u64 | 2.9215 | 1.9309 | 0.66 🟡 |
| bool\|negcol\|u8 | 0.4644 | 0.3079 | 0.66 🟡 |
| f32\|sliced\|u64 | 3.0223 | 2.0104 | 0.67 🟡 |
| f64\|negcol\|u64 | 3.1419 | 2.0931 | 0.67 🟡 |
| c128\|negcol\|char | 1.4606 | 0.9732 | 0.67 🟡 |
| f32\|C\|u64 | 3.0350 | 2.0224 | 0.67 🟡 |
| c128\|bcast\|f16 | 5.7612 | 3.8538 | 0.67 🟡 |
| f32\|T\|u64 | 3.0192 | 2.0328 | 0.67 🟡 |
| f32\|negrow\|u64 | 3.0515 | 2.0565 | 0.67 🟡 |
| f32\|negcol\|i16 | 0.9113 | 0.6191 | 0.68 🟡 |
| f64\|strided\|i32 | 0.9753 | 0.6755 | 0.69 🟡 |
| f32\|bcast\|u64 | 2.7217 | 1.9181 | 0.70 🟡 |
| f64\|negrow\|u64 | 2.9585 | 2.0897 | 0.71 🟡 |
| u8\|F\|u8 | 0.0416 | 0.0297 | 0.71 🟡 |
| f64\|F\|u64 | 2.9260 | 2.0870 | 0.71 🟡 |
| f64\|C\|u64 | 2.9194 | 2.0973 | 0.72 🟡 |
| c128\|negrow\|u64 | 3.3327 | 2.4301 | 0.73 🟡 |
| f32\|negcol\|char | 0.9118 | 0.6683 | 0.73 🟡 |
| i32\|negcol\|char | 0.9126 | 0.6691 | 0.73 🟡 |
| u32\|negcol\|i16 | 0.9133 | 0.6722 | 0.74 🟡 |
| i32\|negcol\|u16 | 0.9132 | 0.6730 | 0.74 🟡 |
| c128\|F\|u64 | 3.2348 | 2.3862 | 0.74 🟡 |
| f32\|negcol\|u16 | 0.9117 | 0.6781 | 0.74 🟡 |
| f32\|bcast\|f16 | 4.0165 | 2.9899 | 0.74 🟡 |
| c128\|negcol\|u64 | 3.3206 | 2.4766 | 0.75 🟡 |
| c128\|sliced\|u64 | 3.2652 | 2.4406 | 0.75 🟡 |
| c128\|T\|u64 | 3.2522 | 2.4435 | 0.75 🟡 |
| c128\|C\|u64 | 3.2326 | 2.4414 | 0.76 🟡 |
| f64\|strided\|u32 | 0.8520 | 0.6451 | 0.76 🟡 |
| c128\|negcol\|u32 | 1.7830 | 1.3516 | 0.76 🟡 |
| u64\|negcol\|i32 | 1.3585 | 1.0384 | 0.76 🟡 |
| f16\|bcast\|u64 | 4.5479 | 3.4803 | 0.77 🟡 |
| f32\|C\|u32 | 1.3182 | 1.0133 | 0.77 🟡 |
| f16\|bcast\|bool | 1.5312 | 1.1783 | 0.77 🟡 |
| f64\|bcast\|i32 | 1.2771 | 0.9844 | 0.77 🟡 |
| f16\|sliced\|u64 | 4.0179 | 3.1650 | 0.79 🟡 |
| f64\|bcast\|u64 | 2.4344 | 1.9181 | 0.79 🟡 |
| f16\|F\|u64 | 4.0126 | 3.1695 | 0.79 🟡 |
| f16\|T\|u64 | 4.0103 | 3.1694 | 0.79 🟡 |
| f32\|sliced\|u32 | 1.3321 | 1.0541 | 0.79 🟡 |
| f32\|negrow\|u32 | 1.3363 | 1.0612 | 0.79 🟡 |
| f16\|C\|u64 | 4.0008 | 3.1783 | 0.79 🟡 |
| f16\|negrow\|u64 | 4.0299 | 3.2041 | 0.80 🟡 |
| f16\|negcol\|u64 | 4.3852 | 3.5055 | 0.80 🟡 |
| f32\|F\|u32 | 1.3159 | 1.0529 | 0.80 🟡 |
| f32\|T\|u32 | 1.3152 | 1.0534 | 0.80 🟡 |
| f16\|strided\|u64 | 2.1899 | 1.7574 | 0.80 🟡 |
| i64\|negcol\|i32 | 1.3580 | 1.1101 | 0.82 🟡 |
| f32\|strided\|bool | 0.4505 | 0.3706 | 0.82 🟡 |
| bool\|strided\|bool | 0.2469 | 0.2057 | 0.83 🟡 |
| c128\|strided\|u32 | 0.9896 | 0.8261 | 0.83 🟡 |
| i64\|negcol\|u32 | 1.3577 | 1.1396 | 0.84 🟡 |
| u64\|negcol\|u32 | 1.3581 | 1.1418 | 0.84 🟡 |
| f32\|bcast\|f64 | 1.5701 | 1.3288 | 0.85 🟡 |
| i64\|C\|i8 | 0.4217 | 0.3577 | 0.85 🟡 |
| f16\|bcast\|f64 | 2.6233 | 2.2281 | 0.85 🟡 |
| i32\|negcol\|i16 | 0.9140 | 0.7919 | 0.87 🟡 |
| u32\|negcol\|u16 | 0.9126 | 0.7937 | 0.87 🟡 |
| c128\|F\|bool | 0.8152 | 0.7101 | 0.87 🟡 |
| c128\|sliced\|bool | 0.8428 | 0.7344 | 0.87 🟡 |
| bool\|strided\|char | 0.2695 | 0.2350 | 0.87 🟡 |
| bool\|strided\|u16 | 0.2695 | 0.2350 | 0.87 🟡 |
| u32\|negcol\|char | 0.9134 | 0.7977 | 0.87 🟡 |
| i64\|T\|i8 | 0.4088 | 0.3575 | 0.87 🟡 |
| bool\|strided\|u8 | 0.2349 | 0.2057 | 0.88 🟡 |
| bool\|strided\|i8 | 0.2350 | 0.2057 | 0.88 🟡 |
| c128\|C\|bool | 0.8317 | 0.7299 | 0.88 🟡 |
| i32\|C\|i8 | 0.4019 | 0.3532 | 0.88 🟡 |
| c128\|negrow\|bool | 0.8622 | 0.7593 | 0.88 🟡 |
| c128\|T\|bool | 0.8196 | 0.7293 | 0.89 🟡 |
| i64\|F\|i8 | 0.4017 | 0.3582 | 0.89 🟡 |
| u32\|T\|i8 | 0.3947 | 0.3534 | 0.90 🟡 |
| bool\|negrow\|u8 | 0.4645 | 0.4161 | 0.90 🟡 |
| i32\|T\|i8 | 0.3945 | 0.3534 | 0.90 🟡 |
| u32\|F\|i8 | 0.3942 | 0.3534 | 0.90 🟡 |
| i32\|F\|i8 | 0.3943 | 0.3535 | 0.90 🟡 |
| bool\|negcol\|i8 | 0.4636 | 0.4157 | 0.90 🟡 |
| bool\|sliced\|u8 | 0.4584 | 0.4112 | 0.90 🟡 |
| u32\|C\|i8 | 0.3942 | 0.3537 | 0.90 🟡 |
| bool\|strided\|i16 | 0.2695 | 0.2423 | 0.90 🟡 |
| u64\|strided\|i32 | 0.6768 | 0.6126 | 0.91 🟡 |
| i64\|strided\|i32 | 0.6766 | 0.6128 | 0.91 🟡 |
| c128\|negcol\|i32 | 1.4434 | 1.3145 | 0.91 🟡 |
| i64\|strided\|u32 | 0.6766 | 0.6195 | 0.92 🟡 |
| u64\|strided\|u32 | 0.6764 | 0.6214 | 0.92 🟡 |
| f32\|negcol\|i64 | 1.9563 | 1.8072 | 0.92 🟡 |
| i32\|bcast\|i64 | 1.5821 | 1.4768 | 0.93 🟡 |
| f32\|F\|i64 | 1.8807 | 1.7581 | 0.93 🟡 |
| f32\|sliced\|i64 | 1.9071 | 1.7939 | 0.94 🟡 |
| f16\|negcol\|f64 | 2.5507 | 2.3999 | 0.94 🟡 |
| f32\|bcast\|i64 | 1.9897 | 1.8787 | 0.94 🟡 |
| f16\|negrow\|f64 | 2.5475 | 2.4116 | 0.95 🟡 |
| f64\|bcast\|i64 | 1.8479 | 1.7577 | 0.95 🟡 |
| f16\|strided\|f64 | 1.2622 | 1.2013 | 0.95 🟡 |
| u32\|bcast\|f64 | 1.6756 | 1.6031 | 0.96 🟡 |
| f32\|T\|i64 | 1.8739 | 1.8118 | 0.97 🟡 |
| u8\|strided\|bool | 0.2460 | 0.2404 | 0.98 🟡 |
| i32\|C\|u8 | 0.3609 | 0.3533 | 0.98 🟡 |
| i8\|strided\|bool | 0.2456 | 0.2405 | 0.98 🟡 |
| c128\|bcast\|f64 | 1.6296 | 1.6120 | 0.99 🟡 |
| i8\|bcast\|i64 | 1.5729 | 1.5633 | 0.99 🟡 |
| u16\|bcast\|i64 | 1.6073 | 1.6008 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 327 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best window (min over ~200 ms):
  a*b+c       fused    3.38 ms   unfused    5.76 ms   (1.70x)
  (a-b)/(a+b) fused    2.83 ms   unfused   12.00 ms   (4.24x)
  sum(a*b)    fused    2.23 ms   unfused    3.64 ms   (1.63x)
  sum(af*bf)  fused    1.20 ms   unfused    1.45 ms   (1.21x)  [f32]
  a*b+c out=  fused    3.26 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.68 ms   unfused    3.79 ms   (1.41x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.38 ms   unfused    5.75 ms   (1.70x)
    [F      ] fused   11.74 ms   unfused   17.66 ms   (1.50x)
    [T      ] fused   11.71 ms   unfused   17.48 ms   (1.49x)
    [strided] fused   11.64 ms   unfused    4.76 ms   (0.41x)
    [bcast  ] fused    2.02 ms   unfused    7.72 ms   (3.81x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best window (min over ~200 ms):
  a*b+c         12.98 ms
  (a-b)/(a+b)   19.67 ms
  sum(a*b)       8.47 ms
  sum(af*bf)     4.15 ms  [f32]
  a*b+c out=     4.43 ms  [two-pass with out=]
  i4*2+f8       10.35 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   13.02 ms
    [F      ]   13.07 ms
    [T      ]   12.60 ms
    [strided]    7.73 ms
    [bcast  ]   12.14 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-6702e17e3d34774811ed31feb79b1be9233097d85018996ddc811efc382c4e33\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b) [ndarray]` | 1,000 | 0.0003 ms | 0.0006 ms | 0.0004 | 1.333x |
| `a.dot(b) [ndarray]` | 100,000 | 0.0090 ms | 0.0090 ms | 0.0068 | 0.756x |
| `a.dot(b) [ndarray]` | 10,000,000 | 3.0736 ms | 2.9441 ms | 3.0018 | 1.020x |
| `np.convolve(a, kernel)` | 1,000 | 0.0036 ms | 0.0156 ms | 0.0110 | 3.056x |
| `np.convolve(a, kernel)` | 100,000 | 0.1138 ms | 0.9192 ms | 0.9811 | 8.621x |
| `np.convolve(a, kernel)` | 10,000,000 | 19.3707 ms | 100.8334 ms | 109.7462 | 5.666x |
| `np.correlate(a, kernel)` | 1,000 | 0.0014 ms | 0.0132 ms | 0.0105 | 7.500x |
| `np.correlate(a, kernel)` | 100,000 | 0.1113 ms | 0.9014 ms | 0.9719 | 8.732x |
| `np.correlate(a, kernel)` | 10,000,000 | 19.3109 ms | 100.8807 ms | 109.5148 | 5.671x |
| `np.dot(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0005 | 1.250x |
| `np.dot(a, b)` | 100,000 | 0.0091 ms | 0.0090 ms | 0.0069 | 0.767x |
| `np.dot(a, b)` | 10,000,000 | 3.0302 ms | 2.9299 ms | 3.0031 | 1.025x |
| `np.einsum('ij,jk->ik', a, b)` | 1,000 | 0.0040 ms | 0.0030 ms | 0.0062 | 2.067x |
| `np.einsum('ij,jk->ik', a, b)` | 100,000 | 1.8205 ms | 0.7969 ms | 3.5330 | 4.433x |
| `np.einsum('ij,jk->ik', a, b)` | 10,000,000 | 2.3765 ms | 1.4162 ms | 6.4002 | 4.519x |
| `np.einsum(subscripts, operands)` | 1,000 | 0.0086 ms | 0.0021 ms | 0.0014 | 0.667x |
| `np.einsum(subscripts, operands)` | 100,000 | 0.6336 ms | 0.0098 ms | 0.0196 | 2.000x |
| `np.einsum(subscripts, operands)` | 10,000,000 | 63.5258 ms | 2.9374 ms | 3.7541 | 1.278x |
| `np.inner(a, b)` | 1,000 | 0.0003 ms | 0.0007 ms | 0.0006 | 2.000x |
| `np.inner(a, b)` | 100,000 | 0.0090 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.inner(a, b)` | 10,000,000 | 2.9977 ms | 2.9246 ms | 3.0051 | 1.028x |
| `np.linalg.cholesky(a)` | 1,000 | missing_backend | 0.0017 ms | 0.0046 | 2.706x |
| `np.linalg.cholesky(a)` | 100,000 | missing_backend | 0.0144 ms | 0.0188 | 1.306x |
| `np.linalg.cholesky(a)` | 10,000,000 | missing_backend | 0.0315 ms | 0.0572 | 1.816x |
| `np.linalg.cond(a)` | 1,000 | missing_backend | 0.0318 ms | 0.0351 | 1.104x |
| `np.linalg.cond(a)` | 100,000 | missing_backend | 0.2608 ms | 0.2590 | 0.993x |
| `np.linalg.cond(a)` | 10,000,000 | missing_backend | 0.5530 ms | 0.5514 | 0.997x |
| `np.linalg.det(a)` | 1,000 | missing_backend | 0.0044 ms | 0.0058 | 1.318x |
| `np.linalg.det(a)` | 100,000 | missing_backend | 0.0256 ms | 0.0268 | 1.047x |
| `np.linalg.det(a)` | 10,000,000 | missing_backend | 0.0531 ms | 0.0543 | 1.023x |
| `np.linalg.eig(a)` | 1,000 | missing_backend | 0.1034 ms | 0.1003 | 0.970x |
| `np.linalg.eig(a)` | 100,000 | missing_backend | 1.3699 ms | 1.3776 | 1.006x |
| `np.linalg.eig(a)` | 10,000,000 | missing_backend | 2.8455 ms | 3.0686 | 1.078x |
| `np.linalg.eigh(a)` | 1,000 | missing_backend | 0.0451 ms | 0.0478 | 1.060x |
| `np.linalg.eigh(a)` | 100,000 | missing_backend | 0.3887 ms | 0.3925 | 1.010x |
| `np.linalg.eigh(a)` | 10,000,000 | missing_backend | 0.7142 ms | 0.7313 | 1.024x |
| `np.linalg.eigvals(a)` | 1,000 | missing_backend | 0.0599 ms | 0.0639 | 1.067x |
| `np.linalg.eigvals(a)` | 100,000 | missing_backend | 0.8128 ms | 0.7854 | 0.966x |
| `np.linalg.eigvals(a)` | 10,000,000 | missing_backend | 1.5956 ms | 1.5325 | 0.960x |
| `np.linalg.eigvalsh(a)` | 1,000 | missing_backend | 0.0189 ms | 0.0212 | 1.122x |
| `np.linalg.eigvalsh(a)` | 100,000 | missing_backend | 0.1581 ms | 0.1584 | 1.002x |
| `np.linalg.eigvalsh(a)` | 10,000,000 | missing_backend | 0.2950 ms | 0.2934 | 0.995x |
| `np.linalg.inv(a)` | 1,000 | missing_backend | 0.0083 ms | 0.0110 | 1.325x |
| `np.linalg.inv(a)` | 100,000 | missing_backend | 0.0756 ms | 0.0786 | 1.040x |
| `np.linalg.inv(a)` | 10,000,000 | missing_backend | 0.1608 ms | 0.1821 | 1.132x |
| `np.linalg.lstsq(a, b)` | 1,000 | missing_backend | 0.0772 ms | 0.0793 | 1.027x |
| `np.linalg.lstsq(a, b)` | 100,000 | missing_backend | 0.7965 ms | 0.7911 | 0.993x |
| `np.linalg.lstsq(a, b)` | 10,000,000 | missing_backend | 1.4325 ms | 1.4275 | 0.997x |
| `np.linalg.matmul(a, b)` | 1,000 | 0.0037 ms | 0.0019 ms | 0.0024 | 1.263x |
| `np.linalg.matmul(a, b)` | 100,000 | 1.8253 ms | 0.7939 ms | 0.8002 | 1.008x |
| `np.linalg.matmul(a, b)` | 10,000,000 | 2.3761 ms | 1.4111 ms | 1.6060 | 1.138x |
| `np.linalg.matrix_norm(a, ord=2)` | 1,000 | missing_backend | 0.0313 ms | 0.0366 | 1.169x |
| `np.linalg.matrix_norm(a, ord=2)` | 100,000 | missing_backend | 0.2576 ms | 0.2602 | 1.010x |
| `np.linalg.matrix_norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5521 ms | 0.5498 | 0.996x |
| `np.linalg.matrix_power(a, -1)` | 1,000 | missing_backend | 0.0088 ms | 0.0113 | 1.284x |
| `np.linalg.matrix_power(a, -1)` | 100,000 | missing_backend | 0.0774 ms | 0.0787 | 1.017x |
| `np.linalg.matrix_power(a, -1)` | 10,000,000 | missing_backend | 0.1628 ms | 0.1825 | 1.121x |
| `np.linalg.matrix_power(a, 3)` | 1,000 | 0.0071 ms | 0.0034 ms | 0.0052 | 1.529x |
| `np.linalg.matrix_power(a, 3)` | 100,000 | 0.3497 ms | 0.1158 ms | 0.1179 | 1.018x |
| `np.linalg.matrix_power(a, 3)` | 10,000,000 | 0.3465 ms | 0.1159 ms | 0.1176 | 1.015x |
| `np.linalg.matrix_rank(a)` | 1,000 | missing_backend | 0.0323 ms | 0.0368 | 1.139x |
| `np.linalg.matrix_rank(a)` | 100,000 | missing_backend | 0.2600 ms | 0.2617 | 1.007x |
| `np.linalg.matrix_rank(a)` | 10,000,000 | missing_backend | 0.5533 ms | 0.5506 | 0.995x |
| `np.linalg.multi_dot(arrays)` | 1,000 | 0.0071 ms | 0.0034 ms | 0.0051 | 1.500x |
| `np.linalg.multi_dot(arrays)` | 100,000 | 0.3509 ms | 0.1159 ms | 0.1207 | 1.041x |
| `np.linalg.multi_dot(arrays)` | 10,000,000 | 0.3066 ms | 0.1159 ms | 0.1203 | 1.038x |
| `np.linalg.norm(a, ord=2)` | 1,000 | missing_backend | 0.0317 ms | 0.0367 | 1.158x |
| `np.linalg.norm(a, ord=2)` | 100,000 | missing_backend | 0.2578 ms | 0.2619 | 1.016x |
| `np.linalg.norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5519 ms | 0.5515 | 0.999x |
| `np.linalg.pinv(a)` | 1,000 | missing_backend | 0.0804 ms | 0.0850 | 1.057x |
| `np.linalg.pinv(a)` | 100,000 | missing_backend | 0.7248 ms | 0.7320 | 1.010x |
| `np.linalg.pinv(a)` | 10,000,000 | missing_backend | 1.4073 ms | 1.6035 | 1.139x |
| `np.linalg.qr(a)` | 1,000 | missing_backend | 0.0186 ms | 0.0229 | 1.231x |
| `np.linalg.qr(a)` | 100,000 | missing_backend | 0.1474 ms | 0.1609 | 1.092x |
| `np.linalg.qr(a)` | 10,000,000 | missing_backend | 0.3365 ms | 0.3988 | 1.185x |
| `np.linalg.slogdet(a)` | 1,000 | missing_backend | 0.0044 ms | 0.0066 | 1.500x |
| `np.linalg.slogdet(a)` | 100,000 | missing_backend | 0.0258 ms | 0.0277 | 1.074x |
| `np.linalg.slogdet(a)` | 10,000,000 | missing_backend | 0.0533 ms | 0.0552 | 1.036x |
| `np.linalg.solve(a, b)` | 1,000 | missing_backend | 0.0046 ms | 0.0075 | 1.630x |
| `np.linalg.solve(a, b)` | 100,000 | missing_backend | 0.0273 ms | 0.0295 | 1.081x |
| `np.linalg.solve(a, b)` | 10,000,000 | missing_backend | 0.0552 ms | 0.0576 | 1.043x |
| `np.linalg.svd(a)` | 1,000 | missing_backend | 0.0716 ms | 0.0749 | 1.046x |
| `np.linalg.svd(a)` | 100,000 | missing_backend | 0.6798 ms | 0.6804 | 1.001x |
| `np.linalg.svd(a)` | 10,000,000 | missing_backend | 1.2957 ms | 1.4826 | 1.144x |
| `np.linalg.svdvals(a)` | 1,000 | missing_backend | 0.0294 ms | 0.0319 | 1.085x |
| `np.linalg.svdvals(a)` | 100,000 | missing_backend | 0.2572 ms | 0.2569 | 0.999x |
| `np.linalg.svdvals(a)` | 10,000,000 | missing_backend | 0.5506 ms | 0.5445 | 0.989x |
| `np.linalg.tensordot(a, b)` | 1,000 | 0.0043 ms | 0.0023 ms | 0.0056 | 2.435x |
| `np.linalg.tensordot(a, b)` | 100,000 | 0.1702 ms | 0.0587 ms | 0.0633 | 1.078x |
| `np.linalg.tensordot(a, b)` | 10,000,000 | 0.1377 ms | 0.0584 ms | 0.0633 | 1.084x |
| `np.linalg.tensorinv(a)` | 1,000 | missing_backend | 0.0085 ms | 0.0115 | 1.353x |
| `np.linalg.tensorinv(a)` | 100,000 | missing_backend | 0.0757 ms | 0.0790 | 1.044x |
| `np.linalg.tensorinv(a)` | 10,000,000 | missing_backend | 0.1606 ms | 0.1829 | 1.139x |
| `np.linalg.tensorsolve(a, b)` | 1,000 | missing_backend | 0.0050 ms | 0.0083 | 1.660x |
| `np.linalg.tensorsolve(a, b)` | 100,000 | missing_backend | 0.0277 ms | 0.0303 | 1.094x |
| `np.linalg.tensorsolve(a, b)` | 10,000,000 | missing_backend | 0.0557 ms | 0.0586 | 1.052x |
| `np.linalg.vecdot(a, b)` | 1,000 | 0.0024 ms | 0.0007 ms | 0.0008 | 1.143x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0196 ms | 0.0090 ms | 0.0073 | 0.811x |
| `np.linalg.vecdot(a, b)` | 10,000,000 | 18.4952 ms | 2.9730 ms | 3.0120 | 1.013x |
| `np.matmul(a, b)` | 1,000 | 0.0041 ms | 0.0023 ms | 0.0022 | 0.957x |
| `np.matmul(a, b)` | 100,000 | 1.8243 ms | 0.7937 ms | 0.8000 | 1.008x |
| `np.matmul(a, b)` | 10,000,000 | 2.3773 ms | 1.4120 ms | 1.6071 | 1.138x |
| `np.matvec(a, b)` | 1,000 | 0.0015 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.matvec(a, b)` | 100,000 | 0.0692 ms | 0.0068 ms | 0.0072 | 1.059x |
| `np.matvec(a, b)` | 10,000,000 | 0.1008 ms | 0.0069 ms | 0.0071 | 1.029x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.0663 ms | 0.0449 | 0.677x |
| `np.polyfit(x, y, 3)` | 100,000 | missing_backend | 0.6043 ms | 0.4632 | 0.767x |
| `np.polyfit(x, y, 3)` | 10,000,000 | missing_backend | 4.2502 ms | 5.1652 | 1.215x |
| `np.roots(p)` | 1,000 | missing_backend | 0.0341 ms | 0.0366 | 1.073x |
| `np.roots(p)` | 100,000 | missing_backend | 0.0947 ms | 0.0977 | 1.032x |
| `np.roots(p)` | 10,000,000 | missing_backend | 0.4145 ms | 0.4097 | 0.988x |
| `np.tensordot(a, b)` | 1,000 | 0.0076 ms | 0.0013 ms | 0.0036 | 2.769x |
| `np.tensordot(a, b)` | 100,000 | 0.6327 ms | 0.0097 ms | 0.0101 | 1.041x |
| `np.tensordot(a, b)` | 10,000,000 | 63.5058 ms | 3.1128 ms | 3.0330 | 0.974x |
| `np.tensordot(a, b, axes=1)` | 1,000 | 0.0043 ms | 0.0023 ms | 0.0054 | 2.348x |
| `np.tensordot(a, b, axes=1)` | 100,000 | 3.4449 ms | 0.7964 ms | 0.8139 | 1.022x |
| `np.tensordot(a, b, axes=1)` | 10,000,000 | 2.3840 ms | 1.4137 ms | 1.6202 | 1.146x |
| `np.vdot(a, b)` | 1,000 | 0.0009 ms | 0.0006 ms | 0.0005 | 0.833x |
| `np.vdot(a, b)` | 100,000 | 0.0092 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.vdot(a, b)` | 10,000,000 | 2.9873 ms | 2.9578 ms | 2.9930 | 1.012x |
| `np.vecdot(a, b)` | 1,000 | 0.0026 ms | 0.0007 ms | 0.0006 | 0.857x |
| `np.vecdot(a, b)` | 100,000 | 0.0198 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.vecdot(a, b)` | 10,000,000 | 18.3500 ms | 2.9600 ms | 3.0103 | 1.017x |
| `np.vecmat(a, b)` | 1,000 | 0.0007 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.vecmat(a, b)` | 100,000 | 0.0311 ms | 0.0069 ms | 0.0072 | 1.043x |
| `np.vecmat(a, b)` | 10,000,000 | 0.0482 ms | 0.0069 ms | 0.0067 | 0.971x |
