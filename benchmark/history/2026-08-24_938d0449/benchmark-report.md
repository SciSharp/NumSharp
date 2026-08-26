# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell.

- Managed effective cells: **3,434**
- OpenBLAS effective cells: **58**
- Missing backend records: **44**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.1084 ms | openblas | 0.419x |
| `np.roots(p)` | float64 | 16 | missing_backend | 0.05150 ms | openblas | 0.726x |
| `np.array2string(a) (float64)` | float64 | 1 | 0.000523 ms | not_measured | managed | 44.373x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 0.9886 ms | not_measured | managed | 2.062x |
| `np.array_repr(a) (float64)` | float64 | 1 | 0.000538 ms | not_measured | managed | 37.232x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 0.9945 ms | not_measured | managed | 2.083x |
| `np.array_str(a) (float64)` | float64 | 1 | 0.000512 ms | not_measured | managed | 34.914x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 0.9928 ms | not_measured | managed | 2.079x |
| `np.can_cast(from, to) (float64)` | float64 | 1 | 0.000014 ms | not_measured | managed | 15.786x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.000027 ms | not_measured | managed | 7.556x |
| `np.common_type(a, b) (float64)` | float64 | 1 | 0.000020 ms | not_measured | managed | 42.100x |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.000028 ms | not_measured | managed | 29.429x |
| `np.copyto(dst, src) (float64)` | float64 | 1 | 0.000185 ms | not_measured | managed | 1.470x |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.000514 ms | not_measured | managed | 1.062x |
| `np.format_float_positional(x) (float64)` | float64 | 1 | 0.000433 ms | not_measured | managed | 1.127x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.000438 ms | not_measured | managed | 1.128x |
| `np.format_float_scientific(x) (float64)` | float64 | 1 | 0.000388 ms | not_measured | managed | 1.245x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.000400 ms | not_measured | managed | 1.205x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.04125 ms | not_measured | managed | 0.818x |
| `np.get_printoptions() (float64)` | float64 | 1 | 0.000233 ms | not_measured | managed | 1.365x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.000253 ms | not_measured | managed | 1.257x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1 | 0.000039 ms | not_measured | managed | 17.231x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000036 ms | not_measured | managed | 18.694x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1 | 0.000034 ms | not_measured | managed | 5.529x |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.000036 ms | not_measured | managed | 4.750x |
| `np.iterable(a) (float64)` | float64 | 1 | 0.000000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.04933 ms | not_measured | managed | 1.348x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.7831 ms | not_measured | managed | 0.388x |
| `np.min_scalar_type(value) (float64)` | float64 | 1 | 0.000003 ms | not_measured | managed | 89.333x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.000003 ms | not_measured | managed | 92.333x |
| `np.mintypecode(typechars) (float64)` | float64 | 1 | 0.000157 ms | not_measured | managed | 4.446x |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.000152 ms | not_measured | managed | 4.605x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1 | 0.00156 ms | not_measured | managed | 0.304x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.00166 ms | not_measured | managed | 0.290x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.1552 ms | not_measured | managed | 0.148x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.00258 ms | not_measured | managed | 0.685x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.00528 ms | not_measured | managed | 0.490x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.2494 ms | not_measured | managed | 0.473x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.00713 ms | not_measured | managed | 0.444x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.01046 ms | not_measured | managed | 1.454x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.00266 ms | not_measured | managed | 0.671x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1050 ms | not_measured | managed | 0.191x |
| `np.printoptions() (float64)` | float64 | 1 | 0.000119 ms | not_measured | managed | 28.840x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.000120 ms | not_measured | managed | 28.550x |
| `np.promote_types(a, b) (float64)` | float64 | 1 | 0.000015 ms | not_measured | managed | 8.133x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.000016 ms | not_measured | managed | 7.688x |
| `np.result_type(a, b) (float64)` | float64 | 1 | 0.000006 ms | not_measured | managed | 25.667x |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.000008 ms | not_measured | managed | 19.500x |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.2880 ms | not_measured | managed | 0.600x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 0.9027 ms | not_measured | managed | 1.854x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 2.3883 ms | not_measured | managed | 0.110x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.4444 ms | not_measured | managed | 0.865x |
| `np.set_printoptions() (float64)` | float64 | 1 | 0.000057 ms | not_measured | managed | 36.544x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.000057 ms | not_measured | managed | 34.386x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.01693 ms | not_measured | managed | 0.817x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.8188 ms | not_measured | managed | 0.879x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 183.7633 ms | not_measured | managed | 1.576x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.01383 ms | not_measured | managed | 0.866x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.6631 ms | not_measured | managed | 0.831x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 168.4439 ms | not_measured | managed | 1.401x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.00360 ms | not_measured | managed | 0.535x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.6540 ms | not_measured | managed | 0.642x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 67.1160 ms | not_measured | managed | 0.656x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.00580 ms | not_measured | managed | 0.691x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.6629 ms | not_measured | managed | 0.609x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 72.5547 ms | not_measured | managed | 0.688x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.01115 ms | not_measured | managed | 0.993x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 1.5605 ms | not_measured | managed | 0.904x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 157.0878 ms | not_measured | managed | 1.646x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.00876 ms | not_measured | managed | 1.108x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 1.4058 ms | not_measured | managed | 0.879x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 147.9190 ms | not_measured | managed | 1.434x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.00297 ms | not_measured | managed | 0.669x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.5938 ms | not_measured | managed | 0.631x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 59.9643 ms | not_measured | managed | 0.719x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.00334 ms | not_measured | managed | 0.998x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.5913 ms | not_measured | managed | 0.623x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 65.8301 ms | not_measured | managed | 0.728x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.00490 ms | not_measured | managed | 0.190x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.2998 ms | not_measured | managed | 0.945x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 40.5577 ms | not_measured | managed | 1.155x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.01030 ms | not_measured | managed | 0.346x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.8884 ms | not_measured | managed | 0.311x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 89.3042 ms | not_measured | managed | 0.341x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.00297 ms | not_measured | managed | 0.244x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.04941 ms | not_measured | managed | 0.126x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 8.4516 ms | not_measured | managed | 0.964x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.00391 ms | not_measured | managed | 0.203x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.09849 ms | not_measured | managed | 0.131x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 18.8628 ms | not_measured | managed | 1.359x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.00269 ms | not_measured | managed | 0.315x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.02790 ms | not_measured | managed | 0.814x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 3.8720 ms | not_measured | managed | 1.102x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.00176 ms | not_measured | managed | 0.507x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.04901 ms | not_measured | managed | 0.452x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 8.4349 ms | not_measured | managed | 0.977x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.00356 ms | not_measured | managed | 0.232x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.09717 ms | not_measured | managed | 0.224x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 20.1115 ms | not_measured | managed | 0.849x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.00258 ms | not_measured | managed | 0.308x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.01596 ms | not_measured | managed | 1.424x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 1.8532 ms | not_measured | managed | 1.686x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.00278 ms | not_measured | managed | 0.318x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.02733 ms | not_measured | managed | 0.821x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 3.9853 ms | not_measured | managed | 1.081x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.00325 ms | not_measured | managed | 0.268x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.04978 ms | not_measured | managed | 0.447x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 8.3408 ms | not_measured | managed | 0.945x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.00345 ms | not_measured | managed | 0.254x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.09919 ms | not_measured | managed | 0.238x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 19.9947 ms | not_measured | managed | 0.795x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.00291 ms | not_measured | managed | 0.282x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.01684 ms | not_measured | managed | 1.373x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 1.8724 ms | not_measured | managed | 1.768x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.00334 ms | not_measured | managed | 0.205x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.2946 ms | not_measured | managed | 0.963x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 40.9740 ms | not_measured | managed | 1.203x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.00949 ms | not_measured | managed | 0.342x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.9010 ms | not_measured | managed | 0.311x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 90.0827 ms | not_measured | managed | 0.338x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.00149 ms | not_measured | managed | 0.273x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.04674 ms | not_measured | managed | 0.147x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 8.9556 ms | not_measured | managed | 0.916x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.00209 ms | not_measured | managed | 0.221x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.09298 ms | not_measured | managed | 0.155x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 19.8491 ms | not_measured | managed | 1.367x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.00107 ms | not_measured | managed | 0.592x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.02436 ms | not_measured | managed | 1.260x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 4.0962 ms | not_measured | managed | 1.150x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.000977 ms | not_measured | managed | 0.769x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.04745 ms | not_measured | managed | 0.678x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 8.7572 ms | not_measured | managed | 0.973x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.00192 ms | not_measured | managed | 0.340x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.09455 ms | not_measured | managed | 0.301x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 20.3125 ms | not_measured | managed | 0.900x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.000807 ms | not_measured | managed | 0.743x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.01541 ms | not_measured | managed | 1.849x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 1.9384 ms | not_measured | managed | 1.955x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.00108 ms | not_measured | managed | 0.583x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.02498 ms | not_measured | managed | 1.147x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 4.2368 ms | not_measured | managed | 1.169x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.00148 ms | not_measured | managed | 0.501x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.04802 ms | not_measured | managed | 0.658x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 8.8716 ms | not_measured | managed | 0.892x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.00180 ms | not_measured | managed | 0.362x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.09651 ms | not_measured | managed | 0.319x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 20.2183 ms | not_measured | managed | 0.753x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.000834 ms | not_measured | managed | 0.716x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.01606 ms | not_measured | managed | 1.850x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 1.9409 ms | not_measured | managed | 1.920x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.00389 ms | not_measured | managed | 0.183x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.3280 ms | not_measured | managed | 0.964x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 55.2207 ms | not_measured | managed | 1.378x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.00949 ms | not_measured | managed | 0.341x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.9012 ms | not_measured | managed | 0.306x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 90.1024 ms | not_measured | managed | 0.335x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.00156 ms | not_measured | managed | 0.328x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.04762 ms | not_measured | managed | 0.141x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 10.8522 ms | not_measured | managed | 0.784x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.00213 ms | not_measured | managed | 0.224x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.09486 ms | not_measured | managed | 0.298x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 25.2025 ms | not_measured | managed | 1.364x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.000985 ms | not_measured | managed | 0.647x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.02569 ms | not_measured | managed | 1.196x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 5.6486 ms | not_measured | managed | 0.898x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.00154 ms | not_measured | managed | 0.486x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.04735 ms | not_measured | managed | 0.636x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 10.6826 ms | not_measured | managed | 0.824x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.00223 ms | not_measured | managed | 0.296x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.09902 ms | not_measured | managed | 0.338x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 28.2180 ms | not_measured | managed | 0.600x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.000955 ms | not_measured | managed | 0.629x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.01441 ms | not_measured | managed | 1.932x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 2.4216 ms | not_measured | managed | 1.615x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.00108 ms | not_measured | managed | 0.588x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.02558 ms | not_measured | managed | 1.081x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 5.5452 ms | not_measured | managed | 0.912x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.00151 ms | not_measured | managed | 0.485x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.04779 ms | not_measured | managed | 0.625x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.4981 ms | not_measured | managed | 0.810x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.00172 ms | not_measured | managed | 0.389x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.09900 ms | not_measured | managed | 0.366x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 26.8743 ms | not_measured | managed | 0.665x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.000902 ms | not_measured | managed | 0.703x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.01552 ms | not_measured | managed | 1.795x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 2.3983 ms | not_measured | managed | 1.567x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.00371 ms | not_measured | managed | 0.205x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.2976 ms | not_measured | managed | 1.004x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 40.1393 ms | not_measured | managed | 1.354x |
| `a * scalar (float16)` | float16 | 1,000 | 0.00938 ms | not_measured | managed | 0.369x |
| `a * scalar (float16)` | float16 | 100,000 | 0.8957 ms | not_measured | managed | 0.308x |
| `a * scalar (float16)` | float16 | 10,000,000 | 89.2481 ms | not_measured | managed | 0.347x |
| `a * scalar (float32)` | float32 | 1,000 | 0.00148 ms | not_measured | managed | 0.446x |
| `a * scalar (float32)` | float32 | 100,000 | 0.04884 ms | not_measured | managed | 0.124x |
| `a * scalar (float32)` | float32 | 10,000,000 | 8.6271 ms | not_measured | managed | 0.958x |
| `a * scalar (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.320x |
| `a * scalar (float64)` | float64 | 100,000 | 0.09608 ms | not_measured | managed | 0.146x |
| `a * scalar (float64)` | float64 | 10,000,000 | 18.8798 ms | not_measured | managed | 1.345x |
| `a * scalar (int16)` | int16 | 1,000 | 0.00106 ms | not_measured | managed | 0.693x |
| `a * scalar (int16)` | int16 | 100,000 | 0.02484 ms | not_measured | managed | 0.958x |
| `a * scalar (int16)` | int16 | 10,000,000 | 3.7047 ms | not_measured | managed | 1.139x |
| `a * scalar (int32)` | int32 | 1,000 | 0.00130 ms | not_measured | managed | 0.582x |
| `a * scalar (int32)` | int32 | 100,000 | 0.04825 ms | not_measured | managed | 0.459x |
| `a * scalar (int32)` | int32 | 10,000,000 | 8.4826 ms | not_measured | managed | 0.976x |
| `a * scalar (int64)` | int64 | 1,000 | 0.00221 ms | not_measured | managed | 0.338x |
| `a * scalar (int64)` | int64 | 100,000 | 0.09844 ms | not_measured | managed | 0.222x |
| `a * scalar (int64)` | int64 | 10,000,000 | 20.1264 ms | not_measured | managed | 0.737x |
| `a * scalar (int8)` | int8 | 1,000 | 0.000934 ms | not_measured | managed | 0.755x |
| `a * scalar (int8)` | int8 | 100,000 | 0.01464 ms | not_measured | managed | 1.554x |
| `a * scalar (int8)` | int8 | 10,000,000 | 1.8945 ms | not_measured | managed | 1.687x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.00103 ms | not_measured | managed | 0.713x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.02488 ms | not_measured | managed | 0.913x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 4.0576 ms | not_measured | managed | 1.122x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.00148 ms | not_measured | managed | 0.514x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.04823 ms | not_measured | managed | 0.460x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 8.5773 ms | not_measured | managed | 0.934x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.00210 ms | not_measured | managed | 0.359x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.1003 ms | not_measured | managed | 0.240x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 20.1635 ms | not_measured | managed | 0.750x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.000854 ms | not_measured | managed | 0.831x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.01513 ms | not_measured | managed | 1.558x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 1.9025 ms | not_measured | managed | 1.703x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.00428 ms | not_measured | managed | 0.213x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.3012 ms | not_measured | managed | 1.029x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 40.8303 ms | not_measured | managed | 1.176x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.01059 ms | not_measured | managed | 0.351x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.8964 ms | not_measured | managed | 0.335x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 89.3584 ms | not_measured | managed | 0.343x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.00308 ms | not_measured | managed | 0.224x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.05073 ms | not_measured | managed | 0.123x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 8.5920 ms | not_measured | managed | 0.908x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.00373 ms | not_measured | managed | 0.204x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.09446 ms | not_measured | managed | 0.138x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 18.9012 ms | not_measured | managed | 1.287x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.00271 ms | not_measured | managed | 0.328x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.02794 ms | not_measured | managed | 0.899x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 4.0945 ms | not_measured | managed | 1.119x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.00230 ms | not_measured | managed | 0.381x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.05187 ms | not_measured | managed | 0.505x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 8.2009 ms | not_measured | managed | 0.994x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.00377 ms | not_measured | managed | 0.274x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.1042 ms | not_measured | managed | 0.228x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 18.9897 ms | not_measured | managed | 0.772x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.00235 ms | not_measured | managed | 0.357x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.01643 ms | not_measured | managed | 2.077x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 1.7984 ms | not_measured | managed | 1.997x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.00283 ms | not_measured | managed | 0.317x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.02807 ms | not_measured | managed | 0.915x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 4.0632 ms | not_measured | managed | 1.082x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.00310 ms | not_measured | managed | 0.284x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.05132 ms | not_measured | managed | 0.472x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 8.5668 ms | not_measured | managed | 0.894x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.00359 ms | not_measured | managed | 0.264x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.09350 ms | not_measured | managed | 0.277x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 18.9477 ms | not_measured | managed | 0.787x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.00281 ms | not_measured | managed | 0.300x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.01619 ms | not_measured | managed | 1.592x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 1.8263 ms | not_measured | managed | 1.923x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.00334 ms | not_measured | managed | 0.225x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.4483 ms | not_measured | managed | 0.710x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 56.8549 ms | not_measured | managed | 1.084x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.00932 ms | not_measured | managed | 0.361x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.9072 ms | not_measured | managed | 0.320x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 89.7597 ms | not_measured | managed | 0.354x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.00155 ms | not_measured | managed | 0.270x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.05414 ms | not_measured | managed | 0.124x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 10.7341 ms | not_measured | managed | 0.774x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.00214 ms | not_measured | managed | 0.225x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.1126 ms | not_measured | managed | 0.244x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 25.4770 ms | not_measured | managed | 1.336x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.00150 ms | not_measured | managed | 0.467x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.02608 ms | not_measured | managed | 1.192x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 7.3251 ms | not_measured | managed | 0.731x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.00129 ms | not_measured | managed | 0.534x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.05681 ms | not_measured | managed | 0.552x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 11.0592 ms | not_measured | managed | 0.754x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.00256 ms | not_measured | managed | 0.265x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.1165 ms | not_measured | managed | 0.302x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 27.5806 ms | not_measured | managed | 0.594x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.00122 ms | not_measured | managed | 0.504x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.01588 ms | not_measured | managed | 1.837x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 3.5686 ms | not_measured | managed | 1.099x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.00103 ms | not_measured | managed | 0.641x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.02700 ms | not_measured | managed | 1.192x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.2092 ms | not_measured | managed | 0.832x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.00145 ms | not_measured | managed | 0.448x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.05562 ms | not_measured | managed | 0.618x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.4807 ms | not_measured | managed | 0.791x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.00199 ms | not_measured | managed | 0.342x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.1062 ms | not_measured | managed | 0.311x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 33.8152 ms | not_measured | managed | 0.606x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.00114 ms | not_measured | managed | 0.587x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.01590 ms | not_measured | managed | 1.882x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.3718 ms | not_measured | managed | 1.134x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.00344 ms | not_measured | managed | 0.217x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.2958 ms | not_measured | managed | 1.038x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 40.7819 ms | not_measured | managed | 1.156x |
| `a + scalar (float16)` | float16 | 1,000 | 0.00947 ms | not_measured | managed | 0.389x |
| `a + scalar (float16)` | float16 | 100,000 | 0.8943 ms | not_measured | managed | 0.335x |
| `a + scalar (float16)` | float16 | 10,000,000 | 88.9785 ms | not_measured | managed | 0.342x |
| `a + scalar (float32)` | float32 | 1,000 | 0.00131 ms | not_measured | managed | 0.406x |
| `a + scalar (float32)` | float32 | 100,000 | 0.04769 ms | not_measured | managed | 0.127x |
| `a + scalar (float32)` | float32 | 10,000,000 | 8.2334 ms | not_measured | managed | 0.949x |
| `a + scalar (float64)` | float64 | 1,000 | 0.00201 ms | not_measured | managed | 0.303x |
| `a + scalar (float64)` | float64 | 100,000 | 0.1012 ms | not_measured | managed | 0.132x |
| `a + scalar (float64)` | float64 | 10,000,000 | 18.8852 ms | not_measured | managed | 1.312x |
| `a + scalar (int16)` | int16 | 1,000 | 0.000991 ms | not_measured | managed | 0.757x |
| `a + scalar (int16)` | int16 | 100,000 | 0.02441 ms | not_measured | managed | 1.018x |
| `a + scalar (int16)` | int16 | 10,000,000 | 4.0155 ms | not_measured | managed | 1.156x |
| `a + scalar (int32)` | int32 | 1,000 | 0.000984 ms | not_measured | managed | 0.804x |
| `a + scalar (int32)` | int32 | 100,000 | 0.05022 ms | not_measured | managed | 0.534x |
| `a + scalar (int32)` | int32 | 10,000,000 | 8.1527 ms | not_measured | managed | 0.949x |
| `a + scalar (int64)` | int64 | 1,000 | 0.00210 ms | not_measured | managed | 0.399x |
| `a + scalar (int64)` | int64 | 100,000 | 0.1040 ms | not_measured | managed | 0.225x |
| `a + scalar (int64)` | int64 | 10,000,000 | 18.9987 ms | not_measured | managed | 0.788x |
| `a + scalar (int8)` | int8 | 1,000 | 0.000900 ms | not_measured | managed | 0.808x |
| `a + scalar (int8)` | int8 | 100,000 | 0.01466 ms | not_measured | managed | 1.749x |
| `a + scalar (int8)` | int8 | 10,000,000 | 1.7655 ms | not_measured | managed | 2.005x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.00108 ms | not_measured | managed | 0.705x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.02629 ms | not_measured | managed | 0.954x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 4.1164 ms | not_measured | managed | 1.083x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.00138 ms | not_measured | managed | 0.546x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.04998 ms | not_measured | managed | 0.491x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 8.3593 ms | not_measured | managed | 0.918x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.00208 ms | not_measured | managed | 0.388x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.09846 ms | not_measured | managed | 0.262x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 18.9388 ms | not_measured | managed | 0.834x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.000818 ms | not_measured | managed | 0.928x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.01474 ms | not_measured | managed | 1.806x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 1.7927 ms | not_measured | managed | 2.000x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.00318 ms | not_measured | managed | 0.216x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.3288 ms | not_measured | managed | 0.919x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 55.9588 ms | not_measured | managed | 1.229x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.00952 ms | not_measured | managed | 0.386x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.9063 ms | not_measured | managed | 0.308x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 90.5417 ms | not_measured | managed | 0.360x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.00145 ms | not_measured | managed | 0.323x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.04782 ms | not_measured | managed | 0.142x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 10.7295 ms | not_measured | managed | 0.772x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.00191 ms | not_measured | managed | 0.254x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.09322 ms | not_measured | managed | 0.303x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 24.8015 ms | not_measured | managed | 1.330x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.00116 ms | not_measured | managed | 0.552x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.02580 ms | not_measured | managed | 1.159x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 5.6505 ms | not_measured | managed | 0.932x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.00141 ms | not_measured | managed | 0.462x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.04791 ms | not_measured | managed | 0.660x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 10.6481 ms | not_measured | managed | 0.892x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.00227 ms | not_measured | managed | 0.297x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.09892 ms | not_measured | managed | 0.348x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 25.0081 ms | not_measured | managed | 0.649x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.000942 ms | not_measured | managed | 0.653x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.01394 ms | not_measured | managed | 2.108x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 2.2055 ms | not_measured | managed | 1.740x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.000986 ms | not_measured | managed | 0.677x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.02455 ms | not_measured | managed | 1.232x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 5.6878 ms | not_measured | managed | 0.924x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.00158 ms | not_measured | managed | 0.411x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.04846 ms | not_measured | managed | 0.615x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 10.8018 ms | not_measured | managed | 0.774x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.00195 ms | not_measured | managed | 0.339x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.09823 ms | not_measured | managed | 0.369x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 24.9826 ms | not_measured | managed | 0.669x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.00108 ms | not_measured | managed | 0.589x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.01448 ms | not_measured | managed | 2.028x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 2.2078 ms | not_measured | managed | 1.739x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.00281 ms | not_measured | managed | 0.256x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.2933 ms | not_measured | managed | 1.021x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 40.6951 ms | not_measured | managed | 1.263x |
| `a - scalar (float16)` | float16 | 1,000 | 0.00946 ms | not_measured | managed | 0.364x |
| `a - scalar (float16)` | float16 | 100,000 | 0.9004 ms | not_measured | managed | 0.313x |
| `a - scalar (float16)` | float16 | 10,000,000 | 90.0217 ms | not_measured | managed | 0.340x |
| `a - scalar (float32)` | float32 | 1,000 | 0.00147 ms | not_measured | managed | 0.400x |
| `a - scalar (float32)` | float32 | 100,000 | 0.04846 ms | not_measured | managed | 0.127x |
| `a - scalar (float32)` | float32 | 10,000,000 | 8.5533 ms | not_measured | managed | 1.011x |
| `a - scalar (float64)` | float64 | 1,000 | 0.00198 ms | not_measured | managed | 0.308x |
| `a - scalar (float64)` | float64 | 100,000 | 0.09505 ms | not_measured | managed | 0.135x |
| `a - scalar (float64)` | float64 | 10,000,000 | 19.1253 ms | not_measured | managed | 1.366x |
| `a - scalar (int16)` | int16 | 1,000 | 0.000995 ms | not_measured | managed | 0.755x |
| `a - scalar (int16)` | int16 | 100,000 | 0.02574 ms | not_measured | managed | 0.958x |
| `a - scalar (int16)` | int16 | 10,000,000 | 4.0597 ms | not_measured | managed | 1.126x |
| `a - scalar (int32)` | int32 | 1,000 | 0.00145 ms | not_measured | managed | 0.709x |
| `a - scalar (int32)` | int32 | 100,000 | 0.04964 ms | not_measured | managed | 0.485x |
| `a - scalar (int32)` | int32 | 10,000,000 | 8.5851 ms | not_measured | managed | 0.940x |
| `a - scalar (int64)` | int64 | 1,000 | 0.00206 ms | not_measured | managed | 0.364x |
| `a - scalar (int64)` | int64 | 100,000 | 0.09727 ms | not_measured | managed | 0.253x |
| `a - scalar (int64)` | int64 | 10,000,000 | 19.0514 ms | not_measured | managed | 0.773x |
| `a - scalar (int8)` | int8 | 1,000 | 0.000870 ms | not_measured | managed | 0.830x |
| `a - scalar (int8)` | int8 | 100,000 | 0.01456 ms | not_measured | managed | 1.725x |
| `a - scalar (int8)` | int8 | 10,000,000 | 1.8380 ms | not_measured | managed | 1.892x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.00108 ms | not_measured | managed | 0.700x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.02585 ms | not_measured | managed | 0.944x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 4.0803 ms | not_measured | managed | 1.088x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.00149 ms | not_measured | managed | 0.511x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.04988 ms | not_measured | managed | 0.470x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 8.4594 ms | not_measured | managed | 0.922x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.00215 ms | not_measured | managed | 0.360x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.09845 ms | not_measured | managed | 0.291x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 19.0785 ms | not_measured | managed | 0.837x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.000791 ms | not_measured | managed | 0.970x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.01439 ms | not_measured | managed | 1.688x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 1.8054 ms | not_measured | managed | 1.855x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.00150 ms | not_measured | managed | 0.320x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.06176 ms | not_measured | managed | 0.193x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 10.6146 ms | not_measured | managed | 0.788x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.00235 ms | not_measured | managed | 0.316x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.1956 ms | not_measured | managed | 0.198x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 24.7402 ms | not_measured | managed | 1.357x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.00371 ms | not_measured | managed | 0.470x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.1956 ms | not_measured | managed | 0.434x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 23.9443 ms | not_measured | managed | 0.890x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.00446 ms | not_measured | managed | 0.374x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.2028 ms | not_measured | managed | 0.382x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 28.3602 ms | not_measured | managed | 0.898x |
| `a / scalar (float32)` | float32 | 1,000 | 0.00161 ms | not_measured | managed | 0.364x |
| `a / scalar (float32)` | float32 | 100,000 | 0.05793 ms | not_measured | managed | 0.212x |
| `a / scalar (float32)` | float32 | 10,000,000 | 8.5495 ms | not_measured | managed | 1.699x |
| `a / scalar (float64)` | float64 | 1,000 | 0.00255 ms | not_measured | managed | 0.343x |
| `a / scalar (float64)` | float64 | 100,000 | 0.1927 ms | not_measured | managed | 0.195x |
| `a / scalar (float64)` | float64 | 10,000,000 | 23.7174 ms | not_measured | managed | 1.273x |
| `a / scalar (int32)` | int32 | 1,000 | 0.00676 ms | not_measured | managed | 0.231x |
| `a / scalar (int32)` | int32 | 100,000 | 0.1883 ms | not_measured | managed | 0.347x |
| `a / scalar (int32)` | int32 | 10,000,000 | 24.1508 ms | not_measured | managed | 0.691x |
| `a / scalar (int64)` | int64 | 1,000 | 0.00469 ms | not_measured | managed | 0.349x |
| `a / scalar (int64)` | int64 | 100,000 | 0.2019 ms | not_measured | managed | 0.297x |
| `a / scalar (int64)` | int64 | 10,000,000 | 24.1549 ms | not_measured | managed | 0.998x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.00307 ms | not_measured | managed | 0.217x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.4566 ms | not_measured | managed | 0.704x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 56.8528 ms | not_measured | managed | 1.108x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.00950 ms | not_measured | managed | 0.389x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.9051 ms | not_measured | managed | 0.331x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 88.9819 ms | not_measured | managed | 0.347x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.00150 ms | not_measured | managed | 0.275x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.05181 ms | not_measured | managed | 0.129x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 10.5703 ms | not_measured | managed | 0.797x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.00196 ms | not_measured | managed | 0.247x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.1113 ms | not_measured | managed | 0.260x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 25.4658 ms | not_measured | managed | 1.316x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.00127 ms | not_measured | managed | 0.528x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.02800 ms | not_measured | managed | 1.117x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 6.9555 ms | not_measured | managed | 0.765x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.00142 ms | not_measured | managed | 0.488x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.05676 ms | not_measured | managed | 0.542x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 11.0957 ms | not_measured | managed | 0.766x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.00209 ms | not_measured | managed | 0.326x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.1133 ms | not_measured | managed | 0.299x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 33.7556 ms | not_measured | managed | 0.483x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.000988 ms | not_measured | managed | 0.632x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.01544 ms | not_measured | managed | 1.905x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 3.4585 ms | not_measured | managed | 1.123x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.000986 ms | not_measured | managed | 0.689x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.02713 ms | not_measured | managed | 1.208x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 6.2605 ms | not_measured | managed | 0.806x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.00137 ms | not_measured | managed | 0.481x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.05581 ms | not_measured | managed | 0.554x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 11.0214 ms | not_measured | managed | 0.754x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.00182 ms | not_measured | managed | 0.374x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.1014 ms | not_measured | managed | 0.328x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 34.5368 ms | not_measured | managed | 0.492x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.00115 ms | not_measured | managed | 0.593x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.01620 ms | not_measured | managed | 1.847x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 3.5286 ms | not_measured | managed | 1.084x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.00151 ms | not_measured | managed | 0.309x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.06305 ms | not_measured | managed | 0.202x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 10.6998 ms | not_measured | managed | 0.766x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.00254 ms | not_measured | managed | 0.284x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.1933 ms | not_measured | managed | 0.201x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 25.5904 ms | not_measured | managed | 1.333x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.00372 ms | not_measured | managed | 0.444x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.2015 ms | not_measured | managed | 0.426x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 23.9005 ms | not_measured | managed | 0.908x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.00413 ms | not_measured | managed | 0.390x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.1973 ms | not_measured | managed | 0.391x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 28.7220 ms | not_measured | managed | 0.916x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.01144 ms | not_measured | managed | 1.009x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 1.5765 ms | not_measured | managed | 0.899x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 157.2932 ms | not_measured | managed | 1.499x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.00950 ms | not_measured | managed | 1.013x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 1.4163 ms | not_measured | managed | 0.868x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 148.8465 ms | not_measured | managed | 1.542x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.00288 ms | not_measured | managed | 0.579x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.6175 ms | not_measured | managed | 0.599x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 61.7989 ms | not_measured | managed | 0.674x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.00263 ms | not_measured | managed | 0.861x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.6162 ms | not_measured | managed | 0.588x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 67.5486 ms | not_measured | managed | 0.882x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.01122 ms | not_measured | managed | 0.997x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 1.5709 ms | not_measured | managed | 0.908x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 156.9542 ms | not_measured | managed | 1.646x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.00887 ms | not_measured | managed | 1.121x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 1.4317 ms | not_measured | managed | 0.859x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 147.9287 ms | not_measured | managed | 1.407x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.00274 ms | not_measured | managed | 0.690x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.5900 ms | not_measured | managed | 0.647x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 60.1027 ms | not_measured | managed | 0.703x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.00318 ms | not_measured | managed | 1.075x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.5975 ms | not_measured | managed | 0.640x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 65.8445 ms | not_measured | managed | 0.724x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.00398 ms | not_measured | managed | 0.189x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.3244 ms | not_measured | managed | 0.933x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 54.6594 ms | not_measured | managed | 1.293x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.00947 ms | not_measured | managed | 0.338x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.9082 ms | not_measured | managed | 0.304x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 89.9276 ms | not_measured | managed | 0.342x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.00152 ms | not_measured | managed | 0.279x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.04721 ms | not_measured | managed | 0.143x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 10.6128 ms | not_measured | managed | 0.782x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.00219 ms | not_measured | managed | 0.243x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.09587 ms | not_measured | managed | 0.300x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 24.8499 ms | not_measured | managed | 1.294x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.000975 ms | not_measured | managed | 0.678x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.02601 ms | not_measured | managed | 1.247x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 5.4462 ms | not_measured | managed | 0.916x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.00136 ms | not_measured | managed | 0.572x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.05005 ms | not_measured | managed | 0.602x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 10.8757 ms | not_measured | managed | 0.839x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.00189 ms | not_measured | managed | 0.387x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.1015 ms | not_measured | managed | 0.335x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 28.2219 ms | not_measured | managed | 0.597x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.000981 ms | not_measured | managed | 0.636x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.01575 ms | not_measured | managed | 1.800x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 2.4659 ms | not_measured | managed | 1.557x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.00107 ms | not_measured | managed | 0.599x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.02569 ms | not_measured | managed | 1.098x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 5.1401 ms | not_measured | managed | 0.994x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.00154 ms | not_measured | managed | 0.469x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.04787 ms | not_measured | managed | 0.634x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 11.3593 ms | not_measured | managed | 0.765x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.00221 ms | not_measured | managed | 0.321x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.09946 ms | not_measured | managed | 0.369x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 26.6763 ms | not_measured | managed | 0.640x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.00106 ms | not_measured | managed | 0.580x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.01599 ms | not_measured | managed | 1.768x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 2.3748 ms | not_measured | managed | 1.666x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.00320 ms | not_measured | managed | 0.211x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.3348 ms | not_measured | managed | 0.920x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 56.1425 ms | not_measured | managed | 1.377x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.00950 ms | not_measured | managed | 0.373x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.9085 ms | not_measured | managed | 0.309x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 90.6594 ms | not_measured | managed | 0.339x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.00138 ms | not_measured | managed | 0.303x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.04925 ms | not_measured | managed | 0.138x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 10.8132 ms | not_measured | managed | 0.769x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.00200 ms | not_measured | managed | 0.245x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.09733 ms | not_measured | managed | 0.300x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 25.2117 ms | not_measured | managed | 1.387x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.00111 ms | not_measured | managed | 0.589x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.02588 ms | not_measured | managed | 1.223x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 5.9004 ms | not_measured | managed | 0.880x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.00140 ms | not_measured | managed | 0.469x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.04573 ms | not_measured | managed | 0.651x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 10.7255 ms | not_measured | managed | 0.858x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.00205 ms | not_measured | managed | 0.339x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.09605 ms | not_measured | managed | 0.352x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 25.1767 ms | not_measured | managed | 0.652x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.000895 ms | not_measured | managed | 0.714x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.01427 ms | not_measured | managed | 2.063x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 2.3124 ms | not_measured | managed | 1.700x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.00105 ms | not_measured | managed | 0.765x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.02383 ms | not_measured | managed | 1.271x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 5.7048 ms | not_measured | managed | 0.898x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.00133 ms | not_measured | managed | 0.499x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.04821 ms | not_measured | managed | 0.609x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 10.0859 ms | not_measured | managed | 0.842x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.00192 ms | not_measured | managed | 0.349x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.09353 ms | not_measured | managed | 0.378x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 25.3541 ms | not_measured | managed | 0.671x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.000973 ms | not_measured | managed | 0.676x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.01440 ms | not_measured | managed | 2.064x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 2.1793 ms | not_measured | managed | 1.771x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.00152 ms | not_measured | managed | 0.304x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.06400 ms | not_measured | managed | 0.187x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 10.6914 ms | not_measured | managed | 0.781x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.00297 ms | not_measured | managed | 0.302x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.1974 ms | not_measured | managed | 0.196x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 25.4176 ms | not_measured | managed | 1.402x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.00461 ms | not_measured | managed | 0.360x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.2005 ms | not_measured | managed | 0.425x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 23.8985 ms | not_measured | managed | 0.852x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.00452 ms | not_measured | managed | 0.356x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.1989 ms | not_measured | managed | 0.388x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 28.7648 ms | not_measured | managed | 1.309x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.00354 ms | not_measured | managed | 0.221x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.2933 ms | not_measured | managed | 1.008x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 40.5564 ms | not_measured | managed | 1.211x |
| `scalar - a (float16)` | float16 | 1,000 | 0.00953 ms | not_measured | managed | 0.358x |
| `scalar - a (float16)` | float16 | 100,000 | 0.9079 ms | not_measured | managed | 0.304x |
| `scalar - a (float16)` | float16 | 10,000,000 | 90.6570 ms | not_measured | managed | 0.338x |
| `scalar - a (float32)` | float32 | 1,000 | 0.00156 ms | not_measured | managed | 0.382x |
| `scalar - a (float32)` | float32 | 100,000 | 0.04645 ms | not_measured | managed | 0.132x |
| `scalar - a (float32)` | float32 | 10,000,000 | 8.6181 ms | not_measured | managed | 0.875x |
| `scalar - a (float64)` | float64 | 1,000 | 0.00219 ms | not_measured | managed | 0.287x |
| `scalar - a (float64)` | float64 | 100,000 | 0.09775 ms | not_measured | managed | 0.137x |
| `scalar - a (float64)` | float64 | 10,000,000 | 19.1823 ms | not_measured | managed | 1.301x |
| `scalar - a (int16)` | int16 | 1,000 | 0.000682 ms | not_measured | managed | 1.164x |
| `scalar - a (int16)` | int16 | 100,000 | 0.02515 ms | not_measured | managed | 1.009x |
| `scalar - a (int16)` | int16 | 10,000,000 | 3.8181 ms | not_measured | managed | 1.188x |
| `scalar - a (int32)` | int32 | 1,000 | 0.00157 ms | not_measured | managed | 0.492x |
| `scalar - a (int32)` | int32 | 100,000 | 0.04973 ms | not_measured | managed | 0.492x |
| `scalar - a (int32)` | int32 | 10,000,000 | 8.2819 ms | not_measured | managed | 0.967x |
| `scalar - a (int64)` | int64 | 1,000 | 0.00212 ms | not_measured | managed | 0.374x |
| `scalar - a (int64)` | int64 | 100,000 | 0.09752 ms | not_measured | managed | 0.249x |
| `scalar - a (int64)` | int64 | 10,000,000 | 19.0541 ms | not_measured | managed | 0.765x |
| `scalar - a (int8)` | int8 | 1,000 | 0.000847 ms | not_measured | managed | 0.884x |
| `scalar - a (int8)` | int8 | 100,000 | 0.01428 ms | not_measured | managed | 1.720x |
| `scalar - a (int8)` | int8 | 10,000,000 | 1.7793 ms | not_measured | managed | 1.883x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.000902 ms | not_measured | managed | 0.864x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.02588 ms | not_measured | managed | 0.942x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 3.9843 ms | not_measured | managed | 1.154x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.00145 ms | not_measured | managed | 0.538x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.04780 ms | not_measured | managed | 0.512x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 8.6441 ms | not_measured | managed | 0.897x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.00199 ms | not_measured | managed | 0.398x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.09785 ms | not_measured | managed | 0.287x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 19.0859 ms | not_measured | managed | 0.780x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.000826 ms | not_measured | managed | 0.958x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.01422 ms | not_measured | managed | 1.729x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 1.8013 ms | not_measured | managed | 1.850x |
| `scalar / a (float32)` | float32 | 1,000 | 0.00112 ms | not_measured | managed | 0.553x |
| `scalar / a (float32)` | float32 | 100,000 | 0.05711 ms | not_measured | managed | 0.247x |
| `scalar / a (float32)` | float32 | 10,000,000 | 8.7055 ms | not_measured | managed | 1.778x |
| `scalar / a (float64)` | float64 | 1,000 | 0.00267 ms | not_measured | managed | 0.338x |
| `scalar / a (float64)` | float64 | 100,000 | 0.1847 ms | not_measured | managed | 0.204x |
| `scalar / a (float64)` | float64 | 10,000,000 | 23.7872 ms | not_measured | managed | 1.345x |
| `scalar / a (int32)` | int32 | 1,000 | 0.00538 ms | not_measured | managed | 0.292x |
| `scalar / a (int32)` | int32 | 100,000 | 0.2020 ms | not_measured | managed | 0.319x |
| `scalar / a (int32)` | int32 | 10,000,000 | 23.7664 ms | not_measured | managed | 0.700x |
| `scalar / a (int64)` | int64 | 1,000 | 0.00497 ms | not_measured | managed | 0.329x |
| `scalar / a (int64)` | int64 | 100,000 | 0.2000 ms | not_measured | managed | 0.293x |
| `scalar / a (int64)` | int64 | 10,000,000 | 24.0985 ms | not_measured | managed | 0.837x |
| `a & b (bool)` | bool | 1,000 | 0.000862 ms | not_measured | managed | 0.379x |
| `a & b (bool)` | bool | 100,000 | 0.04972 ms | not_measured | managed | 0.053x |
| `a & b (bool)` | bool | 10,000,000 | 5.2631 ms | not_measured | managed | 0.384x |
| `a & b (int16)` | int16 | 1,000 | 0.00120 ms | not_measured | managed | 0.564x |
| `a & b (int16)` | int16 | 100,000 | 0.02527 ms | not_measured | managed | 1.235x |
| `a & b (int16)` | int16 | 10,000,000 | 5.5364 ms | not_measured | managed | 1.594x |
| `a & b (int32)` | int32 | 1,000 | 0.00141 ms | not_measured | managed | 0.504x |
| `a & b (int32)` | int32 | 100,000 | 0.04803 ms | not_measured | managed | 0.599x |
| `a & b (int32)` | int32 | 10,000,000 | 11.1657 ms | not_measured | managed | 1.478x |
| `a & b (int64)` | int64 | 1,000 | 0.00175 ms | not_measured | managed | 0.380x |
| `a & b (int64)` | int64 | 100,000 | 0.09840 ms | not_measured | managed | 0.369x |
| `a & b (int64)` | int64 | 10,000,000 | 25.1819 ms | not_measured | managed | 1.312x |
| `a & b (int8)` | int8 | 1,000 | 0.000870 ms | not_measured | managed | 0.699x |
| `a & b (int8)` | int8 | 100,000 | 0.01318 ms | not_measured | managed | 2.166x |
| `a & b (int8)` | int8 | 10,000,000 | 2.1860 ms | not_measured | managed | 2.824x |
| `a & b (uint16)` | uint16 | 1,000 | 0.00122 ms | not_measured | managed | 0.534x |
| `a & b (uint16)` | uint16 | 100,000 | 0.02417 ms | not_measured | managed | 1.238x |
| `a & b (uint16)` | uint16 | 10,000,000 | 5.7427 ms | not_measured | managed | 1.505x |
| `a & b (uint32)` | uint32 | 1,000 | 0.00135 ms | not_measured | managed | 0.518x |
| `a & b (uint32)` | uint32 | 100,000 | 0.05334 ms | not_measured | managed | 0.546x |
| `a & b (uint32)` | uint32 | 10,000,000 | 10.7503 ms | not_measured | managed | 1.545x |
| `a & b (uint64)` | uint64 | 1,000 | 0.00194 ms | not_measured | managed | 0.348x |
| `a & b (uint64)` | uint64 | 100,000 | 0.09529 ms | not_measured | managed | 0.393x |
| `a & b (uint64)` | uint64 | 10,000,000 | 25.2690 ms | not_measured | managed | 1.237x |
| `a & b (uint8)` | uint8 | 1,000 | 0.000924 ms | not_measured | managed | 0.674x |
| `a & b (uint8)` | uint8 | 100,000 | 0.01296 ms | not_measured | managed | 2.209x |
| `a & b (uint8)` | uint8 | 10,000,000 | 2.1618 ms | not_measured | managed | 2.810x |
| `a ^ b (bool)` | bool | 1,000 | 0.00128 ms | not_measured | managed | 0.256x |
| `a ^ b (bool)` | bool | 100,000 | 0.05061 ms | not_measured | managed | 0.054x |
| `a ^ b (bool)` | bool | 10,000,000 | 5.2375 ms | not_measured | managed | 0.375x |
| `a ^ b (int16)` | int16 | 1,000 | 0.00101 ms | not_measured | managed | 0.625x |
| `a ^ b (int16)` | int16 | 100,000 | 0.02576 ms | not_measured | managed | 1.119x |
| `a ^ b (int16)` | int16 | 10,000,000 | 5.6196 ms | not_measured | managed | 1.562x |
| `a ^ b (int32)` | int32 | 1,000 | 0.00134 ms | not_measured | managed | 0.486x |
| `a ^ b (int32)` | int32 | 100,000 | 0.04703 ms | not_measured | managed | 0.677x |
| `a ^ b (int32)` | int32 | 10,000,000 | 11.0689 ms | not_measured | managed | 1.497x |
| `a ^ b (int64)` | int64 | 1,000 | 0.00198 ms | not_measured | managed | 0.332x |
| `a ^ b (int64)` | int64 | 100,000 | 0.09676 ms | not_measured | managed | 0.365x |
| `a ^ b (int64)` | int64 | 10,000,000 | 24.9526 ms | not_measured | managed | 1.359x |
| `a ^ b (int8)` | int8 | 1,000 | 0.000786 ms | not_measured | managed | 0.782x |
| `a ^ b (int8)` | int8 | 100,000 | 0.01530 ms | not_measured | managed | 1.868x |
| `a ^ b (int8)` | int8 | 10,000,000 | 2.1715 ms | not_measured | managed | 2.810x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.00121 ms | not_measured | managed | 0.544x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.02467 ms | not_measured | managed | 1.216x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 5.7217 ms | not_measured | managed | 1.542x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.00132 ms | not_measured | managed | 0.497x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.04791 ms | not_measured | managed | 0.627x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 11.0675 ms | not_measured | managed | 1.477x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.00184 ms | not_measured | managed | 0.354x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.09378 ms | not_measured | managed | 0.378x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 25.4176 ms | not_measured | managed | 1.231x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.00109 ms | not_measured | managed | 0.564x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.01339 ms | not_measured | managed | 2.151x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 2.1486 ms | not_measured | managed | 2.846x |
| `a | b (bool)` | bool | 1,000 | 0.00130 ms | not_measured | managed | 0.256x |
| `a | b (bool)` | bool | 100,000 | 0.04951 ms | not_measured | managed | 0.052x |
| `a | b (bool)` | bool | 10,000,000 | 5.2453 ms | not_measured | managed | 0.378x |
| `a | b (int16)` | int16 | 1,000 | 0.00128 ms | not_measured | managed | 0.552x |
| `a | b (int16)` | int16 | 100,000 | 0.02554 ms | not_measured | managed | 1.166x |
| `a | b (int16)` | int16 | 10,000,000 | 5.6591 ms | not_measured | managed | 1.505x |
| `a | b (int32)` | int32 | 1,000 | 0.00152 ms | not_measured | managed | 0.425x |
| `a | b (int32)` | int32 | 100,000 | 0.04805 ms | not_measured | managed | 0.656x |
| `a | b (int32)` | int32 | 10,000,000 | 11.1827 ms | not_measured | managed | 1.447x |
| `a | b (int64)` | int64 | 1,000 | 0.00200 ms | not_measured | managed | 0.326x |
| `a | b (int64)` | int64 | 100,000 | 0.09897 ms | not_measured | managed | 0.358x |
| `a | b (int64)` | int64 | 10,000,000 | 25.2992 ms | not_measured | managed | 1.282x |
| `a | b (int8)` | int8 | 1,000 | 0.000922 ms | not_measured | managed | 0.677x |
| `a | b (int8)` | int8 | 100,000 | 0.01308 ms | not_measured | managed | 2.300x |
| `a | b (int8)` | int8 | 10,000,000 | 2.1958 ms | not_measured | managed | 2.732x |
| `a | b (uint16)` | uint16 | 1,000 | 0.00131 ms | not_measured | managed | 0.492x |
| `a | b (uint16)` | uint16 | 100,000 | 0.02534 ms | not_measured | managed | 1.127x |
| `a | b (uint16)` | uint16 | 10,000,000 | 5.7816 ms | not_measured | managed | 1.461x |
| `a | b (uint32)` | uint32 | 1,000 | 0.00128 ms | not_measured | managed | 0.515x |
| `a | b (uint32)` | uint32 | 100,000 | 0.04716 ms | not_measured | managed | 0.622x |
| `a | b (uint32)` | uint32 | 10,000,000 | 11.3730 ms | not_measured | managed | 1.453x |
| `a | b (uint64)` | uint64 | 1,000 | 0.00183 ms | not_measured | managed | 0.358x |
| `a | b (uint64)` | uint64 | 100,000 | 0.09599 ms | not_measured | managed | 0.367x |
| `a | b (uint64)` | uint64 | 10,000,000 | 25.3417 ms | not_measured | managed | 1.337x |
| `a | b (uint8)` | uint8 | 1,000 | 0.000832 ms | not_measured | managed | 0.750x |
| `a | b (uint8)` | uint8 | 100,000 | 0.01329 ms | not_measured | managed | 2.235x |
| `a | b (uint8)` | uint8 | 10,000,000 | 2.1864 ms | not_measured | managed | 2.794x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.00110 ms | not_measured | managed | 0.314x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.04950 ms | not_measured | managed | 0.052x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 5.2471 ms | not_measured | managed | 0.621x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.00103 ms | not_measured | managed | 0.636x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.02508 ms | not_measured | managed | 1.172x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 5.5851 ms | not_measured | managed | 1.611x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.00134 ms | not_measured | managed | 0.511x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.04878 ms | not_measured | managed | 0.605x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 10.8376 ms | not_measured | managed | 1.528x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.00173 ms | not_measured | managed | 0.391x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.09689 ms | not_measured | managed | 0.361x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 25.2386 ms | not_measured | managed | 1.240x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.000842 ms | not_measured | managed | 0.751x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.01295 ms | not_measured | managed | 2.218x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 2.1753 ms | not_measured | managed | 2.816x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.000971 ms | not_measured | managed | 0.671x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.02538 ms | not_measured | managed | 1.130x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 5.8052 ms | not_measured | managed | 1.526x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.00125 ms | not_measured | managed | 0.572x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.04857 ms | not_measured | managed | 0.593x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 11.0317 ms | not_measured | managed | 1.361x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.00200 ms | not_measured | managed | 0.333x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.09395 ms | not_measured | managed | 0.371x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 25.0692 ms | not_measured | managed | 1.251x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.000798 ms | not_measured | managed | 0.807x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.01288 ms | not_measured | managed | 2.249x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 2.2199 ms | not_measured | managed | 2.763x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.00113 ms | not_measured | managed | 0.283x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.04822 ms | not_measured | managed | 0.041x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 5.1428 ms | not_measured | managed | 0.479x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.000985 ms | not_measured | managed | 0.607x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.02510 ms | not_measured | managed | 1.069x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 3.8744 ms | not_measured | managed | 1.751x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.00125 ms | not_measured | managed | 0.492x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.04706 ms | not_measured | managed | 0.607x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 8.2594 ms | not_measured | managed | 1.590x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.00178 ms | not_measured | managed | 0.343x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.09411 ms | not_measured | managed | 0.278x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 19.0318 ms | not_measured | managed | 1.306x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.000921 ms | not_measured | managed | 0.628x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.01336 ms | not_measured | managed | 2.085x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 1.7668 ms | not_measured | managed | 2.935x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.000920 ms | not_measured | managed | 0.650x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.02552 ms | not_measured | managed | 1.009x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 3.7905 ms | not_measured | managed | 1.818x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.00122 ms | not_measured | managed | 0.502x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.04809 ms | not_measured | managed | 0.807x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 8.2112 ms | not_measured | managed | 1.576x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.00202 ms | not_measured | managed | 0.305x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.09211 ms | not_measured | managed | 0.334x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 18.8655 ms | not_measured | managed | 1.287x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.000780 ms | not_measured | managed | 0.747x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.01324 ms | not_measured | managed | 1.964x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 1.7732 ms | not_measured | managed | 2.965x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.00111 ms | not_measured | managed | 0.319x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.04555 ms | not_measured | managed | 0.056x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 5.1878 ms | not_measured | managed | 0.604x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.00106 ms | not_measured | managed | 0.614x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.02559 ms | not_measured | managed | 1.135x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 5.4457 ms | not_measured | managed | 1.686x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.00131 ms | not_measured | managed | 0.501x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.04756 ms | not_measured | managed | 0.611x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 11.1365 ms | not_measured | managed | 1.488x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.00166 ms | not_measured | managed | 0.410x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.09646 ms | not_measured | managed | 0.364x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 25.2146 ms | not_measured | managed | 1.292x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.000850 ms | not_measured | managed | 0.738x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.01225 ms | not_measured | managed | 2.447x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 2.1894 ms | not_measured | managed | 2.658x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.00116 ms | not_measured | managed | 0.583x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.02497 ms | not_measured | managed | 1.154x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 5.6966 ms | not_measured | managed | 1.626x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.00139 ms | not_measured | managed | 0.476x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.04803 ms | not_measured | managed | 0.606x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 10.9581 ms | not_measured | managed | 1.508x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.00178 ms | not_measured | managed | 0.375x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.09432 ms | not_measured | managed | 0.365x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 24.9380 ms | not_measured | managed | 1.242x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.000949 ms | not_measured | managed | 0.664x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.01352 ms | not_measured | managed | 2.196x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 2.1440 ms | not_measured | managed | 2.835x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.00123 ms | not_measured | managed | 0.280x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.04742 ms | not_measured | managed | 0.058x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 5.2546 ms | not_measured | managed | 0.617x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.000965 ms | not_measured | managed | 0.679x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.02561 ms | not_measured | managed | 1.155x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 5.7109 ms | not_measured | managed | 1.618x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.00132 ms | not_measured | managed | 0.497x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.04798 ms | not_measured | managed | 0.628x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 11.5850 ms | not_measured | managed | 1.457x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.00188 ms | not_measured | managed | 0.352x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.09574 ms | not_measured | managed | 0.379x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 24.3263 ms | not_measured | managed | 1.382x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.00101 ms | not_measured | managed | 0.617x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.01645 ms | not_measured | managed | 1.875x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 2.1579 ms | not_measured | managed | 2.844x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.00128 ms | not_measured | managed | 0.517x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.02515 ms | not_measured | managed | 1.166x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 5.7281 ms | not_measured | managed | 1.573x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.00154 ms | not_measured | managed | 0.429x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.04810 ms | not_measured | managed | 0.657x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 10.6020 ms | not_measured | managed | 1.575x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.00167 ms | not_measured | managed | 0.398x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.09370 ms | not_measured | managed | 0.377x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 25.2029 ms | not_measured | managed | 1.350x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.000825 ms | not_measured | managed | 0.759x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.01353 ms | not_measured | managed | 2.115x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 2.1680 ms | not_measured | managed | 2.818x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.00138 ms | not_measured | managed | 0.234x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.05047 ms | not_measured | managed | 0.039x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 5.1415 ms | not_measured | managed | 0.339x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.00114 ms | not_measured | managed | 0.525x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.02582 ms | not_measured | managed | 1.063x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 3.8852 ms | not_measured | managed | 1.690x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.00130 ms | not_measured | managed | 0.464x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.04669 ms | not_measured | managed | 0.624x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 8.1457 ms | not_measured | managed | 1.607x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.00181 ms | not_measured | managed | 0.342x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.09546 ms | not_measured | managed | 0.278x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 19.0362 ms | not_measured | managed | 1.324x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.000995 ms | not_measured | managed | 0.588x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.01335 ms | not_measured | managed | 1.930x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 1.7544 ms | not_measured | managed | 2.961x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.000957 ms | not_measured | managed | 0.627x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.02395 ms | not_measured | managed | 1.153x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 4.0226 ms | not_measured | managed | 1.680x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.00131 ms | not_measured | managed | 0.460x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.04790 ms | not_measured | managed | 0.552x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 8.2454 ms | not_measured | managed | 1.564x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.00187 ms | not_measured | managed | 0.333x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.09236 ms | not_measured | managed | 0.284x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 19.0412 ms | not_measured | managed | 1.263x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.000868 ms | not_measured | managed | 0.673x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.01346 ms | not_measured | managed | 1.941x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 1.7455 ms | not_measured | managed | 2.979x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.00520 ms | not_measured | managed | 0.267x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.1087 ms | not_measured | managed | 0.401x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 14.6392 ms | not_measured | managed | 1.066x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.00177 ms | not_measured | managed | 0.523x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.02621 ms | not_measured | managed | 1.082x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 3.8503 ms | not_measured | managed | 1.828x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.00197 ms | not_measured | managed | 0.423x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.04830 ms | not_measured | managed | 0.409x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.2444 ms | not_measured | managed | 1.593x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.00281 ms | not_measured | managed | 0.278x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.09576 ms | not_measured | managed | 0.201x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 19.0042 ms | not_measured | managed | 1.334x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.00173 ms | not_measured | managed | 0.502x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.01424 ms | not_measured | managed | 1.999x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.7639 ms | not_measured | managed | 3.156x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.00178 ms | not_measured | managed | 0.504x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.02636 ms | not_measured | managed | 1.119x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 3.9105 ms | not_measured | managed | 1.851x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00239 ms | not_measured | managed | 0.353x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.04645 ms | not_measured | managed | 0.420x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.1719 ms | not_measured | managed | 1.587x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00299 ms | not_measured | managed | 0.281x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.09014 ms | not_measured | managed | 0.214x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 19.0178 ms | not_measured | managed | 1.261x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.00175 ms | not_measured | managed | 0.495x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.01459 ms | not_measured | managed | 1.958x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.7746 ms | not_measured | managed | 3.102x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.00532 ms | not_measured | managed | 0.267x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.1034 ms | not_measured | managed | 0.542x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | 14.6144 ms | not_measured | managed | 1.099x |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.00161 ms | not_measured | managed | 0.636x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.02644 ms | not_measured | managed | 1.605x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 3.8585 ms | not_measured | managed | 2.234x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.00195 ms | not_measured | managed | 0.470x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.04880 ms | not_measured | managed | 0.588x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.2896 ms | not_measured | managed | 1.626x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.00252 ms | not_measured | managed | 0.349x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.09538 ms | not_measured | managed | 0.301x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 20.2438 ms | not_measured | managed | 1.220x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.00166 ms | not_measured | managed | 0.583x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.01339 ms | not_measured | managed | 3.188x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 1.7880 ms | not_measured | managed | 4.681x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.00183 ms | not_measured | managed | 0.496x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.02607 ms | not_measured | managed | 1.085x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 3.8980 ms | not_measured | managed | 1.820x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.00211 ms | not_measured | managed | 0.398x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.04715 ms | not_measured | managed | 0.410x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.2173 ms | not_measured | managed | 1.574x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.00260 ms | not_measured | managed | 0.316x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.08943 ms | not_measured | managed | 0.217x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 18.9018 ms | not_measured | managed | 1.285x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.00182 ms | not_measured | managed | 0.476x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.01435 ms | not_measured | managed | 1.989x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.7691 ms | not_measured | managed | 3.300x |
| `matrix * 2.0` | float64 | 1,000 | 0.00288 ms | not_measured | managed | 0.215x |
| `matrix * 2.0` | float64 | 100,000 | 0.1030 ms | not_measured | managed | 0.115x |
| `matrix * 2.0` | float64 | 10,000,000 | 19.8404 ms | not_measured | managed | 0.812x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.00573 ms | not_measured | managed | 0.187x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.1165 ms | not_measured | managed | 0.226x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 19.6055 ms | not_measured | managed | 1.308x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.00425 ms | not_measured | managed | 0.244x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.1064 ms | not_measured | managed | 0.303x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 20.3728 ms | not_measured | managed | 1.299x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.00631 ms | not_measured | managed | 0.167x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.1548 ms | not_measured | managed | 0.172x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 19.8538 ms | not_measured | managed | 1.285x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.00414 ms | not_measured | managed | 0.264x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.1146 ms | not_measured | managed | 0.221x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 20.2881 ms | not_measured | managed | 0.962x |
| `matrix + scalar` | float64 | 1,000 | 0.00203 ms | not_measured | managed | 0.242x |
| `matrix + scalar` | float64 | 100,000 | 0.1110 ms | not_measured | managed | 0.105x |
| `matrix + scalar` | float64 | 10,000,000 | 19.7746 ms | not_measured | managed | 0.817x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.000375 ms | not_measured | managed | 4.715x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.000378 ms | not_measured | managed | 4.812x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.000378 ms | not_measured | managed | 11.447x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.000383 ms | not_measured | managed | 4.676x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.000530 ms | not_measured | managed | 3.579x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.000533 ms | not_measured | managed | 8.068x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.00378 ms | not_measured | managed | 0.271x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.1332 ms | not_measured | managed | 0.196x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 21.2959 ms | not_measured | managed | 1.086x |
| `a != b (float32)` | float32 | 1,000 | 0.000660 ms | not_measured | managed | 0.556x |
| `a != b (float32)` | float32 | 100,000 | 0.01570 ms | not_measured | managed | 0.350x |
| `a != b (float32)` | float32 | 10,000,000 | 8.6313 ms | not_measured | managed | 1.183x |
| `a != b (float64)` | float64 | 1,000 | 0.00100 ms | not_measured | managed | 0.395x |
| `a != b (float64)` | float64 | 100,000 | 0.02273 ms | not_measured | managed | 0.515x |
| `a != b (float64)` | float64 | 10,000,000 | 16.4112 ms | not_measured | managed | 1.086x |
| `a != b (int32)` | int32 | 1,000 | 0.000944 ms | not_measured | managed | 0.403x |
| `a != b (int32)` | int32 | 100,000 | 0.01858 ms | not_measured | managed | 0.369x |
| `a != b (int32)` | int32 | 10,000,000 | 8.9010 ms | not_measured | managed | 0.498x |
| `a != b (int64)` | int64 | 1,000 | 0.00102 ms | not_measured | managed | 0.412x |
| `a != b (int64)` | int64 | 100,000 | 0.02366 ms | not_measured | managed | 0.569x |
| `a != b (int64)` | int64 | 10,000,000 | 16.2545 ms | not_measured | managed | 1.100x |
| `a < b (float32)` | float32 | 1,000 | 0.000895 ms | not_measured | managed | 0.419x |
| `a < b (float32)` | float32 | 100,000 | 0.01493 ms | not_measured | managed | 0.396x |
| `a < b (float32)` | float32 | 10,000,000 | 8.8925 ms | not_measured | managed | 1.116x |
| `a < b (float64)` | float64 | 1,000 | 0.000893 ms | not_measured | managed | 0.438x |
| `a < b (float64)` | float64 | 100,000 | 0.02309 ms | not_measured | managed | 0.486x |
| `a < b (float64)` | float64 | 10,000,000 | 16.2414 ms | not_measured | managed | 1.112x |
| `a < b (int32)` | int32 | 1,000 | 0.000866 ms | not_measured | managed | 0.436x |
| `a < b (int32)` | int32 | 100,000 | 0.01864 ms | not_measured | managed | 0.375x |
| `a < b (int32)` | int32 | 10,000,000 | 8.2419 ms | not_measured | managed | 0.508x |
| `a < b (int64)` | int64 | 1,000 | 0.000903 ms | not_measured | managed | 0.533x |
| `a < b (int64)` | int64 | 100,000 | 0.02324 ms | not_measured | managed | 0.765x |
| `a < b (int64)` | int64 | 10,000,000 | 16.5451 ms | not_measured | managed | 1.105x |
| `a <= b (float32)` | float32 | 1,000 | 0.000855 ms | not_measured | managed | 0.434x |
| `a <= b (float32)` | float32 | 100,000 | 0.01560 ms | not_measured | managed | 0.351x |
| `a <= b (float32)` | float32 | 10,000,000 | 8.4873 ms | not_measured | managed | 1.176x |
| `a <= b (float64)` | float64 | 1,000 | 0.00103 ms | not_measured | managed | 0.378x |
| `a <= b (float64)` | float64 | 100,000 | 0.02212 ms | not_measured | managed | 0.531x |
| `a <= b (float64)` | float64 | 10,000,000 | 15.7637 ms | not_measured | managed | 1.134x |
| `a <= b (int32)` | int32 | 1,000 | 0.00102 ms | not_measured | managed | 0.377x |
| `a <= b (int32)` | int32 | 100,000 | 0.01744 ms | not_measured | managed | 0.394x |
| `a <= b (int32)` | int32 | 10,000,000 | 8.5089 ms | not_measured | managed | 0.502x |
| `a <= b (int64)` | int64 | 1,000 | 0.00100 ms | not_measured | managed | 0.495x |
| `a <= b (int64)` | int64 | 100,000 | 0.02579 ms | not_measured | managed | 0.698x |
| `a <= b (int64)` | int64 | 10,000,000 | 15.9458 ms | not_measured | managed | 1.118x |
| `a == b (float32)` | float32 | 1,000 | 0.000838 ms | not_measured | managed | 0.438x |
| `a == b (float32)` | float32 | 100,000 | 0.01565 ms | not_measured | managed | 0.351x |
| `a == b (float32)` | float32 | 10,000,000 | 8.7487 ms | not_measured | managed | 1.154x |
| `a == b (float64)` | float64 | 1,000 | 0.000942 ms | not_measured | managed | 0.415x |
| `a == b (float64)` | float64 | 100,000 | 0.02322 ms | not_measured | managed | 0.489x |
| `a == b (float64)` | float64 | 10,000,000 | 16.4127 ms | not_measured | managed | 1.087x |
| `a == b (int32)` | int32 | 1,000 | 0.000474 ms | not_measured | managed | 0.787x |
| `a == b (int32)` | int32 | 100,000 | 0.02081 ms | not_measured | managed | 0.335x |
| `a == b (int32)` | int32 | 10,000,000 | 8.9461 ms | not_measured | managed | 0.481x |
| `a == b (int64)` | int64 | 1,000 | 0.00105 ms | not_measured | managed | 0.425x |
| `a == b (int64)` | int64 | 100,000 | 0.02377 ms | not_measured | managed | 0.567x |
| `a == b (int64)` | int64 | 10,000,000 | 16.4816 ms | not_measured | managed | 1.089x |
| `a > b (float32)` | float32 | 1,000 | 0.000812 ms | not_measured | managed | 0.468x |
| `a > b (float32)` | float32 | 100,000 | 0.01553 ms | not_measured | managed | 0.351x |
| `a > b (float32)` | float32 | 10,000,000 | 8.5798 ms | not_measured | managed | 1.174x |
| `a > b (float64)` | float64 | 1,000 | 0.00108 ms | not_measured | managed | 0.374x |
| `a > b (float64)` | float64 | 100,000 | 0.02251 ms | not_measured | managed | 0.499x |
| `a > b (float64)` | float64 | 10,000,000 | 16.4828 ms | not_measured | managed | 1.081x |
| `a > b (int32)` | int32 | 1,000 | 0.000935 ms | not_measured | managed | 0.402x |
| `a > b (int32)` | int32 | 100,000 | 0.01657 ms | not_measured | managed | 0.420x |
| `a > b (int32)` | int32 | 10,000,000 | 8.9463 ms | not_measured | managed | 0.468x |
| `a > b (int64)` | int64 | 1,000 | 0.000956 ms | not_measured | managed | 0.505x |
| `a > b (int64)` | int64 | 100,000 | 0.02304 ms | not_measured | managed | 0.813x |
| `a > b (int64)` | int64 | 10,000,000 | 16.1156 ms | not_measured | managed | 1.138x |
| `a >= b (float32)` | float32 | 1,000 | 0.000944 ms | not_measured | managed | 0.396x |
| `a >= b (float32)` | float32 | 100,000 | 0.01562 ms | not_measured | managed | 0.349x |
| `a >= b (float32)` | float32 | 10,000,000 | 8.8409 ms | not_measured | managed | 1.119x |
| `a >= b (float64)` | float64 | 1,000 | 0.000854 ms | not_measured | managed | 0.460x |
| `a >= b (float64)` | float64 | 100,000 | 0.02274 ms | not_measured | managed | 0.497x |
| `a >= b (float64)` | float64 | 10,000,000 | 16.3819 ms | not_measured | managed | 1.085x |
| `a >= b (int32)` | int32 | 1,000 | 0.000965 ms | not_measured | managed | 0.415x |
| `a >= b (int32)` | int32 | 100,000 | 0.01662 ms | not_measured | managed | 0.416x |
| `a >= b (int32)` | int32 | 10,000,000 | 8.9672 ms | not_measured | managed | 0.473x |
| `a >= b (int64)` | int64 | 1,000 | 0.000805 ms | not_measured | managed | 0.624x |
| `a >= b (int64)` | int64 | 100,000 | 0.02615 ms | not_measured | managed | 0.691x |
| `a >= b (int64)` | int64 | 10,000,000 | 15.8513 ms | not_measured | managed | 1.158x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.000720 ms | not_measured | managed | 0.550x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.01574 ms | not_measured | managed | 0.361x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 8.8814 ms | not_measured | managed | 1.094x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.000867 ms | not_measured | managed | 0.468x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.02261 ms | not_measured | managed | 0.498x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 16.2748 ms | not_measured | managed | 1.104x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.00102 ms | not_measured | managed | 0.379x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.01786 ms | not_measured | managed | 0.417x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 8.7441 ms | not_measured | managed | 0.474x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.000772 ms | not_measured | managed | 0.544x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.02312 ms | not_measured | managed | 0.601x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 15.8333 ms | not_measured | managed | 1.159x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.00102 ms | not_measured | managed | 0.365x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.01503 ms | not_measured | managed | 0.380x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 8.5321 ms | not_measured | managed | 1.185x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.000819 ms | not_measured | managed | 0.496x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.02327 ms | not_measured | managed | 0.511x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 16.2060 ms | not_measured | managed | 1.115x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.000783 ms | not_measured | managed | 0.497x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.01470 ms | not_measured | managed | 0.504x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 8.7779 ms | not_measured | managed | 0.571x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.000750 ms | not_measured | managed | 0.667x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.02288 ms | not_measured | managed | 0.787x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 16.0348 ms | not_measured | managed | 1.157x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.000937 ms | not_measured | managed | 0.415x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.01507 ms | not_measured | managed | 0.366x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 8.8166 ms | not_measured | managed | 1.134x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.000935 ms | not_measured | managed | 0.436x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.02328 ms | not_measured | managed | 0.481x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 16.2122 ms | not_measured | managed | 1.118x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.00104 ms | not_measured | managed | 0.382x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.01655 ms | not_measured | managed | 0.440x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 8.9866 ms | not_measured | managed | 1.162x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.000845 ms | not_measured | managed | 0.607x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.02616 ms | not_measured | managed | 0.704x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 15.9750 ms | not_measured | managed | 1.156x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.000753 ms | not_measured | managed | 0.498x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.01553 ms | not_measured | managed | 0.352x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 8.7974 ms | not_measured | managed | 1.144x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.00105 ms | not_measured | managed | 0.388x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.02352 ms | not_measured | managed | 0.527x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 16.1644 ms | not_measured | managed | 1.108x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.000847 ms | not_measured | managed | 0.456x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.01682 ms | not_measured | managed | 0.413x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 8.9810 ms | not_measured | managed | 0.462x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.000558 ms | not_measured | managed | 0.903x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.02297 ms | not_measured | managed | 0.795x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 15.2017 ms | not_measured | managed | 1.195x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.00104 ms | not_measured | managed | 0.380x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.01582 ms | not_measured | managed | 0.357x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 8.0090 ms | not_measured | managed | 1.254x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.00106 ms | not_measured | managed | 0.383x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.02282 ms | not_measured | managed | 0.581x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 16.2148 ms | not_measured | managed | 1.107x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.000800 ms | not_measured | managed | 0.500x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.01640 ms | not_measured | managed | 0.458x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 9.0195 ms | not_measured | managed | 1.158x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.000896 ms | not_measured | managed | 0.594x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.02642 ms | not_measured | managed | 0.675x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 16.1004 ms | not_measured | managed | 1.146x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.000587 ms | not_measured | managed | 0.654x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.01602 ms | not_measured | managed | 0.352x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 8.7809 ms | not_measured | managed | 1.148x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.000926 ms | not_measured | managed | 0.448x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.02088 ms | not_measured | managed | 0.593x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 16.2174 ms | not_measured | managed | 1.092x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.00101 ms | not_measured | managed | 0.410x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.01523 ms | not_measured | managed | 0.455x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 8.5931 ms | not_measured | managed | 0.507x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.000880 ms | not_measured | managed | 0.502x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.02370 ms | not_measured | managed | 0.607x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 16.2306 ms | not_measured | managed | 1.135x |
| `a.copy() (float32)` | float32 | 1,000 | 0.00148 ms | not_measured | managed | 0.172x |
| `a.copy() (float32)` | float32 | 100,000 | 0.01112 ms | not_measured | managed | 0.523x |
| `a.copy() (float32)` | float32 | 10,000,000 | 3.5488 ms | not_measured | managed | 2.718x |
| `a.copy() (float64)` | float64 | 1,000 | 0.00172 ms | not_measured | managed | 0.176x |
| `a.copy() (float64)` | float64 | 100,000 | 0.02568 ms | not_measured | managed | 0.444x |
| `a.copy() (float64)` | float64 | 10,000,000 | 16.5399 ms | not_measured | managed | 1.073x |
| `a.copy() (int32)` | int32 | 1,000 | 0.00156 ms | not_measured | managed | 0.168x |
| `a.copy() (int32)` | int32 | 100,000 | 0.01152 ms | not_measured | managed | 0.503x |
| `a.copy() (int32)` | int32 | 10,000,000 | 3.5467 ms | not_measured | managed | 1.797x |
| `a.copy() (int64)` | int64 | 1,000 | 0.00160 ms | not_measured | managed | 0.199x |
| `a.copy() (int64)` | int64 | 100,000 | 0.02538 ms | not_measured | managed | 0.454x |
| `a.copy() (int64)` | int64 | 10,000,000 | 16.6232 ms | not_measured | managed | 1.105x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.00151 ms | not_measured | managed | 0.483x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.05235 ms | not_measured | managed | 0.750x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 5.6084 ms | not_measured | managed | 1.855x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.00164 ms | not_measured | managed | 0.376x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.04911 ms | not_measured | managed | 0.656x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 16.3392 ms | not_measured | managed | 1.117x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.00162 ms | not_measured | managed | 0.375x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.05295 ms | not_measured | managed | 0.551x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 5.2013 ms | not_measured | managed | 1.665x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.00159 ms | not_measured | managed | 0.320x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.05105 ms | not_measured | managed | 0.373x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 16.4317 ms | not_measured | managed | 1.193x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.00153 ms | not_measured | managed | 0.208x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.1135 ms | not_measured | managed | 0.105x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.000006 ms | not_measured | managed | 10.000x |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.000006 ms | not_measured | managed | 9.500x |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 60.000x |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 59.000x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.000124 ms | not_measured | managed | 10.984x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.01185 ms | not_measured | managed | 1.033x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.00167 ms | not_measured | managed | 0.211x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.05924 ms | not_measured | managed | 0.169x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.00332 ms | not_measured | managed | 0.236x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.1880 ms | not_measured | managed | 0.137x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.000393 ms | not_measured | managed | 4.794x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.000421 ms | not_measured | managed | 4.511x |
| `np.copy (float32)` | float32 | 1,000 | 0.00161 ms | not_measured | managed | 0.288x |
| `np.copy (float32)` | float32 | 100,000 | 0.01136 ms | not_measured | managed | 0.526x |
| `np.copy (float32)` | float32 | 10,000,000 | 4.0374 ms | not_measured | managed | 2.298x |
| `np.copy (float64)` | float64 | 1,000 | 0.00154 ms | not_measured | managed | 0.328x |
| `np.copy (float64)` | float64 | 100,000 | 0.02509 ms | not_measured | managed | 0.465x |
| `np.copy (float64)` | float64 | 10,000,000 | 16.4932 ms | not_measured | managed | 1.069x |
| `np.copy (int32)` | int32 | 1,000 | 0.00169 ms | not_measured | managed | 0.277x |
| `np.copy (int32)` | int32 | 100,000 | 0.01152 ms | not_measured | managed | 0.519x |
| `np.copy (int32)` | int32 | 10,000,000 | 3.8330 ms | not_measured | managed | 1.719x |
| `np.copy (int64)` | int64 | 1,000 | 0.00158 ms | not_measured | managed | 0.327x |
| `np.copy (int64)` | int64 | 100,000 | 0.02596 ms | not_measured | managed | 0.481x |
| `np.copy (int64)` | int64 | 10,000,000 | 16.6128 ms | not_measured | managed | 1.040x |
| `np.empty (float32)` | float32 | 1,000 | 0.00338 ms | not_measured | managed | 0.059x |
| `np.empty (float32)` | float32 | 100,000 | 0.00301 ms | not_measured | managed | 0.091x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.00390 ms | not_measured | managed | 5.767x |
| `np.empty (float64)` | float64 | 1,000 | 0.00347 ms | not_measured | managed | 0.057x |
| `np.empty (float64)` | float64 | 100,000 | 0.00334 ms | not_measured | managed | 0.088x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.01925 ms | not_measured | managed | 0.869x |
| `np.empty (int32)` | int32 | 1,000 | 0.00252 ms | not_measured | managed | 0.080x |
| `np.empty (int32)` | int32 | 100,000 | 0.00418 ms | not_measured | managed | 0.065x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.00353 ms | not_measured | managed | 9.279x |
| `np.empty (int64)` | int64 | 1,000 | 0.000523 ms | not_measured | managed | 0.379x |
| `np.empty (int64)` | int64 | 100,000 | 0.00508 ms | not_measured | managed | 0.054x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.01741 ms | not_measured | managed | 1.354x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.00339 ms | not_measured | managed | 0.071x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.00330 ms | not_measured | managed | 0.097x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.00428 ms | not_measured | managed | 4.982x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.00499 ms | not_measured | managed | 0.048x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.00467 ms | not_measured | managed | 0.064x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.01913 ms | not_measured | managed | 0.897x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.00417 ms | not_measured | managed | 0.058x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.00365 ms | not_measured | managed | 0.085x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.000296 ms | not_measured | managed | 71.206x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.00386 ms | not_measured | managed | 0.065x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.00364 ms | not_measured | managed | 0.085x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.01890 ms | not_measured | managed | 1.126x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.00862 ms | not_measured | managed | 0.127x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.07689 ms | not_measured | managed | 0.150x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.000538 ms | not_measured | managed | 0.535x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.000584 ms | not_measured | managed | 0.503x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.06301 ms | not_measured | managed | 1.488x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 6.5296 ms | not_measured | managed | 1.406x |
| `np.full (float32)` | float32 | 1,000 | 0.00432 ms | not_measured | managed | 0.180x |
| `np.full (float32)` | float32 | 100,000 | 0.01768 ms | not_measured | managed | 0.301x |
| `np.full (float32)` | float32 | 10,000,000 | 3.6275 ms | not_measured | managed | 2.630x |
| `np.full (float64)` | float64 | 1,000 | 0.00368 ms | not_measured | managed | 0.214x |
| `np.full (float64)` | float64 | 100,000 | 0.01850 ms | not_measured | managed | 0.541x |
| `np.full (float64)` | float64 | 10,000,000 | 16.4285 ms | not_measured | managed | 1.093x |
| `np.full (int32)` | int32 | 1,000 | 0.00304 ms | not_measured | managed | 0.251x |
| `np.full (int32)` | int32 | 100,000 | 0.01703 ms | not_measured | managed | 0.309x |
| `np.full (int32)` | int32 | 10,000,000 | 2.9381 ms | not_measured | managed | 2.560x |
| `np.full (int64)` | int64 | 1,000 | 0.00373 ms | not_measured | managed | 0.203x |
| `np.full (int64)` | int64 | 100,000 | 0.01917 ms | not_measured | managed | 0.516x |
| `np.full (int64)` | int64 | 10,000,000 | 16.3991 ms | not_measured | managed | 1.081x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.00105 ms | not_measured | managed | 0.917x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.01818 ms | not_measured | managed | 0.306x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 3.5177 ms | not_measured | managed | 2.663x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.00528 ms | not_measured | managed | 0.185x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.01949 ms | not_measured | managed | 0.524x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 16.5329 ms | not_measured | managed | 1.131x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.00512 ms | not_measured | managed | 0.185x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.01794 ms | not_measured | managed | 0.302x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 3.1787 ms | not_measured | managed | 2.886x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.00515 ms | not_measured | managed | 0.182x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.01962 ms | not_measured | managed | 0.512x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 16.5832 ms | not_measured | managed | 1.172x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.00951 ms | not_measured | managed | 0.139x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.07809 ms | not_measured | managed | 0.151x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.00213 ms | not_measured | managed | 2.136x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.1756 ms | not_measured | managed | 0.423x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 18.3148 ms | not_measured | managed | 3.246x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.00245 ms | not_measured | managed | 1.636x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.1753 ms | not_measured | managed | 0.280x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 25.6216 ms | not_measured | managed | 1.426x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.00352 ms | not_measured | managed | 1.457x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.2719 ms | not_measured | managed | 1.158x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 31.0108 ms | not_measured | managed | 2.080x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.00307 ms | not_measured | managed | 1.640x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.2856 ms | not_measured | managed | 1.247x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 37.6537 ms | not_measured | managed | 2.026x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.01662 ms | not_measured | managed | 0.487x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.1830 ms | not_measured | managed | 0.176x |
| `np.ones (float32)` | float32 | 1,000 | 0.00396 ms | not_measured | managed | 0.198x |
| `np.ones (float32)` | float32 | 100,000 | 0.01716 ms | not_measured | managed | 0.317x |
| `np.ones (float32)` | float32 | 10,000,000 | 3.1745 ms | not_measured | managed | 3.172x |
| `np.ones (float64)` | float64 | 1,000 | 0.00368 ms | not_measured | managed | 0.213x |
| `np.ones (float64)` | float64 | 100,000 | 0.01928 ms | not_measured | managed | 0.513x |
| `np.ones (float64)` | float64 | 10,000,000 | 16.4869 ms | not_measured | managed | 1.087x |
| `np.ones (int32)` | int32 | 1,000 | 0.00397 ms | not_measured | managed | 0.191x |
| `np.ones (int32)` | int32 | 100,000 | 0.01803 ms | not_measured | managed | 0.292x |
| `np.ones (int32)` | int32 | 10,000,000 | 2.8912 ms | not_measured | managed | 2.631x |
| `np.ones (int64)` | int64 | 1,000 | 0.00450 ms | not_measured | managed | 0.177x |
| `np.ones (int64)` | int64 | 100,000 | 0.01962 ms | not_measured | managed | 0.495x |
| `np.ones (int64)` | int64 | 10,000,000 | 16.5337 ms | not_measured | managed | 1.072x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.00454 ms | not_measured | managed | 0.210x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.01797 ms | not_measured | managed | 0.302x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 3.5638 ms | not_measured | managed | 2.715x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.00260 ms | not_measured | managed | 0.390x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.01944 ms | not_measured | managed | 0.513x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 16.5802 ms | not_measured | managed | 1.093x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.00401 ms | not_measured | managed | 0.232x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.01852 ms | not_measured | managed | 0.302x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 3.0074 ms | not_measured | managed | 3.187x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.00392 ms | not_measured | managed | 0.230x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.01938 ms | not_measured | managed | 0.511x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 16.6493 ms | not_measured | managed | 1.173x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.00451 ms | not_measured | managed | 0.163x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.05749 ms | not_measured | managed | 0.198x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.02608 ms | not_measured | managed | 0.133x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.2258 ms | not_measured | managed | 0.761x |
| `np.zeros (float32)` | float32 | 1,000 | 0.00330 ms | not_measured | managed | 0.081x |
| `np.zeros (float32)` | float32 | 100,000 | 0.00282 ms | not_measured | managed | 1.710x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.01055 ms | not_measured | managed | 1.962x |
| `np.zeros (float64)` | float64 | 1,000 | 0.00365 ms | not_measured | managed | 0.076x |
| `np.zeros (float64)` | float64 | 100,000 | 0.00366 ms | not_measured | managed | 2.641x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.00989 ms | not_measured | managed | 1.659x |
| `np.zeros (int32)` | int32 | 1,000 | 0.00296 ms | not_measured | managed | 0.095x |
| `np.zeros (int32)` | int32 | 100,000 | 0.00304 ms | not_measured | managed | 1.590x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.00891 ms | not_measured | managed | 1.275x |
| `np.zeros (int64)` | int64 | 1,000 | 0.00342 ms | not_measured | managed | 0.097x |
| `np.zeros (int64)` | int64 | 100,000 | 0.00428 ms | not_measured | managed | 2.248x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.00795 ms | not_measured | managed | 2.724x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.00291 ms | not_measured | managed | 0.321x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.00343 ms | not_measured | managed | 1.611x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.01087 ms | not_measured | managed | 895.191x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.00381 ms | not_measured | managed | 0.239x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.00436 ms | not_measured | managed | 2.260x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.01046 ms | not_measured | managed | 1699.862x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.00317 ms | not_measured | managed | 0.281x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.00351 ms | not_measured | managed | 1.543x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.00962 ms | not_measured | managed | 965.927x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.00295 ms | not_measured | managed | 0.302x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.00433 ms | not_measured | managed | 2.346x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.00811 ms | not_measured | managed | 2366.374x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.01352 ms | not_measured | managed | 0.682x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 2.0536 ms | not_measured | managed | 0.764x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.06950 ms | not_measured | managed | 0.550x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 10.8368 ms | not_measured | managed | 0.414x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.01462 ms | not_measured | managed | 0.196x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.4398 ms | not_measured | managed | 1.272x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.06600 ms | not_measured | managed | 0.544x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 10.9019 ms | not_measured | managed | 0.435x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.01038 ms | not_measured | managed | 0.553x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.4285 ms | not_measured | managed | 0.779x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.01098 ms | not_measured | managed | 0.642x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 0.9631 ms | not_measured | managed | 1.063x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.01159 ms | not_measured | managed | 0.807x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 2.2403 ms | not_measured | managed | 0.664x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.06830 ms | not_measured | managed | 0.610x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 10.5770 ms | not_measured | managed | 0.428x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.06789 ms | not_measured | managed | 0.549x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 10.8537 ms | not_measured | managed | 0.429x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.00968 ms | not_measured | managed | 0.448x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.4272 ms | not_measured | managed | 0.802x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.01042 ms | not_measured | managed | 0.767x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 0.9916 ms | not_measured | managed | 0.743x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.00885 ms | not_measured | managed | 0.851x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 1.0014 ms | not_measured | managed | 0.720x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.03562 ms | not_measured | managed | 0.721x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 5.3290 ms | not_measured | managed | 0.445x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.03646 ms | not_measured | managed | 0.911x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 5.2776 ms | not_measured | managed | 0.449x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.00828 ms | not_measured | managed | 0.785x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 0.8831 ms | not_measured | managed | 0.821x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.03551 ms | not_measured | managed | 0.774x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 5.1744 ms | not_measured | managed | 0.438x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.00668 ms | not_measured | managed | 0.226x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.1579 ms | not_measured | managed | 0.170x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.03544 ms | not_measured | managed | 0.691x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 5.1782 ms | not_measured | managed | 0.436x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.000235 ms | not_measured | managed | 2.213x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.000230 ms | not_measured | managed | 2.217x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.000217 ms | not_measured | managed | 2.465x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.000729 ms | not_measured | managed | 0.693x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.01948 ms | 0.00740 ms | openblas | 0.946x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 8.3906 ms | not_measured | managed | 0.112x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 128 | 0.2151 ms | 0.06050 ms | openblas | 3.993x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.02037 ms | not_measured | managed | 0.070x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 1.5449 ms | not_measured | managed | 0.013x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 154.1206 ms | not_measured | managed | 0.029x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.00294 ms | not_measured | managed | 3.089x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.00307 ms | not_measured | managed | 2.975x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.00303 ms | not_measured | managed | 2.955x |
| `np.linalg.cholesky(a)` | float64 | 32 | missing_backend | 0.00250 ms | openblas | 1.880x |
| `np.linalg.cholesky(a)` | float64 | 96 | missing_backend | 0.01600 ms | openblas | 1.144x |
| `np.linalg.cond(a)` | float64 | 32 | missing_backend | 0.03900 ms | openblas | 0.905x |
| `np.linalg.cond(a)` | float64 | 96 | missing_backend | 0.5162 ms | openblas | 0.540x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.03252 ms | not_measured | managed | 0.541x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 1.4298 ms | not_measured | managed | 0.452x |
| `np.linalg.det(a)` | float64 | 32 | missing_backend | 0.00520 ms | openblas | 1.173x |
| `np.linalg.det(a)` | float64 | 96 | missing_backend | 0.05780 ms | openblas | 0.471x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.000241 ms | not_measured | managed | 2.768x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.000218 ms | not_measured | managed | 3.037x |
| `np.linalg.eig(a)` | float64 | 32 | missing_backend | 0.1146 ms | openblas | 0.901x |
| `np.linalg.eig(a)` | float64 | 96 | missing_backend | 2.6840 ms | openblas | 0.508x |
| `np.linalg.eigh(a)` | float64 | 32 | missing_backend | 0.04620 ms | openblas | 1.054x |
| `np.linalg.eigh(a)` | float64 | 96 | missing_backend | 0.8220 ms | openblas | 0.485x |
| `np.linalg.eigvals(a)` | float64 | 32 | missing_backend | 0.06700 ms | openblas | 0.991x |
| `np.linalg.eigvals(a)` | float64 | 96 | missing_backend | 1.5275 ms | openblas | 0.507x |
| `np.linalg.eigvalsh(a)` | float64 | 32 | missing_backend | 0.02070 ms | openblas | 1.101x |
| `np.linalg.eigvalsh(a)` | float64 | 96 | missing_backend | 0.2832 ms | openblas | 0.565x |
| `np.linalg.inv(a)` | float64 | 32 | missing_backend | 0.00900 ms | openblas | 1.233x |
| `np.linalg.inv(a)` | float64 | 96 | missing_backend | 0.1619 ms | openblas | 0.486x |
| `np.linalg.lstsq(a, b)` | float64 | 32 | missing_backend | 0.08320 ms | openblas | 0.964x |
| `np.linalg.lstsq(a, b)` | float64 | 96 | missing_backend | 1.6242 ms | openblas | 0.488x |
| `np.linalg.matmul(a, b)` | float64 | 128 | 0.2128 ms | 0.05860 ms | openblas | 1.002x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.01141 ms | not_measured | managed | 0.224x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 0.3259 ms | not_measured | managed | 0.221x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.00897 ms | not_measured | managed | 0.258x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.05388 ms | not_measured | managed | 0.119x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 32 | missing_backend | 0.03280 ms | openblas | 1.125x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 96 | missing_backend | 0.4903 ms | openblas | 0.590x |
| `np.linalg.matrix_power(a, -1)` | float64 | 32 | missing_backend | 0.01060 ms | openblas | 1.085x |
| `np.linalg.matrix_power(a, -1)` | float64 | 96 | missing_backend | 0.1651 ms | openblas | 0.486x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.01888 ms | not_measured | managed | 0.298x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.7743 ms | not_measured | managed | 0.146x |
| `np.linalg.matrix_rank(a)` | float64 | 32 | missing_backend | 0.03650 ms | openblas | 1.014x |
| `np.linalg.matrix_rank(a)` | float64 | 96 | missing_backend | 0.4982 ms | openblas | 0.573x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.000308 ms | not_measured | managed | 2.146x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.000407 ms | not_measured | managed | 1.644x |
| `np.linalg.multi_dot(arrays)` | float64 | 128 | 0.3826 ms | 0.1175 ms | openblas | 1.031x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.01970 ms | not_measured | managed | 0.270x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.7734 ms | not_measured | managed | 0.203x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.00130 ms | not_measured | managed | 0.754x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.01307 ms | not_measured | managed | 6.285x |
| `np.linalg.norm(a, ord=2)` | float64 | 32 | missing_backend | 0.03320 ms | openblas | 1.108x |
| `np.linalg.norm(a, ord=2)` | float64 | 96 | missing_backend | 0.4903 ms | openblas | 0.577x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.00802 ms | not_measured | managed | 0.265x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.1464 ms | not_measured | managed | 0.266x |
| `np.linalg.pinv(a)` | float64 | 32 | missing_backend | 0.09560 ms | openblas | 0.911x |
| `np.linalg.pinv(a)` | float64 | 96 | missing_backend | 1.5979 ms | openblas | 0.462x |
| `np.linalg.qr(a)` | float64 | 32 | missing_backend | 0.02030 ms | openblas | 1.163x |
| `np.linalg.qr(a)` | float64 | 96 | missing_backend | 0.3164 ms | openblas | 0.508x |
| `np.linalg.slogdet(a)` | float64 | 32 | missing_backend | 0.00570 ms | openblas | 1.193x |
| `np.linalg.slogdet(a)` | float64 | 96 | missing_backend | 0.05640 ms | openblas | 0.500x |
| `np.linalg.solve(a, b)` | float64 | 32 | missing_backend | 0.00530 ms | openblas | 1.453x |
| `np.linalg.solve(a, b)` | float64 | 96 | missing_backend | 0.05920 ms | openblas | 0.505x |
| `np.linalg.svd(a)` | float64 | 32 | missing_backend | 0.07500 ms | openblas | 1.028x |
| `np.linalg.svd(a)` | float64 | 96 | missing_backend | 1.4331 ms | openblas | 0.483x |
| `np.linalg.svdvals(a)` | float64 | 32 | missing_backend | 0.02980 ms | openblas | 1.074x |
| `np.linalg.svdvals(a)` | float64 | 96 | missing_backend | 0.4894 ms | openblas | 0.564x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.01281 ms | not_measured | managed | 0.455x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.3919 ms | not_measured | managed | 0.222x |
| `np.linalg.tensordot(a, b, axes=1)` | float64 | 128 | 0.2166 ms | 0.06060 ms | openblas | 1.048x |
| `np.linalg.tensorinv(a)` | float64 | 32 | missing_backend | 0.00960 ms | openblas | 1.229x |
| `np.linalg.tensorinv(a)` | float64 | 96 | missing_backend | 0.1621 ms | openblas | 0.494x |
| `np.linalg.tensorsolve(a, b)` | float64 | 32 | missing_backend | 0.00670 ms | openblas | 1.284x |
| `np.linalg.tensorsolve(a, b)` | float64 | 96 | missing_backend | 0.06040 ms | openblas | 0.512x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.000558 ms | not_measured | managed | 2.591x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.000639 ms | not_measured | managed | 2.283x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00660 ms | not_measured | managed | 0.129x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1589 ms | 0.00730 ms | openblas | 1.014x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.00587 ms | not_measured | managed | 0.452x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.3327 ms | not_measured | managed | 0.092x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.01013 ms | not_measured | managed | 0.229x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 4.0444 ms | not_measured | managed | 0.160x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 6.9332 ms | not_measured | managed | 0.097x |
| `np.matmul(a, b)` | float64 | 128 | 0.2133 ms | 0.05870 ms | openblas | 0.997x |
| `np.matvec(a, b)` | float64 | 128 | 0.02270 ms | 0.00160 ms | openblas | 1.250x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.00323 ms | not_measured | managed | 0.220x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.1204 ms | not_measured | managed | 0.044x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.1758 ms | not_measured | managed | 0.044x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.00605 ms | not_measured | managed | 0.330x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.1389 ms | not_measured | managed | 0.285x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 15.8508 ms | not_measured | managed | 0.871x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.01990 ms | not_measured | managed | 0.187x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 1.5413 ms | not_measured | managed | 0.063x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 153.5055 ms | not_measured | managed | 0.008x |
| `np.tensordot(a, b, axes=1)` | float64 | 128 | 0.2149 ms | 0.06080 ms | openblas | 1.043x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.00112 ms | not_measured | managed | 0.504x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.01300 ms | 0.00810 ms | openblas | 0.877x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 8.3138 ms | not_measured | managed | 0.133x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.00475 ms | not_measured | managed | 0.123x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1583 ms | 0.00730 ms | openblas | 0.973x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 29.5241 ms | not_measured | managed | 0.038x |
| `np.vecmat(a, b)` | float64 | 128 | 0.00270 ms | 0.00170 ms | openblas | 0.882x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.00168 ms | not_measured | managed | 0.419x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.05370 ms | not_measured | managed | 0.095x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.06858 ms | not_measured | managed | 0.123x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.08314 ms | not_measured | managed | 0.361x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 0.9481 ms | not_measured | managed | 1.851x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.08856 ms | not_measured | managed | 0.145x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.5317 ms | not_measured | managed | 0.723x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.07841 ms | not_measured | managed | 0.171x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.4504 ms | not_measured | managed | 1.529x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.00148 ms | not_measured | managed | 1.657x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.1059 ms | not_measured | managed | 0.799x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 12.6667 ms | not_measured | managed | 1.606x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.00146 ms | not_measured | managed | 1.042x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.00977 ms | not_measured | managed | 0.683x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 8.6462 ms | not_measured | managed | 1.127x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.00193 ms | not_measured | managed | 0.854x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.01928 ms | not_measured | managed | 0.589x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 16.0459 ms | not_measured | managed | 1.071x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.00338 ms | not_measured | managed | 0.829x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.8299 ms | not_measured | managed | 0.925x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 87.5675 ms | not_measured | managed | 1.105x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.00127 ms | not_measured | managed | 0.331x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.05015 ms | not_measured | managed | 0.142x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 11.9229 ms | not_measured | managed | 1.291x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.00167 ms | not_measured | managed | 0.293x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.1022 ms | not_measured | managed | 0.266x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 26.7830 ms | not_measured | managed | 1.156x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.00344 ms | not_measured | managed | 0.853x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.8294 ms | not_measured | managed | 0.898x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 87.6222 ms | not_measured | managed | 1.078x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.00133 ms | not_measured | managed | 0.315x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.05006 ms | not_measured | managed | 0.143x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 11.9212 ms | not_measured | managed | 1.330x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.00177 ms | not_measured | managed | 0.280x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.09495 ms | not_measured | managed | 0.333x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 26.7452 ms | not_measured | managed | 1.160x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.09686 ms | not_measured | managed | 0.305x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 1.0012 ms | not_measured | managed | 1.756x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.06749 ms | not_measured | managed | 0.171x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.5255 ms | not_measured | managed | 0.717x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.06446 ms | not_measured | managed | 0.176x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.4708 ms | not_measured | managed | 1.368x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.00146 ms | not_measured | managed | 0.296x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.00274 ms | not_measured | managed | 0.626x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | 0.00691 ms | not_measured | managed | 2.638x |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.00163 ms | not_measured | managed | 0.275x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.00291 ms | not_measured | managed | 0.571x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | 0.00897 ms | not_measured | managed | 2.107x |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.000974 ms | not_measured | managed | 0.449x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.00276 ms | not_measured | managed | 0.641x |
| `np.iscomplex(a) (float64)` | float64 | 10,000,000 | 0.00874 ms | not_measured | managed | 1.952x |
| `np.iscomplexobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.00136 ms | not_measured | managed | 0.601x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.05162 ms | not_measured | managed | 0.917x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 4.9448 ms | not_measured | managed | 1.202x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.00153 ms | not_measured | managed | 0.243x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.04719 ms | not_measured | managed | 0.108x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 5.9683 ms | not_measured | managed | 1.029x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.00158 ms | not_measured | managed | 0.262x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.05140 ms | not_measured | managed | 0.193x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 9.2724 ms | not_measured | managed | 1.130x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.000001 ms | not_measured | managed | 99.000x |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.000001 ms | not_measured | managed | 149.000x |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.000001 ms | not_measured | managed | 99.000x |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.00128 ms | not_measured | managed | 0.709x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.04942 ms | not_measured | managed | 1.037x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 4.9645 ms | not_measured | managed | 1.225x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.00178 ms | not_measured | managed | 0.210x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.05080 ms | not_measured | managed | 0.104x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 6.4660 ms | not_measured | managed | 0.952x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.00178 ms | not_measured | managed | 0.243x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.05869 ms | not_measured | managed | 0.170x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 9.4804 ms | not_measured | managed | 1.153x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.00144 ms | not_measured | managed | 0.709x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.05231 ms | not_measured | managed | 1.300x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 5.3613 ms | not_measured | managed | 1.470x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.00157 ms | not_measured | managed | 0.230x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.05308 ms | not_measured | managed | 0.076x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 6.3274 ms | not_measured | managed | 0.890x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.00195 ms | not_measured | managed | 0.205x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.05217 ms | not_measured | managed | 0.164x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 9.3291 ms | not_measured | managed | 1.077x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.000878 ms | not_measured | managed | 2.159x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.00282 ms | not_measured | managed | 27.252x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 0.3168 ms | not_measured | managed | 64.626x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.00132 ms | not_measured | managed | 0.848x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.00281 ms | not_measured | managed | 3.021x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 0.3418 ms | not_measured | managed | 31.655x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.00195 ms | not_measured | managed | 0.609x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.00259 ms | not_measured | managed | 6.022x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 0.3204 ms | not_measured | managed | 61.412x |
| `np.isrealobj(a) (float16)` | float16 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 277.000x |
| `np.isrealobj(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 10,000,000 | 0.000000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 1,000 | 0.000001 ms | not_measured | managed | 478.000x |
| `np.isscalar(a) (float16)` | float16 | 100,000 | 0.000001 ms | not_measured | managed | 479.000x |
| `np.isscalar(a) (float16)` | float16 | 10,000,000 | 0.000001 ms | not_measured | managed | 600.000x |
| `np.isscalar(a) (float32)` | float32 | 1,000 | 0.000001 ms | not_measured | managed | 474.000x |
| `np.isscalar(a) (float32)` | float32 | 100,000 | 0.000001 ms | not_measured | managed | 520.000x |
| `np.isscalar(a) (float32)` | float32 | 10,000,000 | 0.000001 ms | not_measured | managed | 606.000x |
| `np.isscalar(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 474.000x |
| `np.isscalar(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 477.000x |
| `np.isscalar(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 605.000x |
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
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.00360 ms | not_measured | managed | 0.794x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.8809 ms | not_measured | managed | 0.864x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 89.7283 ms | not_measured | managed | 1.090x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.00140 ms | not_measured | managed | 0.316x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.04823 ms | not_measured | managed | 0.141x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 11.8266 ms | not_measured | managed | 1.303x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.00192 ms | not_measured | managed | 0.262x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.1006 ms | not_measured | managed | 0.286x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 27.7325 ms | not_measured | managed | 1.114x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.00368 ms | not_measured | managed | 0.876x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.8761 ms | not_measured | managed | 0.865x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 89.5472 ms | not_measured | managed | 1.076x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.00118 ms | not_measured | managed | 0.350x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.04833 ms | not_measured | managed | 0.153x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 11.8102 ms | not_measured | managed | 1.298x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.00176 ms | not_measured | managed | 0.276x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.09769 ms | not_measured | managed | 0.254x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 26.0524 ms | not_measured | managed | 1.176x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.000352 ms | not_measured | managed | 0.233x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.000354 ms | not_measured | managed | 0.229x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.000229 ms | not_measured | managed | 0.502x |
| `a.flatten` | float64 | 1,000 | 0.00172 ms | not_measured | managed | 0.255x |
| `a.flatten` | float64 | 100,000 | 0.09023 ms | not_measured | managed | 0.126x |
| `a.flatten` | float64 | 10,000,000 | 14.2341 ms | not_measured | managed | 1.212x |
| `a.ravel() (view)` | float64 | 1,000 | 0.000445 ms | not_measured | managed | 0.173x |
| `a.ravel() (view)` | float64 | 100,000 | 0.000466 ms | not_measured | managed | 0.170x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.000534 ms | not_measured | managed | 0.200x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.00749 ms | not_measured | managed | 0.199x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.2900 ms | not_measured | managed | 1.406x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.00124 ms | not_measured | managed | 5.269x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.00118 ms | not_measured | managed | 14.321x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 190.000x |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 296.000x |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.000309 ms | not_measured | managed | 1.337x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.000415 ms | not_measured | managed | 1.612x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.000645 ms | not_measured | managed | 0.665x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.000639 ms | not_measured | managed | 1.119x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.01046 ms | not_measured | managed | 0.305x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.3077 ms | not_measured | managed | 1.378x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.000559 ms | not_measured | managed | 5.649x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.000610 ms | not_measured | managed | 14.079x |
| `np.byteswap` | float64 | 1,000 | 0.00165 ms | not_measured | managed | 0.421x |
| `np.byteswap` | float64 | 100,000 | 0.1374 ms | not_measured | managed | 0.246x |
| `np.byteswap` | float64 | 10,000,000 | 14.7391 ms | not_measured | managed | 1.814x |
| `np.c_` | float64 | 1,000 | 0.01017 ms | not_measured | managed | 0.409x |
| `np.c_` | float64 | 100,000 | 0.3085 ms | not_measured | managed | 1.028x |
| `np.c_` | float64 | 10,000,000 | 39.9310 ms | not_measured | managed | 1.648x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.01106 ms | not_measured | managed | 0.190x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.3078 ms | not_measured | managed | 1.571x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.00664 ms | not_measured | managed | 0.124x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.3079 ms | not_measured | managed | 1.410x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.01392 ms | not_measured | managed | 0.087x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.4694 ms | not_measured | managed | 0.963x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.00825 ms | not_measured | managed | 0.109x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.3133 ms | not_measured | managed | 1.042x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.00577 ms | not_measured | managed | 0.159x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.3084 ms | not_measured | managed | 0.999x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00423 ms | not_measured | managed | 0.268x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.2975 ms | not_measured | managed | 0.973x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.00545 ms | not_measured | managed | 0.371x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.1032 ms | not_measured | managed | 0.256x |
| `np.diag` | float64 | 1,000 | 0.00704 ms | not_measured | managed | 0.102x |
| `np.diag` | float64 | 100,000 | 0.07959 ms | not_measured | managed | 0.009x |
| `np.diag` | float64 | 10,000,000 | 3.9981 ms | not_measured | managed | 0.000x |
| `np.diagflat` | float64 | 1,000 | 0.00711 ms | not_measured | managed | 0.399x |
| `np.diagflat` | float64 | 100,000 | 0.08037 ms | not_measured | managed | 0.174x |
| `np.diagflat` | float64 | 10,000,000 | 3.8739 ms | not_measured | managed | 1.270x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.00182 ms | not_measured | managed | 4.954x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.00255 ms | not_measured | managed | 8.514x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.00745 ms | not_measured | managed | 0.325x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.2994 ms | not_measured | managed | 1.005x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.000329 ms | not_measured | managed | 3.678x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.000331 ms | not_measured | managed | 4.459x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.000342 ms | not_measured | managed | 7.792x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.000353 ms | not_measured | managed | 4.516x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.000317 ms | not_measured | managed | 3.618x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.000329 ms | not_measured | managed | 9.784x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.000337 ms | not_measured | managed | 3.813x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.000316 ms | not_measured | managed | 4.085x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.000345 ms | not_measured | managed | 7.968x |
| `np.fill_diagonal` | float64 | 1,000 | 0.00186 ms | not_measured | managed | 0.403x |
| `np.fill_diagonal` | float64 | 100,000 | 0.00263 ms | not_measured | managed | 0.891x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.03860 ms | not_measured | managed | 1.682x |
| `np.flip` | float64 | 1,000 | 0.000258 ms | not_measured | managed | 1.481x |
| `np.flip` | float64 | 100,000 | 0.000250 ms | not_measured | managed | 1.532x |
| `np.flip` | float64 | 10,000,000 | 0.000209 ms | not_measured | managed | 2.813x |
| `np.fliplr` | float64 | 1,000 | 0.000267 ms | not_measured | managed | 1.184x |
| `np.fliplr` | float64 | 100,000 | 0.000250 ms | not_measured | managed | 1.252x |
| `np.fliplr` | float64 | 10,000,000 | 0.000270 ms | not_measured | managed | 1.637x |
| `np.flipud` | float64 | 1,000 | 0.000253 ms | not_measured | managed | 1.103x |
| `np.flipud` | float64 | 100,000 | 0.000274 ms | not_measured | managed | 1.011x |
| `np.flipud` | float64 | 10,000,000 | 0.000262 ms | not_measured | managed | 1.469x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.00276 ms | not_measured | managed | 3.240x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.00280 ms | not_measured | managed | 9.053x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.00307 ms | not_measured | managed | 0.499x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.3044 ms | not_measured | managed | 1.056x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.00855 ms | not_measured | managed | 0.672x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.1004 ms | not_measured | managed | 0.373x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.04150 ms | not_measured | managed | 0.977x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 9.6060 ms | not_measured | managed | 0.797x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.04022 ms | not_measured | managed | 0.411x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 1.0225 ms | not_measured | managed | 0.412x |
| `np.ix_` | float64 | 1,000 | 0.000940 ms | not_measured | managed | 1.831x |
| `np.ix_` | float64 | 100,000 | 0.00120 ms | not_measured | managed | 1.454x |
| `np.ix_` | float64 | 10,000,000 | 0.00120 ms | not_measured | managed | 3.139x |
| `np.matrix_transpose` | float64 | 1,000 | 0.000427 ms | not_measured | managed | 1.279x |
| `np.matrix_transpose` | float64 | 100,000 | 0.000312 ms | not_measured | managed | 1.718x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.000340 ms | not_measured | managed | 2.385x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.000884 ms | not_measured | managed | 2.265x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.000873 ms | not_measured | managed | 2.314x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.000969 ms | not_measured | managed | 4.046x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.00793 ms | not_measured | managed | 1.000x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.1080 ms | not_measured | managed | 0.419x |
| `np.permute_dims` | float64 | 1,000 | 0.000338 ms | not_measured | managed | 0.994x |
| `np.permute_dims` | float64 | 100,000 | 0.000335 ms | not_measured | managed | 1.003x |
| `np.permute_dims` | float64 | 10,000,000 | 0.000341 ms | not_measured | managed | 1.296x |
| `np.r_` | float64 | 1,000 | 0.00762 ms | not_measured | managed | 0.353x |
| `np.r_` | float64 | 100,000 | 0.3056 ms | not_measured | managed | 1.040x |
| `np.r_` | float64 | 10,000,000 | 27.9312 ms | not_measured | managed | 1.280x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.00362 ms | not_measured | managed | 1.545x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.2039 ms | not_measured | managed | 1.716x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 31.9341 ms | not_measured | managed | 1.704x |
| `np.r_(0:N)` | float64 | 1,000 | 0.00873 ms | not_measured | managed | 0.273x |
| `np.r_(0:N)` | float64 | 100,000 | 0.1848 ms | not_measured | managed | 1.730x |
| `np.r_(0:N)` | float64 | 10,000,000 | 27.7859 ms | not_measured | managed | 1.254x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.00942 ms | not_measured | managed | 0.436x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.3086 ms | not_measured | managed | 1.036x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 27.4761 ms | not_measured | managed | 1.304x |
| `np.ravel` | float64 | 1,000 | 0.000479 ms | not_measured | managed | 0.622x |
| `np.ravel` | float64 | 100,000 | 0.000503 ms | not_measured | managed | 0.598x |
| `np.ravel` | float64 | 10,000,000 | 0.000555 ms | not_measured | managed | 0.683x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.00369 ms | not_measured | managed | 0.716x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.3116 ms | not_measured | managed | 2.735x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.000506 ms | not_measured | managed | 1.121x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.000491 ms | not_measured | managed | 1.102x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.000519 ms | not_measured | managed | 2.605x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.00799 ms | not_measured | managed | 0.269x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.3045 ms | not_measured | managed | 1.412x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.01136 ms | not_measured | managed | 0.364x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.09907 ms | not_measured | managed | 0.355x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.000477 ms | not_measured | managed | 1.176x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.000325 ms | not_measured | managed | 1.778x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.000458 ms | not_measured | managed | 2.865x |
| `np.rot90` | float64 | 1,000 | 0.000553 ms | not_measured | managed | 5.662x |
| `np.rot90` | float64 | 100,000 | 0.000524 ms | not_measured | managed | 6.403x |
| `np.rot90` | float64 | 10,000,000 | 0.000495 ms | not_measured | managed | 18.638x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.05017 ms | not_measured | managed | 1.049x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 9.3783 ms | not_measured | managed | 0.789x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.04678 ms | not_measured | managed | 0.840x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 9.6200 ms | not_measured | managed | 0.764x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.000000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.000000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.00184 ms | not_measured | managed | 4.748x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.00268 ms | not_measured | managed | 7.678x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.000223 ms | not_measured | managed | 1.269x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.000262 ms | not_measured | managed | 1.080x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.000274 ms | not_measured | managed | 1.431x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.000264 ms | not_measured | managed | 1.170x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.000258 ms | not_measured | managed | 1.205x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.000261 ms | not_measured | managed | 2.356x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.01000 ms | not_measured | managed | 0.193x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.3032 ms | not_measured | managed | 1.040x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.00557 ms | not_measured | managed | 0.420x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.3007 ms | not_measured | managed | 1.050x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.000305 ms | not_measured | managed | 1.207x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.000416 ms | not_measured | managed | 0.880x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.000305 ms | not_measured | managed | 1.751x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.000447 ms | not_measured | managed | 0.881x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.000454 ms | not_measured | managed | 0.826x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.000463 ms | not_measured | managed | 1.168x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.01196 ms | not_measured | managed | 0.206x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.3105 ms | not_measured | managed | 1.355x |
| `np.transpose(a)` | float64 | 1,000 | 0.000391 ms | not_measured | managed | 0.841x |
| `np.transpose(a)` | float64 | 100,000 | 0.000377 ms | not_measured | managed | 0.859x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.000384 ms | not_measured | managed | 1.208x |
| `np.tri` | float64 | 1,000 | 0.00252 ms | not_measured | managed | 1.217x |
| `np.tri` | float64 | 100,000 | 0.1012 ms | not_measured | managed | 0.485x |
| `np.tri` | float64 | 10,000,000 | 17.3121 ms | not_measured | managed | 1.170x |
| `np.tril` | float64 | 1,000 | 0.00179 ms | not_measured | managed | 2.269x |
| `np.tril` | float64 | 100,000 | 0.09954 ms | not_measured | managed | 0.504x |
| `np.tril` | float64 | 10,000,000 | 19.9021 ms | not_measured | managed | 1.220x |
| `np.tril_indices` | float64 | 1,000 | 0.00409 ms | not_measured | managed | 2.512x |
| `np.tril_indices` | float64 | 100,000 | 0.1112 ms | not_measured | managed | 0.639x |
| `np.tril_indices` | float64 | 10,000,000 | 11.5809 ms | not_measured | managed | 2.254x |
| `np.trim_zeros` | float64 | 1,000 | 0.01195 ms | not_measured | managed | 0.979x |
| `np.trim_zeros` | float64 | 100,000 | 0.02022 ms | not_measured | managed | 48.355x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.09244 ms | not_measured | managed | 2052.367x |
| `np.triu` | float64 | 1,000 | 0.00191 ms | not_measured | managed | 2.115x |
| `np.triu` | float64 | 100,000 | 0.1006 ms | not_measured | managed | 0.469x |
| `np.triu` | float64 | 10,000,000 | 19.7064 ms | not_measured | managed | 1.260x |
| `np.triu_indices` | float64 | 1,000 | 0.00397 ms | not_measured | managed | 2.623x |
| `np.triu_indices` | float64 | 100,000 | 0.1023 ms | not_measured | managed | 0.679x |
| `np.triu_indices` | float64 | 10,000,000 | 11.7699 ms | not_measured | managed | 2.422x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.02012 ms | not_measured | managed | 1.149x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 5.6403 ms | not_measured | managed | 0.829x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.00916 ms | not_measured | managed | 1.834x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.3221 ms | not_measured | managed | 0.675x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.03132 ms | not_measured | managed | 1.144x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 3.0514 ms | not_measured | managed | 2.435x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.01424 ms | not_measured | managed | 0.686x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 5.0325 ms | not_measured | managed | 0.238x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.02969 ms | not_measured | managed | 0.787x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 2.4482 ms | not_measured | managed | 2.250x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.00907 ms | not_measured | managed | 1.821x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 4.1707 ms | not_measured | managed | 1.123x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.00264 ms | not_measured | managed | 1.177x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.00279 ms | not_measured | managed | 2.470x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 4.829x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.00194 ms | not_measured | managed | 11.009x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.00675 ms | not_measured | managed | 0.283x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.2941 ms | not_measured | managed | 1.099x |
| `reshape 1D->2D` | float64 | 1,000 | 0.000612 ms | not_measured | managed | 0.235x |
| `reshape 1D->2D` | float64 | 100,000 | 0.000506 ms | not_measured | managed | 0.275x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.000496 ms | not_measured | managed | 0.363x |
| `reshape 1D->3D` | float64 | 1,000 | 0.000655 ms | not_measured | managed | 0.231x |
| `reshape 1D->3D` | float64 | 100,000 | 0.000585 ms | not_measured | managed | 0.260x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.000568 ms | not_measured | managed | 0.410x |
| `reshape 2D->1D` | float64 | 1,000 | 0.000552 ms | not_measured | managed | 0.243x |
| `reshape 2D->1D` | float64 | 100,000 | 0.000517 ms | not_measured | managed | 0.261x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.000516 ms | not_measured | managed | 0.322x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.000751 ms | not_measured | managed | 1.903x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.01529 ms | not_measured | managed | 2.649x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.00113 ms | not_measured | managed | 1.436x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.000998 ms | not_measured | managed | 39.794x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.000844 ms | not_measured | managed | 0.468x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.05694 ms | not_measured | managed | 0.284x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.000841 ms | not_measured | managed | 0.447x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.05698 ms | not_measured | managed | 0.275x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.01200 ms | not_measured | managed | 0.297x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.8682 ms | not_measured | managed | 0.163x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.02959 ms | not_measured | managed | 0.311x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 3.2837 ms | not_measured | managed | 0.428x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.01088 ms | not_measured | managed | 0.434x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.3041 ms | not_measured | managed | 1.890x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.00371 ms | not_measured | managed | 0.272x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.1463 ms | not_measured | managed | 0.084x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.00186 ms | not_measured | managed | 0.831x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.1451 ms | not_measured | managed | 0.676x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 58.000x |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.000002 ms | not_measured | managed | 33.000x |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.000002 ms | not_measured | managed | 29.500x |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.000003 ms | not_measured | managed | 19.333x |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.00589 ms | not_measured | managed | 0.496x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.2905 ms | not_measured | managed | 0.588x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.00526 ms | not_measured | managed | 0.424x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.2796 ms | not_measured | managed | 0.625x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.000279 ms | not_measured | managed | 0.348x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.000231 ms | not_measured | managed | 0.442x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.000582 ms | not_measured | managed | 0.730x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.01897 ms | 0.00730 ms | openblas | 0.945x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.000109 ms | not_measured | managed | 1.606x |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.02073 ms | not_measured | managed | 0.482x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.000320 ms | not_measured | managed | 0.444x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.000298 ms | not_measured | managed | 0.483x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.000009 ms | not_measured | managed | 9.667x |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.000009 ms | not_measured | managed | 9.667x |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.000693 ms | not_measured | managed | 1.492x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.01685 ms | not_measured | managed | 0.561x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.000668 ms | not_measured | managed | 1.546x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.01688 ms | not_measured | managed | 0.560x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.00293 ms | not_measured | managed | 0.716x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.1467 ms | not_measured | managed | 1.225x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.01137 ms | not_measured | managed | 0.230x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.7434 ms | not_measured | managed | 0.187x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.000658 ms | not_measured | managed | 3.474x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.00792 ms | not_measured | managed | 9.536x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.00681 ms | not_measured | managed | 0.255x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.1747 ms | not_measured | managed | 0.617x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.00572 ms | not_measured | managed | 0.414x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.5030 ms | not_measured | managed | 0.745x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.000524 ms | not_measured | managed | 0.292x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.000503 ms | not_measured | managed | 0.328x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.00427 ms | not_measured | managed | 0.158x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.2703 ms | not_measured | managed | 0.041x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.00215 ms | not_measured | managed | 0.227x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.1222 ms | not_measured | managed | 0.093x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.01473 ms | not_measured | managed | 0.417x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 2.8983 ms | not_measured | managed | 0.706x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.195x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.05096 ms | not_measured | managed | 0.183x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.000012 ms | not_measured | managed | 10.417x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.000014 ms | not_measured | managed | 8.857x |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.02241 ms | not_measured | managed | 0.132x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 1.7659 ms | not_measured | managed | 0.273x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.000229 ms | not_measured | managed | 0.511x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.000248 ms | not_measured | managed | 0.484x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.000354 ms | not_measured | managed | 0.359x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.000436 ms | not_measured | managed | 0.314x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.00189 ms | not_measured | managed | 0.546x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.09919 ms | not_measured | managed | 0.519x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 76.000x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 77.000x |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.000377 ms | not_measured | managed | 0.382x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.1668 ms | not_measured | managed | 0.067x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.2170 ms | not_measured | managed | 0.679x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 3.1017 ms | not_measured | managed | 0.121x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.00996 ms | not_measured | managed | 0.763x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 2.9461 ms | not_measured | managed | 0.275x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.000482 ms | not_measured | managed | 2.100x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.000513 ms | not_measured | managed | 2.033x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.000253 ms | not_measured | managed | 0.360x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.000353 ms | not_measured | managed | 0.261x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.000255 ms | not_measured | managed | 0.298x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.000246 ms | not_measured | managed | 0.301x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.04235 ms | not_measured | managed | 0.990x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 8.2285 ms | not_measured | managed | 0.481x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.1371 ms | not_measured | managed | 0.172x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 13.6501 ms | not_measured | managed | 0.247x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.02404 ms | not_measured | managed | 0.920x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 4.3466 ms | not_measured | managed | 0.458x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.02659 ms | not_measured | managed | 0.395x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 1.3751 ms | not_measured | managed | 0.972x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.06777 ms | not_measured | managed | 0.819x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 7.7575 ms | not_measured | managed | 1.190x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.03089 ms | not_measured | managed | 0.289x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 2.4413 ms | not_measured | managed | 0.341x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.1026 ms | not_measured | managed | 0.399x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 8.9262 ms | not_measured | managed | 0.438x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.04854 ms | not_measured | managed | 0.432x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 4.0950 ms | not_measured | managed | 0.486x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.01845 ms | not_measured | managed | 0.988x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 1.9438 ms | not_measured | managed | 1.202x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.03225 ms | not_measured | managed | 0.459x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 2.8683 ms | not_measured | managed | 0.478x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1546 ms | not_measured | managed | 0.417x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 15.4663 ms | not_measured | managed | 0.613x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.02729 ms | not_measured | managed | 0.535x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 2.4148 ms | not_measured | managed | 0.550x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.02117 ms | not_measured | managed | 0.442x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 1.9553 ms | not_measured | managed | 0.436x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.03355 ms | not_measured | managed | 0.544x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 2.8382 ms | not_measured | managed | 0.612x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.04001 ms | not_measured | managed | 0.672x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 3.7464 ms | not_measured | managed | 1.076x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.2149 ms | not_measured | managed | 0.353x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 21.4565 ms | not_measured | managed | 0.538x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.06952 ms | not_measured | managed | 0.944x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 7.1095 ms | not_measured | managed | 1.014x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.1375 ms | not_measured | managed | 0.518x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 13.8807 ms | not_measured | managed | 0.781x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.06601 ms | not_measured | managed | 0.513x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 6.5760 ms | not_measured | managed | 0.492x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.1008 ms | not_measured | managed | 0.564x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 9.9743 ms | not_measured | managed | 0.524x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.02462 ms | not_measured | managed | 0.475x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 2.2754 ms | not_measured | managed | 0.473x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.03032 ms | not_measured | managed | 0.500x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 2.8110 ms | not_measured | managed | 0.515x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.01661 ms | not_measured | managed | 0.841x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 1.7069 ms | not_measured | managed | 1.142x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.05482 ms | not_measured | managed | 0.600x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 4.6245 ms | not_measured | managed | 0.988x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.03177 ms | not_measured | managed | 0.913x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 2.8364 ms | not_measured | managed | 0.998x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.01467 ms | not_measured | managed | 0.279x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 1.3751 ms | not_measured | managed | 0.257x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.00783 ms | not_measured | managed | 1.015x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.8697 ms | not_measured | managed | 0.760x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.02388 ms | not_measured | managed | 0.450x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 2.2069 ms | not_measured | managed | 0.462x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.01507 ms | not_measured | managed | 0.270x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 1.3478 ms | not_measured | managed | 0.268x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.01464 ms | not_measured | managed | 0.276x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 1.3628 ms | not_measured | managed | 0.262x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.02141 ms | not_measured | managed | 0.480x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 2.0629 ms | not_measured | managed | 0.464x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.01612 ms | not_measured | managed | 0.826x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 1.7249 ms | not_measured | managed | 1.492x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.02716 ms | not_measured | managed | 0.795x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 2.6635 ms | not_measured | managed | 0.792x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.01999 ms | not_measured | managed | 0.449x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 1.9706 ms | not_measured | managed | 0.416x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.04054 ms | not_measured | managed | 0.509x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 4.1059 ms | not_measured | managed | 0.479x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.02377 ms | not_measured | managed | 0.448x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 2.3054 ms | not_measured | managed | 0.443x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.06236 ms | not_measured | managed | 0.559x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 6.2383 ms | not_measured | managed | 0.610x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.01541 ms | not_measured | managed | 0.623x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 1.5677 ms | not_measured | managed | 0.795x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.01287 ms | not_measured | managed | 0.387x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 1.2158 ms | not_measured | managed | 0.536x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.08525 ms | not_measured | managed | 0.590x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 8.5649 ms | not_measured | managed | 0.970x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.04641 ms | not_measured | managed | 0.534x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 4.6507 ms | not_measured | managed | 0.828x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.03655 ms | not_measured | managed | 0.588x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 3.7003 ms | not_measured | managed | 0.969x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.08020 ms | not_measured | managed | 0.427x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 8.0322 ms | not_measured | managed | 0.765x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.00215 ms | not_measured | managed | 1.090x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1430 ms | not_measured | managed | 1.111x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 18.9899 ms | not_measured | managed | 1.071x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.00217 ms | not_measured | managed | 1.346x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4429 ms | not_measured | managed | 1.117x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 46.5590 ms | not_measured | managed | 1.353x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.000689 ms | not_measured | managed | 1.353x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.00862 ms | not_measured | managed | 0.643x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 3.0844 ms | not_measured | managed | 1.176x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.000709 ms | not_measured | managed | 1.331x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.01673 ms | not_measured | managed | 0.597x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 7.9642 ms | not_measured | managed | 0.999x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.000623 ms | not_measured | managed | 1.377x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.00256 ms | not_measured | managed | 0.900x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.5171 ms | not_measured | managed | 1.105x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.000612 ms | not_measured | managed | 1.395x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.00490 ms | not_measured | managed | 0.709x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 2.5722 ms | not_measured | managed | 1.209x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.000870 ms | not_measured | managed | 1.077x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.02712 ms | not_measured | managed | 0.327x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 7.7666 ms | not_measured | managed | 1.151x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.000612 ms | not_measured | managed | 1.397x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.00145 ms | not_measured | managed | 1.101x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.2289 ms | not_measured | managed | 1.092x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.000624 ms | not_measured | managed | 1.367x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.00253 ms | not_measured | managed | 0.912x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 0.5061 ms | not_measured | managed | 1.255x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.000621 ms | not_measured | managed | 1.594x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.00457 ms | not_measured | managed | 0.776x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 2.5729 ms | not_measured | managed | 1.177x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.000996 ms | not_measured | managed | 0.949x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.03715 ms | not_measured | managed | 0.379x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 7.9770 ms | not_measured | managed | 1.051x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.000617 ms | not_measured | managed | 1.370x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.00157 ms | not_measured | managed | 1.023x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.2376 ms | not_measured | managed | 1.056x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.00222 ms | not_measured | managed | 1.165x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1436 ms | not_measured | managed | 1.168x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 18.8795 ms | not_measured | managed | 1.085x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.00179 ms | not_measured | managed | 1.790x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.4050 ms | not_measured | managed | 1.243x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 41.8896 ms | not_measured | managed | 1.491x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.000658 ms | not_measured | managed | 1.353x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.00897 ms | not_measured | managed | 0.626x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 3.1396 ms | not_measured | managed | 1.154x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.000707 ms | not_measured | managed | 1.314x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.01768 ms | not_measured | managed | 0.560x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 8.1687 ms | not_measured | managed | 1.063x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.000553 ms | not_measured | managed | 1.559x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.00257 ms | not_measured | managed | 0.918x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 0.5204 ms | not_measured | managed | 1.071x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.000612 ms | not_measured | managed | 1.490x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.00416 ms | not_measured | managed | 1.010x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 2.6745 ms | not_measured | managed | 1.140x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.000858 ms | not_measured | managed | 1.090x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.02727 ms | not_measured | managed | 0.384x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 8.1990 ms | not_measured | managed | 1.172x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.000620 ms | not_measured | managed | 1.413x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.00147 ms | not_measured | managed | 1.081x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.2393 ms | not_measured | managed | 1.077x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.000615 ms | not_measured | managed | 1.377x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.00258 ms | not_measured | managed | 0.892x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 0.5269 ms | not_measured | managed | 1.130x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.000628 ms | not_measured | managed | 1.360x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.00461 ms | not_measured | managed | 0.833x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 2.6394 ms | not_measured | managed | 1.105x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.000916 ms | not_measured | managed | 1.039x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.03734 ms | not_measured | managed | 0.307x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 7.9881 ms | not_measured | managed | 1.051x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.000803 ms | not_measured | managed | 1.054x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.00145 ms | not_measured | managed | 1.120x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.2392 ms | not_measured | managed | 1.121x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.00103 ms | not_measured | managed | 2.162x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.02153 ms | not_measured | managed | 1.804x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 15.5661 ms | not_measured | managed | 1.090x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.00202 ms | not_measured | managed | 2.261x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.1629 ms | not_measured | managed | 0.670x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 16.5403 ms | not_measured | managed | 0.993x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.000647 ms | not_measured | managed | 5.034x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.00512 ms | not_measured | managed | 3.601x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 2.6278 ms | not_measured | managed | 3.136x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.000685 ms | not_measured | managed | 2.835x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.00903 ms | not_measured | managed | 2.593x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 7.4006 ms | not_measured | managed | 1.967x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.000792 ms | not_measured | managed | 3.125x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.04471 ms | not_measured | managed | 1.202x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 4.4482 ms | not_measured | managed | 1.826x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.000794 ms | not_measured | managed | 3.162x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.04480 ms | not_measured | managed | 1.028x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 5.0912 ms | not_measured | managed | 1.599x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.000657 ms | not_measured | managed | 3.621x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.00782 ms | not_measured | managed | 4.881x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 7.3716 ms | not_measured | managed | 1.531x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.000785 ms | not_measured | managed | 3.180x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.04500 ms | not_measured | managed | 1.143x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 4.7170 ms | not_measured | managed | 1.757x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.000959 ms | not_measured | managed | 2.688x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.03937 ms | not_measured | managed | 1.320x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 4.2149 ms | not_measured | managed | 1.897x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.000976 ms | not_measured | managed | 2.494x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.03946 ms | not_measured | managed | 1.111x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 4.8778 ms | not_measured | managed | 1.608x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.000672 ms | not_measured | managed | 3.656x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.00814 ms | not_measured | managed | 7.584x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 7.3958 ms | not_measured | managed | 1.501x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.000981 ms | not_measured | managed | 2.576x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.04094 ms | not_measured | managed | 1.255x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 3.7494 ms | not_measured | managed | 2.114x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.00422 ms | not_measured | managed | 4.521x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.3821 ms | not_measured | managed | 2.295x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 38.4016 ms | not_measured | managed | 4.232x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.000946 ms | not_measured | managed | 8.273x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.01821 ms | not_measured | managed | 2.662x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 6.0339 ms | not_measured | managed | 5.096x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.000964 ms | not_measured | managed | 6.735x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.03589 ms | not_measured | managed | 1.807x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 15.9149 ms | not_measured | managed | 3.791x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.000868 ms | not_measured | managed | 1.218x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.02130 ms | not_measured | managed | 1.498x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 15.9752 ms | not_measured | managed | 1.073x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.000983 ms | not_measured | managed | 2.887x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.06731 ms | not_measured | managed | 2.904x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 6.8480 ms | not_measured | managed | 3.486x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.000629 ms | not_measured | managed | 1.431x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.00582 ms | not_measured | managed | 2.697x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 2.6383 ms | not_measured | managed | 3.297x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.000636 ms | not_measured | managed | 1.417x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.00906 ms | not_measured | managed | 1.820x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 7.4946 ms | not_measured | managed | 1.972x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.000798 ms | not_measured | managed | 1.515x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.04479 ms | not_measured | managed | 0.746x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 4.4453 ms | not_measured | managed | 1.452x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.000797 ms | not_measured | managed | 1.639x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.04466 ms | not_measured | managed | 0.870x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 5.1496 ms | not_measured | managed | 1.663x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.000657 ms | not_measured | managed | 1.518x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.00785 ms | not_measured | managed | 2.382x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 7.4278 ms | not_measured | managed | 1.255x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.000797 ms | not_measured | managed | 1.513x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.04507 ms | not_measured | managed | 0.718x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 4.7225 ms | not_measured | managed | 1.754x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.000964 ms | not_measured | managed | 1.271x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.03921 ms | not_measured | managed | 0.826x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 4.2539 ms | not_measured | managed | 1.989x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.000986 ms | not_measured | managed | 1.229x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.03818 ms | not_measured | managed | 0.880x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 4.9164 ms | not_measured | managed | 1.504x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.000647 ms | not_measured | managed | 1.516x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.00766 ms | not_measured | managed | 2.037x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 7.4666 ms | not_measured | managed | 1.243x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.000983 ms | not_measured | managed | 1.419x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.04095 ms | not_measured | managed | 0.809x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 3.7532 ms | not_measured | managed | 2.081x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.00422 ms | not_measured | managed | 4.403x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.3820 ms | not_measured | managed | 2.389x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 38.3175 ms | not_measured | managed | 4.242x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.000811 ms | not_measured | managed | 8.750x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.01828 ms | not_measured | managed | 2.614x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 6.0145 ms | not_measured | managed | 5.186x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.000962 ms | not_measured | managed | 6.030x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.03588 ms | not_measured | managed | 2.019x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 16.4243 ms | not_measured | managed | 3.494x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.00216 ms | not_measured | managed | 1.405x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1422 ms | not_measured | managed | 1.173x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 18.9191 ms | not_measured | managed | 1.085x |
| `np.amax (float16)` | float16 | 1,000 | 0.00217 ms | not_measured | managed | 1.665x |
| `np.amax (float16)` | float16 | 100,000 | 0.4431 ms | not_measured | managed | 1.146x |
| `np.amax (float16)` | float16 | 10,000,000 | 46.5958 ms | not_measured | managed | 1.358x |
| `np.amax (float32)` | float32 | 1,000 | 0.000671 ms | not_measured | managed | 2.396x |
| `np.amax (float32)` | float32 | 100,000 | 0.00863 ms | not_measured | managed | 0.737x |
| `np.amax (float32)` | float32 | 10,000,000 | 3.1121 ms | not_measured | managed | 1.103x |
| `np.amax (float64)` | float64 | 1,000 | 0.000719 ms | not_measured | managed | 2.270x |
| `np.amax (float64)` | float64 | 100,000 | 0.01673 ms | not_measured | managed | 0.651x |
| `np.amax (float64)` | float64 | 10,000,000 | 7.7532 ms | not_measured | managed | 1.012x |
| `np.amax (int16)` | int16 | 1,000 | 0.000602 ms | not_measured | managed | 2.560x |
| `np.amax (int16)` | int16 | 100,000 | 0.00255 ms | not_measured | managed | 1.220x |
| `np.amax (int16)` | int16 | 10,000,000 | 0.5092 ms | not_measured | managed | 1.085x |
| `np.amax (int32)` | int32 | 1,000 | 0.000794 ms | not_measured | managed | 1.932x |
| `np.amax (int32)` | int32 | 100,000 | 0.00416 ms | not_measured | managed | 1.095x |
| `np.amax (int32)` | int32 | 10,000,000 | 2.5552 ms | not_measured | managed | 1.191x |
| `np.amax (int64)` | int64 | 1,000 | 0.000871 ms | not_measured | managed | 1.831x |
| `np.amax (int64)` | int64 | 100,000 | 0.02646 ms | not_measured | managed | 1.663x |
| `np.amax (int64)` | int64 | 10,000,000 | 7.7139 ms | not_measured | managed | 1.106x |
| `np.amax (int8)` | int8 | 1,000 | 0.000614 ms | not_measured | managed | 2.668x |
| `np.amax (int8)` | int8 | 100,000 | 0.00146 ms | not_measured | managed | 1.684x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2276 ms | not_measured | managed | 1.113x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.000611 ms | not_measured | managed | 2.573x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.00257 ms | not_measured | managed | 1.212x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 0.5253 ms | not_measured | managed | 1.177x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.000791 ms | not_measured | managed | 1.932x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.00453 ms | not_measured | managed | 0.951x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 2.5913 ms | not_measured | managed | 1.175x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.000969 ms | not_measured | managed | 1.806x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.03717 ms | not_measured | managed | 0.441x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 8.0134 ms | not_measured | managed | 1.070x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.000617 ms | not_measured | managed | 2.452x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.00146 ms | not_measured | managed | 1.688x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.2371 ms | not_measured | managed | 1.073x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.00415 ms | not_measured | managed | 0.703x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1830 ms | not_measured | managed | 0.765x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 21.1181 ms | not_measured | managed | 0.951x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.00338 ms | not_measured | managed | 1.153x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6557 ms | not_measured | managed | 0.791x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 139.3714 ms | not_measured | managed | 0.453x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.000905 ms | not_measured | managed | 2.473x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.02820 ms | not_measured | managed | 0.337x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 3.8763 ms | not_measured | managed | 0.993x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.00101 ms | not_measured | managed | 1.977x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.05417 ms | not_measured | managed | 0.367x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 8.3721 ms | not_measured | managed | 1.052x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.000748 ms | not_measured | managed | 2.508x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.00718 ms | not_measured | managed | 1.156x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 0.6794 ms | not_measured | managed | 1.124x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.000765 ms | not_measured | managed | 2.494x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.00989 ms | not_measured | managed | 0.929x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 3.1203 ms | not_measured | managed | 1.102x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.000801 ms | not_measured | managed | 2.479x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.02794 ms | not_measured | managed | 0.633x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 7.9715 ms | not_measured | managed | 1.113x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.000696 ms | not_measured | managed | 2.695x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.00757 ms | not_measured | managed | 1.063x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3398 ms | not_measured | managed | 1.060x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.000731 ms | not_measured | managed | 2.592x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.00694 ms | not_measured | managed | 1.259x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 0.6583 ms | not_measured | managed | 1.124x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.000747 ms | not_measured | managed | 2.557x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.01026 ms | not_measured | managed | 0.826x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 3.1762 ms | not_measured | managed | 1.143x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.000858 ms | not_measured | managed | 2.307x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.03552 ms | not_measured | managed | 0.614x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 8.0026 ms | not_measured | managed | 1.068x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.000700 ms | not_measured | managed | 2.691x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.00733 ms | not_measured | managed | 1.100x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.3373 ms | not_measured | managed | 1.121x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.00216 ms | not_measured | managed | 1.392x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1440 ms | not_measured | managed | 1.232x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 18.8598 ms | not_measured | managed | 1.086x |
| `np.amin (float16)` | float16 | 1,000 | 0.00179 ms | not_measured | managed | 2.168x |
| `np.amin (float16)` | float16 | 100,000 | 0.4052 ms | not_measured | managed | 1.242x |
| `np.amin (float16)` | float16 | 10,000,000 | 41.9022 ms | not_measured | managed | 1.489x |
| `np.amin (float32)` | float32 | 1,000 | 0.000654 ms | not_measured | managed | 2.391x |
| `np.amin (float32)` | float32 | 100,000 | 0.00895 ms | not_measured | managed | 0.701x |
| `np.amin (float32)` | float32 | 10,000,000 | 3.1673 ms | not_measured | managed | 1.096x |
| `np.amin (float64)` | float64 | 1,000 | 0.000700 ms | not_measured | managed | 2.309x |
| `np.amin (float64)` | float64 | 100,000 | 0.01703 ms | not_measured | managed | 0.736x |
| `np.amin (float64)` | float64 | 10,000,000 | 8.1744 ms | not_measured | managed | 1.144x |
| `np.amin (int16)` | int16 | 1,000 | 0.000621 ms | not_measured | managed | 2.546x |
| `np.amin (int16)` | int16 | 100,000 | 0.00257 ms | not_measured | managed | 1.203x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.5208 ms | not_measured | managed | 1.091x |
| `np.amin (int32)` | int32 | 1,000 | 0.000606 ms | not_measured | managed | 2.528x |
| `np.amin (int32)` | int32 | 100,000 | 0.00416 ms | not_measured | managed | 1.019x |
| `np.amin (int32)` | int32 | 10,000,000 | 2.6911 ms | not_measured | managed | 1.100x |
| `np.amin (int64)` | int64 | 1,000 | 0.000870 ms | not_measured | managed | 1.833x |
| `np.amin (int64)` | int64 | 100,000 | 0.02676 ms | not_measured | managed | 0.422x |
| `np.amin (int64)` | int64 | 10,000,000 | 8.2582 ms | not_measured | managed | 1.190x |
| `np.amin (int8)` | int8 | 1,000 | 0.000612 ms | not_measured | managed | 2.490x |
| `np.amin (int8)` | int8 | 100,000 | 0.00146 ms | not_measured | managed | 1.561x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.2419 ms | not_measured | managed | 1.107x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.000618 ms | not_measured | managed | 2.458x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.00258 ms | not_measured | managed | 1.171x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 0.5140 ms | not_measured | managed | 1.214x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.000629 ms | not_measured | managed | 2.439x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.00453 ms | not_measured | managed | 0.983x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 2.6349 ms | not_measured | managed | 1.147x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.000971 ms | not_measured | managed | 1.830x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.03736 ms | not_measured | managed | 0.338x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 7.9171 ms | not_measured | managed | 1.066x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.000787 ms | not_measured | managed | 1.935x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.00147 ms | not_measured | managed | 1.550x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.2345 ms | not_measured | managed | 1.086x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.00456 ms | not_measured | managed | 0.642x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.2264 ms | not_measured | managed | 0.650x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 24.2527 ms | not_measured | managed | 0.820x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.00337 ms | not_measured | managed | 1.241x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.6596 ms | not_measured | managed | 0.733x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 139.1055 ms | not_measured | managed | 0.444x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.000703 ms | not_measured | managed | 2.808x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.02832 ms | not_measured | managed | 0.344x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 3.8055 ms | not_measured | managed | 1.027x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.00100 ms | not_measured | managed | 2.052x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.05437 ms | not_measured | managed | 0.311x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 8.3348 ms | not_measured | managed | 1.044x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.000737 ms | not_measured | managed | 2.744x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.00711 ms | not_measured | managed | 1.180x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 0.6820 ms | not_measured | managed | 1.052x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.000763 ms | not_measured | managed | 2.488x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.01019 ms | not_measured | managed | 0.906x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 3.1780 ms | not_measured | managed | 1.054x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.000814 ms | not_measured | managed | 2.407x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.02763 ms | not_measured | managed | 0.718x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 7.9491 ms | not_measured | managed | 1.124x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.000707 ms | not_measured | managed | 2.724x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.00771 ms | not_measured | managed | 0.948x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.3373 ms | not_measured | managed | 1.063x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.000723 ms | not_measured | managed | 2.707x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.00697 ms | not_measured | managed | 1.215x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 0.6642 ms | not_measured | managed | 1.139x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.000754 ms | not_measured | managed | 2.593x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.01028 ms | not_measured | managed | 0.928x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 3.2266 ms | not_measured | managed | 1.457x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.000866 ms | not_measured | managed | 2.267x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.03714 ms | not_measured | managed | 0.587x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 8.0031 ms | not_measured | managed | 1.066x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.000702 ms | not_measured | managed | 2.651x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.00778 ms | not_measured | managed | 0.889x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.3364 ms | not_measured | managed | 1.063x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.00207 ms | not_measured | managed | 0.931x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1428 ms | not_measured | managed | 0.854x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 18.8189 ms | not_measured | managed | 0.976x |
| `np.argmax (float16)` | float16 | 1,000 | 0.00278 ms | not_measured | managed | 0.997x |
| `np.argmax (float16)` | float16 | 100,000 | 0.2163 ms | not_measured | managed | 2.118x |
| `np.argmax (float16)` | float16 | 10,000,000 | 21.6441 ms | not_measured | managed | 2.597x |
| `np.argmax (float32)` | float32 | 1,000 | 0.000898 ms | not_measured | managed | 0.914x |
| `np.argmax (float32)` | float32 | 100,000 | 0.05694 ms | not_measured | managed | 0.177x |
| `np.argmax (float32)` | float32 | 10,000,000 | 6.1625 ms | not_measured | managed | 0.697x |
| `np.argmax (float64)` | float64 | 1,000 | 0.000896 ms | not_measured | managed | 1.105x |
| `np.argmax (float64)` | float64 | 100,000 | 0.05707 ms | not_measured | managed | 0.552x |
| `np.argmax (float64)` | float64 | 10,000,000 | 8.5654 ms | not_measured | managed | 1.129x |
| `np.argmax (int16)` | int16 | 1,000 | 0.000618 ms | not_measured | managed | 1.303x |
| `np.argmax (int16)` | int16 | 100,000 | 0.00311 ms | not_measured | managed | 1.065x |
| `np.argmax (int16)` | int16 | 10,000,000 | 0.5679 ms | not_measured | managed | 2.003x |
| `np.argmax (int32)` | int32 | 1,000 | 0.000588 ms | not_measured | managed | 1.389x |
| `np.argmax (int32)` | int32 | 100,000 | 0.00563 ms | not_measured | managed | 0.995x |
| `np.argmax (int32)` | int32 | 10,000,000 | 2.8263 ms | not_measured | managed | 1.251x |
| `np.argmax (int64)` | int64 | 1,000 | 0.000838 ms | not_measured | managed | 1.073x |
| `np.argmax (int64)` | int64 | 100,000 | 0.05184 ms | not_measured | managed | 0.281x |
| `np.argmax (int64)` | int64 | 10,000,000 | 8.2316 ms | not_measured | managed | 1.035x |
| `np.argmax (int8)` | int8 | 1,000 | 0.000668 ms | not_measured | managed | 1.249x |
| `np.argmax (int8)` | int8 | 100,000 | 0.00180 ms | not_measured | managed | 1.210x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.2539 ms | not_measured | managed | 2.125x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.000613 ms | not_measured | managed | 1.367x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.00310 ms | not_measured | managed | 1.607x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 0.5663 ms | not_measured | managed | 3.234x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.000596 ms | not_measured | managed | 1.389x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.00560 ms | not_measured | managed | 1.679x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 2.7733 ms | not_measured | managed | 1.501x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.000919 ms | not_measured | managed | 1.046x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.05924 ms | not_measured | managed | 0.372x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 8.4889 ms | not_measured | managed | 1.040x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.000660 ms | not_measured | managed | 1.305x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.00179 ms | not_measured | managed | 1.725x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.2555 ms | not_measured | managed | 3.437x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.00207 ms | not_measured | managed | 0.996x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1426 ms | not_measured | managed | 0.833x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 18.7563 ms | not_measured | managed | 1.011x |
| `np.argmin (float16)` | float16 | 1,000 | 0.00275 ms | not_measured | managed | 0.983x |
| `np.argmin (float16)` | float16 | 100,000 | 0.2171 ms | not_measured | managed | 1.958x |
| `np.argmin (float16)` | float16 | 10,000,000 | 21.6507 ms | not_measured | managed | 2.556x |
| `np.argmin (float32)` | float32 | 1,000 | 0.000887 ms | not_measured | managed | 0.936x |
| `np.argmin (float32)` | float32 | 100,000 | 0.05719 ms | not_measured | managed | 0.177x |
| `np.argmin (float32)` | float32 | 10,000,000 | 6.1614 ms | not_measured | managed | 0.703x |
| `np.argmin (float64)` | float64 | 1,000 | 0.000911 ms | not_measured | managed | 1.023x |
| `np.argmin (float64)` | float64 | 100,000 | 0.05751 ms | not_measured | managed | 0.537x |
| `np.argmin (float64)` | float64 | 10,000,000 | 8.7333 ms | not_measured | managed | 1.076x |
| `np.argmin (int16)` | int16 | 1,000 | 0.000623 ms | not_measured | managed | 1.294x |
| `np.argmin (int16)` | int16 | 100,000 | 0.00298 ms | not_measured | managed | 1.105x |
| `np.argmin (int16)` | int16 | 10,000,000 | 0.5598 ms | not_measured | managed | 2.019x |
| `np.argmin (int32)` | int32 | 1,000 | 0.000610 ms | not_measured | managed | 1.302x |
| `np.argmin (int32)` | int32 | 100,000 | 0.00556 ms | not_measured | managed | 1.005x |
| `np.argmin (int32)` | int32 | 10,000,000 | 2.8503 ms | not_measured | managed | 1.223x |
| `np.argmin (int64)` | int64 | 1,000 | 0.000839 ms | not_measured | managed | 1.057x |
| `np.argmin (int64)` | int64 | 100,000 | 0.05157 ms | not_measured | managed | 0.300x |
| `np.argmin (int64)` | int64 | 10,000,000 | 8.2310 ms | not_measured | managed | 1.028x |
| `np.argmin (int8)` | int8 | 1,000 | 0.000653 ms | not_measured | managed | 1.280x |
| `np.argmin (int8)` | int8 | 100,000 | 0.00180 ms | not_measured | managed | 1.220x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.2517 ms | not_measured | managed | 2.177x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.000631 ms | not_measured | managed | 1.372x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.00305 ms | not_measured | managed | 1.626x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 0.5661 ms | not_measured | managed | 3.268x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.000599 ms | not_measured | managed | 1.379x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.00557 ms | not_measured | managed | 1.830x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 2.7824 ms | not_measured | managed | 1.500x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.000938 ms | not_measured | managed | 0.975x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.05985 ms | not_measured | managed | 0.308x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 8.5004 ms | not_measured | managed | 1.057x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.000658 ms | not_measured | managed | 1.261x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.00179 ms | not_measured | managed | 1.733x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.2500 ms | not_measured | managed | 3.541x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.01646 ms | not_measured | managed | 0.857x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 1.6383 ms | not_measured | managed | 0.635x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 161.7085 ms | not_measured | managed | 0.457x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.04919 ms | not_measured | managed | 0.407x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 6.6701 ms | not_measured | managed | 1.138x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 661.4792 ms | not_measured | managed | 1.059x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.00486 ms | not_measured | managed | 0.591x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 6.5741 ms | not_measured | managed | 1.123x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 669.0225 ms | not_measured | managed | 1.091x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.00355 ms | not_measured | managed | 0.852x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.3070 ms | not_measured | managed | 1.212x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 39.8311 ms | not_measured | managed | 1.349x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.01683 ms | not_measured | managed | 0.413x |
| `np.cumsum (float16)` | float16 | 100,000 | 1.6487 ms | not_measured | managed | 0.302x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 161.5344 ms | not_measured | managed | 0.503x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.00155 ms | not_measured | managed | 1.762x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.1146 ms | not_measured | managed | 1.476x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 10.9514 ms | not_measured | managed | 3.112x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.00216 ms | not_measured | managed | 1.281x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.1660 ms | not_measured | managed | 1.078x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 20.2263 ms | not_measured | managed | 1.880x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.00207 ms | not_measured | managed | 1.090x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.1591 ms | not_measured | managed | 2.158x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 14.8026 ms | not_measured | managed | 2.951x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.00211 ms | not_measured | managed | 1.108x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1581 ms | not_measured | managed | 2.250x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 16.5463 ms | not_measured | managed | 2.709x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.00194 ms | not_measured | managed | 0.816x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1620 ms | not_measured | managed | 0.196x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 20.3484 ms | not_measured | managed | 1.271x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.00257 ms | not_measured | managed | 0.882x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.1565 ms | not_measured | managed | 2.176x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 14.1615 ms | not_measured | managed | 3.046x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.00206 ms | not_measured | managed | 1.094x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.1591 ms | not_measured | managed | 2.117x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 14.8183 ms | not_measured | managed | 2.952x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.00206 ms | not_measured | managed | 1.079x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1606 ms | not_measured | managed | 2.088x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 16.5320 ms | not_measured | managed | 2.807x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.00188 ms | not_measured | managed | 0.855x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1621 ms | not_measured | managed | 0.221x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 20.5836 ms | not_measured | managed | 1.259x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.00218 ms | not_measured | managed | 1.090x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.1576 ms | not_measured | managed | 2.189x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 14.1124 ms | not_measured | managed | 3.041x |
| `np.max (complex128)` | complex128 | 1,000 | 0.00217 ms | not_measured | managed | 1.534x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1433 ms | not_measured | managed | 1.098x |
| `np.max (complex128)` | complex128 | 10,000,000 | 18.8329 ms | not_measured | managed | 1.130x |
| `np.max (float16)` | float16 | 1,000 | 0.00215 ms | not_measured | managed | 1.702x |
| `np.max (float16)` | float16 | 100,000 | 0.4379 ms | not_measured | managed | 1.140x |
| `np.max (float16)` | float16 | 10,000,000 | 46.6044 ms | not_measured | managed | 1.373x |
| `np.max (float32)` | float32 | 1,000 | 0.000654 ms | not_measured | managed | 2.440x |
| `np.max (float32)` | float32 | 100,000 | 0.00864 ms | not_measured | managed | 0.728x |
| `np.max (float32)` | float32 | 10,000,000 | 3.1076 ms | not_measured | managed | 1.146x |
| `np.max (float64)` | float64 | 1,000 | 0.000702 ms | not_measured | managed | 2.553x |
| `np.max (float64)` | float64 | 100,000 | 0.01683 ms | not_measured | managed | 0.645x |
| `np.max (float64)` | float64 | 10,000,000 | 7.6035 ms | not_measured | managed | 1.040x |
| `np.max (int16)` | int16 | 1,000 | 0.000614 ms | not_measured | managed | 2.601x |
| `np.max (int16)` | int16 | 100,000 | 0.00255 ms | not_measured | managed | 1.223x |
| `np.max (int16)` | int16 | 10,000,000 | 0.5064 ms | not_measured | managed | 1.105x |
| `np.max (int32)` | int32 | 1,000 | 0.000615 ms | not_measured | managed | 2.480x |
| `np.max (int32)` | int32 | 100,000 | 0.00630 ms | not_measured | managed | 0.659x |
| `np.max (int32)` | int32 | 10,000,000 | 2.4901 ms | not_measured | managed | 1.231x |
| `np.max (int64)` | int64 | 1,000 | 0.000877 ms | not_measured | managed | 1.860x |
| `np.max (int64)` | int64 | 100,000 | 0.02685 ms | not_measured | managed | 0.407x |
| `np.max (int64)` | int64 | 10,000,000 | 7.6577 ms | not_measured | managed | 1.113x |
| `np.max (int8)` | int8 | 1,000 | 0.000613 ms | not_measured | managed | 2.458x |
| `np.max (int8)` | int8 | 100,000 | 0.00146 ms | not_measured | managed | 1.577x |
| `np.max (int8)` | int8 | 10,000,000 | 0.2282 ms | not_measured | managed | 1.148x |
| `np.max (uint16)` | uint16 | 1,000 | 0.000610 ms | not_measured | managed | 2.479x |
| `np.max (uint16)` | uint16 | 100,000 | 0.00253 ms | not_measured | managed | 1.197x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.5056 ms | not_measured | managed | 1.219x |
| `np.max (uint32)` | uint32 | 1,000 | 0.000626 ms | not_measured | managed | 2.546x |
| `np.max (uint32)` | uint32 | 100,000 | 0.00454 ms | not_measured | managed | 1.119x |
| `np.max (uint32)` | uint32 | 10,000,000 | 2.5732 ms | not_measured | managed | 1.190x |
| `np.max (uint64)` | uint64 | 1,000 | 0.000973 ms | not_measured | managed | 1.711x |
| `np.max (uint64)` | uint64 | 100,000 | 0.03733 ms | not_measured | managed | 0.338x |
| `np.max (uint64)` | uint64 | 10,000,000 | 7.9786 ms | not_measured | managed | 1.093x |
| `np.max (uint8)` | uint8 | 1,000 | 0.000613 ms | not_measured | managed | 2.546x |
| `np.max (uint8)` | uint8 | 100,000 | 0.00149 ms | not_measured | managed | 1.670x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.2275 ms | not_measured | managed | 1.151x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.00105 ms | not_measured | managed | 2.649x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.02154 ms | not_measured | managed | 1.815x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 15.6555 ms | not_measured | managed | 1.072x |
| `np.mean (float16)` | float16 | 1,000 | 0.00201 ms | not_measured | managed | 2.473x |
| `np.mean (float16)` | float16 | 100,000 | 0.1650 ms | not_measured | managed | 0.624x |
| `np.mean (float16)` | float16 | 10,000,000 | 16.5638 ms | not_measured | managed | 1.004x |
| `np.mean (float32)` | float32 | 1,000 | 0.000670 ms | not_measured | managed | 5.203x |
| `np.mean (float32)` | float32 | 100,000 | 0.00521 ms | not_measured | managed | 3.566x |
| `np.mean (float32)` | float32 | 10,000,000 | 2.5973 ms | not_measured | managed | 3.236x |
| `np.mean (float64)` | float64 | 1,000 | 0.000685 ms | not_measured | managed | 3.368x |
| `np.mean (float64)` | float64 | 100,000 | 0.00779 ms | not_measured | managed | 2.655x |
| `np.mean (float64)` | float64 | 10,000,000 | 7.3388 ms | not_measured | managed | 1.962x |
| `np.mean (int16)` | int16 | 1,000 | 0.000788 ms | not_measured | managed | 3.628x |
| `np.mean (int16)` | int16 | 100,000 | 0.04483 ms | not_measured | managed | 1.208x |
| `np.mean (int16)` | int16 | 10,000,000 | 4.4534 ms | not_measured | managed | 1.809x |
| `np.mean (int32)` | int32 | 1,000 | 0.000797 ms | not_measured | managed | 3.565x |
| `np.mean (int32)` | int32 | 100,000 | 0.04481 ms | not_measured | managed | 0.956x |
| `np.mean (int32)` | int32 | 10,000,000 | 5.1126 ms | not_measured | managed | 1.588x |
| `np.mean (int64)` | int64 | 1,000 | 0.000676 ms | not_measured | managed | 4.368x |
| `np.mean (int64)` | int64 | 100,000 | 0.00792 ms | not_measured | managed | 4.851x |
| `np.mean (int64)` | int64 | 10,000,000 | 7.3425 ms | not_measured | managed | 1.518x |
| `np.mean (int8)` | int8 | 1,000 | 0.000787 ms | not_measured | managed | 3.712x |
| `np.mean (int8)` | int8 | 100,000 | 0.04402 ms | not_measured | managed | 1.209x |
| `np.mean (int8)` | int8 | 10,000,000 | 4.7158 ms | not_measured | managed | 1.734x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.000963 ms | not_measured | managed | 3.012x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.03944 ms | not_measured | managed | 1.386x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 4.2530 ms | not_measured | managed | 1.877x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.000981 ms | not_measured | managed | 2.844x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.03954 ms | not_measured | managed | 1.116x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 4.8809 ms | not_measured | managed | 1.670x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.000662 ms | not_measured | managed | 4.260x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.00824 ms | not_measured | managed | 7.577x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 7.3985 ms | not_measured | managed | 1.607x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.000755 ms | not_measured | managed | 3.783x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.04092 ms | not_measured | managed | 1.291x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 3.7054 ms | not_measured | managed | 2.146x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.00378 ms | not_measured | managed | 0.802x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.04428 ms | not_measured | managed | 0.407x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 16.2075 ms | not_measured | managed | 1.077x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.00690 ms | not_measured | managed | 0.911x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.2837 ms | not_measured | managed | 0.295x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 27.2533 ms | not_measured | managed | 0.581x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.00306 ms | not_measured | managed | 1.173x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.01776 ms | not_measured | managed | 0.553x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 3.4474 ms | not_measured | managed | 1.196x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.00335 ms | not_measured | managed | 0.896x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.02701 ms | not_measured | managed | 0.453x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 8.2174 ms | not_measured | managed | 1.019x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.000886 ms | not_measured | managed | 3.857x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.02000 ms | not_measured | managed | 2.539x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 1.9052 ms | not_measured | managed | 4.130x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.000875 ms | not_measured | managed | 3.774x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.01900 ms | not_measured | managed | 2.701x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 3.4531 ms | not_measured | managed | 2.317x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.00203 ms | not_measured | managed | 1.706x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.1449 ms | not_measured | managed | 0.241x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 111.4093 ms | not_measured | managed | 0.104x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.000681 ms | not_measured | managed | 5.389x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.01895 ms | not_measured | managed | 2.656x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 1.8184 ms | not_measured | managed | 4.461x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.000878 ms | not_measured | managed | 4.023x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.01897 ms | not_measured | managed | 2.948x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 1.9027 ms | not_measured | managed | 4.145x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.000752 ms | not_measured | managed | 5.186x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.02485 ms | not_measured | managed | 1.682x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 3.6673 ms | not_measured | managed | 2.093x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.00217 ms | not_measured | managed | 1.572x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.1582 ms | not_measured | managed | 0.343x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 116.8058 ms | not_measured | managed | 0.101x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.000881 ms | not_measured | managed | 3.961x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.01896 ms | not_measured | managed | 2.700x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 1.8148 ms | not_measured | managed | 4.414x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.00379 ms | not_measured | managed | 0.879x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.05163 ms | not_measured | managed | 0.630x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 17.2697 ms | not_measured | managed | 1.036x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.00693 ms | not_measured | managed | 0.763x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.2717 ms | not_measured | managed | 0.329x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 26.0420 ms | not_measured | managed | 0.641x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.00322 ms | not_measured | managed | 1.056x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.02328 ms | not_measured | managed | 0.853x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 3.3827 ms | not_measured | managed | 2.323x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.00345 ms | not_measured | managed | 0.823x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.02662 ms | not_measured | managed | 0.775x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 8.4147 ms | not_measured | managed | 1.440x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.000729 ms | not_measured | managed | 4.735x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.01956 ms | not_measured | managed | 2.943x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 1.8752 ms | not_measured | managed | 4.394x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.000735 ms | not_measured | managed | 4.435x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.01957 ms | not_measured | managed | 2.354x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 3.3879 ms | not_measured | managed | 2.477x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.00201 ms | not_measured | managed | 1.722x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.1443 ms | not_measured | managed | 0.321x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 14.6062 ms | not_measured | managed | 0.794x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.000715 ms | not_measured | managed | 4.961x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.01952 ms | not_measured | managed | 2.954x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 1.8002 ms | not_measured | managed | 4.541x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.000730 ms | not_measured | managed | 4.659x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.01954 ms | not_measured | managed | 3.654x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 1.8679 ms | not_measured | managed | 4.367x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.000792 ms | not_measured | managed | 4.208x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.02545 ms | not_measured | managed | 1.893x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 3.6609 ms | not_measured | managed | 2.231x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.00229 ms | not_measured | managed | 1.478x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.1557 ms | not_measured | managed | 0.354x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 15.7836 ms | not_measured | managed | 0.763x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.000724 ms | not_measured | managed | 4.695x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.01955 ms | not_measured | managed | 3.097x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 1.8003 ms | not_measured | managed | 4.530x |
| `np.min (complex128)` | complex128 | 1,000 | 0.00215 ms | not_measured | managed | 1.444x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1433 ms | not_measured | managed | 1.175x |
| `np.min (complex128)` | complex128 | 10,000,000 | 18.8279 ms | not_measured | managed | 1.090x |
| `np.min (float16)` | float16 | 1,000 | 0.00181 ms | not_measured | managed | 2.080x |
| `np.min (float16)` | float16 | 100,000 | 0.4043 ms | not_measured | managed | 1.255x |
| `np.min (float16)` | float16 | 10,000,000 | 41.8187 ms | not_measured | managed | 1.496x |
| `np.min (float32)` | float32 | 1,000 | 0.000665 ms | not_measured | managed | 2.380x |
| `np.min (float32)` | float32 | 100,000 | 0.00893 ms | not_measured | managed | 0.704x |
| `np.min (float32)` | float32 | 10,000,000 | 3.1617 ms | not_measured | managed | 1.113x |
| `np.min (float64)` | float64 | 1,000 | 0.000725 ms | not_measured | managed | 2.214x |
| `np.min (float64)` | float64 | 100,000 | 0.01761 ms | not_measured | managed | 0.624x |
| `np.min (float64)` | float64 | 10,000,000 | 8.2083 ms | not_measured | managed | 0.961x |
| `np.min (int16)` | int16 | 1,000 | 0.000616 ms | not_measured | managed | 2.481x |
| `np.min (int16)` | int16 | 100,000 | 0.00257 ms | not_measured | managed | 1.285x |
| `np.min (int16)` | int16 | 10,000,000 | 0.5082 ms | not_measured | managed | 1.164x |
| `np.min (int32)` | int32 | 1,000 | 0.000799 ms | not_measured | managed | 2.018x |
| `np.min (int32)` | int32 | 100,000 | 0.00418 ms | not_measured | managed | 1.143x |
| `np.min (int32)` | int32 | 10,000,000 | 2.5735 ms | not_measured | managed | 1.207x |
| `np.min (int64)` | int64 | 1,000 | 0.000879 ms | not_measured | managed | 1.818x |
| `np.min (int64)` | int64 | 100,000 | 0.02727 ms | not_measured | managed | 0.746x |
| `np.min (int64)` | int64 | 10,000,000 | 8.2103 ms | not_measured | managed | 1.206x |
| `np.min (int8)` | int8 | 1,000 | 0.000622 ms | not_measured | managed | 2.442x |
| `np.min (int8)` | int8 | 100,000 | 0.00147 ms | not_measured | managed | 1.739x |
| `np.min (int8)` | int8 | 10,000,000 | 0.2280 ms | not_measured | managed | 1.110x |
| `np.min (uint16)` | uint16 | 1,000 | 0.000626 ms | not_measured | managed | 2.431x |
| `np.min (uint16)` | uint16 | 100,000 | 0.00263 ms | not_measured | managed | 1.147x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.5161 ms | not_measured | managed | 1.147x |
| `np.min (uint32)` | uint32 | 1,000 | 0.000626 ms | not_measured | managed | 2.439x |
| `np.min (uint32)` | uint32 | 100,000 | 0.00462 ms | not_measured | managed | 0.921x |
| `np.min (uint32)` | uint32 | 10,000,000 | 2.5991 ms | not_measured | managed | 1.170x |
| `np.min (uint64)` | uint64 | 1,000 | 0.000976 ms | not_measured | managed | 1.700x |
| `np.min (uint64)` | uint64 | 100,000 | 0.03676 ms | not_measured | managed | 0.348x |
| `np.min (uint64)` | uint64 | 10,000,000 | 8.0209 ms | not_measured | managed | 1.059x |
| `np.min (uint8)` | uint8 | 1,000 | 0.000606 ms | not_measured | managed | 2.517x |
| `np.min (uint8)` | uint8 | 100,000 | 0.00149 ms | not_measured | managed | 1.528x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.2283 ms | not_measured | managed | 1.163x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.00502 ms | not_measured | managed | 1.720x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.2733 ms | not_measured | managed | 2.758x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 27.3994 ms | not_measured | managed | 2.859x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.00313 ms | not_measured | managed | 1.977x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.1110 ms | not_measured | managed | 0.691x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 12.7969 ms | not_measured | managed | 1.652x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.00317 ms | not_measured | managed | 1.921x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.1130 ms | not_measured | managed | 2.167x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 18.6593 ms | not_measured | managed | 2.183x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.00502 ms | not_measured | managed | 1.806x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.2735 ms | not_measured | managed | 2.691x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 27.4033 ms | not_measured | managed | 2.719x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.00315 ms | not_measured | managed | 1.937x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.1114 ms | not_measured | managed | 0.884x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 12.7675 ms | not_measured | managed | 1.656x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.00325 ms | not_measured | managed | 1.878x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.1141 ms | not_measured | managed | 1.497x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 19.0238 ms | not_measured | managed | 2.156x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.00181 ms | not_measured | managed | 2.849x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.4241 ms | not_measured | managed | 1.182x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 43.7152 ms | not_measured | managed | 1.426x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.000836 ms | not_measured | managed | 3.476x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.05074 ms | not_measured | managed | 0.593x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 5.5659 ms | not_measured | managed | 0.703x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.00133 ms | not_measured | managed | 2.300x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.1021 ms | not_measured | managed | 0.973x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 11.3877 ms | not_measured | managed | 0.794x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.00263 ms | not_measured | managed | 4.864x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.2017 ms | not_measured | managed | 1.755x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 20.3824 ms | not_measured | managed | 2.618x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.00135 ms | not_measured | managed | 7.402x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.07142 ms | not_measured | managed | 2.397x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 7.3096 ms | not_measured | managed | 4.943x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.00130 ms | not_measured | managed | 6.453x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.07149 ms | not_measured | managed | 3.705x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 9.1967 ms | not_measured | managed | 5.925x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.01221 ms | not_measured | managed | 1.511x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.6896 ms | not_measured | managed | 0.767x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 125.9341 ms | not_measured | managed | 1.237x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.00463 ms | not_measured | managed | 2.969x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 0.3481 ms | not_measured | managed | 2.012x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 49.9440 ms | not_measured | managed | 2.076x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.00478 ms | not_measured | managed | 2.530x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.3585 ms | not_measured | managed | 2.146x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 61.0615 ms | not_measured | managed | 2.096x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.00186 ms | not_measured | managed | 2.767x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.4261 ms | not_measured | managed | 1.174x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 44.0127 ms | not_measured | managed | 1.383x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.000830 ms | not_measured | managed | 3.681x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.05074 ms | not_measured | managed | 0.592x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 5.5190 ms | not_measured | managed | 0.688x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.00134 ms | not_measured | managed | 2.241x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.1024 ms | not_measured | managed | 0.599x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 11.3876 ms | not_measured | managed | 0.657x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.01212 ms | not_measured | managed | 2.581x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.6956 ms | not_measured | managed | 1.596x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 126.0054 ms | not_measured | managed | 1.345x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.00462 ms | not_measured | managed | 5.905x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.3478 ms | not_measured | managed | 2.901x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 49.9313 ms | not_measured | managed | 1.437x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.00478 ms | not_measured | managed | 5.407x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.3599 ms | not_measured | managed | 3.553x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 61.0070 ms | not_measured | managed | 1.558x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.00222 ms | not_measured | managed | 10.200x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.2094 ms | not_measured | managed | 1.038x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 18.7996 ms | not_measured | managed | 2.139x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.000795 ms | not_measured | managed | 26.904x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 1.9537 ms | not_measured | managed | 3.642x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 200.4294 ms | not_measured | managed | 3.349x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.000895 ms | not_measured | managed | 4.731x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 3.0753 ms | not_measured | managed | 2.449x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 338.2646 ms | not_measured | managed | 2.159x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.01214 ms | not_measured | managed | 2.564x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6937 ms | not_measured | managed | 1.562x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 125.9003 ms | not_measured | managed | 1.358x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00463 ms | not_measured | managed | 5.802x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.3478 ms | not_measured | managed | 2.789x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 49.8835 ms | not_measured | managed | 1.433x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00473 ms | not_measured | managed | 5.251x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3606 ms | not_measured | managed | 3.024x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 60.8848 ms | not_measured | managed | 1.555x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.00494 ms | not_measured | managed | 6.831x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.4046 ms | not_measured | managed | 3.406x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 40.3707 ms | not_measured | managed | 6.296x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.00242 ms | not_measured | managed | 8.245x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.1482 ms | not_measured | managed | 3.310x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 15.1705 ms | not_measured | managed | 4.111x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.00228 ms | not_measured | managed | 8.278x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.1428 ms | not_measured | managed | 3.759x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 18.5441 ms | not_measured | managed | 5.188x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.00222 ms | not_measured | managed | 2.785x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.1889 ms | not_measured | managed | 1.523x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 18.7324 ms | not_measured | managed | 3.939x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.000643 ms | not_measured | managed | 6.184x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.00931 ms | not_measured | managed | 6.606x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 3.2420 ms | not_measured | managed | 7.744x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.000724 ms | not_measured | managed | 5.351x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.01815 ms | not_measured | managed | 5.656x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 8.5320 ms | not_measured | managed | 5.061x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.00502 ms | not_measured | managed | 6.060x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.4041 ms | not_measured | managed | 2.889x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 40.3647 ms | not_measured | managed | 5.726x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.00242 ms | not_measured | managed | 7.930x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.1482 ms | not_measured | managed | 2.191x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 15.1836 ms | not_measured | managed | 4.157x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.00229 ms | not_measured | managed | 8.467x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.1427 ms | not_measured | managed | 2.707x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 18.5229 ms | not_measured | managed | 5.271x |
| `np.prod (float64)` | float64 | 1,000 | 0.000655 ms | not_measured | managed | 3.347x |
| `np.prod (float64)` | float64 | 100,000 | 1.1157 ms | not_measured | managed | 6.209x |
| `np.prod (float64)` | float64 | 10,000,000 | 410.2949 ms | not_measured | managed | 1.710x |
| `np.prod (int64)` | int64 | 1,000 | 0.000974 ms | not_measured | managed | 2.069x |
| `np.prod (int64)` | int64 | 100,000 | 0.03804 ms | not_measured | managed | 3.446x |
| `np.prod (int64)` | int64 | 10,000,000 | 8.1000 ms | not_measured | managed | 1.589x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.000867 ms | not_measured | managed | 2.113x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.01904 ms | not_measured | managed | 1.215x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 120.9367 ms | not_measured | managed | 0.904x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.00100 ms | not_measured | managed | 1.884x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.04049 ms | not_measured | managed | 1.468x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 8.0962 ms | not_measured | managed | 1.124x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.000727 ms | not_measured | managed | 2.508x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.01208 ms | not_measured | managed | 10.626x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 7.9920 ms | not_measured | managed | 23.969x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.00143 ms | not_measured | managed | 1.279x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.04400 ms | not_measured | managed | 2.838x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 7.9943 ms | not_measured | managed | 1.626x |
| `np.std (float16)` | float16 | 1,000 | 0.00421 ms | not_measured | managed | 4.627x |
| `np.std (float16)` | float16 | 100,000 | 0.3822 ms | not_measured | managed | 2.323x |
| `np.std (float16)` | float16 | 10,000,000 | 38.3851 ms | not_measured | managed | 4.251x |
| `np.std (float32)` | float32 | 1,000 | 0.000816 ms | not_measured | managed | 10.145x |
| `np.std (float32)` | float32 | 100,000 | 0.01823 ms | not_measured | managed | 2.696x |
| `np.std (float32)` | float32 | 10,000,000 | 5.9856 ms | not_measured | managed | 5.068x |
| `np.std (float64)` | float64 | 1,000 | 0.000955 ms | not_measured | managed | 7.290x |
| `np.std (float64)` | float64 | 100,000 | 0.03545 ms | not_measured | managed | 1.825x |
| `np.std (float64)` | float64 | 10,000,000 | 16.0914 ms | not_measured | managed | 3.584x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.01085 ms | not_measured | managed | 1.965x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.8085 ms | not_measured | managed | 1.553x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 187.5538 ms | not_measured | managed | 1.349x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.00283 ms | not_measured | managed | 2.998x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.06258 ms | not_measured | managed | 0.652x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 8.0214 ms | not_measured | managed | 3.110x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.00110 ms | not_measured | managed | 6.570x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.03800 ms | not_measured | managed | 1.718x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 15.4400 ms | not_measured | managed | 3.544x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.000889 ms | not_measured | managed | 1.974x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.02133 ms | not_measured | managed | 1.949x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 16.0544 ms | not_measured | managed | 1.074x |
| `np.sum (float16)` | float16 | 1,000 | 0.000983 ms | not_measured | managed | 3.579x |
| `np.sum (float16)` | float16 | 100,000 | 0.06728 ms | not_measured | managed | 3.013x |
| `np.sum (float16)` | float16 | 10,000,000 | 6.9402 ms | not_measured | managed | 3.447x |
| `np.sum (float32)` | float32 | 1,000 | 0.000619 ms | not_measured | managed | 2.603x |
| `np.sum (float32)` | float32 | 100,000 | 0.00568 ms | not_measured | managed | 2.934x |
| `np.sum (float32)` | float32 | 10,000,000 | 2.6749 ms | not_measured | managed | 2.847x |
| `np.sum (float64)` | float64 | 1,000 | 0.00486 ms | not_measured | managed | 0.334x |
| `np.sum (float64)` | float64 | 100,000 | 0.1884 ms | not_measured | managed | 0.094x |
| `np.sum (float64)` | float64 | 10,000,000 | 10.5938 ms | not_measured | managed | 1.384x |
| `np.sum (int16)` | int16 | 1,000 | 0.000792 ms | not_measured | managed | 2.477x |
| `np.sum (int16)` | int16 | 100,000 | 0.04480 ms | not_measured | managed | 0.758x |
| `np.sum (int16)` | int16 | 10,000,000 | 4.4479 ms | not_measured | managed | 1.455x |
| `np.sum (int32)` | int32 | 1,000 | 0.000799 ms | not_measured | managed | 2.742x |
| `np.sum (int32)` | int32 | 100,000 | 0.04467 ms | not_measured | managed | 0.849x |
| `np.sum (int32)` | int32 | 10,000,000 | 5.1193 ms | not_measured | managed | 1.679x |
| `np.sum (int64)` | int64 | 1,000 | 0.000658 ms | not_measured | managed | 2.579x |
| `np.sum (int64)` | int64 | 100,000 | 0.00778 ms | not_measured | managed | 2.180x |
| `np.sum (int64)` | int64 | 10,000,000 | 7.3627 ms | not_measured | managed | 1.267x |
| `np.sum (int8)` | int8 | 1,000 | 0.000796 ms | not_measured | managed | 2.431x |
| `np.sum (int8)` | int8 | 100,000 | 0.04507 ms | not_measured | managed | 0.746x |
| `np.sum (int8)` | int8 | 10,000,000 | 4.7192 ms | not_measured | managed | 1.692x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.000962 ms | not_measured | managed | 1.994x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.03947 ms | not_measured | managed | 0.839x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 4.2528 ms | not_measured | managed | 1.884x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.000962 ms | not_measured | managed | 2.024x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.03935 ms | not_measured | managed | 0.876x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 4.9146 ms | not_measured | managed | 1.517x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.000659 ms | not_measured | managed | 2.863x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.00778 ms | not_measured | managed | 2.080x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 7.4806 ms | not_measured | managed | 1.238x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.000983 ms | not_measured | managed | 2.297x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.04098 ms | not_measured | managed | 0.836x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 3.7492 ms | not_measured | managed | 2.099x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.00278 ms | not_measured | managed | 0.735x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.03882 ms | not_measured | managed | 0.517x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 16.3261 ms | not_measured | managed | 1.029x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.00505 ms | not_measured | managed | 0.959x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1539 ms | not_measured | managed | 1.894x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 13.6878 ms | not_measured | managed | 4.796x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.00213 ms | not_measured | managed | 0.864x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.01477 ms | not_measured | managed | 0.525x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 3.2653 ms | not_measured | managed | 1.074x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.00235 ms | not_measured | managed | 0.791x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.02293 ms | not_measured | managed | 0.677x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 8.1732 ms | not_measured | managed | 1.031x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.00124 ms | not_measured | managed | 1.855x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.00946 ms | not_measured | managed | 5.209x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 0.8653 ms | not_measured | managed | 10.132x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.000852 ms | not_measured | managed | 2.719x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.01419 ms | not_measured | managed | 4.083x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 3.3979 ms | not_measured | managed | 3.181x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.000857 ms | not_measured | managed | 2.253x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.01915 ms | not_measured | managed | 1.772x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 7.9070 ms | not_measured | managed | 1.093x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.00125 ms | not_measured | managed | 1.864x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.00938 ms | not_measured | managed | 5.143x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.7451 ms | not_measured | managed | 13.409x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.00125 ms | not_measured | managed | 1.867x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.00948 ms | not_measured | managed | 5.083x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 0.8641 ms | not_measured | managed | 11.773x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.000853 ms | not_measured | managed | 2.646x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.01417 ms | not_measured | managed | 3.444x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 3.4276 ms | not_measured | managed | 2.839x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.000877 ms | not_measured | managed | 2.195x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.01894 ms | not_measured | managed | 1.685x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 7.9564 ms | not_measured | managed | 1.104x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.00128 ms | not_measured | managed | 1.941x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.00936 ms | not_measured | managed | 5.081x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.7491 ms | not_measured | managed | 13.358x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.00279 ms | not_measured | managed | 0.729x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.04659 ms | not_measured | managed | 0.731x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 17.2534 ms | not_measured | managed | 0.991x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.00490 ms | not_measured | managed | 0.768x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.08169 ms | not_measured | managed | 2.517x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 6.8748 ms | not_measured | managed | 3.659x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.00233 ms | not_measured | managed | 0.804x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.01997 ms | not_measured | managed | 0.876x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 3.3652 ms | not_measured | managed | 2.299x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.00238 ms | not_measured | managed | 0.767x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.02247 ms | not_measured | managed | 1.036x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 8.3494 ms | not_measured | managed | 1.451x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.000797 ms | not_measured | managed | 2.917x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.00676 ms | not_measured | managed | 5.544x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 0.7035 ms | not_measured | managed | 9.385x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.000797 ms | not_measured | managed | 2.949x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.00958 ms | not_measured | managed | 4.621x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 3.4102 ms | not_measured | managed | 2.621x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.000734 ms | not_measured | managed | 2.530x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.01149 ms | not_measured | managed | 1.718x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 8.0642 ms | not_measured | managed | 1.163x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.000798 ms | not_measured | managed | 2.783x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.00671 ms | not_measured | managed | 5.805x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.4941 ms | not_measured | managed | 15.987x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.000779 ms | not_measured | managed | 2.800x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.00676 ms | not_measured | managed | 5.625x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 0.7042 ms | not_measured | managed | 11.418x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.000799 ms | not_measured | managed | 3.068x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.00962 ms | not_measured | managed | 3.873x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 3.4130 ms | not_measured | managed | 2.231x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.000740 ms | not_measured | managed | 2.505x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.01158 ms | not_measured | managed | 1.515x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 8.1678 ms | not_measured | managed | 1.002x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.000782 ms | not_measured | managed | 3.084x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.00670 ms | not_measured | managed | 5.530x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.4846 ms | not_measured | managed | 14.803x |
| `np.var (float16)` | float16 | 1,000 | 0.00416 ms | not_measured | managed | 4.571x |
| `np.var (float16)` | float16 | 100,000 | 0.3821 ms | not_measured | managed | 2.412x |
| `np.var (float16)` | float16 | 10,000,000 | 38.3820 ms | not_measured | managed | 4.211x |
| `np.var (float32)` | float32 | 1,000 | 0.000826 ms | not_measured | managed | 9.287x |
| `np.var (float32)` | float32 | 100,000 | 0.01833 ms | not_measured | managed | 2.653x |
| `np.var (float32)` | float32 | 10,000,000 | 5.9971 ms | not_measured | managed | 5.067x |
| `np.var (float64)` | float64 | 1,000 | 0.000957 ms | not_measured | managed | 6.408x |
| `np.var (float64)` | float64 | 100,000 | 0.03588 ms | not_measured | managed | 1.885x |
| `np.var (float64)` | float64 | 10,000,000 | 16.4890 ms | not_measured | managed | 3.528x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.01106 ms | not_measured | managed | 1.739x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.8082 ms | not_measured | managed | 1.442x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 187.8369 ms | not_measured | managed | 1.326x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.00288 ms | not_measured | managed | 2.830x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.06185 ms | not_measured | managed | 0.659x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 8.0176 ms | not_measured | managed | 3.193x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.00102 ms | not_measured | managed | 7.548x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.03737 ms | not_measured | managed | 1.728x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 15.6539 ms | not_measured | managed | 3.328x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.00551 ms | not_measured | managed | 1.144x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.2950 ms | not_measured | managed | 1.997x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.00209 ms | not_measured | managed | 0.919x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.1216 ms | not_measured | managed | 0.680x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.000964 ms | not_measured | managed | 0.419x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.00144 ms | not_measured | managed | 0.359x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.000889 ms | not_measured | managed | 4.871x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.00154 ms | not_measured | managed | 2.764x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.00215 ms | not_measured | managed | 1.356x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.1228 ms | not_measured | managed | 0.681x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.00385 ms | not_measured | managed | 0.649x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.4172 ms | not_measured | managed | 0.738x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.01145 ms | not_measured | managed | 0.777x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.6101 ms | not_measured | managed | 0.975x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.000731 ms | not_measured | managed | 1.878x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.3111 ms | not_measured | managed | 1.018x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.00747 ms | not_measured | managed | 0.252x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.1517 ms | not_measured | managed | 0.714x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.00281 ms | not_measured | managed | 0.529x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.1863 ms | not_measured | managed | 0.390x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.00368 ms | not_measured | managed | 3.140x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.1466 ms | not_measured | managed | 4.756x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.00234 ms | not_measured | managed | 0.634x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.1367 ms | not_measured | managed | 0.394x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.00172 ms | not_measured | managed | 1.045x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.08462 ms | not_measured | managed | 0.328x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.00392 ms | not_measured | managed | 2.662x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.1127 ms | not_measured | managed | 0.641x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.00390 ms | not_measured | managed | 2.702x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.1122 ms | not_measured | managed | 0.620x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.00186 ms | not_measured | managed | 0.962x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1436 ms | not_measured | managed | 0.649x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.00180 ms | not_measured | managed | 0.440x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.1136 ms | not_measured | managed | 0.270x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 9.0380 ms | not_measured | managed | 0.830x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.00228 ms | not_measured | managed | 0.557x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.1426 ms | not_measured | managed | 0.297x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 28.3354 ms | not_measured | managed | 0.613x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.00173 ms | not_measured | managed | 0.059x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.00184 ms | not_measured | managed | 0.055x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.00175 ms | not_measured | managed | 0.071x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.00280 ms | not_measured | managed | 0.056x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.00272 ms | not_measured | managed | 0.058x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.00272 ms | not_measured | managed | 0.058x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.00267 ms | not_measured | managed | 0.058x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.00261 ms | not_measured | managed | 0.060x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.00257 ms | not_measured | managed | 0.060x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.00164 ms | not_measured | managed | 0.061x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.00159 ms | not_measured | managed | 0.063x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.00159 ms | not_measured | managed | 0.062x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.00160 ms | not_measured | managed | 0.064x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.00162 ms | not_measured | managed | 0.060x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.00159 ms | not_measured | managed | 0.062x |
| `a[::2] * 2` | float64 | 1,000 | 0.01029 ms | not_measured | managed | 0.086x |
| `a[::2] * 2` | float64 | 100,000 | 0.2365 ms | not_measured | managed | 0.062x |
| `a[::2] * 2` | float64 | 10,000,000 | 19.2500 ms | not_measured | managed | 0.490x |
| `a[N/10:9N/10] * 2` | float64 | 1,000 | 0.00652 ms | not_measured | managed | 0.140x |
| `a[N/10:9N/10] * 2` | float64 | 100,000 | 0.1044 ms | not_measured | managed | 0.100x |
| `a[N/10:9N/10] * 2` | float64 | 10,000,000 | 12.7705 ms | not_measured | managed | 0.997x |
| `a[N/10:9N/10].copy()` | float64 | 1,000 | 0.00397 ms | not_measured | managed | 0.099x |
| `a[N/10:9N/10].copy()` | float64 | 100,000 | 0.1558 ms | not_measured | managed | 0.058x |
| `a[N/10:9N/10].copy()` | float64 | 10,000,000 | 8.1949 ms | not_measured | managed | 1.299x |
| `np.copy(a[N/10:9N/10])` | float64 | 1,000 | 0.00605 ms | not_measured | managed | 0.099x |
| `np.copy(a[N/10:9N/10])` | float64 | 100,000 | 0.1245 ms | not_measured | managed | 0.075x |
| `np.copy(a[N/10:9N/10])` | float64 | 10,000,000 | 8.3476 ms | not_measured | managed | 1.517x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.01214 ms | not_measured | managed | 0.357x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 0.9884 ms | not_measured | managed | 0.144x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.01153 ms | not_measured | managed | 0.346x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.7325 ms | not_measured | managed | 0.472x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.01150 ms | not_measured | managed | 0.266x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 1.0548 ms | not_measured | managed | 0.113x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.01130 ms | not_measured | managed | 0.297x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 0.6635 ms | not_measured | managed | 0.216x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.02809 ms | not_measured | managed | 0.414x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.7645 ms | not_measured | managed | 0.903x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 235.4466 ms | not_measured | managed | 5.224x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.04133 ms | not_measured | managed | 0.268x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 3.2694 ms | not_measured | managed | 1.437x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 503.7280 ms | not_measured | managed | 3.195x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.02994 ms | not_measured | managed | 0.444x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 1.4629 ms | not_measured | managed | 0.293x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 205.1319 ms | not_measured | managed | 1.384x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.03429 ms | not_measured | managed | 0.382x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 2.7244 ms | not_measured | managed | 0.179x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 426.7371 ms | not_measured | managed | 1.045x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.00235 ms | not_measured | managed | 1.925x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.1491 ms | not_measured | managed | 2.833x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.00241 ms | not_measured | managed | 1.740x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.1325 ms | not_measured | managed | 3.048x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.00376 ms | not_measured | managed | 0.812x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.1789 ms | not_measured | managed | 2.131x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.00293 ms | not_measured | managed | 1.172x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.1764 ms | not_measured | managed | 2.170x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.00268 ms | not_measured | managed | 1.268x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.2173 ms | not_measured | managed | 0.878x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.00289 ms | not_measured | managed | 1.091x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.1835 ms | not_measured | managed | 1.693x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.00397 ms | not_measured | managed | 0.507x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.2892 ms | not_measured | managed | 0.351x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.00314 ms | not_measured | managed | 0.727x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.2930 ms | not_measured | managed | 0.370x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.04367 ms | not_measured | managed | 1.060x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 3.4485 ms | not_measured | managed | 4.284x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.06634 ms | not_measured | managed | 0.678x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 6.9975 ms | not_measured | managed | 2.599x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.04963 ms | not_measured | managed | 0.813x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 3.1237 ms | not_measured | managed | 3.334x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.06992 ms | not_measured | managed | 0.633x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 5.1599 ms | not_measured | managed | 2.195x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.00221 ms | not_measured | managed | 1.163x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.1152 ms | not_measured | managed | 1.604x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 21.5097 ms | not_measured | managed | 1.975x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.00266 ms | not_measured | managed | 0.983x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.1178 ms | not_measured | managed | 2.568x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 30.6533 ms | not_measured | managed | 1.358x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.00288 ms | not_measured | managed | 0.573x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.1121 ms | not_measured | managed | 0.916x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 22.0003 ms | not_measured | managed | 1.354x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.00233 ms | not_measured | managed | 0.707x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.1167 ms | not_measured | managed | 1.017x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 30.9483 ms | not_measured | managed | 1.229x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.01007 ms | not_measured | managed | 0.222x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.6975 ms | not_measured | managed | 0.149x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.01165 ms | not_measured | managed | 0.243x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.6264 ms | not_measured | managed | 0.497x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.00779 ms | not_measured | managed | 0.195x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.5090 ms | not_measured | managed | 0.055x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.00904 ms | not_measured | managed | 0.224x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.6101 ms | not_measured | managed | 0.137x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.01198 ms | not_measured | managed | 0.651x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 2.8584 ms | not_measured | managed | 0.713x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 330.0203 ms | not_measured | managed | 0.905x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.01208 ms | not_measured | managed | 0.607x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 2.8704 ms | not_measured | managed | 0.922x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 330.3494 ms | not_measured | managed | 0.904x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.00880 ms | not_measured | managed | 2.151x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 2.8538 ms | not_measured | managed | 1.011x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 315.8572 ms | not_measured | managed | 1.791x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.00954 ms | not_measured | managed | 1.954x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 2.8701 ms | not_measured | managed | 1.002x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 316.2494 ms | not_measured | managed | 1.725x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.01487 ms | not_measured | managed | 0.135x |
| `np.sort(a) (float32)` | float32 | 100,000 | 1.1103 ms | not_measured | managed | 0.206x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.02210 ms | not_measured | managed | 0.147x |
| `np.sort(a) (float64)` | float64 | 100,000 | 1.7506 ms | not_measured | managed | 0.719x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.00885 ms | not_measured | managed | 0.249x |
| `np.sort(a) (int32)` | int32 | 100,000 | 1.2447 ms | not_measured | managed | 0.077x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.02573 ms | not_measured | managed | 0.230x |
| `np.sort(a) (int64)` | int64 | 100,000 | 1.8574 ms | not_measured | managed | 0.148x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.02106 ms | not_measured | managed | 0.132x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 1.3907 ms | not_measured | managed | 0.362x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.03077 ms | not_measured | managed | 0.128x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 1.9669 ms | not_measured | managed | 0.838x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.02487 ms | not_measured | managed | 0.122x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 1.4623 ms | not_measured | managed | 0.250x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.03367 ms | not_measured | managed | 0.190x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 2.1908 ms | not_measured | managed | 0.256x |
| `np.average(a) (float16)` | float16 | 1,000 | 0.00199 ms | not_measured | managed | 2.562x |
| `np.average(a) (float16)` | float16 | 100,000 | 0.1652 ms | not_measured | managed | 0.656x |
| `np.average(a) (float16)` | float16 | 10,000,000 | 16.5337 ms | not_measured | managed | 1.005x |
| `np.average(a) (float32)` | float32 | 1,000 | 0.000620 ms | not_measured | managed | 6.474x |
| `np.average(a) (float32)` | float32 | 100,000 | 0.00594 ms | not_measured | managed | 3.256x |
| `np.average(a) (float32)` | float32 | 10,000,000 | 2.5288 ms | not_measured | managed | 3.706x |
| `np.average(a) (float64)` | float64 | 1,000 | 0.000659 ms | not_measured | managed | 4.026x |
| `np.average(a) (float64)` | float64 | 100,000 | 0.00850 ms | not_measured | managed | 2.054x |
| `np.average(a) (float64)` | float64 | 10,000,000 | 7.4254 ms | not_measured | managed | 1.983x |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.00401 ms | not_measured | managed | 0.348x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.3611 ms | not_measured | managed | 0.272x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | 0.01330 ms | not_measured | managed | 0.895x |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | 0.4193 ms | not_measured | managed | 2.374x |
| `np.convolve(a, v, mode='same')` | float64 | 100,000 | 0.1165 ms | 0.9656 ms | managed | 8.533x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | 0.05297 ms | not_measured | managed | 0.358x |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | 1.3086 ms | not_measured | managed | 0.463x |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | 0.00642 ms | not_measured | managed | 1.688x |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | 0.3315 ms | not_measured | managed | 2.978x |
| `np.correlate(a, v, mode='same')` | float64 | 100,000 | 0.1173 ms | 0.9917 ms | managed | 9.493x |
| `np.count_nonzero(a) (float16)` | float16 | 1,000 | 0.00101 ms | not_measured | managed | 1.735x |
| `np.count_nonzero(a) (float16)` | float16 | 100,000 | 0.08269 ms | not_measured | managed | 1.795x |
| `np.count_nonzero(a) (float16)` | float16 | 10,000,000 | 8.2710 ms | not_measured | managed | 2.972x |
| `np.count_nonzero(a) (float32)` | float32 | 1,000 | 0.000317 ms | not_measured | managed | 1.874x |
| `np.count_nonzero(a) (float32)` | float32 | 100,000 | 0.01004 ms | not_measured | managed | 3.911x |
| `np.count_nonzero(a) (float32)` | float32 | 10,000,000 | 3.3180 ms | not_measured | managed | 2.330x |
| `np.count_nonzero(a) (float64)` | float64 | 1,000 | 0.000390 ms | not_measured | managed | 1.490x |
| `np.count_nonzero(a) (float64)` | float64 | 100,000 | 0.01829 ms | not_measured | managed | 2.074x |
| `np.count_nonzero(a) (float64)` | float64 | 10,000,000 | 8.7823 ms | not_measured | managed | 1.120x |
| `np.cov(a, b) (float64)` | float64 | 1,000 | 0.03986 ms | not_measured | managed | 0.344x |
| `np.cov(a, b) (float64)` | float64 | 100,000 | 1.2190 ms | not_measured | managed | 0.466x |
| `np.cross(a, b) (float64)` | float64 | 1,000 | 0.06730 ms | not_measured | managed | 0.199x |
| `np.cross(a, b) (float64)` | float64 | 100,000 | 0.9431 ms | not_measured | managed | 0.753x |
| `np.diff(a) (float64)` | float64 | 1,000 | 0.00523 ms | not_measured | managed | 0.318x |
| `np.diff(a) (float64)` | float64 | 100,000 | 0.09655 ms | not_measured | managed | 0.134x |
| `np.digitize(a, bins) (float64)` | float64 | 1,000 | 0.01682 ms | not_measured | managed | 1.038x |
| `np.digitize(a, bins) (float64)` | float64 | 100,000 | 4.1585 ms | not_measured | managed | 0.796x |
| `np.ediff1d(a) (float64)` | float64 | 1,000 | 0.00484 ms | not_measured | managed | 0.205x |
| `np.ediff1d(a) (float64)` | float64 | 100,000 | 0.1051 ms | not_measured | managed | 0.115x |
| `np.inner(a, b) (float64)` | float64 | 1,000 | 0.000540 ms | not_measured | managed | 1.143x |
| `np.inner(a, b) (float64)` | float64 | 100,000 | 0.01826 ms | 0.00740 ms | openblas | 0.959x |
| `np.kron(a, b) (float64)` | float64 | 1,000 | 0.00758 ms | not_measured | managed | 1.169x |
| `np.kron(a, b) (float64)` | float64 | 100,000 | 0.09041 ms | not_measured | managed | 0.487x |
| `np.median(a) (float16)` | float16 | 1,000 | 0.01194 ms | not_measured | managed | 1.164x |
| `np.median(a) (float16)` | float16 | 100,000 | 1.6594 ms | not_measured | managed | 0.535x |
| `np.median(a) (float16)` | float16 | 10,000,000 | 123.3255 ms | not_measured | managed | 1.109x |
| `np.median(a) (float32)` | float32 | 1,000 | 0.00396 ms | not_measured | managed | 2.971x |
| `np.median(a) (float32)` | float32 | 100,000 | 0.2831 ms | not_measured | managed | 1.612x |
| `np.median(a) (float32)` | float32 | 10,000,000 | 48.6873 ms | not_measured | managed | 1.930x |
| `np.median(a) (float64)` | float64 | 1,000 | 0.00411 ms | not_measured | managed | 2.350x |
| `np.median(a) (float64)` | float64 | 100,000 | 0.3086 ms | not_measured | managed | 1.561x |
| `np.median(a) (float64)` | float64 | 10,000,000 | 59.2493 ms | not_measured | managed | 1.862x |
| `np.percentile(a, 50) (float16)` | float16 | 1,000 | 0.01181 ms | not_measured | managed | 2.440x |
| `np.percentile(a, 50) (float16)` | float16 | 100,000 | 1.6561 ms | not_measured | managed | 1.074x |
| `np.percentile(a, 50) (float16)` | float16 | 10,000,000 | 123.3723 ms | not_measured | managed | 1.204x |
| `np.percentile(a, 50) (float32)` | float32 | 1,000 | 0.00387 ms | not_measured | managed | 6.068x |
| `np.percentile(a, 50) (float32)` | float32 | 100,000 | 0.2753 ms | not_measured | managed | 2.572x |
| `np.percentile(a, 50) (float32)` | float32 | 10,000,000 | 48.6748 ms | not_measured | managed | 1.293x |
| `np.percentile(a, 50) (float64)` | float64 | 1,000 | 0.00405 ms | not_measured | managed | 5.724x |
| `np.percentile(a, 50) (float64)` | float64 | 100,000 | 0.3093 ms | not_measured | managed | 2.288x |
| `np.percentile(a, 50) (float64)` | float64 | 10,000,000 | 59.2612 ms | not_measured | managed | 1.334x |
| `np.ptp(a) (float16)` | float16 | 1,000 | 0.00469 ms | not_measured | managed | 1.553x |
| `np.ptp(a) (float16)` | float16 | 100,000 | 0.8840 ms | not_measured | managed | 1.127x |
| `np.ptp(a) (float16)` | float16 | 10,000,000 | 90.2191 ms | not_measured | managed | 1.398x |
| `np.ptp(a) (float32)` | float32 | 1,000 | 0.00184 ms | not_measured | managed | 1.348x |
| `np.ptp(a) (float32)` | float32 | 100,000 | 0.01787 ms | not_measured | managed | 0.722x |
| `np.ptp(a) (float32)` | float32 | 10,000,000 | 6.2727 ms | not_measured | managed | 1.128x |
| `np.ptp(a) (float64)` | float64 | 1,000 | 0.00190 ms | not_measured | managed | 1.349x |
| `np.ptp(a) (float64)` | float64 | 100,000 | 0.03467 ms | not_measured | managed | 0.557x |
| `np.ptp(a) (float64)` | float64 | 10,000,000 | 16.1987 ms | not_measured | managed | 0.996x |
| `np.quantile(a, 0.5) (float16)` | float16 | 1,000 | 0.01182 ms | not_measured | managed | 2.402x |
| `np.quantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6565 ms | not_measured | managed | 1.074x |
| `np.quantile(a, 0.5) (float16)` | float16 | 10,000,000 | 123.3871 ms | not_measured | managed | 1.205x |
| `np.quantile(a, 0.5) (float32)` | float32 | 1,000 | 0.00391 ms | not_measured | managed | 5.824x |
| `np.quantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2865 ms | not_measured | managed | 2.438x |
| `np.quantile(a, 0.5) (float32)` | float32 | 10,000,000 | 48.1968 ms | not_measured | managed | 1.290x |
| `np.quantile(a, 0.5) (float64)` | float64 | 1,000 | 0.00402 ms | not_measured | managed | 5.824x |
| `np.quantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3089 ms | not_measured | managed | 2.276x |
| `np.quantile(a, 0.5) (float64)` | float64 | 10,000,000 | 59.6724 ms | not_measured | managed | 1.327x |
| `np.trace(a) (float64)` | float64 | 1,000 | 0.000589 ms | not_measured | managed | 2.270x |
| `np.trace(a) (float64)` | float64 | 100,000 | 0.000641 ms | not_measured | managed | 2.092x |
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.00950 ms | not_measured | managed | 0.338x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.9049 ms | not_measured | managed | 0.316x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 89.9597 ms | not_measured | managed | 0.748x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.00129 ms | not_measured | managed | 0.337x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.04766 ms | not_measured | managed | 0.265x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 8.8880 ms | not_measured | managed | 1.409x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.00234 ms | not_measured | managed | 0.198x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.1018 ms | not_measured | managed | 0.209x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 19.9778 ms | not_measured | managed | 1.411x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.01981 ms | not_measured | managed | 0.516x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 2.2430 ms | not_measured | managed | 0.396x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 223.7398 ms | not_measured | managed | 0.756x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.00274 ms | not_measured | managed | 0.291x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.1007 ms | not_measured | managed | 0.245x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 18.7789 ms | not_measured | managed | 1.595x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.00423 ms | not_measured | managed | 0.219x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.2001 ms | not_measured | managed | 2.284x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 46.1099 ms | not_measured | managed | 1.340x |
| `np.abs (float16)` | float16 | 1,000 | 0.00133 ms | not_measured | managed | 0.462x |
| `np.abs (float16)` | float16 | 100,000 | 0.05085 ms | not_measured | managed | 0.513x |
| `np.abs (float16)` | float16 | 10,000,000 | 5.6962 ms | not_measured | managed | 1.279x |
| `np.abs (float32)` | float32 | 1,000 | 0.00144 ms | not_measured | managed | 0.290x |
| `np.abs (float32)` | float32 | 100,000 | 0.04976 ms | not_measured | managed | 0.249x |
| `np.abs (float32)` | float32 | 10,000,000 | 8.1220 ms | not_measured | managed | 1.500x |
| `np.abs (float64)` | float64 | 1,000 | 0.00225 ms | not_measured | managed | 0.198x |
| `np.abs (float64)` | float64 | 100,000 | 0.09923 ms | not_measured | managed | 0.217x |
| `np.abs (float64)` | float64 | 10,000,000 | 18.7756 ms | not_measured | managed | 1.269x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.00143 ms | not_measured | managed | 0.424x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.05332 ms | not_measured | managed | 0.546x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 5.8649 ms | not_measured | managed | 1.252x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.00131 ms | not_measured | managed | 0.298x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.04763 ms | not_measured | managed | 0.243x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 8.6514 ms | not_measured | managed | 1.373x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.00184 ms | not_measured | managed | 0.242x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.09499 ms | not_measured | managed | 0.239x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 18.7208 ms | not_measured | managed | 1.295x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.02014 ms | not_measured | managed | 0.429x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 1.9750 ms | not_measured | managed | 1.110x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 196.3513 ms | not_measured | managed | 0.870x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.01093 ms | not_measured | managed | 0.508x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 1.0267 ms | not_measured | managed | 1.134x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 103.3345 ms | not_measured | managed | 1.384x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.01387 ms | not_measured | managed | 0.590x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 1.3219 ms | not_measured | managed | 1.140x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 134.6807 ms | not_measured | managed | 1.066x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.01847 ms | not_measured | managed | 0.612x |
| `np.angle(a) (float16)` | float16 | 100,000 | 1.4341 ms | not_measured | managed | 1.202x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 161.3017 ms | not_measured | managed | 1.084x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.01473 ms | not_measured | managed | 0.268x |
| `np.angle(a) (float32)` | float32 | 100,000 | 1.1537 ms | not_measured | managed | 0.920x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 132.4335 ms | not_measured | managed | 0.804x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.01314 ms | not_measured | managed | 0.329x |
| `np.angle(a) (float64)` | float64 | 100,000 | 1.1178 ms | not_measured | managed | 0.941x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 119.6004 ms | not_measured | managed | 1.149x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.01675 ms | not_measured | managed | 0.395x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 2.0821 ms | not_measured | managed | 0.521x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 207.6575 ms | not_measured | managed | 0.923x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.00845 ms | not_measured | managed | 0.421x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 1.2984 ms | not_measured | managed | 0.858x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 131.3112 ms | not_measured | managed | 0.936x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.00942 ms | not_measured | managed | 0.395x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 1.3683 ms | not_measured | managed | 0.913x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 140.1688 ms | not_measured | managed | 1.105x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.02012 ms | not_measured | managed | 0.426x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 1.9759 ms | not_measured | managed | 0.988x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 197.5556 ms | not_measured | managed | 0.889x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.01043 ms | not_measured | managed | 0.533x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 1.0553 ms | not_measured | managed | 1.493x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 102.3527 ms | not_measured | managed | 1.373x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.01393 ms | not_measured | managed | 0.476x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 1.3214 ms | not_measured | managed | 1.035x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 132.9119 ms | not_measured | managed | 1.099x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.01714 ms | not_measured | managed | 0.429x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 2.1621 ms | not_measured | managed | 0.512x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 215.3242 ms | not_measured | managed | 1.028x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.00935 ms | not_measured | managed | 0.418x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 1.3474 ms | not_measured | managed | 0.936x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 133.9240 ms | not_measured | managed | 1.143x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.00940 ms | not_measured | managed | 0.402x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 1.4131 ms | not_measured | managed | 0.929x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 145.1251 ms | not_measured | managed | 1.135x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.02519 ms | not_measured | managed | 0.425x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 2.5482 ms | not_measured | managed | 0.711x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 253.2510 ms | not_measured | managed | 0.859x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.01471 ms | not_measured | managed | 0.480x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 1.5554 ms | not_measured | managed | 1.017x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 155.4148 ms | not_measured | managed | 1.085x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.01964 ms | not_measured | managed | 0.422x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 2.0569 ms | not_measured | managed | 1.114x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 206.9168 ms | not_measured | managed | 1.076x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.01631 ms | not_measured | managed | 0.457x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 2.3231 ms | not_measured | managed | 0.555x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 229.0151 ms | not_measured | managed | 0.930x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.00783 ms | not_measured | managed | 0.591x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 1.3938 ms | not_measured | managed | 0.918x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 140.9135 ms | not_measured | managed | 1.028x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.00897 ms | not_measured | managed | 0.438x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 1.5735 ms | not_measured | managed | 0.942x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 160.1650 ms | not_measured | managed | 1.018x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.02913 ms | not_measured | managed | 0.356x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 3.7154 ms | not_measured | managed | 0.459x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 369.4010 ms | not_measured | managed | 0.815x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.02572 ms | not_measured | managed | 0.165x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 3.3054 ms | not_measured | managed | 0.604x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 330.6456 ms | not_measured | managed | 0.642x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.01652 ms | not_measured | managed | 0.349x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 2.5383 ms | not_measured | managed | 1.085x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 258.6393 ms | not_measured | managed | 1.088x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.02090 ms | not_measured | managed | 0.434x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 2.1346 ms | not_measured | managed | 0.964x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 213.4826 ms | not_measured | managed | 0.886x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.01203 ms | not_measured | managed | 0.436x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 1.3069 ms | not_measured | managed | 0.968x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 129.6613 ms | not_measured | managed | 1.102x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.01331 ms | not_measured | managed | 0.455x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 1.4812 ms | not_measured | managed | 1.159x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 149.6877 ms | not_measured | managed | 1.085x |
| `np.around (float16)` | float16 | 1,000 | 0.00716 ms | not_measured | managed | 0.693x |
| `np.around (float16)` | float16 | 100,000 | 0.6880 ms | not_measured | managed | 0.598x |
| `np.around (float16)` | float16 | 10,000,000 | 67.0655 ms | not_measured | managed | 0.909x |
| `np.around (float32)` | float32 | 1,000 | 0.00130 ms | not_measured | managed | 0.747x |
| `np.around (float32)` | float32 | 100,000 | 0.04874 ms | not_measured | managed | 0.316x |
| `np.around (float32)` | float32 | 10,000,000 | 8.2870 ms | not_measured | managed | 1.414x |
| `np.around (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.533x |
| `np.around (float64)` | float64 | 100,000 | 0.09673 ms | not_measured | managed | 0.238x |
| `np.around (float64)` | float64 | 10,000,000 | 18.7808 ms | not_measured | managed | 1.284x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.02520 ms | not_measured | managed | 0.430x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 2.5444 ms | not_measured | managed | 0.959x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 252.0890 ms | not_measured | managed | 0.866x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.01472 ms | not_measured | managed | 0.662x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 1.5554 ms | not_measured | managed | 1.083x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 155.8715 ms | not_measured | managed | 1.083x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.01921 ms | not_measured | managed | 0.430x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 2.0573 ms | not_measured | managed | 1.075x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 206.5408 ms | not_measured | managed | 1.061x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.02088 ms | not_measured | managed | 0.439x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 2.1150 ms | not_measured | managed | 0.937x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 213.4723 ms | not_measured | managed | 0.878x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.01175 ms | not_measured | managed | 0.438x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 1.3053 ms | not_measured | managed | 1.114x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 129.3214 ms | not_measured | managed | 1.096x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.01323 ms | not_measured | managed | 0.449x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 1.4862 ms | not_measured | managed | 1.015x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 150.0611 ms | not_measured | managed | 1.061x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.02312 ms | not_measured | managed | 0.412x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 2.3294 ms | not_measured | managed | 0.927x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 230.9339 ms | not_measured | managed | 0.939x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.01319 ms | not_measured | managed | 0.444x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 1.4516 ms | not_measured | managed | 0.732x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 143.3171 ms | not_measured | managed | 1.054x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.01875 ms | not_measured | managed | 0.548x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 2.0253 ms | not_measured | managed | 1.134x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 205.5419 ms | not_measured | managed | 1.047x |
| `np.ceil (float16)` | float16 | 1,000 | 0.00648 ms | not_measured | managed | 0.630x |
| `np.ceil (float16)` | float16 | 100,000 | 0.6179 ms | not_measured | managed | 0.623x |
| `np.ceil (float16)` | float16 | 10,000,000 | 60.0743 ms | not_measured | managed | 0.968x |
| `np.ceil (float32)` | float32 | 1,000 | 0.00121 ms | not_measured | managed | 0.334x |
| `np.ceil (float32)` | float32 | 100,000 | 0.04889 ms | not_measured | managed | 0.267x |
| `np.ceil (float32)` | float32 | 10,000,000 | 8.3380 ms | not_measured | managed | 1.391x |
| `np.ceil (float64)` | float64 | 1,000 | 0.00204 ms | not_measured | managed | 0.218x |
| `np.ceil (float64)` | float64 | 100,000 | 0.09742 ms | not_measured | managed | 0.222x |
| `np.ceil (float64)` | float64 | 10,000,000 | 18.7553 ms | not_measured | managed | 1.267x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.00926 ms | not_measured | managed | 0.897x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 1.0037 ms | not_measured | managed | 0.938x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 100.6777 ms | not_measured | managed | 1.237x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.00325 ms | not_measured | managed | 0.592x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.06317 ms | not_measured | managed | 0.257x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 8.7191 ms | not_measured | managed | 1.291x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.00346 ms | not_measured | managed | 0.519x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.1051 ms | not_measured | managed | 0.262x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 20.0970 ms | not_measured | managed | 1.180x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.00125 ms | not_measured | managed | 0.442x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.04296 ms | not_measured | managed | 0.890x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 5.0779 ms | not_measured | managed | 1.203x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.00137 ms | not_measured | managed | 0.401x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.05821 ms | not_measured | managed | 0.613x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 9.2864 ms | not_measured | managed | 1.458x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.00209 ms | not_measured | managed | 0.257x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.09726 ms | not_measured | managed | 0.319x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 20.4169 ms | not_measured | managed | 1.267x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.00126 ms | not_measured | managed | 0.440x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.04414 ms | not_measured | managed | 0.871x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 5.0817 ms | not_measured | managed | 1.204x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.00142 ms | not_measured | managed | 0.380x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.05731 ms | not_measured | managed | 0.491x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 9.2858 ms | not_measured | managed | 1.373x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.00197 ms | not_measured | managed | 0.274x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.09949 ms | not_measured | managed | 0.299x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 20.2820 ms | not_measured | managed | 1.264x |
| `np.cos (float16)` | float16 | 1,000 | 0.01637 ms | not_measured | managed | 0.421x |
| `np.cos (float16)` | float16 | 100,000 | 1.8826 ms | not_measured | managed | 0.520x |
| `np.cos (float16)` | float16 | 10,000,000 | 188.0757 ms | not_measured | managed | 0.854x |
| `np.cos (float32)` | float32 | 1,000 | 0.00315 ms | not_measured | managed | 0.345x |
| `np.cos (float32)` | float32 | 100,000 | 0.2740 ms | not_measured | managed | 1.018x |
| `np.cos (float32)` | float32 | 10,000,000 | 26.7950 ms | not_measured | managed | 0.923x |
| `np.cos (float64)` | float64 | 1,000 | 0.01126 ms | not_measured | managed | 0.423x |
| `np.cos (float64)` | float64 | 100,000 | 1.2250 ms | not_measured | managed | 0.933x |
| `np.cos (float64)` | float64 | 10,000,000 | 127.4585 ms | not_measured | managed | 1.002x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.02124 ms | not_measured | managed | 0.369x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 2.1125 ms | not_measured | managed | 0.727x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 212.1583 ms | not_measured | managed | 0.733x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.01063 ms | not_measured | managed | 0.364x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 1.1183 ms | not_measured | managed | 0.952x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 111.7542 ms | not_measured | managed | 1.016x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.01175 ms | not_measured | managed | 0.324x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 1.1880 ms | not_measured | managed | 1.049x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 124.0984 ms | not_measured | managed | 1.074x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.00707 ms | not_measured | managed | 0.635x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.6433 ms | not_measured | managed | 0.733x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 66.7482 ms | not_measured | managed | 1.014x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.00134 ms | not_measured | managed | 0.808x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.04587 ms | not_measured | managed | 3.949x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 8.5030 ms | not_measured | managed | 2.319x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.00204 ms | not_measured | managed | 0.661x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.09558 ms | not_measured | managed | 1.168x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 18.8732 ms | not_measured | managed | 1.551x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.00709 ms | not_measured | managed | 0.581x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.6718 ms | not_measured | managed | 0.686x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 66.6525 ms | not_measured | managed | 0.939x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.00129 ms | not_measured | managed | 0.841x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.04719 ms | not_measured | managed | 2.131x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 8.3400 ms | not_measured | managed | 2.260x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.00153 ms | not_measured | managed | 0.825x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.09402 ms | not_measured | managed | 1.088x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 19.1153 ms | not_measured | managed | 1.579x |
| `np.exp (float16)` | float16 | 1,000 | 0.01120 ms | not_measured | managed | 0.395x |
| `np.exp (float16)` | float16 | 100,000 | 0.9200 ms | not_measured | managed | 0.427x |
| `np.exp (float16)` | float16 | 10,000,000 | 107.9826 ms | not_measured | managed | 0.683x |
| `np.exp (float32)` | float32 | 1,000 | 0.00296 ms | not_measured | managed | 0.306x |
| `np.exp (float32)` | float32 | 100,000 | 0.1808 ms | not_measured | managed | 1.540x |
| `np.exp (float32)` | float32 | 10,000,000 | 20.0182 ms | not_measured | managed | 1.691x |
| `np.exp (float64)` | float64 | 1,000 | 0.00663 ms | not_measured | managed | 0.434x |
| `np.exp (float64)` | float64 | 100,000 | 0.4771 ms | not_measured | managed | 0.926x |
| `np.exp (float64)` | float64 | 10,000,000 | 51.3222 ms | not_measured | managed | 1.055x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.01827 ms | not_measured | managed | 0.246x |
| `np.exp2 (float16)` | float16 | 100,000 | 1.7868 ms | not_measured | managed | 0.241x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 178.5182 ms | not_measured | managed | 0.384x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.01614 ms | not_measured | managed | 0.119x |
| `np.exp2 (float32)` | float32 | 100,000 | 1.3309 ms | not_measured | managed | 0.268x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 149.3123 ms | not_measured | managed | 0.281x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.01639 ms | not_measured | managed | 0.134x |
| `np.exp2 (float64)` | float64 | 100,000 | 1.4320 ms | not_measured | managed | 0.295x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 146.1767 ms | not_measured | managed | 0.406x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.01384 ms | not_measured | managed | 0.396x |
| `np.expm1 (float16)` | float16 | 100,000 | 1.2675 ms | not_measured | managed | 0.405x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 134.3473 ms | not_measured | managed | 0.667x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.00427 ms | not_measured | managed | 0.649x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.3425 ms | not_measured | managed | 1.340x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 33.4575 ms | not_measured | managed | 1.481x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.00651 ms | not_measured | managed | 0.575x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.4736 ms | not_measured | managed | 1.053x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 51.0942 ms | not_measured | managed | 1.442x |
| `np.floor (float16)` | float16 | 1,000 | 0.00648 ms | not_measured | managed | 0.644x |
| `np.floor (float16)` | float16 | 100,000 | 0.6080 ms | not_measured | managed | 0.635x |
| `np.floor (float16)` | float16 | 10,000,000 | 59.6820 ms | not_measured | managed | 0.969x |
| `np.floor (float32)` | float32 | 1,000 | 0.00142 ms | not_measured | managed | 0.348x |
| `np.floor (float32)` | float32 | 100,000 | 0.04848 ms | not_measured | managed | 0.293x |
| `np.floor (float32)` | float32 | 10,000,000 | 8.2982 ms | not_measured | managed | 1.430x |
| `np.floor (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.239x |
| `np.floor (float64)` | float64 | 100,000 | 0.09858 ms | not_measured | managed | 0.220x |
| `np.floor (float64)` | float64 | 10,000,000 | 18.8360 ms | not_measured | managed | 1.270x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.000772 ms | not_measured | managed | 0.427x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.00295 ms | not_measured | managed | 2.483x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | 0.00987 ms | not_measured | managed | 2.301x |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.000733 ms | not_measured | managed | 0.457x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.00271 ms | not_measured | managed | 4.894x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | 0.00310 ms | not_measured | managed | 6.595x |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.000689 ms | not_measured | managed | 0.489x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.00445 ms | not_measured | managed | 5.212x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | 0.01481 ms | not_measured | managed | 1.286x |
| `np.log (float16)` | float16 | 1,000 | 0.01258 ms | not_measured | managed | 0.366x |
| `np.log (float16)` | float16 | 100,000 | 1.0017 ms | not_measured | managed | 0.407x |
| `np.log (float16)` | float16 | 10,000,000 | 121.2002 ms | not_measured | managed | 0.691x |
| `np.log (float32)` | float32 | 1,000 | 0.00287 ms | not_measured | managed | 0.433x |
| `np.log (float32)` | float32 | 100,000 | 0.2306 ms | not_measured | managed | 1.754x |
| `np.log (float32)` | float32 | 10,000,000 | 22.5586 ms | not_measured | managed | 2.046x |
| `np.log (float64)` | float64 | 1,000 | 0.00549 ms | not_measured | managed | 0.484x |
| `np.log (float64)` | float64 | 100,000 | 0.5235 ms | not_measured | managed | 1.740x |
| `np.log (float64)` | float64 | 10,000,000 | 56.8165 ms | not_measured | managed | 1.421x |
| `np.log10 (float16)` | float16 | 1,000 | 0.01289 ms | not_measured | managed | 0.363x |
| `np.log10 (float16)` | float16 | 100,000 | 1.2434 ms | not_measured | managed | 0.347x |
| `np.log10 (float16)` | float16 | 10,000,000 | 124.2899 ms | not_measured | managed | 0.705x |
| `np.log10 (float32)` | float32 | 1,000 | 0.00527 ms | not_measured | managed | 0.430x |
| `np.log10 (float32)` | float32 | 100,000 | 0.4614 ms | not_measured | managed | 0.797x |
| `np.log10 (float32)` | float32 | 10,000,000 | 47.1216 ms | not_measured | managed | 0.925x |
| `np.log10 (float64)` | float64 | 1,000 | 0.00581 ms | not_measured | managed | 0.480x |
| `np.log10 (float64)` | float64 | 100,000 | 0.5572 ms | not_measured | managed | 0.815x |
| `np.log10 (float64)` | float64 | 10,000,000 | 60.0514 ms | not_measured | managed | 1.369x |
| `np.log1p (float16)` | float16 | 1,000 | 0.01315 ms | not_measured | managed | 0.448x |
| `np.log1p (float16)` | float16 | 100,000 | 1.2320 ms | not_measured | managed | 0.451x |
| `np.log1p (float16)` | float16 | 10,000,000 | 125.9269 ms | not_measured | managed | 0.823x |
| `np.log1p (float32)` | float32 | 1,000 | 0.00447 ms | not_measured | managed | 0.704x |
| `np.log1p (float32)` | float32 | 100,000 | 0.4350 ms | not_measured | managed | 1.346x |
| `np.log1p (float32)` | float32 | 10,000,000 | 42.9848 ms | not_measured | managed | 1.463x |
| `np.log1p (float64)` | float64 | 1,000 | 0.00586 ms | not_measured | managed | 0.625x |
| `np.log1p (float64)` | float64 | 100,000 | 0.5050 ms | not_measured | managed | 1.336x |
| `np.log1p (float64)` | float64 | 10,000,000 | 57.1168 ms | not_measured | managed | 1.318x |
| `np.log2 (float16)` | float16 | 1,000 | 0.01104 ms | not_measured | managed | 0.440x |
| `np.log2 (float16)` | float16 | 100,000 | 1.0102 ms | not_measured | managed | 0.452x |
| `np.log2 (float16)` | float16 | 10,000,000 | 106.3960 ms | not_measured | managed | 0.719x |
| `np.log2 (float32)` | float32 | 1,000 | 0.00481 ms | not_measured | managed | 0.464x |
| `np.log2 (float32)` | float32 | 100,000 | 0.4690 ms | not_measured | managed | 0.694x |
| `np.log2 (float32)` | float32 | 10,000,000 | 47.4170 ms | not_measured | managed | 0.813x |
| `np.log2 (float64)` | float64 | 1,000 | 0.00822 ms | not_measured | managed | 0.504x |
| `np.log2 (float64)` | float64 | 100,000 | 0.7678 ms | not_measured | managed | 1.008x |
| `np.log2 (float64)` | float64 | 10,000,000 | 79.1865 ms | not_measured | managed | 1.085x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.00367 ms | not_measured | managed | 0.557x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.1435 ms | not_measured | managed | 2.984x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 17.1254 ms | not_measured | managed | 2.904x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.00593 ms | not_measured | managed | 0.366x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.2860 ms | not_measured | managed | 1.628x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 45.5797 ms | not_measured | managed | 1.413x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.00143 ms | not_measured | managed | 0.431x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.05384 ms | not_measured | managed | 0.503x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 5.7009 ms | not_measured | managed | 1.247x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.00127 ms | not_measured | managed | 0.311x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.04859 ms | not_measured | managed | 0.245x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 8.6283 ms | not_measured | managed | 1.328x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.00216 ms | not_measured | managed | 0.209x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.09680 ms | not_measured | managed | 0.251x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 18.9068 ms | not_measured | managed | 1.200x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.00108 ms | not_measured | managed | 0.516x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.02449 ms | not_measured | managed | 0.861x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 2.4141 ms | not_measured | managed | 2.504x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.00134 ms | not_measured | managed | 0.404x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.04434 ms | not_measured | managed | 2.147x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 5.1882 ms | not_measured | managed | 2.315x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.00180 ms | not_measured | managed | 0.314x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.09260 ms | not_measured | managed | 0.352x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 13.6912 ms | not_measured | managed | 1.752x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.00724 ms | not_measured | managed | 1.206x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.6444 ms | not_measured | managed | 1.254x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 65.5231 ms | not_measured | managed | 3.078x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.00231 ms | not_measured | managed | 0.784x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.07368 ms | not_measured | managed | 2.018x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 8.8303 ms | not_measured | managed | 2.262x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.00406 ms | not_measured | managed | 0.486x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.2798 ms | not_measured | managed | 2.022x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 33.1255 ms | not_measured | managed | 1.979x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.00974 ms | not_measured | managed | 1.035x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.9028 ms | not_measured | managed | 1.167x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 89.9158 ms | not_measured | managed | 2.442x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.00186 ms | not_measured | managed | 1.148x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.04908 ms | not_measured | managed | 3.417x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 9.0635 ms | not_measured | managed | 2.439x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.00294 ms | not_measured | managed | 0.723x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.09481 ms | not_measured | managed | 2.034x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 19.6010 ms | not_measured | managed | 1.498x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.03438 ms | not_measured | managed | 0.335x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 3.9041 ms | not_measured | managed | 0.381x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 394.1282 ms | not_measured | managed | 0.777x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.01303 ms | not_measured | managed | 0.439x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 1.3025 ms | not_measured | managed | 1.012x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 127.6769 ms | not_measured | managed | 1.274x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.02274 ms | not_measured | managed | 0.418x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 2.0669 ms | not_measured | managed | 1.124x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 205.8681 ms | not_measured | managed | 1.181x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.00705 ms | not_measured | managed | 0.631x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.6380 ms | not_measured | managed | 0.778x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 66.7026 ms | not_measured | managed | 1.053x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.00136 ms | not_measured | managed | 0.800x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.04802 ms | not_measured | managed | 2.130x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 8.3630 ms | not_measured | managed | 2.462x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.00210 ms | not_measured | managed | 0.602x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.09504 ms | not_measured | managed | 1.094x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 19.1214 ms | not_measured | managed | 1.613x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.00707 ms | not_measured | managed | 0.634x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.6758 ms | not_measured | managed | 0.792x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 66.7513 ms | not_measured | managed | 0.984x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.00136 ms | not_measured | managed | 0.845x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.04779 ms | not_measured | managed | 2.079x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 8.5965 ms | not_measured | managed | 2.399x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.00187 ms | not_measured | managed | 0.679x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.09360 ms | not_measured | managed | 1.127x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 18.8645 ms | not_measured | managed | 1.529x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.000001 ms | not_measured | managed | 137.000x |
| `np.real(a) (float16)` | float16 | 100,000 | 0.000001 ms | not_measured | managed | 210.000x |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.000001 ms | not_measured | managed | 209.000x |
| `np.real(a) (float32)` | float32 | 1,000 | 0.000001 ms | not_measured | managed | 151.000x |
| `np.real(a) (float32)` | float32 | 100,000 | 0.000001 ms | not_measured | managed | 222.000x |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.000001 ms | not_measured | managed | 209.000x |
| `np.real(a) (float64)` | float64 | 1,000 | 0.000001 ms | not_measured | managed | 137.000x |
| `np.real(a) (float64)` | float64 | 100,000 | 0.000001 ms | not_measured | managed | 221.000x |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.000001 ms | not_measured | managed | 209.000x |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.00766 ms | not_measured | managed | 0.312x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.7370 ms | not_measured | managed | 0.283x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 72.5199 ms | not_measured | managed | 0.644x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.00142 ms | not_measured | managed | 0.329x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.06037 ms | not_measured | managed | 1.038x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 8.6113 ms | not_measured | managed | 1.347x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.00231 ms | not_measured | managed | 0.305x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.1962 ms | not_measured | managed | 1.121x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 23.6292 ms | not_measured | managed | 1.151x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.00802 ms | not_measured | managed | 0.638x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.7903 ms | not_measured | managed | 0.810x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 79.2852 ms | not_measured | managed | 1.010x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.00125 ms | not_measured | managed | 0.318x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.04831 ms | not_measured | managed | 0.253x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 8.4865 ms | not_measured | managed | 1.390x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.00202 ms | not_measured | managed | 0.214x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.09774 ms | not_measured | managed | 0.254x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 19.0169 ms | not_measured | managed | 1.259x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.00799 ms | not_measured | managed | 0.731x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.7907 ms | not_measured | managed | 0.946x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 79.2939 ms | not_measured | managed | 0.940x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.00130 ms | not_measured | managed | 0.764x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.04375 ms | not_measured | managed | 0.348x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 8.3743 ms | not_measured | managed | 1.444x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.00177 ms | not_measured | managed | 0.574x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.09388 ms | not_measured | managed | 0.269x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 18.7347 ms | not_measured | managed | 1.308x |
| `np.sign (float16)` | float16 | 1,000 | 0.00769 ms | not_measured | managed | 0.159x |
| `np.sign (float16)` | float16 | 100,000 | 0.9764 ms | not_measured | managed | 0.091x |
| `np.sign (float16)` | float16 | 10,000,000 | 97.7029 ms | not_measured | managed | 0.208x |
| `np.sign (float32)` | float32 | 1,000 | 0.00239 ms | not_measured | managed | 0.454x |
| `np.sign (float32)` | float32 | 100,000 | 0.5621 ms | not_measured | managed | 0.767x |
| `np.sign (float32)` | float32 | 10,000,000 | 57.3671 ms | not_measured | managed | 0.849x |
| `np.sign (float64)` | float64 | 1,000 | 0.00283 ms | not_measured | managed | 0.356x |
| `np.sign (float64)` | float64 | 100,000 | 0.5652 ms | not_measured | managed | 0.743x |
| `np.sign (float64)` | float64 | 10,000,000 | 61.3028 ms | not_measured | managed | 0.865x |
| `np.sin (float16)` | float16 | 1,000 | 0.01629 ms | not_measured | managed | 0.394x |
| `np.sin (float16)` | float16 | 100,000 | 1.8851 ms | not_measured | managed | 0.501x |
| `np.sin (float16)` | float16 | 10,000,000 | 188.7435 ms | not_measured | managed | 0.877x |
| `np.sin (float32)` | float32 | 1,000 | 0.00313 ms | not_measured | managed | 0.340x |
| `np.sin (float32)` | float32 | 100,000 | 0.2717 ms | not_measured | managed | 1.031x |
| `np.sin (float32)` | float32 | 10,000,000 | 26.7708 ms | not_measured | managed | 1.288x |
| `np.sin (float64)` | float64 | 1,000 | 0.01053 ms | not_measured | managed | 0.443x |
| `np.sin (float64)` | float64 | 100,000 | 1.2046 ms | not_measured | managed | 0.969x |
| `np.sin (float64)` | float64 | 10,000,000 | 125.2839 ms | not_measured | managed | 1.040x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.02194 ms | not_measured | managed | 0.378x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 2.1928 ms | not_measured | managed | 0.578x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 218.1030 ms | not_measured | managed | 0.754x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.01084 ms | not_measured | managed | 0.341x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 1.1843 ms | not_measured | managed | 0.956x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 117.3990 ms | not_measured | managed | 1.028x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.01254 ms | not_measured | managed | 0.393x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 1.3046 ms | not_measured | managed | 0.993x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 132.7978 ms | not_measured | managed | 1.226x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.00694 ms | not_measured | managed | 0.674x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.6553 ms | not_measured | managed | 0.626x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 65.5245 ms | not_measured | managed | 0.829x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.00155 ms | not_measured | managed | 0.317x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.07513 ms | not_measured | managed | 0.978x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 8.4589 ms | not_measured | managed | 1.462x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.00315 ms | not_measured | managed | 0.296x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.2914 ms | not_measured | managed | 1.031x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 33.0161 ms | not_measured | managed | 1.164x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.00879 ms | not_measured | managed | 0.273x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.8379 ms | not_measured | managed | 0.254x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 83.5909 ms | not_measured | managed | 0.535x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.00125 ms | not_measured | managed | 0.313x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.04845 ms | not_measured | managed | 0.244x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 8.4613 ms | not_measured | managed | 1.327x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.00210 ms | not_measured | managed | 0.206x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.1003 ms | not_measured | managed | 0.245x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 18.9046 ms | not_measured | managed | 1.255x |
| `np.tan (float16)` | float16 | 1,000 | 0.01640 ms | not_measured | managed | 0.412x |
| `np.tan (float16)` | float16 | 100,000 | 1.9046 ms | not_measured | managed | 0.502x |
| `np.tan (float16)` | float16 | 10,000,000 | 189.9889 ms | not_measured | managed | 0.869x |
| `np.tan (float32)` | float32 | 1,000 | 0.00751 ms | not_measured | managed | 0.441x |
| `np.tan (float32)` | float32 | 100,000 | 1.1412 ms | not_measured | managed | 0.902x |
| `np.tan (float32)` | float32 | 10,000,000 | 114.2654 ms | not_measured | managed | 1.008x |
| `np.tan (float64)` | float64 | 1,000 | 0.01218 ms | not_measured | managed | 0.372x |
| `np.tan (float64)` | float64 | 100,000 | 1.5600 ms | not_measured | managed | 0.996x |
| `np.tan (float64)` | float64 | 10,000,000 | 158.0826 ms | not_measured | managed | 1.088x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.02114 ms | not_measured | managed | 0.466x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 2.0739 ms | not_measured | managed | 0.832x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 206.8754 ms | not_measured | managed | 0.867x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.00405 ms | not_measured | managed | 0.392x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.2084 ms | not_measured | managed | 2.848x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 36.9141 ms | not_measured | managed | 1.632x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.00883 ms | not_measured | managed | 0.704x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.7957 ms | not_measured | managed | 3.200x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 85.8167 ms | not_measured | managed | 2.967x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.00641 ms | not_measured | managed | 0.759x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.6128 ms | not_measured | managed | 0.779x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 60.0007 ms | not_measured | managed | 1.020x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.00130 ms | not_measured | managed | 0.302x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.04757 ms | not_measured | managed | 0.321x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 8.6994 ms | not_measured | managed | 1.367x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.00208 ms | not_measured | managed | 0.236x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.09265 ms | not_measured | managed | 0.254x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 18.8464 ms | not_measured | managed | 1.199x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-25 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.45× geomean · 69%🕐 of NumPy's time · 106 win / 59 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ███████████████▎ ...   1.53×    65%🕐  ( 20 win / 13 lose)
1K         ███████████████▎ ...   1.53×    66%🕐  ( 21 win / 12 lose)
100K       ████████████▎ ......   1.24×    81%🕐  ( 17 win / 16 lose)
1M         ███████████████▋ ...   1.57×    64%🕐  ( 22 win / 11 lose)
10M        ██████████████ .....   1.41×    71%🕐  ( 26 win /  7 lose)
ALL        ██████████████▌ ....   1.45×    69%🕐  (106 win / 59 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise██████████████▏ ....   1.42×    70%🕐  ( 35 win /  5 lose)
reductions ███████████████████▶   2.92×    34%🕐  ( 34 win /  6 lose)
selection  ██████████████ .....   1.41×    71%🕐  ( 22 win / 13 lose)
copy/cast  ███████▏ ...........   0.72×   138%🕐  (  8 win / 17 lose)  ◄ SLOWER
index-math ████████▌ ..........   0.86×   117%🕐  (  4 win /  6 lose)  ◄ SLOWER
dtypes     ███████████▍ .......   1.14×    88%🕐  (  3 win / 12 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.67×    2.51×    1.18×    1.09×    1.10×
reductions      6.93×    4.46×    1.79×    1.93×    1.98×
selection       0.79×    1.43×    1.40×    2.04×    1.73×
copy/cast       0.72×    0.30×    0.64×    1.39×    1.04×
index-math      0.54×    0.67×    0.94×    1.16×    1.17×
dtypes          0.73×    0.70×    1.42×    2.01×    1.33×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.54×    1.39×    1.01×    0.94×    1.02×     1.16×
  sqrt         1.54×    1.16×    1.02×    1.02×    1.04×     1.14×
  copy         1.72×    4.92×    1.40×    1.07×    1.68×     1.84×
  strided      1.71×    2.46×    0.87×    1.02×    0.86×     1.26×
  bcast        1.68×    3.26×    1.07×    1.01×    1.05×     1.44×
  reversed     1.70×    2.57×    0.83×    0.97×    1.07×     1.30×
  castbuf      2.06×    3.36×    2.11×    1.38×    1.09×     1.86×
  mixbuf       1.44×    2.89×    1.57×    1.35×    1.13×     1.58×
-- reductions
  sum          2.81×    4.62×    3.24×    2.02×    1.98×     2.78×
  sum ax0      2.74×    1.78×    0.47×    0.97×    1.01×     1.17×
  sum ax1      8.02×    1.90×    0.72×    1.40×    1.74×     1.92×
  sum dt=      8.08×    4.12×    1.12×    1.10×    1.12×     2.15×
  amin         6.42×    2.77×    0.81×    0.76×    0.95×     1.60×
  cumsum       3.09×    1.36×    1.23×    1.95×    1.75×     1.77×
  any(F)      19.74×   26.21×    3.52×    1.92×    1.48×     5.53×
  any(hit)    27.40×   24.85×   24.72×   22.35×   24.12×    24.63×
-- selection
  where        0.96×    0.70×    0.88×    1.97×    1.06×     1.05×
  a[mask]      0.30×    1.51×    2.13×    2.90×    2.23×     1.44×
  a[mask]=     0.50×    3.23×    6.30×    6.20×    5.77×     3.25×
  count_nz     1.52×    2.95×    4.10×    3.32×    1.09×     2.32×
  argwhere     1.57×    3.56×    1.16×    3.01×    3.48×     2.32×
  a[idx]       0.32×    0.49×    0.40×    0.67×    1.02×     0.53×
  a[idx]=      1.78×    0.68×    0.47×    0.62×    0.86×     0.79×
-- copy/cast
  flatten      0.57×    0.11×    0.31×    2.27×    1.03×     0.54×
  astype       0.40×    0.25×    1.32×    2.14×    1.25×     0.81×
  ravel.T      0.79×    0.32×    0.66×    1.68×    0.93×     0.77×
  in-place     1.71×    0.41×    1.00×    0.98×    1.02×     0.93×
  less->b      0.65×    0.66×    0.38×    0.65×    0.98×     0.63×
-- index-math
  unravel      0.52×    0.44×    0.90×    0.97×    1.05×     0.73×
  ravel_mi     0.55×    1.03×    0.99×    1.37×    1.31×     1.00×
-- dtypes
  complex      0.75×    0.61×    0.74×    0.92×    0.92×     0.78×
  float16      0.74×    0.65×    0.54×    0.61×    0.60×     0.62×
  int8         0.70×    0.86×    7.23×   14.58×    4.25×     3.06×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▏   1.92×    52%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.54×    22%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   5.08×    20%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.69×    27%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   6.78×    15%🕐  (  1 win /  0 lose)
8op          ███████████████████▶  13.26×     8%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   5.49×    18%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   6.04×    17%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.26×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   4.89×    20%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ██████▌ ............   0.66×   152%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         █████████▏ .........   0.92×   109%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=64         ██████████▋ ........   1.07×    93%🕐  (  1 win /  0 lose)
w=256        ██████████▊ ........   1.09×    92%🕐  (  1 win /  0 lose)
w=1024       █████████████▌ .....   1.36×    74%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    640.95×   (641.0× faster, faster)
  allocate          1.08×   (1.1× faster, faster)
  overlap_copy      1.91×   (1.9× faster, faster)
  forder_out        1.62×   (1.6× faster, faster)
  zerodim           1.46×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.77×    7.45×    2.70×    4.75×    1.92×   vs chained 6× add
reuse            8.64×   13.49×    1.08×    1.93×    1.03×   vs rebuild each call
par8                 -    1.54×    6.81×    6.88×    5.19×   vs single-thread

biggest NumSharp wins: anyeh@1 27.40× · anyff@1K 26.21× · anyeh@1K 24.85× · anyeh@100K 24.72× · anyeh@10M 24.12×
most behind:           flatten@1K 0.11× · astype@1K 0.25× · bread@1 0.30× · flatten@100K 0.31× · gather@1 0.32×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.06 ✅ | 1.04 ✅ | 0.59 🟡 | 0.91 🟡 | 0.70 🟡 | 0.88 🟡 | 0.81 🟡 |
| 1M | 1.01 ✅ | 1.09 ✅ | 1.15 ✅ | 0.60 🟡 | 0.94 🟡 | 0.72 🟡 | 0.82 🟡 | 0.75 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.97 🟡 | 1.17 ✅ | 0.91 🟡 | 0.12 🔴 | 1.04 ✅ | 1.48 ✅ | 1.13 ✅ |
| 1M | 1.05 ✅ | 1.24 ✅ | 0.95 🟡 | 0.11 🔴 | 1.05 ✅ | 1.47 ✅ | 1.01 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.88 🟡 | 0.70 🟡 | 0.71 🟡 | 1.36 ✅ |
| 1M | 0.95 🟡 | 0.67 🟡 | 0.69 🟡 | 1.36 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.6021 | 0.1420 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7671 | 0.0207 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.6152 | 0.2246 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7698 | 0.0230 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7638 | 0.0233 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7652 | 0.0233 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7615 | 0.0233 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.7864 | 0.2428 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.6415 | 0.2418 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.6214 | 0.2428 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7507 | 0.0254 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.5704 | 0.2672 | 0.04 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.2179 | 0.1732 | 0.04 🔴 |
| 100K\|dec\|bcast\|sum\|ax1 | 0.4289 | 0.0240 | 0.06 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 0.8931 | 0.0607 | 0.07 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.18 ✅ | 1.20 ✅ | 1.22 ✅ | 0.70 🟡 | 1.44 ✅ | 1.42 ✅ | 1.79 ✅ | 1.58 ✅ |
| 1M | 3.49 ✅ | 3.47 ✅ | 3.49 ✅ | 2.44 ✅ | 3.25 ✅ | 3.25 ✅ | 3.81 ✅ | 3.46 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 1.00 ✅ | 1.42 ✅ | 1.80 ✅ | 1.99 ✅ | 0.90 🟡 | 1.11 ✅ | 1.59 ✅ | 1.11 ✅ | 1.89 ✅ | 1.52 ✅ | 0.75 🟡 | 0.62 🟡 | 1.96 ✅ |
| 1M | 5.42 ✅ | 5.46 ✅ | 2.54 ✅ | 2.57 ✅ | 2.68 ✅ | 2.68 ✅ | 3.83 ✅ | 3.69 ✅ | 2.52 ✅ | 2.39 ✅ | 2.47 ✅ | 3.81 ✅ | 5.13 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|pos | 0.0433 | 0.0100 | 0.23 🟠 |
| 100K\|f32\|strided\|pos | 0.0379 | 0.0103 | 0.27 🟠 |
| 100K\|u64\|strided\|pos | 0.0259 | 0.0109 | 0.42 🟠 |
| 100K\|i32\|strided\|pos | 0.0237 | 0.0109 | 0.46 🟠 |
| 100K\|f64\|C\|pos | 0.0506 | 0.0252 | 0.50 🟠 |
| 100K\|f32\|negcol\|pos | 0.0774 | 0.0405 | 0.52 🟡 |
| 100K\|u8\|negrow\|pos | 0.0439 | 0.0232 | 0.53 🟡 |
| 100K\|f64\|T\|pos | 0.0448 | 0.0237 | 0.53 🟡 |
| 100K\|i64\|strided\|pos | 0.0205 | 0.0110 | 0.54 🟡 |
| 100K\|u8\|sliced\|pos | 0.0429 | 0.0230 | 0.54 🟡 |
| 100K\|f64\|F\|pos | 0.0447 | 0.0271 | 0.61 🟡 |
| 100K\|u32\|strided\|pos | 0.0178 | 0.0110 | 0.62 🟡 |
| 100K\|f64\|negcol\|pos | 0.0833 | 0.0638 | 0.77 🟡 |
| 100K\|f32\|C\|pos | 0.0252 | 0.0196 | 0.77 🟡 |
| 100K\|u64\|C\|pos | 0.0357 | 0.0290 | 0.81 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.62 🟡 | 0.71 🟡 | 0.66 🟡 | 0.37 🟠 | 0.85 🟡 | 0.83 🟡 | 0.84 🟡 | 0.81 🟡 |
| 1M | 1.77 ✅ | 1.69 ✅ | 1.68 ✅ | 0.97 🟡 | 1.83 ✅ | 1.87 ✅ | 1.41 ✅ | 2.05 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.66 🟡 | 0.29 🟠 | 1.42 ✅ | 0.75 🟡 | 0.78 🟡 | 0.68 🟡 |
| 1M | 1.98 ✅ | 1.65 ✅ | 2.02 ✅ | 0.97 🟡 | 1.63 ✅ | 1.75 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.81 🟡 | 0.84 🟡 | 0.64 🟡 | 0.70 🟡 | 0.74 🟡 | 0.61 🟡 | 0.53 🟡 |
| 1M | 1.83 ✅ | 1.82 ✅ | 2.21 ✅ | 1.96 ✅ | 1.33 ✅ | 0.74 🟡 | 2.10 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|neg | 0.0670 | 0.0060 | 0.09 🔴 |
| 100K\|f32\|strided\|abs | 0.0658 | 0.0059 | 0.09 🔴 |
| 100K\|i64\|strided\|add | 0.2312 | 0.0258 | 0.11 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0672 | 0.0077 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1142 | 0.0136 | 0.12 🔴 |
| 100K\|f32\|strided\|add | 0.1108 | 0.0134 | 0.12 🔴 |
| 100K\|i64\|strided\|mul | 0.2314 | 0.0302 | 0.13 🔴 |
| 100K\|f64\|strided\|neg | 0.0559 | 0.0076 | 0.14 🔴 |
| 100K\|f64\|strided\|abs | 0.0545 | 0.0076 | 0.14 🔴 |
| 100K\|f32\|bcast\|copy | 0.0326 | 0.0050 | 0.15 🔴 |
| 100K\|f32\|T\|sqrt | 0.0845 | 0.0142 | 0.17 🔴 |
| 100K\|f32\|C\|neg | 0.0358 | 0.0065 | 0.18 🔴 |
| 100K\|f32\|bcast\|sqrt | 0.1033 | 0.0188 | 0.18 🔴 |
| 100K\|i32\|C\|copy | 0.0313 | 0.0059 | 0.19 🔴 |
| 100K\|f32\|sliced\|sqrt | 0.1016 | 0.0203 | 0.20 🟠 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 2.96 ✅ | 2.45 ✅ | 0.65 🟡 | 2.91 ✅ | 3.79 ✅ | 2.22 ✅ | 2.21 ✅ |
| 1-D strided a[::2] | 1.95 ✅ | 1.69 ✅ | 1.23 ✅ | 0.60 🟡 | 0.65 🟡 | 1.93 ✅ | 1.20 ✅ |
| 1-D reversed a[::-1] | 2.97 ✅ | 2.44 ✅ | 1.28 ✅ | 0.70 🟡 | 0.62 🟡 | 2.20 ✅ | 1.44 ✅ |
| array + scalar | 3.01 ✅ | 2.45 ✅ | 0.73 🟡 | 2.23 ✅ | 3.31 ✅ | 2.21 ✅ | 2.11 ✅ |
| scalar + array | 2.91 ✅ | 2.39 ✅ | 0.73 🟡 | 2.94 ✅ | 3.09 ✅ | 2.49 ✅ | 2.20 ✅ |
| mixed C + F | 2.39 ✅ | 2.26 ✅ | 0.75 🟡 | 0.99 🟡 | 0.83 🟡 | 1.80 ✅ | 1.35 ✅ |
| mixed C + T | 2.99 ✅ | 2.41 ✅ | 0.75 🟡 | 0.96 🟡 | 0.83 🟡 | 2.64 ✅ | 1.50 ✅ |
| binary broadcast +row(1,C) | 3.53 ✅ | 2.62 ✅ | 0.72 🟡 | 2.57 ✅ | 3.89 ✅ | 2.63 ✅ | 2.37 ✅ |
| binary broadcast +col(R,1) | 2.90 ✅ | 2.32 ✅ | 0.74 🟡 | 2.37 ✅ | 3.86 ✅ | 3.20 ✅ | 2.29 ✅ |
| col-broadcast unary (inner stride-0) | 3.14 ✅ | 2.01 ✅ | 0.92 🟡 | 2.21 ✅ | 3.51 ✅ | 7.71 ✅ | 2.65 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_strided|i32 | 1.2096 | 0.7228 | 0.60 🟡 |
| 1d_rev|i64 | 4.0291 | 2.5065 | 0.62 🟡 |
| 1d_strided|i64 | 2.0305 | 1.3158 | 0.65 🟡 |
| 1d_C|f16 | 4.8542 | 3.1463 | 0.65 🟡 |
| 1d_rev|i32 | 1.9800 | 1.3912 | 0.70 🟡 |
| bcast_row|f16 | 9.5472 | 6.9126 | 0.72 🟡 |
| scalar_lhs|f16 | 9.3919 | 6.8484 | 0.73 🟡 |
| scalar_rhs|f16 | 9.3420 | 6.8644 | 0.73 🟡 |
| bcast_col|f16 | 9.3589 | 6.9455 | 0.74 🟡 |
| mix_C_T|f16 | 10.1275 | 7.6230 | 0.75 🟡 |
| mix_C_F|f16 | 10.1880 | 7.6716 | 0.75 🟡 |
| mix_C_T|i64 | 4.1313 | 3.4157 | 0.83 🟡 |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **366 / 1568** comparable cells lag (<1.0); **1202** win (≥1.0).
- **🔴 <0.2** — 12 cells. Top: 8× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 2× * → bool; 2× same-type diagonal (copy)
- **🟠 0.2–0.5** — 144 cells. Top: 68× int → sub-word (narrow); 33× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 17× * → bool
- **🟡 0.5–1.0** — 210 cells. Top: 51× int → sub-word (narrow); 26× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 12× * → bool

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **1.00**, `f64`→narrow **0.88**, `f16`→narrow **2.71**, `c128`→narrow **0.75**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 2.02 ✅ | 2.06 ✅ | 2.14 ✅ | 2.10 ✅ | 2.14 ✅ | 1.49 ✅ | 1.28 ✅ | 0.51 🟡 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.27 ✅ | 2.22 ✅ | 2.14 ✅ | 2.35 ✅ | 2.28 ✅ | 1.63 ✅ | 1.59 ✅ | 1.28 ✅ | 1.21 ✅ | 2.02 ✅ | 1.67 ✅ | 1.19 ✅ | 1.16 ✅ | nan ? | 1.00 ✅ |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.46 ✅ | 1.23 ✅ | 1.21 ✅ | 1.23 ✅ | 1.21 ✅ | 1.75 ✅ | 1.42 ✅ | 2.17 ✅ | 1.87 ✅ | 1.23 ✅ | 1.40 ✅ | 1.61 ✅ | 2.13 ✅ | nan ? | 3.12 ✅ |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 3.77✅ | 0.97🟡 | 1.05✅ | 0.88🟡 | 0.89🟡 | 1.50✅ | 1.45✅ | 2.28✅ | 2.51✅ | 0.95🟡 | 0.25🟠 | 1.79✅ | 2.41✅ | — | 3.37✅ |
| **u8** | 4.76✅ | 1.08✅ | 2.37✅ | 1.37✅ | 1.39✅ | 1.83✅ | 2.84✅ | 4.15✅ | 4.23✅ | 2.64✅ | 2.52✅ | 2.06✅ | 3.02✅ | — | 4.86✅ |
| **i8** | 4.73✅ | 3.85✅ | 1.16✅ | 2.51✅ | 2.16✅ | 2.78✅ | 2.70✅ | 3.53✅ | 3.34✅ | 2.38✅ | 2.50✅ | 1.92✅ | 2.78✅ | — | 4.86✅ |
| **i16** | 4.80✅ | 3.78✅ | 4.66✅ | 1.43✅ | 2.24✅ | 2.80✅ | 2.73✅ | 3.51✅ | 3.86✅ | 2.06✅ | 2.46✅ | 3.11✅ | 2.79✅ | — | 4.49✅ |
| **u16** | 4.70✅ | 4.63✅ | 2.23✅ | 2.28✅ | 1.46✅ | 2.72✅ | 2.64✅ | 4.54✅ | 4.44✅ | 1.17✅ | 2.47✅ | 2.87✅ | 2.64✅ | — | 4.23✅ |
| **i32** | 0.97🟡 | 0.40🟠 | 0.70🟡 | 1.69✅ | 1.99✅ | 2.07✅ | 2.94✅ | 3.83✅ | 4.10✅ | 3.12✅ | 7.36✅ | 2.16✅ | 3.49✅ | — | 4.29✅ |
| **u32** | 2.25✅ | 1.03✅ | 0.69🟡 | 1.69✅ | 1.95✅ | 2.91✅ | 2.20✅ | 3.93✅ | 3.98✅ | 1.97✅ | 2.27✅ | 1.81✅ | 2.48✅ | — | 4.33✅ |
| **i64** | 0.98🟡 | 1.10✅ | 0.69🟡 | 1.25✅ | 1.14✅ | 2.59✅ | 2.58✅ | 2.90✅ | 3.86✅ | 1.42✅ | 1.29✅ | 2.13✅ | 2.74✅ | — | 3.38✅ |
| **u64** | 0.97🟡 | 0.76🟡 | 0.36🟠 | 0.62🟡 | 0.88🟡 | 2.67✅ | 2.41✅ | 3.22✅ | 2.65✅ | 1.38✅ | 1.62✅ | 1.47✅ | 1.72✅ | — | 2.70✅ |
| **char** | 3.90✅ | 4.84✅ | 4.70✅ | 1.85✅ | 1.13✅ | 2.00✅ | 1.97✅ | 3.26✅ | 3.39✅ | 1.29✅ | 2.44✅ | 1.76✅ | 2.42✅ | — | 4.44✅ |
| **f16** | 9.16✅ | 9.68✅ | 7.07✅ | 3.39✅ | 3.49✅ | 4.05✅ | 1.25✅ | 2.15✅ | 0.62🟡 | 3.53✅ | 1.04✅ | 2.20✅ | 0.67🟡 | — | 1.04✅ |
| **f32** | 2.97✅ | 1.68✅ | 1.72✅ | 1.52✅ | 1.51✅ | 1.81✅ | 0.59🟡 | 0.70🟡 | 0.48🟠 | 1.45✅ | 2.31✅ | 1.43✅ | 2.60✅ | — | 2.74✅ |
| **f64** | 1.82✅ | 1.80✅ | 1.84✅ | 1.69✅ | 1.40✅ | 2.61✅ | 1.02✅ | 1.11✅ | 0.72🟡 | 1.32✅ | 1.51✅ | 2.35✅ | 2.69✅ | — | 3.07✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.81🟡 | 0.66🟡 | 0.63🟡 | 1.39✅ | 1.34✅ | 2.06✅ | 1.11✅ | 1.19✅ | 0.74🟡 | 1.36✅ | 1.36✅ | 1.59✅ | 2.23✅ | — | 2.91✅ |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.19🔴 | 0.56🟡 | 0.67🟡 | 0.88🟡 | 0.85🟡 | 1.46✅ | 1.46✅ | 2.46✅ | 2.39✅ | 0.86🟡 | 0.25🟠 | 1.82✅ | 2.44✅ | — | 3.88✅ |
| **u8** | 5.52✅ | 1.23✅ | 5.95✅ | 2.71✅ | 2.56✅ | 2.39✅ | 1.98✅ | 3.79✅ | 4.01✅ | 2.31✅ | 2.51✅ | 1.91✅ | 2.92✅ | — | 4.91✅ |
| **i8** | 4.60✅ | 3.78✅ | 1.02✅ | 2.55✅ | 2.29✅ | 2.66✅ | 2.81✅ | 3.53✅ | 3.47✅ | 2.17✅ | 2.48✅ | 1.96✅ | 2.90✅ | — | 4.62✅ |
| **i16** | 4.52✅ | 4.08✅ | 5.91✅ | 1.49✅ | 2.36✅ | 3.01✅ | 2.57✅ | 4.32✅ | 4.59✅ | 2.20✅ | 2.39✅ | 3.06✅ | 2.62✅ | — | 4.54✅ |
| **u16** | 4.12✅ | 4.81✅ | 5.15✅ | 2.26✅ | 1.38✅ | 2.63✅ | 2.79✅ | 4.54✅ | 4.64✅ | 1.46✅ | 2.47✅ | 3.02✅ | 2.68✅ | — | 4.22✅ |
| **i32** | 2.24✅ | 0.74🟡 | 0.71🟡 | 2.00✅ | 2.07✅ | 2.08✅ | 2.99✅ | 3.97✅ | 3.99✅ | 1.95✅ | 2.54✅ | 2.29✅ | 3.34✅ | — | 4.53✅ |
| **u32** | 2.30✅ | 1.04✅ | 0.72🟡 | 1.68✅ | 2.01✅ | 2.95✅ | 2.09✅ | 3.79✅ | 3.75✅ | 1.89✅ | 2.29✅ | 1.75✅ | 2.32✅ | — | 4.29✅ |
| **i64** | 1.01✅ | 1.11✅ | 0.71🟡 | 1.36✅ | 1.19✅ | 2.58✅ | 2.53✅ | 2.70✅ | 3.39✅ | 1.42✅ | 1.30✅ | 2.29✅ | 2.99✅ | — | 3.63✅ |
| **u64** | 1.00🟡 | 0.78🟡 | 1.02✅ | 1.09✅ | 1.20✅ | 2.62✅ | 2.57✅ | 3.22✅ | 2.94✅ | 1.45✅ | 1.62✅ | 1.51✅ | 1.79✅ | — | 2.94✅ |
| **char** | 3.75✅ | 4.43✅ | 4.64✅ | 2.11✅ | 1.31✅ | 2.12✅ | 2.06✅ | 3.47✅ | 3.35✅ | 1.26✅ | 2.34✅ | 1.66✅ | 2.23✅ | — | 4.51✅ |
| **f16** | 8.60✅ | 3.70✅ | 3.72✅ | 2.99✅ | 3.30✅ | 4.05✅ | 1.26✅ | 2.16✅ | 0.64🟡 | 3.27✅ | 0.88🟡 | 2.20✅ | 0.76🟡 | — | 1.01✅ |
| **f32** | 2.91✅ | 1.57✅ | 1.67✅ | 1.60✅ | 1.61✅ | 2.07✅ | 0.60🟡 | 0.69🟡 | 0.57🟡 | 1.62✅ | 2.52✅ | 2.12✅ | 3.87✅ | — | 4.05✅ |
| **f64** | 1.81✅ | 1.85✅ | 1.69✅ | 1.58✅ | 1.47✅ | 2.59✅ | 1.04✅ | 1.13✅ | 0.72🟡 | 1.43✅ | 1.54✅ | 2.39✅ | 2.69✅ | — | 3.26✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.82🟡 | 1.31✅ | 1.27✅ | 1.36✅ | 1.27✅ | 1.90✅ | 1.07✅ | 1.10✅ | 0.72🟡 | 1.42✅ | 1.37✅ | 1.62✅ | 2.21✅ | — | 3.01✅ |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.66✅ | 1.02✅ | 1.11✅ | 1.05✅ | 1.07✅ | 1.78✅ | 1.78✅ | 2.67✅ | 2.78✅ | 1.05✅ | 0.26🟠 | 2.13✅ | 2.85✅ | — | 3.48✅ |
| **u8** | 5.31✅ | 1.16✅ | 5.41✅ | 2.65✅ | 2.62✅ | 2.79✅ | 2.85✅ | 4.34✅ | 4.11✅ | 2.70✅ | 2.46✅ | 2.03✅ | 3.02✅ | — | 4.84✅ |
| **i8** | 4.79✅ | 3.88✅ | 1.17✅ | 2.71✅ | 2.21✅ | 2.79✅ | 2.82✅ | 3.50✅ | 3.42✅ | 2.26✅ | 2.41✅ | 1.98✅ | 2.92✅ | — | 4.46✅ |
| **i16** | 4.40✅ | 3.96✅ | 6.41✅ | 1.45✅ | 2.41✅ | 3.07✅ | 2.70✅ | 3.88✅ | 4.66✅ | 2.26✅ | 2.45✅ | 3.08✅ | 2.88✅ | — | 4.58✅ |
| **u16** | 4.40✅ | 4.20✅ | 5.24✅ | 2.28✅ | 1.37✅ | 2.65✅ | 2.42✅ | 3.77✅ | 4.10✅ | 1.36✅ | 2.40✅ | 2.99✅ | 2.82✅ | — | 4.65✅ |
| **i32** | 2.33✅ | 0.79🟡 | 0.69🟡 | 1.72✅ | 2.06✅ | 2.24✅ | 2.83✅ | 3.09✅ | 3.97✅ | 2.09✅ | 2.50✅ | 2.22✅ | 3.89✅ | — | 4.46✅ |
| **u32** | 2.31✅ | 1.03✅ | 0.66🟡 | 1.71✅ | 1.87✅ | 2.78✅ | 2.00✅ | 3.67✅ | 3.76✅ | 2.00✅ | 2.32✅ | 1.93✅ | 2.45✅ | — | 4.26✅ |
| **i64** | 0.99🟡 | 1.10✅ | 0.72🟡 | 1.33✅ | 1.22✅ | 2.70✅ | 2.71✅ | 2.68✅ | 3.86✅ | 1.44✅ | 1.31✅ | 1.84✅ | 2.62✅ | — | 3.36✅ |
| **u64** | 1.03✅ | 0.74🟡 | 1.00✅ | 1.11✅ | 1.16✅ | 2.61✅ | 2.45✅ | 3.83✅ | 2.57✅ | 1.39✅ | 1.60✅ | 1.49✅ | 1.78✅ | — | 2.75✅ |
| **char** | 3.79✅ | 4.87✅ | 4.96✅ | 2.15✅ | 1.25✅ | 1.98✅ | 1.98✅ | 3.47✅ | 3.41✅ | 1.36✅ | 2.40✅ | 1.75✅ | 2.23✅ | — | 4.31✅ |
| **f16** | 10.60✅ | 3.65✅ | 3.76✅ | 3.11✅ | 3.16✅ | 4.09✅ | 1.26✅ | 2.14✅ | 0.64🟡 | 3.33✅ | 0.91🟡 | 2.31✅ | 0.74🟡 | — | 1.01✅ |
| **f32** | 3.31✅ | 1.70✅ | 1.70✅ | 1.89✅ | 1.93✅ | 2.75✅ | 0.81🟡 | 1.00🟡 | 0.68🟡 | 1.88✅ | 2.44✅ | 1.97✅ | 4.06✅ | — | 3.57✅ |
| **f64** | 1.84✅ | 1.78✅ | 1.78✅ | 1.63✅ | 1.45✅ | 2.51✅ | 1.04✅ | 1.16✅ | 0.73🟡 | 1.48✅ | 1.54✅ | 2.54✅ | 2.37✅ | — | 3.10✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.81🟡 | 1.26✅ | 1.33✅ | 1.35✅ | 1.41✅ | 2.18✅ | 1.11✅ | 1.25✅ | 0.79🟡 | 1.42✅ | 1.37✅ | 1.51✅ | 1.91✅ | — | 2.91✅ |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.27✅ | 0.65🟡 | 0.75🟡 | 0.71🟡 | 0.66🟡 | 1.07✅ | 1.21✅ | 3.13✅ | 3.02✅ | 1.22✅ | 0.32🟠 | 2.11✅ | 3.06✅ | — | 4.78✅ |
| **u8** | 4.51✅ | 1.16✅ | 5.76✅ | 2.62✅ | 2.63✅ | 2.61✅ | 2.97✅ | 2.79✅ | 2.72✅ | 2.54✅ | 2.42✅ | 1.94✅ | 2.94✅ | — | 4.75✅ |
| **i8** | 4.23✅ | 3.64✅ | 0.96🟡 | 2.63✅ | 2.17✅ | 2.78✅ | 2.87✅ | 2.89✅ | 3.05✅ | 2.14✅ | 2.49✅ | 1.97✅ | 2.83✅ | — | 4.46✅ |
| **i16** | 4.88✅ | 4.01✅ | 5.44✅ | 1.56✅ | 1.99✅ | 2.98✅ | 2.71✅ | 4.16✅ | 4.01✅ | 1.95✅ | 2.39✅ | 3.04✅ | 3.09✅ | — | 4.45✅ |
| **u16** | 4.28✅ | 4.91✅ | 5.03✅ | 1.91✅ | 1.54✅ | 2.68✅ | 2.62✅ | 3.41✅ | 4.14✅ | 1.59✅ | 2.38✅ | 2.65✅ | 3.43✅ | — | 4.37✅ |
| **i32** | 2.09✅ | 1.87✅ | 1.77✅ | 1.66✅ | 2.08✅ | 2.48✅ | 2.67✅ | 4.07✅ | 4.36✅ | 2.00✅ | 2.42✅ | 1.66✅ | 4.12✅ | — | 4.24✅ |
| **u32** | 1.93✅ | 2.56✅ | 1.84✅ | 1.72✅ | 2.04✅ | 2.57✅ | 2.17✅ | 3.86✅ | 3.85✅ | 1.86✅ | 2.20✅ | 2.12✅ | 3.18✅ | — | 4.49✅ |
| **i64** | 0.94🟡 | 1.46✅ | 0.94🟡 | 1.55✅ | 1.40✅ | 2.49✅ | 2.49✅ | 2.74✅ | 2.92✅ | 1.34✅ | 1.30✅ | 1.63✅ | 2.23✅ | — | 3.17✅ |
| **u64** | 0.98🟡 | 1.00✅ | 1.41✅ | 1.31✅ | 1.35✅ | 2.55✅ | 2.59✅ | 2.74✅ | 2.68✅ | 1.31✅ | 1.60✅ | 1.43✅ | 1.62✅ | — | 2.81✅ |
| **char** | 3.21✅ | 4.69✅ | 4.63✅ | 1.78✅ | 1.43✅ | 1.94✅ | 1.93✅ | 3.06✅ | 3.23✅ | 1.44✅ | 2.33✅ | 1.74✅ | 2.24✅ | — | 4.01✅ |
| **f16** | 8.81✅ | 3.50✅ | 3.57✅ | 3.00✅ | 3.07✅ | 3.74✅ | 1.27✅ | 2.03✅ | 0.63🟡 | 3.08✅ | 1.05✅ | 2.01✅ | 0.73🟡 | — | 1.03✅ |
| **f32** | 2.97✅ | 1.54✅ | 1.56✅ | 1.77✅ | 1.89✅ | 2.46✅ | 0.81🟡 | 0.97🟡 | 0.68🟡 | 1.89✅ | 2.39✅ | 2.40✅ | 3.79✅ | — | 2.97✅ |
| **f64** | 1.68✅ | 1.60✅ | 1.62✅ | 1.53✅ | 1.34✅ | 2.58✅ | 1.04✅ | 1.13✅ | 0.71🟡 | 1.37✅ | 1.53✅ | 2.36✅ | 2.63✅ | — | 3.76✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.80🟡 | 1.26✅ | 1.17✅ | 1.35✅ | 1.36✅ | 1.81✅ | 1.10✅ | 0.98🟡 | 0.61🟡 | 1.15✅ | 1.30✅ | 1.27✅ | 1.39✅ | — | 2.16✅ |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 3.27✅ | 1.22✅ | 1.35✅ | 1.17✅ | 1.19✅ | 1.90✅ | 1.86✅ | 3.06✅ | 2.90✅ | 1.19✅ | 0.31🟠 | 2.07✅ | 2.90✅ | — | 4.91✅ |
| **u8** | 1.75✅ | 1.17✅ | 5.63✅ | 2.62✅ | 2.67✅ | 2.61✅ | 2.58✅ | 2.82✅ | 2.98✅ | 2.55✅ | 2.41✅ | 1.96✅ | 2.93✅ | — | 4.52✅ |
| **i8** | 4.32✅ | 3.88✅ | 0.88🟡 | 2.58✅ | 2.10✅ | 2.92✅ | 2.26✅ | 2.76✅ | 2.82✅ | 2.21✅ | 2.43✅ | 1.90✅ | 2.79✅ | — | 4.55✅ |
| **i16** | 4.47✅ | 3.54✅ | 6.25✅ | 1.56✅ | 2.02✅ | 2.61✅ | 2.64✅ | 3.84✅ | 4.23✅ | 2.14✅ | 2.38✅ | 3.12✅ | 3.04✅ | — | 4.22✅ |
| **u16** | 4.25✅ | 4.84✅ | 4.23✅ | 1.83✅ | 1.46✅ | 2.65✅ | 2.61✅ | 3.94✅ | 4.17✅ | 1.56✅ | 2.36✅ | 2.77✅ | 3.22✅ | — | 4.34✅ |
| **i32** | 2.07✅ | 1.94✅ | 1.86✅ | 1.73✅ | 1.97✅ | 2.74✅ | 2.73✅ | 3.28✅ | 4.03✅ | 1.90✅ | 2.43✅ | 1.75✅ | 3.66✅ | — | 4.32✅ |
| **u32** | 2.03✅ | 2.44✅ | 1.87✅ | 1.69✅ | 2.10✅ | 2.45✅ | 2.38✅ | 3.44✅ | 3.46✅ | 1.91✅ | 2.27✅ | 2.00✅ | 2.89✅ | — | 4.14✅ |
| **i64** | 0.94🟡 | 1.37✅ | 1.03✅ | 1.60✅ | 1.34✅ | 2.33✅ | 2.34✅ | 2.87✅ | 3.14✅ | 1.44✅ | 1.31✅ | 1.62✅ | 2.43✅ | — | 2.82✅ |
| **u64** | 0.97🟡 | 1.04✅ | 1.43✅ | 1.39✅ | 1.40✅ | 2.48✅ | 2.57✅ | 2.96✅ | 2.77✅ | 1.39✅ | 1.59✅ | 1.35✅ | 1.66✅ | — | 2.68✅ |
| **char** | 4.24✅ | 4.85✅ | 4.90✅ | 2.08✅ | 1.84✅ | 2.00✅ | 1.98✅ | 2.51✅ | 2.70✅ | 1.82✅ | 2.43✅ | 1.73✅ | 2.23✅ | — | 3.96✅ |
| **f16** | 8.35✅ | 3.47✅ | 3.53✅ | 2.88✅ | 3.11✅ | 3.98✅ | 1.29✅ | 2.07✅ | 0.63🟡 | 2.97✅ | 1.00✅ | 2.03✅ | 0.74🟡 | — | 1.02✅ |
| **f32** | 2.72✅ | 1.49✅ | 1.49✅ | 1.83✅ | 1.81✅ | 2.52✅ | 0.77🟡 | 0.94🟡 | 0.67🟡 | 1.82✅ | 2.39✅ | 2.52✅ | 3.40✅ | — | 3.53✅ |
| **f64** | 1.63✅ | 1.67✅ | 1.59✅ | 1.65✅ | 1.41✅ | 2.71✅ | 1.05✅ | 1.08✅ | 0.73🟡 | 1.47✅ | 1.51✅ | 2.51✅ | 2.77✅ | — | 3.36✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.77🟡 | 1.11✅ | 1.20✅ | 1.51✅ | 1.40✅ | 1.90✅ | 1.05✅ | 1.12✅ | 0.75🟡 | 1.51✅ | 1.35✅ | 1.60✅ | 2.12✅ | — | 2.40✅ |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 1.20✅ | 1.25✅ | 1.23✅ | 1.16✅ | 1.17✅ | 1.80✅ | 1.86✅ | 2.90✅ | 3.06✅ | 1.16✅ | 0.31🟠 | 2.08✅ | 2.93✅ | — | 4.67✅ |
| **u8** | 1.07✅ | 5.30✅ | 4.34✅ | 2.66✅ | 2.53✅ | 2.22✅ | 2.11✅ | 4.13✅ | 4.26✅ | 2.36✅ | 1.81✅ | 2.03✅ | 3.04✅ | — | 4.80✅ |
| **i8** | 1.03✅ | 5.45✅ | 4.48✅ | 2.14✅ | 2.38✅ | 1.96✅ | 2.16✅ | 3.38✅ | 3.41✅ | 2.45✅ | 1.79✅ | 1.67✅ | 2.96✅ | — | 4.69✅ |
| **i16** | 2.65✅ | 3.47✅ | 2.35✅ | 2.30✅ | 2.34✅ | 2.10✅ | 2.13✅ | 3.10✅ | 3.06✅ | 2.26✅ | 1.89✅ | 2.18✅ | 2.73✅ | — | 4.01✅ |
| **u16** | 2.40✅ | 3.35✅ | 3.41✅ | 1.94✅ | 2.27✅ | 2.22✅ | 2.21✅ | 3.59✅ | 3.50✅ | 2.17✅ | 1.83✅ | 1.90✅ | 2.88✅ | — | 4.14✅ |
| **i32** | 0.35🟠 | 0.54🟡 | 0.51🟡 | 0.75🟡 | 0.66🟡 | 1.79✅ | 1.81✅ | 2.69✅ | 2.99✅ | 0.63🟡 | 1.89✅ | 2.06✅ | 2.53✅ | — | 4.06✅ |
| **u32** | 0.44🟠 | 0.39🟠 | 0.52🟡 | 0.63🟡 | 0.74🟡 | 1.62✅ | 1.66✅ | 3.20✅ | 3.22✅ | 0.73🟡 | 1.79✅ | 1.83✅ | 2.61✅ | — | 3.98✅ |
| **i64** | 0.23🟠 | 0.39🟠 | 0.39🟠 | 0.56🟡 | 0.47🟠 | 0.80🟡 | 0.83🟡 | 2.27✅ | 2.33✅ | 0.48🟠 | 1.10✅ | 2.07✅ | 2.85✅ | — | 2.84✅ |
| **u64** | 0.40🟠 | 0.39🟠 | 0.39🟠 | 0.56🟡 | 0.57🟡 | 0.80🟡 | 0.80🟡 | 2.24✅ | 2.16✅ | 0.55🟡 | 1.38✅ | 1.21✅ | 1.82✅ | — | 2.76✅ |
| **char** | 2.15✅ | 2.61✅ | 2.70✅ | 1.99✅ | 2.36✅ | 2.02✅ | 2.07✅ | 3.02✅ | 2.11✅ | 2.57✅ | 1.88✅ | 1.74✅ | 2.20✅ | — | 4.12✅ |
| **f16** | 5.69✅ | 2.03✅ | 2.12✅ | 1.96✅ | 1.88✅ | 2.54✅ | 1.13✅ | 1.64✅ | 0.65🟡 | 1.93✅ | 1.63✅ | 1.57✅ | 0.69🟡 | — | 1.01✅ |
| **f32** | 0.77🟡 | 0.40🟠 | 0.50🟡 | 0.66🟡 | 0.63🟡 | 1.07✅ | 0.51🟡 | 0.97🟡 | 0.58🟡 | 0.59🟡 | 1.86✅ | 1.58✅ | 1.95✅ | — | 2.77✅ |
| **f64** | 0.58🟡 | 0.30🟠 | 0.30🟠 | 0.55🟡 | 0.49🟠 | 2.38✅ | 0.68🟡 | 1.09✅ | 0.69🟡 | 0.48🟠 | 1.21✅ | 1.37✅ | 2.35✅ | — | 2.89✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.82🟡 | 0.39🟠 | 0.38🟠 | 0.65🟡 | 0.59🟡 | 0.93🟡 | 0.76🟡 | 1.18✅ | 0.77🟡 | 0.63🟡 | 1.22✅ | 1.67✅ | 2.11✅ | — | 2.50✅ |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.75🟡 | 0.78🟡 | 0.76🟡 | 1.42✅ | 1.58✅ | 1.43✅ | 1.35✅ | 2.62✅ | 2.55✅ | 1.39✅ | 0.31🟠 | 1.44✅ | 2.55✅ | — | 3.96✅ |
| **u8** | 0.87🟡 | 3.80✅ | 2.81✅ | 3.20✅ | 2.97✅ | 1.44✅ | 1.51✅ | 2.49✅ | 2.70✅ | 3.03✅ | 1.78✅ | 1.54✅ | 2.47✅ | — | 4.19✅ |
| **i8** | 0.83🟡 | 3.22✅ | 2.51✅ | 2.35✅ | 2.98✅ | 1.36✅ | 1.39✅ | 2.63✅ | 2.46✅ | 2.87✅ | 1.74✅ | 1.46✅ | 2.68✅ | — | 5.02✅ |
| **i16** | 1.94✅ | 2.50✅ | 1.85✅ | 2.57✅ | 2.80✅ | 1.35✅ | 1.48✅ | 2.60✅ | 2.45✅ | 2.45✅ | 1.73✅ | 1.36✅ | 2.37✅ | — | 3.33✅ |
| **u16** | 1.99✅ | 2.47✅ | 2.51✅ | 1.97✅ | 2.50✅ | 1.37✅ | 1.46✅ | 2.62✅ | 2.88✅ | 2.53✅ | 1.76✅ | 1.44✅ | 2.52✅ | — | 4.53✅ |
| **i32** | 0.42🟠 | 0.54🟡 | 0.53🟡 | 0.51🟡 | 0.36🟠 | 1.21✅ | 1.23✅ | 2.55✅ | 2.76✅ | 0.37🟠 | 1.78✅ | 1.34✅ | 2.16✅ | — | 3.63✅ |
| **u32** | 0.41🟠 | 0.37🟠 | 0.52🟡 | 0.36🟠 | 0.51🟡 | 1.17✅ | 1.20✅ | 2.33✅ | 2.44✅ | 0.52🟡 | 1.67✅ | 1.32✅ | 2.33✅ | — | 3.62✅ |
| **i64** | 0.43🟠 | 0.44🟠 | 0.44🟠 | 0.41🟠 | 0.30🟠 | 0.74🟡 | 0.75🟡 | 2.43✅ | 2.29✅ | 0.31🟠 | 1.00✅ | 1.37✅ | 2.33✅ | — | 2.57✅ |
| **u64** | 0.40🟠 | 0.41🟠 | 0.42🟠 | 0.40🟠 | 0.41🟠 | 0.75🟡 | 0.74🟡 | 2.11✅ | 2.11✅ | 0.39🟠 | 1.25✅ | 1.06✅ | 1.88✅ | — | 2.47✅ |
| **char** | 1.89✅ | 2.34✅ | 2.38✅ | 1.85✅ | 2.60✅ | 1.56✅ | 1.57✅ | 2.72✅ | 2.67✅ | 2.57✅ | 1.71✅ | 1.39✅ | 2.37✅ | — | 3.90✅ |
| **f16** | 4.70✅ | 1.76✅ | 1.83✅ | 2.04✅ | 1.92✅ | 1.81✅ | 0.97🟡 | 1.57✅ | 0.65🟡 | 1.99✅ | 2.71✅ | 1.24✅ | 0.70🟡 | — | 0.93🟡 |
| **f32** | 0.72🟡 | 0.32🟠 | 0.47🟠 | 0.34🟠 | 0.33🟠 | 0.82🟡 | 0.44🟠 | 1.00✅ | 0.56🟡 | 0.36🟠 | 1.76✅ | 1.14✅ | 2.02✅ | — | 3.12✅ |
| **f64** | 0.58🟡 | 0.36🟠 | 0.37🟠 | 0.40🟠 | 0.36🟠 | 0.55🟡 | 0.65🟡 | 1.16✅ | 0.69🟡 | 0.37🟠 | 1.15✅ | 1.02✅ | 2.18✅ | — | 2.51✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.35🟠 | 0.59🟡 | 0.59🟡 | 0.55🟡 | 0.55🟡 | 0.86🟡 | 0.74🟡 | 1.26✅ | 0.61🟡 | 0.52🟡 | 1.03✅ | 1.25✅ | 1.82✅ | — | 2.36✅ |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.08🔴 | 0.31🟠 | 0.31🟠 | 0.42🟠 | 0.43🟠 | 0.69🟡 | 0.68🟡 | 1.26✅ | 1.25✅ | 0.41🟠 | 0.31🟠 | 0.74🟡 | 1.12✅ | — | 2.09✅ |
| **u8** | 0.30🟠 | 0.08🔴 | 0.27🟠 | 0.41🟠 | 0.42🟠 | 0.62🟡 | 0.62🟡 | 1.18✅ | 1.16✅ | 0.41🟠 | 0.58🟡 | 0.72🟡 | 1.18✅ | — | 1.93✅ |
| **i8** | 0.25🟠 | 0.35🟠 | 0.05🔴 | 0.35🟠 | 0.42🟠 | 0.63🟡 | 0.66🟡 | 1.16✅ | 1.16✅ | 0.41🟠 | 0.57🟡 | 0.70🟡 | 1.19✅ | — | 1.97✅ |
| **i16** | 0.27🟠 | 0.31🟠 | 0.22🟠 | 0.30🟠 | 0.42🟠 | 0.64🟡 | 0.64🟡 | 1.17✅ | 1.19✅ | 0.42🟠 | 0.56🟡 | 0.67🟡 | 1.10✅ | — | 2.01✅ |
| **u16** | 0.27🟠 | 0.30🟠 | 0.31🟠 | 0.37🟠 | 0.29🟠 | 0.64🟡 | 0.62🟡 | 1.15✅ | 1.14✅ | 0.29🟠 | 0.56🟡 | 0.67🟡 | 1.11✅ | — | 2.00✅ |
| **i32** | 0.23🟠 | 0.29🟠 | 0.29🟠 | 0.40🟠 | 0.35🟠 | 0.54🟡 | 0.58🟡 | 1.13✅ | 1.16✅ | 0.35🟠 | 0.62🟡 | 0.59🟡 | 1.14✅ | — | 1.86✅ |
| **u32** | 0.24🟠 | 0.21🟠 | 0.30🟠 | 0.35🟠 | 0.41🟠 | 0.55🟡 | 0.53🟡 | 1.10✅ | 1.13✅ | 0.41🟠 | 0.58🟡 | 0.60🟡 | 1.15✅ | — | 1.95✅ |
| **i64** | 0.28🟠 | 0.26🟠 | 0.25🟠 | 0.33🟠 | 0.31🟠 | 0.54🟡 | 0.58🟡 | 1.10✅ | 1.12✅ | 0.30🟠 | 0.53🟡 | 0.51🟡 | 1.16✅ | — | 1.95✅ |
| **u64** | 0.28🟠 | 0.25🟠 | 0.26🟠 | 0.36🟠 | 0.34🟠 | 0.56🟡 | 0.58🟡 | 1.15✅ | 1.16✅ | 0.35🟠 | 0.53🟡 | 0.64🟡 | 1.26✅ | — | 1.89✅ |
| **char** | 0.27🟠 | 0.30🟠 | 0.31🟠 | 0.36🟠 | 0.29🟠 | 0.68🟡 | 0.66🟡 | 1.05✅ | 0.86🟡 | 0.22🟠 | 0.56🟡 | 0.47🟠 | 0.77🟡 | — | 1.30✅ |
| **f16** | 0.71🟡 | 0.40🟠 | 0.38🟠 | 0.41🟠 | 0.41🟠 | 0.46🟠 | 0.47🟠 | 0.51🟡 | 0.62🟡 | 0.42🟠 | 0.20🟠 | 0.44🟠 | 0.68🟡 | — | 0.99🟡 |
| **f32** | 0.34🟠 | 0.12🔴 | 0.15🔴 | 0.21🟠 | 0.21🟠 | 0.41🟠 | 0.42🟠 | 0.86🟡 | 0.71🟡 | 0.21🟠 | 0.70🟡 | 0.55🟡 | 1.09✅ | — | 1.81✅ |
| **f64** | 0.40🟠 | 0.12🔴 | 0.12🔴 | 0.26🟠 | 0.22🟠 | 0.47🟠 | 0.48🟠 | 1.01✅ | 0.79🟡 | 0.23🟠 | 0.64🟡 | 0.48🟠 | 1.13✅ | — | 1.80✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.35🟠 | 0.10🔴 | 0.09🔴 | 0.22🟠 | 0.18🔴 | 0.34🟠 | 0.31🟠 | 1.10✅ | 0.51🟡 | 0.20🔴 | 0.65🟡 | 0.41🟠 | 1.10✅ | — | 1.65✅ |

## Lagging cells (<1.0) — the worklist  (366 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| i8\|bcast\|i8 | 1.5608 | 0.0849 | 0.05 🔴 |
| u8\|bcast\|u8 | 1.5429 | 0.1167 | 0.08 🔴 |
| bool\|bcast\|bool | 1.5743 | 0.1266 | 0.08 🔴 |
| c128\|bcast\|i8 | 4.0364 | 0.3833 | 0.09 🔴 |
| c128\|bcast\|u8 | 3.8167 | 0.3859 | 0.10 🔴 |
| f64\|bcast\|u8 | 3.2523 | 0.3770 | 0.12 🔴 |
| f64\|bcast\|i8 | 3.2777 | 0.3801 | 0.12 🔴 |
| f32\|bcast\|u8 | 3.3015 | 0.3844 | 0.12 🔴 |
| f32\|bcast\|i8 | 3.3156 | 0.5018 | 0.15 🔴 |
| c128\|bcast\|u16 | 4.3163 | 0.7712 | 0.18 🔴 |
| bool\|F\|bool | 0.0774 | 0.0144 | 0.19 🔴 |
| c128\|bcast\|char | 3.9601 | 0.7734 | 0.20 🔴 |
| f16\|bcast\|f16 | 2.1558 | 0.4331 | 0.20 🟠 |
| f32\|bcast\|i16 | 3.6410 | 0.7674 | 0.21 🟠 |
| f32\|bcast\|u16 | 3.6420 | 0.7682 | 0.21 🟠 |
| u32\|bcast\|u8 | 1.7715 | 0.3758 | 0.21 🟠 |
| f32\|bcast\|char | 3.6021 | 0.7680 | 0.21 🟠 |
| char\|bcast\|char | 2.1072 | 0.4578 | 0.22 🟠 |
| f64\|bcast\|u16 | 3.4848 | 0.7659 | 0.22 🟠 |
| i16\|bcast\|i8 | 1.6813 | 0.3781 | 0.22 🟠 |
| c128\|bcast\|i16 | 4.0480 | 0.9107 | 0.22 🟠 |
| f64\|bcast\|char | 3.3940 | 0.7640 | 0.23 🟠 |
| i32\|bcast\|bool | 1.9221 | 0.4351 | 0.23 🟠 |
| i64\|negcol\|bool | 1.5082 | 0.3496 | 0.23 🟠 |
| u32\|bcast\|bool | 1.8156 | 0.4391 | 0.24 🟠 |
| bool\|C\|f16 | 6.8871 | 1.6948 | 0.25 🟠 |
| bool\|F\|f16 | 6.8751 | 1.6940 | 0.25 🟠 |
| i8\|bcast\|bool | 2.1068 | 0.5319 | 0.25 🟠 |
| u64\|bcast\|u8 | 2.0032 | 0.5062 | 0.25 🟠 |
| i64\|bcast\|i8 | 1.9919 | 0.5069 | 0.25 🟠 |
| u64\|bcast\|i8 | 1.9821 | 0.5055 | 0.26 🟠 |
| i64\|bcast\|u8 | 2.0039 | 0.5118 | 0.26 🟠 |
| bool\|T\|f16 | 6.8586 | 1.7722 | 0.26 🟠 |
| f64\|bcast\|i16 | 3.4020 | 0.8978 | 0.26 🟠 |
| u16\|bcast\|bool | 1.6564 | 0.4417 | 0.27 🟠 |
| char\|bcast\|bool | 1.6662 | 0.4457 | 0.27 🟠 |
| u8\|bcast\|i8 | 1.5475 | 0.4233 | 0.27 🟠 |
| i16\|bcast\|bool | 1.6363 | 0.4489 | 0.27 🟠 |
| i64\|bcast\|bool | 1.8535 | 0.5120 | 0.28 🟠 |
| u64\|bcast\|bool | 1.8850 | 0.5217 | 0.28 🟠 |
| u16\|bcast\|u16 | 2.1568 | 0.6181 | 0.29 🟠 |
| char\|bcast\|u16 | 2.1005 | 0.6080 | 0.29 🟠 |
| u16\|bcast\|char | 2.1301 | 0.6211 | 0.29 🟠 |
| i32\|bcast\|u8 | 1.7253 | 0.5062 | 0.29 🟠 |
| i32\|bcast\|i8 | 1.7193 | 0.5048 | 0.29 🟠 |
| u32\|bcast\|i8 | 1.7204 | 0.5106 | 0.30 🟠 |
| f64\|negcol\|i8 | 1.4589 | 0.4331 | 0.30 🟠 |
| f64\|negcol\|u8 | 1.4737 | 0.4389 | 0.30 🟠 |
| i64\|bcast\|char | 2.5634 | 0.7637 | 0.30 🟠 |
| char\|bcast\|u8 | 1.6874 | 0.5041 | 0.30 🟠 |
| i16\|bcast\|i16 | 2.1260 | 0.6359 | 0.30 🟠 |
| u8\|bcast\|bool | 1.8195 | 0.5487 | 0.30 🟠 |
| u16\|bcast\|u8 | 1.6640 | 0.5026 | 0.30 🟠 |
| i64\|strided\|u16 | 0.7920 | 0.2409 | 0.30 🟠 |
| c128\|bcast\|u32 | 3.9498 | 1.2105 | 0.31 🟠 |
| bool\|bcast\|f16 | 11.2536 | 3.4501 | 0.31 🟠 |
| u16\|bcast\|i8 | 1.6604 | 0.5103 | 0.31 🟠 |
| i16\|bcast\|u8 | 1.6409 | 0.5045 | 0.31 🟠 |
| char\|bcast\|i8 | 1.6751 | 0.5168 | 0.31 🟠 |
| i64\|strided\|char | 0.7875 | 0.2430 | 0.31 🟠 |
| i64\|bcast\|u16 | 2.5189 | 0.7787 | 0.31 🟠 |
| bool\|bcast\|u8 | 1.5903 | 0.4927 | 0.31 🟠 |
| bool\|negcol\|f16 | 11.1971 | 3.4701 | 0.31 🟠 |
| bool\|negrow\|f16 | 11.2467 | 3.5075 | 0.31 🟠 |
| bool\|strided\|f16 | 5.5321 | 1.7357 | 0.31 🟠 |
| bool\|bcast\|i8 | 1.5970 | 0.5014 | 0.31 🟠 |
| bool\|sliced\|f16 | 11.0503 | 3.5039 | 0.32 🟠 |
| f32\|strided\|u8 | 0.6081 | 0.1938 | 0.32 🟠 |
| f32\|strided\|u16 | 0.6169 | 0.2028 | 0.33 🟠 |
| i64\|bcast\|i16 | 2.6700 | 0.8919 | 0.33 🟠 |
| f32\|bcast\|bool | 2.1831 | 0.7459 | 0.34 🟠 |
| f32\|strided\|i16 | 0.5933 | 0.2028 | 0.34 🟠 |
| u64\|bcast\|u16 | 2.5852 | 0.8882 | 0.34 🟠 |
| c128\|bcast\|i32 | 4.2049 | 1.4462 | 0.34 🟠 |
| i32\|negcol\|bool | 1.0077 | 0.3495 | 0.35 🟠 |
| c128\|bcast\|bool | 2.1639 | 0.7510 | 0.35 🟠 |
| i8\|bcast\|u8 | 1.5371 | 0.5343 | 0.35 🟠 |
| i32\|bcast\|char | 2.2054 | 0.7693 | 0.35 🟠 |
| u32\|bcast\|i16 | 2.2036 | 0.7725 | 0.35 🟠 |
| i32\|bcast\|u16 | 2.1897 | 0.7687 | 0.35 🟠 |
| i8\|bcast\|i16 | 2.1826 | 0.7696 | 0.35 🟠 |
| u64\|bcast\|char | 2.5635 | 0.9051 | 0.35 🟠 |
| c128\|strided\|bool | 1.4600 | 0.5172 | 0.35 🟠 |
| u64\|bcast\|i16 | 2.5374 | 0.9008 | 0.36 🟠 |
| u64\|C\|i8 | 0.5567 | 0.2004 | 0.36 🟠 |
| f32\|strided\|char | 0.5769 | 0.2082 | 0.36 🟠 |
| i32\|strided\|u16 | 0.5568 | 0.2015 | 0.36 🟠 |
| f64\|strided\|u16 | 0.7887 | 0.2858 | 0.36 🟠 |
| f64\|strided\|u8 | 0.7237 | 0.2634 | 0.36 🟠 |
| char\|bcast\|i16 | 2.0933 | 0.7632 | 0.36 🟠 |
| u32\|strided\|i16 | 0.5483 | 0.2000 | 0.36 🟠 |
| f64\|strided\|i8 | 0.7186 | 0.2632 | 0.37 🟠 |
| i32\|strided\|char | 0.5540 | 0.2032 | 0.37 🟠 |
| u16\|bcast\|i16 | 2.1091 | 0.7754 | 0.37 🟠 |
| u32\|strided\|u8 | 0.5206 | 0.1921 | 0.37 🟠 |
| f64\|strided\|char | 0.7564 | 0.2832 | 0.37 🟠 |
| c128\|negcol\|i8 | 1.5870 | 0.5974 | 0.38 🟠 |
| f16\|bcast\|i8 | 5.0106 | 1.9233 | 0.38 🟠 |
| i64\|negcol\|u8 | 1.4461 | 0.5577 | 0.39 🟠 |
| u64\|negcol\|i8 | 1.4493 | 0.5621 | 0.39 🟠 |
| c128\|negcol\|u8 | 1.5515 | 0.6025 | 0.39 🟠 |
| i64\|negcol\|i8 | 1.4392 | 0.5595 | 0.39 🟠 |
| u64\|strided\|char | 0.7937 | 0.3106 | 0.39 🟠 |
| u32\|negcol\|u8 | 1.0191 | 0.4005 | 0.39 🟠 |
| u64\|negcol\|u8 | 1.4542 | 0.5725 | 0.39 🟠 |
| f16\|bcast\|u8 | 4.7591 | 1.8822 | 0.40 🟠 |
| f32\|negcol\|u8 | 1.0518 | 0.4162 | 0.40 🟠 |
| i32\|C\|u8 | 0.5019 | 0.1987 | 0.40 🟠 |
| i32\|bcast\|i16 | 2.2647 | 0.8968 | 0.40 🟠 |
| u64\|negcol\|bool | 1.4841 | 0.5909 | 0.40 🟠 |
| f64\|strided\|i16 | 0.7897 | 0.3169 | 0.40 🟠 |
| f64\|bcast\|bool | 1.8493 | 0.7444 | 0.40 🟠 |
| u64\|strided\|i16 | 0.7941 | 0.3200 | 0.40 🟠 |
| u64\|strided\|bool | 0.7639 | 0.3087 | 0.40 🟠 |
| i8\|bcast\|char | 2.1851 | 0.8872 | 0.41 🟠 |
| u32\|bcast\|char | 2.2045 | 0.8954 | 0.41 🟠 |
| i64\|strided\|i16 | 0.7945 | 0.3232 | 0.41 🟠 |
| u64\|strided\|u8 | 0.7455 | 0.3044 | 0.41 🟠 |
| u32\|strided\|bool | 0.5416 | 0.2212 | 0.41 🟠 |
| u8\|bcast\|i16 | 2.2086 | 0.9024 | 0.41 🟠 |
| u8\|bcast\|char | 2.1808 | 0.8912 | 0.41 🟠 |
| f32\|bcast\|i32 | 2.8998 | 1.1869 | 0.41 🟠 |
| u64\|strided\|u16 | 0.7764 | 0.3181 | 0.41 🟠 |
| c128\|bcast\|f32 | 3.0590 | 1.2551 | 0.41 🟠 |
| bool\|bcast\|char | 2.1789 | 0.8943 | 0.41 🟠 |
| u32\|bcast\|u16 | 2.2268 | 0.9152 | 0.41 🟠 |
| f16\|bcast\|i16 | 5.1513 | 2.1279 | 0.41 🟠 |
| f16\|bcast\|u16 | 5.0237 | 2.0793 | 0.41 🟠 |
| f16\|bcast\|char | 4.9877 | 2.0789 | 0.42 🟠 |
| f32\|bcast\|u32 | 2.8639 | 1.1956 | 0.42 🟠 |
| i16\|bcast\|char | 2.1788 | 0.9099 | 0.42 🟠 |
| i32\|strided\|bool | 0.5274 | 0.2203 | 0.42 🟠 |
| u8\|bcast\|u16 | 2.1490 | 0.9009 | 0.42 🟠 |
| i8\|bcast\|u16 | 2.1836 | 0.9163 | 0.42 🟠 |
| i16\|bcast\|u16 | 2.1332 | 0.8991 | 0.42 🟠 |
| u64\|strided\|i8 | 0.7295 | 0.3084 | 0.42 🟠 |
| bool\|bcast\|i16 | 2.1406 | 0.9052 | 0.42 🟠 |
| i64\|strided\|bool | 0.7450 | 0.3191 | 0.43 🟠 |
| bool\|bcast\|u16 | 2.1276 | 0.9253 | 0.43 🟠 |
| i64\|strided\|i8 | 0.7361 | 0.3207 | 0.44 🟠 |
| f32\|strided\|u32 | 1.5167 | 0.6655 | 0.44 🟠 |
| f16\|bcast\|f32 | 4.2047 | 1.8589 | 0.44 🟠 |
| i64\|strided\|u8 | 0.7166 | 0.3174 | 0.44 🟠 |
| u32\|negcol\|bool | 1.0266 | 0.4550 | 0.44 🟠 |
| f16\|bcast\|i32 | 5.0931 | 2.3607 | 0.46 🟠 |
| i64\|negcol\|u16 | 1.7325 | 0.8081 | 0.47 🟠 |
| char\|bcast\|f32 | 2.0757 | 0.9706 | 0.47 🟠 |
| f16\|bcast\|u32 | 4.9348 | 2.3314 | 0.47 🟠 |
| f32\|strided\|i8 | 0.5735 | 0.2713 | 0.47 🟠 |
| f64\|bcast\|i32 | 2.5009 | 1.1866 | 0.47 🟠 |
| i64\|negcol\|char | 1.6731 | 0.7995 | 0.48 🟠 |
| f64\|negcol\|char | 1.7531 | 0.8420 | 0.48 🟠 |
| f32\|C\|u64 | 3.6129 | 1.7405 | 0.48 🟠 |
| f64\|bcast\|f32 | 2.5235 | 1.2207 | 0.48 🟠 |
| f64\|bcast\|u32 | 2.4197 | 1.1735 | 0.48 🟠 |
| f64\|negcol\|u16 | 1.7444 | 0.8602 | 0.49 🟠 |
| f32\|negcol\|i8 | 1.0755 | 0.5400 | 0.50 🟡 |
| i32\|strided\|i16 | 0.5564 | 0.2810 | 0.51 🟡 |
| c128\|bcast\|u64 | 4.7504 | 2.4018 | 0.51 🟡 |
| f32\|negcol\|u32 | 2.5384 | 1.2861 | 0.51 🟡 |
| u32\|strided\|u16 | 0.5554 | 0.2818 | 0.51 🟡 |
| f16\|bcast\|i64 | 5.3594 | 2.7219 | 0.51 🟡 |
| i64\|bcast\|f32 | 2.5172 | 1.2809 | 0.51 🟡 |
| i32\|negcol\|i8 | 1.0247 | 0.5229 | 0.51 🟡 |
| u32\|strided\|char | 0.5572 | 0.2883 | 0.52 🟡 |
| u32\|strided\|i8 | 0.5219 | 0.2708 | 0.52 🟡 |
| c128\|strided\|char | 0.8799 | 0.4569 | 0.52 🟡 |
| u32\|negcol\|i8 | 1.0257 | 0.5357 | 0.52 🟡 |
| u32\|bcast\|u32 | 2.2419 | 1.1805 | 0.53 🟡 |
| i64\|bcast\|f16 | 6.2360 | 3.2843 | 0.53 🟡 |
| u64\|bcast\|f16 | 6.6596 | 3.5502 | 0.53 🟡 |
| i32\|strided\|i8 | 0.5048 | 0.2696 | 0.53 🟡 |
| i32\|strided\|u8 | 0.5099 | 0.2745 | 0.54 🟡 |
| i64\|bcast\|i32 | 2.1965 | 1.1860 | 0.54 🟡 |
| i32\|negcol\|u8 | 1.0035 | 0.5460 | 0.54 🟡 |
| i32\|bcast\|i32 | 2.1616 | 1.1763 | 0.54 🟡 |
| u64\|negcol\|char | 1.7218 | 0.9389 | 0.55 🟡 |
| c128\|strided\|i16 | 0.8574 | 0.4677 | 0.55 🟡 |
| u32\|bcast\|i32 | 2.1729 | 1.1886 | 0.55 🟡 |
| c128\|strided\|u16 | 0.8472 | 0.4634 | 0.55 🟡 |
| f64\|strided\|i32 | 1.3859 | 0.7597 | 0.55 🟡 |
| f32\|bcast\|f32 | 2.1660 | 1.1887 | 0.55 🟡 |
| f64\|negcol\|i16 | 1.7773 | 0.9760 | 0.55 🟡 |
| char\|bcast\|f16 | 5.6567 | 3.1437 | 0.56 🟡 |
| u64\|bcast\|i32 | 2.1748 | 1.2110 | 0.56 🟡 |
| u64\|negcol\|i16 | 1.6824 | 0.9381 | 0.56 🟡 |
| f32\|strided\|u64 | 2.2906 | 1.2775 | 0.56 🟡 |
| bool\|F\|u8 | 0.3321 | 0.1853 | 0.56 🟡 |
| i64\|negcol\|i16 | 1.7015 | 0.9515 | 0.56 🟡 |
| u16\|bcast\|f16 | 5.9721 | 3.3521 | 0.56 🟡 |
| i16\|bcast\|f16 | 5.9323 | 3.3445 | 0.56 🟡 |
| f32\|F\|u64 | 3.5561 | 2.0218 | 0.57 🟡 |
| u64\|negcol\|u16 | 1.6479 | 0.9414 | 0.57 🟡 |
| i8\|bcast\|f16 | 5.8088 | 3.3379 | 0.57 🟡 |
| f64\|negcol\|bool | 1.4980 | 0.8624 | 0.58 🟡 |
| u32\|bcast\|f16 | 5.7222 | 3.3147 | 0.58 🟡 |
| u64\|bcast\|u32 | 2.2020 | 1.2781 | 0.58 🟡 |
| f32\|negcol\|u64 | 4.2455 | 2.4671 | 0.58 🟡 |
| f64\|strided\|bool | 0.7512 | 0.4366 | 0.58 🟡 |
| i64\|bcast\|u32 | 2.1939 | 1.2760 | 0.58 🟡 |
| i32\|bcast\|u32 | 2.1955 | 1.2777 | 0.58 🟡 |
| u8\|bcast\|f16 | 5.7207 | 3.3419 | 0.58 🟡 |
| c128\|strided\|i8 | 0.7772 | 0.4558 | 0.59 🟡 |
| i32\|bcast\|f32 | 2.1568 | 1.2733 | 0.59 🟡 |
| f32\|C\|u32 | 1.6146 | 0.9539 | 0.59 🟡 |
| c128\|strided\|u8 | 0.7623 | 0.4514 | 0.59 🟡 |
| f32\|negcol\|char | 1.3412 | 0.7953 | 0.59 🟡 |
| c128\|negcol\|u16 | 1.8832 | 1.1178 | 0.59 🟡 |
| u32\|bcast\|f32 | 2.1461 | 1.2812 | 0.60 🟡 |
| f32\|F\|u32 | 1.5960 | 0.9633 | 0.60 🟡 |
| c128\|strided\|u64 | 2.5484 | 1.5469 | 0.61 🟡 |
| c128\|sliced\|u64 | 3.8466 | 2.3453 | 0.61 🟡 |
| f16\|bcast\|u64 | 5.1628 | 3.1946 | 0.62 🟡 |
| u64\|C\|i16 | 0.7863 | 0.4868 | 0.62 🟡 |
| u8\|bcast\|i32 | 1.9046 | 1.1816 | 0.62 🟡 |
| i32\|bcast\|f16 | 5.3892 | 3.3486 | 0.62 🟡 |
| u16\|bcast\|u32 | 2.0366 | 1.2705 | 0.62 🟡 |
| u8\|bcast\|u32 | 1.8811 | 1.1744 | 0.62 🟡 |
| f16\|C\|u64 | 4.6370 | 2.8961 | 0.62 🟡 |
| c128\|negcol\|char | 1.7751 | 1.1112 | 0.63 🟡 |
| u32\|negcol\|i16 | 1.2324 | 0.7775 | 0.63 🟡 |
| i32\|negcol\|char | 1.2235 | 0.7734 | 0.63 🟡 |
| f32\|negcol\|u16 | 1.2602 | 0.7967 | 0.63 🟡 |
| i8\|bcast\|i32 | 1.8888 | 1.1949 | 0.63 🟡 |
| f16\|negrow\|u64 | 4.6066 | 2.9144 | 0.63 🟡 |
| c128\|C\|i8 | 0.4808 | 0.3043 | 0.63 🟡 |
| f16\|sliced\|u64 | 4.5457 | 2.8796 | 0.63 🟡 |
| i16\|bcast\|i32 | 1.9984 | 1.2697 | 0.64 🟡 |
| f16\|T\|u64 | 4.5446 | 2.8912 | 0.64 🟡 |
| u16\|bcast\|i32 | 2.0022 | 1.2757 | 0.64 🟡 |
| f64\|bcast\|f16 | 5.9209 | 3.7727 | 0.64 🟡 |
| f16\|F\|u64 | 4.5613 | 2.9065 | 0.64 🟡 |
| u64\|bcast\|f32 | 2.3463 | 1.5032 | 0.64 🟡 |
| i16\|bcast\|u32 | 2.0319 | 1.3043 | 0.64 🟡 |
| f64\|strided\|u32 | 1.1950 | 0.7765 | 0.65 🟡 |
| bool\|sliced\|u8 | 0.5217 | 0.3390 | 0.65 🟡 |
| f16\|strided\|u64 | 2.4900 | 1.6185 | 0.65 🟡 |
| f16\|negcol\|u64 | 5.0032 | 3.2629 | 0.65 🟡 |
| c128\|negcol\|i16 | 1.7522 | 1.1440 | 0.65 🟡 |
| c128\|bcast\|f16 | 6.1492 | 4.0247 | 0.65 🟡 |
| char\|bcast\|u32 | 2.0558 | 1.3475 | 0.66 🟡 |
| f32\|negcol\|i16 | 1.2323 | 0.8097 | 0.66 🟡 |
| i32\|negcol\|u16 | 1.2018 | 0.7934 | 0.66 🟡 |
| bool\|sliced\|u16 | 0.7926 | 0.5236 | 0.66 🟡 |
| i8\|bcast\|u32 | 1.9171 | 1.2695 | 0.66 🟡 |
| c128\|C\|u8 | 0.4935 | 0.3273 | 0.66 🟡 |
| u32\|T\|i8 | 0.5645 | 0.3748 | 0.66 🟡 |
| bool\|F\|i8 | 0.3330 | 0.2216 | 0.67 🟡 |
| i16\|bcast\|f32 | 2.0048 | 1.3495 | 0.67 🟡 |
| f32\|negrow\|u64 | 3.6373 | 2.4486 | 0.67 🟡 |
| f16\|C\|f64 | 3.3476 | 2.2541 | 0.67 🟡 |
| u16\|bcast\|f32 | 1.9731 | 1.3304 | 0.67 🟡 |
| char\|bcast\|i32 | 2.0367 | 1.3783 | 0.68 🟡 |
| f16\|bcast\|f64 | 3.0953 | 2.0983 | 0.68 🟡 |
| f32\|T\|u64 | 3.5957 | 2.4390 | 0.68 🟡 |
| f32\|sliced\|u64 | 3.6288 | 2.4617 | 0.68 🟡 |
| f64\|negcol\|u32 | 2.0459 | 1.3912 | 0.68 🟡 |
| bool\|bcast\|u32 | 1.8705 | 1.2783 | 0.68 🟡 |
| f64\|strided\|u64 | 1.9584 | 1.3457 | 0.69 🟡 |
| f64\|negcol\|u64 | 3.7170 | 2.5568 | 0.69 🟡 |
| f16\|negcol\|f64 | 3.1022 | 2.1351 | 0.69 🟡 |
| i32\|T\|i8 | 0.5446 | 0.3750 | 0.69 🟡 |
| u32\|C\|i8 | 0.5474 | 0.3779 | 0.69 🟡 |
| f32\|F\|i64 | 2.4365 | 1.6836 | 0.69 🟡 |
| bool\|bcast\|i32 | 1.8802 | 1.3043 | 0.69 🟡 |
| i64\|C\|i8 | 0.5566 | 0.3865 | 0.69 🟡 |
| f16\|strided\|f64 | 1.5349 | 1.0676 | 0.70 🟡 |
| i32\|C\|i8 | 0.5401 | 0.3759 | 0.70 🟡 |
| f32\|C\|i64 | 2.3729 | 1.6670 | 0.70 🟡 |
| f32\|bcast\|f16 | 4.4567 | 3.1332 | 0.70 🟡 |
| i8\|bcast\|f32 | 1.9031 | 1.3397 | 0.70 🟡 |
| i64\|F\|i8 | 0.5421 | 0.3824 | 0.71 🟡 |
| f32\|bcast\|u64 | 3.2544 | 2.3027 | 0.71 🟡 |
| bool\|sliced\|i16 | 0.7633 | 0.5438 | 0.71 🟡 |
| f64\|sliced\|u64 | 3.5317 | 2.5198 | 0.71 🟡 |
| i32\|F\|i8 | 0.5279 | 0.3771 | 0.71 🟡 |
| f16\|bcast\|bool | 1.6962 | 1.2124 | 0.71 🟡 |
| u32\|F\|i8 | 0.5301 | 0.3791 | 0.72 🟡 |
| f64\|C\|u64 | 3.5038 | 2.5098 | 0.72 🟡 |
| u8\|bcast\|f32 | 1.8672 | 1.3376 | 0.72 🟡 |
| c128\|F\|u64 | 3.8528 | 2.7735 | 0.72 🟡 |
| f32\|strided\|bool | 0.5397 | 0.3887 | 0.72 🟡 |
| f64\|F\|u64 | 3.4632 | 2.4968 | 0.72 🟡 |
| i64\|T\|i8 | 0.5455 | 0.3948 | 0.72 🟡 |
| f64\|T\|u64 | 3.4503 | 2.5160 | 0.73 🟡 |
| u32\|negcol\|char | 1.2528 | 0.9136 | 0.73 🟡 |
| f16\|sliced\|f64 | 3.0696 | 2.2425 | 0.73 🟡 |
| f64\|negrow\|u64 | 3.5334 | 2.5820 | 0.73 🟡 |
| f16\|T\|f64 | 3.0508 | 2.2428 | 0.74 🟡 |
| c128\|C\|u64 | 3.8176 | 2.8143 | 0.74 🟡 |
| bool\|bcast\|f32 | 1.8642 | 1.3768 | 0.74 🟡 |
| u64\|T\|u8 | 0.5171 | 0.3820 | 0.74 🟡 |
| c128\|strided\|u32 | 1.3239 | 0.9782 | 0.74 🟡 |
| u32\|negcol\|u16 | 1.2217 | 0.9036 | 0.74 🟡 |
| u64\|strided\|u32 | 1.0015 | 0.7407 | 0.74 🟡 |
| i64\|strided\|i32 | 0.9776 | 0.7231 | 0.74 🟡 |
| i32\|F\|u8 | 0.5068 | 0.3753 | 0.74 🟡 |
| f16\|negrow\|f64 | 3.0767 | 2.2810 | 0.74 🟡 |
| bool\|strided\|bool | 0.2803 | 0.2089 | 0.75 🟡 |
| bool\|sliced\|i8 | 0.5245 | 0.3921 | 0.75 🟡 |
| u64\|strided\|i32 | 0.9759 | 0.7298 | 0.75 🟡 |
| i32\|negcol\|i16 | 1.2160 | 0.9100 | 0.75 🟡 |
| c128\|negrow\|u64 | 3.9047 | 2.9341 | 0.75 🟡 |
| i64\|strided\|u32 | 0.9849 | 0.7433 | 0.75 🟡 |
| bool\|strided\|i8 | 0.2721 | 0.2061 | 0.76 🟡 |
| f16\|F\|f64 | 2.9866 | 2.2626 | 0.76 🟡 |
| u64\|C\|u8 | 0.5131 | 0.3895 | 0.76 🟡 |
| c128\|negcol\|u32 | 2.1182 | 1.6109 | 0.76 🟡 |
| char\|bcast\|f64 | 2.0552 | 1.5794 | 0.77 🟡 |
| f32\|negcol\|bool | 1.0264 | 0.7898 | 0.77 🟡 |
| c128\|negcol\|u64 | 3.8828 | 2.9929 | 0.77 🟡 |
| f32\|negrow\|u32 | 1.6370 | 1.2630 | 0.77 🟡 |
| c128\|negrow\|bool | 1.0263 | 0.7951 | 0.77 🟡 |
| u64\|F\|u8 | 0.4940 | 0.3835 | 0.78 🟡 |
| bool\|strided\|u8 | 0.2641 | 0.2061 | 0.78 🟡 |
| i32\|T\|u8 | 0.4925 | 0.3882 | 0.79 🟡 |
| c128\|T\|u64 | 3.8104 | 3.0083 | 0.79 🟡 |
| f64\|bcast\|u64 | 2.9126 | 2.3151 | 0.79 🟡 |
| u64\|negcol\|u32 | 1.6911 | 1.3492 | 0.80 🟡 |
| u64\|negcol\|i32 | 1.6377 | 1.3107 | 0.80 🟡 |
| c128\|sliced\|bool | 0.9471 | 0.7588 | 0.80 🟡 |
| i64\|negcol\|i32 | 1.6454 | 1.3210 | 0.80 🟡 |
| c128\|C\|bool | 0.9554 | 0.7694 | 0.81 🟡 |
| f32\|sliced\|u32 | 1.5847 | 1.2805 | 0.81 🟡 |
| f32\|T\|u32 | 1.5657 | 1.2736 | 0.81 🟡 |
| c128\|T\|bool | 0.9459 | 0.7699 | 0.81 🟡 |
| c128\|F\|bool | 0.9316 | 0.7622 | 0.82 🟡 |
| f32\|strided\|i32 | 0.8243 | 0.6744 | 0.82 🟡 |
| c128\|negcol\|bool | 1.1075 | 0.9096 | 0.82 🟡 |
| i8\|strided\|bool | 0.2926 | 0.2418 | 0.83 🟡 |
| i64\|negcol\|u32 | 1.6485 | 1.3689 | 0.83 🟡 |
| bool\|F\|u16 | 0.5049 | 0.4314 | 0.85 🟡 |
| bool\|F\|char | 0.5012 | 0.4300 | 0.86 🟡 |
| char\|bcast\|u64 | 2.0802 | 1.7886 | 0.86 🟡 |
| c128\|strided\|i32 | 1.0730 | 0.9240 | 0.86 🟡 |
| f32\|bcast\|i64 | 2.6355 | 2.2751 | 0.86 🟡 |
| u8\|strided\|bool | 0.2796 | 0.2423 | 0.87 🟡 |
| i8\|negrow\|i8 | 0.1102 | 0.0967 | 0.88 🟡 |
| u64\|C\|u16 | 0.7097 | 0.6236 | 0.88 🟡 |
| f16\|F\|f16 | 0.3692 | 0.3245 | 0.88 🟡 |
| bool\|C\|i16 | 0.5189 | 0.4561 | 0.88 🟡 |
| bool\|F\|i16 | 0.4995 | 0.4408 | 0.88 🟡 |
| bool\|C\|u16 | 0.5035 | 0.4496 | 0.89 🟡 |
| f16\|T\|f16 | 0.3590 | 0.3261 | 0.91 🟡 |
| f16\|strided\|c128 | 1.8063 | 1.6720 | 0.93 🟡 |
| c128\|negcol\|i32 | 1.7322 | 1.6059 | 0.93 🟡 |
| i64\|sliced\|i8 | 0.4059 | 0.3816 | 0.94 🟡 |
| f32\|negrow\|i64 | 2.4984 | 2.3533 | 0.94 🟡 |
| i64\|negrow\|bool | 0.5033 | 0.4741 | 0.94 🟡 |
| i64\|sliced\|bool | 0.4706 | 0.4438 | 0.94 🟡 |
| bool\|C\|char | 0.4910 | 0.4647 | 0.95 🟡 |
| i8\|sliced\|i8 | 0.0925 | 0.0884 | 0.96 🟡 |
| u64\|C\|bool | 0.4514 | 0.4372 | 0.97 🟡 |
| u64\|negrow\|bool | 0.4865 | 0.4717 | 0.97 🟡 |
| f32\|negcol\|i64 | 2.4984 | 2.4264 | 0.97 🟡 |
| bool\|C\|u8 | 0.3354 | 0.3259 | 0.97 🟡 |
| f32\|sliced\|i64 | 2.4306 | 2.3633 | 0.97 🟡 |
| f16\|strided\|u32 | 1.2508 | 1.2162 | 0.97 🟡 |
| i32\|C\|bool | 0.2272 | 0.2212 | 0.97 🟡 |
| c128\|sliced\|i64 | 2.4976 | 2.4385 | 0.98 🟡 |
| u64\|sliced\|bool | 0.4535 | 0.4437 | 0.98 🟡 |
| i64\|C\|bool | 0.4504 | 0.4413 | 0.98 🟡 |
| f16\|bcast\|c128 | 3.3496 | 3.3038 | 0.99 🟡 |
| i64\|T\|bool | 0.4467 | 0.4423 | 0.99 🟡 |
| u64\|F\|bool | 0.4498 | 0.4485 | 1.00 🟡 |
| f32\|T\|i64 | 2.3578 | 2.3519 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 366 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.56 ms   unfused    6.18 ms   (1.73x)
  (a-b)/(a+b) fused    2.88 ms   unfused   13.76 ms   (4.78x)
  sum(a*b)    fused    2.31 ms   unfused    3.90 ms   (1.69x)
  sum(af*bf)  fused    1.34 ms   unfused    1.69 ms   (1.26x)  [f32]
  a*b+c out=  fused    3.54 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.74 ms   unfused    3.91 ms   (1.43x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.40 ms   unfused    6.12 ms   (1.80x)
    [F      ] fused    3.52 ms   unfused    6.09 ms   (1.73x)
    [T      ] fused    3.44 ms   unfused    6.12 ms   (1.78x)
    [strided] fused    3.35 ms   unfused    4.69 ms   (1.40x)
    [bcast  ] fused    1.05 ms   unfused    3.44 ms   (3.27x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         13.01 ms
  (a-b)/(a+b)   18.89 ms
  sum(a*b)       8.96 ms
  sum(af*bf)     4.29 ms  [f32]
  a*b+c out=     4.90 ms  [two-pass with out=]
  i4*2+f8        9.78 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   13.33 ms
    [F      ]   13.04 ms
    [T      ]   13.53 ms
    [strided]    8.48 ms
    [bcast  ]   12.56 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-6f2338a9b13e5e0c63650ad0eac116091195bf72a3489e1e18824d08b5dda7c1\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0070 ms | 0.0073 ms | 0.0069 | 0.986x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1165 ms | 0.9656 ms | 0.9941 | 8.533x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1173 ms | 0.9917 ms | 1.1135 | 9.493x |
| `np.dot(a, b)` | 100,000 | 0.0070 ms | 0.0074 ms | 0.0070 | 1.000x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2151 ms | 0.0605 ms | 0.2416 | 3.993x |
| `np.inner(a, b)` | 100,000 | 0.0070 ms | 0.0074 ms | 0.0071 | 1.014x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0025 ms | 0.0047 | 1.880x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0160 ms | 0.0183 | 1.144x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0390 ms | 0.0353 | 0.905x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.5162 ms | 0.2785 | 0.540x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0052 ms | 0.0061 | 1.173x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0578 ms | 0.0272 | 0.471x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1146 ms | 0.1032 | 0.901x |
| `np.linalg.eig(a)` | 96 | missing_backend | 2.6840 ms | 1.3641 | 0.508x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0462 ms | 0.0487 | 1.054x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.8220 ms | 0.3986 | 0.485x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0670 ms | 0.0664 | 0.991x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.5275 ms | 0.7744 | 0.507x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0207 ms | 0.0228 | 1.101x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.2832 ms | 0.1599 | 0.565x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0090 ms | 0.0111 | 1.233x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.1619 ms | 0.0787 | 0.486x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0832 ms | 0.0802 | 0.964x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 1.6242 ms | 0.7920 | 0.488x |
| `np.linalg.matmul(a, b)` | 128 | 0.2128 ms | 0.0586 ms | 0.0587 | 1.002x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0328 ms | 0.0369 | 1.125x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.4903 ms | 0.2894 | 0.590x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0106 ms | 0.0115 | 1.085x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1651 ms | 0.0803 | 0.486x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0365 ms | 0.0370 | 1.014x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.4982 ms | 0.2853 | 0.573x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3826 ms | 0.1175 ms | 0.1212 | 1.031x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0332 ms | 0.0368 | 1.108x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.4903 ms | 0.2827 | 0.577x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.0956 ms | 0.0871 | 0.911x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 1.5979 ms | 0.7379 | 0.462x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0203 ms | 0.0236 | 1.163x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.3164 ms | 0.1607 | 0.508x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0057 ms | 0.0068 | 1.193x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0564 ms | 0.0282 | 0.500x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0053 ms | 0.0077 | 1.453x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0592 ms | 0.0299 | 0.505x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0750 ms | 0.0771 | 1.028x |
| `np.linalg.svd(a)` | 96 | missing_backend | 1.4331 ms | 0.6929 | 0.483x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0298 ms | 0.0320 | 1.074x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.4894 ms | 0.2761 | 0.564x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2166 ms | 0.0606 ms | 0.0635 | 1.048x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0096 ms | 0.0118 | 1.229x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.1621 ms | 0.0801 | 0.494x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0067 ms | 0.0086 | 1.284x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0604 ms | 0.0309 | 0.512x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0329 ms | 0.0073 ms | 0.0074 | 1.014x |
| `np.matmul(a, b)` | 128 | 0.2133 ms | 0.0587 ms | 0.0585 | 0.997x |
| `np.matvec(a, b)` | 128 | 0.0227 ms | 0.0016 ms | 0.0020 | 1.250x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.1084 ms | 0.0454 | 0.419x |
| `np.roots(p)` | 16 | missing_backend | 0.0515 ms | 0.0374 | 0.726x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2149 ms | 0.0608 ms | 0.0634 | 1.043x |
| `np.vdot(a, b)` | 100,000 | 0.0076 ms | 0.0081 ms | 0.0071 | 0.934x |
| `np.vecdot(a, b)` | 100,000 | 0.0333 ms | 0.0073 ms | 0.0071 | 0.973x |
| `np.vecmat(a, b)` | 128 | 0.0027 ms | 0.0017 ms | 0.0015 | 0.882x |
