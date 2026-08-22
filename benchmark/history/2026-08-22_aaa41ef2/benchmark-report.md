# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell.

- Managed effective cells: **3,334**
- OpenBLAS effective cells: **58**
- Missing backend records: **44**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.2522 ms | openblas | 0.211x |
| `np.roots(p)` | float64 | 16 | missing_backend | 0.0652 ms | openblas | 0.868x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 1.0892 ms | not_measured | managed | 1.943x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 1.1179 ms | not_measured | managed | 1.921x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 1.0885 ms | not_measured | managed | 1.986x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.0520 ms | not_measured | managed | 0.625x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.0003 ms | not_measured | managed | 1.333x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.0604 ms | not_measured | managed | 1.131x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.8095 ms | not_measured | managed | 0.348x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.0002 ms | not_measured | managed | 4.000x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.1436 ms | not_measured | managed | 0.169x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.0038 ms | not_measured | managed | 0.605x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.0109 ms | not_measured | managed | 0.239x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.4359 ms | not_measured | managed | 0.263x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.0152 ms | not_measured | managed | 0.197x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.0196 ms | not_measured | managed | 0.770x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.321x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1240 ms | not_measured | managed | 0.181x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 37.000x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.3718 ms | not_measured | managed | 0.766x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 1.2266 ms | not_measured | managed | 1.353x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 7.2840 ms | not_measured | managed | 0.047x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.5410 ms | not_measured | managed | 0.634x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 20.000x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.0081 ms | not_measured | managed | 1.901x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.2696 ms | not_measured | managed | 1.290x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 114.9640 ms | not_measured | managed | 1.452x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 2.016x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.1321 ms | not_measured | managed | 1.284x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 116.3861 ms | not_measured | managed | 1.429x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.579x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.4741 ms | not_measured | managed | 0.876x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 47.0786 ms | not_measured | managed | 0.980x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.0052 ms | not_measured | managed | 0.827x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.5449 ms | not_measured | managed | 0.783x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 58.8679 ms | not_measured | managed | 0.998x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.0122 ms | not_measured | managed | 1.582x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 1.6725 ms | not_measured | managed | 0.900x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 103.3305 ms | not_measured | managed | 1.539x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.0101 ms | not_measured | managed | 1.059x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 1.4814 ms | not_measured | managed | 0.845x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 98.5315 ms | not_measured | managed | 1.371x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.0026 ms | not_measured | managed | 0.808x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.6009 ms | not_measured | managed | 0.632x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 62.7471 ms | not_measured | managed | 0.754x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.0034 ms | not_measured | managed | 1.118x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.6253 ms | not_measured | managed | 0.614x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 68.4734 ms | not_measured | managed | 0.893x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.0052 ms | not_measured | managed | 0.231x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.2486 ms | not_measured | managed | 1.594x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 25.8379 ms | not_measured | managed | 1.171x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.0109 ms | not_measured | managed | 0.349x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.4731 ms | not_measured | managed | 0.641x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 47.5270 ms | not_measured | managed | 0.721x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.0057 ms | not_measured | managed | 0.140x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.0291 ms | not_measured | managed | 0.234x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 5.0387 ms | not_measured | managed | 2.376x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.125x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.0569 ms | not_measured | managed | 0.239x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 13.7544 ms | not_measured | managed | 1.150x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.0037 ms | not_measured | managed | 0.297x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.0222 ms | not_measured | managed | 1.000x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 3.3664 ms | not_measured | managed | 1.854x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.435x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.0279 ms | not_measured | managed | 0.835x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 5.1467 ms | not_measured | managed | 2.610x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.0044 ms | not_measured | managed | 0.273x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.0574 ms | not_measured | managed | 0.416x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 13.4681 ms | not_measured | managed | 2.022x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.0137 ms | not_measured | managed | 1.679x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 1.3353 ms | not_measured | managed | 3.479x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.0034 ms | not_measured | managed | 0.324x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.0205 ms | not_measured | managed | 1.117x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 3.1828 ms | not_measured | managed | 1.794x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.0039 ms | not_measured | managed | 0.256x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.0285 ms | not_measured | managed | 0.926x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 5.1835 ms | not_measured | managed | 2.344x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.0046 ms | not_measured | managed | 0.196x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.0579 ms | not_measured | managed | 0.380x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 12.9265 ms | not_measured | managed | 1.512x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.0147 ms | not_measured | managed | 1.871x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 1.4423 ms | not_measured | managed | 2.877x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.0037 ms | not_measured | managed | 0.216x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.3228 ms | not_measured | managed | 1.147x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 25.9967 ms | not_measured | managed | 1.145x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.0099 ms | not_measured | managed | 0.333x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.9172 ms | not_measured | managed | 0.329x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 48.2161 ms | not_measured | managed | 0.674x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.127x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 5.0090 ms | not_measured | managed | 2.458x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.1206 ms | not_measured | managed | 0.131x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 15.6805 ms | not_measured | managed | 0.994x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.0283 ms | not_measured | managed | 1.007x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 4.5688 ms | not_measured | managed | 1.249x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.0527 ms | not_measured | managed | 0.571x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 9.3490 ms | not_measured | managed | 1.336x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.0998 ms | not_measured | managed | 0.301x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 14.0536 ms | not_measured | managed | 2.025x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.0167 ms | not_measured | managed | 1.784x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 2.2560 ms | not_measured | managed | 2.133x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.0278 ms | not_measured | managed | 1.104x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 4.3980 ms | not_measured | managed | 1.581x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.0561 ms | not_measured | managed | 0.515x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 8.3760 ms | not_measured | managed | 0.975x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.233x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.1025 ms | not_measured | managed | 0.285x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 13.4719 ms | not_measured | managed | 1.819x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.0168 ms | not_measured | managed | 1.839x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 2.2140 ms | not_measured | managed | 2.143x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.0032 ms | not_measured | managed | 0.312x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.3553 ms | not_measured | managed | 1.086x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 29.3659 ms | not_measured | managed | 1.241x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.0100 ms | not_measured | managed | 0.330x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.9214 ms | not_measured | managed | 0.325x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 47.2843 ms | not_measured | managed | 0.735x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.0530 ms | not_measured | managed | 0.168x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 5.7948 ms | not_measured | managed | 2.611x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.296x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.1055 ms | not_measured | managed | 0.267x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 19.1572 ms | not_measured | managed | 0.905x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.0278 ms | not_measured | managed | 1.018x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 6.1369 ms | not_measured | managed | 0.891x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.579x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.0535 ms | not_measured | managed | 0.593x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 10.8943 ms | not_measured | managed | 1.223x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.1076 ms | not_measured | managed | 0.324x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 29.1131 ms | not_measured | managed | 0.888x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.0167 ms | not_measured | managed | 1.695x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 3.5679 ms | not_measured | managed | 1.356x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.0283 ms | not_measured | managed | 1.449x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.0529 ms | not_measured | managed | 1.184x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.0532 ms | not_measured | managed | 0.556x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.0235 ms | not_measured | managed | 0.833x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.0026 ms | not_measured | managed | 0.308x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.1085 ms | not_measured | managed | 0.330x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 21.2758 ms | not_measured | managed | 1.235x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.0171 ms | not_measured | managed | 1.661x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.5028 ms | not_measured | managed | 1.604x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.0031 ms | not_measured | managed | 0.323x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.2235 ms | not_measured | managed | 1.660x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 25.6947 ms | not_measured | managed | 1.174x |
| `a * scalar (float16)` | float16 | 1,000 | 0.0095 ms | not_measured | managed | 0.358x |
| `a * scalar (float16)` | float16 | 100,000 | 0.4905 ms | not_measured | managed | 0.601x |
| `a * scalar (float16)` | float16 | 10,000,000 | 48.3856 ms | not_measured | managed | 0.685x |
| `a * scalar (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.316x |
| `a * scalar (float32)` | float32 | 100,000 | 0.0283 ms | not_measured | managed | 0.233x |
| `a * scalar (float32)` | float32 | 10,000,000 | 5.3557 ms | not_measured | managed | 2.418x |
| `a * scalar (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.250x |
| `a * scalar (float64)` | float64 | 100,000 | 0.0562 ms | not_measured | managed | 0.233x |
| `a * scalar (float64)` | float64 | 10,000,000 | 14.7864 ms | not_measured | managed | 1.090x |
| `a * scalar (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.846x |
| `a * scalar (int16)` | int16 | 100,000 | 0.0172 ms | not_measured | managed | 1.297x |
| `a * scalar (int16)` | int16 | 10,000,000 | 4.0757 ms | not_measured | managed | 1.687x |
| `a * scalar (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `a * scalar (int32)` | int32 | 100,000 | 0.0298 ms | not_measured | managed | 0.836x |
| `a * scalar (int32)` | int32 | 10,000,000 | 5.1711 ms | not_measured | managed | 2.129x |
| `a * scalar (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 1.111x |
| `a * scalar (int64)` | int64 | 100,000 | 0.0560 ms | not_measured | managed | 0.434x |
| `a * scalar (int64)` | int64 | 10,000,000 | 13.4129 ms | not_measured | managed | 1.930x |
| `a * scalar (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `a * scalar (int8)` | int8 | 100,000 | 0.0088 ms | not_measured | managed | 2.614x |
| `a * scalar (int8)` | int8 | 10,000,000 | 1.5632 ms | not_measured | managed | 2.635x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.0006 ms | not_measured | managed | 1.500x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.0151 ms | not_measured | managed | 1.477x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 3.1311 ms | not_measured | managed | 2.210x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.0274 ms | not_measured | managed | 0.836x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 4.9531 ms | not_measured | managed | 1.990x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.0597 ms | not_measured | managed | 0.372x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 13.4308 ms | not_measured | managed | 1.799x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.0088 ms | not_measured | managed | 2.659x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 1.3105 ms | not_measured | managed | 3.122x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.0044 ms | not_measured | managed | 0.250x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.2249 ms | not_measured | managed | 1.683x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 25.7761 ms | not_measured | managed | 1.219x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.0056 ms | not_measured | managed | 1.232x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.4700 ms | not_measured | managed | 0.629x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 48.9655 ms | not_measured | managed | 0.633x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.0037 ms | not_measured | managed | 0.703x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.0297 ms | not_measured | managed | 0.354x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 5.2672 ms | not_measured | managed | 2.502x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.0050 ms | not_measured | managed | 0.500x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.0642 ms | not_measured | managed | 0.263x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 13.7741 ms | not_measured | managed | 1.168x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.0033 ms | not_measured | managed | 0.333x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.0203 ms | not_measured | managed | 1.232x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 2.5643 ms | not_measured | managed | 2.830x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.500x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.0312 ms | not_measured | managed | 0.817x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 5.6259 ms | not_measured | managed | 1.945x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.0050 ms | not_measured | managed | 0.520x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.0570 ms | not_measured | managed | 0.442x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 14.2447 ms | not_measured | managed | 1.713x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.0034 ms | not_measured | managed | 0.265x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.0135 ms | not_measured | managed | 2.022x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 1.3962 ms | not_measured | managed | 3.217x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.0035 ms | not_measured | managed | 0.314x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.0237 ms | not_measured | managed | 1.046x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 2.9461 ms | not_measured | managed | 2.364x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.0035 ms | not_measured | managed | 0.314x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.0305 ms | not_measured | managed | 0.816x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 5.1491 ms | not_measured | managed | 1.991x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.0074 ms | not_measured | managed | 0.351x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.0709 ms | not_measured | managed | 0.343x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 15.6084 ms | not_measured | managed | 1.552x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.0033 ms | not_measured | managed | 0.273x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.0157 ms | not_measured | managed | 1.643x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 1.4361 ms | not_measured | managed | 3.201x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.0030 ms | not_measured | managed | 0.367x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.2614 ms | not_measured | managed | 1.534x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 29.7505 ms | not_measured | managed | 1.725x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.0053 ms | not_measured | managed | 1.264x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.4969 ms | not_measured | managed | 0.601x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 47.9527 ms | not_measured | managed | 0.650x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 1.125x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.0295 ms | not_measured | managed | 0.261x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 6.7644 ms | not_measured | managed | 1.447x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.889x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.0637 ms | not_measured | managed | 0.609x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 15.3827 ms | not_measured | managed | 1.213x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.0197 ms | not_measured | managed | 1.645x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 3.4156 ms | not_measured | managed | 2.411x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.0417 ms | not_measured | managed | 0.818x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 6.0852 ms | not_measured | managed | 2.091x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.955x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.0690 ms | not_measured | managed | 0.528x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 15.5928 ms | not_measured | managed | 1.645x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.778x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.0120 ms | not_measured | managed | 2.483x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 1.7703 ms | not_measured | managed | 2.851x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.0178 ms | not_measured | managed | 1.798x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.1462 ms | not_measured | managed | 2.341x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.0330 ms | not_measured | managed | 0.876x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 6.5457 ms | not_measured | managed | 2.016x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.0019 ms | not_measured | managed | 1.053x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.0639 ms | not_measured | managed | 0.565x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 15.5836 ms | not_measured | managed | 1.736x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.0092 ms | not_measured | managed | 3.196x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.7608 ms | not_measured | managed | 2.783x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.0029 ms | not_measured | managed | 0.345x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.2236 ms | not_measured | managed | 1.678x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 25.0926 ms | not_measured | managed | 1.697x |
| `a + scalar (float16)` | float16 | 1,000 | 0.0052 ms | not_measured | managed | 1.250x |
| `a + scalar (float16)` | float16 | 100,000 | 0.4874 ms | not_measured | managed | 0.593x |
| `a + scalar (float16)` | float16 | 10,000,000 | 46.5946 ms | not_measured | managed | 0.715x |
| `a + scalar (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 1.000x |
| `a + scalar (float32)` | float32 | 100,000 | 0.0288 ms | not_measured | managed | 0.229x |
| `a + scalar (float32)` | float32 | 10,000,000 | 5.3454 ms | not_measured | managed | 2.246x |
| `a + scalar (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.667x |
| `a + scalar (float64)` | float64 | 100,000 | 0.0620 ms | not_measured | managed | 0.237x |
| `a + scalar (float64)` | float64 | 10,000,000 | 13.8986 ms | not_measured | managed | 1.243x |
| `a + scalar (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a + scalar (int16)` | int16 | 100,000 | 0.0150 ms | not_measured | managed | 1.760x |
| `a + scalar (int16)` | int16 | 10,000,000 | 2.8942 ms | not_measured | managed | 2.631x |
| `a + scalar (int32)` | int32 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a + scalar (int32)` | int32 | 100,000 | 0.0294 ms | not_measured | managed | 0.993x |
| `a + scalar (int32)` | int32 | 10,000,000 | 5.3132 ms | not_measured | managed | 1.969x |
| `a + scalar (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.897x |
| `a + scalar (int64)` | int64 | 100,000 | 0.0605 ms | not_measured | managed | 0.441x |
| `a + scalar (int64)` | int64 | 10,000,000 | 13.9601 ms | not_measured | managed | 1.785x |
| `a + scalar (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a + scalar (int8)` | int8 | 100,000 | 0.0083 ms | not_measured | managed | 3.373x |
| `a + scalar (int8)` | int8 | 10,000,000 | 1.3664 ms | not_measured | managed | 3.278x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.0161 ms | not_measured | managed | 1.540x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 2.6049 ms | not_measured | managed | 2.540x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.0273 ms | not_measured | managed | 0.886x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 5.3194 ms | not_measured | managed | 2.303x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.0026 ms | not_measured | managed | 0.923x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.0543 ms | not_measured | managed | 0.446x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 14.0505 ms | not_measured | managed | 1.749x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.0082 ms | not_measured | managed | 3.171x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 1.3920 ms | not_measured | managed | 3.213x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.0035 ms | not_measured | managed | 0.257x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.2445 ms | not_measured | managed | 1.598x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 29.2271 ms | not_measured | managed | 1.148x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.439x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.4774 ms | not_measured | managed | 0.676x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 47.2297 ms | not_measured | managed | 0.667x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.875x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.0303 ms | not_measured | managed | 0.257x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 5.9531 ms | not_measured | managed | 2.270x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.895x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.0650 ms | not_measured | managed | 0.432x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 17.4328 ms | not_measured | managed | 1.005x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.0167 ms | not_measured | managed | 1.808x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 3.1396 ms | not_measured | managed | 1.912x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.0282 ms | not_measured | managed | 1.067x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 5.7422 ms | not_measured | managed | 2.153x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 1.111x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.0635 ms | not_measured | managed | 0.532x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 15.0578 ms | not_measured | managed | 1.752x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.0138 ms | not_measured | managed | 2.210x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 1.6013 ms | not_measured | managed | 3.076x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.0160 ms | not_measured | managed | 1.913x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 3.1554 ms | not_measured | managed | 2.279x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.0291 ms | not_measured | managed | 1.014x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 5.8350 ms | not_measured | managed | 1.577x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 1.167x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.0658 ms | not_measured | managed | 0.552x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 15.1429 ms | not_measured | managed | 1.733x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.0099 ms | not_measured | managed | 3.000x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 1.6898 ms | not_measured | managed | 2.932x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.0027 ms | not_measured | managed | 0.444x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.2855 ms | not_measured | managed | 1.323x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 25.3641 ms | not_measured | managed | 1.257x |
| `a - scalar (float16)` | float16 | 1,000 | 0.0051 ms | not_measured | managed | 1.137x |
| `a - scalar (float16)` | float16 | 100,000 | 0.5023 ms | not_measured | managed | 0.608x |
| `a - scalar (float16)` | float16 | 10,000,000 | 46.2797 ms | not_measured | managed | 0.706x |
| `a - scalar (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `a - scalar (float32)` | float32 | 100,000 | 0.0274 ms | not_measured | managed | 0.339x |
| `a - scalar (float32)` | float32 | 10,000,000 | 5.1269 ms | not_measured | managed | 2.600x |
| `a - scalar (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 1.111x |
| `a - scalar (float64)` | float64 | 100,000 | 0.0640 ms | not_measured | managed | 0.206x |
| `a - scalar (float64)` | float64 | 10,000,000 | 14.6529 ms | not_measured | managed | 1.071x |
| `a - scalar (int16)` | int16 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a - scalar (int16)` | int16 | 100,000 | 0.0163 ms | not_measured | managed | 1.466x |
| `a - scalar (int16)` | int16 | 10,000,000 | 3.0499 ms | not_measured | managed | 1.683x |
| `a - scalar (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a - scalar (int32)` | int32 | 100,000 | 0.0293 ms | not_measured | managed | 0.881x |
| `a - scalar (int32)` | int32 | 10,000,000 | 5.2769 ms | not_measured | managed | 2.108x |
| `a - scalar (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `a - scalar (int64)` | int64 | 100,000 | 0.0722 ms | not_measured | managed | 0.391x |
| `a - scalar (int64)` | int64 | 10,000,000 | 13.9315 ms | not_measured | managed | 1.770x |
| `a - scalar (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `a - scalar (int8)` | int8 | 100,000 | 0.0087 ms | not_measured | managed | 3.023x |
| `a - scalar (int8)` | int8 | 10,000,000 | 1.6264 ms | not_measured | managed | 2.789x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.0171 ms | not_measured | managed | 1.427x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 2.9479 ms | not_measured | managed | 2.214x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.0361 ms | not_measured | managed | 0.665x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 5.3876 ms | not_measured | managed | 1.569x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.0559 ms | not_measured | managed | 0.456x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 17.2005 ms | not_measured | managed | 1.423x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.0089 ms | not_measured | managed | 2.809x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 2.1504 ms | not_measured | managed | 2.027x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.0300 ms | not_measured | managed | 0.417x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 7.1868 ms | not_measured | managed | 1.882x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.0620 ms | not_measured | managed | 0.665x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 17.4062 ms | not_measured | managed | 0.965x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.0055 ms | not_measured | managed | 0.364x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.0972 ms | not_measured | managed | 0.931x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 13.8569 ms | not_measured | managed | 2.150x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.0047 ms | not_measured | managed | 0.553x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.0870 ms | not_measured | managed | 0.911x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 29.6318 ms | not_measured | managed | 1.169x |
| `a / scalar (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `a / scalar (float32)` | float32 | 100,000 | 0.0290 ms | not_measured | managed | 0.441x |
| `a / scalar (float32)` | float32 | 10,000,000 | 5.3267 ms | not_measured | managed | 2.002x |
| `a / scalar (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.562x |
| `a / scalar (float64)` | float64 | 100,000 | 0.0545 ms | not_measured | managed | 0.728x |
| `a / scalar (float64)` | float64 | 10,000,000 | 13.5076 ms | not_measured | managed | 1.207x |
| `a / scalar (int32)` | int32 | 1,000 | 0.0039 ms | not_measured | managed | 0.641x |
| `a / scalar (int32)` | int32 | 100,000 | 0.0856 ms | not_measured | managed | 0.771x |
| `a / scalar (int32)` | int32 | 10,000,000 | 13.7869 ms | not_measured | managed | 1.916x |
| `a / scalar (int64)` | int64 | 1,000 | 0.0039 ms | not_measured | managed | 0.436x |
| `a / scalar (int64)` | int64 | 100,000 | 0.0850 ms | not_measured | managed | 0.779x |
| `a / scalar (int64)` | int64 | 10,000,000 | 23.5373 ms | not_measured | managed | 1.173x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.0029 ms | not_measured | managed | 0.310x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.2429 ms | not_measured | managed | 1.714x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 31.0259 ms | not_measured | managed | 1.591x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.0053 ms | not_measured | managed | 1.113x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.8797 ms | not_measured | managed | 0.335x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 47.2287 ms | not_measured | managed | 0.656x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.0283 ms | not_measured | managed | 0.261x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 5.6715 ms | not_measured | managed | 1.715x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 2.000x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.0649 ms | not_measured | managed | 0.438x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 15.9144 ms | not_measured | managed | 1.076x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.0239 ms | not_measured | managed | 1.397x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 3.3751 ms | not_measured | managed | 2.450x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.0350 ms | not_measured | managed | 0.946x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 5.9742 ms | not_measured | managed | 2.218x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 1.222x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.0643 ms | not_measured | managed | 0.510x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 15.5402 ms | not_measured | managed | 1.637x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.0096 ms | not_measured | managed | 3.104x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 1.8792 ms | not_measured | managed | 2.694x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.0194 ms | not_measured | managed | 1.711x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 3.0623 ms | not_measured | managed | 2.345x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 0.688x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.0308 ms | not_measured | managed | 0.935x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 5.6546 ms | not_measured | managed | 2.335x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.0646 ms | not_measured | managed | 0.576x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 20.0462 ms | not_measured | managed | 1.289x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 0.875x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.0135 ms | not_measured | managed | 2.259x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 1.6222 ms | not_measured | managed | 3.174x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.0277 ms | not_measured | managed | 0.444x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 11.3991 ms | not_measured | managed | 1.076x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.0939 ms | not_measured | managed | 0.430x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 26.6795 ms | not_measured | managed | 0.619x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 1.312x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.0828 ms | not_measured | managed | 1.082x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 14.3681 ms | not_measured | managed | 2.053x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.690x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.0838 ms | not_measured | managed | 0.936x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 29.4501 ms | not_measured | managed | 1.180x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.0059 ms | not_measured | managed | 1.966x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 0.9920 ms | not_measured | managed | 1.519x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 160.8759 ms | not_measured | managed | 0.958x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.0071 ms | not_measured | managed | 1.423x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 0.9572 ms | not_measured | managed | 1.388x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 149.8676 ms | not_measured | managed | 0.924x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 1.125x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.4983 ms | not_measured | managed | 0.762x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 64.2754 ms | not_measured | managed | 0.717x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.0036 ms | not_measured | managed | 0.694x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.5589 ms | not_measured | managed | 0.683x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 69.6646 ms | not_measured | managed | 0.840x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.0062 ms | not_measured | managed | 1.919x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 1.0101 ms | not_measured | managed | 1.513x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 99.4888 ms | not_measured | managed | 1.507x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.0061 ms | not_measured | managed | 1.656x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 0.9035 ms | not_measured | managed | 1.444x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 97.1023 ms | not_measured | managed | 1.417x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.870x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.4228 ms | not_measured | managed | 0.903x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 44.5787 ms | not_measured | managed | 1.042x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.0035 ms | not_measured | managed | 1.086x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.5174 ms | not_measured | managed | 0.731x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 57.2229 ms | not_measured | managed | 1.034x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.0028 ms | not_measured | managed | 0.357x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.2323 ms | not_measured | managed | 1.654x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 29.6384 ms | not_measured | managed | 1.141x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.0050 ms | not_measured | managed | 0.660x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.4732 ms | not_measured | managed | 0.637x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 47.2299 ms | not_measured | managed | 0.664x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.417x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.0302 ms | not_measured | managed | 0.278x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 6.2146 ms | not_measured | managed | 2.479x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.0687 ms | not_measured | managed | 0.403x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 15.7201 ms | not_measured | managed | 1.183x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.0166 ms | not_measured | managed | 1.693x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 6.2832 ms | not_measured | managed | 0.876x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.0323 ms | not_measured | managed | 0.978x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 8.8694 ms | not_measured | managed | 1.517x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.0679 ms | not_measured | managed | 0.501x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 15.7140 ms | not_measured | managed | 1.799x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.0005 ms | not_measured | managed | 1.400x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.0106 ms | not_measured | managed | 2.632x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 1.5069 ms | not_measured | managed | 3.123x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 0.800x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.0165 ms | not_measured | managed | 1.727x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 4.5061 ms | not_measured | managed | 1.372x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.0361 ms | not_measured | managed | 0.817x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 6.4491 ms | not_measured | managed | 1.474x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.0653 ms | not_measured | managed | 0.545x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 15.8284 ms | not_measured | managed | 1.655x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.0107 ms | not_measured | managed | 2.645x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 1.6676 ms | not_measured | managed | 2.874x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.0028 ms | not_measured | managed | 0.321x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.2370 ms | not_measured | managed | 1.615x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 67.2047 ms | not_measured | managed | 0.502x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.0051 ms | not_measured | managed | 0.843x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.4736 ms | not_measured | managed | 0.620x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 94.8897 ms | not_measured | managed | 0.321x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.231x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.0305 ms | not_measured | managed | 0.289x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 7.3520 ms | not_measured | managed | 1.838x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.944x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.0646 ms | not_measured | managed | 0.423x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 18.5637 ms | not_measured | managed | 0.903x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.0158 ms | not_measured | managed | 1.943x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 3.2354 ms | not_measured | managed | 1.804x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.0291 ms | not_measured | managed | 1.041x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 7.1299 ms | not_measured | managed | 1.809x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.0633 ms | not_measured | managed | 0.532x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 19.5408 ms | not_measured | managed | 1.318x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.778x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.0090 ms | not_measured | managed | 3.433x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 2.3187 ms | not_measured | managed | 2.000x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.0157 ms | not_measured | managed | 1.975x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 3.9901 ms | not_measured | managed | 1.811x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.0304 ms | not_measured | managed | 0.987x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 6.4765 ms | not_measured | managed | 1.394x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.0017 ms | not_measured | managed | 1.235x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.0625 ms | not_measured | managed | 0.589x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 21.5750 ms | not_measured | managed | 1.225x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.0091 ms | not_measured | managed | 3.275x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 1.9355 ms | not_measured | managed | 2.606x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.0278 ms | not_measured | managed | 0.460x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 11.2681 ms | not_measured | managed | 1.117x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.0701 ms | not_measured | managed | 0.593x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 26.8739 ms | not_measured | managed | 0.620x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.0031 ms | not_measured | managed | 0.645x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.0833 ms | not_measured | managed | 1.049x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 29.7452 ms | not_measured | managed | 0.979x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.633x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.0867 ms | not_measured | managed | 0.924x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 31.5159 ms | not_measured | managed | 1.156x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.0026 ms | not_measured | managed | 0.385x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.2767 ms | not_measured | managed | 1.369x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 25.8775 ms | not_measured | managed | 1.225x |
| `scalar - a (float16)` | float16 | 1,000 | 0.0050 ms | not_measured | managed | 1.080x |
| `scalar - a (float16)` | float16 | 100,000 | 0.4839 ms | not_measured | managed | 0.634x |
| `scalar - a (float16)` | float16 | 10,000,000 | 45.6205 ms | not_measured | managed | 0.691x |
| `scalar - a (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `scalar - a (float32)` | float32 | 100,000 | 0.0272 ms | not_measured | managed | 0.283x |
| `scalar - a (float32)` | float32 | 10,000,000 | 5.5845 ms | not_measured | managed | 2.389x |
| `scalar - a (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `scalar - a (float64)` | float64 | 100,000 | 0.0600 ms | not_measured | managed | 0.253x |
| `scalar - a (float64)` | float64 | 10,000,000 | 13.6157 ms | not_measured | managed | 1.154x |
| `scalar - a (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `scalar - a (int16)` | int16 | 100,000 | 0.0175 ms | not_measured | managed | 1.423x |
| `scalar - a (int16)` | int16 | 10,000,000 | 2.8236 ms | not_measured | managed | 1.775x |
| `scalar - a (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `scalar - a (int32)` | int32 | 100,000 | 0.0286 ms | not_measured | managed | 0.871x |
| `scalar - a (int32)` | int32 | 10,000,000 | 5.8525 ms | not_measured | managed | 2.134x |
| `scalar - a (int64)` | int64 | 1,000 | 0.0020 ms | not_measured | managed | 0.500x |
| `scalar - a (int64)` | int64 | 100,000 | 0.0695 ms | not_measured | managed | 0.371x |
| `scalar - a (int64)` | int64 | 10,000,000 | 17.5333 ms | not_measured | managed | 1.384x |
| `scalar - a (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 0.800x |
| `scalar - a (int8)` | int8 | 100,000 | 0.0093 ms | not_measured | managed | 2.581x |
| `scalar - a (int8)` | int8 | 10,000,000 | 2.1699 ms | not_measured | managed | 1.951x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.0168 ms | not_measured | managed | 1.494x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 2.9692 ms | not_measured | managed | 2.208x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.0290 ms | not_measured | managed | 0.828x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 5.2487 ms | not_measured | managed | 1.571x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.0565 ms | not_measured | managed | 0.434x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 17.3005 ms | not_measured | managed | 1.475x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 0.800x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.0093 ms | not_measured | managed | 2.656x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 1.7818 ms | not_measured | managed | 2.442x |
| `scalar / a (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `scalar / a (float32)` | float32 | 100,000 | 0.0338 ms | not_measured | managed | 0.373x |
| `scalar / a (float32)` | float32 | 10,000,000 | 5.1912 ms | not_measured | managed | 2.034x |
| `scalar / a (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.500x |
| `scalar / a (float64)` | float64 | 100,000 | 0.0680 ms | not_measured | managed | 0.590x |
| `scalar / a (float64)` | float64 | 10,000,000 | 13.9563 ms | not_measured | managed | 1.139x |
| `scalar / a (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.474x |
| `scalar / a (int32)` | int32 | 100,000 | 0.0847 ms | not_measured | managed | 0.758x |
| `scalar / a (int32)` | int32 | 10,000,000 | 13.8389 ms | not_measured | managed | 1.906x |
| `scalar / a (int64)` | int64 | 1,000 | 0.0034 ms | not_measured | managed | 0.500x |
| `scalar / a (int64)` | int64 | 100,000 | 0.0852 ms | not_measured | managed | 0.891x |
| `scalar / a (int64)` | int64 | 10,000,000 | 15.0967 ms | not_measured | managed | 1.886x |
| `a & b (bool)` | bool | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `a & b (bool)` | bool | 100,000 | 0.0560 ms | not_measured | managed | 0.061x |
| `a & b (bool)` | bool | 10,000,000 | 5.8648 ms | not_measured | managed | 0.350x |
| `a & b (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `a & b (int16)` | int16 | 100,000 | 0.0301 ms | not_measured | managed | 0.990x |
| `a & b (int16)` | int16 | 10,000,000 | 6.3181 ms | not_measured | managed | 0.807x |
| `a & b (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `a & b (int32)` | int32 | 100,000 | 0.0771 ms | not_measured | managed | 0.375x |
| `a & b (int32)` | int32 | 10,000,000 | 11.8863 ms | not_measured | managed | 0.819x |
| `a & b (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.242x |
| `a & b (int64)` | int64 | 100,000 | 0.1183 ms | not_measured | managed | 0.290x |
| `a & b (int64)` | int64 | 10,000,000 | 28.4663 ms | not_measured | managed | 0.595x |
| `a & b (int8)` | int8 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `a & b (int8)` | int8 | 100,000 | 0.0156 ms | not_measured | managed | 1.833x |
| `a & b (int8)` | int8 | 10,000,000 | 3.8546 ms | not_measured | managed | 0.982x |
| `a & b (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a & b (uint16)` | uint16 | 100,000 | 0.0356 ms | not_measured | managed | 0.843x |
| `a & b (uint16)` | uint16 | 10,000,000 | 6.9550 ms | not_measured | managed | 0.738x |
| `a & b (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a & b (uint32)` | uint32 | 100,000 | 0.0686 ms | not_measured | managed | 0.417x |
| `a & b (uint32)` | uint32 | 10,000,000 | 12.9490 ms | not_measured | managed | 0.643x |
| `a & b (uint64)` | uint64 | 1,000 | 0.0033 ms | not_measured | managed | 0.273x |
| `a & b (uint64)` | uint64 | 100,000 | 0.1238 ms | not_measured | managed | 0.321x |
| `a & b (uint64)` | uint64 | 10,000,000 | 28.6289 ms | not_measured | managed | 0.670x |
| `a & b (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a & b (uint8)` | uint8 | 100,000 | 0.0194 ms | not_measured | managed | 1.479x |
| `a & b (uint8)` | uint8 | 10,000,000 | 3.7909 ms | not_measured | managed | 1.012x |
| `a ^ b (bool)` | bool | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a ^ b (bool)` | bool | 100,000 | 0.0539 ms | not_measured | managed | 0.058x |
| `a ^ b (bool)` | bool | 10,000,000 | 6.4347 ms | not_measured | managed | 0.323x |
| `a ^ b (int16)` | int16 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `a ^ b (int16)` | int16 | 100,000 | 0.0287 ms | not_measured | managed | 1.014x |
| `a ^ b (int16)` | int16 | 10,000,000 | 6.6677 ms | not_measured | managed | 0.782x |
| `a ^ b (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `a ^ b (int32)` | int32 | 100,000 | 0.0652 ms | not_measured | managed | 0.443x |
| `a ^ b (int32)` | int32 | 10,000,000 | 12.8917 ms | not_measured | managed | 0.687x |
| `a ^ b (int64)` | int64 | 1,000 | 0.0034 ms | not_measured | managed | 0.235x |
| `a ^ b (int64)` | int64 | 100,000 | 0.1284 ms | not_measured | managed | 0.274x |
| `a ^ b (int64)` | int64 | 10,000,000 | 35.2017 ms | not_measured | managed | 0.518x |
| `a ^ b (int8)` | int8 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `a ^ b (int8)` | int8 | 100,000 | 0.0158 ms | not_measured | managed | 1.848x |
| `a ^ b (int8)` | int8 | 10,000,000 | 3.8045 ms | not_measured | managed | 0.997x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.0330 ms | not_measured | managed | 0.876x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 6.5946 ms | not_measured | managed | 0.767x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.0599 ms | not_measured | managed | 0.479x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 12.2722 ms | not_measured | managed | 0.688x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.1302 ms | not_measured | managed | 0.260x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 34.5739 ms | not_measured | managed | 0.535x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.0173 ms | not_measured | managed | 1.665x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 3.1801 ms | not_measured | managed | 1.191x |
| `a | b (bool)` | bool | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `a | b (bool)` | bool | 100,000 | 0.0561 ms | not_measured | managed | 0.053x |
| `a | b (bool)` | bool | 10,000,000 | 6.6117 ms | not_measured | managed | 0.307x |
| `a | b (int16)` | int16 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `a | b (int16)` | int16 | 100,000 | 0.0321 ms | not_measured | managed | 0.910x |
| `a | b (int16)` | int16 | 10,000,000 | 6.3806 ms | not_measured | managed | 0.805x |
| `a | b (int32)` | int32 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `a | b (int32)` | int32 | 100,000 | 0.0627 ms | not_measured | managed | 0.458x |
| `a | b (int32)` | int32 | 10,000,000 | 12.3715 ms | not_measured | managed | 0.735x |
| `a | b (int64)` | int64 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `a | b (int64)` | int64 | 100,000 | 0.1242 ms | not_measured | managed | 0.277x |
| `a | b (int64)` | int64 | 10,000,000 | 29.6977 ms | not_measured | managed | 0.596x |
| `a | b (int8)` | int8 | 1,000 | 0.0026 ms | not_measured | managed | 0.269x |
| `a | b (int8)` | int8 | 100,000 | 0.0153 ms | not_measured | managed | 2.000x |
| `a | b (int8)` | int8 | 10,000,000 | 3.9582 ms | not_measured | managed | 0.992x |
| `a | b (uint16)` | uint16 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `a | b (uint16)` | uint16 | 100,000 | 0.0352 ms | not_measured | managed | 0.815x |
| `a | b (uint16)` | uint16 | 10,000,000 | 7.0718 ms | not_measured | managed | 0.727x |
| `a | b (uint32)` | uint32 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `a | b (uint32)` | uint32 | 100,000 | 0.0589 ms | not_measured | managed | 0.487x |
| `a | b (uint32)` | uint32 | 10,000,000 | 12.2409 ms | not_measured | managed | 0.681x |
| `a | b (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a | b (uint64)` | uint64 | 100,000 | 0.1183 ms | not_measured | managed | 0.313x |
| `a | b (uint64)` | uint64 | 10,000,000 | 31.3380 ms | not_measured | managed | 0.554x |
| `a | b (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a | b (uint8)` | uint8 | 100,000 | 0.0166 ms | not_measured | managed | 1.771x |
| `a | b (uint8)` | uint8 | 10,000,000 | 3.6036 ms | not_measured | managed | 1.117x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.0030 ms | not_measured | managed | 0.133x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.0557 ms | not_measured | managed | 0.056x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 6.1728 ms | not_measured | managed | 0.347x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.0312 ms | not_measured | managed | 0.949x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 6.5475 ms | not_measured | managed | 0.771x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.0625 ms | not_measured | managed | 0.459x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 12.3488 ms | not_measured | managed | 0.690x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.1250 ms | not_measured | managed | 0.263x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 32.5932 ms | not_measured | managed | 0.523x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.0192 ms | not_measured | managed | 1.500x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 3.7347 ms | not_measured | managed | 1.018x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.0327 ms | not_measured | managed | 0.890x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 6.9221 ms | not_measured | managed | 0.729x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.0613 ms | not_measured | managed | 0.485x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 12.0652 ms | not_measured | managed | 0.697x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.1526 ms | not_measured | managed | 0.221x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 28.1371 ms | not_measured | managed | 0.613x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.0161 ms | not_measured | managed | 1.789x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 3.6808 ms | not_measured | managed | 1.044x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.0025 ms | not_measured | managed | 0.160x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.0542 ms | not_measured | managed | 0.042x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 5.9899 ms | not_measured | managed | 0.285x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.0337 ms | not_measured | managed | 0.869x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 5.0159 ms | not_measured | managed | 0.927x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.0576 ms | not_measured | managed | 0.453x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 9.0233 ms | not_measured | managed | 0.859x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.0031 ms | not_measured | managed | 0.226x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.1137 ms | not_measured | managed | 0.231x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 22.2517 ms | not_measured | managed | 0.696x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.0158 ms | not_measured | managed | 1.665x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 2.9494 ms | not_measured | managed | 1.190x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 1.000x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.0322 ms | not_measured | managed | 0.826x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 4.8219 ms | not_measured | managed | 0.963x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.0026 ms | not_measured | managed | 0.308x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.0570 ms | not_measured | managed | 0.463x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 8.9782 ms | not_measured | managed | 0.849x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.1167 ms | not_measured | managed | 0.225x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 21.8891 ms | not_measured | managed | 0.702x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.0153 ms | not_measured | managed | 1.686x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 2.4929 ms | not_measured | managed | 1.420x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.0028 ms | not_measured | managed | 0.143x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.0555 ms | not_measured | managed | 0.054x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 6.2329 ms | not_measured | managed | 0.318x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.0363 ms | not_measured | managed | 0.793x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 6.6520 ms | not_measured | managed | 0.773x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.0026 ms | not_measured | managed | 0.308x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.0627 ms | not_measured | managed | 0.472x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 12.2088 ms | not_measured | managed | 0.700x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.1344 ms | not_measured | managed | 0.250x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 29.6976 ms | not_measured | managed | 0.584x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.0160 ms | not_measured | managed | 1.844x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 3.8837 ms | not_measured | managed | 0.995x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.0317 ms | not_measured | managed | 0.905x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 6.2448 ms | not_measured | managed | 0.813x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.0605 ms | not_measured | managed | 0.499x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 12.0356 ms | not_measured | managed | 0.700x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.0027 ms | not_measured | managed | 0.296x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.1198 ms | not_measured | managed | 0.283x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 29.2974 ms | not_measured | managed | 0.577x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.0019 ms | not_measured | managed | 0.368x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.0161 ms | not_measured | managed | 1.814x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 3.6788 ms | not_measured | managed | 1.061x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.0029 ms | not_measured | managed | 0.138x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.0560 ms | not_measured | managed | 0.055x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 6.0434 ms | not_measured | managed | 0.325x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.0430 ms | not_measured | managed | 0.672x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 6.6387 ms | not_measured | managed | 0.775x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.0675 ms | not_measured | managed | 0.436x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 12.7413 ms | not_measured | managed | 0.657x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.1281 ms | not_measured | managed | 0.258x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 32.5830 ms | not_measured | managed | 0.550x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.0161 ms | not_measured | managed | 1.795x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 4.1683 ms | not_measured | managed | 0.926x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.0354 ms | not_measured | managed | 0.862x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 6.4238 ms | not_measured | managed | 0.778x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.0628 ms | not_measured | managed | 0.482x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 12.1869 ms | not_measured | managed | 0.694x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.1335 ms | not_measured | managed | 0.255x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 27.7808 ms | not_measured | managed | 0.648x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.0153 ms | not_measured | managed | 1.869x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 3.3477 ms | not_measured | managed | 1.133x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.0546 ms | not_measured | managed | 0.042x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 6.1000 ms | not_measured | managed | 0.295x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 0.875x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.0283 ms | not_measured | managed | 0.929x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 4.7978 ms | not_measured | managed | 0.984x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.0815 ms | not_measured | managed | 0.323x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 9.1047 ms | not_measured | managed | 0.881x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.1194 ms | not_measured | managed | 0.221x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 23.7658 ms | not_measured | managed | 0.649x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.0150 ms | not_measured | managed | 1.747x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 3.0665 ms | not_measured | managed | 1.143x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.0335 ms | not_measured | managed | 0.773x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 5.3370 ms | not_measured | managed | 0.863x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.0585 ms | not_measured | managed | 0.451x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 10.0599 ms | not_measured | managed | 0.757x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.1348 ms | not_measured | managed | 0.194x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 26.9743 ms | not_measured | managed | 0.566x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.0183 ms | not_measured | managed | 1.437x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 2.6690 ms | not_measured | managed | 1.301x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.0060 ms | not_measured | managed | 0.250x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.1297 ms | not_measured | managed | 0.344x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 16.8286 ms | not_measured | managed | 0.912x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.0036 ms | not_measured | managed | 0.306x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.0316 ms | not_measured | managed | 0.899x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.7349 ms | not_measured | managed | 1.032x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.0043 ms | not_measured | managed | 0.233x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.0647 ms | not_measured | managed | 0.295x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 9.0874 ms | not_measured | managed | 0.859x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.0048 ms | not_measured | managed | 0.188x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.1261 ms | not_measured | managed | 0.166x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 22.1847 ms | not_measured | managed | 0.823x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.0027 ms | not_measured | managed | 0.333x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.0185 ms | not_measured | managed | 1.562x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 2.3903 ms | not_measured | managed | 1.548x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0042 ms | not_measured | managed | 0.262x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0344 ms | not_measured | managed | 0.858x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.8970 ms | not_measured | managed | 0.970x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0042 ms | not_measured | managed | 0.238x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0555 ms | not_measured | managed | 0.346x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.6482 ms | not_measured | managed | 0.869x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0036 ms | not_measured | managed | 0.278x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.1183 ms | not_measured | managed | 0.165x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 23.4199 ms | not_measured | managed | 0.654x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0055 ms | not_measured | managed | 0.164x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0201 ms | not_measured | managed | 1.408x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 2.8818 ms | not_measured | managed | 1.273x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.0087 ms | not_measured | managed | 0.207x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.1268 ms | not_measured | managed | 0.421x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | 16.9605 ms | not_measured | managed | 0.904x |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.0040 ms | not_measured | managed | 0.300x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.0349 ms | not_measured | managed | 1.072x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.8798 ms | not_measured | managed | 1.163x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.0042 ms | not_measured | managed | 0.262x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.0602 ms | not_measured | managed | 0.478x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 9.0761 ms | not_measured | managed | 0.875x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.0039 ms | not_measured | managed | 0.282x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.1329 ms | not_measured | managed | 0.219x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 28.7993 ms | not_measured | managed | 0.570x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.0027 ms | not_measured | managed | 0.370x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.0184 ms | not_measured | managed | 2.060x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 3.4144 ms | not_measured | managed | 1.353x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0057 ms | not_measured | managed | 0.211x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0342 ms | not_measured | managed | 0.833x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.9817 ms | not_measured | managed | 0.953x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0045 ms | not_measured | managed | 0.222x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0568 ms | not_measured | managed | 0.340x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.8058 ms | not_measured | managed | 0.858x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0044 ms | not_measured | managed | 0.227x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.1399 ms | not_measured | managed | 0.140x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 25.8004 ms | not_measured | managed | 0.629x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0041 ms | not_measured | managed | 0.220x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0188 ms | not_measured | managed | 1.543x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 3.6806 ms | not_measured | managed | 1.015x |
| `matrix * 2.0` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `matrix * 2.0` | float64 | 100,000 | 0.0444 ms | not_measured | managed | 0.270x |
| `matrix * 2.0` | float64 | 10,000,000 | 14.5212 ms | not_measured | managed | 1.379x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.371x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.0549 ms | not_measured | managed | 0.499x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 14.4097 ms | not_measured | managed | 1.753x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.464x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.0463 ms | not_measured | managed | 0.553x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 13.6564 ms | not_measured | managed | 1.616x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.500x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.0580 ms | not_measured | managed | 0.469x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 14.5435 ms | not_measured | managed | 1.651x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.429x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.0468 ms | not_measured | managed | 0.545x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 13.7227 ms | not_measured | managed | 1.517x |
| `matrix + scalar` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `matrix + scalar` | float64 | 100,000 | 0.0456 ms | not_measured | managed | 0.274x |
| `matrix + scalar` | float64 | 10,000,000 | 14.0757 ms | not_measured | managed | 1.220x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 3.000x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 2.571x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 3.000x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 3.000x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 3.000x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 2.571x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.244x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.0929 ms | not_measured | managed | 0.270x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 14.1971 ms | not_measured | managed | 1.462x |
| `a != b (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.462x |
| `a != b (float32)` | float32 | 100,000 | 0.0118 ms | not_measured | managed | 0.636x |
| `a != b (float32)` | float32 | 10,000,000 | 11.4788 ms | not_measured | managed | 0.372x |
| `a != b (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.417x |
| `a != b (float64)` | float64 | 100,000 | 0.0298 ms | not_measured | managed | 0.359x |
| `a != b (float64)` | float64 | 10,000,000 | 22.7525 ms | not_measured | managed | 0.296x |
| `a != b (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `a != b (int32)` | int32 | 100,000 | 0.0182 ms | not_measured | managed | 0.429x |
| `a != b (int32)` | int32 | 10,000,000 | 11.6553 ms | not_measured | managed | 0.366x |
| `a != b (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `a != b (int64)` | int64 | 100,000 | 0.0220 ms | not_measured | managed | 0.614x |
| `a != b (int64)` | int64 | 10,000,000 | 21.1853 ms | not_measured | managed | 0.342x |
| `a < b (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `a < b (float32)` | float32 | 100,000 | 0.0110 ms | not_measured | managed | 0.636x |
| `a < b (float32)` | float32 | 10,000,000 | 10.8252 ms | not_measured | managed | 0.380x |
| `a < b (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a < b (float64)` | float64 | 100,000 | 0.0281 ms | not_measured | managed | 0.384x |
| `a < b (float64)` | float64 | 10,000,000 | 21.6593 ms | not_measured | managed | 0.326x |
| `a < b (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `a < b (int32)` | int32 | 100,000 | 0.0119 ms | not_measured | managed | 0.588x |
| `a < b (int32)` | int32 | 10,000,000 | 11.0188 ms | not_measured | managed | 0.382x |
| `a < b (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 0.500x |
| `a < b (int64)` | int64 | 100,000 | 0.0283 ms | not_measured | managed | 0.654x |
| `a < b (int64)` | int64 | 10,000,000 | 20.3228 ms | not_measured | managed | 0.338x |
| `a <= b (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `a <= b (float32)` | float32 | 100,000 | 0.0107 ms | not_measured | managed | 0.645x |
| `a <= b (float32)` | float32 | 10,000,000 | 12.9963 ms | not_measured | managed | 0.333x |
| `a <= b (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a <= b (float64)` | float64 | 100,000 | 0.0291 ms | not_measured | managed | 0.357x |
| `a <= b (float64)` | float64 | 10,000,000 | 21.9882 ms | not_measured | managed | 0.309x |
| `a <= b (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a <= b (int32)` | int32 | 100,000 | 0.0180 ms | not_measured | managed | 0.383x |
| `a <= b (int32)` | int32 | 10,000,000 | 11.3990 ms | not_measured | managed | 0.374x |
| `a <= b (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `a <= b (int64)` | int64 | 100,000 | 0.0209 ms | not_measured | managed | 0.866x |
| `a <= b (int64)` | int64 | 10,000,000 | 20.3791 ms | not_measured | managed | 0.351x |
| `a == b (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `a == b (float32)` | float32 | 100,000 | 0.0122 ms | not_measured | managed | 0.582x |
| `a == b (float32)` | float32 | 10,000,000 | 10.7173 ms | not_measured | managed | 0.376x |
| `a == b (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a == b (float64)` | float64 | 100,000 | 0.0324 ms | not_measured | managed | 0.330x |
| `a == b (float64)` | float64 | 10,000,000 | 20.6593 ms | not_measured | managed | 0.333x |
| `a == b (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `a == b (int32)` | int32 | 100,000 | 0.0189 ms | not_measured | managed | 0.381x |
| `a == b (int32)` | int32 | 10,000,000 | 12.1659 ms | not_measured | managed | 0.376x |
| `a == b (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `a == b (int64)` | int64 | 100,000 | 0.0188 ms | not_measured | managed | 0.734x |
| `a == b (int64)` | int64 | 10,000,000 | 20.8245 ms | not_measured | managed | 0.357x |
| `a > b (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `a > b (float32)` | float32 | 100,000 | 0.0111 ms | not_measured | managed | 0.640x |
| `a > b (float32)` | float32 | 10,000,000 | 11.6422 ms | not_measured | managed | 0.338x |
| `a > b (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `a > b (float64)` | float64 | 100,000 | 0.0289 ms | not_measured | managed | 0.370x |
| `a > b (float64)` | float64 | 10,000,000 | 21.4919 ms | not_measured | managed | 0.342x |
| `a > b (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `a > b (int32)` | int32 | 100,000 | 0.0113 ms | not_measured | managed | 0.628x |
| `a > b (int32)` | int32 | 10,000,000 | 10.7765 ms | not_measured | managed | 0.393x |
| `a > b (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a > b (int64)` | int64 | 100,000 | 0.0211 ms | not_measured | managed | 0.872x |
| `a > b (int64)` | int64 | 10,000,000 | 20.6120 ms | not_measured | managed | 0.337x |
| `a >= b (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `a >= b (float32)` | float32 | 100,000 | 0.0117 ms | not_measured | managed | 0.598x |
| `a >= b (float32)` | float32 | 10,000,000 | 11.7650 ms | not_measured | managed | 0.363x |
| `a >= b (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `a >= b (float64)` | float64 | 100,000 | 0.0286 ms | not_measured | managed | 0.353x |
| `a >= b (float64)` | float64 | 10,000,000 | 23.5193 ms | not_measured | managed | 0.291x |
| `a >= b (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `a >= b (int32)` | int32 | 100,000 | 0.0135 ms | not_measured | managed | 0.548x |
| `a >= b (int32)` | int32 | 10,000,000 | 9.8543 ms | not_measured | managed | 0.456x |
| `a >= b (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a >= b (int64)` | int64 | 100,000 | 0.0212 ms | not_measured | managed | 0.896x |
| `a >= b (int64)` | int64 | 10,000,000 | 20.8227 ms | not_measured | managed | 0.350x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.0139 ms | not_measured | managed | 0.489x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 11.4896 ms | not_measured | managed | 0.351x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.0313 ms | not_measured | managed | 0.323x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 22.1563 ms | not_measured | managed | 0.290x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.0144 ms | not_measured | managed | 0.514x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 9.6667 ms | not_measured | managed | 0.475x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.0201 ms | not_measured | managed | 0.652x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 21.7217 ms | not_measured | managed | 0.323x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.0255 ms | not_measured | managed | 0.267x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 11.2991 ms | not_measured | managed | 0.355x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.0312 ms | not_measured | managed | 0.324x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 11.5786 ms | not_measured | managed | 0.559x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.0186 ms | not_measured | managed | 0.478x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 11.4566 ms | not_measured | managed | 0.405x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 0.462x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.0208 ms | not_measured | managed | 0.880x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 20.4487 ms | not_measured | managed | 0.348x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.0250 ms | not_measured | managed | 0.264x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 11.5138 ms | not_measured | managed | 0.350x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.0325 ms | not_measured | managed | 0.323x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 10.4863 ms | not_measured | managed | 0.646x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.0148 ms | not_measured | managed | 0.642x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 11.5367 ms | not_measured | managed | 0.402x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.0258 ms | not_measured | managed | 0.709x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 21.0064 ms | not_measured | managed | 0.355x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.0218 ms | not_measured | managed | 0.303x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 11.1296 ms | not_measured | managed | 0.360x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.0324 ms | not_measured | managed | 0.315x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 14.1860 ms | not_measured | managed | 0.453x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.0161 ms | not_measured | managed | 0.509x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 11.1667 ms | not_measured | managed | 0.427x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 0.500x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.0182 ms | not_measured | managed | 1.011x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 20.9079 ms | not_measured | managed | 0.343x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.0254 ms | not_measured | managed | 0.260x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 11.7340 ms | not_measured | managed | 0.354x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.0333 ms | not_measured | managed | 0.318x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 11.4363 ms | not_measured | managed | 0.574x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.0186 ms | not_measured | managed | 0.634x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 11.4904 ms | not_measured | managed | 0.428x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.0239 ms | not_measured | managed | 0.762x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 20.1472 ms | not_measured | managed | 0.381x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.0177 ms | not_measured | managed | 0.373x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 12.4622 ms | not_measured | managed | 0.331x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.0301 ms | not_measured | managed | 0.339x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 23.1326 ms | not_measured | managed | 0.278x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.417x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.0199 ms | not_measured | managed | 0.372x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 11.4977 ms | not_measured | managed | 0.463x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.0207 ms | not_measured | managed | 0.643x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 21.3541 ms | not_measured | managed | 0.343x |
| `a.copy() (float32)` | float32 | 1,000 | 0.0088 ms | not_measured | managed | 0.045x |
| `a.copy() (float32)` | float32 | 100,000 | 0.0073 ms | not_measured | managed | 0.808x |
| `a.copy() (float32)` | float32 | 10,000,000 | 2.2766 ms | not_measured | managed | 2.746x |
| `a.copy() (float64)` | float64 | 1,000 | 0.0146 ms | not_measured | managed | 0.027x |
| `a.copy() (float64)` | float64 | 100,000 | 0.0131 ms | not_measured | managed | 0.847x |
| `a.copy() (float64)` | float64 | 10,000,000 | 13.8165 ms | not_measured | managed | 0.915x |
| `a.copy() (int32)` | int32 | 1,000 | 0.0082 ms | not_measured | managed | 0.049x |
| `a.copy() (int32)` | int32 | 100,000 | 0.0073 ms | not_measured | managed | 0.767x |
| `a.copy() (int32)` | int32 | 10,000,000 | 2.5813 ms | not_measured | managed | 2.511x |
| `a.copy() (int64)` | int64 | 1,000 | 0.0051 ms | not_measured | managed | 0.078x |
| `a.copy() (int64)` | int64 | 100,000 | 0.0132 ms | not_measured | managed | 0.856x |
| `a.copy() (int64)` | int64 | 10,000,000 | 13.2569 ms | not_measured | managed | 0.976x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.0051 ms | not_measured | managed | 0.176x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.0451 ms | not_measured | managed | 0.876x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 4.2989 ms | not_measured | managed | 1.769x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.0081 ms | not_measured | managed | 0.086x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.0310 ms | not_measured | managed | 1.019x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 14.4098 ms | not_measured | managed | 0.865x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.0260 ms | not_measured | managed | 1.119x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 3.0559 ms | not_measured | managed | 2.273x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.0059 ms | not_measured | managed | 0.119x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.0220 ms | not_measured | managed | 0.855x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 12.8459 ms | not_measured | managed | 0.961x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.0131 ms | not_measured | managed | 0.031x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.0390 ms | not_measured | managed | 0.305x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 14.000x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.0066 ms | not_measured | managed | 1.848x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.098x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.0263 ms | not_measured | managed | 0.376x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.191x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.1194 ms | not_measured | managed | 0.226x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 2.714x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 3.167x |
| `np.copy (float32)` | float32 | 1,000 | 0.0102 ms | not_measured | managed | 0.059x |
| `np.copy (float32)` | float32 | 100,000 | 0.0073 ms | not_measured | managed | 0.822x |
| `np.copy (float32)` | float32 | 10,000,000 | 2.4215 ms | not_measured | managed | 2.539x |
| `np.copy (float64)` | float64 | 1,000 | 0.0140 ms | not_measured | managed | 0.043x |
| `np.copy (float64)` | float64 | 100,000 | 0.0130 ms | not_measured | managed | 0.962x |
| `np.copy (float64)` | float64 | 10,000,000 | 13.4404 ms | not_measured | managed | 0.943x |
| `np.copy (int32)` | int32 | 1,000 | 0.0101 ms | not_measured | managed | 0.059x |
| `np.copy (int32)` | int32 | 100,000 | 0.0074 ms | not_measured | managed | 0.797x |
| `np.copy (int32)` | int32 | 10,000,000 | 3.2580 ms | not_measured | managed | 2.000x |
| `np.copy (int64)` | int64 | 1,000 | 0.0157 ms | not_measured | managed | 0.051x |
| `np.copy (int64)` | int64 | 100,000 | 0.0133 ms | not_measured | managed | 0.872x |
| `np.copy (int64)` | int64 | 10,000,000 | 13.8573 ms | not_measured | managed | 0.934x |
| `np.empty (float32)` | float32 | 1,000 | 0.0009 ms | not_measured | managed | 0.333x |
| `np.empty (float32)` | float32 | 100,000 | 0.0011 ms | not_measured | managed | 0.273x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.0029 ms | not_measured | managed | 3.379x |
| `np.empty (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.231x |
| `np.empty (float64)` | float64 | 100,000 | 0.0016 ms | not_measured | managed | 0.188x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.0137 ms | not_measured | managed | 0.672x |
| `np.empty (int32)` | int32 | 1,000 | 0.0055 ms | not_measured | managed | 0.055x |
| `np.empty (int32)` | int32 | 100,000 | 0.0012 ms | not_measured | managed | 0.250x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.0014 ms | not_measured | managed | 6.714x |
| `np.empty (int64)` | int64 | 1,000 | 0.0069 ms | not_measured | managed | 0.043x |
| `np.empty (int64)` | int64 | 100,000 | 0.0014 ms | not_measured | managed | 0.214x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.0121 ms | not_measured | managed | 0.587x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.0051 ms | not_measured | managed | 0.078x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.0090 ms | not_measured | managed | 1.033x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.054x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.0056 ms | not_measured | managed | 0.054x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.0113 ms | not_measured | managed | 0.841x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.0045 ms | not_measured | managed | 0.089x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.0079 ms | not_measured | managed | 0.051x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.0053 ms | not_measured | managed | 1.717x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.0043 ms | not_measured | managed | 0.093x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.0111 ms | not_measured | managed | 0.036x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.0132 ms | not_measured | managed | 0.508x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.0174 ms | not_measured | managed | 0.075x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.0967 ms | not_measured | managed | 0.121x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.375x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.429x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.0339 ms | not_measured | managed | 2.841x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 9.8479 ms | not_measured | managed | 0.983x |
| `np.full (float32)` | float32 | 1,000 | 0.0059 ms | not_measured | managed | 0.153x |
| `np.full (float32)` | float32 | 100,000 | 0.0124 ms | not_measured | managed | 0.427x |
| `np.full (float32)` | float32 | 10,000,000 | 2.7515 ms | not_measured | managed | 2.616x |
| `np.full (float64)` | float64 | 1,000 | 0.0142 ms | not_measured | managed | 0.063x |
| `np.full (float64)` | float64 | 100,000 | 0.0153 ms | not_measured | managed | 0.634x |
| `np.full (float64)` | float64 | 10,000,000 | 12.6659 ms | not_measured | managed | 1.162x |
| `np.full (int32)` | int32 | 1,000 | 0.0033 ms | not_measured | managed | 0.273x |
| `np.full (int32)` | int32 | 100,000 | 0.0125 ms | not_measured | managed | 0.424x |
| `np.full (int32)` | int32 | 10,000,000 | 2.3814 ms | not_measured | managed | 3.196x |
| `np.full (int64)` | int64 | 1,000 | 0.0115 ms | not_measured | managed | 0.078x |
| `np.full (int64)` | int64 | 100,000 | 0.0148 ms | not_measured | managed | 0.655x |
| `np.full (int64)` | int64 | 10,000,000 | 12.6573 ms | not_measured | managed | 1.164x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.478x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.0115 ms | not_measured | managed | 0.478x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 2.6691 ms | not_measured | managed | 2.714x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.333x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.0115 ms | not_measured | managed | 0.870x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 2.6115 ms | not_measured | managed | 5.789x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.289x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.0115 ms | not_measured | managed | 0.478x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 2.9009 ms | not_measured | managed | 2.500x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.0050 ms | not_measured | managed | 0.220x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.0117 ms | not_measured | managed | 0.855x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 2.6640 ms | not_measured | managed | 5.479x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.0172 ms | not_measured | managed | 0.093x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.0962 ms | not_measured | managed | 0.125x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.0059 ms | not_measured | managed | 0.831x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.0451 ms | not_measured | managed | 6.330x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 4.4164 ms | not_measured | managed | 6.890x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 0.667x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.0677 ms | not_measured | managed | 0.736x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 12.3408 ms | not_measured | managed | 1.768x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.0051 ms | not_measured | managed | 1.078x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.1311 ms | not_measured | managed | 2.185x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 12.4531 ms | not_measured | managed | 2.875x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 1.667x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.1316 ms | not_measured | managed | 2.741x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 18.9200 ms | not_measured | managed | 2.198x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.0143 ms | not_measured | managed | 0.601x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.1032 ms | not_measured | managed | 3.419x |
| `np.ones (float32)` | float32 | 1,000 | 0.0067 ms | not_measured | managed | 0.134x |
| `np.ones (float32)` | float32 | 100,000 | 0.0125 ms | not_measured | managed | 0.424x |
| `np.ones (float32)` | float32 | 10,000,000 | 2.7862 ms | not_measured | managed | 2.646x |
| `np.ones (float64)` | float64 | 1,000 | 0.0084 ms | not_measured | managed | 0.107x |
| `np.ones (float64)` | float64 | 100,000 | 0.0154 ms | not_measured | managed | 0.688x |
| `np.ones (float64)` | float64 | 10,000,000 | 12.8499 ms | not_measured | managed | 1.151x |
| `np.ones (int32)` | int32 | 1,000 | 0.0053 ms | not_measured | managed | 0.170x |
| `np.ones (int32)` | int32 | 100,000 | 0.0123 ms | not_measured | managed | 0.431x |
| `np.ones (int32)` | int32 | 10,000,000 | 2.3303 ms | not_measured | managed | 3.192x |
| `np.ones (int64)` | int64 | 1,000 | 0.0110 ms | not_measured | managed | 0.082x |
| `np.ones (int64)` | int64 | 100,000 | 0.0142 ms | not_measured | managed | 0.697x |
| `np.ones (int64)` | int64 | 10,000,000 | 13.5040 ms | not_measured | managed | 1.084x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.550x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.0117 ms | not_measured | managed | 0.470x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 2.6201 ms | not_measured | managed | 2.740x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.172x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.0138 ms | not_measured | managed | 0.725x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 12.7516 ms | not_measured | managed | 1.164x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.0044 ms | not_measured | managed | 0.250x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.0117 ms | not_measured | managed | 0.479x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 2.3442 ms | not_measured | managed | 3.127x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.0046 ms | not_measured | managed | 0.217x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.0136 ms | not_measured | managed | 0.728x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 13.1624 ms | not_measured | managed | 1.098x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.208x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.0288 ms | not_measured | managed | 0.424x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.0254 ms | not_measured | managed | 0.177x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.1981 ms | not_measured | managed | 0.844x |
| `np.zeros (float32)` | float32 | 1,000 | 0.0055 ms | not_measured | managed | 0.073x |
| `np.zeros (float32)` | float32 | 100,000 | 0.0037 ms | not_measured | managed | 1.378x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.0040 ms | not_measured | managed | 2.350x |
| `np.zeros (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 1.250x |
| `np.zeros (float64)` | float64 | 100,000 | 0.0025 ms | not_measured | managed | 3.720x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.0037 ms | not_measured | managed | 2.649x |
| `np.zeros (int32)` | int32 | 1,000 | 0.0062 ms | not_measured | managed | 0.065x |
| `np.zeros (int32)` | int32 | 100,000 | 0.0021 ms | not_measured | managed | 2.286x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.0029 ms | not_measured | managed | 3.276x |
| `np.zeros (int64)` | int64 | 1,000 | 0.0097 ms | not_measured | managed | 0.041x |
| `np.zeros (int64)` | int64 | 100,000 | 0.0026 ms | not_measured | managed | 3.577x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.0087 ms | not_measured | managed | 0.770x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.0040 ms | not_measured | managed | 0.250x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.0024 ms | not_measured | managed | 2.250x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.0037 ms | not_measured | managed | 1956.378x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.435x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.0025 ms | not_measured | managed | 4.000x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.0038 ms | not_measured | managed | 3981.579x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.0049 ms | not_measured | managed | 0.204x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.0024 ms | not_measured | managed | 2.250x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.0030 ms | not_measured | managed | 2420.967x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.0070 ms | not_measured | managed | 0.143x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.0025 ms | not_measured | managed | 3.960x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.0091 ms | not_measured | managed | 1583.681x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.0152 ms | not_measured | managed | 0.546x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 1.4521 ms | not_measured | managed | 1.089x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.0453 ms | not_measured | managed | 0.863x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 6.8639 ms | not_measured | managed | 0.664x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.0089 ms | not_measured | managed | 0.393x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.2775 ms | not_measured | managed | 2.046x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.0442 ms | not_measured | managed | 0.808x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 7.1109 ms | not_measured | managed | 0.629x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.0078 ms | not_measured | managed | 0.577x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.2918 ms | not_measured | managed | 1.179x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.0081 ms | not_measured | managed | 0.877x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 0.7936 ms | not_measured | managed | 1.123x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.0161 ms | not_measured | managed | 0.571x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 1.4422 ms | not_measured | managed | 1.118x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.0464 ms | not_measured | managed | 0.869x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 7.0775 ms | not_measured | managed | 0.652x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.0464 ms | not_measured | managed | 0.802x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 7.1635 ms | not_measured | managed | 0.644x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.0087 ms | not_measured | managed | 0.540x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.2973 ms | not_measured | managed | 1.139x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.0081 ms | not_measured | managed | 1.025x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 0.7949 ms | not_measured | managed | 1.141x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.0073 ms | not_measured | managed | 0.986x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 0.7151 ms | not_measured | managed | 1.241x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.0252 ms | not_measured | managed | 1.052x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 3.4142 ms | not_measured | managed | 0.695x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.0240 ms | not_measured | managed | 1.096x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 3.4587 ms | not_measured | managed | 0.690x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 0.943x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 0.6708 ms | not_measured | managed | 1.294x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.0228 ms | not_measured | managed | 1.382x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 3.3874 ms | not_measured | managed | 0.692x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.486x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.1102 ms | not_measured | managed | 0.243x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.0236 ms | not_measured | managed | 1.038x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 3.3509 ms | not_measured | managed | 0.677x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.667x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.667x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.0196 ms | 0.0082 ms | openblas | 1.110x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 4.7925 ms | not_measured | managed | 0.259x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 128 | 0.3470 ms | 0.1030 ms | openblas | 2.407x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.073x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 1.6355 ms | not_measured | managed | 0.015x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 71.7141 ms | not_measured | managed | 0.062x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 2.556x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.0034 ms | not_measured | managed | 2.824x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.0018 ms | not_measured | managed | 4.944x |
| `np.inner(a, b)` | float64 | 100,000 | 0.0165 ms | 0.0089 ms | openblas | 1.034x |
| `np.linalg.cholesky(a)` | float64 | 32 | missing_backend | 0.0159 ms | openblas | 0.308x |
| `np.linalg.cholesky(a)` | float64 | 96 | missing_backend | 0.0624 ms | openblas | 0.321x |
| `np.linalg.cond(a)` | float64 | 32 | missing_backend | 0.0525 ms | openblas | 0.707x |
| `np.linalg.cond(a)` | float64 | 96 | missing_backend | 0.3435 ms | openblas | 0.866x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.0465 ms | not_measured | managed | 0.331x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 1.6649 ms | not_measured | managed | 0.410x |
| `np.linalg.det(a)` | float64 | 32 | missing_backend | 0.0100 ms | openblas | 0.670x |
| `np.linalg.det(a)` | float64 | 96 | missing_backend | 0.0351 ms | openblas | 1.188x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 1.143x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.linalg.eig(a)` | float64 | 32 | missing_backend | 0.1592 ms | openblas | 0.702x |
| `np.linalg.eig(a)` | float64 | 96 | missing_backend | 1.6014 ms | openblas | 0.916x |
| `np.linalg.eigh(a)` | float64 | 32 | missing_backend | 0.0634 ms | openblas | 0.787x |
| `np.linalg.eigh(a)` | float64 | 96 | missing_backend | 0.4731 ms | openblas | 0.901x |
| `np.linalg.eigvals(a)` | float64 | 32 | missing_backend | 0.1330 ms | openblas | 0.538x |
| `np.linalg.eigvals(a)` | float64 | 96 | missing_backend | 1.0080 ms | openblas | 0.824x |
| `np.linalg.eigvalsh(a)` | float64 | 32 | missing_backend | 0.0280 ms | openblas | 0.825x |
| `np.linalg.eigvalsh(a)` | float64 | 96 | missing_backend | 0.1874 ms | openblas | 0.948x |
| `np.linalg.inv(a)` | float64 | 32 | missing_backend | 0.0231 ms | openblas | 0.519x |
| `np.linalg.inv(a)` | float64 | 96 | missing_backend | 0.0976 ms | openblas | 0.942x |
| `np.linalg.lstsq(a, b)` | float64 | 32 | missing_backend | 0.0992 ms | openblas | 0.844x |
| `np.linalg.lstsq(a, b)` | float64 | 96 | missing_backend | 0.9049 ms | openblas | 1.000x |
| `np.linalg.matmul(a, b)` | float64 | 128 | 0.3131 ms | 0.1032 ms | openblas | 0.588x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.0136 ms | not_measured | managed | 0.199x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 0.4949 ms | not_measured | managed | 0.161x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.0167 ms | not_measured | managed | 0.168x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.0751 ms | not_measured | managed | 0.097x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0399 ms | openblas | 0.965x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 96 | missing_backend | 0.3253 ms | openblas | 1.406x |
| `np.linalg.matrix_power(a, -1)` | float64 | 32 | missing_backend | 0.0205 ms | openblas | 0.590x |
| `np.linalg.matrix_power(a, -1)` | float64 | 96 | missing_backend | 0.1032 ms | openblas | 0.852x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.0261 ms | not_measured | managed | 0.222x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.9435 ms | not_measured | managed | 0.148x |
| `np.linalg.matrix_rank(a)` | float64 | 32 | missing_backend | 0.0502 ms | openblas | 0.765x |
| `np.linalg.matrix_rank(a)` | float64 | 96 | missing_backend | 0.3256 ms | openblas | 0.988x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.linalg.multi_dot(arrays)` | float64 | 128 | 0.4392 ms | 0.2166 ms | openblas | 0.587x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.0268 ms | not_measured | managed | 0.209x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.9438 ms | not_measured | managed | 0.181x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.278x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.0141 ms | not_measured | managed | 8.113x |
| `np.linalg.norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0400 ms | openblas | 0.960x |
| `np.linalg.norm(a, ord=2)` | float64 | 96 | missing_backend | 0.3202 ms | openblas | 1.053x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.224x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.1731 ms | not_measured | managed | 0.245x |
| `np.linalg.pinv(a)` | float64 | 32 | missing_backend | 0.1311 ms | openblas | 0.717x |
| `np.linalg.pinv(a)` | float64 | 96 | missing_backend | 0.9779 ms | openblas | 0.830x |
| `np.linalg.qr(a)` | float64 | 32 | missing_backend | 0.0390 ms | openblas | 0.638x |
| `np.linalg.qr(a)` | float64 | 96 | missing_backend | 0.2086 ms | openblas | 0.876x |
| `np.linalg.slogdet(a)` | float64 | 32 | missing_backend | 0.0099 ms | openblas | 0.717x |
| `np.linalg.slogdet(a)` | float64 | 96 | missing_backend | 0.0357 ms | openblas | 0.888x |
| `np.linalg.solve(a, b)` | float64 | 32 | missing_backend | 0.0096 ms | openblas | 0.906x |
| `np.linalg.solve(a, b)` | float64 | 96 | missing_backend | 0.0350 ms | openblas | 0.917x |
| `np.linalg.svd(a)` | float64 | 32 | missing_backend | 0.0959 ms | openblas | 0.856x |
| `np.linalg.svd(a)` | float64 | 96 | missing_backend | 0.8168 ms | openblas | 0.898x |
| `np.linalg.svdvals(a)` | float64 | 32 | missing_backend | 0.0371 ms | openblas | 0.895x |
| `np.linalg.svdvals(a)` | float64 | 96 | missing_backend | 0.3204 ms | openblas | 0.929x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0176 ms | not_measured | managed | 0.347x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.5093 ms | not_measured | managed | 0.165x |
| `np.linalg.tensordot(a, b, axes=1)` | float64 | 128 | 0.2382 ms | 0.1074 ms | openblas | 0.616x |
| `np.linalg.tensorinv(a)` | float64 | 32 | missing_backend | 0.0187 ms | openblas | 0.690x |
| `np.linalg.tensorinv(a)` | float64 | 96 | missing_backend | 0.0993 ms | openblas | 0.872x |
| `np.linalg.tensorsolve(a, b)` | float64 | 32 | missing_backend | 0.0104 ms | openblas | 0.894x |
| `np.linalg.tensorsolve(a, b)` | float64 | 96 | missing_backend | 0.0360 ms | openblas | 0.967x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.941x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.0017 ms | not_measured | managed | 1.000x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0117 ms | not_measured | managed | 0.077x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1898 ms | 0.0163 ms | openblas | 0.589x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.0160 ms | not_measured | managed | 0.169x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.3384 ms | not_measured | managed | 0.096x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.0125 ms | not_measured | managed | 0.264x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 4.5232 ms | not_measured | managed | 0.149x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 3.1763 ms | not_measured | managed | 0.220x |
| `np.matmul(a, b)` | float64 | 128 | 0.3168 ms | 0.1024 ms | openblas | 0.586x |
| `np.matvec(a, b)` | float64 | 128 | 0.0343 ms | 0.0024 ms | openblas | 0.958x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.157x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.1326 ms | not_measured | managed | 0.046x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.1155 ms | not_measured | managed | 0.068x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 0.300x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.1677 ms | not_measured | managed | 0.247x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 15.6758 ms | not_measured | managed | 0.862x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0209 ms | not_measured | managed | 0.187x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 1.1357 ms | not_measured | managed | 0.101x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 68.5345 ms | not_measured | managed | 0.015x |
| `np.tensordot(a, b, axes=1)` | float64 | 128 | 0.2355 ms | 0.1136 ms | openblas | 0.588x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.111x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.0113 ms | 0.0089 ms | openblas | 1.022x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 3.9233 ms | not_measured | managed | 0.254x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.132x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1441 ms | 0.0080 ms | openblas | 1.150x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 19.6962 ms | not_measured | managed | 0.056x |
| `np.vecmat(a, b)` | float64 | 128 | 0.0051 ms | 0.0024 ms | openblas | 0.792x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.185x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.0391 ms | not_measured | managed | 0.159x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.0507 ms | not_measured | managed | 0.170x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.0665 ms | not_measured | managed | 0.468x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 1.2541 ms | not_measured | managed | 1.426x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.0339 ms | not_measured | managed | 0.392x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.7046 ms | not_measured | managed | 0.134x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.0328 ms | not_measured | managed | 0.418x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.6101 ms | not_measured | managed | 1.030x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.0028 ms | not_measured | managed | 0.893x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.1399 ms | not_measured | managed | 0.609x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 13.5604 ms | not_measured | managed | 0.710x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 1.067x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.0221 ms | not_measured | managed | 0.321x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 10.8867 ms | not_measured | managed | 0.371x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.889x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.0288 ms | not_measured | managed | 0.410x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 17.9924 ms | not_measured | managed | 0.366x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.0046 ms | not_measured | managed | 0.609x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.9578 ms | not_measured | managed | 0.783x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 98.1281 ms | not_measured | managed | 0.826x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.222x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.0612 ms | not_measured | managed | 0.129x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 17.3502 ms | not_measured | managed | 0.506x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.162x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.1402 ms | not_measured | managed | 0.181x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 38.9178 ms | not_measured | managed | 0.430x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.0050 ms | not_measured | managed | 0.620x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.9569 ms | not_measured | managed | 0.851x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 95.6452 ms | not_measured | managed | 0.840x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.0717 ms | not_measured | managed | 0.100x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 14.7234 ms | not_measured | managed | 0.586x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.167x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.1394 ms | not_measured | managed | 0.184x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 35.2939 ms | not_measured | managed | 0.468x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.0768 ms | not_measured | managed | 0.378x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 1.1880 ms | not_measured | managed | 1.489x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.0386 ms | not_measured | managed | 0.306x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.6911 ms | not_measured | managed | 0.113x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.0326 ms | not_measured | managed | 0.359x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.5677 ms | not_measured | managed | 1.173x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.0173 ms | not_measured | managed | 0.098x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.0177 ms | not_measured | managed | 0.096x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.0161 ms | not_measured | managed | 0.106x |
| `np.iscomplex(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.iscomplexobj(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.0595 ms | not_measured | managed | 0.832x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 6.2993 ms | not_measured | managed | 0.922x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.0031 ms | not_measured | managed | 0.129x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.0634 ms | not_measured | managed | 0.084x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 7.8118 ms | not_measured | managed | 0.449x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.143x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.0638 ms | not_measured | managed | 0.161x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 14.3288 ms | not_measured | managed | 0.354x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.0610 ms | not_measured | managed | 0.836x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 6.4027 ms | not_measured | managed | 0.953x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.0638 ms | not_measured | managed | 0.083x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 8.8333 ms | not_measured | managed | 0.391x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.0721 ms | not_measured | managed | 0.153x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 14.5706 ms | not_measured | managed | 0.351x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 0.609x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.0612 ms | not_measured | managed | 1.119x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 6.4116 ms | not_measured | managed | 1.217x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.0032 ms | not_measured | managed | 0.156x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.0617 ms | not_measured | managed | 0.066x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 7.1799 ms | not_measured | managed | 0.463x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.0657 ms | not_measured | managed | 0.132x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 12.5055 ms | not_measured | managed | 0.411x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 1.150x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.0161 ms | not_measured | managed | 4.783x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 2.0134 ms | not_measured | managed | 5.530x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.929x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.0193 ms | not_measured | managed | 0.451x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 2.0951 ms | not_measured | managed | 3.635x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 1.300x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.0159 ms | not_measured | managed | 1.088x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 1.6778 ms | not_measured | managed | 8.313x |
| `np.isrealobj(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isrealobj(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isscalar(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
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
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.0048 ms | not_measured | managed | 0.625x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.9993 ms | not_measured | managed | 0.762x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 98.7765 ms | not_measured | managed | 0.818x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.222x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.0802 ms | not_measured | managed | 0.092x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 13.4484 ms | not_measured | managed | 0.651x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.0038 ms | not_measured | managed | 0.158x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.1361 ms | not_measured | managed | 0.209x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 32.4967 ms | not_measured | managed | 0.506x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.0051 ms | not_measured | managed | 0.627x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.9707 ms | not_measured | managed | 0.783x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 97.8707 ms | not_measured | managed | 0.835x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.0636 ms | not_measured | managed | 0.119x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 14.2823 ms | not_measured | managed | 0.635x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.176x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.1387 ms | not_measured | managed | 0.185x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 35.7805 ms | not_measured | managed | 0.471x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 0.200x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.200x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `a.flatten` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `a.flatten` | float64 | 100,000 | 0.0582 ms | not_measured | managed | 0.206x |
| `a.flatten` | float64 | 10,000,000 | 12.6884 ms | not_measured | managed | 1.040x |
| `a.ravel() (view)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `a.ravel() (view)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.0085 ms | not_measured | managed | 0.176x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.2382 ms | not_measured | managed | 1.341x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 2.030x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.0029 ms | not_measured | managed | 2.276x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.000x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.0067 ms | not_measured | managed | 0.507x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.2391 ms | not_measured | managed | 1.323x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 2.267x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 3.100x |
| `np.byteswap` | float64 | 1,000 | 0.0072 ms | not_measured | managed | 0.111x |
| `np.byteswap` | float64 | 100,000 | 0.1073 ms | not_measured | managed | 0.320x |
| `np.byteswap` | float64 | 10,000,000 | 11.9288 ms | not_measured | managed | 1.748x |
| `np.c_` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.506x |
| `np.c_` | float64 | 100,000 | 0.2390 ms | not_measured | managed | 1.450x |
| `np.c_` | float64 | 10,000,000 | 25.1148 ms | not_measured | managed | 1.535x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.0136 ms | not_measured | managed | 0.169x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.2252 ms | not_measured | managed | 1.347x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.0106 ms | not_measured | managed | 0.094x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.2552 ms | not_measured | managed | 1.179x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.0097 ms | not_measured | managed | 0.134x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.3844 ms | not_measured | managed | 1.266x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 0.204x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.2404 ms | not_measured | managed | 1.334x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.0046 ms | not_measured | managed | 0.239x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.2562 ms | not_measured | managed | 1.277x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0081 ms | not_measured | managed | 0.160x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.2384 ms | not_measured | managed | 1.258x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.759x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.0671 ms | not_measured | managed | 0.204x |
| `np.diag` | float64 | 1,000 | 0.0082 ms | not_measured | managed | 0.110x |
| `np.diag` | float64 | 100,000 | 0.1340 ms | not_measured | managed | 0.006x |
| `np.diag` | float64 | 10,000,000 | 2.7026 ms | not_measured | managed | 0.000x |
| `np.diagflat` | float64 | 1,000 | 0.0089 ms | not_measured | managed | 0.348x |
| `np.diagflat` | float64 | 100,000 | 0.1378 ms | not_measured | managed | 0.112x |
| `np.diagflat` | float64 | 10,000,000 | 2.6110 ms | not_measured | managed | 1.047x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 1.979x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.0042 ms | not_measured | managed | 2.238x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 0.510x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.2441 ms | not_measured | managed | 1.254x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 2.500x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 2.600x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 2.167x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 2.167x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 2.400x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 2.167x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 3.500x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 2.600x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 2.167x |
| `np.fill_diagonal` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.435x |
| `np.fill_diagonal` | float64 | 100,000 | 0.0027 ms | not_measured | managed | 0.926x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.0128 ms | not_measured | managed | 1.383x |
| `np.flip` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 1.000x |
| `np.flip` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.flip` | float64 | 10,000,000 | 0.0004 ms | not_measured | managed | 1.000x |
| `np.fliplr` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 1.000x |
| `np.fliplr` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.fliplr` | float64 | 10,000,000 | 0.0004 ms | not_measured | managed | 1.000x |
| `np.flipud` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.flipud` | float64 | 100,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.flipud` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 2.325x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.0048 ms | not_measured | managed | 1.958x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.321x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.2491 ms | not_measured | managed | 1.357x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.0081 ms | not_measured | managed | 0.642x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.0636 ms | not_measured | managed | 0.261x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.0430 ms | not_measured | managed | 0.914x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 10.2505 ms | not_measured | managed | 0.754x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.0113 ms | not_measured | managed | 1.531x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 0.5989 ms | not_measured | managed | 0.684x |
| `np.ix_` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.214x |
| `np.ix_` | float64 | 100,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.ix_` | float64 | 10,000,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.matrix_transpose` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.matrix_transpose` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 3.000x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 2.500x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 3.286x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.0133 ms | not_measured | managed | 0.632x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.0668 ms | not_measured | managed | 0.305x |
| `np.permute_dims` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.permute_dims` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.permute_dims` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.r_` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.308x |
| `np.r_` | float64 | 100,000 | 0.2373 ms | not_measured | managed | 1.344x |
| `np.r_` | float64 | 10,000,000 | 25.3716 ms | not_measured | managed | 1.034x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.678x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.1355 ms | not_measured | managed | 2.828x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 26.0924 ms | not_measured | managed | 1.507x |
| `np.r_(0:N)` | float64 | 1,000 | 0.0075 ms | not_measured | managed | 0.320x |
| `np.r_(0:N)` | float64 | 100,000 | 0.1343 ms | not_measured | managed | 2.578x |
| `np.r_(0:N)` | float64 | 10,000,000 | 28.7818 ms | not_measured | managed | 0.909x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.0096 ms | not_measured | managed | 0.438x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.2479 ms | not_measured | managed | 1.291x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 28.3493 ms | not_measured | managed | 0.918x |
| `np.ravel` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.ravel` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.500x |
| `np.ravel` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.500x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.683x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.2536 ms | not_measured | managed | 1.417x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.0155 ms | not_measured | managed | 0.155x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.2305 ms | not_measured | managed | 1.337x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.0133 ms | not_measured | managed | 0.338x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.0636 ms | not_measured | managed | 0.248x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.400x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.rot90` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 2.000x |
| `np.rot90` | float64 | 100,000 | 0.0015 ms | not_measured | managed | 2.133x |
| `np.rot90` | float64 | 10,000,000 | 0.0016 ms | not_measured | managed | 2.062x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.0490 ms | not_measured | managed | 1.027x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 9.6382 ms | not_measured | managed | 0.813x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.0576 ms | not_measured | managed | 0.693x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 9.8072 ms | not_measured | managed | 0.750x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.0046 ms | not_measured | managed | 1.957x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.0051 ms | not_measured | managed | 1.765x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.0003 ms | not_measured | managed | 1.000x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.0004 ms | not_measured | managed | 1.000x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.0004 ms | not_measured | managed | 0.750x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 0.442x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.2397 ms | not_measured | managed | 1.415x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.471x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.2719 ms | not_measured | managed | 1.260x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.000x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.000x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.0164 ms | not_measured | managed | 0.159x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.2637 ms | not_measured | managed | 1.182x |
| `np.transpose(a)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.571x |
| `np.transpose(a)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.tri` | float64 | 1,000 | 0.0096 ms | not_measured | managed | 0.406x |
| `np.tri` | float64 | 100,000 | 0.0688 ms | not_measured | managed | 0.717x |
| `np.tri` | float64 | 10,000,000 | 14.0887 ms | not_measured | managed | 1.326x |
| `np.tril` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.778x |
| `np.tril` | float64 | 100,000 | 0.0654 ms | not_measured | managed | 0.729x |
| `np.tril` | float64 | 10,000,000 | 13.3962 ms | not_measured | managed | 1.646x |
| `np.tril_indices` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 1.812x |
| `np.tril_indices` | float64 | 100,000 | 0.0836 ms | not_measured | managed | 0.850x |
| `np.tril_indices` | float64 | 10,000,000 | 6.7624 ms | not_measured | managed | 3.790x |
| `np.trim_zeros` | float64 | 1,000 | 0.0123 ms | not_measured | managed | 1.057x |
| `np.trim_zeros` | float64 | 100,000 | 0.0200 ms | not_measured | managed | 47.165x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.0506 ms | not_measured | managed | 2673.194x |
| `np.triu` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.641x |
| `np.triu` | float64 | 100,000 | 0.0611 ms | not_measured | managed | 0.784x |
| `np.triu` | float64 | 10,000,000 | 12.6570 ms | not_measured | managed | 1.856x |
| `np.triu_indices` | float64 | 1,000 | 0.0086 ms | not_measured | managed | 1.291x |
| `np.triu_indices` | float64 | 100,000 | 0.0771 ms | not_measured | managed | 0.904x |
| `np.triu_indices` | float64 | 10,000,000 | 6.7442 ms | not_measured | managed | 4.351x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.0494 ms | not_measured | managed | 0.455x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 9.0406 ms | not_measured | managed | 0.523x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.0103 ms | not_measured | managed | 1.602x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.4890 ms | not_measured | managed | 0.685x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.0141 ms | not_measured | managed | 2.872x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 6.2572 ms | not_measured | managed | 1.026x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.0113 ms | not_measured | managed | 0.929x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 4.1726 ms | not_measured | managed | 0.149x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.0204 ms | not_measured | managed | 1.142x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 5.2136 ms | not_measured | managed | 0.478x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.0081 ms | not_measured | managed | 1.889x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 3.5246 ms | not_measured | managed | 0.853x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 0.762x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.0045 ms | not_measured | managed | 0.711x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.0043 ms | not_measured | managed | 2.186x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.0043 ms | not_measured | managed | 2.116x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.412x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.2439 ms | not_measured | managed | 1.269x |
| `reshape 1D->2D` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `reshape 1D->2D` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `reshape 1D->3D` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.250x |
| `reshape 1D->3D` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `reshape 2D->1D` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `reshape 2D->1D` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.0082 ms | not_measured | managed | 4.927x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 1.067x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.0015 ms | not_measured | managed | 26.067x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.625x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.0592 ms | not_measured | managed | 0.264x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.0584 ms | not_measured | managed | 0.298x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0095 ms | not_measured | managed | 0.505x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 1.1217 ms | not_measured | managed | 0.132x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.0235 ms | not_measured | managed | 0.413x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 3.1652 ms | not_measured | managed | 0.442x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.0090 ms | not_measured | managed | 0.567x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.2582 ms | not_measured | managed | 2.275x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.429x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.1511 ms | not_measured | managed | 0.116x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.0954 ms | not_measured | managed | 0.866x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.805x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.2425 ms | not_measured | managed | 0.701x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.676x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.2264 ms | not_measured | managed | 0.739x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 0.500x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.625x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.0143 ms | 0.0103 ms | openblas | 0.864x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.0132 ms | not_measured | managed | 0.697x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 1.111x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.0069 ms | not_measured | managed | 1.464x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 1.111x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.0080 ms | not_measured | managed | 1.163x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.786x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.1538 ms | not_measured | managed | 1.146x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0082 ms | not_measured | managed | 0.354x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 1.4494 ms | not_measured | managed | 0.097x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.0044 ms | not_measured | managed | 17.091x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.486x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.0885 ms | not_measured | managed | 1.240x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.625x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.4618 ms | not_measured | managed | 0.849x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.250x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.200x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.1015 ms | not_measured | managed | 0.137x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.0597 ms | not_measured | managed | 0.256x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.0131 ms | not_measured | managed | 0.519x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 3.1620 ms | not_measured | managed | 0.664x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.167x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0339 ms | not_measured | managed | 0.277x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.0169 ms | not_measured | managed | 0.183x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 2.1625 ms | not_measured | managed | 0.228x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 0.400x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.400x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.786x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.0799 ms | not_measured | managed | 0.673x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.0003 ms | not_measured | managed | 1.000x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.0878 ms | not_measured | managed | 0.146x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.2826 ms | not_measured | managed | 0.685x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 4.2019 ms | not_measured | managed | 0.075x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.0073 ms | not_measured | managed | 1.082x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 3.5529 ms | not_measured | managed | 0.236x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 1.667x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.167x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 0.250x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.0004 ms | not_measured | managed | 0.250x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.0431 ms | not_measured | managed | 0.949x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 4.6955 ms | not_measured | managed | 0.856x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.0983 ms | not_measured | managed | 0.245x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 9.6003 ms | not_measured | managed | 0.238x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.0258 ms | not_measured | managed | 0.802x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 2.5166 ms | not_measured | managed | 0.802x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.0093 ms | not_measured | managed | 1.151x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 0.9641 ms | not_measured | managed | 0.905x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.0558 ms | not_measured | managed | 1.073x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 5.5241 ms | not_measured | managed | 1.027x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.0176 ms | not_measured | managed | 0.511x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 1.4844 ms | not_measured | managed | 0.571x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.0641 ms | not_measured | managed | 0.744x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 5.3942 ms | not_measured | managed | 0.743x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.0303 ms | not_measured | managed | 0.686x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 2.4800 ms | not_measured | managed | 0.811x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.0174 ms | not_measured | managed | 1.052x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 1.6235 ms | not_measured | managed | 1.237x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.0184 ms | not_measured | managed | 0.788x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 1.7164 ms | not_measured | managed | 0.800x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1135 ms | not_measured | managed | 0.565x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 10.7262 ms | not_measured | managed | 0.607x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.0202 ms | not_measured | managed | 0.718x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 1.7580 ms | not_measured | managed | 0.756x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.0158 ms | not_measured | managed | 0.601x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 1.2127 ms | not_measured | managed | 0.710x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.0246 ms | not_measured | managed | 0.748x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 1.7424 ms | not_measured | managed | 1.007x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.0291 ms | not_measured | managed | 0.938x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 2.7279 ms | not_measured | managed | 1.044x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.1437 ms | not_measured | managed | 0.509x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 14.4138 ms | not_measured | managed | 0.515x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.0474 ms | not_measured | managed | 1.380x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 4.4186 ms | not_measured | managed | 1.149x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.0874 ms | not_measured | managed | 0.857x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 8.3986 ms | not_measured | managed | 0.869x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.0411 ms | not_measured | managed | 0.810x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 3.5568 ms | not_measured | managed | 0.935x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.0550 ms | not_measured | managed | 0.973x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 6.1293 ms | not_measured | managed | 0.862x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.0161 ms | not_measured | managed | 0.720x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 1.5253 ms | not_measured | managed | 0.710x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.0190 ms | not_measured | managed | 0.800x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 1.8742 ms | not_measured | managed | 0.786x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.0131 ms | not_measured | managed | 1.069x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 1.3505 ms | not_measured | managed | 1.056x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.0325 ms | not_measured | managed | 1.006x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 3.3760 ms | not_measured | managed | 0.989x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.0186 ms | not_measured | managed | 1.581x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 1.8380 ms | not_measured | managed | 1.897x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.0090 ms | not_measured | managed | 0.489x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 0.7934 ms | not_measured | managed | 0.497x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.0067 ms | not_measured | managed | 1.194x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.6324 ms | not_measured | managed | 0.721x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.0152 ms | not_measured | managed | 0.711x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 1.4755 ms | not_measured | managed | 0.857x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.0086 ms | not_measured | managed | 0.488x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 0.7844 ms | not_measured | managed | 0.500x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.0086 ms | not_measured | managed | 0.477x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 0.7780 ms | not_measured | managed | 0.497x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.0128 ms | not_measured | managed | 0.805x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 1.3646 ms | not_measured | managed | 0.779x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.0108 ms | not_measured | managed | 1.250x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 1.2283 ms | not_measured | managed | 1.127x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.0184 ms | not_measured | managed | 1.174x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 2.1421 ms | not_measured | managed | 1.072x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.0130 ms | not_measured | managed | 0.669x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 1.3616 ms | not_measured | managed | 0.643x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.0225 ms | not_measured | managed | 0.907x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 2.3189 ms | not_measured | managed | 0.879x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.0149 ms | not_measured | managed | 0.718x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 1.4634 ms | not_measured | managed | 0.711x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.0370 ms | not_measured | managed | 0.911x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 3.7461 ms | not_measured | managed | 0.907x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.0108 ms | not_measured | managed | 0.898x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 1.1432 ms | not_measured | managed | 0.762x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.0079 ms | not_measured | managed | 0.671x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 0.7641 ms | not_measured | managed | 0.503x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.0525 ms | not_measured | managed | 0.966x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 6.6863 ms | not_measured | managed | 0.851x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.0325 ms | not_measured | managed | 0.763x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 3.6366 ms | not_measured | managed | 0.784x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.0268 ms | not_measured | managed | 0.802x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 2.4200 ms | not_measured | managed | 1.051x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.0493 ms | not_measured | managed | 0.759x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 4.9276 ms | not_measured | managed | 0.729x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.0019 ms | not_measured | managed | 1.263x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1443 ms | not_measured | managed | 1.011x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 13.9465 ms | not_measured | managed | 1.173x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 2.231x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4545 ms | not_measured | managed | 1.100x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 34.4684 ms | not_measured | managed | 1.458x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.0090 ms | not_measured | managed | 0.578x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 2.2157 ms | not_measured | managed | 0.703x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.0175 ms | not_measured | managed | 0.531x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 3.4988 ms | not_measured | managed | 0.999x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.0006 ms | not_measured | managed | 1.500x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.0016 ms | not_measured | managed | 1.500x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.6875 ms | not_measured | managed | 0.420x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 1.125x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.0024 ms | not_measured | managed | 1.458x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 4.6402 ms | not_measured | managed | 0.312x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.0289 ms | not_measured | managed | 0.287x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 8.9058 ms | not_measured | managed | 0.458x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.2396 ms | not_measured | managed | 0.612x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.0007 ms | not_measured | managed | 1.286x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 1.438x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 0.7903 ms | not_measured | managed | 0.415x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.0051 ms | not_measured | managed | 0.686x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 4.1621 ms | not_measured | managed | 0.263x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.0379 ms | not_measured | managed | 0.303x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 4.3654 ms | not_measured | managed | 0.867x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.2466 ms | not_measured | managed | 0.521x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.043x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1160 ms | not_measured | managed | 1.272x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 14.1740 ms | not_measured | managed | 1.153x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.0012 ms | not_measured | managed | 2.750x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.3048 ms | not_measured | managed | 1.663x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 31.6067 ms | not_measured | managed | 1.636x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.0035 ms | not_measured | managed | 1.486x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 1.5934 ms | not_measured | managed | 0.994x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.0077 ms | not_measured | managed | 1.286x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 3.5970 ms | not_measured | managed | 0.967x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 1.125x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.0017 ms | not_measured | managed | 1.353x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 0.3424 ms | not_measured | managed | 0.907x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.0007 ms | not_measured | managed | 1.429x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.0023 ms | not_measured | managed | 1.522x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 1.2577 ms | not_measured | managed | 0.892x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.0079 ms | not_measured | managed | 1.139x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 4.1948 ms | not_measured | managed | 1.068x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.0008 ms | not_measured | managed | 1.125x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.1479 ms | not_measured | managed | 1.006x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 1.125x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 1.438x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 0.3512 ms | not_measured | managed | 0.869x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.0033 ms | not_measured | managed | 1.061x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 1.6262 ms | not_measured | managed | 0.639x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.0009 ms | not_measured | managed | 1.333x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.0103 ms | not_measured | managed | 1.117x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 4.3699 ms | not_measured | managed | 0.858x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 1.000x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.0013 ms | not_measured | managed | 1.308x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.1496 ms | not_measured | managed | 0.919x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.0017 ms | not_measured | managed | 1.353x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.0226 ms | not_measured | managed | 1.394x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 16.5594 ms | not_measured | managed | 0.575x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 2.286x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.1670 ms | not_measured | managed | 0.637x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 16.3639 ms | not_measured | managed | 0.613x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 2.462x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.0055 ms | not_measured | managed | 3.218x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 3.6319 ms | not_measured | managed | 0.826x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.500x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.0094 ms | not_measured | managed | 1.936x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 7.7117 ms | not_measured | managed | 0.683x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 1.500x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.0448 ms | not_measured | managed | 1.183x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 4.5618 ms | not_measured | managed | 1.097x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 2.077x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.0459 ms | not_measured | managed | 1.000x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 5.4193 ms | not_measured | managed | 0.855x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 1.786x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.0094 ms | not_measured | managed | 3.628x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 7.9498 ms | not_measured | managed | 0.806x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 1.800x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.0456 ms | not_measured | managed | 1.129x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 4.6835 ms | not_measured | managed | 1.057x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.455x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.0415 ms | not_measured | managed | 1.239x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 4.4499 ms | not_measured | managed | 1.133x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 2.000x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.0399 ms | not_measured | managed | 1.008x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 5.2138 ms | not_measured | managed | 0.863x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.0015 ms | not_measured | managed | 1.733x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.0099 ms | not_measured | managed | 5.434x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 7.9590 ms | not_measured | managed | 0.916x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 2.250x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.0412 ms | not_measured | managed | 1.274x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 3.8597 ms | not_measured | managed | 1.277x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 8.429x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.1815 ms | not_measured | managed | 4.852x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 18.5230 ms | not_measured | managed | 4.965x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 9.875x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.0097 ms | not_measured | managed | 4.660x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 3.0762 ms | not_measured | managed | 6.869x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 6.200x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.0192 ms | not_measured | managed | 2.995x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 7.3790 ms | not_measured | managed | 5.268x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.0010 ms | not_measured | managed | 1.100x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.0108 ms | not_measured | managed | 2.963x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 6.8088 ms | not_measured | managed | 1.249x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 1.867x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.0824 ms | not_measured | managed | 2.449x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 8.2324 ms | not_measured | managed | 2.405x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.0009 ms | not_measured | managed | 1.111x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.0033 ms | not_measured | managed | 5.030x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 1.2341 ms | not_measured | managed | 2.610x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.0043 ms | not_measured | managed | 5.349x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 3.1935 ms | not_measured | managed | 1.568x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 1.556x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.0195 ms | not_measured | managed | 1.682x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 2.0058 ms | not_measured | managed | 1.636x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 1.500x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.0196 ms | not_measured | managed | 1.898x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 2.8628 ms | not_measured | managed | 1.459x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.0066 ms | not_measured | managed | 2.773x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 3.2381 ms | not_measured | managed | 1.327x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 1.400x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.0189 ms | not_measured | managed | 1.720x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 1.8734 ms | not_measured | managed | 1.667x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 1.444x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.0191 ms | not_measured | managed | 1.691x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 2.1023 ms | not_measured | managed | 1.551x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 1.444x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.0195 ms | not_measured | managed | 1.636x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 2.8411 ms | not_measured | managed | 1.435x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.0066 ms | not_measured | managed | 2.515x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 3.1011 ms | not_measured | managed | 1.561x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 1.875x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.0190 ms | not_measured | managed | 1.937x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 1.8922 ms | not_measured | managed | 1.641x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.0043 ms | not_measured | managed | 4.372x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.2467 ms | not_measured | managed | 3.662x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 18.4713 ms | not_measured | managed | 4.886x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 9.375x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.0192 ms | not_measured | managed | 2.583x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 3.8988 ms | not_measured | managed | 4.824x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 5.455x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.0366 ms | not_measured | managed | 1.585x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 10.1773 ms | not_measured | managed | 3.883x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.0019 ms | not_measured | managed | 1.579x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1456 ms | not_measured | managed | 1.012x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 14.0663 ms | not_measured | managed | 1.173x |
| `np.amax (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 2.769x |
| `np.amax (float16)` | float16 | 100,000 | 0.4518 ms | not_measured | managed | 1.140x |
| `np.amax (float16)` | float16 | 10,000,000 | 33.0509 ms | not_measured | managed | 1.520x |
| `np.amax (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amax (float32)` | float32 | 100,000 | 0.0091 ms | not_measured | managed | 0.648x |
| `np.amax (float32)` | float32 | 10,000,000 | 1.4771 ms | not_measured | managed | 1.099x |
| `np.amax (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amax (float64)` | float64 | 100,000 | 0.0176 ms | not_measured | managed | 0.568x |
| `np.amax (float64)` | float64 | 10,000,000 | 4.0884 ms | not_measured | managed | 0.886x |
| `np.amax (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amax (int16)` | int16 | 100,000 | 0.0016 ms | not_measured | managed | 2.062x |
| `np.amax (int16)` | int16 | 10,000,000 | 0.7051 ms | not_measured | managed | 0.410x |
| `np.amax (int32)` | int32 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.amax (int32)` | int32 | 100,000 | 0.0023 ms | not_measured | managed | 1.826x |
| `np.amax (int32)` | int32 | 10,000,000 | 4.2928 ms | not_measured | managed | 0.303x |
| `np.amax (int64)` | int64 | 1,000 | 0.0007 ms | not_measured | managed | 2.429x |
| `np.amax (int64)` | int64 | 100,000 | 0.0298 ms | not_measured | managed | 0.443x |
| `np.amax (int64)` | int64 | 10,000,000 | 9.6091 ms | not_measured | managed | 0.396x |
| `np.amax (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `np.amax (int8)` | int8 | 100,000 | 0.0013 ms | not_measured | managed | 1.769x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2397 ms | not_measured | managed | 0.627x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 0.6563 ms | not_measured | managed | 0.492x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.0008 ms | not_measured | managed | 2.125x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.0048 ms | not_measured | managed | 0.875x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 3.6563 ms | not_measured | managed | 0.282x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.0009 ms | not_measured | managed | 2.556x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.0392 ms | not_measured | managed | 0.306x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 4.5504 ms | not_measured | managed | 0.823x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.0012 ms | not_measured | managed | 1.917x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.2409 ms | not_measured | managed | 0.551x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.0031 ms | not_measured | managed | 0.935x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1948 ms | not_measured | managed | 0.778x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 13.9381 ms | not_measured | managed | 1.145x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.0019 ms | not_measured | managed | 2.105x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6672 ms | not_measured | managed | 0.749x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 84.4157 ms | not_measured | managed | 0.582x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 2.500x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.0287 ms | not_measured | managed | 0.334x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 2.1077 ms | not_measured | managed | 0.930x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 2.625x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.0561 ms | not_measured | managed | 0.273x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 5.5064 ms | not_measured | managed | 0.752x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.0041 ms | not_measured | managed | 1.585x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 1.8620 ms | not_measured | managed | 0.283x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.0105 ms | not_measured | managed | 0.838x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 4.0301 ms | not_measured | managed | 0.335x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.0295 ms | not_measured | managed | 0.471x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 9.9613 ms | not_measured | managed | 0.424x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 1.583x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.0040 ms | not_measured | managed | 1.950x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3481 ms | not_measured | managed | 0.576x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.0037 ms | not_measured | managed | 1.703x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 1.5303 ms | not_measured | managed | 0.325x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.0008 ms | not_measured | managed | 2.500x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.0112 ms | not_measured | managed | 0.768x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 4.2247 ms | not_measured | managed | 0.380x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.0370 ms | not_measured | managed | 0.446x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 4.4919 ms | not_measured | managed | 0.951x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.0031 ms | not_measured | managed | 2.581x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.3421 ms | not_measured | managed | 0.594x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.0020 ms | not_measured | managed | 1.500x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1166 ms | not_measured | managed | 1.287x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 14.4006 ms | not_measured | managed | 1.175x |
| `np.amin (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 3.077x |
| `np.amin (float16)` | float16 | 100,000 | 0.3057 ms | not_measured | managed | 1.658x |
| `np.amin (float16)` | float16 | 10,000,000 | 31.4513 ms | not_measured | managed | 1.648x |
| `np.amin (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amin (float32)` | float32 | 100,000 | 0.0035 ms | not_measured | managed | 1.714x |
| `np.amin (float32)` | float32 | 10,000,000 | 1.7711 ms | not_measured | managed | 0.898x |
| `np.amin (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amin (float64)` | float64 | 100,000 | 0.0076 ms | not_measured | managed | 1.329x |
| `np.amin (float64)` | float64 | 10,000,000 | 3.6108 ms | not_measured | managed | 0.994x |
| `np.amin (int16)` | int16 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.amin (int16)` | int16 | 100,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.3612 ms | not_measured | managed | 0.874x |
| `np.amin (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amin (int32)` | int32 | 100,000 | 0.0023 ms | not_measured | managed | 2.043x |
| `np.amin (int32)` | int32 | 10,000,000 | 1.2932 ms | not_measured | managed | 0.872x |
| `np.amin (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 2.000x |
| `np.amin (int64)` | int64 | 100,000 | 0.0080 ms | not_measured | managed | 1.238x |
| `np.amin (int64)` | int64 | 10,000,000 | 3.9387 ms | not_measured | managed | 1.059x |
| `np.amin (int8)` | int8 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.amin (int8)` | int8 | 100,000 | 0.0010 ms | not_measured | managed | 2.800x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.1477 ms | not_measured | managed | 1.008x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 2.312x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 0.3787 ms | not_measured | managed | 0.842x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.0010 ms | not_measured | managed | 1.600x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.0033 ms | not_measured | managed | 1.394x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 1.3259 ms | not_measured | managed | 0.843x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.0010 ms | not_measured | managed | 1.900x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.0102 ms | not_measured | managed | 1.245x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 4.0308 ms | not_measured | managed | 0.935x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.0013 ms | not_measured | managed | 2.154x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.1476 ms | not_measured | managed | 0.900x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.0032 ms | not_measured | managed | 0.906x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.1511 ms | not_measured | managed | 0.997x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 15.9933 ms | not_measured | managed | 1.011x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.0019 ms | not_measured | managed | 2.211x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.5060 ms | not_measured | managed | 0.935x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 78.1015 ms | not_measured | managed | 0.596x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.0100 ms | not_measured | managed | 0.980x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 2.2498 ms | not_measured | managed | 0.845x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.909x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.0174 ms | not_measured | managed | 0.879x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 4.5676 ms | not_measured | managed | 0.897x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.0010 ms | not_measured | managed | 1.900x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.0038 ms | not_measured | managed | 1.737x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 0.5419 ms | not_measured | managed | 0.986x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.0056 ms | not_measured | managed | 1.679x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 1.4870 ms | not_measured | managed | 0.981x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 2.000x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.0133 ms | not_measured | managed | 1.120x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 3.9150 ms | not_measured | managed | 1.265x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.0039 ms | not_measured | managed | 1.923x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.1726 ms | not_measured | managed | 1.186x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.0037 ms | not_measured | managed | 1.757x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 0.4280 ms | not_measured | managed | 1.193x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.0056 ms | not_measured | managed | 1.625x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 2.0717 ms | not_measured | managed | 0.834x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.0153 ms | not_measured | managed | 1.078x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 4.3629 ms | not_measured | managed | 1.059x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.0039 ms | not_measured | managed | 1.821x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.1764 ms | not_measured | managed | 1.151x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 0.826x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1482 ms | not_measured | managed | 0.778x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 19.8474 ms | not_measured | managed | 0.705x |
| `np.argmax (float16)` | float16 | 1,000 | 0.0030 ms | not_measured | managed | 0.967x |
| `np.argmax (float16)` | float16 | 100,000 | 0.2256 ms | not_measured | managed | 1.935x |
| `np.argmax (float16)` | float16 | 10,000,000 | 22.5409 ms | not_measured | managed | 1.958x |
| `np.argmax (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `np.argmax (float32)` | float32 | 100,000 | 0.0583 ms | not_measured | managed | 0.151x |
| `np.argmax (float32)` | float32 | 10,000,000 | 6.5028 ms | not_measured | managed | 0.312x |
| `np.argmax (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (float64)` | float64 | 100,000 | 0.0599 ms | not_measured | managed | 0.270x |
| `np.argmax (float64)` | float64 | 10,000,000 | 9.2697 ms | not_measured | managed | 0.495x |
| `np.argmax (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (int16)` | int16 | 100,000 | 0.0033 ms | not_measured | managed | 1.152x |
| `np.argmax (int16)` | int16 | 10,000,000 | 0.8026 ms | not_measured | managed | 0.530x |
| `np.argmax (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 1.000x |
| `np.argmax (int32)` | int32 | 100,000 | 0.0060 ms | not_measured | managed | 1.050x |
| `np.argmax (int32)` | int32 | 10,000,000 | 3.6697 ms | not_measured | managed | 0.546x |
| `np.argmax (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmax (int64)` | int64 | 100,000 | 0.0549 ms | not_measured | managed | 0.262x |
| `np.argmax (int64)` | int64 | 10,000,000 | 9.1638 ms | not_measured | managed | 0.517x |
| `np.argmax (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (int8)` | int8 | 100,000 | 0.0020 ms | not_measured | managed | 1.200x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.2514 ms | not_measured | managed | 0.623x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.0033 ms | not_measured | managed | 1.515x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 0.6969 ms | not_measured | managed | 0.764x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.0058 ms | not_measured | managed | 1.534x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 4.0151 ms | not_measured | managed | 0.474x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.0627 ms | not_measured | managed | 0.274x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 9.0844 ms | not_measured | managed | 0.493x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 0.846x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.0019 ms | not_measured | managed | 1.579x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.2590 ms | not_measured | managed | 0.881x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.0022 ms | not_measured | managed | 0.864x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1471 ms | not_measured | managed | 0.820x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 14.1414 ms | not_measured | managed | 1.000x |
| `np.argmin (float16)` | float16 | 1,000 | 0.0029 ms | not_measured | managed | 0.966x |
| `np.argmin (float16)` | float16 | 100,000 | 0.2194 ms | not_measured | managed | 1.907x |
| `np.argmin (float16)` | float16 | 10,000,000 | 14.2288 ms | not_measured | managed | 2.952x |
| `np.argmin (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `np.argmin (float32)` | float32 | 100,000 | 0.0583 ms | not_measured | managed | 0.146x |
| `np.argmin (float32)` | float32 | 10,000,000 | 6.6478 ms | not_measured | managed | 0.309x |
| `np.argmin (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmin (float64)` | float64 | 100,000 | 0.0590 ms | not_measured | managed | 0.276x |
| `np.argmin (float64)` | float64 | 10,000,000 | 9.6821 ms | not_measured | managed | 0.466x |
| `np.argmin (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (int16)` | int16 | 100,000 | 0.0036 ms | not_measured | managed | 0.972x |
| `np.argmin (int16)` | int16 | 10,000,000 | 0.7009 ms | not_measured | managed | 0.651x |
| `np.argmin (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `np.argmin (int32)` | int32 | 100,000 | 0.0058 ms | not_measured | managed | 0.983x |
| `np.argmin (int32)` | int32 | 10,000,000 | 3.9569 ms | not_measured | managed | 0.498x |
| `np.argmin (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (int64)` | int64 | 100,000 | 0.0535 ms | not_measured | managed | 0.273x |
| `np.argmin (int64)` | int64 | 10,000,000 | 9.8948 ms | not_measured | managed | 0.498x |
| `np.argmin (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.917x |
| `np.argmin (int8)` | int8 | 100,000 | 0.0024 ms | not_measured | managed | 0.917x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.2529 ms | not_measured | managed | 0.644x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.0032 ms | not_measured | managed | 1.562x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 0.6806 ms | not_measured | managed | 0.957x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.0060 ms | not_measured | managed | 1.467x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 3.9563 ms | not_measured | managed | 0.499x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.0617 ms | not_measured | managed | 0.284x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 9.4625 ms | not_measured | managed | 0.501x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.0023 ms | not_measured | managed | 1.348x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.2543 ms | not_measured | managed | 0.862x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.0099 ms | not_measured | managed | 0.606x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 0.9687 ms | not_measured | managed | 0.424x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 95.2685 ms | not_measured | managed | 0.495x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.0032 ms | not_measured | managed | 1.156x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 0.1161 ms | not_measured | managed | 1.442x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 10.2345 ms | not_measured | managed | 3.868x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.854x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 0.1497 ms | not_measured | managed | 1.124x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 14.0385 ms | not_measured | managed | 1.758x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.0038 ms | not_measured | managed | 0.816x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.2418 ms | not_measured | managed | 1.579x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 27.7060 ms | not_measured | managed | 1.278x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.653x |
| `np.cumsum (float16)` | float16 | 100,000 | 0.9881 ms | not_measured | managed | 0.487x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 94.8567 ms | not_measured | managed | 0.521x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 2.333x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.0774 ms | not_measured | managed | 2.080x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 8.9004 ms | not_measured | managed | 2.748x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 1.429x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.1261 ms | not_measured | managed | 1.317x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 13.3570 ms | not_measured | managed | 2.455x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.0026 ms | not_measured | managed | 0.962x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.1181 ms | not_measured | managed | 2.772x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 11.6795 ms | not_measured | managed | 2.487x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 1.250x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1244 ms | not_measured | managed | 2.633x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 13.1607 ms | not_measured | managed | 2.235x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.842x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1570 ms | not_measured | managed | 0.189x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 14.7734 ms | not_measured | managed | 1.190x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.0031 ms | not_measured | managed | 0.871x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.1306 ms | not_measured | managed | 2.279x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 11.7362 ms | not_measured | managed | 2.468x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.0022 ms | not_measured | managed | 1.182x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.1461 ms | not_measured | managed | 2.170x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 10.9433 ms | not_measured | managed | 2.588x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 1.600x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1248 ms | not_measured | managed | 2.639x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 12.0493 ms | not_measured | managed | 2.420x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.0015 ms | not_measured | managed | 1.133x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1352 ms | not_measured | managed | 0.227x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 13.5145 ms | not_measured | managed | 1.172x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.0026 ms | not_measured | managed | 1.000x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.1335 ms | not_measured | managed | 2.340x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 11.2758 ms | not_measured | managed | 2.489x |
| `np.max (complex128)` | complex128 | 1,000 | 0.0016 ms | not_measured | managed | 2.000x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1424 ms | not_measured | managed | 1.037x |
| `np.max (complex128)` | complex128 | 10,000,000 | 20.2501 ms | not_measured | managed | 0.824x |
| `np.max (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 2.769x |
| `np.max (float16)` | float16 | 100,000 | 0.4440 ms | not_measured | managed | 1.130x |
| `np.max (float16)` | float16 | 10,000,000 | 47.0745 ms | not_measured | managed | 1.072x |
| `np.max (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.max (float32)` | float32 | 100,000 | 0.0092 ms | not_measured | managed | 0.630x |
| `np.max (float32)` | float32 | 10,000,000 | 4.5130 ms | not_measured | managed | 0.369x |
| `np.max (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 2.429x |
| `np.max (float64)` | float64 | 100,000 | 0.0175 ms | not_measured | managed | 0.571x |
| `np.max (float64)` | float64 | 10,000,000 | 8.6328 ms | not_measured | managed | 0.407x |
| `np.max (int16)` | int16 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.max (int16)` | int16 | 100,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.max (int16)` | int16 | 10,000,000 | 0.6097 ms | not_measured | managed | 0.497x |
| `np.max (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.max (int32)` | int32 | 100,000 | 0.0023 ms | not_measured | managed | 1.826x |
| `np.max (int32)` | int32 | 10,000,000 | 3.9465 ms | not_measured | managed | 0.328x |
| `np.max (int64)` | int64 | 1,000 | 0.0008 ms | not_measured | managed | 2.125x |
| `np.max (int64)` | int64 | 100,000 | 0.0180 ms | not_measured | managed | 0.500x |
| `np.max (int64)` | int64 | 10,000,000 | 9.1844 ms | not_measured | managed | 0.518x |
| `np.max (int8)` | int8 | 1,000 | 0.0008 ms | not_measured | managed | 1.875x |
| `np.max (int8)` | int8 | 100,000 | 0.0010 ms | not_measured | managed | 2.300x |
| `np.max (int8)` | int8 | 10,000,000 | 0.2425 ms | not_measured | managed | 0.572x |
| `np.max (uint16)` | uint16 | 1,000 | 0.0006 ms | not_measured | managed | 2.667x |
| `np.max (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.6303 ms | not_measured | managed | 0.501x |
| `np.max (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 1.778x |
| `np.max (uint32)` | uint32 | 100,000 | 0.0033 ms | not_measured | managed | 1.273x |
| `np.max (uint32)` | uint32 | 10,000,000 | 3.7798 ms | not_measured | managed | 0.288x |
| `np.max (uint64)` | uint64 | 1,000 | 0.0008 ms | not_measured | managed | 2.125x |
| `np.max (uint64)` | uint64 | 100,000 | 0.0385 ms | not_measured | managed | 0.322x |
| `np.max (uint64)` | uint64 | 10,000,000 | 8.6250 ms | not_measured | managed | 0.439x |
| `np.max (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.max (uint8)` | uint8 | 100,000 | 0.0013 ms | not_measured | managed | 1.846x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.2427 ms | not_measured | managed | 0.489x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.0018 ms | not_measured | managed | 1.500x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.0227 ms | not_measured | managed | 1.489x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 16.2303 ms | not_measured | managed | 0.567x |
| `np.mean (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 2.182x |
| `np.mean (float16)` | float16 | 100,000 | 0.1665 ms | not_measured | managed | 0.628x |
| `np.mean (float16)` | float16 | 10,000,000 | 16.3238 ms | not_measured | managed | 0.610x |
| `np.mean (float32)` | float32 | 1,000 | 0.0009 ms | not_measured | managed | 4.111x |
| `np.mean (float32)` | float32 | 100,000 | 0.0048 ms | not_measured | managed | 3.604x |
| `np.mean (float32)` | float32 | 10,000,000 | 3.3511 ms | not_measured | managed | 0.900x |
| `np.mean (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.714x |
| `np.mean (float64)` | float64 | 100,000 | 0.0097 ms | not_measured | managed | 1.876x |
| `np.mean (float64)` | float64 | 10,000,000 | 8.0379 ms | not_measured | managed | 0.646x |
| `np.mean (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 2.385x |
| `np.mean (int16)` | int16 | 100,000 | 0.0459 ms | not_measured | managed | 1.176x |
| `np.mean (int16)` | int16 | 10,000,000 | 4.6005 ms | not_measured | managed | 1.093x |
| `np.mean (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 1.824x |
| `np.mean (int32)` | int32 | 100,000 | 0.0456 ms | not_measured | managed | 0.950x |
| `np.mean (int32)` | int32 | 10,000,000 | 5.7313 ms | not_measured | managed | 0.813x |
| `np.mean (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 2.071x |
| `np.mean (int64)` | int64 | 100,000 | 0.0094 ms | not_measured | managed | 3.638x |
| `np.mean (int64)` | int64 | 10,000,000 | 8.0662 ms | not_measured | managed | 0.801x |
| `np.mean (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 2.750x |
| `np.mean (int8)` | int8 | 100,000 | 0.0460 ms | not_measured | managed | 1.133x |
| `np.mean (int8)` | int8 | 10,000,000 | 4.7465 ms | not_measured | managed | 1.059x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.818x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.0397 ms | not_measured | managed | 1.315x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 4.4259 ms | not_measured | managed | 1.153x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.0397 ms | not_measured | managed | 1.015x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 5.3677 ms | not_measured | managed | 0.857x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.0019 ms | not_measured | managed | 1.579x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.0097 ms | not_measured | managed | 5.577x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 7.7581 ms | not_measured | managed | 0.946x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.0021 ms | not_measured | managed | 1.476x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.0412 ms | not_measured | managed | 1.260x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 3.7156 ms | not_measured | managed | 1.335x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.0028 ms | not_measured | managed | 1.214x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.0476 ms | not_measured | managed | 0.380x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 16.9882 ms | not_measured | managed | 0.433x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.0036 ms | not_measured | managed | 1.472x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.2891 ms | not_measured | managed | 0.266x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 27.4741 ms | not_measured | managed | 0.267x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 1.727x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.0102 ms | not_measured | managed | 0.971x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 3.9723 ms | not_measured | managed | 0.425x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 1.500x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.0161 ms | not_measured | managed | 0.938x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 8.4480 ms | not_measured | managed | 0.441x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 4.667x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.0087 ms | not_measured | managed | 6.034x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 2.0976 ms | not_measured | managed | 2.440x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 4.778x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.0080 ms | not_measured | managed | 4.325x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 4.1692 ms | not_measured | managed | 1.105x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 2.714x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.0623 ms | not_measured | managed | 0.485x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 120.9824 ms | not_measured | managed | 0.050x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 4.444x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.0085 ms | not_measured | managed | 5.659x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 1.9052 ms | not_measured | managed | 2.702x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 5.125x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.0086 ms | not_measured | managed | 5.895x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 2.2646 ms | not_measured | managed | 2.257x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 4.444x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.0095 ms | not_measured | managed | 3.800x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 5.1473 ms | not_measured | managed | 0.893x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.0017 ms | not_measured | managed | 2.471x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.0924 ms | not_measured | managed | 0.516x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 123.7309 ms | not_measured | managed | 0.058x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 4.333x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.0085 ms | not_measured | managed | 5.765x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 1.8241 ms | not_measured | managed | 2.812x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.0027 ms | not_measured | managed | 1.185x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.0538 ms | not_measured | managed | 0.623x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 18.5322 ms | not_measured | managed | 0.447x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.0042 ms | not_measured | managed | 1.214x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.2809 ms | not_measured | managed | 0.298x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 26.2373 ms | not_measured | managed | 0.304x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 1.682x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.0135 ms | not_measured | managed | 1.400x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 3.8867 ms | not_measured | managed | 0.830x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 1.250x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.0156 ms | not_measured | managed | 1.333x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 8.8560 ms | not_measured | managed | 0.587x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 4.500x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.0081 ms | not_measured | managed | 7.173x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 2.1316 ms | not_measured | managed | 2.416x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 4.375x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.0049 ms | not_measured | managed | 8.776x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 4.0947 ms | not_measured | managed | 1.128x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 2.500x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.0606 ms | not_measured | managed | 0.653x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 14.9375 ms | not_measured | managed | 0.412x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 4.111x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.0081 ms | not_measured | managed | 7.765x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 1.8158 ms | not_measured | managed | 2.822x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 4.000x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.0081 ms | not_measured | managed | 6.963x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 2.1118 ms | not_measured | managed | 2.442x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 3.889x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.0092 ms | not_measured | managed | 4.957x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 5.0409 ms | not_measured | managed | 0.918x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 2.312x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.0874 ms | not_measured | managed | 0.604x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 16.4151 ms | not_measured | managed | 0.452x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 4.000x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.0081 ms | not_measured | managed | 7.012x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 1.7860 ms | not_measured | managed | 2.844x |
| `np.min (complex128)` | complex128 | 1,000 | 0.0016 ms | not_measured | managed | 1.875x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1170 ms | not_measured | managed | 1.260x |
| `np.min (complex128)` | complex128 | 10,000,000 | 17.6634 ms | not_measured | managed | 0.920x |
| `np.min (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 3.000x |
| `np.min (float16)` | float16 | 100,000 | 0.3275 ms | not_measured | managed | 1.558x |
| `np.min (float16)` | float16 | 10,000,000 | 34.1288 ms | not_measured | managed | 1.515x |
| `np.min (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 1.600x |
| `np.min (float32)` | float32 | 100,000 | 0.0042 ms | not_measured | managed | 1.381x |
| `np.min (float32)` | float32 | 10,000,000 | 1.9138 ms | not_measured | managed | 0.822x |
| `np.min (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 2.125x |
| `np.min (float64)` | float64 | 100,000 | 0.0080 ms | not_measured | managed | 1.288x |
| `np.min (float64)` | float64 | 10,000,000 | 3.6297 ms | not_measured | managed | 0.983x |
| `np.min (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 1.778x |
| `np.min (int16)` | int16 | 100,000 | 0.0016 ms | not_measured | managed | 2.062x |
| `np.min (int16)` | int16 | 10,000,000 | 0.4611 ms | not_measured | managed | 0.665x |
| `np.min (int32)` | int32 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.min (int32)` | int32 | 100,000 | 0.0024 ms | not_measured | managed | 1.750x |
| `np.min (int32)` | int32 | 10,000,000 | 1.9056 ms | not_measured | managed | 0.633x |
| `np.min (int64)` | int64 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.min (int64)` | int64 | 100,000 | 0.0173 ms | not_measured | managed | 0.549x |
| `np.min (int64)` | int64 | 10,000,000 | 4.0159 ms | not_measured | managed | 1.317x |
| `np.min (int8)` | int8 | 1,000 | 0.0007 ms | not_measured | managed | 2.286x |
| `np.min (int8)` | int8 | 100,000 | 0.0010 ms | not_measured | managed | 2.300x |
| `np.min (int8)` | int8 | 10,000,000 | 0.1623 ms | not_measured | managed | 0.911x |
| `np.min (uint16)` | uint16 | 1,000 | 0.0008 ms | not_measured | managed | 2.750x |
| `np.min (uint16)` | uint16 | 100,000 | 0.0016 ms | not_measured | managed | 2.938x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.7057 ms | not_measured | managed | 0.431x |
| `np.min (uint32)` | uint32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.min (uint32)` | uint32 | 100,000 | 0.0033 ms | not_measured | managed | 1.303x |
| `np.min (uint32)` | uint32 | 10,000,000 | 1.7844 ms | not_measured | managed | 0.589x |
| `np.min (uint64)` | uint64 | 1,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `np.min (uint64)` | uint64 | 100,000 | 0.0112 ms | not_measured | managed | 1.080x |
| `np.min (uint64)` | uint64 | 10,000,000 | 4.7782 ms | not_measured | managed | 0.789x |
| `np.min (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.min (uint8)` | uint8 | 100,000 | 0.0010 ms | not_measured | managed | 2.300x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.1577 ms | not_measured | managed | 0.875x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.0033 ms | not_measured | managed | 2.697x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.1700 ms | not_measured | managed | 2.999x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 17.4884 ms | not_measured | managed | 3.347x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 2.792x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.1190 ms | not_measured | managed | 0.238x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 11.3861 ms | not_measured | managed | 2.155x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 1.970x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.1096 ms | not_measured | managed | 0.427x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 13.0850 ms | not_measured | managed | 1.804x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.0033 ms | not_measured | managed | 2.697x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.2175 ms | not_measured | managed | 2.246x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 17.5234 ms | not_measured | managed | 5.059x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.0030 ms | not_measured | managed | 2.067x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.1180 ms | not_measured | managed | 0.243x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 11.0258 ms | not_measured | managed | 2.220x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 1.722x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.1084 ms | not_measured | managed | 0.437x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 13.1649 ms | not_measured | managed | 1.794x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.0012 ms | not_measured | managed | 4.583x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.3207 ms | not_measured | managed | 1.581x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 32.7279 ms | not_measured | managed | 1.534x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 2.231x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.0289 ms | not_measured | managed | 0.263x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 3.5759 ms | not_measured | managed | 0.432x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 2.500x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.1037 ms | not_measured | managed | 0.110x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 7.3201 ms | not_measured | managed | 1.451x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.0018 ms | not_measured | managed | 7.000x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.1070 ms | not_measured | managed | 3.059x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 10.7347 ms | not_measured | managed | 3.387x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 7.071x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.0386 ms | not_measured | managed | 1.972x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 4.6376 ms | not_measured | managed | 4.089x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 6.538x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.0758 ms | not_measured | managed | 1.152x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 6.0739 ms | not_measured | managed | 9.641x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 3.977x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.2997 ms | not_measured | managed | 0.741x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 93.0858 ms | not_measured | managed | 1.295x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 5.609x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 1.0728 ms | not_measured | managed | 0.429x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 84.9262 ms | not_measured | managed | 0.914x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 5.130x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.8777 ms | not_measured | managed | 0.553x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 91.2608 ms | not_measured | managed | 1.004x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.0012 ms | not_measured | managed | 4.500x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.3040 ms | not_measured | managed | 1.634x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 32.7687 ms | not_measured | managed | 1.553x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 2.900x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.0296 ms | not_measured | managed | 0.247x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 3.5204 ms | not_measured | managed | 0.391x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 2.385x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.0762 ms | not_measured | managed | 0.151x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 7.2328 ms | not_measured | managed | 1.558x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 7.568x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.2995 ms | not_measured | managed | 1.397x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 94.7482 ms | not_measured | managed | 1.296x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 11.435x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.9976 ms | not_measured | managed | 0.713x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 87.6764 ms | not_measured | managed | 0.899x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 11.348x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.7197 ms | not_measured | managed | 1.016x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 94.0169 ms | not_measured | managed | 0.688x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 4.143x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.0900 ms | not_measured | managed | 1.808x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 9.3987 ms | not_measured | managed | 2.432x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 4.636x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 0.0172 ms | not_measured | managed | 5.308x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 2.0839 ms | not_measured | managed | 8.691x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 5.556x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 0.0290 ms | not_measured | managed | 3.566x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 4.2175 ms | not_measured | managed | 6.622x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 7.136x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.2987 ms | not_measured | managed | 1.404x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 94.4987 ms | not_measured | managed | 1.299x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 11.087x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.9838 ms | not_measured | managed | 0.730x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 92.9530 ms | not_measured | managed | 0.879x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 11.000x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.7137 ms | not_measured | managed | 1.011x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 96.9676 ms | not_measured | managed | 0.662x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.0028 ms | not_measured | managed | 11.464x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.2168 ms | not_measured | managed | 5.274x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 22.1098 ms | not_measured | managed | 5.416x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 11.000x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.0909 ms | not_measured | managed | 1.793x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 9.5040 ms | not_measured | managed | 3.353x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 11.562x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.0877 ms | not_measured | managed | 2.125x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 11.6153 ms | not_measured | managed | 9.436x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 4.429x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.0921 ms | not_measured | managed | 3.036x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 9.0413 ms | not_measured | managed | 3.435x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.0009 ms | not_measured | managed | 4.111x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.0050 ms | not_measured | managed | 6.300x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 2.4539 ms | not_measured | managed | 5.475x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 5.286x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.0152 ms | not_measured | managed | 2.921x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 3.9053 ms | not_measured | managed | 12.727x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.0028 ms | not_measured | managed | 10.786x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.2171 ms | not_measured | managed | 5.369x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 22.9876 ms | not_measured | managed | 5.220x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 8.682x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.1335 ms | not_measured | managed | 1.204x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 9.6098 ms | not_measured | managed | 3.312x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 10.938x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.0889 ms | not_measured | managed | 2.123x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 13.7866 ms | not_measured | managed | 5.010x |
| `np.prod (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 3.000x |
| `np.prod (float64)` | float64 | 100,000 | 0.1756 ms | not_measured | managed | 13.169x |
| `np.prod (float64)` | float64 | 10,000,000 | 65.5690 ms | not_measured | managed | 3.611x |
| `np.prod (int64)` | int64 | 1,000 | 0.0008 ms | not_measured | managed | 2.500x |
| `np.prod (int64)` | int64 | 100,000 | 0.0147 ms | not_measured | managed | 3.946x |
| `np.prod (int64)` | int64 | 10,000,000 | 4.1030 ms | not_measured | managed | 1.536x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 1.900x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.0102 ms | not_measured | managed | 1.157x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 20.1355 ms | not_measured | managed | 0.979x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 1.900x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.0175 ms | not_measured | managed | 1.823x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 4.6070 ms | not_measured | managed | 1.134x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 1.900x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.0074 ms | not_measured | managed | 7.905x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 3.6647 ms | not_measured | managed | 19.162x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.0180 ms | not_measured | managed | 2.878x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 4.5146 ms | not_measured | managed | 1.380x |
| `np.std (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 8.905x |
| `np.std (float16)` | float16 | 100,000 | 0.1837 ms | not_measured | managed | 4.865x |
| `np.std (float16)` | float16 | 10,000,000 | 18.2044 ms | not_measured | managed | 4.932x |
| `np.std (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 11.000x |
| `np.std (float32)` | float32 | 100,000 | 0.0098 ms | not_measured | managed | 4.704x |
| `np.std (float32)` | float32 | 10,000,000 | 3.3145 ms | not_measured | managed | 6.283x |
| `np.std (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 7.444x |
| `np.std (float64)` | float64 | 100,000 | 0.0196 ms | not_measured | managed | 3.092x |
| `np.std (float64)` | float64 | 10,000,000 | 7.0738 ms | not_measured | managed | 5.348x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.0046 ms | not_measured | managed | 4.304x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.3645 ms | not_measured | managed | 2.937x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 93.3237 ms | not_measured | managed | 1.170x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 5.118x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.0227 ms | not_measured | managed | 1.722x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 5.0479 ms | not_measured | managed | 3.746x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 7.600x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.0262 ms | not_measured | managed | 2.473x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 8.0044 ms | not_measured | managed | 4.699x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.0108 ms | not_measured | managed | 2.861x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 7.3665 ms | not_measured | managed | 1.175x |
| `np.sum (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 2.867x |
| `np.sum (float16)` | float16 | 100,000 | 0.0853 ms | not_measured | managed | 2.411x |
| `np.sum (float16)` | float16 | 10,000,000 | 8.2718 ms | not_measured | managed | 2.393x |
| `np.sum (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 2.625x |
| `np.sum (float32)` | float32 | 100,000 | 0.0033 ms | not_measured | managed | 5.212x |
| `np.sum (float32)` | float32 | 10,000,000 | 1.2714 ms | not_measured | managed | 2.459x |
| `np.sum (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.309x |
| `np.sum (float64)` | float64 | 100,000 | 0.2114 ms | not_measured | managed | 0.086x |
| `np.sum (float64)` | float64 | 10,000,000 | 3.5982 ms | not_measured | managed | 1.413x |
| `np.sum (int16)` | int16 | 1,000 | 0.0008 ms | not_measured | managed | 2.625x |
| `np.sum (int16)` | int16 | 100,000 | 0.0192 ms | not_measured | managed | 1.766x |
| `np.sum (int16)` | int16 | 10,000,000 | 2.0714 ms | not_measured | managed | 1.598x |
| `np.sum (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 2.444x |
| `np.sum (int32)` | int32 | 100,000 | 0.0195 ms | not_measured | managed | 1.785x |
| `np.sum (int32)` | int32 | 10,000,000 | 2.8203 ms | not_measured | managed | 1.421x |
| `np.sum (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 1.889x |
| `np.sum (int64)` | int64 | 100,000 | 0.0065 ms | not_measured | managed | 2.800x |
| `np.sum (int64)` | int64 | 10,000,000 | 3.0656 ms | not_measured | managed | 1.387x |
| `np.sum (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 2.100x |
| `np.sum (int8)` | int8 | 100,000 | 0.0190 ms | not_measured | managed | 1.768x |
| `np.sum (int8)` | int8 | 10,000,000 | 1.8994 ms | not_measured | managed | 1.660x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 2.333x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.0193 ms | not_measured | managed | 1.741x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 2.0862 ms | not_measured | managed | 1.548x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.0010 ms | not_measured | managed | 2.100x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.0196 ms | not_measured | managed | 1.668x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 2.8613 ms | not_measured | managed | 1.387x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.0066 ms | not_measured | managed | 2.924x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 3.2217 ms | not_measured | managed | 1.408x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 2.556x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.0189 ms | not_measured | managed | 1.852x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 1.8715 ms | not_measured | managed | 1.683x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.0024 ms | not_measured | managed | 0.833x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.0263 ms | not_measured | managed | 0.593x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 20.1702 ms | not_measured | managed | 0.365x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.0050 ms | not_measured | managed | 0.980x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1562 ms | not_measured | managed | 1.805x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 28.3685 ms | not_measured | managed | 1.019x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 1.118x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.0096 ms | not_measured | managed | 1.010x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 4.8321 ms | not_measured | managed | 0.344x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 1.118x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.0143 ms | not_measured | managed | 0.909x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 11.6976 ms | not_measured | managed | 0.318x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 2.182x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.0049 ms | not_measured | managed | 9.490x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 1.1118 ms | not_measured | managed | 4.144x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 3.125x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.0108 ms | not_measured | managed | 4.463x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 2.3010 ms | not_measured | managed | 2.334x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 2.000x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.0112 ms | not_measured | managed | 2.438x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 9.4970 ms | not_measured | managed | 0.564x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 1.846x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.0053 ms | not_measured | managed | 8.774x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.3800 ms | not_measured | managed | 11.727x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.182x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.0057 ms | not_measured | managed | 8.158x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 1.3825 ms | not_measured | managed | 3.313x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 2.667x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.0091 ms | not_measured | managed | 5.110x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 5.7330 ms | not_measured | managed | 0.930x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.0127 ms | not_measured | managed | 2.283x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 10.4738 ms | not_measured | managed | 0.510x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 2.167x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.0054 ms | not_measured | managed | 8.981x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.5415 ms | not_measured | managed | 8.165x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.0024 ms | not_measured | managed | 0.833x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.0350 ms | not_measured | managed | 0.980x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 21.1146 ms | not_measured | managed | 0.408x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.0047 ms | not_measured | managed | 0.809x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.1392 ms | not_measured | managed | 1.430x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 27.0763 ms | not_measured | managed | 0.712x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.905x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.0129 ms | not_measured | managed | 1.271x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 4.8726 ms | not_measured | managed | 0.650x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 1.118x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.0142 ms | not_measured | managed | 1.345x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 9.6967 ms | not_measured | managed | 0.567x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 2.556x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.0039 ms | not_measured | managed | 9.359x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 1.5556 ms | not_measured | managed | 2.125x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 2.778x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.0065 ms | not_measured | managed | 5.723x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 4.9789 ms | not_measured | managed | 0.855x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 2.111x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.0079 ms | not_measured | managed | 2.089x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 10.6399 ms | not_measured | managed | 0.450x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 2.556x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.0040 ms | not_measured | managed | 9.300x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.4189 ms | not_measured | managed | 7.538x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 2.300x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.0048 ms | not_measured | managed | 7.854x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 1.1473 ms | not_measured | managed | 2.873x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.0009 ms | not_measured | managed | 2.556x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.0054 ms | not_measured | managed | 6.833x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 5.7427 ms | not_measured | managed | 0.707x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.0008 ms | not_measured | managed | 2.375x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.0086 ms | not_measured | managed | 2.163x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 9.2382 ms | not_measured | managed | 0.508x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 2.667x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.0039 ms | not_measured | managed | 9.590x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.3096 ms | not_measured | managed | 10.042x |
| `np.var (float16)` | float16 | 1,000 | 0.0042 ms | not_measured | managed | 4.476x |
| `np.var (float16)` | float16 | 100,000 | 0.1841 ms | not_measured | managed | 4.809x |
| `np.var (float16)` | float16 | 10,000,000 | 18.5710 ms | not_measured | managed | 4.859x |
| `np.var (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 11.500x |
| `np.var (float32)` | float32 | 100,000 | 0.0189 ms | not_measured | managed | 2.534x |
| `np.var (float32)` | float32 | 10,000,000 | 3.6398 ms | not_measured | managed | 4.622x |
| `np.var (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 7.111x |
| `np.var (float64)` | float64 | 100,000 | 0.0379 ms | not_measured | managed | 1.607x |
| `np.var (float64)` | float64 | 10,000,000 | 8.2599 ms | not_measured | managed | 5.303x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.0110 ms | not_measured | managed | 1.836x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.3760 ms | not_measured | managed | 2.833x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 85.9102 ms | not_measured | managed | 1.273x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 4.474x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.0617 ms | not_measured | managed | 0.635x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 5.9343 ms | not_measured | managed | 3.188x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 7.300x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.0474 ms | not_measured | managed | 1.386x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 9.5502 ms | not_measured | managed | 3.945x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.0091 ms | not_measured | managed | 0.648x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.1882 ms | not_measured | managed | 3.084x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 1.733x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.0833 ms | not_measured | managed | 1.137x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 2.350x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.0019 ms | not_measured | managed | 2.421x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 2.067x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.0866 ms | not_measured | managed | 1.020x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 1.000x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.2774 ms | not_measured | managed | 1.182x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.0094 ms | not_measured | managed | 1.021x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.4534 ms | not_measured | managed | 1.505x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 3.250x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.2729 ms | not_measured | managed | 1.158x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.611x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.1065 ms | not_measured | managed | 1.027x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 1.174x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.1212 ms | not_measured | managed | 0.618x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 3.588x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.1371 ms | not_measured | managed | 5.174x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.938x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.0977 ms | not_measured | managed | 0.620x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.0764 ms | not_measured | managed | 0.380x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 2.224x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.0609 ms | not_measured | managed | 1.218x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 2.264x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.0686 ms | not_measured | managed | 1.022x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 1.067x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1116 ms | not_measured | managed | 0.851x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.0842 ms | not_measured | managed | 0.347x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 5.7192 ms | not_measured | managed | 1.690x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.850x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.1181 ms | not_measured | managed | 0.360x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 18.4614 ms | not_measured | managed | 0.994x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.118x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.0016 ms | not_measured | managed | 0.125x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.0014 ms | not_measured | managed | 0.143x |
| `a[100:1000] * 2` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 0.173x |
| `a[100:1000] * 2` | float64 | 100,000 | 0.0047 ms | not_measured | managed | 0.234x |
| `a[100:1000] * 2` | float64 | 10,000,000 | 0.0058 ms | not_measured | managed | 0.190x |
| `a[100:1000].copy()` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.078x |
| `a[100:1000].copy()` | float64 | 100,000 | 0.0045 ms | not_measured | managed | 0.089x |
| `a[100:1000].copy()` | float64 | 10,000,000 | 0.0039 ms | not_measured | managed | 0.128x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.095x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 0.095x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.0022 ms | not_measured | managed | 0.091x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.105x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.0019 ms | not_measured | managed | 0.105x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.0027 ms | not_measured | managed | 0.074x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.071x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 0.154x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.0016 ms | not_measured | managed | 0.125x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.154x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 0.077x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.0014 ms | not_measured | managed | 0.143x |
| `a[::2] * 2` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.145x |
| `a[::2] * 2` | float64 | 100,000 | 0.0410 ms | not_measured | managed | 0.366x |
| `a[::2] * 2` | float64 | 10,000,000 | 8.2016 ms | not_measured | managed | 1.395x |
| `np.copy(a[100:1000])` | float64 | 1,000 | 0.0043 ms | not_measured | managed | 0.163x |
| `np.copy(a[100:1000])` | float64 | 100,000 | 0.0042 ms | not_measured | managed | 0.167x |
| `np.copy(a[100:1000])` | float64 | 10,000,000 | 0.0037 ms | not_measured | managed | 0.189x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.0090 ms | not_measured | managed | 0.511x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 0.9097 ms | not_measured | managed | 0.155x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.0125 ms | not_measured | managed | 0.336x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.9968 ms | not_measured | managed | 0.147x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.0134 ms | not_measured | managed | 0.254x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 0.8762 ms | not_measured | managed | 0.134x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.0116 ms | not_measured | managed | 0.328x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 0.8482 ms | not_measured | managed | 0.166x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.0273 ms | not_measured | managed | 0.429x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.1759 ms | not_measured | managed | 1.324x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 314.1613 ms | not_measured | managed | 2.179x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.0297 ms | not_measured | managed | 0.367x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 2.1830 ms | not_measured | managed | 0.653x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 553.2263 ms | not_measured | managed | 1.322x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.0256 ms | not_measured | managed | 0.469x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 0.9977 ms | not_measured | managed | 0.413x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 265.7193 ms | not_measured | managed | 0.781x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.0250 ms | not_measured | managed | 0.528x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 1.8210 ms | not_measured | managed | 0.263x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 504.5494 ms | not_measured | managed | 0.651x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 2.095x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.1330 ms | not_measured | managed | 3.595x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 2.444x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.0749 ms | not_measured | managed | 3.880x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.0044 ms | not_measured | managed | 0.750x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.1188 ms | not_measured | managed | 3.295x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.0027 ms | not_measured | managed | 1.222x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.1217 ms | not_measured | managed | 3.248x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 1.737x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.1934 ms | not_measured | managed | 1.053x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 1.571x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.1810 ms | not_measured | managed | 1.048x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.0057 ms | not_measured | managed | 0.421x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.2369 ms | not_measured | managed | 0.431x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.0027 ms | not_measured | managed | 0.815x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.2412 ms | not_measured | managed | 0.459x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.0334 ms | not_measured | managed | 1.392x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 2.8782 ms | not_measured | managed | 4.666x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.0562 ms | not_measured | managed | 0.730x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 4.6279 ms | not_measured | managed | 2.913x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.0548 ms | not_measured | managed | 0.763x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 2.2401 ms | not_measured | managed | 4.583x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.0404 ms | not_measured | managed | 0.985x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 3.8592 ms | not_measured | managed | 2.850x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.0083 ms | not_measured | managed | 0.325x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.0670 ms | not_measured | managed | 2.799x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 37.8406 ms | not_measured | managed | 0.697x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.622x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.0871 ms | not_measured | managed | 2.002x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 35.4112 ms | not_measured | managed | 0.784x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.304x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.0627 ms | not_measured | managed | 1.589x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 26.7773 ms | not_measured | managed | 1.014x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.0117 ms | not_measured | managed | 0.154x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.0893 ms | not_measured | managed | 1.185x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 34.3408 ms | not_measured | managed | 0.833x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.0108 ms | not_measured | managed | 0.231x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.8077 ms | not_measured | managed | 0.122x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.0076 ms | not_measured | managed | 0.421x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.8903 ms | not_measured | managed | 0.157x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.0063 ms | not_measured | managed | 0.365x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.6836 ms | not_measured | managed | 0.038x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.0108 ms | not_measured | managed | 0.213x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.6718 ms | not_measured | managed | 0.136x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.0111 ms | not_measured | managed | 0.721x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 2.7686 ms | not_measured | managed | 0.725x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 356.6051 ms | not_measured | managed | 0.710x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.0110 ms | not_measured | managed | 0.691x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 2.7222 ms | not_measured | managed | 0.760x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 347.4465 ms | not_measured | managed | 0.710x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.0055 ms | not_measured | managed | 3.382x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 2.3337 ms | not_measured | managed | 1.222x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 348.3851 ms | not_measured | managed | 1.142x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.0072 ms | not_measured | managed | 2.611x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 2.3386 ms | not_measured | managed | 1.221x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 352.0518 ms | not_measured | managed | 1.160x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.0127 ms | not_measured | managed | 0.189x |
| `np.sort(a) (float32)` | float32 | 100,000 | 0.6355 ms | not_measured | managed | 0.351x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.0151 ms | not_measured | managed | 0.252x |
| `np.sort(a) (float64)` | float64 | 100,000 | 1.4116 ms | not_measured | managed | 0.355x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.0119 ms | not_measured | managed | 0.202x |
| `np.sort(a) (int32)` | int32 | 100,000 | 0.6780 ms | not_measured | managed | 0.132x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.0270 ms | not_measured | managed | 0.274x |
| `np.sort(a) (int64)` | int64 | 100,000 | 1.1565 ms | not_measured | managed | 0.237x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.0150 ms | not_measured | managed | 0.200x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 1.0617 ms | not_measured | managed | 0.485x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.214x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 1.4510 ms | not_measured | managed | 0.523x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.0325 ms | not_measured | managed | 0.102x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 1.1490 ms | not_measured | managed | 0.328x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.0412 ms | not_measured | managed | 0.165x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 1.5443 ms | not_measured | managed | 0.352x |
| `np.average(a) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.0053 ms | not_measured | managed | 0.283x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.2905 ms | not_measured | managed | 0.339x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.convolve(a, v, mode='same')` | float64 | 100,000 | 0.1367 ms | 2.8086 ms | managed | 8.811x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, v, mode='same')` | float64 | 100,000 | 0.1287 ms | 2.6770 ms | managed | 8.318x |
| `np.count_nonzero(a) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.count_nonzero(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.cov(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.cov(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.cross(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.cross(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.diff(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.diff(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.digitize(a, bins) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.digitize(a, bins) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.ediff1d(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.ediff1d(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.inner(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.kron(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.kron(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.median(a) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.median(a) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.median(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.median(a) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.median(a) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.median(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.median(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.median(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.median(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.percentile(a, 50) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.ptp(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.quantile(a, 0.5) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.trace(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.trace(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.0051 ms | not_measured | managed | 0.627x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.4761 ms | not_measured | managed | 0.582x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 48.9641 ms | not_measured | managed | 0.657x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.0291 ms | not_measured | managed | 0.258x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 5.9319 ms | not_measured | managed | 3.348x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.316x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.0651 ms | not_measured | managed | 0.201x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 13.9056 ms | not_measured | managed | 3.082x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.0103 ms | not_measured | managed | 0.990x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 1.4264 ms | not_measured | managed | 0.623x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 133.3360 ms | not_measured | managed | 0.737x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.0032 ms | not_measured | managed | 0.344x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.0565 ms | not_measured | managed | 0.310x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 11.2398 ms | not_measured | managed | 2.960x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.389x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.1290 ms | not_measured | managed | 2.605x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 31.1959 ms | not_measured | managed | 2.935x |
| `np.abs (float16)` | float16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.abs (float16)` | float16 | 100,000 | 0.0233 ms | not_measured | managed | 1.219x |
| `np.abs (float16)` | float16 | 10,000,000 | 3.1708 ms | not_measured | managed | 1.489x |
| `np.abs (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `np.abs (float32)` | float32 | 100,000 | 0.0399 ms | not_measured | managed | 0.173x |
| `np.abs (float32)` | float32 | 10,000,000 | 5.4299 ms | not_measured | managed | 2.741x |
| `np.abs (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.abs (float64)` | float64 | 100,000 | 0.0765 ms | not_measured | managed | 0.158x |
| `np.abs (float64)` | float64 | 10,000,000 | 14.0634 ms | not_measured | managed | 2.687x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.0220 ms | not_measured | managed | 1.255x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 3.0176 ms | not_measured | managed | 2.271x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.0267 ms | not_measured | managed | 0.258x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 5.2011 ms | not_measured | managed | 3.208x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.0591 ms | not_measured | managed | 0.190x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 14.1981 ms | not_measured | managed | 3.056x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.0110 ms | not_measured | managed | 0.791x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 1.0623 ms | not_measured | managed | 0.787x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 106.3626 ms | not_measured | managed | 1.652x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.0061 ms | not_measured | managed | 0.918x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 0.5542 ms | not_measured | managed | 0.943x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 54.7233 ms | not_measured | managed | 2.241x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.0073 ms | not_measured | managed | 0.918x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 0.6723 ms | not_measured | managed | 0.933x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 70.8437 ms | not_measured | managed | 0.996x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.0120 ms | not_measured | managed | 0.958x |
| `np.angle(a) (float16)` | float16 | 100,000 | 0.8519 ms | not_measured | managed | 1.257x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 95.4108 ms | not_measured | managed | 1.854x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.0114 ms | not_measured | managed | 0.351x |
| `np.angle(a) (float32)` | float32 | 100,000 | 0.7114 ms | not_measured | managed | 0.908x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 84.1726 ms | not_measured | managed | 1.487x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.0085 ms | not_measured | managed | 0.518x |
| `np.angle(a) (float64)` | float64 | 100,000 | 0.6797 ms | not_measured | managed | 0.858x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 78.2705 ms | not_measured | managed | 0.858x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.0079 ms | not_measured | managed | 0.873x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 1.2351 ms | not_measured | managed | 0.879x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 124.2961 ms | not_measured | managed | 1.623x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.0060 ms | not_measured | managed | 0.583x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 0.8056 ms | not_measured | managed | 0.938x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 79.1855 ms | not_measured | managed | 1.848x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.606x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 0.8706 ms | not_measured | managed | 0.911x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 92.3990 ms | not_measured | managed | 1.713x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.0110 ms | not_measured | managed | 0.791x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 1.0638 ms | not_measured | managed | 0.787x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 106.8178 ms | not_measured | managed | 1.631x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.0061 ms | not_measured | managed | 0.934x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 0.5684 ms | not_measured | managed | 0.931x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 54.7337 ms | not_measured | managed | 2.193x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.0073 ms | not_measured | managed | 0.918x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 0.6718 ms | not_measured | managed | 0.944x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 70.9531 ms | not_measured | managed | 0.994x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.0082 ms | not_measured | managed | 0.976x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 1.2283 ms | not_measured | managed | 0.886x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 123.7359 ms | not_measured | managed | 1.307x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.0067 ms | not_measured | managed | 0.731x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 0.8377 ms | not_measured | managed | 0.952x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 83.4208 ms | not_measured | managed | 1.855x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.0067 ms | not_measured | managed | 0.582x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 0.8655 ms | not_measured | managed | 0.934x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 91.8120 ms | not_measured | managed | 1.969x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.0124 ms | not_measured | managed | 0.871x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 1.4248 ms | not_measured | managed | 0.820x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 143.7835 ms | not_measured | managed | 1.627x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.0073 ms | not_measured | managed | 0.959x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 0.8964 ms | not_measured | managed | 1.035x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 88.9640 ms | not_measured | managed | 2.030x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.0088 ms | not_measured | managed | 1.034x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 0.9854 ms | not_measured | managed | 0.956x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 101.9965 ms | not_measured | managed | 1.014x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.0085 ms | not_measured | managed | 0.918x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 1.4223 ms | not_measured | managed | 0.879x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 143.3053 ms | not_measured | managed | 1.532x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.0065 ms | not_measured | managed | 0.631x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 1.0006 ms | not_measured | managed | 0.927x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 98.1614 ms | not_measured | managed | 1.590x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.661x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 0.9990 ms | not_measured | managed | 0.952x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 105.2760 ms | not_measured | managed | 1.863x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.0126 ms | not_measured | managed | 0.817x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 1.8228 ms | not_measured | managed | 0.910x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 183.3922 ms | not_measured | managed | 1.648x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.0174 ms | not_measured | managed | 0.253x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 2.2456 ms | not_measured | managed | 0.597x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 222.1751 ms | not_measured | managed | 1.061x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.855x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 1.4286 ms | not_measured | managed | 0.977x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 146.5170 ms | not_measured | managed | 2.069x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.0105 ms | not_measured | managed | 0.876x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 1.2408 ms | not_measured | managed | 0.838x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 124.4793 ms | not_measured | managed | 1.641x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.0056 ms | not_measured | managed | 0.964x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 0.8511 ms | not_measured | managed | 0.904x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 81.2980 ms | not_measured | managed | 2.105x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.0065 ms | not_measured | managed | 0.923x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 0.8613 ms | not_measured | managed | 0.944x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 91.1225 ms | not_measured | managed | 1.001x |
| `np.around (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 1.222x |
| `np.around (float16)` | float16 | 100,000 | 0.4026 ms | not_measured | managed | 1.061x |
| `np.around (float16)` | float16 | 10,000,000 | 39.4776 ms | not_measured | managed | 1.017x |
| `np.around (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.733x |
| `np.around (float32)` | float32 | 100,000 | 0.0329 ms | not_measured | managed | 0.204x |
| `np.around (float32)` | float32 | 10,000,000 | 5.2993 ms | not_measured | managed | 3.983x |
| `np.around (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.800x |
| `np.around (float64)` | float64 | 100,000 | 0.0896 ms | not_measured | managed | 0.146x |
| `np.around (float64)` | float64 | 10,000,000 | 14.9037 ms | not_measured | managed | 2.525x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.0126 ms | not_measured | managed | 0.849x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 1.4276 ms | not_measured | managed | 0.820x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 143.5365 ms | not_measured | managed | 1.650x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.0073 ms | not_measured | managed | 0.945x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 0.8962 ms | not_measured | managed | 1.008x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 88.8929 ms | not_measured | managed | 2.038x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 1.000x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 0.9813 ms | not_measured | managed | 0.958x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 101.8759 ms | not_measured | managed | 1.072x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.0105 ms | not_measured | managed | 0.876x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 1.2402 ms | not_measured | managed | 0.836x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 124.6002 ms | not_measured | managed | 1.645x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.0065 ms | not_measured | managed | 0.815x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 0.8256 ms | not_measured | managed | 0.929x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 81.2740 ms | not_measured | managed | 2.024x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.938x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 0.8631 ms | not_measured | managed | 0.951x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 91.7413 ms | not_measured | managed | 1.001x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.0245 ms | not_measured | managed | 0.392x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 1.3497 ms | not_measured | managed | 0.850x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 135.8927 ms | not_measured | managed | 0.872x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.0065 ms | not_measured | managed | 0.938x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 0.9005 ms | not_measured | managed | 0.958x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 86.0522 ms | not_measured | managed | 1.053x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 0.932x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 1.0850 ms | not_measured | managed | 0.998x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 109.8133 ms | not_measured | managed | 1.024x |
| `np.ceil (float16)` | float16 | 1,000 | 0.0058 ms | not_measured | managed | 0.828x |
| `np.ceil (float16)` | float16 | 100,000 | 0.3380 ms | not_measured | managed | 1.150x |
| `np.ceil (float16)` | float16 | 10,000,000 | 33.1625 ms | not_measured | managed | 1.230x |
| `np.ceil (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `np.ceil (float32)` | float32 | 100,000 | 0.0353 ms | not_measured | managed | 0.167x |
| `np.ceil (float32)` | float32 | 10,000,000 | 5.3526 ms | not_measured | managed | 3.874x |
| `np.ceil (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.ceil (float64)` | float64 | 100,000 | 0.1106 ms | not_measured | managed | 0.112x |
| `np.ceil (float64)` | float64 | 10,000,000 | 15.7422 ms | not_measured | managed | 2.032x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.0064 ms | not_measured | managed | 1.312x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 0.7297 ms | not_measured | managed | 1.268x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 72.0016 ms | not_measured | managed | 1.448x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.0059 ms | not_measured | managed | 0.356x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.0434 ms | not_measured | managed | 0.194x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 5.3964 ms | not_measured | managed | 3.818x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.463x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.0993 ms | not_measured | managed | 0.161x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 13.9517 ms | not_measured | managed | 1.907x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.0220 ms | not_measured | managed | 0.959x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 3.1266 ms | not_measured | managed | 2.296x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.0322 ms | not_measured | managed | 0.596x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 5.2949 ms | not_measured | managed | 3.973x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.0588 ms | not_measured | managed | 0.367x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 13.1280 ms | not_measured | managed | 1.181x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.0222 ms | not_measured | managed | 0.982x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 3.1348 ms | not_measured | managed | 3.090x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.462x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.0314 ms | not_measured | managed | 0.605x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 5.4299 ms | not_measured | managed | 3.788x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.0580 ms | not_measured | managed | 0.350x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 13.3721 ms | not_measured | managed | 1.141x |
| `np.cos (float16)` | float16 | 1,000 | 0.0174 ms | not_measured | managed | 0.402x |
| `np.cos (float16)` | float16 | 100,000 | 2.0334 ms | not_measured | managed | 0.474x |
| `np.cos (float16)` | float16 | 10,000,000 | 113.8530 ms | not_measured | managed | 0.971x |
| `np.cos (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.571x |
| `np.cos (float32)` | float32 | 100,000 | 0.2851 ms | not_measured | managed | 0.255x |
| `np.cos (float32)` | float32 | 10,000,000 | 28.5550 ms | not_measured | managed | 1.486x |
| `np.cos (float64)` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.754x |
| `np.cos (float64)` | float64 | 100,000 | 1.2828 ms | not_measured | managed | 0.625x |
| `np.cos (float64)` | float64 | 10,000,000 | 100.8689 ms | not_measured | managed | 1.491x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.0117 ms | not_measured | managed | 0.692x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 1.1512 ms | not_measured | managed | 0.786x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 116.3568 ms | not_measured | managed | 1.467x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.0059 ms | not_measured | managed | 0.712x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 0.6401 ms | not_measured | managed | 0.971x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 62.8572 ms | not_measured | managed | 1.993x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.722x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 0.6034 ms | not_measured | managed | 0.941x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 65.2325 ms | not_measured | managed | 2.013x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 0.909x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.3702 ms | not_measured | managed | 1.086x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 36.9885 ms | not_measured | managed | 1.474x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.846x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.0266 ms | not_measured | managed | 2.778x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 5.2740 ms | not_measured | managed | 4.512x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.933x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.0565 ms | not_measured | managed | 1.653x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 14.4804 ms | not_measured | managed | 1.264x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.0043 ms | not_measured | managed | 1.023x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.3705 ms | not_measured | managed | 1.065x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 36.7713 ms | not_measured | managed | 1.537x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.917x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.0263 ms | not_measured | managed | 2.810x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 5.2403 ms | not_measured | managed | 3.693x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.765x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.0577 ms | not_measured | managed | 1.594x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 14.3475 ms | not_measured | managed | 1.267x |
| `np.exp (float16)` | float16 | 1,000 | 0.0064 ms | not_measured | managed | 0.734x |
| `np.exp (float16)` | float16 | 100,000 | 0.5945 ms | not_measured | managed | 0.694x |
| `np.exp (float16)` | float16 | 10,000,000 | 58.2744 ms | not_measured | managed | 0.747x |
| `np.exp (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.542x |
| `np.exp (float32)` | float32 | 100,000 | 0.0736 ms | not_measured | managed | 0.791x |
| `np.exp (float32)` | float32 | 10,000,000 | 5.9886 ms | not_measured | managed | 7.042x |
| `np.exp (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.536x |
| `np.exp (float64)` | float64 | 100,000 | 0.2817 ms | not_measured | managed | 0.896x |
| `np.exp (float64)` | float64 | 10,000,000 | 30.9391 ms | not_measured | managed | 2.325x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.0105 ms | not_measured | managed | 0.467x |
| `np.exp2 (float16)` | float16 | 100,000 | 0.9956 ms | not_measured | managed | 0.417x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 96.1563 ms | not_measured | managed | 0.456x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.0097 ms | not_measured | managed | 0.216x |
| `np.exp2 (float32)` | float32 | 100,000 | 0.9104 ms | not_measured | managed | 0.177x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 87.4687 ms | not_measured | managed | 0.573x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.0095 ms | not_measured | managed | 0.253x |
| `np.exp2 (float64)` | float64 | 100,000 | 0.8473 ms | not_measured | managed | 0.303x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 91.5751 ms | not_measured | managed | 0.724x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.0097 ms | not_measured | managed | 0.567x |
| `np.expm1 (float16)` | float16 | 100,000 | 0.8620 ms | not_measured | managed | 0.589x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 85.7992 ms | not_measured | managed | 0.612x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.0042 ms | not_measured | managed | 0.690x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.1892 ms | not_measured | managed | 1.304x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 17.4985 ms | not_measured | managed | 3.411x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.594x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.2736 ms | not_measured | managed | 1.404x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 32.2897 ms | not_measured | managed | 2.727x |
| `np.floor (float16)` | float16 | 1,000 | 0.0046 ms | not_measured | managed | 1.043x |
| `np.floor (float16)` | float16 | 100,000 | 0.3521 ms | not_measured | managed | 1.087x |
| `np.floor (float16)` | float16 | 10,000,000 | 35.2976 ms | not_measured | managed | 1.124x |
| `np.floor (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.floor (float32)` | float32 | 100,000 | 0.0313 ms | not_measured | managed | 0.208x |
| `np.floor (float32)` | float32 | 10,000,000 | 5.0679 ms | not_measured | managed | 4.089x |
| `np.floor (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `np.floor (float64)` | float64 | 100,000 | 0.0683 ms | not_measured | managed | 0.193x |
| `np.floor (float64)` | float64 | 10,000,000 | 14.8149 ms | not_measured | managed | 2.328x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.0020 ms | not_measured | managed | 1.350x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.0021 ms | not_measured | managed | 2.333x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.0017 ms | not_measured | managed | 5.529x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.log (float16)` | float16 | 1,000 | 0.0064 ms | not_measured | managed | 0.766x |
| `np.log (float16)` | float16 | 100,000 | 0.6657 ms | not_measured | managed | 0.629x |
| `np.log (float16)` | float16 | 10,000,000 | 62.1328 ms | not_measured | managed | 0.739x |
| `np.log (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.667x |
| `np.log (float32)` | float32 | 100,000 | 0.0727 ms | not_measured | managed | 1.224x |
| `np.log (float32)` | float32 | 10,000,000 | 7.0109 ms | not_measured | managed | 7.853x |
| `np.log (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.875x |
| `np.log (float64)` | float64 | 100,000 | 0.2596 ms | not_measured | managed | 0.911x |
| `np.log (float64)` | float64 | 10,000,000 | 35.0972 ms | not_measured | managed | 1.961x |
| `np.log10 (float16)` | float16 | 1,000 | 0.0066 ms | not_measured | managed | 0.727x |
| `np.log10 (float16)` | float16 | 100,000 | 0.6578 ms | not_measured | managed | 0.671x |
| `np.log10 (float16)` | float16 | 10,000,000 | 63.7823 ms | not_measured | managed | 0.769x |
| `np.log10 (float32)` | float32 | 1,000 | 0.0038 ms | not_measured | managed | 0.632x |
| `np.log10 (float32)` | float32 | 100,000 | 0.2111 ms | not_measured | managed | 0.952x |
| `np.log10 (float32)` | float32 | 10,000,000 | 21.2660 ms | not_measured | managed | 2.265x |
| `np.log10 (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.617x |
| `np.log10 (float64)` | float64 | 100,000 | 0.2704 ms | not_measured | managed | 0.920x |
| `np.log10 (float64)` | float64 | 10,000,000 | 31.1273 ms | not_measured | managed | 2.263x |
| `np.log1p (float16)` | float16 | 1,000 | 0.0080 ms | not_measured | managed | 0.825x |
| `np.log1p (float16)` | float16 | 100,000 | 0.8619 ms | not_measured | managed | 0.674x |
| `np.log1p (float16)` | float16 | 10,000,000 | 78.2089 ms | not_measured | managed | 0.760x |
| `np.log1p (float32)` | float32 | 1,000 | 0.0042 ms | not_measured | managed | 0.786x |
| `np.log1p (float32)` | float32 | 100,000 | 0.2449 ms | not_measured | managed | 1.187x |
| `np.log1p (float32)` | float32 | 10,000,000 | 23.0794 ms | not_measured | managed | 3.473x |
| `np.log1p (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.822x |
| `np.log1p (float64)` | float64 | 100,000 | 0.2744 ms | not_measured | managed | 1.225x |
| `np.log1p (float64)` | float64 | 10,000,000 | 42.1826 ms | not_measured | managed | 2.081x |
| `np.log2 (float16)` | float16 | 1,000 | 0.0065 ms | not_measured | managed | 0.769x |
| `np.log2 (float16)` | float16 | 100,000 | 0.6743 ms | not_measured | managed | 0.671x |
| `np.log2 (float16)` | float16 | 10,000,000 | 63.0828 ms | not_measured | managed | 0.766x |
| `np.log2 (float32)` | float32 | 1,000 | 0.0038 ms | not_measured | managed | 0.632x |
| `np.log2 (float32)` | float32 | 100,000 | 0.1987 ms | not_measured | managed | 0.989x |
| `np.log2 (float32)` | float32 | 10,000,000 | 19.7195 ms | not_measured | managed | 2.689x |
| `np.log2 (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.688x |
| `np.log2 (float64)` | float64 | 100,000 | 0.3908 ms | not_measured | managed | 1.089x |
| `np.log2 (float64)` | float64 | 10,000,000 | 44.9690 ms | not_measured | managed | 2.265x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.957x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.0954 ms | not_measured | managed | 2.661x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 14.7267 ms | not_measured | managed | 3.525x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.426x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.1908 ms | not_measured | managed | 1.407x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 31.9956 ms | not_measured | managed | 1.406x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.0237 ms | not_measured | managed | 1.485x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 3.2889 ms | not_measured | managed | 1.418x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.0287 ms | not_measured | managed | 0.226x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 5.3502 ms | not_measured | managed | 1.424x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.0558 ms | not_measured | managed | 0.215x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 17.3354 ms | not_measured | managed | 0.945x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.0133 ms | not_measured | managed | 1.594x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 3.2699 ms | not_measured | managed | 1.280x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.0297 ms | not_measured | managed | 0.731x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 4.1695 ms | not_measured | managed | 1.815x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.0541 ms | not_measured | managed | 0.360x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 11.1425 ms | not_measured | managed | 1.342x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.0040 ms | not_measured | managed | 2.200x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.3590 ms | not_measured | managed | 2.298x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 34.0882 ms | not_measured | managed | 2.650x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.870x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.0297 ms | not_measured | managed | 4.461x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 5.5082 ms | not_measured | managed | 3.963x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.0031 ms | not_measured | managed | 0.581x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.0954 ms | not_measured | managed | 1.266x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 14.2105 ms | not_measured | managed | 6.295x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.0053 ms | not_measured | managed | 1.925x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.5145 ms | not_measured | managed | 2.015x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 48.5500 ms | not_measured | managed | 2.246x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.885x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.0284 ms | not_measured | managed | 5.405x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 5.4999 ms | not_measured | managed | 5.864x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.852x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.0593 ms | not_measured | managed | 2.627x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 14.3203 ms | not_measured | managed | 2.259x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.0183 ms | not_measured | managed | 0.645x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 2.5142 ms | not_measured | managed | 0.588x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 278.2987 ms | not_measured | managed | 0.564x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.0085 ms | not_measured | managed | 0.671x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 0.6672 ms | not_measured | managed | 1.017x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 68.4348 ms | not_measured | managed | 2.261x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.0142 ms | not_measured | managed | 0.690x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 1.2610 ms | not_measured | managed | 0.842x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 111.4562 ms | not_measured | managed | 2.194x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.0042 ms | not_measured | managed | 1.000x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.3719 ms | not_measured | managed | 0.972x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 36.8199 ms | not_measured | managed | 1.542x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.846x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.0265 ms | not_measured | managed | 2.785x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 5.3829 ms | not_measured | managed | 4.560x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.875x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.0567 ms | not_measured | managed | 1.654x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 14.1734 ms | not_measured | managed | 1.291x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 1.023x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.3715 ms | not_measured | managed | 1.013x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 36.8833 ms | not_measured | managed | 1.514x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.786x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.0274 ms | not_measured | managed | 2.704x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 5.7892 ms | not_measured | managed | 4.013x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.867x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.0545 ms | not_measured | managed | 1.699x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 14.4599 ms | not_measured | managed | 1.275x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.0082 ms | not_measured | managed | 0.293x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.4075 ms | not_measured | managed | 0.501x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 40.3766 ms | not_measured | managed | 0.556x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 0.500x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.0275 ms | not_measured | managed | 0.596x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 5.5016 ms | not_measured | managed | 1.324x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.156x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.0610 ms | not_measured | managed | 0.615x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 14.3531 ms | not_measured | managed | 1.041x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 1.045x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.4994 ms | not_measured | managed | 0.958x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 49.6190 ms | not_measured | managed | 1.361x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.417x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.0260 ms | not_measured | managed | 0.219x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 5.2482 ms | not_measured | managed | 2.692x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.500x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.0556 ms | not_measured | managed | 0.203x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 13.7375 ms | not_measured | managed | 1.119x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.0043 ms | not_measured | managed | 1.349x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.5019 ms | not_measured | managed | 1.022x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 49.7460 ms | not_measured | managed | 1.388x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.0278 ms | not_measured | managed | 0.234x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 5.1736 ms | not_measured | managed | 2.694x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.611x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.0533 ms | not_measured | managed | 0.240x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 13.8444 ms | not_measured | managed | 1.113x |
| `np.sign (float16)` | float16 | 1,000 | 0.0052 ms | not_measured | managed | 0.269x |
| `np.sign (float16)` | float16 | 100,000 | 0.6880 ms | not_measured | managed | 0.122x |
| `np.sign (float16)` | float16 | 10,000,000 | 65.1572 ms | not_measured | managed | 0.160x |
| `np.sign (float32)` | float32 | 1,000 | 0.0030 ms | not_measured | managed | 0.367x |
| `np.sign (float32)` | float32 | 100,000 | 0.4072 ms | not_measured | managed | 0.736x |
| `np.sign (float32)` | float32 | 10,000,000 | 38.0703 ms | not_measured | managed | 1.468x |
| `np.sign (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.333x |
| `np.sign (float64)` | float64 | 100,000 | 0.4070 ms | not_measured | managed | 0.726x |
| `np.sign (float64)` | float64 | 10,000,000 | 47.0745 ms | not_measured | managed | 1.459x |
| `np.sin (float16)` | float16 | 1,000 | 0.0174 ms | not_measured | managed | 0.443x |
| `np.sin (float16)` | float16 | 100,000 | 2.0354 ms | not_measured | managed | 0.473x |
| `np.sin (float16)` | float16 | 10,000,000 | 125.7852 ms | not_measured | managed | 0.793x |
| `np.sin (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.600x |
| `np.sin (float32)` | float32 | 100,000 | 0.2824 ms | not_measured | managed | 0.278x |
| `np.sin (float32)` | float32 | 10,000,000 | 28.2802 ms | not_measured | managed | 1.520x |
| `np.sin (float64)` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.848x |
| `np.sin (float64)` | float64 | 100,000 | 1.3318 ms | not_measured | managed | 0.527x |
| `np.sin (float64)` | float64 | 10,000,000 | 124.9465 ms | not_measured | managed | 1.146x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.0111 ms | not_measured | managed | 0.703x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 1.1787 ms | not_measured | managed | 0.801x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 118.8931 ms | not_measured | managed | 1.495x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.0057 ms | not_measured | managed | 0.684x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 0.6588 ms | not_measured | managed | 0.952x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 64.6743 ms | not_measured | managed | 2.093x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.652x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 0.6485 ms | not_measured | managed | 0.945x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 69.2488 ms | not_measured | managed | 2.553x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.0061 ms | not_measured | managed | 0.803x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.3366 ms | not_measured | managed | 1.236x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 38.5199 ms | not_measured | managed | 1.054x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.0304 ms | not_measured | managed | 0.484x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 5.3493 ms | not_measured | managed | 3.694x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.722x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.0671 ms | not_measured | managed | 0.833x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 13.8325 ms | not_measured | managed | 2.923x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.0062 ms | not_measured | managed | 0.387x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.4423 ms | not_measured | managed | 0.463x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 44.2040 ms | not_measured | managed | 0.506x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.0272 ms | not_measured | managed | 0.243x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 5.3751 ms | not_measured | managed | 1.328x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.0577 ms | not_measured | managed | 0.196x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 14.3966 ms | not_measured | managed | 1.071x |
| `np.tan (float16)` | float16 | 1,000 | 0.0176 ms | not_measured | managed | 0.415x |
| `np.tan (float16)` | float16 | 100,000 | 2.0505 ms | not_measured | managed | 0.464x |
| `np.tan (float16)` | float16 | 10,000,000 | 118.1311 ms | not_measured | managed | 0.847x |
| `np.tan (float32)` | float32 | 1,000 | 0.0048 ms | not_measured | managed | 0.854x |
| `np.tan (float32)` | float32 | 100,000 | 1.2388 ms | not_measured | managed | 0.534x |
| `np.tan (float32)` | float32 | 10,000,000 | 119.0199 ms | not_measured | managed | 1.136x |
| `np.tan (float64)` | float64 | 1,000 | 0.0072 ms | not_measured | managed | 0.653x |
| `np.tan (float64)` | float64 | 100,000 | 1.6336 ms | not_measured | managed | 0.639x |
| `np.tan (float64)` | float64 | 10,000,000 | 101.9934 ms | not_measured | managed | 1.849x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.0109 ms | not_measured | managed | 0.908x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 1.0647 ms | not_measured | managed | 0.937x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 107.7083 ms | not_measured | managed | 1.796x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 1.062x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.0723 ms | not_measured | managed | 1.781x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 7.5772 ms | not_measured | managed | 9.469x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 1.778x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.3387 ms | not_measured | managed | 1.731x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 35.7917 ms | not_measured | managed | 1.988x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.0052 ms | not_measured | managed | 0.885x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.3315 ms | not_measured | managed | 1.288x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 34.3889 ms | not_measured | managed | 1.207x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.0279 ms | not_measured | managed | 0.233x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 5.2639 ms | not_measured | managed | 1.394x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.214x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.0594 ms | not_measured | managed | 0.200x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 13.8114 ms | not_measured | managed | 1.047x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-22 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.40× geomean · 72%🕐 of NumPy's time · 83 win / 47 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████▊ ........   1.08×    93%🕐  ( 16 win / 10 lose)
1K         ████████████▉ ......   1.29×    78%🕐  ( 14 win / 12 lose)
100K       █████████████ ......   1.31×    76%🕐  ( 15 win / 11 lose)
1M         ███████████████████▶   2.02×    49%🕐  ( 19 win /  7 lose)
10M        ██████████████▎ ....   1.44×    70%🕐  ( 19 win /  7 lose)
ALL        █████████████▉ .....   1.40×    72%🕐  ( 83 win / 47 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████████████▶   2.24×    45%🕐  ( 39 win /  1 lose)
reductions █████████████████▍     1.74×    57%🕐  ( 29 win / 11 lose)
selection  (no data)
copy/cast  ███████▋ ...........   0.77×   130%🕐  ( 10 win / 15 lose)  ◄ SLOWER
index-math ██████ .............   0.61×   164%🕐  (  1 win /  9 lose)  ◄ SLOWER
dtypes     ██████████▏ ........   1.02×    98%🕐  (  4 win / 11 lose)  ◄ PARITY

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.46×    2.73×    1.92×    3.61×    2.05×
reductions      2.48×    1.87×    1.45×    1.61×    1.49×
selection           -        -        -        -        -
copy/cast       0.42×    0.46×    0.61×    2.13×    1.10×
index-math      0.18×    0.52×    0.86×    0.93×    1.11×
dtypes          0.80×    0.67×    1.73×    1.22×    0.95×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.43×    2.66×    0.99×    2.68×    3.94×     2.09×
  sqrt         1.42×    4.15×    4.63×    5.81×    3.32×     3.50×
  copy         1.50×    3.16×    2.38×    7.39×    3.93×     3.18×
  strided      1.47×    1.77×    1.14×    1.99×    2.49×     1.71×
  bcast        1.47×    2.53×    1.63×    3.19×    1.95×     2.07×
  reversed     1.51×    1.58×    1.18×    2.97×    1.00×     1.53×
  castbuf      1.63×    3.49×    3.10×    4.25×    1.13×     2.43×
  mixbuf       1.31×    3.57×    2.46×    3.14×    1.13×     2.10×
-- reductions
  sum          1.58×    1.57×    2.82×    2.84×    1.65×     2.01×
  sum ax0      1.49×    0.89×    0.87×    0.97×    1.06×     1.04×
  sum ax1      1.39×    0.85×    1.28×    3.15×    1.75×     1.53×
  sum dt=      1.49×    1.32×    0.49×    0.55×    0.64×     0.80×
  amin         1.65×    1.38×    0.65×    0.90×    0.87×     1.03×
  cumsum       1.22×    0.94×    1.03×    1.86×    1.73×     1.31×
  any(F)      12.48×    8.48×    2.21×    1.19×    1.03×     3.10×
  any(hit)    11.80×    8.69×    8.59×    4.58×    7.84×     7.94×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.43×    0.16×    0.30×    2.61×    1.09×     0.57×
  astype       0.27×    0.24×    1.13×    3.01×    1.98×     0.85×
  ravel.T      0.34×    0.72×    0.72×    1.72×    0.96×     0.78×
  in-place     0.78×    0.94×    1.00×    2.08×    1.00×     1.09×
  less->b      0.42×    0.74×    0.34×    1.54×    0.76×     0.66×
-- index-math
  unravel      0.26×    0.42×    0.83×    0.94×    0.97×     0.61×
  ravel_mi     0.13×    0.64×    0.88×    0.92×    1.27×     0.61×
-- dtypes
  complex      0.89×    0.59×    0.87×    0.63×    0.77×     0.74×
  float16      0.83×    0.32×    0.58×    0.27×    0.62×     0.48×
  int8         0.70×    1.60×   10.24×   10.82×    1.78×     2.95×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ██████████████████▋    1.87×    53%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.38×    23%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   4.77×    21%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.51×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.48×    40%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.07×    20%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.66×    38%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.20×    46%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.08×    32%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   3.16×    32%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          █████████▏ .........   0.92×   108%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████████▋ ....   1.47×    68%🕐  (  1 win /  0 lose)
w=64         █████████████▊ .....   1.38×    72%🕐  (  1 win /  0 lose)
w=256        █████████████████▎     1.73×    58%🕐  (  1 win /  0 lose)
w=1024       ███████████████████▶   2.07×    48%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    530.73×   (530.7× faster, faster)
  allocate          1.00×   (1.0× slower, parity)
  overlap_copy      1.76×   (1.8× faster, faster)
  forder_out        1.34×   (1.3× faster, faster)
  zerodim           1.51×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.85×    3.36×    1.55×    2.34×    2.15×   vs chained 6× add
reuse            6.05×    5.09×    1.21×    1.97×    1.30×   vs rebuild each call
par8                 -    0.75×    3.22×    5.69×    5.27×   vs single-thread

biggest NumSharp wins: anyff@1 12.48× · anyeh@1 11.80× · i8@1M 10.82× · i8@100K 10.24× · anyeh@1K 8.69×
most behind:           ravelmi@1 0.13× · flatten@1K 0.16× · astype@1K 0.24× · unravel@1 0.26× · f16@1M 0.27×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.12 ✅ | 1.09 ✅ | 0.58 🟡 | 0.80 🟡 | 0.75 🟡 | 0.77 🟡 | 0.78 🟡 |
| 1M | 1.35 ✅ | 1.37 ✅ | 1.39 ✅ | 1.00 ✅ | 1.37 ✅ | 1.21 ✅ | 1.30 ✅ | 0.84 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.96 🟡 | 1.04 ✅ | 0.96 🟡 | 0.09 🔴 | 1.09 ✅ | 1.51 ✅ | 1.32 ✅ |
| 1M | 1.90 ✅ | 1.52 ✅ | 1.80 ✅ | 0.13 🔴 | 1.38 ✅ | 1.50 ✅ | 1.60 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.83 🟡 | 0.72 🟡 | 0.70 🟡 | 1.30 ✅ |
| 1M | 1.38 ✅ | 0.94 🟡 | 0.93 🟡 | 1.91 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|dec\|bcast\|sum\|ax0 | 0.5777 | 0.0111 | 0.02 🔴 |
| 1M\|dec\|bcast\|sum\|ax0 | 5.3967 | 0.1121 | 0.02 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.5420 | 0.0129 | 0.02 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.5645 | 0.0137 | 0.02 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.5576 | 0.0136 | 0.02 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.5461 | 0.0135 | 0.02 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.5533 | 0.0138 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 5.5946 | 0.1709 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.5519 | 0.1968 | 0.04 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 5.5506 | 0.1982 | 0.04 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 5.4491 | 0.2040 | 0.04 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 5.4765 | 0.2199 | 0.04 🔴 |
| 100K\|dec\|C\|min\|ax0 | 0.3278 | 0.0164 | 0.05 🔴 |
| 100K\|dec\|negcol\|sum\|ax0 | 0.5598 | 0.0310 | 0.06 🔴 |
| 100K\|dec\|F\|max\|ax1 | 0.2259 | 0.0144 | 0.06 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.04 ✅ | 0.99 🟡 | 1.03 ✅ | 0.57 🟡 | 1.09 ✅ | 1.20 ✅ | 1.57 ✅ | 1.10 ✅ |
| 1M | 2.19 ✅ | 2.19 ✅ | 2.17 ✅ | 1.01 ✅ | 1.73 ✅ | 1.85 ✅ | 1.71 ✅ | 2.08 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.86 🟡 | 1.17 ✅ | 1.25 ✅ | 1.51 ✅ | 0.63 🟡 | 0.54 🟡 | 1.17 ✅ | 0.89 🟡 | 1.50 ✅ | 1.87 ✅ | 0.82 🟡 | 0.58 🟡 | 1.84 ✅ |
| 1M | 2.39 ✅ | 1.94 ✅ | 1.26 ✅ | 1.24 ✅ | 1.82 ✅ | 1.56 ✅ | 1.71 ✅ | 1.73 ✅ | 1.50 ✅ | 2.06 ✅ | 2.76 ✅ | 1.54 ✅ | 2.95 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|pos | 0.0683 | 0.0137 | 0.20 🟠 |
| 100K\|f64\|strided\|pos | 0.0695 | 0.0201 | 0.29 🟠 |
| 100K\|u32\|strided\|pos | 0.0451 | 0.0131 | 0.29 🟠 |
| 100K\|f64\|T\|pos | 0.0867 | 0.0321 | 0.37 🟠 |
| 100K\|f64\|F\|pos | 0.0914 | 0.0351 | 0.38 🟠 |
| 100K\|f64\|C\|pos | 0.0755 | 0.0309 | 0.41 🟠 |
| 100K\|u32\|C\|pos | 0.0564 | 0.0252 | 0.45 🟠 |
| 100K\|i32\|sliced\|pos | 0.0726 | 0.0325 | 0.45 🟠 |
| 100K\|u64\|strided\|pos | 0.0398 | 0.0181 | 0.46 🟠 |
| 100K\|i32\|strided\|pos | 0.0480 | 0.0219 | 0.46 🟠 |
| 100K\|f32\|strided\|pos | 0.0280 | 0.0147 | 0.53 🟡 |
| 100K\|u32\|T\|pos | 0.0454 | 0.0243 | 0.54 🟡 |
| 100K\|i32\|negcol\|pos | 0.1055 | 0.0575 | 0.54 🟡 |
| 100K\|i32\|bcast\|pos | 0.0533 | 0.0304 | 0.57 🟡 |
| 100K\|u32\|F\|pos | 0.0576 | 0.0339 | 0.59 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.60 🟡 | 0.58 🟡 | 0.63 🟡 | 0.38 🟠 | 0.69 🟡 | 0.66 🟡 | 0.82 🟡 | 0.54 🟡 |
| 1M | 1.53 ✅ | 1.44 ✅ | 1.65 ✅ | 1.31 ✅ | 1.77 ✅ | 1.67 ✅ | 1.81 ✅ | 2.00 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.45 🟠 | 0.17 🔴 | 0.55 🟡 | 1.06 ✅ | 0.94 🟡 | 1.10 ✅ |
| 1M | 1.45 ✅ | 1.58 ✅ | 1.88 ✅ | 0.98 🟡 | 1.75 ✅ | 2.55 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.71 🟡 | 0.72 🟡 | 0.53 🟡 | 0.52 🟡 | 0.82 🟡 | 0.59 🟡 | 0.41 🟠 |
| 1M | 1.85 ✅ | 1.88 ✅ | 2.01 ✅ | 1.84 ✅ | 1.82 ✅ | 0.75 🟡 | 1.77 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|abs | 0.0803 | 0.0067 | 0.08 🔴 |
| 100K\|f32\|sliced\|copy | 0.0773 | 0.0069 | 0.09 🔴 |
| 100K\|f32\|C\|copy | 0.0738 | 0.0066 | 0.09 🔴 |
| 100K\|f32\|strided\|neg | 0.0794 | 0.0074 | 0.09 🔴 |
| 100K\|f32\|negrow\|copy | 0.0701 | 0.0069 | 0.10 🔴 |
| 100K\|f32\|bcast\|copy | 0.0552 | 0.0057 | 0.10 🔴 |
| 100K\|f32\|T\|add | 0.0785 | 0.0081 | 0.10 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0849 | 0.0090 | 0.11 🔴 |
| 100K\|f32\|C\|add | 0.0690 | 0.0075 | 0.11 🔴 |
| 100K\|c128\|strided\|mul | 0.3327 | 0.0364 | 0.11 🔴 |
| 100K\|f32\|C\|less | 0.0402 | 0.0045 | 0.11 🔴 |
| 100K\|f32\|C\|neg | 0.0641 | 0.0072 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1398 | 0.0158 | 0.11 🔴 |
| 100K\|f32\|C\|mul | 0.0703 | 0.0084 | 0.12 🔴 |
| 100K\|f64\|strided\|neg | 0.0726 | 0.0087 | 0.12 🔴 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 1.88 ✅ | 2.13 ✅ | 0.58 🟡 | 2.46 ✅ | 2.15 ✅ | 2.00 ✅ | 1.71 ✅ |
| 1-D strided a[::2] | 1.20 ✅ | 1.14 ✅ | 0.59 🟡 | 1.13 ✅ | 1.60 ✅ | 1.17 ✅ | 1.09 ✅ |
| 1-D reversed a[::-1] | 1.67 ✅ | 1.32 ✅ | 0.56 🟡 | 1.84 ✅ | 1.65 ✅ | 1.71 ✅ | 1.36 ✅ |
| array + scalar | 3.14 ✅ | 2.90 ✅ | 0.75 🟡 | 3.03 ✅ | 2.40 ✅ | 1.58 ✅ | 2.07 ✅ |
| scalar + array | 2.64 ✅ | 3.28 ✅ | 0.61 🟡 | 2.83 ✅ | 2.26 ✅ | 1.58 ✅ | 1.94 ✅ |
| mixed C + F | 1.67 ✅ | 0.93 🟡 | 0.53 🟡 | 1.65 ✅ | 1.42 ✅ | 1.52 ✅ | 1.20 ✅ |
| mixed C + T | 2.24 ✅ | 1.38 ✅ | 0.52 🟡 | 2.43 ✅ | 1.58 ✅ | 1.40 ✅ | 1.43 ✅ |
| binary broadcast +row(1,C) | 1.78 ✅ | 2.62 ✅ | 0.43 🟠 | 3.00 ✅ | 2.05 ✅ | 1.31 ✅ | 1.59 ✅ |
| binary broadcast +col(R,1) | 1.84 ✅ | 2.92 ✅ | 0.67 🟡 | 2.97 ✅ | 1.71 ✅ | 1.83 ✅ | 1.79 ✅ |
| col-broadcast unary (inner stride-0) | 2.52 ✅ | 1.61 ✅ | 0.92 🟡 | 1.91 ✅ | 2.67 ✅ | 4.69 ✅ | 2.12 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| bcast_row|f16 | 9.6428 | 4.1885 | 0.43 🟠 |
| mix_C_T|f16 | 10.9229 | 5.6328 | 0.52 🟡 |
| mix_C_F|f16 | 10.4320 | 5.5790 | 0.53 🟡 |
| 1d_rev|f16 | 10.5810 | 5.9071 | 0.56 🟡 |
| 1d_C|f16 | 9.5644 | 5.5501 | 0.58 🟡 |
| 1d_strided|f16 | 5.1822 | 3.0524 | 0.59 🟡 |
| scalar_lhs|f16 | 9.5668 | 5.8104 | 0.61 🟡 |
| bcast_col|f16 | 10.0513 | 6.6901 | 0.67 🟡 |
| scalar_rhs|f16 | 9.5391 | 7.1765 | 0.75 🟡 |
| colbcast_unary|f16 | 0.9429 | 0.8696 | 0.92 🟡 |
| mix_C_F|f32 | 3.2395 | 3.0024 | 0.93 🟡 |
| 1d_strided|i32 | 0.9778 | 1.1070 | 1.13 ✅ |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **1128 / 1568** comparable cells lag (<1.0); **440** win (≥1.0).
- **🔴 <0.2** — 138 cells. Top: 50× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 39× int → sub-word (narrow); 17× * → bool
- **🟠 0.2–0.5** — 496 cells. Top: 107× int → sub-word (narrow); 94× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 23× same-type diagonal (copy)
- **🟡 0.5–1.0** — 494 cells. Top: 130× int → sub-word (narrow); 55× same-type diagonal (copy); 24× * → bool

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **0.29**, `f64`→narrow **0.26**, `f16`→narrow **0.89**, `c128`→narrow **0.22**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 0.76 🟡 | 0.81 🟡 | 0.75 🟡 | 0.67 🟡 | 0.67 🟡 | 0.60 🟡 | 0.47 🟠 | 0.28 🟠 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.79 🟡 | 1.04 ✅ | 0.99 🟡 | 1.08 ✅ | 1.02 ✅ | 0.74 🟡 | 0.69 🟡 | 0.38 🟠 | 0.40 🟠 | 0.46 🟠 | 0.67 🟡 | 0.34 🟠 | 0.34 🟠 | nan ? | 0.29 🟠 |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.46 🟠 | 0.42 🟠 | 0.44 🟠 | 0.51 🟡 | 0.49 🟠 | 0.65 🟡 | 0.61 🟡 | 0.75 🟡 | 0.71 🟡 | 0.49 🟠 | 0.80 🟡 | 0.58 🟡 | 0.74 🟡 | nan ? | 0.95 🟡 |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.31🟠 | 0.46🟠 | 0.56🟡 | 0.90🟡 | 1.03✅ | 1.97✅ | 1.69✅ | 2.39✅ | 2.33✅ | 1.03✅ | 0.33🟠 | 2.75✅ | 3.50✅ | — | 4.62✅ |
| **u8** | 1.52✅ | 0.16🔴 | 1.95✅ | 1.24✅ | 1.38✅ | 1.55✅ | 1.53✅ | 1.46✅ | 1.87✅ | 0.88🟡 | 1.93✅ | 0.89🟡 | 1.71✅ | — | 1.54✅ |
| **i8** | 0.75🟡 | 1.48✅ | 0.18🔴 | 1.45✅ | 1.12✅ | 2.32✅ | 2.00✅ | 2.03✅ | 2.04✅ | 1.14✅ | 2.70✅ | 1.33✅ | 1.75✅ | — | 1.54✅ |
| **i16** | 2.54✅ | 1.95✅ | 1.72✅ | 0.70🟡 | 1.28✅ | 1.47✅ | 1.70✅ | 1.07✅ | 1.41✅ | 0.95🟡 | 2.41✅ | 1.24✅ | 1.28✅ | — | 1.42✅ |
| **u16** | 1.75✅ | 1.72✅ | 1.77✅ | 1.70✅ | 1.01✅ | 2.10✅ | 1.98✅ | 1.68✅ | 1.87✅ | 0.74🟡 | 2.46✅ | 1.08✅ | 1.69✅ | — | 1.69✅ |
| **i32** | 0.69🟡 | 0.50🟠 | 0.54🟡 | 0.81🟡 | 0.91🟡 | 1.47✅ | 1.88✅ | 1.38✅ | 2.01✅ | 0.83🟡 | 2.14✅ | 0.96🟡 | 1.09✅ | — | 1.34✅ |
| **u32** | 0.57🟡 | 0.48🟠 | 0.51🟡 | 0.61🟡 | 0.54🟡 | 1.14✅ | 0.76🟡 | 1.10✅ | 1.06✅ | 0.44🟠 | 1.72✅ | 0.95🟡 | 1.58✅ | — | 2.45✅ |
| **i64** | 0.24🟠 | 0.15🔴 | 0.22🟠 | 0.28🟠 | 0.32🟠 | 0.52🟡 | 0.46🟠 | 0.60🟡 | 0.70🟡 | 0.25🟠 | 0.80🟡 | 0.50🟠 | 0.71🟡 | — | 0.90🟡 |
| **u64** | 0.33🟠 | 0.36🟠 | 0.27🟠 | 0.53🟡 | 0.38🟠 | 0.84🟡 | 0.97🟡 | 1.33✅ | 1.06✅ | 0.29🟠 | 0.97🟡 | 0.49🟠 | 0.72🟡 | — | 1.43✅ |
| **char** | 0.55🟡 | 0.68🟡 | 0.71🟡 | 0.63🟡 | 0.44🟠 | 0.43🟠 | 0.41🟠 | 0.47🟠 | 0.54🟡 | 0.21🟠 | 1.09✅ | 0.45🟠 | 0.51🟡 | — | 0.56🟡 |
| **f16** | 1.35✅ | 1.51✅ | 1.65✅ | 1.46✅ | 1.47✅ | 1.07✅ | 0.64🟡 | 0.56🟡 | 0.37🟠 | 1.00🟡 | 0.47🟠 | 0.23🟠 | 0.43🟠 | — | 0.68🟡 |
| **f32** | 0.73🟡 | 0.46🟠 | 0.45🟠 | 0.45🟠 | 0.44🟠 | 0.51🟡 | 0.32🟠 | 0.31🟠 | 0.36🟠 | 0.39🟠 | 0.72🟡 | 0.30🟠 | 0.52🟡 | — | 0.79🟡 |
| **f64** | 0.23🟠 | 0.32🟠 | 0.33🟠 | 0.35🟠 | 0.29🟠 | 0.36🟠 | 0.47🟠 | 0.44🟠 | 0.37🟠 | 0.24🟠 | 0.42🟠 | 0.37🟠 | 0.46🟠 | — | 0.45🟠 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.24🟠 | 0.10🔴 | 0.21🟠 | 0.17🔴 | 0.38🟠 | 0.49🟠 | 0.45🟠 | 0.34🟠 | 0.35🟠 | 0.38🟠 | 0.46🟠 | 0.24🟠 | 0.35🟠 | — | 0.61🟡 |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.91✅ | 0.96🟡 | 1.00✅ | 1.23✅ | 1.05✅ | 2.06✅ | 1.88✅ | 2.77✅ | 3.23✅ | 1.14✅ | 0.30🟠 | 2.59✅ | 3.30✅ | — | 4.07✅ |
| **u8** | 2.13✅ | 2.32✅ | 3.63✅ | 1.16✅ | 1.31✅ | 1.54✅ | 1.91✅ | 1.55✅ | 1.62✅ | 1.11✅ | 2.11✅ | 1.02✅ | 1.28✅ | — | 1.88✅ |
| **i8** | 3.06✅ | 3.04✅ | 2.22✅ | 1.30✅ | 1.15✅ | 1.72✅ | 1.47✅ | 2.01✅ | 2.08✅ | 1.28✅ | 2.48✅ | 1.60✅ | 2.06✅ | — | 1.87✅ |
| **i16** | 0.75🟡 | 1.05✅ | 1.29✅ | 0.64🟡 | 1.23✅ | 1.26✅ | 1.21✅ | 1.11✅ | 1.21✅ | 1.28✅ | 2.39✅ | 1.18✅ | 1.15✅ | — | 1.39✅ |
| **u16** | 1.34✅ | 1.31✅ | 1.46✅ | 1.01✅ | 0.62🟡 | 1.16✅ | 1.50✅ | 1.48✅ | 1.37✅ | 0.79🟡 | 2.14✅ | 1.55✅ | 1.03✅ | — | 1.52✅ |
| **i32** | 0.75🟡 | 0.50🟡 | 0.50🟠 | 0.64🟡 | 0.77🟡 | 0.96🟡 | 1.59✅ | 1.56✅ | 1.59✅ | 0.89🟡 | 2.19✅ | 0.88🟡 | 1.21✅ | — | 1.83✅ |
| **u32** | 0.92🟡 | 0.49🟠 | 0.44🟠 | 0.73🟡 | 0.94🟡 | 2.38✅ | 1.58✅ | 1.90✅ | 1.91✅ | 0.97🟡 | 2.50✅ | 1.69✅ | 1.72✅ | — | 1.84✅ |
| **i64** | 0.15🔴 | 0.16🔴 | 0.13🔴 | 0.29🟠 | 0.27🟠 | 0.56🟡 | 0.43🟠 | 0.63🟡 | 0.72🟡 | 0.37🟠 | 0.96🟡 | 0.49🟠 | 0.72🟡 | — | 0.95🟡 |
| **u64** | 0.30🟠 | 0.20🔴 | 0.18🔴 | 0.36🟠 | 0.27🟠 | 0.61🟡 | 0.58🟡 | 0.95🟡 | 0.77🟡 | 0.58🟡 | 0.99🟡 | 0.77🟡 | 1.30✅ | — | 1.74✅ |
| **char** | 0.67🟡 | 0.52🟡 | 0.62🟡 | 0.93🟡 | 0.71🟡 | 0.55🟡 | 0.48🟠 | 0.55🟡 | 0.58🟡 | 0.62🟡 | 1.01✅ | 0.58🟡 | 0.57🟡 | — | 0.50🟠 |
| **f16** | 1.22✅ | 1.63✅ | 1.40✅ | 1.37✅ | 1.32✅ | 0.95🟡 | 0.67🟡 | 0.73🟡 | 0.41🟠 | 1.44✅ | 0.72🟡 | 0.93🟡 | 0.51🟡 | — | 0.78🟡 |
| **f32** | 0.86🟡 | 0.44🟠 | 0.48🟠 | 0.61🟡 | 0.42🟠 | 0.53🟡 | 0.35🟠 | 0.51🟡 | 0.50🟠 | 0.50🟠 | 0.81🟡 | 0.65🟡 | 0.77🟡 | — | 0.84🟡 |
| **f64** | 0.18🔴 | 0.26🟠 | 0.18🔴 | 0.23🟠 | 0.23🟠 | 0.27🟠 | 0.22🟠 | 0.32🟠 | 0.27🟠 | 0.23🟠 | 0.43🟠 | 0.29🟠 | 0.42🟠 | — | 0.42🟠 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.28🟠 | 0.23🟠 | 0.08🔴 | 0.29🟠 | 0.19🔴 | 0.30🟠 | 0.33🟠 | 0.34🟠 | 0.31🟠 | 0.32🟠 | 0.48🟠 | 0.27🟠 | 0.38🟠 | — | 0.71🟡 |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.21🟠 | 0.51🟡 | 0.49🟠 | 0.99🟡 | 0.80🟡 | 1.59✅ | 1.89✅ | 2.18✅ | 2.61✅ | 0.64🟡 | 0.28🟠 | 1.70✅ | 3.20✅ | — | 3.37✅ |
| **u8** | 1.50✅ | 0.13🔴 | 2.16✅ | 1.01✅ | 1.00✅ | 1.63✅ | 1.49✅ | 1.42✅ | 1.46✅ | 1.03✅ | 2.56✅ | 0.99🟡 | 1.33✅ | — | 1.80✅ |
| **i8** | 2.03✅ | 1.82✅ | 0.17🔴 | 1.28✅ | 1.09✅ | 1.38✅ | 1.35✅ | 1.27✅ | 1.30✅ | 0.89🟡 | 2.39✅ | 0.91🟡 | 1.22✅ | — | 1.27✅ |
| **i16** | 1.12✅ | 1.63✅ | 2.40✅ | 0.72🟡 | 1.39✅ | 1.18✅ | 1.42✅ | 0.66🟡 | 0.88🟡 | 1.28✅ | 2.53✅ | 1.53✅ | 1.20✅ | — | 1.22✅ |
| **u16** | 1.00🟡 | 1.57✅ | 1.30✅ | 1.55✅ | 1.07✅ | 1.55✅ | 1.56✅ | 1.41✅ | 1.37✅ | 0.99🟡 | 1.67✅ | 1.23✅ | 1.02✅ | — | 1.37✅ |
| **i32** | 0.99🟡 | 0.56🟡 | 0.54🟡 | 0.68🟡 | 0.78🟡 | 1.37✅ | 1.30✅ | 1.30✅ | 1.51✅ | 1.02✅ | 2.50✅ | 0.89🟡 | 1.34✅ | — | 1.51✅ |
| **u32** | 0.93🟡 | 0.66🟡 | 0.79🟡 | 1.10✅ | 0.75🟡 | 1.21✅ | 0.96🟡 | 0.92🟡 | 0.78🟡 | 0.49🟠 | 1.61✅ | 0.77🟡 | 0.91🟡 | — | 1.43✅ |
| **i64** | 0.27🟠 | 0.17🔴 | 0.19🔴 | 0.34🟠 | 0.24🟠 | 0.42🟠 | 0.47🟠 | 0.86🟡 | 0.73🟡 | 0.31🟠 | 0.83🟡 | 0.41🟠 | 0.58🟡 | — | 0.94🟡 |
| **u64** | 0.33🟠 | 0.20🟠 | 0.21🟠 | 0.27🟠 | 0.31🟠 | 0.57🟡 | 0.52🟡 | 0.89🟡 | 0.59🟡 | 0.27🟠 | 0.85🟡 | 0.46🟠 | 0.86🟡 | — | 1.48✅ |
| **char** | 0.37🟠 | 0.71🟡 | 0.60🟡 | 0.95🟡 | 0.68🟡 | 0.25🟠 | 0.26🟠 | 0.55🟡 | 0.31🟠 | 0.75🟡 | 1.01✅ | 0.59🟡 | 0.56🟡 | — | 0.49🟠 |
| **f16** | 1.98✅ | 1.72✅ | 1.87✅ | 1.53✅ | 1.64✅ | 1.10✅ | 0.71🟡 | 0.87🟡 | 0.43🟠 | 1.90✅ | 0.89🟡 | 1.05✅ | 0.54🟡 | — | 0.83🟡 |
| **f32** | 0.94🟡 | 0.48🟠 | 0.47🟠 | 0.47🟠 | 0.44🟠 | 0.49🟠 | 0.30🟠 | 0.46🟠 | 0.40🟠 | 0.41🟠 | 0.74🟡 | 0.71🟡 | 0.57🟡 | — | 0.59🟡 |
| **f64** | 0.19🔴 | 0.28🟠 | 0.46🟠 | 0.41🟠 | 0.38🟠 | 0.39🟠 | 0.26🟠 | 0.39🟠 | 0.30🟠 | 0.47🟠 | 0.43🟠 | 0.35🟠 | 0.58🟡 | — | 0.60🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.23🟠 | 0.17🔴 | 0.21🟠 | 0.37🟠 | 0.40🟠 | 0.38🟠 | 0.35🟠 | 0.32🟠 | 0.27🟠 | 0.21🟠 | 0.45🟠 | 0.22🟠 | 0.36🟠 | — | 0.64🟡 |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.21🟠 | 0.39🟠 | 0.57🟡 | 0.81🟡 | 0.84🟡 | 1.59✅ | 0.95🟡 | 1.93✅ | 1.52✅ | 0.72🟡 | 0.24🟠 | 1.26✅ | 1.45✅ | — | 1.74✅ |
| **u8** | 0.92🟡 | 0.13🔴 | 1.33✅ | 0.65🟡 | 0.65🟡 | 1.43✅ | 1.88✅ | 1.54✅ | 1.35✅ | 0.83🟡 | 2.28✅ | 1.25✅ | 1.05✅ | — | 1.57✅ |
| **i8** | 1.17✅ | 1.17✅ | 0.18🔴 | 1.02✅ | 1.24✅ | 1.56✅ | 1.38✅ | 1.18✅ | 1.22✅ | 1.06✅ | 2.21✅ | 1.24✅ | 0.94🟡 | — | 1.33✅ |
| **i16** | 1.10✅ | 1.69✅ | 1.58✅ | 0.59🟡 | 0.91🟡 | 1.44✅ | 0.98🟡 | 1.18✅ | 1.00✅ | 0.99🟡 | 2.24✅ | 0.95🟡 | 0.97🟡 | — | 1.61✅ |
| **u16** | 1.38✅ | 0.97🟡 | 1.07✅ | 0.83🟡 | 0.88🟡 | 1.39✅ | 1.24✅ | 1.13✅ | 0.88🟡 | 0.78🟡 | 2.10✅ | 1.23✅ | 1.04✅ | — | 1.46✅ |
| **i32** | 0.72🟡 | 0.69🟡 | 0.67🟡 | 0.73🟡 | 0.68🟡 | 0.89🟡 | 0.97🟡 | 1.10✅ | 0.97🟡 | 0.56🟡 | 1.99✅ | 0.68🟡 | 0.76🟡 | — | 1.55✅ |
| **u32** | 0.49🟠 | 0.57🟡 | 0.78🟡 | 0.67🟡 | 0.93🟡 | 0.86🟡 | 0.93🟡 | 0.89🟡 | 0.83🟡 | 0.56🟡 | 1.87✅ | 0.62🟡 | 0.77🟡 | — | 1.16✅ |
| **i64** | 0.17🔴 | 0.16🔴 | 0.25🟠 | 0.26🟠 | 0.22🟠 | 0.32🟠 | 0.36🟠 | 0.48🟠 | 0.54🟡 | 0.23🟠 | 0.71🟡 | 0.41🟠 | 0.42🟠 | — | 0.80🟡 |
| **u64** | 0.19🔴 | 0.20🔴 | 0.22🟠 | 0.34🟠 | 0.34🟠 | 0.60🟡 | 0.56🟡 | 0.97🟡 | 0.95🟡 | 0.30🟠 | 0.84🟡 | 0.46🟠 | 0.59🟡 | — | 1.31✅ |
| **char** | 0.61🟡 | 0.60🟡 | 0.67🟡 | 0.84🟡 | 0.66🟡 | 0.85🟡 | 0.60🟡 | 0.51🟡 | 0.63🟡 | 0.62🟡 | 1.05✅ | 0.20🟠 | 0.52🟡 | — | 0.69🟡 |
| **f16** | 1.43✅ | 1.65✅ | 1.59✅ | 1.34✅ | 1.31✅ | 1.12✅ | 0.66🟡 | 1.14✅ | 0.42🟠 | 1.47✅ | 0.67🟡 | 1.04✅ | 0.55🟡 | — | 0.89🟡 |
| **f32** | 0.79🟡 | 0.55🟡 | 0.48🟠 | 0.47🟠 | 0.40🟠 | 0.43🟠 | 0.27🟠 | 0.39🟠 | 0.33🟠 | 0.35🟠 | 0.66🟡 | 0.33🟠 | 0.44🟠 | — | 0.39🟠 |
| **f64** | 0.27🟠 | 0.39🟠 | 0.25🟠 | 0.38🟠 | 0.38🟠 | 0.37🟠 | 0.26🟠 | 0.33🟠 | 0.30🟠 | 0.40🟠 | 0.47🟠 | 0.37🟠 | 0.51🟡 | — | 0.56🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.16🔴 | 0.13🔴 | 0.13🔴 | 0.22🟠 | 0.26🟠 | 0.31🟠 | 0.29🟠 | 0.32🟠 | 0.31🟠 | 0.33🟠 | 0.46🟠 | 0.39🟠 | 0.46🟠 | — | 0.53🟡 |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.13🔴 | 0.39🟠 | 0.44🟠 | 0.66🟡 | 0.62🟡 | 1.00🟡 | 1.18✅ | 1.25✅ | 1.26✅ | 0.58🟡 | 0.23🟠 | 1.23✅ | 1.55✅ | — | 1.75✅ |
| **u8** | 1.56✅ | 0.11🔴 | 1.78✅ | 1.11✅ | 0.85🟡 | 1.54✅ | 1.64✅ | 1.32✅ | 1.19✅ | 1.00✅ | 2.25✅ | 0.87🟡 | 1.33✅ | — | 1.69✅ |
| **i8** | 1.16✅ | 1.87✅ | 0.19🔴 | 1.22✅ | 0.86🟡 | 1.41✅ | 1.40✅ | 1.17✅ | 1.66✅ | 1.08✅ | 3.04✅ | 1.01✅ | 1.12✅ | — | 1.43✅ |
| **i16** | 1.59✅ | 2.52✅ | 1.93✅ | 0.98🟡 | 0.96🟡 | 1.68✅ | 1.43✅ | 1.27✅ | 0.90🟡 | 0.90🟡 | 1.99✅ | 1.11✅ | 1.15✅ | — | 1.46✅ |
| **u16** | 1.36✅ | 1.40✅ | 1.06✅ | 0.80🟡 | 0.83🟡 | 0.84🟡 | 0.99🟡 | 0.94🟡 | 0.99🟡 | 0.77🟡 | 2.17✅ | 1.07✅ | 1.13✅ | — | 1.39✅ |
| **i32** | 0.73🟡 | 0.62🟡 | 0.70🟡 | 0.70🟡 | 0.56🟡 | 0.67🟡 | 0.68🟡 | 0.85🟡 | 0.79🟡 | 0.46🟠 | 1.19✅ | 0.46🟠 | 0.71🟡 | — | 1.01✅ |
| **u32** | 0.51🟡 | 0.52🟡 | 0.67🟡 | 0.54🟡 | 0.58🟡 | 0.66🟡 | 0.73🟡 | 0.72🟡 | 1.00🟡 | 0.74🟡 | 1.90✅ | 0.49🟠 | 0.70🟡 | — | 1.04✅ |
| **i64** | 0.15🔴 | 0.17🔴 | 0.15🔴 | 0.24🟠 | 0.26🟠 | 0.53🟡 | 0.54🟡 | 0.67🟡 | 0.64🟡 | 0.33🟠 | 0.65🟡 | 0.44🟠 | 0.84🟡 | — | 1.12✅ |
| **u64** | 0.16🔴 | 0.21🟠 | 0.17🔴 | 0.48🟠 | 0.44🟠 | 0.67🟡 | 0.86🟡 | 1.00✅ | 0.95🟡 | 0.40🟠 | 0.80🟡 | 0.42🟠 | 0.87🟡 | — | 1.27✅ |
| **char** | 0.77🟡 | 0.70🟡 | 0.63🟡 | 0.62🟡 | 0.53🟡 | 0.62🟡 | 0.59🟡 | 0.54🟡 | 0.60🟡 | 0.63🟡 | 0.84🟡 | 0.55🟡 | 0.56🟡 | — | 0.65🟡 |
| **f16** | 1.56✅ | 1.53✅ | 1.58✅ | 1.37✅ | 1.20✅ | 1.08✅ | 0.73🟡 | 0.95🟡 | 0.42🟠 | 0.99🟡 | 0.61🟡 | 0.76🟡 | 0.43🟠 | — | 0.74🟡 |
| **f32** | 0.48🟠 | 0.35🟠 | 0.41🟠 | 0.29🟠 | 0.36🟠 | 0.39🟠 | 0.22🟠 | 0.27🟠 | 0.27🟠 | 0.35🟠 | 0.54🟡 | 0.38🟠 | 0.41🟠 | — | 0.34🟠 |
| **f64** | 0.47🟠 | 0.34🟠 | 0.40🟠 | 0.58🟡 | 0.53🟡 | 0.55🟡 | 0.43🟠 | 0.44🟠 | 0.38🟠 | 0.57🟡 | 0.55🟡 | 0.66🟡 | 0.76🟡 | — | 1.17✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.19🔴 | 0.23🟠 | 0.21🟠 | 0.33🟠 | 0.28🟠 | 0.39🟠 | 0.24🟠 | 0.31🟠 | 0.26🟠 | 0.26🟠 | 0.28🟠 | 0.18🔴 | 0.37🟠 | — | 0.35🟠 |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.30🟠 | 0.75🟡 | 0.65🟡 | 0.74🟡 | 0.59🟡 | 0.93🟡 | 0.98🟡 | 0.97🟡 | 1.14✅ | 0.65🟡 | 0.22🟠 | 0.85🟡 | 0.64🟡 | — | 1.50✅ |
| **u8** | 0.52🟡 | 1.05✅ | 1.16✅ | 1.08✅ | 1.21✅ | 0.96🟡 | 1.18✅ | 1.48✅ | 1.68✅ | 1.35✅ | 1.61✅ | 0.90🟡 | 1.05✅ | — | 1.97✅ |
| **i8** | 0.55🟡 | 1.49✅ | 1.97✅ | 1.05✅ | 1.06✅ | 0.89🟡 | 1.17✅ | 0.93🟡 | 1.13✅ | 0.97🟡 | 1.60✅ | 0.96🟡 | 1.17✅ | — | 1.22✅ |
| **i16** | 1.38✅ | 1.02✅ | 1.28✅ | 0.91🟡 | 0.93🟡 | 0.93🟡 | 0.95🟡 | 1.05✅ | 1.11✅ | 0.84🟡 | 1.43✅ | 0.87🟡 | 1.00🟡 | — | 1.52✅ |
| **u16** | 1.36✅ | 1.04✅ | 1.15✅ | 0.86🟡 | 0.94🟡 | 1.23✅ | 0.91🟡 | 1.19✅ | 1.47✅ | 1.06✅ | 1.24✅ | 0.80🟡 | 0.85🟡 | — | 1.29✅ |
| **i32** | 0.69🟡 | 0.59🟡 | 0.40🟠 | 0.59🟡 | 0.61🟡 | 0.82🟡 | 1.23✅ | 0.83🟡 | 0.75🟡 | 0.62🟡 | 1.23✅ | 0.65🟡 | 0.88🟡 | — | 1.06✅ |
| **u32** | 0.56🟡 | 0.38🟠 | 0.42🟠 | 0.42🟠 | 0.30🟠 | 0.40🟠 | 0.60🟡 | 0.63🟡 | 0.74🟡 | 0.42🟠 | 1.04✅ | 0.66🟡 | 0.79🟡 | — | 0.99🟡 |
| **i64** | 0.44🟠 | 0.30🟠 | 0.28🟠 | 0.50🟡 | 0.30🟠 | 0.42🟠 | 0.40🟠 | 0.69🟡 | 0.69🟡 | 0.28🟠 | 0.53🟡 | 0.60🟡 | 0.83🟡 | — | 1.26✅ |
| **u64** | 0.16🔴 | 0.15🔴 | 0.20🔴 | 0.44🟠 | 0.27🟠 | 0.60🟡 | 0.61🟡 | 1.14✅ | 0.89🟡 | 0.36🟠 | 0.42🟠 | 0.36🟠 | 0.44🟠 | — | 0.67🟡 |
| **char** | 0.84🟡 | 0.69🟡 | 0.78🟡 | 0.52🟡 | 0.59🟡 | 0.40🟠 | 0.34🟠 | 0.50🟡 | 0.53🟡 | 0.70🟡 | 0.68🟡 | 0.28🟠 | 0.46🟠 | — | 0.34🟠 |
| **f16** | 1.10✅ | 0.40🟠 | 0.31🟠 | 0.63🟡 | 0.72🟡 | 0.76🟡 | 0.51🟡 | 0.64🟡 | 0.35🟠 | 0.73🟡 | 0.78🟡 | 0.54🟡 | 0.44🟠 | — | 0.34🟠 |
| **f32** | 0.16🔴 | 0.12🔴 | 0.15🔴 | 0.12🔴 | 0.14🔴 | 0.25🟠 | 0.17🔴 | 0.27🟠 | 0.22🟠 | 0.18🔴 | 0.49🟠 | 0.26🟠 | 0.41🟠 | — | 0.49🟠 |
| **f64** | 0.26🟠 | 0.17🔴 | 0.20🟠 | 0.40🟠 | 0.40🟠 | 0.86🟡 | 0.56🟡 | 0.80🟡 | 0.54🟡 | 0.33🟠 | 0.45🟠 | 0.59🟡 | 0.62🟡 | — | 0.72🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.19🔴 | 0.21🟠 | 0.20🟠 | 0.23🟠 | 0.19🔴 | 0.30🟠 | 0.22🟠 | 0.09🔴 | 0.32🟠 | 0.28🟠 | 0.73🟡 | 0.43🟠 | 0.66🟡 | — | 0.84🟡 |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.26🟠 | 0.55🟡 | 0.57🟡 | 0.39🟠 | 0.41🟠 | 0.66🟡 | 0.74🟡 | 1.25✅ | 1.05✅ | 0.38🟠 | 0.18🔴 | 0.72🟡 | 1.19✅ | — | 1.33✅ |
| **u8** | 0.55🟡 | 0.92🟡 | 0.90🟡 | 0.61🟡 | 1.08✅ | 0.93🟡 | 0.85🟡 | 2.05✅ | 2.13✅ | 0.80🟡 | 1.57✅ | 0.93🟡 | 1.36✅ | — | 2.14✅ |
| **i8** | 0.49🟠 | 0.98🟡 | 1.19✅ | 0.88🟡 | 0.86🟡 | 0.82🟡 | 0.63🟡 | 1.22✅ | 1.14✅ | 0.49🟠 | 1.20✅ | 0.59🟡 | 1.16✅ | — | 1.44✅ |
| **i16** | 1.74✅ | 1.21✅ | 1.10✅ | 1.13✅ | 1.18✅ | 0.71🟡 | 0.71🟡 | 1.20✅ | 1.28✅ | 0.91🟡 | 1.23✅ | 0.67🟡 | 1.21✅ | — | 1.33✅ |
| **u16** | 1.08✅ | 0.80🟡 | 0.88🟡 | 0.62🟡 | 0.57🟡 | 0.59🟡 | 0.61🟡 | 1.14✅ | 1.14✅ | 0.71🟡 | 1.33✅ | 0.55🟡 | 0.67🟡 | — | 1.09✅ |
| **i32** | 0.52🟡 | 0.53🟡 | 0.40🟠 | 0.26🟠 | 0.27🟠 | 0.39🟠 | 0.46🟠 | 0.72🟡 | 0.75🟡 | 0.26🟠 | 0.96🟡 | 0.41🟠 | 0.72🟡 | — | 1.07✅ |
| **u32** | 0.39🟠 | 0.48🟠 | 0.50🟡 | 0.47🟠 | 0.40🟠 | 0.39🟠 | 0.37🟠 | 0.69🟡 | 0.71🟡 | 0.52🟡 | 1.34✅ | 0.39🟠 | 0.32🟠 | — | 0.92🟡 |
| **i64** | 0.18🔴 | 0.14🔴 | 0.30🟠 | 0.12🔴 | 0.19🔴 | 0.34🟠 | 0.33🟠 | 0.43🟠 | 0.40🟠 | 0.10🔴 | 0.41🟠 | 0.27🟠 | 0.51🟡 | — | 0.63🟡 |
| **u64** | 0.09🔴 | 0.06🔴 | 0.10🔴 | 0.08🔴 | 0.08🔴 | 0.25🟠 | 0.27🟠 | 0.47🟠 | 0.48🟠 | 0.08🔴 | 0.36🟠 | 0.42🟠 | 0.56🟡 | — | 0.72🟡 |
| **char** | 0.21🟠 | 0.55🟡 | 0.49🟠 | 0.13🔴 | 0.13🔴 | 0.09🔴 | 0.16🔴 | 0.34🟠 | 0.45🟠 | 0.34🟠 | 0.65🟡 | 0.43🟠 | 0.41🟠 | — | 0.23🟠 |
| **f16** | 0.32🟠 | 0.70🟡 | 0.29🟠 | 0.39🟠 | 0.55🟡 | 0.52🟡 | 0.36🟠 | 0.63🟡 | 0.38🟠 | 0.91🟡 | 0.32🟠 | 0.35🟠 | 0.27🟠 | — | 0.51🟡 |
| **f32** | 0.15🔴 | 0.15🔴 | 0.17🔴 | 0.13🔴 | 0.14🔴 | 0.24🟠 | 0.16🔴 | 0.26🟠 | 0.22🟠 | 0.15🔴 | 0.42🟠 | 0.18🔴 | 0.33🟠 | — | 0.48🟠 |
| **f64** | 0.27🟠 | 0.21🟠 | 0.19🔴 | 0.20🔴 | 0.25🟠 | 0.29🟠 | 0.34🟠 | 0.37🟠 | 0.35🟠 | 0.11🔴 | 0.31🟠 | 0.34🟠 | 0.38🟠 | — | 0.60🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.39🟠 | 0.60🟡 | 0.70🟡 | 0.30🟠 | 0.26🟠 | 0.57🟡 | 0.55🟡 | 0.76🟡 | 0.39🟠 | 0.21🟠 | 0.41🟠 | 0.33🟠 | 0.50🟡 | — | 0.67🟡 |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.01🔴 | 0.31🟠 | 0.28🟠 | 0.36🟠 | 0.32🟠 | 0.41🟠 | 0.38🟠 | 0.43🟠 | 0.45🟠 | 0.35🟠 | 0.21🟠 | 0.37🟠 | 0.45🟠 | — | 0.47🟠 |
| **u8** | 0.16🔴 | 0.01🔴 | 0.19🔴 | 0.33🟠 | 0.39🟠 | 0.62🟡 | 0.65🟡 | 0.79🟡 | 0.80🟡 | 0.44🟠 | 0.43🟠 | 0.56🟡 | 0.79🟡 | — | 0.86🟡 |
| **i8** | 0.15🔴 | 0.19🔴 | 0.01🔴 | 0.25🟠 | 0.27🟠 | 0.30🟠 | 0.38🟠 | 0.52🟡 | 0.45🟠 | 0.24🟠 | 0.36🟠 | 0.35🟠 | 0.56🟡 | — | 0.52🟡 |
| **i16** | 0.28🟠 | 0.27🟠 | 0.26🟠 | 0.32🟠 | 0.39🟠 | 0.43🟠 | 0.49🟠 | 0.61🟡 | 0.86🟡 | 0.39🟠 | 0.44🟠 | 0.62🟡 | 0.69🟡 | — | 0.96🟡 |
| **u16** | 0.29🟠 | 0.27🟠 | 0.22🟠 | 0.37🟠 | 0.34🟠 | 0.50🟠 | 0.54🟡 | 0.79🟡 | 0.55🟡 | 0.35🟠 | 0.41🟠 | 0.48🟠 | 0.60🟡 | — | 0.60🟡 |
| **i32** | 0.14🔴 | 0.22🟠 | 0.16🔴 | 0.31🟠 | 0.35🟠 | 0.50🟠 | 0.48🟠 | 0.57🟡 | 0.41🟠 | 0.29🟠 | 0.45🟠 | 0.37🟠 | 0.47🟠 | — | 0.50🟡 |
| **u32** | 0.18🔴 | 0.20🔴 | 0.18🔴 | 0.29🟠 | 0.28🟠 | 0.35🟠 | 0.40🟠 | 0.47🟠 | 0.45🟠 | 0.26🟠 | 0.36🟠 | 0.45🟠 | 0.49🟠 | — | 0.41🟠 |
| **i64** | 0.15🔴 | 0.21🟠 | 0.20🟠 | 0.29🟠 | 0.33🟠 | 0.37🟠 | 0.38🟠 | 0.62🟡 | 0.57🟡 | 0.28🟠 | 0.35🟠 | 0.40🟠 | 0.54🟡 | — | 0.73🟡 |
| **u64** | 0.14🔴 | 0.09🔴 | 0.09🔴 | 0.18🔴 | 0.07🔴 | 0.25🟠 | 0.21🟠 | 0.49🟠 | 0.47🟠 | 0.14🔴 | 0.19🔴 | 0.20🔴 | 0.38🟠 | — | 0.53🟡 |
| **char** | 0.15🔴 | 0.12🔴 | 0.11🔴 | 0.17🔴 | 0.14🔴 | 0.18🔴 | 0.19🔴 | 0.33🟠 | 0.39🟠 | 0.14🔴 | 0.26🟠 | 0.18🔴 | 0.39🟠 | — | 0.50🟡 |
| **f16** | 0.23🟠 | 0.13🔴 | 0.18🔴 | 0.23🟠 | 0.21🟠 | 0.24🟠 | 0.25🟠 | 0.24🟠 | 0.28🟠 | 0.23🟠 | 0.15🔴 | 0.25🟠 | 0.35🟠 | — | 0.65🟡 |
| **f32** | 0.16🔴 | 0.07🔴 | 0.07🔴 | 0.11🔴 | 0.11🔴 | 0.16🔴 | 0.16🔴 | 0.32🟠 | 0.31🟠 | 0.12🔴 | 0.33🟠 | 0.18🔴 | 0.33🟠 | — | 0.43🟠 |
| **f64** | 0.18🔴 | 0.08🔴 | 0.10🔴 | 0.13🔴 | 0.13🔴 | 0.19🔴 | 0.19🔴 | 0.38🟠 | 0.36🟠 | 0.13🔴 | 0.30🟠 | 0.21🟠 | 0.49🟠 | — | 0.55🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.12🔴 | 0.07🔴 | 0.07🔴 | 0.12🔴 | 0.12🔴 | 0.15🔴 | 0.18🔴 | 0.46🟠 | 0.31🟠 | 0.13🔴 | 0.32🟠 | 0.46🟠 | 0.64🟡 | — | 0.61🟡 |

## Lagging cells (<1.0) — the worklist  (1128 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| i8\|bcast\|i8 | 1.3217 | 0.0145 | 0.01 🔴 |
| u8\|bcast\|u8 | 1.3055 | 0.0151 | 0.01 🔴 |
| bool\|bcast\|bool | 1.0800 | 0.0145 | 0.01 🔴 |
| u64\|strided\|u8 | 2.6992 | 0.1746 | 0.06 🔴 |
| c128\|bcast\|u8 | 4.2042 | 0.2903 | 0.07 🔴 |
| c128\|bcast\|i8 | 4.1309 | 0.2900 | 0.07 🔴 |
| f32\|bcast\|i8 | 3.9932 | 0.2829 | 0.07 🔴 |
| f32\|bcast\|u8 | 3.9020 | 0.2799 | 0.07 🔴 |
| u64\|bcast\|u16 | 8.7928 | 0.6318 | 0.07 🔴 |
| c128\|F\|i8 | 5.4094 | 0.4096 | 0.08 🔴 |
| u64\|strided\|i16 | 2.9655 | 0.2252 | 0.08 🔴 |
| f64\|bcast\|u8 | 3.8798 | 0.2949 | 0.08 🔴 |
| u64\|strided\|char | 2.4511 | 0.1938 | 0.08 🔴 |
| u64\|strided\|u16 | 2.5960 | 0.2135 | 0.08 🔴 |
| u64\|strided\|bool | 2.5300 | 0.2197 | 0.09 🔴 |
| char\|strided\|i32 | 4.1850 | 0.3641 | 0.09 🔴 |
| c128\|negcol\|i64 | 25.0167 | 2.2620 | 0.09 🔴 |
| u64\|bcast\|i8 | 2.8607 | 0.2640 | 0.09 🔴 |
| u64\|bcast\|u8 | 2.7242 | 0.2564 | 0.09 🔴 |
| u64\|strided\|i8 | 1.8687 | 0.1789 | 0.10 🔴 |
| f64\|bcast\|i8 | 3.7976 | 0.3680 | 0.10 🔴 |
| c128\|C\|u8 | 5.0435 | 0.5157 | 0.10 🔴 |
| i64\|strided\|char | 1.6594 | 0.1704 | 0.10 🔴 |
| f64\|strided\|char | 2.0751 | 0.2191 | 0.11 🔴 |
| f32\|bcast\|u16 | 4.5011 | 0.4766 | 0.11 🔴 |
| f32\|bcast\|i16 | 4.4146 | 0.4805 | 0.11 🔴 |
| char\|bcast\|i8 | 2.2641 | 0.2475 | 0.11 🔴 |
| u8\|negrow\|u8 | 0.1758 | 0.0195 | 0.11 🔴 |
| f32\|negcol\|i16 | 4.4204 | 0.5102 | 0.12 🔴 |
| c128\|bcast\|u16 | 4.3258 | 0.5124 | 0.12 🔴 |
| f32\|negcol\|u8 | 2.4741 | 0.2954 | 0.12 🔴 |
| f32\|bcast\|char | 4.5303 | 0.5450 | 0.12 🔴 |
| i64\|strided\|i16 | 1.2737 | 0.1562 | 0.12 🔴 |
| char\|bcast\|u8 | 2.2505 | 0.2779 | 0.12 🔴 |
| c128\|bcast\|bool | 3.4724 | 0.4292 | 0.12 🔴 |
| c128\|bcast\|i16 | 4.2009 | 0.5227 | 0.12 🔴 |
| c128\|sliced\|u8 | 3.1504 | 0.3958 | 0.13 🔴 |
| c128\|bcast\|char | 4.2080 | 0.5351 | 0.13 🔴 |
| i64\|F\|i8 | 1.6716 | 0.2135 | 0.13 🔴 |
| f64\|bcast\|char | 4.0212 | 0.5156 | 0.13 🔴 |
| char\|strided\|i16 | 0.9754 | 0.1253 | 0.13 🔴 |
| u8\|sliced\|u8 | 0.1488 | 0.0193 | 0.13 🔴 |
| u8\|T\|u8 | 0.1282 | 0.0166 | 0.13 🔴 |
| f64\|bcast\|u16 | 3.9753 | 0.5155 | 0.13 🔴 |
| f16\|bcast\|u8 | 8.1969 | 1.0714 | 0.13 🔴 |
| f32\|strided\|i16 | 1.1397 | 0.1496 | 0.13 🔴 |
| bool\|negrow\|bool | 0.1539 | 0.0202 | 0.13 🔴 |
| c128\|sliced\|i8 | 3.1410 | 0.4156 | 0.13 🔴 |
| f64\|bcast\|i16 | 3.9112 | 0.5198 | 0.13 🔴 |
| char\|strided\|u16 | 0.8625 | 0.1152 | 0.13 🔴 |
| f32\|strided\|u16 | 1.0865 | 0.1468 | 0.14 🔴 |
| i32\|bcast\|bool | 2.4379 | 0.3307 | 0.14 🔴 |
| char\|bcast\|char | 3.3380 | 0.4570 | 0.14 🔴 |
| u64\|bcast\|char | 3.8219 | 0.5284 | 0.14 🔴 |
| i64\|strided\|u8 | 1.0553 | 0.1470 | 0.14 🔴 |
| f32\|negcol\|u16 | 3.4680 | 0.4872 | 0.14 🔴 |
| char\|bcast\|u16 | 3.2473 | 0.4596 | 0.14 🔴 |
| u64\|bcast\|bool | 2.5063 | 0.3564 | 0.14 🔴 |
| f16\|bcast\|f16 | 3.7060 | 0.5405 | 0.15 🔴 |
| i64\|negrow\|i8 | 1.6061 | 0.2350 | 0.15 🔴 |
| i64\|negrow\|bool | 1.8644 | 0.2742 | 0.15 🔴 |
| c128\|bcast\|i32 | 5.3480 | 0.7924 | 0.15 🔴 |
| u64\|negcol\|u8 | 2.0623 | 0.3056 | 0.15 🔴 |
| f32\|strided\|u8 | 0.9536 | 0.1418 | 0.15 🔴 |
| i8\|bcast\|bool | 2.1921 | 0.3265 | 0.15 🔴 |
| i64\|C\|u8 | 1.3993 | 0.2086 | 0.15 🔴 |
| f32\|negcol\|i8 | 1.9289 | 0.2886 | 0.15 🔴 |
| f32\|strided\|char | 0.9824 | 0.1481 | 0.15 🔴 |
| char\|bcast\|bool | 2.2889 | 0.3472 | 0.15 🔴 |
| i64\|bcast\|bool | 2.1719 | 0.3313 | 0.15 🔴 |
| i64\|F\|bool | 1.4982 | 0.2299 | 0.15 🔴 |
| f32\|strided\|bool | 1.3431 | 0.2071 | 0.15 🔴 |
| i64\|sliced\|u8 | 1.5042 | 0.2332 | 0.16 🔴 |
| u64\|negrow\|bool | 1.9772 | 0.3074 | 0.16 🔴 |
| u8\|C\|u8 | 0.1140 | 0.0178 | 0.16 🔴 |
| u64\|negcol\|bool | 2.7778 | 0.4361 | 0.16 🔴 |
| char\|strided\|u32 | 2.3595 | 0.3724 | 0.16 🔴 |
| i64\|F\|u8 | 1.3813 | 0.2195 | 0.16 🔴 |
| c128\|sliced\|bool | 3.7024 | 0.5890 | 0.16 🔴 |
| f32\|bcast\|i32 | 4.7704 | 0.7628 | 0.16 🔴 |
| i32\|bcast\|i8 | 1.4255 | 0.2284 | 0.16 🔴 |
| f32\|negcol\|bool | 2.7460 | 0.4409 | 0.16 🔴 |
| f32\|bcast\|u32 | 4.8426 | 0.7850 | 0.16 🔴 |
| f32\|strided\|u32 | 2.6375 | 0.4307 | 0.16 🔴 |
| f32\|bcast\|bool | 2.5153 | 0.4121 | 0.16 🔴 |
| u8\|bcast\|bool | 2.3486 | 0.3866 | 0.16 🔴 |
| f32\|strided\|i8 | 0.8405 | 0.1389 | 0.17 🔴 |
| u64\|negrow\|i8 | 1.6675 | 0.2765 | 0.17 🔴 |
| c128\|C\|i16 | 7.6648 | 1.2749 | 0.17 🔴 |
| f32\|negcol\|u32 | 4.3666 | 0.7292 | 0.17 🔴 |
| i64\|negrow\|u8 | 1.4274 | 0.2386 | 0.17 🔴 |
| c128\|T\|u8 | 2.7261 | 0.4560 | 0.17 🔴 |
| i8\|T\|i8 | 0.0919 | 0.0155 | 0.17 🔴 |
| i64\|T\|u8 | 1.2950 | 0.2186 | 0.17 🔴 |
| f64\|negcol\|u8 | 2.0511 | 0.3475 | 0.17 🔴 |
| char\|bcast\|i16 | 3.2846 | 0.5618 | 0.17 🔴 |
| i64\|sliced\|bool | 1.3931 | 0.2430 | 0.17 🔴 |
| u32\|bcast\|i8 | 1.3250 | 0.2319 | 0.18 🔴 |
| bool\|strided\|f16 | 4.7037 | 0.8239 | 0.18 🔴 |
| f64\|F\|bool | 2.2975 | 0.4029 | 0.18 🔴 |
| u32\|bcast\|bool | 1.9603 | 0.3438 | 0.18 🔴 |
| f32\|bcast\|f32 | 4.8573 | 0.8524 | 0.18 🔴 |
| i8\|sliced\|i8 | 0.1111 | 0.0195 | 0.18 🔴 |
| u64\|bcast\|i16 | 3.7157 | 0.6557 | 0.18 🔴 |
| c128\|negrow\|f32 | 8.4542 | 1.4934 | 0.18 🔴 |
| c128\|bcast\|u32 | 4.4208 | 0.7825 | 0.18 🔴 |
| f64\|F\|i8 | 1.4829 | 0.2643 | 0.18 🔴 |
| char\|bcast\|i32 | 4.2335 | 0.7552 | 0.18 🔴 |
| i64\|strided\|bool | 1.0232 | 0.1846 | 0.18 🔴 |
| f16\|bcast\|i8 | 5.8728 | 1.0635 | 0.18 🔴 |
| i8\|C\|i8 | 0.1052 | 0.0192 | 0.18 🔴 |
| u64\|F\|i8 | 1.2926 | 0.2364 | 0.18 🔴 |
| f64\|bcast\|bool | 2.4311 | 0.4457 | 0.18 🔴 |
| f32\|strided\|f32 | 1.9691 | 0.3617 | 0.18 🔴 |
| f32\|negcol\|char | 2.7563 | 0.5073 | 0.18 🔴 |
| char\|bcast\|f32 | 4.3301 | 0.7984 | 0.18 🔴 |
| c128\|negrow\|bool | 4.6557 | 0.8619 | 0.19 🔴 |
| f64\|strided\|i8 | 1.6078 | 0.2985 | 0.19 🔴 |
| u64\|sliced\|bool | 1.3411 | 0.2511 | 0.19 🔴 |
| i64\|strided\|u16 | 0.8286 | 0.1553 | 0.19 🔴 |
| i64\|T\|i8 | 1.1935 | 0.2258 | 0.19 🔴 |
| char\|bcast\|u32 | 4.1051 | 0.7770 | 0.19 🔴 |
| c128\|negcol\|bool | 5.0853 | 0.9627 | 0.19 🔴 |
| f64\|bcast\|i32 | 4.2035 | 0.8036 | 0.19 🔴 |
| c128\|negcol\|u16 | 5.8263 | 1.1140 | 0.19 🔴 |
| u8\|bcast\|i8 | 1.2459 | 0.2382 | 0.19 🔴 |
| i8\|negrow\|i8 | 0.1002 | 0.0192 | 0.19 🔴 |
| u64\|bcast\|f16 | 10.8430 | 2.0749 | 0.19 🔴 |
| c128\|F\|u16 | 4.9438 | 0.9572 | 0.19 🔴 |
| f64\|bcast\|u32 | 4.0150 | 0.7800 | 0.19 🔴 |
| i8\|bcast\|u8 | 1.1792 | 0.2294 | 0.19 🔴 |
| f64\|T\|bool | 2.0875 | 0.4070 | 0.19 🔴 |
| u64\|negcol\|i8 | 1.6067 | 0.3135 | 0.20 🔴 |
| u32\|bcast\|u8 | 1.2278 | 0.2409 | 0.20 🔴 |
| f64\|strided\|i16 | 1.3715 | 0.2700 | 0.20 🔴 |
| u64\|bcast\|f32 | 4.5457 | 0.8997 | 0.20 🔴 |
| u64\|sliced\|u8 | 1.2218 | 0.2435 | 0.20 🔴 |
| u64\|F\|u8 | 1.1424 | 0.2281 | 0.20 🔴 |
| f64\|negcol\|i8 | 2.1312 | 0.4268 | 0.20 🟠 |
| i64\|bcast\|i8 | 1.1815 | 0.2366 | 0.20 🟠 |
| char\|sliced\|f32 | 3.8002 | 0.7660 | 0.20 🟠 |
| c128\|negcol\|i8 | 4.3563 | 0.8786 | 0.20 🟠 |
| u64\|T\|u8 | 1.1010 | 0.2256 | 0.20 🟠 |
| char\|C\|char | 1.3778 | 0.2834 | 0.21 🟠 |
| f64\|strided\|u8 | 1.3181 | 0.2715 | 0.21 🟠 |
| bool\|bcast\|f16 | 9.0282 | 1.8623 | 0.21 🟠 |
| u64\|T\|i8 | 1.2028 | 0.2483 | 0.21 🟠 |
| i64\|bcast\|u8 | 1.1402 | 0.2356 | 0.21 🟠 |
| f16\|bcast\|u16 | 6.9429 | 1.4435 | 0.21 🟠 |
| c128\|T\|char | 3.5987 | 0.7507 | 0.21 🟠 |
| c128\|strided\|char | 2.2170 | 0.4650 | 0.21 🟠 |
| char\|strided\|bool | 0.7873 | 0.1661 | 0.21 🟠 |
| bool\|sliced\|bool | 0.1448 | 0.0306 | 0.21 🟠 |
| c128\|negcol\|u8 | 4.3626 | 0.9235 | 0.21 🟠 |
| bool\|T\|bool | 0.1168 | 0.0247 | 0.21 🟠 |
| u64\|negrow\|u8 | 1.3146 | 0.2789 | 0.21 🟠 |
| u64\|bcast\|u32 | 4.4624 | 0.9483 | 0.21 🟠 |
| f64\|bcast\|f32 | 4.3061 | 0.9184 | 0.21 🟠 |
| c128\|C\|i8 | 3.0191 | 0.6448 | 0.21 🟠 |
| c128\|negrow\|i8 | 4.4241 | 0.9462 | 0.21 🟠 |
| c128\|T\|i8 | 2.7910 | 0.5982 | 0.21 🟠 |
| bool\|negcol\|f16 | 8.7949 | 1.9134 | 0.22 🟠 |
| f32\|negrow\|u32 | 3.2549 | 0.7092 | 0.22 🟠 |
| i64\|sliced\|u16 | 1.9318 | 0.4217 | 0.22 🟠 |
| i64\|C\|i8 | 0.9750 | 0.2129 | 0.22 🟠 |
| i32\|bcast\|u8 | 1.3205 | 0.2889 | 0.22 🟠 |
| c128\|T\|f32 | 4.6471 | 1.0257 | 0.22 🟠 |
| c128\|sliced\|i16 | 3.7548 | 0.8292 | 0.22 🟠 |
| u16\|bcast\|i8 | 1.1760 | 0.2605 | 0.22 🟠 |
| f64\|F\|u32 | 3.4776 | 0.7724 | 0.22 🟠 |
| c128\|negcol\|u32 | 7.8043 | 1.7389 | 0.22 🟠 |
| f32\|strided\|u64 | 3.3208 | 0.7414 | 0.22 🟠 |
| f32\|negcol\|u64 | 6.3999 | 1.4343 | 0.22 🟠 |
| u64\|sliced\|i8 | 1.0869 | 0.2439 | 0.22 🟠 |
| bool\|negrow\|f16 | 8.4937 | 1.9261 | 0.23 🟠 |
| f64\|C\|bool | 1.8478 | 0.4192 | 0.23 🟠 |
| f16\|bcast\|bool | 2.2310 | 0.5068 | 0.23 🟠 |
| f16\|bcast\|char | 6.3091 | 1.4340 | 0.23 🟠 |
| c128\|negrow\|u8 | 3.4398 | 0.7836 | 0.23 🟠 |
| c128\|T\|bool | 3.5378 | 0.8073 | 0.23 🟠 |
| c128\|F\|u8 | 2.9553 | 0.6754 | 0.23 🟠 |
| i64\|sliced\|char | 2.0964 | 0.4818 | 0.23 🟠 |
| f64\|F\|char | 1.9723 | 0.4534 | 0.23 🟠 |
| f64\|F\|i16 | 2.2005 | 0.5061 | 0.23 🟠 |
| f16\|C\|f32 | 4.8982 | 1.1278 | 0.23 🟠 |
| c128\|negcol\|i16 | 5.5301 | 1.2841 | 0.23 🟠 |
| f64\|F\|u16 | 2.0528 | 0.4768 | 0.23 🟠 |
| f16\|bcast\|i16 | 6.2607 | 1.4560 | 0.23 🟠 |
| char\|strided\|c128 | 6.5280 | 1.5328 | 0.23 🟠 |
| f32\|strided\|i32 | 1.6293 | 0.3835 | 0.24 🟠 |
| f16\|bcast\|i32 | 7.3636 | 1.7397 | 0.24 🟠 |
| c128\|C\|bool | 3.9839 | 0.9474 | 0.24 🟠 |
| i64\|negrow\|i16 | 1.9694 | 0.4692 | 0.24 🟠 |
| i64\|C\|bool | 0.9763 | 0.2330 | 0.24 🟠 |
| i8\|bcast\|char | 1.8237 | 0.4359 | 0.24 🟠 |
| c128\|C\|f32 | 6.5724 | 1.5788 | 0.24 🟠 |
| i64\|T\|u16 | 2.0297 | 0.4879 | 0.24 🟠 |
| bool\|sliced\|f16 | 8.4919 | 2.0414 | 0.24 🟠 |
| c128\|negrow\|u32 | 5.6617 | 1.3616 | 0.24 🟠 |
| f64\|C\|char | 2.0849 | 0.5102 | 0.24 🟠 |
| f16\|bcast\|i64 | 8.1765 | 2.0017 | 0.24 🟠 |
| i8\|bcast\|i16 | 1.7960 | 0.4403 | 0.25 🟠 |
| f16\|bcast\|u32 | 6.5594 | 1.6257 | 0.25 🟠 |
| f32\|negcol\|i32 | 3.1340 | 0.7770 | 0.25 🟠 |
| u64\|strided\|i32 | 3.0723 | 0.7628 | 0.25 🟠 |
| f16\|bcast\|f32 | 5.8073 | 1.4506 | 0.25 🟠 |
| u64\|bcast\|i32 | 4.1810 | 1.0481 | 0.25 🟠 |
| i64\|C\|char | 1.9009 | 0.4774 | 0.25 🟠 |
| char\|T\|i32 | 3.3344 | 0.8412 | 0.25 🟠 |
| f64\|strided\|u16 | 1.5716 | 0.3967 | 0.25 🟠 |
| f64\|sliced\|i8 | 1.0119 | 0.2570 | 0.25 🟠 |
| i64\|sliced\|i8 | 0.8532 | 0.2168 | 0.25 🟠 |
| f64\|negcol\|bool | 2.4930 | 0.6358 | 0.26 🟠 |
| f64\|sliced\|u32 | 3.1226 | 0.7973 | 0.26 🟠 |
| f64\|F\|u8 | 1.0144 | 0.2621 | 0.26 🟠 |
| char\|bcast\|f16 | 6.8089 | 1.7640 | 0.26 🟠 |
| i64\|sliced\|i16 | 1.5404 | 0.3994 | 0.26 🟠 |
| bool\|strided\|bool | 0.3751 | 0.0973 | 0.26 🟠 |
| f32\|strided\|i64 | 2.7597 | 0.7164 | 0.26 🟠 |
| f32\|negcol\|f32 | 3.0209 | 0.7845 | 0.26 🟠 |
| i64\|negrow\|u16 | 1.9578 | 0.5094 | 0.26 🟠 |
| u32\|bcast\|char | 1.7814 | 0.4636 | 0.26 🟠 |
| char\|T\|u32 | 3.2739 | 0.8528 | 0.26 🟠 |
| c128\|negrow\|u64 | 9.1271 | 2.3785 | 0.26 🟠 |
| i32\|strided\|i16 | 0.4693 | 0.1223 | 0.26 🟠 |
| i32\|strided\|char | 0.4773 | 0.1245 | 0.26 🟠 |
| c128\|sliced\|u16 | 3.0969 | 0.8090 | 0.26 🟠 |
| i16\|bcast\|i8 | 1.1388 | 0.2981 | 0.26 🟠 |
| f64\|T\|u32 | 3.4839 | 0.9122 | 0.26 🟠 |
| c128\|strided\|u16 | 1.9382 | 0.5122 | 0.26 🟠 |
| c128\|negrow\|char | 4.6106 | 1.2184 | 0.26 🟠 |
| u64\|strided\|u32 | 2.9316 | 0.7778 | 0.27 🟠 |
| f16\|strided\|f64 | 4.5362 | 1.2044 | 0.27 🟠 |
| u64\|F\|u16 | 2.1498 | 0.5710 | 0.27 🟠 |
| i16\|bcast\|u8 | 1.1790 | 0.3137 | 0.27 🟠 |
| f32\|negcol\|i64 | 5.2240 | 1.3915 | 0.27 🟠 |
| c128\|F\|f32 | 4.4728 | 1.1949 | 0.27 🟠 |
| u64\|T\|i16 | 2.0630 | 0.5521 | 0.27 🟠 |
| i64\|F\|u16 | 1.7243 | 0.4616 | 0.27 🟠 |
| u64\|C\|i8 | 0.8336 | 0.2236 | 0.27 🟠 |
| i64\|T\|bool | 1.0022 | 0.2692 | 0.27 🟠 |
| u64\|negcol\|u16 | 2.6681 | 0.7166 | 0.27 🟠 |
| f64\|strided\|bool | 1.2084 | 0.3249 | 0.27 🟠 |
| i32\|strided\|u16 | 0.4633 | 0.1251 | 0.27 🟠 |
| u16\|bcast\|u8 | 1.0567 | 0.2856 | 0.27 🟠 |
| f32\|negrow\|u64 | 5.3382 | 1.4530 | 0.27 🟠 |
| i8\|bcast\|u16 | 1.5731 | 0.4286 | 0.27 🟠 |
| f64\|sliced\|bool | 1.6422 | 0.4475 | 0.27 🟠 |
| c128\|T\|u64 | 7.1480 | 1.9498 | 0.27 🟠 |
| f64\|F\|u64 | 4.9541 | 1.3526 | 0.27 🟠 |
| f32\|negrow\|i64 | 4.9890 | 1.3624 | 0.27 🟠 |
| u64\|T\|char | 2.0596 | 0.5625 | 0.27 🟠 |
| f32\|sliced\|u32 | 2.6775 | 0.7315 | 0.27 🟠 |
| f64\|F\|i32 | 2.7355 | 0.7505 | 0.27 🟠 |
| i64\|strided\|f32 | 1.9021 | 0.5229 | 0.27 🟠 |
| c128\|F\|bool | 3.5073 | 0.9651 | 0.28 🟠 |
| i64\|C\|i16 | 1.6308 | 0.4500 | 0.28 🟠 |
| i64\|negcol\|char | 2.4810 | 0.6871 | 0.28 🟠 |
| f16\|bcast\|u64 | 9.0727 | 2.5135 | 0.28 🟠 |
| bool\|bcast\|i8 | 1.2416 | 0.3459 | 0.28 🟠 |
| f64\|T\|u8 | 0.9595 | 0.2679 | 0.28 🟠 |
| c128\|negrow\|f16 | 8.3078 | 2.3232 | 0.28 🟠 |
| i64\|negcol\|i8 | 1.0183 | 0.2848 | 0.28 🟠 |
| char\|negcol\|f32 | 3.1489 | 0.8810 | 0.28 🟠 |
| c128\|negrow\|u16 | 3.8608 | 1.0842 | 0.28 🟠 |
| u32\|bcast\|u16 | 1.7071 | 0.4826 | 0.28 🟠 |
| bool\|T\|f16 | 8.6002 | 2.4365 | 0.28 🟠 |
| c128\|negcol\|char | 5.0016 | 1.4198 | 0.28 🟠 |
| i16\|bcast\|bool | 1.3705 | 0.3892 | 0.28 🟠 |
| i64\|bcast\|char | 1.7017 | 0.4848 | 0.28 🟠 |
| f32\|negrow\|i16 | 1.7314 | 0.4938 | 0.29 🟠 |
| u32\|bcast\|i16 | 1.5570 | 0.4446 | 0.29 🟠 |
| i64\|F\|i16 | 1.6461 | 0.4701 | 0.29 🟠 |
| f64\|strided\|i32 | 2.5671 | 0.7344 | 0.29 🟠 |
| u16\|bcast\|bool | 1.2376 | 0.3564 | 0.29 🟠 |
| u64\|C\|char | 1.7483 | 0.5054 | 0.29 🟠 |
| c128\|sliced\|u32 | 4.5705 | 1.3263 | 0.29 🟠 |
| f64\|C\|u16 | 1.9952 | 0.5792 | 0.29 🟠 |
| f64\|F\|f32 | 2.6510 | 0.7751 | 0.29 🟠 |
| i64\|bcast\|i16 | 1.6188 | 0.4753 | 0.29 🟠 |
| i32\|bcast\|char | 1.6340 | 0.4802 | 0.29 🟠 |
| f16\|strided\|i8 | 1.7858 | 0.5261 | 0.29 🟠 |
| c128\|F\|i16 | 3.9554 | 1.1661 | 0.29 🟠 |
| u64\|F\|bool | 0.8324 | 0.2456 | 0.30 🟠 |
| f64\|T\|u64 | 5.0758 | 1.5003 | 0.30 🟠 |
| f64\|sliced\|u64 | 4.9647 | 1.4728 | 0.30 🟠 |
| c128\|F\|i32 | 5.0464 | 1.5044 | 0.30 🟠 |
| c128\|strided\|i16 | 1.8245 | 0.5456 | 0.30 🟠 |
| i64\|negcol\|u8 | 0.9205 | 0.2758 | 0.30 🟠 |
| u64\|sliced\|char | 1.7454 | 0.5240 | 0.30 🟠 |
| f64\|bcast\|f16 | 6.8440 | 2.0554 | 0.30 🟠 |
| bool\|negcol\|bool | 0.7207 | 0.2167 | 0.30 🟠 |
| bool\|F\|f16 | 8.1422 | 2.4511 | 0.30 🟠 |
| i64\|strided\|i8 | 0.4708 | 0.1419 | 0.30 🟠 |
| i8\|bcast\|i32 | 2.3663 | 0.7140 | 0.30 🟠 |
| f32\|T\|u32 | 3.1096 | 0.9430 | 0.30 🟠 |
| c128\|negcol\|i32 | 5.5719 | 1.6918 | 0.30 🟠 |
| f32\|C\|f32 | 2.7169 | 0.8266 | 0.30 🟠 |
| i64\|negcol\|u16 | 2.3354 | 0.7109 | 0.30 🟠 |
| u32\|negcol\|u16 | 1.5510 | 0.4723 | 0.30 🟠 |
| bool\|bcast\|u8 | 1.0925 | 0.3359 | 0.31 🟠 |
| c128\|sliced\|i32 | 3.9650 | 1.2197 | 0.31 🟠 |
| c128\|negrow\|i64 | 7.5982 | 2.3380 | 0.31 🟠 |
| c128\|F\|u64 | 7.0308 | 2.1639 | 0.31 🟠 |
| c128\|bcast\|u64 | 5.1699 | 1.5928 | 0.31 🟠 |
| f32\|bcast\|u64 | 5.1509 | 1.5880 | 0.31 🟠 |
| bool\|C\|bool | 0.0583 | 0.0180 | 0.31 🟠 |
| u64\|T\|u16 | 1.7443 | 0.5394 | 0.31 🟠 |
| f16\|negcol\|i8 | 3.3103 | 1.0305 | 0.31 🟠 |
| i32\|bcast\|i16 | 1.7329 | 0.5396 | 0.31 🟠 |
| c128\|sliced\|u64 | 7.3130 | 2.2789 | 0.31 🟠 |
| char\|T\|u64 | 5.4010 | 1.6844 | 0.31 🟠 |
| f64\|strided\|f16 | 2.9634 | 0.9261 | 0.31 🟠 |
| f32\|C\|i64 | 4.7495 | 1.4898 | 0.31 🟠 |
| i64\|T\|char | 1.5743 | 0.4940 | 0.31 🟠 |
| f32\|bcast\|i64 | 4.8077 | 1.5151 | 0.32 🟠 |
| f16\|strided\|bool | 0.8396 | 0.2651 | 0.32 🟠 |
| f16\|strided\|f16 | 0.3909 | 0.1238 | 0.32 🟠 |
| i64\|C\|u16 | 1.5237 | 0.4825 | 0.32 🟠 |
| c128\|sliced\|i64 | 6.1889 | 1.9603 | 0.32 🟠 |
| f64\|C\|u8 | 0.8498 | 0.2693 | 0.32 🟠 |
| f64\|F\|i64 | 4.2821 | 1.3608 | 0.32 🟠 |
| c128\|negcol\|u64 | 7.3758 | 2.3471 | 0.32 🟠 |
| c128\|T\|i64 | 5.4933 | 1.7482 | 0.32 🟠 |
| c128\|bcast\|f16 | 6.0472 | 1.9309 | 0.32 🟠 |
| bool\|bcast\|u16 | 1.6651 | 0.5317 | 0.32 🟠 |
| u32\|strided\|f64 | 2.3902 | 0.7661 | 0.32 🟠 |
| i64\|sliced\|i32 | 2.1280 | 0.6866 | 0.32 🟠 |
| c128\|F\|char | 3.3768 | 1.0917 | 0.32 🟠 |
| f32\|C\|u32 | 2.7584 | 0.8950 | 0.32 🟠 |
| i16\|bcast\|i16 | 1.4436 | 0.4686 | 0.32 🟠 |
| u8\|bcast\|i16 | 1.8400 | 0.5987 | 0.33 🟠 |
| f64\|C\|i8 | 0.8778 | 0.2858 | 0.33 🟠 |
| f64\|negcol\|char | 3.0846 | 1.0070 | 0.33 🟠 |
| c128\|negrow\|i16 | 3.3486 | 1.0937 | 0.33 🟠 |
| f32\|bcast\|f64 | 4.1708 | 1.3655 | 0.33 🟠 |
| u64\|T\|bool | 0.7863 | 0.2575 | 0.33 🟠 |
| bool\|C\|f16 | 8.2221 | 2.7037 | 0.33 🟠 |
| c128\|strided\|f32 | 3.0228 | 0.9946 | 0.33 🟠 |
| f32\|sliced\|u64 | 4.7470 | 1.5631 | 0.33 🟠 |
| f32\|bcast\|f16 | 5.1588 | 1.6990 | 0.33 🟠 |
| u64\|C\|bool | 0.7284 | 0.2406 | 0.33 🟠 |
| c128\|F\|u32 | 4.8442 | 1.6031 | 0.33 🟠 |
| i64\|negrow\|char | 1.8105 | 0.5999 | 0.33 🟠 |
| f32\|sliced\|f32 | 2.1322 | 0.7081 | 0.33 🟠 |
| char\|bcast\|i64 | 4.3484 | 1.4448 | 0.33 🟠 |
| f32\|strided\|f64 | 2.1017 | 0.6996 | 0.33 🟠 |
| i64\|bcast\|u16 | 1.6108 | 0.5365 | 0.33 🟠 |
| f64\|sliced\|i64 | 4.2900 | 1.4298 | 0.33 🟠 |
| i64\|strided\|u32 | 1.5836 | 0.5282 | 0.33 🟠 |
| c128\|sliced\|char | 3.3087 | 1.1043 | 0.33 🟠 |
| f64\|negrow\|u8 | 0.8157 | 0.2748 | 0.34 🟠 |
| f32\|negrow\|c128 | 8.5894 | 2.8948 | 0.34 🟠 |
| f64\|strided\|f32 | 1.8197 | 0.6133 | 0.34 🟠 |
| u64\|sliced\|i16 | 1.6438 | 0.5543 | 0.34 🟠 |
| i64\|T\|i16 | 1.4549 | 0.4907 | 0.34 🟠 |
| u16\|bcast\|u16 | 1.5237 | 0.5143 | 0.34 🟠 |
| u64\|sliced\|u16 | 1.5992 | 0.5398 | 0.34 🟠 |
| i64\|strided\|i32 | 1.5624 | 0.5302 | 0.34 🟠 |
| char\|strided\|i64 | 2.2098 | 0.7512 | 0.34 🟠 |
| char\|negcol\|u32 | 2.3253 | 0.7936 | 0.34 🟠 |
| c128\|C\|i64 | 6.1738 | 2.1090 | 0.34 🟠 |
| c128\|F\|i64 | 5.8512 | 2.0006 | 0.34 🟠 |
| f16\|negcol\|c128 | 10.1173 | 3.4597 | 0.34 🟠 |
| f64\|strided\|u32 | 2.3786 | 0.8138 | 0.34 🟠 |
| char\|negcol\|c128 | 8.6496 | 2.9697 | 0.34 🟠 |
| char\|strided\|char | 0.3422 | 0.1176 | 0.34 🟠 |
| u32\|bcast\|i32 | 2.1581 | 0.7448 | 0.35 🟠 |
| f32\|negrow\|char | 1.3410 | 0.4633 | 0.35 🟠 |
| c128\|T\|u32 | 3.8555 | 1.3350 | 0.35 🟠 |
| f64\|T\|f32 | 2.2269 | 0.7712 | 0.35 🟠 |
| c128\|C\|f64 | 5.5272 | 1.9144 | 0.35 🟠 |
| f32\|F\|u32 | 3.1362 | 1.0877 | 0.35 🟠 |
| i64\|bcast\|f16 | 5.0995 | 1.7708 | 0.35 🟠 |
| c128\|C\|u64 | 6.7818 | 2.3573 | 0.35 🟠 |
| f32\|negrow\|u8 | 0.7549 | 0.2628 | 0.35 🟠 |
| f16\|bcast\|f64 | 6.1308 | 2.1456 | 0.35 🟠 |
| bool\|bcast\|char | 1.4596 | 0.5117 | 0.35 🟠 |
| f16\|strided\|f32 | 2.2495 | 0.7888 | 0.35 🟠 |
| c128\|negrow\|c128 | 10.3333 | 3.6272 | 0.35 🟠 |
| f32\|sliced\|char | 1.3542 | 0.4760 | 0.35 🟠 |
| i8\|bcast\|f32 | 2.1681 | 0.7621 | 0.35 🟠 |
| i32\|bcast\|u16 | 1.5247 | 0.5387 | 0.35 🟠 |
| f16\|negcol\|u64 | 6.3331 | 2.2394 | 0.35 🟠 |
| u16\|bcast\|char | 1.3568 | 0.4808 | 0.35 🟠 |
| f64\|C\|i16 | 1.6973 | 0.6023 | 0.35 🟠 |
| f64\|strided\|u64 | 3.1253 | 1.1092 | 0.35 🟠 |
| f64\|bcast\|u64 | 4.3190 | 1.5333 | 0.36 🟠 |
| bool\|bcast\|i16 | 1.5477 | 0.5510 | 0.36 🟠 |
| f32\|negrow\|u16 | 1.2860 | 0.4578 | 0.36 🟠 |
| u64\|negcol\|char | 1.8773 | 0.6696 | 0.36 🟠 |
| i64\|sliced\|u32 | 2.0991 | 0.7530 | 0.36 🟠 |
| c128\|T\|f64 | 5.3736 | 1.9313 | 0.36 🟠 |
| f16\|strided\|u32 | 2.0234 | 0.7288 | 0.36 🟠 |
| u64\|negcol\|f32 | 4.1199 | 1.4841 | 0.36 🟠 |
| u64\|F\|i16 | 1.8691 | 0.6766 | 0.36 🟠 |
| u32\|bcast\|f16 | 4.9755 | 1.8043 | 0.36 🟠 |
| u64\|strided\|f16 | 2.9334 | 1.0665 | 0.36 🟠 |
| f64\|C\|i32 | 2.7216 | 0.9909 | 0.36 🟠 |
| f32\|C\|u64 | 4.8271 | 1.7580 | 0.36 🟠 |
| u64\|C\|u8 | 0.6444 | 0.2347 | 0.36 🟠 |
| i8\|bcast\|f16 | 4.7737 | 1.7419 | 0.36 🟠 |
| i32\|bcast\|f32 | 2.1111 | 0.7709 | 0.37 🟠 |
| i64\|bcast\|i32 | 2.3307 | 0.8547 | 0.37 🟠 |
| c128\|T\|i16 | 3.0732 | 1.1273 | 0.37 🟠 |
| char\|T\|bool | 0.6209 | 0.2280 | 0.37 🟠 |
| f16\|C\|u64 | 5.6710 | 2.0823 | 0.37 🟠 |
| u32\|strided\|u32 | 1.0518 | 0.3872 | 0.37 🟠 |
| u16\|bcast\|i16 | 1.4545 | 0.5371 | 0.37 🟠 |
| c128\|negrow\|f64 | 5.5129 | 2.0375 | 0.37 🟠 |
| i64\|F\|char | 1.4244 | 0.5264 | 0.37 🟠 |
| f64\|strided\|i64 | 2.8441 | 1.0512 | 0.37 🟠 |
| f64\|C\|u64 | 4.6581 | 1.7231 | 0.37 🟠 |
| f64\|sliced\|i32 | 2.0817 | 0.7705 | 0.37 🟠 |
| f64\|sliced\|f32 | 2.0856 | 0.7732 | 0.37 🟠 |
| f64\|C\|f32 | 2.5229 | 0.9364 | 0.37 🟠 |
| bool\|bcast\|f32 | 2.2572 | 0.8391 | 0.37 🟠 |
| c128\|C\|char | 3.3911 | 1.2739 | 0.38 🟠 |
| f32\|negrow\|f32 | 1.9247 | 0.7233 | 0.38 🟠 |
| f16\|strided\|u64 | 3.8893 | 1.4637 | 0.38 🟠 |
| f64\|T\|u16 | 1.2070 | 0.4548 | 0.38 🟠 |
| f64\|bcast\|i64 | 3.9440 | 1.4895 | 0.38 🟠 |
| c128\|C\|u16 | 3.7648 | 1.4225 | 0.38 🟠 |
| f64\|negrow\|u64 | 4.9214 | 1.8645 | 0.38 🟠 |
| i64\|bcast\|u32 | 2.1792 | 0.8265 | 0.38 🟠 |
| bool\|bcast\|u32 | 1.9986 | 0.7584 | 0.38 🟠 |
| u64\|bcast\|f64 | 4.0634 | 1.5440 | 0.38 🟠 |
| i8\|bcast\|u32 | 1.9001 | 0.7233 | 0.38 🟠 |
| bool\|strided\|char | 0.3825 | 0.1460 | 0.38 🟠 |
| u64\|C\|u16 | 1.4606 | 0.5578 | 0.38 🟠 |
| c128\|F\|f64 | 5.0489 | 1.9309 | 0.38 🟠 |
| u32\|negcol\|u8 | 0.6589 | 0.2520 | 0.38 🟠 |
| f64\|sliced\|i16 | 1.2778 | 0.4903 | 0.38 🟠 |
| f64\|sliced\|u16 | 1.3985 | 0.5368 | 0.38 🟠 |
| c128\|T\|i32 | 3.2692 | 1.2566 | 0.38 🟠 |
| f64\|strided\|f64 | 2.3739 | 0.9134 | 0.38 🟠 |
| i32\|strided\|i32 | 1.1080 | 0.4270 | 0.39 🟠 |
| f64\|sliced\|u8 | 0.6804 | 0.2628 | 0.39 🟠 |
| f64\|T\|i32 | 2.2580 | 0.8729 | 0.39 🟠 |
| bool\|sliced\|u8 | 0.5910 | 0.2285 | 0.39 🟠 |
| bool\|strided\|i16 | 0.3720 | 0.1440 | 0.39 🟠 |
| f32\|sliced\|i64 | 3.9060 | 1.5136 | 0.39 🟠 |
| u32\|strided\|bool | 0.4583 | 0.1777 | 0.39 🟠 |
| f32\|negrow\|i32 | 1.9099 | 0.7420 | 0.39 🟠 |
| f32\|sliced\|c128 | 6.8724 | 2.6731 | 0.39 🟠 |
| f32\|C\|char | 1.3319 | 0.5182 | 0.39 🟠 |
| char\|bcast\|u64 | 3.8814 | 1.5103 | 0.39 🟠 |
| c128\|negrow\|i32 | 3.7791 | 1.4709 | 0.39 🟠 |
| char\|bcast\|f64 | 4.0236 | 1.5753 | 0.39 🟠 |
| u32\|strided\|i32 | 1.0102 | 0.3957 | 0.39 🟠 |
| i16\|bcast\|u16 | 1.3486 | 0.5297 | 0.39 🟠 |
| u32\|strided\|f32 | 1.0474 | 0.4118 | 0.39 🟠 |
| i16\|bcast\|char | 1.6007 | 0.6294 | 0.39 🟠 |
| c128\|strided\|u64 | 3.6508 | 1.4368 | 0.39 🟠 |
| c128\|sliced\|f32 | 3.7098 | 1.4607 | 0.39 🟠 |
| c128\|strided\|bool | 1.6523 | 0.6508 | 0.39 🟠 |
| u8\|bcast\|u16 | 1.5368 | 0.6058 | 0.39 🟠 |
| f16\|strided\|i16 | 1.3950 | 0.5501 | 0.39 🟠 |
| f64\|T\|i64 | 3.9763 | 1.5702 | 0.39 🟠 |
| bool\|negrow\|u8 | 0.5358 | 0.2116 | 0.39 🟠 |
| c128\|T\|u16 | 3.1827 | 1.2573 | 0.40 🟠 |
| i64\|bcast\|f32 | 2.2846 | 0.9030 | 0.40 🟠 |
| f32\|T\|u64 | 4.9167 | 1.9490 | 0.40 🟠 |
| u32\|negcol\|i32 | 2.0630 | 0.8200 | 0.40 🟠 |
| u32\|bcast\|u32 | 2.0568 | 0.8189 | 0.40 🟠 |
| u32\|strided\|u16 | 0.3504 | 0.1399 | 0.40 🟠 |
| i64\|negcol\|u32 | 2.9639 | 1.1858 | 0.40 🟠 |
| char\|negcol\|i32 | 1.9536 | 0.7825 | 0.40 🟠 |
| f64\|negrow\|i8 | 0.9662 | 0.3871 | 0.40 🟠 |
| f64\|negcol\|u16 | 2.8136 | 1.1297 | 0.40 🟠 |
| i32\|strided\|i8 | 0.2971 | 0.1193 | 0.40 🟠 |
| i64\|strided\|u64 | 2.4064 | 0.9665 | 0.40 🟠 |
| i32\|negcol\|i8 | 0.6211 | 0.2496 | 0.40 🟠 |
| f16\|negcol\|u8 | 2.5717 | 1.0358 | 0.40 🟠 |
| f64\|negcol\|i16 | 2.6970 | 1.0865 | 0.40 🟠 |
| f32\|sliced\|u16 | 1.1478 | 0.4627 | 0.40 🟠 |
| f64\|sliced\|char | 1.2547 | 0.5079 | 0.40 🟠 |
| u64\|negrow\|char | 2.3584 | 0.9548 | 0.40 🟠 |
| char\|C\|u32 | 1.8217 | 0.7396 | 0.41 🟠 |
| bool\|bcast\|i32 | 1.9925 | 0.8097 | 0.41 🟠 |
| i32\|strided\|f32 | 1.0438 | 0.4245 | 0.41 🟠 |
| i64\|strided\|f16 | 1.9520 | 0.7942 | 0.41 🟠 |
| i32\|bcast\|u64 | 3.2776 | 1.3391 | 0.41 🟠 |
| f64\|T\|i16 | 1.2051 | 0.4925 | 0.41 🟠 |
| f16\|F\|u64 | 5.4919 | 2.2483 | 0.41 🟠 |
| i64\|T\|f32 | 2.2153 | 0.9080 | 0.41 🟠 |
| f32\|T\|char | 1.2135 | 0.4976 | 0.41 🟠 |
| f32\|negrow\|f64 | 3.3369 | 1.3694 | 0.41 🟠 |
| i64\|sliced\|f32 | 2.2250 | 0.9131 | 0.41 🟠 |
| c128\|strided\|f16 | 3.0796 | 1.2664 | 0.41 🟠 |
| u16\|bcast\|f16 | 4.6594 | 1.9186 | 0.41 🟠 |
| f32\|negcol\|f64 | 3.7730 | 1.5567 | 0.41 🟠 |
| char\|strided\|f64 | 1.7123 | 0.7068 | 0.41 🟠 |
| bool\|strided\|u16 | 0.3631 | 0.1502 | 0.41 🟠 |
| f32\|negrow\|i8 | 0.6491 | 0.2691 | 0.41 🟠 |
| u32\|bcast\|c128 | 6.5434 | 2.7139 | 0.41 🟠 |
| u32\|negcol\|i8 | 0.5823 | 0.2420 | 0.42 🟠 |
| u32\|negcol\|i16 | 1.1049 | 0.4598 | 0.42 🟠 |
| f32\|F\|u16 | 1.3874 | 0.5777 | 0.42 🟠 |
| i64\|negcol\|i32 | 2.7758 | 1.1637 | 0.42 🟠 |
| u32\|negcol\|char | 1.1666 | 0.4894 | 0.42 🟠 |
| f64\|F\|f64 | 2.5780 | 1.0821 | 0.42 🟠 |
| u64\|strided\|f32 | 2.3599 | 0.9942 | 0.42 🟠 |
| f64\|C\|f16 | 4.4053 | 1.8567 | 0.42 🟠 |
| i64\|T\|i32 | 1.9229 | 0.8123 | 0.42 🟠 |
| u64\|negcol\|f16 | 5.5085 | 2.3271 | 0.42 🟠 |
| f16\|sliced\|u64 | 5.4846 | 2.3188 | 0.42 🟠 |
| u64\|negrow\|f32 | 3.7487 | 1.5872 | 0.42 🟠 |
| i64\|sliced\|f64 | 3.5010 | 1.4824 | 0.42 🟠 |
| f16\|negrow\|u64 | 5.8758 | 2.4910 | 0.42 🟠 |
| f64\|F\|c128 | 6.5955 | 2.7986 | 0.42 🟠 |
| f32\|strided\|f16 | 1.7176 | 0.7289 | 0.42 🟠 |
| i16\|bcast\|i32 | 2.0619 | 0.8774 | 0.43 🟠 |
| char\|C\|i32 | 1.6855 | 0.7173 | 0.43 🟠 |
| u8\|bcast\|f16 | 4.7459 | 2.0245 | 0.43 🟠 |
| f32\|bcast\|c128 | 6.4360 | 2.7514 | 0.43 🟠 |
| f64\|negrow\|u32 | 2.8007 | 1.1979 | 0.43 🟠 |
| f64\|T\|f16 | 4.1697 | 1.7836 | 0.43 🟠 |
| f32\|sliced\|i32 | 1.8570 | 0.7946 | 0.43 🟠 |
| i64\|F\|u32 | 2.2100 | 0.9464 | 0.43 🟠 |
| c128\|negcol\|f32 | 3.6279 | 1.5562 | 0.43 🟠 |
| f64\|F\|f16 | 4.1583 | 1.7856 | 0.43 🟠 |
| f16\|negrow\|f64 | 3.8738 | 1.6642 | 0.43 🟠 |
| i64\|strided\|i64 | 2.2565 | 0.9712 | 0.43 🟠 |
| char\|strided\|f32 | 0.8998 | 0.3874 | 0.43 🟠 |
| bool\|bcast\|i64 | 3.1416 | 1.3535 | 0.43 🟠 |
| f16\|C\|f64 | 3.7119 | 1.6102 | 0.43 🟠 |
| f16\|T\|u64 | 5.6644 | 2.4602 | 0.43 🟠 |
| u64\|negrow\|u16 | 2.0077 | 0.8749 | 0.44 🟠 |
| i64\|negcol\|bool | 0.9813 | 0.4280 | 0.44 🟠 |
| bool\|negrow\|i8 | 0.5565 | 0.2431 | 0.44 🟠 |
| f64\|C\|i64 | 4.3685 | 1.9133 | 0.44 🟠 |
| u8\|bcast\|char | 1.5511 | 0.6798 | 0.44 🟠 |
| f32\|T\|u16 | 1.2149 | 0.5326 | 0.44 🟠 |
| u32\|C\|char | 1.1943 | 0.5238 | 0.44 🟠 |
| i16\|bcast\|f16 | 4.8659 | 2.1355 | 0.44 🟠 |
| f32\|sliced\|f64 | 2.9603 | 1.3019 | 0.44 🟠 |
| f32\|F\|u8 | 0.6499 | 0.2860 | 0.44 🟠 |
| char\|C\|u16 | 0.7702 | 0.3395 | 0.44 🟠 |
| f32\|C\|u16 | 1.5035 | 0.6627 | 0.44 🟠 |
| f64\|negrow\|i64 | 4.1013 | 1.8117 | 0.44 🟠 |
| f16\|negcol\|f64 | 4.2708 | 1.8887 | 0.44 🟠 |
| u32\|F\|i8 | 0.4955 | 0.2193 | 0.44 🟠 |
| i64\|negrow\|f32 | 2.2347 | 0.9906 | 0.44 🟠 |
| u64\|negcol\|f64 | 5.8994 | 2.6190 | 0.44 🟠 |
| u64\|negcol\|i16 | 1.6422 | 0.7304 | 0.44 🟠 |
| c128\|C\|u32 | 3.9633 | 1.7676 | 0.45 🟠 |
| u32\|bcast\|f32 | 1.9256 | 0.8593 | 0.45 🟠 |
| bool\|bcast\|u64 | 3.1097 | 1.3881 | 0.45 🟠 |
| c128\|T\|f16 | 4.3694 | 1.9531 | 0.45 🟠 |
| f64\|C\|c128 | 6.5405 | 2.9248 | 0.45 🟠 |
| char\|strided\|u64 | 1.5465 | 0.6918 | 0.45 🟠 |
| f32\|C\|i8 | 0.6004 | 0.2689 | 0.45 🟠 |
| f64\|negcol\|f16 | 4.8440 | 2.1701 | 0.45 🟠 |
| char\|C\|f32 | 1.5269 | 0.6908 | 0.45 🟠 |
| f32\|C\|i16 | 1.2992 | 0.5879 | 0.45 🟠 |
| i8\|bcast\|u64 | 2.9565 | 1.3407 | 0.45 🟠 |
| bool\|bcast\|f64 | 3.0558 | 1.3870 | 0.45 🟠 |
| i32\|bcast\|f16 | 3.9379 | 1.7892 | 0.45 🟠 |
| u32\|bcast\|u64 | 3.1650 | 1.4384 | 0.45 🟠 |
| c128\|sliced\|f16 | 4.6371 | 2.1099 | 0.46 🟠 |
| c128\|sliced\|f64 | 4.6479 | 2.1184 | 0.46 🟠 |
| u64\|sliced\|f32 | 2.3686 | 1.0799 | 0.46 🟠 |
| f32\|T\|i64 | 4.2963 | 1.9589 | 0.46 🟠 |
| char\|negcol\|f64 | 3.5025 | 1.5978 | 0.46 🟠 |
| i32\|negrow\|f32 | 1.7688 | 0.8086 | 0.46 🟠 |
| c128\|C\|f16 | 4.8594 | 2.2227 | 0.46 🟠 |
| f32\|C\|u8 | 0.5910 | 0.2706 | 0.46 🟠 |
| c128\|bcast\|f32 | 1.9243 | 0.8824 | 0.46 🟠 |
| bool\|C\|u8 | 0.4807 | 0.2207 | 0.46 🟠 |
| i32\|negrow\|char | 1.0948 | 0.5028 | 0.46 🟠 |
| u64\|T\|f32 | 2.6382 | 1.2143 | 0.46 🟠 |
| i32\|strided\|u32 | 0.9565 | 0.4407 | 0.46 🟠 |
| f64\|C\|f64 | 2.4760 | 1.1410 | 0.46 🟠 |
| f64\|T\|i8 | 0.5404 | 0.2497 | 0.46 🟠 |
| c128\|bcast\|i64 | 3.3084 | 1.5301 | 0.46 🟠 |
| i64\|C\|u32 | 1.7815 | 0.8253 | 0.46 🟠 |
| f64\|sliced\|f16 | 3.8675 | 1.7986 | 0.47 🟠 |
| i32\|bcast\|f64 | 3.1283 | 1.4551 | 0.47 🟠 |
| i64\|T\|u32 | 2.1286 | 0.9902 | 0.47 🟠 |
| f32\|T\|i16 | 1.1864 | 0.5521 | 0.47 🟠 |
| u64\|bcast\|u64 | 3.9605 | 1.8440 | 0.47 🟠 |
| char\|C\|i64 | 3.0384 | 1.4159 | 0.47 🟠 |
| f32\|T\|i8 | 0.5672 | 0.2645 | 0.47 🟠 |
| f16\|C\|f16 | 0.5850 | 0.2729 | 0.47 🟠 |
| u32\|bcast\|i64 | 3.0949 | 1.4517 | 0.47 🟠 |
| f64\|negrow\|bool | 1.1455 | 0.5382 | 0.47 🟠 |
| u32\|strided\|i16 | 0.3031 | 0.1424 | 0.47 🟠 |
| u64\|strided\|i64 | 2.9026 | 1.3641 | 0.47 🟠 |
| f64\|T\|char | 1.1569 | 0.5444 | 0.47 🟠 |
| bool\|bcast\|c128 | 5.7155 | 2.6960 | 0.47 🟠 |
| f64\|C\|u32 | 2.4615 | 1.1652 | 0.47 🟠 |
| f32\|sliced\|i16 | 1.0285 | 0.4883 | 0.47 🟠 |
| f32\|F\|i8 | 0.5663 | 0.2694 | 0.48 🟠 |
| char\|F\|u32 | 1.4975 | 0.7129 | 0.48 🟠 |
| f32\|strided\|c128 | 2.9771 | 1.4217 | 0.48 🟠 |
| i32\|bcast\|u32 | 1.8270 | 0.8740 | 0.48 🟠 |
| f32\|sliced\|i8 | 0.5556 | 0.2661 | 0.48 🟠 |
| c128\|F\|f16 | 4.9718 | 2.3836 | 0.48 🟠 |
| u64\|negrow\|i16 | 2.2009 | 1.0572 | 0.48 🟠 |
| u32\|C\|u8 | 0.4541 | 0.2185 | 0.48 🟠 |
| f32\|negrow\|bool | 0.8854 | 0.4260 | 0.48 🟠 |
| u16\|bcast\|f32 | 1.8521 | 0.8919 | 0.48 🟠 |
| u32\|strided\|u8 | 0.2702 | 0.1303 | 0.48 🟠 |
| f32\|T\|u8 | 0.5378 | 0.2606 | 0.48 🟠 |
| u64\|strided\|u64 | 2.7541 | 1.3352 | 0.48 🟠 |
| i64\|sliced\|i64 | 3.1489 | 1.5267 | 0.48 🟠 |
| f32\|negcol\|f16 | 3.5310 | 1.7137 | 0.49 🟠 |
| char\|strided\|i8 | 0.2358 | 0.1148 | 0.49 🟠 |
| f64\|bcast\|f64 | 3.5380 | 1.7219 | 0.49 🟠 |
| u32\|sliced\|bool | 0.4656 | 0.2272 | 0.49 🟠 |
| i16\|bcast\|u32 | 1.8661 | 0.9108 | 0.49 🟠 |
| u32\|bcast\|f64 | 2.8792 | 1.4059 | 0.49 🟠 |
| i8\|strided\|char | 0.2164 | 0.1057 | 0.49 🟠 |
| u32\|negrow\|f32 | 1.8073 | 0.8841 | 0.49 🟠 |
| i8\|strided\|bool | 0.3270 | 0.1600 | 0.49 🟠 |
| c128\|C\|i32 | 3.7216 | 1.8237 | 0.49 🟠 |
| f32\|negcol\|c128 | 5.8826 | 2.8852 | 0.49 🟠 |
| u32\|T\|char | 0.9333 | 0.4580 | 0.49 🟠 |
| char\|T\|c128 | 6.1961 | 3.0564 | 0.49 🟠 |
| i64\|F\|f32 | 1.9953 | 0.9847 | 0.49 🟠 |
| f32\|T\|i32 | 1.8587 | 0.9173 | 0.49 🟠 |
| u64\|bcast\|i64 | 4.2229 | 2.0848 | 0.49 🟠 |
| u32\|F\|u8 | 0.4436 | 0.2192 | 0.49 🟠 |
| u64\|C\|f32 | 2.0177 | 0.9980 | 0.49 🟠 |
| bool\|T\|i8 | 0.5079 | 0.2512 | 0.49 🟠 |
| char\|F\|c128 | 6.3434 | 3.1457 | 0.50 🟠 |
| i32\|C\|u8 | 0.4970 | 0.2471 | 0.50 🟠 |
| f32\|F\|u64 | 4.6562 | 2.3174 | 0.50 🟠 |
| f32\|F\|char | 1.2491 | 0.6224 | 0.50 🟠 |
| u16\|bcast\|i32 | 1.9360 | 0.9661 | 0.50 🟠 |
| i64\|C\|f32 | 1.7955 | 0.8961 | 0.50 🟠 |
| i32\|F\|i8 | 0.5003 | 0.2498 | 0.50 🟠 |
| i32\|bcast\|i32 | 1.9096 | 0.9546 | 0.50 🟠 |
| char\|negcol\|i64 | 2.9943 | 1.4997 | 0.50 🟡 |
| u32\|strided\|i8 | 0.2599 | 0.1302 | 0.50 🟡 |
| i32\|bcast\|c128 | 7.0365 | 3.5260 | 0.50 🟡 |
| i64\|negcol\|i16 | 1.5811 | 0.7928 | 0.50 🟡 |
| char\|bcast\|c128 | 6.0878 | 3.0601 | 0.50 🟡 |
| i32\|F\|u8 | 0.4749 | 0.2389 | 0.50 🟡 |
| c128\|strided\|f64 | 3.1412 | 1.5811 | 0.50 🟡 |
| f32\|C\|i32 | 2.0787 | 1.0510 | 0.51 🟡 |
| bool\|T\|u8 | 0.5332 | 0.2701 | 0.51 🟡 |
| f16\|negcol\|u32 | 2.9870 | 1.5141 | 0.51 🟡 |
| f16\|strided\|c128 | 4.3677 | 2.2161 | 0.51 🟡 |
| f16\|F\|f64 | 3.6382 | 1.8505 | 0.51 🟡 |
| i64\|strided\|f64 | 1.8539 | 0.9440 | 0.51 🟡 |
| f32\|F\|i64 | 4.6598 | 2.3739 | 0.51 🟡 |
| char\|sliced\|i64 | 3.1753 | 1.6218 | 0.51 🟡 |
| char\|C\|f64 | 2.8447 | 1.4547 | 0.51 🟡 |
| u32\|C\|i8 | 0.4406 | 0.2255 | 0.51 🟡 |
| f64\|sliced\|f64 | 2.9481 | 1.5096 | 0.51 🟡 |
| u32\|negrow\|bool | 0.4563 | 0.2338 | 0.51 🟡 |
| f16\|strided\|i32 | 1.4900 | 0.7684 | 0.52 🟡 |
| i8\|bcast\|i64 | 2.5370 | 1.3088 | 0.52 🟡 |
| char\|sliced\|f64 | 2.9009 | 1.5028 | 0.52 🟡 |
| i32\|strided\|bool | 0.3236 | 0.1681 | 0.52 🟡 |
| char\|negcol\|i16 | 0.9084 | 0.4719 | 0.52 🟡 |
| i8\|bcast\|c128 | 5.0791 | 2.6396 | 0.52 🟡 |
| u32\|negrow\|u8 | 0.4375 | 0.2279 | 0.52 🟡 |
| u32\|strided\|char | 0.2441 | 0.1273 | 0.52 🟡 |
| i64\|C\|i32 | 1.5550 | 0.8117 | 0.52 🟡 |
| u64\|T\|u32 | 1.8445 | 0.9652 | 0.52 🟡 |
| f32\|C\|f64 | 3.9052 | 2.0470 | 0.52 🟡 |
| u8\|negcol\|bool | 0.6520 | 0.3418 | 0.52 🟡 |
| char\|F\|u8 | 0.4138 | 0.2170 | 0.52 🟡 |
| i64\|negcol\|f16 | 3.7366 | 1.9633 | 0.53 🟡 |
| c128\|sliced\|c128 | 6.6236 | 3.4829 | 0.53 🟡 |
| i64\|negrow\|i32 | 1.6957 | 0.8938 | 0.53 🟡 |
| f64\|negrow\|u16 | 1.2793 | 0.6759 | 0.53 🟡 |
| u64\|C\|i16 | 1.2046 | 0.6374 | 0.53 🟡 |
| f32\|F\|i32 | 1.9467 | 1.0329 | 0.53 🟡 |
| char\|negcol\|u64 | 3.0000 | 1.5957 | 0.53 🟡 |
| char\|negrow\|u16 | 0.7803 | 0.4158 | 0.53 🟡 |
| u64\|bcast\|c128 | 6.0032 | 3.2087 | 0.53 🟡 |
| i32\|strided\|u8 | 0.2054 | 0.1098 | 0.53 🟡 |
| f32\|negrow\|f16 | 3.0176 | 1.6148 | 0.54 🟡 |
| u32\|negrow\|i16 | 0.8977 | 0.4806 | 0.54 🟡 |
| i32\|C\|i8 | 0.4631 | 0.2480 | 0.54 🟡 |
| i64\|negrow\|u32 | 1.6828 | 0.9030 | 0.54 🟡 |
| u16\|bcast\|u32 | 1.8803 | 1.0120 | 0.54 🟡 |
| f16\|negcol\|f32 | 2.4033 | 1.2936 | 0.54 🟡 |
| f64\|negcol\|u64 | 5.7322 | 3.0902 | 0.54 🟡 |
| i64\|bcast\|f64 | 3.2832 | 1.7713 | 0.54 🟡 |
| u32\|C\|u16 | 1.0022 | 0.5408 | 0.54 🟡 |
| char\|C\|u64 | 2.6174 | 1.4129 | 0.54 🟡 |
| f16\|T\|f64 | 3.6134 | 1.9526 | 0.54 🟡 |
| char\|negrow\|i64 | 2.9548 | 1.6043 | 0.54 🟡 |
| i64\|sliced\|u64 | 2.6126 | 1.4208 | 0.54 🟡 |
| i32\|T\|i8 | 0.4113 | 0.2240 | 0.54 🟡 |
| char\|C\|bool | 0.3969 | 0.2164 | 0.55 🟡 |
| bool\|strided\|u8 | 0.2854 | 0.1558 | 0.55 🟡 |
| char\|F\|i64 | 2.6350 | 1.4410 | 0.55 🟡 |
| u16\|strided\|f32 | 0.7329 | 0.4009 | 0.55 🟡 |
| i8\|negcol\|bool | 0.5860 | 0.3206 | 0.55 🟡 |
| f64\|bcast\|c128 | 5.4507 | 2.9817 | 0.55 🟡 |
| c128\|strided\|u32 | 1.8131 | 0.9932 | 0.55 🟡 |
| char\|T\|i64 | 2.9302 | 1.6101 | 0.55 🟡 |
| char\|F\|i32 | 1.4173 | 0.7793 | 0.55 🟡 |
| f16\|sliced\|f64 | 3.6279 | 1.9954 | 0.55 🟡 |
| char\|negrow\|f32 | 1.5072 | 0.8294 | 0.55 🟡 |
| f32\|sliced\|u8 | 0.4850 | 0.2671 | 0.55 🟡 |
| f64\|negrow\|f16 | 3.8987 | 2.1496 | 0.55 🟡 |
| u16\|bcast\|u64 | 2.7717 | 1.5296 | 0.55 🟡 |
| f64\|negrow\|i32 | 2.0264 | 1.1192 | 0.55 🟡 |
| u8\|strided\|bool | 0.3380 | 0.1867 | 0.55 🟡 |
| char\|strided\|u8 | 0.2131 | 0.1181 | 0.55 🟡 |
| f16\|strided\|u16 | 0.9584 | 0.5318 | 0.55 🟡 |
| i64\|F\|i32 | 1.6586 | 0.9210 | 0.56 🟡 |
| i32\|negrow\|u16 | 0.8890 | 0.4954 | 0.56 🟡 |
| char\|negrow\|f64 | 2.9310 | 1.6349 | 0.56 🟡 |
| bool\|C\|i8 | 0.4824 | 0.2693 | 0.56 🟡 |
| char\|T\|f64 | 2.7345 | 1.5268 | 0.56 🟡 |
| u32\|sliced\|char | 0.7510 | 0.4199 | 0.56 🟡 |
| u8\|bcast\|f32 | 1.9811 | 1.1079 | 0.56 🟡 |
| u64\|strided\|f64 | 2.5752 | 1.4413 | 0.56 🟡 |
| f16\|C\|i64 | 3.4514 | 1.9341 | 0.56 🟡 |
| u32\|negcol\|bool | 0.6317 | 0.3542 | 0.56 🟡 |
| char\|C\|c128 | 4.8609 | 2.7257 | 0.56 🟡 |
| f64\|negcol\|u32 | 3.0563 | 1.7157 | 0.56 🟡 |
| i32\|sliced\|char | 0.7766 | 0.4360 | 0.56 🟡 |
| u64\|sliced\|u32 | 2.0558 | 1.1547 | 0.56 🟡 |
| i32\|T\|u8 | 0.4098 | 0.2306 | 0.56 🟡 |
| i8\|bcast\|f64 | 2.3640 | 1.3340 | 0.56 🟡 |
| f64\|sliced\|c128 | 5.2338 | 2.9550 | 0.56 🟡 |
| i32\|bcast\|i64 | 3.0072 | 1.7024 | 0.57 🟡 |
| u64\|T\|i32 | 1.8832 | 1.0670 | 0.57 🟡 |
| f32\|T\|f64 | 3.0741 | 1.7418 | 0.57 🟡 |
| bool\|strided\|i8 | 0.2758 | 0.1564 | 0.57 🟡 |
| c128\|strided\|i32 | 1.9092 | 1.0840 | 0.57 🟡 |
| f64\|negrow\|char | 1.2903 | 0.7338 | 0.57 🟡 |
| char\|F\|f64 | 2.7509 | 1.5668 | 0.57 🟡 |
| bool\|sliced\|i8 | 0.6114 | 0.3502 | 0.57 🟡 |
| u32\|C\|bool | 0.4028 | 0.2309 | 0.57 🟡 |
| u16\|strided\|u16 | 0.2402 | 0.1379 | 0.57 🟡 |
| i64\|bcast\|u64 | 2.8663 | 1.6468 | 0.57 🟡 |
| u32\|sliced\|u8 | 0.3847 | 0.2212 | 0.57 🟡 |
| u64\|F\|u32 | 2.1079 | 1.2146 | 0.58 🟡 |
| u32\|negrow\|u16 | 0.8320 | 0.4800 | 0.58 🟡 |
| char\|F\|f32 | 1.3787 | 0.7967 | 0.58 🟡 |
| f64\|negrow\|i16 | 1.3908 | 0.8064 | 0.58 🟡 |
| f64\|T\|f64 | 2.1831 | 1.2681 | 0.58 🟡 |
| i64\|T\|f64 | 2.6474 | 1.5380 | 0.58 🟡 |
| u64\|F\|char | 1.2904 | 0.7499 | 0.58 🟡 |
| char\|F\|u64 | 2.4117 | 1.4018 | 0.58 🟡 |
| bool\|negrow\|char | 0.8608 | 0.5016 | 0.58 🟡 |
| f32\|T\|c128 | 5.5951 | 3.2745 | 0.59 🟡 |
| u16\|strided\|i32 | 0.6874 | 0.4030 | 0.59 🟡 |
| i32\|negcol\|u8 | 0.5196 | 0.3051 | 0.59 🟡 |
| u64\|T\|u64 | 2.7457 | 1.6124 | 0.59 🟡 |
| bool\|negcol\|u16 | 0.9471 | 0.5564 | 0.59 🟡 |
| char\|T\|f32 | 1.3254 | 0.7792 | 0.59 🟡 |
| f64\|negcol\|f32 | 2.5045 | 1.4731 | 0.59 🟡 |
| i16\|sliced\|i16 | 0.6772 | 0.3999 | 0.59 🟡 |
| i8\|strided\|f32 | 0.6408 | 0.3789 | 0.59 🟡 |
| char\|negcol\|u16 | 0.7731 | 0.4584 | 0.59 🟡 |
| u64\|sliced\|f64 | 3.2258 | 1.9145 | 0.59 🟡 |
| char\|negrow\|u32 | 1.5764 | 0.9371 | 0.59 🟡 |
| i32\|negcol\|i16 | 0.9343 | 0.5554 | 0.59 🟡 |
| char\|sliced\|u32 | 1.4281 | 0.8499 | 0.60 🟡 |
| char\|negrow\|u64 | 2.6076 | 1.5525 | 0.60 🟡 |
| u64\|negcol\|i32 | 2.1115 | 1.2586 | 0.60 🟡 |
| u16\|bcast\|f64 | 2.7980 | 1.6700 | 0.60 🟡 |
| u32\|negcol\|u32 | 1.4073 | 0.8408 | 0.60 🟡 |
| char\|T\|i8 | 0.3717 | 0.2224 | 0.60 🟡 |
| u64\|sliced\|i32 | 1.6697 | 0.9990 | 0.60 🟡 |
| u16\|bcast\|c128 | 6.0340 | 3.6139 | 0.60 🟡 |
| i64\|negcol\|f32 | 2.4330 | 1.4631 | 0.60 🟡 |
| f64\|T\|c128 | 4.9031 | 2.9541 | 0.60 🟡 |
| f64\|strided\|c128 | 3.1100 | 1.8765 | 0.60 🟡 |
| c128\|strided\|u8 | 1.3380 | 0.8083 | 0.60 🟡 |
| i64\|C\|i64 | 1.8816 | 1.1376 | 0.60 🟡 |
| char\|sliced\|u8 | 0.3877 | 0.2346 | 0.60 🟡 |
| u64\|negcol\|u32 | 2.3581 | 1.4351 | 0.61 🟡 |
| c128\|bcast\|c128 | 5.7398 | 3.4936 | 0.61 🟡 |
| f16\|negrow\|f16 | 0.6796 | 0.4141 | 0.61 🟡 |
| f32\|F\|i16 | 1.0205 | 0.6219 | 0.61 🟡 |
| u16\|strided\|u32 | 0.6586 | 0.4015 | 0.61 🟡 |
| c128\|C\|c128 | 4.6570 | 2.8407 | 0.61 🟡 |
| i32\|negcol\|u16 | 0.9045 | 0.5518 | 0.61 🟡 |
| u8\|strided\|i16 | 0.2129 | 0.1299 | 0.61 🟡 |
| u32\|C\|i16 | 0.8331 | 0.5086 | 0.61 🟡 |
| char\|sliced\|bool | 0.3772 | 0.2310 | 0.61 🟡 |
| u64\|F\|i32 | 1.9547 | 1.1990 | 0.61 🟡 |
| i16\|bcast\|i64 | 2.9610 | 1.8194 | 0.61 🟡 |
| u16\|F\|u16 | 0.4800 | 0.2953 | 0.62 🟡 |
| f64\|negcol\|f64 | 3.4584 | 2.1316 | 0.62 🟡 |
| u32\|sliced\|f32 | 1.3059 | 0.8051 | 0.62 🟡 |
| bool\|negrow\|u16 | 0.9056 | 0.5591 | 0.62 🟡 |
| u8\|bcast\|i32 | 1.8360 | 1.1338 | 0.62 🟡 |
| char\|sliced\|char | 0.7510 | 0.4640 | 0.62 🟡 |
| char\|negrow\|i32 | 1.2826 | 0.7926 | 0.62 🟡 |
| i32\|negrow\|u8 | 0.4054 | 0.2506 | 0.62 🟡 |
| char\|F\|char | 0.4439 | 0.2747 | 0.62 🟡 |
| char\|F\|i8 | 0.3359 | 0.2084 | 0.62 🟡 |
| i32\|negcol\|char | 0.8637 | 0.5361 | 0.62 🟡 |
| char\|negrow\|i16 | 0.7858 | 0.4880 | 0.62 🟡 |
| u16\|strided\|i16 | 0.2351 | 0.1462 | 0.62 🟡 |
| i64\|bcast\|i64 | 2.8978 | 1.8058 | 0.62 🟡 |
| i16\|bcast\|f32 | 1.9687 | 1.2289 | 0.62 🟡 |
| u32\|negcol\|i64 | 2.3762 | 1.4869 | 0.63 🟡 |
| char\|negrow\|i8 | 0.3840 | 0.2411 | 0.63 🟡 |
| i64\|strided\|c128 | 2.8815 | 1.8125 | 0.63 🟡 |
| f16\|negcol\|i16 | 2.0072 | 1.2627 | 0.63 🟡 |
| char\|sliced\|u64 | 2.8226 | 1.7789 | 0.63 🟡 |
| char\|negrow\|char | 0.6580 | 0.4147 | 0.63 🟡 |
| i64\|F\|i64 | 2.3718 | 1.4988 | 0.63 🟡 |
| i8\|strided\|u32 | 0.5766 | 0.3656 | 0.63 🟡 |
| char\|C\|i16 | 0.7283 | 0.4622 | 0.63 🟡 |
| f16\|strided\|i64 | 1.8885 | 1.1992 | 0.63 🟡 |
| i16\|F\|i16 | 0.4534 | 0.2885 | 0.64 🟡 |
| f16\|negcol\|i64 | 3.1478 | 2.0095 | 0.64 🟡 |
| i32\|F\|i16 | 0.8449 | 0.5406 | 0.64 🟡 |
| c128\|T\|c128 | 4.0449 | 2.5911 | 0.64 🟡 |
| i64\|negrow\|u64 | 2.2966 | 1.4712 | 0.64 🟡 |
| f16\|C\|u32 | 2.3040 | 1.4779 | 0.64 🟡 |
| bool\|T\|char | 0.9515 | 0.6110 | 0.64 🟡 |
| bool\|negcol\|f64 | 2.4236 | 1.5572 | 0.64 🟡 |
| c128\|bcast\|f64 | 2.9652 | 1.9078 | 0.64 🟡 |
| u8\|sliced\|i16 | 0.7038 | 0.4543 | 0.65 🟡 |
| bool\|negcol\|char | 0.8619 | 0.5569 | 0.65 🟡 |
| u8\|sliced\|u16 | 0.7080 | 0.4580 | 0.65 🟡 |
| char\|negrow\|c128 | 4.7337 | 3.0662 | 0.65 🟡 |
| char\|strided\|f16 | 1.2006 | 0.7782 | 0.65 🟡 |
| u8\|bcast\|u32 | 1.8170 | 1.1795 | 0.65 🟡 |
| f16\|bcast\|c128 | 7.4819 | 4.8591 | 0.65 🟡 |
| i32\|negcol\|f32 | 1.4110 | 0.9204 | 0.65 🟡 |
| f32\|F\|f32 | 1.2215 | 0.7980 | 0.65 🟡 |
| bool\|negcol\|i8 | 0.5309 | 0.3469 | 0.65 🟡 |
| i64\|negrow\|f16 | 3.0108 | 1.9709 | 0.65 🟡 |
| u32\|T\|u8 | 0.3446 | 0.2259 | 0.66 🟡 |
| u32\|negrow\|i32 | 1.2241 | 0.8041 | 0.66 🟡 |
| f32\|sliced\|f16 | 2.3909 | 1.5753 | 0.66 🟡 |
| char\|sliced\|u16 | 0.6723 | 0.4436 | 0.66 🟡 |
| f64\|negrow\|f32 | 1.9400 | 1.2809 | 0.66 🟡 |
| bool\|negrow\|i16 | 0.8256 | 0.5451 | 0.66 🟡 |
| c128\|negcol\|f64 | 3.3823 | 2.2349 | 0.66 🟡 |
| f16\|sliced\|u32 | 2.3032 | 1.5239 | 0.66 🟡 |
| bool\|strided\|i32 | 0.5816 | 0.3848 | 0.66 🟡 |
| i16\|T\|i64 | 2.0230 | 1.3418 | 0.66 🟡 |
| u32\|negcol\|f32 | 1.4247 | 0.9453 | 0.66 🟡 |
| char\|F\|bool | 0.3330 | 0.2215 | 0.67 🟡 |
| i64\|negrow\|i64 | 2.4861 | 1.6544 | 0.67 🟡 |
| u32\|negrow\|i8 | 0.3664 | 0.2438 | 0.67 🟡 |
| f16\|sliced\|f16 | 0.6791 | 0.4523 | 0.67 🟡 |
| char\|sliced\|i8 | 0.3387 | 0.2258 | 0.67 🟡 |
| f16\|F\|u32 | 2.1682 | 1.4467 | 0.67 🟡 |
| i16\|strided\|f32 | 0.6513 | 0.4349 | 0.67 🟡 |
| i32\|sliced\|i8 | 0.3229 | 0.2163 | 0.67 🟡 |
| u64\|negrow\|i32 | 2.1629 | 1.4503 | 0.67 🟡 |
| i32\|negrow\|i32 | 1.3462 | 0.9036 | 0.67 🟡 |
| c128\|strided\|c128 | 3.4626 | 2.3273 | 0.67 🟡 |
| u16\|strided\|f64 | 1.1488 | 0.7733 | 0.67 🟡 |
| u64\|negcol\|c128 | 7.8373 | 5.2772 | 0.67 🟡 |
| u32\|sliced\|i16 | 0.6696 | 0.4516 | 0.67 🟡 |
| i32\|sliced\|f32 | 1.1576 | 0.7834 | 0.68 🟡 |
| f16\|C\|c128 | 4.2738 | 2.9105 | 0.68 🟡 |
| char\|T\|u16 | 0.4515 | 0.3079 | 0.68 🟡 |
| char\|negcol\|f16 | 2.7592 | 1.8822 | 0.68 🟡 |
| i32\|T\|i16 | 0.8139 | 0.5556 | 0.68 🟡 |
| i32\|sliced\|u16 | 0.6652 | 0.4544 | 0.68 🟡 |
| char\|C\|u8 | 0.3405 | 0.2328 | 0.68 🟡 |
| i32\|negrow\|u32 | 1.4159 | 0.9680 | 0.68 🟡 |
| i32\|sliced\|u8 | 0.3271 | 0.2245 | 0.69 🟡 |
| i32\|C\|bool | 0.3567 | 0.2453 | 0.69 🟡 |
| i16\|bcast\|f64 | 2.6935 | 1.8601 | 0.69 🟡 |
| char\|sliced\|c128 | 4.2990 | 2.9742 | 0.69 🟡 |
| i32\|negcol\|bool | 0.5236 | 0.3623 | 0.69 🟡 |
| u32\|strided\|i64 | 1.1056 | 0.7651 | 0.69 🟡 |
| i64\|negcol\|i64 | 3.4070 | 2.3580 | 0.69 🟡 |
| char\|negcol\|u8 | 0.3611 | 0.2500 | 0.69 🟡 |
| i64\|negcol\|u64 | 3.3173 | 2.3000 | 0.69 🟡 |
| i32\|negrow\|i8 | 0.3591 | 0.2496 | 0.70 🟡 |
| char\|negrow\|u8 | 0.3555 | 0.2473 | 0.70 🟡 |
| char\|negcol\|char | 0.6592 | 0.4594 | 0.70 🟡 |
| i64\|C\|u64 | 2.1149 | 1.4808 | 0.70 🟡 |
| i16\|C\|i16 | 0.3805 | 0.2667 | 0.70 🟡 |
| c128\|strided\|i8 | 1.0316 | 0.7235 | 0.70 🟡 |
| i32\|negrow\|i16 | 0.7736 | 0.5427 | 0.70 🟡 |
| f16\|strided\|u8 | 0.7543 | 0.5309 | 0.70 🟡 |
| u32\|negrow\|f64 | 2.1666 | 1.5256 | 0.70 🟡 |
| i16\|strided\|i32 | 0.5638 | 0.3983 | 0.71 🟡 |
| char\|C\|i8 | 0.3088 | 0.2182 | 0.71 🟡 |
| c128\|F\|c128 | 3.7659 | 2.6609 | 0.71 🟡 |
| i32\|negrow\|f64 | 2.3044 | 1.6309 | 0.71 🟡 |
| i64\|sliced\|f16 | 2.4689 | 1.7477 | 0.71 🟡 |
| char\|F\|u16 | 0.3833 | 0.2714 | 0.71 🟡 |
| u32\|strided\|u64 | 1.0729 | 0.7600 | 0.71 🟡 |
| u16\|strided\|char | 0.1954 | 0.1385 | 0.71 🟡 |
| i64\|C\|f64 | 2.2931 | 1.6315 | 0.71 🟡 |
| i16\|strided\|u32 | 0.5772 | 0.4124 | 0.71 🟡 |
| f16\|T\|u32 | 2.1105 | 1.5088 | 0.71 🟡 |
| f32\|T\|f32 | 1.0334 | 0.7388 | 0.71 🟡 |
| char\|T\|u8 | 0.3447 | 0.2465 | 0.71 🟡 |
| i64\|F\|u64 | 2.6062 | 1.8637 | 0.72 🟡 |
| i32\|strided\|i64 | 1.1733 | 0.8390 | 0.72 🟡 |
| u32\|negrow\|i64 | 2.0065 | 1.4358 | 0.72 🟡 |
| i64\|F\|f64 | 2.4619 | 1.7657 | 0.72 🟡 |
| f16\|negcol\|u16 | 1.7453 | 1.2524 | 0.72 🟡 |
| bool\|strided\|f32 | 0.5922 | 0.4254 | 0.72 🟡 |
| f64\|negcol\|c128 | 4.8306 | 3.4727 | 0.72 🟡 |
| u64\|C\|f64 | 2.6261 | 1.8879 | 0.72 🟡 |
| i16\|T\|i16 | 0.4139 | 0.2983 | 0.72 🟡 |
| i32\|strided\|f64 | 1.1021 | 0.7962 | 0.72 🟡 |
| i32\|sliced\|bool | 0.3168 | 0.2290 | 0.72 🟡 |
| bool\|sliced\|char | 0.8378 | 0.6062 | 0.72 🟡 |
| f32\|C\|f16 | 2.4685 | 1.7877 | 0.72 🟡 |
| u64\|strided\|c128 | 3.9194 | 2.8409 | 0.72 🟡 |
| f16\|F\|f16 | 0.4206 | 0.3049 | 0.72 🟡 |
| f32\|C\|bool | 0.6208 | 0.4517 | 0.73 🟡 |
| c128\|negcol\|f16 | 3.1650 | 2.3038 | 0.73 🟡 |
| u32\|negrow\|u32 | 1.0560 | 0.7687 | 0.73 🟡 |
| i64\|bcast\|c128 | 5.7023 | 4.1511 | 0.73 🟡 |
| f16\|negrow\|u32 | 2.5152 | 1.8316 | 0.73 🟡 |
| i64\|T\|u64 | 2.3422 | 1.7084 | 0.73 🟡 |
| f16\|F\|i64 | 2.6902 | 1.9700 | 0.73 🟡 |
| u32\|F\|i16 | 0.7153 | 0.5239 | 0.73 🟡 |
| f16\|negcol\|char | 1.9432 | 1.4237 | 0.73 🟡 |
| i32\|negrow\|bool | 0.3866 | 0.2834 | 0.73 🟡 |
| i32\|sliced\|i16 | 0.6224 | 0.4570 | 0.73 🟡 |
| u32\|negrow\|char | 0.5907 | 0.4355 | 0.74 🟡 |
| f16\|negrow\|c128 | 5.8129 | 4.2919 | 0.74 🟡 |
| f32\|T\|f16 | 2.3317 | 1.7219 | 0.74 🟡 |
| bool\|negcol\|i16 | 0.8350 | 0.6180 | 0.74 🟡 |
| u16\|C\|char | 0.4390 | 0.3253 | 0.74 🟡 |
| bool\|strided\|u32 | 0.5255 | 0.3895 | 0.74 🟡 |
| u32\|negcol\|u64 | 2.0715 | 1.5398 | 0.74 🟡 |
| bool\|negcol\|u8 | 0.4666 | 0.3479 | 0.75 🟡 |
| i16\|F\|bool | 0.3472 | 0.2592 | 0.75 🟡 |
| i32\|F\|bool | 0.3354 | 0.2508 | 0.75 🟡 |
| i32\|strided\|u64 | 1.1747 | 0.8802 | 0.75 🟡 |
| i8\|C\|bool | 0.2669 | 0.2001 | 0.75 🟡 |
| u32\|T\|u16 | 0.5906 | 0.4435 | 0.75 🟡 |
| i32\|negcol\|u64 | 2.1912 | 1.6465 | 0.75 🟡 |
| char\|T\|char | 0.4623 | 0.3481 | 0.75 🟡 |
| f64\|negrow\|f64 | 3.1643 | 2.3920 | 0.76 🟡 |
| i32\|sliced\|f64 | 1.9321 | 1.4608 | 0.76 🟡 |
| c128\|strided\|i64 | 2.1569 | 1.6361 | 0.76 🟡 |
| f16\|negrow\|f32 | 1.4925 | 1.1360 | 0.76 🟡 |
| u32\|C\|u32 | 0.9433 | 0.7200 | 0.76 🟡 |
| f16\|negcol\|i32 | 1.9317 | 1.4757 | 0.76 🟡 |
| i32\|F\|u16 | 0.6970 | 0.5365 | 0.77 🟡 |
| u16\|negrow\|char | 0.5530 | 0.4263 | 0.77 🟡 |
| u32\|T\|f32 | 1.1732 | 0.9049 | 0.77 🟡 |
| char\|negrow\|bool | 0.3190 | 0.2462 | 0.77 🟡 |
| u64\|F\|f32 | 1.6910 | 1.3052 | 0.77 🟡 |
| u32\|sliced\|f64 | 1.7015 | 1.3142 | 0.77 🟡 |
| u64\|F\|u64 | 2.0873 | 1.6153 | 0.77 🟡 |
| f32\|F\|f64 | 2.9845 | 2.3107 | 0.77 🟡 |
| u16\|sliced\|char | 0.4925 | 0.3827 | 0.78 🟡 |
| char\|negcol\|i8 | 0.3350 | 0.2604 | 0.78 🟡 |
| i32\|T\|u16 | 0.7325 | 0.5704 | 0.78 🟡 |
| u32\|T\|u64 | 1.9187 | 1.4955 | 0.78 🟡 |
| f16\|negcol\|f16 | 0.6171 | 0.4815 | 0.78 🟡 |
| u32\|sliced\|i8 | 0.2848 | 0.2223 | 0.78 🟡 |
| f16\|F\|c128 | 4.3665 | 3.4249 | 0.78 🟡 |
| u16\|F\|char | 0.4603 | 0.3616 | 0.79 🟡 |
| f32\|C\|c128 | 6.0290 | 4.7378 | 0.79 🟡 |
| f32\|sliced\|bool | 0.5543 | 0.4363 | 0.79 🟡 |
| u16\|bcast\|i64 | 2.5945 | 2.0444 | 0.79 🟡 |
| u32\|T\|i8 | 0.2935 | 0.2314 | 0.79 🟡 |
| u32\|negcol\|f64 | 2.0711 | 1.6351 | 0.79 🟡 |
| u8\|bcast\|i64 | 3.0537 | 2.4127 | 0.79 🟡 |
| i32\|negrow\|u64 | 2.4789 | 1.9603 | 0.79 🟡 |
| u8\|bcast\|f64 | 2.7358 | 2.1690 | 0.79 🟡 |
| u16\|negcol\|f32 | 1.1055 | 0.8797 | 0.80 🟡 |
| u8\|bcast\|u64 | 3.0218 | 2.4197 | 0.80 🟡 |
| i64\|C\|f16 | 2.1610 | 1.7308 | 0.80 🟡 |
| f64\|negcol\|i64 | 3.7948 | 3.0428 | 0.80 🟡 |
| u8\|strided\|char | 0.1487 | 0.1193 | 0.80 🟡 |
| u16\|strided\|u8 | 0.1689 | 0.1355 | 0.80 🟡 |
| u64\|negrow\|f16 | 3.1541 | 2.5335 | 0.80 🟡 |
| i64\|sliced\|c128 | 3.7025 | 2.9753 | 0.80 🟡 |
| u16\|negrow\|i16 | 0.6589 | 0.5298 | 0.80 🟡 |
| bool\|T\|u16 | 0.8804 | 0.7082 | 0.80 🟡 |
| f32\|F\|f16 | 2.4525 | 1.9747 | 0.81 🟡 |
| i32\|C\|i16 | 0.7351 | 0.5930 | 0.81 🟡 |
| bool\|sliced\|i16 | 0.8356 | 0.6758 | 0.81 🟡 |
| i32\|negcol\|i32 | 1.1658 | 0.9572 | 0.82 🟡 |
| i8\|strided\|i32 | 0.5016 | 0.4131 | 0.82 🟡 |
| i64\|T\|f16 | 2.2138 | 1.8275 | 0.83 🟡 |
| f16\|T\|c128 | 4.5474 | 3.7607 | 0.83 🟡 |
| i32\|C\|char | 0.6794 | 0.5621 | 0.83 🟡 |
| u32\|sliced\|u64 | 1.6823 | 1.3920 | 0.83 🟡 |
| u16\|negrow\|u16 | 0.5008 | 0.4147 | 0.83 🟡 |
| i64\|negcol\|f64 | 2.8317 | 2.3481 | 0.83 🟡 |
| u16\|sliced\|i16 | 0.5435 | 0.4509 | 0.83 🟡 |
| u8\|sliced\|char | 0.5855 | 0.4863 | 0.83 🟡 |
| i32\|negcol\|i64 | 2.2802 | 1.8946 | 0.83 🟡 |
| char\|negcol\|bool | 0.4115 | 0.3442 | 0.84 🟡 |
| i64\|negrow\|f64 | 2.3145 | 1.9369 | 0.84 🟡 |
| char\|negrow\|f16 | 2.1964 | 1.8385 | 0.84 🟡 |
| i16\|negcol\|char | 0.5575 | 0.4667 | 0.84 🟡 |
| u64\|C\|i32 | 1.3003 | 1.0893 | 0.84 🟡 |
| f32\|F\|c128 | 5.7911 | 4.8597 | 0.84 🟡 |
| u64\|sliced\|f16 | 2.5115 | 2.1092 | 0.84 🟡 |
| c128\|negcol\|c128 | 4.4314 | 3.7292 | 0.84 🟡 |
| bool\|sliced\|u16 | 0.9326 | 0.7850 | 0.84 🟡 |
| char\|sliced\|i16 | 0.6531 | 0.5498 | 0.84 🟡 |
| u16\|negrow\|i32 | 0.9232 | 0.7782 | 0.84 🟡 |
| u64\|T\|f16 | 2.4710 | 2.0882 | 0.85 🟡 |
| bool\|negcol\|f32 | 1.0923 | 0.9232 | 0.85 🟡 |
| u16\|negcol\|f64 | 1.9954 | 1.6873 | 0.85 🟡 |
| i32\|negrow\|i64 | 2.0705 | 1.7512 | 0.85 🟡 |
| u8\|strided\|u32 | 0.6660 | 0.5637 | 0.85 🟡 |
| u8\|negrow\|u16 | 0.5307 | 0.4495 | 0.85 🟡 |
| char\|sliced\|i32 | 0.9975 | 0.8473 | 0.85 🟡 |
| i8\|strided\|u16 | 0.1362 | 0.1164 | 0.86 🟡 |
| i8\|negrow\|u16 | 0.5587 | 0.4788 | 0.86 🟡 |
| u64\|negrow\|u32 | 1.8173 | 1.5596 | 0.86 🟡 |
| i64\|T\|i64 | 1.5208 | 1.3058 | 0.86 🟡 |
| u32\|sliced\|i32 | 0.8899 | 0.7644 | 0.86 🟡 |
| u64\|T\|f64 | 2.6737 | 2.2998 | 0.86 🟡 |
| u16\|negcol\|i16 | 0.6150 | 0.5291 | 0.86 🟡 |
| f64\|negcol\|i32 | 1.8451 | 1.5899 | 0.86 🟡 |
| i16\|bcast\|u64 | 2.8597 | 2.4654 | 0.86 🟡 |
| f32\|F\|bool | 0.4994 | 0.4310 | 0.86 🟡 |
| u8\|bcast\|c128 | 5.2854 | 4.5653 | 0.86 🟡 |
| u64\|negrow\|f64 | 3.3815 | 2.9252 | 0.87 🟡 |
| f16\|T\|i64 | 2.4609 | 2.1369 | 0.87 🟡 |
| u8\|negrow\|f32 | 0.9832 | 0.8554 | 0.87 🟡 |
| i16\|negcol\|f32 | 1.0189 | 0.8896 | 0.87 🟡 |
| u16\|sliced\|u16 | 0.4503 | 0.3958 | 0.88 🟡 |
| u16\|strided\|i8 | 0.1524 | 0.1340 | 0.88 🟡 |
| i32\|F\|f32 | 0.9976 | 0.8773 | 0.88 🟡 |
| i16\|T\|u64 | 1.5030 | 1.3222 | 0.88 🟡 |
| i32\|negcol\|f64 | 2.1567 | 1.8979 | 0.88 🟡 |
| u8\|C\|char | 0.4683 | 0.4129 | 0.88 🟡 |
| u16\|sliced\|u64 | 1.5576 | 1.3740 | 0.88 🟡 |
| i8\|strided\|i16 | 0.1324 | 0.1171 | 0.88 🟡 |
| i32\|T\|f32 | 0.8922 | 0.7898 | 0.89 🟡 |
| f16\|sliced\|c128 | 4.7705 | 4.2305 | 0.89 🟡 |
| u8\|C\|f32 | 0.8233 | 0.7311 | 0.89 🟡 |
| u64\|T\|i64 | 2.2362 | 1.9897 | 0.89 🟡 |
| u64\|negcol\|u64 | 2.7513 | 2.4502 | 0.89 🟡 |
| i8\|negcol\|i32 | 0.8343 | 0.7435 | 0.89 🟡 |
| u32\|sliced\|i64 | 1.6313 | 1.4558 | 0.89 🟡 |
| f16\|T\|f16 | 0.4069 | 0.3635 | 0.89 🟡 |
| i32\|sliced\|i32 | 0.8811 | 0.7873 | 0.89 🟡 |
| i8\|T\|char | 0.4519 | 0.4042 | 0.89 🟡 |
| i32\|F\|char | 0.5915 | 0.5291 | 0.89 🟡 |
| bool\|C\|i16 | 0.6710 | 0.6017 | 0.90 🟡 |
| u8\|negcol\|f32 | 1.0486 | 0.9416 | 0.90 🟡 |
| i64\|C\|c128 | 3.2768 | 2.9465 | 0.90 🟡 |
| i16\|negrow\|u64 | 1.7092 | 1.5405 | 0.90 🟡 |
| u8\|strided\|i8 | 0.1480 | 0.1335 | 0.90 🟡 |
| i16\|negrow\|char | 0.5059 | 0.4570 | 0.90 🟡 |
| u16\|negcol\|u32 | 0.9063 | 0.8203 | 0.91 🟡 |
| i8\|T\|f32 | 0.7704 | 0.6974 | 0.91 🟡 |
| f16\|strided\|char | 0.6064 | 0.5500 | 0.91 🟡 |
| i32\|C\|u16 | 0.7032 | 0.6379 | 0.91 🟡 |
| i16\|strided\|char | 0.1837 | 0.1669 | 0.91 🟡 |
| u32\|T\|f64 | 1.7220 | 1.5644 | 0.91 🟡 |
| i16\|sliced\|u16 | 0.4850 | 0.4422 | 0.91 🟡 |
| i16\|negcol\|i16 | 0.4957 | 0.4523 | 0.91 🟡 |
| u32\|F\|bool | 0.2352 | 0.2160 | 0.92 🟡 |
| u32\|T\|i64 | 1.5957 | 1.4674 | 0.92 🟡 |
| u32\|strided\|c128 | 1.5965 | 1.4689 | 0.92 🟡 |
| u8\|strided\|u8 | 0.1498 | 0.1379 | 0.92 🟡 |
| u8\|sliced\|bool | 0.2628 | 0.2427 | 0.92 🟡 |
| u32\|T\|bool | 0.2409 | 0.2230 | 0.93 🟡 |
| u8\|strided\|i32 | 0.6131 | 0.5675 | 0.93 🟡 |
| i8\|negcol\|i64 | 1.4565 | 1.3503 | 0.93 🟡 |
| bool\|negcol\|i32 | 0.8890 | 0.8257 | 0.93 🟡 |
| u8\|strided\|f32 | 0.6287 | 0.5846 | 0.93 🟡 |
| i16\|negcol\|u16 | 0.5193 | 0.4829 | 0.93 🟡 |
| f16\|F\|f32 | 1.2730 | 1.1838 | 0.93 🟡 |
| i16\|negcol\|i32 | 0.8960 | 0.8344 | 0.93 🟡 |
| u32\|sliced\|u16 | 0.4755 | 0.4439 | 0.93 🟡 |
| u32\|sliced\|u32 | 0.8712 | 0.8139 | 0.93 🟡 |
| char\|F\|i16 | 0.4497 | 0.4204 | 0.93 🟡 |
| u16\|negcol\|u16 | 0.5846 | 0.5467 | 0.94 🟡 |
| u16\|negrow\|i64 | 1.6775 | 1.5765 | 0.94 🟡 |
| i64\|T\|c128 | 3.2136 | 3.0202 | 0.94 🟡 |
| u32\|F\|u16 | 0.6288 | 0.5918 | 0.94 🟡 |
| f32\|T\|bool | 0.4575 | 0.4307 | 0.94 🟡 |
| i8\|sliced\|f64 | 1.4360 | 1.3568 | 0.94 🟡 |
| i64\|F\|c128 | 3.3231 | 3.1409 | 0.95 🟡 |
| i16\|sliced\|f32 | 0.9116 | 0.8636 | 0.95 🟡 |
| char\|T\|i16 | 0.4782 | 0.4530 | 0.95 🟡 |
| u64\|negrow\|u64 | 2.6868 | 2.5468 | 0.95 🟡 |
| u32\|C\|f32 | 1.2041 | 1.1436 | 0.95 🟡 |
| f16\|negrow\|i64 | 2.8771 | 2.7341 | 0.95 🟡 |
| bool\|sliced\|u32 | 1.3451 | 1.2801 | 0.95 🟡 |
| f16\|F\|i32 | 1.4771 | 1.4071 | 0.95 🟡 |
| i16\|negcol\|u32 | 0.8421 | 0.8023 | 0.95 🟡 |
| i16\|C\|char | 0.4447 | 0.4240 | 0.95 🟡 |
| u64\|F\|i64 | 2.3560 | 2.2471 | 0.95 🟡 |
| u64\|sliced\|u64 | 2.2950 | 2.1910 | 0.95 🟡 |
| i64\|F\|f16 | 1.9456 | 1.8589 | 0.96 🟡 |
| i16\|bcast\|c128 | 5.6077 | 5.3599 | 0.96 🟡 |
| u32\|T\|u32 | 0.5894 | 0.5643 | 0.96 🟡 |
| u8\|negcol\|i32 | 0.9257 | 0.8867 | 0.96 🟡 |
| i8\|negcol\|f32 | 0.8372 | 0.8022 | 0.96 🟡 |
| i32\|C\|f32 | 0.9805 | 0.9399 | 0.96 🟡 |
| bool\|F\|u8 | 0.4323 | 0.4149 | 0.96 🟡 |
| i32\|strided\|f16 | 0.7963 | 0.7659 | 0.96 🟡 |
| i32\|F\|i32 | 0.7786 | 0.7496 | 0.96 🟡 |
| i16\|negrow\|u16 | 0.5303 | 0.5110 | 0.96 🟡 |
| u16\|sliced\|u8 | 0.2576 | 0.2490 | 0.97 🟡 |
| u64\|sliced\|i64 | 2.2771 | 2.2014 | 0.97 🟡 |
| u64\|C\|u32 | 1.2620 | 1.2216 | 0.97 🟡 |
| bool\|negcol\|i64 | 1.6254 | 1.5744 | 0.97 🟡 |
| i8\|negcol\|char | 0.4822 | 0.4674 | 0.97 🟡 |
| i16\|sliced\|f64 | 1.6335 | 1.5836 | 0.97 🟡 |
| u32\|F\|char | 0.5563 | 0.5405 | 0.97 🟡 |
| i32\|sliced\|u64 | 1.4805 | 1.4399 | 0.97 🟡 |
| i32\|sliced\|u32 | 0.8015 | 0.7800 | 0.97 🟡 |
| u64\|C\|f16 | 2.0987 | 2.0424 | 0.97 🟡 |
| i16\|negrow\|i16 | 0.4061 | 0.3983 | 0.98 🟡 |
| bool\|negcol\|u32 | 0.8833 | 0.8667 | 0.98 🟡 |
| i8\|strided\|u8 | 0.1116 | 0.1098 | 0.98 🟡 |
| i16\|sliced\|u32 | 0.7521 | 0.7407 | 0.98 🟡 |
| i16\|sliced\|char | 0.6652 | 0.6553 | 0.99 🟡 |
| u8\|T\|f32 | 0.7926 | 0.7810 | 0.99 🟡 |
| u64\|F\|f16 | 2.2190 | 2.1870 | 0.99 🟡 |
| u16\|T\|char | 0.3105 | 0.3064 | 0.99 🟡 |
| i32\|T\|bool | 0.2580 | 0.2547 | 0.99 🟡 |
| bool\|T\|i16 | 0.8055 | 0.7956 | 0.99 🟡 |
| f16\|negrow\|char | 1.3296 | 1.3138 | 0.99 🟡 |
| u32\|negcol\|c128 | 3.2083 | 3.1724 | 0.99 🟡 |
| u16\|negrow\|u32 | 0.7623 | 0.7539 | 0.99 🟡 |
| u16\|negrow\|u64 | 1.5086 | 1.4989 | 0.99 🟡 |
| f16\|C\|char | 1.2213 | 1.2156 | 1.00 🟡 |
| u32\|negrow\|u64 | 1.4235 | 1.4170 | 1.00 🟡 |
| bool\|negrow\|i32 | 0.7926 | 0.7896 | 1.00 🟡 |
| u16\|T\|bool | 0.2344 | 0.2339 | 1.00 🟡 |
| i16\|negcol\|f64 | 1.7908 | 1.7887 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 1128 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.25 ms   unfused    7.82 ms   (1.84x)
  (a-b)/(a+b) fused    3.67 ms   unfused   17.36 ms   (4.72x)
  sum(a*b)    fused    2.86 ms   unfused    5.15 ms   (1.80x)
  sum(af*bf)  fused    1.78 ms   unfused    2.64 ms   (1.49x)  [f32]
  a*b+c out=  fused    4.77 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.80 ms   unfused    5.69 ms   (1.50x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    5.30 ms   unfused    8.66 ms   (1.63x)
    [F      ] fused    5.17 ms   unfused    8.59 ms   (1.66x)
    [T      ] fused    4.72 ms   unfused    9.32 ms   (1.97x)
    [strided] fused    4.39 ms   unfused    7.12 ms   (1.62x)
    [bcast  ] fused    2.32 ms   unfused    5.99 ms   (2.58x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         20.43 ms
  (a-b)/(a+b)   34.27 ms
  sum(a*b)      13.21 ms
  sum(af*bf)     7.10 ms  [f32]
  a*b+c out=    10.52 ms  [two-pass with out=]
  i4*2+f8       14.97 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   22.68 ms
    [F      ]   23.92 ms
    [T      ]   17.55 ms
    [strided]   12.41 ms
    [bcast  ]   16.23 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-d37087cf9fdda14af443dc52c5a0ebc7f8066d4f752a929e00b12ec4c6ab1fb4\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0097 ms | 0.0103 ms | 0.0089 | 0.918x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1224 ms | 2.8086 ms | 1.1405 | 9.318x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1176 ms | 2.6770 ms | 1.0250 | 8.716x |
| `np.dot(a, b)` | 100,000 | 0.0086 ms | 0.0082 ms | 0.0091 | 1.110x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2230 ms | 0.1030 ms | 0.2479 | 2.407x |
| `np.inner(a, b)` | 100,000 | 0.0087 ms | 0.0089 ms | 0.0092 | 1.057x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0159 ms | 0.0049 | 0.308x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0624 ms | 0.0200 | 0.321x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0525 ms | 0.0371 | 0.707x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.3435 ms | 0.2975 | 0.866x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0100 ms | 0.0067 | 0.670x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0351 ms | 0.0417 | 1.188x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1592 ms | 0.1117 | 0.702x |
| `np.linalg.eig(a)` | 96 | missing_backend | 1.6014 ms | 1.4666 | 0.916x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0634 ms | 0.0499 | 0.787x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.4731 ms | 0.4264 | 0.901x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.1330 ms | 0.0716 | 0.538x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.0080 ms | 0.8309 | 0.824x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0280 ms | 0.0231 | 0.825x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.1874 ms | 0.1777 | 0.948x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0231 ms | 0.0120 | 0.519x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.0976 ms | 0.0919 | 0.942x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0992 ms | 0.0837 | 0.844x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 0.9049 ms | 0.9046 | 1.000x |
| `np.linalg.matmul(a, b)` | 128 | 0.2218 ms | 0.1032 ms | 0.0607 | 0.588x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0399 ms | 0.0385 | 0.965x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.3253 ms | 0.4574 | 1.406x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0205 ms | 0.0121 | 0.590x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1032 ms | 0.0879 | 0.852x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0502 ms | 0.0384 | 0.765x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.3256 ms | 0.3217 | 0.988x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3926 ms | 0.2166 ms | 0.1271 | 0.587x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0400 ms | 0.0384 | 0.960x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.3202 ms | 0.3372 | 1.053x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.1311 ms | 0.0940 | 0.717x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 0.9779 ms | 0.8117 | 0.830x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0390 ms | 0.0249 | 0.638x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.2086 ms | 0.1827 | 0.876x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0099 ms | 0.0071 | 0.717x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0357 ms | 0.0317 | 0.888x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0096 ms | 0.0087 | 0.906x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0350 ms | 0.0321 | 0.917x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0959 ms | 0.0821 | 0.856x |
| `np.linalg.svd(a)` | 96 | missing_backend | 0.8168 ms | 0.7335 | 0.898x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0371 ms | 0.0332 | 0.895x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.3204 ms | 0.2975 | 0.929x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2095 ms | 0.1074 ms | 0.0662 | 0.616x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0187 ms | 0.0129 | 0.690x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.0993 ms | 0.0866 | 0.872x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0104 ms | 0.0093 | 0.894x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0360 ms | 0.0348 | 0.967x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0772 ms | 0.0163 ms | 0.0096 | 0.589x |
| `np.matmul(a, b)` | 128 | 0.2164 ms | 0.1024 ms | 0.0600 | 0.586x |
| `np.matvec(a, b)` | 128 | 0.0244 ms | 0.0024 ms | 0.0023 | 0.958x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.2522 ms | 0.0533 | 0.211x |
| `np.roots(p)` | 16 | missing_backend | 0.0652 ms | 0.0566 | 0.868x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2113 ms | 0.1136 ms | 0.0668 | 0.588x |
| `np.vdot(a, b)` | 100,000 | 0.0100 ms | 0.0089 ms | 0.0091 | 1.022x |
| `np.vecdot(a, b)` | 100,000 | 0.1569 ms | 0.0080 ms | 0.0092 | 1.150x |
| `np.vecmat(a, b)` | 128 | 0.0036 ms | 0.0024 ms | 0.0019 | 0.792x |
