# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements. 
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell.

- Managed effective cells: **3,335**
- OpenBLAS effective cells: **57**
- Missing backend records: **44**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.4565 ms | openblas | 0.120x |
| `np.roots(p)` | float64 | 16 | missing_backend | 0.0732 ms | openblas | 0.623x |
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
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.0196 ms | 0.0081 ms | openblas | 1.235x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 4.7925 ms | not_measured | managed | 0.259x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 128 | 0.3470 ms | 0.1099 ms | openblas | 2.530x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.073x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 1.6355 ms | not_measured | managed | 0.015x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 71.7141 ms | not_measured | managed | 0.062x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 2.556x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.0034 ms | not_measured | managed | 2.824x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.0018 ms | not_measured | managed | 4.944x |
| `np.inner(a, b)` | float64 | 100,000 | 0.0165 ms | 0.0108 ms | openblas | 0.759x |
| `np.linalg.cholesky(a)` | float64 | 32 | missing_backend | 0.0181 ms | openblas | 0.337x |
| `np.linalg.cholesky(a)` | float64 | 96 | missing_backend | 0.0575 ms | openblas | 0.570x |
| `np.linalg.cond(a)` | float64 | 32 | missing_backend | 0.0523 ms | openblas | 0.763x |
| `np.linalg.cond(a)` | float64 | 96 | missing_backend | 0.3647 ms | openblas | 0.971x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.0465 ms | not_measured | managed | 0.331x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 1.6649 ms | not_measured | managed | 0.410x |
| `np.linalg.det(a)` | float64 | 32 | missing_backend | 0.0102 ms | openblas | 0.686x |
| `np.linalg.det(a)` | float64 | 96 | missing_backend | 0.0567 ms | openblas | 0.573x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 1.143x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.linalg.eig(a)` | float64 | 32 | missing_backend | 0.1765 ms | openblas | 0.687x |
| `np.linalg.eig(a)` | float64 | 96 | missing_backend | 1.9611 ms | openblas | 0.816x |
| `np.linalg.eigh(a)` | float64 | 32 | missing_backend | 0.0646 ms | openblas | 0.844x |
| `np.linalg.eigh(a)` | float64 | 96 | missing_backend | 0.5353 ms | openblas | 0.894x |
| `np.linalg.eigvals(a)` | float64 | 32 | missing_backend | 0.1094 ms | openblas | 0.654x |
| `np.linalg.eigvals(a)` | float64 | 96 | missing_backend | 0.9880 ms | openblas | 0.980x |
| `np.linalg.eigvalsh(a)` | float64 | 32 | missing_backend | 0.0458 ms | openblas | 0.541x |
| `np.linalg.eigvalsh(a)` | float64 | 96 | missing_backend | 0.2229 ms | openblas | 0.842x |
| `np.linalg.inv(a)` | float64 | 32 | missing_backend | 0.0294 ms | openblas | 0.412x |
| `np.linalg.inv(a)` | float64 | 96 | missing_backend | 0.1220 ms | openblas | 0.768x |
| `np.linalg.lstsq(a, b)` | float64 | 32 | missing_backend | 0.2219 ms | openblas | 0.410x |
| `np.linalg.lstsq(a, b)` | float64 | 96 | missing_backend | 1.0107 ms | openblas | 0.856x |
| `np.linalg.matmul(a, b)` | float64 | 128 | 0.3131 ms | 0.1544 ms | openblas | 0.433x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.0136 ms | not_measured | managed | 0.199x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 0.4949 ms | not_measured | managed | 0.161x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.0167 ms | not_measured | managed | 0.168x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.0751 ms | not_measured | managed | 0.097x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0405 ms | openblas | 1.279x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 96 | missing_backend | 0.3460 ms | openblas | 0.951x |
| `np.linalg.matrix_power(a, -1)` | float64 | 32 | missing_backend | 0.0189 ms | openblas | 0.661x |
| `np.linalg.matrix_power(a, -1)` | float64 | 96 | missing_backend | 0.1056 ms | openblas | 0.856x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.0261 ms | not_measured | managed | 0.222x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.9435 ms | not_measured | managed | 0.148x |
| `np.linalg.matrix_rank(a)` | float64 | 32 | missing_backend | 0.0494 ms | openblas | 0.816x |
| `np.linalg.matrix_rank(a)` | float64 | 96 | missing_backend | 0.3409 ms | openblas | 1.006x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.linalg.multi_dot(arrays)` | float64 | 128 | 0.4392 ms | 0.2410 ms | openblas | 0.549x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.0268 ms | not_measured | managed | 0.209x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.9438 ms | not_measured | managed | 0.181x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.278x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.0141 ms | not_measured | managed | 8.113x |
| `np.linalg.norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0415 ms | openblas | 1.198x |
| `np.linalg.norm(a, ord=2)` | float64 | 96 | missing_backend | 0.6113 ms | openblas | 0.519x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.224x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.1731 ms | not_measured | managed | 0.245x |
| `np.linalg.pinv(a)` | float64 | 32 | missing_backend | 0.1637 ms | openblas | 0.631x |
| `np.linalg.pinv(a)` | float64 | 96 | missing_backend | 1.0533 ms | openblas | 0.807x |
| `np.linalg.qr(a)` | float64 | 32 | missing_backend | 0.0408 ms | openblas | 0.657x |
| `np.linalg.qr(a)` | float64 | 96 | missing_backend | 0.2163 ms | openblas | 0.868x |
| `np.linalg.slogdet(a)` | float64 | 32 | missing_backend | 0.0103 ms | openblas | 0.786x |
| `np.linalg.slogdet(a)` | float64 | 96 | missing_backend | 0.0359 ms | openblas | 0.922x |
| `np.linalg.solve(a, b)` | float64 | 32 | missing_backend | 0.0181 ms | openblas | 0.464x |
| `np.linalg.solve(a, b)` | float64 | 96 | missing_backend | 0.0355 ms | openblas | 0.955x |
| `np.linalg.svd(a)` | float64 | 32 | missing_backend | 0.1293 ms | openblas | 0.648x |
| `np.linalg.svd(a)` | float64 | 96 | missing_backend | 0.8141 ms | openblas | 1.073x |
| `np.linalg.svdvals(a)` | float64 | 32 | missing_backend | 0.0378 ms | openblas | 0.979x |
| `np.linalg.svdvals(a)` | float64 | 96 | missing_backend | 0.3335 ms | openblas | 0.993x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0176 ms | not_measured | managed | 0.347x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.5093 ms | not_measured | managed | 0.165x |
| `np.linalg.tensordot(a, b, axes=1)` | float64 | 128 | 0.2382 ms | 0.1300 ms | openblas | 0.563x |
| `np.linalg.tensorinv(a)` | float64 | 32 | missing_backend | 0.0189 ms | openblas | 0.735x |
| `np.linalg.tensorinv(a)` | float64 | 96 | missing_backend | 0.1010 ms | openblas | 0.930x |
| `np.linalg.tensorsolve(a, b)` | float64 | 32 | missing_backend | 0.0116 ms | openblas | 0.862x |
| `np.linalg.tensorsolve(a, b)` | float64 | 96 | missing_backend | 0.0376 ms | openblas | 0.941x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.941x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.0017 ms | not_measured | managed | 1.000x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0117 ms | not_measured | managed | 0.077x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1898 ms | 0.0162 ms | openblas | 0.556x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.0160 ms | not_measured | managed | 0.169x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.3384 ms | not_measured | managed | 0.096x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.0125 ms | not_measured | managed | 0.264x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 4.5232 ms | not_measured | managed | 0.149x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 3.1763 ms | not_measured | managed | 0.220x |
| `np.matmul(a, b)` | float64 | 128 | 0.3168 ms | 0.1284 ms | openblas | 0.501x |
| `np.matvec(a, b)` | float64 | 128 | 0.0343 ms | 0.0027 ms | openblas | 0.704x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.157x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.1326 ms | not_measured | managed | 0.046x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.1155 ms | not_measured | managed | 0.068x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 0.300x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.1677 ms | not_measured | managed | 0.247x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 15.6758 ms | not_measured | managed | 0.862x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0209 ms | not_measured | managed | 0.187x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 1.1357 ms | not_measured | managed | 0.101x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 68.5345 ms | not_measured | managed | 0.015x |
| `np.tensordot(a, b, axes=1)` | float64 | 128 | 0.2355 ms | 0.1447 ms | openblas | 0.514x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.111x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.0113 ms | 0.0169 ms | managed | 9.496x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 3.9233 ms | not_measured | managed | 0.254x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.132x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1441 ms | 0.0081 ms | openblas | 1.062x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 19.6962 ms | not_measured | managed | 0.056x |
| `np.vecmat(a, b)` | float64 | 128 | 0.0051 ms | 0.0031 ms | openblas | 0.516x |
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
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.0143 ms | 0.0104 ms | openblas | 0.952x |
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
| `np.convolve(a, v, mode='same')` | float64 | 100,000 | 0.1367 ms | 4.5802 ms | managed | 8.811x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, v, mode='same')` | float64 | 100,000 | 0.1287 ms | 2.8776 ms | managed | 8.318x |
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
