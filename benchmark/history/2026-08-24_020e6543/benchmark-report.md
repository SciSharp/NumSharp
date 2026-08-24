# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell.

- Managed effective cells: **3,415**
- OpenBLAS effective cells: **58**
- Missing backend records: **44**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.1007 ms | openblas | 0.557x |
| `np.roots(p)` | float64 | 16 | missing_backend | 0.0570 ms | openblas | 0.730x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 1.0412 ms | not_measured | managed | 2.055x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 1.0315 ms | not_measured | managed | 2.089x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 0.9633 ms | not_measured | managed | 2.277x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 1.500x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.0004 ms | not_measured | managed | 1.500x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.0473 ms | not_measured | managed | 0.734x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.0003 ms | not_measured | managed | 1.333x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.0560 ms | not_measured | managed | 1.257x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.7197 ms | not_measured | managed | 0.446x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.0002 ms | not_measured | managed | 4.000x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.2039 ms | not_measured | managed | 0.115x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.852x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.0076 ms | not_measured | managed | 0.382x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.3435 ms | not_measured | managed | 0.342x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.290x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.0122 ms | not_measured | managed | 1.230x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.545x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1174 ms | not_measured | managed | 0.191x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 36.000x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.3732 ms | not_measured | managed | 0.514x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 1.2239 ms | not_measured | managed | 1.361x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 7.1487 ms | not_measured | managed | 0.034x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.5172 ms | not_measured | managed | 0.673x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 21.000x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.0188 ms | not_measured | managed | 0.782x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.8797 ms | not_measured | managed | 0.886x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 186.1864 ms | not_measured | managed | 1.338x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.0151 ms | not_measured | managed | 0.821x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.7011 ms | not_measured | managed | 0.860x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 173.1657 ms | not_measured | managed | 1.550x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.429x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.6726 ms | not_measured | managed | 0.630x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 68.2715 ms | not_measured | managed | 0.663x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.0085 ms | not_measured | managed | 0.553x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.6823 ms | not_measured | managed | 0.659x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 75.0528 ms | not_measured | managed | 0.663x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.0118 ms | not_measured | managed | 1.051x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 1.5681 ms | not_measured | managed | 0.990x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 157.2452 ms | not_measured | managed | 0.951x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.0105 ms | not_measured | managed | 1.057x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 1.4549 ms | not_measured | managed | 0.889x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 152.0806 ms | not_measured | managed | 1.636x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.0030 ms | not_measured | managed | 0.700x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.5972 ms | not_measured | managed | 0.671x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 60.2907 ms | not_measured | managed | 0.710x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.0036 ms | not_measured | managed | 1.028x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.6027 ms | not_measured | managed | 0.655x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 66.5675 ms | not_measured | managed | 0.716x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.0081 ms | not_measured | managed | 0.136x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.3355 ms | not_measured | managed | 0.865x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 41.1246 ms | not_measured | managed | 1.354x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.0108 ms | not_measured | managed | 0.361x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.8808 ms | not_measured | managed | 0.357x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 87.9391 ms | not_measured | managed | 0.352x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.0058 ms | not_measured | managed | 0.138x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.0543 ms | not_measured | managed | 0.129x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 9.0068 ms | not_measured | managed | 0.919x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.141x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.1143 ms | not_measured | managed | 0.121x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 19.3739 ms | not_measured | managed | 1.500x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.0029 ms | not_measured | managed | 0.379x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.0290 ms | not_measured | managed | 0.821x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 4.9792 ms | not_measured | managed | 0.903x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.0032 ms | not_measured | managed | 0.344x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.0518 ms | not_measured | managed | 0.471x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 7.7183 ms | not_measured | managed | 1.012x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.0064 ms | not_measured | managed | 0.156x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.1141 ms | not_measured | managed | 0.240x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 30.9685 ms | not_measured | managed | 0.487x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.0039 ms | not_measured | managed | 0.256x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.0161 ms | not_measured | managed | 1.447x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 2.1932 ms | not_measured | managed | 2.088x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.0029 ms | not_measured | managed | 0.379x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.0334 ms | not_measured | managed | 0.787x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 4.3191 ms | not_measured | managed | 1.045x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.0058 ms | not_measured | managed | 0.224x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.0536 ms | not_measured | managed | 0.502x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 7.7876 ms | not_measured | managed | 1.005x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.0058 ms | not_measured | managed | 0.172x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.1153 ms | not_measured | managed | 0.219x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 21.1917 ms | not_measured | managed | 0.706x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.0028 ms | not_measured | managed | 0.321x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.0238 ms | not_measured | managed | 1.101x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 2.3399 ms | not_measured | managed | 1.387x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.0039 ms | not_measured | managed | 0.205x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.3273 ms | not_measured | managed | 0.847x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 43.2034 ms | not_measured | managed | 1.324x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.367x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.8819 ms | not_measured | managed | 0.346x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 88.8828 ms | not_measured | managed | 0.343x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.0525 ms | not_measured | managed | 0.128x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 11.2137 ms | not_measured | managed | 0.697x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.1060 ms | not_measured | managed | 0.141x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 21.2552 ms | not_measured | managed | 1.325x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.0266 ms | not_measured | managed | 1.429x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 4.5029 ms | not_measured | managed | 1.097x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.0505 ms | not_measured | managed | 0.620x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 9.9714 ms | not_measured | managed | 0.794x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.1114 ms | not_measured | managed | 0.305x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 30.7221 ms | not_measured | managed | 0.495x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 0.467x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.0160 ms | not_measured | managed | 1.837x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 2.1416 ms | not_measured | managed | 1.804x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.0277 ms | not_measured | managed | 1.072x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 4.4012 ms | not_measured | managed | 1.094x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.0515 ms | not_measured | managed | 0.612x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 8.0786 ms | not_measured | managed | 0.998x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.1064 ms | not_measured | managed | 0.286x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 31.3451 ms | not_measured | managed | 0.486x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.467x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.0164 ms | not_measured | managed | 1.774x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 2.1598 ms | not_measured | managed | 1.763x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.0044 ms | not_measured | managed | 0.205x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.3535 ms | not_measured | managed | 0.818x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 56.6006 ms | not_measured | managed | 1.231x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.0096 ms | not_measured | managed | 0.385x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.8655 ms | not_measured | managed | 0.364x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 87.8373 ms | not_measured | managed | 0.346x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.0517 ms | not_measured | managed | 0.147x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 10.6183 ms | not_measured | managed | 0.802x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.1037 ms | not_measured | managed | 0.275x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 33.3261 ms | not_measured | managed | 1.091x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.0017 ms | not_measured | managed | 0.529x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.0276 ms | not_measured | managed | 1.087x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 6.0667 ms | not_measured | managed | 0.828x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.0518 ms | not_measured | managed | 0.730x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 11.1295 ms | not_measured | managed | 0.780x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.1153 ms | not_measured | managed | 0.325x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 37.7497 ms | not_measured | managed | 0.453x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.0161 ms | not_measured | managed | 1.851x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 3.0313 ms | not_measured | managed | 1.288x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.0273 ms | not_measured | managed | 1.172x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.1524 ms | not_measured | managed | 0.830x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.0540 ms | not_measured | managed | 0.643x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.3659 ms | not_measured | managed | 0.762x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.367x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.1143 ms | not_measured | managed | 0.321x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 37.4667 ms | not_measured | managed | 0.452x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.0167 ms | not_measured | managed | 1.826x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.0235 ms | not_measured | managed | 1.241x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.0037 ms | not_measured | managed | 0.243x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.3220 ms | not_measured | managed | 0.903x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 40.9404 ms | not_measured | managed | 1.502x |
| `a * scalar (float16)` | float16 | 1,000 | 0.0095 ms | not_measured | managed | 0.400x |
| `a * scalar (float16)` | float16 | 100,000 | 0.8796 ms | not_measured | managed | 0.352x |
| `a * scalar (float16)` | float16 | 10,000,000 | 88.9203 ms | not_measured | managed | 0.343x |
| `a * scalar (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.304x |
| `a * scalar (float32)` | float32 | 100,000 | 0.0504 ms | not_measured | managed | 0.133x |
| `a * scalar (float32)` | float32 | 10,000,000 | 9.0232 ms | not_measured | managed | 0.877x |
| `a * scalar (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.233x |
| `a * scalar (float64)` | float64 | 100,000 | 0.1117 ms | not_measured | managed | 0.118x |
| `a * scalar (float64)` | float64 | 10,000,000 | 19.5480 ms | not_measured | managed | 1.361x |
| `a * scalar (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.632x |
| `a * scalar (int16)` | int16 | 100,000 | 0.0275 ms | not_measured | managed | 0.873x |
| `a * scalar (int16)` | int16 | 10,000,000 | 4.2122 ms | not_measured | managed | 1.049x |
| `a * scalar (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `a * scalar (int32)` | int32 | 100,000 | 0.0513 ms | not_measured | managed | 0.487x |
| `a * scalar (int32)` | int32 | 10,000,000 | 9.9604 ms | not_measured | managed | 0.779x |
| `a * scalar (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.345x |
| `a * scalar (int64)` | int64 | 100,000 | 0.1185 ms | not_measured | managed | 0.217x |
| `a * scalar (int64)` | int64 | 10,000,000 | 30.9862 ms | not_measured | managed | 0.494x |
| `a * scalar (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a * scalar (int8)` | int8 | 100,000 | 0.0160 ms | not_measured | managed | 1.456x |
| `a * scalar (int8)` | int8 | 10,000,000 | 2.1686 ms | not_measured | managed | 1.496x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.0274 ms | not_measured | managed | 1.000x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 4.2107 ms | not_measured | managed | 1.086x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.0516 ms | not_measured | managed | 0.459x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 8.7568 ms | not_measured | managed | 0.886x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.310x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.1110 ms | not_measured | managed | 0.213x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 30.1447 ms | not_measured | managed | 0.501x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.0161 ms | not_measured | managed | 1.472x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 2.1423 ms | not_measured | managed | 1.484x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.0086 ms | not_measured | managed | 0.128x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.3091 ms | not_measured | managed | 0.952x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 42.1934 ms | not_measured | managed | 1.351x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.0109 ms | not_measured | managed | 0.606x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.8788 ms | not_measured | managed | 0.366x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 89.3755 ms | not_measured | managed | 0.341x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.0061 ms | not_measured | managed | 0.180x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.0534 ms | not_measured | managed | 0.135x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 7.9643 ms | not_measured | managed | 1.000x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 0.145x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.1056 ms | not_measured | managed | 0.140x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 19.9740 ms | not_measured | managed | 1.422x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.0030 ms | not_measured | managed | 0.400x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.0354 ms | not_measured | managed | 0.746x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 5.2750 ms | not_measured | managed | 0.891x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.0035 ms | not_measured | managed | 0.314x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.0515 ms | not_measured | managed | 0.524x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 7.9720 ms | not_measured | managed | 0.974x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.0060 ms | not_measured | managed | 0.167x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.1027 ms | not_measured | managed | 0.274x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 28.2127 ms | not_measured | managed | 0.543x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.0039 ms | not_measured | managed | 0.256x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.0242 ms | not_measured | managed | 1.219x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 2.2194 ms | not_measured | managed | 1.613x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.0029 ms | not_measured | managed | 0.414x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.0342 ms | not_measured | managed | 0.772x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 5.1154 ms | not_measured | managed | 0.960x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.0064 ms | not_measured | managed | 0.188x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.0529 ms | not_measured | managed | 0.463x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 8.0030 ms | not_measured | managed | 0.975x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.0059 ms | not_measured | managed | 0.169x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.1124 ms | not_measured | managed | 0.291x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 29.7935 ms | not_measured | managed | 0.510x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.0055 ms | not_measured | managed | 0.182x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.0226 ms | not_measured | managed | 1.226x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 2.1942 ms | not_measured | managed | 1.643x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.0046 ms | not_measured | managed | 0.239x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.3457 ms | not_measured | managed | 0.910x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 58.4046 ms | not_measured | managed | 1.147x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.0095 ms | not_measured | managed | 0.589x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.9057 ms | not_measured | managed | 0.348x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 93.2564 ms | not_measured | managed | 0.328x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.304x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.0519 ms | not_measured | managed | 0.152x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 10.7868 ms | not_measured | managed | 0.802x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.1127 ms | not_measured | managed | 0.256x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 25.9321 ms | not_measured | managed | 1.348x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.0024 ms | not_measured | managed | 0.458x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.0292 ms | not_measured | managed | 1.096x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 6.2443 ms | not_measured | managed | 1.453x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.0025 ms | not_measured | managed | 0.360x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.0573 ms | not_measured | managed | 0.682x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 11.2771 ms | not_measured | managed | 0.773x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.303x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.1296 ms | not_measured | managed | 0.333x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 35.7110 ms | not_measured | managed | 0.480x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.0181 ms | not_measured | managed | 1.751x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 2.8210 ms | not_measured | managed | 1.393x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.524x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.0305 ms | not_measured | managed | 1.072x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.2306 ms | not_measured | managed | 0.863x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.500x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.0555 ms | not_measured | managed | 0.616x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.1087 ms | not_measured | managed | 0.777x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.0031 ms | not_measured | managed | 0.323x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.1127 ms | not_measured | managed | 0.337x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 36.1716 ms | not_measured | managed | 0.472x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.0163 ms | not_measured | managed | 1.890x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 2.9192 ms | not_measured | managed | 1.372x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.0042 ms | not_measured | managed | 0.333x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.2989 ms | not_measured | managed | 1.002x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 42.0749 ms | not_measured | managed | 1.236x |
| `a + scalar (float16)` | float16 | 1,000 | 0.0096 ms | not_measured | managed | 0.688x |
| `a + scalar (float16)` | float16 | 100,000 | 0.8884 ms | not_measured | managed | 0.355x |
| `a + scalar (float16)` | float16 | 10,000,000 | 89.5701 ms | not_measured | managed | 0.340x |
| `a + scalar (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a + scalar (float32)` | float32 | 100,000 | 0.0526 ms | not_measured | managed | 0.127x |
| `a + scalar (float32)` | float32 | 10,000,000 | 7.8354 ms | not_measured | managed | 1.033x |
| `a + scalar (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.259x |
| `a + scalar (float64)` | float64 | 100,000 | 0.1015 ms | not_measured | managed | 0.136x |
| `a + scalar (float64)` | float64 | 10,000,000 | 26.1723 ms | not_measured | managed | 1.047x |
| `a + scalar (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.526x |
| `a + scalar (int16)` | int16 | 100,000 | 0.0270 ms | not_measured | managed | 1.052x |
| `a + scalar (int16)` | int16 | 10,000,000 | 5.1082 ms | not_measured | managed | 1.049x |
| `a + scalar (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.476x |
| `a + scalar (int32)` | int32 | 100,000 | 0.0497 ms | not_measured | managed | 0.616x |
| `a + scalar (int32)` | int32 | 10,000,000 | 10.3550 ms | not_measured | managed | 0.771x |
| `a + scalar (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.321x |
| `a + scalar (int64)` | int64 | 100,000 | 0.1008 ms | not_measured | managed | 0.292x |
| `a + scalar (int64)` | int64 | 10,000,000 | 29.8820 ms | not_measured | managed | 0.509x |
| `a + scalar (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `a + scalar (int8)` | int8 | 100,000 | 0.0141 ms | not_measured | managed | 1.858x |
| `a + scalar (int8)` | int8 | 10,000,000 | 1.9899 ms | not_measured | managed | 1.911x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.526x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.0265 ms | not_measured | managed | 1.053x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 5.2876 ms | not_measured | managed | 0.975x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.0501 ms | not_measured | managed | 0.513x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 10.0826 ms | not_measured | managed | 0.776x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.345x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.1006 ms | not_measured | managed | 0.262x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 30.0565 ms | not_measured | managed | 0.550x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.0149 ms | not_measured | managed | 1.879x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 2.3236 ms | not_measured | managed | 1.542x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.0042 ms | not_measured | managed | 0.214x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.3508 ms | not_measured | managed | 0.871x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 58.0571 ms | not_measured | managed | 1.279x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.0096 ms | not_measured | managed | 0.479x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.9031 ms | not_measured | managed | 0.349x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 90.4985 ms | not_measured | managed | 0.339x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.0522 ms | not_measured | managed | 0.167x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 11.2168 ms | not_measured | managed | 0.755x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.200x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.1074 ms | not_measured | managed | 0.264x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 26.0328 ms | not_measured | managed | 1.341x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.0017 ms | not_measured | managed | 0.529x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.0276 ms | not_measured | managed | 1.149x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 6.0515 ms | not_measured | managed | 0.872x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.0518 ms | not_measured | managed | 0.716x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 10.8129 ms | not_measured | managed | 0.800x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.1075 ms | not_measured | managed | 0.340x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 25.9102 ms | not_measured | managed | 0.666x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.0158 ms | not_measured | managed | 1.943x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 2.6224 ms | not_measured | managed | 1.493x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.0017 ms | not_measured | managed | 0.529x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.0273 ms | not_measured | managed | 1.220x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 5.9662 ms | not_measured | managed | 0.937x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.0516 ms | not_measured | managed | 0.669x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.2764 ms | not_measured | managed | 0.773x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.1061 ms | not_measured | managed | 0.352x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 25.9694 ms | not_measured | managed | 0.656x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.0156 ms | not_measured | managed | 2.026x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 2.6057 ms | not_measured | managed | 1.539x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.0037 ms | not_measured | managed | 0.243x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.3156 ms | not_measured | managed | 0.972x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 41.9335 ms | not_measured | managed | 1.292x |
| `a - scalar (float16)` | float16 | 1,000 | 0.0096 ms | not_measured | managed | 0.635x |
| `a - scalar (float16)` | float16 | 100,000 | 0.9049 ms | not_measured | managed | 0.343x |
| `a - scalar (float16)` | float16 | 10,000,000 | 89.3447 ms | not_measured | managed | 0.344x |
| `a - scalar (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `a - scalar (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.133x |
| `a - scalar (float32)` | float32 | 10,000,000 | 9.0930 ms | not_measured | managed | 0.875x |
| `a - scalar (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.241x |
| `a - scalar (float64)` | float64 | 100,000 | 0.1036 ms | not_measured | managed | 0.127x |
| `a - scalar (float64)` | float64 | 10,000,000 | 19.5380 ms | not_measured | managed | 1.409x |
| `a - scalar (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.579x |
| `a - scalar (int16)` | int16 | 100,000 | 0.0269 ms | not_measured | managed | 0.993x |
| `a - scalar (int16)` | int16 | 10,000,000 | 4.2894 ms | not_measured | managed | 1.091x |
| `a - scalar (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `a - scalar (int32)` | int32 | 100,000 | 0.0511 ms | not_measured | managed | 0.577x |
| `a - scalar (int32)` | int32 | 10,000,000 | 9.1958 ms | not_measured | managed | 0.847x |
| `a - scalar (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.300x |
| `a - scalar (int64)` | int64 | 100,000 | 0.1042 ms | not_measured | managed | 0.254x |
| `a - scalar (int64)` | int64 | 10,000,000 | 19.6435 ms | not_measured | managed | 0.790x |
| `a - scalar (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a - scalar (int8)` | int8 | 100,000 | 0.0154 ms | not_measured | managed | 1.890x |
| `a - scalar (int8)` | int8 | 10,000,000 | 1.9590 ms | not_measured | managed | 1.734x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.579x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.0270 ms | not_measured | managed | 1.011x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 4.4785 ms | not_measured | managed | 1.159x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.0511 ms | not_measured | managed | 0.548x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 9.0546 ms | not_measured | managed | 0.872x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.300x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.1034 ms | not_measured | managed | 0.263x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 19.4358 ms | not_measured | managed | 0.787x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.0150 ms | not_measured | managed | 1.793x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 1.9575 ms | not_measured | managed | 1.768x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.0643 ms | not_measured | managed | 0.204x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 11.3991 ms | not_measured | managed | 0.753x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.300x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.1977 ms | not_measured | managed | 0.200x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 27.1703 ms | not_measured | managed | 1.298x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.0045 ms | not_measured | managed | 0.467x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.2047 ms | not_measured | managed | 0.491x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 24.4591 ms | not_measured | managed | 0.826x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.0056 ms | not_measured | managed | 0.375x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.1970 ms | not_measured | managed | 0.433x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 29.9473 ms | not_measured | managed | 0.798x |
| `a / scalar (float32)` | float32 | 1,000 | 0.0025 ms | not_measured | managed | 0.280x |
| `a / scalar (float32)` | float32 | 100,000 | 0.0641 ms | not_measured | managed | 0.228x |
| `a / scalar (float32)` | float32 | 10,000,000 | 9.1445 ms | not_measured | managed | 0.878x |
| `a / scalar (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.370x |
| `a / scalar (float64)` | float64 | 100,000 | 0.1925 ms | not_measured | managed | 0.230x |
| `a / scalar (float64)` | float64 | 10,000,000 | 23.5382 ms | not_measured | managed | 1.590x |
| `a / scalar (int32)` | int32 | 1,000 | 0.0067 ms | not_measured | managed | 0.343x |
| `a / scalar (int32)` | int32 | 100,000 | 0.1977 ms | not_measured | managed | 0.336x |
| `a / scalar (int32)` | int32 | 10,000,000 | 23.4103 ms | not_measured | managed | 0.716x |
| `a / scalar (int64)` | int64 | 1,000 | 0.0059 ms | not_measured | managed | 0.644x |
| `a / scalar (int64)` | int64 | 100,000 | 0.1957 ms | not_measured | managed | 0.349x |
| `a / scalar (int64)` | int64 | 10,000,000 | 24.8160 ms | not_measured | managed | 0.757x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.0044 ms | not_measured | managed | 0.205x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.3449 ms | not_measured | managed | 0.881x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 58.2536 ms | not_measured | managed | 1.296x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.0095 ms | not_measured | managed | 0.653x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.8954 ms | not_measured | managed | 0.355x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 90.0937 ms | not_measured | managed | 0.342x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.166x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 10.8352 ms | not_measured | managed | 0.788x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.1100 ms | not_measured | managed | 0.289x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 25.7172 ms | not_measured | managed | 1.442x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.0398 ms | not_measured | managed | 0.917x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 6.3948 ms | not_measured | managed | 1.594x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.0553 ms | not_measured | managed | 0.687x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 10.9208 ms | not_measured | managed | 0.837x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.345x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.1143 ms | not_measured | managed | 0.338x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 34.7952 ms | not_measured | managed | 0.485x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.0177 ms | not_measured | managed | 1.785x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 2.8566 ms | not_measured | managed | 1.395x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.0288 ms | not_measured | managed | 1.170x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 6.1441 ms | not_measured | managed | 0.894x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.0563 ms | not_measured | managed | 0.544x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 11.1112 ms | not_measured | managed | 0.777x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.1106 ms | not_measured | managed | 0.336x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 36.0699 ms | not_measured | managed | 0.470x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.0162 ms | not_measured | managed | 2.062x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 2.9138 ms | not_measured | managed | 1.418x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.0653 ms | not_measured | managed | 0.205x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 11.2537 ms | not_measured | managed | 0.755x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.275x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.1957 ms | not_measured | managed | 0.215x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 35.5752 ms | not_measured | managed | 1.149x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.375x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.2027 ms | not_measured | managed | 0.451x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 24.5236 ms | not_measured | managed | 0.823x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.0062 ms | not_measured | managed | 0.339x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.2019 ms | not_measured | managed | 0.409x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 41.2452 ms | not_measured | managed | 0.579x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.0120 ms | not_measured | managed | 1.025x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 1.5615 ms | not_measured | managed | 1.001x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 157.6573 ms | not_measured | managed | 0.951x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.0100 ms | not_measured | managed | 1.030x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 1.4513 ms | not_measured | managed | 0.891x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 153.4305 ms | not_measured | managed | 1.684x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.0035 ms | not_measured | managed | 0.686x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.6245 ms | not_measured | managed | 0.648x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 62.5407 ms | not_measured | managed | 0.688x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.0037 ms | not_measured | managed | 0.676x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.6244 ms | not_measured | managed | 0.639x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 78.8368 ms | not_measured | managed | 0.600x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.0122 ms | not_measured | managed | 1.025x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 1.5494 ms | not_measured | managed | 0.992x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 163.5778 ms | not_measured | managed | 0.921x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 1.000x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 1.4305 ms | not_measured | managed | 0.904x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 147.5571 ms | not_measured | managed | 1.580x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.0035 ms | not_measured | managed | 0.600x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.5940 ms | not_measured | managed | 0.665x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 59.9227 ms | not_measured | managed | 0.708x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.0037 ms | not_measured | managed | 1.000x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.6054 ms | not_measured | managed | 0.649x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 80.4375 ms | not_measured | managed | 0.592x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.0032 ms | not_measured | managed | 0.250x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.3432 ms | not_measured | managed | 0.856x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 56.7251 ms | not_measured | managed | 1.288x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.388x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.8983 ms | not_measured | managed | 0.347x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 89.6936 ms | not_measured | managed | 0.339x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.0521 ms | not_measured | managed | 0.148x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 10.9481 ms | not_measured | managed | 0.775x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.316x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.1105 ms | not_measured | managed | 0.271x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 25.1430 ms | not_measured | managed | 1.371x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.0279 ms | not_measured | managed | 1.065x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 6.1057 ms | not_measured | managed | 0.915x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.0532 ms | not_measured | managed | 0.590x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 11.2237 ms | not_measured | managed | 0.778x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.1213 ms | not_measured | managed | 0.288x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 34.4814 ms | not_measured | managed | 0.550x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.0164 ms | not_measured | managed | 1.805x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 3.1428 ms | not_measured | managed | 1.199x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.0283 ms | not_measured | managed | 1.078x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 6.1445 ms | not_measured | managed | 0.817x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.0536 ms | not_measured | managed | 0.599x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 11.0852 ms | not_measured | managed | 0.787x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.1179 ms | not_measured | managed | 0.293x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 27.6827 ms | not_measured | managed | 0.615x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.0163 ms | not_measured | managed | 1.877x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 3.0914 ms | not_measured | managed | 1.240x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.0040 ms | not_measured | managed | 0.225x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.3447 ms | not_measured | managed | 0.859x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 57.3264 ms | not_measured | managed | 1.387x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.0094 ms | not_measured | managed | 0.489x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.8981 ms | not_measured | managed | 0.348x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 88.7192 ms | not_measured | managed | 0.354x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.0507 ms | not_measured | managed | 0.176x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 10.9446 ms | not_measured | managed | 0.786x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.200x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.1074 ms | not_measured | managed | 0.262x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 25.9451 ms | not_measured | managed | 1.487x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.0269 ms | not_measured | managed | 1.297x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 6.1620 ms | not_measured | managed | 0.930x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.0517 ms | not_measured | managed | 0.768x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 11.4189 ms | not_measured | managed | 0.775x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.1052 ms | not_measured | managed | 0.362x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 25.9726 ms | not_measured | managed | 0.658x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.0154 ms | not_measured | managed | 2.058x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 2.5956 ms | not_measured | managed | 1.526x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.0275 ms | not_measured | managed | 1.284x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 5.8956 ms | not_measured | managed | 0.926x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.0515 ms | not_measured | managed | 0.728x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 11.0977 ms | not_measured | managed | 0.773x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.1048 ms | not_measured | managed | 0.347x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 26.0540 ms | not_measured | managed | 0.659x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.0159 ms | not_measured | managed | 1.969x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 2.6240 ms | not_measured | managed | 1.499x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.250x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.0653 ms | not_measured | managed | 0.194x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 11.4353 ms | not_measured | managed | 0.738x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.220x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.1959 ms | not_measured | managed | 0.209x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 36.0341 ms | not_measured | managed | 1.052x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.0060 ms | not_measured | managed | 0.350x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.1980 ms | not_measured | managed | 0.461x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 25.1650 ms | not_measured | managed | 0.810x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.0062 ms | not_measured | managed | 0.323x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.2004 ms | not_measured | managed | 0.422x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 40.9624 ms | not_measured | managed | 0.575x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.0038 ms | not_measured | managed | 0.263x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.3181 ms | not_measured | managed | 0.903x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 41.6641 ms | not_measured | managed | 1.222x |
| `scalar - a (float16)` | float16 | 1,000 | 0.0093 ms | not_measured | managed | 0.591x |
| `scalar - a (float16)` | float16 | 100,000 | 0.9026 ms | not_measured | managed | 0.343x |
| `scalar - a (float16)` | float16 | 10,000,000 | 94.0145 ms | not_measured | managed | 0.326x |
| `scalar - a (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `scalar - a (float32)` | float32 | 100,000 | 0.0506 ms | not_measured | managed | 0.132x |
| `scalar - a (float32)` | float32 | 10,000,000 | 9.0549 ms | not_measured | managed | 0.885x |
| `scalar - a (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `scalar - a (float64)` | float64 | 100,000 | 0.1040 ms | not_measured | managed | 0.126x |
| `scalar - a (float64)` | float64 | 10,000,000 | 19.5370 ms | not_measured | managed | 1.372x |
| `scalar - a (int16)` | int16 | 1,000 | 0.0015 ms | not_measured | managed | 0.733x |
| `scalar - a (int16)` | int16 | 100,000 | 0.0273 ms | not_measured | managed | 1.004x |
| `scalar - a (int16)` | int16 | 10,000,000 | 4.2795 ms | not_measured | managed | 1.101x |
| `scalar - a (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `scalar - a (int32)` | int32 | 100,000 | 0.0509 ms | not_measured | managed | 0.621x |
| `scalar - a (int32)` | int32 | 10,000,000 | 9.1653 ms | not_measured | managed | 0.861x |
| `scalar - a (int64)` | int64 | 1,000 | 0.0027 ms | not_measured | managed | 0.333x |
| `scalar - a (int64)` | int64 | 100,000 | 0.1033 ms | not_measured | managed | 0.331x |
| `scalar - a (int64)` | int64 | 10,000,000 | 19.5781 ms | not_measured | managed | 0.777x |
| `scalar - a (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `scalar - a (int8)` | int8 | 100,000 | 0.0155 ms | not_measured | managed | 1.768x |
| `scalar - a (int8)` | int8 | 10,000,000 | 1.9847 ms | not_measured | managed | 1.688x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.579x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.0274 ms | not_measured | managed | 1.230x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 4.5772 ms | not_measured | managed | 1.043x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.0509 ms | not_measured | managed | 0.540x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 9.1636 ms | not_measured | managed | 0.855x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.400x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.1034 ms | not_measured | managed | 0.254x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 19.3762 ms | not_measured | managed | 0.788x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.0159 ms | not_measured | managed | 1.642x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 1.9500 ms | not_measured | managed | 1.708x |
| `scalar / a (float32)` | float32 | 1,000 | 0.0025 ms | not_measured | managed | 0.280x |
| `scalar / a (float32)` | float32 | 100,000 | 0.0632 ms | not_measured | managed | 0.210x |
| `scalar / a (float32)` | float32 | 10,000,000 | 9.3504 ms | not_measured | managed | 0.852x |
| `scalar / a (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.357x |
| `scalar / a (float64)` | float64 | 100,000 | 0.1899 ms | not_measured | managed | 0.217x |
| `scalar / a (float64)` | float64 | 10,000,000 | 23.8872 ms | not_measured | managed | 1.370x |
| `scalar / a (int32)` | int32 | 1,000 | 0.0049 ms | not_measured | managed | 0.388x |
| `scalar / a (int32)` | int32 | 100,000 | 0.1974 ms | not_measured | managed | 0.380x |
| `scalar / a (int32)` | int32 | 10,000,000 | 23.2313 ms | not_measured | managed | 0.722x |
| `scalar / a (int64)` | int64 | 1,000 | 0.0063 ms | not_measured | managed | 0.365x |
| `scalar / a (int64)` | int64 | 100,000 | 0.2007 ms | not_measured | managed | 0.553x |
| `scalar / a (int64)` | int64 | 10,000,000 | 24.4431 ms | not_measured | managed | 0.767x |
| `a & b (bool)` | bool | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a & b (bool)` | bool | 100,000 | 0.0528 ms | not_measured | managed | 0.055x |
| `a & b (bool)` | bool | 10,000,000 | 5.8949 ms | not_measured | managed | 0.361x |
| `a & b (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a & b (int16)` | int16 | 100,000 | 0.0261 ms | not_measured | managed | 1.153x |
| `a & b (int16)` | int16 | 10,000,000 | 5.9060 ms | not_measured | managed | 1.535x |
| `a & b (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a & b (int32)` | int32 | 100,000 | 0.0502 ms | not_measured | managed | 0.715x |
| `a & b (int32)` | int32 | 10,000,000 | 11.5441 ms | not_measured | managed | 1.450x |
| `a & b (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a & b (int64)` | int64 | 100,000 | 0.1087 ms | not_measured | managed | 0.345x |
| `a & b (int64)` | int64 | 10,000,000 | 26.7589 ms | not_measured | managed | 1.263x |
| `a & b (int8)` | int8 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `a & b (int8)` | int8 | 100,000 | 0.0140 ms | not_measured | managed | 2.057x |
| `a & b (int8)` | int8 | 10,000,000 | 2.5273 ms | not_measured | managed | 2.566x |
| `a & b (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a & b (uint16)` | uint16 | 100,000 | 0.0261 ms | not_measured | managed | 1.103x |
| `a & b (uint16)` | uint16 | 10,000,000 | 5.8288 ms | not_measured | managed | 1.582x |
| `a & b (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a & b (uint32)` | uint32 | 100,000 | 0.0528 ms | not_measured | managed | 0.545x |
| `a & b (uint32)` | uint32 | 10,000,000 | 10.7802 ms | not_measured | managed | 1.569x |
| `a & b (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a & b (uint64)` | uint64 | 100,000 | 0.0990 ms | not_measured | managed | 0.380x |
| `a & b (uint64)` | uint64 | 10,000,000 | 25.8685 ms | not_measured | managed | 1.295x |
| `a & b (uint8)` | uint8 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `a & b (uint8)` | uint8 | 100,000 | 0.0136 ms | not_measured | managed | 2.096x |
| `a & b (uint8)` | uint8 | 10,000,000 | 2.5781 ms | not_measured | managed | 2.598x |
| `a ^ b (bool)` | bool | 1,000 | 0.0024 ms | not_measured | managed | 0.167x |
| `a ^ b (bool)` | bool | 100,000 | 0.0492 ms | not_measured | managed | 0.061x |
| `a ^ b (bool)` | bool | 10,000,000 | 5.2862 ms | not_measured | managed | 0.394x |
| `a ^ b (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `a ^ b (int16)` | int16 | 100,000 | 0.0262 ms | not_measured | managed | 1.103x |
| `a ^ b (int16)` | int16 | 10,000,000 | 5.9158 ms | not_measured | managed | 1.542x |
| `a ^ b (int32)` | int32 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a ^ b (int32)` | int32 | 100,000 | 0.0504 ms | not_measured | managed | 0.565x |
| `a ^ b (int32)` | int32 | 10,000,000 | 11.3236 ms | not_measured | managed | 1.444x |
| `a ^ b (int64)` | int64 | 1,000 | 0.0035 ms | not_measured | managed | 0.229x |
| `a ^ b (int64)` | int64 | 100,000 | 0.0974 ms | not_measured | managed | 0.369x |
| `a ^ b (int64)` | int64 | 10,000,000 | 25.7343 ms | not_measured | managed | 1.250x |
| `a ^ b (int8)` | int8 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `a ^ b (int8)` | int8 | 100,000 | 0.0144 ms | not_measured | managed | 2.000x |
| `a ^ b (int8)` | int8 | 10,000,000 | 2.5803 ms | not_measured | managed | 2.561x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.0266 ms | not_measured | managed | 1.109x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 5.8233 ms | not_measured | managed | 1.583x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.0511 ms | not_measured | managed | 0.562x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 11.2066 ms | not_measured | managed | 1.501x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.0988 ms | not_measured | managed | 0.315x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 25.6681 ms | not_measured | managed | 1.314x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.0138 ms | not_measured | managed | 2.174x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 2.7696 ms | not_measured | managed | 2.484x |
| `a | b (bool)` | bool | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `a | b (bool)` | bool | 100,000 | 0.0512 ms | not_measured | managed | 0.057x |
| `a | b (bool)` | bool | 10,000,000 | 5.3706 ms | not_measured | managed | 0.403x |
| `a | b (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a | b (int16)` | int16 | 100,000 | 0.0265 ms | not_measured | managed | 1.083x |
| `a | b (int16)` | int16 | 10,000,000 | 5.7106 ms | not_measured | managed | 1.567x |
| `a | b (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `a | b (int32)` | int32 | 100,000 | 0.0525 ms | not_measured | managed | 0.543x |
| `a | b (int32)` | int32 | 10,000,000 | 11.2423 ms | not_measured | managed | 1.486x |
| `a | b (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `a | b (int64)` | int64 | 100,000 | 0.1344 ms | not_measured | managed | 0.265x |
| `a | b (int64)` | int64 | 10,000,000 | 26.1583 ms | not_measured | managed | 1.229x |
| `a | b (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a | b (int8)` | int8 | 100,000 | 0.0136 ms | not_measured | managed | 2.228x |
| `a | b (int8)` | int8 | 10,000,000 | 2.5167 ms | not_measured | managed | 2.661x |
| `a | b (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a | b (uint16)` | uint16 | 100,000 | 0.0259 ms | not_measured | managed | 1.108x |
| `a | b (uint16)` | uint16 | 10,000,000 | 5.8316 ms | not_measured | managed | 1.553x |
| `a | b (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a | b (uint32)` | uint32 | 100,000 | 0.0525 ms | not_measured | managed | 0.545x |
| `a | b (uint32)` | uint32 | 10,000,000 | 11.4706 ms | not_measured | managed | 1.410x |
| `a | b (uint64)` | uint64 | 1,000 | 0.0033 ms | not_measured | managed | 0.242x |
| `a | b (uint64)` | uint64 | 100,000 | 0.1033 ms | not_measured | managed | 0.324x |
| `a | b (uint64)` | uint64 | 10,000,000 | 25.5691 ms | not_measured | managed | 1.236x |
| `a | b (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `a | b (uint8)` | uint8 | 100,000 | 0.0143 ms | not_measured | managed | 2.084x |
| `a | b (uint8)` | uint8 | 10,000,000 | 2.7444 ms | not_measured | managed | 2.500x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.0528 ms | not_measured | managed | 0.062x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 5.2754 ms | not_measured | managed | 0.430x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.0264 ms | not_measured | managed | 1.121x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 6.0351 ms | not_measured | managed | 1.544x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.0508 ms | not_measured | managed | 0.596x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 11.1018 ms | not_measured | managed | 1.644x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.1035 ms | not_measured | managed | 0.318x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 25.4866 ms | not_measured | managed | 1.302x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 0.467x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.0130 ms | not_measured | managed | 2.269x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 2.7840 ms | not_measured | managed | 2.332x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.0258 ms | not_measured | managed | 1.120x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 5.8885 ms | not_measured | managed | 1.561x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.0497 ms | not_measured | managed | 0.598x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 11.4946 ms | not_measured | managed | 1.470x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.0033 ms | not_measured | managed | 0.242x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.0991 ms | not_measured | managed | 0.308x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 25.5002 ms | not_measured | managed | 1.274x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.0142 ms | not_measured | managed | 2.007x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 2.8242 ms | not_measured | managed | 2.391x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.0570 ms | not_measured | managed | 0.046x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 5.1119 ms | not_measured | managed | 0.567x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.0276 ms | not_measured | managed | 0.964x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 4.4139 ms | not_measured | managed | 1.598x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 1.300x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.0495 ms | not_measured | managed | 0.525x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 9.4836 ms | not_measured | managed | 1.800x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.250x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.1043 ms | not_measured | managed | 0.256x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 19.1018 ms | not_measured | managed | 1.330x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.0138 ms | not_measured | managed | 1.870x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 1.9391 ms | not_measured | managed | 2.785x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.0259 ms | not_measured | managed | 0.996x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 4.8434 ms | not_measured | managed | 1.461x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.0487 ms | not_measured | managed | 0.585x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 8.6892 ms | not_measured | managed | 1.489x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.0032 ms | not_measured | managed | 0.219x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.0970 ms | not_measured | managed | 0.274x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 19.4933 ms | not_measured | managed | 1.306x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.0146 ms | not_measured | managed | 1.795x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 1.9209 ms | not_measured | managed | 2.867x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.0516 ms | not_measured | managed | 0.058x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 5.1157 ms | not_measured | managed | 0.791x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.0262 ms | not_measured | managed | 1.095x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 5.6839 ms | not_measured | managed | 1.572x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.0502 ms | not_measured | managed | 0.588x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 11.4320 ms | not_measured | managed | 2.106x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.0036 ms | not_measured | managed | 0.222x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.1037 ms | not_measured | managed | 0.320x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 26.0127 ms | not_measured | managed | 1.194x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.0137 ms | not_measured | managed | 2.131x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 2.7381 ms | not_measured | managed | 2.287x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.0257 ms | not_measured | managed | 1.121x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 5.9806 ms | not_measured | managed | 1.484x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.0496 ms | not_measured | managed | 0.631x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 11.3180 ms | not_measured | managed | 1.472x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.0997 ms | not_measured | managed | 0.313x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 25.5696 ms | not_measured | managed | 1.271x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.0019 ms | not_measured | managed | 0.368x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.0140 ms | not_measured | managed | 2.143x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 3.1264 ms | not_measured | managed | 2.151x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.0522 ms | not_measured | managed | 0.057x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 5.0756 ms | not_measured | managed | 0.851x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.0257 ms | not_measured | managed | 1.233x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 5.8851 ms | not_measured | managed | 1.582x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.0517 ms | not_measured | managed | 0.582x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 11.0355 ms | not_measured | managed | 2.309x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.0026 ms | not_measured | managed | 0.308x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.0971 ms | not_measured | managed | 0.344x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 25.7072 ms | not_measured | managed | 1.270x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.0019 ms | not_measured | managed | 0.368x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.0136 ms | not_measured | managed | 2.110x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 2.7897 ms | not_measured | managed | 2.243x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.0257 ms | not_measured | managed | 1.132x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 6.3395 ms | not_measured | managed | 1.457x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.0491 ms | not_measured | managed | 0.629x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 11.0211 ms | not_measured | managed | 1.469x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.0984 ms | not_measured | managed | 0.315x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 26.8352 ms | not_measured | managed | 1.201x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.0137 ms | not_measured | managed | 2.175x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 2.6159 ms | not_measured | managed | 2.491x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.0025 ms | not_measured | managed | 0.160x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.0498 ms | not_measured | managed | 0.052x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 5.1907 ms | not_measured | managed | 0.374x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.0257 ms | not_measured | managed | 1.004x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 4.2485 ms | not_measured | managed | 1.587x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.0591 ms | not_measured | managed | 0.440x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 8.6419 ms | not_measured | managed | 1.520x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.212x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.1026 ms | not_measured | managed | 0.264x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 19.9562 ms | not_measured | managed | 1.220x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 0.400x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.0163 ms | not_measured | managed | 1.620x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 2.0247 ms | not_measured | managed | 2.725x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.0253 ms | not_measured | managed | 1.028x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 5.2406 ms | not_measured | managed | 1.337x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.0497 ms | not_measured | managed | 0.523x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 7.8396 ms | not_measured | managed | 1.652x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.0950 ms | not_measured | managed | 0.277x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 19.2084 ms | not_measured | managed | 1.309x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.0135 ms | not_measured | managed | 1.904x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 1.9799 ms | not_measured | managed | 2.832x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.0081 ms | not_measured | managed | 0.185x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.1254 ms | not_measured | managed | 0.341x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 14.8281 ms | not_measured | managed | 1.087x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.0036 ms | not_measured | managed | 0.306x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.0314 ms | not_measured | managed | 1.035x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.3602 ms | not_measured | managed | 1.662x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.0039 ms | not_measured | managed | 0.256x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.0557 ms | not_measured | managed | 0.346x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.7192 ms | not_measured | managed | 1.507x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.0046 ms | not_measured | managed | 0.196x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.0958 ms | not_measured | managed | 0.210x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 20.3719 ms | not_measured | managed | 1.234x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.0151 ms | not_measured | managed | 1.874x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 2.1366 ms | not_measured | managed | 2.604x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0041 ms | not_measured | managed | 0.244x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0266 ms | not_measured | managed | 1.117x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.5757 ms | not_measured | managed | 1.635x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0043 ms | not_measured | managed | 0.233x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0508 ms | not_measured | managed | 0.421x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 7.9027 ms | not_measured | managed | 1.747x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0051 ms | not_measured | managed | 0.196x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.0999 ms | not_measured | managed | 0.191x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 19.3671 ms | not_measured | managed | 1.326x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0038 ms | not_measured | managed | 0.237x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0158 ms | not_measured | managed | 1.778x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.9449 ms | not_measured | managed | 3.013x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.0088 ms | not_measured | managed | 0.182x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.1159 ms | not_measured | managed | 0.457x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | 14.9298 ms | not_measured | managed | 1.138x |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.0034 ms | not_measured | managed | 0.353x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.0267 ms | not_measured | managed | 1.423x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.1755 ms | not_measured | managed | 2.180x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.0036 ms | not_measured | managed | 0.306x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.0560 ms | not_measured | managed | 0.539x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 9.4687 ms | not_measured | managed | 1.411x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.0046 ms | not_measured | managed | 0.304x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.1000 ms | not_measured | managed | 0.290x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 21.1537 ms | not_measured | managed | 1.156x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.0041 ms | not_measured | managed | 0.244x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.0188 ms | not_measured | managed | 2.032x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 2.3973 ms | not_measured | managed | 3.657x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0043 ms | not_measured | managed | 0.256x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0264 ms | not_measured | managed | 1.076x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.5134 ms | not_measured | managed | 1.635x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0040 ms | not_measured | managed | 0.250x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0507 ms | not_measured | managed | 0.394x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 9.2777 ms | not_measured | managed | 1.398x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0044 ms | not_measured | managed | 0.227x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.0982 ms | not_measured | managed | 0.195x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 19.5102 ms | not_measured | managed | 1.352x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0155 ms | not_measured | managed | 1.884x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 1.9603 ms | not_measured | managed | 3.201x |
| `matrix * 2.0` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.212x |
| `matrix * 2.0` | float64 | 100,000 | 0.0820 ms | not_measured | managed | 0.167x |
| `matrix * 2.0` | float64 | 10,000,000 | 20.4812 ms | not_measured | managed | 0.817x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.236x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.1032 ms | not_measured | managed | 0.277x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 21.1808 ms | not_measured | managed | 1.249x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 0.250x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.1008 ms | not_measured | managed | 0.258x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 21.0518 ms | not_measured | managed | 1.297x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.237x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.1032 ms | not_measured | managed | 0.291x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 20.6669 ms | not_measured | managed | 1.283x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.0044 ms | not_measured | managed | 0.273x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.1121 ms | not_measured | managed | 0.260x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 20.4765 ms | not_measured | managed | 0.947x |
| `matrix + scalar` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `matrix + scalar` | float64 | 100,000 | 0.0780 ms | not_measured | managed | 0.146x |
| `matrix + scalar` | float64 | 10,000,000 | 19.9475 ms | not_measured | managed | 0.823x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 6.636x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.583x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 4.700x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.234x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.1414 ms | not_measured | managed | 0.178x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 20.8609 ms | not_measured | managed | 1.223x |
| `a != b (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `a != b (float32)` | float32 | 100,000 | 0.0165 ms | not_measured | managed | 0.861x |
| `a != b (float32)` | float32 | 10,000,000 | 9.1763 ms | not_measured | managed | 1.113x |
| `a != b (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a != b (float64)` | float64 | 100,000 | 0.0254 ms | not_measured | managed | 0.421x |
| `a != b (float64)` | float64 | 10,000,000 | 15.9548 ms | not_measured | managed | 1.165x |
| `a != b (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `a != b (int32)` | int32 | 100,000 | 0.0177 ms | not_measured | managed | 0.475x |
| `a != b (int32)` | int32 | 10,000,000 | 8.8670 ms | not_measured | managed | 0.520x |
| `a != b (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `a != b (int64)` | int64 | 100,000 | 0.0297 ms | not_measured | managed | 0.508x |
| `a != b (int64)` | int64 | 10,000,000 | 16.3374 ms | not_measured | managed | 1.191x |
| `a < b (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `a < b (float32)` | float32 | 100,000 | 0.0166 ms | not_measured | managed | 0.560x |
| `a < b (float32)` | float32 | 10,000,000 | 10.1088 ms | not_measured | managed | 1.048x |
| `a < b (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `a < b (float64)` | float64 | 100,000 | 0.0278 ms | not_measured | managed | 0.385x |
| `a < b (float64)` | float64 | 10,000,000 | 16.9025 ms | not_measured | managed | 1.097x |
| `a < b (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `a < b (int32)` | int32 | 100,000 | 0.0179 ms | not_measured | managed | 0.391x |
| `a < b (int32)` | int32 | 10,000,000 | 9.2041 ms | not_measured | managed | 0.484x |
| `a < b (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `a < b (int64)` | int64 | 100,000 | 0.0276 ms | not_measured | managed | 0.663x |
| `a < b (int64)` | int64 | 10,000,000 | 15.5211 ms | not_measured | managed | 1.240x |
| `a <= b (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `a <= b (float32)` | float32 | 100,000 | 0.0167 ms | not_measured | managed | 0.581x |
| `a <= b (float32)` | float32 | 10,000,000 | 10.6303 ms | not_measured | managed | 0.995x |
| `a <= b (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a <= b (float64)` | float64 | 100,000 | 0.0262 ms | not_measured | managed | 0.435x |
| `a <= b (float64)` | float64 | 10,000,000 | 16.9999 ms | not_measured | managed | 1.555x |
| `a <= b (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `a <= b (int32)` | int32 | 100,000 | 0.0185 ms | not_measured | managed | 0.378x |
| `a <= b (int32)` | int32 | 10,000,000 | 11.0013 ms | not_measured | managed | 0.424x |
| `a <= b (int64)` | int64 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `a <= b (int64)` | int64 | 100,000 | 0.0302 ms | not_measured | managed | 0.616x |
| `a <= b (int64)` | int64 | 10,000,000 | 15.6130 ms | not_measured | managed | 1.267x |
| `a == b (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `a == b (float32)` | float32 | 100,000 | 0.0169 ms | not_measured | managed | 0.751x |
| `a == b (float32)` | float32 | 10,000,000 | 8.9514 ms | not_measured | managed | 1.157x |
| `a == b (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a == b (float64)` | float64 | 100,000 | 0.0244 ms | not_measured | managed | 0.422x |
| `a == b (float64)` | float64 | 10,000,000 | 15.8940 ms | not_measured | managed | 1.253x |
| `a == b (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `a == b (int32)` | int32 | 100,000 | 0.0221 ms | not_measured | managed | 0.339x |
| `a == b (int32)` | int32 | 10,000,000 | 8.7582 ms | not_measured | managed | 0.509x |
| `a == b (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `a == b (int64)` | int64 | 100,000 | 0.0288 ms | not_measured | managed | 0.438x |
| `a == b (int64)` | int64 | 10,000,000 | 15.7141 ms | not_measured | managed | 1.231x |
| `a > b (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `a > b (float32)` | float32 | 100,000 | 0.0163 ms | not_measured | managed | 0.613x |
| `a > b (float32)` | float32 | 10,000,000 | 10.4151 ms | not_measured | managed | 1.012x |
| `a > b (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a > b (float64)` | float64 | 100,000 | 0.0244 ms | not_measured | managed | 0.451x |
| `a > b (float64)` | float64 | 10,000,000 | 16.6324 ms | not_measured | managed | 1.153x |
| `a > b (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `a > b (int32)` | int32 | 100,000 | 0.0172 ms | not_measured | managed | 0.401x |
| `a > b (int32)` | int32 | 10,000,000 | 9.1228 ms | not_measured | managed | 0.509x |
| `a > b (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a > b (int64)` | int64 | 100,000 | 0.0268 ms | not_measured | managed | 0.683x |
| `a > b (int64)` | int64 | 10,000,000 | 16.0825 ms | not_measured | managed | 1.174x |
| `a >= b (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `a >= b (float32)` | float32 | 100,000 | 0.0170 ms | not_measured | managed | 0.412x |
| `a >= b (float32)` | float32 | 10,000,000 | 10.4222 ms | not_measured | managed | 1.027x |
| `a >= b (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `a >= b (float64)` | float64 | 100,000 | 0.0323 ms | not_measured | managed | 0.529x |
| `a >= b (float64)` | float64 | 10,000,000 | 16.3643 ms | not_measured | managed | 1.147x |
| `a >= b (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `a >= b (int32)` | int32 | 100,000 | 0.0180 ms | not_measured | managed | 0.383x |
| `a >= b (int32)` | int32 | 10,000,000 | 9.5301 ms | not_measured | managed | 0.478x |
| `a >= b (int64)` | int64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `a >= b (int64)` | int64 | 100,000 | 0.0294 ms | not_measured | managed | 0.619x |
| `a >= b (int64)` | int64 | 10,000,000 | 16.2601 ms | not_measured | managed | 1.155x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.0162 ms | not_measured | managed | 0.537x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 10.9479 ms | not_measured | managed | 0.979x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.0260 ms | not_measured | managed | 0.465x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 17.1136 ms | not_measured | managed | 1.050x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.0175 ms | not_measured | managed | 0.400x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 8.7519 ms | not_measured | managed | 0.507x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.0271 ms | not_measured | managed | 0.517x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 16.3267 ms | not_measured | managed | 1.166x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.0198 ms | not_measured | managed | 0.455x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 9.2807 ms | not_measured | managed | 1.155x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.0250 ms | not_measured | managed | 0.636x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 16.5794 ms | not_measured | managed | 1.148x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.0172 ms | not_measured | managed | 0.459x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 9.2540 ms | not_measured | managed | 0.652x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.0269 ms | not_measured | managed | 0.803x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 15.3942 ms | not_measured | managed | 1.237x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.250x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.0178 ms | not_measured | managed | 0.803x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 8.6376 ms | not_measured | managed | 1.243x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.0281 ms | not_measured | managed | 0.537x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 16.0657 ms | not_measured | managed | 1.161x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.0181 ms | not_measured | managed | 0.403x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 9.0928 ms | not_measured | managed | 1.198x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.0287 ms | not_measured | managed | 0.693x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 18.3576 ms | not_measured | managed | 1.012x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.0164 ms | not_measured | managed | 0.421x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 9.5024 ms | not_measured | managed | 1.120x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.0277 ms | not_measured | managed | 0.394x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 17.7934 ms | not_measured | managed | 1.056x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.0169 ms | not_measured | managed | 0.426x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 8.7655 ms | not_measured | managed | 0.521x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.0273 ms | not_measured | managed | 0.766x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 16.7905 ms | not_measured | managed | 1.134x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.0186 ms | not_measured | managed | 0.382x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 9.2772 ms | not_measured | managed | 1.178x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.0274 ms | not_measured | managed | 0.438x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 16.5357 ms | not_measured | managed | 1.134x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.0176 ms | not_measured | managed | 0.438x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 9.2549 ms | not_measured | managed | 1.142x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.200x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.0296 ms | not_measured | managed | 0.723x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 15.5919 ms | not_measured | managed | 1.228x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.0169 ms | not_measured | managed | 0.479x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 9.4803 ms | not_measured | managed | 1.138x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.417x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.0238 ms | not_measured | managed | 0.655x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 16.3539 ms | not_measured | managed | 1.119x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.0171 ms | not_measured | managed | 0.404x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 8.8105 ms | not_measured | managed | 0.519x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.0273 ms | not_measured | managed | 0.575x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 17.8131 ms | not_measured | managed | 1.041x |
| `a.copy() (float32)` | float32 | 1,000 | 0.0077 ms | not_measured | managed | 0.052x |
| `a.copy() (float32)` | float32 | 100,000 | 0.0117 ms | not_measured | managed | 0.496x |
| `a.copy() (float32)` | float32 | 10,000,000 | 5.0426 ms | not_measured | managed | 1.774x |
| `a.copy() (float64)` | float64 | 1,000 | 0.0139 ms | not_measured | managed | 0.029x |
| `a.copy() (float64)` | float64 | 100,000 | 0.0247 ms | not_measured | managed | 0.482x |
| `a.copy() (float64)` | float64 | 10,000,000 | 18.1872 ms | not_measured | managed | 1.034x |
| `a.copy() (int32)` | int32 | 1,000 | 0.0076 ms | not_measured | managed | 0.053x |
| `a.copy() (int32)` | int32 | 100,000 | 0.0132 ms | not_measured | managed | 0.455x |
| `a.copy() (int32)` | int32 | 10,000,000 | 4.8610 ms | not_measured | managed | 1.579x |
| `a.copy() (int64)` | int64 | 1,000 | 0.0122 ms | not_measured | managed | 0.041x |
| `a.copy() (int64)` | int64 | 100,000 | 0.0236 ms | not_measured | managed | 0.479x |
| `a.copy() (int64)` | int64 | 10,000,000 | 16.8558 ms | not_measured | managed | 1.084x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.0081 ms | not_measured | managed | 0.099x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.0555 ms | not_measured | managed | 0.712x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 5.7305 ms | not_measured | managed | 1.906x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.0098 ms | not_measured | managed | 0.092x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.0523 ms | not_measured | managed | 0.623x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 16.9090 ms | not_measured | managed | 1.162x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.0026 ms | not_measured | managed | 0.269x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.0555 ms | not_measured | managed | 0.539x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 5.6140 ms | not_measured | managed | 1.761x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.0141 ms | not_measured | managed | 0.043x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.0570 ms | not_measured | managed | 0.333x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 16.7159 ms | not_measured | managed | 1.106x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.0078 ms | not_measured | managed | 0.064x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.1185 ms | not_measured | managed | 0.097x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 15.000x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.0120 ms | not_measured | managed | 1.033x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 0.083x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.0622 ms | not_measured | managed | 0.158x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.161x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.2031 ms | not_measured | managed | 0.139x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 2.222x |
| `np.copy (float32)` | float32 | 1,000 | 0.0076 ms | not_measured | managed | 0.079x |
| `np.copy (float32)` | float32 | 100,000 | 0.0122 ms | not_measured | managed | 0.500x |
| `np.copy (float32)` | float32 | 10,000,000 | 5.0500 ms | not_measured | managed | 1.921x |
| `np.copy (float64)` | float64 | 1,000 | 0.0148 ms | not_measured | managed | 0.041x |
| `np.copy (float64)` | float64 | 100,000 | 0.0240 ms | not_measured | managed | 0.483x |
| `np.copy (float64)` | float64 | 10,000,000 | 17.1287 ms | not_measured | managed | 1.116x |
| `np.copy (int32)` | int32 | 1,000 | 0.0069 ms | not_measured | managed | 0.087x |
| `np.copy (int32)` | int32 | 100,000 | 0.0130 ms | not_measured | managed | 0.515x |
| `np.copy (int32)` | int32 | 10,000,000 | 4.8774 ms | not_measured | managed | 1.712x |
| `np.copy (int64)` | int64 | 1,000 | 0.0146 ms | not_measured | managed | 0.041x |
| `np.copy (int64)` | int64 | 100,000 | 0.0247 ms | not_measured | managed | 0.474x |
| `np.copy (int64)` | int64 | 10,000,000 | 16.8803 ms | not_measured | managed | 1.086x |
| `np.empty (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.176x |
| `np.empty (float32)` | float32 | 100,000 | 0.0015 ms | not_measured | managed | 0.200x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.0055 ms | not_measured | managed | 4.836x |
| `np.empty (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.231x |
| `np.empty (float64)` | float64 | 100,000 | 0.0051 ms | not_measured | managed | 0.059x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.0117 ms | not_measured | managed | 1.889x |
| `np.empty (int32)` | int32 | 1,000 | 0.0028 ms | not_measured | managed | 0.107x |
| `np.empty (int32)` | int32 | 100,000 | 0.0014 ms | not_measured | managed | 0.214x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.0019 ms | not_measured | managed | 6.579x |
| `np.empty (int64)` | int64 | 1,000 | 0.0037 ms | not_measured | managed | 0.081x |
| `np.empty (int64)` | int64 | 100,000 | 0.0016 ms | not_measured | managed | 0.188x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.0231 ms | not_measured | managed | 0.922x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.0039 ms | not_measured | managed | 0.103x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.0163 ms | not_measured | managed | 0.025x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.0075 ms | not_measured | managed | 2.600x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.150x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.0065 ms | not_measured | managed | 0.046x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.0120 ms | not_measured | managed | 1.775x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.0046 ms | not_measured | managed | 0.087x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.0092 ms | not_measured | managed | 0.043x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.0077 ms | not_measured | managed | 2.896x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.0126 ms | not_measured | managed | 0.032x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.0234 ms | not_measured | managed | 0.889x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.0111 ms | not_measured | managed | 0.117x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.0856 ms | not_measured | managed | 0.147x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.300x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.300x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.0654 ms | not_measured | managed | 1.430x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 6.9039 ms | not_measured | managed | 1.343x |
| `np.full (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.full (float32)` | float32 | 100,000 | 0.0194 ms | not_measured | managed | 0.289x |
| `np.full (float32)` | float32 | 10,000,000 | 4.2875 ms | not_measured | managed | 2.122x |
| `np.full (float64)` | float64 | 1,000 | 0.0097 ms | not_measured | managed | 0.093x |
| `np.full (float64)` | float64 | 100,000 | 0.0209 ms | not_measured | managed | 0.464x |
| `np.full (float64)` | float64 | 10,000,000 | 16.6372 ms | not_measured | managed | 1.188x |
| `np.full (int32)` | int32 | 1,000 | 0.0044 ms | not_measured | managed | 0.205x |
| `np.full (int32)` | int32 | 100,000 | 0.0192 ms | not_measured | managed | 0.276x |
| `np.full (int32)` | int32 | 10,000,000 | 4.1453 ms | not_measured | managed | 2.087x |
| `np.full (int64)` | int64 | 1,000 | 0.0107 ms | not_measured | managed | 0.084x |
| `np.full (int64)` | int64 | 100,000 | 0.0212 ms | not_measured | managed | 0.462x |
| `np.full (int64)` | int64 | 10,000,000 | 16.6937 ms | not_measured | managed | 1.088x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.0035 ms | not_measured | managed | 0.314x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.0197 ms | not_measured | managed | 0.279x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 4.5751 ms | not_measured | managed | 1.985x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.0050 ms | not_measured | managed | 0.280x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.0211 ms | not_measured | managed | 0.479x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 16.9170 ms | not_measured | managed | 1.143x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.0044 ms | not_measured | managed | 0.250x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.0194 ms | not_measured | managed | 0.284x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 4.5070 ms | not_measured | managed | 2.028x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.579x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.0212 ms | not_measured | managed | 0.481x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 17.0479 ms | not_measured | managed | 1.123x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.0076 ms | not_measured | managed | 0.211x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.0839 ms | not_measured | managed | 0.148x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.0049 ms | not_measured | managed | 0.959x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.1859 ms | not_measured | managed | 1.492x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 18.6489 ms | not_measured | managed | 3.155x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.0158 ms | not_measured | managed | 0.266x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.1826 ms | not_measured | managed | 0.302x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 26.3771 ms | not_measured | managed | 1.530x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.0039 ms | not_measured | managed | 1.385x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.3132 ms | not_measured | managed | 0.927x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 30.7353 ms | not_measured | managed | 2.208x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.0038 ms | not_measured | managed | 1.447x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.3118 ms | not_measured | managed | 1.094x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 37.2486 ms | not_measured | managed | 2.082x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.0122 ms | not_measured | managed | 0.730x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.2069 ms | not_measured | managed | 2.010x |
| `np.ones (float32)` | float32 | 1,000 | 0.0032 ms | not_measured | managed | 0.281x |
| `np.ones (float32)` | float32 | 100,000 | 0.0192 ms | not_measured | managed | 0.281x |
| `np.ones (float32)` | float32 | 10,000,000 | 4.2339 ms | not_measured | managed | 2.328x |
| `np.ones (float64)` | float64 | 1,000 | 0.0111 ms | not_measured | managed | 0.081x |
| `np.ones (float64)` | float64 | 100,000 | 0.0213 ms | not_measured | managed | 0.465x |
| `np.ones (float64)` | float64 | 10,000,000 | 16.9250 ms | not_measured | managed | 1.179x |
| `np.ones (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.237x |
| `np.ones (int32)` | int32 | 100,000 | 0.0193 ms | not_measured | managed | 0.275x |
| `np.ones (int32)` | int32 | 10,000,000 | 4.0588 ms | not_measured | managed | 1.895x |
| `np.ones (int64)` | int64 | 1,000 | 0.0085 ms | not_measured | managed | 0.106x |
| `np.ones (int64)` | int64 | 100,000 | 0.0212 ms | not_measured | managed | 0.472x |
| `np.ones (int64)` | int64 | 10,000,000 | 16.6046 ms | not_measured | managed | 1.124x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.0007 ms | not_measured | managed | 1.571x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.0193 ms | not_measured | managed | 0.280x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 4.3154 ms | not_measured | managed | 2.125x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 0.175x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.0214 ms | not_measured | managed | 0.472x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 16.8165 ms | not_measured | managed | 1.139x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.0032 ms | not_measured | managed | 0.344x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.0189 ms | not_measured | managed | 0.296x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 4.6095 ms | not_measured | managed | 1.926x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.0060 ms | not_measured | managed | 0.183x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.0212 ms | not_measured | managed | 0.481x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 16.8513 ms | not_measured | managed | 1.206x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.222x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.0579 ms | not_measured | managed | 0.235x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.0521 ms | not_measured | managed | 0.083x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.2320 ms | not_measured | managed | 0.759x |
| `np.zeros (float32)` | float32 | 1,000 | 0.0057 ms | not_measured | managed | 0.088x |
| `np.zeros (float32)` | float32 | 100,000 | 0.0038 ms | not_measured | managed | 1.316x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.0050 ms | not_measured | managed | 4.020x |
| `np.zeros (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `np.zeros (float64)` | float64 | 100,000 | 0.0045 ms | not_measured | managed | 2.067x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.0054 ms | not_measured | managed | 4.037x |
| `np.zeros (int32)` | int32 | 1,000 | 0.0029 ms | not_measured | managed | 0.138x |
| `np.zeros (int32)` | int32 | 100,000 | 0.0039 ms | not_measured | managed | 1.256x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.0037 ms | not_measured | managed | 3.081x |
| `np.zeros (int64)` | int64 | 1,000 | 0.0076 ms | not_measured | managed | 0.053x |
| `np.zeros (int64)` | int64 | 100,000 | 0.0048 ms | not_measured | managed | 1.979x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.0140 ms | not_measured | managed | 1.443x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.0030 ms | not_measured | managed | 0.333x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.0042 ms | not_measured | managed | 1.286x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.0054 ms | not_measured | managed | 1727.056x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.0123 ms | not_measured | managed | 0.081x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.0050 ms | not_measured | managed | 2.000x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.0056 ms | not_measured | managed | 3469.607x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.0030 ms | not_measured | managed | 0.367x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.0042 ms | not_measured | managed | 1.286x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.0040 ms | not_measured | managed | 2227.900x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.0069 ms | not_measured | managed | 0.145x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.0053 ms | not_measured | managed | 1.925x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.0150 ms | not_measured | managed | 1325.593x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.0295 ms | not_measured | managed | 0.292x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 2.4859 ms | not_measured | managed | 0.702x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.0726 ms | not_measured | managed | 0.533x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 11.9209 ms | not_measured | managed | 0.399x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.0185 ms | not_measured | managed | 0.189x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.5433 ms | not_measured | managed | 1.410x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.0768 ms | not_measured | managed | 0.477x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 11.8184 ms | not_measured | managed | 0.445x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.0120 ms | not_measured | managed | 0.417x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.4393 ms | not_measured | managed | 0.864x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.0132 ms | not_measured | managed | 0.606x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 1.1810 ms | not_measured | managed | 1.095x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.0199 ms | not_measured | managed | 0.462x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 2.4936 ms | not_measured | managed | 0.677x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.0765 ms | not_measured | managed | 0.527x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 11.8049 ms | not_measured | managed | 0.426x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.0759 ms | not_measured | managed | 0.549x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 12.4428 ms | not_measured | managed | 0.425x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.0119 ms | not_measured | managed | 0.395x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.3862 ms | not_measured | managed | 1.042x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.0125 ms | not_measured | managed | 0.648x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 1.1573 ms | not_measured | managed | 0.718x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.0111 ms | not_measured | managed | 0.658x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 1.1963 ms | not_measured | managed | 0.713x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.0391 ms | not_measured | managed | 0.668x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 5.8310 ms | not_measured | managed | 0.482x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.0418 ms | not_measured | managed | 0.651x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 5.7815 ms | not_measured | managed | 0.456x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.0097 ms | not_measured | managed | 0.691x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 1.1414 ms | not_measured | managed | 0.759x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.0432 ms | not_measured | managed | 0.669x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 5.6609 ms | not_measured | managed | 0.499x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 0.175x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.1991 ms | not_measured | managed | 0.148x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.0432 ms | not_measured | managed | 0.586x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 5.7007 ms | not_measured | managed | 0.459x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.500x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.0201 ms | 0.0074 ms | openblas | 1.027x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 8.8045 ms | not_measured | managed | 0.111x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 128 | 0.2096 ms | 0.0619 ms | openblas | 4.281x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.0195 ms | not_measured | managed | 0.077x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 1.5504 ms | not_measured | managed | 0.014x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 152.1234 ms | not_measured | managed | 0.030x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 2.432x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.0032 ms | not_measured | managed | 2.812x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.0030 ms | not_measured | managed | 3.067x |
| `np.linalg.cholesky(a)` | float64 | 32 | missing_backend | 0.0026 ms | openblas | 2.077x |
| `np.linalg.cholesky(a)` | float64 | 96 | missing_backend | 0.0356 ms | openblas | 0.834x |
| `np.linalg.cond(a)` | float64 | 32 | missing_backend | 0.0378 ms | openblas | 1.061x |
| `np.linalg.cond(a)` | float64 | 96 | missing_backend | 0.5259 ms | openblas | 0.608x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.0291 ms | not_measured | managed | 0.536x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 1.3797 ms | not_measured | managed | 0.481x |
| `np.linalg.det(a)` | float64 | 32 | missing_backend | 0.0052 ms | openblas | 1.269x |
| `np.linalg.det(a)` | float64 | 96 | missing_backend | 0.0600 ms | openblas | 0.500x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 10.333x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 1.167x |
| `np.linalg.eig(a)` | float64 | 32 | missing_backend | 0.1160 ms | openblas | 1.029x |
| `np.linalg.eig(a)` | float64 | 96 | missing_backend | 2.7234 ms | openblas | 0.564x |
| `np.linalg.eigh(a)` | float64 | 32 | missing_backend | 0.0468 ms | openblas | 1.370x |
| `np.linalg.eigh(a)` | float64 | 96 | missing_backend | 0.8513 ms | openblas | 0.520x |
| `np.linalg.eigvals(a)` | float64 | 32 | missing_backend | 0.0675 ms | openblas | 1.105x |
| `np.linalg.eigvals(a)` | float64 | 96 | missing_backend | 1.5264 ms | openblas | 0.568x |
| `np.linalg.eigvalsh(a)` | float64 | 32 | missing_backend | 0.0209 ms | openblas | 1.206x |
| `np.linalg.eigvalsh(a)` | float64 | 96 | missing_backend | 0.2885 ms | openblas | 0.805x |
| `np.linalg.inv(a)` | float64 | 32 | missing_backend | 0.0094 ms | openblas | 1.340x |
| `np.linalg.inv(a)` | float64 | 96 | missing_backend | 0.1642 ms | openblas | 0.750x |
| `np.linalg.lstsq(a, b)` | float64 | 32 | missing_backend | 0.0834 ms | openblas | 1.254x |
| `np.linalg.lstsq(a, b)` | float64 | 96 | missing_backend | 1.6352 ms | openblas | 0.684x |
| `np.linalg.matmul(a, b)` | float64 | 128 | 0.2078 ms | 0.0635 ms | openblas | 1.063x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.262x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 0.4073 ms | not_measured | managed | 0.172x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.0095 ms | not_measured | managed | 0.263x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.0461 ms | not_measured | managed | 0.134x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0581 ms | openblas | 0.761x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 96 | missing_backend | 0.4981 ms | openblas | 0.633x |
| `np.linalg.matrix_power(a, -1)` | float64 | 32 | missing_backend | 0.0214 ms | openblas | 0.617x |
| `np.linalg.matrix_power(a, -1)` | float64 | 96 | missing_backend | 0.1657 ms | openblas | 0.534x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.0196 ms | not_measured | managed | 0.291x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.7909 ms | not_measured | managed | 0.168x |
| `np.linalg.matrix_rank(a)` | float64 | 32 | missing_backend | 0.0372 ms | openblas | 1.121x |
| `np.linalg.matrix_rank(a)` | float64 | 96 | missing_backend | 0.5264 ms | openblas | 0.781x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 1.000x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 1.167x |
| `np.linalg.multi_dot(arrays)` | float64 | 128 | 0.4100 ms | 0.1199 ms | openblas | 1.119x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.0210 ms | not_measured | managed | 0.300x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.7925 ms | not_measured | managed | 0.180x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.526x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.0140 ms | not_measured | managed | 6.614x |
| `np.linalg.norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0645 ms | openblas | 0.730x |
| `np.linalg.norm(a, ord=2)` | float64 | 96 | missing_backend | 0.4958 ms | openblas | 0.622x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.0085 ms | not_measured | managed | 0.282x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.1599 ms | not_measured | managed | 0.243x |
| `np.linalg.pinv(a)` | float64 | 32 | missing_backend | 0.0967 ms | openblas | 1.040x |
| `np.linalg.pinv(a)` | float64 | 96 | missing_backend | 1.6728 ms | openblas | 0.618x |
| `np.linalg.qr(a)` | float64 | 32 | missing_backend | 0.0212 ms | openblas | 1.401x |
| `np.linalg.qr(a)` | float64 | 96 | missing_backend | 0.3196 ms | openblas | 0.589x |
| `np.linalg.slogdet(a)` | float64 | 32 | missing_backend | 0.0058 ms | openblas | 1.328x |
| `np.linalg.slogdet(a)` | float64 | 96 | missing_backend | 0.0585 ms | openblas | 0.538x |
| `np.linalg.solve(a, b)` | float64 | 32 | missing_backend | 0.0053 ms | openblas | 1.642x |
| `np.linalg.solve(a, b)` | float64 | 96 | missing_backend | 0.0609 ms | openblas | 0.544x |
| `np.linalg.svd(a)` | float64 | 32 | missing_backend | 0.0747 ms | openblas | 1.167x |
| `np.linalg.svd(a)` | float64 | 96 | missing_backend | 1.4758 ms | openblas | 0.634x |
| `np.linalg.svdvals(a)` | float64 | 32 | missing_backend | 0.0307 ms | openblas | 1.140x |
| `np.linalg.svdvals(a)` | float64 | 96 | missing_backend | 0.4991 ms | openblas | 0.619x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0143 ms | not_measured | managed | 0.420x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.4014 ms | not_measured | managed | 0.194x |
| `np.linalg.tensordot(a, b, axes=1)` | float64 | 128 | 0.2149 ms | 0.0614 ms | openblas | 1.132x |
| `np.linalg.tensorinv(a)` | float64 | 32 | missing_backend | 0.0100 ms | openblas | 1.270x |
| `np.linalg.tensorinv(a)` | float64 | 96 | missing_backend | 0.1621 ms | openblas | 0.567x |
| `np.linalg.tensorsolve(a, b)` | float64 | 32 | missing_backend | 0.0065 ms | openblas | 1.523x |
| `np.linalg.tensorsolve(a, b)` | float64 | 96 | missing_backend | 0.0593 ms | openblas | 0.580x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0068 ms | not_measured | managed | 0.147x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1809 ms | 0.0075 ms | openblas | 1.440x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.0072 ms | not_measured | managed | 0.361x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.3147 ms | not_measured | managed | 0.096x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.0110 ms | not_measured | managed | 0.282x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 4.3440 ms | not_measured | managed | 0.153x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 7.0651 ms | not_measured | managed | 0.100x |
| `np.matmul(a, b)` | float64 | 128 | 0.2122 ms | 0.0638 ms | openblas | 1.009x |
| `np.matvec(a, b)` | float64 | 128 | 0.0232 ms | 0.0021 ms | openblas | 1.190x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.179x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.1248 ms | not_measured | managed | 0.045x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.1766 ms | not_measured | managed | 0.063x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.0080 ms | not_measured | managed | 0.275x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.1546 ms | not_measured | managed | 0.255x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 23.4806 ms | not_measured | managed | 0.636x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.204x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 1.5228 ms | not_measured | managed | 0.067x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 151.0315 ms | not_measured | managed | 0.009x |
| `np.tensordot(a, b, axes=1)` | float64 | 128 | 0.2118 ms | 0.0614 ms | openblas | 1.161x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.0138 ms | 0.0081 ms | openblas | 1.185x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 9.3725 ms | not_measured | managed | 0.128x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0057 ms | not_measured | managed | 0.123x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1821 ms | 0.0073 ms | openblas | 1.918x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 41.2993 ms | not_measured | managed | 0.029x |
| `np.vecmat(a, b)` | float64 | 128 | 0.0031 ms | 0.0021 ms | openblas | 0.952x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.286x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.0540 ms | not_measured | managed | 0.113x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.0721 ms | not_measured | managed | 0.112x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.0232 ms | not_measured | managed | 1.297x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 1.0170 ms | not_measured | managed | 1.723x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.0421 ms | not_measured | managed | 0.311x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.5673 ms | not_measured | managed | 0.708x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.0488 ms | not_measured | managed | 0.285x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.5204 ms | not_measured | managed | 1.224x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.0028 ms | not_measured | managed | 0.893x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.1301 ms | not_measured | managed | 0.662x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 13.3505 ms | not_measured | managed | 1.625x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 1.000x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.0112 ms | not_measured | managed | 0.598x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 9.2203 ms | not_measured | managed | 1.193x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.941x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.0239 ms | not_measured | managed | 0.477x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 17.3401 ms | not_measured | managed | 0.966x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.0043 ms | not_measured | managed | 0.721x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.8763 ms | not_measured | managed | 0.870x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 91.5244 ms | not_measured | managed | 1.102x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.0561 ms | not_measured | managed | 0.127x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 13.4459 ms | not_measured | managed | 1.224x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.154x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.1060 ms | not_measured | managed | 0.267x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 37.2586 ms | not_measured | managed | 0.870x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.0040 ms | not_measured | managed | 0.750x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.8770 ms | not_measured | managed | 0.880x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 91.6410 ms | not_measured | managed | 1.086x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.0543 ms | not_measured | managed | 0.131x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 13.3755 ms | not_measured | managed | 1.227x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.171x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.1137 ms | not_measured | managed | 0.249x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 41.2037 ms | not_measured | managed | 0.788x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.0922 ms | not_measured | managed | 0.311x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 1.0236 ms | not_measured | managed | 1.764x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.0845 ms | not_measured | managed | 0.154x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.5502 ms | not_measured | managed | 0.684x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.0711 ms | not_measured | managed | 0.159x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.5608 ms | not_measured | managed | 1.127x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.0035 ms | not_measured | managed | 0.486x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | 0.0095 ms | not_measured | managed | 2.189x |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.0033 ms | not_measured | managed | 0.515x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | 0.0060 ms | not_measured | managed | 3.550x |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.0033 ms | not_measured | managed | 0.515x |
| `np.iscomplex(a) (float64)` | float64 | 10,000,000 | 0.0059 ms | not_measured | managed | 3.237x |
| `np.iscomplexobj(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.iscomplexobj(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.0536 ms | not_measured | managed | 0.927x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 6.0048 ms | not_measured | managed | 0.966x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.0542 ms | not_measured | managed | 0.094x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 8.3496 ms | not_measured | managed | 0.774x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.0555 ms | not_measured | managed | 0.177x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 11.2716 ms | not_measured | managed | 1.019x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.0568 ms | not_measured | managed | 0.879x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 6.0694 ms | not_measured | managed | 0.993x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.0583 ms | not_measured | managed | 0.091x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 9.5788 ms | not_measured | managed | 0.695x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.0642 ms | not_measured | managed | 0.157x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 10.8356 ms | not_measured | managed | 0.986x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.0031 ms | not_measured | managed | 0.419x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.0573 ms | not_measured | managed | 1.180x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 5.9339 ms | not_measured | managed | 1.315x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.0574 ms | not_measured | managed | 0.073x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 6.8278 ms | not_measured | managed | 0.878x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.0547 ms | not_measured | managed | 0.155x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 10.9001 ms | not_measured | managed | 0.991x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.0016 ms | not_measured | managed | 1.250x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.0035 ms | not_measured | managed | 22.000x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 0.3904 ms | not_measured | managed | 53.681x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.706x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.0036 ms | not_measured | managed | 2.333x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 0.3693 ms | not_measured | managed | 30.503x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.182x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.0035 ms | not_measured | managed | 4.686x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 0.4615 ms | not_measured | managed | 43.268x |
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
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.0041 ms | not_measured | managed | 0.732x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.9065 ms | not_measured | managed | 0.860x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 94.8747 ms | not_measured | managed | 1.103x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.250x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.0559 ms | not_measured | managed | 0.122x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 13.8913 ms | not_measured | managed | 1.166x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.167x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.1068 ms | not_measured | managed | 0.285x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 28.0617 ms | not_measured | managed | 1.143x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.0042 ms | not_measured | managed | 0.857x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.9287 ms | not_measured | managed | 0.823x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 93.9371 ms | not_measured | managed | 1.074x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.286x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.0541 ms | not_measured | managed | 0.128x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 13.7495 ms | not_measured | managed | 1.190x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 0.143x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.1018 ms | not_measured | managed | 0.283x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 28.3840 ms | not_measured | managed | 1.133x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.125x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 0.250x |
| `a.flatten` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `a.flatten` | float64 | 100,000 | 0.0999 ms | not_measured | managed | 0.113x |
| `a.flatten` | float64 | 10,000,000 | 14.7159 ms | not_measured | managed | 1.398x |
| `a.ravel() (view)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.111x |
| `a.ravel() (view)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.100x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.268x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.4295 ms | not_measured | managed | 1.315x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 1.354x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.0052 ms | not_measured | managed | 3.615x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.750x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 1.250x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.0017 ms | not_measured | managed | 0.588x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.600x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.4354 ms | not_measured | managed | 1.656x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 1.571x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.0016 ms | not_measured | managed | 8.938x |
| `np.byteswap` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.143x |
| `np.byteswap` | float64 | 100,000 | 0.1641 ms | not_measured | managed | 0.209x |
| `np.byteswap` | float64 | 10,000,000 | 14.7308 ms | not_measured | managed | 1.947x |
| `np.c_` | float64 | 1,000 | 0.0104 ms | not_measured | managed | 0.423x |
| `np.c_` | float64 | 100,000 | 0.3212 ms | not_measured | managed | 1.007x |
| `np.c_` | float64 | 10,000,000 | 48.0018 ms | not_measured | managed | 1.438x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.0079 ms | not_measured | managed | 0.278x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.4331 ms | not_measured | managed | 1.537x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.222x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.3307 ms | not_measured | managed | 1.660x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.0077 ms | not_measured | managed | 0.169x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.4875 ms | not_measured | managed | 0.924x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.0060 ms | not_measured | managed | 0.183x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.3355 ms | not_measured | managed | 0.925x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 0.224x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.3312 ms | not_measured | managed | 0.940x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.245x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.3142 ms | not_measured | managed | 0.921x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 0.404x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.1206 ms | not_measured | managed | 0.273x |
| `np.diag` | float64 | 1,000 | 0.0092 ms | not_measured | managed | 0.087x |
| `np.diag` | float64 | 100,000 | 0.0851 ms | not_measured | managed | 0.009x |
| `np.diag` | float64 | 10,000,000 | 4.1864 ms | not_measured | managed | 0.000x |
| `np.diagflat` | float64 | 1,000 | 0.0085 ms | not_measured | managed | 0.376x |
| `np.diagflat` | float64 | 100,000 | 0.0871 ms | not_measured | managed | 0.160x |
| `np.diagflat` | float64 | 10,000,000 | 4.1722 ms | not_measured | managed | 1.210x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 1.673x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.0066 ms | not_measured | managed | 3.697x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.0088 ms | not_measured | managed | 0.284x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.3276 ms | not_measured | managed | 0.945x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 1.500x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 1.500x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 4.429x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 1.333x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 1.444x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 3.875x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 3.143x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 1.625x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 3.875x |
| `np.fill_diagonal` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.270x |
| `np.fill_diagonal` | float64 | 100,000 | 0.0045 ms | not_measured | managed | 0.556x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.0397 ms | not_measured | managed | 1.050x |
| `np.flip` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.flip` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.500x |
| `np.flip` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 1.800x |
| `np.fliplr` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.fliplr` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.600x |
| `np.fliplr` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.333x |
| `np.flipud` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.500x |
| `np.flipud` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.500x |
| `np.flipud` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 1.400x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 1.429x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.0074 ms | not_measured | managed | 3.189x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.362x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.3277 ms | not_measured | managed | 0.944x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.0090 ms | not_measured | managed | 0.578x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.1143 ms | not_measured | managed | 0.391x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.0388 ms | not_measured | managed | 0.985x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 9.8950 ms | not_measured | managed | 0.782x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.0364 ms | not_measured | managed | 0.464x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 1.0338 ms | not_measured | managed | 0.693x |
| `np.ix_` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.857x |
| `np.ix_` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 0.857x |
| `np.ix_` | float64 | 10,000,000 | 0.0020 ms | not_measured | managed | 2.100x |
| `np.matrix_transpose` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.750x |
| `np.matrix_transpose` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 1.857x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.909x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 2.100x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 4.364x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.0130 ms | not_measured | managed | 0.738x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.1105 ms | not_measured | managed | 0.849x |
| `np.permute_dims` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.permute_dims` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.571x |
| `np.permute_dims` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.000x |
| `np.r_` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 0.729x |
| `np.r_` | float64 | 100,000 | 0.3198 ms | not_measured | managed | 0.977x |
| `np.r_` | float64 | 10,000,000 | 36.6016 ms | not_measured | managed | 0.985x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 1.439x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.2156 ms | not_measured | managed | 1.635x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 37.3187 ms | not_measured | managed | 1.512x |
| `np.r_(0:N)` | float64 | 1,000 | 0.0065 ms | not_measured | managed | 0.385x |
| `np.r_(0:N)` | float64 | 100,000 | 0.2078 ms | not_measured | managed | 1.518x |
| `np.r_(0:N)` | float64 | 10,000,000 | 35.7730 ms | not_measured | managed | 1.016x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.0107 ms | not_measured | managed | 0.393x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.3213 ms | not_measured | managed | 1.063x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 37.1718 ms | not_measured | managed | 0.988x |
| `np.ravel` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.444x |
| `np.ravel` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.300x |
| `np.ravel` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 1.000x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.0084 ms | not_measured | managed | 0.345x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.4009 ms | not_measured | managed | 2.863x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.667x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.600x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.0061 ms | not_measured | managed | 0.443x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.3650 ms | not_measured | managed | 1.597x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.0111 ms | not_measured | managed | 0.414x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.1136 ms | not_measured | managed | 0.359x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.750x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 2.000x |
| `np.rot90` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 2.200x |
| `np.rot90` | float64 | 100,000 | 0.0014 ms | not_measured | managed | 2.286x |
| `np.rot90` | float64 | 10,000,000 | 0.0015 ms | not_measured | managed | 6.400x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.0740 ms | not_measured | managed | 0.666x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 9.5440 ms | not_measured | managed | 0.816x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.0566 ms | not_measured | managed | 0.691x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 9.8140 ms | not_measured | managed | 0.784x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 1.438x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.0060 ms | not_measured | managed | 3.750x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 0.600x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.429x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.333x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.667x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.600x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.500x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.0078 ms | not_measured | managed | 0.282x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.3314 ms | not_measured | managed | 0.986x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0079 ms | not_measured | managed | 0.304x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.3300 ms | not_measured | managed | 0.978x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.444x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.444x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 1.143x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.500x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.571x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 1.143x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.0102 ms | not_measured | managed | 0.255x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.3348 ms | not_measured | managed | 1.851x |
| `np.transpose(a)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.500x |
| `np.transpose(a)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.400x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 0.750x |
| `np.tri` | float64 | 1,000 | 0.0046 ms | not_measured | managed | 0.696x |
| `np.tri` | float64 | 100,000 | 0.1106 ms | not_measured | managed | 0.448x |
| `np.tri` | float64 | 10,000,000 | 17.3390 ms | not_measured | managed | 1.189x |
| `np.tril` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 1.344x |
| `np.tril` | float64 | 100,000 | 0.1044 ms | not_measured | managed | 0.463x |
| `np.tril` | float64 | 10,000,000 | 20.2680 ms | not_measured | managed | 1.255x |
| `np.tril_indices` | float64 | 1,000 | 0.0076 ms | not_measured | managed | 1.421x |
| `np.tril_indices` | float64 | 100,000 | 0.1153 ms | not_measured | managed | 0.633x |
| `np.tril_indices` | float64 | 10,000,000 | 10.8206 ms | not_measured | managed | 2.551x |
| `np.trim_zeros` | float64 | 1,000 | 0.0138 ms | not_measured | managed | 0.877x |
| `np.trim_zeros` | float64 | 100,000 | 0.0220 ms | not_measured | managed | 43.586x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.1178 ms | not_measured | managed | 1748.288x |
| `np.triu` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 1.189x |
| `np.triu` | float64 | 100,000 | 0.0991 ms | not_measured | managed | 0.541x |
| `np.triu` | float64 | 10,000,000 | 20.1464 ms | not_measured | managed | 1.298x |
| `np.triu_indices` | float64 | 1,000 | 0.0096 ms | not_measured | managed | 1.104x |
| `np.triu_indices` | float64 | 100,000 | 0.1084 ms | not_measured | managed | 0.633x |
| `np.triu_indices` | float64 | 10,000,000 | 11.6132 ms | not_measured | managed | 2.667x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.0257 ms | not_measured | managed | 0.837x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 5.8396 ms | not_measured | managed | 0.832x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.0098 ms | not_measured | managed | 1.643x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.4773 ms | not_measured | managed | 0.708x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.0356 ms | not_measured | managed | 1.006x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 3.3528 ms | not_measured | managed | 1.933x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.0196 ms | not_measured | managed | 0.505x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 5.1066 ms | not_measured | managed | 0.134x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.0339 ms | not_measured | managed | 0.667x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 2.6997 ms | not_measured | managed | 2.088x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.0099 ms | not_measured | managed | 1.525x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 4.6014 ms | not_measured | managed | 1.810x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 0.516x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.0060 ms | not_measured | managed | 1.267x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.0079 ms | not_measured | managed | 1.152x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.0060 ms | not_measured | managed | 3.800x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.0080 ms | not_measured | managed | 0.263x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.3279 ms | not_measured | managed | 0.974x |
| `reshape 1D->2D` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `reshape 1D->2D` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `reshape 1D->3D` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.167x |
| `reshape 1D->3D` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 0.167x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.333x |
| `reshape 2D->1D` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `reshape 2D->1D` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 0.273x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.875x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.0177 ms | not_measured | managed | 2.288x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.700x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.0024 ms | not_measured | managed | 16.417x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.400x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.0612 ms | not_measured | managed | 0.257x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.0608 ms | not_measured | managed | 0.258x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0145 ms | not_measured | managed | 0.297x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.7855 ms | not_measured | managed | 0.186x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.0346 ms | not_measured | managed | 0.298x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 3.6515 ms | not_measured | managed | 0.396x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.0135 ms | not_measured | managed | 0.370x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.3511 ms | not_measured | managed | 1.750x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.308x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.1486 ms | not_measured | managed | 0.082x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.808x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.1146 ms | not_measured | managed | 0.700x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.0058 ms | not_measured | managed | 0.517x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.3281 ms | not_measured | managed | 0.515x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.434x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.3154 ms | not_measured | managed | 0.515x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.556x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.0194 ms | 0.0093 ms | openblas | 0.849x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 2.000x |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.0216 ms | not_measured | managed | 0.421x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.0181 ms | not_measured | managed | 0.680x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.0197 ms | not_measured | managed | 1.117x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.0046 ms | not_measured | managed | 0.609x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.1491 ms | not_measured | managed | 1.195x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0157 ms | not_measured | managed | 0.210x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.7073 ms | not_measured | managed | 0.202x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.909x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.0139 ms | not_measured | managed | 5.374x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.0065 ms | not_measured | managed | 0.277x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.1429 ms | not_measured | managed | 0.747x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.322x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.4218 ms | not_measured | managed | 1.116x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.167x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.156x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.2160 ms | not_measured | managed | 0.054x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 0.240x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.1353 ms | not_measured | managed | 0.081x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.0174 ms | not_measured | managed | 0.368x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 3.0126 ms | not_measured | managed | 0.696x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.103x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0532 ms | not_measured | managed | 0.175x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.0288 ms | not_measured | managed | 0.111x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 1.9682 ms | not_measured | managed | 0.251x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.286x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.333x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.1152 ms | not_measured | managed | 0.446x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.0922 ms | not_measured | managed | 0.117x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.3628 ms | not_measured | managed | 0.421x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 3.9354 ms | not_measured | managed | 0.088x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.0111 ms | not_measured | managed | 0.703x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 3.9615 ms | not_measured | managed | 0.210x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.111x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.200x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.0445 ms | not_measured | managed | 1.004x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 8.6808 ms | not_measured | managed | 0.491x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.1489 ms | not_measured | managed | 0.162x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 13.9775 ms | not_measured | managed | 0.248x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.0465 ms | not_measured | managed | 0.497x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 4.4503 ms | not_measured | managed | 0.484x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.0465 ms | not_measured | managed | 0.243x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 1.6229 ms | not_measured | managed | 0.804x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.1046 ms | not_measured | managed | 0.669x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 9.9404 ms | not_measured | managed | 0.947x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.0321 ms | not_measured | managed | 0.308x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 2.7524 ms | not_measured | managed | 0.328x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.1201 ms | not_measured | managed | 0.359x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 10.1436 ms | not_measured | managed | 0.415x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.0511 ms | not_measured | managed | 0.434x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 4.6059 ms | not_measured | managed | 0.462x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.0234 ms | not_measured | managed | 0.808x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 2.2896 ms | not_measured | managed | 0.968x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.0346 ms | not_measured | managed | 0.434x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 3.2546 ms | not_measured | managed | 0.454x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1613 ms | not_measured | managed | 0.425x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 16.9447 ms | not_measured | managed | 0.565x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.0306 ms | not_measured | managed | 0.490x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 2.7183 ms | not_measured | managed | 0.522x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.0239 ms | not_measured | managed | 0.414x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 2.1956 ms | not_measured | managed | 0.407x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.0356 ms | not_measured | managed | 0.565x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 3.0780 ms | not_measured | managed | 0.583x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.0439 ms | not_measured | managed | 0.656x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 4.3833 ms | not_measured | managed | 0.863x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.2337 ms | not_measured | managed | 0.350x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 22.7929 ms | not_measured | managed | 0.515x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.0874 ms | not_measured | managed | 0.941x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 8.4879 ms | not_measured | managed | 0.924x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.1454 ms | not_measured | managed | 0.556x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 14.6470 ms | not_measured | managed | 0.750x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.0667 ms | not_measured | managed | 0.531x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 6.7991 ms | not_measured | managed | 0.494x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.1026 ms | not_measured | managed | 0.568x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 10.5679 ms | not_measured | managed | 0.516x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.0258 ms | not_measured | managed | 0.504x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 2.5672 ms | not_measured | managed | 0.446x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.0331 ms | not_measured | managed | 0.508x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 3.1277 ms | not_measured | managed | 0.487x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.0202 ms | not_measured | managed | 0.757x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 2.0554 ms | not_measured | managed | 1.037x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.0555 ms | not_measured | managed | 0.634x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 5.4420 ms | not_measured | managed | 0.836x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.0339 ms | not_measured | managed | 0.906x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 2.9671 ms | not_measured | managed | 0.976x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.0173 ms | not_measured | managed | 0.295x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 1.4522 ms | not_measured | managed | 0.253x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.0108 ms | not_measured | managed | 0.759x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.9544 ms | not_measured | managed | 0.657x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.0266 ms | not_measured | managed | 0.410x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 2.3662 ms | not_measured | managed | 0.446x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.0162 ms | not_measured | managed | 0.265x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 1.4768 ms | not_measured | managed | 0.248x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.0159 ms | not_measured | managed | 0.270x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 1.4643 ms | not_measured | managed | 0.247x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.0236 ms | not_measured | managed | 0.483x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 2.2769 ms | not_measured | managed | 0.461x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.0169 ms | not_measured | managed | 0.828x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 1.9751 ms | not_measured | managed | 1.089x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.0318 ms | not_measured | managed | 0.736x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 2.8898 ms | not_measured | managed | 0.770x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.0241 ms | not_measured | managed | 0.386x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 2.1647 ms | not_measured | managed | 0.383x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.0435 ms | not_measured | managed | 0.522x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 4.5468 ms | not_measured | managed | 0.437x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.0261 ms | not_measured | managed | 0.444x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 2.4722 ms | not_measured | managed | 0.427x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.0661 ms | not_measured | managed | 0.548x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 6.6805 ms | not_measured | managed | 0.579x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.0177 ms | not_measured | managed | 0.576x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 1.6935 ms | not_measured | managed | 0.786x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.0149 ms | not_measured | managed | 0.369x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 1.3129 ms | not_measured | managed | 0.507x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.0910 ms | not_measured | managed | 0.603x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 9.0151 ms | not_measured | managed | 0.953x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.0494 ms | not_measured | managed | 0.536x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 4.9547 ms | not_measured | managed | 0.797x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.0389 ms | not_measured | managed | 0.584x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 4.0043 ms | not_measured | managed | 0.888x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.0831 ms | not_measured | managed | 0.427x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 8.2310 ms | not_measured | managed | 0.753x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.043x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1465 ms | not_measured | managed | 1.042x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 19.4148 ms | not_measured | managed | 1.391x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 1.364x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4420 ms | not_measured | managed | 1.127x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 46.6380 ms | not_measured | managed | 1.447x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.0087 ms | not_measured | managed | 0.621x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 3.8133 ms | not_measured | managed | 1.177x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.0170 ms | not_measured | managed | 0.576x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 8.2188 ms | not_measured | managed | 1.097x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.0030 ms | not_measured | managed | 0.767x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.6632 ms | not_measured | managed | 1.008x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.0051 ms | not_measured | managed | 0.725x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 3.4165 ms | not_measured | managed | 1.128x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.0266 ms | not_measured | managed | 0.312x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 8.6870 ms | not_measured | managed | 1.146x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 0.842x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.2270 ms | not_measured | managed | 1.125x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.0028 ms | not_measured | managed | 0.821x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 0.6788 ms | not_measured | managed | 0.986x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.0043 ms | not_measured | managed | 0.814x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 3.2421 ms | not_measured | managed | 1.325x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.0372 ms | not_measured | managed | 0.301x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 8.4484 ms | not_measured | managed | 1.262x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.0019 ms | not_measured | managed | 0.842x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.2302 ms | not_measured | managed | 1.289x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.0024 ms | not_measured | managed | 1.000x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1464 ms | not_measured | managed | 1.041x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 22.2022 ms | not_measured | managed | 1.154x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 1.391x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.4087 ms | not_measured | managed | 1.238x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 42.9831 ms | not_measured | managed | 1.783x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.0089 ms | not_measured | managed | 0.596x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 3.9529 ms | not_measured | managed | 1.268x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.0177 ms | not_measured | managed | 0.548x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 8.8823 ms | not_measured | managed | 0.987x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 0.821x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 0.6949 ms | not_measured | managed | 1.035x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.0043 ms | not_measured | managed | 0.837x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 3.8595 ms | not_measured | managed | 1.077x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.0267 ms | not_measured | managed | 0.352x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 8.0316 ms | not_measured | managed | 1.529x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 0.895x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.3464 ms | not_measured | managed | 0.739x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.0029 ms | not_measured | managed | 1.586x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 0.7870 ms | not_measured | managed | 0.378x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.0045 ms | not_measured | managed | 0.778x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 3.4362 ms | not_measured | managed | 0.834x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.0378 ms | not_measured | managed | 0.415x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 8.2983 ms | not_measured | managed | 1.323x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.0020 ms | not_measured | managed | 0.800x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.2287 ms | not_measured | managed | 1.143x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.0016 ms | not_measured | managed | 1.375x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.0224 ms | not_measured | managed | 1.317x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 16.8702 ms | not_measured | managed | 1.204x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 2.000x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.1613 ms | not_measured | managed | 0.658x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 16.1575 ms | not_measured | managed | 1.232x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 2.909x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.0055 ms | not_measured | managed | 3.127x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 3.2409 ms | not_measured | managed | 3.224x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.0087 ms | not_measured | managed | 2.057x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 7.4510 ms | not_measured | managed | 2.548x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 2.250x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.0466 ms | not_measured | managed | 1.105x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 4.4159 ms | not_measured | managed | 2.300x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 2.167x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.0449 ms | not_measured | managed | 0.873x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 5.2430 ms | not_measured | managed | 1.935x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 2.182x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.0089 ms | not_measured | managed | 3.888x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 7.6139 ms | not_measured | managed | 1.922x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 2.364x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.0441 ms | not_measured | managed | 1.161x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 4.6594 ms | not_measured | managed | 1.825x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.364x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.0397 ms | not_measured | managed | 1.343x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 4.1746 ms | not_measured | managed | 2.327x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 2.273x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.0399 ms | not_measured | managed | 0.990x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 5.0884 ms | not_measured | managed | 1.534x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.0012 ms | not_measured | managed | 2.083x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.0092 ms | not_measured | managed | 5.511x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 7.4433 ms | not_measured | managed | 1.834x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 2.364x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.0408 ms | not_measured | managed | 1.275x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 3.6912 ms | not_measured | managed | 2.320x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 3.933x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.4004 ms | not_measured | managed | 2.246x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 40.6214 ms | not_measured | managed | 4.479x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 7.600x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.0200 ms | not_measured | managed | 2.315x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 8.9418 ms | not_measured | managed | 4.927x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 4.286x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.0391 ms | not_measured | managed | 1.604x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 17.9911 ms | not_measured | managed | 4.097x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.0014 ms | not_measured | managed | 0.786x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.0228 ms | not_measured | managed | 1.289x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 16.0405 ms | not_measured | managed | 1.270x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 2.071x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.0671 ms | not_measured | managed | 3.042x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 6.7273 ms | not_measured | managed | 4.189x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.0060 ms | not_measured | managed | 2.433x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 3.6996 ms | not_measured | managed | 3.450x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.0094 ms | not_measured | managed | 1.702x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 7.6501 ms | not_measured | managed | 2.489x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.0436 ms | not_measured | managed | 0.736x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 4.8192 ms | not_measured | managed | 1.771x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 1.250x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.0471 ms | not_measured | managed | 0.707x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 5.5929 ms | not_measured | managed | 1.778x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.0091 ms | not_measured | managed | 1.967x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 7.9808 ms | not_measured | managed | 1.441x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 1.167x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.0469 ms | not_measured | managed | 0.712x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 4.6138 ms | not_measured | managed | 1.982x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 1.077x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.0412 ms | not_measured | managed | 0.774x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 4.4138 ms | not_measured | managed | 2.192x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 1.167x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.0434 ms | not_measured | managed | 0.737x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 5.2924 ms | not_measured | managed | 1.629x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.0089 ms | not_measured | managed | 2.101x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 7.6351 ms | not_measured | managed | 1.269x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.364x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.0425 ms | not_measured | managed | 0.793x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 3.7468 ms | not_measured | managed | 2.242x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 3.822x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.3923 ms | not_measured | managed | 2.258x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 38.1121 ms | not_measured | managed | 4.869x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 5.917x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.0192 ms | not_measured | managed | 2.354x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 9.0338 ms | not_measured | managed | 3.852x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 2.947x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.0367 ms | not_measured | managed | 1.668x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 17.9969 ms | not_measured | managed | 3.989x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.348x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1459 ms | not_measured | managed | 1.063x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 19.3953 ms | not_measured | managed | 1.261x |
| `np.amax (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 1.682x |
| `np.amax (float16)` | float16 | 100,000 | 0.4411 ms | not_measured | managed | 1.138x |
| `np.amax (float16)` | float16 | 10,000,000 | 46.3431 ms | not_measured | managed | 1.654x |
| `np.amax (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.amax (float32)` | float32 | 100,000 | 0.0087 ms | not_measured | managed | 0.724x |
| `np.amax (float32)` | float32 | 10,000,000 | 3.8165 ms | not_measured | managed | 1.167x |
| `np.amax (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 1.308x |
| `np.amax (float64)` | float64 | 100,000 | 0.0169 ms | not_measured | managed | 0.633x |
| `np.amax (float64)` | float64 | 10,000,000 | 8.1764 ms | not_measured | managed | 1.100x |
| `np.amax (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amax (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 1.071x |
| `np.amax (int16)` | int16 | 10,000,000 | 0.6553 ms | not_measured | managed | 1.060x |
| `np.amax (int32)` | int32 | 1,000 | 0.0008 ms | not_measured | managed | 2.000x |
| `np.amax (int32)` | int32 | 100,000 | 0.0043 ms | not_measured | managed | 1.000x |
| `np.amax (int32)` | int32 | 10,000,000 | 3.3807 ms | not_measured | managed | 1.144x |
| `np.amax (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 1.133x |
| `np.amax (int64)` | int64 | 100,000 | 0.0272 ms | not_measured | managed | 0.327x |
| `np.amax (int64)` | int64 | 10,000,000 | 8.6767 ms | not_measured | managed | 1.130x |
| `np.amax (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 1.600x |
| `np.amax (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 1.211x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2321 ms | not_measured | managed | 1.102x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 1.600x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.0029 ms | not_measured | managed | 1.034x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 0.6733 ms | not_measured | managed | 0.432x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.0044 ms | not_measured | managed | 0.955x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 3.3142 ms | not_measured | managed | 1.193x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.0367 ms | not_measured | managed | 0.327x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 8.5682 ms | not_measured | managed | 1.426x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.0020 ms | not_measured | managed | 1.150x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.2313 ms | not_measured | managed | 1.240x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.0044 ms | not_measured | managed | 0.659x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1897 ms | not_measured | managed | 0.761x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 21.4350 ms | not_measured | managed | 1.010x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.0035 ms | not_measured | managed | 1.114x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6599 ms | not_measured | managed | 0.753x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 137.8232 ms | not_measured | managed | 0.593x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 2.000x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.0276 ms | not_measured | managed | 0.391x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 4.3235 ms | not_measured | managed | 1.023x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.429x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.0549 ms | not_measured | managed | 0.281x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 8.6485 ms | not_measured | managed | 1.224x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.0073 ms | not_measured | managed | 0.932x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 0.7794 ms | not_measured | managed | 1.683x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.0104 ms | not_measured | managed | 1.058x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 3.9053 ms | not_measured | managed | 1.101x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.0286 ms | not_measured | managed | 0.490x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 8.6821 ms | not_measured | managed | 1.275x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 1.583x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.0078 ms | not_measured | managed | 0.987x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3401 ms | not_measured | managed | 1.159x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.0070 ms | not_measured | managed | 1.043x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 0.7641 ms | not_measured | managed | 1.098x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 1.538x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.0104 ms | not_measured | managed | 1.058x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 3.7564 ms | not_measured | managed | 1.191x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 1.312x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.0370 ms | not_measured | managed | 0.443x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 8.6624 ms | not_measured | managed | 1.178x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.0075 ms | not_measured | managed | 1.067x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.3251 ms | not_measured | managed | 1.239x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.304x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1452 ms | not_measured | managed | 1.079x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 23.3065 ms | not_measured | managed | 1.123x |
| `np.amin (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 1.739x |
| `np.amin (float16)` | float16 | 100,000 | 0.4080 ms | not_measured | managed | 1.292x |
| `np.amin (float16)` | float16 | 10,000,000 | 43.1390 ms | not_measured | managed | 1.766x |
| `np.amin (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.amin (float32)` | float32 | 100,000 | 0.0090 ms | not_measured | managed | 0.667x |
| `np.amin (float32)` | float32 | 10,000,000 | 3.9518 ms | not_measured | managed | 1.352x |
| `np.amin (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.amin (float64)` | float64 | 100,000 | 0.0175 ms | not_measured | managed | 0.606x |
| `np.amin (float64)` | float64 | 10,000,000 | 8.7899 ms | not_measured | managed | 1.074x |
| `np.amin (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.amin (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 1.250x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.7167 ms | not_measured | managed | 0.973x |
| `np.amin (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amin (int32)` | int32 | 100,000 | 0.0044 ms | not_measured | managed | 0.977x |
| `np.amin (int32)` | int32 | 10,000,000 | 3.5229 ms | not_measured | managed | 0.707x |
| `np.amin (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 1.267x |
| `np.amin (int64)` | int64 | 100,000 | 0.0271 ms | not_measured | managed | 0.351x |
| `np.amin (int64)` | int64 | 10,000,000 | 8.1100 ms | not_measured | managed | 1.181x |
| `np.amin (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amin (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 1.421x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.2474 ms | not_measured | managed | 1.040x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.0030 ms | not_measured | managed | 1.167x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 0.7243 ms | not_measured | managed | 1.097x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.0044 ms | not_measured | managed | 1.023x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 3.3603 ms | not_measured | managed | 0.289x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.0379 ms | not_measured | managed | 0.317x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 8.1311 ms | not_measured | managed | 1.061x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.0020 ms | not_measured | managed | 1.350x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.2335 ms | not_measured | managed | 1.191x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.0045 ms | not_measured | managed | 0.667x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.2240 ms | not_measured | managed | 0.629x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 26.3954 ms | not_measured | managed | 0.887x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.0036 ms | not_measured | managed | 1.111x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.6658 ms | not_measured | managed | 0.709x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 142.1820 ms | not_measured | managed | 0.548x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 1.818x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.0276 ms | not_measured | managed | 0.399x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 4.4230 ms | not_measured | managed | 0.976x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.429x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.0558 ms | not_measured | managed | 0.267x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 9.1703 ms | not_measured | managed | 1.184x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 1.583x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.0072 ms | not_measured | managed | 0.986x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 0.8187 ms | not_measured | managed | 1.181x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.0102 ms | not_measured | managed | 1.078x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 5.0503 ms | not_measured | managed | 0.860x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.0286 ms | not_measured | managed | 0.521x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 8.8740 ms | not_measured | managed | 0.982x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.0079 ms | not_measured | managed | 0.886x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.3634 ms | not_measured | managed | 1.103x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.0072 ms | not_measured | managed | 1.000x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 1.6855 ms | not_measured | managed | 0.521x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.0105 ms | not_measured | managed | 1.048x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 4.0702 ms | not_measured | managed | 1.325x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 1.538x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.0373 ms | not_measured | managed | 0.432x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 10.0507 ms | not_measured | managed | 1.128x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.0079 ms | not_measured | managed | 0.861x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.3259 ms | not_measured | managed | 1.235x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.0022 ms | not_measured | managed | 0.864x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1473 ms | not_measured | managed | 0.788x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 19.1650 ms | not_measured | managed | 1.184x |
| `np.argmax (float16)` | float16 | 1,000 | 0.0029 ms | not_measured | managed | 1.069x |
| `np.argmax (float16)` | float16 | 100,000 | 0.2180 ms | not_measured | managed | 1.985x |
| `np.argmax (float16)` | float16 | 10,000,000 | 21.4268 ms | not_measured | managed | 3.362x |
| `np.argmax (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmax (float32)` | float32 | 100,000 | 0.0587 ms | not_measured | managed | 0.153x |
| `np.argmax (float32)` | float32 | 10,000,000 | 6.7330 ms | not_measured | managed | 0.747x |
| `np.argmax (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (float64)` | float64 | 100,000 | 0.0584 ms | not_measured | managed | 0.296x |
| `np.argmax (float64)` | float64 | 10,000,000 | 8.9012 ms | not_measured | managed | 1.145x |
| `np.argmax (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `np.argmax (int16)` | int16 | 100,000 | 0.0032 ms | not_measured | managed | 1.188x |
| `np.argmax (int16)` | int16 | 10,000,000 | 0.7236 ms | not_measured | managed | 1.871x |
| `np.argmax (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmax (int32)` | int32 | 100,000 | 0.0057 ms | not_measured | managed | 1.105x |
| `np.argmax (int32)` | int32 | 10,000,000 | 3.6440 ms | not_measured | managed | 1.317x |
| `np.argmax (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `np.argmax (int64)` | int64 | 100,000 | 0.0529 ms | not_measured | managed | 0.267x |
| `np.argmax (int64)` | int64 | 10,000,000 | 8.3276 ms | not_measured | managed | 1.319x |
| `np.argmax (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `np.argmax (int8)` | int8 | 100,000 | 0.0023 ms | not_measured | managed | 1.043x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.2447 ms | not_measured | managed | 2.429x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.0033 ms | not_measured | managed | 1.515x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 0.7659 ms | not_measured | managed | 2.864x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.0058 ms | not_measured | managed | 1.517x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 3.5760 ms | not_measured | managed | 1.441x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.0608 ms | not_measured | managed | 0.280x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 8.7310 ms | not_measured | managed | 1.374x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.0023 ms | not_measured | managed | 1.304x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.2388 ms | not_measured | managed | 3.964x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.0022 ms | not_measured | managed | 0.864x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1457 ms | not_measured | managed | 0.786x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 19.3506 ms | not_measured | managed | 1.137x |
| `np.argmin (float16)` | float16 | 1,000 | 0.0029 ms | not_measured | managed | 1.000x |
| `np.argmin (float16)` | float16 | 100,000 | 0.2176 ms | not_measured | managed | 1.880x |
| `np.argmin (float16)` | float16 | 10,000,000 | 21.5659 ms | not_measured | managed | 3.214x |
| `np.argmin (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmin (float32)` | float32 | 100,000 | 0.0580 ms | not_measured | managed | 0.153x |
| `np.argmin (float32)` | float32 | 10,000,000 | 6.5660 ms | not_measured | managed | 0.757x |
| `np.argmin (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmin (float64)` | float64 | 100,000 | 0.0586 ms | not_measured | managed | 0.295x |
| `np.argmin (float64)` | float64 | 10,000,000 | 8.9070 ms | not_measured | managed | 1.137x |
| `np.argmin (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmin (int16)` | int16 | 100,000 | 0.0031 ms | not_measured | managed | 1.129x |
| `np.argmin (int16)` | int16 | 10,000,000 | 0.7434 ms | not_measured | managed | 1.761x |
| `np.argmin (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.argmin (int32)` | int32 | 100,000 | 0.0056 ms | not_measured | managed | 1.018x |
| `np.argmin (int32)` | int32 | 10,000,000 | 3.6560 ms | not_measured | managed | 1.282x |
| `np.argmin (int64)` | int64 | 1,000 | 0.0037 ms | not_measured | managed | 0.243x |
| `np.argmin (int64)` | int64 | 100,000 | 0.0517 ms | not_measured | managed | 0.271x |
| `np.argmin (int64)` | int64 | 10,000,000 | 8.5333 ms | not_measured | managed | 1.288x |
| `np.argmin (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `np.argmin (int8)` | int8 | 100,000 | 0.0023 ms | not_measured | managed | 1.000x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.2476 ms | not_measured | managed | 2.315x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.0031 ms | not_measured | managed | 1.613x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 0.7371 ms | not_measured | managed | 3.234x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.0055 ms | not_measured | managed | 1.600x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 3.5349 ms | not_measured | managed | 1.436x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.0591 ms | not_measured | managed | 0.288x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 8.8719 ms | not_measured | managed | 1.166x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.0023 ms | not_measured | managed | 1.304x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.2524 ms | not_measured | managed | 3.764x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.0167 ms | not_measured | managed | 0.910x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 1.6204 ms | not_measured | managed | 0.235x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 163.1712 ms | not_measured | managed | 0.574x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.0497 ms | not_measured | managed | 0.386x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 6.7023 ms | not_measured | managed | 0.365x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 657.5351 ms | not_measured | managed | 1.215x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.0072 ms | not_measured | managed | 0.403x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 6.3835 ms | not_measured | managed | 1.146x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 670.6892 ms | not_measured | managed | 1.171x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.0038 ms | not_measured | managed | 0.816x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.3263 ms | not_measured | managed | 1.055x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 41.9600 ms | not_measured | managed | 1.553x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.0172 ms | not_measured | managed | 0.349x |
| `np.cumsum (float16)` | float16 | 100,000 | 1.6813 ms | not_measured | managed | 0.277x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 165.6600 ms | not_measured | managed | 0.590x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 1.556x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.1206 ms | not_measured | managed | 1.318x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 11.4729 ms | not_measured | managed | 3.000x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.757x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.2115 ms | not_measured | managed | 0.782x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 21.6011 ms | not_measured | managed | 2.233x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.0023 ms | not_measured | managed | 1.087x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.2162 ms | not_measured | managed | 1.509x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 15.1643 ms | not_measured | managed | 3.250x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 1.182x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1882 ms | not_measured | managed | 1.773x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 19.0520 ms | not_measured | managed | 2.703x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 0.810x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1709 ms | not_measured | managed | 0.175x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 21.7224 ms | not_measured | managed | 1.601x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.0031 ms | not_measured | managed | 0.806x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.2480 ms | not_measured | managed | 1.321x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 14.4197 ms | not_measured | managed | 3.311x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 1.286x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.2124 ms | not_measured | managed | 1.637x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 15.9418 ms | not_measured | managed | 3.262x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 1.143x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1951 ms | not_measured | managed | 1.670x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 17.0458 ms | not_measured | managed | 3.014x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 0.810x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1667 ms | not_measured | managed | 0.178x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 21.0507 ms | not_measured | managed | 1.363x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.0027 ms | not_measured | managed | 0.926x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.2112 ms | not_measured | managed | 1.554x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 14.7416 ms | not_measured | managed | 3.003x |
| `np.max (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.348x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1462 ms | not_measured | managed | 1.033x |
| `np.max (complex128)` | complex128 | 10,000,000 | 19.4857 ms | not_measured | managed | 1.266x |
| `np.max (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 1.565x |
| `np.max (float16)` | float16 | 100,000 | 0.4377 ms | not_measured | managed | 1.145x |
| `np.max (float16)` | float16 | 10,000,000 | 46.3558 ms | not_measured | managed | 1.604x |
| `np.max (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.231x |
| `np.max (float32)` | float32 | 100,000 | 0.0086 ms | not_measured | managed | 0.744x |
| `np.max (float32)` | float32 | 10,000,000 | 3.6158 ms | not_measured | managed | 1.201x |
| `np.max (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 1.231x |
| `np.max (float64)` | float64 | 100,000 | 0.0168 ms | not_measured | managed | 0.643x |
| `np.max (float64)` | float64 | 10,000,000 | 8.2030 ms | not_measured | managed | 1.070x |
| `np.max (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.max (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 1.143x |
| `np.max (int16)` | int16 | 10,000,000 | 0.6750 ms | not_measured | managed | 0.977x |
| `np.max (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.max (int32)` | int32 | 100,000 | 0.0043 ms | not_measured | managed | 1.209x |
| `np.max (int32)` | int32 | 10,000,000 | 3.2775 ms | not_measured | managed | 1.230x |
| `np.max (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 1.133x |
| `np.max (int64)` | int64 | 100,000 | 0.0270 ms | not_measured | managed | 0.330x |
| `np.max (int64)` | int64 | 10,000,000 | 8.3643 ms | not_measured | managed | 1.328x |
| `np.max (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.max (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 1.211x |
| `np.max (int8)` | int8 | 10,000,000 | 0.2565 ms | not_measured | managed | 1.144x |
| `np.max (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 1.917x |
| `np.max (uint16)` | uint16 | 100,000 | 0.0028 ms | not_measured | managed | 1.071x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.6791 ms | not_measured | managed | 1.371x |
| `np.max (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.max (uint32)` | uint32 | 100,000 | 0.0045 ms | not_measured | managed | 0.933x |
| `np.max (uint32)` | uint32 | 10,000,000 | 3.2714 ms | not_measured | managed | 0.464x |
| `np.max (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.max (uint64)` | uint64 | 100,000 | 0.0365 ms | not_measured | managed | 0.326x |
| `np.max (uint64)` | uint64 | 10,000,000 | 8.1652 ms | not_measured | managed | 1.348x |
| `np.max (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.max (uint8)` | uint8 | 100,000 | 0.0020 ms | not_measured | managed | 1.200x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.2421 ms | not_measured | managed | 1.312x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.0016 ms | not_measured | managed | 1.625x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.0220 ms | not_measured | managed | 1.459x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 16.7710 ms | not_measured | managed | 1.143x |
| `np.mean (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 2.238x |
| `np.mean (float16)` | float16 | 100,000 | 0.1636 ms | not_measured | managed | 0.620x |
| `np.mean (float16)` | float16 | 10,000,000 | 16.2840 ms | not_measured | managed | 1.087x |
| `np.mean (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 3.000x |
| `np.mean (float32)` | float32 | 100,000 | 0.0051 ms | not_measured | managed | 3.588x |
| `np.mean (float32)` | float32 | 10,000,000 | 3.3387 ms | not_measured | managed | 3.389x |
| `np.mean (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 1.846x |
| `np.mean (float64)` | float64 | 100,000 | 0.0086 ms | not_measured | managed | 2.198x |
| `np.mean (float64)` | float64 | 10,000,000 | 7.3631 ms | not_measured | managed | 2.438x |
| `np.mean (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 2.500x |
| `np.mean (int16)` | int16 | 100,000 | 0.0443 ms | not_measured | managed | 1.178x |
| `np.mean (int16)` | int16 | 10,000,000 | 4.5044 ms | not_measured | managed | 2.124x |
| `np.mean (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 2.500x |
| `np.mean (int32)` | int32 | 100,000 | 0.0455 ms | not_measured | managed | 0.960x |
| `np.mean (int32)` | int32 | 10,000,000 | 5.3051 ms | not_measured | managed | 1.867x |
| `np.mean (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 2.333x |
| `np.mean (int64)` | int64 | 100,000 | 0.0088 ms | not_measured | managed | 3.943x |
| `np.mean (int64)` | int64 | 10,000,000 | 7.4994 ms | not_measured | managed | 1.932x |
| `np.mean (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 2.500x |
| `np.mean (int8)` | int8 | 100,000 | 0.0453 ms | not_measured | managed | 1.201x |
| `np.mean (int8)` | int8 | 10,000,000 | 4.6523 ms | not_measured | managed | 3.025x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.727x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.0418 ms | not_measured | managed | 1.321x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 4.2267 ms | not_measured | managed | 2.399x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 1.933x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.0393 ms | not_measured | managed | 1.018x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 5.0930 ms | not_measured | managed | 1.711x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.0012 ms | not_measured | managed | 2.417x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.0096 ms | not_measured | managed | 5.573x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 7.6024 ms | not_measured | managed | 1.582x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 2.727x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.0411 ms | not_measured | managed | 1.294x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 3.7184 ms | not_measured | managed | 2.266x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.0045 ms | not_measured | managed | 0.778x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.0453 ms | not_measured | managed | 0.618x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 16.9481 ms | not_measured | managed | 1.157x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.0068 ms | not_measured | managed | 0.824x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.2802 ms | not_measured | managed | 0.278x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 27.2283 ms | not_measured | managed | 0.663x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.0033 ms | not_measured | managed | 1.061x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.0177 ms | not_measured | managed | 0.644x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 3.7939 ms | not_measured | managed | 1.169x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.744x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.0280 ms | not_measured | managed | 0.511x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 8.7225 ms | not_measured | managed | 1.275x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.0010 ms | not_measured | managed | 3.900x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.0189 ms | not_measured | managed | 2.762x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 2.0347 ms | not_measured | managed | 3.732x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 3.900x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.0193 ms | not_measured | managed | 1.855x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 3.9741 ms | not_measured | managed | 2.540x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 1.714x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.1460 ms | not_measured | managed | 0.229x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 114.3394 ms | not_measured | managed | 0.081x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 3.545x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.0190 ms | not_measured | managed | 2.537x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 1.8142 ms | not_measured | managed | 4.691x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 2.375x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.0191 ms | not_measured | managed | 2.686x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 2.0286 ms | not_measured | managed | 5.044x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 3.364x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.0251 ms | not_measured | managed | 1.685x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 4.1907 ms | not_measured | managed | 2.411x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.0025 ms | not_measured | managed | 1.560x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.1583 ms | not_measured | managed | 0.361x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 119.3040 ms | not_measured | managed | 0.127x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 3.800x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.0191 ms | not_measured | managed | 2.911x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 1.7713 ms | not_measured | managed | 4.516x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.0047 ms | not_measured | managed | 0.660x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.0536 ms | not_measured | managed | 0.679x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 18.1618 ms | not_measured | managed | 1.024x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.0067 ms | not_measured | managed | 0.746x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.2710 ms | not_measured | managed | 0.310x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 26.0187 ms | not_measured | managed | 0.718x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.0034 ms | not_measured | managed | 1.029x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.0232 ms | not_measured | managed | 0.789x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 3.9328 ms | not_measured | managed | 2.055x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.732x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.0282 ms | not_measured | managed | 0.695x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 8.9008 ms | not_measured | managed | 1.648x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 2.917x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.0195 ms | not_measured | managed | 2.892x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 2.0290 ms | not_measured | managed | 4.659x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 3.182x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.0196 ms | not_measured | managed | 2.158x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 3.8722 ms | not_measured | managed | 2.814x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 1.619x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.1441 ms | not_measured | managed | 0.291x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 14.6226 ms | not_measured | managed | 0.970x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 2.917x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.0195 ms | not_measured | managed | 3.164x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 1.7997 ms | not_measured | managed | 4.366x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 3.500x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.0196 ms | not_measured | managed | 2.939x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 1.9660 ms | not_measured | managed | 4.906x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 3.091x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.0254 ms | not_measured | managed | 1.732x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 4.1422 ms | not_measured | managed | 2.607x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.0024 ms | not_measured | managed | 1.458x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.1565 ms | not_measured | managed | 0.392x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 15.9105 ms | not_measured | managed | 0.773x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 3.500x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.0196 ms | not_measured | managed | 2.883x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 1.7880 ms | not_measured | managed | 4.927x |
| `np.min (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.391x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1467 ms | not_measured | managed | 1.050x |
| `np.min (complex128)` | complex128 | 10,000,000 | 19.3300 ms | not_measured | managed | 1.344x |
| `np.min (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 1.696x |
| `np.min (float16)` | float16 | 100,000 | 0.4069 ms | not_measured | managed | 1.241x |
| `np.min (float16)` | float16 | 10,000,000 | 42.2388 ms | not_measured | managed | 1.938x |
| `np.min (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.min (float32)` | float32 | 100,000 | 0.0089 ms | not_measured | managed | 0.685x |
| `np.min (float32)` | float32 | 10,000,000 | 3.7886 ms | not_measured | managed | 1.397x |
| `np.min (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 1.462x |
| `np.min (float64)` | float64 | 100,000 | 0.0175 ms | not_measured | managed | 0.600x |
| `np.min (float64)` | float64 | 10,000,000 | 8.5034 ms | not_measured | managed | 1.044x |
| `np.min (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.min (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 1.071x |
| `np.min (int16)` | int16 | 10,000,000 | 0.7077 ms | not_measured | managed | 0.968x |
| `np.min (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.min (int32)` | int32 | 100,000 | 0.0048 ms | not_measured | managed | 0.896x |
| `np.min (int32)` | int32 | 10,000,000 | 3.2775 ms | not_measured | managed | 1.257x |
| `np.min (int64)` | int64 | 1,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `np.min (int64)` | int64 | 100,000 | 0.0268 ms | not_measured | managed | 0.354x |
| `np.min (int64)` | int64 | 10,000,000 | 8.3687 ms | not_measured | managed | 1.078x |
| `np.min (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 1.364x |
| `np.min (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 1.211x |
| `np.min (int8)` | int8 | 10,000,000 | 0.2397 ms | not_measured | managed | 1.141x |
| `np.min (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 1.333x |
| `np.min (uint16)` | uint16 | 100,000 | 0.0029 ms | not_measured | managed | 1.448x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.6885 ms | not_measured | managed | 0.425x |
| `np.min (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 1.583x |
| `np.min (uint32)` | uint32 | 100,000 | 0.0044 ms | not_measured | managed | 0.955x |
| `np.min (uint32)` | uint32 | 10,000,000 | 3.3482 ms | not_measured | managed | 1.288x |
| `np.min (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.min (uint64)` | uint64 | 100,000 | 0.0366 ms | not_measured | managed | 0.328x |
| `np.min (uint64)` | uint64 | 10,000,000 | 8.2811 ms | not_measured | managed | 1.417x |
| `np.min (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.min (uint8)` | uint8 | 100,000 | 0.0020 ms | not_measured | managed | 1.150x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.2418 ms | not_measured | managed | 1.156x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.0057 ms | not_measured | managed | 2.246x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.2725 ms | not_measured | managed | 1.937x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 28.1648 ms | not_measured | managed | 3.488x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.0037 ms | not_measured | managed | 1.649x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.1172 ms | not_measured | managed | 0.234x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 14.4077 ms | not_measured | managed | 1.542x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.0038 ms | not_measured | managed | 1.684x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.1161 ms | not_measured | managed | 1.213x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 21.3726 ms | not_measured | managed | 2.012x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.0054 ms | not_measured | managed | 1.630x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.2696 ms | not_measured | managed | 1.851x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 28.7859 ms | not_measured | managed | 3.342x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.0037 ms | not_measured | managed | 1.649x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.1154 ms | not_measured | managed | 0.248x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 14.2928 ms | not_measured | managed | 1.765x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 1.622x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.1157 ms | not_measured | managed | 1.086x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 20.8469 ms | not_measured | managed | 2.036x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 2.167x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.4238 ms | not_measured | managed | 1.174x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 47.0978 ms | not_measured | managed | 1.633x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 2.500x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.0514 ms | not_measured | managed | 0.140x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 6.0021 ms | not_measured | managed | 0.675x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 1.579x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.1058 ms | not_measured | managed | 0.112x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 12.7438 ms | not_measured | managed | 0.749x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.0028 ms | not_measured | managed | 4.714x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.2035 ms | not_measured | managed | 1.614x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 21.0167 ms | not_measured | managed | 3.178x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 5.105x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.0718 ms | not_measured | managed | 1.061x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 7.5240 ms | not_measured | managed | 6.070x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 4.421x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.0738 ms | not_measured | managed | 1.183x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 10.4202 ms | not_measured | managed | 6.301x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.0123 ms | not_measured | managed | 1.350x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.6992 ms | not_measured | managed | 0.573x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 128.0458 ms | not_measured | managed | 1.516x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.0050 ms | not_measured | managed | 2.560x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 0.3544 ms | not_measured | managed | 1.354x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 53.3593 ms | not_measured | managed | 2.074x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 2.750x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.3810 ms | not_measured | managed | 2.340x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 64.9356 ms | not_measured | managed | 2.028x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 2.125x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.4232 ms | not_measured | managed | 1.205x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 44.9680 ms | not_measured | managed | 1.740x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 2.417x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.0522 ms | not_measured | managed | 0.138x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 6.2408 ms | not_measured | managed | 0.818x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 1.526x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.1052 ms | not_measured | managed | 0.108x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 12.5390 ms | not_measured | managed | 0.814x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.0122 ms | not_measured | managed | 2.828x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.6889 ms | not_measured | managed | 1.094x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 128.1226 ms | not_measured | managed | 1.591x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.0048 ms | not_measured | managed | 5.396x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.3609 ms | not_measured | managed | 1.982x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 52.7205 ms | not_measured | managed | 1.517x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 5.265x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.3689 ms | not_measured | managed | 3.287x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 64.7813 ms | not_measured | managed | 1.582x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 9.391x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.2036 ms | not_measured | managed | 1.069x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 19.6099 ms | not_measured | managed | 2.425x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 12.875x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 1.9742 ms | not_measured | managed | 1.207x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 203.8725 ms | not_measured | managed | 3.903x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 4.300x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 3.2518 ms | not_measured | managed | 1.611x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 344.3129 ms | not_measured | managed | 2.468x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.0122 ms | not_measured | managed | 2.492x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6875 ms | not_measured | managed | 1.090x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 129.7749 ms | not_measured | managed | 1.568x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.0048 ms | not_measured | managed | 5.271x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.3682 ms | not_measured | managed | 1.917x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 53.1184 ms | not_measured | managed | 1.460x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 5.208x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3674 ms | not_measured | managed | 3.245x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 65.0209 ms | not_measured | managed | 1.525x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.0057 ms | not_measured | managed | 5.456x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.3970 ms | not_measured | managed | 2.953x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 41.3113 ms | not_measured | managed | 5.969x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 7.222x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.1489 ms | not_measured | managed | 1.124x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 16.0975 ms | not_measured | managed | 4.897x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 7.500x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.1456 ms | not_measured | managed | 1.296x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 19.9158 ms | not_measured | managed | 5.422x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 2.696x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.1884 ms | not_measured | managed | 1.490x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 19.0367 ms | not_measured | managed | 2.784x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 2.312x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.0094 ms | not_measured | managed | 3.309x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 4.1024 ms | not_measured | managed | 7.935x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 2.643x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.0187 ms | not_measured | managed | 2.422x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 9.0631 ms | not_measured | managed | 5.800x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.0056 ms | not_measured | managed | 5.375x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.4001 ms | not_measured | managed | 2.901x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 43.3223 ms | not_measured | managed | 6.293x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 7.269x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.1491 ms | not_measured | managed | 1.064x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 15.9214 ms | not_measured | managed | 4.637x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 8.625x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.1451 ms | not_measured | managed | 1.355x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 20.2653 ms | not_measured | managed | 5.495x |
| `np.prod (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 2.154x |
| `np.prod (float64)` | float64 | 100,000 | 1.1493 ms | not_measured | managed | 2.732x |
| `np.prod (float64)` | float64 | 10,000,000 | 432.8116 ms | not_measured | managed | 1.807x |
| `np.prod (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `np.prod (int64)` | int64 | 100,000 | 0.0412 ms | not_measured | managed | 3.034x |
| `np.prod (int64)` | int64 | 10,000,000 | 8.8045 ms | not_measured | managed | 1.632x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.714x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.0200 ms | not_measured | managed | 1.145x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 115.4239 ms | not_measured | managed | 1.183x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.0411 ms | not_measured | managed | 1.479x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 8.8008 ms | not_measured | managed | 1.183x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.0116 ms | not_measured | managed | 11.655x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 8.3839 ms | not_measured | managed | 24.549x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 1.000x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.0465 ms | not_measured | managed | 2.841x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 8.7016 ms | not_measured | managed | 1.628x |
| `np.std (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 4.089x |
| `np.std (float16)` | float16 | 100,000 | 0.3964 ms | not_measured | managed | 2.247x |
| `np.std (float16)` | float16 | 10,000,000 | 39.1442 ms | not_measured | managed | 4.907x |
| `np.std (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 4.500x |
| `np.std (float32)` | float32 | 100,000 | 0.0194 ms | not_measured | managed | 2.624x |
| `np.std (float32)` | float32 | 10,000,000 | 10.8179 ms | not_measured | managed | 3.414x |
| `np.std (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 5.077x |
| `np.std (float64)` | float64 | 100,000 | 0.0386 ms | not_measured | managed | 1.562x |
| `np.std (float64)` | float64 | 10,000,000 | 17.6173 ms | not_measured | managed | 4.016x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.0116 ms | not_measured | managed | 1.741x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.8486 ms | not_measured | managed | 1.283x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 198.9534 ms | not_measured | managed | 1.412x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.0045 ms | not_measured | managed | 1.889x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.0675 ms | not_measured | managed | 0.578x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 9.9118 ms | not_measured | managed | 3.039x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 4.944x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.0466 ms | not_measured | managed | 1.500x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 17.5379 ms | not_measured | managed | 3.611x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.0224 ms | not_measured | managed | 1.496x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 16.2803 ms | not_measured | managed | 1.204x |
| `np.sum (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 2.400x |
| `np.sum (float16)` | float16 | 100,000 | 0.0692 ms | not_measured | managed | 2.921x |
| `np.sum (float16)` | float16 | 10,000,000 | 6.7499 ms | not_measured | managed | 4.048x |
| `np.sum (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `np.sum (float32)` | float32 | 100,000 | 0.0060 ms | not_measured | managed | 2.733x |
| `np.sum (float32)` | float32 | 10,000,000 | 3.5366 ms | not_measured | managed | 4.013x |
| `np.sum (float64)` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.333x |
| `np.sum (float64)` | float64 | 100,000 | 0.1965 ms | not_measured | managed | 0.084x |
| `np.sum (float64)` | float64 | 10,000,000 | 13.0861 ms | not_measured | managed | 1.302x |
| `np.sum (int16)` | int16 | 1,000 | 0.0012 ms | not_measured | managed | 2.417x |
| `np.sum (int16)` | int16 | 100,000 | 0.0444 ms | not_measured | managed | 0.745x |
| `np.sum (int16)` | int16 | 10,000,000 | 4.8896 ms | not_measured | managed | 1.693x |
| `np.sum (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 1.000x |
| `np.sum (int32)` | int32 | 100,000 | 0.0469 ms | not_measured | managed | 0.744x |
| `np.sum (int32)` | int32 | 10,000,000 | 5.5298 ms | not_measured | managed | 1.642x |
| `np.sum (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 1.062x |
| `np.sum (int64)` | int64 | 100,000 | 0.0089 ms | not_measured | managed | 2.202x |
| `np.sum (int64)` | int64 | 10,000,000 | 7.7586 ms | not_measured | managed | 1.234x |
| `np.sum (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `np.sum (int8)` | int8 | 100,000 | 0.0476 ms | not_measured | managed | 0.689x |
| `np.sum (int8)` | int8 | 10,000,000 | 4.8994 ms | not_measured | managed | 2.042x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 1.909x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.0392 ms | not_measured | managed | 0.852x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 4.3095 ms | not_measured | managed | 1.837x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.0019 ms | not_measured | managed | 1.105x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.0440 ms | not_measured | managed | 0.743x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 5.3606 ms | not_measured | managed | 1.740x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.0090 ms | not_measured | managed | 2.211x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 8.3894 ms | not_measured | managed | 1.305x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 2.000x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.0407 ms | not_measured | managed | 0.845x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 3.7122 ms | not_measured | managed | 2.241x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.0041 ms | not_measured | managed | 0.512x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.0406 ms | not_measured | managed | 0.608x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 17.9939 ms | not_measured | managed | 1.142x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.0077 ms | not_measured | managed | 0.649x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1531 ms | not_measured | managed | 1.917x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 13.9083 ms | not_measured | managed | 5.144x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.0039 ms | not_measured | managed | 0.487x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.0154 ms | not_measured | managed | 0.617x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 4.5630 ms | not_measured | managed | 1.036x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.0037 ms | not_measured | managed | 0.514x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.0232 ms | not_measured | managed | 0.591x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 8.9235 ms | not_measured | managed | 1.187x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 1.263x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.0101 ms | not_measured | managed | 4.614x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 1.3052 ms | not_measured | managed | 5.834x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 1.923x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.0148 ms | not_measured | managed | 3.236x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 4.2797 ms | not_measured | managed | 3.055x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 1.250x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.0200 ms | not_measured | managed | 1.365x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 9.0723 ms | not_measured | managed | 1.028x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 1.333x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.0097 ms | not_measured | managed | 4.794x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.7991 ms | not_measured | managed | 13.913x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 1.263x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.0101 ms | not_measured | managed | 4.594x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 2.3995 ms | not_measured | managed | 5.573x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 2.182x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.0148 ms | not_measured | managed | 3.149x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 4.3908 ms | not_measured | managed | 2.777x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.818x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.0201 ms | not_measured | managed | 1.353x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 9.0387 ms | not_measured | managed | 1.150x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.0017 ms | not_measured | managed | 1.471x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.0097 ms | not_measured | managed | 5.216x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.7581 ms | not_measured | managed | 14.284x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.0042 ms | not_measured | managed | 0.476x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.0500 ms | not_measured | managed | 0.686x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 18.5878 ms | not_measured | managed | 1.100x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.0072 ms | not_measured | managed | 0.528x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.0845 ms | not_measured | managed | 2.488x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 6.8456 ms | not_measured | managed | 4.742x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.0036 ms | not_measured | managed | 0.528x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.0207 ms | not_measured | managed | 0.792x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 4.5761 ms | not_measured | managed | 2.156x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.487x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.0235 ms | not_measured | managed | 0.753x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 9.1455 ms | not_measured | managed | 1.726x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 2.091x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.0073 ms | not_measured | managed | 4.986x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 1.2739 ms | not_measured | managed | 6.531x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 2.500x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.0104 ms | not_measured | managed | 3.615x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 4.9116 ms | not_measured | managed | 1.896x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.0117 ms | not_measured | managed | 1.556x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 10.0112 ms | not_measured | managed | 0.922x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 2.091x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.0071 ms | not_measured | managed | 5.394x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.5039 ms | not_measured | managed | 17.467x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 2.091x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.0072 ms | not_measured | managed | 5.056x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 1.7307 ms | not_measured | managed | 5.810x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.0010 ms | not_measured | managed | 2.300x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.0098 ms | not_measured | managed | 3.714x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 5.7486 ms | not_measured | managed | 1.628x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.0122 ms | not_measured | managed | 1.352x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 9.0228 ms | not_measured | managed | 1.154x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 2.500x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.0073 ms | not_measured | managed | 5.178x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.5054 ms | not_measured | managed | 16.839x |
| `np.var (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 3.933x |
| `np.var (float16)` | float16 | 100,000 | 0.3902 ms | not_measured | managed | 2.301x |
| `np.var (float16)` | float16 | 10,000,000 | 40.2336 ms | not_measured | managed | 4.664x |
| `np.var (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 7.091x |
| `np.var (float32)` | float32 | 100,000 | 0.0189 ms | not_measured | managed | 2.413x |
| `np.var (float32)` | float32 | 10,000,000 | 8.6727 ms | not_measured | managed | 4.026x |
| `np.var (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 4.571x |
| `np.var (float64)` | float64 | 100,000 | 0.0364 ms | not_measured | managed | 1.676x |
| `np.var (float64)` | float64 | 10,000,000 | 20.3768 ms | not_measured | managed | 3.507x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.0112 ms | not_measured | managed | 1.929x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.8299 ms | not_measured | managed | 1.313x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 199.6234 ms | not_measured | managed | 1.441x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.0041 ms | not_measured | managed | 2.049x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.0656 ms | not_measured | managed | 0.587x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 10.2918 ms | not_measured | managed | 3.190x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 5.071x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.0455 ms | not_measured | managed | 1.433x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 17.5236 ms | not_measured | managed | 3.788x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.0072 ms | not_measured | managed | 0.819x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.3110 ms | not_measured | managed | 1.922x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.962x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.1359 ms | not_measured | managed | 0.713x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 2.316x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.0024 ms | not_measured | managed | 2.000x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 1.280x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.1365 ms | not_measured | managed | 0.714x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.667x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.4648 ms | not_measured | managed | 0.764x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.0145 ms | not_measured | managed | 0.600x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.7025 ms | not_measured | managed | 1.083x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 2.000x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.3169 ms | not_measured | managed | 1.004x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.0058 ms | not_measured | managed | 0.362x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.1553 ms | not_measured | managed | 0.724x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.517x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.2274 ms | not_measured | managed | 0.318x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 2.441x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.1941 ms | not_measured | managed | 3.714x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.0031 ms | not_measured | managed | 0.484x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.1404 ms | not_measured | managed | 0.417x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.792x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.0995 ms | not_measured | managed | 0.277x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 1.964x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.1265 ms | not_measured | managed | 0.572x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 2.159x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.1214 ms | not_measured | managed | 0.596x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.818x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1932 ms | not_measured | managed | 0.526x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.1407 ms | not_measured | managed | 0.212x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 10.5911 ms | not_measured | managed | 1.331x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.515x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.1834 ms | not_measured | managed | 0.220x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 38.2127 ms | not_measured | managed | 0.742x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.100x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.0022 ms | not_measured | managed | 0.091x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.0019 ms | not_measured | managed | 0.105x |
| `a[100:1000] * 2` | float64 | 1,000 | 0.0077 ms | not_measured | managed | 0.117x |
| `a[100:1000] * 2` | float64 | 100,000 | 0.0079 ms | not_measured | managed | 0.127x |
| `a[100:1000] * 2` | float64 | 10,000,000 | 0.0096 ms | not_measured | managed | 0.104x |
| `a[100:1000].copy()` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.078x |
| `a[100:1000].copy()` | float64 | 100,000 | 0.0074 ms | not_measured | managed | 0.068x |
| `a[100:1000].copy()` | float64 | 10,000,000 | 0.0071 ms | not_measured | managed | 0.070x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.069x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.0030 ms | not_measured | managed | 0.067x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.0033 ms | not_measured | managed | 0.061x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.071x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.0029 ms | not_measured | managed | 0.069x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.0029 ms | not_measured | managed | 0.069x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.048x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 0.048x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.0021 ms | not_measured | managed | 0.048x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.105x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.0020 ms | not_measured | managed | 0.100x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.0021 ms | not_measured | managed | 0.095x |
| `a[::2] * 2` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 0.107x |
| `a[::2] * 2` | float64 | 100,000 | 0.2743 ms | not_measured | managed | 0.055x |
| `a[::2] * 2` | float64 | 10,000,000 | 24.1133 ms | not_measured | managed | 0.410x |
| `np.copy(a[100:1000])` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.091x |
| `np.copy(a[100:1000])` | float64 | 100,000 | 0.0068 ms | not_measured | managed | 0.103x |
| `np.copy(a[100:1000])` | float64 | 10,000,000 | 0.0069 ms | not_measured | managed | 0.101x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.0142 ms | not_measured | managed | 0.324x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 1.0098 ms | not_measured | managed | 0.143x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.0197 ms | not_measured | managed | 0.218x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.7554 ms | not_measured | managed | 0.461x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.0126 ms | not_measured | managed | 0.270x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 1.0814 ms | not_measured | managed | 0.111x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.0125 ms | not_measured | managed | 0.296x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 1.3041 ms | not_measured | managed | 0.104x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.0308 ms | not_measured | managed | 0.377x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.8744 ms | not_measured | managed | 0.841x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 242.2219 ms | not_measured | managed | 5.699x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.0362 ms | not_measured | managed | 0.287x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 3.5464 ms | not_measured | managed | 1.516x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 527.7505 ms | not_measured | managed | 3.189x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.0321 ms | not_measured | managed | 0.371x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 1.5398 ms | not_measured | managed | 0.272x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 217.8821 ms | not_measured | managed | 1.443x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.0419 ms | not_measured | managed | 0.315x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 2.6753 ms | not_measured | managed | 0.189x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 436.8174 ms | not_measured | managed | 1.120x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 1.654x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.1858 ms | not_measured | managed | 1.854x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 1.800x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.1705 ms | not_measured | managed | 1.968x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.868x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.1757 ms | not_measured | managed | 2.257x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 1.000x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.2021 ms | not_measured | managed | 1.873x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.0033 ms | not_measured | managed | 1.000x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.2720 ms | not_measured | managed | 1.133x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 1.029x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.2054 ms | not_measured | managed | 1.599x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.393x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.2840 ms | not_measured | managed | 0.357x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.667x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.2857 ms | not_measured | managed | 0.418x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.0509 ms | not_measured | managed | 0.972x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 3.9166 ms | not_measured | managed | 3.484x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.0662 ms | not_measured | managed | 0.625x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 7.5438 ms | not_measured | managed | 2.540x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.0591 ms | not_measured | managed | 0.687x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 3.4205 ms | not_measured | managed | 3.042x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.0686 ms | not_measured | managed | 0.593x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 5.9445 ms | not_measured | managed | 1.888x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.0056 ms | not_measured | managed | 0.482x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.1246 ms | not_measured | managed | 1.474x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 22.7602 ms | not_measured | managed | 1.878x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.0071 ms | not_measured | managed | 0.366x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.1304 ms | not_measured | managed | 2.352x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 31.8898 ms | not_measured | managed | 1.384x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.0082 ms | not_measured | managed | 0.207x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.1302 ms | not_measured | managed | 0.769x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 27.9948 ms | not_measured | managed | 1.094x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.0083 ms | not_measured | managed | 0.205x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.1319 ms | not_measured | managed | 0.851x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 31.4035 ms | not_measured | managed | 1.172x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.0122 ms | not_measured | managed | 0.189x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.7230 ms | not_measured | managed | 0.151x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.0136 ms | not_measured | managed | 0.235x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.6587 ms | not_measured | managed | 0.524x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.0085 ms | not_measured | managed | 0.200x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.5004 ms | not_measured | managed | 0.054x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.0102 ms | not_measured | managed | 0.216x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.6516 ms | not_measured | managed | 0.129x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.0143 ms | not_measured | managed | 0.531x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 2.9488 ms | not_measured | managed | 0.704x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 335.8515 ms | not_measured | managed | 1.003x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.0149 ms | not_measured | managed | 0.490x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 2.9649 ms | not_measured | managed | 0.893x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 335.3476 ms | not_measured | managed | 0.958x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.0111 ms | not_measured | managed | 1.676x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 2.9930 ms | not_measured | managed | 0.960x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 325.3454 ms | not_measured | managed | 1.783x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.0126 ms | not_measured | managed | 1.476x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 3.0458 ms | not_measured | managed | 0.946x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 329.9762 ms | not_measured | managed | 1.870x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.0159 ms | not_measured | managed | 0.189x |
| `np.sort(a) (float32)` | float32 | 100,000 | 1.1990 ms | not_measured | managed | 0.200x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.0268 ms | not_measured | managed | 0.134x |
| `np.sort(a) (float64)` | float64 | 100,000 | 1.8880 ms | not_measured | managed | 0.686x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.0099 ms | not_measured | managed | 0.232x |
| `np.sort(a) (int32)` | int32 | 100,000 | 1.2747 ms | not_measured | managed | 0.074x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.0267 ms | not_measured | managed | 0.213x |
| `np.sort(a) (int64)` | int64 | 100,000 | 2.6997 ms | not_measured | managed | 0.102x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.0218 ms | not_measured | managed | 0.138x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 1.4824 ms | not_measured | managed | 0.383x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.0345 ms | not_measured | managed | 0.128x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 2.3338 ms | not_measured | managed | 0.729x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.0245 ms | not_measured | managed | 0.122x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 1.6426 ms | not_measured | managed | 0.231x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.0378 ms | not_measured | managed | 0.172x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 2.7547 ms | not_measured | managed | 0.198x |
| `np.average(a) (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 2.619x |
| `np.average(a) (float16)` | float16 | 100,000 | 0.1518 ms | not_measured | managed | 0.715x |
| `np.average(a) (float16)` | float16 | 10,000,000 | 16.0725 ms | not_measured | managed | 1.089x |
| `np.average(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 3.077x |
| `np.average(a) (float32)` | float32 | 100,000 | 0.0056 ms | not_measured | managed | 3.161x |
| `np.average(a) (float32)` | float32 | 10,000,000 | 3.2942 ms | not_measured | managed | 3.054x |
| `np.average(a) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.929x |
| `np.average(a) (float64)` | float64 | 100,000 | 0.0088 ms | not_measured | managed | 1.977x |
| `np.average(a) (float64)` | float64 | 10,000,000 | 7.5822 ms | not_measured | managed | 1.999x |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.0044 ms | not_measured | managed | 0.364x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.3550 ms | not_measured | managed | 0.286x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | 0.0158 ms | not_measured | managed | 0.728x |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | 0.4447 ms | not_measured | managed | 2.322x |
| `np.convolve(a, v, mode='same')` | float64 | 100,000 | 0.1170 ms | 0.9291 ms | managed | 9.692x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | 0.0692 ms | not_measured | managed | 0.273x |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | 1.6385 ms | not_measured | managed | 0.381x |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | 0.0068 ms | not_measured | managed | 1.618x |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | 0.4346 ms | not_measured | managed | 2.393x |
| `np.correlate(a, v, mode='same')` | float64 | 100,000 | 0.1155 ms | 0.9218 ms | managed | 10.739x |
| `np.count_nonzero(a) (float16)` | float16 | 1,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.count_nonzero(a) (float16)` | float16 | 100,000 | 0.0809 ms | not_measured | managed | 1.828x |
| `np.count_nonzero(a) (float16)` | float16 | 10,000,000 | 7.8276 ms | not_measured | managed | 3.341x |
| `np.count_nonzero(a) (float32)` | float32 | 1,000 | 0.0004 ms | not_measured | managed | 1.500x |
| `np.count_nonzero(a) (float32)` | float32 | 100,000 | 0.0101 ms | not_measured | managed | 3.693x |
| `np.count_nonzero(a) (float32)` | float32 | 10,000,000 | 4.2777 ms | not_measured | managed | 1.920x |
| `np.count_nonzero(a) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.count_nonzero(a) (float64)` | float64 | 100,000 | 0.0181 ms | not_measured | managed | 2.166x |
| `np.count_nonzero(a) (float64)` | float64 | 10,000,000 | 9.1126 ms | not_measured | managed | 1.112x |
| `np.cov(a, b) (float64)` | float64 | 1,000 | 0.0527 ms | not_measured | managed | 0.269x |
| `np.cov(a, b) (float64)` | float64 | 100,000 | 1.6193 ms | not_measured | managed | 0.373x |
| `np.cross(a, b) (float64)` | float64 | 1,000 | 0.0763 ms | not_measured | managed | 0.176x |
| `np.cross(a, b) (float64)` | float64 | 100,000 | 1.0805 ms | not_measured | managed | 0.614x |
| `np.diff(a) (float64)` | float64 | 1,000 | 0.0060 ms | not_measured | managed | 0.283x |
| `np.diff(a) (float64)` | float64 | 100,000 | 0.1413 ms | not_measured | managed | 0.109x |
| `np.digitize(a, bins) (float64)` | float64 | 1,000 | 0.0204 ms | not_measured | managed | 1.118x |
| `np.digitize(a, bins) (float64)` | float64 | 100,000 | 4.5550 ms | not_measured | managed | 0.735x |
| `np.ediff1d(a) (float64)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 0.194x |
| `np.ediff1d(a) (float64)` | float64 | 100,000 | 0.1138 ms | not_measured | managed | 0.126x |
| `np.inner(a, b) (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.500x |
| `np.inner(a, b) (float64)` | float64 | 100,000 | 0.0203 ms | 0.0075 ms | openblas | 1.080x |
| `np.kron(a, b) (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 1.400x |
| `np.kron(a, b) (float64)` | float64 | 100,000 | 0.1110 ms | not_measured | managed | 0.423x |
| `np.median(a) (float16)` | float16 | 1,000 | 0.0123 ms | not_measured | managed | 1.106x |
| `np.median(a) (float16)` | float16 | 100,000 | 1.6642 ms | not_measured | managed | 0.552x |
| `np.median(a) (float16)` | float16 | 10,000,000 | 124.3057 ms | not_measured | managed | 1.132x |
| `np.median(a) (float32)` | float32 | 1,000 | 0.0041 ms | not_measured | managed | 2.707x |
| `np.median(a) (float32)` | float32 | 100,000 | 0.3027 ms | not_measured | managed | 1.583x |
| `np.median(a) (float32)` | float32 | 10,000,000 | 48.6859 ms | not_measured | managed | 2.037x |
| `np.median(a) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 2.333x |
| `np.median(a) (float64)` | float64 | 100,000 | 0.3106 ms | not_measured | managed | 1.571x |
| `np.median(a) (float64)` | float64 | 10,000,000 | 60.8506 ms | not_measured | managed | 1.899x |
| `np.percentile(a, 50) (float16)` | float16 | 1,000 | 0.0122 ms | not_measured | managed | 2.311x |
| `np.percentile(a, 50) (float16)` | float16 | 100,000 | 1.6572 ms | not_measured | managed | 1.077x |
| `np.percentile(a, 50) (float16)` | float16 | 10,000,000 | 123.4448 ms | not_measured | managed | 1.256x |
| `np.percentile(a, 50) (float32)` | float32 | 1,000 | 0.0041 ms | not_measured | managed | 5.756x |
| `np.percentile(a, 50) (float32)` | float32 | 100,000 | 0.1844 ms | not_measured | managed | 3.879x |
| `np.percentile(a, 50) (float32)` | float32 | 10,000,000 | 47.5559 ms | not_measured | managed | 1.370x |
| `np.percentile(a, 50) (float64)` | float64 | 1,000 | 0.0043 ms | not_measured | managed | 5.581x |
| `np.percentile(a, 50) (float64)` | float64 | 100,000 | 0.3298 ms | not_measured | managed | 2.139x |
| `np.percentile(a, 50) (float64)` | float64 | 10,000,000 | 60.0197 ms | not_measured | managed | 1.411x |
| `np.ptp(a) (float16)` | float16 | 1,000 | 0.0054 ms | not_measured | managed | 1.389x |
| `np.ptp(a) (float16)` | float16 | 100,000 | 0.8936 ms | not_measured | managed | 1.127x |
| `np.ptp(a) (float16)` | float16 | 10,000,000 | 89.3665 ms | not_measured | managed | 1.488x |
| `np.ptp(a) (float32)` | float32 | 1,000 | 0.0025 ms | not_measured | managed | 1.280x |
| `np.ptp(a) (float32)` | float32 | 100,000 | 0.0179 ms | not_measured | managed | 0.810x |
| `np.ptp(a) (float32)` | float32 | 10,000,000 | 7.5219 ms | not_measured | managed | 1.065x |
| `np.ptp(a) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 1.259x |
| `np.ptp(a) (float64)` | float64 | 100,000 | 0.0361 ms | not_measured | managed | 0.548x |
| `np.ptp(a) (float64)` | float64 | 10,000,000 | 16.8288 ms | not_measured | managed | 0.998x |
| `np.quantile(a, 0.5) (float16)` | float16 | 1,000 | 0.0117 ms | not_measured | managed | 2.359x |
| `np.quantile(a, 0.5) (float16)` | float16 | 100,000 | 1.6678 ms | not_measured | managed | 1.074x |
| `np.quantile(a, 0.5) (float16)` | float16 | 10,000,000 | 125.3833 ms | not_measured | managed | 1.240x |
| `np.quantile(a, 0.5) (float32)` | float32 | 1,000 | 0.0041 ms | not_measured | managed | 5.537x |
| `np.quantile(a, 0.5) (float32)` | float32 | 100,000 | 0.2878 ms | not_measured | managed | 2.485x |
| `np.quantile(a, 0.5) (float32)` | float32 | 10,000,000 | 49.2731 ms | not_measured | managed | 1.338x |
| `np.quantile(a, 0.5) (float64)` | float64 | 1,000 | 0.0043 ms | not_measured | managed | 5.209x |
| `np.quantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3086 ms | not_measured | managed | 2.264x |
| `np.quantile(a, 0.5) (float64)` | float64 | 10,000,000 | 60.9959 ms | not_measured | managed | 1.345x |
| `np.trace(a) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 1.625x |
| `np.trace(a) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 1.400x |
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.0095 ms | not_measured | managed | 0.442x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.8983 ms | not_measured | managed | 0.342x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 89.0537 ms | not_measured | managed | 0.809x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.286x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.0531 ms | not_measured | managed | 0.134x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 9.1038 ms | not_measured | managed | 1.594x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.310x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.1049 ms | not_measured | managed | 0.244x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 20.3563 ms | not_measured | managed | 1.416x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.0201 ms | not_measured | managed | 0.557x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 2.2267 ms | not_measured | managed | 0.418x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 223.6628 ms | not_measured | managed | 0.810x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.0042 ms | not_measured | managed | 0.286x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.1048 ms | not_measured | managed | 0.135x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 18.9740 ms | not_measured | managed | 1.605x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.288x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.2143 ms | not_measured | managed | 2.216x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 46.9175 ms | not_measured | managed | 1.336x |
| `np.abs (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.abs (float16)` | float16 | 100,000 | 0.0539 ms | not_measured | managed | 0.486x |
| `np.abs (float16)` | float16 | 10,000,000 | 6.0097 ms | not_measured | managed | 1.272x |
| `np.abs (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `np.abs (float32)` | float32 | 100,000 | 0.0648 ms | not_measured | managed | 0.100x |
| `np.abs (float32)` | float32 | 10,000,000 | 8.8411 ms | not_measured | managed | 1.589x |
| `np.abs (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.200x |
| `np.abs (float64)` | float64 | 100,000 | 0.1157 ms | not_measured | managed | 0.208x |
| `np.abs (float64)` | float64 | 10,000,000 | 19.4369 ms | not_measured | managed | 1.358x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 0.500x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.0537 ms | not_measured | managed | 0.499x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 5.9375 ms | not_measured | managed | 1.329x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.0489 ms | not_measured | managed | 0.131x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 7.4766 ms | not_measured | managed | 1.644x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.214x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.1016 ms | not_measured | managed | 0.236x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 28.7456 ms | not_measured | managed | 0.892x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.0197 ms | not_measured | managed | 0.457x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 1.9561 ms | not_measured | managed | 0.433x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 195.7513 ms | not_measured | managed | 0.856x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.0110 ms | not_measured | managed | 0.545x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 1.0458 ms | not_measured | managed | 0.521x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 103.1075 ms | not_measured | managed | 1.263x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.0142 ms | not_measured | managed | 0.479x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 1.2841 ms | not_measured | managed | 1.101x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 134.7672 ms | not_measured | managed | 1.263x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.0234 ms | not_measured | managed | 0.590x |
| `np.angle(a) (float16)` | float16 | 100,000 | 1.5265 ms | not_measured | managed | 0.735x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 160.9273 ms | not_measured | managed | 1.053x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.0209 ms | not_measured | managed | 0.201x |
| `np.angle(a) (float32)` | float32 | 100,000 | 1.1738 ms | not_measured | managed | 0.898x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 136.2011 ms | not_measured | managed | 0.830x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.0170 ms | not_measured | managed | 0.265x |
| `np.angle(a) (float64)` | float64 | 100,000 | 1.1371 ms | not_measured | managed | 0.945x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 119.7345 ms | not_measured | managed | 1.128x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.0165 ms | not_measured | managed | 0.436x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 2.0931 ms | not_measured | managed | 0.527x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 208.2670 ms | not_measured | managed | 0.918x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.0085 ms | not_measured | managed | 0.482x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 1.2682 ms | not_measured | managed | 0.623x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 130.2267 ms | not_measured | managed | 1.130x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.0098 ms | not_measured | managed | 0.408x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 1.3701 ms | not_measured | managed | 1.023x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 148.9016 ms | not_measured | managed | 1.139x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.0199 ms | not_measured | managed | 0.452x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 1.9352 ms | not_measured | managed | 0.433x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 197.2797 ms | not_measured | managed | 0.842x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.0108 ms | not_measured | managed | 0.574x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 1.0445 ms | not_measured | managed | 0.519x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 103.5303 ms | not_measured | managed | 1.251x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.0142 ms | not_measured | managed | 0.493x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 1.2973 ms | not_measured | managed | 1.120x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 135.0792 ms | not_measured | managed | 1.229x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.0173 ms | not_measured | managed | 0.434x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 2.1358 ms | not_measured | managed | 0.540x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 216.1600 ms | not_measured | managed | 0.912x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.0095 ms | not_measured | managed | 0.474x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 1.3212 ms | not_measured | managed | 0.626x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 133.8658 ms | not_measured | managed | 1.123x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.0099 ms | not_measured | managed | 0.424x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 1.4075 ms | not_measured | managed | 0.988x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 146.2234 ms | not_measured | managed | 1.236x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.0251 ms | not_measured | managed | 0.454x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 2.5039 ms | not_measured | managed | 0.487x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 254.7398 ms | not_measured | managed | 0.884x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.0150 ms | not_measured | managed | 0.527x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 1.5359 ms | not_measured | managed | 0.588x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 155.9902 ms | not_measured | managed | 1.237x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.0195 ms | not_measured | managed | 0.467x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 1.9801 ms | not_measured | managed | 1.321x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 206.5524 ms | not_measured | managed | 1.233x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.0160 ms | not_measured | managed | 0.494x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 2.3449 ms | not_measured | managed | 0.557x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 233.6849 ms | not_measured | managed | 0.915x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.0087 ms | not_measured | managed | 0.517x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 1.3977 ms | not_measured | managed | 0.700x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 141.1608 ms | not_measured | managed | 1.289x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.0104 ms | not_measured | managed | 0.385x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 1.5484 ms | not_measured | managed | 1.106x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 159.5703 ms | not_measured | managed | 1.142x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.0299 ms | not_measured | managed | 0.378x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 3.6646 ms | not_measured | managed | 0.478x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 369.7733 ms | not_measured | managed | 0.815x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.0262 ms | not_measured | managed | 0.176x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 3.2392 ms | not_measured | managed | 0.429x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 331.7892 ms | not_measured | managed | 0.697x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.0170 ms | not_measured | managed | 0.353x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 2.5269 ms | not_measured | managed | 1.103x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 257.8978 ms | not_measured | managed | 1.214x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.0206 ms | not_measured | managed | 0.466x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 2.1303 ms | not_measured | managed | 0.509x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 212.6427 ms | not_measured | managed | 0.916x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.0123 ms | not_measured | managed | 0.447x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 1.2865 ms | not_measured | managed | 0.622x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 128.8282 ms | not_measured | managed | 1.184x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.0129 ms | not_measured | managed | 0.488x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 1.4712 ms | not_measured | managed | 1.295x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 150.7087 ms | not_measured | managed | 1.176x |
| `np.around (float16)` | float16 | 1,000 | 0.0073 ms | not_measured | managed | 0.808x |
| `np.around (float16)` | float16 | 100,000 | 0.6829 ms | not_measured | managed | 0.615x |
| `np.around (float16)` | float16 | 10,000,000 | 66.9759 ms | not_measured | managed | 0.817x |
| `np.around (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.550x |
| `np.around (float32)` | float32 | 100,000 | 0.0522 ms | not_measured | managed | 0.132x |
| `np.around (float32)` | float32 | 10,000,000 | 7.6555 ms | not_measured | managed | 1.697x |
| `np.around (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.464x |
| `np.around (float64)` | float64 | 100,000 | 0.1067 ms | not_measured | managed | 0.257x |
| `np.around (float64)` | float64 | 10,000,000 | 19.5545 ms | not_measured | managed | 1.393x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.0251 ms | not_measured | managed | 0.446x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 2.4990 ms | not_measured | managed | 0.486x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 253.4573 ms | not_measured | managed | 0.890x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.0148 ms | not_measured | managed | 0.500x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 1.5602 ms | not_measured | managed | 0.583x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 155.7774 ms | not_measured | managed | 1.256x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.0197 ms | not_measured | managed | 0.482x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 2.0182 ms | not_measured | managed | 1.151x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 206.9368 ms | not_measured | managed | 1.233x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.0208 ms | not_measured | managed | 0.452x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 2.1020 ms | not_measured | managed | 0.507x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 211.7560 ms | not_measured | managed | 0.919x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.0122 ms | not_measured | managed | 0.443x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 1.3133 ms | not_measured | managed | 0.708x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 128.7435 ms | not_measured | managed | 1.244x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.0137 ms | not_measured | managed | 0.453x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 1.4750 ms | not_measured | managed | 0.911x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 152.1865 ms | not_measured | managed | 1.250x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.0232 ms | not_measured | managed | 0.435x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 2.3012 ms | not_measured | managed | 0.933x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 233.3927 ms | not_measured | managed | 1.061x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.0131 ms | not_measured | managed | 0.748x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 1.4259 ms | not_measured | managed | 1.106x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 141.5350 ms | not_measured | managed | 1.209x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.0193 ms | not_measured | managed | 0.585x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 2.0288 ms | not_measured | managed | 1.145x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 207.6808 ms | not_measured | managed | 1.238x |
| `np.ceil (float16)` | float16 | 1,000 | 0.0066 ms | not_measured | managed | 0.758x |
| `np.ceil (float16)` | float16 | 100,000 | 0.6192 ms | not_measured | managed | 0.653x |
| `np.ceil (float16)` | float16 | 10,000,000 | 59.9849 ms | not_measured | managed | 0.926x |
| `np.ceil (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.ceil (float32)` | float32 | 100,000 | 0.0554 ms | not_measured | managed | 0.117x |
| `np.ceil (float32)` | float32 | 10,000,000 | 8.9631 ms | not_measured | managed | 1.391x |
| `np.ceil (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.ceil (float64)` | float64 | 100,000 | 0.1122 ms | not_measured | managed | 0.250x |
| `np.ceil (float64)` | float64 | 10,000,000 | 19.4373 ms | not_measured | managed | 1.343x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.0099 ms | not_measured | managed | 0.909x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 1.0154 ms | not_measured | managed | 0.984x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 101.7041 ms | not_measured | managed | 1.274x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.0097 ms | not_measured | managed | 0.227x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.0606 ms | not_measured | managed | 0.149x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 9.1869 ms | not_measured | managed | 1.517x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.0065 ms | not_measured | managed | 0.308x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.1239 ms | not_measured | managed | 0.228x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 21.1339 ms | not_measured | managed | 1.205x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.0440 ms | not_measured | managed | 0.518x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 5.4225 ms | not_measured | managed | 1.196x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.0593 ms | not_measured | managed | 0.523x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 8.7516 ms | not_measured | managed | 1.508x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.1016 ms | not_measured | managed | 0.288x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 20.7380 ms | not_measured | managed | 1.367x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.0437 ms | not_measured | managed | 0.529x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 5.4648 ms | not_measured | managed | 1.183x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.0604 ms | not_measured | managed | 0.520x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 8.8127 ms | not_measured | managed | 1.543x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.1024 ms | not_measured | managed | 0.284x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 20.6281 ms | not_measured | managed | 1.295x |
| `np.cos (float16)` | float16 | 1,000 | 0.0161 ms | not_measured | managed | 0.460x |
| `np.cos (float16)` | float16 | 100,000 | 1.8781 ms | not_measured | managed | 0.542x |
| `np.cos (float16)` | float16 | 10,000,000 | 188.2764 ms | not_measured | managed | 0.880x |
| `np.cos (float32)` | float32 | 1,000 | 0.0044 ms | not_measured | managed | 0.273x |
| `np.cos (float32)` | float32 | 100,000 | 0.2571 ms | not_measured | managed | 0.294x |
| `np.cos (float32)` | float32 | 10,000,000 | 25.9983 ms | not_measured | managed | 1.394x |
| `np.cos (float64)` | float64 | 1,000 | 0.0119 ms | not_measured | managed | 0.479x |
| `np.cos (float64)` | float64 | 100,000 | 1.2287 ms | not_measured | managed | 1.046x |
| `np.cos (float64)` | float64 | 10,000,000 | 124.2762 ms | not_measured | managed | 1.145x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.0214 ms | not_measured | managed | 0.388x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 2.0967 ms | not_measured | managed | 0.448x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 212.3485 ms | not_measured | managed | 0.758x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.0102 ms | not_measured | managed | 0.422x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 1.0943 ms | not_measured | managed | 0.582x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 110.1705 ms | not_measured | managed | 1.126x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.0120 ms | not_measured | managed | 0.358x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 1.1741 ms | not_measured | managed | 1.152x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 122.4276 ms | not_measured | managed | 1.272x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.0072 ms | not_measured | managed | 0.653x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.6586 ms | not_measured | managed | 0.573x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 66.9683 ms | not_measured | managed | 0.786x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.600x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.0498 ms | not_measured | managed | 2.221x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 8.9818 ms | not_measured | managed | 2.172x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.536x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.0994 ms | not_measured | managed | 2.027x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 19.6691 ms | not_measured | managed | 1.573x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.0073 ms | not_measured | managed | 0.658x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.6564 ms | not_measured | managed | 0.622x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 66.8015 ms | not_measured | managed | 0.785x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.600x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.0503 ms | not_measured | managed | 2.141x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 8.7882 ms | not_measured | managed | 1.780x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.464x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.0980 ms | not_measured | managed | 1.101x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 19.1633 ms | not_measured | managed | 1.648x |
| `np.exp (float16)` | float16 | 1,000 | 0.0113 ms | not_measured | managed | 0.442x |
| `np.exp (float16)` | float16 | 100,000 | 1.0730 ms | not_measured | managed | 0.413x |
| `np.exp (float16)` | float16 | 10,000,000 | 107.9796 ms | not_measured | managed | 0.710x |
| `np.exp (float32)` | float32 | 1,000 | 0.0045 ms | not_measured | managed | 0.267x |
| `np.exp (float32)` | float32 | 100,000 | 0.2174 ms | not_measured | managed | 0.290x |
| `np.exp (float32)` | float32 | 10,000,000 | 19.9004 ms | not_measured | managed | 1.842x |
| `np.exp (float64)` | float64 | 1,000 | 0.0100 ms | not_measured | managed | 0.320x |
| `np.exp (float64)` | float64 | 100,000 | 0.4796 ms | not_measured | managed | 0.961x |
| `np.exp (float64)` | float64 | 10,000,000 | 51.1050 ms | not_measured | managed | 0.813x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.0185 ms | not_measured | managed | 0.249x |
| `np.exp2 (float16)` | float16 | 100,000 | 1.7820 ms | not_measured | managed | 0.260x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 178.1323 ms | not_measured | managed | 0.398x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.0162 ms | not_measured | managed | 0.130x |
| `np.exp2 (float32)` | float32 | 100,000 | 1.5216 ms | not_measured | managed | 0.111x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 150.8896 ms | not_measured | managed | 0.312x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.0169 ms | not_measured | managed | 0.148x |
| `np.exp2 (float64)` | float64 | 100,000 | 1.4642 ms | not_measured | managed | 0.299x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 147.2939 ms | not_measured | managed | 0.386x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.0139 ms | not_measured | managed | 0.424x |
| `np.expm1 (float16)` | float16 | 100,000 | 1.3397 ms | not_measured | managed | 0.418x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 133.9596 ms | not_measured | managed | 0.684x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.0065 ms | not_measured | managed | 0.446x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.3536 ms | not_measured | managed | 0.738x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 33.2553 ms | not_measured | managed | 1.613x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.0086 ms | not_measured | managed | 0.465x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.4887 ms | not_measured | managed | 1.366x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 50.9595 ms | not_measured | managed | 1.685x |
| `np.floor (float16)` | float16 | 1,000 | 0.0066 ms | not_measured | managed | 0.758x |
| `np.floor (float16)` | float16 | 100,000 | 0.6127 ms | not_measured | managed | 0.638x |
| `np.floor (float16)` | float16 | 10,000,000 | 59.6124 ms | not_measured | managed | 0.900x |
| `np.floor (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.floor (float32)` | float32 | 100,000 | 0.0570 ms | not_measured | managed | 0.151x |
| `np.floor (float32)` | float32 | 10,000,000 | 7.5884 ms | not_measured | managed | 1.702x |
| `np.floor (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.214x |
| `np.floor (float64)` | float64 | 100,000 | 0.1087 ms | not_measured | managed | 0.270x |
| `np.floor (float64)` | float64 | 10,000,000 | 19.4308 ms | not_measured | managed | 1.286x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.0009 ms | not_measured | managed | 0.444x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.0032 ms | not_measured | managed | 0.844x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | 0.0058 ms | not_measured | managed | 3.414x |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 0.400x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.0052 ms | not_measured | managed | 2.442x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | 0.0064 ms | not_measured | managed | 2.688x |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.0065 ms | not_measured | managed | 3.400x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | 0.0082 ms | not_measured | managed | 2.598x |
| `np.log (float16)` | float16 | 1,000 | 0.0125 ms | not_measured | managed | 0.400x |
| `np.log (float16)` | float16 | 100,000 | 1.2146 ms | not_measured | managed | 0.372x |
| `np.log (float16)` | float16 | 10,000,000 | 121.1419 ms | not_measured | managed | 0.716x |
| `np.log (float32)` | float32 | 1,000 | 0.0043 ms | not_measured | managed | 0.302x |
| `np.log (float32)` | float32 | 100,000 | 0.2384 ms | not_measured | managed | 0.405x |
| `np.log (float32)` | float32 | 10,000,000 | 21.8829 ms | not_measured | managed | 2.199x |
| `np.log (float64)` | float64 | 1,000 | 0.0082 ms | not_measured | managed | 0.402x |
| `np.log (float64)` | float64 | 100,000 | 0.5266 ms | not_measured | managed | 0.860x |
| `np.log (float64)` | float64 | 10,000,000 | 55.3229 ms | not_measured | managed | 1.031x |
| `np.log10 (float16)` | float16 | 1,000 | 0.0130 ms | not_measured | managed | 0.392x |
| `np.log10 (float16)` | float16 | 100,000 | 1.2570 ms | not_measured | managed | 0.367x |
| `np.log10 (float16)` | float16 | 10,000,000 | 124.3444 ms | not_measured | managed | 0.727x |
| `np.log10 (float32)` | float32 | 1,000 | 0.0075 ms | not_measured | managed | 0.320x |
| `np.log10 (float32)` | float32 | 100,000 | 0.4545 ms | not_measured | managed | 0.455x |
| `np.log10 (float32)` | float32 | 10,000,000 | 44.7537 ms | not_measured | managed | 1.009x |
| `np.log10 (float64)` | float64 | 1,000 | 0.0085 ms | not_measured | managed | 0.365x |
| `np.log10 (float64)` | float64 | 100,000 | 0.5484 ms | not_measured | managed | 0.850x |
| `np.log10 (float64)` | float64 | 10,000,000 | 57.8679 ms | not_measured | managed | 1.115x |
| `np.log1p (float16)` | float16 | 1,000 | 0.0132 ms | not_measured | managed | 0.508x |
| `np.log1p (float16)` | float16 | 100,000 | 1.2682 ms | not_measured | managed | 0.457x |
| `np.log1p (float16)` | float16 | 10,000,000 | 126.7996 ms | not_measured | managed | 0.839x |
| `np.log1p (float32)` | float32 | 1,000 | 0.0049 ms | not_measured | managed | 0.694x |
| `np.log1p (float32)` | float32 | 100,000 | 0.4411 ms | not_measured | managed | 0.684x |
| `np.log1p (float32)` | float32 | 10,000,000 | 43.0089 ms | not_measured | managed | 1.726x |
| `np.log1p (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.448x |
| `np.log1p (float64)` | float64 | 100,000 | 0.5255 ms | not_measured | managed | 1.343x |
| `np.log1p (float64)` | float64 | 10,000,000 | 55.5434 ms | not_measured | managed | 1.473x |
| `np.log2 (float16)` | float16 | 1,000 | 0.0111 ms | not_measured | managed | 0.514x |
| `np.log2 (float16)` | float16 | 100,000 | 1.0718 ms | not_measured | managed | 0.457x |
| `np.log2 (float16)` | float16 | 10,000,000 | 105.6358 ms | not_measured | managed | 0.745x |
| `np.log2 (float32)` | float32 | 1,000 | 0.0077 ms | not_measured | managed | 0.338x |
| `np.log2 (float32)` | float32 | 100,000 | 0.4688 ms | not_measured | managed | 0.441x |
| `np.log2 (float32)` | float32 | 10,000,000 | 45.0238 ms | not_measured | managed | 0.939x |
| `np.log2 (float64)` | float64 | 1,000 | 0.0084 ms | not_measured | managed | 0.536x |
| `np.log2 (float64)` | float64 | 100,000 | 0.7806 ms | not_measured | managed | 1.004x |
| `np.log2 (float64)` | float64 | 10,000,000 | 78.1854 ms | not_measured | managed | 1.190x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.0051 ms | not_measured | managed | 0.431x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.1472 ms | not_measured | managed | 2.950x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 18.8824 ms | not_measured | managed | 3.078x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.0083 ms | not_measured | managed | 0.277x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.2911 ms | not_measured | managed | 1.825x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 46.3796 ms | not_measured | managed | 1.512x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.0530 ms | not_measured | managed | 1.338x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 6.0166 ms | not_measured | managed | 1.314x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.0515 ms | not_measured | managed | 0.398x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 8.9438 ms | not_measured | managed | 1.360x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.172x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.1026 ms | not_measured | managed | 0.220x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 19.6483 ms | not_measured | managed | 1.356x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.0281 ms | not_measured | managed | 1.317x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 2.6873 ms | not_measured | managed | 2.551x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.0585 ms | not_measured | managed | 0.480x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 6.0583 ms | not_measured | managed | 2.297x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.1011 ms | not_measured | managed | 0.281x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 16.8109 ms | not_measured | managed | 1.717x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.0076 ms | not_measured | managed | 1.211x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.6743 ms | not_measured | managed | 1.255x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 65.3230 ms | not_measured | managed | 2.835x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.0042 ms | not_measured | managed | 0.500x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.0769 ms | not_measured | managed | 1.657x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 9.0667 ms | not_measured | managed | 2.324x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.288x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.3015 ms | not_measured | managed | 2.043x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 32.5269 ms | not_measured | managed | 2.342x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.0104 ms | not_measured | managed | 1.019x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.8896 ms | not_measured | managed | 1.264x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 89.7952 ms | not_measured | managed | 2.509x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.0037 ms | not_measured | managed | 0.649x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.0538 ms | not_measured | managed | 2.916x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 9.2622 ms | not_measured | managed | 2.675x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.644x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.1085 ms | not_measured | managed | 1.922x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 20.5265 ms | not_measured | managed | 1.544x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.0350 ms | not_measured | managed | 0.360x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 3.8740 ms | not_measured | managed | 0.409x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 390.0583 ms | not_measured | managed | 0.718x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.0133 ms | not_measured | managed | 0.474x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 1.3292 ms | not_measured | managed | 0.513x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 127.0359 ms | not_measured | managed | 1.345x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.0224 ms | not_measured | managed | 0.446x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 2.1654 ms | not_measured | managed | 1.001x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 203.8046 ms | not_measured | managed | 1.187x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.0072 ms | not_measured | managed | 0.653x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.6680 ms | not_measured | managed | 0.573x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 67.3751 ms | not_measured | managed | 0.766x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.600x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.0501 ms | not_measured | managed | 2.200x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 8.8103 ms | not_measured | managed | 1.835x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.812x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.1000 ms | not_measured | managed | 1.304x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 19.0930 ms | not_measured | managed | 1.622x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.0071 ms | not_measured | managed | 0.620x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.6635 ms | not_measured | managed | 0.590x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 66.1986 ms | not_measured | managed | 0.790x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.600x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.0501 ms | not_measured | managed | 2.220x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 8.8403 ms | not_measured | managed | 2.195x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.464x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.1012 ms | not_measured | managed | 1.148x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 19.5040 ms | not_measured | managed | 1.527x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.0078 ms | not_measured | managed | 0.333x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.7367 ms | not_measured | managed | 0.623x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 72.5847 ms | not_measured | managed | 0.709x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.0627 ms | not_measured | managed | 0.963x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 8.9959 ms | not_measured | managed | 1.539x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.225x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.1978 ms | not_measured | managed | 1.017x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 23.8074 ms | not_measured | managed | 1.328x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.0079 ms | not_measured | managed | 0.709x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.7954 ms | not_measured | managed | 0.629x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 79.5745 ms | not_measured | managed | 0.820x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.0497 ms | not_measured | managed | 0.276x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 8.8757 ms | not_measured | managed | 1.392x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.179x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.0974 ms | not_measured | managed | 0.227x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 19.4009 ms | not_measured | managed | 1.449x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.0079 ms | not_measured | managed | 0.709x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.7922 ms | not_measured | managed | 0.652x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 79.2901 ms | not_measured | managed | 0.833x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.500x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.0506 ms | not_measured | managed | 0.308x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 8.9845 ms | not_measured | managed | 1.402x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.393x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.1004 ms | not_measured | managed | 0.233x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 19.1692 ms | not_measured | managed | 1.507x |
| `np.sign (float16)` | float16 | 1,000 | 0.0078 ms | not_measured | managed | 0.179x |
| `np.sign (float16)` | float16 | 100,000 | 0.9705 ms | not_measured | managed | 0.092x |
| `np.sign (float16)` | float16 | 10,000,000 | 98.5637 ms | not_measured | managed | 0.201x |
| `np.sign (float32)` | float32 | 1,000 | 0.0040 ms | not_measured | managed | 0.300x |
| `np.sign (float32)` | float32 | 100,000 | 0.5697 ms | not_measured | managed | 0.560x |
| `np.sign (float32)` | float32 | 10,000,000 | 58.0816 ms | not_measured | managed | 0.919x |
| `np.sign (float64)` | float64 | 1,000 | 0.0043 ms | not_measured | managed | 0.279x |
| `np.sign (float64)` | float64 | 100,000 | 0.5758 ms | not_measured | managed | 0.754x |
| `np.sign (float64)` | float64 | 10,000,000 | 62.0712 ms | not_measured | managed | 0.980x |
| `np.sin (float16)` | float16 | 1,000 | 0.0163 ms | not_measured | managed | 0.509x |
| `np.sin (float16)` | float16 | 100,000 | 1.8840 ms | not_measured | managed | 0.546x |
| `np.sin (float16)` | float16 | 10,000,000 | 188.3727 ms | not_measured | managed | 0.898x |
| `np.sin (float32)` | float32 | 1,000 | 0.0045 ms | not_measured | managed | 0.267x |
| `np.sin (float32)` | float32 | 100,000 | 0.2655 ms | not_measured | managed | 0.265x |
| `np.sin (float32)` | float32 | 10,000,000 | 25.3736 ms | not_measured | managed | 1.365x |
| `np.sin (float64)` | float64 | 1,000 | 0.0115 ms | not_measured | managed | 0.461x |
| `np.sin (float64)` | float64 | 100,000 | 1.2034 ms | not_measured | managed | 1.039x |
| `np.sin (float64)` | float64 | 10,000,000 | 121.7944 ms | not_measured | managed | 1.082x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.0220 ms | not_measured | managed | 0.373x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 2.1911 ms | not_measured | managed | 0.453x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 219.1652 ms | not_measured | managed | 0.786x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.0107 ms | not_measured | managed | 0.421x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 1.1693 ms | not_measured | managed | 0.548x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 115.9714 ms | not_measured | managed | 1.030x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.0126 ms | not_measured | managed | 0.373x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 1.2991 ms | not_measured | managed | 1.178x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 131.5665 ms | not_measured | managed | 1.309x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.0071 ms | not_measured | managed | 0.718x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.6754 ms | not_measured | managed | 0.565x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 65.1094 ms | not_measured | managed | 0.800x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.0759 ms | not_measured | managed | 0.203x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 9.1052 ms | not_measured | managed | 1.559x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 0.232x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.2863 ms | not_measured | managed | 1.076x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 32.3711 ms | not_measured | managed | 1.225x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.0087 ms | not_measured | managed | 0.299x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.8309 ms | not_measured | managed | 0.458x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 82.3769 ms | not_measured | managed | 0.611x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.429x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 8.9069 ms | not_measured | managed | 1.398x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.214x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.1039 ms | not_measured | managed | 0.210x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 19.7600 ms | not_measured | managed | 1.352x |
| `np.tan (float16)` | float16 | 1,000 | 0.0165 ms | not_measured | managed | 0.455x |
| `np.tan (float16)` | float16 | 100,000 | 1.8645 ms | not_measured | managed | 0.547x |
| `np.tan (float16)` | float16 | 10,000,000 | 188.9694 ms | not_measured | managed | 0.896x |
| `np.tan (float32)` | float32 | 1,000 | 0.0082 ms | not_measured | managed | 0.476x |
| `np.tan (float32)` | float32 | 100,000 | 1.1465 ms | not_measured | managed | 0.652x |
| `np.tan (float32)` | float32 | 10,000,000 | 111.2932 ms | not_measured | managed | 1.163x |
| `np.tan (float64)` | float64 | 1,000 | 0.0126 ms | not_measured | managed | 0.389x |
| `np.tan (float64)` | float64 | 100,000 | 1.5601 ms | not_measured | managed | 1.049x |
| `np.tan (float64)` | float64 | 10,000,000 | 155.9469 ms | not_measured | managed | 1.261x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.0211 ms | not_measured | managed | 0.526x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 2.0928 ms | not_measured | managed | 0.502x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 208.9086 ms | not_measured | managed | 0.896x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.0028 ms | not_measured | managed | 0.643x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.2093 ms | not_measured | managed | 0.612x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 36.1234 ms | not_measured | managed | 1.896x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.0101 ms | not_measured | managed | 0.822x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.8211 ms | not_measured | managed | 4.165x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 83.8608 ms | not_measured | managed | 3.701x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.0066 ms | not_measured | managed | 1.152x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.6153 ms | not_measured | managed | 0.796x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 60.2253 ms | not_measured | managed | 1.086x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.368x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.0536 ms | not_measured | managed | 0.194x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 9.0125 ms | not_measured | managed | 1.558x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.214x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.0988 ms | not_measured | managed | 0.218x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 19.7507 ms | not_measured | managed | 1.443x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-24 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (40 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across elementwise.

HEADLINE — operation matrix: 1.30× geomean · 77%🕐 of NumPy's time · 72 win / 53 lose over 125 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     █████████▎ .........   0.93×   108%🕐  (  9 win / 16 lose)  ◄ SLOWER
1K         ███████████▋ .......   1.17×    86%🕐  ( 13 win / 12 lose)
100K       █████████████▊ .....   1.38×    72%🕐  ( 15 win / 10 lose)
1M         ████████████████▍ ..   1.64×    61%🕐  ( 17 win /  8 lose)
10M        ███████████████▏ ...   1.52×    66%🕐  ( 18 win /  7 lose)
ALL        █████████████ ......   1.30×    77%🕐  ( 72 win / 53 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise(no data)
reductions ███████████████████▶   2.55×    39%🕐  ( 38 win /  2 lose)
selection  ███████████▋ .......   1.17×    86%🕐  ( 19 win / 16 lose)
copy/cast  ███████▍ ...........   0.74×   134%🕐  (  6 win / 19 lose)  ◄ SLOWER
index-math ██████▊ ............   0.68×   146%🕐  (  5 win /  5 lose)  ◄ SLOWER
dtypes     ██████████▉ ........   1.09×    91%🕐  (  4 win / 11 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise         -        -        -        -        -
reductions      3.80×    3.26×    2.15×    2.07×    1.95×
selection       0.57×    1.12×    1.29×    1.64×    1.61×
copy/cast       0.47×    0.48×    0.69×    1.29×    1.15×
index-math      0.18×    0.45×    1.20×    1.09×    1.41×
dtypes          0.64×    0.70×    1.76×    1.74×    1.15×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add             NA       NA       NA       NA       NA
  sqrt            NA       NA       NA       NA       NA
  copy            NA       NA       NA       NA       NA
  strided         NA       NA       NA       NA       NA
  bcast           NA       NA       NA       NA       NA
  reversed        NA       NA       NA       NA       NA
  castbuf         NA       NA       NA       NA       NA
  mixbuf          NA       NA       NA       NA       NA
-- reductions
  sum          1.79×    1.84×    2.76×    2.18×    1.89×     2.06×
  sum ax0      1.87×    1.14×    1.02×    1.17×    1.04×     1.22×
  sum ax1      4.35×    1.30×    1.45×    2.06×    1.65×     1.95×
  sum dt=      2.08×    2.52×    1.14×    1.10×    1.06×     1.48×
  amin         2.24×    2.04×    0.85×    0.73×    1.29×     1.30×
  cumsum       1.53×    1.36×    1.25×    1.91×    1.68×     1.53×
  any(F)      17.93×   26.03×    3.54×    1.88×    1.38×     5.33×
  any(hit)    23.57×   25.79×   25.82×   21.86×   19.84×    23.26×
-- selection
  where        0.62×    0.87×    0.81×    1.33×    1.05×     0.91×
  a[mask]      0.21×    1.01×    2.35×    2.07×    1.65×     1.11×
  a[mask]=     0.18×    2.69×    7.56×    5.84×    4.88×     2.54×
  count_nz     0.77×    2.08×    2.71×    3.57×    1.14×     1.78×
  argwhere     0.98×    2.32×    1.07×    2.78×    3.90×     1.92×
  a[idx]       0.65×    0.32×    0.33×    0.41×    0.93×     0.48×
  a[idx]=      1.67×    0.62×    0.43×    0.48×    0.80×     0.70×
-- copy/cast
  flatten      0.43×    0.19×    0.97×    2.70×    1.29×     0.77×
  astype       0.31×    0.28×    0.57×    1.19×    1.87×     0.65×
  ravel.T      0.43×    0.68×    0.73×    1.47×    0.96×     0.79×
  in-place     0.91×    0.88×    1.05×    0.93×    0.98×     0.95×
  less->b      0.44×    0.75×    0.37×    0.81×    0.87×     0.61×
-- index-math
  unravel      0.29×    0.36×    1.13×    0.94×    1.25×     0.68×
  ravel_mi     0.11×    0.57×    1.28×    1.26×    1.59×     0.69×
-- dtypes
  complex      0.66×    0.56×    0.88×    0.90×    0.80×     0.75×
  float16      0.63×    0.42×    0.62×    0.62×    0.65×     0.58×
  int8         0.64×    1.44×   10.07×    9.38×    2.94×     3.04×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▏   1.92×    52%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   5.07×    20%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   5.63×    18%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   9.06×    11%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   7.12×    14%🕐  (  1 win /  0 lose)
8op          ███████████████████▶  10.08×    10%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   6.90×    14%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   5.37×    19%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   4.98×    20%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   5.72×    17%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████████▌ ..........   0.86×   116%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ███████████▉ .......   1.20×    84%🕐  (  1 win /  0 lose)
w=64         ████████████▋ ......   1.27×    79%🕐  (  1 win /  0 lose)
w=256        ███████████████▉ ...   1.59×    63%🕐  (  1 win /  0 lose)
w=1024       ███████████████████▊   1.98×    50%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    489.52×   (489.5× faster, faster)
  allocate          1.10×   (1.1× faster, faster)
  overlap_copy      1.79×   (1.8× faster, faster)
  forder_out        1.51×   (1.5× faster, faster)
  zerodim           1.73×   (1.7× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           13.72×    7.03×    2.54×    3.83×    1.67×   vs chained 6× add
reuse            7.40×    9.38×    1.31×    0.77×    1.03×   vs rebuild each call
par8                 -    1.52×    5.99×    3.42×    6.61×   vs single-thread

biggest NumSharp wins: anyff@1K 26.03× · anyeh@100K 25.82× · anyeh@1K 25.79× · anyeh@1 23.57× · anyeh@1M 21.86×
most behind:           ravelmi@1 0.11× · bassign@1 0.18× · flatten@1K 0.19× · bread@1 0.21× · astype@1K 0.28×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.02 ✅ | 1.01 ✅ | 1.03 ✅ | 0.59 🟡 | 0.89 🟡 | 0.67 🟡 | 0.83 🟡 | 0.78 🟡 |
| 1M | 0.72 🟡 | 0.83 🟡 | 0.81 🟡 | 0.51 🟡 | 0.90 🟡 | 0.67 🟡 | 0.75 🟡 | 0.74 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.92 🟡 | 1.13 ✅ | 0.96 🟡 | 0.13 🔴 | 1.08 ✅ | 1.36 ✅ | 0.97 🟡 |
| 1M | 0.63 🟡 | 1.04 ✅ | 0.73 🟡 | 0.11 🔴 | 1.03 ✅ | 1.25 ✅ | 1.02 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.83 🟡 | 0.67 🟡 | 0.70 🟡 | 1.35 ✅ |
| 1M | 0.80 🟡 | 0.60 🟡 | 0.59 🟡 | 1.09 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.9751 | 0.1504 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7699 | 0.0219 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7752 | 0.0245 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 8.0090 | 0.2607 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7849 | 0.0256 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7657 | 0.0251 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7843 | 0.0261 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.9411 | 0.2661 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.7552 | 0.2717 | 0.04 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7654 | 0.0268 | 0.04 🔴 |
| 1M\|c128\|C\|prod\|ax0 | 26.1258 | 0.9165 | 0.04 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.9508 | 0.2983 | 0.04 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.9802 | 0.3039 | 0.04 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.3544 | 0.1812 | 0.04 🔴 |
| 100K\|dec\|bcast\|sum\|ax1 | 0.4394 | 0.0254 | 0.06 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.67 🟡 | 0.60 🟡 | 0.61 🟡 | 0.42 🟠 | 0.83 🟡 | 0.88 🟡 | 1.04 ✅ | 0.78 🟡 |
| 1M | 2.66 ✅ | 2.64 ✅ | 2.53 ✅ | 1.70 ✅ | 1.96 ✅ | 1.90 ✅ | 2.50 ✅ | 2.14 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.96 🟡 | 1.05 ✅ | 1.10 ✅ | 1.08 ✅ | 0.49 🟠 | 0.39 🟠 | 0.63 🟡 | 0.40 🟠 | 0.83 🟡 | 0.77 🟡 | 0.38 🟠 | 0.29 🟠 | 2.53 ✅ |
| 1M | 7.08 ✅ | 5.27 ✅ | 2.36 ✅ | 1.67 ✅ | 1.50 ✅ | 1.77 ✅ | 1.75 ✅ | 1.87 ✅ | 2.07 ✅ | 2.06 ✅ | 1.83 ✅ | 1.75 ✅ | 1.89 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|F\|pos | 0.1655 | 0.0251 | 0.15 🔴 |
| 100K\|f64\|strided\|pos | 0.0767 | 0.0117 | 0.15 🔴 |
| 100K\|i64\|strided\|pos | 0.0755 | 0.0117 | 0.15 🔴 |
| 100K\|f64\|T\|pos | 0.1474 | 0.0247 | 0.17 🔴 |
| 100K\|u64\|strided\|pos | 0.0711 | 0.0128 | 0.18 🔴 |
| 100K\|f32\|strided\|pos | 0.0585 | 0.0113 | 0.19 🔴 |
| 100K\|i32\|strided\|pos | 0.0600 | 0.0136 | 0.23 🟠 |
| 100K\|u32\|strided\|pos | 0.0503 | 0.0126 | 0.25 🟠 |
| 100K\|f64\|C\|pos | 0.0962 | 0.0262 | 0.27 🟠 |
| 100K\|f32\|T\|pos | 0.0709 | 0.0211 | 0.30 🟠 |
| 100K\|u32\|F\|pos | 0.0732 | 0.0233 | 0.32 🟠 |
| 100K\|u64\|F\|pos | 0.1068 | 0.0343 | 0.32 🟠 |
| 100K\|u64\|T\|pos | 0.0952 | 0.0313 | 0.33 🟠 |
| 100K\|u64\|C\|pos | 0.0963 | 0.0339 | 0.35 🟠 |
| 100K\|u32\|bcast\|pos | 0.0760 | 0.0272 | 0.36 🟠 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.63 🟡 | 0.56 🟡 | 0.58 🟡 | 0.37 🟠 | 0.76 🟡 | 0.75 🟡 | 0.82 🟡 | 0.73 🟡 |
| 1M | 1.33 ✅ | 1.30 ✅ | 1.26 ✅ | 0.80 🟡 | 1.38 ✅ | 1.40 ✅ | 1.14 ✅ | 1.51 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.62 🟡 | 0.24 🟠 | 1.23 ✅ | 0.74 🟡 | 0.78 🟡 | 0.61 🟡 |
| 1M | 1.34 ✅ | 1.36 ✅ | 1.36 ✅ | 0.93 🟡 | 1.34 ✅ | 1.21 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.74 🟡 | 0.76 🟡 | 0.58 🟡 | 0.63 🟡 | 0.70 🟡 | 0.60 🟡 | 0.47 🟠 |
| 1M | 1.29 ✅ | 1.31 ✅ | 1.52 ✅ | 1.45 ✅ | 1.09 ✅ | 0.79 🟡 | 1.47 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|abs | 0.0688 | 0.0061 | 0.09 🔴 |
| 100K\|f32\|strided\|neg | 0.0705 | 0.0063 | 0.09 🔴 |
| 100K\|i64\|strided\|add | 0.2354 | 0.0258 | 0.11 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0713 | 0.0079 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1190 | 0.0136 | 0.11 🔴 |
| 100K\|f32\|strided\|add | 0.1156 | 0.0139 | 0.12 🔴 |
| 100K\|i64\|strided\|mul | 0.2403 | 0.0312 | 0.13 🔴 |
| 100K\|f32\|F\|sqrt | 0.1038 | 0.0142 | 0.14 🔴 |
| 100K\|f32\|T\|sqrt | 0.0985 | 0.0143 | 0.14 🔴 |
| 100K\|f32\|bcast\|copy | 0.0342 | 0.0050 | 0.15 🔴 |
| 100K\|f64\|strided\|abs | 0.0519 | 0.0078 | 0.15 🔴 |
| 100K\|f32\|T\|abs | 0.0363 | 0.0057 | 0.16 🔴 |
| 100K\|f32\|C\|abs | 0.0354 | 0.0057 | 0.16 🔴 |
| 100K\|i32\|F\|neg | 0.0398 | 0.0065 | 0.16 🔴 |
| 100K\|f32\|C\|add | 0.0389 | 0.0067 | 0.17 🔴 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 1.73 ✅ | 1.96 ✅ | 0.61 🟡 | 2.24 ✅ | 1.23 ✅ | 1.18 ✅ | 1.37 ✅ |
| 1-D strided a[::2] | 1.10 ✅ | 1.29 ✅ | 0.51 🟡 | 0.49 🟠 | 0.48 🟠 | 0.98 🟡 | 0.74 🟡 |
| 1-D reversed a[::-1] | 1.34 ✅ | 1.82 ✅ | 0.31 🟠 | 0.52 🟡 | 0.48 🟠 | 1.17 ✅ | 0.78 🟡 |
| array + scalar | 1.62 ✅ | 1.94 ✅ | 0.74 🟡 | 1.98 ✅ | 2.20 ✅ | 1.08 ✅ | 1.49 ✅ |
| scalar + array | 1.73 ✅ | 1.96 ✅ | 0.75 🟡 | 2.32 ✅ | 1.58 ✅ | 1.13 ✅ | 1.48 ✅ |
| mixed C + F | 1.08 ✅ | 1.78 ✅ | 0.75 🟡 | 0.75 🟡 | 0.66 🟡 | 1.42 ✅ | 1.00 ✅ |
| mixed C + T | 1.70 ✅ | 1.85 ✅ | 0.76 🟡 | 0.77 🟡 | 0.56 🟡 | 1.67 ✅ | 1.09 ✅ |
| binary broadcast +row(1,C) | 2.38 ✅ | 2.03 ✅ | 0.74 🟡 | 2.34 ✅ | 1.90 ✅ | 1.34 ✅ | 1.66 ✅ |
| binary broadcast +col(R,1) | 1.89 ✅ | 2.10 ✅ | 0.71 🟡 | 2.54 ✅ | 2.00 ✅ | 1.74 ✅ | 1.71 ✅ |
| col-broadcast unary (inner stride-0) | 2.29 ✅ | 1.56 ✅ | 0.78 🟡 | 1.60 ✅ | 2.25 ✅ | 3.11 ✅ | 1.78 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_rev|f16 | 10.1367 | 3.0923 | 0.31 🟠 |
| 1d_rev|i64 | 4.1858 | 1.9939 | 0.48 🟠 |
| 1d_strided|i64 | 2.2447 | 1.0786 | 0.48 🟠 |
| 1d_strided|i32 | 1.1653 | 0.5690 | 0.49 🟠 |
| 1d_strided|f16 | 2.7960 | 1.4127 | 0.51 🟡 |
| 1d_rev|i32 | 2.0741 | 1.0886 | 0.52 🟡 |
| mix_C_T|i64 | 5.4057 | 3.0449 | 0.56 🟡 |
| 1d_C|f16 | 5.0186 | 3.0662 | 0.61 🟡 |
| mix_C_F|i64 | 5.3856 | 3.5580 | 0.66 🟡 |
| bcast_col|f16 | 10.2398 | 7.2599 | 0.71 🟡 |
| scalar_rhs|f16 | 9.3804 | 6.9071 | 0.74 🟡 |
| bcast_row|f16 | 9.6966 | 7.1450 | 0.74 🟡 |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **514 / 1568** comparable cells lag (<1.0); **1054** win (≥1.0).
- **🔴 <0.2** — 28 cells. Top: 14× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 5× same-type diagonal (copy); 4× int → sub-word (narrow)
- **🟠 0.2–0.5** — 190 cells. Top: 77× int → sub-word (narrow); 22× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 19× * → bool
- **🟡 0.5–1.0** — 296 cells. Top: 77× int → sub-word (narrow); 28× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 18× * → bool

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **1.08**, `f64`→narrow **0.98**, `f16`→narrow **2.51**, `c128`→narrow **0.73**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 1.52 ✅ | 1.66 ✅ | 1.54 ✅ | 1.50 ✅ | 1.52 ✅ | 1.19 ✅ | 1.12 ✅ | 0.35 🟠 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.83 🟡 | 1.65 ✅ | 1.52 ✅ | 1.84 ✅ | 1.67 ✅ | 1.18 ✅ | 1.00 ✅ | 0.85 🟡 | 0.87 🟡 | 1.49 ✅ | 1.61 ✅ | 1.04 ✅ | 1.01 ✅ | nan ? | 0.87 🟡 |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.26 ✅ | 1.08 ✅ | 1.16 ✅ | 1.11 ✅ | 1.11 ✅ | 1.29 ✅ | 1.06 ✅ | 1.31 ✅ | 1.13 ✅ | 1.09 ✅ | 1.31 ✅ | 1.17 ✅ | 1.31 ✅ | nan ? | 1.40 ✅ |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.21🟠 | 0.53🟡 | 0.57🟡 | 0.84🟡 | 0.81🟡 | 1.25✅ | 1.05✅ | 1.32✅ | 1.36✅ | 0.78🟡 | 0.24🟠 | 1.78✅ | 2.28✅ | — | 2.04✅ |
| **u8** | 4.21✅ | 0.26🟠 | 5.00✅ | 2.53✅ | 2.79✅ | 2.18✅ | 2.39✅ | 2.03✅ | 2.17✅ | 2.68✅ | 2.46✅ | 1.50✅ | 1.89✅ | — | 1.35✅ |
| **i8** | 4.32✅ | 2.38✅ | 0.14🔴 | 2.07✅ | 1.99✅ | 2.43✅ | 2.02✅ | 1.74✅ | 1.83✅ | 1.78✅ | 1.79✅ | 1.15✅ | 1.65✅ | — | 1.62✅ |
| **i16** | 6.12✅ | 5.62✅ | 5.15✅ | 1.11✅ | 2.33✅ | 2.41✅ | 1.48✅ | 2.11✅ | 1.96✅ | 1.82✅ | 2.38✅ | 2.19✅ | 1.73✅ | — | 1.59✅ |
| **u16** | 4.09✅ | 5.47✅ | 5.38✅ | 2.34✅ | 1.29✅ | 1.99✅ | 1.73✅ | 1.41✅ | 1.70✅ | 0.93🟡 | 2.48✅ | 2.18✅ | 1.62✅ | — | 1.53✅ |
| **i32** | 0.82🟡 | 0.75🟡 | 0.76🟡 | 1.48✅ | 1.76✅ | 1.56✅ | 2.38✅ | 1.44✅ | 1.78✅ | 1.98✅ | 2.43✅ | 1.42✅ | 1.40✅ | — | 1.37✅ |
| **u32** | 2.12✅ | 1.04✅ | 0.67🟡 | 1.64✅ | 2.08✅ | 2.23✅ | 1.58✅ | 1.54✅ | 1.54✅ | 2.06✅ | 2.24✅ | 1.36✅ | 1.38✅ | — | 1.47✅ |
| **i64** | 0.98🟡 | 0.35🟠 | 0.57🟡 | 0.94🟡 | 0.56🟡 | 1.16✅ | 1.36✅ | 1.32✅ | 1.93✅ | 1.15✅ | 1.28✅ | 1.37✅ | 0.98🟡 | — | 1.36✅ |
| **u64** | 0.88🟡 | 0.65🟡 | 0.82🟡 | 0.55🟡 | 0.95🟡 | 1.55✅ | 1.34✅ | 2.28✅ | 1.42✅ | 1.27✅ | 1.52✅ | 1.19✅ | 1.15✅ | — | 1.18✅ |
| **char** | 4.38✅ | 4.21✅ | 4.70✅ | 1.75✅ | 0.99🟡 | 1.53✅ | 1.32✅ | 1.67✅ | 1.83✅ | 0.92🟡 | 2.42✅ | 1.01✅ | 1.54✅ | — | 1.41✅ |
| **f16** | 4.06✅ | 3.56✅ | 3.70✅ | 3.33✅ | 2.96✅ | 3.39✅ | 1.25✅ | 2.11✅ | 0.64🟡 | 3.42✅ | 1.08✅ | 2.31✅ | 0.72🟡 | — | 1.09✅ |
| **f32** | 3.22✅ | 1.69✅ | 1.63✅ | 1.71✅ | 1.86✅ | 1.90✅ | 0.58🟡 | 0.69🟡 | 0.50🟠 | 1.85✅ | 2.39✅ | 1.52✅ | 1.64✅ | — | 1.28✅ |
| **f64** | 1.90✅ | 1.85✅ | 2.03✅ | 1.65✅ | 1.32✅ | 1.78✅ | 0.79🟡 | 0.92🟡 | 0.59🟡 | 1.40✅ | 1.60✅ | 1.74✅ | 1.66✅ | — | 1.70✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.79🟡 | 1.28✅ | 1.29✅ | 1.58✅ | 1.33✅ | 1.62✅ | 0.83🟡 | 1.09✅ | 0.72🟡 | 1.55✅ | 1.40✅ | 1.39✅ | 1.16✅ | — | 1.76✅ |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.35✅ | 0.85🟡 | 0.97🟡 | 0.84🟡 | 0.87🟡 | 1.39✅ | 1.38✅ | 2.22✅ | 2.44✅ | 0.85🟡 | 0.24🟠 | 1.77✅ | 2.23✅ | — | 2.47✅ |
| **u8** | 6.11✅ | 2.53✅ | 6.85✅ | 2.41✅ | 2.38✅ | 2.39✅ | 2.49✅ | 1.89✅ | 1.90✅ | 2.44✅ | 2.42✅ | 1.46✅ | 1.62✅ | — | 1.96✅ |
| **i8** | 6.43✅ | 5.50✅ | 2.74✅ | 2.16✅ | 1.49✅ | 2.14✅ | 2.14✅ | 1.65✅ | 1.62✅ | 1.86✅ | 2.40✅ | 1.31✅ | 1.66✅ | — | 1.72✅ |
| **i16** | 5.75✅ | 3.91✅ | 5.52✅ | 1.13✅ | 2.32✅ | 2.01✅ | 2.06✅ | 1.77✅ | 2.27✅ | 2.29✅ | 2.15✅ | 2.40✅ | 1.70✅ | — | 1.66✅ |
| **u16** | 4.68✅ | 2.70✅ | 4.83✅ | 2.36✅ | 1.17✅ | 1.96✅ | 2.02✅ | 1.89✅ | 2.04✅ | 1.18✅ | 2.33✅ | 2.28✅ | 1.62✅ | — | 2.00✅ |
| **i32** | 2.12✅ | 0.74🟡 | 0.77🟡 | 1.67✅ | 1.80✅ | 1.50✅ | 2.40✅ | 1.97✅ | 2.14✅ | 2.03✅ | 1.73✅ | 1.54✅ | 1.72✅ | — | 1.58✅ |
| **u32** | 2.14✅ | 0.99🟡 | 0.68🟡 | 1.61✅ | 1.94✅ | 2.43✅ | 1.55✅ | 1.57✅ | 1.61✅ | 2.04✅ | 2.30✅ | 1.08✅ | 1.24✅ | — | 1.21✅ |
| **i64** | 0.94🟡 | 0.92🟡 | 0.60🟡 | 1.14✅ | 1.02✅ | 1.57✅ | 1.70✅ | 1.52✅ | 2.27✅ | 1.09✅ | 1.26✅ | 1.02✅ | 1.30✅ | — | 1.12✅ |
| **u64** | 0.83🟡 | 0.72🟡 | 1.00🟡 | 1.03✅ | 0.89🟡 | 1.46✅ | 1.53✅ | 2.24✅ | 1.46✅ | 1.23✅ | 1.40✅ | 1.18✅ | 1.18✅ | — | 1.57✅ |
| **char** | 3.42✅ | 3.83✅ | 5.57✅ | 1.80✅ | 1.04✅ | 1.50✅ | 1.39✅ | 1.43✅ | 1.90✅ | 1.19✅ | 2.41✅ | 1.32✅ | 1.52✅ | — | 1.50✅ |
| **f16** | 10.60✅ | 3.42✅ | 3.74✅ | 3.42✅ | 3.53✅ | 3.15✅ | 1.28✅ | 2.11✅ | 0.63🟡 | 2.66✅ | 0.98🟡 | 2.11✅ | 0.68🟡 | — | 1.00🟡 |
| **f32** | 3.00✅ | 1.72✅ | 1.65✅ | 1.23✅ | 1.88✅ | 1.77✅ | 0.59🟡 | 0.76🟡 | 0.48🟠 | 1.78✅ | 2.64✅ | 1.50✅ | 1.82✅ | — | 1.34✅ |
| **f64** | 1.70✅ | 1.88✅ | 1.72✅ | 1.49✅ | 1.46✅ | 1.71✅ | 0.84🟡 | 0.92🟡 | 0.58🟡 | 1.19✅ | 1.50✅ | 1.89✅ | 1.61✅ | — | 1.39✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.75🟡 | 1.42✅ | 2.13✅ | 1.18✅ | 1.90✅ | 2.91✅ | 0.94🟡 | 0.90🟡 | 0.66🟡 | 1.30✅ | 1.41✅ | 1.26✅ | 1.24✅ | — | 1.54✅ |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.22🟠 | 0.56🟡 | 0.64🟡 | 0.88🟡 | 0.87🟡 | 1.43✅ | 1.45✅ | 2.50✅ | 2.38✅ | 0.85🟡 | 0.23🟠 | 0.96🟡 | 1.30✅ | — | 1.73✅ |
| **u8** | 4.01✅ | 0.27🟠 | 5.54✅ | 2.86✅ | 2.32✅ | 2.20✅ | 2.09✅ | 2.05✅ | 2.12✅ | 2.41✅ | 2.29✅ | 1.49✅ | 1.71✅ | — | 2.18✅ |
| **i8** | 4.39✅ | 3.78✅ | 0.47🟠 | 2.53✅ | 2.08✅ | 1.74✅ | 2.34✅ | 1.98✅ | 1.92✅ | 1.93✅ | 2.53✅ | 1.38✅ | 1.77✅ | — | 1.61✅ |
| **i16** | 3.67✅ | 3.70✅ | 5.40✅ | 1.27✅ | 2.25✅ | 2.25✅ | 1.93✅ | 1.96✅ | 2.02✅ | 2.36✅ | 2.47✅ | 2.39✅ | 1.65✅ | — | 1.55✅ |
| **u16** | 5.00✅ | 5.78✅ | 6.38✅ | 2.19✅ | 1.28✅ | 2.12✅ | 1.84✅ | 2.23✅ | 2.14✅ | 1.37✅ | 2.16✅ | 2.23✅ | 1.72✅ | — | 1.77✅ |
| **i32** | 2.29✅ | 0.74🟡 | 0.73🟡 | 1.52✅ | 1.86✅ | 1.43✅ | 2.38✅ | 2.03✅ | 1.57✅ | 1.79✅ | 2.45✅ | 1.41✅ | 1.85✅ | — | 1.64✅ |
| **u32** | 2.18✅ | 1.02✅ | 0.67🟡 | 1.33✅ | 1.84✅ | 2.47✅ | 1.70✅ | 1.55✅ | 1.30✅ | 1.90✅ | 1.82✅ | 1.31✅ | 1.58✅ | — | 1.40✅ |
| **i64** | 0.84🟡 | 0.83🟡 | 0.64🟡 | 0.97🟡 | 0.90🟡 | 1.12✅ | 1.31✅ | 1.36✅ | 2.14✅ | 1.09✅ | 1.25✅ | 0.99🟡 | 0.86🟡 | — | 1.15✅ |
| **u64** | 0.99🟡 | 0.36🟠 | 0.85🟡 | 0.83🟡 | 0.96🟡 | 1.29✅ | 1.59✅ | 2.34✅ | 1.69✅ | 1.35✅ | 1.59✅ | 1.19✅ | 1.21✅ | — | 0.74🟡 |
| **char** | 3.63✅ | 4.67✅ | 5.14✅ | 1.98✅ | 0.91🟡 | 1.36✅ | 1.44✅ | 1.35✅ | 1.10✅ | 1.01✅ | 2.29✅ | 1.26✅ | 1.32✅ | — | 1.52✅ |
| **f16** | 8.15✅ | 3.58✅ | 3.66✅ | 2.41✅ | 3.27✅ | 3.72✅ | 1.33✅ | 2.02✅ | 0.60🟡 | 3.93✅ | 0.99🟡 | 2.19✅ | 0.67🟡 | — | 1.06✅ |
| **f32** | 2.84✅ | 1.70✅ | 1.80✅ | 1.81✅ | 1.92✅ | 2.24✅ | 0.59🟡 | 0.71🟡 | 0.54🟡 | 1.68✅ | 2.50✅ | 1.53✅ | 1.55✅ | — | 1.59✅ |
| **f64** | 1.74✅ | 2.05✅ | 1.80✅ | 1.75✅ | 1.30✅ | 1.93✅ | 0.76🟡 | 0.86🟡 | 0.54🟡 | 1.44✅ | 2.02✅ | 1.85✅ | 1.72✅ | — | 1.48✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.59🟡 | 0.44🟠 | 0.92🟡 | 0.99🟡 | 0.80🟡 | 1.31✅ | 0.82🟡 | 0.97🟡 | 0.72🟡 | 1.27✅ | 1.25✅ | 0.93🟡 | 1.35✅ | — | 1.80✅ |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.18🔴 | 0.37🟠 | 0.44🟠 | 0.59🟡 | 0.59🟡 | 0.91🟡 | 1.01✅ | 1.48✅ | 1.30✅ | 0.58🟡 | 0.15🔴 | 1.53✅ | 1.95✅ | — | 2.18✅ |
| **u8** | 4.40✅ | 0.41🟠 | 4.96✅ | 1.74✅ | 2.40✅ | 2.19✅ | 1.88✅ | 1.84✅ | 1.65✅ | 2.55✅ | 2.36✅ | 1.45✅ | 1.94✅ | — | 1.73✅ |
| **i8** | 3.89✅ | 3.55✅ | 0.31🟠 | 2.41✅ | 2.05✅ | 2.01✅ | 1.92✅ | 1.59✅ | 1.63✅ | 2.00✅ | 2.29✅ | 1.48✅ | 1.74✅ | — | 1.64✅ |
| **i16** | 4.09✅ | 4.02✅ | 5.19✅ | 1.34✅ | 1.93✅ | 2.28✅ | 2.12✅ | 1.88✅ | 1.91✅ | 1.81✅ | 2.40✅ | 2.55✅ | 1.86✅ | — | 1.86✅ |
| **u16** | 3.77✅ | 1.98✅ | 2.16✅ | 1.85✅ | 1.53✅ | 2.18✅ | 2.16✅ | 1.73✅ | 1.77✅ | 1.43✅ | 2.33✅ | 2.06✅ | 1.59✅ | — | 1.62✅ |
| **i32** | 1.95✅ | 1.47✅ | 1.44✅ | 1.47✅ | 1.76✅ | 1.67✅ | 1.96✅ | 1.31✅ | 1.25✅ | 1.88✅ | 2.35✅ | 1.24✅ | 1.71✅ | — | 1.47✅ |
| **u32** | 0.81🟡 | 0.95🟡 | 1.94✅ | 1.27✅ | 1.87✅ | 1.18✅ | 1.67✅ | 1.68✅ | 1.28✅ | 1.91✅ | 2.21✅ | 1.43✅ | 1.61✅ | — | 1.32✅ |
| **i64** | 0.92🟡 | 1.57✅ | 0.91🟡 | 1.38✅ | 1.01✅ | 1.16✅ | 1.45✅ | 1.23✅ | 1.26✅ | 0.95🟡 | 1.24✅ | 1.15✅ | 1.62✅ | — | 1.37✅ |
| **u64** | 0.76🟡 | 0.83🟡 | 1.11✅ | 0.92🟡 | 0.96🟡 | 1.40✅ | 1.26✅ | 1.07✅ | 1.07✅ | 1.05✅ | 1.20✅ | 0.84🟡 | 0.96🟡 | — | 1.24✅ |
| **char** | 3.50✅ | 4.84✅ | 3.88✅ | 1.56✅ | 1.38✅ | 1.34✅ | 1.24✅ | 1.54✅ | 1.66✅ | 1.31✅ | 2.25✅ | 1.30✅ | 1.57✅ | — | 1.51✅ |
| **f16** | 7.22✅ | 3.43✅ | 3.49✅ | 2.83✅ | 3.53✅ | 3.68✅ | 1.25✅ | 1.98✅ | 0.61🟡 | 3.26✅ | 1.33✅ | 1.96✅ | 0.72🟡 | — | 1.03✅ |
| **f32** | 2.90✅ | 1.52✅ | 1.54✅ | 2.03✅ | 1.75✅ | 1.99✅ | 0.65🟡 | 0.70🟡 | 0.52🟡 | 1.94✅ | 2.16✅ | 1.65✅ | 1.96✅ | — | 1.60✅ |
| **f64** | 1.67✅ | 1.63✅ | 1.57✅ | 1.47✅ | 1.31✅ | 1.87✅ | 0.78🟡 | 0.88🟡 | 0.55🟡 | 1.22✅ | 1.57✅ | 1.83✅ | 1.44✅ | — | 1.93✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.73🟡 | 0.57🟡 | 1.22✅ | 1.20✅ | 1.46✅ | 1.70✅ | 0.90🟡 | 1.11✅ | 0.69🟡 | 1.08✅ | 1.37✅ | 1.38✅ | 1.26✅ | — | 1.08✅ |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.32🟠 | 0.74🟡 | 0.94🟡 | 1.02✅ | 1.11✅ | 1.21✅ | 1.52✅ | 1.72✅ | 1.65✅ | 1.03✅ | 0.27🟠 | 1.50✅ | 1.59✅ | — | 1.69✅ |
| **u8** | 2.99✅ | 0.36🟠 | 4.96✅ | 2.53✅ | 2.20✅ | 2.28✅ | 2.13✅ | 1.91✅ | 2.16✅ | 1.16✅ | 2.39✅ | 1.40✅ | 1.84✅ | — | 2.25✅ |
| **i8** | 4.05✅ | 3.57✅ | 0.39🟠 | 2.21✅ | 2.13✅ | 2.03✅ | 2.04✅ | 1.61✅ | 1.54✅ | 1.91✅ | 2.42✅ | 1.44✅ | 1.54✅ | — | 1.58✅ |
| **i16** | 5.06✅ | 3.95✅ | 5.72✅ | 1.46✅ | 1.88✅ | 2.22✅ | 2.13✅ | 2.36✅ | 2.37✅ | 1.99✅ | 2.35✅ | 2.48✅ | 1.98✅ | — | 2.09✅ |
| **u16** | 3.85✅ | 5.22✅ | 2.11✅ | 1.75✅ | 1.51✅ | 2.12✅ | 2.06✅ | 2.03✅ | 1.64✅ | 1.27✅ | 2.54✅ | 2.36✅ | 1.40✅ | — | 1.54✅ |
| **i32** | 1.88✅ | 1.92✅ | 1.76✅ | 1.63✅ | 2.23✅ | 1.80✅ | 1.97✅ | 1.68✅ | 1.59✅ | 1.69✅ | 1.67✅ | 1.02✅ | 1.61✅ | — | 1.77✅ |
| **u32** | 1.21✅ | 0.78🟡 | 1.52✅ | 0.35🟠 | 1.26✅ | 0.50🟡 | 1.27✅ | 0.55🟡 | 0.71🟡 | 1.46✅ | 1.51✅ | 1.18✅ | 0.86🟡 | — | 1.27✅ |
| **i64** | 0.85🟡 | 1.34✅ | 0.96🟡 | 1.62✅ | 1.10✅ | 1.28✅ | 1.04✅ | 1.14✅ | 1.28✅ | 0.92🟡 | 1.21✅ | 0.95🟡 | 1.21✅ | — | 1.24✅ |
| **u64** | 0.86🟡 | 0.94🟡 | 1.26✅ | 1.15✅ | 1.20✅ | 1.71✅ | 1.41✅ | 1.26✅ | 1.22✅ | 1.10✅ | 1.45✅ | 0.99🟡 | 0.99🟡 | — | 1.20✅ |
| **char** | 2.81✅ | 4.65✅ | 3.72✅ | 1.19✅ | 1.57✅ | 1.45✅ | 1.59✅ | 1.72✅ | 1.42✅ | 1.16✅ | 2.16✅ | 1.25✅ | 1.61✅ | — | 1.48✅ |
| **f16** | 7.36✅ | 3.35✅ | 3.56✅ | 3.49✅ | 3.42✅ | 4.07✅ | 1.24✅ | 1.97✅ | 0.66🟡 | 3.37✅ | 1.35✅ | 2.19✅ | 0.77🟡 | — | 1.11✅ |
| **f32** | 3.04✅ | 1.47✅ | 1.53✅ | 2.10✅ | 2.00✅ | 2.08✅ | 0.62🟡 | 0.81🟡 | 0.52🟡 | 1.76✅ | 2.46✅ | 1.87✅ | 1.91✅ | — | 1.65✅ |
| **f64** | 1.73✅ | 1.64✅ | 1.61✅ | 1.39✅ | 1.38✅ | 1.79✅ | 0.75🟡 | 0.78🟡 | 0.53🟡 | 1.53✅ | 1.53✅ | 1.59✅ | 1.43✅ | — | 1.30✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.59🟡 | 0.94🟡 | 1.00✅ | 1.38✅ | 1.23✅ | 1.54✅ | 0.91🟡 | 0.93🟡 | 0.86🟡 | 2.10✅ | 1.40✅ | 1.09✅ | 1.23✅ | — | 1.80✅ |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.75🟡 | 0.78🟡 | 0.80🟡 | 0.67🟡 | 1.00🟡 | 1.29✅ | 1.32✅ | 1.82✅ | 1.81✅ | 1.02✅ | 0.21🟠 | 1.55✅ | 1.86✅ | — | 1.76✅ |
| **u8** | 0.97🟡 | 4.55✅ | 4.03✅ | 2.51✅ | 2.49✅ | 1.59✅ | 1.72✅ | 2.74✅ | 2.46✅ | 2.62✅ | 1.85✅ | 1.47✅ | 2.05✅ | — | 2.10✅ |
| **i8** | 0.86🟡 | 5.16✅ | 4.83✅ | 1.82✅ | 2.23✅ | 1.42✅ | 1.40✅ | 1.77✅ | 2.03✅ | 2.47✅ | 1.77✅ | 1.23✅ | 1.80✅ | — | 1.75✅ |
| **i16** | 3.07✅ | 3.40✅ | 2.68✅ | 2.55✅ | 2.61✅ | 1.66✅ | 1.56✅ | 1.84✅ | 2.01✅ | 2.36✅ | 1.94✅ | 1.61✅ | 1.75✅ | — | 1.39✅ |
| **u16** | 2.22✅ | 2.73✅ | 2.61✅ | 2.15✅ | 2.17✅ | 1.59✅ | 1.66✅ | 2.01✅ | 1.94✅ | 2.36✅ | 1.72✅ | 1.38✅ | 1.63✅ | — | 1.66✅ |
| **i32** | 0.43🟠 | 0.50🟡 | 3.14✅ | 2.47✅ | 0.59🟡 | 1.19✅ | 1.24✅ | 1.57✅ | 1.99✅ | 0.51🟡 | 1.68✅ | 1.37✅ | 1.60✅ | — | 1.42✅ |
| **u32** | 0.44🟠 | 0.36🟠 | 0.49🟠 | 0.37🟠 | 0.56🟡 | 1.17✅ | 1.15✅ | 1.21✅ | 1.15✅ | 0.64🟡 | 1.50✅ | 1.19✅ | 1.25✅ | — | 1.19✅ |
| **i64** | 0.36🟠 | 0.35🟠 | 0.38🟠 | 0.45🟠 | 0.42🟠 | 0.52🟡 | 0.60🟡 | 0.79🟡 | 0.75🟡 | 0.35🟠 | 0.88🟡 | 1.28✅ | 1.26✅ | — | 1.29✅ |
| **u64** | 0.40🟠 | 0.36🟠 | 0.19🔴 | 0.44🟠 | 0.36🟠 | 0.57🟡 | 0.54🟡 | 1.43✅ | 1.21✅ | 0.48🟠 | 1.15✅ | 0.86🟡 | 1.06✅ | — | 2.16✅ |
| **char** | 2.13✅ | 2.74✅ | 2.89✅ | 1.98✅ | 1.96✅ | 1.47✅ | 1.49✅ | 1.77✅ | 1.79✅ | 2.04✅ | 1.84✅ | 1.24✅ | 1.51✅ | — | 1.59✅ |
| **f16** | 2.95✅ | 2.11✅ | 2.23✅ | 2.29✅ | 2.25✅ | 2.61✅ | 1.16✅ | 1.65✅ | 0.64🟡 | 1.89✅ | 1.85✅ | 1.72✅ | 0.74🟡 | — | 1.02✅ |
| **f32** | 0.79🟡 | 0.64🟡 | 0.52🟡 | 0.56🟡 | 0.52🟡 | 0.83🟡 | 0.41🟠 | 0.66🟡 | 0.42🟠 | 0.62🟡 | 1.77✅ | 1.28✅ | 1.50✅ | — | 1.44✅ |
| **f64** | 0.56🟡 | 1.73✅ | 1.59✅ | 0.48🟠 | 0.42🟠 | 1.60✅ | 0.52🟡 | 0.82🟡 | 0.61🟡 | 0.45🟠 | 1.16✅ | 0.92🟡 | 1.29✅ | — | 1.38✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.75🟡 | 0.48🟠 | 0.47🟠 | 0.63🟡 | 0.78🟡 | 1.06✅ | 0.74🟡 | 0.91🟡 | 0.82🟡 | 0.58🟡 | 1.19✅ | 1.13✅ | 1.05✅ | — | 1.35✅ |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.70🟡 | 0.71🟡 | 0.75🟡 | 0.73🟡 | 0.72🟡 | 1.07✅ | 1.14✅ | 2.13✅ | 2.07✅ | 0.69🟡 | 0.23🟠 | 1.15✅ | 1.45✅ | — | 1.24✅ |
| **u8** | 0.88🟡 | 3.34✅ | 2.68✅ | 2.50✅ | 2.47✅ | 1.09✅ | 1.17✅ | 2.02✅ | 1.89✅ | 2.30✅ | 1.67✅ | 1.18✅ | 1.91✅ | — | 2.69✅ |
| **i8** | 0.82🟡 | 4.41✅ | 3.99✅ | 1.56✅ | 1.26✅ | 1.14✅ | 1.26✅ | 1.94✅ | 1.91✅ | 2.23✅ | 1.78✅ | 1.15✅ | 1.74✅ | — | 2.69✅ |
| **i16** | 1.91✅ | 2.45✅ | 1.83✅ | 4.75✅ | 3.31✅ | 1.21✅ | 1.11✅ | 2.07✅ | 1.83✅ | 2.77✅ | 1.40✅ | 1.10✅ | 1.85✅ | — | 2.27✅ |
| **u16** | 1.69✅ | 2.40✅ | 2.38✅ | 1.96✅ | 2.15✅ | 1.18✅ | 1.30✅ | 2.10✅ | 2.13✅ | 2.71✅ | 1.80✅ | 1.26✅ | 1.83✅ | — | 2.11✅ |
| **i32** | 0.38🟠 | 0.49🟠 | 0.48🟠 | 0.50🟠 | 0.36🟠 | 1.16✅ | 1.02✅ | 1.98✅ | 1.72✅ | 0.36🟠 | 1.66✅ | 1.18✅ | 1.49✅ | — | 1.31✅ |
| **u32** | 0.36🟠 | 0.34🟠 | 0.49🟠 | 0.36🟠 | 0.50🟠 | 0.82🟡 | 0.86🟡 | 1.22✅ | 1.43✅ | 0.45🟠 | 1.38✅ | 0.80🟡 | 1.34✅ | — | 1.39✅ |
| **i64** | 0.41🟠 | 1.15✅ | 0.38🟠 | 0.39🟠 | 0.31🟠 | 0.64🟡 | 0.64🟡 | 1.57✅ | 1.63✅ | 0.35🟠 | 0.98🟡 | 0.95🟡 | 1.29✅ | — | 1.52✅ |
| **u64** | 0.39🟠 | 0.41🟠 | 0.37🟠 | 0.37🟠 | 0.38🟠 | 0.60🟡 | 0.67🟡 | 1.55✅ | 1.49✅ | 0.40🟠 | 1.12✅ | 0.91🟡 | 1.29✅ | — | 1.41✅ |
| **char** | 1.55✅ | 2.26✅ | 2.20✅ | 1.82✅ | 2.44✅ | 1.20✅ | 1.25✅ | 1.92✅ | 1.65✅ | 2.20✅ | 1.62✅ | 1.08✅ | 1.68✅ | — | 1.97✅ |
| **f16** | 4.43✅ | 1.68✅ | 1.80✅ | 2.00✅ | 1.82✅ | 2.15✅ | 1.01✅ | 1.53✅ | 0.63🟡 | 1.21✅ | 2.38✅ | 1.30✅ | 0.71🟡 | — | 0.89🟡 |
| **f32** | 0.81🟡 | 0.33🟠 | 0.45🟠 | 1.75✅ | 1.59✅ | 0.75🟡 | 0.39🟠 | 0.79🟡 | 0.43🟠 | 0.38🟠 | 1.73✅ | 0.94🟡 | 1.58✅ | — | 1.65✅ |
| **f64** | 0.65🟡 | 0.34🟠 | 1.80✅ | 0.43🟠 | 1.54✅ | 1.89✅ | 0.55🟡 | 0.74🟡 | 0.47🟠 | 0.32🟠 | 1.17✅ | 0.96🟡 | 1.53✅ | — | 1.27✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.39🟠 | 0.60🟡 | 0.50🟡 | 1.35✅ | 0.61🟡 | 0.98🟡 | 1.59✅ | 0.83🟡 | 0.54🟡 | 0.53🟡 | 0.93🟡 | 0.99🟡 | 1.32✅ | — | 1.10✅ |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.02🔴 | 0.23🟠 | 0.23🟠 | 0.28🟠 | 0.27🟠 | 0.31🟠 | 0.30🟠 | 0.60🟡 | 0.64🟡 | 0.28🟠 | 0.27🟠 | 0.43🟠 | 0.81🟡 | — | 0.86🟡 |
| **u8** | 0.25🟠 | 0.02🔴 | 0.18🔴 | 0.32🟠 | 0.32🟠 | 0.35🟠 | 0.37🟠 | 0.75🟡 | 0.83🟡 | 0.32🟠 | 0.53🟡 | 0.42🟠 | 0.73🟡 | — | 0.90🟡 |
| **i8** | 0.27🟠 | 0.31🟠 | 0.02🔴 | 0.27🟠 | 0.37🟠 | 0.30🟠 | 0.38🟠 | 0.70🟡 | 0.68🟡 | 0.33🟠 | 0.69🟡 | 0.40🟠 | 0.69🟡 | — | 0.88🟡 |
| **i16** | 0.26🟠 | 0.31🟠 | 0.24🟠 | 0.21🟠 | 0.31🟠 | 0.34🟠 | 0.35🟠 | 0.75🟡 | 0.71🟡 | 0.78🟡 | 0.55🟡 | 0.35🟠 | 0.65🟡 | — | 0.78🟡 |
| **u16** | 0.19🔴 | 0.31🟠 | 0.30🟠 | 0.27🟠 | 0.21🟠 | 0.36🟠 | 0.38🟠 | 0.69🟡 | 0.66🟡 | 0.21🟠 | 0.49🟠 | 0.36🟠 | 0.76🟡 | — | 0.76🟡 |
| **i32** | 0.20🟠 | 0.27🟠 | 0.27🟠 | 0.30🟠 | 0.28🟠 | 0.30🟠 | 0.28🟠 | 0.50🟠 | 0.39🟠 | 0.25🟠 | 0.58🟡 | 0.44🟠 | 0.60🟡 | — | 0.79🟡 |
| **u32** | 0.21🟠 | 0.18🔴 | 0.25🟠 | 0.22🟠 | 0.27🟠 | 0.24🟠 | 0.25🟠 | 0.59🟡 | 0.69🟡 | 0.31🟠 | 0.56🟡 | 0.34🟠 | 0.66🟡 | — | 0.78🟡 |
| **i64** | 0.27🟠 | 0.25🟠 | 0.25🟠 | 0.22🟠 | 0.23🟠 | 0.31🟠 | 0.27🟠 | 0.58🟡 | 0.64🟡 | 0.21🟠 | 0.51🟡 | 0.26🟠 | 0.57🟡 | — | 0.75🟡 |
| **u64** | 0.27🟠 | 0.27🟠 | 0.25🟠 | 0.24🟠 | 0.28🟠 | 0.30🟠 | 0.33🟠 | 0.73🟡 | 0.72🟡 | 0.27🟠 | 0.46🟠 | 0.37🟠 | 0.92🟡 | — | 0.86🟡 |
| **char** | 0.31🟠 | 0.31🟠 | 0.29🟠 | 0.24🟠 | 0.18🔴 | 0.30🟠 | 0.38🟠 | 0.74🟡 | 0.71🟡 | 0.19🔴 | 0.44🟠 | 0.39🟠 | 0.62🟡 | — | 0.84🟡 |
| **f16** | 0.69🟡 | 0.39🟠 | 0.39🟠 | 0.43🟠 | 0.43🟠 | 0.48🟠 | 0.58🟡 | 0.51🟡 | 0.60🟡 | 0.40🟠 | 0.20🔴 | 0.44🟠 | 0.52🟡 | — | 0.62🟡 |
| **f32** | 0.33🟠 | 0.12🔴 | 0.15🔴 | 0.17🔴 | 0.19🔴 | 0.25🟠 | 0.29🟠 | 0.47🟠 | 0.48🟠 | 0.18🔴 | 0.73🟡 | 0.38🟠 | 0.73🟡 | — | 1.00✅ |
| **f64** | 0.39🟠 | 0.13🔴 | 0.12🔴 | 0.23🟠 | 0.18🔴 | 0.27🟠 | 0.32🟠 | 0.56🟡 | 0.48🟠 | 0.17🔴 | 0.70🟡 | 0.29🟠 | 0.74🟡 | — | 0.84🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.35🟠 | 0.09🔴 | 0.09🔴 | 0.19🔴 | 0.14🔴 | 0.24🟠 | 0.19🔴 | 0.65🟡 | 0.34🟠 | 0.15🔴 | 0.75🟡 | 0.27🟠 | 0.75🟡 | — | 0.94🟡 |

## Lagging cells (<1.0) — the worklist  (514 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| bool\|bcast\|bool | 1.8113 | 0.0272 | 0.02 🔴 |
| u8\|bcast\|u8 | 1.6243 | 0.0257 | 0.02 🔴 |
| i8\|bcast\|i8 | 1.5715 | 0.0273 | 0.02 🔴 |
| c128\|bcast\|u8 | 4.5139 | 0.3900 | 0.09 🔴 |
| c128\|bcast\|i8 | 4.2811 | 0.3841 | 0.09 🔴 |
| f32\|bcast\|u8 | 3.5565 | 0.4136 | 0.12 🔴 |
| f64\|bcast\|i8 | 3.3925 | 0.4078 | 0.12 🔴 |
| f64\|bcast\|u8 | 3.0507 | 0.4105 | 0.13 🔴 |
| c128\|bcast\|u16 | 4.3201 | 0.6075 | 0.14 🔴 |
| i8\|C\|i8 | 0.1046 | 0.0150 | 0.14 🔴 |
| c128\|bcast\|char | 4.0644 | 0.6110 | 0.15 🔴 |
| bool\|sliced\|f16 | 11.4558 | 1.7272 | 0.15 🔴 |
| f32\|bcast\|i8 | 3.5592 | 0.5431 | 0.15 🔴 |
| f64\|bcast\|char | 3.4871 | 0.5871 | 0.17 🔴 |
| f32\|bcast\|i16 | 3.6609 | 0.6197 | 0.17 🔴 |
| f32\|bcast\|char | 3.6922 | 0.6479 | 0.18 🔴 |
| u8\|bcast\|i8 | 1.6194 | 0.2858 | 0.18 🔴 |
| bool\|sliced\|bool | 0.1003 | 0.0177 | 0.18 🔴 |
| f64\|bcast\|u16 | 3.3832 | 0.6096 | 0.18 🔴 |
| u32\|bcast\|u8 | 2.0400 | 0.3716 | 0.18 🔴 |
| char\|bcast\|u16 | 2.5620 | 0.4719 | 0.18 🔴 |
| c128\|bcast\|u32 | 4.6413 | 0.8598 | 0.19 🔴 |
| char\|bcast\|char | 2.3169 | 0.4308 | 0.19 🔴 |
| c128\|bcast\|i16 | 4.0141 | 0.7541 | 0.19 🔴 |
| f32\|bcast\|u16 | 3.5609 | 0.6707 | 0.19 🔴 |
| u16\|bcast\|bool | 1.7689 | 0.3341 | 0.19 🔴 |
| u64\|negcol\|i8 | 1.6407 | 0.3101 | 0.19 🔴 |
| f16\|bcast\|f16 | 2.2311 | 0.4369 | 0.20 🔴 |
| i32\|bcast\|bool | 2.1236 | 0.4301 | 0.20 🟠 |
| bool\|negcol\|f16 | 11.6741 | 2.4007 | 0.21 🟠 |
| bool\|C\|bool | 0.0718 | 0.0148 | 0.21 🟠 |
| i16\|bcast\|i16 | 2.2078 | 0.4566 | 0.21 🟠 |
| u32\|bcast\|bool | 2.1188 | 0.4449 | 0.21 🟠 |
| i64\|bcast\|char | 2.8487 | 0.6039 | 0.21 🟠 |
| u16\|bcast\|u16 | 2.0921 | 0.4440 | 0.21 🟠 |
| u16\|bcast\|char | 2.1947 | 0.4672 | 0.21 🟠 |
| i64\|bcast\|i16 | 2.7912 | 0.6121 | 0.22 🟠 |
| bool\|T\|bool | 0.0691 | 0.0152 | 0.22 🟠 |
| u32\|bcast\|i16 | 2.7109 | 0.5971 | 0.22 🟠 |
| bool\|bcast\|i8 | 1.8670 | 0.4236 | 0.23 🟠 |
| bool\|bcast\|u8 | 1.8447 | 0.4199 | 0.23 🟠 |
| i64\|bcast\|u16 | 2.7804 | 0.6387 | 0.23 🟠 |
| bool\|T\|f16 | 7.2373 | 1.6754 | 0.23 🟠 |
| f64\|bcast\|i16 | 3.3888 | 0.7895 | 0.23 🟠 |
| bool\|strided\|f16 | 5.7726 | 1.3480 | 0.23 🟠 |
| i16\|bcast\|i8 | 1.7868 | 0.4222 | 0.24 🟠 |
| u64\|bcast\|i16 | 2.6624 | 0.6354 | 0.24 🟠 |
| bool\|C\|f16 | 7.0703 | 1.6945 | 0.24 🟠 |
| c128\|bcast\|i32 | 3.9803 | 0.9573 | 0.24 🟠 |
| char\|bcast\|i16 | 2.4031 | 0.5811 | 0.24 🟠 |
| u32\|bcast\|i32 | 3.4858 | 0.8480 | 0.24 🟠 |
| bool\|F\|f16 | 6.9657 | 1.6992 | 0.24 🟠 |
| f32\|bcast\|i32 | 3.5744 | 0.8775 | 0.25 🟠 |
| i64\|bcast\|u8 | 2.1083 | 0.5185 | 0.25 🟠 |
| i32\|bcast\|char | 2.6240 | 0.6495 | 0.25 🟠 |
| u8\|bcast\|bool | 1.9844 | 0.4914 | 0.25 🟠 |
| i64\|bcast\|i8 | 2.1563 | 0.5343 | 0.25 🟠 |
| u32\|bcast\|i8 | 2.0545 | 0.5110 | 0.25 🟠 |
| u32\|bcast\|u32 | 3.3853 | 0.8552 | 0.25 🟠 |
| u64\|bcast\|i8 | 2.0183 | 0.5102 | 0.25 🟠 |
| u8\|C\|u8 | 0.0973 | 0.0251 | 0.26 🟠 |
| i64\|bcast\|f32 | 3.6905 | 0.9665 | 0.26 🟠 |
| i16\|bcast\|bool | 1.7515 | 0.4636 | 0.26 🟠 |
| u16\|bcast\|i16 | 2.2582 | 0.5986 | 0.27 🟠 |
| u64\|bcast\|u8 | 2.0283 | 0.5394 | 0.27 🟠 |
| i8\|bcast\|bool | 1.9593 | 0.5222 | 0.27 🟠 |
| i64\|bcast\|u32 | 2.8639 | 0.7640 | 0.27 🟠 |
| u8\|T\|u8 | 0.0937 | 0.0250 | 0.27 🟠 |
| u64\|bcast\|bool | 1.8669 | 0.4994 | 0.27 🟠 |
| i8\|bcast\|i16 | 2.2644 | 0.6112 | 0.27 🟠 |
| bool\|bcast\|f16 | 11.0593 | 2.9995 | 0.27 🟠 |
| f64\|bcast\|i32 | 3.2458 | 0.8805 | 0.27 🟠 |
| c128\|bcast\|f32 | 3.7639 | 1.0218 | 0.27 🟠 |
| i64\|bcast\|bool | 1.8589 | 0.5051 | 0.27 🟠 |
| bool\|negrow\|f16 | 11.5863 | 3.1538 | 0.27 🟠 |
| bool\|bcast\|u16 | 2.6599 | 0.7241 | 0.27 🟠 |
| i32\|bcast\|i8 | 1.8775 | 0.5130 | 0.27 🟠 |
| u32\|bcast\|u16 | 2.7146 | 0.7424 | 0.27 🟠 |
| u64\|bcast\|char | 2.7200 | 0.7443 | 0.27 🟠 |
| i32\|bcast\|u8 | 1.9465 | 0.5341 | 0.27 🟠 |
| i32\|bcast\|u16 | 2.5024 | 0.6882 | 0.28 🟠 |
| u64\|bcast\|u16 | 2.7183 | 0.7524 | 0.28 🟠 |
| bool\|bcast\|char | 2.6037 | 0.7249 | 0.28 🟠 |
| i32\|bcast\|u32 | 3.3740 | 0.9457 | 0.28 🟠 |
| bool\|bcast\|i16 | 2.7184 | 0.7641 | 0.28 🟠 |
| f32\|bcast\|u32 | 3.1311 | 0.8977 | 0.29 🟠 |
| f64\|bcast\|f32 | 3.3738 | 0.9724 | 0.29 🟠 |
| char\|bcast\|i8 | 1.7592 | 0.5102 | 0.29 🟠 |
| i8\|bcast\|i32 | 2.7741 | 0.8248 | 0.30 🟠 |
| u16\|bcast\|i8 | 1.7383 | 0.5192 | 0.30 🟠 |
| bool\|bcast\|u32 | 3.0995 | 0.9265 | 0.30 🟠 |
| i32\|bcast\|i16 | 2.4715 | 0.7396 | 0.30 🟠 |
| char\|bcast\|i32 | 3.1435 | 0.9417 | 0.30 🟠 |
| i32\|bcast\|i32 | 3.1778 | 0.9532 | 0.30 🟠 |
| u64\|bcast\|i32 | 2.7831 | 0.8453 | 0.30 🟠 |
| i16\|bcast\|u16 | 2.3119 | 0.7057 | 0.31 🟠 |
| u32\|bcast\|char | 2.4585 | 0.7518 | 0.31 🟠 |
| i64\|bcast\|i32 | 2.8399 | 0.8714 | 0.31 🟠 |
| i16\|bcast\|u8 | 1.7976 | 0.5520 | 0.31 🟠 |
| i64\|strided\|u16 | 0.8416 | 0.2590 | 0.31 🟠 |
| char\|bcast\|bool | 1.4886 | 0.4593 | 0.31 🟠 |
| bool\|bcast\|i32 | 3.0236 | 0.9367 | 0.31 🟠 |
| i8\|bcast\|u8 | 1.6101 | 0.5026 | 0.31 🟠 |
| char\|bcast\|u8 | 1.7315 | 0.5408 | 0.31 🟠 |
| i8\|sliced\|i8 | 0.0965 | 0.0302 | 0.31 🟠 |
| u16\|bcast\|u8 | 1.7153 | 0.5403 | 0.31 🟠 |
| u8\|bcast\|char | 2.3111 | 0.7353 | 0.32 🟠 |
| bool\|negrow\|bool | 0.1116 | 0.0355 | 0.32 🟠 |
| u8\|bcast\|i16 | 2.2569 | 0.7199 | 0.32 🟠 |
| u8\|bcast\|u16 | 2.2698 | 0.7326 | 0.32 🟠 |
| f64\|bcast\|u32 | 2.9459 | 0.9509 | 0.32 🟠 |
| f64\|strided\|char | 0.8457 | 0.2748 | 0.32 🟠 |
| f32\|bcast\|bool | 2.3635 | 0.7753 | 0.33 🟠 |
| i8\|bcast\|char | 2.2664 | 0.7469 | 0.33 🟠 |
| f32\|strided\|u8 | 0.5497 | 0.1819 | 0.33 🟠 |
| u64\|bcast\|u32 | 2.7916 | 0.9289 | 0.33 🟠 |
| c128\|bcast\|u64 | 5.1173 | 1.7213 | 0.34 🟠 |
| i16\|bcast\|i32 | 2.7791 | 0.9393 | 0.34 🟠 |
| u32\|bcast\|f32 | 2.7734 | 0.9398 | 0.34 🟠 |
| u32\|strided\|u8 | 0.5625 | 0.1928 | 0.34 🟠 |
| f64\|strided\|u8 | 0.7721 | 0.2652 | 0.34 🟠 |
| i16\|bcast\|f32 | 3.0589 | 1.0592 | 0.35 🟠 |
| i64\|strided\|char | 0.8259 | 0.2862 | 0.35 🟠 |
| i64\|C\|u8 | 0.5745 | 0.1997 | 0.35 🟠 |
| i64\|negcol\|char | 1.9591 | 0.6814 | 0.35 🟠 |
| u8\|bcast\|i32 | 2.4305 | 0.8459 | 0.35 🟠 |
| c128\|bcast\|bool | 2.3255 | 0.8145 | 0.35 🟠 |
| u32\|negrow\|i16 | 1.7705 | 0.6250 | 0.35 🟠 |
| i16\|bcast\|u32 | 2.6490 | 0.9371 | 0.35 🟠 |
| i64\|negcol\|u8 | 1.6651 | 0.5893 | 0.35 🟠 |
| u16\|bcast\|f32 | 2.7968 | 0.9939 | 0.36 🟠 |
| u32\|strided\|i16 | 0.5858 | 0.2084 | 0.36 🟠 |
| i32\|strided\|u16 | 0.5470 | 0.1948 | 0.36 🟠 |
| u32\|negcol\|u8 | 1.1060 | 0.3942 | 0.36 🟠 |
| u8\|negrow\|u8 | 0.0951 | 0.0340 | 0.36 🟠 |
| u64\|negcol\|u16 | 1.7729 | 0.6348 | 0.36 🟠 |
| i32\|strided\|char | 0.5849 | 0.2095 | 0.36 🟠 |
| u16\|bcast\|i32 | 2.6378 | 0.9469 | 0.36 🟠 |
| u32\|strided\|bool | 0.5802 | 0.2084 | 0.36 🟠 |
| u64\|T\|u8 | 0.5598 | 0.2012 | 0.36 🟠 |
| u64\|negcol\|u8 | 1.5432 | 0.5597 | 0.36 🟠 |
| i64\|negcol\|bool | 1.6658 | 0.6064 | 0.36 🟠 |
| bool\|sliced\|u8 | 0.5202 | 0.1928 | 0.37 🟠 |
| u32\|negcol\|i16 | 1.2710 | 0.4713 | 0.37 🟠 |
| u64\|strided\|i16 | 0.8298 | 0.3079 | 0.37 🟠 |
| u64\|bcast\|f32 | 3.0971 | 1.1528 | 0.37 🟠 |
| u8\|bcast\|u32 | 2.4204 | 0.9017 | 0.37 🟠 |
| i8\|bcast\|u16 | 2.0154 | 0.7527 | 0.37 🟠 |
| u64\|strided\|i8 | 0.7747 | 0.2903 | 0.37 🟠 |
| char\|bcast\|u32 | 2.6092 | 0.9844 | 0.38 🟠 |
| u64\|strided\|u16 | 0.8260 | 0.3120 | 0.38 🟠 |
| f32\|strided\|char | 0.5832 | 0.2206 | 0.38 🟠 |
| i64\|negcol\|i8 | 1.6162 | 0.6113 | 0.38 🟠 |
| u16\|bcast\|u32 | 2.5220 | 0.9551 | 0.38 🟠 |
| i8\|bcast\|u32 | 2.4750 | 0.9402 | 0.38 🟠 |
| i32\|strided\|bool | 0.5481 | 0.2085 | 0.38 🟠 |
| f32\|bcast\|f32 | 2.4511 | 0.9341 | 0.38 🟠 |
| i64\|strided\|i8 | 0.7761 | 0.2963 | 0.38 🟠 |
| f64\|bcast\|bool | 1.9621 | 0.7569 | 0.39 🟠 |
| c128\|strided\|bool | 1.6879 | 0.6511 | 0.39 🟠 |
| u64\|strided\|bool | 0.7849 | 0.3037 | 0.39 🟠 |
| i64\|strided\|i16 | 0.8435 | 0.3266 | 0.39 🟠 |
| char\|bcast\|f32 | 2.5828 | 1.0005 | 0.39 🟠 |
| f32\|strided\|u32 | 1.3758 | 0.5353 | 0.39 🟠 |
| i8\|negrow\|i8 | 0.0933 | 0.0365 | 0.39 🟠 |
| f16\|bcast\|u8 | 4.9190 | 1.9265 | 0.39 🟠 |
| i32\|bcast\|u64 | 3.7448 | 1.4756 | 0.39 🟠 |
| f16\|bcast\|i8 | 5.1087 | 2.0156 | 0.39 🟠 |
| i8\|bcast\|f32 | 2.4859 | 0.9965 | 0.40 🟠 |
| f16\|bcast\|char | 5.3223 | 2.1358 | 0.40 🟠 |
| u64\|negcol\|bool | 1.6328 | 0.6569 | 0.40 🟠 |
| u64\|strided\|char | 0.8114 | 0.3274 | 0.40 🟠 |
| u8\|sliced\|u8 | 0.0901 | 0.0365 | 0.41 🟠 |
| i64\|strided\|bool | 0.7818 | 0.3185 | 0.41 🟠 |
| f32\|negcol\|u32 | 2.4260 | 0.9942 | 0.41 🟠 |
| u64\|strided\|u8 | 0.7362 | 0.3046 | 0.41 🟠 |
| i64\|negcol\|u16 | 1.8071 | 0.7564 | 0.42 🟠 |
| f64\|negcol\|u16 | 1.6882 | 0.7076 | 0.42 🟠 |
| u8\|bcast\|f32 | 2.4795 | 1.0434 | 0.42 🟠 |
| f32\|negcol\|u64 | 4.4013 | 1.8546 | 0.42 🟠 |
| f16\|bcast\|u16 | 5.0953 | 2.1698 | 0.43 🟠 |
| f64\|strided\|i16 | 0.8160 | 0.3476 | 0.43 🟠 |
| f32\|strided\|u64 | 2.4301 | 1.0354 | 0.43 🟠 |
| i32\|negcol\|bool | 1.0534 | 0.4492 | 0.43 🟠 |
| bool\|bcast\|f32 | 2.4279 | 1.0360 | 0.43 🟠 |
| f16\|bcast\|i16 | 5.3072 | 2.2952 | 0.43 🟠 |
| u32\|negcol\|bool | 1.0691 | 0.4656 | 0.44 🟠 |
| f16\|bcast\|f32 | 4.4827 | 1.9569 | 0.44 🟠 |
| char\|bcast\|f16 | 5.8495 | 2.5640 | 0.44 🟠 |
| u64\|negcol\|i16 | 1.6909 | 0.7442 | 0.44 🟠 |
| i32\|bcast\|f32 | 2.1742 | 0.9584 | 0.44 🟠 |
| c128\|T\|u8 | 0.8163 | 0.3603 | 0.44 🟠 |
| bool\|sliced\|i8 | 0.5379 | 0.2389 | 0.44 🟠 |
| i64\|negcol\|i16 | 1.8461 | 0.8242 | 0.45 🟠 |
| f64\|negcol\|char | 1.7182 | 0.7733 | 0.45 🟠 |
| f32\|strided\|i8 | 0.5652 | 0.2551 | 0.45 🟠 |
| u32\|strided\|char | 0.6257 | 0.2836 | 0.45 🟠 |
| u64\|bcast\|f16 | 6.7282 | 3.1028 | 0.46 🟠 |
| f32\|bcast\|i64 | 3.5219 | 1.6383 | 0.47 🟠 |
| i8\|T\|i8 | 0.1011 | 0.0471 | 0.47 🟠 |
| c128\|negcol\|i8 | 1.8387 | 0.8657 | 0.47 🟠 |
| f64\|strided\|u64 | 2.0043 | 0.9519 | 0.47 🟠 |
| f32\|bcast\|u64 | 3.8671 | 1.8449 | 0.48 🟠 |
| u64\|negcol\|char | 1.7593 | 0.8415 | 0.48 🟠 |
| f64\|bcast\|u64 | 3.5876 | 1.7215 | 0.48 🟠 |
| i32\|strided\|i8 | 0.5293 | 0.2544 | 0.48 🟠 |
| f64\|negcol\|i16 | 1.7079 | 0.8222 | 0.48 🟠 |
| f32\|F\|u64 | 3.7357 | 1.8056 | 0.48 🟠 |
| f16\|bcast\|i32 | 5.2505 | 2.5396 | 0.48 🟠 |
| c128\|negcol\|u8 | 1.6970 | 0.8219 | 0.48 🟠 |
| i32\|strided\|u8 | 0.5406 | 0.2627 | 0.49 🟠 |
| u32\|negcol\|i8 | 1.0888 | 0.5294 | 0.49 🟠 |
| u16\|bcast\|f16 | 5.7882 | 2.8366 | 0.49 🟠 |
| u32\|strided\|i8 | 0.5386 | 0.2641 | 0.49 🟠 |
| f32\|C\|u64 | 3.8036 | 1.8845 | 0.50 🟠 |
| i32\|bcast\|i64 | 3.0837 | 1.5343 | 0.50 🟠 |
| i32\|strided\|i16 | 0.5666 | 0.2820 | 0.50 🟠 |
| u32\|strided\|u16 | 0.6256 | 0.3117 | 0.50 🟠 |
| u32\|negrow\|i32 | 2.1058 | 1.0552 | 0.50 🟡 |
| i32\|negcol\|u8 | 1.0751 | 0.5388 | 0.50 🟡 |
| c128\|strided\|i8 | 0.9689 | 0.4868 | 0.50 🟡 |
| i64\|bcast\|f16 | 6.3213 | 3.1970 | 0.51 🟡 |
| i32\|negcol\|char | 1.1644 | 0.5985 | 0.51 🟡 |
| f16\|bcast\|i64 | 5.5429 | 2.8529 | 0.51 🟡 |
| f32\|negcol\|u16 | 1.2573 | 0.6485 | 0.52 🟡 |
| f32\|negcol\|i8 | 1.0869 | 0.5610 | 0.52 🟡 |
| f64\|negcol\|u32 | 2.2021 | 1.1391 | 0.52 🟡 |
| f32\|negrow\|u64 | 3.7309 | 1.9323 | 0.52 🟡 |
| f16\|bcast\|f64 | 4.3463 | 2.2625 | 0.52 🟡 |
| f32\|sliced\|u64 | 3.7155 | 1.9385 | 0.52 🟡 |
| i64\|negcol\|i32 | 1.9113 | 1.0006 | 0.52 🟡 |
| u8\|bcast\|f16 | 5.7223 | 3.0116 | 0.53 🟡 |
| f64\|negrow\|u64 | 3.8488 | 2.0345 | 0.53 🟡 |
| c128\|strided\|char | 1.1107 | 0.5890 | 0.53 🟡 |
| bool\|C\|u8 | 0.3497 | 0.1859 | 0.53 🟡 |
| f32\|T\|u64 | 3.7593 | 2.0231 | 0.54 🟡 |
| u64\|negcol\|u32 | 1.7581 | 0.9494 | 0.54 🟡 |
| c128\|strided\|u64 | 2.8413 | 1.5408 | 0.54 🟡 |
| f64\|T\|u64 | 3.5936 | 1.9572 | 0.54 🟡 |
| u64\|C\|i16 | 0.7588 | 0.4144 | 0.55 🟡 |
| i16\|bcast\|f16 | 5.8756 | 3.2231 | 0.55 🟡 |
| u32\|negrow\|i64 | 3.3915 | 1.8748 | 0.55 🟡 |
| f64\|strided\|u32 | 1.1081 | 0.6138 | 0.55 🟡 |
| f64\|sliced\|u64 | 3.6950 | 2.0488 | 0.55 🟡 |
| u32\|bcast\|f16 | 5.8626 | 3.2614 | 0.56 🟡 |
| u32\|negcol\|u16 | 1.1913 | 0.6659 | 0.56 🟡 |
| f32\|negcol\|i16 | 1.2112 | 0.6789 | 0.56 🟡 |
| bool\|T\|u8 | 0.3279 | 0.1844 | 0.56 🟡 |
| i64\|C\|u16 | 0.7935 | 0.4465 | 0.56 🟡 |
| f64\|bcast\|i64 | 3.0590 | 1.7237 | 0.56 🟡 |
| f64\|negcol\|bool | 1.5276 | 0.8622 | 0.56 🟡 |
| bool\|C\|i8 | 0.3422 | 0.1935 | 0.57 🟡 |
| i64\|C\|i8 | 0.6560 | 0.3712 | 0.57 🟡 |
| i64\|bcast\|f64 | 2.6903 | 1.5294 | 0.57 🟡 |
| u64\|negcol\|i32 | 1.7573 | 0.9991 | 0.57 🟡 |
| c128\|sliced\|u8 | 0.6670 | 0.3832 | 0.57 🟡 |
| f16\|bcast\|u32 | 4.3775 | 2.5274 | 0.58 🟡 |
| c128\|negcol\|char | 2.0530 | 1.1863 | 0.58 🟡 |
| i32\|bcast\|f16 | 5.6147 | 3.2488 | 0.58 🟡 |
| f32\|C\|u32 | 1.6943 | 0.9816 | 0.58 🟡 |
| f64\|F\|u64 | 3.4152 | 1.9920 | 0.58 🟡 |
| bool\|sliced\|char | 0.7710 | 0.4498 | 0.58 🟡 |
| i64\|bcast\|i64 | 2.8002 | 1.6376 | 0.58 🟡 |
| f32\|T\|u32 | 1.6757 | 0.9827 | 0.59 🟡 |
| bool\|sliced\|i16 | 0.7463 | 0.4377 | 0.59 🟡 |
| u32\|bcast\|i64 | 2.8626 | 1.6826 | 0.59 🟡 |
| f32\|F\|u32 | 1.7074 | 1.0051 | 0.59 🟡 |
| i32\|negcol\|u16 | 1.1351 | 0.6693 | 0.59 🟡 |
| f64\|C\|u64 | 3.5217 | 2.0817 | 0.59 🟡 |
| c128\|T\|bool | 1.4900 | 0.8823 | 0.59 🟡 |
| bool\|sliced\|u16 | 0.7695 | 0.4575 | 0.59 🟡 |
| c128\|negrow\|bool | 1.6111 | 0.9582 | 0.59 🟡 |
| u64\|strided\|i32 | 0.9350 | 0.5582 | 0.60 🟡 |
| i32\|bcast\|f64 | 2.8118 | 1.6834 | 0.60 🟡 |
| bool\|bcast\|i64 | 2.7130 | 1.6255 | 0.60 🟡 |
| f16\|bcast\|u64 | 5.6975 | 3.4148 | 0.60 🟡 |
| i64\|negcol\|u32 | 1.7985 | 1.0795 | 0.60 🟡 |
| c128\|strided\|u8 | 0.9051 | 0.5468 | 0.60 🟡 |
| f16\|T\|u64 | 4.6909 | 2.8343 | 0.60 🟡 |
| i64\|F\|i8 | 0.6266 | 0.3787 | 0.60 🟡 |
| c128\|strided\|u16 | 0.9700 | 0.5897 | 0.61 🟡 |
| f16\|sliced\|u64 | 4.9586 | 3.0181 | 0.61 🟡 |
| f64\|negcol\|u64 | 3.5336 | 2.1659 | 0.61 🟡 |
| char\|bcast\|f64 | 2.8011 | 1.7294 | 0.62 🟡 |
| f32\|negcol\|char | 1.1707 | 0.7263 | 0.62 🟡 |
| f32\|negrow\|u32 | 1.6565 | 1.0279 | 0.62 🟡 |
| f16\|bcast\|c128 | 5.4566 | 3.3861 | 0.62 🟡 |
| c128\|negcol\|i16 | 2.0261 | 1.2751 | 0.63 🟡 |
| f16\|strided\|u64 | 2.6354 | 1.6590 | 0.63 🟡 |
| f16\|F\|u64 | 4.7007 | 2.9695 | 0.63 🟡 |
| bool\|bcast\|u64 | 2.6368 | 1.6829 | 0.64 🟡 |
| i64\|T\|i8 | 0.6418 | 0.4100 | 0.64 🟡 |
| f32\|negcol\|u8 | 0.6450 | 0.4121 | 0.64 🟡 |
| f16\|negcol\|u64 | 5.3587 | 3.4310 | 0.64 🟡 |
| u32\|negcol\|char | 1.2548 | 0.8035 | 0.64 🟡 |
| bool\|T\|i8 | 0.3332 | 0.2134 | 0.64 🟡 |
| i64\|strided\|u32 | 0.9887 | 0.6350 | 0.64 🟡 |
| i64\|bcast\|u64 | 2.5177 | 1.6182 | 0.64 🟡 |
| i64\|strided\|i32 | 0.9275 | 0.5966 | 0.64 🟡 |
| f16\|C\|u64 | 4.7007 | 3.0269 | 0.64 🟡 |
| f64\|strided\|bool | 0.7741 | 0.5011 | 0.65 🟡 |
| u64\|C\|u8 | 0.6106 | 0.3964 | 0.65 🟡 |
| f32\|sliced\|u32 | 1.5058 | 0.9797 | 0.65 🟡 |
| c128\|bcast\|i64 | 2.7178 | 1.7689 | 0.65 🟡 |
| i16\|bcast\|f64 | 2.5707 | 1.6837 | 0.65 🟡 |
| u16\|bcast\|u64 | 2.6671 | 1.7470 | 0.66 🟡 |
| f32\|negcol\|i64 | 2.6411 | 1.7405 | 0.66 🟡 |
| c128\|F\|u64 | 4.1488 | 2.7416 | 0.66 🟡 |
| u32\|bcast\|f64 | 2.4685 | 1.6365 | 0.66 🟡 |
| f16\|negrow\|u64 | 4.7441 | 3.1478 | 0.66 🟡 |
| u32\|C\|i8 | 0.5702 | 0.3795 | 0.67 🟡 |
| f16\|T\|f64 | 3.1805 | 2.1186 | 0.67 🟡 |
| bool\|negcol\|i16 | 0.7417 | 0.4959 | 0.67 🟡 |
| u32\|T\|i8 | 0.5712 | 0.3825 | 0.67 🟡 |
| u64\|strided\|u32 | 0.9361 | 0.6312 | 0.67 🟡 |
| i8\|bcast\|u64 | 2.3376 | 1.5828 | 0.68 🟡 |
| f16\|F\|f64 | 3.0944 | 2.1005 | 0.68 🟡 |
| u32\|F\|i8 | 0.5533 | 0.3775 | 0.68 🟡 |
| c128\|sliced\|u64 | 4.2477 | 2.9123 | 0.69 🟡 |
| u16\|bcast\|i64 | 2.5167 | 1.7276 | 0.69 🟡 |
| u32\|bcast\|u64 | 2.4663 | 1.6931 | 0.69 🟡 |
| i8\|bcast\|f16 | 4.8134 | 3.3047 | 0.69 🟡 |
| f32\|C\|i64 | 2.5915 | 1.7922 | 0.69 🟡 |
| f16\|bcast\|bool | 1.8309 | 1.2671 | 0.69 🟡 |
| bool\|strided\|char | 0.3408 | 0.2366 | 0.69 🟡 |
| i8\|bcast\|f64 | 2.3422 | 1.6271 | 0.69 🟡 |
| f32\|sliced\|i64 | 2.4843 | 1.7270 | 0.70 🟡 |
| i8\|bcast\|i64 | 2.3008 | 1.6008 | 0.70 🟡 |
| bool\|strided\|bool | 0.2975 | 0.2086 | 0.70 🟡 |
| f64\|bcast\|f16 | 5.5022 | 3.8630 | 0.70 🟡 |
| char\|bcast\|u64 | 2.3215 | 1.6398 | 0.71 🟡 |
| u32\|negrow\|u64 | 2.6394 | 1.8709 | 0.71 🟡 |
| i16\|bcast\|u64 | 2.1522 | 1.5301 | 0.71 🟡 |
| bool\|strided\|u8 | 0.2924 | 0.2087 | 0.71 🟡 |
| f32\|T\|i64 | 2.5297 | 1.8076 | 0.71 🟡 |
| f16\|strided\|f64 | 1.5592 | 1.1143 | 0.71 🟡 |
| c128\|C\|u64 | 4.0127 | 2.8737 | 0.72 🟡 |
| c128\|T\|u64 | 3.9980 | 2.8651 | 0.72 🟡 |
| u64\|bcast\|u64 | 2.3333 | 1.6722 | 0.72 🟡 |
| bool\|strided\|u16 | 0.3399 | 0.2438 | 0.72 🟡 |
| u64\|F\|u8 | 0.5728 | 0.4132 | 0.72 🟡 |
| f16\|sliced\|f64 | 3.2931 | 2.3778 | 0.72 🟡 |
| f16\|C\|f64 | 3.3250 | 2.4045 | 0.72 🟡 |
| u64\|bcast\|i64 | 2.3472 | 1.7018 | 0.73 🟡 |
| c128\|sliced\|bool | 1.2007 | 0.8740 | 0.73 🟡 |
| f32\|bcast\|f64 | 2.3882 | 1.7397 | 0.73 🟡 |
| bool\|strided\|i16 | 0.3418 | 0.2499 | 0.73 🟡 |
| u8\|bcast\|f64 | 2.2628 | 1.6560 | 0.73 🟡 |
| i32\|T\|i8 | 0.5531 | 0.4060 | 0.73 🟡 |
| f32\|bcast\|f16 | 4.1749 | 3.0656 | 0.73 🟡 |
| char\|bcast\|i64 | 2.2951 | 1.6885 | 0.74 🟡 |
| f16\|negcol\|f64 | 3.1453 | 2.3148 | 0.74 🟡 |
| f64\|bcast\|f64 | 2.4392 | 1.7958 | 0.74 🟡 |
| c128\|negcol\|u32 | 2.4242 | 1.7875 | 0.74 🟡 |
| i32\|T\|u8 | 0.5478 | 0.4048 | 0.74 🟡 |
| f64\|strided\|i64 | 1.3003 | 0.9634 | 0.74 🟡 |
| u64\|T\|c128 | 3.6771 | 2.7352 | 0.74 🟡 |
| i32\|F\|u8 | 0.5357 | 0.3987 | 0.74 🟡 |
| bool\|negrow\|u8 | 0.5900 | 0.4392 | 0.74 🟡 |
| c128\|F\|bool | 1.1235 | 0.8381 | 0.75 🟡 |
| c128\|bcast\|f64 | 2.3316 | 1.7423 | 0.75 🟡 |
| bool\|strided\|i8 | 0.2766 | 0.2068 | 0.75 🟡 |
| bool\|negcol\|bool | 0.5870 | 0.4391 | 0.75 🟡 |
| i64\|negcol\|u64 | 2.5236 | 1.8877 | 0.75 🟡 |
| f32\|strided\|i32 | 0.7051 | 0.5276 | 0.75 🟡 |
| i64\|bcast\|c128 | 4.3297 | 3.2404 | 0.75 🟡 |
| u8\|bcast\|i64 | 2.1958 | 1.6471 | 0.75 🟡 |
| f64\|negrow\|u32 | 1.4857 | 1.1161 | 0.75 🟡 |
| i16\|bcast\|i64 | 2.2019 | 1.6549 | 0.75 🟡 |
| i32\|C\|u8 | 0.5413 | 0.4074 | 0.75 🟡 |
| c128\|bcast\|f16 | 5.2744 | 3.9703 | 0.75 🟡 |
| c128\|negcol\|bool | 1.4688 | 1.1074 | 0.75 🟡 |
| i32\|C\|i8 | 0.5658 | 0.4278 | 0.76 🟡 |
| u16\|bcast\|f64 | 2.2939 | 1.7357 | 0.76 🟡 |
| u64\|sliced\|bool | 0.5856 | 0.4436 | 0.76 🟡 |
| f32\|F\|i64 | 2.4693 | 1.8727 | 0.76 🟡 |
| u16\|bcast\|c128 | 4.3842 | 3.3408 | 0.76 🟡 |
| f64\|T\|u32 | 1.4715 | 1.1256 | 0.76 🟡 |
| i32\|F\|i8 | 0.5569 | 0.4274 | 0.77 🟡 |
| f16\|negrow\|f64 | 3.2501 | 2.4965 | 0.77 🟡 |
| u32\|negrow\|u8 | 0.4284 | 0.3322 | 0.78 🟡 |
| u32\|bcast\|c128 | 4.4322 | 3.4408 | 0.78 🟡 |
| i16\|bcast\|c128 | 4.1175 | 3.1966 | 0.78 🟡 |
| f64\|negrow\|i64 | 2.5587 | 1.9901 | 0.78 🟡 |
| c128\|negcol\|u16 | 1.9920 | 1.5527 | 0.78 🟡 |
| bool\|C\|char | 0.5402 | 0.4211 | 0.78 🟡 |
| bool\|negcol\|u8 | 0.5511 | 0.4306 | 0.78 🟡 |
| i16\|bcast\|char | 0.9038 | 0.7069 | 0.78 🟡 |
| f64\|sliced\|u32 | 1.3915 | 1.0902 | 0.78 🟡 |
| i32\|bcast\|c128 | 4.0782 | 3.2067 | 0.79 🟡 |
| f32\|strided\|i64 | 1.2444 | 0.9815 | 0.79 🟡 |
| f32\|negcol\|bool | 1.1332 | 0.8960 | 0.79 🟡 |
| i64\|negcol\|i64 | 1.8238 | 1.4422 | 0.79 🟡 |
| f64\|C\|u32 | 1.3480 | 1.0686 | 0.79 🟡 |
| c128\|C\|bool | 1.1371 | 0.9039 | 0.79 🟡 |
| c128\|T\|u16 | 1.5004 | 1.1953 | 0.80 🟡 |
| bool\|negcol\|i8 | 0.5341 | 0.4283 | 0.80 🟡 |
| u32\|strided\|f32 | 0.5955 | 0.4778 | 0.80 🟡 |
| f32\|negrow\|i64 | 2.3350 | 1.8847 | 0.81 🟡 |
| bool\|C\|u16 | 0.5255 | 0.4247 | 0.81 🟡 |
| u32\|sliced\|bool | 0.2591 | 0.2096 | 0.81 🟡 |
| bool\|bcast\|f64 | 2.0576 | 1.6716 | 0.81 🟡 |
| f32\|strided\|bool | 0.5502 | 0.4471 | 0.81 🟡 |
| c128\|negcol\|u64 | 3.8777 | 3.1623 | 0.82 🟡 |
| u32\|strided\|i32 | 0.6642 | 0.5416 | 0.82 🟡 |
| i32\|C\|bool | 0.2607 | 0.2128 | 0.82 🟡 |
| i8\|strided\|bool | 0.3201 | 0.2620 | 0.82 🟡 |
| c128\|T\|u32 | 2.1248 | 1.7424 | 0.82 🟡 |
| u64\|C\|i8 | 0.6848 | 0.5637 | 0.82 🟡 |
| f64\|negcol\|i64 | 2.5392 | 2.0917 | 0.82 🟡 |
| c128\|strided\|i64 | 1.5979 | 1.3197 | 0.83 🟡 |
| i64\|T\|u8 | 0.6912 | 0.5718 | 0.83 🟡 |
| f32\|negcol\|i32 | 1.2429 | 1.0312 | 0.83 🟡 |
| c128\|C\|u32 | 1.7805 | 1.4781 | 0.83 🟡 |
| u64\|sliced\|u8 | 0.4741 | 0.3937 | 0.83 🟡 |
| u64\|T\|i16 | 0.7920 | 0.6584 | 0.83 🟡 |
| u64\|F\|bool | 0.5302 | 0.4412 | 0.83 🟡 |
| u8\|bcast\|u64 | 2.1717 | 1.8087 | 0.83 🟡 |
| f64\|F\|u32 | 1.4037 | 1.1738 | 0.84 🟡 |
| i64\|T\|bool | 0.5566 | 0.4677 | 0.84 🟡 |
| f64\|bcast\|c128 | 3.9167 | 3.2976 | 0.84 🟡 |
| bool\|C\|i16 | 0.5106 | 0.4301 | 0.84 🟡 |
| u64\|sliced\|f32 | 1.7952 | 1.5147 | 0.84 🟡 |
| bool\|F\|i16 | 0.5008 | 0.4225 | 0.84 🟡 |
| char\|bcast\|c128 | 4.0624 | 3.4320 | 0.84 🟡 |
| bool\|F\|u8 | 0.3627 | 0.3074 | 0.85 🟡 |
| u64\|T\|i8 | 0.6318 | 0.5379 | 0.85 🟡 |
| bool\|T\|char | 0.5136 | 0.4378 | 0.85 🟡 |
| i64\|negrow\|bool | 0.5454 | 0.4654 | 0.85 🟡 |
| bool\|F\|char | 0.4925 | 0.4207 | 0.85 🟡 |
| f64\|T\|i64 | 2.2624 | 1.9388 | 0.86 🟡 |
| u64\|bcast\|c128 | 4.0373 | 3.4675 | 0.86 🟡 |
| u32\|strided\|u32 | 0.5895 | 0.5073 | 0.86 🟡 |
| i8\|negcol\|bool | 0.5749 | 0.4951 | 0.86 🟡 |
| u64\|negcol\|f32 | 1.3853 | 1.1941 | 0.86 🟡 |
| c128\|negrow\|u64 | 3.3655 | 2.9013 | 0.86 🟡 |
| i64\|T\|f64 | 2.0951 | 1.8089 | 0.86 🟡 |
| u32\|negrow\|f64 | 2.0436 | 1.7662 | 0.86 🟡 |
| bool\|bcast\|c128 | 3.8001 | 3.2847 | 0.86 🟡 |
| u64\|negrow\|bool | 0.5380 | 0.4653 | 0.86 🟡 |
| bool\|T\|u16 | 0.4855 | 0.4224 | 0.87 🟡 |
| bool\|F\|u16 | 0.5002 | 0.4361 | 0.87 🟡 |
| f64\|sliced\|i64 | 2.2737 | 1.9971 | 0.88 🟡 |
| bool\|T\|i16 | 0.4802 | 0.4234 | 0.88 🟡 |
| u8\|strided\|bool | 0.2861 | 0.2523 | 0.88 🟡 |
| i8\|bcast\|c128 | 3.7171 | 3.2794 | 0.88 🟡 |
| i64\|negcol\|f16 | 3.6136 | 3.1907 | 0.88 🟡 |
| u64\|C\|bool | 0.5253 | 0.4647 | 0.88 🟡 |
| f16\|strided\|c128 | 1.9790 | 1.7545 | 0.89 🟡 |
| u64\|F\|u16 | 0.7345 | 0.6561 | 0.89 🟡 |
| c128\|F\|i64 | 2.8951 | 2.5917 | 0.90 🟡 |
| i64\|T\|u16 | 0.7881 | 0.7068 | 0.90 🟡 |
| u8\|bcast\|c128 | 3.5666 | 3.1991 | 0.90 🟡 |
| c128\|sliced\|u32 | 1.6378 | 1.4805 | 0.90 🟡 |
| bool\|sliced\|i32 | 0.7739 | 0.7012 | 0.91 🟡 |
| i64\|sliced\|i8 | 0.4276 | 0.3888 | 0.91 🟡 |
| u64\|strided\|f32 | 0.7505 | 0.6838 | 0.91 🟡 |
| c128\|negrow\|u32 | 2.0685 | 1.8856 | 0.91 🟡 |
| c128\|negcol\|i64 | 3.2441 | 2.9593 | 0.91 🟡 |
| char\|T\|u16 | 0.3684 | 0.3365 | 0.91 🟡 |
| u64\|bcast\|f64 | 2.2297 | 2.0406 | 0.92 🟡 |
| c128\|T\|i8 | 0.8224 | 0.7552 | 0.92 🟡 |
| i64\|negrow\|char | 0.7221 | 0.6632 | 0.92 🟡 |
| f64\|F\|i64 | 2.3153 | 2.1273 | 0.92 🟡 |
| i64\|F\|u8 | 0.6526 | 0.6006 | 0.92 🟡 |
| char\|C\|char | 0.3452 | 0.3182 | 0.92 🟡 |
| f64\|negcol\|f32 | 1.1443 | 1.0572 | 0.92 🟡 |
| i64\|sliced\|bool | 0.5021 | 0.4641 | 0.92 🟡 |
| f64\|C\|i64 | 2.1799 | 2.0154 | 0.92 🟡 |
| u64\|sliced\|i16 | 0.6912 | 0.6392 | 0.92 🟡 |
| u16\|C\|char | 0.3512 | 0.3257 | 0.93 🟡 |
| c128\|T\|f32 | 1.7054 | 1.5835 | 0.93 🟡 |
| c128\|negrow\|i64 | 2.9753 | 2.7697 | 0.93 🟡 |
| c128\|strided\|f16 | 2.2252 | 2.0726 | 0.93 🟡 |
| bool\|negrow\|i8 | 0.5493 | 0.5145 | 0.94 🟡 |
| u64\|negrow\|u8 | 0.4365 | 0.4091 | 0.94 🟡 |
| c128\|negrow\|u8 | 0.7838 | 0.7354 | 0.94 🟡 |
| i64\|C\|i16 | 0.8089 | 0.7595 | 0.94 🟡 |
| c128\|bcast\|c128 | 4.1583 | 3.9170 | 0.94 🟡 |
| f32\|strided\|f32 | 0.5630 | 0.5308 | 0.94 🟡 |
| c128\|F\|u32 | 1.6975 | 1.6033 | 0.94 🟡 |
| i64\|F\|bool | 0.5487 | 0.5184 | 0.94 🟡 |
| u64\|C\|u16 | 0.7461 | 0.7067 | 0.95 🟡 |
| u32\|sliced\|u8 | 0.2199 | 0.2083 | 0.95 🟡 |
| i64\|sliced\|char | 0.5561 | 0.5284 | 0.95 🟡 |
| i64\|negrow\|f32 | 1.0800 | 1.0264 | 0.95 🟡 |
| i64\|strided\|f32 | 0.6332 | 0.6019 | 0.95 🟡 |
| u64\|sliced\|f64 | 1.9956 | 1.9062 | 0.96 🟡 |
| u64\|T\|u16 | 0.6694 | 0.6412 | 0.96 🟡 |
| bool\|T\|f32 | 0.7489 | 0.7186 | 0.96 🟡 |
| f64\|strided\|f32 | 0.6821 | 0.6545 | 0.96 🟡 |
| i64\|negrow\|i8 | 0.4061 | 0.3898 | 0.96 🟡 |
| u64\|sliced\|u16 | 0.6650 | 0.6412 | 0.96 🟡 |
| i64\|T\|i16 | 0.8115 | 0.7849 | 0.97 🟡 |
| c128\|T\|i64 | 3.0399 | 2.9473 | 0.97 🟡 |
| bool\|F\|i8 | 0.3466 | 0.3365 | 0.97 🟡 |
| u8\|negcol\|bool | 0.5171 | 0.5038 | 0.97 🟡 |
| i64\|C\|bool | 0.4891 | 0.4769 | 0.98 🟡 |
| f16\|F\|f16 | 0.3347 | 0.3265 | 0.98 🟡 |
| i64\|strided\|f16 | 1.5713 | 1.5331 | 0.98 🟡 |
| c128\|strided\|i32 | 0.9023 | 0.8816 | 0.98 🟡 |
| i64\|C\|f64 | 1.8939 | 1.8586 | 0.98 🟡 |
| f16\|T\|f16 | 0.3400 | 0.3354 | 0.99 🟡 |
| u64\|negrow\|f32 | 1.5392 | 1.5196 | 0.99 🟡 |
| char\|C\|u16 | 0.3201 | 0.3161 | 0.99 🟡 |
| i64\|T\|f32 | 1.1148 | 1.1010 | 0.99 🟡 |
| c128\|strided\|f32 | 0.9237 | 0.9158 | 0.99 🟡 |
| u64\|T\|bool | 0.4655 | 0.4615 | 0.99 🟡 |
| u64\|negrow\|f64 | 1.9676 | 1.9512 | 0.99 🟡 |
| u32\|F\|u8 | 0.5335 | 0.5297 | 0.99 🟡 |
| c128\|T\|i16 | 1.1951 | 1.1875 | 0.99 🟡 |
| u64\|F\|i8 | 0.6177 | 0.6150 | 1.00 🟡 |
| bool\|negcol\|u16 | 0.7809 | 0.7782 | 1.00 🟡 |
| f16\|F\|c128 | 3.3467 | 3.3361 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 514 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.42 ms   unfused    7.42 ms   (1.68x)
  (a-b)/(a+b) fused    3.67 ms   unfused   14.74 ms   (4.02x)
  sum(a*b)    fused    2.63 ms   unfused    4.69 ms   (1.78x)
  sum(af*bf)  fused    1.63 ms   unfused    2.19 ms   (1.34x)  [f32]
  a*b+c out=  fused    4.15 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.31 ms   unfused    4.89 ms   (1.48x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.35 ms   unfused    7.44 ms   (1.71x)
    [F      ] fused    4.34 ms   unfused    7.39 ms   (1.70x)
    [T      ] fused    4.16 ms   unfused    7.40 ms   (1.78x)
    [strided] fused    3.89 ms   unfused    5.56 ms   (1.43x)
    [bcast  ] fused    1.56 ms   unfused    4.35 ms   (2.80x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         13.48 ms
  (a-b)/(a+b)   19.61 ms
  sum(a*b)       8.90 ms
  sum(af*bf)     4.42 ms  [f32]
  a*b+c out=     5.13 ms  [two-pass with out=]
  i4*2+f8       10.10 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.93 ms
    [F      ]   12.90 ms
    [T      ]   13.02 ms
    [strided]    7.94 ms
    [bcast  ]   12.48 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-7ec7c6ad0c58f826c3d04fc8c1f325edd5ddd512868ff2c0d77027c2d96ba979\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0071 ms | 0.0093 ms | 0.0079 | 1.113x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1170 ms | 0.9291 ms | 1.1340 | 9.692x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1155 ms | 0.9218 ms | 1.2404 | 10.739x |
| `np.dot(a, b)` | 100,000 | 0.0072 ms | 0.0074 ms | 0.0076 | 1.056x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2096 ms | 0.0619 ms | 0.2650 | 4.281x |
| `np.inner(a, b)` | 100,000 | 0.0070 ms | 0.0075 ms | 0.0081 | 1.157x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0026 ms | 0.0054 | 2.077x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0356 ms | 0.0297 | 0.834x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0378 ms | 0.0401 | 1.061x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.5259 ms | 0.3198 | 0.608x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0052 ms | 0.0066 | 1.269x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0600 ms | 0.0300 | 0.500x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1160 ms | 0.1194 | 1.029x |
| `np.linalg.eig(a)` | 96 | missing_backend | 2.7234 ms | 1.5348 | 0.564x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0468 ms | 0.0641 | 1.370x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.8513 ms | 0.4429 | 0.520x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0675 ms | 0.0746 | 1.105x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.5264 ms | 0.8668 | 0.568x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0209 ms | 0.0252 | 1.206x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.2885 ms | 0.2323 | 0.805x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0094 ms | 0.0126 | 1.340x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.1642 ms | 0.1231 | 0.750x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0834 ms | 0.1046 | 1.254x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 1.6352 ms | 1.1186 | 0.684x |
| `np.linalg.matmul(a, b)` | 128 | 0.2078 ms | 0.0635 ms | 0.0675 | 1.063x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0581 ms | 0.0442 | 0.761x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.4981 ms | 0.3155 | 0.633x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0214 ms | 0.0132 | 0.617x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1657 ms | 0.0885 | 0.534x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0372 ms | 0.0417 | 1.121x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.5264 ms | 0.4112 | 0.781x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.4100 ms | 0.1199 ms | 0.1342 | 1.119x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0645 ms | 0.0471 | 0.730x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.4958 ms | 0.3085 | 0.622x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.0967 ms | 0.1006 | 1.040x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 1.6728 ms | 1.0330 | 0.618x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0212 ms | 0.0297 | 1.401x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.3196 ms | 0.1882 | 0.589x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0058 ms | 0.0077 | 1.328x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0585 ms | 0.0315 | 0.538x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0053 ms | 0.0087 | 1.641x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0609 ms | 0.0331 | 0.544x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0747 ms | 0.0872 | 1.167x |
| `np.linalg.svd(a)` | 96 | missing_backend | 1.4758 ms | 0.9363 | 0.634x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0307 ms | 0.0350 | 1.140x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.4991 ms | 0.3091 | 0.619x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2149 ms | 0.0614 ms | 0.0695 | 1.132x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0100 ms | 0.0127 | 1.270x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.1621 ms | 0.0919 | 0.567x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0065 ms | 0.0099 | 1.523x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0593 ms | 0.0344 | 0.580x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0339 ms | 0.0075 ms | 0.0108 | 1.440x |
| `np.matmul(a, b)` | 128 | 0.2122 ms | 0.0638 ms | 0.0644 | 1.009x |
| `np.matvec(a, b)` | 128 | 0.0232 ms | 0.0021 ms | 0.0025 | 1.191x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.1007 ms | 0.0561 | 0.557x |
| `np.roots(p)` | 16 | missing_backend | 0.0570 ms | 0.0416 | 0.730x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2118 ms | 0.0614 ms | 0.0713 | 1.161x |
| `np.vdot(a, b)` | 100,000 | 0.0077 ms | 0.0081 ms | 0.0096 | 1.247x |
| `np.vecdot(a, b)` | 100,000 | 0.0380 ms | 0.0073 ms | 0.0140 | 1.918x |
| `np.vecmat(a, b)` | 128 | 0.0031 ms | 0.0021 ms | 0.0020 | 0.952x |
