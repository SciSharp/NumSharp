# NumSharp vs NumPy Performance — backend profiles

One canonical cell model with separate Managed C# and OpenBLAS measurements.
**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell.

- Managed effective cells: **3,335**
- OpenBLAS effective cells: **57**
- Missing backend records: **44**
- Not supported records: **0**

| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |
|---|---|---:|---:|---:|---|---:|
| `np.polyfit(x, y, 3)` | float64 | 1,000 | missing_backend | 0.2248 ms | openblas | 0.231x |
| `np.roots(p)` | float64 | 16 | missing_backend | 0.0606 ms | openblas | 0.729x |
| `np.array2string(a) (float64)` | float64 | 1,000 | 1.1123 ms | not_measured | managed | 2.450x |
| `np.array_repr(a) (float64)` | float64 | 1,000 | 1.0340 ms | not_measured | managed | 2.512x |
| `np.array_str(a) (float64)` | float64 | 1,000 | 1.0502 ms | not_measured | managed | 2.410x |
| `np.can_cast(from, to) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.common_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.copyto(dst, src) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.format_float_positional(x) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.format_float_scientific(x) (float64)` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 1.200x |
| `np.fromfile(path) (float64)` | float64 | 1,000 | 0.0500 ms | not_measured | managed | 1.192x |
| `np.get_printoptions() (float64)` | float64 | 1,000 | 0.0003 ms | not_measured | managed | 1.333x |
| `np.isdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.issubdtype(dtype, kind) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.iterable(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.load(path) (float64)` | float64 | 1,000 | 0.0659 ms | not_measured | managed | 1.530x |
| `np.loadtxt(path) (float64)` | float64 | 1,000 | 0.8369 ms | not_measured | managed | 0.397x |
| `np.min_scalar_type(value) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.mintypecode(typechars) (float64)` | float64 | 1,000 | 0.0002 ms | not_measured | managed | 4.000x |
| `np.nested_iters(a, axes) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.poly(roots) (float64)` | float64 | 1,000 | 0.1495 ms | not_measured | managed | 0.168x |
| `np.polyadd(p, q) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.463x |
| `np.polyder(p) (float64)` | float64 | 1,000 | 0.0096 ms | not_measured | managed | 0.302x |
| `np.polydiv(p, q) (float64)` | float64 | 1,000 | 0.3921 ms | not_measured | managed | 0.316x |
| `np.polyint(p) (float64)` | float64 | 1,000 | 0.0133 ms | not_measured | managed | 0.233x |
| `np.polymul(p, q) (float64)` | float64 | 1,000 | 0.0165 ms | not_measured | managed | 1.000x |
| `np.polysub(p, q) (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.600x |
| `np.polyval(p, x) (float64)` | float64 | 1,000 | 0.1137 ms | not_measured | managed | 0.357x |
| `np.printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 41.000x |
| `np.promote_types(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.result_type(a, b) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.save(path, a) (float64)` | float64 | 1,000 | 0.4388 ms | not_measured | managed | 0.601x |
| `np.savetxt(path, a) (float64)` | float64 | 1,000 | 1.3931 ms | not_measured | managed | 1.470x |
| `np.savez(path, a) (float64)` | float64 | 1,000 | 6.9250 ms | not_measured | managed | 0.049x |
| `np.savez_compressed(path, a) (float64)` | float64 | 1,000 | 0.5783 ms | not_measured | managed | 0.763x |
| `np.set_printoptions() (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 22.000x |
| `a % 7 (literal) (float32)` | float32 | 1,000 | 0.0183 ms | not_measured | managed | 0.792x |
| `a % 7 (literal) (float32)` | float32 | 100,000 | 1.9294 ms | not_measured | managed | 0.898x |
| `a % 7 (literal) (float32)` | float32 | 10,000,000 | 186.8877 ms | not_measured | managed | 1.355x |
| `a % 7 (literal) (float64)` | float64 | 1,000 | 0.0148 ms | not_measured | managed | 0.851x |
| `a % 7 (literal) (float64)` | float64 | 100,000 | 1.7390 ms | not_measured | managed | 0.835x |
| `a % 7 (literal) (float64)` | float64 | 10,000,000 | 186.1513 ms | not_measured | managed | 1.333x |
| `a % 7 (literal) (int32)` | int32 | 1,000 | 0.0058 ms | not_measured | managed | 0.466x |
| `a % 7 (literal) (int32)` | int32 | 100,000 | 0.6793 ms | not_measured | managed | 0.615x |
| `a % 7 (literal) (int32)` | int32 | 10,000,000 | 71.3434 ms | not_measured | managed | 0.635x |
| `a % 7 (literal) (int64)` | int64 | 1,000 | 0.0075 ms | not_measured | managed | 0.560x |
| `a % 7 (literal) (int64)` | int64 | 100,000 | 0.6832 ms | not_measured | managed | 0.606x |
| `a % 7 (literal) (int64)` | int64 | 10,000,000 | 77.4725 ms | not_measured | managed | 0.641x |
| `a % b (element-wise) (float32)` | float32 | 1,000 | 0.0129 ms | not_measured | managed | 0.946x |
| `a % b (element-wise) (float32)` | float32 | 100,000 | 1.6594 ms | not_measured | managed | 0.945x |
| `a % b (element-wise) (float32)` | float32 | 10,000,000 | 166.1644 ms | not_measured | managed | 1.427x |
| `a % b (element-wise) (float64)` | float64 | 1,000 | 0.0109 ms | not_measured | managed | 1.046x |
| `a % b (element-wise) (float64)` | float64 | 100,000 | 1.4692 ms | not_measured | managed | 0.889x |
| `a % b (element-wise) (float64)` | float64 | 10,000,000 | 155.6740 ms | not_measured | managed | 1.523x |
| `a % b (element-wise) (int32)` | int32 | 1,000 | 0.0025 ms | not_measured | managed | 0.800x |
| `a % b (element-wise) (int32)` | int32 | 100,000 | 0.6232 ms | not_measured | managed | 0.608x |
| `a % b (element-wise) (int32)` | int32 | 10,000,000 | 62.2534 ms | not_measured | managed | 0.676x |
| `a % b (element-wise) (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 1.250x |
| `a % b (element-wise) (int64)` | int64 | 100,000 | 0.6186 ms | not_measured | managed | 0.625x |
| `a % b (element-wise) (int64)` | int64 | 10,000,000 | 69.5313 ms | not_measured | managed | 0.693x |
| `a * 2 (literal) (complex128)` | complex128 | 1,000 | 0.0078 ms | not_measured | managed | 0.141x |
| `a * 2 (literal) (complex128)` | complex128 | 100,000 | 0.3512 ms | not_measured | managed | 1.066x |
| `a * 2 (literal) (complex128)` | complex128 | 10,000,000 | 66.6010 ms | not_measured | managed | 0.839x |
| `a * 2 (literal) (float16)` | float16 | 1,000 | 0.0122 ms | not_measured | managed | 0.311x |
| `a * 2 (literal) (float16)` | float16 | 100,000 | 0.9425 ms | not_measured | managed | 0.319x |
| `a * 2 (literal) (float16)` | float16 | 10,000,000 | 94.1869 ms | not_measured | managed | 0.781x |
| `a * 2 (literal) (float32)` | float32 | 1,000 | 0.0067 ms | not_measured | managed | 0.149x |
| `a * 2 (literal) (float32)` | float32 | 100,000 | 0.0654 ms | not_measured | managed | 0.131x |
| `a * 2 (literal) (float32)` | float32 | 10,000,000 | 10.9649 ms | not_measured | managed | 1.147x |
| `a * 2 (literal) (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.145x |
| `a * 2 (literal) (float64)` | float64 | 100,000 | 0.1189 ms | not_measured | managed | 0.110x |
| `a * 2 (literal) (float64)` | float64 | 10,000,000 | 22.3078 ms | not_measured | managed | 1.175x |
| `a * 2 (literal) (int16)` | int16 | 1,000 | 0.0033 ms | not_measured | managed | 0.303x |
| `a * 2 (literal) (int16)` | int16 | 100,000 | 0.0292 ms | not_measured | managed | 0.818x |
| `a * 2 (literal) (int16)` | int16 | 10,000,000 | 4.6802 ms | not_measured | managed | 0.958x |
| `a * 2 (literal) (int32)` | int32 | 1,000 | 0.0041 ms | not_measured | managed | 0.244x |
| `a * 2 (literal) (int32)` | int32 | 100,000 | 0.0520 ms | not_measured | managed | 0.515x |
| `a * 2 (literal) (int32)` | int32 | 10,000,000 | 9.2244 ms | not_measured | managed | 0.864x |
| `a * 2 (literal) (int64)` | int64 | 1,000 | 0.0044 ms | not_measured | managed | 0.250x |
| `a * 2 (literal) (int64)` | int64 | 100,000 | 0.1165 ms | not_measured | managed | 0.205x |
| `a * 2 (literal) (int64)` | int64 | 10,000,000 | 26.0221 ms | not_measured | managed | 0.596x |
| `a * 2 (literal) (int8)` | int8 | 1,000 | 0.0029 ms | not_measured | managed | 0.310x |
| `a * 2 (literal) (int8)` | int8 | 100,000 | 0.0182 ms | not_measured | managed | 1.253x |
| `a * 2 (literal) (int8)` | int8 | 10,000,000 | 2.3654 ms | not_measured | managed | 1.302x |
| `a * 2 (literal) (uint16)` | uint16 | 1,000 | 0.0066 ms | not_measured | managed | 0.167x |
| `a * 2 (literal) (uint16)` | uint16 | 100,000 | 0.0279 ms | not_measured | managed | 0.849x |
| `a * 2 (literal) (uint16)` | uint16 | 10,000,000 | 5.7877 ms | not_measured | managed | 0.911x |
| `a * 2 (literal) (uint32)` | uint32 | 1,000 | 0.0075 ms | not_measured | managed | 0.147x |
| `a * 2 (literal) (uint32)` | uint32 | 100,000 | 0.0547 ms | not_measured | managed | 0.430x |
| `a * 2 (literal) (uint32)` | uint32 | 10,000,000 | 9.4926 ms | not_measured | managed | 0.823x |
| `a * 2 (literal) (uint64)` | uint64 | 1,000 | 0.0052 ms | not_measured | managed | 0.192x |
| `a * 2 (literal) (uint64)` | uint64 | 100,000 | 0.1132 ms | not_measured | managed | 0.209x |
| `a * 2 (literal) (uint64)` | uint64 | 10,000,000 | 28.4173 ms | not_measured | managed | 0.955x |
| `a * 2 (literal) (uint8)` | uint8 | 1,000 | 0.0034 ms | not_measured | managed | 0.265x |
| `a * 2 (literal) (uint8)` | uint8 | 100,000 | 0.0185 ms | not_measured | managed | 1.254x |
| `a * 2 (literal) (uint8)` | uint8 | 10,000,000 | 2.3750 ms | not_measured | managed | 1.488x |
| `a * a (square) (complex128)` | complex128 | 1,000 | 0.0035 ms | not_measured | managed | 0.257x |
| `a * a (square) (complex128)` | complex128 | 100,000 | 0.3341 ms | not_measured | managed | 1.091x |
| `a * a (square) (complex128)` | complex128 | 10,000,000 | 49.9347 ms | not_measured | managed | 1.169x |
| `a * a (square) (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.327x |
| `a * a (square) (float16)` | float16 | 100,000 | 0.9233 ms | not_measured | managed | 0.325x |
| `a * a (square) (float16)` | float16 | 10,000,000 | 93.7215 ms | not_measured | managed | 0.852x |
| `a * a (square) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `a * a (square) (float32)` | float32 | 100,000 | 0.0508 ms | not_measured | managed | 0.185x |
| `a * a (square) (float32)` | float32 | 10,000,000 | 10.3990 ms | not_measured | managed | 1.652x |
| `a * a (square) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.185x |
| `a * a (square) (float64)` | float64 | 100,000 | 0.1098 ms | not_measured | managed | 0.133x |
| `a * a (square) (float64)` | float64 | 10,000,000 | 20.7307 ms | not_measured | managed | 1.275x |
| `a * a (square) (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a * a (square) (int16)` | int16 | 100,000 | 0.0276 ms | not_measured | managed | 1.098x |
| `a * a (square) (int16)` | int16 | 10,000,000 | 4.5731 ms | not_measured | managed | 1.048x |
| `a * a (square) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a * a (square) (int32)` | int32 | 100,000 | 0.0544 ms | not_measured | managed | 0.564x |
| `a * a (square) (int32)` | int32 | 10,000,000 | 9.6691 ms | not_measured | managed | 0.887x |
| `a * a (square) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a * a (square) (int64)` | int64 | 100,000 | 0.1208 ms | not_measured | managed | 0.252x |
| `a * a (square) (int64)` | int64 | 10,000,000 | 20.8623 ms | not_measured | managed | 0.774x |
| `a * a (square) (int8)` | int8 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `a * a (square) (int8)` | int8 | 100,000 | 0.0163 ms | not_measured | managed | 1.822x |
| `a * a (square) (int8)` | int8 | 10,000,000 | 2.4319 ms | not_measured | managed | 1.566x |
| `a * a (square) (uint16)` | uint16 | 1,000 | 0.0007 ms | not_measured | managed | 1.143x |
| `a * a (square) (uint16)` | uint16 | 100,000 | 0.0273 ms | not_measured | managed | 1.070x |
| `a * a (square) (uint16)` | uint16 | 10,000,000 | 4.5880 ms | not_measured | managed | 1.076x |
| `a * a (square) (uint32)` | uint32 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `a * a (square) (uint32)` | uint32 | 100,000 | 0.0511 ms | not_measured | managed | 0.599x |
| `a * a (square) (uint32)` | uint32 | 10,000,000 | 10.0014 ms | not_measured | managed | 0.822x |
| `a * a (square) (uint64)` | uint64 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `a * a (square) (uint64)` | uint64 | 100,000 | 0.1169 ms | not_measured | managed | 0.252x |
| `a * a (square) (uint64)` | uint64 | 10,000,000 | 21.4738 ms | not_measured | managed | 1.401x |
| `a * a (square) (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `a * a (square) (uint8)` | uint8 | 100,000 | 0.0162 ms | not_measured | managed | 1.765x |
| `a * a (square) (uint8)` | uint8 | 10,000,000 | 2.3575 ms | not_measured | managed | 2.276x |
| `a * b (element-wise) (complex128)` | complex128 | 1,000 | 0.0032 ms | not_measured | managed | 0.281x |
| `a * b (element-wise) (complex128)` | complex128 | 100,000 | 0.3674 ms | not_measured | managed | 1.023x |
| `a * b (element-wise) (complex128)` | complex128 | 10,000,000 | 67.3492 ms | not_measured | managed | 1.045x |
| `a * b (element-wise) (float16)` | float16 | 1,000 | 0.0100 ms | not_measured | managed | 0.320x |
| `a * b (element-wise) (float16)` | float16 | 100,000 | 0.9435 ms | not_measured | managed | 0.311x |
| `a * b (element-wise) (float16)` | float16 | 10,000,000 | 93.8892 ms | not_measured | managed | 0.840x |
| `a * b (element-wise) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.467x |
| `a * b (element-wise) (float32)` | float32 | 100,000 | 0.0515 ms | not_measured | managed | 0.173x |
| `a * b (element-wise) (float32)` | float32 | 10,000,000 | 11.6341 ms | not_measured | managed | 1.455x |
| `a * b (element-wise) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `a * b (element-wise) (float64)` | float64 | 100,000 | 0.1141 ms | not_measured | managed | 0.273x |
| `a * b (element-wise) (float64)` | float64 | 10,000,000 | 27.6905 ms | not_measured | managed | 1.230x |
| `a * b (element-wise) (int16)` | int16 | 1,000 | 0.0024 ms | not_measured | managed | 0.417x |
| `a * b (element-wise) (int16)` | int16 | 100,000 | 0.0264 ms | not_measured | managed | 1.163x |
| `a * b (element-wise) (int16)` | int16 | 10,000,000 | 6.1329 ms | not_measured | managed | 0.822x |
| `a * b (element-wise) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.733x |
| `a * b (element-wise) (int32)` | int32 | 100,000 | 0.0524 ms | not_measured | managed | 0.594x |
| `a * b (element-wise) (int32)` | int32 | 10,000,000 | 11.3914 ms | not_measured | managed | 1.196x |
| `a * b (element-wise) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `a * b (element-wise) (int64)` | int64 | 100,000 | 0.1112 ms | not_measured | managed | 0.328x |
| `a * b (element-wise) (int64)` | int64 | 10,000,000 | 32.2430 ms | not_measured | managed | 0.527x |
| `a * b (element-wise) (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `a * b (element-wise) (int8)` | int8 | 100,000 | 0.0163 ms | not_measured | managed | 1.748x |
| `a * b (element-wise) (int8)` | int8 | 10,000,000 | 4.6123 ms | not_measured | managed | 0.914x |
| `a * b (element-wise) (uint16)` | uint16 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `a * b (element-wise) (uint16)` | uint16 | 100,000 | 0.0275 ms | not_measured | managed | 1.069x |
| `a * b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.1128 ms | not_measured | managed | 0.852x |
| `a * b (element-wise) (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `a * b (element-wise) (uint32)` | uint32 | 100,000 | 0.0525 ms | not_measured | managed | 0.562x |
| `a * b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.2064 ms | not_measured | managed | 0.799x |
| `a * b (element-wise) (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a * b (element-wise) (uint64)` | uint64 | 100,000 | 0.1164 ms | not_measured | managed | 0.311x |
| `a * b (element-wise) (uint64)` | uint64 | 10,000,000 | 28.1256 ms | not_measured | managed | 1.461x |
| `a * b (element-wise) (uint8)` | uint8 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `a * b (element-wise) (uint8)` | uint8 | 100,000 | 0.0160 ms | not_measured | managed | 1.775x |
| `a * b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.8695 ms | not_measured | managed | 1.208x |
| `a * scalar (complex128)` | complex128 | 1,000 | 0.0035 ms | not_measured | managed | 0.257x |
| `a * scalar (complex128)` | complex128 | 100,000 | 0.3499 ms | not_measured | managed | 1.048x |
| `a * scalar (complex128)` | complex128 | 10,000,000 | 43.5793 ms | not_measured | managed | 1.202x |
| `a * scalar (float16)` | float16 | 1,000 | 0.0107 ms | not_measured | managed | 0.327x |
| `a * scalar (float16)` | float16 | 100,000 | 0.9274 ms | not_measured | managed | 0.317x |
| `a * scalar (float16)` | float16 | 10,000,000 | 97.5616 ms | not_measured | managed | 0.773x |
| `a * scalar (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `a * scalar (float32)` | float32 | 100,000 | 0.0510 ms | not_measured | managed | 0.131x |
| `a * scalar (float32)` | float32 | 10,000,000 | 9.9382 ms | not_measured | managed | 1.362x |
| `a * scalar (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `a * scalar (float64)` | float64 | 100,000 | 0.1005 ms | not_measured | managed | 0.131x |
| `a * scalar (float64)` | float64 | 10,000,000 | 28.5102 ms | not_measured | managed | 0.974x |
| `a * scalar (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `a * scalar (int16)` | int16 | 100,000 | 0.0284 ms | not_measured | managed | 0.835x |
| `a * scalar (int16)` | int16 | 10,000,000 | 5.3182 ms | not_measured | managed | 0.854x |
| `a * scalar (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.529x |
| `a * scalar (int32)` | int32 | 100,000 | 0.0543 ms | not_measured | managed | 0.444x |
| `a * scalar (int32)` | int32 | 10,000,000 | 9.1394 ms | not_measured | managed | 0.910x |
| `a * scalar (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `a * scalar (int64)` | int64 | 100,000 | 0.1109 ms | not_measured | managed | 0.210x |
| `a * scalar (int64)` | int64 | 10,000,000 | 32.8280 ms | not_measured | managed | 0.624x |
| `a * scalar (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `a * scalar (int8)` | int8 | 100,000 | 0.0180 ms | not_measured | managed | 1.250x |
| `a * scalar (int8)` | int8 | 10,000,000 | 2.3829 ms | not_measured | managed | 1.299x |
| `a * scalar (uint16)` | uint16 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a * scalar (uint16)` | uint16 | 100,000 | 0.0285 ms | not_measured | managed | 0.874x |
| `a * scalar (uint16)` | uint16 | 10,000,000 | 4.6171 ms | not_measured | managed | 0.957x |
| `a * scalar (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `a * scalar (uint32)` | uint32 | 100,000 | 0.0515 ms | not_measured | managed | 0.464x |
| `a * scalar (uint32)` | uint32 | 10,000,000 | 9.1022 ms | not_measured | managed | 0.852x |
| `a * scalar (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a * scalar (uint64)` | uint64 | 100,000 | 0.1165 ms | not_measured | managed | 0.205x |
| `a * scalar (uint64)` | uint64 | 10,000,000 | 28.2317 ms | not_measured | managed | 1.114x |
| `a * scalar (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 0.800x |
| `a * scalar (uint8)` | uint8 | 100,000 | 0.0168 ms | not_measured | managed | 1.405x |
| `a * scalar (uint8)` | uint8 | 10,000,000 | 2.1467 ms | not_measured | managed | 1.573x |
| `a + 5 (literal) (complex128)` | complex128 | 1,000 | 0.0075 ms | not_measured | managed | 0.147x |
| `a + 5 (literal) (complex128)` | complex128 | 100,000 | 0.3503 ms | not_measured | managed | 1.049x |
| `a + 5 (literal) (complex128)` | complex128 | 10,000,000 | 25.3508 ms | not_measured | managed | 2.104x |
| `a + 5 (literal) (float16)` | float16 | 1,000 | 0.0108 ms | not_measured | managed | 0.574x |
| `a + 5 (literal) (float16)` | float16 | 100,000 | 0.9225 ms | not_measured | managed | 0.322x |
| `a + 5 (literal) (float16)` | float16 | 10,000,000 | 46.6935 ms | not_measured | managed | 1.564x |
| `a + 5 (literal) (float32)` | float32 | 1,000 | 0.0065 ms | not_measured | managed | 0.138x |
| `a + 5 (literal) (float32)` | float32 | 100,000 | 0.0590 ms | not_measured | managed | 0.120x |
| `a + 5 (literal) (float32)` | float32 | 10,000,000 | 5.0671 ms | not_measured | managed | 2.657x |
| `a + 5 (literal) (float64)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 0.154x |
| `a + 5 (literal) (float64)` | float64 | 100,000 | 0.1177 ms | not_measured | managed | 0.118x |
| `a + 5 (literal) (float64)` | float64 | 10,000,000 | 13.5038 ms | not_measured | managed | 1.927x |
| `a + 5 (literal) (int16)` | int16 | 1,000 | 0.0036 ms | not_measured | managed | 0.306x |
| `a + 5 (literal) (int16)` | int16 | 100,000 | 0.0311 ms | not_measured | managed | 0.868x |
| `a + 5 (literal) (int16)` | int16 | 10,000,000 | 5.6348 ms | not_measured | managed | 0.788x |
| `a + 5 (literal) (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 0.289x |
| `a + 5 (literal) (int32)` | int32 | 100,000 | 0.0590 ms | not_measured | managed | 0.480x |
| `a + 5 (literal) (int32)` | int32 | 10,000,000 | 5.4590 ms | not_measured | managed | 1.612x |
| `a + 5 (literal) (int64)` | int64 | 1,000 | 0.0054 ms | not_measured | managed | 0.204x |
| `a + 5 (literal) (int64)` | int64 | 100,000 | 0.1131 ms | not_measured | managed | 0.245x |
| `a + 5 (literal) (int64)` | int64 | 10,000,000 | 16.3404 ms | not_measured | managed | 0.935x |
| `a + 5 (literal) (int8)` | int8 | 1,000 | 0.0030 ms | not_measured | managed | 0.300x |
| `a + 5 (literal) (int8)` | int8 | 100,000 | 0.0170 ms | not_measured | managed | 1.594x |
| `a + 5 (literal) (int8)` | int8 | 10,000,000 | 2.5881 ms | not_measured | managed | 1.502x |
| `a + 5 (literal) (uint16)` | uint16 | 1,000 | 0.0060 ms | not_measured | managed | 0.183x |
| `a + 5 (literal) (uint16)` | uint16 | 100,000 | 0.0312 ms | not_measured | managed | 0.833x |
| `a + 5 (literal) (uint16)` | uint16 | 10,000,000 | 4.7756 ms | not_measured | managed | 0.991x |
| `a + 5 (literal) (uint32)` | uint32 | 1,000 | 0.0059 ms | not_measured | managed | 0.186x |
| `a + 5 (literal) (uint32)` | uint32 | 100,000 | 0.0588 ms | not_measured | managed | 0.446x |
| `a + 5 (literal) (uint32)` | uint32 | 10,000,000 | 5.6691 ms | not_measured | managed | 1.403x |
| `a + 5 (literal) (uint64)` | uint64 | 1,000 | 0.0054 ms | not_measured | managed | 0.185x |
| `a + 5 (literal) (uint64)` | uint64 | 100,000 | 0.1673 ms | not_measured | managed | 0.169x |
| `a + 5 (literal) (uint64)` | uint64 | 10,000,000 | 13.9885 ms | not_measured | managed | 2.026x |
| `a + 5 (literal) (uint8)` | uint8 | 1,000 | 0.0033 ms | not_measured | managed | 0.273x |
| `a + 5 (literal) (uint8)` | uint8 | 100,000 | 0.0190 ms | not_measured | managed | 1.374x |
| `a + 5 (literal) (uint8)` | uint8 | 10,000,000 | 2.4842 ms | not_measured | managed | 2.066x |
| `a + b (element-wise) (complex128)` | complex128 | 1,000 | 0.0037 ms | not_measured | managed | 0.297x |
| `a + b (element-wise) (complex128)` | complex128 | 100,000 | 0.3930 ms | not_measured | managed | 0.973x |
| `a + b (element-wise) (complex128)` | complex128 | 10,000,000 | 70.1479 ms | not_measured | managed | 0.936x |
| `a + b (element-wise) (float16)` | float16 | 1,000 | 0.0105 ms | not_measured | managed | 0.571x |
| `a + b (element-wise) (float16)` | float16 | 100,000 | 0.9632 ms | not_measured | managed | 0.308x |
| `a + b (element-wise) (float16)` | float16 | 10,000,000 | 94.3281 ms | not_measured | managed | 0.756x |
| `a + b (element-wise) (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a + b (element-wise) (float32)` | float32 | 100,000 | 0.0538 ms | not_measured | managed | 0.138x |
| `a + b (element-wise) (float32)` | float32 | 10,000,000 | 11.1473 ms | not_measured | managed | 1.733x |
| `a + b (element-wise) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `a + b (element-wise) (float64)` | float64 | 100,000 | 0.1149 ms | not_measured | managed | 0.272x |
| `a + b (element-wise) (float64)` | float64 | 10,000,000 | 27.7865 ms | not_measured | managed | 1.166x |
| `a + b (element-wise) (int16)` | int16 | 1,000 | 0.0022 ms | not_measured | managed | 0.591x |
| `a + b (element-wise) (int16)` | int16 | 100,000 | 0.0313 ms | not_measured | managed | 0.968x |
| `a + b (element-wise) (int16)` | int16 | 10,000,000 | 6.1888 ms | not_measured | managed | 0.869x |
| `a + b (element-wise) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `a + b (element-wise) (int32)` | int32 | 100,000 | 0.0630 ms | not_measured | managed | 0.506x |
| `a + b (element-wise) (int32)` | int32 | 10,000,000 | 11.6492 ms | not_measured | managed | 0.735x |
| `a + b (element-wise) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.303x |
| `a + b (element-wise) (int64)` | int64 | 100,000 | 0.1235 ms | not_measured | managed | 0.296x |
| `a + b (element-wise) (int64)` | int64 | 10,000,000 | 27.2986 ms | not_measured | managed | 0.650x |
| `a + b (element-wise) (int8)` | int8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `a + b (element-wise) (int8)` | int8 | 100,000 | 0.0203 ms | not_measured | managed | 1.591x |
| `a + b (element-wise) (int8)` | int8 | 10,000,000 | 3.3838 ms | not_measured | managed | 1.282x |
| `a + b (element-wise) (uint16)` | uint16 | 1,000 | 0.0014 ms | not_measured | managed | 0.571x |
| `a + b (element-wise) (uint16)` | uint16 | 100,000 | 0.0318 ms | not_measured | managed | 1.025x |
| `a + b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.3501 ms | not_measured | managed | 0.870x |
| `a + b (element-wise) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a + b (element-wise) (uint32)` | uint32 | 100,000 | 0.0635 ms | not_measured | managed | 0.480x |
| `a + b (element-wise) (uint32)` | uint32 | 10,000,000 | 12.1666 ms | not_measured | managed | 0.934x |
| `a + b (element-wise) (uint64)` | uint64 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `a + b (element-wise) (uint64)` | uint64 | 100,000 | 0.1133 ms | not_measured | managed | 0.355x |
| `a + b (element-wise) (uint64)` | uint64 | 10,000,000 | 27.6866 ms | not_measured | managed | 0.661x |
| `a + b (element-wise) (uint8)` | uint8 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `a + b (element-wise) (uint8)` | uint8 | 100,000 | 0.0184 ms | not_measured | managed | 1.614x |
| `a + b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.0451 ms | not_measured | managed | 1.571x |
| `a + scalar (complex128)` | complex128 | 1,000 | 0.0036 ms | not_measured | managed | 0.250x |
| `a + scalar (complex128)` | complex128 | 100,000 | 0.3667 ms | not_measured | managed | 1.014x |
| `a + scalar (complex128)` | complex128 | 10,000,000 | 25.5910 ms | not_measured | managed | 2.365x |
| `a + scalar (float16)` | float16 | 1,000 | 0.0101 ms | not_measured | managed | 0.634x |
| `a + scalar (float16)` | float16 | 100,000 | 0.9532 ms | not_measured | managed | 0.316x |
| `a + scalar (float16)` | float16 | 10,000,000 | 46.9215 ms | not_measured | managed | 1.623x |
| `a + scalar (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a + scalar (float32)` | float32 | 100,000 | 0.0542 ms | not_measured | managed | 0.124x |
| `a + scalar (float32)` | float32 | 10,000,000 | 5.2780 ms | not_measured | managed | 2.610x |
| `a + scalar (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `a + scalar (float64)` | float64 | 100,000 | 0.1198 ms | not_measured | managed | 0.109x |
| `a + scalar (float64)` | float64 | 10,000,000 | 13.8581 ms | not_measured | managed | 1.787x |
| `a + scalar (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.429x |
| `a + scalar (int16)` | int16 | 100,000 | 0.0306 ms | not_measured | managed | 0.817x |
| `a + scalar (int16)` | int16 | 10,000,000 | 6.0762 ms | not_measured | managed | 0.754x |
| `a + scalar (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `a + scalar (int32)` | int32 | 100,000 | 0.0534 ms | not_measured | managed | 0.534x |
| `a + scalar (int32)` | int32 | 10,000,000 | 7.8419 ms | not_measured | managed | 1.064x |
| `a + scalar (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.321x |
| `a + scalar (int64)` | int64 | 100,000 | 0.1204 ms | not_measured | managed | 0.217x |
| `a + scalar (int64)` | int64 | 10,000,000 | 19.4043 ms | not_measured | managed | 0.787x |
| `a + scalar (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a + scalar (int8)` | int8 | 100,000 | 0.0164 ms | not_measured | managed | 1.628x |
| `a + scalar (int8)` | int8 | 10,000,000 | 2.8284 ms | not_measured | managed | 1.312x |
| `a + scalar (uint16)` | uint16 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a + scalar (uint16)` | uint16 | 100,000 | 0.0283 ms | not_measured | managed | 0.922x |
| `a + scalar (uint16)` | uint16 | 10,000,000 | 5.0920 ms | not_measured | managed | 0.924x |
| `a + scalar (uint32)` | uint32 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `a + scalar (uint32)` | uint32 | 100,000 | 0.0576 ms | not_measured | managed | 0.441x |
| `a + scalar (uint32)` | uint32 | 10,000,000 | 5.5006 ms | not_measured | managed | 1.700x |
| `a + scalar (uint64)` | uint64 | 1,000 | 0.0027 ms | not_measured | managed | 0.333x |
| `a + scalar (uint64)` | uint64 | 100,000 | 0.1050 ms | not_measured | managed | 0.277x |
| `a + scalar (uint64)` | uint64 | 10,000,000 | 14.6908 ms | not_measured | managed | 1.827x |
| `a + scalar (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a + scalar (uint8)` | uint8 | 100,000 | 0.0160 ms | not_measured | managed | 1.637x |
| `a + scalar (uint8)` | uint8 | 10,000,000 | 2.9996 ms | not_measured | managed | 1.445x |
| `a - b (element-wise) (complex128)` | complex128 | 1,000 | 0.0031 ms | not_measured | managed | 0.258x |
| `a - b (element-wise) (complex128)` | complex128 | 100,000 | 0.4024 ms | not_measured | managed | 0.948x |
| `a - b (element-wise) (complex128)` | complex128 | 10,000,000 | 64.5398 ms | not_measured | managed | 1.022x |
| `a - b (element-wise) (float16)` | float16 | 1,000 | 0.0099 ms | not_measured | managed | 0.424x |
| `a - b (element-wise) (float16)` | float16 | 100,000 | 0.9942 ms | not_measured | managed | 0.297x |
| `a - b (element-wise) (float16)` | float16 | 10,000,000 | 95.3852 ms | not_measured | managed | 0.923x |
| `a - b (element-wise) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `a - b (element-wise) (float32)` | float32 | 100,000 | 0.0538 ms | not_measured | managed | 0.151x |
| `a - b (element-wise) (float32)` | float32 | 10,000,000 | 11.8841 ms | not_measured | managed | 1.410x |
| `a - b (element-wise) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `a - b (element-wise) (float64)` | float64 | 100,000 | 0.1476 ms | not_measured | managed | 0.210x |
| `a - b (element-wise) (float64)` | float64 | 10,000,000 | 28.0915 ms | not_measured | managed | 1.328x |
| `a - b (element-wise) (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `a - b (element-wise) (int16)` | int16 | 100,000 | 0.0265 ms | not_measured | managed | 1.223x |
| `a - b (element-wise) (int16)` | int16 | 10,000,000 | 7.9486 ms | not_measured | managed | 0.650x |
| `a - b (element-wise) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a - b (element-wise) (int32)` | int32 | 100,000 | 0.0524 ms | not_measured | managed | 0.634x |
| `a - b (element-wise) (int32)` | int32 | 10,000,000 | 11.6284 ms | not_measured | managed | 0.884x |
| `a - b (element-wise) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a - b (element-wise) (int64)` | int64 | 100,000 | 0.1077 ms | not_measured | managed | 0.333x |
| `a - b (element-wise) (int64)` | int64 | 10,000,000 | 28.0115 ms | not_measured | managed | 0.632x |
| `a - b (element-wise) (int8)` | int8 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `a - b (element-wise) (int8)` | int8 | 100,000 | 0.0151 ms | not_measured | managed | 1.967x |
| `a - b (element-wise) (int8)` | int8 | 10,000,000 | 4.2231 ms | not_measured | managed | 0.930x |
| `a - b (element-wise) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a - b (element-wise) (uint16)` | uint16 | 100,000 | 0.0270 ms | not_measured | managed | 1.167x |
| `a - b (element-wise) (uint16)` | uint16 | 10,000,000 | 6.3404 ms | not_measured | managed | 0.852x |
| `a - b (element-wise) (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a - b (element-wise) (uint32)` | uint32 | 100,000 | 0.0508 ms | not_measured | managed | 0.671x |
| `a - b (element-wise) (uint32)` | uint32 | 10,000,000 | 11.6455 ms | not_measured | managed | 0.770x |
| `a - b (element-wise) (uint64)` | uint64 | 1,000 | 0.0031 ms | not_measured | managed | 0.226x |
| `a - b (element-wise) (uint64)` | uint64 | 100,000 | 0.1080 ms | not_measured | managed | 0.331x |
| `a - b (element-wise) (uint64)` | uint64 | 10,000,000 | 26.9651 ms | not_measured | managed | 1.405x |
| `a - b (element-wise) (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `a - b (element-wise) (uint8)` | uint8 | 100,000 | 0.0148 ms | not_measured | managed | 1.973x |
| `a - b (element-wise) (uint8)` | uint8 | 10,000,000 | 3.4727 ms | not_measured | managed | 1.645x |
| `a - scalar (complex128)` | complex128 | 1,000 | 0.0035 ms | not_measured | managed | 0.257x |
| `a - scalar (complex128)` | complex128 | 100,000 | 0.3283 ms | not_measured | managed | 1.137x |
| `a - scalar (complex128)` | complex128 | 10,000,000 | 25.5075 ms | not_measured | managed | 2.146x |
| `a - scalar (float16)` | float16 | 1,000 | 0.0102 ms | not_measured | managed | 0.627x |
| `a - scalar (float16)` | float16 | 100,000 | 0.9209 ms | not_measured | managed | 0.320x |
| `a - scalar (float16)` | float16 | 10,000,000 | 48.8542 ms | not_measured | managed | 1.579x |
| `a - scalar (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `a - scalar (float32)` | float32 | 100,000 | 0.0516 ms | not_measured | managed | 0.136x |
| `a - scalar (float32)` | float32 | 10,000,000 | 5.4220 ms | not_measured | managed | 2.576x |
| `a - scalar (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.316x |
| `a - scalar (float64)` | float64 | 100,000 | 0.1052 ms | not_measured | managed | 0.127x |
| `a - scalar (float64)` | float64 | 10,000,000 | 14.2505 ms | not_measured | managed | 1.761x |
| `a - scalar (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a - scalar (int16)` | int16 | 100,000 | 0.0273 ms | not_measured | managed | 0.941x |
| `a - scalar (int16)` | int16 | 10,000,000 | 3.0203 ms | not_measured | managed | 1.476x |
| `a - scalar (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a - scalar (int32)` | int32 | 100,000 | 0.0529 ms | not_measured | managed | 0.495x |
| `a - scalar (int32)` | int32 | 10,000,000 | 5.3366 ms | not_measured | managed | 1.617x |
| `a - scalar (int64)` | int64 | 1,000 | 0.0023 ms | not_measured | managed | 0.478x |
| `a - scalar (int64)` | int64 | 100,000 | 0.1175 ms | not_measured | managed | 0.218x |
| `a - scalar (int64)` | int64 | 10,000,000 | 14.5652 ms | not_measured | managed | 1.055x |
| `a - scalar (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `a - scalar (int8)` | int8 | 100,000 | 0.0178 ms | not_measured | managed | 1.427x |
| `a - scalar (int8)` | int8 | 10,000,000 | 1.4378 ms | not_measured | managed | 2.358x |
| `a - scalar (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a - scalar (uint16)` | uint16 | 100,000 | 0.0270 ms | not_measured | managed | 0.956x |
| `a - scalar (uint16)` | uint16 | 10,000,000 | 3.2264 ms | not_measured | managed | 1.416x |
| `a - scalar (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `a - scalar (uint32)` | uint32 | 100,000 | 0.0510 ms | not_measured | managed | 0.478x |
| `a - scalar (uint32)` | uint32 | 10,000,000 | 5.8337 ms | not_measured | managed | 1.574x |
| `a - scalar (uint64)` | uint64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `a - scalar (uint64)` | uint64 | 100,000 | 0.1543 ms | not_measured | managed | 0.165x |
| `a - scalar (uint64)` | uint64 | 10,000,000 | 14.0091 ms | not_measured | managed | 2.670x |
| `a - scalar (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `a - scalar (uint8)` | uint8 | 100,000 | 0.0168 ms | not_measured | managed | 1.506x |
| `a - scalar (uint8)` | uint8 | 10,000,000 | 1.3294 ms | not_measured | managed | 3.040x |
| `a / b (element-wise) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.185x |
| `a / b (element-wise) (float32)` | float32 | 100,000 | 0.0658 ms | not_measured | managed | 0.199x |
| `a / b (element-wise) (float32)` | float32 | 10,000,000 | 11.7430 ms | not_measured | managed | 1.380x |
| `a / b (element-wise) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `a / b (element-wise) (float64)` | float64 | 100,000 | 0.2063 ms | not_measured | managed | 0.203x |
| `a / b (element-wise) (float64)` | float64 | 10,000,000 | 27.3738 ms | not_measured | managed | 1.205x |
| `a / b (element-wise) (int32)` | int32 | 1,000 | 0.0045 ms | not_measured | managed | 0.533x |
| `a / b (element-wise) (int32)` | int32 | 100,000 | 0.2114 ms | not_measured | managed | 0.427x |
| `a / b (element-wise) (int32)` | int32 | 10,000,000 | 25.7646 ms | not_measured | managed | 0.803x |
| `a / b (element-wise) (int64)` | int64 | 1,000 | 0.0046 ms | not_measured | managed | 0.413x |
| `a / b (element-wise) (int64)` | int64 | 100,000 | 0.2084 ms | not_measured | managed | 0.401x |
| `a / b (element-wise) (int64)` | int64 | 10,000,000 | 29.8175 ms | not_measured | managed | 0.834x |
| `a / scalar (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.292x |
| `a / scalar (float32)` | float32 | 100,000 | 0.0703 ms | not_measured | managed | 0.186x |
| `a / scalar (float32)` | float32 | 10,000,000 | 8.2356 ms | not_measured | managed | 1.551x |
| `a / scalar (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.367x |
| `a / scalar (float64)` | float64 | 100,000 | 0.2052 ms | not_measured | managed | 0.204x |
| `a / scalar (float64)` | float64 | 10,000,000 | 24.6504 ms | not_measured | managed | 1.718x |
| `a / scalar (int32)` | int32 | 1,000 | 0.0052 ms | not_measured | managed | 0.346x |
| `a / scalar (int32)` | int32 | 100,000 | 0.2101 ms | not_measured | managed | 0.317x |
| `a / scalar (int32)` | int32 | 10,000,000 | 24.4953 ms | not_measured | managed | 0.686x |
| `a / scalar (int64)` | int64 | 1,000 | 0.0049 ms | not_measured | managed | 0.327x |
| `a / scalar (int64)` | int64 | 100,000 | 0.2088 ms | not_measured | managed | 0.311x |
| `a / scalar (int64)` | int64 | 10,000,000 | 26.4072 ms | not_measured | managed | 0.701x |
| `np.add(a, b) (complex128)` | complex128 | 1,000 | 0.0032 ms | not_measured | managed | 0.312x |
| `np.add(a, b) (complex128)` | complex128 | 100,000 | 0.3802 ms | not_measured | managed | 1.001x |
| `np.add(a, b) (complex128)` | complex128 | 10,000,000 | 62.6251 ms | not_measured | managed | 1.055x |
| `np.add(a, b) (float16)` | float16 | 1,000 | 0.0101 ms | not_measured | managed | 0.624x |
| `np.add(a, b) (float16)` | float16 | 100,000 | 0.9251 ms | not_measured | managed | 0.320x |
| `np.add(a, b) (float16)` | float16 | 10,000,000 | 96.8722 ms | not_measured | managed | 0.920x |
| `np.add(a, b) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.add(a, b) (float32)` | float32 | 100,000 | 0.0545 ms | not_measured | managed | 0.145x |
| `np.add(a, b) (float32)` | float32 | 10,000,000 | 12.1086 ms | not_measured | managed | 1.445x |
| `np.add(a, b) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.add(a, b) (float64)` | float64 | 100,000 | 0.1105 ms | not_measured | managed | 0.288x |
| `np.add(a, b) (float64)` | float64 | 10,000,000 | 28.3575 ms | not_measured | managed | 1.134x |
| `np.add(a, b) (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `np.add(a, b) (int16)` | int16 | 100,000 | 0.0390 ms | not_measured | managed | 0.792x |
| `np.add(a, b) (int16)` | int16 | 10,000,000 | 6.3066 ms | not_measured | managed | 0.854x |
| `np.add(a, b) (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `np.add(a, b) (int32)` | int32 | 100,000 | 0.0629 ms | not_measured | managed | 0.564x |
| `np.add(a, b) (int32)` | int32 | 10,000,000 | 11.7717 ms | not_measured | managed | 0.835x |
| `np.add(a, b) (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.add(a, b) (int64)` | int64 | 100,000 | 0.1140 ms | not_measured | managed | 0.307x |
| `np.add(a, b) (int64)` | int64 | 10,000,000 | 33.0584 ms | not_measured | managed | 0.653x |
| `np.add(a, b) (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.583x |
| `np.add(a, b) (int8)` | int8 | 100,000 | 0.0190 ms | not_measured | managed | 1.616x |
| `np.add(a, b) (int8)` | int8 | 10,000,000 | 3.1147 ms | not_measured | managed | 1.292x |
| `np.add(a, b) (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `np.add(a, b) (uint16)` | uint16 | 100,000 | 0.0343 ms | not_measured | managed | 0.953x |
| `np.add(a, b) (uint16)` | uint16 | 10,000,000 | 6.2812 ms | not_measured | managed | 0.930x |
| `np.add(a, b) (uint32)` | uint32 | 1,000 | 0.0023 ms | not_measured | managed | 0.348x |
| `np.add(a, b) (uint32)` | uint32 | 100,000 | 0.0590 ms | not_measured | managed | 0.602x |
| `np.add(a, b) (uint32)` | uint32 | 10,000,000 | 12.1761 ms | not_measured | managed | 0.947x |
| `np.add(a, b) (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.add(a, b) (uint64)` | uint64 | 100,000 | 0.1189 ms | not_measured | managed | 0.384x |
| `np.add(a, b) (uint64)` | uint64 | 10,000,000 | 28.5135 ms | not_measured | managed | 0.634x |
| `np.add(a, b) (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `np.add(a, b) (uint8)` | uint8 | 100,000 | 0.0189 ms | not_measured | managed | 1.582x |
| `np.add(a, b) (uint8)` | uint8 | 10,000,000 | 2.8119 ms | not_measured | managed | 1.509x |
| `np.divide(a, b) (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `np.divide(a, b) (float32)` | float32 | 100,000 | 0.0755 ms | not_measured | managed | 0.172x |
| `np.divide(a, b) (float32)` | float32 | 10,000,000 | 11.5676 ms | not_measured | managed | 1.448x |
| `np.divide(a, b) (float64)` | float64 | 1,000 | 0.0039 ms | not_measured | managed | 0.205x |
| `np.divide(a, b) (float64)` | float64 | 100,000 | 0.2078 ms | not_measured | managed | 0.215x |
| `np.divide(a, b) (float64)` | float64 | 10,000,000 | 27.7701 ms | not_measured | managed | 1.312x |
| `np.divide(a, b) (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 1.118x |
| `np.divide(a, b) (int32)` | int32 | 100,000 | 0.2185 ms | not_measured | managed | 0.403x |
| `np.divide(a, b) (int32)` | int32 | 10,000,000 | 26.0952 ms | not_measured | managed | 0.785x |
| `np.divide(a, b) (int64)` | int64 | 1,000 | 0.0017 ms | not_measured | managed | 1.235x |
| `np.divide(a, b) (int64)` | int64 | 100,000 | 0.2101 ms | not_measured | managed | 0.381x |
| `np.divide(a, b) (int64)` | int64 | 10,000,000 | 30.8500 ms | not_measured | managed | 0.783x |
| `np.floor_divide(a, b) (float32)` | float32 | 1,000 | 0.0063 ms | not_measured | managed | 1.825x |
| `np.floor_divide(a, b) (float32)` | float32 | 100,000 | 1.6499 ms | not_measured | managed | 0.929x |
| `np.floor_divide(a, b) (float32)` | float32 | 10,000,000 | 164.6854 ms | not_measured | managed | 1.410x |
| `np.floor_divide(a, b) (float64)` | float64 | 1,000 | 0.0108 ms | not_measured | managed | 0.870x |
| `np.floor_divide(a, b) (float64)` | float64 | 100,000 | 1.5334 ms | not_measured | managed | 0.870x |
| `np.floor_divide(a, b) (float64)` | float64 | 10,000,000 | 157.1928 ms | not_measured | managed | 1.446x |
| `np.floor_divide(a, b) (int32)` | int32 | 1,000 | 0.0025 ms | not_measured | managed | 0.720x |
| `np.floor_divide(a, b) (int32)` | int32 | 100,000 | 0.6760 ms | not_measured | managed | 0.576x |
| `np.floor_divide(a, b) (int32)` | int32 | 10,000,000 | 62.6822 ms | not_measured | managed | 0.676x |
| `np.floor_divide(a, b) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.793x |
| `np.floor_divide(a, b) (int64)` | int64 | 100,000 | 0.6799 ms | not_measured | managed | 0.559x |
| `np.floor_divide(a, b) (int64)` | int64 | 10,000,000 | 71.0474 ms | not_measured | managed | 0.666x |
| `np.mod(a, b) (float32)` | float32 | 1,000 | 0.0124 ms | not_measured | managed | 0.944x |
| `np.mod(a, b) (float32)` | float32 | 100,000 | 1.6041 ms | not_measured | managed | 0.969x |
| `np.mod(a, b) (float32)` | float32 | 10,000,000 | 162.9179 ms | not_measured | managed | 1.462x |
| `np.mod(a, b) (float64)` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 1.049x |
| `np.mod(a, b) (float64)` | float64 | 100,000 | 1.4560 ms | not_measured | managed | 0.946x |
| `np.mod(a, b) (float64)` | float64 | 10,000,000 | 155.2454 ms | not_measured | managed | 1.457x |
| `np.mod(a, b) (int32)` | int32 | 1,000 | 0.0025 ms | not_measured | managed | 0.800x |
| `np.mod(a, b) (int32)` | int32 | 100,000 | 0.6048 ms | not_measured | managed | 0.620x |
| `np.mod(a, b) (int32)` | int32 | 10,000,000 | 61.1395 ms | not_measured | managed | 0.690x |
| `np.mod(a, b) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 1.133x |
| `np.mod(a, b) (int64)` | int64 | 100,000 | 0.6172 ms | not_measured | managed | 0.628x |
| `np.mod(a, b) (int64)` | int64 | 10,000,000 | 69.7648 ms | not_measured | managed | 0.686x |
| `np.multiply(a, b) (complex128)` | complex128 | 1,000 | 0.0033 ms | not_measured | managed | 0.273x |
| `np.multiply(a, b) (complex128)` | complex128 | 100,000 | 0.4642 ms | not_measured | managed | 0.808x |
| `np.multiply(a, b) (complex128)` | complex128 | 10,000,000 | 57.1746 ms | not_measured | managed | 1.179x |
| `np.multiply(a, b) (float16)` | float16 | 1,000 | 0.0100 ms | not_measured | managed | 0.320x |
| `np.multiply(a, b) (float16)` | float16 | 100,000 | 0.9260 ms | not_measured | managed | 0.323x |
| `np.multiply(a, b) (float16)` | float16 | 10,000,000 | 92.2221 ms | not_measured | managed | 0.800x |
| `np.multiply(a, b) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.multiply(a, b) (float32)` | float32 | 100,000 | 0.0624 ms | not_measured | managed | 0.162x |
| `np.multiply(a, b) (float32)` | float32 | 10,000,000 | 11.1890 ms | not_measured | managed | 1.712x |
| `np.multiply(a, b) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `np.multiply(a, b) (float64)` | float64 | 100,000 | 0.1107 ms | not_measured | managed | 0.288x |
| `np.multiply(a, b) (float64)` | float64 | 10,000,000 | 27.4395 ms | not_measured | managed | 1.214x |
| `np.multiply(a, b) (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `np.multiply(a, b) (int16)` | int16 | 100,000 | 0.0268 ms | not_measured | managed | 1.183x |
| `np.multiply(a, b) (int16)` | int16 | 10,000,000 | 6.8326 ms | not_measured | managed | 0.738x |
| `np.multiply(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.multiply(a, b) (int32)` | int32 | 100,000 | 0.0508 ms | not_measured | managed | 0.616x |
| `np.multiply(a, b) (int32)` | int32 | 10,000,000 | 11.9980 ms | not_measured | managed | 0.815x |
| `np.multiply(a, b) (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.multiply(a, b) (int64)` | int64 | 100,000 | 0.1148 ms | not_measured | managed | 0.308x |
| `np.multiply(a, b) (int64)` | int64 | 10,000,000 | 30.4784 ms | not_measured | managed | 0.553x |
| `np.multiply(a, b) (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.778x |
| `np.multiply(a, b) (int8)` | int8 | 100,000 | 0.0162 ms | not_measured | managed | 1.858x |
| `np.multiply(a, b) (int8)` | int8 | 10,000,000 | 3.4538 ms | not_measured | managed | 1.097x |
| `np.multiply(a, b) (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 0.615x |
| `np.multiply(a, b) (uint16)` | uint16 | 100,000 | 0.0261 ms | not_measured | managed | 1.149x |
| `np.multiply(a, b) (uint16)` | uint16 | 10,000,000 | 6.3619 ms | not_measured | managed | 0.803x |
| `np.multiply(a, b) (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.multiply(a, b) (uint32)` | uint32 | 100,000 | 0.0506 ms | not_measured | managed | 0.668x |
| `np.multiply(a, b) (uint32)` | uint32 | 10,000,000 | 11.1765 ms | not_measured | managed | 0.805x |
| `np.multiply(a, b) (uint64)` | uint64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.multiply(a, b) (uint64)` | uint64 | 100,000 | 0.1162 ms | not_measured | managed | 0.303x |
| `np.multiply(a, b) (uint64)` | uint64 | 10,000,000 | 28.0677 ms | not_measured | managed | 1.661x |
| `np.multiply(a, b) (uint8)` | uint8 | 1,000 | 0.0024 ms | not_measured | managed | 0.292x |
| `np.multiply(a, b) (uint8)` | uint8 | 100,000 | 0.0167 ms | not_measured | managed | 1.713x |
| `np.multiply(a, b) (uint8)` | uint8 | 10,000,000 | 3.3911 ms | not_measured | managed | 1.426x |
| `np.subtract(a, b) (complex128)` | complex128 | 1,000 | 0.0035 ms | not_measured | managed | 0.229x |
| `np.subtract(a, b) (complex128)` | complex128 | 100,000 | 0.3600 ms | not_measured | managed | 1.089x |
| `np.subtract(a, b) (complex128)` | complex128 | 10,000,000 | 66.0251 ms | not_measured | managed | 1.106x |
| `np.subtract(a, b) (float16)` | float16 | 1,000 | 0.0101 ms | not_measured | managed | 0.426x |
| `np.subtract(a, b) (float16)` | float16 | 100,000 | 0.9495 ms | not_measured | managed | 0.320x |
| `np.subtract(a, b) (float16)` | float16 | 10,000,000 | 96.7641 ms | not_measured | managed | 0.799x |
| `np.subtract(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.subtract(a, b) (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.150x |
| `np.subtract(a, b) (float32)` | float32 | 10,000,000 | 11.4268 ms | not_measured | managed | 1.480x |
| `np.subtract(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `np.subtract(a, b) (float64)` | float64 | 100,000 | 0.1110 ms | not_measured | managed | 0.281x |
| `np.subtract(a, b) (float64)` | float64 | 10,000,000 | 26.0326 ms | not_measured | managed | 1.298x |
| `np.subtract(a, b) (int16)` | int16 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.subtract(a, b) (int16)` | int16 | 100,000 | 0.0275 ms | not_measured | managed | 1.175x |
| `np.subtract(a, b) (int16)` | int16 | 10,000,000 | 6.9187 ms | not_measured | managed | 0.759x |
| `np.subtract(a, b) (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.533x |
| `np.subtract(a, b) (int32)` | int32 | 100,000 | 0.0512 ms | not_measured | managed | 0.615x |
| `np.subtract(a, b) (int32)` | int32 | 10,000,000 | 11.6797 ms | not_measured | managed | 0.775x |
| `np.subtract(a, b) (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.subtract(a, b) (int64)` | int64 | 100,000 | 0.1050 ms | not_measured | managed | 0.359x |
| `np.subtract(a, b) (int64)` | int64 | 10,000,000 | 27.6089 ms | not_measured | managed | 0.642x |
| `np.subtract(a, b) (int8)` | int8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.subtract(a, b) (int8)` | int8 | 100,000 | 0.0148 ms | not_measured | managed | 2.000x |
| `np.subtract(a, b) (int8)` | int8 | 10,000,000 | 3.0308 ms | not_measured | managed | 1.302x |
| `np.subtract(a, b) (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `np.subtract(a, b) (uint16)` | uint16 | 100,000 | 0.0288 ms | not_measured | managed | 1.215x |
| `np.subtract(a, b) (uint16)` | uint16 | 10,000,000 | 6.6545 ms | not_measured | managed | 0.808x |
| `np.subtract(a, b) (uint32)` | uint32 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.subtract(a, b) (uint32)` | uint32 | 100,000 | 0.0514 ms | not_measured | managed | 0.619x |
| `np.subtract(a, b) (uint32)` | uint32 | 10,000,000 | 11.3774 ms | not_measured | managed | 0.779x |
| `np.subtract(a, b) (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.subtract(a, b) (uint64)` | uint64 | 100,000 | 0.1094 ms | not_measured | managed | 0.328x |
| `np.subtract(a, b) (uint64)` | uint64 | 10,000,000 | 26.5571 ms | not_measured | managed | 1.434x |
| `np.subtract(a, b) (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `np.subtract(a, b) (uint8)` | uint8 | 100,000 | 0.0156 ms | not_measured | managed | 1.949x |
| `np.subtract(a, b) (uint8)` | uint8 | 10,000,000 | 3.5428 ms | not_measured | managed | 1.370x |
| `np.true_divide(a, b) (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 0.500x |
| `np.true_divide(a, b) (float32)` | float32 | 100,000 | 0.0698 ms | not_measured | managed | 0.186x |
| `np.true_divide(a, b) (float32)` | float32 | 10,000,000 | 11.8506 ms | not_measured | managed | 1.442x |
| `np.true_divide(a, b) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `np.true_divide(a, b) (float64)` | float64 | 100,000 | 0.2094 ms | not_measured | managed | 0.192x |
| `np.true_divide(a, b) (float64)` | float64 | 10,000,000 | 29.9997 ms | not_measured | managed | 1.095x |
| `np.true_divide(a, b) (int32)` | int32 | 1,000 | 0.0032 ms | not_measured | managed | 0.594x |
| `np.true_divide(a, b) (int32)` | int32 | 100,000 | 0.2198 ms | not_measured | managed | 0.389x |
| `np.true_divide(a, b) (int32)` | int32 | 10,000,000 | 27.9825 ms | not_measured | managed | 0.741x |
| `np.true_divide(a, b) (int64)` | int64 | 1,000 | 0.0029 ms | not_measured | managed | 0.690x |
| `np.true_divide(a, b) (int64)` | int64 | 100,000 | 0.2145 ms | not_measured | managed | 0.379x |
| `np.true_divide(a, b) (int64)` | int64 | 10,000,000 | 30.6698 ms | not_measured | managed | 0.772x |
| `scalar - a (complex128)` | complex128 | 1,000 | 0.0031 ms | not_measured | managed | 0.290x |
| `scalar - a (complex128)` | complex128 | 100,000 | 0.2829 ms | not_measured | managed | 1.321x |
| `scalar - a (complex128)` | complex128 | 10,000,000 | 25.0545 ms | not_measured | managed | 2.017x |
| `scalar - a (float16)` | float16 | 1,000 | 0.0102 ms | not_measured | managed | 0.500x |
| `scalar - a (float16)` | float16 | 100,000 | 0.9139 ms | not_measured | managed | 0.322x |
| `scalar - a (float16)` | float16 | 10,000,000 | 47.7781 ms | not_measured | managed | 1.537x |
| `scalar - a (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `scalar - a (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.142x |
| `scalar - a (float32)` | float32 | 10,000,000 | 5.5919 ms | not_measured | managed | 2.543x |
| `scalar - a (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.231x |
| `scalar - a (float64)` | float64 | 100,000 | 0.1142 ms | not_measured | managed | 0.116x |
| `scalar - a (float64)` | float64 | 10,000,000 | 15.1896 ms | not_measured | managed | 1.763x |
| `scalar - a (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `scalar - a (int16)` | int16 | 100,000 | 0.0274 ms | not_measured | managed | 0.956x |
| `scalar - a (int16)` | int16 | 10,000,000 | 3.1127 ms | not_measured | managed | 1.499x |
| `scalar - a (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.588x |
| `scalar - a (int32)` | int32 | 100,000 | 0.0566 ms | not_measured | managed | 0.489x |
| `scalar - a (int32)` | int32 | 10,000,000 | 8.9213 ms | not_measured | managed | 1.198x |
| `scalar - a (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `scalar - a (int64)` | int64 | 100,000 | 0.1089 ms | not_measured | managed | 0.235x |
| `scalar - a (int64)` | int64 | 10,000,000 | 14.5704 ms | not_measured | managed | 1.040x |
| `scalar - a (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `scalar - a (int8)` | int8 | 100,000 | 0.0147 ms | not_measured | managed | 1.673x |
| `scalar - a (int8)` | int8 | 10,000,000 | 1.5380 ms | not_measured | managed | 2.361x |
| `scalar - a (uint16)` | uint16 | 1,000 | 0.0024 ms | not_measured | managed | 0.417x |
| `scalar - a (uint16)` | uint16 | 100,000 | 0.0275 ms | not_measured | managed | 0.967x |
| `scalar - a (uint16)` | uint16 | 10,000,000 | 3.1298 ms | not_measured | managed | 1.489x |
| `scalar - a (uint32)` | uint32 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `scalar - a (uint32)` | uint32 | 100,000 | 0.0513 ms | not_measured | managed | 0.483x |
| `scalar - a (uint32)` | uint32 | 10,000,000 | 5.4999 ms | not_measured | managed | 1.451x |
| `scalar - a (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `scalar - a (uint64)` | uint64 | 100,000 | 0.1028 ms | not_measured | managed | 0.252x |
| `scalar - a (uint64)` | uint64 | 10,000,000 | 14.2218 ms | not_measured | managed | 2.590x |
| `scalar - a (uint8)` | uint8 | 1,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `scalar - a (uint8)` | uint8 | 100,000 | 0.0151 ms | not_measured | managed | 1.636x |
| `scalar - a (uint8)` | uint8 | 10,000,000 | 1.3770 ms | not_measured | managed | 2.864x |
| `scalar / a (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `scalar / a (float32)` | float32 | 100,000 | 0.0679 ms | not_measured | managed | 0.194x |
| `scalar / a (float32)` | float32 | 10,000,000 | 9.4193 ms | not_measured | managed | 1.381x |
| `scalar / a (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.345x |
| `scalar / a (float64)` | float64 | 100,000 | 0.2059 ms | not_measured | managed | 0.199x |
| `scalar / a (float64)` | float64 | 10,000,000 | 24.7051 ms | not_measured | managed | 1.731x |
| `scalar / a (int32)` | int32 | 1,000 | 0.0055 ms | not_measured | managed | 0.327x |
| `scalar / a (int32)` | int32 | 100,000 | 0.2072 ms | not_measured | managed | 0.313x |
| `scalar / a (int32)` | int32 | 10,000,000 | 24.9825 ms | not_measured | managed | 0.668x |
| `scalar / a (int64)` | int64 | 1,000 | 0.0048 ms | not_measured | managed | 0.354x |
| `scalar / a (int64)` | int64 | 100,000 | 0.2168 ms | not_measured | managed | 0.314x |
| `scalar / a (int64)` | int64 | 10,000,000 | 26.9115 ms | not_measured | managed | 0.688x |
| `a & b (bool)` | bool | 1,000 | 0.0021 ms | not_measured | managed | 0.190x |
| `a & b (bool)` | bool | 100,000 | 0.0563 ms | not_measured | managed | 0.062x |
| `a & b (bool)` | bool | 10,000,000 | 5.9482 ms | not_measured | managed | 0.369x |
| `a & b (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `a & b (int16)` | int16 | 100,000 | 0.0326 ms | not_measured | managed | 0.951x |
| `a & b (int16)` | int16 | 10,000,000 | 6.5480 ms | not_measured | managed | 1.416x |
| `a & b (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a & b (int32)` | int32 | 100,000 | 0.0584 ms | not_measured | managed | 0.502x |
| `a & b (int32)` | int32 | 10,000,000 | 12.6206 ms | not_measured | managed | 1.345x |
| `a & b (int64)` | int64 | 1,000 | 0.0036 ms | not_measured | managed | 0.222x |
| `a & b (int64)` | int64 | 100,000 | 0.1175 ms | not_measured | managed | 0.309x |
| `a & b (int64)` | int64 | 10,000,000 | 31.4366 ms | not_measured | managed | 1.075x |
| `a & b (int8)` | int8 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `a & b (int8)` | int8 | 100,000 | 0.0173 ms | not_measured | managed | 1.740x |
| `a & b (int8)` | int8 | 10,000,000 | 3.3982 ms | not_measured | managed | 1.850x |
| `a & b (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `a & b (uint16)` | uint16 | 100,000 | 0.0270 ms | not_measured | managed | 1.093x |
| `a & b (uint16)` | uint16 | 10,000,000 | 7.1208 ms | not_measured | managed | 1.393x |
| `a & b (uint32)` | uint32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `a & b (uint32)` | uint32 | 100,000 | 0.0535 ms | not_measured | managed | 0.555x |
| `a & b (uint32)` | uint32 | 10,000,000 | 13.1233 ms | not_measured | managed | 1.345x |
| `a & b (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a & b (uint64)` | uint64 | 100,000 | 0.1048 ms | not_measured | managed | 0.391x |
| `a & b (uint64)` | uint64 | 10,000,000 | 27.6571 ms | not_measured | managed | 1.251x |
| `a & b (uint8)` | uint8 | 1,000 | 0.0024 ms | not_measured | managed | 0.292x |
| `a & b (uint8)` | uint8 | 100,000 | 0.0155 ms | not_measured | managed | 1.903x |
| `a & b (uint8)` | uint8 | 10,000,000 | 3.1778 ms | not_measured | managed | 2.206x |
| `a ^ b (bool)` | bool | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `a ^ b (bool)` | bool | 100,000 | 0.0556 ms | not_measured | managed | 0.056x |
| `a ^ b (bool)` | bool | 10,000,000 | 5.4316 ms | not_measured | managed | 0.425x |
| `a ^ b (int16)` | int16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `a ^ b (int16)` | int16 | 100,000 | 0.0278 ms | not_measured | managed | 1.083x |
| `a ^ b (int16)` | int16 | 10,000,000 | 7.1747 ms | not_measured | managed | 1.272x |
| `a ^ b (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `a ^ b (int32)` | int32 | 100,000 | 0.0605 ms | not_measured | managed | 0.499x |
| `a ^ b (int32)` | int32 | 10,000,000 | 12.4006 ms | not_measured | managed | 1.376x |
| `a ^ b (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a ^ b (int64)` | int64 | 100,000 | 0.1267 ms | not_measured | managed | 0.275x |
| `a ^ b (int64)` | int64 | 10,000,000 | 27.3200 ms | not_measured | managed | 1.221x |
| `a ^ b (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.583x |
| `a ^ b (int8)` | int8 | 100,000 | 0.0154 ms | not_measured | managed | 1.942x |
| `a ^ b (int8)` | int8 | 10,000,000 | 3.4846 ms | not_measured | managed | 1.872x |
| `a ^ b (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `a ^ b (uint16)` | uint16 | 100,000 | 0.0280 ms | not_measured | managed | 1.050x |
| `a ^ b (uint16)` | uint16 | 10,000,000 | 6.0472 ms | not_measured | managed | 1.499x |
| `a ^ b (uint32)` | uint32 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a ^ b (uint32)` | uint32 | 100,000 | 0.0567 ms | not_measured | managed | 0.529x |
| `a ^ b (uint32)` | uint32 | 10,000,000 | 12.5158 ms | not_measured | managed | 1.446x |
| `a ^ b (uint64)` | uint64 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `a ^ b (uint64)` | uint64 | 100,000 | 0.1041 ms | not_measured | managed | 0.345x |
| `a ^ b (uint64)` | uint64 | 10,000,000 | 30.5433 ms | not_measured | managed | 1.329x |
| `a ^ b (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `a ^ b (uint8)` | uint8 | 100,000 | 0.0169 ms | not_measured | managed | 1.757x |
| `a ^ b (uint8)` | uint8 | 10,000,000 | 4.3137 ms | not_measured | managed | 1.532x |
| `a | b (bool)` | bool | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `a | b (bool)` | bool | 100,000 | 0.0571 ms | not_measured | managed | 0.054x |
| `a | b (bool)` | bool | 10,000,000 | 5.5658 ms | not_measured | managed | 0.415x |
| `a | b (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a | b (int16)` | int16 | 100,000 | 0.0289 ms | not_measured | managed | 1.035x |
| `a | b (int16)` | int16 | 10,000,000 | 6.4875 ms | not_measured | managed | 1.413x |
| `a | b (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `a | b (int32)` | int32 | 100,000 | 0.0579 ms | not_measured | managed | 0.516x |
| `a | b (int32)` | int32 | 10,000,000 | 12.7017 ms | not_measured | managed | 1.313x |
| `a | b (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.242x |
| `a | b (int64)` | int64 | 100,000 | 0.1219 ms | not_measured | managed | 0.292x |
| `a | b (int64)` | int64 | 10,000,000 | 30.5362 ms | not_measured | managed | 1.132x |
| `a | b (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.538x |
| `a | b (int8)` | int8 | 100,000 | 0.0160 ms | not_measured | managed | 1.962x |
| `a | b (int8)` | int8 | 10,000,000 | 3.4496 ms | not_measured | managed | 1.870x |
| `a | b (uint16)` | uint16 | 1,000 | 0.0020 ms | not_measured | managed | 0.400x |
| `a | b (uint16)` | uint16 | 100,000 | 0.0277 ms | not_measured | managed | 1.072x |
| `a | b (uint16)` | uint16 | 10,000,000 | 6.5982 ms | not_measured | managed | 1.460x |
| `a | b (uint32)` | uint32 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `a | b (uint32)` | uint32 | 100,000 | 0.0522 ms | not_measured | managed | 0.628x |
| `a | b (uint32)` | uint32 | 10,000,000 | 13.1683 ms | not_measured | managed | 1.273x |
| `a | b (uint64)` | uint64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `a | b (uint64)` | uint64 | 100,000 | 0.1082 ms | not_measured | managed | 0.322x |
| `a | b (uint64)` | uint64 | 10,000,000 | 31.3410 ms | not_measured | managed | 1.081x |
| `a | b (uint8)` | uint8 | 1,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `a | b (uint8)` | uint8 | 100,000 | 0.0159 ms | not_measured | managed | 1.943x |
| `a | b (uint8)` | uint8 | 10,000,000 | 4.5140 ms | not_measured | managed | 1.497x |
| `np.bitwise_and(a, b) (bool)` | bool | 1,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.bitwise_and(a, b) (bool)` | bool | 100,000 | 0.0579 ms | not_measured | managed | 0.057x |
| `np.bitwise_and(a, b) (bool)` | bool | 10,000,000 | 5.6002 ms | not_measured | managed | 0.456x |
| `np.bitwise_and(a, b) (int16)` | int16 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `np.bitwise_and(a, b) (int16)` | int16 | 100,000 | 0.0314 ms | not_measured | managed | 1.102x |
| `np.bitwise_and(a, b) (int16)` | int16 | 10,000,000 | 6.7229 ms | not_measured | managed | 1.417x |
| `np.bitwise_and(a, b) (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.bitwise_and(a, b) (int32)` | int32 | 100,000 | 0.0604 ms | not_measured | managed | 0.498x |
| `np.bitwise_and(a, b) (int32)` | int32 | 10,000,000 | 13.1521 ms | not_measured | managed | 1.364x |
| `np.bitwise_and(a, b) (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_and(a, b) (int64)` | int64 | 100,000 | 0.1309 ms | not_measured | managed | 0.261x |
| `np.bitwise_and(a, b) (int64)` | int64 | 10,000,000 | 27.1095 ms | not_measured | managed | 1.251x |
| `np.bitwise_and(a, b) (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `np.bitwise_and(a, b) (int8)` | int8 | 100,000 | 0.0172 ms | not_measured | managed | 1.692x |
| `np.bitwise_and(a, b) (int8)` | int8 | 10,000,000 | 3.2638 ms | not_measured | managed | 1.991x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 100,000 | 0.0300 ms | not_measured | managed | 1.040x |
| `np.bitwise_and(a, b) (uint16)` | uint16 | 10,000,000 | 7.0740 ms | not_measured | managed | 1.297x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 100,000 | 0.0525 ms | not_measured | managed | 0.602x |
| `np.bitwise_and(a, b) (uint32)` | uint32 | 10,000,000 | 12.1282 ms | not_measured | managed | 1.377x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 100,000 | 0.1133 ms | not_measured | managed | 0.311x |
| `np.bitwise_and(a, b) (uint64)` | uint64 | 10,000,000 | 43.9084 ms | not_measured | managed | 0.777x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 1,000 | 0.0021 ms | not_measured | managed | 0.333x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 100,000 | 0.0156 ms | not_measured | managed | 1.846x |
| `np.bitwise_and(a, b) (uint8)` | uint8 | 10,000,000 | 3.2311 ms | not_measured | managed | 2.005x |
| `np.bitwise_not(a) (bool)` | bool | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.bitwise_not(a) (bool)` | bool | 100,000 | 0.0591 ms | not_measured | managed | 0.046x |
| `np.bitwise_not(a) (bool)` | bool | 10,000,000 | 5.4267 ms | not_measured | managed | 0.670x |
| `np.bitwise_not(a) (int16)` | int16 | 1,000 | 0.0019 ms | not_measured | managed | 0.368x |
| `np.bitwise_not(a) (int16)` | int16 | 100,000 | 0.0281 ms | not_measured | managed | 0.964x |
| `np.bitwise_not(a) (int16)` | int16 | 10,000,000 | 4.8423 ms | not_measured | managed | 1.524x |
| `np.bitwise_not(a) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `np.bitwise_not(a) (int32)` | int32 | 100,000 | 0.0589 ms | not_measured | managed | 0.477x |
| `np.bitwise_not(a) (int32)` | int32 | 10,000,000 | 9.2474 ms | not_measured | managed | 1.525x |
| `np.bitwise_not(a) (int64)` | int64 | 1,000 | 0.0025 ms | not_measured | managed | 0.320x |
| `np.bitwise_not(a) (int64)` | int64 | 100,000 | 0.1051 ms | not_measured | managed | 0.269x |
| `np.bitwise_not(a) (int64)` | int64 | 10,000,000 | 20.6610 ms | not_measured | managed | 1.234x |
| `np.bitwise_not(a) (int8)` | int8 | 1,000 | 0.0025 ms | not_measured | managed | 0.280x |
| `np.bitwise_not(a) (int8)` | int8 | 100,000 | 0.0166 ms | not_measured | managed | 2.524x |
| `np.bitwise_not(a) (int8)` | int8 | 10,000,000 | 2.0726 ms | not_measured | managed | 2.669x |
| `np.bitwise_not(a) (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.500x |
| `np.bitwise_not(a) (uint16)` | uint16 | 100,000 | 0.0322 ms | not_measured | managed | 0.826x |
| `np.bitwise_not(a) (uint16)` | uint16 | 10,000,000 | 5.9274 ms | not_measured | managed | 1.195x |
| `np.bitwise_not(a) (uint32)` | uint32 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.bitwise_not(a) (uint32)` | uint32 | 100,000 | 0.0582 ms | not_measured | managed | 0.455x |
| `np.bitwise_not(a) (uint32)` | uint32 | 10,000,000 | 8.8725 ms | not_measured | managed | 1.587x |
| `np.bitwise_not(a) (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.bitwise_not(a) (uint64)` | uint64 | 100,000 | 0.1051 ms | not_measured | managed | 0.251x |
| `np.bitwise_not(a) (uint64)` | uint64 | 10,000,000 | 25.8994 ms | not_measured | managed | 0.973x |
| `np.bitwise_not(a) (uint8)` | uint8 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.bitwise_not(a) (uint8)` | uint8 | 100,000 | 0.0147 ms | not_measured | managed | 1.823x |
| `np.bitwise_not(a) (uint8)` | uint8 | 10,000,000 | 2.0881 ms | not_measured | managed | 2.643x |
| `np.bitwise_or(a, b) (bool)` | bool | 1,000 | 0.0025 ms | not_measured | managed | 0.160x |
| `np.bitwise_or(a, b) (bool)` | bool | 100,000 | 0.0596 ms | not_measured | managed | 0.050x |
| `np.bitwise_or(a, b) (bool)` | bool | 10,000,000 | 5.8893 ms | not_measured | managed | 0.793x |
| `np.bitwise_or(a, b) (int16)` | int16 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `np.bitwise_or(a, b) (int16)` | int16 | 100,000 | 0.0319 ms | not_measured | managed | 0.937x |
| `np.bitwise_or(a, b) (int16)` | int16 | 10,000,000 | 6.4807 ms | not_measured | managed | 1.448x |
| `np.bitwise_or(a, b) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.409x |
| `np.bitwise_or(a, b) (int32)` | int32 | 100,000 | 0.0510 ms | not_measured | managed | 0.600x |
| `np.bitwise_or(a, b) (int32)` | int32 | 10,000,000 | 12.8907 ms | not_measured | managed | 1.391x |
| `np.bitwise_or(a, b) (int64)` | int64 | 1,000 | 0.0030 ms | not_measured | managed | 0.267x |
| `np.bitwise_or(a, b) (int64)` | int64 | 100,000 | 0.1060 ms | not_measured | managed | 0.322x |
| `np.bitwise_or(a, b) (int64)` | int64 | 10,000,000 | 28.7835 ms | not_measured | managed | 1.154x |
| `np.bitwise_or(a, b) (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `np.bitwise_or(a, b) (int8)` | int8 | 100,000 | 0.0175 ms | not_measured | managed | 1.737x |
| `np.bitwise_or(a, b) (int8)` | int8 | 10,000,000 | 3.7679 ms | not_measured | managed | 1.726x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 100,000 | 0.0278 ms | not_measured | managed | 1.090x |
| `np.bitwise_or(a, b) (uint16)` | uint16 | 10,000,000 | 8.4491 ms | not_measured | managed | 1.082x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 1,000 | 0.0022 ms | not_measured | managed | 0.364x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 100,000 | 0.0550 ms | not_measured | managed | 0.538x |
| `np.bitwise_or(a, b) (uint32)` | uint32 | 10,000,000 | 11.6152 ms | not_measured | managed | 1.472x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 100,000 | 0.1139 ms | not_measured | managed | 0.297x |
| `np.bitwise_or(a, b) (uint64)` | uint64 | 10,000,000 | 38.2804 ms | not_measured | managed | 0.885x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 1,000 | 0.0024 ms | not_measured | managed | 0.292x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 100,000 | 0.0149 ms | not_measured | managed | 1.960x |
| `np.bitwise_or(a, b) (uint8)` | uint8 | 10,000,000 | 2.8426 ms | not_measured | managed | 2.281x |
| `np.bitwise_xor(a, b) (bool)` | bool | 1,000 | 0.0031 ms | not_measured | managed | 0.129x |
| `np.bitwise_xor(a, b) (bool)` | bool | 100,000 | 0.0590 ms | not_measured | managed | 0.054x |
| `np.bitwise_xor(a, b) (bool)` | bool | 10,000,000 | 5.7041 ms | not_measured | managed | 0.845x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 100,000 | 0.0286 ms | not_measured | managed | 1.077x |
| `np.bitwise_xor(a, b) (int16)` | int16 | 10,000,000 | 6.9091 ms | not_measured | managed | 1.315x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 100,000 | 0.0552 ms | not_measured | managed | 0.582x |
| `np.bitwise_xor(a, b) (int32)` | int32 | 10,000,000 | 12.3360 ms | not_measured | managed | 1.431x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 100,000 | 0.1113 ms | not_measured | managed | 0.365x |
| `np.bitwise_xor(a, b) (int64)` | int64 | 10,000,000 | 27.4588 ms | not_measured | managed | 1.237x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 1,000 | 0.0015 ms | not_measured | managed | 0.467x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 100,000 | 0.0159 ms | not_measured | managed | 1.969x |
| `np.bitwise_xor(a, b) (int8)` | int8 | 10,000,000 | 3.2968 ms | not_measured | managed | 1.970x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 1,000 | 0.0018 ms | not_measured | managed | 0.444x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 100,000 | 0.0294 ms | not_measured | managed | 1.020x |
| `np.bitwise_xor(a, b) (uint16)` | uint16 | 10,000,000 | 7.3591 ms | not_measured | managed | 1.259x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 0.667x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 100,000 | 0.0540 ms | not_measured | managed | 0.557x |
| `np.bitwise_xor(a, b) (uint32)` | uint32 | 10,000,000 | 12.0062 ms | not_measured | managed | 1.501x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 1,000 | 0.0027 ms | not_measured | managed | 0.296x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 100,000 | 0.1085 ms | not_measured | managed | 0.332x |
| `np.bitwise_xor(a, b) (uint64)` | uint64 | 10,000,000 | 36.0222 ms | not_measured | managed | 0.915x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.389x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 100,000 | 0.0155 ms | not_measured | managed | 1.890x |
| `np.bitwise_xor(a, b) (uint8)` | uint8 | 10,000,000 | 3.1463 ms | not_measured | managed | 1.999x |
| `np.invert(a) (bool)` | bool | 1,000 | 0.0017 ms | not_measured | managed | 0.235x |
| `np.invert(a) (bool)` | bool | 100,000 | 0.0564 ms | not_measured | managed | 0.050x |
| `np.invert(a) (bool)` | bool | 10,000,000 | 5.6450 ms | not_measured | managed | 0.332x |
| `np.invert(a) (int16)` | int16 | 1,000 | 0.0016 ms | not_measured | managed | 0.438x |
| `np.invert(a) (int16)` | int16 | 100,000 | 0.0282 ms | not_measured | managed | 0.950x |
| `np.invert(a) (int16)` | int16 | 10,000,000 | 5.4051 ms | not_measured | managed | 1.332x |
| `np.invert(a) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.381x |
| `np.invert(a) (int32)` | int32 | 100,000 | 0.0578 ms | not_measured | managed | 0.528x |
| `np.invert(a) (int32)` | int32 | 10,000,000 | 9.7582 ms | not_measured | managed | 1.344x |
| `np.invert(a) (int64)` | int64 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `np.invert(a) (int64)` | int64 | 100,000 | 0.1037 ms | not_measured | managed | 0.268x |
| `np.invert(a) (int64)` | int64 | 10,000,000 | 20.4604 ms | not_measured | managed | 1.267x |
| `np.invert(a) (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.500x |
| `np.invert(a) (int8)` | int8 | 100,000 | 0.0142 ms | not_measured | managed | 1.845x |
| `np.invert(a) (int8)` | int8 | 10,000,000 | 2.6882 ms | not_measured | managed | 2.049x |
| `np.invert(a) (uint16)` | uint16 | 1,000 | 0.0017 ms | not_measured | managed | 0.412x |
| `np.invert(a) (uint16)` | uint16 | 100,000 | 0.0274 ms | not_measured | managed | 0.978x |
| `np.invert(a) (uint16)` | uint16 | 10,000,000 | 4.6254 ms | not_measured | managed | 1.529x |
| `np.invert(a) (uint32)` | uint32 | 1,000 | 0.0019 ms | not_measured | managed | 0.421x |
| `np.invert(a) (uint32)` | uint32 | 100,000 | 0.0537 ms | not_measured | managed | 0.521x |
| `np.invert(a) (uint32)` | uint32 | 10,000,000 | 9.0231 ms | not_measured | managed | 1.483x |
| `np.invert(a) (uint64)` | uint64 | 1,000 | 0.0029 ms | not_measured | managed | 0.276x |
| `np.invert(a) (uint64)` | uint64 | 100,000 | 0.1084 ms | not_measured | managed | 0.247x |
| `np.invert(a) (uint64)` | uint64 | 10,000,000 | 22.4604 ms | not_measured | managed | 1.221x |
| `np.invert(a) (uint8)` | uint8 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.invert(a) (uint8)` | uint8 | 100,000 | 0.0180 ms | not_measured | managed | 1.489x |
| `np.invert(a) (uint8)` | uint8 | 10,000,000 | 2.7357 ms | not_measured | managed | 2.054x |
| `np.left_shift(a, 2) (bool)` | bool | 1,000 | 0.0077 ms | not_measured | managed | 0.195x |
| `np.left_shift(a, 2) (bool)` | bool | 100,000 | 0.1411 ms | not_measured | managed | 0.311x |
| `np.left_shift(a, 2) (bool)` | bool | 10,000,000 | 16.2429 ms | not_measured | managed | 1.009x |
| `np.left_shift(a, 2) (int16)` | int16 | 1,000 | 0.0046 ms | not_measured | managed | 0.239x |
| `np.left_shift(a, 2) (int16)` | int16 | 100,000 | 0.0281 ms | not_measured | managed | 1.025x |
| `np.left_shift(a, 2) (int16)` | int16 | 10,000,000 | 5.1364 ms | not_measured | managed | 1.450x |
| `np.left_shift(a, 2) (int32)` | int32 | 1,000 | 0.0043 ms | not_measured | managed | 0.256x |
| `np.left_shift(a, 2) (int32)` | int32 | 100,000 | 0.0605 ms | not_measured | managed | 0.350x |
| `np.left_shift(a, 2) (int32)` | int32 | 10,000,000 | 8.2502 ms | not_measured | managed | 1.623x |
| `np.left_shift(a, 2) (int64)` | int64 | 1,000 | 0.0047 ms | not_measured | managed | 0.213x |
| `np.left_shift(a, 2) (int64)` | int64 | 100,000 | 0.1094 ms | not_measured | managed | 0.186x |
| `np.left_shift(a, 2) (int64)` | int64 | 10,000,000 | 20.2476 ms | not_measured | managed | 1.292x |
| `np.left_shift(a, 2) (int8)` | int8 | 1,000 | 0.0024 ms | not_measured | managed | 0.375x |
| `np.left_shift(a, 2) (int8)` | int8 | 100,000 | 0.0200 ms | not_measured | managed | 1.480x |
| `np.left_shift(a, 2) (int8)` | int8 | 10,000,000 | 2.8295 ms | not_measured | managed | 2.056x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0039 ms | not_measured | managed | 0.333x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0300 ms | not_measured | managed | 1.013x |
| `np.left_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.6326 ms | not_measured | managed | 1.665x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0041 ms | not_measured | managed | 0.244x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0531 ms | not_measured | managed | 0.401x |
| `np.left_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 9.7158 ms | not_measured | managed | 1.354x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0044 ms | not_measured | managed | 0.227x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.1035 ms | not_measured | managed | 0.192x |
| `np.left_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 21.6478 ms | not_measured | managed | 1.197x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0038 ms | not_measured | managed | 0.237x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0192 ms | not_measured | managed | 1.505x |
| `np.left_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 2.8602 ms | not_measured | managed | 2.062x |
| `np.right_shift(a, 2) (bool)` | bool | 1,000 | 0.0103 ms | not_measured | managed | 0.155x |
| `np.right_shift(a, 2) (bool)` | bool | 100,000 | 0.1382 ms | not_measured | managed | 0.402x |
| `np.right_shift(a, 2) (bool)` | bool | 10,000,000 | 16.1229 ms | not_measured | managed | 1.005x |
| `np.right_shift(a, 2) (int16)` | int16 | 1,000 | 0.0035 ms | not_measured | managed | 0.343x |
| `np.right_shift(a, 2) (int16)` | int16 | 100,000 | 0.0289 ms | not_measured | managed | 1.308x |
| `np.right_shift(a, 2) (int16)` | int16 | 10,000,000 | 4.9103 ms | not_measured | managed | 2.016x |
| `np.right_shift(a, 2) (int32)` | int32 | 1,000 | 0.0035 ms | not_measured | managed | 0.343x |
| `np.right_shift(a, 2) (int32)` | int32 | 100,000 | 0.0613 ms | not_measured | managed | 0.485x |
| `np.right_shift(a, 2) (int32)` | int32 | 10,000,000 | 9.1702 ms | not_measured | managed | 1.531x |
| `np.right_shift(a, 2) (int64)` | int64 | 1,000 | 0.0047 ms | not_measured | managed | 0.234x |
| `np.right_shift(a, 2) (int64)` | int64 | 100,000 | 0.1217 ms | not_measured | managed | 0.242x |
| `np.right_shift(a, 2) (int64)` | int64 | 10,000,000 | 21.1698 ms | not_measured | managed | 1.223x |
| `np.right_shift(a, 2) (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `np.right_shift(a, 2) (int8)` | int8 | 100,000 | 0.0192 ms | not_measured | managed | 1.984x |
| `np.right_shift(a, 2) (int8)` | int8 | 10,000,000 | 2.4257 ms | not_measured | managed | 3.692x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 1,000 | 0.0044 ms | not_measured | managed | 0.386x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 100,000 | 0.0283 ms | not_measured | managed | 1.088x |
| `np.right_shift(a, 2) (uint16)` | uint16 | 10,000,000 | 4.8827 ms | not_measured | managed | 1.524x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 1,000 | 0.0032 ms | not_measured | managed | 0.312x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 100,000 | 0.0523 ms | not_measured | managed | 0.390x |
| `np.right_shift(a, 2) (uint32)` | uint32 | 10,000,000 | 8.8079 ms | not_measured | managed | 1.494x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 1,000 | 0.0047 ms | not_measured | managed | 0.213x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 100,000 | 0.1036 ms | not_measured | managed | 0.187x |
| `np.right_shift(a, 2) (uint64)` | uint64 | 10,000,000 | 26.7529 ms | not_measured | managed | 0.944x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 1,000 | 0.0041 ms | not_measured | managed | 0.220x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 100,000 | 0.0211 ms | not_measured | managed | 1.351x |
| `np.right_shift(a, 2) (uint8)` | uint8 | 10,000,000 | 2.8033 ms | not_measured | managed | 2.223x |
| `matrix * 2.0` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.303x |
| `matrix * 2.0` | float64 | 100,000 | 0.1171 ms | not_measured | managed | 0.118x |
| `matrix * 2.0` | float64 | 10,000,000 | 20.2857 ms | not_measured | managed | 0.946x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 0.254x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 100,000 | 0.1503 ms | not_measured | managed | 0.194x |
| `matrix * col_vector (N,M)*(N,1)` | float64 | 10,000,000 | 20.6466 ms | not_measured | managed | 0.895x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 1,000 | 0.0057 ms | not_measured | managed | 0.228x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 100,000 | 0.1026 ms | not_measured | managed | 0.262x |
| `matrix * row_vector (N,M)*(M,)` | float64 | 10,000,000 | 21.6123 ms | not_measured | managed | 0.793x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 1,000 | 0.0065 ms | not_measured | managed | 0.215x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 100,000 | 0.1656 ms | not_measured | managed | 0.178x |
| `matrix + col_vector (N,M)+(N,1)` | float64 | 10,000,000 | 26.7054 ms | not_measured | managed | 0.672x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 1,000 | 0.0057 ms | not_measured | managed | 0.228x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 100,000 | 0.1094 ms | not_measured | managed | 0.253x |
| `matrix + row_vector (N,M)+(M,)` | float64 | 10,000,000 | 20.7618 ms | not_measured | managed | 0.993x |
| `matrix + scalar` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.296x |
| `matrix + scalar` | float64 | 100,000 | 0.0974 ms | not_measured | managed | 0.151x |
| `matrix + scalar` | float64 | 10,000,000 | 20.7957 ms | not_measured | managed | 0.841x |
| `np.broadcast_to(col, (N,M))` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.429x |
| `np.broadcast_to(col, (N,M))` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 2.000x |
| `np.broadcast_to(col, (N,M))` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 1.818x |
| `np.broadcast_to(row, (N,M))` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.500x |
| `np.broadcast_to(row, (N,M))` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 1.667x |
| `np.broadcast_to(row, (N,M))` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 7.000x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 1,000 | 0.0073 ms | not_measured | managed | 0.164x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 100,000 | 0.1491 ms | not_measured | managed | 0.195x |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | float64 | 10,000,000 | 22.2680 ms | not_measured | managed | 0.859x |
| `a != b (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a != b (float32)` | float32 | 100,000 | 0.0200 ms | not_measured | managed | 0.330x |
| `a != b (float32)` | float32 | 10,000,000 | 11.8171 ms | not_measured | managed | 0.942x |
| `a != b (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a != b (float64)` | float64 | 100,000 | 0.0306 ms | not_measured | managed | 0.392x |
| `a != b (float64)` | float64 | 10,000,000 | 19.8239 ms | not_measured | managed | 0.939x |
| `a != b (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.385x |
| `a != b (int32)` | int32 | 100,000 | 0.0210 ms | not_measured | managed | 0.395x |
| `a != b (int32)` | int32 | 10,000,000 | 10.5581 ms | not_measured | managed | 0.493x |
| `a != b (int64)` | int64 | 1,000 | 0.0027 ms | not_measured | managed | 0.222x |
| `a != b (int64)` | int64 | 100,000 | 0.0299 ms | not_measured | managed | 0.552x |
| `a != b (int64)` | int64 | 10,000,000 | 18.2666 ms | not_measured | managed | 1.087x |
| `a < b (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a < b (float32)` | float32 | 100,000 | 0.0174 ms | not_measured | managed | 0.374x |
| `a < b (float32)` | float32 | 10,000,000 | 11.8203 ms | not_measured | managed | 0.931x |
| `a < b (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `a < b (float64)` | float64 | 100,000 | 0.0310 ms | not_measured | managed | 0.365x |
| `a < b (float64)` | float64 | 10,000,000 | 18.4534 ms | not_measured | managed | 1.026x |
| `a < b (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `a < b (int32)` | int32 | 100,000 | 0.0225 ms | not_measured | managed | 0.329x |
| `a < b (int32)` | int32 | 10,000,000 | 11.4470 ms | not_measured | managed | 0.422x |
| `a < b (int64)` | int64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `a < b (int64)` | int64 | 100,000 | 0.0296 ms | not_measured | managed | 0.696x |
| `a < b (int64)` | int64 | 10,000,000 | 17.8995 ms | not_measured | managed | 1.062x |
| `a <= b (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `a <= b (float32)` | float32 | 100,000 | 0.0205 ms | not_measured | managed | 0.307x |
| `a <= b (float32)` | float32 | 10,000,000 | 10.3600 ms | not_measured | managed | 1.026x |
| `a <= b (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `a <= b (float64)` | float64 | 100,000 | 0.0314 ms | not_measured | managed | 0.742x |
| `a <= b (float64)` | float64 | 10,000,000 | 20.4372 ms | not_measured | managed | 0.898x |
| `a <= b (int32)` | int32 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `a <= b (int32)` | int32 | 100,000 | 0.0252 ms | not_measured | managed | 0.313x |
| `a <= b (int32)` | int32 | 10,000,000 | 10.0360 ms | not_measured | managed | 0.494x |
| `a <= b (int64)` | int64 | 1,000 | 0.0031 ms | not_measured | managed | 0.194x |
| `a <= b (int64)` | int64 | 100,000 | 0.0332 ms | not_measured | managed | 0.572x |
| `a <= b (int64)` | int64 | 10,000,000 | 17.9054 ms | not_measured | managed | 1.116x |
| `a == b (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `a == b (float32)` | float32 | 100,000 | 0.0208 ms | not_measured | managed | 0.308x |
| `a == b (float32)` | float32 | 10,000,000 | 11.9641 ms | not_measured | managed | 0.919x |
| `a == b (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a == b (float64)` | float64 | 100,000 | 0.0322 ms | not_measured | managed | 0.425x |
| `a == b (float64)` | float64 | 10,000,000 | 19.3072 ms | not_measured | managed | 0.981x |
| `a == b (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.278x |
| `a == b (int32)` | int32 | 100,000 | 0.0258 ms | not_measured | managed | 0.295x |
| `a == b (int32)` | int32 | 10,000,000 | 9.6938 ms | not_measured | managed | 0.545x |
| `a == b (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `a == b (int64)` | int64 | 100,000 | 0.0299 ms | not_measured | managed | 0.495x |
| `a == b (int64)` | int64 | 10,000,000 | 18.0719 ms | not_measured | managed | 1.062x |
| `a > b (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `a > b (float32)` | float32 | 100,000 | 0.0190 ms | not_measured | managed | 0.321x |
| `a > b (float32)` | float32 | 10,000,000 | 11.7163 ms | not_measured | managed | 0.921x |
| `a > b (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `a > b (float64)` | float64 | 100,000 | 0.0318 ms | not_measured | managed | 0.506x |
| `a > b (float64)` | float64 | 10,000,000 | 20.3535 ms | not_measured | managed | 0.913x |
| `a > b (int32)` | int32 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `a > b (int32)` | int32 | 100,000 | 0.0236 ms | not_measured | managed | 0.322x |
| `a > b (int32)` | int32 | 10,000,000 | 10.4530 ms | not_measured | managed | 0.463x |
| `a > b (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `a > b (int64)` | int64 | 100,000 | 0.0300 ms | not_measured | managed | 0.670x |
| `a > b (int64)` | int64 | 10,000,000 | 17.7827 ms | not_measured | managed | 1.069x |
| `a >= b (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `a >= b (float32)` | float32 | 100,000 | 0.0199 ms | not_measured | managed | 0.312x |
| `a >= b (float32)` | float32 | 10,000,000 | 9.9757 ms | not_measured | managed | 1.048x |
| `a >= b (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `a >= b (float64)` | float64 | 100,000 | 0.0287 ms | not_measured | managed | 0.833x |
| `a >= b (float64)` | float64 | 10,000,000 | 18.2340 ms | not_measured | managed | 1.003x |
| `a >= b (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `a >= b (int32)` | int32 | 100,000 | 0.0223 ms | not_measured | managed | 0.534x |
| `a >= b (int32)` | int32 | 10,000,000 | 10.1032 ms | not_measured | managed | 0.473x |
| `a >= b (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `a >= b (int64)` | int64 | 100,000 | 0.0319 ms | not_measured | managed | 0.596x |
| `a >= b (int64)` | int64 | 10,000,000 | 18.5932 ms | not_measured | managed | 1.023x |
| `np.equal(a, b) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.equal(a, b) (float32)` | float32 | 100,000 | 0.0187 ms | not_measured | managed | 0.326x |
| `np.equal(a, b) (float32)` | float32 | 10,000,000 | 10.6466 ms | not_measured | managed | 1.034x |
| `np.equal(a, b) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 0.280x |
| `np.equal(a, b) (float64)` | float64 | 100,000 | 0.0298 ms | not_measured | managed | 0.466x |
| `np.equal(a, b) (float64)` | float64 | 10,000,000 | 18.0829 ms | not_measured | managed | 1.039x |
| `np.equal(a, b) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.equal(a, b) (int32)` | int32 | 100,000 | 0.0193 ms | not_measured | managed | 0.606x |
| `np.equal(a, b) (int32)` | int32 | 10,000,000 | 10.2628 ms | not_measured | managed | 0.437x |
| `np.equal(a, b) (int64)` | int64 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.equal(a, b) (int64)` | int64 | 100,000 | 0.0321 ms | not_measured | managed | 0.439x |
| `np.equal(a, b) (int64)` | int64 | 10,000,000 | 20.1780 ms | not_measured | managed | 0.924x |
| `np.greater(a, b) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.294x |
| `np.greater(a, b) (float32)` | float32 | 100,000 | 0.0185 ms | not_measured | managed | 0.335x |
| `np.greater(a, b) (float32)` | float32 | 10,000,000 | 9.4059 ms | not_measured | managed | 1.229x |
| `np.greater(a, b) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.greater(a, b) (float64)` | float64 | 100,000 | 0.0292 ms | not_measured | managed | 0.507x |
| `np.greater(a, b) (float64)` | float64 | 10,000,000 | 20.9399 ms | not_measured | managed | 0.921x |
| `np.greater(a, b) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.greater(a, b) (int32)` | int32 | 100,000 | 0.0184 ms | not_measured | managed | 0.516x |
| `np.greater(a, b) (int32)` | int32 | 10,000,000 | 9.6072 ms | not_measured | managed | 1.099x |
| `np.greater(a, b) (int64)` | int64 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.greater(a, b) (int64)` | int64 | 100,000 | 0.0284 ms | not_measured | managed | 0.754x |
| `np.greater(a, b) (int64)` | int64 | 10,000,000 | 19.7592 ms | not_measured | managed | 0.961x |
| `np.greater_equal(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.greater_equal(a, b) (float32)` | float32 | 100,000 | 0.0247 ms | not_measured | managed | 0.247x |
| `np.greater_equal(a, b) (float32)` | float32 | 10,000,000 | 10.0384 ms | not_measured | managed | 1.038x |
| `np.greater_equal(a, b) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.179x |
| `np.greater_equal(a, b) (float64)` | float64 | 100,000 | 0.0284 ms | not_measured | managed | 0.461x |
| `np.greater_equal(a, b) (float64)` | float64 | 10,000,000 | 18.0184 ms | not_measured | managed | 1.026x |
| `np.greater_equal(a, b) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.greater_equal(a, b) (int32)` | int32 | 100,000 | 0.0181 ms | not_measured | managed | 0.569x |
| `np.greater_equal(a, b) (int32)` | int32 | 10,000,000 | 11.8380 ms | not_measured | managed | 0.925x |
| `np.greater_equal(a, b) (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.273x |
| `np.greater_equal(a, b) (int64)` | int64 | 100,000 | 0.0308 ms | not_measured | managed | 0.656x |
| `np.greater_equal(a, b) (int64)` | int64 | 10,000,000 | 20.0608 ms | not_measured | managed | 0.962x |
| `np.less(a, b) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.less(a, b) (float32)` | float32 | 100,000 | 0.0189 ms | not_measured | managed | 0.339x |
| `np.less(a, b) (float32)` | float32 | 10,000,000 | 9.6447 ms | not_measured | managed | 1.165x |
| `np.less(a, b) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.less(a, b) (float64)` | float64 | 100,000 | 0.0314 ms | not_measured | managed | 0.459x |
| `np.less(a, b) (float64)` | float64 | 10,000,000 | 20.9613 ms | not_measured | managed | 0.888x |
| `np.less(a, b) (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `np.less(a, b) (int32)` | int32 | 100,000 | 0.0181 ms | not_measured | managed | 0.580x |
| `np.less(a, b) (int32)` | int32 | 10,000,000 | 10.5430 ms | not_measured | managed | 0.429x |
| `np.less(a, b) (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `np.less(a, b) (int64)` | int64 | 100,000 | 0.0277 ms | not_measured | managed | 0.751x |
| `np.less(a, b) (int64)` | int64 | 10,000,000 | 21.2633 ms | not_measured | managed | 0.884x |
| `np.less_equal(a, b) (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.less_equal(a, b) (float32)` | float32 | 100,000 | 0.0227 ms | not_measured | managed | 0.286x |
| `np.less_equal(a, b) (float32)` | float32 | 10,000,000 | 9.6309 ms | not_measured | managed | 1.184x |
| `np.less_equal(a, b) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.less_equal(a, b) (float64)` | float64 | 100,000 | 0.0287 ms | not_measured | managed | 0.443x |
| `np.less_equal(a, b) (float64)` | float64 | 10,000,000 | 21.5897 ms | not_measured | managed | 0.872x |
| `np.less_equal(a, b) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.263x |
| `np.less_equal(a, b) (int32)` | int32 | 100,000 | 0.0187 ms | not_measured | managed | 0.695x |
| `np.less_equal(a, b) (int32)` | int32 | 10,000,000 | 10.0708 ms | not_measured | managed | 1.083x |
| `np.less_equal(a, b) (int64)` | int64 | 1,000 | 0.0023 ms | not_measured | managed | 0.261x |
| `np.less_equal(a, b) (int64)` | int64 | 100,000 | 0.0325 ms | not_measured | managed | 0.618x |
| `np.less_equal(a, b) (int64)` | int64 | 10,000,000 | 19.0234 ms | not_measured | managed | 0.993x |
| `np.not_equal(a, b) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.not_equal(a, b) (float32)` | float32 | 100,000 | 0.0180 ms | not_measured | managed | 0.367x |
| `np.not_equal(a, b) (float32)` | float32 | 10,000,000 | 10.4045 ms | not_measured | managed | 1.051x |
| `np.not_equal(a, b) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 0.200x |
| `np.not_equal(a, b) (float64)` | float64 | 100,000 | 0.0299 ms | not_measured | managed | 0.441x |
| `np.not_equal(a, b) (float64)` | float64 | 10,000,000 | 19.2934 ms | not_measured | managed | 0.964x |
| `np.not_equal(a, b) (int32)` | int32 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.not_equal(a, b) (int32)` | int32 | 100,000 | 0.0179 ms | not_measured | managed | 1.318x |
| `np.not_equal(a, b) (int32)` | int32 | 10,000,000 | 10.6374 ms | not_measured | managed | 0.438x |
| `np.not_equal(a, b) (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.not_equal(a, b) (int64)` | int64 | 100,000 | 0.0304 ms | not_measured | managed | 0.464x |
| `np.not_equal(a, b) (int64)` | int64 | 10,000,000 | 21.1925 ms | not_measured | managed | 0.898x |
| `a.copy() (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.148x |
| `a.copy() (float32)` | float32 | 100,000 | 0.0128 ms | not_measured | managed | 0.477x |
| `a.copy() (float32)` | float32 | 10,000,000 | 5.7953 ms | not_measured | managed | 1.912x |
| `a.copy() (float64)` | float64 | 1,000 | 0.0105 ms | not_measured | managed | 0.038x |
| `a.copy() (float64)` | float64 | 100,000 | 0.0266 ms | not_measured | managed | 0.801x |
| `a.copy() (float64)` | float64 | 10,000,000 | 17.0494 ms | not_measured | managed | 1.699x |
| `a.copy() (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.071x |
| `a.copy() (int32)` | int32 | 100,000 | 0.0143 ms | not_measured | managed | 0.413x |
| `a.copy() (int32)` | int32 | 10,000,000 | 5.2267 ms | not_measured | managed | 1.505x |
| `a.copy() (int64)` | int64 | 1,000 | 0.0028 ms | not_measured | managed | 0.179x |
| `a.copy() (int64)` | int64 | 100,000 | 0.0244 ms | not_measured | managed | 1.004x |
| `a.copy() (int64)` | int64 | 10,000,000 | 17.2795 ms | not_measured | managed | 0.777x |
| `np.arange(N) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `np.arange(N) (float32)` | float32 | 100,000 | 0.0578 ms | not_measured | managed | 0.746x |
| `np.arange(N) (float32)` | float32 | 10,000,000 | 5.8966 ms | not_measured | managed | 1.797x |
| `np.arange(N) (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.471x |
| `np.arange(N) (float64)` | float64 | 100,000 | 0.0564 ms | not_measured | managed | 0.509x |
| `np.arange(N) (float64)` | float64 | 10,000,000 | 16.2531 ms | not_measured | managed | 1.154x |
| `np.arange(N) (int32)` | int32 | 1,000 | 0.0054 ms | not_measured | managed | 0.148x |
| `np.arange(N) (int32)` | int32 | 100,000 | 0.0575 ms | not_measured | managed | 0.525x |
| `np.arange(N) (int32)` | int32 | 10,000,000 | 5.6977 ms | not_measured | managed | 1.934x |
| `np.arange(N) (int64)` | int64 | 1,000 | 0.0023 ms | not_measured | managed | 0.304x |
| `np.arange(N) (int64)` | int64 | 100,000 | 0.0594 ms | not_measured | managed | 0.338x |
| `np.arange(N) (int64)` | int64 | 10,000,000 | 28.6929 ms | not_measured | managed | 0.448x |
| `np.array(a) (float64)` | float64 | 1,000 | 0.0104 ms | not_measured | managed | 0.038x |
| `np.array(a) (float64)` | float64 | 100,000 | 0.1170 ms | not_measured | managed | 0.206x |
| `np.asanyarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asanyarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.asarray_chkfinite(a) (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 14.000x |
| `np.asarray_chkfinite(a) (float64)` | float64 | 100,000 | 0.0122 ms | not_measured | managed | 1.049x |
| `np.ascontiguousarray(a) (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.091x |
| `np.ascontiguousarray(a) (float64)` | float64 | 100,000 | 0.0670 ms | not_measured | managed | 0.178x |
| `np.asfortranarray(a) (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.191x |
| `np.asfortranarray(a) (float64)` | float64 | 100,000 | 0.2357 ms | not_measured | managed | 0.186x |
| `np.asmatrix(a) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 2.714x |
| `np.asmatrix(a) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 2.200x |
| `np.copy (float32)` | float32 | 1,000 | 0.0078 ms | not_measured | managed | 0.103x |
| `np.copy (float32)` | float32 | 100,000 | 0.0139 ms | not_measured | managed | 0.482x |
| `np.copy (float32)` | float32 | 10,000,000 | 5.9584 ms | not_measured | managed | 1.619x |
| `np.copy (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.092x |
| `np.copy (float64)` | float64 | 100,000 | 0.0255 ms | not_measured | managed | 0.859x |
| `np.copy (float64)` | float64 | 10,000,000 | 17.0477 ms | not_measured | managed | 1.249x |
| `np.copy (int32)` | int32 | 1,000 | 0.0078 ms | not_measured | managed | 0.077x |
| `np.copy (int32)` | int32 | 100,000 | 0.0137 ms | not_measured | managed | 0.460x |
| `np.copy (int32)` | int32 | 10,000,000 | 5.1975 ms | not_measured | managed | 1.601x |
| `np.copy (int64)` | int64 | 1,000 | 0.0134 ms | not_measured | managed | 0.052x |
| `np.copy (int64)` | int64 | 100,000 | 0.0276 ms | not_measured | managed | 0.931x |
| `np.copy (int64)` | int64 | 10,000,000 | 18.5321 ms | not_measured | managed | 0.770x |
| `np.empty (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `np.empty (float32)` | float32 | 100,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.empty (float32)` | float32 | 10,000,000 | 0.0011 ms | not_measured | managed | 18.091x |
| `np.empty (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.273x |
| `np.empty (float64)` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 0.250x |
| `np.empty (float64)` | float64 | 10,000,000 | 0.0233 ms | not_measured | managed | 0.824x |
| `np.empty (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.273x |
| `np.empty (int32)` | int32 | 100,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.empty (int32)` | int32 | 10,000,000 | 0.0013 ms | not_measured | managed | 8.462x |
| `np.empty (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 0.231x |
| `np.empty (int64)` | int64 | 100,000 | 0.0014 ms | not_measured | managed | 0.214x |
| `np.empty (int64)` | int64 | 10,000,000 | 0.0225 ms | not_measured | managed | 0.516x |
| `np.empty_like(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.286x |
| `np.empty_like(a) (float32)` | float32 | 100,000 | 0.0068 ms | not_measured | managed | 0.059x |
| `np.empty_like(a) (float32)` | float32 | 10,000,000 | 0.0065 ms | not_measured | managed | 3.015x |
| `np.empty_like(a) (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `np.empty_like(a) (float64)` | float64 | 100,000 | 0.0028 ms | not_measured | managed | 0.143x |
| `np.empty_like(a) (float64)` | float64 | 10,000,000 | 0.0246 ms | not_measured | managed | 0.809x |
| `np.empty_like(a) (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.211x |
| `np.empty_like(a) (int32)` | int32 | 100,000 | 0.0052 ms | not_measured | managed | 0.077x |
| `np.empty_like(a) (int32)` | int32 | 10,000,000 | 0.0016 ms | not_measured | managed | 6.812x |
| `np.empty_like(a) (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.empty_like(a) (int64)` | int64 | 100,000 | 0.0018 ms | not_measured | managed | 0.222x |
| `np.empty_like(a) (int64)` | int64 | 10,000,000 | 0.0225 ms | not_measured | managed | 0.889x |
| `np.eye(n) (float64)` | float64 | 1,000 | 0.0098 ms | not_measured | managed | 0.143x |
| `np.eye(n) (float64)` | float64 | 100,000 | 0.0931 ms | not_measured | managed | 0.129x |
| `np.frombuffer(buffer) (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.231x |
| `np.frombuffer(buffer) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.fromstring(text) (float64)` | float64 | 1,000 | 0.0679 ms | not_measured | managed | 1.454x |
| `np.fromstring(text) (float64)` | float64 | 100,000 | 14.4034 ms | not_measured | managed | 0.670x |
| `np.full (float32)` | float32 | 1,000 | 0.0028 ms | not_measured | managed | 0.321x |
| `np.full (float32)` | float32 | 100,000 | 0.0221 ms | not_measured | managed | 0.258x |
| `np.full (float32)` | float32 | 10,000,000 | 4.9181 ms | not_measured | managed | 1.883x |
| `np.full (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.647x |
| `np.full (float64)` | float64 | 100,000 | 0.0222 ms | not_measured | managed | 0.468x |
| `np.full (float64)` | float64 | 10,000,000 | 17.0292 ms | not_measured | managed | 1.177x |
| `np.full (int32)` | int32 | 1,000 | 0.0018 ms | not_measured | managed | 0.556x |
| `np.full (int32)` | int32 | 100,000 | 0.0204 ms | not_measured | managed | 0.294x |
| `np.full (int32)` | int32 | 10,000,000 | 4.4706 ms | not_measured | managed | 2.010x |
| `np.full (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `np.full (int64)` | int64 | 100,000 | 0.0259 ms | not_measured | managed | 0.394x |
| `np.full (int64)` | int64 | 10,000,000 | 17.1537 ms | not_measured | managed | 0.858x |
| `np.full_like(a, 42) (float32)` | float32 | 1,000 | 0.0029 ms | not_measured | managed | 0.379x |
| `np.full_like(a, 42) (float32)` | float32 | 100,000 | 0.0211 ms | not_measured | managed | 0.270x |
| `np.full_like(a, 42) (float32)` | float32 | 10,000,000 | 4.8007 ms | not_measured | managed | 1.904x |
| `np.full_like(a, 42) (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.733x |
| `np.full_like(a, 42) (float64)` | float64 | 100,000 | 0.0220 ms | not_measured | managed | 0.477x |
| `np.full_like(a, 42) (float64)` | float64 | 10,000,000 | 17.2611 ms | not_measured | managed | 1.113x |
| `np.full_like(a, 42) (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.647x |
| `np.full_like(a, 42) (int32)` | int32 | 100,000 | 0.0205 ms | not_measured | managed | 0.293x |
| `np.full_like(a, 42) (int32)` | int32 | 10,000,000 | 5.3460 ms | not_measured | managed | 1.398x |
| `np.full_like(a, 42) (int64)` | int64 | 1,000 | 0.0055 ms | not_measured | managed | 0.200x |
| `np.full_like(a, 42) (int64)` | int64 | 100,000 | 0.0219 ms | not_measured | managed | 0.466x |
| `np.full_like(a, 42) (int64)` | int64 | 10,000,000 | 16.4409 ms | not_measured | managed | 1.152x |
| `np.identity(n) (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 0.895x |
| `np.identity(n) (float64)` | float64 | 100,000 | 0.0834 ms | not_measured | managed | 0.145x |
| `np.linspace(0, N, N) (float32)` | float32 | 1,000 | 0.0045 ms | not_measured | managed | 1.178x |
| `np.linspace(0, N, N) (float32)` | float32 | 100,000 | 0.1956 ms | not_measured | managed | 1.476x |
| `np.linspace(0, N, N) (float32)` | float32 | 10,000,000 | 19.0131 ms | not_measured | managed | 2.994x |
| `np.linspace(0, N, N) (float64)` | float64 | 1,000 | 0.0057 ms | not_measured | managed | 0.825x |
| `np.linspace(0, N, N) (float64)` | float64 | 100,000 | 0.1932 ms | not_measured | managed | 0.300x |
| `np.linspace(0, N, N) (float64)` | float64 | 10,000,000 | 25.2390 ms | not_measured | managed | 1.562x |
| `np.linspace(0, N, N) (int32)` | int32 | 1,000 | 0.0038 ms | not_measured | managed | 1.421x |
| `np.linspace(0, N, N) (int32)` | int32 | 100,000 | 0.3225 ms | not_measured | managed | 0.951x |
| `np.linspace(0, N, N) (int32)` | int32 | 10,000,000 | 33.5228 ms | not_measured | managed | 1.348x |
| `np.linspace(0, N, N) (int64)` | int64 | 1,000 | 0.0057 ms | not_measured | managed | 0.947x |
| `np.linspace(0, N, N) (int64)` | int64 | 100,000 | 0.3249 ms | not_measured | managed | 1.118x |
| `np.linspace(0, N, N) (int64)` | int64 | 10,000,000 | 38.7834 ms | not_measured | managed | 1.313x |
| `np.meshgrid(x, y) (float64)` | float64 | 1,000 | 0.0138 ms | not_measured | managed | 0.667x |
| `np.meshgrid(x, y) (float64)` | float64 | 100,000 | 0.2155 ms | not_measured | managed | 1.710x |
| `np.ones (float32)` | float32 | 1,000 | 0.0028 ms | not_measured | managed | 0.429x |
| `np.ones (float32)` | float32 | 100,000 | 0.0209 ms | not_measured | managed | 0.263x |
| `np.ones (float32)` | float32 | 10,000,000 | 4.4204 ms | not_measured | managed | 2.224x |
| `np.ones (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.ones (float64)` | float64 | 100,000 | 0.0225 ms | not_measured | managed | 0.484x |
| `np.ones (float64)` | float64 | 10,000,000 | 18.7917 ms | not_measured | managed | 1.157x |
| `np.ones (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 0.588x |
| `np.ones (int32)` | int32 | 100,000 | 0.0202 ms | not_measured | managed | 0.277x |
| `np.ones (int32)` | int32 | 10,000,000 | 4.5086 ms | not_measured | managed | 2.585x |
| `np.ones (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `np.ones (int64)` | int64 | 100,000 | 0.0251 ms | not_measured | managed | 0.410x |
| `np.ones (int64)` | int64 | 10,000,000 | 16.8460 ms | not_measured | managed | 0.873x |
| `np.ones_like(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.688x |
| `np.ones_like(a) (float32)` | float32 | 100,000 | 0.0212 ms | not_measured | managed | 0.269x |
| `np.ones_like(a) (float32)` | float32 | 10,000,000 | 4.3830 ms | not_measured | managed | 2.033x |
| `np.ones_like(a) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 0.229x |
| `np.ones_like(a) (float64)` | float64 | 100,000 | 0.0221 ms | not_measured | managed | 0.507x |
| `np.ones_like(a) (float64)` | float64 | 10,000,000 | 17.4020 ms | not_measured | managed | 1.089x |
| `np.ones_like(a) (int32)` | int32 | 1,000 | 0.0047 ms | not_measured | managed | 0.234x |
| `np.ones_like(a) (int32)` | int32 | 100,000 | 0.0201 ms | not_measured | managed | 0.294x |
| `np.ones_like(a) (int32)` | int32 | 10,000,000 | 5.7821 ms | not_measured | managed | 1.399x |
| `np.ones_like(a) (int64)` | int64 | 1,000 | 0.0033 ms | not_measured | managed | 0.333x |
| `np.ones_like(a) (int64)` | int64 | 100,000 | 0.0219 ms | not_measured | managed | 0.466x |
| `np.ones_like(a) (int64)` | int64 | 10,000,000 | 17.6440 ms | not_measured | managed | 0.903x |
| `np.require(a, requirements=C) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 0.238x |
| `np.require(a, requirements=C) (float64)` | float64 | 100,000 | 0.0639 ms | not_measured | managed | 0.199x |
| `np.vander(x) (float64)` | float64 | 1,000 | 0.0438 ms | not_measured | managed | 0.100x |
| `np.vander(x) (float64)` | float64 | 100,000 | 0.3458 ms | not_measured | managed | 0.487x |
| `np.zeros (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `np.zeros (float32)` | float32 | 100,000 | 0.0053 ms | not_measured | managed | 0.981x |
| `np.zeros (float32)` | float32 | 10,000,000 | 0.0050 ms | not_measured | managed | 4.460x |
| `np.zeros (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.182x |
| `np.zeros (float64)` | float64 | 100,000 | 0.0050 ms | not_measured | managed | 1.880x |
| `np.zeros (float64)` | float64 | 10,000,000 | 0.0147 ms | not_measured | managed | 1.286x |
| `np.zeros (int32)` | int32 | 1,000 | 0.0036 ms | not_measured | managed | 0.111x |
| `np.zeros (int32)` | int32 | 100,000 | 0.0040 ms | not_measured | managed | 1.300x |
| `np.zeros (int32)` | int32 | 10,000,000 | 0.0136 ms | not_measured | managed | 0.824x |
| `np.zeros (int64)` | int64 | 1,000 | 0.0051 ms | not_measured | managed | 0.078x |
| `np.zeros (int64)` | int64 | 100,000 | 0.0051 ms | not_measured | managed | 1.941x |
| `np.zeros (int64)` | int64 | 10,000,000 | 0.0116 ms | not_measured | managed | 0.862x |
| `np.zeros_like (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.611x |
| `np.zeros_like (float32)` | float32 | 100,000 | 0.0047 ms | not_measured | managed | 1.191x |
| `np.zeros_like (float32)` | float32 | 10,000,000 | 0.0052 ms | not_measured | managed | 1880.423x |
| `np.zeros_like (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 0.143x |
| `np.zeros_like (float64)` | float64 | 100,000 | 0.0052 ms | not_measured | managed | 2.135x |
| `np.zeros_like (float64)` | float64 | 10,000,000 | 0.0140 ms | not_measured | managed | 1364.629x |
| `np.zeros_like (int32)` | int32 | 1,000 | 0.0019 ms | not_measured | managed | 0.526x |
| `np.zeros_like (int32)` | int32 | 100,000 | 0.0043 ms | not_measured | managed | 1.349x |
| `np.zeros_like (int32)` | int32 | 10,000,000 | 0.0142 ms | not_measured | managed | 540.092x |
| `np.zeros_like (int64)` | int64 | 1,000 | 0.0088 ms | not_measured | managed | 0.148x |
| `np.zeros_like (int64)` | int64 | 100,000 | 0.0055 ms | not_measured | managed | 1.873x |
| `np.zeros_like (int64)` | int64 | 10,000,000 | 0.0124 ms | not_measured | managed | 1229.798x |
| `np.fft.fft(a) (complex128)` | complex128 | 1,000 | 0.0134 ms | not_measured | managed | 0.619x |
| `np.fft.fft(a) (complex128)` | complex128 | 100,000 | 2.6356 ms | not_measured | managed | 0.633x |
| `np.fft.fft2(a) (complex128)` | complex128 | 1,000 | 0.0840 ms | not_measured | managed | 0.474x |
| `np.fft.fft2(a) (complex128)` | complex128 | 100,000 | 12.2773 ms | not_measured | managed | 0.382x |
| `np.fft.fftfreq(n) (float64)` | float64 | 1,000 | 0.0173 ms | not_measured | managed | 0.214x |
| `np.fft.fftfreq(n) (float64)` | float64 | 100,000 | 0.5204 ms | not_measured | managed | 1.107x |
| `np.fft.fftn(a) (complex128)` | complex128 | 1,000 | 0.0902 ms | not_measured | managed | 0.410x |
| `np.fft.fftn(a) (complex128)` | complex128 | 100,000 | 12.0817 ms | not_measured | managed | 0.392x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 1,000 | 0.0171 ms | not_measured | managed | 0.287x |
| `np.fft.fftshift(a) (complex128)` | complex128 | 100,000 | 0.3990 ms | not_measured | managed | 0.915x |
| `np.fft.hfft(a) (float64)` | float64 | 1,000 | 0.0144 ms | not_measured | managed | 0.500x |
| `np.fft.hfft(a) (float64)` | float64 | 100,000 | 1.2580 ms | not_measured | managed | 0.805x |
| `np.fft.ifft(a) (complex128)` | complex128 | 1,000 | 0.0149 ms | not_measured | managed | 0.617x |
| `np.fft.ifft(a) (complex128)` | complex128 | 100,000 | 2.6423 ms | not_measured | managed | 0.647x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 1,000 | 0.0921 ms | not_measured | managed | 0.445x |
| `np.fft.ifft2(a) (complex128)` | complex128 | 100,000 | 12.2657 ms | not_measured | managed | 0.400x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 1,000 | 0.0931 ms | not_measured | managed | 0.426x |
| `np.fft.ifftn(a) (complex128)` | complex128 | 100,000 | 12.3028 ms | not_measured | managed | 0.383x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 1,000 | 0.0131 ms | not_measured | managed | 0.366x |
| `np.fft.ifftshift(a) (complex128)` | complex128 | 100,000 | 0.3828 ms | not_measured | managed | 0.977x |
| `np.fft.ihfft(a) (float64)` | float64 | 1,000 | 0.0140 ms | not_measured | managed | 0.600x |
| `np.fft.ihfft(a) (float64)` | float64 | 100,000 | 1.3247 ms | not_measured | managed | 0.582x |
| `np.fft.irfft(a) (float64)` | float64 | 1,000 | 0.0112 ms | not_measured | managed | 0.643x |
| `np.fft.irfft(a) (float64)` | float64 | 100,000 | 1.1853 ms | not_measured | managed | 0.652x |
| `np.fft.irfft2(a) (float64)` | float64 | 1,000 | 0.0394 ms | not_measured | managed | 0.665x |
| `np.fft.irfft2(a) (float64)` | float64 | 100,000 | 6.3602 ms | not_measured | managed | 0.387x |
| `np.fft.irfftn(a) (float64)` | float64 | 1,000 | 0.0450 ms | not_measured | managed | 0.587x |
| `np.fft.irfftn(a) (float64)` | float64 | 100,000 | 6.0528 ms | not_measured | managed | 0.422x |
| `np.fft.rfft(a) (float64)` | float64 | 1,000 | 0.0094 ms | not_measured | managed | 0.691x |
| `np.fft.rfft(a) (float64)` | float64 | 100,000 | 1.1505 ms | not_measured | managed | 0.630x |
| `np.fft.rfft2(a) (float64)` | float64 | 1,000 | 0.0436 ms | not_measured | managed | 0.693x |
| `np.fft.rfft2(a) (float64)` | float64 | 100,000 | 6.0298 ms | not_measured | managed | 0.406x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 1,000 | 0.0091 ms | not_measured | managed | 0.209x |
| `np.fft.rfftfreq(n) (float64)` | float64 | 100,000 | 0.2016 ms | not_measured | managed | 0.131x |
| `np.fft.rfftn(a) (float64)` | float64 | 1,000 | 0.0442 ms | not_measured | managed | 0.597x |
| `np.fft.rfftn(a) (float64)` | float64 | 100,000 | 6.1110 ms | not_measured | managed | 0.393x |
| `np.diagonal(a) (float64)` | float64 | 1,000 | 0.0006 ms | not_measured | managed | 0.833x |
| `np.diagonal(a) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.diagonal(a) (float64)` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 0.857x |
| `np.dot(a, b) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.dot(a, b) (float64)` | float64 | 100,000 | 0.0183 ms | 0.0076 ms | openblas | 1.092x |
| `np.dot(a, b) (float64)` | float64 | 10,000,000 | 11.4800 ms | not_measured | managed | 0.091x |
| `np.einsum('ij,jk->ik', a, b)` | float64 | 128 | 0.2206 ms | 0.0939 ms | openblas | 2.823x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 1,000 | 0.0204 ms | not_measured | managed | 0.074x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 100,000 | 1.6906 ms | not_measured | managed | 0.012x |
| `np.einsum(subscripts, operands) (float64)` | float64 | 10,000,000 | 165.6081 ms | not_measured | managed | 0.029x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 2.500x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 100,000 | 0.0039 ms | not_measured | managed | 2.667x |
| `np.einsum_path(subscripts, operands) (float64)` | float64 | 10,000,000 | 0.0034 ms | not_measured | managed | 2.676x |
| `np.inner(a, b)` | float64 | 100,000 | 0.0075 ms | 0.0078 ms | managed | 1.013x |
| `np.linalg.cholesky(a)` | float64 | 32 | missing_backend | 0.0166 ms | openblas | 0.307x |
| `np.linalg.cholesky(a)` | float64 | 96 | missing_backend | 0.0414 ms | openblas | 0.553x |
| `np.linalg.cond(a)` | float64 | 32 | missing_backend | 0.0454 ms | openblas | 0.934x |
| `np.linalg.cond(a)` | float64 | 96 | missing_backend | 0.3186 ms | openblas | 1.048x |
| `np.linalg.cross(a, b) (float64)` | float64 | 1,000 | 0.0531 ms | not_measured | managed | 0.286x |
| `np.linalg.cross(a, b) (float64)` | float64 | 100,000 | 1.6998 ms | not_measured | managed | 0.407x |
| `np.linalg.det(a)` | float64 | 32 | missing_backend | 0.0100 ms | openblas | 0.700x |
| `np.linalg.det(a)` | float64 | 96 | missing_backend | 0.0322 ms | openblas | 0.997x |
| `np.linalg.diagonal(a) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.875x |
| `np.linalg.diagonal(a) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.778x |
| `np.linalg.eig(a)` | float64 | 32 | missing_backend | 0.1395 ms | openblas | 0.823x |
| `np.linalg.eig(a)` | float64 | 96 | missing_backend | 1.4870 ms | openblas | 1.014x |
| `np.linalg.eigh(a)` | float64 | 32 | missing_backend | 0.0577 ms | openblas | 0.927x |
| `np.linalg.eigh(a)` | float64 | 96 | missing_backend | 0.4273 ms | openblas | 1.109x |
| `np.linalg.eigvals(a)` | float64 | 32 | missing_backend | 0.0800 ms | openblas | 0.875x |
| `np.linalg.eigvals(a)` | float64 | 96 | missing_backend | 0.8608 ms | openblas | 1.060x |
| `np.linalg.eigvalsh(a)` | float64 | 32 | missing_backend | 0.0245 ms | openblas | 0.976x |
| `np.linalg.eigvalsh(a)` | float64 | 96 | missing_backend | 0.1664 ms | openblas | 1.096x |
| `np.linalg.inv(a)` | float64 | 32 | missing_backend | 0.0209 ms | openblas | 0.550x |
| `np.linalg.inv(a)` | float64 | 96 | missing_backend | 0.0881 ms | openblas | 1.002x |
| `np.linalg.lstsq(a, b)` | float64 | 32 | missing_backend | 0.0946 ms | openblas | 0.970x |
| `np.linalg.lstsq(a, b)` | float64 | 96 | missing_backend | 0.8382 ms | openblas | 1.046x |
| `np.linalg.matmul(a, b)` | float64 | 128 | 0.2107 ms | 0.0915 ms | openblas | 0.732x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 1,000 | 0.0149 ms | not_measured | managed | 0.181x |
| `np.linalg.matmul(a, b) (float64)` | float64 | 100,000 | 0.5020 ms | not_measured | managed | 0.106x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 1,000 | 0.0156 ms | not_measured | managed | 0.160x |
| `np.linalg.matrix_norm(a) (float64)` | float64 | 100,000 | 0.0769 ms | not_measured | managed | 0.082x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0364 ms | openblas | 1.283x |
| `np.linalg.matrix_norm(a, ord=2)` | float64 | 96 | missing_backend | 0.3064 ms | openblas | 1.077x |
| `np.linalg.matrix_power(a, -1)` | float64 | 32 | missing_backend | 0.0164 ms | openblas | 0.817x |
| `np.linalg.matrix_power(a, -1)` | float64 | 96 | missing_backend | 0.0919 ms | openblas | 1.055x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 1,000 | 0.0264 ms | not_measured | managed | 0.220x |
| `np.linalg.matrix_power(a, 3) (float64)` | float64 | 100,000 | 0.9700 ms | not_measured | managed | 0.128x |
| `np.linalg.matrix_rank(a)` | float64 | 32 | missing_backend | 0.0446 ms | openblas | 0.944x |
| `np.linalg.matrix_rank(a)` | float64 | 96 | missing_backend | 0.3003 ms | openblas | 1.010x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.636x |
| `np.linalg.matrix_transpose(a) (float64)` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 0.583x |
| `np.linalg.multi_dot(arrays)` | float64 | 128 | 0.3759 ms | 0.1970 ms | openblas | 0.672x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 1,000 | 0.0280 ms | not_measured | managed | 0.204x |
| `np.linalg.multi_dot(arrays) (float64)` | float64 | 100,000 | 0.9732 ms | not_measured | managed | 0.124x |
| `np.linalg.norm(a) (float64)` | float64 | 1,000 | 0.0031 ms | not_measured | managed | 0.323x |
| `np.linalg.norm(a) (float64)` | float64 | 100,000 | 0.0188 ms | not_measured | managed | 5.287x |
| `np.linalg.norm(a, ord=2)` | float64 | 32 | missing_backend | 0.0356 ms | openblas | 1.315x |
| `np.linalg.norm(a, ord=2)` | float64 | 96 | missing_backend | 0.2922 ms | openblas | 1.056x |
| `np.linalg.outer(a, b) (float64)` | float64 | 1,000 | 0.0112 ms | not_measured | managed | 0.223x |
| `np.linalg.outer(a, b) (float64)` | float64 | 100,000 | 0.1636 ms | not_measured | managed | 0.243x |
| `np.linalg.pinv(a)` | float64 | 32 | missing_backend | 0.1116 ms | openblas | 0.880x |
| `np.linalg.pinv(a)` | float64 | 96 | missing_backend | 0.8636 ms | openblas | 0.968x |
| `np.linalg.qr(a)` | float64 | 32 | missing_backend | 0.0360 ms | openblas | 0.731x |
| `np.linalg.qr(a)` | float64 | 96 | missing_backend | 0.1876 ms | openblas | 1.001x |
| `np.linalg.slogdet(a)` | float64 | 32 | missing_backend | 0.0088 ms | openblas | 0.898x |
| `np.linalg.slogdet(a)` | float64 | 96 | missing_backend | 0.0319 ms | openblas | 1.028x |
| `np.linalg.solve(a, b)` | float64 | 32 | missing_backend | 0.0089 ms | openblas | 1.202x |
| `np.linalg.solve(a, b)` | float64 | 96 | missing_backend | 0.0320 ms | openblas | 1.047x |
| `np.linalg.svd(a)` | float64 | 32 | missing_backend | 0.0857 ms | openblas | 1.090x |
| `np.linalg.svd(a)` | float64 | 96 | missing_backend | 0.7232 ms | openblas | 1.106x |
| `np.linalg.svdvals(a)` | float64 | 32 | missing_backend | 0.0329 ms | openblas | 1.198x |
| `np.linalg.svdvals(a)` | float64 | 96 | missing_backend | 0.2928 ms | openblas | 1.131x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0185 ms | not_measured | managed | 0.324x |
| `np.linalg.tensordot(a, b) (float64)` | float64 | 100,000 | 0.4766 ms | not_measured | managed | 0.138x |
| `np.linalg.tensordot(a, b, axes=1)` | float64 | 128 | 0.2046 ms | 0.0895 ms | openblas | 0.741x |
| `np.linalg.tensorinv(a)` | float64 | 32 | missing_backend | 0.0165 ms | openblas | 0.891x |
| `np.linalg.tensorinv(a)` | float64 | 96 | missing_backend | 0.0869 ms | openblas | 1.137x |
| `np.linalg.tensorsolve(a, b)` | float64 | 32 | missing_backend | 0.0094 ms | openblas | 0.979x |
| `np.linalg.tensorsolve(a, b)` | float64 | 96 | missing_backend | 0.0326 ms | openblas | 1.199x |
| `np.linalg.trace(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.455x |
| `np.linalg.trace(a) (float64)` | float64 | 100,000 | 0.0014 ms | not_measured | managed | 1.143x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0115 ms | not_measured | managed | 0.078x |
| `np.linalg.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1864 ms | 0.0145 ms | openblas | 0.545x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 1,000 | 0.0180 ms | not_measured | managed | 0.150x |
| `np.linalg.vector_norm(a) (float64)` | float64 | 100,000 | 0.3644 ms | not_measured | managed | 0.086x |
| `np.matmul(A, B) (float64)` | float64 | 1,000 | 0.0116 ms | not_measured | managed | 0.216x |
| `np.matmul(A, B) (float64)` | float64 | 100,000 | 4.5719 ms | not_measured | managed | 0.125x |
| `np.matmul(A, B) (float64)` | float64 | 10,000,000 | 8.5667 ms | not_measured | managed | 0.072x |
| `np.matmul(a, b)` | float64 | 128 | 0.2112 ms | 0.0933 ms | openblas | 0.682x |
| `np.matvec(a, b)` | float64 | 128 | 0.0231 ms | 0.0022 ms | openblas | 1.000x |
| `np.matvec(a, b) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.189x |
| `np.matvec(a, b) (float64)` | float64 | 100,000 | 0.1351 ms | not_measured | managed | 0.044x |
| `np.matvec(a, b) (float64)` | float64 | 10,000,000 | 0.1907 ms | not_measured | managed | 0.041x |
| `np.outer(a, b) (float64)` | float64 | 1,000 | 0.0123 ms | not_measured | managed | 0.195x |
| `np.outer(a, b) (float64)` | float64 | 100,000 | 0.1559 ms | not_measured | managed | 0.250x |
| `np.outer(a, b) (float64)` | float64 | 10,000,000 | 15.9524 ms | not_measured | managed | 0.902x |
| `np.tensordot(a, b) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.184x |
| `np.tensordot(a, b) (float64)` | float64 | 100,000 | 1.7343 ms | not_measured | managed | 0.061x |
| `np.tensordot(a, b) (float64)` | float64 | 10,000,000 | 166.5067 ms | not_measured | managed | 0.008x |
| `np.tensordot(a, b, axes=1)` | float64 | 128 | 0.2107 ms | 0.0943 ms | openblas | 0.704x |
| `np.vdot(a, b) (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.176x |
| `np.vdot(a, b) (float64)` | float64 | 100,000 | 0.0147 ms | 0.0083 ms | openblas | 0.904x |
| `np.vdot(a, b) (float64)` | float64 | 10,000,000 | 11.3037 ms | not_measured | managed | 0.124x |
| `np.vecdot(a, b) (float64)` | float64 | 1,000 | 0.0060 ms | not_measured | managed | 0.117x |
| `np.vecdot(a, b) (float64)` | float64 | 100,000 | 0.1817 ms | 0.0078 ms | openblas | 0.974x |
| `np.vecdot(a, b) (float64)` | float64 | 10,000,000 | 37.0803 ms | not_measured | managed | 0.036x |
| `np.vecmat(a, b)` | float64 | 128 | 0.0036 ms | 0.0022 ms | openblas | 0.773x |
| `np.vecmat(a, b) (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.127x |
| `np.vecmat(a, b) (float64)` | float64 | 100,000 | 0.0671 ms | not_measured | managed | 0.086x |
| `np.vecmat(a, b) (float64)` | float64 | 10,000,000 | 0.0831 ms | not_measured | managed | 0.105x |
| `np.all(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.all(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.allclose(a, b) (float16)` | float16 | 1,000 | 0.1052 ms | not_measured | managed | 0.305x |
| `np.allclose(a, b) (float16)` | float16 | 100,000 | 1.0727 ms | not_measured | managed | 1.681x |
| `np.allclose(a, b) (float32)` | float32 | 1,000 | 0.0281 ms | not_measured | managed | 0.562x |
| `np.allclose(a, b) (float32)` | float32 | 100,000 | 0.6146 ms | not_measured | managed | 0.657x |
| `np.allclose(a, b) (float64)` | float64 | 1,000 | 0.0446 ms | not_measured | managed | 0.307x |
| `np.allclose(a, b) (float64)` | float64 | 100,000 | 0.5379 ms | not_measured | managed | 1.228x |
| `np.any(a) (bool)` | bool | 1,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 100,000 | not_measured | not_measured | — | — |
| `np.any(a) (bool)` | bool | 10,000,000 | not_measured | not_measured | — | — |
| `np.array_equal(a, b) (float16)` | float16 | 1,000 | 0.0031 ms | not_measured | managed | 0.839x |
| `np.array_equal(a, b) (float16)` | float16 | 100,000 | 0.1328 ms | not_measured | managed | 0.667x |
| `np.array_equal(a, b) (float16)` | float16 | 10,000,000 | 13.2102 ms | not_measured | managed | 1.742x |
| `np.array_equal(a, b) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 1.235x |
| `np.array_equal(a, b) (float32)` | float32 | 100,000 | 0.0189 ms | not_measured | managed | 0.413x |
| `np.array_equal(a, b) (float32)` | float32 | 10,000,000 | 10.6271 ms | not_measured | managed | 0.989x |
| `np.array_equal(a, b) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.727x |
| `np.array_equal(a, b) (float64)` | float64 | 100,000 | 0.0296 ms | not_measured | managed | 0.422x |
| `np.array_equal(a, b) (float64)` | float64 | 10,000,000 | 17.0187 ms | not_measured | managed | 1.057x |
| `np.fmax(a, b) (float16)` | float16 | 1,000 | 0.0041 ms | not_measured | managed | 0.732x |
| `np.fmax(a, b) (float16)` | float16 | 100,000 | 0.9329 ms | not_measured | managed | 0.812x |
| `np.fmax(a, b) (float16)` | float16 | 10,000,000 | 90.2446 ms | not_measured | managed | 1.141x |
| `np.fmax(a, b) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.222x |
| `np.fmax(a, b) (float32)` | float32 | 100,000 | 0.0653 ms | not_measured | managed | 0.124x |
| `np.fmax(a, b) (float32)` | float32 | 10,000,000 | 12.6556 ms | not_measured | managed | 1.263x |
| `np.fmax(a, b) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.171x |
| `np.fmax(a, b) (float64)` | float64 | 100,000 | 0.1486 ms | not_measured | managed | 0.218x |
| `np.fmax(a, b) (float64)` | float64 | 10,000,000 | 32.2104 ms | not_measured | managed | 1.028x |
| `np.fmin(a, b) (float16)` | float16 | 1,000 | 0.0050 ms | not_measured | managed | 0.580x |
| `np.fmin(a, b) (float16)` | float16 | 100,000 | 0.9324 ms | not_measured | managed | 0.823x |
| `np.fmin(a, b) (float16)` | float16 | 10,000,000 | 90.9450 ms | not_measured | managed | 1.087x |
| `np.fmin(a, b) (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.250x |
| `np.fmin(a, b) (float32)` | float32 | 100,000 | 0.0590 ms | not_measured | managed | 0.132x |
| `np.fmin(a, b) (float32)` | float32 | 10,000,000 | 12.2661 ms | not_measured | managed | 1.337x |
| `np.fmin(a, b) (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.150x |
| `np.fmin(a, b) (float64)` | float64 | 100,000 | 0.1510 ms | not_measured | managed | 0.209x |
| `np.fmin(a, b) (float64)` | float64 | 10,000,000 | 36.3312 ms | not_measured | managed | 0.911x |
| `np.isclose(a, b) (float16)` | float16 | 1,000 | 0.0941 ms | not_measured | managed | 0.357x |
| `np.isclose(a, b) (float16)` | float16 | 100,000 | 1.1131 ms | not_measured | managed | 1.597x |
| `np.isclose(a, b) (float32)` | float32 | 1,000 | 0.0387 ms | not_measured | managed | 0.310x |
| `np.isclose(a, b) (float32)` | float32 | 100,000 | 0.6096 ms | not_measured | managed | 0.628x |
| `np.isclose(a, b) (float64)` | float64 | 1,000 | 0.0562 ms | not_measured | managed | 0.212x |
| `np.isclose(a, b) (float64)` | float64 | 100,000 | 0.5700 ms | not_measured | managed | 1.282x |
| `np.iscomplex(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.iscomplex(a) (float16)` | float16 | 100,000 | 0.0188 ms | not_measured | managed | 0.090x |
| `np.iscomplex(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.iscomplex(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.357x |
| `np.iscomplex(a) (float32)` | float32 | 100,000 | 0.0174 ms | not_measured | managed | 0.098x |
| `np.iscomplex(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.iscomplex(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.250x |
| `np.iscomplex(a) (float64)` | float64 | 100,000 | 0.0174 ms | not_measured | managed | 0.109x |
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
| `np.isfinite(a) (float16)` | float16 | 1,000 | 0.0019 ms | not_measured | managed | 0.474x |
| `np.isfinite(a) (float16)` | float16 | 100,000 | 0.0588 ms | not_measured | managed | 0.821x |
| `np.isfinite(a) (float16)` | float16 | 10,000,000 | 6.6886 ms | not_measured | managed | 0.876x |
| `np.isfinite(a) (float32)` | float32 | 1,000 | 0.0028 ms | not_measured | managed | 0.143x |
| `np.isfinite(a) (float32)` | float32 | 100,000 | 0.0558 ms | not_measured | managed | 0.093x |
| `np.isfinite(a) (float32)` | float32 | 10,000,000 | 7.3553 ms | not_measured | managed | 0.867x |
| `np.isfinite(a) (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.156x |
| `np.isfinite(a) (float64)` | float64 | 100,000 | 0.0618 ms | not_measured | managed | 0.167x |
| `np.isfinite(a) (float64)` | float64 | 10,000,000 | 10.6654 ms | not_measured | managed | 1.247x |
| `np.isfortran(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.isfortran(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.isinf(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 0.391x |
| `np.isinf(a) (float16)` | float16 | 100,000 | 0.0622 ms | not_measured | managed | 0.815x |
| `np.isinf(a) (float16)` | float16 | 10,000,000 | 6.3973 ms | not_measured | managed | 0.968x |
| `np.isinf(a) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.185x |
| `np.isinf(a) (float32)` | float32 | 100,000 | 0.0599 ms | not_measured | managed | 0.088x |
| `np.isinf(a) (float32)` | float32 | 10,000,000 | 6.7826 ms | not_measured | managed | 0.954x |
| `np.isinf(a) (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.isinf(a) (float64)` | float64 | 100,000 | 0.0709 ms | not_measured | managed | 0.147x |
| `np.isinf(a) (float64)` | float64 | 10,000,000 | 10.8165 ms | not_measured | managed | 1.178x |
| `np.isnan(a) (float16)` | float16 | 1,000 | 0.0026 ms | not_measured | managed | 0.538x |
| `np.isnan(a) (float16)` | float16 | 100,000 | 0.0617 ms | not_measured | managed | 1.126x |
| `np.isnan(a) (float16)` | float16 | 10,000,000 | 6.2036 ms | not_measured | managed | 1.304x |
| `np.isnan(a) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.238x |
| `np.isnan(a) (float32)` | float32 | 100,000 | 0.0586 ms | not_measured | managed | 0.070x |
| `np.isnan(a) (float32)` | float32 | 10,000,000 | 7.2419 ms | not_measured | managed | 0.894x |
| `np.isnan(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.isnan(a) (float64)` | float64 | 100,000 | 0.0613 ms | not_measured | managed | 0.142x |
| `np.isnan(a) (float64)` | float64 | 10,000,000 | 11.1416 ms | not_measured | managed | 1.038x |
| `np.isreal(a) (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 1.333x |
| `np.isreal(a) (float16)` | float16 | 100,000 | 0.0171 ms | not_measured | managed | 4.544x |
| `np.isreal(a) (float16)` | float16 | 10,000,000 | 1.7465 ms | not_measured | managed | 13.142x |
| `np.isreal(a) (float32)` | float32 | 1,000 | 0.0010 ms | not_measured | managed | 1.300x |
| `np.isreal(a) (float32)` | float32 | 100,000 | 0.0160 ms | not_measured | managed | 0.544x |
| `np.isreal(a) (float32)` | float32 | 10,000,000 | 1.7310 ms | not_measured | managed | 6.550x |
| `np.isreal(a) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.650x |
| `np.isreal(a) (float64)` | float64 | 100,000 | 0.0165 ms | not_measured | managed | 1.042x |
| `np.isreal(a) (float64)` | float64 | 10,000,000 | 2.3786 ms | not_measured | managed | 8.830x |
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
| `np.maximum(a, b) (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 0.711x |
| `np.maximum(a, b) (float16)` | float16 | 100,000 | 0.9799 ms | not_measured | managed | 0.786x |
| `np.maximum(a, b) (float16)` | float16 | 10,000,000 | 95.4103 ms | not_measured | managed | 1.083x |
| `np.maximum(a, b) (float32)` | float32 | 1,000 | 0.0030 ms | not_measured | managed | 0.200x |
| `np.maximum(a, b) (float32)` | float32 | 100,000 | 0.0604 ms | not_measured | managed | 0.129x |
| `np.maximum(a, b) (float32)` | float32 | 10,000,000 | 15.1897 ms | not_measured | managed | 1.086x |
| `np.maximum(a, b) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.171x |
| `np.maximum(a, b) (float64)` | float64 | 100,000 | 0.1476 ms | not_measured | managed | 0.205x |
| `np.maximum(a, b) (float64)` | float64 | 10,000,000 | 30.0723 ms | not_measured | managed | 1.211x |
| `np.minimum(a, b) (float16)` | float16 | 1,000 | 0.0042 ms | not_measured | managed | 0.738x |
| `np.minimum(a, b) (float16)` | float16 | 100,000 | 0.9733 ms | not_measured | managed | 0.779x |
| `np.minimum(a, b) (float16)` | float16 | 10,000,000 | 93.8227 ms | not_measured | managed | 1.097x |
| `np.minimum(a, b) (float32)` | float32 | 1,000 | 0.0025 ms | not_measured | managed | 0.240x |
| `np.minimum(a, b) (float32)` | float32 | 100,000 | 0.0598 ms | not_measured | managed | 0.134x |
| `np.minimum(a, b) (float32)` | float32 | 10,000,000 | 12.8489 ms | not_measured | managed | 1.269x |
| `np.minimum(a, b) (float64)` | float64 | 1,000 | 0.0038 ms | not_measured | managed | 0.158x |
| `np.minimum(a, b) (float64)` | float64 | 100,000 | 0.1429 ms | not_measured | managed | 0.209x |
| `np.minimum(a, b) (float64)` | float64 | 10,000,000 | 30.1096 ms | not_measured | managed | 1.052x |
| `a.T (transpose 2D)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.111x |
| `a.T (transpose 2D)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.125x |
| `a.T (transpose 2D)` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.222x |
| `a.flatten` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.152x |
| `a.flatten` | float64 | 100,000 | 0.1144 ms | not_measured | managed | 0.099x |
| `a.flatten` | float64 | 10,000,000 | 16.4230 ms | not_measured | managed | 1.177x |
| `a.ravel() (view)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.083x |
| `a.ravel() (view)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `a.ravel() (view)` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `np.append(a, b) (float64)` | float64 | 1,000 | 0.0061 ms | not_measured | managed | 0.262x |
| `np.append(a, b) (float64)` | float64 | 100,000 | 0.3376 ms | not_measured | managed | 1.395x |
| `np.array_split(a, sections) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 1.222x |
| `np.array_split(a, sections) (float64)` | float64 | 100,000 | 0.0055 ms | not_measured | managed | 3.255x |
| `np.atleast_1d(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_1d(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.atleast_2d(a) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.400x |
| `np.atleast_2d(a) (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 1.000x |
| `np.atleast_3d(a) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.atleast_3d(a) (float64)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 0.524x |
| `np.block([a, b]) (float64)` | float64 | 1,000 | 0.0060 ms | not_measured | managed | 0.550x |
| `np.block([a, b]) (float64)` | float64 | 100,000 | 0.3474 ms | not_measured | managed | 1.420x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 1.889x |
| `np.broadcast_arrays(a, b) (float64)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 4.524x |
| `np.byteswap` | float64 | 1,000 | 0.0060 ms | not_measured | managed | 0.133x |
| `np.byteswap` | float64 | 100,000 | 0.2107 ms | not_measured | managed | 0.167x |
| `np.byteswap` | float64 | 10,000,000 | 16.3488 ms | not_measured | managed | 2.633x |
| `np.c_` | float64 | 1,000 | 0.0108 ms | not_measured | managed | 0.398x |
| `np.c_` | float64 | 100,000 | 0.3620 ms | not_measured | managed | 0.928x |
| `np.c_` | float64 | 10,000,000 | 44.2182 ms | not_measured | managed | 1.570x |
| `np.column_stack((a, b)) (float64)` | float64 | 1,000 | 0.0088 ms | not_measured | managed | 0.250x |
| `np.column_stack((a, b)) (float64)` | float64 | 100,000 | 0.3549 ms | not_measured | managed | 1.382x |
| `np.concat((a, b)) (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.189x |
| `np.concat((a, b)) (float64)` | float64 | 100,000 | 0.3369 ms | not_measured | managed | 1.324x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 1,000 | 0.0101 ms | not_measured | managed | 0.129x |
| `np.concatenate([a, b, c]) (float64)` | float64 | 100,000 | 0.6118 ms | not_measured | managed | 0.773x |
| `np.concatenate([a, b]) (float64)` | float64 | 1,000 | 0.0071 ms | not_measured | managed | 0.141x |
| `np.concatenate([a, b]) (float64)` | float64 | 100,000 | 0.4008 ms | not_measured | managed | 0.819x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.186x |
| `np.concatenate([a, b], axis=0) (float64)` | float64 | 100,000 | 0.4204 ms | not_measured | managed | 0.763x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0063 ms | not_measured | managed | 0.190x |
| `np.concatenate([a, b], axis=1) (float64)` | float64 | 100,000 | 0.3860 ms | not_measured | managed | 0.791x |
| `np.delete(a, index) (float64)` | float64 | 1,000 | 0.0054 ms | not_measured | managed | 0.407x |
| `np.delete(a, index) (float64)` | float64 | 100,000 | 0.1398 ms | not_measured | managed | 0.218x |
| `np.diag` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 0.087x |
| `np.diag` | float64 | 100,000 | 0.1374 ms | not_measured | managed | 0.006x |
| `np.diag` | float64 | 10,000,000 | 3.7297 ms | not_measured | managed | 0.001x |
| `np.diagflat` | float64 | 1,000 | 0.0100 ms | not_measured | managed | 0.310x |
| `np.diagflat` | float64 | 100,000 | 0.0804 ms | not_measured | managed | 0.174x |
| `np.diagflat` | float64 | 10,000,000 | 4.1278 ms | not_measured | managed | 1.539x |
| `np.dsplit(a, sections) (float64)` | float64 | 1,000 | 0.0089 ms | not_measured | managed | 1.056x |
| `np.dsplit(a, sections) (float64)` | float64 | 100,000 | 0.0073 ms | not_measured | managed | 3.438x |
| `np.dstack([a, b]) (float64)` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.348x |
| `np.dstack([a, b]) (float64)` | float64 | 100,000 | 0.4391 ms | not_measured | managed | 0.756x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 1.333x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 1.500x |
| `np.expand_dims(a, axis=-1) (float64)` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 3.500x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 1.091x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 2.167x |
| `np.expand_dims(a, axis=0) (float64)` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 4.125x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 1.538x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 1.667x |
| `np.expand_dims(a2D, axis=1) (float64)` | float64 | 10,000,000 | 0.0012 ms | not_measured | managed | 2.583x |
| `np.fill_diagonal` | float64 | 1,000 | 0.0045 ms | not_measured | managed | 0.222x |
| `np.fill_diagonal` | float64 | 100,000 | 0.0046 ms | not_measured | managed | 0.522x |
| `np.fill_diagonal` | float64 | 10,000,000 | 0.0437 ms | not_measured | managed | 1.128x |
| `np.flip` | float64 | 1,000 | 0.0005 ms | not_measured | managed | 0.800x |
| `np.flip` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 0.308x |
| `np.flip` | float64 | 10,000,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.fliplr` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.571x |
| `np.fliplr` | float64 | 100,000 | 0.0015 ms | not_measured | managed | 0.267x |
| `np.fliplr` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 1.125x |
| `np.flipud` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.300x |
| `np.flipud` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 0.231x |
| `np.flipud` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 0.727x |
| `np.hsplit(a, sections) (float64)` | float64 | 1,000 | 0.0093 ms | not_measured | managed | 1.000x |
| `np.hsplit(a, sections) (float64)` | float64 | 100,000 | 0.0077 ms | not_measured | managed | 3.039x |
| `np.hstack([a, b]) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 0.354x |
| `np.hstack([a, b]) (float64)` | float64 | 100,000 | 0.3738 ms | not_measured | managed | 0.902x |
| `np.insert(a, index, value) (float64)` | float64 | 1,000 | 0.0093 ms | not_measured | managed | 0.559x |
| `np.insert(a, index, value) (float64)` | float64 | 100,000 | 0.1397 ms | not_measured | managed | 0.271x |
| `np.intersect1d(a, b) (int32)` | int32 | 1,000 | 0.0476 ms | not_measured | managed | 0.853x |
| `np.intersect1d(a, b) (int32)` | int32 | 100,000 | 10.5536 ms | not_measured | managed | 0.796x |
| `np.isin(a, b) (int32)` | int32 | 1,000 | 0.0442 ms | not_measured | managed | 0.396x |
| `np.isin(a, b) (int32)` | int32 | 100,000 | 1.3420 ms | not_measured | managed | 0.614x |
| `np.ix_` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.630x |
| `np.ix_` | float64 | 100,000 | 0.0028 ms | not_measured | managed | 0.643x |
| `np.ix_` | float64 | 10,000,000 | 0.0025 ms | not_measured | managed | 2.000x |
| `np.matrix_transpose` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.667x |
| `np.matrix_transpose` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.750x |
| `np.matrix_transpose` | float64 | 10,000,000 | 0.0013 ms | not_measured | managed | 0.923x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 1.312x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 100,000 | 0.0014 ms | not_measured | managed | 1.500x |
| `np.moveaxis(a, 0, -1) (float64)` | float64 | 10,000,000 | 0.0015 ms | not_measured | managed | 3.333x |
| `np.pad(a, width) (float64)` | float64 | 1,000 | 0.0150 ms | not_measured | managed | 0.613x |
| `np.pad(a, width) (float64)` | float64 | 100,000 | 0.1486 ms | not_measured | managed | 0.349x |
| `np.permute_dims` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.333x |
| `np.permute_dims` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.500x |
| `np.permute_dims` | float64 | 10,000,000 | 0.0008 ms | not_measured | managed | 0.875x |
| `np.r_` | float64 | 1,000 | 0.0051 ms | not_measured | managed | 0.588x |
| `np.r_` | float64 | 100,000 | 0.3909 ms | not_measured | managed | 0.899x |
| `np.r_` | float64 | 10,000,000 | 30.7325 ms | not_measured | managed | 1.310x |
| `np.r_(0:1:Nj)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 1.538x |
| `np.r_(0:1:Nj)` | float64 | 100,000 | 0.2447 ms | not_measured | managed | 1.612x |
| `np.r_(0:1:Nj)` | float64 | 10,000,000 | 32.2724 ms | not_measured | managed | 2.749x |
| `np.r_(0:N)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 0.387x |
| `np.r_(0:N)` | float64 | 100,000 | 0.2549 ms | not_measured | managed | 1.357x |
| `np.r_(0:N)` | float64 | 10,000,000 | 31.4479 ms | not_measured | managed | 1.165x |
| `np.r_(a, 0, 0, b)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.792x |
| `np.r_(a, 0, 0, b)` | float64 | 100,000 | 0.3824 ms | not_measured | managed | 0.897x |
| `np.r_(a, 0, 0, b)` | float64 | 10,000,000 | 30.6441 ms | not_measured | managed | 1.224x |
| `np.ravel` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.273x |
| `np.ravel` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.ravel` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 0.700x |
| `np.repeat(a, repeats) (float64)` | float64 | 1,000 | 0.0082 ms | not_measured | managed | 0.341x |
| `np.repeat(a, repeats) (float64)` | float64 | 100,000 | 0.4075 ms | not_measured | managed | 2.421x |
| `np.reshape(a, shape)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.500x |
| `np.reshape(a, shape)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.reshape(a, shape)` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 1.600x |
| `np.resize(a, shape) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.390x |
| `np.resize(a, shape) (float64)` | float64 | 100,000 | 0.3426 ms | not_measured | managed | 1.658x |
| `np.roll(a, shift) (float64)` | float64 | 1,000 | 0.0143 ms | not_measured | managed | 0.336x |
| `np.roll(a, shift) (float64)` | float64 | 100,000 | 0.1340 ms | not_measured | managed | 0.531x |
| `np.rollaxis(a, 2) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.600x |
| `np.rollaxis(a, 2) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.rollaxis(a, 2) (float64)` | float64 | 10,000,000 | 0.0010 ms | not_measured | managed | 1.400x |
| `np.rot90` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 1.333x |
| `np.rot90` | float64 | 100,000 | 0.0023 ms | not_measured | managed | 1.391x |
| `np.rot90` | float64 | 10,000,000 | 0.0016 ms | not_measured | managed | 6.625x |
| `np.setdiff1d(a, b) (int32)` | int32 | 1,000 | 0.0738 ms | not_measured | managed | 0.680x |
| `np.setdiff1d(a, b) (int32)` | int32 | 100,000 | 10.2635 ms | not_measured | managed | 0.784x |
| `np.setxor1d(a, b) (int32)` | int32 | 1,000 | 0.0790 ms | not_measured | managed | 0.487x |
| `np.setxor1d(a, b) (int32)` | int32 | 100,000 | 10.0405 ms | not_measured | managed | 0.741x |
| `np.size(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.size(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.split(a, sections) (float64)` | float64 | 1,000 | 0.0068 ms | not_measured | managed | 1.309x |
| `np.split(a, sections) (float64)` | float64 | 100,000 | 0.0081 ms | not_measured | managed | 2.840x |
| `np.squeeze(a) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.375x |
| `np.squeeze(a) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.429x |
| `np.squeeze(a) (float64)` | float64 | 10,000,000 | 0.0006 ms | not_measured | managed | 1.333x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.429x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 100,000 | 0.0007 ms | not_measured | managed | 0.571x |
| `np.squeeze(a, axis=1) (float64)` | float64 | 10,000,000 | 0.0005 ms | not_measured | managed | 1.800x |
| `np.stack([a, b]) (float64)` | float64 | 1,000 | 0.0071 ms | not_measured | managed | 0.310x |
| `np.stack([a, b]) (float64)` | float64 | 100,000 | 0.3663 ms | not_measured | managed | 0.866x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 0.571x |
| `np.stack([a, b], axis=1) (float64)` | float64 | 100,000 | 0.3469 ms | not_measured | managed | 0.950x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.swapaxes(a, 0, 1) (float64)` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.400x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.364x |
| `np.swapaxes(a3D, 0, 2) (float64)` | float64 | 10,000,000 | 0.0009 ms | not_measured | managed | 0.889x |
| `np.tile(a, reps) (float64)` | float64 | 1,000 | 0.0118 ms | not_measured | managed | 0.220x |
| `np.tile(a, reps) (float64)` | float64 | 100,000 | 0.3452 ms | not_measured | managed | 1.437x |
| `np.transpose(a)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 0.500x |
| `np.transpose(a)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.444x |
| `np.transpose(a)` | float64 | 10,000,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `np.tri` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.736x |
| `np.tri` | float64 | 100,000 | 0.1551 ms | not_measured | managed | 0.363x |
| `np.tri` | float64 | 10,000,000 | 15.9671 ms | not_measured | managed | 1.691x |
| `np.tril` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 1.400x |
| `np.tril` | float64 | 100,000 | 0.1327 ms | not_measured | managed | 0.378x |
| `np.tril` | float64 | 10,000,000 | 22.2531 ms | not_measured | managed | 1.507x |
| `np.tril_indices` | float64 | 1,000 | 0.0088 ms | not_measured | managed | 1.193x |
| `np.tril_indices` | float64 | 100,000 | 0.1321 ms | not_measured | managed | 0.564x |
| `np.tril_indices` | float64 | 10,000,000 | 12.9492 ms | not_measured | managed | 3.002x |
| `np.trim_zeros` | float64 | 1,000 | 0.0179 ms | not_measured | managed | 0.709x |
| `np.trim_zeros` | float64 | 100,000 | 0.0248 ms | not_measured | managed | 38.685x |
| `np.trim_zeros` | float64 | 10,000,000 | 0.1246 ms | not_measured | managed | 1901.631x |
| `np.triu` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 1.235x |
| `np.triu` | float64 | 100,000 | 0.1322 ms | not_measured | managed | 0.377x |
| `np.triu` | float64 | 10,000,000 | 19.3437 ms | not_measured | managed | 1.463x |
| `np.triu_indices` | float64 | 1,000 | 0.0113 ms | not_measured | managed | 1.035x |
| `np.triu_indices` | float64 | 100,000 | 0.1207 ms | not_measured | managed | 0.630x |
| `np.triu_indices` | float64 | 10,000,000 | 12.8489 ms | not_measured | managed | 2.421x |
| `np.union1d(a, b) (int32)` | int32 | 1,000 | 0.0276 ms | not_measured | managed | 0.772x |
| `np.union1d(a, b) (int32)` | int32 | 100,000 | 5.9013 ms | not_measured | managed | 0.821x |
| `np.unique(a) (int32)` | int32 | 1,000 | 0.0110 ms | not_measured | managed | 1.527x |
| `np.unique(a) (int32)` | int32 | 100,000 | 4.8563 ms | not_measured | managed | 0.645x |
| `np.unique_all(a) (int32)` | int32 | 1,000 | 0.0473 ms | not_measured | managed | 0.820x |
| `np.unique_all(a) (int32)` | int32 | 100,000 | 3.6148 ms | not_measured | managed | 1.809x |
| `np.unique_counts(a) (int32)` | int32 | 1,000 | 0.0206 ms | not_measured | managed | 0.490x |
| `np.unique_counts(a) (int32)` | int32 | 100,000 | 5.4617 ms | not_measured | managed | 0.118x |
| `np.unique_inverse(a) (int32)` | int32 | 1,000 | 0.0306 ms | not_measured | managed | 0.758x |
| `np.unique_inverse(a) (int32)` | int32 | 100,000 | 3.0989 ms | not_measured | managed | 1.870x |
| `np.unique_values(a) (int32)` | int32 | 1,000 | 0.0103 ms | not_measured | managed | 1.495x |
| `np.unique_values(a) (int32)` | int32 | 100,000 | 5.2385 ms | not_measured | managed | 1.106x |
| `np.unstack(a) (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.368x |
| `np.unstack(a) (float64)` | float64 | 100,000 | 0.0084 ms | not_measured | managed | 1.000x |
| `np.vsplit(a, sections) (float64)` | float64 | 1,000 | 0.0084 ms | not_measured | managed | 1.071x |
| `np.vsplit(a, sections) (float64)` | float64 | 100,000 | 0.0071 ms | not_measured | managed | 3.239x |
| `np.vstack([a, b]) (float64)` | float64 | 1,000 | 0.0078 ms | not_measured | managed | 0.269x |
| `np.vstack([a, b]) (float64)` | float64 | 100,000 | 0.4751 ms | not_measured | managed | 0.696x |
| `reshape 1D->2D` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.133x |
| `reshape 1D->2D` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `reshape 1D->2D` | float64 | 10,000,000 | 0.0007 ms | not_measured | managed | 0.429x |
| `reshape 1D->3D` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `reshape 1D->3D` | float64 | 100,000 | 0.0013 ms | not_measured | managed | 0.154x |
| `reshape 1D->3D` | float64 | 10,000,000 | 0.0012 ms | not_measured | managed | 0.250x |
| `reshape 2D->1D` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `reshape 2D->1D` | float64 | 100,000 | 0.0011 ms | not_measured | managed | 0.182x |
| `reshape 2D->1D` | float64 | 10,000,000 | 0.0012 ms | not_measured | managed | 0.250x |
| `a.all() [ndarray] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `a.all() [ndarray] (float64)` | float64 | 100,000 | 0.0174 ms | not_measured | managed | 2.379x |
| `a.any() [ndarray] (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.615x |
| `a.any() [ndarray] (float64)` | float64 | 100,000 | 0.0025 ms | not_measured | managed | 16.200x |
| `a.argmax() [ndarray] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.455x |
| `a.argmax() [ndarray] (float64)` | float64 | 100,000 | 0.0637 ms | not_measured | managed | 0.257x |
| `a.argmin() [ndarray] (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 0.545x |
| `a.argmin() [ndarray] (float64)` | float64 | 100,000 | 0.0630 ms | not_measured | managed | 0.263x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0175 ms | not_measured | managed | 0.251x |
| `a.argpartition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.7896 ms | not_measured | managed | 0.197x |
| `a.argsort() [ndarray] (float64)` | float64 | 1,000 | 0.0367 ms | not_measured | managed | 0.264x |
| `a.argsort() [ndarray] (float64)` | float64 | 100,000 | 3.8751 ms | not_measured | managed | 0.387x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 1,000 | 0.0168 ms | not_measured | managed | 0.310x |
| `a.choose(choices) [ndarray] (float64)` | float64 | 100,000 | 0.3158 ms | not_measured | managed | 1.886x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.300x |
| `a.clip(min, max) [ndarray] (float64)` | float64 | 100,000 | 0.1375 ms | not_measured | managed | 0.093x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.586x |
| `a.compress(mask) [ndarray] (float64)` | float64 | 100,000 | 0.1117 ms | not_measured | managed | 0.781x |
| `a.conj() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conj() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.conjugate() [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.cumprod() [ndarray] (float64)` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.545x |
| `a.cumprod() [ndarray] (float64)` | float64 | 100,000 | 0.2510 ms | not_measured | managed | 0.680x |
| `a.cumsum() [ndarray] (float64)` | float64 | 1,000 | 0.0062 ms | not_measured | managed | 0.371x |
| `a.cumsum() [ndarray] (float64)` | float64 | 100,000 | 0.2385 ms | not_measured | managed | 0.699x |
| `a.diagonal() [ndarray] (float64)` | float64 | 1,000 | 0.0007 ms | not_measured | managed | 0.143x |
| `a.diagonal() [ndarray] (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.dot(b) [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.333x |
| `a.dot(b) [ndarray] (float64)` | float64 | 100,000 | 0.0202 ms | 0.0077 ms | openblas | 1.130x |
| `a.fill(value) [ndarray] (float64)` | float64 | 1,000 | 0.0001 ms | not_measured | managed | 2.000x |
| `a.fill(value) [ndarray] (float64)` | float64 | 100,000 | 0.0217 ms | not_measured | managed | 0.452x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.154x |
| `a.getfield(dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0012 ms | not_measured | managed | 0.167x |
| `a.item(index) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.item(index) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.max() [ndarray] (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 0.867x |
| `a.max() [ndarray] (float64)` | float64 | 100,000 | 0.0176 ms | not_measured | managed | 0.608x |
| `a.min() [ndarray] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.786x |
| `a.min() [ndarray] (float64)` | float64 | 100,000 | 0.0178 ms | not_measured | managed | 0.545x |
| `a.nonzero() [ndarray] (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.600x |
| `a.nonzero() [ndarray] (float64)` | float64 | 100,000 | 0.1567 ms | not_measured | managed | 1.195x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 1,000 | 0.0157 ms | not_measured | managed | 0.204x |
| `a.partition(kth) [ndarray] (float64)` | float64 | 100,000 | 0.7104 ms | not_measured | managed | 0.212x |
| `a.prod() [ndarray] (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 1.048x |
| `a.prod() [ndarray] (float64)` | float64 | 100,000 | 0.0098 ms | not_measured | managed | 9.418x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 1,000 | 0.0095 ms | not_measured | managed | 0.189x |
| `a.put(indices, values) [ndarray] (float64)` | float64 | 100,000 | 0.1390 ms | not_measured | managed | 0.793x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 1,000 | 0.0094 ms | not_measured | managed | 0.330x |
| `a.repeat(repeats) [ndarray] (float64)` | float64 | 100,000 | 0.4188 ms | not_measured | managed | 0.936x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.125x |
| `a.reshape(shape) [ndarray] (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.132x |
| `a.resize(shape) [ndarray] (float64)` | float64 | 100,000 | 0.1623 ms | not_measured | managed | 0.092x |
| `a.round() [ndarray] (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.188x |
| `a.round() [ndarray] (float64)` | float64 | 100,000 | 0.1075 ms | not_measured | managed | 0.133x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 1,000 | 0.0166 ms | not_measured | managed | 0.500x |
| `a.searchsorted(v) [ndarray] (float64)` | float64 | 100,000 | 3.1382 ms | not_measured | managed | 0.674x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.125x |
| `a.setfield(value, dtype) [ndarray] (float64)` | float64 | 100,000 | 0.0507 ms | not_measured | managed | 0.211x |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.setflags(write=True) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.sort() [ndarray] (float64)` | float64 | 1,000 | 0.0287 ms | not_measured | managed | 0.153x |
| `a.sort() [ndarray] (float64)` | float64 | 100,000 | 2.1277 ms | not_measured | managed | 0.229x |
| `a.squeeze() [ndarray] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.squeeze() [ndarray] (float64)` | float64 | 100,000 | 0.0006 ms | not_measured | managed | 0.333x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 0.200x |
| `a.swapaxes(0, 1) [ndarray] (float64)` | float64 | 100,000 | 0.0008 ms | not_measured | managed | 0.250x |
| `a.take(indices) [ndarray] (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.393x |
| `a.take(indices) [ndarray] (float64)` | float64 | 100,000 | 0.1140 ms | not_measured | managed | 0.537x |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `a.to_device(cpu) [ndarray] (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `a.tobytes() [ndarray] (float64)` | float64 | 1,000 | 0.0013 ms | not_measured | managed | 0.154x |
| `a.tobytes() [ndarray] (float64)` | float64 | 100,000 | 0.0810 ms | not_measured | managed | 0.146x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 1,000 | 0.4546 ms | not_measured | managed | 0.469x |
| `a.tofile(path) [ndarray] (float64)` | float64 | 100,000 | 3.5482 ms | not_measured | managed | 0.096x |
| `a.tolist() [ndarray] (float64)` | float64 | 1,000 | 0.0135 ms | not_measured | managed | 0.593x |
| `a.tolist() [ndarray] (float64)` | float64 | 100,000 | 3.0869 ms | not_measured | managed | 0.287x |
| `a.trace() [ndarray] (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 0.588x |
| `a.trace() [ndarray] (float64)` | float64 | 100,000 | 0.0010 ms | not_measured | managed | 1.100x |
| `a.transpose() [ndarray] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.071x |
| `a.transpose() [ndarray] (float64)` | float64 | 100,000 | 0.0009 ms | not_measured | managed | 0.111x |
| `a.view() [ndarray] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 0.111x |
| `a.view() [ndarray] (float64)` | float64 | 100,000 | 0.0005 ms | not_measured | managed | 0.200x |
| `np.random.beta(a, b) (float64)` | float64 | 1,000 | 0.0447 ms | not_measured | managed | 0.975x |
| `np.random.beta(a, b) (float64)` | float64 | 100,000 | 8.3629 ms | not_measured | managed | 0.489x |
| `np.random.binomial(n, p) (int64)` | int64 | 1,000 | 0.1082 ms | not_measured | managed | 0.217x |
| `np.random.binomial(n, p) (int64)` | int64 | 100,000 | 15.3358 ms | not_measured | managed | 0.153x |
| `np.random.chisquare(df) (float64)` | float64 | 1,000 | 0.0324 ms | not_measured | managed | 0.657x |
| `np.random.chisquare(df) (float64)` | float64 | 100,000 | 4.6899 ms | not_measured | managed | 0.455x |
| `np.random.choice(a, size) (float64)` | float64 | 1,000 | 0.0164 ms | not_measured | managed | 0.726x |
| `np.random.choice(a, size) (float64)` | float64 | 100,000 | 1.5563 ms | not_measured | managed | 0.602x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 1,000 | 0.1006 ms | not_measured | managed | 0.598x |
| `np.random.dirichlet(alpha, size) (float64)` | float64 | 100,000 | 9.4869 ms | not_measured | managed | 0.611x |
| `np.random.exponential(scale) (float64)` | float64 | 1,000 | 0.0368 ms | not_measured | managed | 0.253x |
| `np.random.exponential(scale) (float64)` | float64 | 100,000 | 2.6837 ms | not_measured | managed | 0.344x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 1,000 | 0.1231 ms | not_measured | managed | 0.354x |
| `np.random.f(dfnum, dfden) (float64)` | float64 | 100,000 | 10.0843 ms | not_measured | managed | 0.406x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 1,000 | 0.0502 ms | not_measured | managed | 0.416x |
| `np.random.gamma(shape, scale) (float64)` | float64 | 100,000 | 4.6816 ms | not_measured | managed | 0.449x |
| `np.random.geometric(p) (int64)` | int64 | 1,000 | 0.0195 ms | not_measured | managed | 0.933x |
| `np.random.geometric(p) (int64)` | int64 | 100,000 | 2.3824 ms | not_measured | managed | 0.775x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 1,000 | 0.0362 ms | not_measured | managed | 0.420x |
| `np.random.gumbel(loc, scale) (float64)` | float64 | 100,000 | 3.0331 ms | not_measured | managed | 0.462x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 1,000 | 0.1267 ms | not_measured | managed | 0.511x |
| `np.random.hypergeometric(ngood, nbad, nsample) (int64)` | int64 | 100,000 | 16.8501 ms | not_measured | managed | 0.394x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 1,000 | 0.0305 ms | not_measured | managed | 0.472x |
| `np.random.laplace(loc, scale) (float64)` | float64 | 100,000 | 2.6108 ms | not_measured | managed | 0.519x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 1,000 | 0.0240 ms | not_measured | managed | 0.396x |
| `np.random.logistic(loc, scale) (float64)` | float64 | 100,000 | 2.0894 ms | not_measured | managed | 0.419x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 1,000 | 0.0384 ms | not_measured | managed | 0.477x |
| `np.random.lognormal(mean, sigma) (float64)` | float64 | 100,000 | 3.4366 ms | not_measured | managed | 0.532x |
| `np.random.logseries(p) (int64)` | int64 | 1,000 | 0.0355 ms | not_measured | managed | 0.800x |
| `np.random.logseries(p) (int64)` | int64 | 100,000 | 4.7123 ms | not_measured | managed | 0.607x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 1,000 | 0.2210 ms | not_measured | managed | 0.327x |
| `np.random.multinomial(n, pvals, size) (int64)` | int64 | 100,000 | 21.9239 ms | not_measured | managed | 0.342x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 1,000 | 0.0882 ms | not_measured | managed | 0.712x |
| `np.random.multivariate_normal(mean, cov, size) (float64)` | float64 | 100,000 | 8.5216 ms | not_measured | managed | 0.553x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 1,000 | 0.1419 ms | not_measured | managed | 0.515x |
| `np.random.negative_binomial(n, p) (int64)` | int64 | 100,000 | 15.0325 ms | not_measured | managed | 0.492x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 1,000 | 0.0687 ms | not_measured | managed | 0.537x |
| `np.random.noncentral_chisquare(df, nonc) (float64)` | float64 | 100,000 | 6.8566 ms | not_measured | managed | 0.497x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 1,000 | 0.1044 ms | not_measured | managed | 0.533x |
| `np.random.noncentral_f(dfnum, dfden, nonc) (float64)` | float64 | 100,000 | 10.7901 ms | not_measured | managed | 0.500x |
| `np.random.normal(loc, scale) (float64)` | float64 | 1,000 | 0.0268 ms | not_measured | managed | 0.448x |
| `np.random.normal(loc, scale) (float64)` | float64 | 100,000 | 2.5071 ms | not_measured | managed | 0.452x |
| `np.random.pareto(a) (float64)` | float64 | 1,000 | 0.0346 ms | not_measured | managed | 0.439x |
| `np.random.pareto(a) (float64)` | float64 | 100,000 | 3.3190 ms | not_measured | managed | 0.466x |
| `np.random.permutation(a) (float64)` | float64 | 1,000 | 0.0206 ms | not_measured | managed | 0.699x |
| `np.random.permutation(a) (float64)` | float64 | 100,000 | 2.1285 ms | not_measured | managed | 0.668x |
| `np.random.poisson(lam) (int64)` | int64 | 1,000 | 0.0501 ms | not_measured | managed | 0.659x |
| `np.random.poisson(lam) (int64)` | int64 | 100,000 | 4.6583 ms | not_measured | managed | 0.730x |
| `np.random.power(a) (float64)` | float64 | 1,000 | 0.0335 ms | not_measured | managed | 0.869x |
| `np.random.power(a) (float64)` | float64 | 100,000 | 3.2747 ms | not_measured | managed | 0.902x |
| `np.random.rand(n) (float64)` | float64 | 1,000 | 0.0180 ms | not_measured | managed | 0.239x |
| `np.random.rand(n) (float64)` | float64 | 100,000 | 1.5895 ms | not_measured | managed | 0.236x |
| `np.random.randint(low, high) (int64)` | int64 | 1,000 | 0.0122 ms | not_measured | managed | 0.656x |
| `np.random.randint(low, high) (int64)` | int64 | 100,000 | 0.9905 ms | not_measured | managed | 0.470x |
| `np.random.randn(n) (float64)` | float64 | 1,000 | 0.0256 ms | not_measured | managed | 0.422x |
| `np.random.randn(n) (float64)` | float64 | 100,000 | 2.4820 ms | not_measured | managed | 0.431x |
| `np.random.random(n) (float64)` | float64 | 1,000 | 0.0168 ms | not_measured | managed | 0.250x |
| `np.random.random(n) (float64)` | float64 | 100,000 | 1.5210 ms | not_measured | managed | 0.239x |
| `np.random.random_sample(n) (float64)` | float64 | 1,000 | 0.0161 ms | not_measured | managed | 0.255x |
| `np.random.random_sample(n) (float64)` | float64 | 100,000 | 1.5336 ms | not_measured | managed | 0.238x |
| `np.random.rayleigh(scale) (float64)` | float64 | 1,000 | 0.0246 ms | not_measured | managed | 0.415x |
| `np.random.rayleigh(scale) (float64)` | float64 | 100,000 | 2.4364 ms | not_measured | managed | 0.404x |
| `np.random.shuffle(a) (float64)` | float64 | 1,000 | 0.0173 ms | not_measured | managed | 0.798x |
| `np.random.shuffle(a) (float64)` | float64 | 100,000 | 2.0284 ms | not_measured | managed | 0.698x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 1,000 | 0.0323 ms | not_measured | managed | 0.721x |
| `np.random.standard_cauchy(n) (float64)` | float64 | 100,000 | 3.0585 ms | not_measured | managed | 0.718x |
| `np.random.standard_exponential(n) (float64)` | float64 | 1,000 | 0.0224 ms | not_measured | managed | 0.397x |
| `np.random.standard_exponential(n) (float64)` | float64 | 100,000 | 2.3078 ms | not_measured | managed | 0.374x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 1,000 | 0.0443 ms | not_measured | managed | 0.479x |
| `np.random.standard_gamma(shape) (float64)` | float64 | 100,000 | 4.4169 ms | not_measured | managed | 0.471x |
| `np.random.standard_normal(n) (float64)` | float64 | 1,000 | 0.0252 ms | not_measured | managed | 0.433x |
| `np.random.standard_normal(n) (float64)` | float64 | 100,000 | 2.5524 ms | not_measured | managed | 0.416x |
| `np.random.standard_t(df) (float64)` | float64 | 1,000 | 0.0666 ms | not_measured | managed | 0.515x |
| `np.random.standard_t(df) (float64)` | float64 | 100,000 | 6.6590 ms | not_measured | managed | 0.511x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 1,000 | 0.0162 ms | not_measured | managed | 0.611x |
| `np.random.triangular(left, mode, right) (float64)` | float64 | 100,000 | 1.5733 ms | not_measured | managed | 0.563x |
| `np.random.uniform(low, high) (float64)` | float64 | 1,000 | 0.0145 ms | not_measured | managed | 0.359x |
| `np.random.uniform(low, high) (float64)` | float64 | 100,000 | 1.2691 ms | not_measured | managed | 0.294x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 1,000 | 0.0896 ms | not_measured | managed | 0.579x |
| `np.random.vonmises(mu, kappa) (float64)` | float64 | 100,000 | 9.3747 ms | not_measured | managed | 0.552x |
| `np.random.wald(mean, scale) (float64)` | float64 | 1,000 | 0.0494 ms | not_measured | managed | 0.502x |
| `np.random.wald(mean, scale) (float64)` | float64 | 100,000 | 4.5259 ms | not_measured | managed | 0.557x |
| `np.random.weibull(a) (float64)` | float64 | 1,000 | 0.0390 ms | not_measured | managed | 0.559x |
| `np.random.weibull(a) (float64)` | float64 | 100,000 | 2.4515 ms | not_measured | managed | 0.876x |
| `np.random.zipf(a) (int64)` | int64 | 1,000 | 0.0897 ms | not_measured | managed | 0.399x |
| `np.random.zipf(a) (int64)` | int64 | 100,000 | 8.4061 ms | not_measured | managed | 0.444x |
| `a.amax() [method] (complex128)` | complex128 | 1,000 | 0.0024 ms | not_measured | managed | 1.000x |
| `a.amax() [method] (complex128)` | complex128 | 100,000 | 0.1464 ms | not_measured | managed | 1.133x |
| `a.amax() [method] (complex128)` | complex128 | 10,000,000 | 21.4518 ms | not_measured | managed | 0.775x |
| `a.amax() [method] (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 1.348x |
| `a.amax() [method] (float16)` | float16 | 100,000 | 0.4638 ms | not_measured | managed | 1.105x |
| `a.amax() [method] (float16)` | float16 | 10,000,000 | 51.3650 ms | not_measured | managed | 0.974x |
| `a.amax() [method] (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amax() [method] (float32)` | float32 | 100,000 | 0.0092 ms | not_measured | managed | 0.576x |
| `a.amax() [method] (float32)` | float32 | 10,000,000 | 4.4086 ms | not_measured | managed | 0.346x |
| `a.amax() [method] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.786x |
| `a.amax() [method] (float64)` | float64 | 100,000 | 0.0184 ms | not_measured | managed | 0.538x |
| `a.amax() [method] (float64)` | float64 | 10,000,000 | 8.8355 ms | not_measured | managed | 0.376x |
| `a.amax() [method] (int16)` | int16 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `a.amax() [method] (int16)` | int16 | 100,000 | 0.0028 ms | not_measured | managed | 0.893x |
| `a.amax() [method] (int16)` | int16 | 10,000,000 | 0.8340 ms | not_measured | managed | 0.409x |
| `a.amax() [method] (int32)` | int32 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.amax() [method] (int32)` | int32 | 100,000 | 0.0050 ms | not_measured | managed | 0.700x |
| `a.amax() [method] (int32)` | int32 | 10,000,000 | 3.9705 ms | not_measured | managed | 0.323x |
| `a.amax() [method] (int64)` | int64 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.amax() [method] (int64)` | int64 | 100,000 | 0.0294 ms | not_measured | managed | 0.282x |
| `a.amax() [method] (int64)` | int64 | 10,000,000 | 9.3032 ms | not_measured | managed | 0.346x |
| `a.amax() [method] (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a.amax() [method] (int8)` | int8 | 100,000 | 0.0024 ms | not_measured | managed | 0.708x |
| `a.amax() [method] (int8)` | int8 | 10,000,000 | 0.2420 ms | not_measured | managed | 0.728x |
| `a.amax() [method] (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `a.amax() [method] (uint16)` | uint16 | 100,000 | 0.0027 ms | not_measured | managed | 0.889x |
| `a.amax() [method] (uint16)` | uint16 | 10,000,000 | 1.0649 ms | not_measured | managed | 0.280x |
| `a.amax() [method] (uint32)` | uint32 | 1,000 | 0.0016 ms | not_measured | managed | 0.562x |
| `a.amax() [method] (uint32)` | uint32 | 100,000 | 0.0059 ms | not_measured | managed | 0.627x |
| `a.amax() [method] (uint32)` | uint32 | 10,000,000 | 3.8872 ms | not_measured | managed | 0.391x |
| `a.amax() [method] (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 0.556x |
| `a.amax() [method] (uint64)` | uint64 | 100,000 | 0.0402 ms | not_measured | managed | 0.286x |
| `a.amax() [method] (uint64)` | uint64 | 10,000,000 | 8.5777 ms | not_measured | managed | 0.436x |
| `a.amax() [method] (uint8)` | uint8 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `a.amax() [method] (uint8)` | uint8 | 100,000 | 0.0023 ms | not_measured | managed | 0.739x |
| `a.amax() [method] (uint8)` | uint8 | 10,000,000 | 0.2554 ms | not_measured | managed | 0.493x |
| `a.amin() [method] (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.043x |
| `a.amin() [method] (complex128)` | complex128 | 100,000 | 0.1493 ms | not_measured | managed | 1.078x |
| `a.amin() [method] (complex128)` | complex128 | 10,000,000 | 28.2649 ms | not_measured | managed | 0.587x |
| `a.amin() [method] (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 1.375x |
| `a.amin() [method] (float16)` | float16 | 100,000 | 0.4275 ms | not_measured | managed | 1.212x |
| `a.amin() [method] (float16)` | float16 | 10,000,000 | 43.8867 ms | not_measured | managed | 1.182x |
| `a.amin() [method] (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.amin() [method] (float32)` | float32 | 100,000 | 0.0096 ms | not_measured | managed | 0.562x |
| `a.amin() [method] (float32)` | float32 | 10,000,000 | 4.6412 ms | not_measured | managed | 0.373x |
| `a.amin() [method] (float64)` | float64 | 1,000 | 0.0009 ms | not_measured | managed | 1.111x |
| `a.amin() [method] (float64)` | float64 | 100,000 | 0.0200 ms | not_measured | managed | 0.480x |
| `a.amin() [method] (float64)` | float64 | 10,000,000 | 9.5190 ms | not_measured | managed | 0.368x |
| `a.amin() [method] (int16)` | int16 | 1,000 | 0.0015 ms | not_measured | managed | 0.600x |
| `a.amin() [method] (int16)` | int16 | 100,000 | 0.0027 ms | not_measured | managed | 0.889x |
| `a.amin() [method] (int16)` | int16 | 10,000,000 | 1.0047 ms | not_measured | managed | 0.340x |
| `a.amin() [method] (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `a.amin() [method] (int32)` | int32 | 100,000 | 0.0048 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (int32)` | int32 | 10,000,000 | 4.3261 ms | not_measured | managed | 0.281x |
| `a.amin() [method] (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 0.909x |
| `a.amin() [method] (int64)` | int64 | 100,000 | 0.0292 ms | not_measured | managed | 0.325x |
| `a.amin() [method] (int64)` | int64 | 10,000,000 | 10.6777 ms | not_measured | managed | 0.322x |
| `a.amin() [method] (int8)` | int8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (int8)` | int8 | 100,000 | 0.0018 ms | not_measured | managed | 0.944x |
| `a.amin() [method] (int8)` | int8 | 10,000,000 | 0.2590 ms | not_measured | managed | 0.727x |
| `a.amin() [method] (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (uint16)` | uint16 | 100,000 | 0.0027 ms | not_measured | managed | 0.926x |
| `a.amin() [method] (uint16)` | uint16 | 10,000,000 | 1.0238 ms | not_measured | managed | 0.297x |
| `a.amin() [method] (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `a.amin() [method] (uint32)` | uint32 | 100,000 | 0.0055 ms | not_measured | managed | 0.691x |
| `a.amin() [method] (uint32)` | uint32 | 10,000,000 | 5.0412 ms | not_measured | managed | 0.268x |
| `a.amin() [method] (uint64)` | uint64 | 1,000 | 0.0018 ms | not_measured | managed | 0.556x |
| `a.amin() [method] (uint64)` | uint64 | 100,000 | 0.0396 ms | not_measured | managed | 0.303x |
| `a.amin() [method] (uint64)` | uint64 | 10,000,000 | 11.9088 ms | not_measured | managed | 0.403x |
| `a.amin() [method] (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 0.750x |
| `a.amin() [method] (uint8)` | uint8 | 100,000 | 0.0026 ms | not_measured | managed | 0.654x |
| `a.amin() [method] (uint8)` | uint8 | 10,000,000 | 0.2560 ms | not_measured | managed | 0.586x |
| `a.mean() [method] (complex128)` | complex128 | 1,000 | 0.0017 ms | not_measured | managed | 1.353x |
| `a.mean() [method] (complex128)` | complex128 | 100,000 | 0.0108 ms | not_measured | managed | 3.231x |
| `a.mean() [method] (complex128)` | complex128 | 10,000,000 | 20.4819 ms | not_measured | managed | 0.399x |
| `a.mean() [method] (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 2.091x |
| `a.mean() [method] (float16)` | float16 | 100,000 | 0.0849 ms | not_measured | managed | 1.245x |
| `a.mean() [method] (float16)` | float16 | 10,000,000 | 17.3424 ms | not_measured | managed | 0.635x |
| `a.mean() [method] (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 1.391x |
| `a.mean() [method] (float32)` | float32 | 100,000 | 0.0034 ms | not_measured | managed | 5.324x |
| `a.mean() [method] (float32)` | float32 | 10,000,000 | 4.7879 ms | not_measured | managed | 0.650x |
| `a.mean() [method] (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 1.111x |
| `a.mean() [method] (float64)` | float64 | 100,000 | 0.0042 ms | not_measured | managed | 4.595x |
| `a.mean() [method] (float64)` | float64 | 10,000,000 | 9.1760 ms | not_measured | managed | 0.527x |
| `a.mean() [method] (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 1.929x |
| `a.mean() [method] (int16)` | int16 | 100,000 | 0.0197 ms | not_measured | managed | 2.635x |
| `a.mean() [method] (int16)` | int16 | 10,000,000 | 2.1297 ms | not_measured | managed | 2.576x |
| `a.mean() [method] (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 1.688x |
| `a.mean() [method] (int32)` | int32 | 100,000 | 0.0195 ms | not_measured | managed | 2.333x |
| `a.mean() [method] (int32)` | int32 | 10,000,000 | 3.0728 ms | not_measured | managed | 1.565x |
| `a.mean() [method] (int64)` | int64 | 1,000 | 0.0017 ms | not_measured | managed | 1.529x |
| `a.mean() [method] (int64)` | int64 | 100,000 | 0.0047 ms | not_measured | managed | 7.234x |
| `a.mean() [method] (int64)` | int64 | 10,000,000 | 7.9461 ms | not_measured | managed | 0.817x |
| `a.mean() [method] (int8)` | int8 | 1,000 | 0.0019 ms | not_measured | managed | 1.474x |
| `a.mean() [method] (int8)` | int8 | 100,000 | 0.0471 ms | not_measured | managed | 1.127x |
| `a.mean() [method] (int8)` | int8 | 10,000,000 | 1.8881 ms | not_measured | managed | 3.210x |
| `a.mean() [method] (uint16)` | uint16 | 1,000 | 0.0019 ms | not_measured | managed | 1.421x |
| `a.mean() [method] (uint16)` | uint16 | 100,000 | 0.0236 ms | not_measured | managed | 2.212x |
| `a.mean() [method] (uint16)` | uint16 | 10,000,000 | 2.2713 ms | not_measured | managed | 2.210x |
| `a.mean() [method] (uint32)` | uint32 | 1,000 | 0.0019 ms | not_measured | managed | 1.368x |
| `a.mean() [method] (uint32)` | uint32 | 100,000 | 0.0198 ms | not_measured | managed | 2.030x |
| `a.mean() [method] (uint32)` | uint32 | 10,000,000 | 3.0409 ms | not_measured | managed | 1.530x |
| `a.mean() [method] (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 1.562x |
| `a.mean() [method] (uint64)` | uint64 | 100,000 | 0.0046 ms | not_measured | managed | 11.543x |
| `a.mean() [method] (uint64)` | uint64 | 10,000,000 | 8.9506 ms | not_measured | managed | 0.825x |
| `a.mean() [method] (uint8)` | uint8 | 1,000 | 0.0020 ms | not_measured | managed | 1.350x |
| `a.mean() [method] (uint8)` | uint8 | 100,000 | 0.0190 ms | not_measured | managed | 3.074x |
| `a.mean() [method] (uint8)` | uint8 | 10,000,000 | 1.9110 ms | not_measured | managed | 2.619x |
| `a.std() [method] (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 4.068x |
| `a.std() [method] (float16)` | float16 | 100,000 | 0.3994 ms | not_measured | managed | 2.231x |
| `a.std() [method] (float16)` | float16 | 10,000,000 | 39.2848 ms | not_measured | managed | 2.342x |
| `a.std() [method] (float32)` | float32 | 1,000 | 0.0015 ms | not_measured | managed | 5.333x |
| `a.std() [method] (float32)` | float32 | 100,000 | 0.0202 ms | not_measured | managed | 2.411x |
| `a.std() [method] (float32)` | float32 | 10,000,000 | 9.8328 ms | not_measured | managed | 1.780x |
| `a.std() [method] (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 3.500x |
| `a.std() [method] (float64)` | float64 | 100,000 | 0.0377 ms | not_measured | managed | 1.714x |
| `a.std() [method] (float64)` | float64 | 10,000,000 | 18.0062 ms | not_measured | managed | 1.748x |
| `a.sum() [method] (complex128)` | complex128 | 1,000 | 0.0015 ms | not_measured | managed | 0.867x |
| `a.sum() [method] (complex128)` | complex128 | 100,000 | 0.0269 ms | not_measured | managed | 1.309x |
| `a.sum() [method] (complex128)` | complex128 | 10,000,000 | 17.2355 ms | not_measured | managed | 0.482x |
| `a.sum() [method] (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 1.933x |
| `a.sum() [method] (float16)` | float16 | 100,000 | 0.0684 ms | not_measured | managed | 2.927x |
| `a.sum() [method] (float16)` | float16 | 10,000,000 | 7.5091 ms | not_measured | managed | 2.677x |
| `a.sum() [method] (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `a.sum() [method] (float32)` | float32 | 100,000 | 0.0052 ms | not_measured | managed | 2.962x |
| `a.sum() [method] (float32)` | float32 | 10,000,000 | 4.3007 ms | not_measured | managed | 0.750x |
| `a.sum() [method] (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 0.833x |
| `a.sum() [method] (float64)` | float64 | 100,000 | 0.0096 ms | not_measured | managed | 1.729x |
| `a.sum() [method] (float64)` | float64 | 10,000,000 | 8.2028 ms | not_measured | managed | 0.637x |
| `a.sum() [method] (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 1.071x |
| `a.sum() [method] (int16)` | int16 | 100,000 | 0.0457 ms | not_measured | managed | 0.729x |
| `a.sum() [method] (int16)` | int16 | 10,000,000 | 4.8288 ms | not_measured | managed | 0.754x |
| `a.sum() [method] (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 1.231x |
| `a.sum() [method] (int32)` | int32 | 100,000 | 0.0470 ms | not_measured | managed | 0.721x |
| `a.sum() [method] (int32)` | int32 | 10,000,000 | 5.9087 ms | not_measured | managed | 0.772x |
| `a.sum() [method] (int64)` | int64 | 1,000 | 0.0009 ms | not_measured | managed | 1.222x |
| `a.sum() [method] (int64)` | int64 | 100,000 | 0.0092 ms | not_measured | managed | 2.022x |
| `a.sum() [method] (int64)` | int64 | 10,000,000 | 8.3188 ms | not_measured | managed | 0.588x |
| `a.sum() [method] (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 1.000x |
| `a.sum() [method] (int8)` | int8 | 100,000 | 0.0463 ms | not_measured | managed | 0.698x |
| `a.sum() [method] (int8)` | int8 | 10,000,000 | 4.9438 ms | not_measured | managed | 0.676x |
| `a.sum() [method] (uint16)` | uint16 | 1,000 | 0.0012 ms | not_measured | managed | 1.167x |
| `a.sum() [method] (uint16)` | uint16 | 100,000 | 0.0429 ms | not_measured | managed | 0.755x |
| `a.sum() [method] (uint16)` | uint16 | 10,000,000 | 4.5945 ms | not_measured | managed | 0.720x |
| `a.sum() [method] (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 1.077x |
| `a.sum() [method] (uint32)` | uint32 | 100,000 | 0.0421 ms | not_measured | managed | 0.760x |
| `a.sum() [method] (uint32)` | uint32 | 10,000,000 | 5.8905 ms | not_measured | managed | 0.746x |
| `a.sum() [method] (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `a.sum() [method] (uint64)` | uint64 | 100,000 | 0.0096 ms | not_measured | managed | 1.885x |
| `a.sum() [method] (uint64)` | uint64 | 10,000,000 | 8.2856 ms | not_measured | managed | 0.499x |
| `a.sum() [method] (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 1.154x |
| `a.sum() [method] (uint8)` | uint8 | 100,000 | 0.0424 ms | not_measured | managed | 0.823x |
| `a.sum() [method] (uint8)` | uint8 | 10,000,000 | 3.9977 ms | not_measured | managed | 0.785x |
| `a.var() [method] (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 3.933x |
| `a.var() [method] (float16)` | float16 | 100,000 | 0.3833 ms | not_measured | managed | 2.352x |
| `a.var() [method] (float16)` | float16 | 10,000,000 | 40.0156 ms | not_measured | managed | 2.236x |
| `a.var() [method] (float32)` | float32 | 1,000 | 0.0011 ms | not_measured | managed | 6.636x |
| `a.var() [method] (float32)` | float32 | 100,000 | 0.0202 ms | not_measured | managed | 2.530x |
| `a.var() [method] (float32)` | float32 | 10,000,000 | 9.3164 ms | not_measured | managed | 1.842x |
| `a.var() [method] (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 4.214x |
| `a.var() [method] (float64)` | float64 | 100,000 | 0.0381 ms | not_measured | managed | 1.798x |
| `a.var() [method] (float64)` | float64 | 10,000,000 | 17.5915 ms | not_measured | managed | 1.849x |
| `np.amax (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.391x |
| `np.amax (complex128)` | complex128 | 100,000 | 0.1444 ms | not_measured | managed | 1.210x |
| `np.amax (complex128)` | complex128 | 10,000,000 | 22.8853 ms | not_measured | managed | 0.715x |
| `np.amax (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 1.667x |
| `np.amax (float16)` | float16 | 100,000 | 0.4720 ms | not_measured | managed | 1.070x |
| `np.amax (float16)` | float16 | 10,000,000 | 46.6954 ms | not_measured | managed | 1.071x |
| `np.amax (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.947x |
| `np.amax (float32)` | float32 | 100,000 | 0.0098 ms | not_measured | managed | 0.633x |
| `np.amax (float32)` | float32 | 10,000,000 | 4.3040 ms | not_measured | managed | 0.356x |
| `np.amax (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.286x |
| `np.amax (float64)` | float64 | 100,000 | 0.0184 ms | not_measured | managed | 0.571x |
| `np.amax (float64)` | float64 | 10,000,000 | 8.3981 ms | not_measured | managed | 0.399x |
| `np.amax (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 1.308x |
| `np.amax (int16)` | int16 | 100,000 | 0.0025 ms | not_measured | managed | 1.280x |
| `np.amax (int16)` | int16 | 10,000,000 | 1.1386 ms | not_measured | managed | 0.292x |
| `np.amax (int32)` | int32 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.amax (int32)` | int32 | 100,000 | 0.0044 ms | not_measured | managed | 1.023x |
| `np.amax (int32)` | int32 | 10,000,000 | 4.0828 ms | not_measured | managed | 0.320x |
| `np.amax (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 1.636x |
| `np.amax (int64)` | int64 | 100,000 | 0.0291 ms | not_measured | managed | 0.309x |
| `np.amax (int64)` | int64 | 10,000,000 | 10.7917 ms | not_measured | managed | 0.293x |
| `np.amax (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 1.889x |
| `np.amax (int8)` | int8 | 100,000 | 0.0024 ms | not_measured | managed | 1.042x |
| `np.amax (int8)` | int8 | 10,000,000 | 0.2510 ms | not_measured | managed | 0.686x |
| `np.amax (uint16)` | uint16 | 1,000 | 0.0015 ms | not_measured | managed | 1.533x |
| `np.amax (uint16)` | uint16 | 100,000 | 0.0026 ms | not_measured | managed | 1.192x |
| `np.amax (uint16)` | uint16 | 10,000,000 | 1.5487 ms | not_measured | managed | 0.191x |
| `np.amax (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `np.amax (uint32)` | uint32 | 100,000 | 0.0050 ms | not_measured | managed | 0.860x |
| `np.amax (uint32)` | uint32 | 10,000,000 | 4.0437 ms | not_measured | managed | 0.337x |
| `np.amax (uint64)` | uint64 | 1,000 | 0.0020 ms | not_measured | managed | 0.900x |
| `np.amax (uint64)` | uint64 | 100,000 | 0.0376 ms | not_measured | managed | 0.330x |
| `np.amax (uint64)` | uint64 | 10,000,000 | 10.7815 ms | not_measured | managed | 0.342x |
| `np.amax (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `np.amax (uint8)` | uint8 | 100,000 | 0.0023 ms | not_measured | managed | 1.043x |
| `np.amax (uint8)` | uint8 | 10,000,000 | 0.2588 ms | not_measured | managed | 0.508x |
| `np.amax axis=0 (complex128)` | complex128 | 1,000 | 0.0043 ms | not_measured | managed | 0.744x |
| `np.amax axis=0 (complex128)` | complex128 | 100,000 | 0.1972 ms | not_measured | managed | 0.766x |
| `np.amax axis=0 (complex128)` | complex128 | 10,000,000 | 28.8413 ms | not_measured | managed | 0.575x |
| `np.amax axis=0 (float16)` | float16 | 1,000 | 0.0036 ms | not_measured | managed | 1.194x |
| `np.amax axis=0 (float16)` | float16 | 100,000 | 0.6869 ms | not_measured | managed | 0.754x |
| `np.amax axis=0 (float16)` | float16 | 10,000,000 | 142.5118 ms | not_measured | managed | 0.360x |
| `np.amax axis=0 (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 1.278x |
| `np.amax axis=0 (float32)` | float32 | 100,000 | 0.0297 ms | not_measured | managed | 0.330x |
| `np.amax axis=0 (float32)` | float32 | 10,000,000 | 4.4998 ms | not_measured | managed | 0.499x |
| `np.amax axis=0 (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 1.150x |
| `np.amax axis=0 (float64)` | float64 | 100,000 | 0.0594 ms | not_measured | managed | 0.258x |
| `np.amax axis=0 (float64)` | float64 | 10,000,000 | 10.5427 ms | not_measured | managed | 0.362x |
| `np.amax axis=0 (int16)` | int16 | 1,000 | 0.0011 ms | not_measured | managed | 2.000x |
| `np.amax axis=0 (int16)` | int16 | 100,000 | 0.0076 ms | not_measured | managed | 0.934x |
| `np.amax axis=0 (int16)` | int16 | 10,000,000 | 1.8156 ms | not_measured | managed | 0.275x |
| `np.amax axis=0 (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 2.200x |
| `np.amax axis=0 (int32)` | int32 | 100,000 | 0.0115 ms | not_measured | managed | 0.896x |
| `np.amax axis=0 (int32)` | int32 | 10,000,000 | 4.4742 ms | not_measured | managed | 0.379x |
| `np.amax axis=0 (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 1.048x |
| `np.amax axis=0 (int64)` | int64 | 100,000 | 0.0305 ms | not_measured | managed | 0.505x |
| `np.amax axis=0 (int64)` | int64 | 10,000,000 | 9.6665 ms | not_measured | managed | 0.437x |
| `np.amax axis=0 (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.amax axis=0 (int8)` | int8 | 100,000 | 0.0080 ms | not_measured | managed | 0.963x |
| `np.amax axis=0 (int8)` | int8 | 10,000,000 | 0.3485 ms | not_measured | managed | 0.711x |
| `np.amax axis=0 (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 2.444x |
| `np.amax axis=0 (uint16)` | uint16 | 100,000 | 0.0074 ms | not_measured | managed | 0.959x |
| `np.amax axis=0 (uint16)` | uint16 | 10,000,000 | 2.5225 ms | not_measured | managed | 0.199x |
| `np.amax axis=0 (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 1.467x |
| `np.amax axis=0 (uint32)` | uint32 | 100,000 | 0.0107 ms | not_measured | managed | 0.907x |
| `np.amax axis=0 (uint32)` | uint32 | 10,000,000 | 4.7594 ms | not_measured | managed | 0.322x |
| `np.amax axis=0 (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 1.095x |
| `np.amax axis=0 (uint64)` | uint64 | 100,000 | 0.0401 ms | not_measured | managed | 0.454x |
| `np.amax axis=0 (uint64)` | uint64 | 10,000,000 | 9.4798 ms | not_measured | managed | 0.496x |
| `np.amax axis=0 (uint8)` | uint8 | 1,000 | 0.0014 ms | not_measured | managed | 1.571x |
| `np.amax axis=0 (uint8)` | uint8 | 100,000 | 0.0076 ms | not_measured | managed | 1.079x |
| `np.amax axis=0 (uint8)` | uint8 | 10,000,000 | 0.3569 ms | not_measured | managed | 0.610x |
| `np.amin (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.435x |
| `np.amin (complex128)` | complex128 | 100,000 | 0.1506 ms | not_measured | managed | 1.023x |
| `np.amin (complex128)` | complex128 | 10,000,000 | 24.5960 ms | not_measured | managed | 0.675x |
| `np.amin (float16)` | float16 | 1,000 | 0.0025 ms | not_measured | managed | 1.720x |
| `np.amin (float16)` | float16 | 100,000 | 0.4218 ms | not_measured | managed | 1.202x |
| `np.amin (float16)` | float16 | 10,000,000 | 44.1931 ms | not_measured | managed | 1.202x |
| `np.amin (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 1.118x |
| `np.amin (float32)` | float32 | 100,000 | 0.0096 ms | not_measured | managed | 0.656x |
| `np.amin (float32)` | float32 | 10,000,000 | 4.9684 ms | not_measured | managed | 0.318x |
| `np.amin (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.214x |
| `np.amin (float64)` | float64 | 100,000 | 0.0195 ms | not_measured | managed | 0.554x |
| `np.amin (float64)` | float64 | 10,000,000 | 9.3097 ms | not_measured | managed | 0.412x |
| `np.amin (int16)` | int16 | 1,000 | 0.0010 ms | not_measured | managed | 1.700x |
| `np.amin (int16)` | int16 | 100,000 | 0.0033 ms | not_measured | managed | 1.091x |
| `np.amin (int16)` | int16 | 10,000,000 | 0.9468 ms | not_measured | managed | 0.382x |
| `np.amin (int32)` | int32 | 1,000 | 0.0014 ms | not_measured | managed | 1.357x |
| `np.amin (int32)` | int32 | 100,000 | 0.0051 ms | not_measured | managed | 0.902x |
| `np.amin (int32)` | int32 | 10,000,000 | 3.9885 ms | not_measured | managed | 0.319x |
| `np.amin (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.amin (int64)` | int64 | 100,000 | 0.0301 ms | not_measured | managed | 0.355x |
| `np.amin (int64)` | int64 | 10,000,000 | 10.0397 ms | not_measured | managed | 0.371x |
| `np.amin (int8)` | int8 | 1,000 | 0.0008 ms | not_measured | managed | 2.250x |
| `np.amin (int8)` | int8 | 100,000 | 0.0026 ms | not_measured | managed | 1.154x |
| `np.amin (int8)` | int8 | 10,000,000 | 0.2617 ms | not_measured | managed | 0.760x |
| `np.amin (uint16)` | uint16 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.amin (uint16)` | uint16 | 100,000 | 0.0028 ms | not_measured | managed | 1.250x |
| `np.amin (uint16)` | uint16 | 10,000,000 | 0.9080 ms | not_measured | managed | 0.341x |
| `np.amin (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 1.133x |
| `np.amin (uint32)` | uint32 | 100,000 | 0.0051 ms | not_measured | managed | 0.941x |
| `np.amin (uint32)` | uint32 | 10,000,000 | 4.1891 ms | not_measured | managed | 0.347x |
| `np.amin (uint64)` | uint64 | 1,000 | 0.0021 ms | not_measured | managed | 0.952x |
| `np.amin (uint64)` | uint64 | 100,000 | 0.0390 ms | not_measured | managed | 0.344x |
| `np.amin (uint64)` | uint64 | 10,000,000 | 10.2403 ms | not_measured | managed | 0.369x |
| `np.amin (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `np.amin (uint8)` | uint8 | 100,000 | 0.0019 ms | not_measured | managed | 1.526x |
| `np.amin (uint8)` | uint8 | 10,000,000 | 0.2573 ms | not_measured | managed | 0.588x |
| `np.amin axis=0 (complex128)` | complex128 | 1,000 | 0.0045 ms | not_measured | managed | 0.689x |
| `np.amin axis=0 (complex128)` | complex128 | 100,000 | 0.2391 ms | not_measured | managed | 0.622x |
| `np.amin axis=0 (complex128)` | complex128 | 10,000,000 | 32.8407 ms | not_measured | managed | 0.571x |
| `np.amin axis=0 (float16)` | float16 | 1,000 | 0.0036 ms | not_measured | managed | 1.222x |
| `np.amin axis=0 (float16)` | float16 | 100,000 | 0.7042 ms | not_measured | managed | 0.685x |
| `np.amin axis=0 (float16)` | float16 | 10,000,000 | 157.0288 ms | not_measured | managed | 0.331x |
| `np.amin axis=0 (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.769x |
| `np.amin axis=0 (float32)` | float32 | 100,000 | 0.0310 ms | not_measured | managed | 0.319x |
| `np.amin axis=0 (float32)` | float32 | 10,000,000 | 5.3184 ms | not_measured | managed | 0.385x |
| `np.amin axis=0 (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 1.533x |
| `np.amin axis=0 (float64)` | float64 | 100,000 | 0.0590 ms | not_measured | managed | 0.242x |
| `np.amin axis=0 (float64)` | float64 | 10,000,000 | 13.6506 ms | not_measured | managed | 0.285x |
| `np.amin axis=0 (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 1.692x |
| `np.amin axis=0 (int16)` | int16 | 100,000 | 0.0076 ms | not_measured | managed | 1.039x |
| `np.amin axis=0 (int16)` | int16 | 10,000,000 | 1.7162 ms | not_measured | managed | 0.336x |
| `np.amin axis=0 (int32)` | int32 | 1,000 | 0.0017 ms | not_measured | managed | 1.294x |
| `np.amin axis=0 (int32)` | int32 | 100,000 | 0.0111 ms | not_measured | managed | 1.027x |
| `np.amin axis=0 (int32)` | int32 | 10,000,000 | 4.2881 ms | not_measured | managed | 0.423x |
| `np.amin axis=0 (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 1.692x |
| `np.amin axis=0 (int64)` | int64 | 100,000 | 0.0309 ms | not_measured | managed | 0.540x |
| `np.amin axis=0 (int64)` | int64 | 10,000,000 | 11.7529 ms | not_measured | managed | 0.378x |
| `np.amin axis=0 (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.amin axis=0 (int8)` | int8 | 100,000 | 0.0086 ms | not_measured | managed | 0.895x |
| `np.amin axis=0 (int8)` | int8 | 10,000,000 | 0.3936 ms | not_measured | managed | 0.652x |
| `np.amin axis=0 (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.amin axis=0 (uint16)` | uint16 | 100,000 | 0.0075 ms | not_measured | managed | 1.080x |
| `np.amin axis=0 (uint16)` | uint16 | 10,000,000 | 1.4658 ms | not_measured | managed | 0.355x |
| `np.amin axis=0 (uint32)` | uint32 | 1,000 | 0.0012 ms | not_measured | managed | 1.833x |
| `np.amin axis=0 (uint32)` | uint32 | 100,000 | 0.0117 ms | not_measured | managed | 0.889x |
| `np.amin axis=0 (uint32)` | uint32 | 10,000,000 | 5.4079 ms | not_measured | managed | 0.283x |
| `np.amin axis=0 (uint64)` | uint64 | 1,000 | 0.0017 ms | not_measured | managed | 1.294x |
| `np.amin axis=0 (uint64)` | uint64 | 100,000 | 0.0394 ms | not_measured | managed | 0.472x |
| `np.amin axis=0 (uint64)` | uint64 | 10,000,000 | 9.5983 ms | not_measured | managed | 0.498x |
| `np.amin axis=0 (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 1.833x |
| `np.amin axis=0 (uint8)` | uint8 | 100,000 | 0.0084 ms | not_measured | managed | 0.905x |
| `np.amin axis=0 (uint8)` | uint8 | 10,000,000 | 0.3645 ms | not_measured | managed | 0.565x |
| `np.argmax (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 0.826x |
| `np.argmax (complex128)` | complex128 | 100,000 | 0.1489 ms | not_measured | managed | 0.787x |
| `np.argmax (complex128)` | complex128 | 10,000,000 | 23.2878 ms | not_measured | managed | 0.596x |
| `np.argmax (float16)` | float16 | 1,000 | 0.0031 ms | not_measured | managed | 0.935x |
| `np.argmax (float16)` | float16 | 100,000 | 0.2223 ms | not_measured | managed | 2.005x |
| `np.argmax (float16)` | float16 | 10,000,000 | 22.7083 ms | not_measured | managed | 1.921x |
| `np.argmax (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmax (float32)` | float32 | 100,000 | 0.0581 ms | not_measured | managed | 0.150x |
| `np.argmax (float32)` | float32 | 10,000,000 | 7.7719 ms | not_measured | managed | 0.290x |
| `np.argmax (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 0.714x |
| `np.argmax (float64)` | float64 | 100,000 | 0.0601 ms | not_measured | managed | 0.316x |
| `np.argmax (float64)` | float64 | 10,000,000 | 14.5839 ms | not_measured | managed | 0.289x |
| `np.argmax (int16)` | int16 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `np.argmax (int16)` | int16 | 100,000 | 0.0033 ms | not_measured | managed | 1.212x |
| `np.argmax (int16)` | int16 | 10,000,000 | 2.5907 ms | not_measured | managed | 0.216x |
| `np.argmax (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmax (int32)` | int32 | 100,000 | 0.0049 ms | not_measured | managed | 1.306x |
| `np.argmax (int32)` | int32 | 10,000,000 | 4.1916 ms | not_measured | managed | 0.430x |
| `np.argmax (int64)` | int64 | 1,000 | 0.0015 ms | not_measured | managed | 0.667x |
| `np.argmax (int64)` | int64 | 100,000 | 0.0527 ms | not_measured | managed | 0.292x |
| `np.argmax (int64)` | int64 | 10,000,000 | 10.0642 ms | not_measured | managed | 0.409x |
| `np.argmax (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 0.769x |
| `np.argmax (int8)` | int8 | 100,000 | 0.0019 ms | not_measured | managed | 1.316x |
| `np.argmax (int8)` | int8 | 10,000,000 | 0.2553 ms | not_measured | managed | 1.177x |
| `np.argmax (uint16)` | uint16 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.argmax (uint16)` | uint16 | 100,000 | 0.0032 ms | not_measured | managed | 1.562x |
| `np.argmax (uint16)` | uint16 | 10,000,000 | 2.5783 ms | not_measured | managed | 0.209x |
| `np.argmax (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmax (uint32)` | uint32 | 100,000 | 0.0058 ms | not_measured | managed | 1.517x |
| `np.argmax (uint32)` | uint32 | 10,000,000 | 4.4120 ms | not_measured | managed | 0.517x |
| `np.argmax (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.argmax (uint64)` | uint64 | 100,000 | 0.0623 ms | not_measured | managed | 0.278x |
| `np.argmax (uint64)` | uint64 | 10,000,000 | 10.7633 ms | not_measured | managed | 0.446x |
| `np.argmax (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `np.argmax (uint8)` | uint8 | 100,000 | 0.0026 ms | not_measured | managed | 1.192x |
| `np.argmax (uint8)` | uint8 | 10,000,000 | 0.2548 ms | not_measured | managed | 0.862x |
| `np.argmin (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 0.826x |
| `np.argmin (complex128)` | complex128 | 100,000 | 0.1525 ms | not_measured | managed | 0.760x |
| `np.argmin (complex128)` | complex128 | 10,000,000 | 23.6089 ms | not_measured | managed | 0.585x |
| `np.argmin (float16)` | float16 | 1,000 | 0.0033 ms | not_measured | managed | 0.939x |
| `np.argmin (float16)` | float16 | 100,000 | 0.2365 ms | not_measured | managed | 1.853x |
| `np.argmin (float16)` | float16 | 10,000,000 | 23.9573 ms | not_measured | managed | 1.796x |
| `np.argmin (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `np.argmin (float32)` | float32 | 100,000 | 0.0605 ms | not_measured | managed | 0.145x |
| `np.argmin (float32)` | float32 | 10,000,000 | 7.6342 ms | not_measured | managed | 0.287x |
| `np.argmin (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.argmin (float64)` | float64 | 100,000 | 0.0602 ms | not_measured | managed | 0.277x |
| `np.argmin (float64)` | float64 | 10,000,000 | 14.1938 ms | not_measured | managed | 0.307x |
| `np.argmin (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 0.450x |
| `np.argmin (int16)` | int16 | 100,000 | 0.0033 ms | not_measured | managed | 1.121x |
| `np.argmin (int16)` | int16 | 10,000,000 | 1.1599 ms | not_measured | managed | 0.516x |
| `np.argmin (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 0.900x |
| `np.argmin (int32)` | int32 | 100,000 | 0.0054 ms | not_measured | managed | 1.074x |
| `np.argmin (int32)` | int32 | 10,000,000 | 4.0627 ms | not_measured | managed | 0.453x |
| `np.argmin (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `np.argmin (int64)` | int64 | 100,000 | 0.0541 ms | not_measured | managed | 0.270x |
| `np.argmin (int64)` | int64 | 10,000,000 | 13.9466 ms | not_measured | managed | 0.303x |
| `np.argmin (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 0.643x |
| `np.argmin (int8)` | int8 | 100,000 | 0.0020 ms | not_measured | managed | 1.150x |
| `np.argmin (int8)` | int8 | 10,000,000 | 0.2587 ms | not_measured | managed | 1.242x |
| `np.argmin (uint16)` | uint16 | 1,000 | 0.0018 ms | not_measured | managed | 0.500x |
| `np.argmin (uint16)` | uint16 | 100,000 | 0.0032 ms | not_measured | managed | 1.594x |
| `np.argmin (uint16)` | uint16 | 10,000,000 | 1.1630 ms | not_measured | managed | 0.462x |
| `np.argmin (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 0.818x |
| `np.argmin (uint32)` | uint32 | 100,000 | 0.0057 ms | not_measured | managed | 1.596x |
| `np.argmin (uint32)` | uint32 | 10,000,000 | 4.3178 ms | not_measured | managed | 0.508x |
| `np.argmin (uint64)` | uint64 | 1,000 | 0.0016 ms | not_measured | managed | 0.625x |
| `np.argmin (uint64)` | uint64 | 100,000 | 0.0613 ms | not_measured | managed | 0.284x |
| `np.argmin (uint64)` | uint64 | 10,000,000 | 14.2426 ms | not_measured | managed | 0.331x |
| `np.argmin (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 0.692x |
| `np.argmin (uint8)` | uint8 | 100,000 | 0.0022 ms | not_measured | managed | 1.409x |
| `np.argmin (uint8)` | uint8 | 10,000,000 | 0.2613 ms | not_measured | managed | 0.855x |
| `np.cumprod(a) (float16)` | float16 | 1,000 | 0.0099 ms | not_measured | managed | 0.626x |
| `np.cumprod(a) (float16)` | float16 | 100,000 | 0.9618 ms | not_measured | managed | 0.433x |
| `np.cumprod(a) (float16)` | float16 | 10,000,000 | 94.4352 ms | not_measured | managed | 0.489x |
| `np.cumprod(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 1.636x |
| `np.cumprod(a) (float32)` | float32 | 100,000 | 0.1141 ms | not_measured | managed | 1.594x |
| `np.cumprod(a) (float32)` | float32 | 10,000,000 | 10.6752 ms | not_measured | managed | 1.951x |
| `np.cumprod(a) (float64)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 0.712x |
| `np.cumprod(a) (float64)` | float64 | 100,000 | 0.1462 ms | not_measured | managed | 1.153x |
| `np.cumprod(a) (float64)` | float64 | 10,000,000 | 14.4049 ms | not_measured | managed | 1.873x |
| `np.cumsum (complex128)` | complex128 | 1,000 | 0.0041 ms | not_measured | managed | 0.732x |
| `np.cumsum (complex128)` | complex128 | 100,000 | 0.3255 ms | not_measured | managed | 1.192x |
| `np.cumsum (complex128)` | complex128 | 10,000,000 | 44.4570 ms | not_measured | managed | 0.955x |
| `np.cumsum (float16)` | float16 | 1,000 | 0.0174 ms | not_measured | managed | 0.356x |
| `np.cumsum (float16)` | float16 | 100,000 | 1.6703 ms | not_measured | managed | 0.293x |
| `np.cumsum (float16)` | float16 | 10,000,000 | 167.9535 ms | not_measured | managed | 0.293x |
| `np.cumsum (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 1.304x |
| `np.cumsum (float32)` | float32 | 100,000 | 0.1178 ms | not_measured | managed | 1.380x |
| `np.cumsum (float32)` | float32 | 10,000,000 | 12.3104 ms | not_measured | managed | 1.610x |
| `np.cumsum (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 1.217x |
| `np.cumsum (float64)` | float64 | 100,000 | 0.1720 ms | not_measured | managed | 1.009x |
| `np.cumsum (float64)` | float64 | 10,000,000 | 21.6830 ms | not_measured | managed | 1.121x |
| `np.cumsum (int16)` | int16 | 1,000 | 0.0026 ms | not_measured | managed | 1.000x |
| `np.cumsum (int16)` | int16 | 100,000 | 0.1622 ms | not_measured | managed | 2.033x |
| `np.cumsum (int16)` | int16 | 10,000,000 | 16.5649 ms | not_measured | managed | 1.751x |
| `np.cumsum (int32)` | int32 | 1,000 | 0.0024 ms | not_measured | managed | 1.125x |
| `np.cumsum (int32)` | int32 | 100,000 | 0.1681 ms | not_measured | managed | 1.907x |
| `np.cumsum (int32)` | int32 | 10,000,000 | 17.2159 ms | not_measured | managed | 1.808x |
| `np.cumsum (int64)` | int64 | 1,000 | 0.0022 ms | not_measured | managed | 0.773x |
| `np.cumsum (int64)` | int64 | 100,000 | 0.1702 ms | not_measured | managed | 0.180x |
| `np.cumsum (int64)` | int64 | 10,000,000 | 21.5475 ms | not_measured | managed | 0.724x |
| `np.cumsum (int8)` | int8 | 1,000 | 0.0027 ms | not_measured | managed | 0.963x |
| `np.cumsum (int8)` | int8 | 100,000 | 0.1968 ms | not_measured | managed | 1.669x |
| `np.cumsum (int8)` | int8 | 10,000,000 | 14.5603 ms | not_measured | managed | 2.713x |
| `np.cumsum (uint16)` | uint16 | 1,000 | 0.0024 ms | not_measured | managed | 1.167x |
| `np.cumsum (uint16)` | uint16 | 100,000 | 0.1581 ms | not_measured | managed | 2.126x |
| `np.cumsum (uint16)` | uint16 | 10,000,000 | 15.4563 ms | not_measured | managed | 1.854x |
| `np.cumsum (uint32)` | uint32 | 1,000 | 0.0023 ms | not_measured | managed | 1.087x |
| `np.cumsum (uint32)` | uint32 | 100,000 | 0.1605 ms | not_measured | managed | 1.963x |
| `np.cumsum (uint32)` | uint32 | 10,000,000 | 16.9712 ms | not_measured | managed | 1.748x |
| `np.cumsum (uint64)` | uint64 | 1,000 | 0.0022 ms | not_measured | managed | 0.864x |
| `np.cumsum (uint64)` | uint64 | 100,000 | 0.1661 ms | not_measured | managed | 0.183x |
| `np.cumsum (uint64)` | uint64 | 10,000,000 | 22.6918 ms | not_measured | managed | 0.740x |
| `np.cumsum (uint8)` | uint8 | 1,000 | 0.0028 ms | not_measured | managed | 0.964x |
| `np.cumsum (uint8)` | uint8 | 100,000 | 0.1685 ms | not_measured | managed | 2.069x |
| `np.cumsum (uint8)` | uint8 | 10,000,000 | 14.8503 ms | not_measured | managed | 1.918x |
| `np.max (complex128)` | complex128 | 1,000 | 0.0024 ms | not_measured | managed | 1.375x |
| `np.max (complex128)` | complex128 | 100,000 | 0.1494 ms | not_measured | managed | 1.097x |
| `np.max (complex128)` | complex128 | 10,000,000 | 20.7188 ms | not_measured | managed | 0.808x |
| `np.max (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 1.625x |
| `np.max (float16)` | float16 | 100,000 | 0.4656 ms | not_measured | managed | 1.104x |
| `np.max (float16)` | float16 | 10,000,000 | 47.5786 ms | not_measured | managed | 1.074x |
| `np.max (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.max (float32)` | float32 | 100,000 | 0.0097 ms | not_measured | managed | 0.629x |
| `np.max (float32)` | float32 | 10,000,000 | 4.0679 ms | not_measured | managed | 0.381x |
| `np.max (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 1.200x |
| `np.max (float64)` | float64 | 100,000 | 0.0194 ms | not_measured | managed | 0.562x |
| `np.max (float64)` | float64 | 10,000,000 | 9.0596 ms | not_measured | managed | 0.384x |
| `np.max (int16)` | int16 | 1,000 | 0.0014 ms | not_measured | managed | 1.214x |
| `np.max (int16)` | int16 | 100,000 | 0.0033 ms | not_measured | managed | 0.970x |
| `np.max (int16)` | int16 | 10,000,000 | 0.9657 ms | not_measured | managed | 0.342x |
| `np.max (int32)` | int32 | 1,000 | 0.0020 ms | not_measured | managed | 1.000x |
| `np.max (int32)` | int32 | 100,000 | 0.0047 ms | not_measured | managed | 0.915x |
| `np.max (int32)` | int32 | 10,000,000 | 4.2432 ms | not_measured | managed | 0.331x |
| `np.max (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 1.545x |
| `np.max (int64)` | int64 | 100,000 | 0.0301 ms | not_measured | managed | 0.299x |
| `np.max (int64)` | int64 | 10,000,000 | 10.9978 ms | not_measured | managed | 0.314x |
| `np.max (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 1.214x |
| `np.max (int8)` | int8 | 100,000 | 0.0026 ms | not_measured | managed | 0.962x |
| `np.max (int8)` | int8 | 10,000,000 | 0.2412 ms | not_measured | managed | 0.665x |
| `np.max (uint16)` | uint16 | 1,000 | 0.0010 ms | not_measured | managed | 1.800x |
| `np.max (uint16)` | uint16 | 100,000 | 0.0027 ms | not_measured | managed | 1.185x |
| `np.max (uint16)` | uint16 | 10,000,000 | 0.8530 ms | not_measured | managed | 0.341x |
| `np.max (uint32)` | uint32 | 1,000 | 0.0017 ms | not_measured | managed | 1.000x |
| `np.max (uint32)` | uint32 | 100,000 | 0.0056 ms | not_measured | managed | 0.750x |
| `np.max (uint32)` | uint32 | 10,000,000 | 5.1535 ms | not_measured | managed | 0.267x |
| `np.max (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 1.538x |
| `np.max (uint64)` | uint64 | 100,000 | 0.0400 ms | not_measured | managed | 0.307x |
| `np.max (uint64)` | uint64 | 10,000,000 | 10.7945 ms | not_measured | managed | 0.361x |
| `np.max (uint8)` | uint8 | 1,000 | 0.0012 ms | not_measured | managed | 1.417x |
| `np.max (uint8)` | uint8 | 100,000 | 0.0024 ms | not_measured | managed | 1.042x |
| `np.max (uint8)` | uint8 | 10,000,000 | 0.2732 ms | not_measured | managed | 0.450x |
| `np.mean (complex128)` | complex128 | 1,000 | 0.0022 ms | not_measured | managed | 1.273x |
| `np.mean (complex128)` | complex128 | 100,000 | 0.0119 ms | not_measured | managed | 3.067x |
| `np.mean (complex128)` | complex128 | 10,000,000 | 19.0264 ms | not_measured | managed | 0.475x |
| `np.mean (float16)` | float16 | 1,000 | 0.0022 ms | not_measured | managed | 2.227x |
| `np.mean (float16)` | float16 | 100,000 | 0.0855 ms | not_measured | managed | 1.244x |
| `np.mean (float16)` | float16 | 10,000,000 | 17.1652 ms | not_measured | managed | 0.602x |
| `np.mean (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 1.947x |
| `np.mean (float32)` | float32 | 100,000 | 0.0033 ms | not_measured | managed | 5.636x |
| `np.mean (float32)` | float32 | 10,000,000 | 3.9873 ms | not_measured | managed | 0.832x |
| `np.mean (float64)` | float64 | 1,000 | 0.0014 ms | not_measured | managed | 1.786x |
| `np.mean (float64)` | float64 | 100,000 | 0.0042 ms | not_measured | managed | 4.357x |
| `np.mean (float64)` | float64 | 10,000,000 | 8.6582 ms | not_measured | managed | 0.550x |
| `np.mean (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 2.538x |
| `np.mean (int16)` | int16 | 100,000 | 0.0192 ms | not_measured | managed | 2.750x |
| `np.mean (int16)` | int16 | 10,000,000 | 2.1535 ms | not_measured | managed | 2.557x |
| `np.mean (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 1.938x |
| `np.mean (int32)` | int32 | 100,000 | 0.0213 ms | not_measured | managed | 2.000x |
| `np.mean (int32)` | int32 | 10,000,000 | 3.0139 ms | not_measured | managed | 1.541x |
| `np.mean (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 1.611x |
| `np.mean (int64)` | int64 | 100,000 | 0.0047 ms | not_measured | managed | 7.340x |
| `np.mean (int64)` | int64 | 10,000,000 | 3.3658 ms | not_measured | managed | 1.931x |
| `np.mean (int8)` | int8 | 1,000 | 0.0014 ms | not_measured | managed | 2.357x |
| `np.mean (int8)` | int8 | 100,000 | 0.0479 ms | not_measured | managed | 1.121x |
| `np.mean (int8)` | int8 | 10,000,000 | 1.9057 ms | not_measured | managed | 3.173x |
| `np.mean (uint16)` | uint16 | 1,000 | 0.0014 ms | not_measured | managed | 2.214x |
| `np.mean (uint16)` | uint16 | 100,000 | 0.0225 ms | not_measured | managed | 2.360x |
| `np.mean (uint16)` | uint16 | 10,000,000 | 2.0661 ms | not_measured | managed | 2.470x |
| `np.mean (uint32)` | uint32 | 1,000 | 0.0019 ms | not_measured | managed | 1.632x |
| `np.mean (uint32)` | uint32 | 100,000 | 0.0205 ms | not_measured | managed | 1.995x |
| `np.mean (uint32)` | uint32 | 10,000,000 | 3.2003 ms | not_measured | managed | 1.555x |
| `np.mean (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 2.727x |
| `np.mean (uint64)` | uint64 | 100,000 | 0.0048 ms | not_measured | managed | 10.854x |
| `np.mean (uint64)` | uint64 | 10,000,000 | 8.3626 ms | not_measured | managed | 0.864x |
| `np.mean (uint8)` | uint8 | 1,000 | 0.0013 ms | not_measured | managed | 2.385x |
| `np.mean (uint8)` | uint8 | 100,000 | 0.0191 ms | not_measured | managed | 2.832x |
| `np.mean (uint8)` | uint8 | 10,000,000 | 1.9221 ms | not_measured | managed | 2.606x |
| `np.mean axis=0 (complex128)` | complex128 | 1,000 | 0.0046 ms | not_measured | managed | 0.913x |
| `np.mean axis=0 (complex128)` | complex128 | 100,000 | 0.0485 ms | not_measured | managed | 0.367x |
| `np.mean axis=0 (complex128)` | complex128 | 10,000,000 | 18.2055 ms | not_measured | managed | 0.425x |
| `np.mean axis=0 (float16)` | float16 | 1,000 | 0.0072 ms | not_measured | managed | 0.819x |
| `np.mean axis=0 (float16)` | float16 | 100,000 | 0.2952 ms | not_measured | managed | 0.266x |
| `np.mean axis=0 (float16)` | float16 | 10,000,000 | 28.0556 ms | not_measured | managed | 0.252x |
| `np.mean axis=0 (float32)` | float32 | 1,000 | 0.0035 ms | not_measured | managed | 1.114x |
| `np.mean axis=0 (float32)` | float32 | 100,000 | 0.0189 ms | not_measured | managed | 0.540x |
| `np.mean axis=0 (float32)` | float32 | 10,000,000 | 5.9516 ms | not_measured | managed | 0.359x |
| `np.mean axis=0 (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.660x |
| `np.mean axis=0 (float64)` | float64 | 100,000 | 0.0316 ms | not_measured | managed | 0.446x |
| `np.mean axis=0 (float64)` | float64 | 10,000,000 | 9.2269 ms | not_measured | managed | 0.361x |
| `np.mean axis=0 (int16)` | int16 | 1,000 | 0.0027 ms | not_measured | managed | 1.519x |
| `np.mean axis=0 (int16)` | int16 | 100,000 | 0.0199 ms | not_measured | managed | 2.583x |
| `np.mean axis=0 (int16)` | int16 | 10,000,000 | 2.7348 ms | not_measured | managed | 1.913x |
| `np.mean axis=0 (int32)` | int32 | 1,000 | 0.0009 ms | not_measured | managed | 4.556x |
| `np.mean axis=0 (int32)` | int32 | 100,000 | 0.0208 ms | not_measured | managed | 1.952x |
| `np.mean axis=0 (int32)` | int32 | 10,000,000 | 4.7227 ms | not_measured | managed | 1.011x |
| `np.mean axis=0 (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 1.857x |
| `np.mean axis=0 (int64)` | int64 | 100,000 | 0.1525 ms | not_measured | managed | 0.216x |
| `np.mean axis=0 (int64)` | int64 | 10,000,000 | 139.0226 ms | not_measured | managed | 0.045x |
| `np.mean axis=0 (int8)` | int8 | 1,000 | 0.0009 ms | not_measured | managed | 4.556x |
| `np.mean axis=0 (int8)` | int8 | 100,000 | 0.0210 ms | not_measured | managed | 2.500x |
| `np.mean axis=0 (int8)` | int8 | 10,000,000 | 1.9504 ms | not_measured | managed | 3.122x |
| `np.mean axis=0 (uint16)` | uint16 | 1,000 | 0.0015 ms | not_measured | managed | 3.667x |
| `np.mean axis=0 (uint16)` | uint16 | 100,000 | 0.0206 ms | not_measured | managed | 2.558x |
| `np.mean axis=0 (uint16)` | uint16 | 10,000,000 | 2.6100 ms | not_measured | managed | 2.027x |
| `np.mean axis=0 (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 2.867x |
| `np.mean axis=0 (uint32)` | uint32 | 100,000 | 0.0280 ms | not_measured | managed | 1.439x |
| `np.mean axis=0 (uint32)` | uint32 | 10,000,000 | 4.5148 ms | not_measured | managed | 1.029x |
| `np.mean axis=0 (uint64)` | uint64 | 1,000 | 0.0022 ms | not_measured | managed | 1.864x |
| `np.mean axis=0 (uint64)` | uint64 | 100,000 | 0.1795 ms | not_measured | managed | 0.256x |
| `np.mean axis=0 (uint64)` | uint64 | 10,000,000 | 140.0372 ms | not_measured | managed | 0.054x |
| `np.mean axis=0 (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 5.125x |
| `np.mean axis=0 (uint8)` | uint8 | 100,000 | 0.0211 ms | not_measured | managed | 2.545x |
| `np.mean axis=0 (uint8)` | uint8 | 10,000,000 | 1.9927 ms | not_measured | managed | 2.848x |
| `np.mean axis=1 (complex128)` | complex128 | 1,000 | 0.0050 ms | not_measured | managed | 0.680x |
| `np.mean axis=1 (complex128)` | complex128 | 100,000 | 0.0562 ms | not_measured | managed | 0.626x |
| `np.mean axis=1 (complex128)` | complex128 | 10,000,000 | 19.7724 ms | not_measured | managed | 0.422x |
| `np.mean axis=1 (float16)` | float16 | 1,000 | 0.0065 ms | not_measured | managed | 0.862x |
| `np.mean axis=1 (float16)` | float16 | 100,000 | 0.2796 ms | not_measured | managed | 0.311x |
| `np.mean axis=1 (float16)` | float16 | 10,000,000 | 27.1315 ms | not_measured | managed | 0.292x |
| `np.mean axis=1 (float32)` | float32 | 1,000 | 0.0040 ms | not_measured | managed | 0.950x |
| `np.mean axis=1 (float32)` | float32 | 100,000 | 0.0258 ms | not_measured | managed | 0.895x |
| `np.mean axis=1 (float32)` | float32 | 10,000,000 | 4.6932 ms | not_measured | managed | 0.688x |
| `np.mean axis=1 (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.660x |
| `np.mean axis=1 (float64)` | float64 | 100,000 | 0.0299 ms | not_measured | managed | 0.706x |
| `np.mean axis=1 (float64)` | float64 | 10,000,000 | 10.1108 ms | not_measured | managed | 0.505x |
| `np.mean axis=1 (int16)` | int16 | 1,000 | 0.0020 ms | not_measured | managed | 1.950x |
| `np.mean axis=1 (int16)` | int16 | 100,000 | 0.0206 ms | not_measured | managed | 2.782x |
| `np.mean axis=1 (int16)` | int16 | 10,000,000 | 2.6301 ms | not_measured | managed | 1.992x |
| `np.mean axis=1 (int32)` | int32 | 1,000 | 0.0016 ms | not_measured | managed | 2.438x |
| `np.mean axis=1 (int32)` | int32 | 100,000 | 0.0219 ms | not_measured | managed | 2.000x |
| `np.mean axis=1 (int32)` | int32 | 10,000,000 | 4.6992 ms | not_measured | managed | 1.072x |
| `np.mean axis=1 (int64)` | int64 | 1,000 | 0.0021 ms | not_measured | managed | 1.714x |
| `np.mean axis=1 (int64)` | int64 | 100,000 | 0.1471 ms | not_measured | managed | 0.285x |
| `np.mean axis=1 (int64)` | int64 | 10,000,000 | 16.9707 ms | not_measured | managed | 0.384x |
| `np.mean axis=1 (int8)` | int8 | 1,000 | 0.0008 ms | not_measured | managed | 4.750x |
| `np.mean axis=1 (int8)` | int8 | 100,000 | 0.0211 ms | not_measured | managed | 2.777x |
| `np.mean axis=1 (int8)` | int8 | 10,000,000 | 1.8933 ms | not_measured | managed | 3.367x |
| `np.mean axis=1 (uint16)` | uint16 | 1,000 | 0.0009 ms | not_measured | managed | 4.111x |
| `np.mean axis=1 (uint16)` | uint16 | 100,000 | 0.0211 ms | not_measured | managed | 2.777x |
| `np.mean axis=1 (uint16)` | uint16 | 10,000,000 | 2.4331 ms | not_measured | managed | 2.276x |
| `np.mean axis=1 (uint32)` | uint32 | 1,000 | 0.0015 ms | not_measured | managed | 2.467x |
| `np.mean axis=1 (uint32)` | uint32 | 100,000 | 0.0275 ms | not_measured | managed | 1.771x |
| `np.mean axis=1 (uint32)` | uint32 | 10,000,000 | 4.7657 ms | not_measured | managed | 1.000x |
| `np.mean axis=1 (uint64)` | uint64 | 1,000 | 0.0024 ms | not_measured | managed | 2.042x |
| `np.mean axis=1 (uint64)` | uint64 | 100,000 | 0.1649 ms | not_measured | managed | 0.351x |
| `np.mean axis=1 (uint64)` | uint64 | 10,000,000 | 17.1868 ms | not_measured | managed | 0.429x |
| `np.mean axis=1 (uint8)` | uint8 | 1,000 | 0.0011 ms | not_measured | managed | 3.727x |
| `np.mean axis=1 (uint8)` | uint8 | 100,000 | 0.0200 ms | not_measured | managed | 2.865x |
| `np.mean axis=1 (uint8)` | uint8 | 10,000,000 | 1.9554 ms | not_measured | managed | 2.648x |
| `np.min (complex128)` | complex128 | 1,000 | 0.0023 ms | not_measured | managed | 1.435x |
| `np.min (complex128)` | complex128 | 100,000 | 0.1461 ms | not_measured | managed | 1.086x |
| `np.min (complex128)` | complex128 | 10,000,000 | 20.8606 ms | not_measured | managed | 0.793x |
| `np.min (float16)` | float16 | 1,000 | 0.0019 ms | not_measured | managed | 2.105x |
| `np.min (float16)` | float16 | 100,000 | 0.4187 ms | not_measured | managed | 1.251x |
| `np.min (float16)` | float16 | 10,000,000 | 45.3168 ms | not_measured | managed | 1.141x |
| `np.min (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.944x |
| `np.min (float32)` | float32 | 100,000 | 0.0093 ms | not_measured | managed | 0.677x |
| `np.min (float32)` | float32 | 10,000,000 | 4.1696 ms | not_measured | managed | 0.396x |
| `np.min (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 1.800x |
| `np.min (float64)` | float64 | 100,000 | 0.0183 ms | not_measured | managed | 0.568x |
| `np.min (float64)` | float64 | 10,000,000 | 8.5502 ms | not_measured | managed | 0.400x |
| `np.min (int16)` | int16 | 1,000 | 0.0009 ms | not_measured | managed | 1.889x |
| `np.min (int16)` | int16 | 100,000 | 0.0026 ms | not_measured | managed | 1.231x |
| `np.min (int16)` | int16 | 10,000,000 | 0.6511 ms | not_measured | managed | 0.526x |
| `np.min (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 1.308x |
| `np.min (int32)` | int32 | 100,000 | 0.0045 ms | not_measured | managed | 1.022x |
| `np.min (int32)` | int32 | 10,000,000 | 3.4914 ms | not_measured | managed | 0.372x |
| `np.min (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 1.727x |
| `np.min (int64)` | int64 | 100,000 | 0.0284 ms | not_measured | managed | 0.342x |
| `np.min (int64)` | int64 | 10,000,000 | 8.9260 ms | not_measured | managed | 0.363x |
| `np.min (int8)` | int8 | 1,000 | 0.0010 ms | not_measured | managed | 1.800x |
| `np.min (int8)` | int8 | 100,000 | 0.0023 ms | not_measured | managed | 1.087x |
| `np.min (int8)` | int8 | 10,000,000 | 0.2440 ms | not_measured | managed | 0.784x |
| `np.min (uint16)` | uint16 | 1,000 | 0.0018 ms | not_measured | managed | 0.944x |
| `np.min (uint16)` | uint16 | 100,000 | 0.0028 ms | not_measured | managed | 1.143x |
| `np.min (uint16)` | uint16 | 10,000,000 | 0.6375 ms | not_measured | managed | 0.468x |
| `np.min (uint32)` | uint32 | 1,000 | 0.0014 ms | not_measured | managed | 1.500x |
| `np.min (uint32)` | uint32 | 100,000 | 0.0051 ms | not_measured | managed | 0.863x |
| `np.min (uint32)` | uint32 | 10,000,000 | 3.9324 ms | not_measured | managed | 0.334x |
| `np.min (uint64)` | uint64 | 1,000 | 0.0014 ms | not_measured | managed | 1.286x |
| `np.min (uint64)` | uint64 | 100,000 | 0.0381 ms | not_measured | managed | 0.331x |
| `np.min (uint64)` | uint64 | 10,000,000 | 8.4767 ms | not_measured | managed | 0.468x |
| `np.min (uint8)` | uint8 | 1,000 | 0.0008 ms | not_measured | managed | 2.125x |
| `np.min (uint8)` | uint8 | 100,000 | 0.0025 ms | not_measured | managed | 1.040x |
| `np.min (uint8)` | uint8 | 10,000,000 | 0.2406 ms | not_measured | managed | 0.619x |
| `np.nanargmax(a) (float16)` | float16 | 1,000 | 0.0057 ms | not_measured | managed | 1.737x |
| `np.nanargmax(a) (float16)` | float16 | 100,000 | 0.2891 ms | not_measured | managed | 1.791x |
| `np.nanargmax(a) (float16)` | float16 | 10,000,000 | 29.0918 ms | not_measured | managed | 1.947x |
| `np.nanargmax(a) (float32)` | float32 | 1,000 | 0.0042 ms | not_measured | managed | 2.119x |
| `np.nanargmax(a) (float32)` | float32 | 100,000 | 0.1224 ms | not_measured | managed | 0.255x |
| `np.nanargmax(a) (float32)` | float32 | 10,000,000 | 16.2447 ms | not_measured | managed | 0.909x |
| `np.nanargmax(a) (float64)` | float64 | 1,000 | 0.0052 ms | not_measured | managed | 1.308x |
| `np.nanargmax(a) (float64)` | float64 | 100,000 | 0.1264 ms | not_measured | managed | 0.373x |
| `np.nanargmax(a) (float64)` | float64 | 10,000,000 | 21.3015 ms | not_measured | managed | 1.322x |
| `np.nanargmin(a) (float16)` | float16 | 1,000 | 0.0053 ms | not_measured | managed | 1.811x |
| `np.nanargmin(a) (float16)` | float16 | 100,000 | 0.2935 ms | not_measured | managed | 1.729x |
| `np.nanargmin(a) (float16)` | float16 | 10,000,000 | 28.7053 ms | not_measured | managed | 1.867x |
| `np.nanargmin(a) (float32)` | float32 | 1,000 | 0.0047 ms | not_measured | managed | 1.383x |
| `np.nanargmin(a) (float32)` | float32 | 100,000 | 0.1227 ms | not_measured | managed | 0.230x |
| `np.nanargmin(a) (float32)` | float32 | 10,000,000 | 14.4274 ms | not_measured | managed | 0.959x |
| `np.nanargmin(a) (float64)` | float64 | 1,000 | 0.0048 ms | not_measured | managed | 1.333x |
| `np.nanargmin(a) (float64)` | float64 | 100,000 | 0.1248 ms | not_measured | managed | 0.385x |
| `np.nanargmin(a) (float64)` | float64 | 10,000,000 | 20.9830 ms | not_measured | managed | 1.167x |
| `np.nanmax(a) (float16)` | float16 | 1,000 | 0.0020 ms | not_measured | managed | 2.800x |
| `np.nanmax(a) (float16)` | float16 | 100,000 | 0.4420 ms | not_measured | managed | 1.143x |
| `np.nanmax(a) (float16)` | float16 | 10,000,000 | 45.9507 ms | not_measured | managed | 1.113x |
| `np.nanmax(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 2.062x |
| `np.nanmax(a) (float32)` | float32 | 100,000 | 0.0528 ms | not_measured | managed | 0.153x |
| `np.nanmax(a) (float32)` | float32 | 10,000,000 | 5.9821 ms | not_measured | managed | 0.329x |
| `np.nanmax(a) (float64)` | float64 | 1,000 | 0.0018 ms | not_measured | managed | 1.722x |
| `np.nanmax(a) (float64)` | float64 | 100,000 | 0.1038 ms | not_measured | managed | 0.117x |
| `np.nanmax(a) (float64)` | float64 | 10,000,000 | 13.6955 ms | not_measured | managed | 0.409x |
| `np.nanmean(a) (float16)` | float16 | 1,000 | 0.0029 ms | not_measured | managed | 4.483x |
| `np.nanmean(a) (float16)` | float16 | 100,000 | 0.2123 ms | not_measured | managed | 1.554x |
| `np.nanmean(a) (float16)` | float16 | 10,000,000 | 21.1416 ms | not_measured | managed | 1.809x |
| `np.nanmean(a) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 4.952x |
| `np.nanmean(a) (float32)` | float32 | 100,000 | 0.0745 ms | not_measured | managed | 1.024x |
| `np.nanmean(a) (float32)` | float32 | 10,000,000 | 7.7300 ms | not_measured | managed | 2.692x |
| `np.nanmean(a) (float64)` | float64 | 1,000 | 0.0023 ms | not_measured | managed | 4.000x |
| `np.nanmean(a) (float64)` | float64 | 100,000 | 0.0744 ms | not_measured | managed | 1.187x |
| `np.nanmean(a) (float64)` | float64 | 10,000,000 | 9.8835 ms | not_measured | managed | 4.911x |
| `np.nanmedian(a) (float16)` | float16 | 1,000 | 0.0133 ms | not_measured | managed | 1.714x |
| `np.nanmedian(a) (float16)` | float16 | 100,000 | 1.7130 ms | not_measured | managed | 0.560x |
| `np.nanmedian(a) (float16)` | float16 | 10,000,000 | 131.1002 ms | not_measured | managed | 0.889x |
| `np.nanmedian(a) (float32)` | float32 | 1,000 | 0.0047 ms | not_measured | managed | 3.000x |
| `np.nanmedian(a) (float32)` | float32 | 100,000 | 0.3725 ms | not_measured | managed | 1.298x |
| `np.nanmedian(a) (float32)` | float32 | 10,000,000 | 55.8867 ms | not_measured | managed | 1.462x |
| `np.nanmedian(a) (float64)` | float64 | 1,000 | 0.0050 ms | not_measured | managed | 2.580x |
| `np.nanmedian(a) (float64)` | float64 | 100,000 | 0.3899 ms | not_measured | managed | 1.257x |
| `np.nanmedian(a) (float64)` | float64 | 10,000,000 | 64.0212 ms | not_measured | managed | 1.714x |
| `np.nanmin(a) (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 2.857x |
| `np.nanmin(a) (float16)` | float16 | 100,000 | 0.4450 ms | not_measured | managed | 1.139x |
| `np.nanmin(a) (float16)` | float16 | 10,000,000 | 45.8740 ms | not_measured | managed | 1.094x |
| `np.nanmin(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 2.286x |
| `np.nanmin(a) (float32)` | float32 | 100,000 | 0.0534 ms | not_measured | managed | 0.139x |
| `np.nanmin(a) (float32)` | float32 | 10,000,000 | 5.9093 ms | not_measured | managed | 0.306x |
| `np.nanmin(a) (float64)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 1.455x |
| `np.nanmin(a) (float64)` | float64 | 100,000 | 0.1035 ms | not_measured | managed | 0.112x |
| `np.nanmin(a) (float64)` | float64 | 10,000,000 | 11.5795 ms | not_measured | managed | 0.401x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 1,000 | 0.0134 ms | not_measured | managed | 2.627x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 100,000 | 1.7669 ms | not_measured | managed | 1.053x |
| `np.nanpercentile(a, 50) (float16)` | float16 | 10,000,000 | 129.4004 ms | not_measured | managed | 0.925x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 1,000 | 0.0051 ms | not_measured | managed | 5.745x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 100,000 | 0.3767 ms | not_measured | managed | 2.047x |
| `np.nanpercentile(a, 50) (float32)` | float32 | 10,000,000 | 54.7743 ms | not_measured | managed | 1.268x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 5.755x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 100,000 | 0.4218 ms | not_measured | managed | 1.858x |
| `np.nanpercentile(a, 50) (float64)` | float64 | 10,000,000 | 64.7989 ms | not_measured | managed | 0.982x |
| `np.nanprod(a) (float16)` | float16 | 1,000 | 0.0026 ms | not_measured | managed | 2.538x |
| `np.nanprod(a) (float16)` | float16 | 100,000 | 0.1947 ms | not_measured | managed | 0.855x |
| `np.nanprod(a) (float16)` | float16 | 10,000,000 | 19.4095 ms | not_measured | managed | 1.049x |
| `np.nanprod(a) (float32)` | float32 | 1,000 | 0.0014 ms | not_measured | managed | 3.857x |
| `np.nanprod(a) (float32)` | float32 | 100,000 | 0.0169 ms | not_measured | managed | 5.568x |
| `np.nanprod(a) (float32)` | float32 | 10,000,000 | 4.7322 ms | not_measured | managed | 3.853x |
| `np.nanprod(a) (float64)` | float64 | 1,000 | 0.0011 ms | not_measured | managed | 5.000x |
| `np.nanprod(a) (float64)` | float64 | 100,000 | 0.0325 ms | not_measured | managed | 3.160x |
| `np.nanprod(a) (float64)` | float64 | 10,000,000 | 9.8603 ms | not_measured | managed | 2.729x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 1,000 | 0.0141 ms | not_measured | managed | 2.411x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 100,000 | 1.7488 ms | not_measured | managed | 1.038x |
| `np.nanquantile(a, 0.5) (float16)` | float16 | 10,000,000 | 131.9225 ms | not_measured | managed | 0.910x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 1,000 | 0.0052 ms | not_measured | managed | 5.635x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 100,000 | 0.3774 ms | not_measured | managed | 1.995x |
| `np.nanquantile(a, 0.5) (float32)` | float32 | 10,000,000 | 54.4836 ms | not_measured | managed | 1.137x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 1,000 | 0.0050 ms | not_measured | managed | 5.360x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 100,000 | 0.3908 ms | not_measured | managed | 1.873x |
| `np.nanquantile(a, 0.5) (float64)` | float64 | 10,000,000 | 63.4764 ms | not_measured | managed | 1.064x |
| `np.nanstd(a) (float16)` | float16 | 1,000 | 0.0056 ms | not_measured | managed | 8.161x |
| `np.nanstd(a) (float16)` | float16 | 100,000 | 0.4231 ms | not_measured | managed | 2.812x |
| `np.nanstd(a) (float16)` | float16 | 10,000,000 | 41.2492 ms | not_measured | managed | 3.142x |
| `np.nanstd(a) (float32)` | float32 | 1,000 | 0.0028 ms | not_measured | managed | 7.643x |
| `np.nanstd(a) (float32)` | float32 | 100,000 | 0.1566 ms | not_measured | managed | 1.058x |
| `np.nanstd(a) (float32)` | float32 | 10,000,000 | 16.3418 ms | not_measured | managed | 2.299x |
| `np.nanstd(a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 7.538x |
| `np.nanstd(a) (float64)` | float64 | 100,000 | 0.1527 ms | not_measured | managed | 1.219x |
| `np.nanstd(a) (float64)` | float64 | 10,000,000 | 19.8716 ms | not_measured | managed | 2.778x |
| `np.nansum(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 2.826x |
| `np.nansum(a) (float16)` | float16 | 100,000 | 0.1962 ms | not_measured | managed | 1.497x |
| `np.nansum(a) (float16)` | float16 | 10,000,000 | 19.3425 ms | not_measured | managed | 1.633x |
| `np.nansum(a) (float32)` | float32 | 1,000 | 0.0008 ms | not_measured | managed | 4.875x |
| `np.nansum(a) (float32)` | float32 | 100,000 | 0.0098 ms | not_measured | managed | 3.194x |
| `np.nansum(a) (float32)` | float32 | 10,000,000 | 5.3510 ms | not_measured | managed | 2.746x |
| `np.nansum(a) (float64)` | float64 | 1,000 | 0.0010 ms | not_measured | managed | 3.900x |
| `np.nansum(a) (float64)` | float64 | 100,000 | 0.0195 ms | not_measured | managed | 2.262x |
| `np.nansum(a) (float64)` | float64 | 10,000,000 | 9.4570 ms | not_measured | managed | 2.580x |
| `np.nanvar(a) (float16)` | float16 | 1,000 | 0.0056 ms | not_measured | managed | 7.071x |
| `np.nanvar(a) (float16)` | float16 | 100,000 | 0.4246 ms | not_measured | managed | 2.815x |
| `np.nanvar(a) (float16)` | float16 | 10,000,000 | 41.1379 ms | not_measured | managed | 2.917x |
| `np.nanvar(a) (float32)` | float32 | 1,000 | 0.0026 ms | not_measured | managed | 8.192x |
| `np.nanvar(a) (float32)` | float32 | 100,000 | 0.1569 ms | not_measured | managed | 1.134x |
| `np.nanvar(a) (float32)` | float32 | 10,000,000 | 17.4582 ms | not_measured | managed | 1.939x |
| `np.nanvar(a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 7.269x |
| `np.nanvar(a) (float64)` | float64 | 100,000 | 0.1507 ms | not_measured | managed | 1.228x |
| `np.nanvar(a) (float64)` | float64 | 10,000,000 | 20.8325 ms | not_measured | managed | 2.490x |
| `np.prod (float64)` | float64 | 1,000 | 0.0017 ms | not_measured | managed | 1.412x |
| `np.prod (float64)` | float64 | 100,000 | 1.0256 ms | not_measured | managed | 2.256x |
| `np.prod (float64)` | float64 | 10,000,000 | 417.5831 ms | not_measured | managed | 0.898x |
| `np.prod (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 2.154x |
| `np.prod (int64)` | int64 | 100,000 | 0.0388 ms | not_measured | managed | 1.482x |
| `np.prod (int64)` | int64 | 10,000,000 | 8.7327 ms | not_measured | managed | 0.733x |
| `np.prod axis=0 (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.750x |
| `np.prod axis=0 (float64)` | float64 | 100,000 | 0.0201 ms | not_measured | managed | 0.483x |
| `np.prod axis=0 (float64)` | float64 | 10,000,000 | 117.8719 ms | not_measured | managed | 0.171x |
| `np.prod axis=0 (int64)` | int64 | 1,000 | 0.0018 ms | not_measured | managed | 1.222x |
| `np.prod axis=0 (int64)` | int64 | 100,000 | 0.0436 ms | not_measured | managed | 0.773x |
| `np.prod axis=0 (int64)` | int64 | 10,000,000 | 9.3039 ms | not_measured | managed | 0.607x |
| `np.prod axis=1 (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 1.833x |
| `np.prod axis=1 (float64)` | float64 | 100,000 | 0.0126 ms | not_measured | managed | 4.563x |
| `np.prod axis=1 (float64)` | float64 | 10,000,000 | 8.7521 ms | not_measured | managed | 8.855x |
| `np.prod axis=1 (int64)` | int64 | 1,000 | 0.0020 ms | not_measured | managed | 1.050x |
| `np.prod axis=1 (int64)` | int64 | 100,000 | 0.0446 ms | not_measured | managed | 1.087x |
| `np.prod axis=1 (int64)` | int64 | 10,000,000 | 10.4504 ms | not_measured | managed | 0.670x |
| `np.std (float16)` | float16 | 1,000 | 0.0045 ms | not_measured | managed | 4.156x |
| `np.std (float16)` | float16 | 100,000 | 0.4024 ms | not_measured | managed | 2.205x |
| `np.std (float16)` | float16 | 10,000,000 | 40.1047 ms | not_measured | managed | 2.230x |
| `np.std (float32)` | float32 | 1,000 | 0.0012 ms | not_measured | managed | 7.250x |
| `np.std (float32)` | float32 | 100,000 | 0.0192 ms | not_measured | managed | 2.734x |
| `np.std (float32)` | float32 | 10,000,000 | 9.3572 ms | not_measured | managed | 1.813x |
| `np.std (float64)` | float64 | 1,000 | 0.0019 ms | not_measured | managed | 3.632x |
| `np.std (float64)` | float64 | 100,000 | 0.0383 ms | not_measured | managed | 1.768x |
| `np.std (float64)` | float64 | 10,000,000 | 19.2119 ms | not_measured | managed | 1.718x |
| `np.std axis=0 (float16)` | float16 | 1,000 | 0.0110 ms | not_measured | managed | 1.864x |
| `np.std axis=0 (float16)` | float16 | 100,000 | 0.8497 ms | not_measured | managed | 1.336x |
| `np.std axis=0 (float16)` | float16 | 10,000,000 | 221.3413 ms | not_measured | managed | 0.499x |
| `np.std axis=0 (float32)` | float32 | 1,000 | 0.0039 ms | not_measured | managed | 2.333x |
| `np.std axis=0 (float32)` | float32 | 100,000 | 0.0668 ms | not_measured | managed | 0.696x |
| `np.std axis=0 (float32)` | float32 | 10,000,000 | 10.9038 ms | not_measured | managed | 1.356x |
| `np.std axis=0 (float64)` | float64 | 1,000 | 0.0015 ms | not_measured | managed | 6.733x |
| `np.std axis=0 (float64)` | float64 | 100,000 | 0.0489 ms | not_measured | managed | 1.458x |
| `np.std axis=0 (float64)` | float64 | 10,000,000 | 17.3995 ms | not_measured | managed | 1.650x |
| `np.sum (complex128)` | complex128 | 1,000 | 0.0015 ms | not_measured | managed | 1.533x |
| `np.sum (complex128)` | complex128 | 100,000 | 0.0255 ms | not_measured | managed | 1.278x |
| `np.sum (complex128)` | complex128 | 10,000,000 | 17.3907 ms | not_measured | managed | 0.478x |
| `np.sum (float16)` | float16 | 1,000 | 0.0015 ms | not_measured | managed | 3.933x |
| `np.sum (float16)` | float16 | 100,000 | 0.0705 ms | not_measured | managed | 2.841x |
| `np.sum (float16)` | float16 | 10,000,000 | 7.1664 ms | not_measured | managed | 2.775x |
| `np.sum (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.sum (float32)` | float32 | 100,000 | 0.0057 ms | not_measured | managed | 2.877x |
| `np.sum (float32)` | float32 | 10,000,000 | 4.5815 ms | not_measured | managed | 0.760x |
| `np.sum (float64)` | float64 | 1,000 | 0.0053 ms | not_measured | managed | 0.340x |
| `np.sum (float64)` | float64 | 100,000 | 0.1938 ms | not_measured | managed | 0.088x |
| `np.sum (float64)` | float64 | 10,000,000 | 11.1030 ms | not_measured | managed | 0.428x |
| `np.sum (int16)` | int16 | 1,000 | 0.0013 ms | not_measured | managed | 1.846x |
| `np.sum (int16)` | int16 | 100,000 | 0.0461 ms | not_measured | managed | 0.770x |
| `np.sum (int16)` | int16 | 10,000,000 | 5.1019 ms | not_measured | managed | 0.742x |
| `np.sum (int32)` | int32 | 1,000 | 0.0013 ms | not_measured | managed | 1.846x |
| `np.sum (int32)` | int32 | 100,000 | 0.0475 ms | not_measured | managed | 0.758x |
| `np.sum (int32)` | int32 | 10,000,000 | 5.8087 ms | not_measured | managed | 0.670x |
| `np.sum (int64)` | int64 | 1,000 | 0.0014 ms | not_measured | managed | 1.429x |
| `np.sum (int64)` | int64 | 100,000 | 0.0089 ms | not_measured | managed | 1.607x |
| `np.sum (int64)` | int64 | 10,000,000 | 7.9687 ms | not_measured | managed | 0.558x |
| `np.sum (int8)` | int8 | 1,000 | 0.0013 ms | not_measured | managed | 1.692x |
| `np.sum (int8)` | int8 | 100,000 | 0.0468 ms | not_measured | managed | 0.729x |
| `np.sum (int8)` | int8 | 10,000,000 | 5.1176 ms | not_measured | managed | 0.642x |
| `np.sum (uint16)` | uint16 | 1,000 | 0.0013 ms | not_measured | managed | 1.923x |
| `np.sum (uint16)` | uint16 | 100,000 | 0.0425 ms | not_measured | managed | 0.776x |
| `np.sum (uint16)` | uint16 | 10,000,000 | 4.5619 ms | not_measured | managed | 0.735x |
| `np.sum (uint32)` | uint32 | 1,000 | 0.0013 ms | not_measured | managed | 1.692x |
| `np.sum (uint32)` | uint32 | 100,000 | 0.0416 ms | not_measured | managed | 0.796x |
| `np.sum (uint32)` | uint32 | 10,000,000 | 5.3761 ms | not_measured | managed | 0.916x |
| `np.sum (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 1.385x |
| `np.sum (uint64)` | uint64 | 100,000 | 0.0088 ms | not_measured | managed | 1.716x |
| `np.sum (uint64)` | uint64 | 10,000,000 | 8.1814 ms | not_measured | managed | 0.546x |
| `np.sum (uint8)` | uint8 | 1,000 | 0.0014 ms | not_measured | managed | 1.714x |
| `np.sum (uint8)` | uint8 | 100,000 | 0.0421 ms | not_measured | managed | 0.829x |
| `np.sum (uint8)` | uint8 | 10,000,000 | 3.8629 ms | not_measured | managed | 0.814x |
| `np.sum axis=0 (complex128)` | complex128 | 1,000 | 0.0045 ms | not_measured | managed | 0.489x |
| `np.sum axis=0 (complex128)` | complex128 | 100,000 | 0.0415 ms | not_measured | managed | 0.475x |
| `np.sum axis=0 (complex128)` | complex128 | 10,000,000 | 19.1184 ms | not_measured | managed | 0.377x |
| `np.sum axis=0 (float16)` | float16 | 1,000 | 0.0051 ms | not_measured | managed | 1.000x |
| `np.sum axis=0 (float16)` | float16 | 100,000 | 0.1630 ms | not_measured | managed | 1.761x |
| `np.sum axis=0 (float16)` | float16 | 10,000,000 | 14.5672 ms | not_measured | managed | 1.960x |
| `np.sum axis=0 (float32)` | float32 | 1,000 | 0.0039 ms | not_measured | managed | 0.538x |
| `np.sum axis=0 (float32)` | float32 | 100,000 | 0.0158 ms | not_measured | managed | 0.544x |
| `np.sum axis=0 (float32)` | float32 | 10,000,000 | 4.5801 ms | not_measured | managed | 0.385x |
| `np.sum axis=0 (float64)` | float64 | 1,000 | 0.0041 ms | not_measured | managed | 0.512x |
| `np.sum axis=0 (float64)` | float64 | 100,000 | 0.0253 ms | not_measured | managed | 0.482x |
| `np.sum axis=0 (float64)` | float64 | 10,000,000 | 9.4367 ms | not_measured | managed | 0.667x |
| `np.sum axis=0 (int16)` | int16 | 1,000 | 0.0016 ms | not_measured | managed | 1.750x |
| `np.sum axis=0 (int16)` | int16 | 100,000 | 0.0102 ms | not_measured | managed | 4.627x |
| `np.sum axis=0 (int16)` | int16 | 10,000,000 | 2.1520 ms | not_measured | managed | 2.543x |
| `np.sum axis=0 (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 2.700x |
| `np.sum axis=0 (int32)` | int32 | 100,000 | 0.0161 ms | not_measured | managed | 3.298x |
| `np.sum axis=0 (int32)` | int32 | 10,000,000 | 4.7464 ms | not_measured | managed | 1.294x |
| `np.sum axis=0 (int64)` | int64 | 1,000 | 0.0011 ms | not_measured | managed | 2.000x |
| `np.sum axis=0 (int64)` | int64 | 100,000 | 0.0205 ms | not_measured | managed | 1.346x |
| `np.sum axis=0 (int64)` | int64 | 10,000,000 | 10.8252 ms | not_measured | managed | 0.546x |
| `np.sum axis=0 (int8)` | int8 | 1,000 | 0.0020 ms | not_measured | managed | 1.300x |
| `np.sum axis=0 (int8)` | int8 | 100,000 | 0.0095 ms | not_measured | managed | 5.021x |
| `np.sum axis=0 (int8)` | int8 | 10,000,000 | 0.8858 ms | not_measured | managed | 6.240x |
| `np.sum axis=0 (uint16)` | uint16 | 1,000 | 0.0018 ms | not_measured | managed | 1.444x |
| `np.sum axis=0 (uint16)` | uint16 | 100,000 | 0.0103 ms | not_measured | managed | 4.825x |
| `np.sum axis=0 (uint16)` | uint16 | 10,000,000 | 1.7566 ms | not_measured | managed | 2.717x |
| `np.sum axis=0 (uint32)` | uint32 | 1,000 | 0.0011 ms | not_measured | managed | 2.455x |
| `np.sum axis=0 (uint32)` | uint32 | 100,000 | 0.0152 ms | not_measured | managed | 3.197x |
| `np.sum axis=0 (uint32)` | uint32 | 10,000,000 | 6.7684 ms | not_measured | managed | 0.816x |
| `np.sum axis=0 (uint64)` | uint64 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.sum axis=0 (uint64)` | uint64 | 100,000 | 0.0206 ms | not_measured | managed | 1.393x |
| `np.sum axis=0 (uint64)` | uint64 | 10,000,000 | 11.7334 ms | not_measured | managed | 0.435x |
| `np.sum axis=0 (uint8)` | uint8 | 1,000 | 0.0018 ms | not_measured | managed | 1.611x |
| `np.sum axis=0 (uint8)` | uint8 | 100,000 | 0.0099 ms | not_measured | managed | 4.980x |
| `np.sum axis=0 (uint8)` | uint8 | 10,000,000 | 0.8396 ms | not_measured | managed | 5.444x |
| `np.sum axis=1 (complex128)` | complex128 | 1,000 | 0.0041 ms | not_measured | managed | 0.537x |
| `np.sum axis=1 (complex128)` | complex128 | 100,000 | 0.0500 ms | not_measured | managed | 0.712x |
| `np.sum axis=1 (complex128)` | complex128 | 10,000,000 | 19.1907 ms | not_measured | managed | 0.440x |
| `np.sum axis=1 (float16)` | float16 | 1,000 | 0.0078 ms | not_measured | managed | 0.513x |
| `np.sum axis=1 (float16)` | float16 | 100,000 | 0.0899 ms | not_measured | managed | 2.257x |
| `np.sum axis=1 (float16)` | float16 | 10,000,000 | 7.1678 ms | not_measured | managed | 2.736x |
| `np.sum axis=1 (float32)` | float32 | 1,000 | 0.0036 ms | not_measured | managed | 0.583x |
| `np.sum axis=1 (float32)` | float32 | 100,000 | 0.0219 ms | not_measured | managed | 0.817x |
| `np.sum axis=1 (float32)` | float32 | 10,000,000 | 4.9847 ms | not_measured | managed | 0.687x |
| `np.sum axis=1 (float64)` | float64 | 1,000 | 0.0040 ms | not_measured | managed | 0.525x |
| `np.sum axis=1 (float64)` | float64 | 100,000 | 0.0242 ms | not_measured | managed | 0.826x |
| `np.sum axis=1 (float64)` | float64 | 10,000,000 | 9.7793 ms | not_measured | managed | 0.550x |
| `np.sum axis=1 (int16)` | int16 | 1,000 | 0.0017 ms | not_measured | managed | 1.647x |
| `np.sum axis=1 (int16)` | int16 | 100,000 | 0.0073 ms | not_measured | managed | 5.247x |
| `np.sum axis=1 (int16)` | int16 | 10,000,000 | 1.7137 ms | not_measured | managed | 2.121x |
| `np.sum axis=1 (int32)` | int32 | 1,000 | 0.0010 ms | not_measured | managed | 2.600x |
| `np.sum axis=1 (int32)` | int32 | 100,000 | 0.0104 ms | not_measured | managed | 3.875x |
| `np.sum axis=1 (int32)` | int32 | 10,000,000 | 5.7912 ms | not_measured | managed | 0.830x |
| `np.sum axis=1 (int64)` | int64 | 1,000 | 0.0013 ms | not_measured | managed | 1.615x |
| `np.sum axis=1 (int64)` | int64 | 100,000 | 0.0125 ms | not_measured | managed | 1.328x |
| `np.sum axis=1 (int64)` | int64 | 10,000,000 | 8.9492 ms | not_measured | managed | 0.521x |
| `np.sum axis=1 (int8)` | int8 | 1,000 | 0.0011 ms | not_measured | managed | 2.273x |
| `np.sum axis=1 (int8)` | int8 | 100,000 | 0.0072 ms | not_measured | managed | 5.472x |
| `np.sum axis=1 (int8)` | int8 | 10,000,000 | 0.5724 ms | not_measured | managed | 7.046x |
| `np.sum axis=1 (uint16)` | uint16 | 1,000 | 0.0015 ms | not_measured | managed | 1.800x |
| `np.sum axis=1 (uint16)` | uint16 | 100,000 | 0.0073 ms | not_measured | managed | 5.068x |
| `np.sum axis=1 (uint16)` | uint16 | 10,000,000 | 1.2575 ms | not_measured | managed | 2.678x |
| `np.sum axis=1 (uint32)` | uint32 | 1,000 | 0.0010 ms | not_measured | managed | 2.600x |
| `np.sum axis=1 (uint32)` | uint32 | 100,000 | 0.0105 ms | not_measured | managed | 3.552x |
| `np.sum axis=1 (uint32)` | uint32 | 10,000,000 | 4.9584 ms | not_measured | managed | 0.945x |
| `np.sum axis=1 (uint64)` | uint64 | 1,000 | 0.0011 ms | not_measured | managed | 1.909x |
| `np.sum axis=1 (uint64)` | uint64 | 100,000 | 0.0125 ms | not_measured | managed | 1.464x |
| `np.sum axis=1 (uint64)` | uint64 | 10,000,000 | 11.3436 ms | not_measured | managed | 0.368x |
| `np.sum axis=1 (uint8)` | uint8 | 1,000 | 0.0010 ms | not_measured | managed | 2.800x |
| `np.sum axis=1 (uint8)` | uint8 | 100,000 | 0.0071 ms | not_measured | managed | 5.535x |
| `np.sum axis=1 (uint8)` | uint8 | 10,000,000 | 0.6393 ms | not_measured | managed | 4.968x |
| `np.var (float16)` | float16 | 1,000 | 0.0044 ms | not_measured | managed | 4.114x |
| `np.var (float16)` | float16 | 100,000 | 0.3917 ms | not_measured | managed | 2.342x |
| `np.var (float16)` | float16 | 10,000,000 | 39.4131 ms | not_measured | managed | 2.425x |
| `np.var (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 6.231x |
| `np.var (float32)` | float32 | 100,000 | 0.0201 ms | not_measured | managed | 2.811x |
| `np.var (float32)` | float32 | 10,000,000 | 10.0555 ms | not_measured | managed | 1.754x |
| `np.var (float64)` | float64 | 1,000 | 0.0012 ms | not_measured | managed | 5.750x |
| `np.var (float64)` | float64 | 100,000 | 0.0382 ms | not_measured | managed | 1.602x |
| `np.var (float64)` | float64 | 10,000,000 | 19.0249 ms | not_measured | managed | 1.625x |
| `np.var axis=0 (float16)` | float16 | 1,000 | 0.0109 ms | not_measured | managed | 1.826x |
| `np.var axis=0 (float16)` | float16 | 100,000 | 0.8441 ms | not_measured | managed | 1.409x |
| `np.var axis=0 (float16)` | float16 | 10,000,000 | 220.0234 ms | not_measured | managed | 0.509x |
| `np.var axis=0 (float32)` | float32 | 1,000 | 0.0036 ms | not_measured | managed | 2.417x |
| `np.var axis=0 (float32)` | float32 | 100,000 | 0.0665 ms | not_measured | managed | 0.665x |
| `np.var axis=0 (float32)` | float32 | 10,000,000 | 9.6872 ms | not_measured | managed | 1.624x |
| `np.var axis=0 (float64)` | float64 | 1,000 | 0.0016 ms | not_measured | managed | 4.750x |
| `np.var axis=0 (float64)` | float64 | 100,000 | 0.0468 ms | not_measured | managed | 1.630x |
| `np.var axis=0 (float64)` | float64 | 10,000,000 | 25.5786 ms | not_measured | managed | 1.134x |
| `np.choose(selector, choices) (float64)` | float64 | 1,000 | 0.0071 ms | not_measured | managed | 0.901x |
| `np.choose(selector, choices) (float64)` | float64 | 100,000 | 0.3663 ms | not_measured | managed | 1.641x |
| `np.compress(mask, a) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 1.250x |
| `np.compress(mask, a) (float64)` | float64 | 100,000 | 0.1463 ms | not_measured | managed | 0.580x |
| `np.diag_indices(n) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `np.diag_indices(n) (float64)` | float64 | 100,000 | 0.0022 ms | not_measured | managed | 0.318x |
| `np.diag_indices_from(a) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 1.760x |
| `np.diag_indices_from(a) (float64)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 2.238x |
| `np.extract(mask, a) (float64)` | float64 | 1,000 | 0.0025 ms | not_measured | managed | 1.400x |
| `np.extract(mask, a) (float64)` | float64 | 100,000 | 0.1450 ms | not_measured | managed | 0.601x |
| `np.indices(shape) (float64)` | float64 | 1,000 | 0.0044 ms | not_measured | managed | 0.682x |
| `np.indices(shape) (float64)` | float64 | 100,000 | 0.4353 ms | not_measured | managed | 0.810x |
| `np.mask_indices(n, triu) (float64)` | float64 | 1,000 | 0.0137 ms | not_measured | managed | 0.650x |
| `np.mask_indices(n, triu) (float64)` | float64 | 100,000 | 0.7391 ms | not_measured | managed | 1.023x |
| `np.place(a, mask, values) (float64)` | float64 | 1,000 | 0.0008 ms | not_measured | managed | 1.750x |
| `np.place(a, mask, values) (float64)` | float64 | 100,000 | 0.3309 ms | not_measured | managed | 0.982x |
| `np.put(a, indices, values) (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.241x |
| `np.put(a, indices, values) (float64)` | float64 | 100,000 | 0.1682 ms | not_measured | managed | 0.671x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 1,000 | 0.0036 ms | not_measured | managed | 0.444x |
| `np.ravel_multi_index(coords, dims) (float64)` | float64 | 100,000 | 0.2101 ms | not_measured | managed | 0.365x |
| `np.select(condlist, choicelist) (float64)` | float64 | 1,000 | 0.0056 ms | not_measured | managed | 2.321x |
| `np.select(condlist, choicelist) (float64)` | float64 | 100,000 | 0.2054 ms | not_measured | managed | 3.618x |
| `np.take(a, indices) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.630x |
| `np.take(a, indices) (float64)` | float64 | 100,000 | 0.1578 ms | not_measured | managed | 0.356x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.875x |
| `np.take_along_axis(a, indices) (float64)` | float64 | 100,000 | 0.1077 ms | not_measured | managed | 0.245x |
| `np.tril_indices_from(a) (float64)` | float64 | 1,000 | 0.0075 ms | not_measured | managed | 1.560x |
| `np.tril_indices_from(a) (float64)` | float64 | 100,000 | 0.1695 ms | not_measured | managed | 0.467x |
| `np.triu_indices_from(a) (float64)` | float64 | 1,000 | 0.0070 ms | not_measured | managed | 2.314x |
| `np.triu_indices_from(a) (float64)` | float64 | 100,000 | 0.1647 ms | not_measured | managed | 0.449x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.750x |
| `np.unravel_index(indices, shape) (float64)` | float64 | 100,000 | 0.1640 ms | not_measured | managed | 0.596x |
| `np.where(cond) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.314x |
| `np.where(cond) (float64)` | float64 | 100,000 | 0.1303 ms | not_measured | managed | 0.230x |
| `np.where(cond) (float64)` | float64 | 10,000,000 | 10.7353 ms | not_measured | managed | 0.762x |
| `np.where(cond, a, b) (float64)` | float64 | 1,000 | 0.0035 ms | not_measured | managed | 0.571x |
| `np.where(cond, a, b) (float64)` | float64 | 100,000 | 0.1908 ms | not_measured | managed | 0.213x |
| `np.where(cond, a, b) (float64)` | float64 | 10,000,000 | 37.0600 ms | not_measured | managed | 0.491x |
| `a[100:1000] (contiguous slice)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.083x |
| `a[100:1000] (contiguous slice)` | float64 | 100,000 | 0.0022 ms | not_measured | managed | 0.091x |
| `a[100:1000] (contiguous slice)` | float64 | 10,000,000 | 0.0031 ms | not_measured | managed | 0.065x |
| `a[100:1000] * 2` | float64 | 1,000 | 0.0078 ms | not_measured | managed | 0.154x |
| `a[100:1000] * 2` | float64 | 100,000 | 0.0091 ms | not_measured | managed | 0.110x |
| `a[100:1000] * 2` | float64 | 10,000,000 | 0.0081 ms | not_measured | managed | 0.247x |
| `a[100:1000].copy()` | float64 | 1,000 | 0.0069 ms | not_measured | managed | 0.087x |
| `a[100:1000].copy()` | float64 | 100,000 | 0.0072 ms | not_measured | managed | 0.069x |
| `a[100:1000].copy()` | float64 | 10,000,000 | 0.0081 ms | not_measured | managed | 0.062x |
| `a[10:100, :] (row slice 2D)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.059x |
| `a[10:100, :] (row slice 2D)` | float64 | 100,000 | 0.0046 ms | not_measured | managed | 0.043x |
| `a[10:100, :] (row slice 2D)` | float64 | 10,000,000 | 0.0035 ms | not_measured | managed | 0.057x |
| `a[:, 10:100] (col slice 2D)` | float64 | 1,000 | 0.0033 ms | not_measured | managed | 0.061x |
| `a[:, 10:100] (col slice 2D)` | float64 | 100,000 | 0.0042 ms | not_measured | managed | 0.048x |
| `a[:, 10:100] (col slice 2D)` | float64 | 10,000,000 | 0.0032 ms | not_measured | managed | 0.062x |
| `a[::-1] (reversed)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.100x |
| `a[::-1] (reversed)` | float64 | 100,000 | 0.0022 ms | not_measured | managed | 0.091x |
| `a[::-1] (reversed)` | float64 | 10,000,000 | 0.0020 ms | not_measured | managed | 0.100x |
| `a[::2] (strided slice)` | float64 | 1,000 | 0.0022 ms | not_measured | managed | 0.091x |
| `a[::2] (strided slice)` | float64 | 100,000 | 0.0021 ms | not_measured | managed | 0.095x |
| `a[::2] (strided slice)` | float64 | 10,000,000 | 0.0021 ms | not_measured | managed | 0.095x |
| `a[::2] * 2` | float64 | 1,000 | 0.0117 ms | not_measured | managed | 0.103x |
| `a[::2] * 2` | float64 | 100,000 | 0.2647 ms | not_measured | managed | 0.068x |
| `a[::2] * 2` | float64 | 10,000,000 | 20.2275 ms | not_measured | managed | 0.629x |
| `np.copy(a[100:1000])` | float64 | 1,000 | 0.0075 ms | not_measured | managed | 0.093x |
| `np.copy(a[100:1000])` | float64 | 100,000 | 0.0083 ms | not_measured | managed | 0.084x |
| `np.copy(a[100:1000])` | float64 | 10,000,000 | 0.0067 ms | not_measured | managed | 0.104x |
| `np.argpartition(a, kth) (float32)` | float32 | 1,000 | 0.0169 ms | not_measured | managed | 0.302x |
| `np.argpartition(a, kth) (float32)` | float32 | 100,000 | 1.1138 ms | not_measured | managed | 0.154x |
| `np.argpartition(a, kth) (float64)` | float64 | 1,000 | 0.0176 ms | not_measured | managed | 0.250x |
| `np.argpartition(a, kth) (float64)` | float64 | 100,000 | 0.8752 ms | not_measured | managed | 0.262x |
| `np.argpartition(a, kth) (int32)` | int32 | 1,000 | 0.0155 ms | not_measured | managed | 0.226x |
| `np.argpartition(a, kth) (int32)` | int32 | 100,000 | 1.2725 ms | not_measured | managed | 0.108x |
| `np.argpartition(a, kth) (int64)` | int64 | 1,000 | 0.0216 ms | not_measured | managed | 0.213x |
| `np.argpartition(a, kth) (int64)` | int64 | 100,000 | 0.8868 ms | not_measured | managed | 0.173x |
| `np.argsort(a) (float32)` | float32 | 1,000 | 0.0273 ms | not_measured | managed | 0.436x |
| `np.argsort(a) (float32)` | float32 | 100,000 | 1.9523 ms | not_measured | managed | 0.862x |
| `np.argsort(a) (float32)` | float32 | 10,000,000 | 268.1215 ms | not_measured | managed | 5.512x |
| `np.argsort(a) (float64)` | float64 | 1,000 | 0.0504 ms | not_measured | managed | 0.220x |
| `np.argsort(a) (float64)` | float64 | 100,000 | 3.6474 ms | not_measured | managed | 1.403x |
| `np.argsort(a) (float64)` | float64 | 10,000,000 | 731.8750 ms | not_measured | managed | 1.920x |
| `np.argsort(a) (int32)` | int32 | 1,000 | 0.0293 ms | not_measured | managed | 0.427x |
| `np.argsort(a) (int32)` | int32 | 100,000 | 1.8009 ms | not_measured | managed | 0.238x |
| `np.argsort(a) (int32)` | int32 | 10,000,000 | 232.6069 ms | not_measured | managed | 1.829x |
| `np.argsort(a) (int64)` | int64 | 1,000 | 0.0474 ms | not_measured | managed | 0.304x |
| `np.argsort(a) (int64)` | int64 | 100,000 | 3.2061 ms | not_measured | managed | 0.179x |
| `np.argsort(a) (int64)` | int64 | 10,000,000 | 505.3822 ms | not_measured | managed | 1.014x |
| `np.argwhere(a) (float32)` | float32 | 1,000 | 0.0033 ms | not_measured | managed | 1.364x |
| `np.argwhere(a) (float32)` | float32 | 100,000 | 0.1663 ms | not_measured | managed | 4.575x |
| `np.argwhere(a) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 1.533x |
| `np.argwhere(a) (float64)` | float64 | 100,000 | 0.1619 ms | not_measured | managed | 2.794x |
| `np.argwhere(a) (int32)` | int32 | 1,000 | 0.0043 ms | not_measured | managed | 0.791x |
| `np.argwhere(a) (int32)` | int32 | 100,000 | 0.1763 ms | not_measured | managed | 2.590x |
| `np.argwhere(a) (int64)` | int64 | 1,000 | 0.0045 ms | not_measured | managed | 0.756x |
| `np.argwhere(a) (int64)` | int64 | 100,000 | 0.1832 ms | not_measured | managed | 2.307x |
| `np.flatnonzero(a) (float32)` | float32 | 1,000 | 0.0040 ms | not_measured | managed | 0.850x |
| `np.flatnonzero(a) (float32)` | float32 | 100,000 | 0.2592 ms | not_measured | managed | 1.159x |
| `np.flatnonzero(a) (float64)` | float64 | 1,000 | 0.0038 ms | not_measured | managed | 0.895x |
| `np.flatnonzero(a) (float64)` | float64 | 100,000 | 0.2982 ms | not_measured | managed | 1.143x |
| `np.flatnonzero(a) (int32)` | int32 | 1,000 | 0.0056 ms | not_measured | managed | 0.411x |
| `np.flatnonzero(a) (int32)` | int32 | 100,000 | 0.2526 ms | not_measured | managed | 0.443x |
| `np.flatnonzero(a) (int64)` | int64 | 1,000 | 0.0047 ms | not_measured | managed | 0.511x |
| `np.flatnonzero(a) (int64)` | int64 | 100,000 | 0.3872 ms | not_measured | managed | 0.309x |
| `np.lexsort((b, a)) (float32)` | float32 | 1,000 | 0.0610 ms | not_measured | managed | 0.859x |
| `np.lexsort((b, a)) (float32)` | float32 | 100,000 | 4.9081 ms | not_measured | managed | 3.228x |
| `np.lexsort((b, a)) (float64)` | float64 | 1,000 | 0.0767 ms | not_measured | managed | 0.786x |
| `np.lexsort((b, a)) (float64)` | float64 | 100,000 | 8.3261 ms | not_measured | managed | 2.295x |
| `np.lexsort((b, a)) (int32)` | int32 | 1,000 | 0.0512 ms | not_measured | managed | 0.805x |
| `np.lexsort((b, a)) (int32)` | int32 | 100,000 | 4.2756 ms | not_measured | managed | 2.576x |
| `np.lexsort((b, a)) (int64)` | int64 | 1,000 | 0.0889 ms | not_measured | managed | 0.469x |
| `np.lexsort((b, a)) (int64)` | int64 | 100,000 | 6.4467 ms | not_measured | managed | 1.868x |
| `np.nonzero(a) (float32)` | float32 | 1,000 | 0.0054 ms | not_measured | managed | 0.500x |
| `np.nonzero(a) (float32)` | float32 | 100,000 | 0.1346 ms | not_measured | managed | 1.421x |
| `np.nonzero(a) (float32)` | float32 | 10,000,000 | 27.2509 ms | not_measured | managed | 1.495x |
| `np.nonzero(a) (float64)` | float64 | 1,000 | 0.0059 ms | not_measured | managed | 0.458x |
| `np.nonzero(a) (float64)` | float64 | 100,000 | 0.1705 ms | not_measured | managed | 1.838x |
| `np.nonzero(a) (float64)` | float64 | 10,000,000 | 50.8295 ms | not_measured | managed | 0.576x |
| `np.nonzero(a) (int32)` | int32 | 1,000 | 0.0073 ms | not_measured | managed | 0.247x |
| `np.nonzero(a) (int32)` | int32 | 100,000 | 0.1454 ms | not_measured | managed | 0.800x |
| `np.nonzero(a) (int32)` | int32 | 10,000,000 | 23.6816 ms | not_measured | managed | 1.460x |
| `np.nonzero(a) (int64)` | int64 | 1,000 | 0.0062 ms | not_measured | managed | 0.306x |
| `np.nonzero(a) (int64)` | int64 | 100,000 | 0.1662 ms | not_measured | managed | 0.769x |
| `np.nonzero(a) (int64)` | int64 | 10,000,000 | 34.0275 ms | not_measured | managed | 1.107x |
| `np.partition(a, kth) (float32)` | float32 | 1,000 | 0.0136 ms | not_measured | managed | 0.184x |
| `np.partition(a, kth) (float32)` | float32 | 100,000 | 0.8508 ms | not_measured | managed | 0.130x |
| `np.partition(a, kth) (float64)` | float64 | 1,000 | 0.0149 ms | not_measured | managed | 0.221x |
| `np.partition(a, kth) (float64)` | float64 | 100,000 | 0.7931 ms | not_measured | managed | 0.404x |
| `np.partition(a, kth) (int32)` | int32 | 1,000 | 0.0087 ms | not_measured | managed | 0.218x |
| `np.partition(a, kth) (int32)` | int32 | 100,000 | 0.5483 ms | not_measured | managed | 0.053x |
| `np.partition(a, kth) (int64)` | int64 | 1,000 | 0.0119 ms | not_measured | managed | 0.210x |
| `np.partition(a, kth) (int64)` | int64 | 100,000 | 0.6671 ms | not_measured | managed | 0.131x |
| `np.searchsorted(a, v) (float32)` | float32 | 1,000 | 0.0151 ms | not_measured | managed | 0.543x |
| `np.searchsorted(a, v) (float32)` | float32 | 100,000 | 3.0937 ms | not_measured | managed | 0.689x |
| `np.searchsorted(a, v) (float32)` | float32 | 10,000,000 | 356.2031 ms | not_measured | managed | 0.900x |
| `np.searchsorted(a, v) (float64)` | float64 | 1,000 | 0.0150 ms | not_measured | managed | 0.540x |
| `np.searchsorted(a, v) (float64)` | float64 | 100,000 | 3.1915 ms | not_measured | managed | 0.853x |
| `np.searchsorted(a, v) (float64)` | float64 | 10,000,000 | 373.0022 ms | not_measured | managed | 0.711x |
| `np.searchsorted(a, v) (int32)` | int32 | 1,000 | 0.0117 ms | not_measured | managed | 1.684x |
| `np.searchsorted(a, v) (int32)` | int32 | 100,000 | 3.1980 ms | not_measured | managed | 0.925x |
| `np.searchsorted(a, v) (int32)` | int32 | 10,000,000 | 338.8701 ms | not_measured | managed | 1.295x |
| `np.searchsorted(a, v) (int64)` | int64 | 1,000 | 0.0127 ms | not_measured | managed | 1.535x |
| `np.searchsorted(a, v) (int64)` | int64 | 100,000 | 3.2601 ms | not_measured | managed | 0.925x |
| `np.searchsorted(a, v) (int64)` | int64 | 10,000,000 | 347.6776 ms | not_measured | managed | 1.641x |
| `np.sort(a) (float32)` | float32 | 1,000 | 0.0190 ms | not_measured | managed | 0.126x |
| `np.sort(a) (float32)` | float32 | 100,000 | 1.2959 ms | not_measured | managed | 0.199x |
| `np.sort(a) (float64)` | float64 | 1,000 | 0.0294 ms | not_measured | managed | 0.122x |
| `np.sort(a) (float64)` | float64 | 100,000 | 2.0432 ms | not_measured | managed | 0.647x |
| `np.sort(a) (int32)` | int32 | 1,000 | 0.0107 ms | not_measured | managed | 0.224x |
| `np.sort(a) (int32)` | int32 | 100,000 | 1.4483 ms | not_measured | managed | 0.074x |
| `np.sort(a) (int64)` | int64 | 1,000 | 0.0279 ms | not_measured | managed | 0.233x |
| `np.sort(a) (int64)` | int64 | 100,000 | 2.9208 ms | not_measured | managed | 0.102x |
| `np.sort_complex(a) (float32)` | float32 | 1,000 | 0.0283 ms | not_measured | managed | 0.106x |
| `np.sort_complex(a) (float32)` | float32 | 100,000 | 1.7326 ms | not_measured | managed | 0.353x |
| `np.sort_complex(a) (float64)` | float64 | 1,000 | 0.0382 ms | not_measured | managed | 0.118x |
| `np.sort_complex(a) (float64)` | float64 | 100,000 | 2.8200 ms | not_measured | managed | 0.609x |
| `np.sort_complex(a) (int32)` | int32 | 1,000 | 0.0289 ms | not_measured | managed | 0.114x |
| `np.sort_complex(a) (int32)` | int32 | 100,000 | 1.7565 ms | not_measured | managed | 0.227x |
| `np.sort_complex(a) (int64)` | int64 | 1,000 | 0.0434 ms | not_measured | managed | 0.166x |
| `np.sort_complex(a) (int64)` | int64 | 100,000 | 3.0723 ms | not_measured | managed | 0.216x |
| `np.average(a) (float16)` | float16 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float16)` | float16 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.average(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.bincount(a) (int32)` | int32 | 1,000 | 0.0058 ms | not_measured | managed | 0.259x |
| `np.bincount(a) (int32)` | int32 | 100,000 | 0.4531 ms | not_measured | managed | 0.253x |
| `np.convolve(a, kernel) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.convolve(a, kernel) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.convolve(a, v, mode='same')` | float64 | 100,000 | 0.1257 ms | 2.4239 ms | managed | 8.733x |
| `np.corrcoef(a, b) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.corrcoef(a, b) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 1,000 | not_measured | not_measured | — | — |
| `np.correlate(a, kernel) (float64)` | float64 | 100,000 | not_measured | not_measured | — | — |
| `np.correlate(a, v, mode='same')` | float64 | 100,000 | 0.1179 ms | 2.4322 ms | managed | 10.425x |
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
| `a * a (square via multiply) (float16)` | float16 | 1,000 | 0.0098 ms | not_measured | managed | 0.429x |
| `a * a (square via multiply) (float16)` | float16 | 100,000 | 0.9317 ms | not_measured | managed | 0.303x |
| `a * a (square via multiply) (float16)` | float16 | 10,000,000 | 90.8875 ms | not_measured | managed | 0.816x |
| `a * a (square via multiply) (float32)` | float32 | 1,000 | 0.0021 ms | not_measured | managed | 0.286x |
| `a * a (square via multiply) (float32)` | float32 | 100,000 | 0.0526 ms | not_measured | managed | 0.148x |
| `a * a (square via multiply) (float32)` | float32 | 10,000,000 | 8.6009 ms | not_measured | managed | 0.902x |
| `a * a (square via multiply) (float64)` | float64 | 1,000 | 0.0020 ms | not_measured | managed | 0.350x |
| `a * a (square via multiply) (float64)` | float64 | 100,000 | 0.1163 ms | not_measured | managed | 0.385x |
| `a * a (square via multiply) (float64)` | float64 | 10,000,000 | 21.8269 ms | not_measured | managed | 0.731x |
| `a * a * a (cube via multiply) (float16)` | float16 | 1,000 | 0.0213 ms | not_measured | managed | 0.498x |
| `a * a * a (cube via multiply) (float16)` | float16 | 100,000 | 2.2539 ms | not_measured | managed | 0.402x |
| `a * a * a (cube via multiply) (float16)` | float16 | 10,000,000 | 234.9409 ms | not_measured | managed | 0.765x |
| `a * a * a (cube via multiply) (float32)` | float32 | 1,000 | 0.0038 ms | not_measured | managed | 0.289x |
| `a * a * a (cube via multiply) (float32)` | float32 | 100,000 | 0.1173 ms | not_measured | managed | 0.137x |
| `a * a * a (cube via multiply) (float32)` | float32 | 10,000,000 | 19.1563 ms | not_measured | managed | 0.846x |
| `a * a * a (cube via multiply) (float64)` | float64 | 1,000 | 0.0042 ms | not_measured | managed | 0.333x |
| `a * a * a (cube via multiply) (float64)` | float64 | 100,000 | 0.2304 ms | not_measured | managed | 2.905x |
| `a * a * a (cube via multiply) (float64)` | float64 | 10,000,000 | 49.9937 ms | not_measured | managed | 0.688x |
| `np.abs (float16)` | float16 | 1,000 | 0.0024 ms | not_measured | managed | 0.333x |
| `np.abs (float16)` | float16 | 100,000 | 0.0545 ms | not_measured | managed | 0.510x |
| `np.abs (float16)` | float16 | 10,000,000 | 6.2464 ms | not_measured | managed | 1.399x |
| `np.abs (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `np.abs (float32)` | float32 | 100,000 | 0.0649 ms | not_measured | managed | 0.094x |
| `np.abs (float32)` | float32 | 10,000,000 | 8.9253 ms | not_measured | managed | 1.034x |
| `np.abs (float64)` | float64 | 1,000 | 0.0021 ms | not_measured | managed | 0.286x |
| `np.abs (float64)` | float64 | 100,000 | 0.1269 ms | not_measured | managed | 0.215x |
| `np.abs (float64)` | float64 | 10,000,000 | 20.2367 ms | not_measured | managed | 1.030x |
| `np.absolute(a) (float16)` | float16 | 1,000 | 0.0023 ms | not_measured | managed | 0.522x |
| `np.absolute(a) (float16)` | float16 | 100,000 | 0.0227 ms | not_measured | managed | 1.211x |
| `np.absolute(a) (float16)` | float16 | 10,000,000 | 3.5416 ms | not_measured | managed | 2.144x |
| `np.absolute(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.375x |
| `np.absolute(a) (float32)` | float32 | 100,000 | 0.0513 ms | not_measured | managed | 0.125x |
| `np.absolute(a) (float32)` | float32 | 10,000,000 | 5.3715 ms | not_measured | managed | 1.381x |
| `np.absolute(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.207x |
| `np.absolute(a) (float64)` | float64 | 100,000 | 0.0569 ms | not_measured | managed | 0.578x |
| `np.absolute(a) (float64)` | float64 | 10,000,000 | 14.4572 ms | not_measured | managed | 1.236x |
| `np.acosh(a) (float16)` | float16 | 1,000 | 0.0216 ms | not_measured | managed | 0.403x |
| `np.acosh(a) (float16)` | float16 | 100,000 | 2.0225 ms | not_measured | managed | 0.441x |
| `np.acosh(a) (float16)` | float16 | 10,000,000 | 108.2171 ms | not_measured | managed | 1.651x |
| `np.acosh(a) (float32)` | float32 | 1,000 | 0.0112 ms | not_measured | managed | 0.518x |
| `np.acosh(a) (float32)` | float32 | 100,000 | 0.5766 ms | not_measured | managed | 1.053x |
| `np.acosh(a) (float32)` | float32 | 10,000,000 | 55.7853 ms | not_measured | managed | 1.650x |
| `np.acosh(a) (float64)` | float64 | 1,000 | 0.0145 ms | not_measured | managed | 0.517x |
| `np.acosh(a) (float64)` | float64 | 100,000 | 0.6708 ms | not_measured | managed | 2.114x |
| `np.acosh(a) (float64)` | float64 | 10,000,000 | 72.2551 ms | not_measured | managed | 1.022x |
| `np.angle(a) (float16)` | float16 | 1,000 | 0.0199 ms | not_measured | managed | 0.658x |
| `np.angle(a) (float16)` | float16 | 100,000 | 1.5989 ms | not_measured | managed | 0.708x |
| `np.angle(a) (float16)` | float16 | 10,000,000 | 96.0198 ms | not_measured | managed | 1.132x |
| `np.angle(a) (float32)` | float32 | 1,000 | 0.0159 ms | not_measured | managed | 0.277x |
| `np.angle(a) (float32)` | float32 | 100,000 | 0.7081 ms | not_measured | managed | 1.634x |
| `np.angle(a) (float32)` | float32 | 10,000,000 | 93.7862 ms | not_measured | managed | 0.876x |
| `np.angle(a) (float64)` | float64 | 1,000 | 0.0161 ms | not_measured | managed | 0.292x |
| `np.angle(a) (float64)` | float64 | 100,000 | 0.6767 ms | not_measured | managed | 1.531x |
| `np.angle(a) (float64)` | float64 | 10,000,000 | 78.2865 ms | not_measured | managed | 1.072x |
| `np.arccos(a) (float16)` | float16 | 1,000 | 0.0170 ms | not_measured | managed | 0.418x |
| `np.arccos(a) (float16)` | float16 | 100,000 | 1.2410 ms | not_measured | managed | 0.922x |
| `np.arccos(a) (float16)` | float16 | 10,000,000 | 131.6949 ms | not_measured | managed | 1.593x |
| `np.arccos(a) (float32)` | float32 | 1,000 | 0.0092 ms | not_measured | managed | 0.424x |
| `np.arccos(a) (float32)` | float32 | 100,000 | 1.3629 ms | not_measured | managed | 0.583x |
| `np.arccos(a) (float32)` | float32 | 10,000,000 | 80.2236 ms | not_measured | managed | 1.014x |
| `np.arccos(a) (float64)` | float64 | 1,000 | 0.0103 ms | not_measured | managed | 0.388x |
| `np.arccos(a) (float64)` | float64 | 100,000 | 0.8597 ms | not_measured | managed | 1.773x |
| `np.arccos(a) (float64)` | float64 | 10,000,000 | 95.6056 ms | not_measured | managed | 0.988x |
| `np.arccosh(a) (float16)` | float16 | 1,000 | 0.0212 ms | not_measured | managed | 0.415x |
| `np.arccosh(a) (float16)` | float16 | 100,000 | 2.0669 ms | not_measured | managed | 0.430x |
| `np.arccosh(a) (float16)` | float16 | 10,000,000 | 107.8454 ms | not_measured | managed | 1.581x |
| `np.arccosh(a) (float32)` | float32 | 1,000 | 0.0114 ms | not_measured | managed | 0.518x |
| `np.arccosh(a) (float32)` | float32 | 100,000 | 0.5886 ms | not_measured | managed | 1.011x |
| `np.arccosh(a) (float32)` | float32 | 10,000,000 | 56.1513 ms | not_measured | managed | 1.330x |
| `np.arccosh(a) (float64)` | float64 | 1,000 | 0.0150 ms | not_measured | managed | 0.487x |
| `np.arccosh(a) (float64)` | float64 | 100,000 | 0.6718 ms | not_measured | managed | 2.090x |
| `np.arccosh(a) (float64)` | float64 | 10,000,000 | 72.5841 ms | not_measured | managed | 0.974x |
| `np.arcsin(a) (float16)` | float16 | 1,000 | 0.0180 ms | not_measured | managed | 0.433x |
| `np.arcsin(a) (float16)` | float16 | 100,000 | 1.2380 ms | not_measured | managed | 0.899x |
| `np.arcsin(a) (float16)` | float16 | 10,000,000 | 136.1456 ms | not_measured | managed | 1.442x |
| `np.arcsin(a) (float32)` | float32 | 1,000 | 0.0104 ms | not_measured | managed | 0.404x |
| `np.arcsin(a) (float32)` | float32 | 100,000 | 1.3544 ms | not_measured | managed | 0.606x |
| `np.arcsin(a) (float32)` | float32 | 10,000,000 | 83.6036 ms | not_measured | managed | 1.027x |
| `np.arcsin(a) (float64)` | float64 | 1,000 | 0.0109 ms | not_measured | managed | 0.376x |
| `np.arcsin(a) (float64)` | float64 | 100,000 | 0.8561 ms | not_measured | managed | 1.879x |
| `np.arcsin(a) (float64)` | float64 | 10,000,000 | 101.3700 ms | not_measured | managed | 1.008x |
| `np.arcsinh(a) (float16)` | float16 | 1,000 | 0.0252 ms | not_measured | managed | 0.440x |
| `np.arcsinh(a) (float16)` | float16 | 100,000 | 2.6832 ms | not_measured | managed | 0.472x |
| `np.arcsinh(a) (float16)` | float16 | 10,000,000 | 145.1794 ms | not_measured | managed | 1.573x |
| `np.arcsinh(a) (float32)` | float32 | 1,000 | 0.0165 ms | not_measured | managed | 0.448x |
| `np.arcsinh(a) (float32)` | float32 | 100,000 | 1.6324 ms | not_measured | managed | 0.619x |
| `np.arcsinh(a) (float32)` | float32 | 10,000,000 | 90.8430 ms | not_measured | managed | 1.523x |
| `np.arcsinh(a) (float64)` | float64 | 1,000 | 0.0205 ms | not_measured | managed | 0.424x |
| `np.arcsinh(a) (float64)` | float64 | 100,000 | 0.9771 ms | not_measured | managed | 2.241x |
| `np.arcsinh(a) (float64)` | float64 | 10,000,000 | 103.8435 ms | not_measured | managed | 1.097x |
| `np.arctan(a) (float16)` | float16 | 1,000 | 0.0173 ms | not_measured | managed | 0.445x |
| `np.arctan(a) (float16)` | float16 | 100,000 | 1.4318 ms | not_measured | managed | 0.935x |
| `np.arctan(a) (float16)` | float16 | 10,000,000 | 147.7143 ms | not_measured | managed | 1.447x |
| `np.arctan(a) (float32)` | float32 | 1,000 | 0.0091 ms | not_measured | managed | 0.440x |
| `np.arctan(a) (float32)` | float32 | 100,000 | 1.4485 ms | not_measured | managed | 0.684x |
| `np.arctan(a) (float32)` | float32 | 10,000,000 | 98.3848 ms | not_measured | managed | 0.992x |
| `np.arctan(a) (float64)` | float64 | 1,000 | 0.0101 ms | not_measured | managed | 0.406x |
| `np.arctan(a) (float64)` | float64 | 100,000 | 0.9910 ms | not_measured | managed | 2.042x |
| `np.arctan(a) (float64)` | float64 | 10,000,000 | 106.4615 ms | not_measured | managed | 1.022x |
| `np.arctan2(a, b) (float16)` | float16 | 1,000 | 0.0303 ms | not_measured | managed | 0.363x |
| `np.arctan2(a, b) (float16)` | float16 | 100,000 | 1.8370 ms | not_measured | managed | 0.960x |
| `np.arctan2(a, b) (float16)` | float16 | 10,000,000 | 188.7485 ms | not_measured | managed | 1.622x |
| `np.arctan2(a, b) (float32)` | float32 | 1,000 | 0.0285 ms | not_measured | managed | 0.158x |
| `np.arctan2(a, b) (float32)` | float32 | 100,000 | 3.4777 ms | not_measured | managed | 0.419x |
| `np.arctan2(a, b) (float32)` | float32 | 10,000,000 | 226.3722 ms | not_measured | managed | 0.731x |
| `np.arctan2(a, b) (float64)` | float64 | 1,000 | 0.0176 ms | not_measured | managed | 0.347x |
| `np.arctan2(a, b) (float64)` | float64 | 100,000 | 1.4207 ms | not_measured | managed | 2.006x |
| `np.arctan2(a, b) (float64)` | float64 | 10,000,000 | 151.9321 ms | not_measured | managed | 1.000x |
| `np.arctanh(a) (float16)` | float16 | 1,000 | 0.0217 ms | not_measured | managed | 0.442x |
| `np.arctanh(a) (float16)` | float16 | 100,000 | 2.1549 ms | not_measured | managed | 0.502x |
| `np.arctanh(a) (float16)` | float16 | 10,000,000 | 125.8855 ms | not_measured | managed | 1.641x |
| `np.arctanh(a) (float32)` | float32 | 1,000 | 0.0125 ms | not_measured | managed | 0.448x |
| `np.arctanh(a) (float32)` | float32 | 100,000 | 0.8486 ms | not_measured | managed | 0.998x |
| `np.arctanh(a) (float32)` | float32 | 10,000,000 | 81.6247 ms | not_measured | managed | 1.292x |
| `np.arctanh(a) (float64)` | float64 | 1,000 | 0.0138 ms | not_measured | managed | 0.457x |
| `np.arctanh(a) (float64)` | float64 | 100,000 | 0.8676 ms | not_measured | managed | 1.819x |
| `np.arctanh(a) (float64)` | float64 | 10,000,000 | 92.6802 ms | not_measured | managed | 0.986x |
| `np.around (float16)` | float16 | 1,000 | 0.0081 ms | not_measured | managed | 0.679x |
| `np.around (float16)` | float16 | 100,000 | 0.7090 ms | not_measured | managed | 0.580x |
| `np.around (float16)` | float16 | 10,000,000 | 70.3858 ms | not_measured | managed | 0.791x |
| `np.around (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.500x |
| `np.around (float32)` | float32 | 100,000 | 0.0538 ms | not_measured | managed | 0.123x |
| `np.around (float32)` | float32 | 10,000,000 | 9.3868 ms | not_measured | managed | 0.888x |
| `np.around (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.462x |
| `np.around (float64)` | float64 | 100,000 | 0.1062 ms | not_measured | managed | 0.272x |
| `np.around (float64)` | float64 | 10,000,000 | 20.6300 ms | not_measured | managed | 1.058x |
| `np.asinh(a) (float16)` | float16 | 1,000 | 0.0264 ms | not_measured | managed | 0.420x |
| `np.asinh(a) (float16)` | float16 | 100,000 | 2.6947 ms | not_measured | managed | 0.479x |
| `np.asinh(a) (float16)` | float16 | 10,000,000 | 147.2079 ms | not_measured | managed | 1.558x |
| `np.asinh(a) (float32)` | float32 | 1,000 | 0.0163 ms | not_measured | managed | 0.466x |
| `np.asinh(a) (float32)` | float32 | 100,000 | 0.9132 ms | not_measured | managed | 1.081x |
| `np.asinh(a) (float32)` | float32 | 10,000,000 | 89.5922 ms | not_measured | managed | 1.203x |
| `np.asinh(a) (float64)` | float64 | 1,000 | 0.0215 ms | not_measured | managed | 0.419x |
| `np.asinh(a) (float64)` | float64 | 100,000 | 0.9759 ms | not_measured | managed | 2.244x |
| `np.asinh(a) (float64)` | float64 | 10,000,000 | 102.9544 ms | not_measured | managed | 1.008x |
| `np.atanh(a) (float16)` | float16 | 1,000 | 0.0219 ms | not_measured | managed | 0.479x |
| `np.atanh(a) (float16)` | float16 | 100,000 | 2.2247 ms | not_measured | managed | 0.491x |
| `np.atanh(a) (float16)` | float16 | 10,000,000 | 126.7734 ms | not_measured | managed | 0.919x |
| `np.atanh(a) (float32)` | float32 | 1,000 | 0.0123 ms | not_measured | managed | 0.496x |
| `np.atanh(a) (float32)` | float32 | 100,000 | 0.8465 ms | not_measured | managed | 1.717x |
| `np.atanh(a) (float32)` | float32 | 10,000,000 | 82.6821 ms | not_measured | managed | 1.293x |
| `np.atanh(a) (float64)` | float64 | 1,000 | 0.0138 ms | not_measured | managed | 0.457x |
| `np.atanh(a) (float64)` | float64 | 100,000 | 0.8612 ms | not_measured | managed | 1.793x |
| `np.atanh(a) (float64)` | float64 | 10,000,000 | 93.6445 ms | not_measured | managed | 0.981x |
| `np.cbrt(a) (float16)` | float16 | 1,000 | 0.0253 ms | not_measured | managed | 0.423x |
| `np.cbrt(a) (float16)` | float16 | 100,000 | 2.4350 ms | not_measured | managed | 0.886x |
| `np.cbrt(a) (float16)` | float16 | 10,000,000 | 249.0228 ms | not_measured | managed | 0.476x |
| `np.cbrt(a) (float32)` | float32 | 1,000 | 0.0147 ms | not_measured | managed | 0.429x |
| `np.cbrt(a) (float32)` | float32 | 100,000 | 1.4943 ms | not_measured | managed | 1.040x |
| `np.cbrt(a) (float32)` | float32 | 10,000,000 | 146.5038 ms | not_measured | managed | 0.634x |
| `np.cbrt(a) (float64)` | float64 | 1,000 | 0.0198 ms | not_measured | managed | 0.495x |
| `np.cbrt(a) (float64)` | float64 | 100,000 | 2.1025 ms | not_measured | managed | 1.064x |
| `np.cbrt(a) (float64)` | float64 | 10,000,000 | 216.8700 ms | not_measured | managed | 0.555x |
| `np.ceil (float16)` | float16 | 1,000 | 0.0066 ms | not_measured | managed | 0.636x |
| `np.ceil (float16)` | float16 | 100,000 | 0.6256 ms | not_measured | managed | 0.668x |
| `np.ceil (float16)` | float16 | 10,000,000 | 62.7053 ms | not_measured | managed | 0.870x |
| `np.ceil (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.300x |
| `np.ceil (float32)` | float32 | 100,000 | 0.0588 ms | not_measured | managed | 0.102x |
| `np.ceil (float32)` | float32 | 10,000,000 | 7.8385 ms | not_measured | managed | 1.409x |
| `np.ceil (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.222x |
| `np.ceil (float64)` | float64 | 100,000 | 0.1134 ms | not_measured | managed | 0.242x |
| `np.ceil (float64)` | float64 | 10,000,000 | 20.2017 ms | not_measured | managed | 1.212x |
| `np.clip(a, -10, 10) (float16)` | float16 | 1,000 | 0.0101 ms | not_measured | managed | 0.881x |
| `np.clip(a, -10, 10) (float16)` | float16 | 100,000 | 1.0309 ms | not_measured | managed | 0.902x |
| `np.clip(a, -10, 10) (float16)` | float16 | 10,000,000 | 104.7051 ms | not_measured | managed | 1.293x |
| `np.clip(a, -10, 10) (float32)` | float32 | 1,000 | 0.0089 ms | not_measured | managed | 0.247x |
| `np.clip(a, -10, 10) (float32)` | float32 | 100,000 | 0.0693 ms | not_measured | managed | 0.121x |
| `np.clip(a, -10, 10) (float32)` | float32 | 10,000,000 | 8.3864 ms | not_measured | managed | 0.980x |
| `np.clip(a, -10, 10) (float64)` | float64 | 1,000 | 0.0066 ms | not_measured | managed | 0.303x |
| `np.clip(a, -10, 10) (float64)` | float64 | 100,000 | 0.1240 ms | not_measured | managed | 0.390x |
| `np.clip(a, -10, 10) (float64)` | float64 | 10,000,000 | 20.9332 ms | not_measured | managed | 0.736x |
| `np.conj(a) (float16)` | float16 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.conj(a) (float16)` | float16 | 100,000 | 0.0226 ms | not_measured | managed | 0.965x |
| `np.conj(a) (float16)` | float16 | 10,000,000 | 3.7158 ms | not_measured | managed | 1.308x |
| `np.conj(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.316x |
| `np.conj(a) (float32)` | float32 | 100,000 | 0.0314 ms | not_measured | managed | 1.169x |
| `np.conj(a) (float32)` | float32 | 10,000,000 | 5.7231 ms | not_measured | managed | 2.119x |
| `np.conj(a) (float64)` | float64 | 1,000 | 0.0024 ms | not_measured | managed | 0.250x |
| `np.conj(a) (float64)` | float64 | 100,000 | 0.0596 ms | not_measured | managed | 0.554x |
| `np.conj(a) (float64)` | float64 | 10,000,000 | 13.8381 ms | not_measured | managed | 1.276x |
| `np.conjugate(a) (float16)` | float16 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.conjugate(a) (float16)` | float16 | 100,000 | 0.0451 ms | not_measured | managed | 0.481x |
| `np.conjugate(a) (float16)` | float16 | 10,000,000 | 3.7965 ms | not_measured | managed | 1.255x |
| `np.conjugate(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.353x |
| `np.conjugate(a) (float32)` | float32 | 100,000 | 0.0325 ms | not_measured | managed | 1.114x |
| `np.conjugate(a) (float32)` | float32 | 10,000,000 | 5.2825 ms | not_measured | managed | 2.109x |
| `np.conjugate(a) (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.281x |
| `np.conjugate(a) (float64)` | float64 | 100,000 | 0.0652 ms | not_measured | managed | 0.479x |
| `np.conjugate(a) (float64)` | float64 | 10,000,000 | 13.3127 ms | not_measured | managed | 1.547x |
| `np.cos (float16)` | float16 | 1,000 | 0.0167 ms | not_measured | managed | 0.437x |
| `np.cos (float16)` | float16 | 100,000 | 1.9044 ms | not_measured | managed | 0.551x |
| `np.cos (float16)` | float16 | 10,000,000 | 192.4945 ms | not_measured | managed | 0.868x |
| `np.cos (float32)` | float32 | 1,000 | 0.0033 ms | not_measured | managed | 0.394x |
| `np.cos (float32)` | float32 | 100,000 | 0.2683 ms | not_measured | managed | 0.291x |
| `np.cos (float32)` | float32 | 10,000,000 | 27.5921 ms | not_measured | managed | 0.441x |
| `np.cos (float64)` | float64 | 1,000 | 0.0116 ms | not_measured | managed | 0.474x |
| `np.cos (float64)` | float64 | 100,000 | 1.2375 ms | not_measured | managed | 1.495x |
| `np.cos (float64)` | float64 | 10,000,000 | 130.0599 ms | not_measured | managed | 0.735x |
| `np.cosh(a) (float16)` | float16 | 1,000 | 0.0218 ms | not_measured | managed | 0.381x |
| `np.cosh(a) (float16)` | float16 | 100,000 | 1.1579 ms | not_measured | managed | 0.830x |
| `np.cosh(a) (float16)` | float16 | 10,000,000 | 116.5615 ms | not_measured | managed | 1.423x |
| `np.cosh(a) (float32)` | float32 | 1,000 | 0.0112 ms | not_measured | managed | 0.375x |
| `np.cosh(a) (float32)` | float32 | 100,000 | 1.2111 ms | not_measured | managed | 0.592x |
| `np.cosh(a) (float32)` | float32 | 10,000,000 | 63.7419 ms | not_measured | managed | 1.316x |
| `np.cosh(a) (float64)` | float64 | 1,000 | 0.0121 ms | not_measured | managed | 0.339x |
| `np.cosh(a) (float64)` | float64 | 100,000 | 0.6007 ms | not_measured | managed | 2.108x |
| `np.cosh(a) (float64)` | float64 | 10,000,000 | 66.2618 ms | not_measured | managed | 1.074x |
| `np.deg2rad(a) (float16)` | float16 | 1,000 | 0.0075 ms | not_measured | managed | 0.560x |
| `np.deg2rad(a) (float16)` | float16 | 100,000 | 0.6980 ms | not_measured | managed | 0.580x |
| `np.deg2rad(a) (float16)` | float16 | 10,000,000 | 37.4599 ms | not_measured | managed | 1.069x |
| `np.deg2rad(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.706x |
| `np.deg2rad(a) (float32)` | float32 | 100,000 | 0.0346 ms | not_measured | managed | 3.604x |
| `np.deg2rad(a) (float32)` | float32 | 10,000,000 | 5.9526 ms | not_measured | managed | 3.121x |
| `np.deg2rad(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.483x |
| `np.deg2rad(a) (float64)` | float64 | 100,000 | 0.0597 ms | not_measured | managed | 2.050x |
| `np.deg2rad(a) (float64)` | float64 | 10,000,000 | 14.3975 ms | not_measured | managed | 1.265x |
| `np.degrees(a) (float16)` | float16 | 1,000 | 0.0075 ms | not_measured | managed | 0.653x |
| `np.degrees(a) (float16)` | float16 | 100,000 | 0.6999 ms | not_measured | managed | 0.571x |
| `np.degrees(a) (float16)` | float16 | 10,000,000 | 37.4024 ms | not_measured | managed | 1.096x |
| `np.degrees(a) (float32)` | float32 | 1,000 | 0.0017 ms | not_measured | managed | 0.706x |
| `np.degrees(a) (float32)` | float32 | 100,000 | 0.0289 ms | not_measured | managed | 4.879x |
| `np.degrees(a) (float32)` | float32 | 10,000,000 | 5.4521 ms | not_measured | managed | 2.669x |
| `np.degrees(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.500x |
| `np.degrees(a) (float64)` | float64 | 100,000 | 0.0554 ms | not_measured | managed | 2.076x |
| `np.degrees(a) (float64)` | float64 | 10,000,000 | 14.6127 ms | not_measured | managed | 1.581x |
| `np.exp (float16)` | float16 | 1,000 | 0.0061 ms | not_measured | managed | 0.770x |
| `np.exp (float16)` | float16 | 100,000 | 0.9304 ms | not_measured | managed | 0.460x |
| `np.exp (float16)` | float16 | 10,000,000 | 111.9641 ms | not_measured | managed | 0.700x |
| `np.exp (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.455x |
| `np.exp (float32)` | float32 | 100,000 | 0.0662 ms | not_measured | managed | 0.884x |
| `np.exp (float32)` | float32 | 10,000,000 | 7.3920 ms | not_measured | managed | 1.425x |
| `np.exp (float64)` | float64 | 1,000 | 0.0064 ms | not_measured | managed | 0.484x |
| `np.exp (float64)` | float64 | 100,000 | 0.2865 ms | not_measured | managed | 1.869x |
| `np.exp (float64)` | float64 | 10,000,000 | 52.9755 ms | not_measured | managed | 0.716x |
| `np.exp2 (float16)` | float16 | 1,000 | 0.0097 ms | not_measured | managed | 0.464x |
| `np.exp2 (float16)` | float16 | 100,000 | 1.5392 ms | not_measured | managed | 0.268x |
| `np.exp2 (float16)` | float16 | 10,000,000 | 182.9342 ms | not_measured | managed | 0.430x |
| `np.exp2 (float32)` | float32 | 1,000 | 0.0102 ms | not_measured | managed | 0.206x |
| `np.exp2 (float32)` | float32 | 100,000 | 0.9176 ms | not_measured | managed | 0.176x |
| `np.exp2 (float32)` | float32 | 10,000,000 | 161.8345 ms | not_measured | managed | 0.123x |
| `np.exp2 (float64)` | float64 | 1,000 | 0.0097 ms | not_measured | managed | 0.247x |
| `np.exp2 (float64)` | float64 | 100,000 | 0.8425 ms | not_measured | managed | 0.566x |
| `np.exp2 (float64)` | float64 | 10,000,000 | 156.3296 ms | not_measured | managed | 0.248x |
| `np.expm1 (float16)` | float16 | 1,000 | 0.0088 ms | not_measured | managed | 0.693x |
| `np.expm1 (float16)` | float16 | 100,000 | 1.0714 ms | not_measured | managed | 0.477x |
| `np.expm1 (float16)` | float16 | 10,000,000 | 135.8391 ms | not_measured | managed | 0.702x |
| `np.expm1 (float32)` | float32 | 1,000 | 0.0040 ms | not_measured | managed | 0.775x |
| `np.expm1 (float32)` | float32 | 100,000 | 0.1945 ms | not_measured | managed | 1.297x |
| `np.expm1 (float32)` | float32 | 10,000,000 | 35.2353 ms | not_measured | managed | 0.790x |
| `np.expm1 (float64)` | float64 | 1,000 | 0.0049 ms | not_measured | managed | 0.816x |
| `np.expm1 (float64)` | float64 | 100,000 | 0.2847 ms | not_measured | managed | 2.596x |
| `np.expm1 (float64)` | float64 | 10,000,000 | 53.4116 ms | not_measured | managed | 1.057x |
| `np.floor (float16)` | float16 | 1,000 | 0.0069 ms | not_measured | managed | 0.725x |
| `np.floor (float16)` | float16 | 100,000 | 0.6266 ms | not_measured | managed | 0.677x |
| `np.floor (float16)` | float16 | 10,000,000 | 60.8301 ms | not_measured | managed | 0.909x |
| `np.floor (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.floor (float32)` | float32 | 100,000 | 0.0578 ms | not_measured | managed | 0.109x |
| `np.floor (float32)` | float32 | 10,000,000 | 9.2293 ms | not_measured | managed | 0.836x |
| `np.floor (float64)` | float64 | 1,000 | 0.0034 ms | not_measured | managed | 0.235x |
| `np.floor (float64)` | float64 | 100,000 | 0.1223 ms | not_measured | managed | 0.331x |
| `np.floor (float64)` | float64 | 10,000,000 | 19.8208 ms | not_measured | managed | 0.992x |
| `np.imag(a) (float16)` | float16 | 1,000 | 0.0021 ms | not_measured | managed | 0.190x |
| `np.imag(a) (float16)` | float16 | 100,000 | 0.0017 ms | not_measured | managed | 1.706x |
| `np.imag(a) (float16)` | float16 | 10,000,000 | not_measured | not_measured | — | — |
| `np.imag(a) (float32)` | float32 | 1,000 | 0.0020 ms | not_measured | managed | 0.200x |
| `np.imag(a) (float32)` | float32 | 100,000 | 0.0015 ms | not_measured | managed | 9.133x |
| `np.imag(a) (float32)` | float32 | 10,000,000 | not_measured | not_measured | — | — |
| `np.imag(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.138x |
| `np.imag(a) (float64)` | float64 | 100,000 | 0.0019 ms | not_measured | managed | 20.053x |
| `np.imag(a) (float64)` | float64 | 10,000,000 | not_measured | not_measured | — | — |
| `np.log (float16)` | float16 | 1,000 | 0.0129 ms | not_measured | managed | 0.372x |
| `np.log (float16)` | float16 | 100,000 | 1.2412 ms | not_measured | managed | 0.349x |
| `np.log (float16)` | float16 | 10,000,000 | 64.0271 ms | not_measured | managed | 1.383x |
| `np.log (float32)` | float32 | 1,000 | 0.0031 ms | not_measured | managed | 0.548x |
| `np.log (float32)` | float32 | 100,000 | 0.2306 ms | not_measured | managed | 0.393x |
| `np.log (float32)` | float32 | 10,000,000 | 22.9332 ms | not_measured | managed | 0.565x |
| `np.log (float64)` | float64 | 1,000 | 0.0078 ms | not_measured | managed | 0.359x |
| `np.log (float64)` | float64 | 100,000 | 0.5483 ms | not_measured | managed | 0.944x |
| `np.log (float64)` | float64 | 10,000,000 | 57.7467 ms | not_measured | managed | 0.659x |
| `np.log10 (float16)` | float16 | 1,000 | 0.0132 ms | not_measured | managed | 0.379x |
| `np.log10 (float16)` | float16 | 100,000 | 1.2709 ms | not_measured | managed | 0.359x |
| `np.log10 (float16)` | float16 | 10,000,000 | 137.6343 ms | not_measured | managed | 0.689x |
| `np.log10 (float32)` | float32 | 1,000 | 0.0070 ms | not_measured | managed | 0.343x |
| `np.log10 (float32)` | float32 | 100,000 | 0.4823 ms | not_measured | managed | 0.434x |
| `np.log10 (float32)` | float32 | 10,000,000 | 45.9167 ms | not_measured | managed | 0.512x |
| `np.log10 (float64)` | float64 | 1,000 | 0.0082 ms | not_measured | managed | 0.354x |
| `np.log10 (float64)` | float64 | 100,000 | 0.5676 ms | not_measured | managed | 0.887x |
| `np.log10 (float64)` | float64 | 10,000,000 | 76.5100 ms | not_measured | managed | 0.620x |
| `np.log1p (float16)` | float16 | 1,000 | 0.0138 ms | not_measured | managed | 0.435x |
| `np.log1p (float16)` | float16 | 100,000 | 1.3093 ms | not_measured | managed | 0.427x |
| `np.log1p (float16)` | float16 | 10,000,000 | 132.9394 ms | not_measured | managed | 0.918x |
| `np.log1p (float32)` | float32 | 1,000 | 0.0056 ms | not_measured | managed | 0.589x |
| `np.log1p (float32)` | float32 | 100,000 | 0.4632 ms | not_measured | managed | 0.640x |
| `np.log1p (float32)` | float32 | 10,000,000 | 45.5056 ms | not_measured | managed | 0.705x |
| `np.log1p (float64)` | float64 | 1,000 | 0.0087 ms | not_measured | managed | 0.437x |
| `np.log1p (float64)` | float64 | 100,000 | 0.5271 ms | not_measured | managed | 1.480x |
| `np.log1p (float64)` | float64 | 10,000,000 | 34.8915 ms | not_measured | managed | 1.250x |
| `np.log2 (float16)` | float16 | 1,000 | 0.0112 ms | not_measured | managed | 0.473x |
| `np.log2 (float16)` | float16 | 100,000 | 1.0931 ms | not_measured | managed | 0.417x |
| `np.log2 (float16)` | float16 | 10,000,000 | 109.9325 ms | not_measured | managed | 0.743x |
| `np.log2 (float32)` | float32 | 1,000 | 0.0064 ms | not_measured | managed | 0.375x |
| `np.log2 (float32)` | float32 | 100,000 | 0.4729 ms | not_measured | managed | 0.414x |
| `np.log2 (float32)` | float32 | 10,000,000 | 46.4819 ms | not_measured | managed | 0.494x |
| `np.log2 (float64)` | float64 | 1,000 | 0.0088 ms | not_measured | managed | 0.500x |
| `np.log2 (float64)` | float64 | 100,000 | 0.7933 ms | not_measured | managed | 1.056x |
| `np.log2 (float64)` | float64 | 10,000,000 | 82.0834 ms | not_measured | managed | 0.652x |
| `np.modf(a) (float32)` | float32 | 1,000 | 0.0063 ms | not_measured | managed | 0.349x |
| `np.modf(a) (float32)` | float32 | 100,000 | 0.1662 ms | not_measured | managed | 2.842x |
| `np.modf(a) (float32)` | float32 | 10,000,000 | 19.8888 ms | not_measured | managed | 2.044x |
| `np.modf(a) (float64)` | float64 | 1,000 | 0.0067 ms | not_measured | managed | 0.358x |
| `np.modf(a) (float64)` | float64 | 100,000 | 0.2979 ms | not_measured | managed | 2.586x |
| `np.modf(a) (float64)` | float64 | 10,000,000 | 81.9793 ms | not_measured | managed | 0.567x |
| `np.negative(a) (float16)` | float16 | 1,000 | 0.0025 ms | not_measured | managed | 0.280x |
| `np.negative(a) (float16)` | float16 | 100,000 | 0.0551 ms | not_measured | managed | 1.069x |
| `np.negative(a) (float16)` | float16 | 10,000,000 | 7.3521 ms | not_measured | managed | 0.655x |
| `np.negative(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.negative(a) (float32)` | float32 | 100,000 | 0.0556 ms | not_measured | managed | 0.255x |
| `np.negative(a) (float32)` | float32 | 10,000,000 | 9.5495 ms | not_measured | managed | 0.826x |
| `np.negative(a) (float64)` | float64 | 1,000 | 0.0027 ms | not_measured | managed | 0.185x |
| `np.negative(a) (float64)` | float64 | 100,000 | 0.1232 ms | not_measured | managed | 0.209x |
| `np.negative(a) (float64)` | float64 | 10,000,000 | 19.9179 ms | not_measured | managed | 0.807x |
| `np.positive(a) (float16)` | float16 | 1,000 | 0.0014 ms | not_measured | managed | 0.429x |
| `np.positive(a) (float16)` | float16 | 100,000 | 0.0268 ms | not_measured | managed | 1.478x |
| `np.positive(a) (float16)` | float16 | 10,000,000 | 3.6901 ms | not_measured | managed | 1.144x |
| `np.positive(a) (float32)` | float32 | 1,000 | 0.0018 ms | not_measured | managed | 0.333x |
| `np.positive(a) (float32)` | float32 | 100,000 | 0.0510 ms | not_measured | managed | 0.582x |
| `np.positive(a) (float32)` | float32 | 10,000,000 | 6.1440 ms | not_measured | managed | 1.265x |
| `np.positive(a) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.233x |
| `np.positive(a) (float64)` | float64 | 100,000 | 0.1054 ms | not_measured | managed | 0.380x |
| `np.positive(a) (float64)` | float64 | 10,000,000 | 14.4107 ms | not_measured | managed | 1.068x |
| `np.power(a, 0.5) (float16)` | float16 | 1,000 | 0.0077 ms | not_measured | managed | 1.182x |
| `np.power(a, 0.5) (float16)` | float16 | 100,000 | 0.6636 ms | not_measured | managed | 1.229x |
| `np.power(a, 0.5) (float16)` | float16 | 10,000,000 | 65.7102 ms | not_measured | managed | 2.847x |
| `np.power(a, 0.5) (float32)` | float32 | 1,000 | 0.0045 ms | not_measured | managed | 0.444x |
| `np.power(a, 0.5) (float32)` | float32 | 100,000 | 0.0761 ms | not_measured | managed | 1.671x |
| `np.power(a, 0.5) (float32)` | float32 | 10,000,000 | 8.6266 ms | not_measured | managed | 1.837x |
| `np.power(a, 0.5) (float64)` | float64 | 1,000 | 0.0058 ms | not_measured | managed | 0.362x |
| `np.power(a, 0.5) (float64)` | float64 | 100,000 | 0.3028 ms | not_measured | managed | 2.476x |
| `np.power(a, 0.5) (float64)` | float64 | 10,000,000 | 34.6539 ms | not_measured | managed | 0.698x |
| `np.power(a, 2) (float16)` | float16 | 1,000 | 0.0107 ms | not_measured | managed | 0.953x |
| `np.power(a, 2) (float16)` | float16 | 100,000 | 0.9377 ms | not_measured | managed | 1.115x |
| `np.power(a, 2) (float16)` | float16 | 10,000,000 | 92.0102 ms | not_measured | managed | 2.438x |
| `np.power(a, 2) (float32)` | float32 | 1,000 | 0.0039 ms | not_measured | managed | 0.590x |
| `np.power(a, 2) (float32)` | float32 | 100,000 | 0.0567 ms | not_measured | managed | 2.829x |
| `np.power(a, 2) (float32)` | float32 | 10,000,000 | 10.1392 ms | not_measured | managed | 1.841x |
| `np.power(a, 2) (float64)` | float64 | 1,000 | 0.0047 ms | not_measured | managed | 0.489x |
| `np.power(a, 2) (float64)` | float64 | 100,000 | 0.1147 ms | not_measured | managed | 2.077x |
| `np.power(a, 2) (float64)` | float64 | 10,000,000 | 22.4176 ms | not_measured | managed | 1.031x |
| `np.power(a, 3) (float16)` | float16 | 1,000 | 0.0357 ms | not_measured | managed | 0.331x |
| `np.power(a, 3) (float16)` | float16 | 100,000 | 4.3412 ms | not_measured | managed | 0.341x |
| `np.power(a, 3) (float16)` | float16 | 10,000,000 | 405.5648 ms | not_measured | managed | 0.688x |
| `np.power(a, 3) (float32)` | float32 | 1,000 | 0.0141 ms | not_measured | managed | 0.418x |
| `np.power(a, 3) (float32)` | float32 | 100,000 | 1.3487 ms | not_measured | managed | 0.506x |
| `np.power(a, 3) (float32)` | float32 | 10,000,000 | 131.1496 ms | not_measured | managed | 0.545x |
| `np.power(a, 3) (float64)` | float64 | 1,000 | 0.0231 ms | not_measured | managed | 0.437x |
| `np.power(a, 3) (float64)` | float64 | 100,000 | 2.0608 ms | not_measured | managed | 1.193x |
| `np.power(a, 3) (float64)` | float64 | 10,000,000 | 209.2461 ms | not_measured | managed | 0.554x |
| `np.rad2deg(a) (float16)` | float16 | 1,000 | 0.0075 ms | not_measured | managed | 0.653x |
| `np.rad2deg(a) (float16)` | float16 | 100,000 | 0.7071 ms | not_measured | managed | 0.570x |
| `np.rad2deg(a) (float16)` | float16 | 10,000,000 | 37.7264 ms | not_measured | managed | 1.046x |
| `np.rad2deg(a) (float32)` | float32 | 1,000 | 0.0013 ms | not_measured | managed | 0.923x |
| `np.rad2deg(a) (float32)` | float32 | 100,000 | 0.0271 ms | not_measured | managed | 4.613x |
| `np.rad2deg(a) (float32)` | float32 | 10,000,000 | 4.1600 ms | not_measured | managed | 3.747x |
| `np.rad2deg(a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.538x |
| `np.rad2deg(a) (float64)` | float64 | 100,000 | 0.0573 ms | not_measured | managed | 2.120x |
| `np.rad2deg(a) (float64)` | float64 | 10,000,000 | 14.4298 ms | not_measured | managed | 1.305x |
| `np.radians(a) (float16)` | float16 | 1,000 | 0.0079 ms | not_measured | managed | 0.658x |
| `np.radians(a) (float16)` | float16 | 100,000 | 0.6991 ms | not_measured | managed | 0.554x |
| `np.radians(a) (float16)` | float16 | 10,000,000 | 37.5809 ms | not_measured | managed | 1.001x |
| `np.radians(a) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.444x |
| `np.radians(a) (float32)` | float32 | 100,000 | 0.0315 ms | not_measured | managed | 3.860x |
| `np.radians(a) (float32)` | float32 | 10,000,000 | 7.4800 ms | not_measured | managed | 1.987x |
| `np.radians(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.483x |
| `np.radians(a) (float64)` | float64 | 100,000 | 0.0594 ms | not_measured | managed | 2.138x |
| `np.radians(a) (float64)` | float64 | 10,000,000 | 14.6615 ms | not_measured | managed | 1.245x |
| `np.real(a) (float16)` | float16 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float16)` | float16 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float32)` | float32 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 1,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 100,000 | 0.0000 ms | not_measured | managed | — |
| `np.real(a) (float64)` | float64 | 10,000,000 | 0.0000 ms | not_measured | managed | — |
| `np.reciprocal(a) (float16)` | float16 | 1,000 | 0.0081 ms | not_measured | managed | 0.309x |
| `np.reciprocal(a) (float16)` | float16 | 100,000 | 0.7684 ms | not_measured | managed | 0.635x |
| `np.reciprocal(a) (float16)` | float16 | 10,000,000 | 78.6298 ms | not_measured | managed | 0.338x |
| `np.reciprocal(a) (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.208x |
| `np.reciprocal(a) (float32)` | float32 | 100,000 | 0.0645 ms | not_measured | managed | 0.983x |
| `np.reciprocal(a) (float32)` | float32 | 10,000,000 | 9.1301 ms | not_measured | managed | 0.805x |
| `np.reciprocal(a) (float64)` | float64 | 1,000 | 0.0028 ms | not_measured | managed | 0.286x |
| `np.reciprocal(a) (float64)` | float64 | 100,000 | 0.2015 ms | not_measured | managed | 1.008x |
| `np.reciprocal(a) (float64)` | float64 | 10,000,000 | 24.5140 ms | not_measured | managed | 0.635x |
| `np.rint(a) (float16)` | float16 | 1,000 | 0.0082 ms | not_measured | managed | 0.646x |
| `np.rint(a) (float16)` | float16 | 100,000 | 0.5072 ms | not_measured | managed | 0.972x |
| `np.rint(a) (float16)` | float16 | 10,000,000 | 50.1083 ms | not_measured | managed | 1.031x |
| `np.rint(a) (float32)` | float32 | 1,000 | 0.0016 ms | not_measured | managed | 0.312x |
| `np.rint(a) (float32)` | float32 | 100,000 | 0.0277 ms | not_measured | managed | 0.581x |
| `np.rint(a) (float32)` | float32 | 10,000,000 | 6.9712 ms | not_measured | managed | 1.642x |
| `np.rint(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.172x |
| `np.rint(a) (float64)` | float64 | 100,000 | 0.0531 ms | not_measured | managed | 0.838x |
| `np.rint(a) (float64)` | float64 | 10,000,000 | 14.4474 ms | not_measured | managed | 1.438x |
| `np.round(a) (float16)` | float16 | 1,000 | 0.0083 ms | not_measured | managed | 0.723x |
| `np.round(a) (float16)` | float16 | 100,000 | 0.4994 ms | not_measured | managed | 0.997x |
| `np.round(a) (float16)` | float16 | 10,000,000 | 49.8550 ms | not_measured | managed | 0.952x |
| `np.round(a) (float32)` | float32 | 1,000 | 0.0019 ms | not_measured | managed | 0.526x |
| `np.round(a) (float32)` | float32 | 100,000 | 0.0267 ms | not_measured | managed | 1.172x |
| `np.round(a) (float32)` | float32 | 10,000,000 | 7.0128 ms | not_measured | managed | 1.460x |
| `np.round(a) (float64)` | float64 | 1,000 | 0.0029 ms | not_measured | managed | 0.379x |
| `np.round(a) (float64)` | float64 | 100,000 | 0.0536 ms | not_measured | managed | 0.750x |
| `np.round(a) (float64)` | float64 | 10,000,000 | 14.2465 ms | not_measured | managed | 1.126x |
| `np.sign (float16)` | float16 | 1,000 | 0.0078 ms | not_measured | managed | 0.179x |
| `np.sign (float16)` | float16 | 100,000 | 1.0024 ms | not_measured | managed | 0.086x |
| `np.sign (float16)` | float16 | 10,000,000 | 103.0400 ms | not_measured | managed | 0.192x |
| `np.sign (float32)` | float32 | 1,000 | 0.0029 ms | not_measured | managed | 0.379x |
| `np.sign (float32)` | float32 | 100,000 | 0.5929 ms | not_measured | managed | 0.558x |
| `np.sign (float32)` | float32 | 10,000,000 | 63.0946 ms | not_measured | managed | 0.593x |
| `np.sign (float64)` | float64 | 1,000 | 0.0032 ms | not_measured | managed | 0.344x |
| `np.sign (float64)` | float64 | 100,000 | 0.6162 ms | not_measured | managed | 0.775x |
| `np.sign (float64)` | float64 | 10,000,000 | 67.3471 ms | not_measured | managed | 0.751x |
| `np.sin (float16)` | float16 | 1,000 | 0.0168 ms | not_measured | managed | 0.583x |
| `np.sin (float16)` | float16 | 100,000 | 1.9105 ms | not_measured | managed | 0.515x |
| `np.sin (float16)` | float16 | 10,000,000 | 198.9163 ms | not_measured | managed | 0.891x |
| `np.sin (float32)` | float32 | 1,000 | 0.0035 ms | not_measured | managed | 0.400x |
| `np.sin (float32)` | float32 | 100,000 | 0.2741 ms | not_measured | managed | 0.279x |
| `np.sin (float32)` | float32 | 10,000,000 | 27.3915 ms | not_measured | managed | 0.460x |
| `np.sin (float64)` | float64 | 1,000 | 0.0117 ms | not_measured | managed | 0.436x |
| `np.sin (float64)` | float64 | 100,000 | 1.2290 ms | not_measured | managed | 1.220x |
| `np.sin (float64)` | float64 | 10,000,000 | 137.2392 ms | not_measured | managed | 0.653x |
| `np.sinh(a) (float16)` | float16 | 1,000 | 0.0225 ms | not_measured | managed | 0.378x |
| `np.sinh(a) (float16)` | float16 | 100,000 | 1.1799 ms | not_measured | managed | 0.868x |
| `np.sinh(a) (float16)` | float16 | 10,000,000 | 119.8326 ms | not_measured | managed | 1.493x |
| `np.sinh(a) (float32)` | float32 | 1,000 | 0.0119 ms | not_measured | managed | 0.345x |
| `np.sinh(a) (float32)` | float32 | 100,000 | 1.2521 ms | not_measured | managed | 0.568x |
| `np.sinh(a) (float32)` | float32 | 10,000,000 | 65.7070 ms | not_measured | managed | 1.346x |
| `np.sinh(a) (float64)` | float64 | 1,000 | 0.0131 ms | not_measured | managed | 0.359x |
| `np.sinh(a) (float64)` | float64 | 100,000 | 0.6481 ms | not_measured | managed | 2.163x |
| `np.sinh(a) (float64)` | float64 | 10,000,000 | 70.3393 ms | not_measured | managed | 1.025x |
| `np.sqrt (float16)` | float16 | 1,000 | 0.0073 ms | not_measured | managed | 0.644x |
| `np.sqrt (float16)` | float16 | 100,000 | 0.7132 ms | not_measured | managed | 0.596x |
| `np.sqrt (float16)` | float16 | 10,000,000 | 66.3762 ms | not_measured | managed | 0.805x |
| `np.sqrt (float32)` | float32 | 1,000 | 0.0024 ms | not_measured | managed | 0.292x |
| `np.sqrt (float32)` | float32 | 100,000 | 0.0784 ms | not_measured | managed | 0.193x |
| `np.sqrt (float32)` | float32 | 10,000,000 | 8.4158 ms | not_measured | managed | 1.272x |
| `np.sqrt (float64)` | float64 | 1,000 | 0.0055 ms | not_measured | managed | 0.236x |
| `np.sqrt (float64)` | float64 | 100,000 | 0.2977 ms | not_measured | managed | 1.109x |
| `np.sqrt (float64)` | float64 | 10,000,000 | 35.6290 ms | not_measured | managed | 0.580x |
| `np.square(a) (float16)` | float16 | 1,000 | 0.0093 ms | not_measured | managed | 0.269x |
| `np.square(a) (float16)` | float16 | 100,000 | 0.8590 ms | not_measured | managed | 0.540x |
| `np.square(a) (float16)` | float16 | 10,000,000 | 90.4239 ms | not_measured | managed | 0.257x |
| `np.square(a) (float32)` | float32 | 1,000 | 0.0022 ms | not_measured | managed | 0.227x |
| `np.square(a) (float32)` | float32 | 100,000 | 0.0607 ms | not_measured | managed | 0.231x |
| `np.square(a) (float32)` | float32 | 10,000,000 | 8.9048 ms | not_measured | managed | 0.815x |
| `np.square(a) (float64)` | float64 | 1,000 | 0.0030 ms | not_measured | managed | 0.167x |
| `np.square(a) (float64)` | float64 | 100,000 | 0.1038 ms | not_measured | managed | 0.235x |
| `np.square(a) (float64)` | float64 | 10,000,000 | 21.3191 ms | not_measured | managed | 0.702x |
| `np.tan (float16)` | float16 | 1,000 | 0.0172 ms | not_measured | managed | 0.430x |
| `np.tan (float16)` | float16 | 100,000 | 1.9277 ms | not_measured | managed | 0.506x |
| `np.tan (float16)` | float16 | 10,000,000 | 198.6763 ms | not_measured | managed | 0.871x |
| `np.tan (float32)` | float32 | 1,000 | 0.0096 ms | not_measured | managed | 0.375x |
| `np.tan (float32)` | float32 | 100,000 | 1.1376 ms | not_measured | managed | 0.594x |
| `np.tan (float32)` | float32 | 10,000,000 | 118.2217 ms | not_measured | managed | 0.659x |
| `np.tan (float64)` | float64 | 1,000 | 0.0127 ms | not_measured | managed | 0.394x |
| `np.tan (float64)` | float64 | 100,000 | 1.5464 ms | not_measured | managed | 1.304x |
| `np.tan (float64)` | float64 | 10,000,000 | 162.3607 ms | not_measured | managed | 1.044x |
| `np.tanh(a) (float16)` | float16 | 1,000 | 0.0217 ms | not_measured | managed | 0.470x |
| `np.tanh(a) (float16)` | float16 | 100,000 | 1.8587 ms | not_measured | managed | 0.571x |
| `np.tanh(a) (float16)` | float16 | 10,000,000 | 109.0487 ms | not_measured | managed | 1.751x |
| `np.tanh(a) (float32)` | float32 | 1,000 | 0.0027 ms | not_measured | managed | 0.667x |
| `np.tanh(a) (float32)` | float32 | 100,000 | 0.3827 ms | not_measured | managed | 0.352x |
| `np.tanh(a) (float32)` | float32 | 10,000,000 | 8.2901 ms | not_measured | managed | 2.278x |
| `np.tanh(a) (float64)` | float64 | 1,000 | 0.0091 ms | not_measured | managed | 0.769x |
| `np.tanh(a) (float64)` | float64 | 100,000 | 0.3083 ms | not_measured | managed | 9.197x |
| `np.tanh(a) (float64)` | float64 | 10,000,000 | 35.6937 ms | not_measured | managed | 2.129x |
| `np.trunc(a) (float16)` | float16 | 1,000 | 0.0072 ms | not_measured | managed | 0.694x |
| `np.trunc(a) (float16)` | float16 | 100,000 | 0.6354 ms | not_measured | managed | 0.785x |
| `np.trunc(a) (float16)` | float16 | 10,000,000 | 62.0921 ms | not_measured | managed | 0.712x |
| `np.trunc(a) (float32)` | float32 | 1,000 | 0.0023 ms | not_measured | managed | 0.217x |
| `np.trunc(a) (float32)` | float32 | 100,000 | 0.0586 ms | not_measured | managed | 0.239x |
| `np.trunc(a) (float32)` | float32 | 10,000,000 | 8.9763 ms | not_measured | managed | 0.840x |
| `np.trunc(a) (float64)` | float64 | 1,000 | 0.0026 ms | not_measured | managed | 0.192x |
| `np.trunc(a) (float64)` | float64 | 100,000 | 0.1011 ms | not_measured | managed | 0.267x |
| `np.trunc(a) (float64)` | float64 | 10,000,000 | 19.3973 ms | not_measured | managed | 0.811x |


---

## NDIter iterator benchmark

```
NumSharp NDIter — canonical benchmark · 2026-08-22 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.32× geomean · 76%🕐 of NumPy's time · 80 win / 50 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ████████████▋ ......   1.27×    78%🕐  ( 16 win / 10 lose)
1K         █████████████▊ .....   1.38×    73%🕐  ( 16 win / 10 lose)
100K       ████████████▋ ......   1.27×    79%🕐  ( 14 win / 12 lose)
1M         ██████████████ .....   1.41×    71%🕐  ( 18 win /  8 lose)
10M        ████████████▌ ......   1.25×    80%🕐  ( 16 win / 10 lose)
ALL        █████████████▏ .....   1.32×    76%🕐  ( 80 win / 50 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████████ ....   1.51×    66%🕐  ( 34 win /  6 lose)
reductions ███████████████████▶   2.42×    41%🕐  ( 34 win /  6 lose)
selection  (no data)
copy/cast  ██████ .............   0.61×   164%🕐  (  5 win / 20 lose)  ◄ SLOWER
index-math ██████▎ ............   0.64×   157%🕐  (  4 win /  6 lose)  ◄ SLOWER
dtypes     ██████████▌ ........   1.06×    94%🕐  (  3 win / 12 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.80×    2.37×    1.24×    1.38×    1.07×
reductions      3.85×    3.29×    2.09×    1.84×    1.70×
selection           -        -        -        -        -
copy/cast       0.42×    0.36×    0.57×    0.94×    1.04×
index-math      0.17×    0.48×    0.99×    1.14×    1.12×
dtypes          0.65×    0.62×    1.58×    1.70×    1.25×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.64×    2.86×    1.14×    1.73×    1.05×     1.58×
  sqrt         1.62×    4.88×    1.04×    1.03×    1.01×     1.53×
  copy         1.58×    2.16×    1.39×    1.68×    1.70×     1.68×
  strided      1.49×    1.28×    1.10×    0.92×    0.98×     1.13×
  bcast        1.51×    2.36×    1.13×    1.64×    0.96×     1.45×
  reversed     1.53×    1.29×    0.84×    1.37×    0.98×     1.17×
  castbuf      1.93×    3.13×    1.83×    1.67×    0.98×     1.79×
  mixbuf       3.91×    2.75×    1.77×    1.27×    1.04×     1.91×
-- reductions
  sum          1.67×    2.33×    2.98×    2.04×    1.69×     2.09×
  sum ax0      1.66×    1.26×    0.98×    0.99×    0.87×     1.12×
  sum ax1      4.51×    1.20×    1.33×    1.46×    1.51×     1.74×
  sum dt=      2.04×    2.46×    1.12×    1.05×    1.02×     1.43×
  amin         2.24×    1.88×    0.80×    0.73×    0.75×     1.13×
  cumsum       1.56×    1.46×    1.23×    1.66×    1.43×     1.46×
  any(F)      20.94×   24.01×    3.48×    1.66×    1.35×     5.23×
  any(hit)    25.91×   24.34×   23.92×   20.94×   21.59×    23.27×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.40×    0.16×    0.26×    2.22×    1.05×     0.52×
  astype       0.26×    0.21×    0.94×    0.94×    1.37×     0.58×
  ravel.T      0.28×    0.26×    0.69×    1.08×    0.99×     0.56×
  in-place     0.91×    0.90×    0.94×    0.57×    1.01×     0.85×
  less->b      0.49×    0.74×    0.39×    0.58×    0.86×     0.59×
-- index-math
  unravel      0.26×    0.38×    1.00×    1.02×    1.08×     0.64×
  ravel_mi     0.11×    0.60×    0.98×    1.28×    1.16×     0.63×
-- dtypes
  complex      0.66×    0.54×    0.78×    0.89×    0.92×     0.75×
  float16      0.67×    0.53×    0.60×    0.62×    0.63×     0.61×
  int8         0.62×    0.81×    8.34×    8.89×    3.33×     2.62×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          █████████▌ .........   0.96×   104%🕐  (  0 win /  1 lose)  ◄ SLOWER
3op_exl      ███████████████████▶   2.60×    38%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   2.99×    33%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   7.22×    14%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   4.15×    24%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   4.04×    25%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.79×    36%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.39×    42%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.26×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   3.00×    33%🕐  (  8 win /  1 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████▍ ...........   0.74×   135%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         █████████▊ .........   0.98×   102%🕐  (  0 win /  1 lose)  ◄ PARITY
w=64         ██████████▊ ........   1.09×    92%🕐  (  1 win /  0 lose)
w=256        █████████████▌ .....   1.35×    74%🕐  (  1 win /  0 lose)
w=1024       █████████████████▍     1.74×    58%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    510.72×   (510.7× faster, faster)
  allocate          1.06×   (1.1× faster, faster)
  overlap_copy      1.89×   (1.9× faster, faster)
  forder_out        1.32×   (1.3× faster, faster)
  zerodim           1.47×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           11.46×    6.17×    2.68×    4.60×    1.81×   vs chained 6× add
reuse            8.44×   10.45×    2.01×    1.27×    1.05×   vs rebuild each call
par8                 -    1.58×    7.20×    6.16×    6.77×   vs single-thread

biggest NumSharp wins: anyeh@1 25.91× · anyeh@1K 24.34× · anyff@1K 24.01× · anyeh@100K 23.92× · anyeh@10M 21.59×
most behind:           ravelmi@1 0.11× · flatten@1K 0.16× · astype@1K 0.21× · astype@1 0.26× · unravel@1 0.26×
```


---

## Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.11 ✅ | 1.19 ✅ | 1.02 ✅ | 0.54 🟡 | 0.78 🟡 | 0.70 🟡 | 0.75 🟡 | 0.73 🟡 |
| 1M | 0.45 🟠 | 0.45 🟠 | 0.44 🟠 | 0.29 🟠 | 0.40 🟠 | 0.34 🟠 | 0.35 🟠 | 0.33 🟠 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.08 ✅ | 1.16 ✅ | 0.97 🟡 | 0.13 🔴 | 1.11 ✅ | 1.46 ✅ | 0.65 🟡 |
| 1M | 0.40 🟠 | 0.50 🟡 | 0.58 🟡 | 0.04 🔴 | 0.61 🟡 | 0.54 🟡 | 0.40 🟠 |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.89 🟡 | 0.67 🟡 | 0.67 🟡 | 1.23 ✅ |
| 1M | 0.46 🟠 | 0.29 🟠 | 0.29 🟠 | 0.58 🟡 |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.4497 | 0.0696 | 0.01 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.7505 | 0.1179 | 0.02 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.8700 | 0.1202 | 0.02 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.9562 | 0.1221 | 0.02 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.8166 | 0.1307 | 0.02 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.7814 | 0.1306 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7930 | 0.0222 | 0.03 🔴 |
| 1M\|dec\|bcast\|max\|ax0 | 3.3884 | 0.0950 | 0.03 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.2607 | 0.1195 | 0.03 🔴 |
| 1M\|dec\|bcast\|min\|ax0 | 3.4004 | 0.0992 | 0.03 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 0.9662 | 0.0298 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.8056 | 0.0253 | 0.03 🔴 |
| 1M\|i32\|bcast\|min\|ax1 | 0.8975 | 0.0301 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7858 | 0.0264 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7817 | 0.0269 | 0.03 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.32 ✅ | 1.25 ✅ | 1.29 ✅ | 0.77 🟡 | 1.45 ✅ | 1.46 ✅ | 2.02 ✅ | 1.43 ✅ |
| 1M | 2.87 ✅ | 2.89 ✅ | 2.91 ✅ | 1.90 ✅ | 2.51 ✅ | 2.49 ✅ | 3.13 ✅ | 2.57 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.92 🟡 | 1.16 ✅ | 1.52 ✅ | 1.88 ✅ | 0.98 🟡 | 1.07 ✅ | 1.64 ✅ | 0.95 🟡 | 2.56 ✅ | 2.42 ✅ | 0.94 🟡 | 0.60 🟡 | 2.40 ✅ |
| 1M | 4.31 ✅ | 4.24 ✅ | 2.18 ✅ | 2.12 ✅ | 2.09 ✅ | 2.06 ✅ | 2.50 ✅ | 2.56 ✅ | 2.05 ✅ | 2.08 ✅ | 2.12 ✅ | 2.46 ✅ | 5.62 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|pos | 0.0442 | 0.0112 | 0.25 🟠 |
| 100K\|f64\|T\|pos | 0.0654 | 0.0277 | 0.42 🟠 |
| 100K\|u64\|strided\|pos | 0.0255 | 0.0111 | 0.43 🟠 |
| 100K\|i64\|strided\|pos | 0.0251 | 0.0110 | 0.44 🟠 |
| 100K\|f64\|C\|pos | 0.0576 | 0.0260 | 0.45 🟠 |
| 100K\|f32\|strided\|pos | 0.0208 | 0.0103 | 0.50 🟠 |
| 100K\|i32\|strided\|pos | 0.0214 | 0.0109 | 0.51 🟡 |
| 100K\|f64\|F\|pos | 0.0543 | 0.0278 | 0.51 🟡 |
| 100K\|u8\|sliced\|pos | 0.0419 | 0.0227 | 0.54 🟡 |
| 100K\|u8\|negrow\|pos | 0.0421 | 0.0229 | 0.54 🟡 |
| 100K\|u32\|strided\|pos | 0.0196 | 0.0110 | 0.56 🟡 |
| 100K\|u64\|F\|pos | 0.0436 | 0.0297 | 0.68 🟡 |
| 100K\|u64\|C\|pos | 0.0391 | 0.0289 | 0.74 🟡 |
| 100K\|i8\|strided\|pos | 0.0165 | 0.0125 | 0.76 🟡 |
| 100K\|f64\|sliced\|pos | 0.0630 | 0.0489 | 0.78 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.61 🟡 | 0.62 🟡 | 0.63 🟡 | 0.49 🟠 | 0.80 🟡 | 0.78 🟡 | 1.10 ✅ | 0.75 🟡 |
| 1M | 1.51 ✅ | 1.47 ✅ | 1.45 ✅ | 1.10 ✅ | 1.57 ✅ | 1.54 ✅ | 1.64 ✅ | 1.66 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.61 🟡 | 0.50 🟠 | 1.03 ✅ | 0.73 🟡 | 0.78 🟡 | 0.68 🟡 |
| 1M | 1.75 ✅ | 1.43 ✅ | 1.58 ✅ | 0.92 🟡 | 1.58 ✅ | 1.84 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.93 🟡 | 0.88 🟡 | 0.66 🟡 | 0.66 🟡 | 0.87 🟡 | 0.57 🟡 | 0.48 🟠 |
| 1M | 1.79 ✅ | 1.71 ✅ | 1.94 ✅ | 1.61 ✅ | 1.45 ✅ | 0.67 🟡 | 1.71 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|abs | 0.0608 | 0.0077 | 0.13 🔴 |
| 100K\|f64\|strided\|neg | 0.0609 | 0.0091 | 0.15 🔴 |
| 100K\|f32\|C\|abs | 0.0380 | 0.0060 | 0.16 🔴 |
| 100K\|i32\|bcast\|copy | 0.0242 | 0.0050 | 0.21 🟠 |
| 100K\|f32\|C\|copy | 0.0283 | 0.0059 | 0.21 🟠 |
| 100K\|f64\|strided\|copy | 0.0502 | 0.0105 | 0.21 🟠 |
| 100K\|f32\|C\|mul | 0.0312 | 0.0068 | 0.22 🟠 |
| 100K\|f16\|C\|copy | 0.0135 | 0.0031 | 0.23 🟠 |
| 100K\|i64\|strided\|neg | 0.0328 | 0.0076 | 0.23 🟠 |
| 100K\|f16\|negrow\|copy | 0.0168 | 0.0039 | 0.23 🟠 |
| 100K\|f64\|strided\|mul | 0.0601 | 0.0141 | 0.23 🟠 |
| 100K\|f64\|strided\|add | 0.0616 | 0.0146 | 0.24 🟠 |
| 100K\|f32\|F\|mul | 0.0279 | 0.0068 | 0.24 🟠 |
| 100K\|f32\|bcast\|copy | 0.0201 | 0.0050 | 0.25 🟠 |
| 100K\|f64\|C\|less | 0.0290 | 0.0073 | 0.25 🟠 |


---

## Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 1.70 ✅ | 2.04 ✅ | 0.57 🟡 | 1.99 ✅ | 2.17 ✅ | 1.82 ✅ | 1.58 ✅ |
| 1-D strided a[::2] | 1.43 ✅ | 1.45 ✅ | 0.48 🟠 | 1.35 ✅ | 1.73 ✅ | 1.69 ✅ | 1.26 ✅ |
| 1-D reversed a[::-1] | 1.82 ✅ | 2.06 ✅ | 0.52 🟡 | 1.92 ✅ | 2.30 ✅ | 1.74 ✅ | 1.57 ✅ |
| array + scalar | 2.37 ✅ | 2.14 ✅ | 0.50 🟡 | 1.92 ✅ | 2.41 ✅ | 1.92 ✅ | 1.68 ✅ |
| scalar + array | 2.39 ✅ | 2.18 ✅ | 0.61 🟡 | 1.81 ✅ | 2.41 ✅ | 2.21 ✅ | 1.77 ✅ |
| mixed C + F | 1.19 ✅ | 2.10 ✅ | 0.55 🟡 | 1.60 ✅ | 1.58 ✅ | 1.69 ✅ | 1.34 ✅ |
| mixed C + T | 1.97 ✅ | 2.01 ✅ | 0.55 🟡 | 1.86 ✅ | 1.94 ✅ | 2.55 ✅ | 1.65 ✅ |
| binary broadcast +row(1,C) | 2.45 ✅ | 1.98 ✅ | 0.56 🟡 | 1.99 ✅ | 2.39 ✅ | 2.15 ✅ | 1.74 ✅ |
| binary broadcast +col(R,1) | 2.84 ✅ | 1.90 ✅ | 0.51 🟡 | 1.99 ✅ | 2.23 ✅ | 2.79 ✅ | 1.80 ✅ |
| col-broadcast unary (inner stride-0) | 2.52 ✅ | 1.56 ✅ | 0.84 🟡 | 1.47 ✅ | 2.27 ✅ | 4.77 ✅ | 1.93 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_strided|f16 | 2.8582 | 1.3829 | 0.48 🟠 |
| scalar_rhs|f16 | 5.9307 | 2.9947 | 0.50 🟡 |
| bcast_col|f16 | 5.8657 | 3.0015 | 0.51 🟡 |
| 1d_rev|f16 | 5.7572 | 3.0003 | 0.52 🟡 |
| mix_C_T|f16 | 5.9972 | 3.3089 | 0.55 🟡 |
| mix_C_F|f16 | 6.0080 | 3.3226 | 0.55 🟡 |
| bcast_row|f16 | 5.3649 | 3.0041 | 0.56 🟡 |
| 1d_C|f16 | 5.3273 | 3.0186 | 0.57 🟡 |
| scalar_lhs|f16 | 4.8620 | 2.9859 | 0.61 🟡 |
| colbcast_unary|f16 | 0.5087 | 0.4285 | 0.84 🟡 |
| mix_C_F|f64 | 1.6559 | 1.9770 | 1.19 ✅ |
| 1d_strided|i32 | 0.2703 | 0.3644 | 1.35 ✅ |

_60 comparable cells._


---

## Cast matrix — astype src→dst × layout × dtype

Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.
✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).

## Summary

- **380 / 1568** comparable cells lag (<1.0); **1188** win (≥1.0).
- **🔴 <0.2** — 22 cells. Top: 10× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 6× same-type diagonal (copy); 4× * → bool
- **🟠 0.2–0.5** — 100 cells. Top: 45× int → sub-word (narrow); 13× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 11× * → bool
- **🟡 0.5–1.0** — 258 cells. Top: 63× int → sub-word (narrow); 28× float/cplx → narrow-int (bool/u8/i8/i16/u16/char); 10× same-type diagonal (copy)

**float/complex → narrow-int geomean by src** (the historical cliff): `f32`→narrow **1.30**, `f64`→narrow **1.02**, `f16`→narrow **2.69**, `c128`→narrow **0.78**.

## Geomean by layout (all src×dst, excl. Decimal)

| C | F | T | sliced | negrow | negcol | strided | bcast |
|---|---|---|---|---|---|---|---|
| 1.53 ✅ | 1.82 ✅ | 1.75 ✅ | 1.72 ✅ | 1.76 ✅ | 1.61 ✅ | 1.32 ✅ | 0.45 🟠 |

## Geomean by src dtype (all layouts×dst)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.96 🟡 | 1.50 ✅ | 1.53 ✅ | 1.73 ✅ | 1.79 ✅ | 1.49 ✅ | 1.49 ✅ | 1.11 ✅ | 1.30 ✅ | 1.62 ✅ | 1.91 ✅ | 1.34 ✅ | 1.15 ✅ | nan ? | 0.97 🟡 |

## Geomean by dst dtype (all layouts×src)

| bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.31 ✅ | 1.12 ✅ | 1.13 ✅ | 1.20 ✅ | 1.16 ✅ | 1.46 ✅ | 1.35 ✅ | 1.61 ✅ | 1.51 ✅ | 1.20 ✅ | 1.67 ✅ | 1.38 ✅ | 1.74 ✅ | nan ? | 1.92 ✅ |

## Layout: C  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.18🔴 | 0.48🟠 | 0.50🟡 | 0.79🟡 | 0.77🟡 | 1.31✅ | 1.43✅ | 1.74✅ | 1.80✅ | 0.84🟡 | 0.24🟠 | 1.70✅ | 2.47✅ | — | 2.49✅ |
| **u8** | 2.25✅ | 0.19🔴 | 2.41✅ | 1.63✅ | 1.71✅ | 2.06✅ | 2.03✅ | 2.46✅ | 2.34✅ | 1.57✅ | 3.01✅ | 1.45✅ | 2.54✅ | — | 2.70✅ |
| **i8** | 2.00✅ | 2.34✅ | 0.19🔴 | 1.79✅ | 1.68✅ | 2.04✅ | 2.08✅ | 2.34✅ | 2.45✅ | 1.67✅ | 3.68✅ | 1.66✅ | 2.38✅ | — | 2.62✅ |
| **i16** | 3.20✅ | 3.41✅ | 2.39✅ | 1.23✅ | 1.90✅ | 1.87✅ | 1.83✅ | 2.08✅ | 2.32✅ | 1.51✅ | 3.60✅ | 1.93✅ | 2.15✅ | — | 2.48✅ |
| **u16** | 2.33✅ | 2.78✅ | 3.07✅ | 1.96✅ | 1.26✅ | 1.96✅ | 1.90✅ | 2.39✅ | 2.63✅ | 1.01✅ | 3.81✅ | 2.00✅ | 2.33✅ | — | 2.76✅ |
| **i32** | 1.74✅ | 1.10✅ | 1.12✅ | 1.37✅ | 0.82🟡 | 1.63✅ | 2.12✅ | 1.43✅ | 1.81✅ | 1.36✅ | 3.37✅ | 1.53✅ | 2.20✅ | — | 1.74✅ |
| **u32** | 1.76✅ | 1.02✅ | 1.03✅ | 1.47✅ | 1.52✅ | 2.21✅ | 1.66✅ | 2.31✅ | 2.33✅ | 1.46✅ | 3.76✅ | 1.56✅ | 2.40✅ | — | 3.06✅ |
| **i64** | 0.17🔴 | 0.16🔴 | 0.18🔴 | 0.35🟠 | 0.43🟠 | 0.43🟠 | 0.43🟠 | 0.62🟡 | 0.97🟡 | 0.66🟡 | 1.75✅ | 1.74✅ | 2.30✅ | — | 2.39✅ |
| **u64** | 1.12✅ | 0.92🟡 | 0.90🟡 | 1.21✅ | 1.22✅ | 1.74✅ | 1.65✅ | 2.57✅ | 2.07✅ | 1.39✅ | 1.60✅ | 1.16✅ | 1.85✅ | — | 2.29✅ |
| **char** | 2.15✅ | 1.98✅ | 1.93✅ | 1.43✅ | 1.00✅ | 1.63✅ | 1.66✅ | 2.16✅ | 2.42✅ | 1.22✅ | 3.63✅ | 1.42✅ | 2.31✅ | — | 2.46✅ |
| **f16** | 4.64✅ | 4.34✅ | 4.34✅ | 3.75✅ | 3.42✅ | 3.14✅ | 1.95✅ | 3.18✅ | 0.98🟡 | 4.08✅ | 1.28✅ | 2.75✅ | 0.90🟡 | — | 1.51✅ |
| **f32** | 3.13✅ | 1.40✅ | 1.61✅ | 1.40✅ | 1.45✅ | 1.65✅ | 1.10✅ | 0.70🟡 | 0.64🟡 | 1.04✅ | 2.95✅ | 1.48✅ | 1.48✅ | — | 1.74✅ |
| **f64** | 1.70✅ | 0.86🟡 | 0.93🟡 | 1.42✅ | 1.38✅ | 1.81✅ | 1.50✅ | 0.86🟡 | 0.98🟡 | 1.28✅ | 1.05✅ | 1.77✅ | 2.13✅ | — | 2.22✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.97🟡 | 0.80🟡 | 0.99🟡 | 1.02✅ | 1.10✅ | 1.42✅ | 1.05✅ | 0.96🟡 | 0.91🟡 | 1.37✅ | 1.54✅ | 1.15✅ | 1.24✅ | — | 2.63✅ |

## Layout: F  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 2.41✅ | 0.91🟡 | 1.14✅ | 0.89🟡 | 0.91🟡 | 1.48✅ | 1.39✅ | 2.24✅ | 1.96✅ | 0.85🟡 | 0.24🟠 | 1.85✅ | 2.74✅ | — | 2.71✅ |
| **u8** | 3.47✅ | 3.03✅ | 4.18✅ | 1.57✅ | 1.56✅ | 1.99✅ | 2.10✅ | 2.15✅ | 2.21✅ | 1.71✅ | 3.86✅ | 1.46✅ | 2.81✅ | — | 3.04✅ |
| **i8** | 4.64✅ | 3.34✅ | 2.43✅ | 1.67✅ | 1.57✅ | 2.00✅ | 1.78✅ | 2.30✅ | 2.53✅ | 1.68✅ | 3.55✅ | 1.61✅ | 1.91✅ | — | 2.39✅ |
| **i16** | 2.36✅ | 2.78✅ | 2.89✅ | 1.23✅ | 1.92✅ | 1.74✅ | 1.93✅ | 2.30✅ | 2.38✅ | 1.89✅ | 3.80✅ | 2.19✅ | 2.28✅ | — | 2.51✅ |
| **u16** | 2.45✅ | 3.11✅ | 2.36✅ | 1.97✅ | 1.31✅ | 2.08✅ | 1.97✅ | 2.19✅ | 2.27✅ | 1.41✅ | 3.79✅ | 2.13✅ | 2.37✅ | — | 3.01✅ |
| **i32** | 1.78✅ | 1.13✅ | 1.22✅ | 1.25✅ | 1.49✅ | 1.72✅ | 2.11✅ | 1.91✅ | 2.29✅ | 1.31✅ | 3.50✅ | 1.59✅ | 0.76🟡 | — | 1.98✅ |
| **u32** | 1.73✅ | 1.16✅ | 1.01✅ | 1.53✅ | 1.53✅ | 2.17✅ | 1.78✅ | 2.22✅ | 2.26✅ | 1.67✅ | 3.69✅ | 1.58✅ | 2.41✅ | — | 2.52✅ |
| **i64** | 1.09✅ | 0.88🟡 | 0.92🟡 | 1.25✅ | 1.24✅ | 1.80✅ | 1.83✅ | 2.26✅ | 2.69✅ | 1.45✅ | 1.87✅ | 2.01✅ | 2.38✅ | — | 2.08✅ |
| **u64** | 1.12✅ | 0.89🟡 | 0.96🟡 | 1.26✅ | 1.31✅ | 2.00✅ | 1.84✅ | 2.59✅ | 2.03✅ | 1.37✅ | 2.38✅ | 1.20✅ | 1.86✅ | — | 2.37✅ |
| **char** | 2.09✅ | 2.12✅ | 2.02✅ | 1.92✅ | 1.23✅ | 1.63✅ | 1.52✅ | 2.11✅ | 2.20✅ | 1.23✅ | 3.56✅ | 1.43✅ | 2.48✅ | — | 2.71✅ |
| **f16** | 4.02✅ | 5.77✅ | 6.15✅ | 4.18✅ | 4.06✅ | 3.70✅ | 2.00✅ | 3.01✅ | 1.10✅ | 4.31✅ | 1.25✅ | 2.74✅ | 1.01✅ | — | 1.47✅ |
| **f32** | 3.30✅ | 2.08✅ | 2.30✅ | 1.64✅ | 1.51✅ | 1.85✅ | 1.33✅ | 0.77🟡 | 0.88🟡 | 1.40✅ | 3.14✅ | 1.52✅ | 2.13✅ | — | 2.12✅ |
| **f64** | 1.83✅ | 1.19✅ | 1.30✅ | 1.31✅ | 1.58✅ | 1.70✅ | 1.48✅ | 0.91🟡 | 0.92🟡 | 1.53✅ | 1.59✅ | 1.73✅ | 2.08✅ | — | 2.09✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 1.18✅ | 0.91🟡 | 0.94🟡 | 1.13✅ | 1.26✅ | 1.57✅ | 1.11✅ | 0.97🟡 | 0.96🟡 | 1.14✅ | 1.49✅ | 1.00✅ | 1.36✅ | — | 3.20✅ |

## Layout: T  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.23🟠 | 0.57🟡 | 0.60🟡 | 0.95🟡 | 0.96🟡 | 1.58✅ | 1.42✅ | 2.29✅ | 2.29✅ | 0.95🟡 | 0.26🟠 | 1.89✅ | 2.60✅ | — | 2.59✅ |
| **u8** | 2.28✅ | 0.13🔴 | 2.21✅ | 1.62✅ | 1.73✅ | 1.98✅ | 2.12✅ | 2.62✅ | 2.15✅ | 1.68✅ | 3.77✅ | 1.35✅ | 2.35✅ | — | 2.68✅ |
| **i8** | 2.44✅ | 2.78✅ | 0.24🟠 | 1.78✅ | 1.61✅ | 1.83✅ | 1.80✅ | 2.25✅ | 2.10✅ | 1.54✅ | 3.55✅ | 1.62✅ | 2.34✅ | — | 2.32✅ |
| **i16** | 2.17✅ | 2.49✅ | 2.77✅ | 1.20✅ | 1.98✅ | 1.83✅ | 1.75✅ | 2.32✅ | 2.05✅ | 1.86✅ | 3.71✅ | 1.79✅ | 2.20✅ | — | 2.48✅ |
| **u16** | 3.45✅ | 3.11✅ | 2.76✅ | 1.96✅ | 1.34✅ | 1.89✅ | 2.06✅ | 2.32✅ | 2.40✅ | 1.28✅ | 3.73✅ | 2.06✅ | 2.22✅ | — | 2.59✅ |
| **i32** | 1.95✅ | 1.22✅ | 1.18✅ | 1.56✅ | 1.50✅ | 1.82✅ | 2.27✅ | 2.32✅ | 2.36✅ | 1.72✅ | 3.69✅ | 1.76✅ | 2.44✅ | — | 2.75✅ |
| **u32** | 1.90✅ | 1.14✅ | 1.08✅ | 1.65✅ | 1.70✅ | 2.22✅ | 1.81✅ | 2.26✅ | 2.32✅ | 1.51✅ | 3.84✅ | 1.54✅ | 2.47✅ | — | 2.44✅ |
| **i64** | 1.18✅ | 0.92🟡 | 0.93🟡 | 1.21✅ | 1.37✅ | 1.97✅ | 1.88✅ | 2.13✅ | 2.72✅ | 1.47✅ | 1.93✅ | 1.72✅ | 2.42✅ | — | 1.85✅ |
| **u64** | 1.27✅ | 0.92🟡 | 0.88🟡 | 1.13✅ | 1.15✅ | 1.71✅ | 1.75✅ | 2.61✅ | 2.08✅ | 1.31✅ | 2.40✅ | 1.13✅ | 1.72✅ | — | 2.09✅ |
| **char** | 2.29✅ | 2.84✅ | 2.72✅ | 1.97✅ | 1.25✅ | 1.60✅ | 1.64✅ | 2.47✅ | 2.57✅ | 1.27✅ | 3.63✅ | 1.51✅ | 2.50✅ | — | 2.58✅ |
| **f16** | 5.00✅ | 6.04✅ | 5.95✅ | 4.06✅ | 4.22✅ | 3.67✅ | 2.06✅ | 3.31✅ | 1.13✅ | 4.14✅ | 1.24✅ | 2.74✅ | 0.96🟡 | — | 1.51✅ |
| **f32** | 3.29✅ | 2.13✅ | 2.19✅ | 1.58✅ | 1.43✅ | 1.83✅ | 1.21✅ | 0.81🟡 | 0.88🟡 | 1.53✅ | 3.72✅ | 1.63✅ | 2.09✅ | — | 2.39✅ |
| **f64** | 1.83✅ | 1.22✅ | 1.21✅ | 1.41✅ | 1.41✅ | 1.77✅ | 1.49✅ | 0.90🟡 | 0.92🟡 | 1.34✅ | 1.60✅ | 2.05✅ | 2.28✅ | — | 2.36✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.94🟡 | 0.82🟡 | 0.93🟡 | 0.95🟡 | 1.03✅ | 1.32✅ | 1.22✅ | 0.96🟡 | 0.89🟡 | 1.03✅ | 1.52✅ | 1.28✅ | 1.28✅ | — | 2.55✅ |

## Layout: sliced  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.25🟠 | 0.60🟡 | 0.65🟡 | 0.96🟡 | 0.65🟡 | 1.56✅ | 1.62✅ | 2.31✅ | 2.45✅ | 0.84🟡 | 0.25🟠 | 1.73✅ | 2.72✅ | — | 2.98✅ |
| **u8** | 2.03✅ | 0.19🔴 | 2.98✅ | 1.74✅ | 1.77✅ | 1.81✅ | 1.58✅ | 2.15✅ | 2.44✅ | 1.78✅ | 3.72✅ | 1.36✅ | 2.46✅ | — | 2.90✅ |
| **i8** | 2.48✅ | 3.04✅ | 0.26🟠 | 1.73✅ | 1.23✅ | 1.95✅ | 1.96✅ | 2.07✅ | 1.86✅ | 1.42✅ | 3.78✅ | 1.48✅ | 2.29✅ | — | 2.88✅ |
| **i16** | 2.63✅ | 2.80✅ | 2.65✅ | 1.35✅ | 1.47✅ | 1.82✅ | 2.01✅ | 2.10✅ | 2.07✅ | 1.54✅ | 3.63✅ | 1.96✅ | 2.63✅ | — | 2.23✅ |
| **u16** | 2.35✅ | 3.22✅ | 3.34✅ | 1.57✅ | 1.38✅ | 2.00✅ | 2.04✅ | 2.50✅ | 2.50✅ | 1.40✅ | 3.70✅ | 1.99✅ | 2.29✅ | — | 2.35✅ |
| **i32** | 1.95✅ | 2.08✅ | 2.02✅ | 1.53✅ | 1.71✅ | 2.01✅ | 0.75🟡 | 1.39✅ | 1.40✅ | 1.36✅ | 3.51✅ | 1.43✅ | 1.80✅ | — | 1.94✅ |
| **u32** | 1.97✅ | 1.99✅ | 2.12✅ | 1.57✅ | 1.57✅ | 1.97✅ | 1.85✅ | 2.37✅ | 2.06✅ | 1.45✅ | 3.68✅ | 1.48✅ | 2.25✅ | — | 2.54✅ |
| **i64** | 0.60🟡 | 0.69🟡 | 1.12✅ | 1.29✅ | 1.26✅ | 1.68✅ | 1.78✅ | 2.69✅ | 2.28✅ | 1.43✅ | 1.38✅ | 1.64✅ | 1.77✅ | — | 2.06✅ |
| **u64** | 1.13✅ | 1.06✅ | 1.23✅ | 1.30✅ | 1.32✅ | 1.59✅ | 1.75✅ | 2.12✅ | 2.37✅ | 1.32✅ | 2.35✅ | 0.99🟡 | 1.49✅ | — | 2.36✅ |
| **char** | 2.16✅ | 2.17✅ | 2.04✅ | 1.44✅ | 1.34✅ | 1.61✅ | 1.65✅ | 1.92✅ | 2.21✅ | 1.34✅ | 3.41✅ | 1.50✅ | 2.15✅ | — | 2.57✅ |
| **f16** | 4.93✅ | 4.89✅ | 4.81✅ | 3.80✅ | 3.91✅ | 3.86✅ | 1.96✅ | 2.93✅ | 1.08✅ | 4.01✅ | 1.37✅ | 2.83✅ | 1.04✅ | — | 1.56✅ |
| **f32** | 3.38✅ | 2.21✅ | 2.38✅ | 1.53✅ | 1.48✅ | 1.88✅ | 1.45✅ | 0.90🟡 | 0.89🟡 | 1.60✅ | 3.68✅ | 1.93✅ | 2.50✅ | — | 2.19✅ |
| **f64** | 2.03✅ | 1.28✅ | 1.27✅ | 1.41✅ | 1.40✅ | 0.78🟡 | 1.51✅ | 0.88🟡 | 0.86🟡 | 1.39✅ | 1.59✅ | 1.73✅ | 2.24✅ | — | 2.20✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.80🟡 | 0.90🟡 | 0.79🟡 | 1.22✅ | 1.07✅ | 1.69✅ | 1.01✅ | 0.99🟡 | 0.90🟡 | 1.21✅ | 1.55✅ | 1.82✅ | 1.69✅ | — | 2.16✅ |

## Layout: negrow  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.23🟠 | 0.59🟡 | 0.67🟡 | 0.89🟡 | 0.93🟡 | 1.48✅ | 1.58✅ | 2.50✅ | 2.45✅ | 0.93🟡 | 0.25🟠 | 1.76✅ | 2.64✅ | — | 2.90✅ |
| **u8** | 2.82✅ | 0.22🟠 | 3.21✅ | 1.66✅ | 1.66✅ | 2.05✅ | 1.92✅ | 2.31✅ | 2.09✅ | 1.37✅ | 2.74✅ | 1.37✅ | 2.37✅ | — | 2.76✅ |
| **i8** | 2.96✅ | 3.36✅ | 0.27🟠 | 1.89✅ | 1.59✅ | 1.93✅ | 1.94✅ | 2.17✅ | 2.10✅ | 1.60✅ | 3.58✅ | 1.63✅ | 2.16✅ | — | 2.58✅ |
| **i16** | 2.23✅ | 2.46✅ | 2.47✅ | 0.95🟡 | 0.83🟡 | 1.24✅ | 1.79✅ | 1.90✅ | 1.40✅ | 1.35✅ | 3.15✅ | 1.79✅ | 2.35✅ | — | 2.29✅ |
| **u16** | 3.33✅ | 2.67✅ | 2.33✅ | 1.66✅ | 1.42✅ | 1.81✅ | 1.91✅ | 2.18✅ | 2.18✅ | 1.47✅ | 3.58✅ | 1.91✅ | 2.27✅ | — | 2.34✅ |
| **i32** | 1.95✅ | 2.12✅ | 2.07✅ | 1.53✅ | 1.41✅ | 1.86✅ | 1.88✅ | 2.26✅ | 2.36✅ | 1.61✅ | 3.33✅ | 1.54✅ | 2.41✅ | — | 2.91✅ |
| **u32** | 1.87✅ | 2.08✅ | 2.53✅ | 1.55✅ | 1.57✅ | 2.02✅ | 2.10✅ | 1.98✅ | 2.27✅ | 1.55✅ | 3.61✅ | 1.44✅ | 2.31✅ | — | 2.79✅ |
| **i64** | 1.14✅ | 1.21✅ | 1.22✅ | 1.42✅ | 1.41✅ | 2.23✅ | 1.90✅ | 2.55✅ | 2.10✅ | 1.31✅ | 1.69✅ | 1.43✅ | 1.90✅ | — | 1.74✅ |
| **u64** | 1.14✅ | 1.17✅ | 1.29✅ | 1.38✅ | 1.38✅ | 1.73✅ | 1.67✅ | 2.41✅ | 2.21✅ | 1.43✅ | 2.35✅ | 1.00🟡 | 1.48✅ | — | 2.22✅ |
| **char** | 3.18✅ | 2.92✅ | 2.28✅ | 1.59✅ | 1.38✅ | 1.67✅ | 1.58✅ | 2.21✅ | 2.09✅ | 1.37✅ | 3.19✅ | 1.43✅ | 2.36✅ | — | 2.34✅ |
| **f16** | 4.79✅ | 4.65✅ | 4.66✅ | 3.79✅ | 3.82✅ | 3.87✅ | 2.04✅ | 3.15✅ | 1.10✅ | 3.94✅ | 1.34✅ | 2.72✅ | 0.97🟡 | — | 1.54✅ |
| **f32** | 3.37✅ | 1.81✅ | 2.19✅ | 1.75✅ | 1.46✅ | 1.88✅ | 1.59✅ | 0.94🟡 | 0.91🟡 | 1.73✅ | 3.60✅ | 2.04✅ | 2.75✅ | — | 2.61✅ |
| **f64** | 1.85✅ | 1.23✅ | 1.29✅ | 1.44✅ | 1.46✅ | 1.76✅ | 1.40✅ | 0.85🟡 | 0.93🟡 | 1.44✅ | 1.54✅ | 1.83✅ | 2.40✅ | — | 2.18✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 1.03✅ | 0.91🟡 | 0.91🟡 | 1.27✅ | 1.09✅ | 1.66✅ | 1.23✅ | 0.99🟡 | 0.93🟡 | 1.03✅ | 1.52✅ | 1.72✅ | 1.75✅ | — | 2.12✅ |

## Layout: negcol  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.72🟡 | 0.94🟡 | 0.93🟡 | 1.09✅ | 1.03✅ | 1.59✅ | 1.65✅ | 2.69✅ | 2.75✅ | 1.17✅ | 0.22🟠 | 1.43✅ | 1.76✅ | — | 2.00✅ |
| **u8** | 0.99🟡 | 2.66✅ | 2.23✅ | 1.55✅ | 1.49✅ | 1.43✅ | 1.54✅ | 2.12✅ | 2.40✅ | 1.78✅ | 2.55✅ | 1.43✅ | 2.33✅ | — | 1.55✅ |
| **i8** | 1.14✅ | 2.46✅ | 2.17✅ | 1.58✅ | 1.56✅ | 1.74✅ | 1.67✅ | 2.24✅ | 2.24✅ | 1.58✅ | 2.39✅ | 1.59✅ | 1.91✅ | — | 2.32✅ |
| **i16** | 3.79✅ | 3.09✅ | 2.82✅ | 1.46✅ | 1.70✅ | 1.68✅ | 1.70✅ | 2.19✅ | 2.37✅ | 1.73✅ | 1.98✅ | 1.57✅ | 2.05✅ | — | 2.79✅ |
| **u16** | 4.53✅ | 3.14✅ | 3.55✅ | 1.70✅ | 1.53✅ | 1.73✅ | 1.74✅ | 2.41✅ | 2.23✅ | 1.71✅ | 2.02✅ | 1.44✅ | 2.13✅ | — | 2.06✅ |
| **i32** | 2.04✅ | 1.51✅ | 1.51✅ | 1.56✅ | 1.57✅ | 1.63✅ | 1.54✅ | 2.22✅ | 2.39✅ | 1.65✅ | 2.06✅ | 1.62✅ | 2.16✅ | — | 2.61✅ |
| **u32** | 2.04✅ | 1.54✅ | 1.50✅ | 1.47✅ | 1.45✅ | 1.52✅ | 1.68✅ | 2.43✅ | 2.00✅ | 1.51✅ | 1.76✅ | 1.38✅ | 1.27✅ | — | 1.59✅ |
| **i64** | 1.21✅ | 0.92🟡 | 0.89🟡 | 1.26✅ | 1.18✅ | 1.70✅ | 1.60✅ | 0.78🟡 | 1.51✅ | 1.05✅ | 1.17✅ | 1.37✅ | 1.50✅ | — | 1.90✅ |
| **u64** | 1.26✅ | 0.92🟡 | 1.02✅ | 1.22✅ | 1.31✅ | 1.81✅ | 1.65✅ | 2.18✅ | 2.44✅ | 1.31✅ | 1.58✅ | 1.14✅ | 1.82✅ | — | 2.57✅ |
| **char** | 3.19✅ | 2.91✅ | 2.53✅ | 1.56✅ | 1.52✅ | 1.68✅ | 1.62✅ | 2.31✅ | 2.27✅ | 1.63✅ | 1.98✅ | 1.52✅ | 2.45✅ | — | 2.32✅ |
| **f16** | 4.57✅ | 1.72✅ | 1.72✅ | 1.73✅ | 1.78✅ | 2.12✅ | 1.28✅ | 1.97✅ | 0.89🟡 | 1.77✅ | 1.57✅ | 1.57✅ | 0.97🟡 | — | 1.56✅ |
| **f32** | 2.32✅ | 1.69✅ | 1.68✅ | 1.71✅ | 1.69✅ | 1.97✅ | 1.15✅ | 1.07✅ | 1.02✅ | 1.82✅ | 2.05✅ | 1.49✅ | 1.45✅ | — | 2.49✅ |
| **f64** | 1.35✅ | 1.06✅ | 1.08✅ | 1.36✅ | 1.26✅ | 1.73✅ | 1.07✅ | 0.85🟡 | 0.82🟡 | 1.36✅ | 1.14✅ | 1.47✅ | 2.05✅ | — | 1.99✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.89🟡 | 1.08✅ | 1.22✅ | 0.97🟡 | 1.06✅ | 1.22✅ | 0.74🟡 | 0.87🟡 | 0.89🟡 | 0.88🟡 | 0.95🟡 | 1.08✅ | 1.48✅ | — | 1.89✅ |

## Layout: strided  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.71🟡 | 0.87🟡 | 0.87🟡 | 0.65🟡 | 0.57🟡 | 1.24✅ | 1.28✅ | 1.81✅ | 1.98✅ | 0.66🟡 | 0.23🟠 | 1.50✅ | 2.03✅ | — | 1.82✅ |
| **u8** | 0.56🟡 | 2.62✅ | 1.50✅ | 1.37✅ | 1.59✅ | 1.10✅ | 1.21✅ | 1.74✅ | 1.92✅ | 1.37✅ | 2.16✅ | 0.91🟡 | 1.59✅ | — | 2.21✅ |
| **i8** | 0.94🟡 | 1.93✅ | 1.53✅ | 1.71✅ | 1.38✅ | 1.27✅ | 1.27✅ | 1.76✅ | 1.75✅ | 1.30✅ | 2.30✅ | 1.15✅ | 1.75✅ | — | 2.06✅ |
| **i16** | 2.16✅ | 1.67✅ | 1.66✅ | 1.67✅ | 1.63✅ | 1.30✅ | 1.29✅ | 2.14✅ | 1.83✅ | 2.14✅ | 1.97✅ | 1.34✅ | 1.89✅ | — | 2.09✅ |
| **u16** | 2.22✅ | 1.52✅ | 1.88✅ | 1.52✅ | 1.67✅ | 1.20✅ | 1.48✅ | 2.15✅ | 1.98✅ | 1.95✅ | 1.90✅ | 1.23✅ | 1.90✅ | — | 2.16✅ |
| **i32** | 1.65✅ | 1.32✅ | 1.29✅ | 1.17✅ | 1.16✅ | 1.27✅ | 1.28✅ | 1.75✅ | 1.83✅ | 1.16✅ | 1.83✅ | 1.23✅ | 1.81✅ | — | 2.38✅ |
| **u32** | 1.24✅ | 1.05✅ | 0.97🟡 | 1.03✅ | 0.85🟡 | 0.61🟡 | 1.08✅ | 1.69✅ | 1.49✅ | 0.99🟡 | 1.81✅ | 1.04✅ | 1.56✅ | — | 2.00✅ |
| **i64** | 0.96🟡 | 0.86🟡 | 0.83🟡 | 0.93🟡 | 0.81🟡 | 1.05✅ | 0.93🟡 | 1.48✅ | 1.36✅ | 0.88🟡 | 1.12✅ | 1.12✅ | 1.65✅ | — | 2.47✅ |
| **u64** | 1.08✅ | 0.95🟡 | 0.93🟡 | 0.96🟡 | 0.93🟡 | 1.34✅ | 1.41✅ | 1.86✅ | 1.86✅ | 0.98🟡 | 1.46✅ | 0.98🟡 | 1.53✅ | — | 2.27✅ |
| **char** | 2.42✅ | 1.69✅ | 1.70✅ | 1.27✅ | 1.08✅ | 1.14✅ | 1.20✅ | 1.80✅ | 1.78✅ | 1.13✅ | 1.79✅ | 1.12✅ | 1.74✅ | — | 2.26✅ |
| **f16** | 3.62✅ | 1.50✅ | 1.46✅ | 1.57✅ | 1.54✅ | 1.66✅ | 1.13✅ | 1.90✅ | 0.90🟡 | 1.60✅ | 1.06✅ | 1.30✅ | 1.05✅ | — | 1.53✅ |
| **f32** | 1.62✅ | 1.05✅ | 1.10✅ | 1.35✅ | 1.32✅ | 1.46✅ | 0.88🟡 | 1.00✅ | 0.83🟡 | 1.38✅ | 1.85✅ | 1.35✅ | 1.31✅ | — | 2.26✅ |
| **f64** | 1.18✅ | 0.99🟡 | 1.02✅ | 1.02✅ | 1.03✅ | 1.20✅ | 0.95🟡 | 0.87🟡 | 0.81🟡 | 0.81🟡 | 1.05✅ | 1.18✅ | 1.73✅ | — | 1.91✅ |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.58🟡 | 0.83🟡 | 0.85🟡 | 0.72🟡 | 0.74🟡 | 1.07✅ | 0.94🟡 | 0.96🟡 | 0.84🟡 | 0.80🟡 | 0.88🟡 | 1.26✅ | 1.63✅ | — | 1.65✅ |

## Layout: bcast  (rows=src, cols=dst)

| src\dst | bool | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | dec | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | 0.02🔴 | 0.34🟠 | 0.36🟠 | 0.51🟡 | 0.52🟡 | 0.47🟠 | 0.48🟠 | 0.73🟡 | 0.76🟡 | 0.52🟡 | 0.24🟠 | 0.58🟡 | 0.93🟡 | — | 0.63🟡 |
| **u8** | 0.21🟠 | 0.02🔴 | 0.26🟠 | 0.45🟠 | 0.29🟠 | 0.59🟡 | 0.65🟡 | 0.92🟡 | 0.79🟡 | 0.45🟠 | 0.65🟡 | 0.68🟡 | 0.94🟡 | — | 0.62🟡 |
| **i8** | 0.21🟠 | 0.26🟠 | 0.02🔴 | 0.46🟠 | 0.47🟠 | 0.67🟡 | 0.68🟡 | 0.85🟡 | 0.91🟡 | 0.46🟠 | 0.66🟡 | 0.69🟡 | 0.88🟡 | — | 0.57🟡 |
| **i16** | 0.34🟠 | 0.29🟠 | 0.32🟠 | 0.44🟠 | 0.48🟠 | 0.67🟡 | 0.66🟡 | 0.97🟡 | 1.09✅ | 0.45🟠 | 0.66🟡 | 0.67🟡 | 0.98🟡 | — | 0.65🟡 |
| **u16** | 0.40🟠 | 0.31🟠 | 0.30🟠 | 0.45🟠 | 0.42🟠 | 0.58🟡 | 0.63🟡 | 0.83🟡 | 0.85🟡 | 0.46🟠 | 0.67🟡 | 0.60🟡 | 0.70🟡 | — | 0.57🟡 |
| **i32** | 0.21🟠 | 0.32🟠 | 0.27🟠 | 0.43🟠 | 0.44🟠 | 0.79🟡 | 0.68🟡 | 1.00🟡 | 0.95🟡 | 0.44🟠 | 0.69🟡 | 0.69🟡 | 0.97🟡 | — | 0.65🟡 |
| **u32** | 0.20🔴 | 0.27🟠 | 0.26🟠 | 0.44🟠 | 0.43🟠 | 0.49🟠 | 0.59🟡 | 0.64🟡 | 0.62🟡 | 0.40🟠 | 0.66🟡 | 0.66🟡 | 0.86🟡 | — | 0.49🟠 |
| **i64** | 0.21🟠 | 0.25🟠 | 0.26🟠 | 0.41🟠 | 0.42🟠 | 0.56🟡 | 0.57🟡 | 0.84🟡 | 0.81🟡 | 0.34🟠 | 0.58🟡 | 0.56🟡 | 0.81🟡 | — | 0.63🟡 |
| **u64** | 0.22🟠 | 0.26🟠 | 0.29🟠 | 0.41🟠 | 0.43🟠 | 0.62🟡 | 0.62🟡 | 0.93🟡 | 1.14✅ | 0.42🟠 | 0.59🟡 | 0.78🟡 | 0.90🟡 | — | 0.60🟡 |
| **char** | 0.39🟠 | 0.31🟠 | 0.27🟠 | 0.43🟠 | 0.41🟠 | 0.58🟡 | 0.57🟡 | 0.73🟡 | 0.82🟡 | 0.41🟠 | 0.64🟡 | 0.64🟡 | 0.88🟡 | — | 0.61🟡 |
| **f16** | 0.56🟡 | 0.36🟠 | 0.38🟠 | 0.45🟠 | 0.47🟠 | 0.68🟡 | 0.61🟡 | 0.78🟡 | 0.61🟡 | 0.43🟠 | 0.54🟡 | 0.52🟡 | 0.65🟡 | — | 0.44🟠 |
| **f32** | 0.23🟠 | 0.10🔴 | 0.10🔴 | 0.16🔴 | 0.16🔴 | 0.26🟠 | 0.47🟠 | 0.66🟡 | 0.61🟡 | 0.16🔴 | 0.74🟡 | 0.72🟡 | 0.85🟡 | — | 0.64🟡 |
| **f64** | 0.25🟠 | 0.13🔴 | 0.13🔴 | 0.21🟠 | 0.21🟠 | 0.45🟠 | 0.44🟠 | 0.58🟡 | 0.56🟡 | 0.21🟠 | 0.50🟠 | 0.57🟡 | 0.88🟡 | — | 0.60🟡 |
| **dec** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **c128** | 0.17🔴 | 0.13🔴 | 0.13🔴 | 0.21🟠 | 0.21🟠 | 0.27🟠 | 0.29🟠 | 0.69🟡 | 0.45🟠 | 0.21🟠 | 0.49🟠 | 0.65🟡 | 0.88🟡 | — | 0.68🟡 |

## Lagging cells (<1.0) — the worklist  (380 cells)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| u8\|bcast\|u8 | 0.8325 | 0.0134 | 0.02 🔴 |
| bool\|bcast\|bool | 0.9129 | 0.0149 | 0.02 🔴 |
| i8\|bcast\|i8 | 0.8017 | 0.0137 | 0.02 🔴 |
| f32\|bcast\|i8 | 2.6924 | 0.2680 | 0.10 🔴 |
| f32\|bcast\|u8 | 2.7084 | 0.2772 | 0.10 🔴 |
| c128\|bcast\|i8 | 2.0643 | 0.2586 | 0.13 🔴 |
| u8\|T\|u8 | 0.1157 | 0.0145 | 0.13 🔴 |
| c128\|bcast\|u8 | 2.0430 | 0.2587 | 0.13 🔴 |
| f64\|bcast\|u8 | 1.9841 | 0.2580 | 0.13 🔴 |
| f64\|bcast\|i8 | 1.9367 | 0.2588 | 0.13 🔴 |
| i64\|C\|u8 | 1.3130 | 0.2058 | 0.16 🔴 |
| f32\|bcast\|u16 | 2.8963 | 0.4620 | 0.16 🔴 |
| f32\|bcast\|char | 2.8658 | 0.4578 | 0.16 🔴 |
| f32\|bcast\|i16 | 2.8808 | 0.4709 | 0.16 🔴 |
| i64\|C\|bool | 1.3914 | 0.2307 | 0.17 🔴 |
| c128\|bcast\|bool | 2.3737 | 0.3946 | 0.17 🔴 |
| bool\|C\|bool | 0.0809 | 0.0144 | 0.18 🔴 |
| i64\|C\|i8 | 1.1174 | 0.2045 | 0.18 🔴 |
| i8\|C\|i8 | 0.0766 | 0.0144 | 0.19 🔴 |
| u8\|sliced\|u8 | 0.0924 | 0.0176 | 0.19 🔴 |
| u8\|C\|u8 | 0.0758 | 0.0146 | 0.19 🔴 |
| u32\|bcast\|bool | 1.6144 | 0.3184 | 0.20 🔴 |
| c128\|bcast\|u16 | 2.1687 | 0.4482 | 0.21 🟠 |
| c128\|bcast\|char | 2.1655 | 0.4493 | 0.21 🟠 |
| u8\|bcast\|bool | 1.4596 | 0.3032 | 0.21 🟠 |
| c128\|bcast\|i16 | 2.1437 | 0.4497 | 0.21 🟠 |
| f64\|bcast\|i16 | 2.0888 | 0.4408 | 0.21 🟠 |
| f64\|bcast\|u16 | 2.0398 | 0.4352 | 0.21 🟠 |
| i32\|bcast\|bool | 1.4830 | 0.3170 | 0.21 🟠 |
| f64\|bcast\|char | 2.0781 | 0.4454 | 0.21 🟠 |
| i8\|bcast\|bool | 1.4151 | 0.3036 | 0.21 🟠 |
| i64\|bcast\|bool | 1.5129 | 0.3252 | 0.21 🟠 |
| bool\|negcol\|f16 | 8.6481 | 1.8823 | 0.22 🟠 |
| u64\|bcast\|bool | 1.4723 | 0.3240 | 0.22 🟠 |
| u8\|negrow\|u8 | 0.0800 | 0.0176 | 0.22 🟠 |
| f32\|bcast\|bool | 1.7423 | 0.4004 | 0.23 🟠 |
| bool\|T\|bool | 0.0647 | 0.0150 | 0.23 🟠 |
| bool\|strided\|f16 | 3.4443 | 0.8035 | 0.23 🟠 |
| bool\|negrow\|bool | 0.0756 | 0.0177 | 0.23 🟠 |
| bool\|C\|f16 | 7.0984 | 1.6776 | 0.24 🟠 |
| bool\|bcast\|f16 | 7.0933 | 1.6862 | 0.24 🟠 |
| bool\|F\|f16 | 7.2805 | 1.7486 | 0.24 🟠 |
| i8\|T\|i8 | 0.0617 | 0.0151 | 0.24 🟠 |
| bool\|sliced\|bool | 0.0736 | 0.0181 | 0.25 🟠 |
| bool\|sliced\|f16 | 6.9678 | 1.7203 | 0.25 🟠 |
| i64\|bcast\|u8 | 0.8727 | 0.2158 | 0.25 🟠 |
| bool\|negrow\|f16 | 6.8697 | 1.7276 | 0.25 🟠 |
| f64\|bcast\|bool | 1.6243 | 0.4106 | 0.25 🟠 |
| bool\|T\|f16 | 6.9721 | 1.7815 | 0.26 🟠 |
| u8\|bcast\|i8 | 0.8244 | 0.2111 | 0.26 🟠 |
| u64\|bcast\|u8 | 0.8624 | 0.2210 | 0.26 🟠 |
| u32\|bcast\|i8 | 0.8443 | 0.2166 | 0.26 🟠 |
| i64\|bcast\|i8 | 0.8762 | 0.2266 | 0.26 🟠 |
| i8\|sliced\|i8 | 0.0668 | 0.0174 | 0.26 🟠 |
| f32\|bcast\|i32 | 2.8015 | 0.7409 | 0.26 🟠 |
| i8\|bcast\|u8 | 0.8027 | 0.2124 | 0.26 🟠 |
| i8\|negrow\|i8 | 0.0658 | 0.0175 | 0.27 🟠 |
| i32\|bcast\|i8 | 0.8442 | 0.2242 | 0.27 🟠 |
| u32\|bcast\|u8 | 0.8461 | 0.2263 | 0.27 🟠 |
| char\|bcast\|i8 | 0.8271 | 0.2247 | 0.27 🟠 |
| c128\|bcast\|i32 | 2.5995 | 0.7075 | 0.27 🟠 |
| u8\|bcast\|u16 | 1.4327 | 0.4095 | 0.29 🟠 |
| u64\|bcast\|i8 | 0.8385 | 0.2415 | 0.29 🟠 |
| i16\|bcast\|u8 | 0.8677 | 0.2539 | 0.29 🟠 |
| c128\|bcast\|u32 | 2.3663 | 0.6959 | 0.29 🟠 |
| u16\|bcast\|i8 | 0.8153 | 0.2441 | 0.30 🟠 |
| char\|bcast\|u8 | 0.8270 | 0.2531 | 0.31 🟠 |
| u16\|bcast\|u8 | 0.8265 | 0.2531 | 0.31 🟠 |
| i16\|bcast\|i8 | 0.8088 | 0.2552 | 0.32 🟠 |
| i32\|bcast\|u8 | 0.8301 | 0.2670 | 0.32 🟠 |
| i64\|bcast\|char | 1.2242 | 0.4118 | 0.34 🟠 |
| i16\|bcast\|bool | 1.0122 | 0.3448 | 0.34 🟠 |
| bool\|bcast\|u8 | 0.8950 | 0.3078 | 0.34 🟠 |
| i64\|C\|i16 | 1.1683 | 0.4137 | 0.35 🟠 |
| f16\|bcast\|u8 | 2.6823 | 0.9598 | 0.36 🟠 |
| bool\|bcast\|i8 | 0.8433 | 0.3067 | 0.36 🟠 |
| f16\|bcast\|i8 | 2.4800 | 0.9541 | 0.38 🟠 |
| char\|bcast\|bool | 0.8193 | 0.3184 | 0.39 🟠 |
| u32\|bcast\|char | 1.0396 | 0.4145 | 0.40 🟠 |
| u16\|bcast\|bool | 0.8228 | 0.3321 | 0.40 🟠 |
| u64\|bcast\|i16 | 1.0019 | 0.4103 | 0.41 🟠 |
| char\|bcast\|u16 | 0.9169 | 0.3764 | 0.41 🟠 |
| i64\|bcast\|i16 | 1.0125 | 0.4164 | 0.41 🟠 |
| char\|bcast\|char | 0.9180 | 0.3790 | 0.41 🟠 |
| i64\|bcast\|u16 | 0.9963 | 0.4153 | 0.42 🟠 |
| u16\|bcast\|u16 | 0.9466 | 0.3973 | 0.42 🟠 |
| u64\|bcast\|char | 0.9942 | 0.4208 | 0.42 🟠 |
| u64\|bcast\|u16 | 0.9745 | 0.4153 | 0.43 🟠 |
| f16\|bcast\|char | 2.7655 | 1.1809 | 0.43 🟠 |
| i64\|C\|u16 | 0.9836 | 0.4208 | 0.43 🟠 |
| i64\|C\|i32 | 1.7525 | 0.7539 | 0.43 🟠 |
| u32\|bcast\|u16 | 0.9570 | 0.4118 | 0.43 🟠 |
| i64\|C\|u32 | 1.6712 | 0.7220 | 0.43 🟠 |
| char\|bcast\|i16 | 0.9451 | 0.4088 | 0.43 🟠 |
| i32\|bcast\|i16 | 0.9498 | 0.4111 | 0.43 🟠 |
| u32\|bcast\|i16 | 0.9662 | 0.4229 | 0.44 🟠 |
| f64\|bcast\|u32 | 1.5152 | 0.6647 | 0.44 🟠 |
| i32\|bcast\|char | 0.9404 | 0.4135 | 0.44 🟠 |
| i32\|bcast\|u16 | 0.9387 | 0.4134 | 0.44 🟠 |
| i16\|bcast\|i16 | 0.8907 | 0.3948 | 0.44 🟠 |
| f16\|bcast\|c128 | 6.4658 | 2.8746 | 0.44 🟠 |
| i16\|bcast\|char | 0.9404 | 0.4189 | 0.45 🟠 |
| f64\|bcast\|i32 | 1.4988 | 0.6680 | 0.45 🟠 |
| c128\|bcast\|u64 | 2.8405 | 1.2676 | 0.45 🟠 |
| u16\|bcast\|i16 | 0.9467 | 0.4261 | 0.45 🟠 |
| u8\|bcast\|i16 | 0.9237 | 0.4165 | 0.45 🟠 |
| f16\|bcast\|i16 | 2.7766 | 1.2557 | 0.45 🟠 |
| u8\|bcast\|char | 0.9136 | 0.4135 | 0.45 🟠 |
| u16\|bcast\|char | 0.9117 | 0.4150 | 0.46 🟠 |
| i8\|bcast\|char | 0.8981 | 0.4100 | 0.46 🟠 |
| i8\|bcast\|i16 | 0.8887 | 0.4083 | 0.46 🟠 |
| i8\|bcast\|u16 | 0.8855 | 0.4122 | 0.47 🟠 |
| f16\|bcast\|u16 | 2.6836 | 1.2495 | 0.47 🟠 |
| bool\|bcast\|i32 | 1.4624 | 0.6871 | 0.47 🟠 |
| f32\|bcast\|u32 | 1.5486 | 0.7350 | 0.47 🟠 |
| i16\|bcast\|u16 | 0.8817 | 0.4200 | 0.48 🟠 |
| bool\|bcast\|u32 | 1.4222 | 0.6847 | 0.48 🟠 |
| bool\|C\|u8 | 0.3813 | 0.1845 | 0.48 🟠 |
| u32\|bcast\|c128 | 5.0994 | 2.4890 | 0.49 🟠 |
| u32\|bcast\|i32 | 1.3498 | 0.6601 | 0.49 🟠 |
| c128\|bcast\|f16 | 3.4685 | 1.6987 | 0.49 🟠 |
| f64\|bcast\|f16 | 3.3072 | 1.6430 | 0.50 🟠 |
| bool\|C\|i8 | 0.3808 | 0.1922 | 0.50 🟡 |
| bool\|bcast\|i16 | 0.9468 | 0.4835 | 0.51 🟡 |
| f16\|bcast\|f32 | 2.1272 | 1.0987 | 0.52 🟡 |
| bool\|bcast\|char | 0.9171 | 0.4784 | 0.52 🟡 |
| bool\|bcast\|u16 | 0.9287 | 0.4861 | 0.52 🟡 |
| f16\|bcast\|f16 | 0.9222 | 0.4950 | 0.54 🟡 |
| u8\|strided\|bool | 0.2679 | 0.1494 | 0.56 🟡 |
| f64\|bcast\|u64 | 2.2572 | 1.2649 | 0.56 🟡 |
| f16\|bcast\|bool | 0.8140 | 0.4572 | 0.56 🟡 |
| i64\|bcast\|f32 | 1.2273 | 0.6912 | 0.56 🟡 |
| i64\|bcast\|i32 | 1.1826 | 0.6681 | 0.56 🟡 |
| i64\|bcast\|u32 | 1.1870 | 0.6712 | 0.57 🟡 |
| f64\|bcast\|f32 | 1.2083 | 0.6847 | 0.57 🟡 |
| i8\|bcast\|c128 | 4.1852 | 2.3719 | 0.57 🟡 |
| u16\|bcast\|c128 | 4.3592 | 2.4778 | 0.57 🟡 |
| bool\|T\|u8 | 0.3327 | 0.1895 | 0.57 🟡 |
| bool\|strided\|u16 | 0.2545 | 0.1451 | 0.57 🟡 |
| char\|bcast\|u32 | 1.1329 | 0.6484 | 0.57 🟡 |
| c128\|strided\|bool | 0.5088 | 0.2936 | 0.58 🟡 |
| f64\|bcast\|i64 | 2.1570 | 1.2489 | 0.58 🟡 |
| u16\|bcast\|i32 | 1.1593 | 0.6716 | 0.58 🟡 |
| char\|bcast\|i32 | 1.1662 | 0.6764 | 0.58 🟡 |
| bool\|bcast\|f32 | 1.2697 | 0.7392 | 0.58 🟡 |
| i64\|bcast\|f16 | 2.7444 | 1.6035 | 0.58 🟡 |
| bool\|negrow\|u8 | 0.3401 | 0.1996 | 0.59 🟡 |
| u64\|bcast\|f16 | 3.1061 | 1.8231 | 0.59 🟡 |
| u8\|bcast\|i32 | 1.1330 | 0.6676 | 0.59 🟡 |
| u32\|bcast\|u32 | 1.2460 | 0.7346 | 0.59 🟡 |
| f64\|bcast\|c128 | 4.1892 | 2.5098 | 0.60 🟡 |
| bool\|sliced\|u8 | 0.3390 | 0.2032 | 0.60 🟡 |
| bool\|T\|i8 | 0.3342 | 0.2007 | 0.60 🟡 |
| u64\|bcast\|c128 | 4.0336 | 2.4231 | 0.60 🟡 |
| u16\|bcast\|f32 | 1.1468 | 0.6906 | 0.60 🟡 |
| i64\|sliced\|bool | 0.4120 | 0.2485 | 0.60 🟡 |
| char\|bcast\|c128 | 4.0805 | 2.4764 | 0.61 🟡 |
| u32\|strided\|i32 | 0.5497 | 0.3337 | 0.61 🟡 |
| f32\|bcast\|u64 | 2.2041 | 1.3408 | 0.61 🟡 |
| f16\|bcast\|u64 | 3.3482 | 2.0500 | 0.61 🟡 |
| f16\|bcast\|u32 | 2.6834 | 1.6431 | 0.61 🟡 |
| u64\|bcast\|i32 | 1.0528 | 0.6499 | 0.62 🟡 |
| u64\|bcast\|u32 | 1.0676 | 0.6620 | 0.62 🟡 |
| i64\|C\|i64 | 1.7065 | 1.0613 | 0.62 🟡 |
| u32\|bcast\|u64 | 2.0083 | 1.2495 | 0.62 🟡 |
| u8\|bcast\|c128 | 3.9115 | 2.4391 | 0.62 🟡 |
| i64\|bcast\|c128 | 3.9430 | 2.4686 | 0.63 🟡 |
| u16\|bcast\|u32 | 1.1140 | 0.6980 | 0.63 🟡 |
| bool\|bcast\|c128 | 3.8419 | 2.4212 | 0.63 🟡 |
| f32\|C\|u64 | 2.1592 | 1.3774 | 0.64 🟡 |
| u32\|bcast\|i64 | 1.9724 | 1.2610 | 0.64 🟡 |
| char\|bcast\|f32 | 1.0561 | 0.6768 | 0.64 🟡 |
| f32\|bcast\|c128 | 4.0937 | 2.6250 | 0.64 🟡 |
| char\|bcast\|f16 | 2.4802 | 1.5994 | 0.64 🟡 |
| c128\|bcast\|f32 | 1.0871 | 0.7024 | 0.65 🟡 |
| bool\|strided\|i16 | 0.2201 | 0.1423 | 0.65 🟡 |
| u8\|bcast\|f16 | 2.4679 | 1.6015 | 0.65 🟡 |
| i32\|bcast\|c128 | 3.8674 | 2.5156 | 0.65 🟡 |
| bool\|sliced\|i8 | 0.3402 | 0.2216 | 0.65 🟡 |
| i16\|bcast\|c128 | 3.9945 | 2.6020 | 0.65 🟡 |
| bool\|sliced\|u16 | 0.7190 | 0.4688 | 0.65 🟡 |
| u8\|bcast\|u32 | 1.0177 | 0.6642 | 0.65 🟡 |
| f16\|bcast\|f64 | 2.3934 | 1.5641 | 0.65 🟡 |
| i64\|C\|char | 0.6699 | 0.4392 | 0.66 🟡 |
| bool\|strided\|char | 0.2216 | 0.1457 | 0.66 🟡 |
| f32\|bcast\|i64 | 1.9899 | 1.3106 | 0.66 🟡 |
| i16\|bcast\|u32 | 1.0175 | 0.6707 | 0.66 🟡 |
| i16\|bcast\|f16 | 2.4289 | 1.6017 | 0.66 🟡 |
| i8\|bcast\|f16 | 2.4364 | 1.6071 | 0.66 🟡 |
| u32\|bcast\|f32 | 1.1459 | 0.7567 | 0.66 🟡 |
| u32\|bcast\|f16 | 2.5458 | 1.6836 | 0.66 🟡 |
| i16\|bcast\|f32 | 1.0683 | 0.7128 | 0.67 🟡 |
| bool\|negrow\|i8 | 0.3418 | 0.2289 | 0.67 🟡 |
| i16\|bcast\|i32 | 0.9812 | 0.6588 | 0.67 🟡 |
| u16\|bcast\|f16 | 2.5049 | 1.6841 | 0.67 🟡 |
| i8\|bcast\|i32 | 0.9817 | 0.6613 | 0.67 🟡 |
| i32\|bcast\|u32 | 1.0512 | 0.7157 | 0.68 🟡 |
| u8\|bcast\|f32 | 0.9977 | 0.6793 | 0.68 🟡 |
| i8\|bcast\|u32 | 0.9942 | 0.6780 | 0.68 🟡 |
| f16\|bcast\|i32 | 2.4819 | 1.6982 | 0.68 🟡 |
| c128\|bcast\|c128 | 3.9324 | 2.6921 | 0.68 🟡 |
| i32\|bcast\|f32 | 0.9953 | 0.6821 | 0.69 🟡 |
| i32\|bcast\|f16 | 2.3526 | 1.6148 | 0.69 🟡 |
| i64\|sliced\|u8 | 0.3287 | 0.2259 | 0.69 🟡 |
| i8\|bcast\|f32 | 1.0035 | 0.6954 | 0.69 🟡 |
| c128\|bcast\|i64 | 1.8514 | 1.2838 | 0.69 🟡 |
| f32\|C\|i64 | 1.7877 | 1.2452 | 0.70 🟡 |
| u16\|bcast\|f64 | 1.7742 | 1.2487 | 0.70 🟡 |
| bool\|strided\|bool | 0.1481 | 0.1050 | 0.71 🟡 |
| f32\|bcast\|f32 | 1.0674 | 0.7683 | 0.72 🟡 |
| bool\|negcol\|bool | 0.2759 | 0.1992 | 0.72 🟡 |
| c128\|strided\|i16 | 0.3703 | 0.2682 | 0.72 🟡 |
| bool\|bcast\|i64 | 1.7032 | 1.2366 | 0.73 🟡 |
| char\|bcast\|i64 | 1.6825 | 1.2360 | 0.73 🟡 |
| c128\|negcol\|u32 | 1.3671 | 1.0097 | 0.74 🟡 |
| f32\|bcast\|f16 | 2.1326 | 1.5761 | 0.74 🟡 |
| c128\|strided\|u16 | 0.3611 | 0.2678 | 0.74 🟡 |
| i32\|sliced\|u32 | 0.8600 | 0.6490 | 0.75 🟡 |
| i32\|F\|f64 | 1.7142 | 1.2944 | 0.76 🟡 |
| bool\|bcast\|u64 | 1.6255 | 1.2427 | 0.76 🟡 |
| f32\|F\|i64 | 1.6977 | 1.3057 | 0.77 🟡 |
| bool\|C\|u16 | 0.5522 | 0.4267 | 0.77 🟡 |
| i64\|negcol\|i64 | 1.6748 | 1.2999 | 0.78 🟡 |
| f64\|sliced\|i32 | 0.9758 | 0.7595 | 0.78 🟡 |
| u64\|bcast\|f32 | 0.9911 | 0.7728 | 0.78 🟡 |
| f16\|bcast\|i64 | 3.0449 | 2.3861 | 0.78 🟡 |
| i32\|bcast\|i32 | 0.9768 | 0.7668 | 0.79 🟡 |
| bool\|C\|i16 | 0.5478 | 0.4306 | 0.79 🟡 |
| u8\|bcast\|u64 | 1.5803 | 1.2447 | 0.79 🟡 |
| c128\|sliced\|i8 | 0.4531 | 0.3590 | 0.79 🟡 |
| c128\|sliced\|bool | 0.6790 | 0.5401 | 0.80 🟡 |
| c128\|C\|u8 | 0.4219 | 0.3394 | 0.80 🟡 |
| c128\|strided\|char | 0.3777 | 0.3039 | 0.80 🟡 |
| f64\|strided\|char | 0.1897 | 0.1527 | 0.81 🟡 |
| f64\|strided\|u64 | 0.8066 | 0.6496 | 0.81 🟡 |
| i64\|bcast\|f64 | 1.5563 | 1.2579 | 0.81 🟡 |
| i64\|strided\|u16 | 0.1860 | 0.1508 | 0.81 🟡 |
| i64\|bcast\|u64 | 1.5573 | 1.2665 | 0.81 🟡 |
| f32\|T\|i64 | 1.5283 | 1.2431 | 0.81 🟡 |
| c128\|T\|u8 | 0.4087 | 0.3338 | 0.82 🟡 |
| i32\|C\|u16 | 0.4761 | 0.3891 | 0.82 🟡 |
| char\|bcast\|u64 | 1.5010 | 1.2275 | 0.82 🟡 |
| f64\|negcol\|u64 | 1.5875 | 1.3031 | 0.82 🟡 |
| f32\|strided\|u64 | 0.8409 | 0.6950 | 0.83 🟡 |
| c128\|strided\|u8 | 0.3201 | 0.2651 | 0.83 🟡 |
| i16\|negrow\|u16 | 0.4745 | 0.3930 | 0.83 🟡 |
| i64\|strided\|i8 | 0.1554 | 0.1288 | 0.83 🟡 |
| u16\|bcast\|i64 | 1.5399 | 1.2786 | 0.83 🟡 |
| bool\|sliced\|char | 0.5168 | 0.4330 | 0.84 🟡 |
| c128\|strided\|u64 | 1.0816 | 0.9099 | 0.84 🟡 |
| bool\|C\|char | 0.5091 | 0.4283 | 0.84 🟡 |
| i64\|bcast\|i64 | 1.7433 | 1.4680 | 0.84 🟡 |
| i8\|bcast\|i64 | 1.4444 | 1.2220 | 0.85 🟡 |
| f32\|bcast\|f64 | 1.5455 | 1.3103 | 0.85 🟡 |
| u32\|strided\|u16 | 0.1382 | 0.1172 | 0.85 🟡 |
| f64\|negcol\|i64 | 1.4945 | 1.2703 | 0.85 🟡 |
| c128\|strided\|i8 | 0.3165 | 0.2691 | 0.85 🟡 |
| f64\|negrow\|i64 | 1.5435 | 1.3147 | 0.85 🟡 |
| bool\|F\|char | 0.5106 | 0.4357 | 0.85 🟡 |
| u16\|bcast\|u64 | 1.5320 | 1.3090 | 0.85 🟡 |
| u32\|bcast\|f64 | 1.4435 | 1.2349 | 0.86 🟡 |
| i64\|strided\|u8 | 0.1479 | 0.1267 | 0.86 🟡 |
| f64\|C\|i64 | 1.5091 | 1.2981 | 0.86 🟡 |
| f64\|C\|u8 | 0.3010 | 0.2593 | 0.86 🟡 |
| f64\|sliced\|u64 | 1.4844 | 1.2828 | 0.86 🟡 |
| bool\|strided\|i8 | 0.1743 | 0.1516 | 0.87 🟡 |
| c128\|negcol\|i64 | 1.8684 | 1.6308 | 0.87 🟡 |
| bool\|strided\|u8 | 0.1836 | 0.1603 | 0.87 🟡 |
| f64\|strided\|i64 | 0.7340 | 0.6415 | 0.87 🟡 |
| c128\|negcol\|char | 0.8050 | 0.7056 | 0.88 🟡 |
| f64\|bcast\|f64 | 1.7139 | 1.5030 | 0.88 🟡 |
| i64\|strided\|char | 0.1763 | 0.1549 | 0.88 🟡 |
| f32\|strided\|u32 | 0.4352 | 0.3827 | 0.88 🟡 |
| char\|bcast\|f64 | 1.4198 | 1.2490 | 0.88 🟡 |
| f32\|T\|u64 | 1.4718 | 1.2953 | 0.88 🟡 |
| u64\|T\|i8 | 0.2311 | 0.2034 | 0.88 🟡 |
| i64\|F\|u8 | 0.2296 | 0.2021 | 0.88 🟡 |
| c128\|bcast\|f64 | 1.3946 | 1.2283 | 0.88 🟡 |
| c128\|strided\|f16 | 0.9395 | 0.8275 | 0.88 🟡 |
| f32\|F\|u64 | 1.5378 | 1.3566 | 0.88 🟡 |
| i8\|bcast\|f64 | 1.4162 | 1.2512 | 0.88 🟡 |
| f64\|sliced\|i64 | 1.4791 | 1.3072 | 0.88 🟡 |
| f16\|negcol\|u64 | 2.1230 | 1.8800 | 0.89 🟡 |
| bool\|negrow\|i16 | 0.4900 | 0.4350 | 0.89 🟡 |
| c128\|negcol\|u64 | 1.9848 | 1.7622 | 0.89 🟡 |
| c128\|negcol\|bool | 0.6935 | 0.6171 | 0.89 🟡 |
| c128\|T\|u64 | 1.7548 | 1.5634 | 0.89 🟡 |
| bool\|F\|i16 | 0.4690 | 0.4185 | 0.89 🟡 |
| i64\|negcol\|i8 | 0.2719 | 0.2431 | 0.89 🟡 |
| f32\|sliced\|u64 | 1.5142 | 1.3543 | 0.89 🟡 |
| u64\|F\|u8 | 0.2228 | 0.1993 | 0.89 🟡 |
| c128\|sliced\|u64 | 1.8684 | 1.6739 | 0.90 🟡 |
| u64\|bcast\|f64 | 1.4326 | 1.2861 | 0.90 🟡 |
| f32\|sliced\|i64 | 1.4776 | 1.3276 | 0.90 🟡 |
| c128\|sliced\|u8 | 0.3866 | 0.3478 | 0.90 🟡 |
| u64\|C\|i8 | 0.2177 | 0.1962 | 0.90 🟡 |
| f16\|strided\|u64 | 1.0364 | 0.9343 | 0.90 🟡 |
| f64\|T\|i64 | 1.4836 | 1.3422 | 0.90 🟡 |
| f16\|C\|f64 | 1.6333 | 1.4779 | 0.90 🟡 |
| f64\|F\|i64 | 1.4677 | 1.3293 | 0.91 🟡 |
| c128\|negrow\|i8 | 0.4157 | 0.3766 | 0.91 🟡 |
| c128\|negrow\|u8 | 0.4075 | 0.3695 | 0.91 🟡 |
| c128\|F\|u8 | 0.3440 | 0.3123 | 0.91 🟡 |
| i8\|bcast\|u64 | 1.3591 | 1.2343 | 0.91 🟡 |
| bool\|F\|u16 | 0.4743 | 0.4308 | 0.91 🟡 |
| c128\|C\|u64 | 1.7535 | 1.5930 | 0.91 🟡 |
| u8\|strided\|f32 | 0.3771 | 0.3448 | 0.91 🟡 |
| f32\|negrow\|u64 | 1.5317 | 1.4006 | 0.91 🟡 |
| bool\|F\|u8 | 0.3439 | 0.3145 | 0.91 🟡 |
| i64\|negcol\|u8 | 0.2637 | 0.2415 | 0.92 🟡 |
| u8\|bcast\|i64 | 1.3234 | 1.2128 | 0.92 🟡 |
| u64\|T\|u8 | 0.2155 | 0.1976 | 0.92 🟡 |
| i64\|F\|i8 | 0.2186 | 0.2005 | 0.92 🟡 |
| i64\|T\|u8 | 0.2162 | 0.1984 | 0.92 🟡 |
| u64\|negcol\|u8 | 0.2492 | 0.2293 | 0.92 🟡 |
| u64\|C\|u8 | 0.2133 | 0.1969 | 0.92 🟡 |
| f64\|T\|u64 | 1.4455 | 1.3351 | 0.92 🟡 |
| f64\|F\|u64 | 1.4401 | 1.3320 | 0.92 🟡 |
| u64\|bcast\|i64 | 1.3091 | 1.2114 | 0.93 🟡 |
| i64\|strided\|u32 | 0.3954 | 0.3660 | 0.93 🟡 |
| u64\|strided\|u16 | 0.1631 | 0.1510 | 0.93 🟡 |
| bool\|negrow\|char | 0.4833 | 0.4478 | 0.93 🟡 |
| bool\|negrow\|u16 | 0.4785 | 0.4442 | 0.93 🟡 |
| f64\|C\|i8 | 0.2922 | 0.2712 | 0.93 🟡 |
| u64\|strided\|i8 | 0.1396 | 0.1297 | 0.93 🟡 |
| c128\|negrow\|u64 | 1.8228 | 1.6940 | 0.93 🟡 |
| i64\|strided\|i16 | 0.1704 | 0.1590 | 0.93 🟡 |
| bool\|bcast\|f64 | 1.3065 | 1.2196 | 0.93 🟡 |
| i64\|T\|i8 | 0.2138 | 0.1996 | 0.93 🟡 |
| bool\|negcol\|i8 | 0.3387 | 0.3163 | 0.93 🟡 |
| c128\|T\|i8 | 0.3749 | 0.3502 | 0.93 🟡 |
| f64\|negrow\|u64 | 1.5034 | 1.4056 | 0.93 🟡 |
| u8\|bcast\|f64 | 1.3104 | 1.2264 | 0.94 🟡 |
| f32\|negrow\|i64 | 1.5615 | 1.4618 | 0.94 🟡 |
| c128\|T\|bool | 0.4813 | 0.4515 | 0.94 🟡 |
| bool\|negcol\|u8 | 0.3361 | 0.3156 | 0.94 🟡 |
| c128\|strided\|u32 | 0.6259 | 0.5879 | 0.94 🟡 |
| c128\|F\|i8 | 0.3747 | 0.3524 | 0.94 🟡 |
| i8\|strided\|bool | 0.1576 | 0.1488 | 0.94 🟡 |
| i32\|bcast\|u64 | 1.3220 | 1.2502 | 0.95 🟡 |
| bool\|T\|i16 | 0.4736 | 0.4482 | 0.95 🟡 |
| c128\|T\|i16 | 0.6104 | 0.5780 | 0.95 🟡 |
| i16\|negrow\|i16 | 0.3694 | 0.3503 | 0.95 🟡 |
| u64\|strided\|u8 | 0.1385 | 0.1314 | 0.95 🟡 |
| f64\|strided\|u32 | 0.3933 | 0.3739 | 0.95 🟡 |
| bool\|T\|char | 0.4770 | 0.4541 | 0.95 🟡 |
| c128\|negcol\|f16 | 1.9606 | 1.8684 | 0.95 🟡 |
| u64\|F\|i8 | 0.2163 | 0.2069 | 0.96 🟡 |
| c128\|F\|u64 | 1.7233 | 1.6485 | 0.96 🟡 |
| i64\|strided\|bool | 0.1728 | 0.1656 | 0.96 🟡 |
| bool\|sliced\|i16 | 0.4849 | 0.4652 | 0.96 🟡 |
| c128\|T\|i64 | 1.6355 | 1.5692 | 0.96 🟡 |
| f16\|T\|f64 | 1.5261 | 1.4654 | 0.96 🟡 |
| bool\|T\|u16 | 0.4712 | 0.4527 | 0.96 🟡 |
| c128\|strided\|i64 | 0.9412 | 0.9049 | 0.96 🟡 |
| c128\|C\|i64 | 1.6848 | 1.6222 | 0.96 🟡 |
| u64\|strided\|i16 | 0.1588 | 0.1532 | 0.96 🟡 |
| f16\|negcol\|f64 | 1.5460 | 1.4943 | 0.97 🟡 |
| f16\|negrow\|f64 | 1.5430 | 1.4929 | 0.97 🟡 |
| c128\|F\|i64 | 1.6300 | 1.5780 | 0.97 🟡 |
| c128\|C\|bool | 0.5680 | 0.5499 | 0.97 🟡 |
| u32\|strided\|i8 | 0.1016 | 0.0984 | 0.97 🟡 |
| i64\|C\|u64 | 1.4194 | 1.3757 | 0.97 🟡 |
| c128\|negcol\|i16 | 0.8341 | 0.8095 | 0.97 🟡 |
| i16\|bcast\|i64 | 1.3572 | 1.3181 | 0.97 🟡 |
| i32\|bcast\|f64 | 1.2875 | 1.2506 | 0.97 🟡 |
| u64\|strided\|f32 | 0.4257 | 0.4158 | 0.98 🟡 |
| f16\|C\|u64 | 1.9227 | 1.8841 | 0.98 🟡 |
| i16\|bcast\|f64 | 1.3197 | 1.2937 | 0.98 🟡 |
| u64\|strided\|char | 0.1514 | 0.1485 | 0.98 🟡 |
| f64\|C\|u64 | 1.4743 | 1.4487 | 0.98 🟡 |
| c128\|C\|i8 | 0.3926 | 0.3871 | 0.99 🟡 |
| u32\|strided\|char | 0.1196 | 0.1180 | 0.99 🟡 |
| f64\|strided\|u8 | 0.1435 | 0.1416 | 0.99 🟡 |
| u8\|negcol\|bool | 0.3082 | 0.3051 | 0.99 🟡 |
| c128\|sliced\|i64 | 1.6716 | 1.6558 | 0.99 🟡 |
| c128\|negrow\|i64 | 1.7452 | 1.7295 | 0.99 🟡 |
| u64\|sliced\|f32 | 0.7778 | 0.7716 | 0.99 🟡 |
| u64\|negrow\|f32 | 0.7943 | 0.7904 | 1.00 🟡 |
| i32\|bcast\|i64 | 1.2645 | 1.2641 | 1.00 🟡 |

_1568 comparable cells (1800 NumSharp rows; 380 lagging <1.0)._


---

## Fusion — np.evaluate vs unfused chains

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.65 ms   unfused    6.65 ms   (1.82x)
  (a-b)/(a+b) fused    3.02 ms   unfused   12.63 ms   (4.18x)
  sum(a*b)    fused    2.60 ms   unfused    4.61 ms   (1.77x)
  sum(af*bf)  fused    1.57 ms   unfused    1.90 ms   (1.21x)  [f32]
  a*b+c out=  fused    3.84 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.07 ms   unfused    4.54 ms   (1.48x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.25 ms   unfused    6.65 ms   (1.56x)
    [F      ] fused    3.89 ms   unfused    6.53 ms   (1.68x)
    [T      ] fused    3.81 ms   unfused    6.61 ms   (1.74x)
    [strided] fused    3.66 ms   unfused    5.00 ms   (1.36x)
    [bcast  ] fused    1.86 ms   unfused    4.10 ms   (2.20x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         12.91 ms
  (a-b)/(a+b)   19.38 ms
  sum(a*b)       8.45 ms
  sum(af*bf)     4.31 ms  [f32]
  a*b+c out=     5.28 ms  [two-pass with out=]
  i4*2+f8       10.05 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.80 ms
    [F      ]   13.45 ms
    [T      ]   14.27 ms
    [strided]    8.58 ms
    [bcast  ]   13.10 ms
```


---

## Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-731fd42e1dd9378fe3a752f5d05d483a3188d66c472f3d268b9febfc8d1b8f5a\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0097 ms | 0.0077 ms | 0.0087 | 1.130x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1257 ms | 2.4239 ms | 1.0977 | 8.733x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1179 ms | 2.4322 ms | 1.2291 | 10.425x |
| `np.dot(a, b)` | 100,000 | 0.0092 ms | 0.0076 ms | 0.0083 | 1.092x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2206 ms | 0.0939 ms | 0.2651 | 2.823x |
| `np.inner(a, b)` | 100,000 | 0.0075 ms | 0.0078 ms | 0.0076 | 1.013x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0166 ms | 0.0051 | 0.307x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0414 ms | 0.0229 | 0.553x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0454 ms | 0.0424 | 0.934x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.3186 ms | 0.3338 | 1.048x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0100 ms | 0.0070 | 0.700x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0322 ms | 0.0321 | 0.997x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1395 ms | 0.1148 | 0.823x |
| `np.linalg.eig(a)` | 96 | missing_backend | 1.4870 ms | 1.5073 | 1.014x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0577 ms | 0.0535 | 0.927x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.4273 ms | 0.4739 | 1.109x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0800 ms | 0.0700 | 0.875x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 0.8608 ms | 0.9121 | 1.060x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0245 ms | 0.0239 | 0.976x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.1664 ms | 0.1824 | 1.096x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0209 ms | 0.0115 | 0.550x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.0881 ms | 0.0883 | 1.002x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0946 ms | 0.0918 | 0.970x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 0.8382 ms | 0.8770 | 1.046x |
| `np.linalg.matmul(a, b)` | 128 | 0.2107 ms | 0.0915 ms | 0.0670 | 0.732x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0364 ms | 0.0467 | 1.283x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.3064 ms | 0.3300 | 1.077x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0164 ms | 0.0134 | 0.817x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.0919 ms | 0.0970 | 1.055x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0446 ms | 0.0421 | 0.944x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.3003 ms | 0.3034 | 1.010x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3759 ms | 0.1970 ms | 0.1324 | 0.672x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0356 ms | 0.0468 | 1.315x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.2922 ms | 0.3086 | 1.056x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.1116 ms | 0.0982 | 0.880x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 0.8636 ms | 0.8362 | 0.968x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0360 ms | 0.0263 | 0.731x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.1876 ms | 0.1878 | 1.001x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0088 ms | 0.0079 | 0.898x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0319 ms | 0.0328 | 1.028x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0089 ms | 0.0107 | 1.202x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0320 ms | 0.0335 | 1.047x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0857 ms | 0.0934 | 1.090x |
| `np.linalg.svd(a)` | 96 | missing_backend | 0.7232 ms | 0.8002 | 1.106x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0329 ms | 0.0394 | 1.198x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.2928 ms | 0.3313 | 1.131x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2046 ms | 0.0895 ms | 0.0663 | 0.741x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0165 ms | 0.0147 | 0.891x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.0869 ms | 0.0988 | 1.137x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0094 ms | 0.0092 | 0.979x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0326 ms | 0.0391 | 1.199x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0722 ms | 0.0145 ms | 0.0079 | 0.545x |
| `np.matmul(a, b)` | 128 | 0.2112 ms | 0.0933 ms | 0.0636 | 0.682x |
| `np.matvec(a, b)` | 128 | 0.0231 ms | 0.0022 ms | 0.0022 | 1.000x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.2248 ms | 0.0520 | 0.231x |
| `np.roots(p)` | 16 | missing_backend | 0.0606 ms | 0.0442 | 0.729x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2107 ms | 0.0943 ms | 0.0664 | 0.704x |
| `np.vdot(a, b)` | 100,000 | 0.0079 ms | 0.0083 ms | 0.0075 | 0.949x |
| `np.vecdot(a, b)` | 100,000 | 0.1582 ms | 0.0078 ms | 0.0076 | 0.974x |
| `np.vecmat(a, b)` | 128 | 0.0036 ms | 0.0022 ms | 0.0017 | 0.773x |
