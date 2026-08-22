# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
