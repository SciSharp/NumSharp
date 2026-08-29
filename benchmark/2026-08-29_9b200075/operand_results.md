# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
