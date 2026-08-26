# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
