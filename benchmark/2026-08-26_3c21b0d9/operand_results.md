# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
