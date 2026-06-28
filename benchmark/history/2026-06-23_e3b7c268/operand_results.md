# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 2.55 ✅ | 2.18 ✅ | 0.62 🟡 | 2.04 ✅ | 2.54 ✅ | 2.38 ✅ | 1.87 ✅ |
| 1-D strided a[::2] | 1.83 ✅ | 1.36 ✅ | 0.52 🟡 | 1.37 ✅ | 1.77 ✅ | 1.86 ✅ | 1.34 ✅ |
| 1-D reversed a[::-1] | 2.26 ✅ | 2.09 ✅ | 0.56 🟡 | 2.00 ✅ | 2.14 ✅ | 2.23 ✅ | 1.71 ✅ |
| array + scalar | 2.63 ✅ | 2.11 ✅ | 0.63 🟡 | 1.80 ✅ | 2.17 ✅ | 2.58 ✅ | 1.81 ✅ |
| scalar + array | 2.35 ✅ | 2.10 ✅ | 0.64 🟡 | 1.98 ✅ | 2.48 ✅ | 2.59 ✅ | 1.85 ✅ |
| mixed C + F | 2.12 ✅ | 2.01 ✅ | 0.62 🟡 | 1.98 ✅ | 2.02 ✅ | 2.26 ✅ | 1.70 ✅ |
| mixed C + T | 2.59 ✅ | 2.13 ✅ | 0.62 🟡 | 2.04 ✅ | 2.20 ✅ | 2.51 ✅ | 1.84 ✅ |
| binary broadcast +row(1,C) | 2.72 ✅ | 2.28 ✅ | 0.63 🟡 | 2.00 ✅ | 2.42 ✅ | 2.42 ✅ | 1.89 ✅ |
| binary broadcast +col(R,1) | 2.68 ✅ | 2.14 ✅ | 0.56 🟡 | 1.99 ✅ | 2.45 ✅ | 2.82 ✅ | 1.88 ✅ |
| col-broadcast unary (inner stride-0) | 2.55 ✅ | 1.76 ✅ | 0.86 🟡 | 1.69 ✅ | 2.66 ✅ | 6.09 ✅ | 2.18 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_strided|f16 | 2.6414 | 1.3865 | 0.52 🟡 |
| 1d_rev|f16 | 5.3054 | 2.9810 | 0.56 🟡 |
| bcast_col|f16 | 5.2996 | 2.9861 | 0.56 🟡 |
| mix_C_T|f16 | 5.3156 | 3.2865 | 0.62 🟡 |
| 1d_C|f16 | 4.7503 | 2.9588 | 0.62 🟡 |
| mix_C_F|f16 | 5.3110 | 3.3155 | 0.62 🟡 |
| bcast_row|f16 | 4.7682 | 2.9983 | 0.63 🟡 |
| scalar_rhs|f16 | 4.7137 | 2.9711 | 0.63 🟡 |
| scalar_lhs|f16 | 4.6575 | 2.9643 | 0.64 🟡 |
| colbcast_unary|f16 | 0.4898 | 0.4232 | 0.86 🟡 |
| 1d_strided|f32 | 0.2652 | 0.3608 | 1.36 ✅ |
| 1d_strided|i32 | 0.2798 | 0.3832 | 1.37 ✅ |

_60 comparable cells._
