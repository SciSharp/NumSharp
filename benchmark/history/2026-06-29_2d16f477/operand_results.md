# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 2.41 ✅ | 2.16 ✅ | 0.63 🟡 | 2.13 ✅ | 2.45 ✅ | 2.35 ✅ | 1.85 ✅ |
| 1-D strided a[::2] | 1.84 ✅ | 1.46 ✅ | 0.53 🟡 | 1.51 ✅ | 1.81 ✅ | 1.93 ✅ | 1.40 ✅ |
| 1-D reversed a[::-1] | 2.49 ✅ | 1.95 ✅ | 0.56 🟡 | 2.14 ✅ | 2.29 ✅ | 2.22 ✅ | 1.76 ✅ |
| array + scalar | 2.72 ✅ | 1.93 ✅ | 0.63 🟡 | 1.91 ✅ | 2.26 ✅ | 2.64 ✅ | 1.83 ✅ |
| scalar + array | 2.40 ✅ | 1.98 ✅ | 0.64 🟡 | 2.01 ✅ | 2.19 ✅ | 2.69 ✅ | 1.82 ✅ |
| mixed C + F | 2.29 ✅ | 2.02 ✅ | 0.62 🟡 | 2.02 ✅ | 2.09 ✅ | 1.87 ✅ | 1.68 ✅ |
| mixed C + T | 2.53 ✅ | 2.04 ✅ | 0.62 🟡 | 2.01 ✅ | 2.33 ✅ | 2.30 ✅ | 1.80 ✅ |
| binary broadcast +row(1,C) | 2.71 ✅ | 2.09 ✅ | 0.63 🟡 | 1.95 ✅ | 2.52 ✅ | 2.58 ✅ | 1.89 ✅ |
| binary broadcast +col(R,1) | 2.61 ✅ | 2.02 ✅ | 0.56 🟡 | 2.07 ✅ | 2.54 ✅ | 2.96 ✅ | 1.89 ✅ |
| col-broadcast unary (inner stride-0) | 2.54 ✅ | 1.56 ✅ | 0.88 🟡 | 1.65 ✅ | 2.64 ✅ | 6.22 ✅ | 2.13 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_strided|f16 | 2.6261 | 1.3926 | 0.53 🟡 |
| 1d_rev|f16 | 5.2838 | 2.9653 | 0.56 🟡 |
| bcast_col|f16 | 5.2887 | 2.9837 | 0.56 🟡 |
| mix_C_T|f16 | 5.3031 | 3.2873 | 0.62 🟡 |
| mix_C_F|f16 | 5.3153 | 3.3015 | 0.62 🟡 |
| 1d_C|f16 | 4.7456 | 2.9682 | 0.63 🟡 |
| bcast_row|f16 | 4.7695 | 2.9999 | 0.63 🟡 |
| scalar_rhs|f16 | 4.7058 | 2.9602 | 0.63 🟡 |
| scalar_lhs|f16 | 4.6383 | 2.9742 | 0.64 🟡 |
| colbcast_unary|f16 | 0.4753 | 0.4173 | 0.88 🟡 |
| 1d_strided|f32 | 0.2366 | 0.3444 | 1.46 ✅ |
| 1d_strided|i32 | 0.2433 | 0.3676 | 1.51 ✅ |

_60 comparable cells._
