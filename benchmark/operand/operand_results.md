# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

The layout classes the per-operand layout grid (benchmark/layout) can't express. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 2.46 ✅ | 1.77 ✅ | 0.62 🟡 | 2.11 ✅ | 2.14 ✅ | 2.20 ✅ | 1.73 ✅ |
| 1-D strided a[::2] | 1.71 ✅ | 1.40 ✅ | 0.57 🟡 | 1.43 ✅ | 1.82 ✅ | 1.86 ✅ | 1.37 ✅ |
| 1-D reversed a[::-1] | 2.36 ✅ | 1.63 ✅ | 0.55 🟡 | 2.23 ✅ | 2.46 ✅ | 2.03 ✅ | 1.69 ✅ |
| array + scalar | 2.44 ✅ | 1.66 ✅ | 0.60 🟡 | 2.05 ✅ | 2.29 ✅ | 2.35 ✅ | 1.73 ✅ |
| scalar + array | 2.10 ✅ | 1.81 ✅ | 0.63 🟡 | 2.09 ✅ | 2.27 ✅ | 2.14 ✅ | 1.70 ✅ |
| mixed C + F | 1.91 ✅ | 1.90 ✅ | 0.61 🟡 | 2.04 ✅ | 2.10 ✅ | 1.97 ✅ | 1.63 ✅ |
| mixed C + T | 2.16 ✅ | 1.75 ✅ | 0.60 🟡 | 2.05 ✅ | 2.22 ✅ | 1.98 ✅ | 1.66 ✅ |
| binary broadcast +row(1,C) | 2.56 ✅ | 1.57 ✅ | 0.62 🟡 | 2.20 ✅ | 2.49 ✅ | 2.05 ✅ | 1.74 ✅ |
| binary broadcast +col(R,1) | 2.58 ✅ | 1.78 ✅ | 0.56 🟡 | 2.14 ✅ | 2.68 ✅ | 2.49 ✅ | 1.82 ✅ |
| col-broadcast unary (inner stride-0) | 2.36 ✅ | 1.78 ✅ | 1.14 ✅ | 1.80 ✅ | 2.58 ✅ | 4.95 ✅ | 2.19 ✅ |

**Worst 12 cells**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1d_rev|f16 | 5.5378 | 3.0571 | 0.55 🟡 |
| bcast_col|f16 | 5.5143 | 3.0736 | 0.56 🟡 |
| 1d_strided|f16 | 2.7227 | 1.5583 | 0.57 🟡 |
| scalar_rhs|f16 | 5.0665 | 3.0530 | 0.60 🟡 |
| mix_C_T|f16 | 5.6143 | 3.3869 | 0.60 🟡 |
| mix_C_F|f16 | 5.5398 | 3.4030 | 0.61 🟡 |
| bcast_row|f16 | 5.0252 | 3.0921 | 0.62 🟡 |
| 1d_C|f16 | 4.9049 | 3.0533 | 0.62 🟡 |
| scalar_lhs|f16 | 4.8426 | 3.0687 | 0.63 🟡 |
| colbcast_unary|f16 | 0.3871 | 0.4409 | 1.14 ✅ |
| 1d_strided|f32 | 0.2779 | 0.3889 | 1.40 ✅ |
| 1d_strided|i32 | 0.2680 | 0.3828 | 1.43 ✅ |

_60 comparable cells._
