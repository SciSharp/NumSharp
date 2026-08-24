# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
