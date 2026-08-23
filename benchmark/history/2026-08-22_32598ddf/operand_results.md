# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)

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
