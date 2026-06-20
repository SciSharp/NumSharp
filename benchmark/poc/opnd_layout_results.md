# POC — operand / extra layout classes (NumSharp vs NumPy 2.4.2)

Layout classes the op×layout×dtype matrix omits. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.

| case | f64 | f32 | f16 | i32 | i64 | c128 | geomean |
|---|---|---|---|---|---|---|---|
| 1-D contiguous (a+a) | 2.58 ✅ | 2.16 ✅ | 0.62 🟡 | 2.03 ✅ | 2.21 ✅ | 2.29 ✅ | 1.81 ✅ |
| 1-D strided a[::2] | 1.94 ✅ | 1.57 ✅ | 0.53 🟡 | 1.53 ✅ | 1.73 ✅ | 1.88 ✅ | 1.41 ✅ |
| 1-D reversed a[::-1] | 2.25 ✅ | 1.98 ✅ | 0.57 🟡 | 1.97 ✅ | 2.46 ✅ | 1.90 ✅ | 1.69 ✅ |
| array + scalar | 2.61 ✅ | 1.99 ✅ | 0.64 🟡 | 1.98 ✅ | 2.19 ✅ | 1.95 ✅ | 1.74 ✅ |
| scalar + array | 2.57 ✅ | 2.06 ✅ | 0.64 🟡 | 1.93 ✅ | 2.26 ✅ | 1.97 ✅ | 1.75 ✅ |
| mixed C + F | 2.30 ✅ | 2.13 ✅ | 0.62 🟡 | 1.98 ✅ | 2.27 ✅ | 1.63 ✅ | 1.68 ✅ |
| mixed C + T | 2.45 ✅ | 2.13 ✅ | 0.62 🟡 | 2.05 ✅ | 2.50 ✅ | 2.37 ✅ | 1.84 ✅ |
| binary broadcast +row(1,C) | 2.67 ✅ | 2.06 ✅ | 0.64 🟡 | 2.05 ✅ | 2.58 ✅ | 2.22 ✅ | 1.86 ✅ |
| binary broadcast +col(R,1) | 2.58 ✅ | 2.22 ✅ | 0.56 🟡 | 2.04 ✅ | 2.48 ✅ | 2.83 ✅ | 1.89 ✅ |
| col-broadcast unary (inner stride-0) | 2.58 ✅ | 1.85 ✅ | 1.14 ✅ | 1.80 ✅ | 2.73 ✅ | 5.24 ✅ | 2.28 ✅ |

**Worst 12 (key · NumSharp ms · NumPy ms · ratio)**

| key | NS ms | NP ms | ratio |
|---|---|---|---|
| 1d_strided|f16 | 2.6370 | 1.4034 | 0.53 🟡 |
| bcast_col|f16 | 5.3293 | 3.0046 | 0.56 🟡 |
| 1d_rev|f16 | 5.2866 | 2.9931 | 0.57 🟡 |
| mix_C_T|f16 | 5.3551 | 3.3238 | 0.62 🟡 |
| 1d_C|f16 | 4.8281 | 2.9999 | 0.62 🟡 |
| mix_C_F|f16 | 5.3843 | 3.3524 | 0.62 🟡 |
| bcast_row|f16 | 4.7781 | 3.0363 | 0.64 🟡 |
| scalar_rhs|f16 | 4.6920 | 2.9875 | 0.64 🟡 |
| scalar_lhs|f16 | 4.6505 | 2.9846 | 0.64 🟡 |
| colbcast_unary|f16 | 0.3829 | 0.4372 | 1.14 ✅ |
| 1d_strided|i32 | 0.2566 | 0.3914 | 1.53 ✅ |
| 1d_strided|f32 | 0.2383 | 0.3738 | 1.57 ✅ |
