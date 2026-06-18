# Reduction × Layout × dtype parity (NumSharp vs NumPy 2.4.2)

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1 🟡≥0.5 🟠≥0.2 🔴<0.2

## Geomean by layout (all dtypes/ops/axes)

| size | C | F | T | strided | negstride | sliced |
|---|---|---|---|---|---|---|
| 100K | 0.82 🟡 | 0.69 🟡 | 0.65 🟡 | 0.44 🟠 | 0.54 🟡 | 0.67 🟡 |
| 1M | 0.84 🟡 | 0.63 🟡 | 0.67 🟡 | 0.48 🟠 | 0.73 🟡 | 0.67 🟡 |

## Geomean by dtype (all layouts/ops/axes)

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.78 🟡 | 0.89 🟡 | 0.85 🟡 | 0.10 🔴 | 0.43 🟠 | 0.80 🟡 | 1.17 ✅ |
| 1M | 0.84 🟡 | 1.07 ✅ | 1.00 ✅ | 0.11 🔴 | 0.41 🟠 | 0.73 🟡 | 1.21 ✅ |

## Geomean by op (all dtypes/layouts/axes)

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.62 🟡 | 0.48 🟠 | 0.51 🟡 | 1.09 ✅ |
| 1M | 0.71 🟡 | 0.51 🟡 | 0.52 🟡 | 1.10 ✅ |

## Worst 30 cells (NumSharp slowest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|dec\|negstride\|sum\|ax0 | 0.7711 | 0.0103 | 0.01 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7656 | 0.0231 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.7629 | 0.2344 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7723 | 0.0236 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.7580 | 0.2464 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7880 | 0.0251 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.6954 | 0.2456 | 0.03 🔴 |
| 1M\|dec\|negstride\|sum\|ax0 | 7.8558 | 0.2564 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.6701 | 0.2601 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7587 | 0.0266 | 0.04 🔴 |
| 100K\|dec\|T\|max\|ax1 | 0.3457 | 0.0134 | 0.04 🔴 |
| 100K\|dec\|negstride\|min\|ax0 | 0.3371 | 0.0134 | 0.04 🔴 |
| 100K\|dec\|F\|sum\|ax0 | 0.4424 | 0.0180 | 0.04 🔴 |
| 100K\|dec\|negstride\|max\|ax0 | 0.3508 | 0.0150 | 0.04 🔴 |
| 100K\|dec\|negstride\|sum\|ax1 | 0.4362 | 0.0188 | 0.04 🔴 |
| 1M\|i32\|F\|sum\|ax0 | 6.0787 | 0.3266 | 0.05 🔴 |
| 100K\|dec\|negstride\|min\|ax1 | 0.2658 | 0.0170 | 0.06 🔴 |
| 100K\|dec\|T\|max\|ax0 | 0.2710 | 0.0174 | 0.06 🔴 |
| 1M\|i32\|F\|sum\|ax1 | 6.3503 | 0.4385 | 0.07 🔴 |
| 100K\|dec\|C\|sum\|ax1 | 0.4382 | 0.0309 | 0.07 🔴 |
| 100K\|dec\|sliced\|sum\|ax1 | 0.4404 | 0.0312 | 0.07 🔴 |
| 100K\|dec\|T\|sum\|ax0 | 0.4369 | 0.0314 | 0.07 🔴 |
| 1M\|dec\|strided\|sum\|ax0 | 3.9868 | 0.3003 | 0.08 🔴 |
| 100K\|i32\|strided\|sum\|ax1 | 0.3015 | 0.0232 | 0.08 🔴 |
| 1M\|dec\|sliced\|sum\|ax1 | 4.3671 | 0.3545 | 0.08 🔴 |
| 100K\|dec\|T\|min\|ax1 | 0.3388 | 0.0290 | 0.09 🔴 |
| 1M\|dec\|T\|sum\|ax0 | 4.3803 | 0.3771 | 0.09 🔴 |
| 1M\|dec\|F\|sum\|ax0 | 4.2594 | 0.3696 | 0.09 🔴 |
| 100K\|dec\|strided\|sum\|ax0 | 0.3896 | 0.0344 | 0.09 🔴 |
| 100K\|dec\|strided\|sum\|ax1 | 0.2233 | 0.0197 | 0.09 🔴 |

## Best 12 cells (NumSharp fastest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|f32\|F\|prod\|ax0 | 0.0501 | 0.9469 | 18.91 ✅ |
| 1M\|f32\|C\|prod\|ax1 | 0.0516 | 0.9392 | 18.20 ✅ |
| 1M\|f32\|T\|prod\|ax0 | 0.0586 | 0.9358 | 15.96 ✅ |
| 100K\|f32\|F\|prod\|ax0 | 0.0037 | 0.0582 | 15.80 ✅ |
| 100K\|f32\|T\|prod\|ax0 | 0.0037 | 0.0584 | 15.58 ✅ |
| 100K\|f32\|C\|prod\|ax1 | 0.0041 | 0.0582 | 14.28 ✅ |
| 100K\|f64\|F\|prod\|ax0 | 0.0050 | 0.0586 | 11.67 ✅ |
| 1M\|i32\|C\|sum\|ax0 | 0.1353 | 1.4989 | 11.08 ✅ |
| 100K\|f64\|C\|prod\|ax1 | 0.0053 | 0.0581 | 10.94 ✅ |
| 100K\|f64\|T\|prod\|ax0 | 0.0054 | 0.0583 | 10.83 ✅ |
| 1M\|i32\|negstride\|sum\|ax0 | 0.1555 | 1.6724 | 10.75 ✅ |
| 1M\|i32\|C\|sum\|ax1 | 0.0887 | 0.8716 | 9.82 ✅ |

_648 cells compared._
