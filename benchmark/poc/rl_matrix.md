# Reduction × Layout × dtype parity (NumSharp vs NumPy 2.4.2)

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1 🟡≥0.5 🟠≥0.2 🔴<0.2

## Geomean by layout (all dtypes/ops/axes)

| size | C | F | T | strided | negstride | sliced |
|---|---|---|---|---|---|---|
| 100K | 1.21 ✅ | 1.33 ✅ | 1.29 ✅ | 0.61 🟡 | 0.78 🟡 | 1.01 ✅ |
| 1M | 1.63 ✅ | 1.53 ✅ | 1.68 ✅ | 0.90 🟡 | 1.31 ✅ | 1.34 ✅ |

## Geomean by dtype (all layouts/ops/axes)

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.78 🟡 | 0.88 🟡 | 1.16 ✅ | 0.16 🔴 | 0.72 🟡 | 2.58 ✅ | 2.63 ✅ |
| 1M | 2.11 ✅ | 2.13 ✅ | 1.60 ✅ | 0.18 🔴 | 0.70 🟡 | 2.21 ✅ | 2.75 ✅ |

## Geomean by op (all dtypes/layouts/axes)

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 1.10 ✅ | 0.71 🟡 | 0.73 🟡 | 1.94 ✅ |
| 1M | 1.55 ✅ | 0.97 🟡 | 1.00 🟡 | 2.56 ✅ |

## Worst 30 cells (NumSharp slowest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|dec\|negstride\|sum\|ax0 | 0.5667 | 0.0103 | 0.02 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 5.5780 | 0.2344 | 0.04 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.5411 | 0.0231 | 0.04 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 5.7330 | 0.2456 | 0.04 🔴 |
| 1M\|dec\|negstride\|sum\|ax0 | 5.8865 | 0.2564 | 0.04 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.5409 | 0.0236 | 0.04 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.4341 | 0.2464 | 0.05 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.5444 | 0.0251 | 0.05 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 5.4982 | 0.2601 | 0.05 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.5345 | 0.0266 | 0.05 🔴 |
| 100K\|dec\|T\|max\|ax1 | 0.2181 | 0.0134 | 0.06 🔴 |
| 100K\|dec\|negstride\|max\|ax0 | 0.2160 | 0.0150 | 0.07 🔴 |
| 100K\|dec\|negstride\|min\|ax0 | 0.1730 | 0.0134 | 0.08 🔴 |
| 100K\|dec\|negstride\|sum\|ax1 | 0.2203 | 0.0188 | 0.09 🔴 |
| 100K\|dec\|F\|sum\|ax0 | 0.2096 | 0.0180 | 0.09 🔴 |
| 100K\|dec\|T\|max\|ax0 | 0.1698 | 0.0174 | 0.10 🔴 |
| 1M\|dec\|strided\|sum\|ax0 | 2.7219 | 0.3003 | 0.11 🔴 |
| 100K\|dec\|negstride\|min\|ax1 | 0.1463 | 0.0170 | 0.12 🔴 |
| 100K\|i32\|strided\|sum\|ax1 | 0.1987 | 0.0232 | 0.12 🔴 |
| 1M\|dec\|F\|max\|ax1 | 2.7517 | 0.3422 | 0.12 🔴 |
| 100K\|dec\|strided\|sum\|ax0 | 0.2745 | 0.0344 | 0.13 🔴 |
| 100K\|i32\|strided\|sum\|ax0 | 0.2061 | 0.0283 | 0.14 🔴 |
| 100K\|dec\|C\|max\|ax0 | 0.2865 | 0.0394 | 0.14 🔴 |
| 100K\|dec\|T\|sum\|ax0 | 0.2180 | 0.0314 | 0.14 🔴 |
| 100K\|dec\|sliced\|sum\|ax1 | 0.2169 | 0.0312 | 0.14 🔴 |
| 100K\|dec\|C\|sum\|ax1 | 0.2106 | 0.0309 | 0.15 🔴 |
| 1M\|dec\|sliced\|max\|ax0 | 2.2509 | 0.3450 | 0.15 🔴 |
| 1M\|dec\|T\|sum\|ax0 | 2.4251 | 0.3771 | 0.16 🔴 |
| 1M\|dec\|C\|max\|ax0 | 2.1834 | 0.3481 | 0.16 🔴 |
| 1M\|dec\|sliced\|sum\|ax1 | 2.2193 | 0.3545 | 0.16 🔴 |

## Best 12 cells (NumSharp fastest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|f32\|T\|prod\|ax0 | 0.0427 | 0.9358 | 21.93 ✅ |
| 1M\|f32\|F\|prod\|ax0 | 0.0471 | 0.9469 | 20.12 ✅ |
| 100K\|i32\|F\|sum\|ax0 | 0.0051 | 0.0928 | 18.27 ✅ |
| 100K\|f32\|T\|prod\|ax0 | 0.0033 | 0.0584 | 17.55 ✅ |
| 1M\|i32\|negstride\|sum\|ax0 | 0.0961 | 1.6724 | 17.40 ✅ |
| 100K\|i32\|T\|sum\|ax0 | 0.0056 | 0.0951 | 17.05 ✅ |
| 100K\|f32\|F\|prod\|ax0 | 0.0034 | 0.0582 | 16.99 ✅ |
| 100K\|i32\|C\|sum\|ax1 | 0.0054 | 0.0898 | 16.57 ✅ |
| 100K\|f32\|C\|prod\|ax1 | 0.0035 | 0.0582 | 16.47 ✅ |
| 1M\|i32\|C\|sum\|ax0 | 0.0922 | 1.4989 | 16.26 ✅ |
| 100K\|i32\|negstride\|sum\|ax0 | 0.0089 | 0.1418 | 15.87 ✅ |
| 1M\|i32\|T\|sum\|ax0 | 0.0595 | 0.9058 | 15.23 ✅ |

_648 cells compared._
