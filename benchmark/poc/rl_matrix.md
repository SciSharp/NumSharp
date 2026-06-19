# Reduction × Layout × dtype parity (NumSharp vs NumPy 2.4.2)

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1 🟡≥0.5 🟠≥0.2 🔴<0.2

## Geomean by layout (all dtypes/ops/axes)

| size | C | F | T | strided | negstride | sliced |
|---|---|---|---|---|---|---|
| 100K | 1.23 ✅ | 1.35 ✅ | 1.32 ✅ | 0.60 🟡 | 0.77 🟡 | 0.99 🟡 |
| 1M | 1.69 ✅ | 1.62 ✅ | 1.77 ✅ | 0.93 🟡 | 1.42 ✅ | 1.32 ✅ |

## Geomean by dtype (all layouts/ops/axes)

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.77 🟡 | 0.92 🟡 | 1.17 ✅ | 0.16 🔴 | 0.73 🟡 | 2.61 ✅ | 2.52 ✅ |
| 1M | 2.32 ✅ | 2.44 ✅ | 1.67 ✅ | 0.18 🔴 | 0.71 🟡 | 2.09 ✅ | 2.83 ✅ |

## Geomean by op (all dtypes/layouts/axes)

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 1.07 ✅ | 0.72 🟡 | 0.75 🟡 | 1.91 ✅ |
| 1M | 1.57 ✅ | 1.03 ✅ | 1.06 ✅ | 2.65 ✅ |

## Worst 30 cells (NumSharp slowest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|dec\|negstride\|sum\|ax0 | 0.5474 | 0.0103 | 0.02 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.5447 | 0.0231 | 0.04 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.5456 | 0.0236 | 0.04 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 5.3773 | 0.2344 | 0.04 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 5.4113 | 0.2456 | 0.05 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.3985 | 0.2464 | 0.05 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.5463 | 0.0251 | 0.05 🔴 |
| 1M\|dec\|negstride\|sum\|ax0 | 5.4202 | 0.2564 | 0.05 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 5.4938 | 0.2601 | 0.05 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.5370 | 0.0266 | 0.05 🔴 |
| 100K\|dec\|T\|max\|ax1 | 0.2175 | 0.0134 | 0.06 🔴 |
| 100K\|dec\|negstride\|max\|ax0 | 0.2187 | 0.0150 | 0.07 🔴 |
| 100K\|dec\|negstride\|min\|ax0 | 0.1791 | 0.0134 | 0.07 🔴 |
| 100K\|dec\|F\|sum\|ax0 | 0.2115 | 0.0180 | 0.09 🔴 |
| 100K\|dec\|negstride\|sum\|ax1 | 0.2155 | 0.0188 | 0.09 🔴 |
| 100K\|dec\|negstride\|min\|ax1 | 0.1565 | 0.0170 | 0.11 🔴 |
| 100K\|dec\|T\|max\|ax0 | 0.1604 | 0.0174 | 0.11 🔴 |
| 1M\|dec\|strided\|sum\|ax0 | 2.7085 | 0.3003 | 0.11 🔴 |
| 100K\|i32\|strided\|sum\|ax1 | 0.1972 | 0.0232 | 0.12 🔴 |
| 100K\|dec\|C\|max\|ax0 | 0.3338 | 0.0394 | 0.12 🔴 |
| 100K\|dec\|strided\|sum\|ax0 | 0.2754 | 0.0344 | 0.12 🔴 |
| 100K\|i32\|strided\|sum\|ax0 | 0.2021 | 0.0283 | 0.14 🔴 |
| 100K\|dec\|C\|sum\|ax1 | 0.2142 | 0.0309 | 0.14 🔴 |
| 100K\|dec\|sliced\|sum\|ax1 | 0.2115 | 0.0312 | 0.15 🔴 |
| 100K\|dec\|T\|sum\|ax0 | 0.2095 | 0.0314 | 0.15 🔴 |
| 1M\|dec\|F\|max\|ax1 | 2.2227 | 0.3422 | 0.15 🔴 |
| 1M\|dec\|C\|max\|ax0 | 2.2539 | 0.3481 | 0.15 🔴 |
| 1M\|dec\|sliced\|max\|ax0 | 2.2260 | 0.3450 | 0.15 🔴 |
| 100K\|f64\|sliced\|min\|ax1 | 0.0957 | 0.0156 | 0.16 🔴 |
| 100K\|f32\|sliced\|max\|ax1 | 0.0655 | 0.0107 | 0.16 🔴 |

## Best 12 cells (NumSharp fastest vs NumPy)

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|f32\|F\|prod\|ax0 | 0.0428 | 0.9469 | 22.14 ✅ |
| 1M\|f32\|C\|prod\|ax1 | 0.0434 | 0.9392 | 21.62 ✅ |
| 1M\|f32\|T\|prod\|ax0 | 0.0451 | 0.9358 | 20.77 ✅ |
| 100K\|i32\|F\|sum\|ax0 | 0.0052 | 0.0928 | 17.90 ✅ |
| 100K\|i32\|T\|sum\|ax0 | 0.0054 | 0.0951 | 17.55 ✅ |
| 100K\|f32\|F\|prod\|ax0 | 0.0034 | 0.0582 | 17.27 ✅ |
| 100K\|i32\|C\|sum\|ax1 | 0.0052 | 0.0898 | 17.18 ✅ |
| 1M\|i32\|negstride\|sum\|ax0 | 0.0978 | 1.6724 | 17.10 ✅ |
| 100K\|f32\|C\|prod\|ax1 | 0.0036 | 0.0582 | 16.34 ✅ |
| 100K\|f32\|T\|prod\|ax0 | 0.0036 | 0.0584 | 16.25 ✅ |
| 1M\|i32\|C\|sum\|ax0 | 0.0933 | 1.4989 | 16.07 ✅ |
| 100K\|i32\|negstride\|sum\|ax0 | 0.0095 | 0.1418 | 14.87 ✅ |

_648 cells compared._
