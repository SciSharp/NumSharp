# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.02 ✅ | 1.01 ✅ | 1.03 ✅ | 0.59 🟡 | 0.89 🟡 | 0.67 🟡 | 0.83 🟡 | 0.78 🟡 |
| 1M | 0.72 🟡 | 0.83 🟡 | 0.81 🟡 | 0.51 🟡 | 0.90 🟡 | 0.67 🟡 | 0.75 🟡 | 0.74 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.92 🟡 | 1.13 ✅ | 0.96 🟡 | 0.13 🔴 | 1.08 ✅ | 1.36 ✅ | 0.97 🟡 |
| 1M | 0.63 🟡 | 1.04 ✅ | 0.73 🟡 | 0.11 🔴 | 1.03 ✅ | 1.25 ✅ | 1.02 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.83 🟡 | 0.67 🟡 | 0.70 🟡 | 1.35 ✅ |
| 1M | 0.80 🟡 | 0.60 🟡 | 0.59 🟡 | 1.09 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.9751 | 0.1504 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7699 | 0.0219 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7752 | 0.0245 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 8.0090 | 0.2607 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7849 | 0.0256 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7657 | 0.0251 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7843 | 0.0261 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.9411 | 0.2661 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.7552 | 0.2717 | 0.04 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7654 | 0.0268 | 0.04 🔴 |
| 1M\|c128\|C\|prod\|ax0 | 26.1258 | 0.9165 | 0.04 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.9508 | 0.2983 | 0.04 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.9802 | 0.3039 | 0.04 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.3544 | 0.1812 | 0.04 🔴 |
| 100K\|dec\|bcast\|sum\|ax1 | 0.4394 | 0.0254 | 0.06 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.67 🟡 | 0.60 🟡 | 0.61 🟡 | 0.42 🟠 | 0.83 🟡 | 0.88 🟡 | 1.04 ✅ | 0.78 🟡 |
| 1M | 2.66 ✅ | 2.64 ✅ | 2.53 ✅ | 1.70 ✅ | 1.96 ✅ | 1.90 ✅ | 2.50 ✅ | 2.14 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.96 🟡 | 1.05 ✅ | 1.10 ✅ | 1.08 ✅ | 0.49 🟠 | 0.39 🟠 | 0.63 🟡 | 0.40 🟠 | 0.83 🟡 | 0.77 🟡 | 0.38 🟠 | 0.29 🟠 | 2.53 ✅ |
| 1M | 7.08 ✅ | 5.27 ✅ | 2.36 ✅ | 1.67 ✅ | 1.50 ✅ | 1.77 ✅ | 1.75 ✅ | 1.87 ✅ | 2.07 ✅ | 2.06 ✅ | 1.83 ✅ | 1.75 ✅ | 1.89 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|F\|pos | 0.1655 | 0.0251 | 0.15 🔴 |
| 100K\|f64\|strided\|pos | 0.0767 | 0.0117 | 0.15 🔴 |
| 100K\|i64\|strided\|pos | 0.0755 | 0.0117 | 0.15 🔴 |
| 100K\|f64\|T\|pos | 0.1474 | 0.0247 | 0.17 🔴 |
| 100K\|u64\|strided\|pos | 0.0711 | 0.0128 | 0.18 🔴 |
| 100K\|f32\|strided\|pos | 0.0585 | 0.0113 | 0.19 🔴 |
| 100K\|i32\|strided\|pos | 0.0600 | 0.0136 | 0.23 🟠 |
| 100K\|u32\|strided\|pos | 0.0503 | 0.0126 | 0.25 🟠 |
| 100K\|f64\|C\|pos | 0.0962 | 0.0262 | 0.27 🟠 |
| 100K\|f32\|T\|pos | 0.0709 | 0.0211 | 0.30 🟠 |
| 100K\|u32\|F\|pos | 0.0732 | 0.0233 | 0.32 🟠 |
| 100K\|u64\|F\|pos | 0.1068 | 0.0343 | 0.32 🟠 |
| 100K\|u64\|T\|pos | 0.0952 | 0.0313 | 0.33 🟠 |
| 100K\|u64\|C\|pos | 0.0963 | 0.0339 | 0.35 🟠 |
| 100K\|u32\|bcast\|pos | 0.0760 | 0.0272 | 0.36 🟠 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.63 🟡 | 0.56 🟡 | 0.58 🟡 | 0.37 🟠 | 0.76 🟡 | 0.75 🟡 | 0.82 🟡 | 0.73 🟡 |
| 1M | 1.33 ✅ | 1.30 ✅ | 1.26 ✅ | 0.80 🟡 | 1.38 ✅ | 1.40 ✅ | 1.14 ✅ | 1.51 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.62 🟡 | 0.24 🟠 | 1.23 ✅ | 0.74 🟡 | 0.78 🟡 | 0.61 🟡 |
| 1M | 1.34 ✅ | 1.36 ✅ | 1.36 ✅ | 0.93 🟡 | 1.34 ✅ | 1.21 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.74 🟡 | 0.76 🟡 | 0.58 🟡 | 0.63 🟡 | 0.70 🟡 | 0.60 🟡 | 0.47 🟠 |
| 1M | 1.29 ✅ | 1.31 ✅ | 1.52 ✅ | 1.45 ✅ | 1.09 ✅ | 0.79 🟡 | 1.47 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|abs | 0.0688 | 0.0061 | 0.09 🔴 |
| 100K\|f32\|strided\|neg | 0.0705 | 0.0063 | 0.09 🔴 |
| 100K\|i64\|strided\|add | 0.2354 | 0.0258 | 0.11 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0713 | 0.0079 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1190 | 0.0136 | 0.11 🔴 |
| 100K\|f32\|strided\|add | 0.1156 | 0.0139 | 0.12 🔴 |
| 100K\|i64\|strided\|mul | 0.2403 | 0.0312 | 0.13 🔴 |
| 100K\|f32\|F\|sqrt | 0.1038 | 0.0142 | 0.14 🔴 |
| 100K\|f32\|T\|sqrt | 0.0985 | 0.0143 | 0.14 🔴 |
| 100K\|f32\|bcast\|copy | 0.0342 | 0.0050 | 0.15 🔴 |
| 100K\|f64\|strided\|abs | 0.0519 | 0.0078 | 0.15 🔴 |
| 100K\|f32\|T\|abs | 0.0363 | 0.0057 | 0.16 🔴 |
| 100K\|f32\|C\|abs | 0.0354 | 0.0057 | 0.16 🔴 |
| 100K\|i32\|F\|neg | 0.0398 | 0.0065 | 0.16 🔴 |
| 100K\|f32\|C\|add | 0.0389 | 0.0067 | 0.17 🔴 |
