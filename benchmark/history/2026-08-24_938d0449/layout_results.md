# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.06 ✅ | 1.04 ✅ | 0.59 🟡 | 0.91 🟡 | 0.70 🟡 | 0.88 🟡 | 0.81 🟡 |
| 1M | 1.01 ✅ | 1.09 ✅ | 1.15 ✅ | 0.60 🟡 | 0.94 🟡 | 0.72 🟡 | 0.82 🟡 | 0.75 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.97 🟡 | 1.17 ✅ | 0.91 🟡 | 0.12 🔴 | 1.04 ✅ | 1.48 ✅ | 1.13 ✅ |
| 1M | 1.05 ✅ | 1.24 ✅ | 0.95 🟡 | 0.11 🔴 | 1.05 ✅ | 1.47 ✅ | 1.01 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.88 🟡 | 0.70 🟡 | 0.71 🟡 | 1.36 ✅ |
| 1M | 0.95 🟡 | 0.67 🟡 | 0.69 🟡 | 1.36 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.6021 | 0.1420 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7671 | 0.0207 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.6152 | 0.2246 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7698 | 0.0230 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7638 | 0.0233 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7652 | 0.0233 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7615 | 0.0233 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.7864 | 0.2428 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.6415 | 0.2418 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.6214 | 0.2428 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7507 | 0.0254 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.5704 | 0.2672 | 0.04 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.2179 | 0.1732 | 0.04 🔴 |
| 100K\|dec\|bcast\|sum\|ax1 | 0.4289 | 0.0240 | 0.06 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 0.8931 | 0.0607 | 0.07 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.18 ✅ | 1.20 ✅ | 1.22 ✅ | 0.70 🟡 | 1.44 ✅ | 1.42 ✅ | 1.79 ✅ | 1.58 ✅ |
| 1M | 3.49 ✅ | 3.47 ✅ | 3.49 ✅ | 2.44 ✅ | 3.25 ✅ | 3.25 ✅ | 3.81 ✅ | 3.46 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 1.00 ✅ | 1.42 ✅ | 1.80 ✅ | 1.99 ✅ | 0.90 🟡 | 1.11 ✅ | 1.59 ✅ | 1.11 ✅ | 1.89 ✅ | 1.52 ✅ | 0.75 🟡 | 0.62 🟡 | 1.96 ✅ |
| 1M | 5.42 ✅ | 5.46 ✅ | 2.54 ✅ | 2.57 ✅ | 2.68 ✅ | 2.68 ✅ | 3.83 ✅ | 3.69 ✅ | 2.52 ✅ | 2.39 ✅ | 2.47 ✅ | 3.81 ✅ | 5.13 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|pos | 0.0433 | 0.0100 | 0.23 🟠 |
| 100K\|f32\|strided\|pos | 0.0379 | 0.0103 | 0.27 🟠 |
| 100K\|u64\|strided\|pos | 0.0259 | 0.0109 | 0.42 🟠 |
| 100K\|i32\|strided\|pos | 0.0237 | 0.0109 | 0.46 🟠 |
| 100K\|f64\|C\|pos | 0.0506 | 0.0252 | 0.50 🟠 |
| 100K\|f32\|negcol\|pos | 0.0774 | 0.0405 | 0.52 🟡 |
| 100K\|u8\|negrow\|pos | 0.0439 | 0.0232 | 0.53 🟡 |
| 100K\|f64\|T\|pos | 0.0448 | 0.0237 | 0.53 🟡 |
| 100K\|i64\|strided\|pos | 0.0205 | 0.0110 | 0.54 🟡 |
| 100K\|u8\|sliced\|pos | 0.0429 | 0.0230 | 0.54 🟡 |
| 100K\|f64\|F\|pos | 0.0447 | 0.0271 | 0.61 🟡 |
| 100K\|u32\|strided\|pos | 0.0178 | 0.0110 | 0.62 🟡 |
| 100K\|f64\|negcol\|pos | 0.0833 | 0.0638 | 0.77 🟡 |
| 100K\|f32\|C\|pos | 0.0252 | 0.0196 | 0.77 🟡 |
| 100K\|u64\|C\|pos | 0.0357 | 0.0290 | 0.81 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.62 🟡 | 0.71 🟡 | 0.66 🟡 | 0.37 🟠 | 0.85 🟡 | 0.83 🟡 | 0.84 🟡 | 0.81 🟡 |
| 1M | 1.77 ✅ | 1.69 ✅ | 1.68 ✅ | 0.97 🟡 | 1.83 ✅ | 1.87 ✅ | 1.41 ✅ | 2.05 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.66 🟡 | 0.29 🟠 | 1.42 ✅ | 0.75 🟡 | 0.78 🟡 | 0.68 🟡 |
| 1M | 1.98 ✅ | 1.65 ✅ | 2.02 ✅ | 0.97 🟡 | 1.63 ✅ | 1.75 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.81 🟡 | 0.84 🟡 | 0.64 🟡 | 0.70 🟡 | 0.74 🟡 | 0.61 🟡 | 0.53 🟡 |
| 1M | 1.83 ✅ | 1.82 ✅ | 2.21 ✅ | 1.96 ✅ | 1.33 ✅ | 0.74 🟡 | 2.10 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|neg | 0.0670 | 0.0060 | 0.09 🔴 |
| 100K\|f32\|strided\|abs | 0.0658 | 0.0059 | 0.09 🔴 |
| 100K\|i64\|strided\|add | 0.2312 | 0.0258 | 0.11 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0672 | 0.0077 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1142 | 0.0136 | 0.12 🔴 |
| 100K\|f32\|strided\|add | 0.1108 | 0.0134 | 0.12 🔴 |
| 100K\|i64\|strided\|mul | 0.2314 | 0.0302 | 0.13 🔴 |
| 100K\|f64\|strided\|neg | 0.0559 | 0.0076 | 0.14 🔴 |
| 100K\|f64\|strided\|abs | 0.0545 | 0.0076 | 0.14 🔴 |
| 100K\|f32\|bcast\|copy | 0.0326 | 0.0050 | 0.15 🔴 |
| 100K\|f32\|T\|sqrt | 0.0845 | 0.0142 | 0.17 🔴 |
| 100K\|f32\|C\|neg | 0.0358 | 0.0065 | 0.18 🔴 |
| 100K\|f32\|bcast\|sqrt | 0.1033 | 0.0188 | 0.18 🔴 |
| 100K\|i32\|C\|copy | 0.0313 | 0.0059 | 0.19 🔴 |
| 100K\|f32\|sliced\|sqrt | 0.1016 | 0.0203 | 0.20 🟠 |
