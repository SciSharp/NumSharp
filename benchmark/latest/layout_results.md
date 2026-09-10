# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.10 ✅ | 1.11 ✅ | 1.10 ✅ | 0.60 🟡 | 1.03 ✅ | 0.73 🟡 | 0.91 🟡 | 0.83 🟡 |
| 1M | 1.05 ✅ | 1.01 ✅ | 0.98 🟡 | 0.63 🟡 | 1.02 ✅ | 0.72 🟡 | 0.81 🟡 | 0.75 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.11 ✅ | 1.30 ✅ | 0.97 🟡 | 0.12 🔴 | 1.05 ✅ | 1.50 ✅ | 1.14 ✅ |
| 1M | 1.02 ✅ | 1.28 ✅ | 0.96 🟡 | 0.11 🔴 | 1.07 ✅ | 1.31 ✅ | 1.07 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.95 🟡 | 0.74 🟡 | 0.76 🟡 | 1.34 ✅ |
| 1M | 0.95 🟡 | 0.67 🟡 | 0.68 🟡 | 1.32 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.3695 | 0.1408 | 0.02 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.1086 | 0.1125 | 0.03 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7283 | 0.0204 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.5294 | 0.2240 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.5579 | 0.2255 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7326 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7286 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7284 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.7282 | 0.0230 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.4363 | 0.2417 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.2451 | 0.2436 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7190 | 0.0248 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.5333 | 0.2184 | 0.04 🔴 |
| 100K\|dec\|negrow\|sum\|ax1 | 0.4159 | 0.0184 | 0.04 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 1.1684 | 0.0603 | 0.05 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 3.78 ✅ | 2.96 ✅ | 3.47 ✅ | 1.55 ✅ | 3.76 ✅ | 3.95 ✅ | 3.58 ✅ | 4.26 ✅ |
| 1M | 6.83 ✅ | 6.29 ✅ | 6.51 ✅ | 3.32 ✅ | 5.41 ✅ | 6.04 ✅ | 6.47 ✅ | 9.24 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 5.43 ✅ | 4.13 ✅ | 4.31 ✅ | 4.58 ✅ | 2.80 ✅ | 2.76 ✅ | 1.87 ✅ | 1.51 ✅ | 4.72 ✅ | 4.89 ✅ | 1.73 ✅ | 1.28 ✅ | 10.68 ✅ |
| 1M | 10.08 ✅ | 10.08 ✅ | 8.28 ✅ | 8.35 ✅ | 4.46 ✅ | 4.98 ✅ | 3.65 ✅ | 4.39 ✅ | 7.96 ✅ | 7.65 ✅ | 4.17 ✅ | 3.93 ✅ | 5.93 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|pos | 0.0285 | 0.0159 | 0.56 🟡 |
| 100K\|f64\|strided\|pos | 0.0312 | 0.0180 | 0.58 🟡 |
| 100K\|i32\|strided\|pos | 0.0286 | 0.0197 | 0.69 🟡 |
| 100K\|u32\|strided\|pos | 0.0285 | 0.0197 | 0.69 🟡 |
| 100K\|f64\|T\|pos | 0.0310 | 0.0277 | 0.89 🟡 |
| 100K\|i64\|strided\|pos | 0.0317 | 0.0299 | 0.94 🟡 |
| 100K\|u64\|strided\|pos | 0.0312 | 0.0298 | 0.96 🟡 |
| 100K\|f64\|C\|pos | 0.0281 | 0.0277 | 0.98 🟡 |
| 100K\|i64\|F\|pos | 0.0370 | 0.0364 | 0.98 🟡 |
| 100K\|f64\|F\|pos | 0.0305 | 0.0329 | 1.08 ✅ |
| 100K\|u64\|F\|pos | 0.0317 | 0.0364 | 1.15 ✅ |
| 100K\|u64\|T\|pos | 0.0310 | 0.0365 | 1.18 ✅ |
| 100K\|u64\|C\|pos | 0.0284 | 0.0364 | 1.28 ✅ |
| 1M\|i32\|strided\|pos | 0.2711 | 0.3486 | 1.29 ✅ |
| 100K\|i8\|strided\|pos | 0.0089 | 0.0117 | 1.31 ✅ |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.26 ✅ | 1.19 ✅ | 1.15 ✅ | 0.54 🟡 | 1.59 ✅ | 1.55 ✅ | 1.13 ✅ | 1.46 ✅ |
| 1M | 2.15 ✅ | 1.96 ✅ | 1.97 ✅ | 0.99 🟡 | 2.38 ✅ | 2.40 ✅ | 1.49 ✅ | 2.89 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 1.07 ✅ | 0.86 🟡 | 2.61 ✅ | 0.91 🟡 | 1.27 ✅ | 0.98 🟡 |
| 1M | 2.14 ✅ | 2.34 ✅ | 1.93 ✅ | 1.12 ✅ | 2.40 ✅ | 2.05 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 1.44 ✅ | 1.39 ✅ | 1.30 ✅ | 1.20 ✅ | 0.93 🟡 | 0.87 🟡 | 1.26 ✅ |
| 1M | 2.18 ✅ | 2.17 ✅ | 2.80 ✅ | 2.44 ✅ | 1.27 ✅ | 0.88 🟡 | 2.88 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|add | 0.1828 | 0.0256 | 0.14 🔴 |
| 100K\|f64\|strided\|add | 0.1900 | 0.0287 | 0.15 🔴 |
| 100K\|f64\|strided\|mul | 0.1833 | 0.0301 | 0.16 🔴 |
| 100K\|i64\|strided\|mul | 0.1831 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|neg | 0.0943 | 0.0163 | 0.17 🔴 |
| 100K\|f64\|strided\|abs | 0.0931 | 0.0165 | 0.18 🔴 |
| 100K\|f64\|T\|sqrt | 0.2797 | 0.0552 | 0.20 🔴 |
| 100K\|f32\|strided\|abs | 0.0492 | 0.0098 | 0.20 🟠 |
| 100K\|i64\|strided\|sqrt | 0.1752 | 0.0408 | 0.23 🟠 |
| 100K\|i32\|bcast\|sqrt | 0.3200 | 0.0804 | 0.25 🟠 |
| 100K\|f32\|negcol\|add | 0.1845 | 0.0494 | 0.27 🟠 |
| 100K\|i32\|F\|sqrt | 0.2797 | 0.0778 | 0.28 🟠 |
| 100K\|i32\|strided\|add | 0.0932 | 0.0279 | 0.30 🟠 |
| 100K\|i32\|strided\|mul | 0.0930 | 0.0279 | 0.30 🟠 |
| 100K\|f32\|strided\|add | 0.0933 | 0.0286 | 0.31 🟠 |
