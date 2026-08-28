# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.02 ✅ | 1.16 ✅ | 1.10 ✅ | 0.65 🟡 | 1.05 ✅ | 0.77 🟡 | 0.92 🟡 | 0.84 🟡 |
| 1M | 1.03 ✅ | 1.03 ✅ | 1.12 ✅ | 0.58 🟡 | 0.94 🟡 | 0.70 🟡 | 0.84 🟡 | 0.69 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.08 ✅ | 1.30 ✅ | 0.98 🟡 | 0.13 🔴 | 1.08 ✅ | 1.46 ✅ | 1.25 ✅ |
| 1M | 1.01 ✅ | 1.24 ✅ | 0.96 🟡 | 0.11 🔴 | 1.05 ✅ | 1.30 ✅ | 1.07 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.99 🟡 | 0.76 🟡 | 0.78 🟡 | 1.31 ✅ |
| 1M | 0.93 🟡 | 0.65 🟡 | 0.70 🟡 | 1.30 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.3704 | 0.0627 | 0.01 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7278 | 0.0205 | 0.03 🔴 |
| 1M\|dec\|bcast\|min\|ax0 | 3.1558 | 0.0892 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.5109 | 0.2257 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.5254 | 0.2269 | 0.03 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.4183 | 0.2266 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.7378 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.7283 | 0.0227 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7281 | 0.0228 | 0.03 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.4767 | 0.2384 | 0.03 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.3708 | 0.2435 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7189 | 0.0248 | 0.03 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.1325 | 0.1709 | 0.04 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.5507 | 0.0229 | 0.04 🔴 |
| 100K\|dec\|C\|max\|ax0 | 0.3266 | 0.0147 | 0.05 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 3.79 ✅ | 3.09 ✅ | 3.14 ✅ | 1.50 ✅ | 3.83 ✅ | 3.75 ✅ | 3.40 ✅ | 4.01 ✅ |
| 1M | 8.07 ✅ | 7.88 ✅ | 8.29 ✅ | 3.88 ✅ | 6.44 ✅ | 6.31 ✅ | 7.22 ✅ | 9.62 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 5.21 ✅ | 4.10 ✅ | 4.91 ✅ | 4.51 ✅ | 2.75 ✅ | 2.73 ✅ | 1.91 ✅ | 1.48 ✅ | 4.92 ✅ | 4.32 ✅ | 1.39 ✅ | 1.16 ✅ | 10.38 ✅ |
| 1M | 9.96 ✅ | 10.11 ✅ | 9.29 ✅ | 9.12 ✅ | 5.52 ✅ | 5.67 ✅ | 4.86 ✅ | 5.15 ✅ | 9.15 ✅ | 8.98 ✅ | 5.12 ✅ | 5.10 ✅ | 6.74 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|pos | 0.0287 | 0.0158 | 0.55 🟡 |
| 100K\|f64\|strided\|pos | 0.0317 | 0.0181 | 0.57 🟡 |
| 100K\|i32\|strided\|pos | 0.0288 | 0.0199 | 0.69 🟡 |
| 100K\|u32\|strided\|pos | 0.0288 | 0.0199 | 0.69 🟡 |
| 100K\|f64\|T\|pos | 0.0302 | 0.0278 | 0.92 🟡 |
| 100K\|i64\|strided\|pos | 0.0318 | 0.0299 | 0.94 🟡 |
| 100K\|u64\|strided\|pos | 0.0314 | 0.0300 | 0.96 🟡 |
| 100K\|f64\|C\|pos | 0.0291 | 0.0278 | 0.96 🟡 |
| 100K\|f64\|F\|pos | 0.0286 | 0.0278 | 0.97 🟡 |
| 100K\|u64\|T\|pos | 0.0334 | 0.0365 | 1.09 ✅ |
| 100K\|u64\|C\|pos | 0.0318 | 0.0365 | 1.15 ✅ |
| 100K\|u64\|F\|pos | 0.0304 | 0.0365 | 1.20 ✅ |
| 100K\|f32\|F\|pos | 0.0157 | 0.0189 | 1.20 ✅ |
| 100K\|i64\|F\|pos | 0.0292 | 0.0366 | 1.25 ✅ |
| 100K\|f32\|negcol\|pos | 0.0538 | 0.0734 | 1.36 ✅ |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.46 ✅ | 1.24 ✅ | 0.56 🟡 | 1.65 ✅ | 1.48 ✅ | 1.14 ✅ | 1.50 ✅ |
| 1M | 2.47 ✅ | 2.22 ✅ | 2.24 ✅ | 1.25 ✅ | 2.61 ✅ | 2.59 ✅ | 1.65 ✅ | 3.04 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 0.92 🟡 | 2.92 ✅ | 0.88 🟡 | 1.28 ✅ | 0.95 🟡 |
| 1M | 2.39 ✅ | 2.82 ✅ | 2.17 ✅ | 1.24 ✅ | 2.57 ✅ | 2.35 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 1.59 ✅ | 1.42 ✅ | 1.27 ✅ | 1.16 ✅ | 0.98 🟡 | 0.89 🟡 | 1.26 ✅ |
| 1M | 2.43 ✅ | 2.41 ✅ | 3.27 ✅ | 2.80 ✅ | 1.45 ✅ | 0.85 🟡 | 3.62 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|add | 0.1834 | 0.0257 | 0.14 🔴 |
| 100K\|i64\|strided\|mul | 0.1832 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|mul | 0.1832 | 0.0301 | 0.16 🔴 |
| 100K\|f64\|strided\|neg | 0.0938 | 0.0166 | 0.18 🔴 |
| 100K\|f64\|strided\|abs | 0.0931 | 0.0168 | 0.18 🔴 |
| 100K\|f32\|strided\|abs | 0.0492 | 0.0099 | 0.20 🟠 |
| 100K\|i32\|strided\|sqrt | 0.1827 | 0.0419 | 0.23 🟠 |
| 100K\|i64\|strided\|sqrt | 0.1759 | 0.0408 | 0.23 🟠 |
| 100K\|i64\|C\|sqrt | 0.2797 | 0.0744 | 0.27 🟠 |
| 100K\|i32\|strided\|add | 0.0932 | 0.0279 | 0.30 🟠 |
| 100K\|i32\|strided\|mul | 0.0931 | 0.0279 | 0.30 🟠 |
| 100K\|f32\|strided\|add | 0.0930 | 0.0286 | 0.31 🟠 |
| 100K\|f32\|strided\|mul | 0.0931 | 0.0287 | 0.31 🟠 |
| 100K\|i32\|strided\|less | 0.0486 | 0.0168 | 0.35 🟠 |
| 100K\|c128\|C\|less | 0.1901 | 0.0685 | 0.36 🟠 |
