# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.11 ✅ | 1.19 ✅ | 1.02 ✅ | 0.54 🟡 | 0.78 🟡 | 0.70 🟡 | 0.75 🟡 | 0.73 🟡 |
| 1M | 0.45 🟠 | 0.45 🟠 | 0.44 🟠 | 0.29 🟠 | 0.40 🟠 | 0.34 🟠 | 0.35 🟠 | 0.33 🟠 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 1.08 ✅ | 1.16 ✅ | 0.97 🟡 | 0.13 🔴 | 1.11 ✅ | 1.46 ✅ | 0.65 🟡 |
| 1M | 0.40 🟠 | 0.50 🟡 | 0.58 🟡 | 0.04 🔴 | 0.61 🟡 | 0.54 🟡 | 0.40 🟠 |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.89 🟡 | 0.67 🟡 | 0.67 🟡 | 1.23 ✅ |
| 1M | 0.46 🟠 | 0.29 🟠 | 0.29 🟠 | 0.58 🟡 |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 7.4497 | 0.0696 | 0.01 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 7.7505 | 0.1179 | 0.02 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 7.8700 | 0.1202 | 0.02 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 7.9562 | 0.1221 | 0.02 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 7.8166 | 0.1307 | 0.02 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 7.7814 | 0.1306 | 0.02 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.7930 | 0.0222 | 0.03 🔴 |
| 1M\|dec\|bcast\|max\|ax0 | 3.3884 | 0.0950 | 0.03 🔴 |
| 1M\|dec\|bcast\|sum\|ax1 | 4.2607 | 0.1195 | 0.03 🔴 |
| 1M\|dec\|bcast\|min\|ax0 | 3.4004 | 0.0992 | 0.03 🔴 |
| 1M\|i32\|bcast\|max\|ax1 | 0.9662 | 0.0298 | 0.03 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.8056 | 0.0253 | 0.03 🔴 |
| 1M\|i32\|bcast\|min\|ax1 | 0.8975 | 0.0301 | 0.03 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.7858 | 0.0264 | 0.03 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.7817 | 0.0269 | 0.03 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.32 ✅ | 1.25 ✅ | 1.29 ✅ | 0.77 🟡 | 1.45 ✅ | 1.46 ✅ | 2.02 ✅ | 1.43 ✅ |
| 1M | 2.87 ✅ | 2.89 ✅ | 2.91 ✅ | 1.90 ✅ | 2.51 ✅ | 2.49 ✅ | 3.13 ✅ | 2.57 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.92 🟡 | 1.16 ✅ | 1.52 ✅ | 1.88 ✅ | 0.98 🟡 | 1.07 ✅ | 1.64 ✅ | 0.95 🟡 | 2.56 ✅ | 2.42 ✅ | 0.94 🟡 | 0.60 🟡 | 2.40 ✅ |
| 1M | 4.31 ✅ | 4.24 ✅ | 2.18 ✅ | 2.12 ✅ | 2.09 ✅ | 2.06 ✅ | 2.50 ✅ | 2.56 ✅ | 2.05 ✅ | 2.08 ✅ | 2.12 ✅ | 2.46 ✅ | 5.62 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|pos | 0.0442 | 0.0112 | 0.25 🟠 |
| 100K\|f64\|T\|pos | 0.0654 | 0.0277 | 0.42 🟠 |
| 100K\|u64\|strided\|pos | 0.0255 | 0.0111 | 0.43 🟠 |
| 100K\|i64\|strided\|pos | 0.0251 | 0.0110 | 0.44 🟠 |
| 100K\|f64\|C\|pos | 0.0576 | 0.0260 | 0.45 🟠 |
| 100K\|f32\|strided\|pos | 0.0208 | 0.0103 | 0.50 🟠 |
| 100K\|i32\|strided\|pos | 0.0214 | 0.0109 | 0.51 🟡 |
| 100K\|f64\|F\|pos | 0.0543 | 0.0278 | 0.51 🟡 |
| 100K\|u8\|sliced\|pos | 0.0419 | 0.0227 | 0.54 🟡 |
| 100K\|u8\|negrow\|pos | 0.0421 | 0.0229 | 0.54 🟡 |
| 100K\|u32\|strided\|pos | 0.0196 | 0.0110 | 0.56 🟡 |
| 100K\|u64\|F\|pos | 0.0436 | 0.0297 | 0.68 🟡 |
| 100K\|u64\|C\|pos | 0.0391 | 0.0289 | 0.74 🟡 |
| 100K\|i8\|strided\|pos | 0.0165 | 0.0125 | 0.76 🟡 |
| 100K\|f64\|sliced\|pos | 0.0630 | 0.0489 | 0.78 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.61 🟡 | 0.62 🟡 | 0.63 🟡 | 0.49 🟠 | 0.80 🟡 | 0.78 🟡 | 1.10 ✅ | 0.75 🟡 |
| 1M | 1.51 ✅ | 1.47 ✅ | 1.45 ✅ | 1.10 ✅ | 1.57 ✅ | 1.54 ✅ | 1.64 ✅ | 1.66 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.61 🟡 | 0.50 🟠 | 1.03 ✅ | 0.73 🟡 | 0.78 🟡 | 0.68 🟡 |
| 1M | 1.75 ✅ | 1.43 ✅ | 1.58 ✅ | 0.92 🟡 | 1.58 ✅ | 1.84 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.93 🟡 | 0.88 🟡 | 0.66 🟡 | 0.66 🟡 | 0.87 🟡 | 0.57 🟡 | 0.48 🟠 |
| 1M | 1.79 ✅ | 1.71 ✅ | 1.94 ✅ | 1.61 ✅ | 1.45 ✅ | 0.67 🟡 | 1.71 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|abs | 0.0608 | 0.0077 | 0.13 🔴 |
| 100K\|f64\|strided\|neg | 0.0609 | 0.0091 | 0.15 🔴 |
| 100K\|f32\|C\|abs | 0.0380 | 0.0060 | 0.16 🔴 |
| 100K\|i32\|bcast\|copy | 0.0242 | 0.0050 | 0.21 🟠 |
| 100K\|f32\|C\|copy | 0.0283 | 0.0059 | 0.21 🟠 |
| 100K\|f64\|strided\|copy | 0.0502 | 0.0105 | 0.21 🟠 |
| 100K\|f32\|C\|mul | 0.0312 | 0.0068 | 0.22 🟠 |
| 100K\|f16\|C\|copy | 0.0135 | 0.0031 | 0.23 🟠 |
| 100K\|i64\|strided\|neg | 0.0328 | 0.0076 | 0.23 🟠 |
| 100K\|f16\|negrow\|copy | 0.0168 | 0.0039 | 0.23 🟠 |
| 100K\|f64\|strided\|mul | 0.0601 | 0.0141 | 0.23 🟠 |
| 100K\|f64\|strided\|add | 0.0616 | 0.0146 | 0.24 🟠 |
| 100K\|f32\|F\|mul | 0.0279 | 0.0068 | 0.24 🟠 |
| 100K\|f32\|bcast\|copy | 0.0201 | 0.0050 | 0.25 🟠 |
| 100K\|f64\|C\|less | 0.0290 | 0.0073 | 0.25 🟠 |
