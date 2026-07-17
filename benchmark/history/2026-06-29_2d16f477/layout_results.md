# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.99 🟡 | 1.02 ✅ | 1.02 ✅ | 0.55 🟡 | 0.70 🟡 | 0.70 🟡 | 0.72 🟡 | 0.74 🟡 |
| 1M | 0.95 🟡 | 0.94 🟡 | 0.95 🟡 | 0.65 🟡 | 0.87 🟡 | 0.73 🟡 | 0.75 🟡 | 0.68 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.87 🟡 | 1.00 🟡 | 0.98 🟡 | 0.08 🔴 | 1.03 ✅ | 1.33 ✅ | 1.14 ✅ |
| 1M | 1.12 ✅ | 1.15 ✅ | 1.04 ✅ | 0.07 🔴 | 1.02 ✅ | 1.19 ✅ | 1.02 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.80 🟡 | 0.66 🟡 | 0.66 🟡 | 1.19 ✅ |
| 1M | 0.84 🟡 | 0.65 🟡 | 0.64 🟡 | 1.28 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 1M\|dec\|bcast\|sum\|ax0 | 5.5269 | 0.0655 | 0.01 🔴 |
| 100K\|dec\|bcast\|sum\|ax0 | 0.5555 | 0.0087 | 0.02 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.5585 | 0.0094 | 0.02 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.5615 | 0.0111 | 0.02 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.5582 | 0.0111 | 0.02 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.5592 | 0.0114 | 0.02 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.5517 | 0.0115 | 0.02 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 5.5802 | 0.1240 | 0.02 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 5.5707 | 0.1242 | 0.02 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 5.6157 | 0.1254 | 0.02 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 5.5849 | 0.1278 | 0.02 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.5736 | 0.1524 | 0.03 🔴 |
| 100K\|dec\|C\|min\|ax0 | 0.3021 | 0.0141 | 0.05 🔴 |
| 1M\|dec\|negcol\|sum\|ax0 | 5.6678 | 0.2866 | 0.05 🔴 |
| 100K\|dec\|negcol\|sum\|ax0 | 0.5594 | 0.0302 | 0.05 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.22 ✅ | 1.51 ✅ | 1.28 ✅ | 0.83 🟡 | 1.54 ✅ | 1.48 ✅ | 2.20 ✅ | 1.62 ✅ |
| 1M | 2.82 ✅ | 2.89 ✅ | 2.88 ✅ | 2.08 ✅ | 2.58 ✅ | 2.64 ✅ | 3.16 ✅ | 2.66 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.92 🟡 | 1.03 ✅ | 1.51 ✅ | 1.99 ✅ | 1.02 ✅ | 1.09 ✅ | 1.65 ✅ | 1.08 ✅ | 2.56 ✅ | 2.47 ✅ | 1.15 ✅ | 0.85 🟡 | 2.56 ✅ |
| 1M | 4.60 ✅ | 4.87 ✅ | 2.13 ✅ | 2.14 ✅ | 2.11 ✅ | 2.12 ✅ | 2.56 ✅ | 2.46 ✅ | 2.20 ✅ | 2.10 ✅ | 2.14 ✅ | 2.59 ✅ | 5.47 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|pos | 0.0243 | 0.0100 | 0.41 🟠 |
| 100K\|u64\|strided\|pos | 0.0241 | 0.0110 | 0.46 🟠 |
| 100K\|i64\|strided\|pos | 0.0237 | 0.0109 | 0.46 🟠 |
| 100K\|f32\|strided\|pos | 0.0216 | 0.0100 | 0.46 🟠 |
| 100K\|i32\|strided\|pos | 0.0213 | 0.0110 | 0.52 🟡 |
| 100K\|i64\|C\|pos | 0.0400 | 0.0210 | 0.53 🟡 |
| 100K\|i64\|T\|pos | 0.0387 | 0.0208 | 0.54 🟡 |
| 100K\|i8\|sliced\|pos | 0.0423 | 0.0228 | 0.54 🟡 |
| 100K\|u8\|bcast\|pos | 0.0416 | 0.0229 | 0.55 🟡 |
| 100K\|u8\|sliced\|pos | 0.0410 | 0.0227 | 0.55 🟡 |
| 100K\|u8\|negrow\|pos | 0.0410 | 0.0231 | 0.56 🟡 |
| 100K\|u32\|strided\|pos | 0.0191 | 0.0109 | 0.57 🟡 |
| 100K\|f64\|T\|pos | 0.0389 | 0.0227 | 0.58 🟡 |
| 100K\|f64\|F\|pos | 0.0362 | 0.0227 | 0.63 🟡 |
| 100K\|f64\|C\|pos | 0.0361 | 0.0228 | 0.63 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.73 🟡 | 0.70 🟡 | 0.68 🟡 | 0.53 🟡 | 0.84 🟡 | 0.85 🟡 | 1.18 ✅ | 0.85 🟡 |
| 1M | 1.54 ✅ | 1.48 ✅ | 1.48 ✅ | 1.12 ✅ | 1.64 ✅ | 1.65 ✅ | 1.80 ✅ | 1.63 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.67 🟡 | 0.57 🟡 | 1.15 ✅ | 0.75 🟡 | 0.82 🟡 | 0.81 🟡 |
| 1M | 1.86 ✅ | 1.53 ✅ | 1.73 ✅ | 0.94 🟡 | 1.54 ✅ | 1.79 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 1.04 ✅ | 0.99 🟡 | 0.76 🟡 | 0.74 🟡 | 0.94 🟡 | 0.61 🟡 | 0.51 🟡 |
| 1M | 1.81 ✅ | 1.78 ✅ | 2.11 ✅ | 1.62 ✅ | 1.50 ✅ | 0.68 🟡 | 1.75 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f64\|strided\|abs | 0.0466 | 0.0077 | 0.17 🔴 |
| 100K\|f64\|strided\|neg | 0.0473 | 0.0088 | 0.19 🔴 |
| 100K\|i32\|bcast\|copy | 0.0233 | 0.0050 | 0.21 🟠 |
| 100K\|f16\|bcast\|copy | 0.0134 | 0.0030 | 0.22 🟠 |
| 100K\|f16\|negrow\|copy | 0.0168 | 0.0038 | 0.23 🟠 |
| 100K\|f32\|C\|copy | 0.0244 | 0.0059 | 0.24 🟠 |
| 100K\|i64\|strided\|neg | 0.0296 | 0.0076 | 0.26 🟠 |
| 100K\|f64\|strided\|copy | 0.0413 | 0.0108 | 0.26 🟠 |
| 100K\|f64\|negrow\|copy | 0.0595 | 0.0160 | 0.27 🟠 |
| 100K\|f16\|sliced\|copy | 0.0134 | 0.0037 | 0.27 🟠 |
| 100K\|f64\|sliced\|copy | 0.0592 | 0.0162 | 0.27 🟠 |
| 100K\|i32\|negrow\|copy | 0.0234 | 0.0064 | 0.28 🟠 |
| 100K\|f32\|C\|add | 0.0240 | 0.0066 | 0.28 🟠 |
| 100K\|f64\|strided\|mul | 0.0509 | 0.0142 | 0.28 🟠 |
| 100K\|f64\|strided\|add | 0.0516 | 0.0147 | 0.28 🟠 |
