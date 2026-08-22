# Layout suite — reduction / copy / elementwise × memory layout × dtype

ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.
Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), `strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, `bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). 100K + 1M elements, best-of-rounds.

### Reduction (sum/min/max/prod, both axes)

**Geomean by lay**

| size | C | F | T | strided | negrow | negcol | sliced | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.05 ✅ | 1.12 ✅ | 1.09 ✅ | 0.58 🟡 | 0.80 🟡 | 0.75 🟡 | 0.77 🟡 | 0.78 🟡 |
| 1M | 1.35 ✅ | 1.37 ✅ | 1.39 ✅ | 1.00 ✅ | 1.37 ✅ | 1.21 ✅ | 1.30 ✅ | 0.84 🟡 |

**Geomean by dt**

| size | f64 | f32 | c128 | dec | f16 | i32 | i64 |
|---|---|---|---|---|---|---|---|
| 100K | 0.96 🟡 | 1.04 ✅ | 0.96 🟡 | 0.09 🔴 | 1.09 ✅ | 1.51 ✅ | 1.32 ✅ |
| 1M | 1.90 ✅ | 1.52 ✅ | 1.80 ✅ | 0.13 🔴 | 1.38 ✅ | 1.50 ✅ | 1.60 ✅ |

**Geomean by op**

| size | sum | min | max | prod |
|---|---|---|---|---|
| 100K | 0.83 🟡 | 0.72 🟡 | 0.70 🟡 | 1.30 ✅ |
| 1M | 1.38 ✅ | 0.94 🟡 | 0.93 🟡 | 1.91 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|dec\|bcast\|sum\|ax0 | 0.5777 | 0.0111 | 0.02 🔴 |
| 1M\|dec\|bcast\|sum\|ax0 | 5.3967 | 0.1121 | 0.02 🔴 |
| 100K\|dec\|F\|sum\|ax1 | 0.5420 | 0.0129 | 0.02 🔴 |
| 100K\|dec\|C\|sum\|ax0 | 0.5645 | 0.0137 | 0.02 🔴 |
| 100K\|dec\|sliced\|sum\|ax0 | 0.5576 | 0.0136 | 0.02 🔴 |
| 100K\|dec\|negrow\|sum\|ax0 | 0.5461 | 0.0135 | 0.02 🔴 |
| 100K\|dec\|T\|sum\|ax1 | 0.5533 | 0.0138 | 0.03 🔴 |
| 1M\|dec\|T\|sum\|ax1 | 5.5946 | 0.1709 | 0.03 🔴 |
| 1M\|dec\|C\|sum\|ax0 | 5.5519 | 0.1968 | 0.04 🔴 |
| 1M\|dec\|negrow\|sum\|ax0 | 5.5506 | 0.1982 | 0.04 🔴 |
| 1M\|dec\|sliced\|sum\|ax0 | 5.4491 | 0.2040 | 0.04 🔴 |
| 1M\|dec\|F\|sum\|ax1 | 5.4765 | 0.2199 | 0.04 🔴 |
| 100K\|dec\|C\|min\|ax0 | 0.3278 | 0.0164 | 0.05 🔴 |
| 100K\|dec\|negcol\|sum\|ax0 | 0.5598 | 0.0310 | 0.06 🔴 |
| 100K\|dec\|F\|max\|ax1 | 0.2259 | 0.0144 | 0.06 🔴 |

### Copy / identity-ufunc (np.positive)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 1.04 ✅ | 0.99 🟡 | 1.03 ✅ | 0.57 🟡 | 1.09 ✅ | 1.20 ✅ | 1.57 ✅ | 1.10 ✅ |
| 1M | 2.19 ✅ | 2.19 ✅ | 2.17 ✅ | 1.01 ✅ | 1.73 ✅ | 1.85 ✅ | 1.71 ✅ | 2.08 ✅ |

**Geomean by dt**

| size | u8 | i8 | i16 | u16 | i32 | u32 | i64 | u64 | char | f16 | f32 | f64 | c128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.86 🟡 | 1.17 ✅ | 1.25 ✅ | 1.51 ✅ | 0.63 🟡 | 0.54 🟡 | 1.17 ✅ | 0.89 🟡 | 1.50 ✅ | 1.87 ✅ | 0.82 🟡 | 0.58 🟡 | 1.84 ✅ |
| 1M | 2.39 ✅ | 1.94 ✅ | 1.26 ✅ | 1.24 ✅ | 1.82 ✅ | 1.56 ✅ | 1.71 ✅ | 1.73 ✅ | 1.50 ✅ | 2.06 ✅ | 2.76 ✅ | 1.54 ✅ | 2.95 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|i64\|strided\|pos | 0.0683 | 0.0137 | 0.20 🟠 |
| 100K\|f64\|strided\|pos | 0.0695 | 0.0201 | 0.29 🟠 |
| 100K\|u32\|strided\|pos | 0.0451 | 0.0131 | 0.29 🟠 |
| 100K\|f64\|T\|pos | 0.0867 | 0.0321 | 0.37 🟠 |
| 100K\|f64\|F\|pos | 0.0914 | 0.0351 | 0.38 🟠 |
| 100K\|f64\|C\|pos | 0.0755 | 0.0309 | 0.41 🟠 |
| 100K\|u32\|C\|pos | 0.0564 | 0.0252 | 0.45 🟠 |
| 100K\|i32\|sliced\|pos | 0.0726 | 0.0325 | 0.45 🟠 |
| 100K\|u64\|strided\|pos | 0.0398 | 0.0181 | 0.46 🟠 |
| 100K\|i32\|strided\|pos | 0.0480 | 0.0219 | 0.46 🟠 |
| 100K\|f32\|strided\|pos | 0.0280 | 0.0147 | 0.53 🟡 |
| 100K\|u32\|T\|pos | 0.0454 | 0.0243 | 0.54 🟡 |
| 100K\|i32\|negcol\|pos | 0.1055 | 0.0575 | 0.54 🟡 |
| 100K\|i32\|bcast\|pos | 0.0533 | 0.0304 | 0.57 🟡 |
| 100K\|u32\|F\|pos | 0.0576 | 0.0339 | 0.59 🟡 |

### Elementwise (add/mul/neg/abs/sqrt/less/copy)

**Geomean by lay**

| size | C | F | T | strided | sliced | negrow | negcol | bcast |
|---|---|---|---|---|---|---|---|---|
| 100K | 0.60 🟡 | 0.58 🟡 | 0.63 🟡 | 0.38 🟠 | 0.69 🟡 | 0.66 🟡 | 0.82 🟡 | 0.54 🟡 |
| 1M | 1.53 ✅ | 1.44 ✅ | 1.65 ✅ | 1.31 ✅ | 1.77 ✅ | 1.67 ✅ | 1.81 ✅ | 2.00 ✅ |

**Geomean by dt**

| size | f64 | f32 | c128 | f16 | i32 | i64 |
|---|---|---|---|---|---|---|
| 100K | 0.45 🟠 | 0.17 🔴 | 0.55 🟡 | 1.06 ✅ | 0.94 🟡 | 1.10 ✅ |
| 1M | 1.45 ✅ | 1.58 ✅ | 1.88 ✅ | 0.98 🟡 | 1.75 ✅ | 2.55 ✅ |

**Geomean by op**

| size | add | mul | neg | abs | sqrt | less | copy |
|---|---|---|---|---|---|---|---|
| 100K | 0.71 🟡 | 0.72 🟡 | 0.53 🟡 | 0.52 🟡 | 0.82 🟡 | 0.59 🟡 | 0.41 🟠 |
| 1M | 1.85 ✅ | 1.88 ✅ | 2.01 ✅ | 1.84 ✅ | 1.82 ✅ | 0.75 🟡 | 1.77 ✅ |

**Worst 15 cells (NumSharp slowest vs NumPy)**

| key | NumSharp ms | NumPy ms | ratio |
|---|---|---|---|
| 100K\|f32\|strided\|abs | 0.0803 | 0.0067 | 0.08 🔴 |
| 100K\|f32\|sliced\|copy | 0.0773 | 0.0069 | 0.09 🔴 |
| 100K\|f32\|C\|copy | 0.0738 | 0.0066 | 0.09 🔴 |
| 100K\|f32\|strided\|neg | 0.0794 | 0.0074 | 0.09 🔴 |
| 100K\|f32\|negrow\|copy | 0.0701 | 0.0069 | 0.10 🔴 |
| 100K\|f32\|bcast\|copy | 0.0552 | 0.0057 | 0.10 🔴 |
| 100K\|f32\|T\|add | 0.0785 | 0.0081 | 0.10 🔴 |
| 100K\|f32\|strided\|sqrt | 0.0849 | 0.0090 | 0.11 🔴 |
| 100K\|f32\|C\|add | 0.0690 | 0.0075 | 0.11 🔴 |
| 100K\|c128\|strided\|mul | 0.3327 | 0.0364 | 0.11 🔴 |
| 100K\|f32\|C\|less | 0.0402 | 0.0045 | 0.11 🔴 |
| 100K\|f32\|C\|neg | 0.0641 | 0.0072 | 0.11 🔴 |
| 100K\|f32\|strided\|mul | 0.1398 | 0.0158 | 0.11 🔴 |
| 100K\|f32\|C\|mul | 0.0703 | 0.0084 | 0.12 🔴 |
| 100K\|f64\|strided\|neg | 0.0726 | 0.0087 | 0.12 🔴 |
