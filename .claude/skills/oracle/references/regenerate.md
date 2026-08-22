# Regenerating the oracle corpus

All generators are **deterministic** — `gen_oracle.py` uses no RNG; `gen_index_oracle.py` and `fuzz_random.py` use a
PINNED seed — and require **`numpy==2.4.2`** (the `precision` tier additionally needs `mpmath`). The corpus is
committed; CI replays it and never runs these. After regenerating, `dotnet build` (in `NumSharp.Tests.Oracle`) copies
the `.jsonl` into the test output via the csproj glob.

## Command matrix

```bash
cd test/oracle

# 1. The op corpus (gen_oracle.py) — one file per mode. Pass the mode(s) you touched:
python gen_oracle.py <mode>
#   surface/kind tiers: conversion creation multioutput iter dtype_text out_where errors_full
#   core value tiers:   smoke astype_full binary divmod_power comparison unary unary_extra bitwise
#                       reduce nanreduce scan stat logic modf manip sort tail rounding params
#                       aliasing copyto place where matmul errors groupa
#   value/parity tiers: specials precision products fft random_parity matmul_parity numpy_f32
#   creation/conversion append their own Char proxy rows; the usual 18 modes use char_tier
# Authoritative list = the `elif mode == ...` branches in gen_oracle.py's main() (and its
# unknown-mode error message). Notes: `numpy_f32` writes BOTH numpy_f32_kernels.jsonl AND
# numpy_f64_kernels.jsonl; `matmul_parity` ALSO writes the host pin matmul_parity.host.jsonl;
# `random_parity` writes both random_parity.jsonl (portable) and random_parity_host.jsonl (win-amd64).
# Regenerate ALL modes by looping them (each writes its own corpus/<...>.jsonl).

# 2. The advanced-indexing oracle (index_curated / index_dtype / index_random tiers):
python gen_index_oracle.py

# 3. The Decimal oracle (no NumPy analog — independent C# scalar oracle). Writes decimal_*.jsonl:
dotnet run gen_decimal_oracle.cs

# 4. (optional) seeded random soak batch:
python fuzz_random.py 1234 2000 random_smoke.jsonl

# then, from the test project:
cd ../NumSharp.Tests.Oracle && dotnet build     # copies corpus/*.jsonl to bin/.../Fuzz/corpus/
```

The `.npy`/`.npz` FORMAT oracle is a **separate** corpus + gate (`gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip`,
`TestCategory=NpyOracle`). Regenerate it with `python test/oracle/gen_npy_oracle.py`. It is unrelated to the op
corpus above.

## Coverage model — what the loops multiply

Every `gen_<mode>` is `for layout in LAYOUTS: for dtype in <MODE>_DTYPES: for job in jobs: record`.

- **Layouts** come from `layout_catalog.py` — the 40 variations (26 single-array + 9 pairwise + 5 where-triple:
  C-contiguous, F-contiguous, strided, reversed, offset, broadcast, transposed, 0-d/empty/high-rank). Each builder
  returns `(base, view)` where `base` is a fresh C-contiguous array whose `.tobytes()` is what gets serialized, and
  `view` is the operand the op sees (reconstructable from `shape/strides/offset` into `base`'s bytes). To add a
  layout, add a builder there and regenerate the affected tiers — **no C# mirror is needed**: `FuzzCorpus.Reconstruct`
  aliases any `(shape, strides, offset)` over the base bytes generically (there is no `LayoutCatalog.cs`).
- **Dtypes** are widened per mode toward `ALL_DTYPES` (the 13 NumPy-representable dtypes). Shape/manip ops are
  dtype-agnostic so `MANIP_DTYPES = ALL_DTYPES`; numeric tiers use narrower axes where a dtype is meaningless.
- **Char** (no NumPy dtype) is woven into the 18 tiers whose `main()` branch calls `char_tier("<mode>")`, which
  re-runs `gen_<mode>` with the Char pool (`[_C]`) and relabels `uint16 → char`. Adding your op to one of those
  `gen_<mode>`s gets Char automatically; modes with no `char_tier` call (`modf`/`place`/`nanreduce`/`params`/… and
  every value-parity/result-kind tier) have no Char coverage.
- **Decimal** (no NumPy analog) rides `gen_decimal_oracle.cs` separately (step 3 above).
- **The value pools** (`_FLOAT_POOL`, `_INT_POOL` in `layout_catalog.py`) front-load the edges that break kernels:
  `nan, inf, -inf, -0.0, 0.0`, type min/max boundaries, narrowing-wrap seams. `_INT_POOL` STARTS with `0` — that's
  why value-dependent ops like `trim_zeros`/`nonzero` get real coverage from the generic tiers.

## The huge-but-harmless diff

Case `id` is `f"{opname}/{layout}/{dtype}/{n}"` where `n` is a **global** running counter incremented per emitted
case. Inserting a job renumbers every following `id`, so `git diff` on the `.jsonl` looks enormous. It is pure
renumbering; the recorded dtypes/shapes/bytes for existing cases are unchanged. Don't try to minimize it — commit it.

## Verify determinism

Regenerating with the same NumPy version + unchanged generator must produce a byte-identical file (only your new
cases differ). If a "no-op" regeneration changes existing bytes, your NumPy is not 2.4.2 (or `gen_oracle.py` /
`layout_catalog.py` changed a fixture) — **or you regenerated on a different OS/CPU**, see below.

## Regenerate on the platform the corpus was authored on

`astype_full` (and any tier casting float/complex → an integer dtype) is **host-sensitive**. C leaves a
float→integer conversion undefined for NaN, ±inf, and out-of-range values; NumPy just does the C cast, so glibc/gcc
and MSVC produce different bytes — as do the vectorized and scalar loops of one build. Those cells are committed on
purpose (they pin NumSharp's hand-written cast kernels against themselves), but regenerating them elsewhere rewrites
them, and the diff is a host difference, not a NumSharp bug. `fuzz_random.py` is the exception: it recomputes
expectations on the host, so it defuses that value class and is portable by construction. Details:
`test/NumSharp.Tests.Oracle/Fuzz/README.md` → "Host-dependent values".
