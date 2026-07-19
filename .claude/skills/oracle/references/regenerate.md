# Regenerating the oracle corpus

All generators are **deterministic** (fixed seeds, no `random` at gen time) and require **`numpy==2.4.2`**. The
corpus is committed; CI replays it and never runs these. After regenerating, `dotnet build` copies the `.jsonl`
into the test output via the csproj glob.

## Command matrix

```bash
cd test/oracle

# 1. The op corpus (gen_oracle.py) — one file per mode. Pass the mode(s) you touched:
python gen_oracle.py <mode>
#   modes: smoke astype_full binary divmod_power comparison unary reduce where place matmul
#          bitwise unary_extra nanreduce scan stat logic modf manip sort tail params aliasing
#          copyto errors
# Regenerate ALL modes by looping them (each writes its own corpus/<...>.jsonl).

# 2. The advanced-indexing oracle (index_curated / index_dtype / index_random tiers):
python gen_index_oracle.py

# 3. The Decimal oracle (no NumPy analog — independent C# scalar oracle). Writes decimal_*.jsonl:
dotnet run gen_decimal_oracle.cs

# 4. (optional) seeded random soak batch:
python fuzz_random.py 1234 2000 random_smoke.jsonl

# then, from the test project:
cd ../NumSharp.UnitTest && dotnet build     # copies corpus/*.jsonl to bin/.../Fuzz/corpus/
```

The `.npy`/`.npz` FORMAT oracle is a **separate** corpus + gate (`gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip`,
`TestCategory=NpyOracle`). Regenerate it with `python test/oracle/gen_npy_oracle.py`. It is unrelated to the op
corpus above.

## Coverage model — what the loops multiply

Every `gen_<mode>` is `for layout in LAYOUTS: for dtype in <MODE>_DTYPES: for job in jobs: record`.

- **Layouts** come from `layout_catalog.py` — the "44 variations" (C-contiguous, F-contiguous, strided, reversed,
  offset, broadcast, transposed, and pair/where builders). Each builder returns `(base, view)` where `base` is a
  fresh C-contiguous array whose `.tobytes()` is what gets serialized, and `view` is the operand the op sees
  (reconstructable from `shape/strides/offset` into `base`'s bytes). To add a layout, add a builder there and mirror
  it in `LayoutCatalog.cs` (same name both sides).
- **Dtypes** are widened per mode toward `ALL_DTYPES`. Shape/manip ops are dtype-agnostic so `MANIP_DTYPES =
  ALL_DTYPES`; numeric tiers use narrower axes where a dtype is meaningless.
- **Char** (no NumPy dtype) is woven into every tier by `char_tier("<mode>")`, which re-runs `gen_<mode>` with the
  Char pool (`[_C]`) and relabels `uint16 → char`. Nothing extra to do — adding your op to `gen_<mode>` gets Char.
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
`layout_catalog.py` changed a fixture).
