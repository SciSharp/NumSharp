# Oracle system map

The full narrative is the project `.claude/CLAUDE.md` → "Differential-Fuzz Pipeline (NumPy oracle)". This is the
condensed structural map. Four independent oracles share one philosophy: **NumPy (or an independent scalar oracle)
is the source of truth; the corpus is committed; no Python at test time.**

## 1. The op oracle (the main one)

**Generate** — `test/oracle/gen_oracle.py` (~1,900 lines). Structure:
- Value pools + layout builders imported from `layout_catalog.py`.
- One `gen_<mode>(dtypes, layout_names)` per op family. Each loops `layout × dtype`, builds `(base, view)`, runs a
  `jobs` list of `(opname, params, lambda)`, and appends a case `{id, op, params, operands:[{dtype,shape,strides,
  offset,bufferSize,buffer(hex)}], expected:{dtype,shape,buffer(hex)}, layout, valueclass}`.
- `char_tier(mode)` re-runs the relevant `gen_<mode>` with the Char pool and relabels `uint16 → char`.
- `main()` dispatches `mode → gen_<mode>(...) + char_tier(mode) → write_jsonl(corpus/<mode>.jsonl)`.

**Replay** — `test/NumSharp.Tests.Oracle/Fuzz/`:
- `FuzzCorpus.cs` — parses each JSONL line and rebuilds the EXACT `NDArray` view from `(dtype, shape, strides,
  offset, bytes)`, so C# sees byte-identical operands.
- `OpRegistry.cs` — `Apply(op, params, ops)` maps opname → the NumSharp call. Pairs 1:1 with `gen_oracle.py`.
- `BitDiff.cs` — bit-exact compare (NaN tokenized, Decimal by value). `Shrinker.cs` — minimizes a failure.
- `FuzzCorpusTests.cs` — one `[FuzzMatrix]` `[TestMethod]` per corpus file, each calling `RunCorpus("<tier>.jsonl")`.
- `MisalignedRegistry.cs` — the excused, documented divergences.

## 2. The advanced-indexing oracle

- `test/oracle/gen_index_oracle.py` → `index_curated` / `index_dtype` / `index_random` tiers (getter/setter over
  base recipes). Replayed by `IndexOracleTests.cs` (also `[FuzzMatrix]`).

## 3. The Decimal oracle (no NumPy analog)

- `test/oracle/gen_decimal_oracle.cs` — an INDEPENDENT C# oracle using naive scalar `System.Decimal` math, since
  NumPy has no 128-bit decimal. Emits `decimal_{unary,binary,reduce,scan,power,varstd,matmul,astype,stat,where,
  sort,manip}.jsonl`, replayed by the same `FuzzCorpusTests` machinery.

## 4. The `.npy`/`.npz` format oracle (separate corpus + gate)

- `test/oracle/gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` (REAL `np.save`/`savez` output + a manifest).
- Replayed by `IO/NpyOracleTests.cs` under `TestCategory=NpyOracle` — the claim is stronger: NumSharp's writer must
  be BYTE-IDENTICAL to `np.save`, not merely readable. Reverse interop (NumPy reading NumSharp) is the manual gate
  `python test/oracle/verify_npy_interop.py`.

## Other harness pieces

- `MetamorphicTests.cs` — NumPy-free invariants (round-trips / involutions / identities); catch bugs the oracle
  can't (no reference needed).
- `HarnessSelfTests.cs` — proves the harness has teeth (BitDiff actually detects value/NaN/-0 diffs; non-vacuous).
- `fuzz_random.py` — the nightly-soak seeded fuzzer (`.github/workflows/fuzz-soak.yml`), ~1M fresh cases/night;
  shrunk failures get pinned under `Fuzz/corpus/regressions/`.

## Where the corpus lives and how it reaches tests

- Generators write to `test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl` (path resolved relative to `test/oracle/`).
- The test `.csproj` has a glob that copies `corpus/*.jsonl` into the build output; `RunCorpus` reads them from
  there. So a regeneration is only "live" after a `dotnet build`.
- The whole point: **CI runs `dotnet test --filter TestCategory=FuzzMatrix` with no Python** — it replays committed
  bytes. Regenerating + committing the `.jsonl` is the entire delivery of new coverage.
