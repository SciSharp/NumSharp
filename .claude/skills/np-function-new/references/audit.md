# Auditing an already-implemented op

The workflow's phases assume you are building; an audit ("verify np.foo", "modernize np.foo",
"why does np.foo differ") reorders them. Everything here was validated by a real audit pass over
five implemented functions (`sum`, `quantile`, `concatenate`, `random.normal`, `allclose/isclose`).

## 1. Inventory existing coverage FIRST

Before probing anything, learn what parity is already *proven* — and its limits:

- `OpRegistry.cs` case + which corpus tiers replay the op (grep the op name in `gen_oracle.py`).
- `MisalignedRegistry.cs` excuses touching it; `OpenBugs*.cs` repros; unit tests; benchmark pair.
- The CLAUDE.md per-function paragraph, if any — it records past probed decisions.

## 2. Signature diff

`inspect.signature(np.foo)` against the C# overload surface (`grep "public static .* foo("`).
Classify every difference:

- **Missing params** — found: `sum` lacks `out=`/`where=`/`initial=`; `quantile` lacks `weights=`.
- **Missing modes of existing params** — found: `sum` axis takes `int?` only (no `int[]` tuple —
  and `axis=()` ≠ `axis=None`: empty tuple reduces NOTHING, an identity pass; probed);
  `allclose` rtol/atol are `double` (NumPy broadcasts array-likes; probed `[False, True]`);
  `normal` loc/scale are `double` (NumPy broadcasts arrays).
- **Pre-convention overload piles** — `sum` has 10 positional overloads predating the house
  one-NumPy-shaped-overload convention; `quantile`/`concatenate` are the modern exemplars.
  Modernizing = add the NumPy-shaped overload; keep the old ones for source compat.

## 3. Behavioral differential OUTSIDE the corpus envelope

**Corpus-green is bounded by the corpus envelope.** The reduce tier's operands are ≤ 24 elements;
NumPy's pairwise-summation blocking (block 128, unroll 8) never engages there, so accumulation-tree
differences are structurally invisible to a green gate. Probed proof: `sum` of 1000× `0.1f` —
NumPy `0x42c80002`, NumSharp `0x42c80004`, 1 ULP apart, tier green the whole time.

Audit probes therefore target what the tiers *cannot* see:

- **Large-N accumulation** — cross the blocking thresholds (129, 1000, 100_003 elements), compare
  BYTES not values.
- **Param combos the tiers skip** — `dtype=` + `out=` together, `casting=` failures, `axis=()`.
- **Call granularity for stateful APIs** — RNG: `normal(size=5)` must equal five `normal()` calls
  (the gauss-cache carry; probed identical, seed 42 bytes exact).
- **Error-order pairs** — two invalid things at once; which text fires first.

## 4. Classify every finding — wins included

- **Real divergence** → fix, or pin under `[OpenBugs]` if not now; check whether
  `MisalignedRegistry` already excuses it (float reduce ops have entries — read them before
  declaring a bug).
- **Intended divergence** → `MisalignedRegistry`, never silent.
- **Missing param/mode** → explicit scope decision + issue (the dependency-audit rule applies).
- **Verbatim/byte-exact wins** → record them (CLAUDE.md paragraph), so the next audit doesn't
  re-prove them. This pass: every `concatenate` error text verbatim (incl. the dtype+out
  mutual-exclusion TypeError and the per-dimension mismatch text), seed-42 `normal` byte-identical,
  `quantile` 9-method values exact, f16 2049-element sum exact.

## Audit evidence table (2026-07 pass)

| Function | Matched | Drifted |
|---|---|---|
| `sum` | i4→i8 dtype, f2 accumulator, empty-min text, AxisError text | 1-ULP f32 tree @N=1000; no out/where/initial; no axis-tuple; overload pile |
| `quantile` | all 9 methods, q-range text, NaN propagation, int→f8 | no `weights=` (2.x) |
| `concatenate` | ALL texts verbatim, axis=None ravel, casting machinery | — |
| `random.normal` | seed-42 bytes, call-granularity, "scale < 0" | scalar-only loc/scale |
| `allclose/isclose` | return types, asymmetry, equal_nan, broadcast | scalar-only rtol/atol |
