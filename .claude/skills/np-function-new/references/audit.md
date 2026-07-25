# Auditing an already-implemented op

The workflow's phases assume you are building; an audit reorders them. Two triggers land here:

- **Auditing code you did not write** — "verify np.foo", "modernize np.foo", "why does np.foo
  differ". Start at §1: you do not yet know what parity is already proven.
- **Sweeping the op you just shipped** — "do a pass over edge cases", "find missing support". Skip
  §1 (you have the inventory in your head) and go straight to §3, whose whole subject is what the
  gates you just turned green structurally cannot see. This is high-yield precisely because it is
  not a re-run of Phase 4: a sweep after a clean Phase-7 commit found a live `InvalidCastException`
  on the second-most-common way to call the function.

Validated by a real audit pass over five implemented functions (`sum`, `quantile`, `concatenate`,
`random.normal`, `allclose/isclose`) and a post-ship sweep over the 13-function diag/tri family.

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

## 3. Run the Phase-5 sweep

The dimension catalog is Phase 5's and lives in **`edge-sweep.md`** — magnitude, structural,
resource, parameter/`null`, acceptance-surface, adversarial-combination, call-form and
threshold-crossing extremes, plus the two mechanical triage rules. Work it from there; auditing
adds exactly one dimension of its own, because only a shipped implementation has a history:

- **The corpus envelope.** Green is bounded by what the tiers actually contain. The reduce tier's
  operands are ≤ 24 elements, so NumPy's pairwise-summation blocking (block 128, unroll 8) never
  engages and accumulation-tree differences cannot show. Probed proof: `sum` of 1000× `0.1f` —
  NumPy `0x42c80002`, NumSharp `0x42c80004`, 1 ULP apart, tier green the whole time. Cross the
  blocking thresholds (129, 1000, 100_003) and compare BYTES, not values. Then ask the same
  question of every other tier your op rides: what is its size, layout and dtype envelope, and what
  lives outside it?

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

## Post-ship sweep evidence (diag/tri family, 13 functions)

Run immediately after a green Phase 7 — 76 unit tests, FuzzMatrix 91/91, full suite 11,904 passing.

| Probe dimension | Result |
|---|---|
| Acceptance surface of `object val` | **BUG** — `int[]`/`long[]`/`double[]`/`List`/`int[,]` all threw `InvalidCastException`; a Phase-6 scalar fast path had narrowed acceptance. Invisible to every gate (all pass `val` as `NDArray`) |
| `null` value argument | **Unspecified** — silently wrote nothing; NumPy's `None` yields nan (float) / TypeError (int). Now throws, documented as a deliberate divergence |
| Size-selected branch (64 MiB) | **Untested half** — every test sat below the threshold; the large path had no coverage. Verified via `tril(k)+triu(k+1)==a` at 59 MB and 78 MB |
| `wrap` × non-contiguous | 37 combinations uncovered by the tier; all 37 bit-exact once probed |
| Extreme `k` (`int.Min/MaxValue`) | Saturates correctly; `diag`'s 1-D branch reports "array is too big" rather than overflowing |
| Allocation ceiling | `tri(int.MaxValue, 2)` genuinely allocates 34 GB and fills correctly — no silent size overflow |
| Char / Decimal, null array args, 0-size hyper-cubes, runtime-rank tuple arity | All clean |

The shape of the finding matters more than the specifics: **the two bugs both lived in code added
AFTER the gates went green**, and both sat in dimensions the corpus cannot express. A sweep that
only re-runs the tiers would have found neither.
