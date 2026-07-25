# Auditing an already-implemented op

The workflow's phases assume you are building; an audit reorders them. Two triggers land here:

- **Auditing code you did not write** — "verify np.foo", "modernize np.foo", "why does np.foo
  differ". Start at §1: you do not yet know what parity is already proven.
- **Sweeping the op you just shipped** — "do a pass over edge cases", "find missing support". Skip
  §1 (you have the inventory in your head) and go straight to §3, whose whole subject is what the
  gates you just turned green structurally cannot see. This is high-yield precisely because it is
  not a re-run of Phase 4: a sweep after a clean Phase-6 commit found a live `InvalidCastException`
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

## 3. Behavioral differential OUTSIDE the corpus envelope

**Corpus-green is bounded by the corpus envelope.** The reduce tier's operands are ≤ 24 elements;
NumPy's pairwise-summation blocking (block 128, unroll 8) never engages there, so accumulation-tree
differences are structurally invisible to a green gate. Probed proof: `sum` of 1000× `0.1f` —
NumPy `0x42c80002`, NumSharp `0x42c80004`, 1 ULP apart, tier green the whole time.

Audit probes therefore target what the tiers *cannot* see:

- **Large-N accumulation** — cross the blocking thresholds (129, 1000, 100_003 elements), compare
  BYTES not values.
- **Param combos the tiers skip** — `dtype=` + `out=` together, `casting=` failures, `axis=()`.
  Generalize this: enumerate your parameters' cross-product and find the cells the generator never
  emits. The diag/tri corpus wrote every non-contiguous `fill_diagonal` destination with
  `wrap=false`, so `wrap` × non-contiguous — 37 real combinations — had zero coverage until probed
  by hand (all 37 turned out bit-exact, which is exactly the outcome that makes the gap invisible).
- **The acceptance surface of polymorphic parameters.** A corpus serializes each argument to ONE
  canonical form, so everything else the declared type admits is untested by construction. For an
  `object`-typed value: scalars, `NDArray`, C# arrays (incl. rectangular 2-D), `IEnumerable`
  collections, non-`IConvertible` scalars (`Half`, `Complex`), and `null`. This is where the
  post-ship sweep found its bug — a scalar fast path had narrowed acceptance so `int[]` threw, with
  every gate still green. Fix it at both ends: unit tests for the surface, plus widen the oracle to
  pass a NON-canonical form so the general path stays gated.
- **Both sides of a size- or layout-selected branch.** Any threshold splits the code in two and
  every ordinary test sits on one side. Cross it deliberately, and prefer an invariant that holds
  regardless of size — `tril(a, k) + triu(a, k+1) == a` catches a missing zero-fill on the large
  path without needing a stored expectation.
- **Extreme and degenerate scalar params** — `int.MinValue`/`int.MaxValue` for offsets that get
  added to an index (do they saturate or wrap?), negative sizes, and the allocation ceiling
  (`diag` of a 1-D with a huge `k` needs an `(n+|k|)²` matrix — report, don't overflow).
- **`null` for every reference parameter.** NumPy's `None` often has real semantics (`fill_diagonal`
  with `None` writes nan to a float array); decide deliberately whether to mirror it or reject, and
  record the choice. Silently no-op'ing is the one answer that is always wrong.
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

## Post-ship sweep evidence (diag/tri family, 13 functions)

Run immediately after a green Phase 6 — 76 unit tests, FuzzMatrix 91/91, full suite 11,904 passing.

| Probe dimension | Result |
|---|---|
| Acceptance surface of `object val` | **BUG** — `int[]`/`long[]`/`double[]`/`List`/`int[,]` all threw `InvalidCastException`; a Phase-5 scalar fast path had narrowed acceptance. Invisible to every gate (all pass `val` as `NDArray`) |
| `null` value argument | **Unspecified** — silently wrote nothing; NumPy's `None` yields nan (float) / TypeError (int). Now throws, documented as a deliberate divergence |
| Size-selected branch (64 MiB) | **Untested half** — every test sat below the threshold; the large path had no coverage. Verified via `tril(k)+triu(k+1)==a` at 59 MB and 78 MB |
| `wrap` × non-contiguous | 37 combinations uncovered by the tier; all 37 bit-exact once probed |
| Extreme `k` (`int.Min/MaxValue`) | Saturates correctly; `diag`'s 1-D branch reports "array is too big" rather than overflowing |
| Allocation ceiling | `tri(int.MaxValue, 2)` genuinely allocates 34 GB and fills correctly — no silent size overflow |
| Char / Decimal, null array args, 0-size hyper-cubes, runtime-rank tuple arity | All clean |

The shape of the finding matters more than the specifics: **the two bugs both lived in code added
AFTER the gates went green**, and both sat in dimensions the corpus cannot express. A sweep that
only re-runs the tiers would have found neither.
