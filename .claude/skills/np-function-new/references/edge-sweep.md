# Phase 5 — the edge & extremes dimension catalog

## Why this is a phase and not a habit

Phase 1 probes what you thought to ask. Phase 4 proves parity over what the corpus can *serialize*:
one canonical spelling per argument, operands in the tens of elements, shapes a generator finds
natural. Both are cooperative. The bugs that survive them are not subtle reasoning errors — they
are the inputs nobody pointed at the code, and they are found by deciding in advance to attack
along fixed dimensions rather than by being clever.

Run the catalog below. Most entries are two lines of probe, and the yield is high because you are
the first person to look.

## The two mechanical triage rules

These fire without knowing what NumPy answers, so run them over every probe you write.

**1. A raw .NET framework exception at the API boundary is always a finding.**
`DivideByZeroException`, `NullReferenceException`, `IndexOutOfRangeException`,
`InvalidCastException`, `OverflowException`, `ArgumentNullException` from deep inside — NumPy
answers with typed, texted errors, so every one of these marks a guard nobody wrote. Probed:
`np.zeros((0,3)).reshape(-1, 0)` throws `DivideByZeroException` (a degenerate dimension reached a
divisor) where NumPy raises `ValueError: cannot reshape array of size 0 into shape (0)`. You do not
need the NumPy comparison to know this one is wrong.

**2. Silent success where NumPy refuses is worse than a crash.** It is the class most likely to be
a memory-safety bug rather than a cosmetic one. Probed: `np.zeros((2^31, 2^31))` constructs
successfully, `[0,0]` reads `0`, and the last element reads garbage — an **out-of-bounds read**
where NumPy raises ``ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than
the maximum possible size``. The mechanism is worth internalizing because it generalizes: `size` is
correctly 2⁶², but `size * 8` is 2⁶⁵, which wraps to **exactly 0** in 64 bits — so the allocator is
asked for zero bytes and cheerfully obliges. One dimension over (`(2^30, 2^33)`) the element count
itself overflows and `size` reports **negative**. That the neighbouring case `np.zeros(2^40)`
correctly throws `OutOfMemoryException` is the tell: a guard that exists on some paths and not
others is exactly what a sweep is for.

The general lesson: **a size computed by multiplication needs its overflow checked before it is
trusted**, and the failure is silent by construction — a wrapped byte count looks like a small
allocation, not like an error. NumPy carries a dedicated guard for this; where NumSharp computes
`size × itemsize`, check that one exists.

## The harness

One `T(label, thunk)` that prints either the result or `Type: message`, so a failure is data
rather than a stack trace that ends the run. Both sides, same case list, diff the transcripts.

```csharp
void T(string label, Func<string> f) {
    try { Console.WriteLine($"  {label}: {f()}"); }
    catch (Exception e) { Console.WriteLine($"  {label}: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); }
}
```

```python
def T(label, f):
    try: print(f"  {label}: {f()}")
    except Exception as e: print(f"  {label}: {type(e).__name__}: {e}")
```

Extremes are where a probe is likeliest to crash the host or allocate the machine to death — keep
each case independent, and put genuinely huge allocations last.

## The dimension catalog

### A. Magnitude & value extremes

Feed each dtype's boundary values and the values that break arithmetic identities: `MinValue`,
`MaxValue`, `MinValue` under negation/abs (no positive counterpart exists), `±inf`, `-0.0`, NaN
(including sNaN payloads), smallest subnormal, and inputs that overflow the accumulator.

**NumPy wraps silently — that is the contract, not a bug to fix.** Probed and matched by NumSharp:
`abs(i64.min)` stays negative, `negative(i64.min)` is itself, `sum([i64.max, 1])` wraps to
`i64.min`, `i64.min // -1` wraps, `u64.max + 1 == 0`, `f64.max * 2 == inf`. A well-intentioned move
to checked arithmetic would break every one, which is why matches here get written down.

### B. Structural extremes

Rank, degeneracy, and shapes that are legal but nobody builds:

- **Rank ceiling.** NumPy 2.x caps at 64 dimensions:
  `ValueError: maximum supported dimension for an ndarray is currently 64, found 65`. Probed:
  NumSharp accepts ndim=65 — no cap. Worth knowing before you design a stack-allocated coord array.
- **Interior zeros** — `(1,0,1)`, `(3,0,4)`: a zero *between* non-zero dims, not just a trailing
  one. Reductions over them still have to produce the right shape (probed: `sum((1,0,1), axis=1)`
  → `(1,1)`, matched).
- **Zero-size reshape/transpose/view chains** — `zeros((0,3)).reshape(3,0)`, `.T`. This is where
  divide-by-a-dimension lives; rule 1 catches it.
- **Size-1 axes, deep view-of-view chains, negative strides on a 0-size axis.**

### C. Resource extremes

The `size × itemsize` overflow class, which is invisible until it corrupts memory:

- Shapes whose element count is fine in 64 bits but whose **byte count** is not (`(2^31, 2^31)` of
  float64 → 2⁶⁵ bytes). NumPy guards up front; check that NumSharp raises rather than returns.
- Shapes that overflow an `int` somewhere in the stride/offset math even when the total is legal.
- The honest allocation ceiling: a request that genuinely needs more RAM than exists should raise
  `OutOfMemoryException`, not truncate. (Probed clean: `zeros(2^40)` does exactly this; a previous
  sweep found `tri(int.MaxValue, 2)` genuinely allocating 34 GB and filling correctly.)

### D. Parameter extremes, including `null`

- `int.MinValue` / `int.MaxValue` / `long.MaxValue` for every scalar parameter — especially offsets
  that get *added* to an index: do they saturate or wrap? (Probed: `axis=int.MinValue` rejected by
  both, though the texts differ — NumSharp's `AxisError` vs NumPy's
  `ValueError: integer won't fit into a C int`.)
- Negative sizes/counts, zero counts, and the value one past every documented bound.
- **`null` for every reference parameter.** NumPy's `None` usually has real semantics
  (`fill_diagonal(a, None)` writes nan to a float array), so decide deliberately: mirror it, or
  reject with a texted error. Silently no-op'ing is the one answer that is always wrong.

### E. Acceptance surface of polymorphic parameters

A corpus serializes each argument to ONE canonical form, so **everything else the declared type
admits is untested by construction**. For an `object`-typed value that means: scalars, `NDArray`,
C# arrays (including rectangular `int[,]`), `IEnumerable` collections, non-`IConvertible` scalars
(`Half`, `Complex`), and `null`. Enumerate what the signature actually admits and probe each.

This is the single highest-yield entry in the catalog. A real regression: a scalar fast path added
for speed narrowed acceptance so `int[]`, `List<int>` and `int[,]` all threw `InvalidCastException`
— while unit tests, the tier and a hand-written parity harness stayed green, because every one of
them passes that argument as an `NDArray`. Close it at both ends: unit tests for the surface, plus
widen the oracle to hand the argument over in a NON-canonical form (a raw `long[]` rather than
`np.array(...)`) so the general path stays gated from then on.

### F. Adversarial combinations

Two legal things that are hostile together:

- **Aliasing and overlap** — `out=` aliasing an input, partially overlapping views (`a[1:]` with
  `a[:-1]`), in-place accumulation. NumPy resolves these with COPY_IF_OVERLAP and the result is
  well-defined, so it is parity. (Probed and matched: `add(a[1:], a[:-1], out=a[:-1])`.)
- **Broadcast + `out=`**, read-only broadcast destinations, `where=` masking a strided `out`.
- **Parameter cross-product cells the generator never emits.** Enumerate your parameters' product
  and find the empty cells: one corpus wrote every non-contiguous destination with `wrap=false`,
  leaving 37 real `wrap` × non-contiguous combinations uncovered — all 37 turned out bit-exact,
  which is precisely what keeps such a gap invisible.
- **Error-order pairs** — make two things wrong at once and check which error fires first.
- **Call granularity for stateful APIs** — RNG: `normal(size=5)` must equal five `normal()` calls.

### G. Call forms your tests never use

The tests use the overload you were thinking about while writing them. Enumerate the others: every
overload, named vs positional arguments, defaults omitted vs passed explicitly, and the NumPy call
forms that ought to exist. Missing API surface shows up here — probed: `np.cumsum` has no `out=`
overload at all, though NumPy's signature is `cumsum(a, axis, dtype, out)`.

### H. Threshold crossings

Any size- or layout-selected branch splits the code in two, and every ordinary test sits on one
side. Cross every threshold deliberately — NumPy's pairwise-summation blocking (128), your own fast
paths, allocation strategy switches. Prefer an **invariant that holds regardless of size**
(`tril(a,k) + triu(a,k+1) == a`) over a stored expectation, since it catches a missing zero-fill on
the large path without needing a 78 MB fixture in the repo.

## Classify every finding — wins included

- **Real divergence** → fix now, or pin under `[OpenBugs]`. Check `MisalignedRegistry` first: float
  reduce ops already have entries, so read before declaring.
- **Intended divergence** → `MisalignedRegistry` or the CLAUDE.md paragraph, never silent.
- **Missing param / call form** → explicit scope decision + issue.
- **Match** → record it. Matches on wrap semantics, aliasing results and error texts are load-bearing
  facts that stop a later "cleanup" from breaking them.

## Evidence — extremes pass (2026-07)

| Dimension | Probe | Result |
|---|---|---|
| Structural | `reshape(-1, 0)` on any 0-size array | **BUG** — `DivideByZeroException` (the `-1` inference divides by the product of the known dims, which is 0); NumPy: `ValueError: cannot reshape array of size 0 into shape (0)`. `reshape(0,-1)` and `arange(0).reshape(-1,0)` too; without the `-1` (`reshape(3,0)`) it is fine |
| Resource | `zeros((2^31,2^31))` | **BUG** — `size × 8` = 2⁶⁵ wraps to **0 bytes** requested; constructs, last element reads garbage (OOB read). `(2^30,2^33)` reports a **negative** `size`. NumPy: `ValueError: array is too big; ...` |
| Structural | ndim = 65 | **Divergence** — accepted; NumPy caps at 64 with a verbatim text |
| Call form | `cumsum(..., out=)` | **Missing** — no such overload |
| Parameter | `axis=int.MinValue` | Both reject; texts differ (`AxisError` vs `integer won't fit into a C int`) |
| Magnitude | i64.min abs/negate, sum/floor-div wrap, u64 wrap, f64 overflow→inf | **All matched** — silent-wrap contract reproduced exactly |
| Structural | `(1,0,1)` axis-sum, 0-size reshape/`.T` | **Matched** |
| Adversarial | overlapping `out=` | **Matched** byte-for-byte |
| Resource | `zeros(2^40)` | **Matched** — clean `OutOfMemoryException` |

The two bugs sit in dimensions no gate expresses, and the one-line reshape repro had been reachable
from public API the whole time. See `audit.md` for the five-function audit and 13-function
post-ship sweep tables, whose two bugs both lived in code added *after* the gates went green.
