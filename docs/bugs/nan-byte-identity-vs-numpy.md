# NaN byte-identity vs NumPy: which `np.*` ops diverge, and which "differences" are correct

**Status:** investigation (2026-08-14). Method, proof, root cause, and classification below. The
localized core fixes are **recommended, not applied** — they touch the bit-for-bit complex/float16
kernels and must be regenerated/verified against the differential-fuzz oracle. The interop is only the
measurement tool here; every finding is a **NumSharp-core** property, not an interop defect.

**Probed on:** CPython 3.12.12 · numpy 2.4.2 · net8.0 · Windows 11, via
`experimental/adversarial-harness` scenarios **`byteident`** and **`nansweep`**.

## Method (numpy vs NumSharp, raw bytes)

For each operation: give NumSharp and numpy **byte-identical inputs** (finite literals, or numpy
specials imported byte-exact through the interop), run the **same** `np.*` op in each engine, then
compare `dtype` + `tobytes()`. Extraction of NumSharp's bytes is the zero-copy `ToNumpy()` view, so the
bytes are exactly what NumSharp stored. 68 + 51 comparisons; **the value is always correct — only NaN
*bit patterns* differ.**

## Result: 3 correct-design, 4 real byte-identity problems

### Correct design (NOT bugs)

1. **Construction from a NaN *literal*.** `np.array(new[]{ double.NaN })` stores `0xFFF8…` while
   Python's `np.array([np.nan])` stores `0x7FF8…`. This is the **language default**: C#'s
   `double.NaN`/`Half.NaN` are the negative-signed quiet NaN, Python's `float('nan')`/`np.nan` are the
   positive one. **Both engines faithfully store whatever NaN they are handed** — numpy also stores
   `0xFFF8…` if given `np.array([np.float64(struct.unpack('<d', bytes.fromhex('000000000000f8ff'))[0])])`.
   A C# caller who wants numpy's pattern passes it explicitly. Not fixable without silently mutating
   user input, which numpy itself does not do.

2. **`GetDouble`/`SetDouble` are typed, double-only accessors** (observation from the prior pass).
   `UnmanagedStorage.GetDouble` is `_arrayDouble[offset]` with an explicit XML contract:
   *"NullReferenceException when DType is not double"* — the general accessors are
   `GetValue<T>`/`SetValue<T>` (and boxed `GetValue`/`SetValue`). So calling `GetDouble` on an `int16`
   array is a caller error, by design. **Correct design**, with one core-robustness nit worth a
   follow-up: `SetDouble` is `*((double*)Address + offset) = value` with **no** dtype guard, so misuse
   *silently corrupts* (writes 8 double-bytes over a narrower element / past the end) where `GetDouble`
   cleanly NREs. This is not a byte-identity issue and is out of interop scope, but `SetDouble` should
   mirror `GetDouble`'s guard.

3. **NaN-producing arithmetic on `float64`/`float32` (minimal inputs).** `0/0`, `inf-inf`, `0*inf`,
   `sqrt(-1)`, `log(-1)`, and all unary/binary/reduction minimal repros are **byte-identical** — both
   engines take the same hardware/`npy_*` path. float64 has **zero** divergences across 34 ops.

### Real byte-identity problems (supported dtypes; value-correct, bit-divergent)

Every one is NumSharp emitting **.NET's negative-signed NaN** where numpy emits the **positive quiet
NaN**:

| Op | Input | NumSharp | numpy | Root cause |
|---|---|---|---|---|
| **`float16 sign`** | `nan` | `0xfe00` | `0x7e00` | Half scalar sign kernel returns `Half.NaN` (0xfe00) |
| **`float16 maximum`** | `(nan, x)` | `0xfe00` | `0x7e00` | Half element-wise `maximum` returns `Half.NaN` |
| **`complex128 sqrt/log`** | component `nan` | real `0xfff8…` | `0x7ff8…` | `NDComplexMath.Hypot`/`HypotInf`/branches return .NET `double.NaN` (0xfff8) where numpy's `npy_hypot`/`npy_csqrt` yield +NaN — e.g. `Sqrt(1+nanj)` → `CsqrtCore(1,nan)` → `Hypot(1,nan)` line 540 `return double.NaN;` flows into the real part |
| **`complex128 reciprocal`** | `1+nanj` | imag `0xfff8…` | `0x7ff8…` | Smith's `nc_recip` produces a .NET-signed NaN component |
| **`float32 sum`** | long mixed `[…inf,-inf,nan,…]` | `0xffc00000` | `0x7fc00000` | SIMD/pairwise **summation order** picks a different NaN operand than numpy's pairwise tree (only appears with enough elements; the 2-element `sum(inf,-inf)` matches) |

Notes that shaped the classification:
- **float64 never diverges**; the problems are float16, complex128, and (order-dependent) float32.
- The complex divergences are **input-specific**, not blanket: `sqrt(nan+1j)` and `exp(1+nanj)` are
  byte-identical, so a naive "canonicalize every NaN component to positive" is **wrong** (it would
  break the agreeing cases). numpy is itself inconsistent — its hardware `0/0` yields `0xFFF8` while
  `npy_hypot` yields `0x7FF8` — so a correct fix must reproduce numpy's per-path NaN, not canonicalize.

## Root cause (one sentence)

Several NumSharp kernels emit a NaN via a **.NET NaN constant/BCL call** (`Half.NaN`=0xfe00,
`double.NaN`=0xfff8 inside `NDComplexMath`, order-dependent float32 SIMD) instead of the exact NaN the
corresponding numpy C code produces at that point — a gap in the otherwise-careful bit-for-bit ports
(`NDComplexMath.cs` is explicitly a port of `npy_csqrt`/`npy_clog`; the ported float32 exp/log/sin/cos/
tanh kernels DO canonicalize NaN and are byte-exact — these paths were simply not covered).

## Recommended fix (follow-up, oracle-gated)

Per-path, not blanket:
- **`NDComplexMath`**: audit each `double.NaN`/`Hypot`/`HypotInf` NaN return against numpy's actual
  output for nan-component inputs (the `byteident`/`nansweep` cases pin the targets), correcting the
  sign where numpy's `npy_hypot`/`npy_csqrt`/`nc_recip` yields +NaN. Verify against the `complex` /
  `decimal_matmul` and unary/binary fuzz tiers.
- **float16 `sign`/`maximum`**: return the numpy-canonical `0x7e00` for a NaN result rather than
  `Half.NaN`.
- **float32 `sum`**: matching numpy's pairwise NaN propagation is order-sensitive; likely accept as a
  documented order-dependent NaN-bit difference (value is NaN either way) unless the reduction is
  reordered to numpy's pairwise tree.

## How it is reproduced

`experimental/adversarial-harness`: `dotnet run -c Release -- byteident` (68 ops, families A/B +
construction) and `dotnet run -c Release -- nansweep` (51 minimal-repro nan ops across f8/f4/f2/c16).
Each prints `NS[dtype]=<hex> | NP[dtype]=<hex>` on any divergence.

## Related

- `docs/bugs/pythonnet-array-interface-validation.md` — the interop defect fixed this branch.
- `Utilities/NDComplexMath.cs` — the complex port (root of the complex NaN-sign divergences).
- CLAUDE.md → "Five float32 unary loops are BIT-EXACT" — the ports that DO canonicalize NaN (0x7fc00000).
