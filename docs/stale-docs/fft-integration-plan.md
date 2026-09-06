# NumSharp FFT Integration — Technical Plan (`np.fft.*`)

Status: **proposal / research** (no FFT code exists in NumSharp yet).
Reference: numpy 2.4.2 (`refs/numpy/numpy/fft/`), pocketfft engine (`github.com/mreineck/pocketfft`).
Author: research pass, 2026-08-14.

---

## 0. TL;DR

NumPy's `numpy.fft` is **four thin Python/C++ layers over one algorithmic engine** (`pocketfft`, a
header-only C++11 library). The public surface is **14 transform functions + 4 helpers**; only **three
1-D transform kernels** (`c2c`, `r2c`, `c2r`) do real work — every 2-D/N-D/Hermitian variant is a pure
composition of the 1-D transform, `roll`, `conjugate`, `swapaxes` and `take`, all of which NumSharp
already has (or nearly has).

**Recommendation: port pocketfft's 1-D engine to pure managed C# inside `NumSharp.Core`** (no native
dependency), exactly the way the team already produced a byte-identical managed GEMM. pocketfft is
portable, algorithmic C++11 (not hand-tuned assembly like OpenBLAS), so a faithful scalar port is
bit-identical to NumPy modulo the same FMA-contraction host-pins already documented for the
`exp`/`log`/`tanh` ports. An optional native `IFftBackend` accelerator package can follow later on the
`IBlasBackend` pattern, but is **not** needed for parity — the managed port *is* NumPy's algorithm.

The single biggest constraint: **NumSharp has only one complex type (`System.Numerics.Complex`, i.e.
complex128) — there is no complex64.** So float32/float16 inputs promote to complex128 (compute in
double), which matches NumPy's long-standing *documented* "promote to double" behaviour but diverges
from NumPy 2.x's newer native-float32 path (which returns complex64). This is the one deliberate
`[Misaligned]` cell and the one open decision worth confirming (§10).

---

## 1. What the FFT is, and how NumPy structures it

The discrete Fourier transform maps `n` samples to `n` frequency bins,
`A_k = Σ_m a_m · exp(-2πi·mk/n)`. The FFT is the O(n log n) algorithm for it. NumPy computes it through
a strict layer cake — understanding the layering is the whole porting story, because **each layer maps
to a different NumSharp home**:

| Layer | File | Responsibility | NumSharp analog |
|-------|------|----------------|-----------------|
| **1. Exports** | `fft/__init__.py` | Re-export names, module docstring | `APIs/np.fft.cs` static class |
| **2. Helpers** | `fft/_helper.py` | `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` — **pure Python**, built on `arange`/`empty`/`roll` | trivial: compose existing `np.*` |
| **3. Python wrapper** | `fft/_pocketfft.py` (64 KB) | axis/`n`/`s`/`norm` normalization; decompose N-D → repeated 1-D; build `hfft`/`ihfft` from `irfft`/`rfft`; pick the gufunc | `Fourier/np.fft.*.cs` (managed dispatch) |
| **4. C++ ufunc glue** | `fft/_pocketfft_umath.cpp` (16 KB) | 5 gufuncs `(n),()->(m)`; the FFTPACK half-complex packing/unpacking; the strided outer-loop driver | `Fourier/PocketFFTDriver.cs` (C# axis driver) |
| **5. Engine** | `pocketfft/pocketfft_hdronly.h` (3 635 lines) | the actual mixed-radix / Bluestein FFT | `Fourier/PocketFFT*.cs` (managed port) |

The crucial architectural fact: **layer 3 already reduces every N-D transform to a sequence of 1-D
transforms along single axes** (`_raw_fftnd` loops `function(a, n=s[ii], axis=axes[ii], ...)`). NumPy's
C++ layer therefore only ever needs a **1-D transform with a strided outer loop** — its fancy vectorized
N-D machinery (`multi_iter`, `general_nd`, SIMD `VTYPE`, the `threading` pool, the DCT/DST classes) is
**dead weight for a `numpy.fft` port** and must not be ported.

---

## 2. The public API surface (18 functions)

From `fft/__init__.py`. All keep NumPy's signatures; `norm ∈ {None,"backward","ortho","forward"}`.

### Standard (complex→complex)
| Function | Signature | Built from |
|----------|-----------|------------|
| `fft`   | `(a, n=None, axis=-1, norm=None, out=None)` | `_raw_fft(..., is_real=False, is_forward=True)` → `c2c` |
| `ifft`  | same | `_raw_fft(..., is_forward=False)` → `c2c` |
| `fft2`  | `(a, s=None, axes=(-2,-1), norm=None, out=None)` | `_raw_fftnd(..., fft)` |
| `ifft2` | same | `_raw_fftnd(..., ifft)` |
| `fftn`  | `(a, s=None, axes=None, norm=None, out=None)` | `_raw_fftnd(..., fft)` |
| `ifftn` | same | `_raw_fftnd(..., ifft)` |

### Real (`rfft`: real→complex half-spectrum; `irfft`: complex half→real)
| Function | Signature | Built from |
|----------|-----------|------------|
| `rfft`   | `(a, n=None, axis=-1, norm=None, out=None)` | `_raw_fft(..., is_real=True, is_forward=True)` → `r2c`; output length `n//2+1` |
| `irfft`  | same | `_raw_fft(..., is_real=True, is_forward=False)` → `c2r`; output length `n` (default `2*(m-1)`) |
| `rfft2`/`irfft2` | `(a, s, axes=(-2,-1), ...)` | `rfftn`/`irfftn` |
| `rfftn`  | `(a, s=None, axes=None, ...)` | `rfft` on last axis, then `fft` on the rest |
| `irfftn` | same | `ifft` on all-but-last, then `irfft` on last axis |

### Hermitian (spectrum is real → signal is Hermitian) — **pure compositions, no new kernel**
| Function | Definition (verbatim from `_pocketfft.py`) |
|----------|--------------------------------------------|
| `hfft`  | `irfft(conjugate(a), n, axis, norm=_swap_direction(norm))` |
| `ihfft` | `conjugate(rfft(a, n, axis, norm=_swap_direction(norm)), out=out)` |

### Helpers (`_helper.py`) — **pure Python, trivial ports**
| Function | Definition |
|----------|------------|
| `fftshift(x, axes=None)` | `roll(x, [dim//2 for dim in shape], axes)` |
| `ifftshift(x, axes=None)` | `roll(x, [-(dim//2) ...], axes)` |
| `fftfreq(n, d=1.0)` | `[0,1,…,n/2-1,-n/2,…,-1] / (d·n)` via `arange` + slice-assign |
| `rfftfreq(n, d=1.0)` | `[0,1,…,n/2] / (d·n)` via `arange` |

### Dependency graph (what actually needs a kernel)
```
fftfreq, rfftfreq, fftshift, ifftshift   → arange, empty, roll         [NO FFT kernel]
fft, ifft                                → c2c   (complex engine)
rfft                                     → r2c   (real engine)
irfft                                    → c2r   (real engine)
hfft   = irfft ∘ conjugate ∘ swap_norm   → c2r
ihfft  = conjugate ∘ rfft ∘ swap_norm    → r2c
fft2/ifft2/fftn/ifftn                    → repeated fft/ifft per axis
rfft2/rfftn                              → rfft (last axis) + fft (others)
irfft2/irfftn                            → ifft (others) + irfft (last axis)
```
So the **entire kernel scope is `c2c` + `r2c` + `c2r` on one axis of an N-D array.**

---

## 3. The original engine source: pocketfft

- **Upstream:** `https://github.com/mreineck/pocketfft` (Martin Reinecke, Max-Planck-Society).
  3-clause BSD (`pocketfft/LICENSE.md`) — compatible with NumSharp's Apache-2.0.
- **Vendored copy already in-tree:** `refs/numpy/numpy/fft/pocketfft/pocketfft_hdronly.h`
  (git submodule pinned at `33ae5dc9`, `release_for_eigen-24-g33ae5dc`).
- **Standalone reference clone (this pass):** cloned to the session scratchpad
  (`…/scratchpad/pocketfft-ref/`, latest `main`, 124 KB header). Use the **in-tree pinned copy** as the
  port's source of truth (it is the exact algorithm numpy 2.4.2 ships); keep the standalone clone only
  for cross-checking upstream changes.

### Engine anatomy (`pocketfft_hdronly.h`, 3 635 lines)
| Component | Lines | Port? | Notes |
|-----------|-------|:-----:|-------|
| `sincos_2pibyn<T>` | 299–372 | ✅ | Accurate twiddle factors — angles reduced to `[0,π/4]`, `2√n` trig calls. **Load-bearing for bit-parity.** |
| `util` (good_size, largest_prime_factor, cost_guess) | 373–530 | ✅ | plan selection + Bluestein sizing |
| `cfftp<T0>` | 878–1603 | ✅ | **complex mixed-radix FFTPACK** — codelets radix 2,3,4,5,7,8,11 + generic. The heart. |
| `rfftp<T0>` | 1604–2412 | ✅ | **real** mixed-radix (codelets 2,3,4,5 + generic) |
| `fftblue<T0>` | 2413–2514 | ✅ | **Bluestein's algorithm** for large prime factors (chirp-z) |
| `pocketfft_c<T0>` / `pocketfft_r<T0>` | 2515–2589 | ✅ | plan dispatch: mixed-radix if `largest_prime_factor(n)² ≤ n`, else cost-compare vs Bluestein |
| `arr<T>`, `cmplx<T>` | 184–298 | ✅(thin) | tiny aligned buffer + complex struct → C# `Complex[]`/pooled buffers |
| `threading::*` | 569–877 | ❌ | numpy builds with `POCKETFFT_NO_MULTITHREADING`; the 1-D path is single-threaded |
| `multi_iter`, `cndarr`, `ndarr`, `general_nd`, `VTYPE`/SIMD, `ExecC2C`/`ExecHartley`/`ExecDcst`/`ExecR2R` | 2876–3600 | ❌ | pocketfft's own N-D + SIMD driver — **replaced by NumSharp's axis driver**; numpy.fft never calls the N-D public entry points for these |
| `T_dct1/T_dst1/T_dcst23/T_dcst4`, `dct`/`dst`/Hartley public API | 2590–2875, 3452–3600 | ❌ | DCT/DST/Hartley — **not part of `numpy.fft`** (that's `scipy.fft`) |

**Net port scope ≈ 1 500–1 800 of the 3 635 lines** — the scalar 1-D engine only.

### Why a scalar port is bit-identical to NumPy
pocketfft's SIMD (`VTYPE`) vectorizes **across `vlen` independent outer transforms** — each lane runs
the *identical* scalar arithmetic on its own 1-D signal. So per-transform results are the same whether
computed scalar or SIMD. A pure-scalar C# port therefore matches native pocketfft (hence numpy)
bit-for-bit, **except** where the C++ compiler contracts `a*b+c` into an FMA. That is the exact
host-pin class already handled for `exp`/`log`/`sin`/`cos`/`tanh` (see CLAUDE.md → "Host pin"), and the
mitigation is the same: spell the butterflies with explicit `Math.FusedMultiplyAdd` where numpy's
Windows MSVC build contracts, and pin discriminating inputs in the corpus.

### What drives the C++ inner loops (layer 4, to reimplement in C#)
The 5 gufuncs in `_pocketfft_umath.cpp` share one shape — `fft_loop`, `rfft_impl`, `irfft_loop`:
1. Outer loop over `n_outer` independent transforms (the "all-but-transform-axis" product).
2. `copy_input` — gather `min(nin, n)` strided elements into a contiguous buffer, zero-pad the rest
   (this is where `n`-truncation/padding physically happens).
3. `plan->exec(buffer, fct, direction)` — the 1-D transform, in place, scaled by `fct`.
4. `copy_output` — scatter the contiguous buffer back to the (possibly strided) output.
5. **Real packing convention (must be reproduced exactly):** pocketfft's real transforms use FFTPACK
   half-complex order `R0,R1,I1,…,Rn-1,In-1,Rn[,In]`. `rfft_impl` writes real data offset-by-one so it
   only has to move `R0` and set `I0=0` (`op_or_buff[0] = op_or_buff[0].imag()`); `irfft_loop` unpacks
   the mirror. Getting this packing wrong is the classic rfft/irfft bug.

`out=` aliasing, `sf==0` fast path, and the "buffer only if output non-contiguous" optimization are
present but are optimizations, not semantics — the C# driver can start with "always buffer" and add the
in-place fast path later.

---

## 4. Normalization, `n`/`s`, axis semantics (layer 3 detail to port verbatim)

- **`fct` (norm factor)** is computed once in `_raw_fft` and passed into the engine, which multiplies
  every output by it:
  - forward: `backward/None → 1`, `ortho → 1/√n`, `forward → 1/n`.
  - inverse: `norm = _swap_direction(norm)` first (`backward↔forward`, `ortho↔ortho`), so `ifft`
    default divides by `n`.
  - `fct` uses `real_dtype = result_type(a.real.dtype, 1.0)` to avoid precision loss — in C# compute
    `1/√n` / `1/n` in the output's real precision (double for the complex128 path).
- **`n` (1-D):** default `a.shape[axis]`; `n<1` → `ValueError("Invalid number of FFT data points…")`.
  `n<size` truncates, `n>size` zero-pads — both realised by `copy_input`'s `min(nin,n)` + zero fill.
  For `irfft`, default output `n = 2*(m-1)` where `m` is the input length along axis.
- **`s`/`axes` (N-D):** `_cook_nd_args` — if `s` given without `axes`, axes default to the **last
  `len(s)`** axes (with a NumPy-2.0 DeprecationWarning); `s[i]==-1` means "use full input length"; for
  `irfftn`, `invreal=1` sets `s[-1] = (a.shape[axes[-1]]-1)*2` when `s` omitted. `len(s)!=len(axes)` →
  `ValueError("Shape and axes have different lengths.")`.
- **axis normalization:** `normalize_axis_index(axis, a.ndim)` → NumSharp's `AxisError` machinery
  (already ported repo-wide).
- **`out=`:** optional pre-allocated output; shape must equal `a.shape` with `axis` replaced by `n_out`,
  else `ValueError("output array has wrong shape.")`. Can be deferred to a later milestone.

---

## 5. NumSharp integration architecture

### 5.1 Recommended: managed port in `NumSharp.Core` (primary path)

Rationale (aligned with CLAUDE.md's stated identity):
- "NumSharp.Core is 100 % managed C# — optional packages are the only native path." FFT-by-default
  should therefore be managed, giving every user FFT with zero native dependency.
- pocketfft is **portable algorithmic C++11**, not arch-specific assembly. Unlike OpenBLAS (62 MB of
  DYNAMIC_ARCH kernels whose *bits* depend on CPU/thread-count), a managed pocketfft port is
  deterministic and reproduces numpy bit-for-bit modulo FMA host-pins.
- Precedent: the memory note "openblas-sgemm-from-scratch — pure-C#/IL GEMM byte-identical to NumPy"
  shows the team already ports numeric kernels to managed code for parity.
- FFT has no "62 MB binary per RID" delivery problem, so the OpenBLAS bundling machinery is unwarranted.

### 5.2 Proposed file layout (`src/NumSharp.Core/Fourier/`)
```
Fourier/
  np.fft.cs                 static np.fft facade (namespace/class holding all 18 entry points)
  np.fft.Standard.cs        fft, ifft, fft2, ifft2, fftn, ifftn
  np.fft.Real.cs            rfft, irfft, rfft2, irfft2, rfftn, irfftn
  np.fft.Hermitian.cs       hfft, ihfft
  np.fft.Helper.cs          fftfreq, rfftfreq, fftshift, ifftshift
  np.fft.RawFft.cs          _raw_fft, _raw_fftnd, _cook_nd_args, _swap_direction  (layer-3 port)
  PocketFFT.Complex.cs      cfftp port  (mixed-radix c2c)
  PocketFFT.Real.cs         rfftp port  (r2c / c2r)
  PocketFFT.Bluestein.cs    fftblue port
  PocketFFT.Twiddle.cs      sincos_2pibyn + util (good_size / prime factors / cost_guess)
  PocketFFT.Plan.cs         pocketfft_c / pocketfft_r dispatch + a per-length plan cache
  PocketFFTDriver.cs        the strided axis driver (layer-4 port; copy_input/exec/copy_output)
```
Exposed as `np.fft.fft(...)` etc. Decide facade shape in §10 (nested `np.fft` class vs `np.fft_*`).

### 5.3 The axis driver — reuse existing machinery
`_pocketfft_umath.cpp`'s outer loop is exactly "iterate all axes except the transform axis, run a 1-D
kernel per lane." NumSharp already has this: **`AxisSort.DriveAllButAxis`** (an NDIter *IterAllButAxis*
drive used by `sort`/`partition`). `PocketFFTDriver` mirrors it — for each lane, `copy_input` gathers
the strided/short/padded signal into a pooled contiguous `Complex[]`/`double[]`, calls
`plan.Exec(buffer, fct, forward)`, then `copy_output` scatters back. This gives correct behaviour on
every layout in the DOD (contiguous, strided, transposed, broadcast-read, sliced-offset) for free.

### 5.4 Plan cache
pocketfft keeps a small per-length plan cache (twiddle tables + factorization). Port a bounded
`ConcurrentDictionary<(int len, bool real), Plan>` so repeated same-length transforms (the common case,
and every N-D transform along a fixed axis) reuse twiddles. Twiddle precompute is the dominant setup
cost, so this matters for the N-D loop.

### 5.5 Optional future accelerator (NOT in scope now)
A later `NumSharp.Interop.PocketFFT` (or FFTW/MKL) package could add an `IFftBackend` seam on
`TensorEngine` mirroring `IBlasBackend` — `TryC2C`/`TryR2C`/`TryC2R`, defaulting to `false`, opt-in by
package reference. **Deferred:** the managed port already delivers parity and adequate speed; a native
FFT backend is a performance option, not a correctness requirement. Note the seam now only so the
managed API doesn't foreclose it.

---

## 6. Prerequisite gap analysis (NumSharp today)

| Needed by | Prerequisite | Status in NumSharp |
|-----------|-------------|--------------------|
| helpers | `np.roll`, `np.arange`, `np.empty`, slice-assign | ✅ present |
| N-D / axes | `np.swapaxes`, `np.take`, `AxisError`/normalize-axis | ✅ present |
| norm | `np.result_type`, `np.sqrt`, `np.reciprocal` | ✅ present |
| **hfft/ihfft** | **`np.conjugate` / `np.conj`** | ⚠️ **only `internal TensorEngine.Conjugate`** (used by vdot) — must be promoted to a public `np.conjugate`/`conj` (its own small pass) |
| output dtype | build/allocate a `Complex` (complex128) NDArray, complex scalar-multiply | ✅ Complex is a first-class dtype (`sort_complex`, `vdot`, complex casts all exist) |
| ergonomics (not strictly required) | `np.real`, `np.imag`, `np.angle` | ❌ absent — worth a companion mini-pass; `hfft`/`ihfft` only need `conjugate`, but users expect `real`/`imag`/`angle` next to FFT |

**Gate on FFT: land a public `np.conjugate`/`conj` first** (thin wrapper over the existing internal
`Conjugate` + a real-dtype identity fast path + `out=`). `real`/`imag`/`angle` are recommended
companions but can ship in the same or an adjacent pass.

---

## 7. Numerical parity strategy

1. **Bit-parity target, FMA host-pins accepted.** Port the butterflies and twiddle recurrences with the
   *same operation order* as pocketfft. Where numpy's win-amd64 MSVC build contracts to FMA, spell
   `Math.FusedMultiplyAdd` and pin the discriminating inputs — identical to the documented
   `exp`/`log`/`tanh` host-pin approach. Expect **exact** agreement on most inputs, ≤1–2 ULP on FMA-
   sensitive ones.
2. **float32/float16 → complex128 (the one deliberate divergence).** NumSharp has no complex64, so real
   float32 input is computed in double and returned as complex128. This matches numpy's *historically
   documented* promote-to-double behaviour (`fft/__init__.py` still says "numpy.fft promotes float32 …
   to … complex128") but diverges from numpy 2.x's newer native single-precision path (which returns
   complex64). Record as `[Misaligned]` on the float32/float16 **dtype** cell only — values are the
   correctly-rounded double result. (Full parity would require adding a complex64 dtype + porting
   pocketfft's float engine — a much larger, separate effort; see §10.)
3. **double / complex128:** target exact bit-parity with numpy 2.4.2.
4. **Edge cases to pin** (from `test_pocketfft.py`): `n<1` error text; `n` truncation/padding;
   round-trip `ifft(fft(x)) == x`; the four `norm` values incl. inverse swap; `rfft` even/odd length
   packing; `irfft` default `2*(m-1)`; the `s`/`axes` deprecation and `-1` sentinel; `out=` wrong-shape
   error; Parseval/norm-preservation for `ortho`.

---

## 8. Phased implementation plan

**M0 — Prerequisite: `np.conjugate`/`conj`** (+ optional `np.real`/`imag`/`angle`). Own pass, own tests.

**M1 — Helpers (no engine):** `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`. Pure compositions;
immediately oracle-testable. Cheap, high-value, unblocks downstream users' frequency-axis code.

**M2 — Complex 1-D engine (`c2c`):** port `sincos_2pibyn` + `util` + `cfftp` (radix 2/3/4/5/7/8/11 +
generic) + `fftblue` + `pocketfft_c` + plan cache + `PocketFFTDriver`. Wire `fft`/`ifft`. This is the
bulk of the work and the parity-critical milestone.

**M3 — Real engine (`r2c`/`c2r`):** port `rfftp` + `pocketfft_r`; reproduce the FFTPACK half-complex
packing exactly. Wire `rfft`/`irfft`, then `hfft`/`ihfft` (compositions over M0's `conjugate`).

**M4 — N-D wrappers:** `_cook_nd_args`/`_raw_fftnd`, then `fft2/ifft2/fftn/ifftn` and
`rfft2/irfft2/rfftn/irfftn`. Pure composition over M2/M3 — mostly axis/`s` bookkeeping.

**M5 — `out=` + fast paths:** `out=` argument + wrong-shape validation; the in-place (`step_out ==
itemsize`) and `sf==0` non-buffered fast paths; benchmark vs numpy and tune the plan cache.

**M6 — Oracle corpus + docs:** add an `fft` tier to the differential-fuzz corpus (§9); write the
CLAUDE.md "Supported np.* APIs → FFT" section and a `docs/FFT_PARITY.md` mirroring `GEMM_PARITY.md`.

---

## 9. Testing / oracle strategy

- **Differential-fuzz tier** (`test/oracle/gen_oracle.py` mode `fft` → `Fuzz/corpus/fft*.jsonl`, replayed
  by `FuzzCorpusTests`): generate NumPy 2.4.2 outputs for `fft/ifft/rfft/irfft/hfft/ihfft` and the
  N-D variants across dtype {float64, complex128, + float32/complex64 recorded as promoted}, `n`
  {default, shorter, longer}, `norm` {all four}, axis {all}, and layouts {C, F, strided, reversed,
  transposed}. Bit-compare (complex128 exact; float32-input cells flagged `[Misaligned]`).
- **Unit tests** ported from `numpy/fft/tests/test_pocketfft.py` (33 cases: identities, out-argument,
  bad-out, per-`norm` inverse, `rfft` even/odd, `s`/`axes` edge cases, norm-preservation).
- **Metamorphic (oracle-free):** `ifft∘fft == id`, `irfft∘rfft == id` (to `n`), `fftshift∘ifftshift ==
  id`, Parseval under `ortho`. These belong in `MetamorphicTests.cs`.
- **DOD sweep:** all-layouts × the dtype matrix (float64/complex128 native; float16/float32 promoted;
  integer/bool promoted-to-double; Decimal → documented unsupported/NSE, since pocketfft is float-only).

---

## 10. Open decisions (need confirmation before M2)

1. **float32 dtype parity.** Accept the complex128 promotion (recommended — matches the still-published
   NumPy docstring, keeps scope sane, one `[Misaligned]` cell), **or** commit to adding a `complex64`
   dtype + porting pocketfft's float engine for exact numpy-2.x parity (a large, separate initiative)?
2. **API facade shape.** Nested `np.fft.fft(...)` (a `fft` static holder class exposed as `np.fft`) to
   match Python's `np.fft.*` verbatim — recommended — vs flat `np.fft_*`? The nested form ports user
   code unchanged; confirm the naming/access pattern the repo prefers.
3. **Managed-only now, native `IFftBackend` later?** Confirm the managed port is the accepted default
   (per §5.1) and the native accelerator is explicitly deferred.
4. **Scope of the M0 complex-accessor pass.** Just `conjugate`/`conj` (minimum for FFT), or bundle
   `real`/`imag`/`angle` in the same pass (recommended for a complete complex-support story)?

---

## Appendix A — exact NumPy call sites worth reading during the port
- Norm + gufunc selection: `_pocketfft.py:58-101` (`_raw_fft`), `:104-113` (`_swap_direction`).
- N-D decomposition: `_pocketfft.py:704-748` (`_cook_nd_args`, `_raw_fftnd`), `:1385-1390` (rfftn),
  `:1604-1609` (irfftn).
- Hermitian: `_pocketfft.py:627-629` (hfft), `:699-701` (ihfft).
- Strided driver + real packing: `_pocketfft_umath.cpp:87-293` (`fft_loop`, `rfft_impl`, `irfft_loop`).
- gufunc registration + type loops: `_pocketfft_umath.cpp:295-388`.
- Plan dispatch (mixed-radix vs Bluestein): `pocketfft_hdronly.h:2515-2589`.
- Twiddles: `pocketfft_hdronly.h:299-372`. Codelets: `cfftp` 878-1603, `rfftp` 1604-2412, `fftblue`
  2413-2514.
