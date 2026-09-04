# `np.fft.*` — a managed pocketfft port, bit-identical to NumPy 2.4.2

> **✅ IMPLEMENTED.** The whole `numpy.fft` module (18 public functions) computes in pure managed
> C# inside `NumSharp.Core` (`src/NumSharp.Core/Fourier/`) — no native dependency, no P/Invoke,
> nothing to install. The engine is a scalar-double port of **pocketfft**, NumPy's own vendored FFT
> library, so the double/complex128 path is **bit-for-bit identical to NumPy 2.4.2** rather than
> merely close. Gate: the portable `fft.jsonl` differential-fuzz tier (2,000 cases, floor 1,700) +
> 330 `Fourier`-namespace unit tests.
>
> **One deliberate divergence, now genuinely dtype-only:** NumSharp has a single complex type
> (`System.Numerics.Complex` = complex128) and **no complex64**, so a float32/float16 input is returned
> as complex128/float64 where NumPy 2.x returns complex64/float32/float16. The **values are bit-identical
> to NumPy 2.4.2** (verified 902/902 across every transform × size × norm × layout), not merely close —
> NumSharp reproduces NumPy's exact per-loop precision (see §7). Only the result *dtype* differs; closing
> that is issue [#569](https://github.com/SciSharp/NumSharp/issues/569) (a complex64 dtype), not FFT work.

Status date: 2026-08-14 · Issue:
[SciSharp/NumSharp#114](https://github.com/SciSharp/NumSharp/issues/114) · Plan:
`docs/plans/fft-integration-plan.md`

---

## 1. What `numpy.fft` is, and why the port is small

The discrete Fourier transform maps `n` samples to `n` frequency bins,
`A_k = Σ_m a_m · exp(-2πi·mk/n)`; the FFT is the O(n log n) algorithm for it. NumPy structures the
whole module as **four thin layers over one engine** (`pocketfft`), and only **three 1-D kernels**
(`c2c`, `r2c`, `c2r`) do real work — every 2-D/N-D/Hermitian variant is a pure composition of a 1-D
transform with `roll`/`conjugate`/`swapaxes`, all of which NumSharp already had:

| NumPy layer | File | NumSharp home |
|---|---|---|
| 1. Exports | `fft/__init__.py` | `Fourier/np.fft.cs` — the `np.fft` facade |
| 2. Helpers | `fft/_helper.py` | `Fourier/np.fft.Helper.cs` — pure `np.*` compositions |
| 3. Python wrapper | `fft/_pocketfft.py` | `Fourier/np.fft.{Standard,Real,Hermitian,RawFft}.cs` |
| 4. C++ ufunc glue | `fft/_pocketfft_umath.cpp` | `Fourier/PocketFFTDriver.cs` — the strided 1-D driver |
| 5. Engine | `pocketfft/pocketfft_hdronly.h` | `Fourier/PocketFFT.{Twiddle,Complex,Real,Bluestein,Plan}.cs` |

Layer 3 already reduces every N-D transform to a sequence of single-axis 1-D transforms
(`_raw_fftnd`), so the entire kernel scope is `c2c + r2c + c2r` on **one axis of an N-D array**.
pocketfft's own N-D machinery, SIMD `VTYPE`, threading pool and DCT/DST classes are dead weight for a
`numpy.fft` port and were **not** ported.

### The 18 functions (all present, `np.fft.fft(x)` verbatim)

| Group | Functions | Built from |
|---|---|---|
| Standard (c↔c) | `fft` `ifft` `fft2` `ifft2` `fftn` `ifftn` | `c2c`; the N-D forms compose 1-D `fft`/`ifft` |
| Real | `rfft` `irfft` `rfft2` `irfft2` `rfftn` `irfftn` | `r2c`/`c2r`; N-D = `rfft` on last axis + `fft` on the rest |
| Hermitian | `hfft` `ihfft` | pure compositions: `hfft = irfft∘conjugate∘swap_norm`, `ihfft = conjugate∘rfft∘swap_norm` |
| Helpers | `fftfreq` `rfftfreq` `fftshift` `ifftshift` | pure `arange`/`roll` compositions — **no engine** |

The facade is a nested module (`np.fft` → `FourierModule`), the `np.random` house shape, so Python
code ports unchanged.

---

## 2. Why a managed port — and why it is bit-identical

`NumSharp.Core` is 100 % managed C#; FFT-by-default should be too, so every user gets it with zero
native dependency (unlike `np.dot`'s OpenBLAS, whose *bits* depend on CPU/thread-count — see
`GEMM_PARITY.md`). pocketfft is **portable algorithmic C++11**, not arch-specific assembly, so a
faithful scalar port reproduces NumPy's arithmetic deterministically. Two facts make it *bit*-exact,
not just close, and both are load-bearing:

1. **NumPy's win-amd64 build is already scalar.** pocketfft leaves `POCKETFFT_NO_VECTORS` defined
   under MSVC, so `VLEN == 1` and the whole engine runs one 1-D transform at a time. (Its SIMD path
   vectorizes *across independent outer transforms* anyway — each lane runs the identical scalar
   arithmetic — so scalar and SIMD agree per transform regardless.) A pure-scalar C# port therefore
   matches the shipping wheel's per-transform arithmetic.
2. **The twiddle trig is the same CRT.** `sincos_2pibyn` reduces every angle into `[0, π/4]` and
   calls `std::cos`/`std::sin`, which on Windows is `ucrtbase` — the *same* library .NET's
   `Math.Cos`/`Math.Sin` call. So the twiddle tables are bit-identical.

**No explicit FMA is needed.** The GEMM port had to spell `Math.FusedMultiplyAdd` where MSVC
contracts `a*b+c`; the pocketfft butterflies are transcribed in pocketfft's **exact operation
order** and the tier passes bit-exact without any FMA intervention — which is why `fft.jsonl` is a
**portable** FuzzMatrix tier, not a host-pinned one (contrast `matmul_parity.host.jsonl`). The parity
basis is the scalar op-order plus the shared Windows CRT trig, as documented in the engine's own
header comments.

---

## 3. The engine port (`PocketFFT.*.cs`)

Managed port of pocketfft as vendored by NumPy 2.4.2 (pinned commit `33ae5dc9`,
`src/pocketfft/pocketfft_hdronly.h`). Double engine only (`T0 == double`); operation order preserved
verbatim.

| File | Ports (pocketfft) | Contents |
|---|---|---|
| `PocketFFT.Twiddle.cs` | `cmplx<T0>`, `sincos_2pibyn<T>`, `util` (lines 236–431) | the `Cmplx` struct (bit-reinterpretable with `System.Numerics.Complex`), `special_mul`, accurate twiddles, `largest_prime_factor`/`cost_guess`/`good_size` |
| `PocketFFT.Complex.cs` | `cfftp<T0>` (878–1598) | complex mixed-radix FFTPACK: codelets radix **2/3/4/5/7/8/11 + generic**, `factorize`, `comp_twiddle`, the two-buffer ping-pong with `fct` scaling |
| `PocketFFT.Real.cs` | `rfftp<T0>` (1604–2407) | real mixed-radix: `radf2/3/4/5 + radfg` (forward r2hc), `radb2/3/4/5 + radbg` (backward hc2r), in FFTPACK half-complex order `R0,R1,I1,…` |
| `PocketFFT.Bluestein.cs` | `fftblue<T0>` (2413–2509) | Bluestein's algorithm (chirp-z) for lengths with large prime factors — a length-`n2` complex convolution via an inner `cfftp` plan |
| `PocketFFT.Plan.cs` | `pocketfft_c`/`pocketfft_r` dispatch (2515–2583) | mixed-radix when `largest_prime_factor(n)² ≤ n`, else cost-compare vs Bluestein; + a bounded plan cache |

**Plan cache.** A bounded `ConcurrentDictionary<long, Plan>` (`MaxEntries = 64`, cleared wholesale
on overflow so a program touching many lengths cannot leak). Plans are immutable and deterministic,
so sharing them across same-length transforms is safe and **cannot change bits** — it only saves the
twiddle setup. NumPy's own build sets `POCKETFFT_CACHE_SIZE == 0` (a fresh plan per call), which is
why the cache is a pure-performance addition rather than a parity concern.

---

## 4. The strided driver (`PocketFFTDriver.cs`)

Port of `_pocketfft_umath.cpp`'s `fft_loop`/`rfft_impl`/`irfft_loop`. For each 1-D lane along the
transform axis it runs NumPy's three steps:

1. **`copy_input`** — gather `min(nin, n)` strided elements into a contiguous pooled buffer and
   zero-pad the rest (this is where `n`-truncation/padding physically happens).
2. **`plan.Exec(buffer, fct, direction)`** — the 1-D transform, in place, scaled by the norm factor.
3. **`copy_output`** — scatter the buffer back to the (possibly strided) output.

The all-but-axis walk is an explicit stride **odometer** (rightmost non-axis dim fastest, mirroring
`AxisSort.DriveAllButAxis`), so every layout in the DOD is handled directly by reading the operand's
own `Shape.strides`/`offset` — **contiguous, strided, transposed, broadcast-read, reversed and
sliced-offset** views all work with no materialization.

**The FFTPACK half-complex packing is the classic rfft/irfft trap, and is reproduced exactly.**
`RunR2C` writes `R0` with `I0 = 0`, then `(R_k, I_k)` pairs, then the Nyquist real term for even `n`;
`RunC2R` unpacks the mirror image, zero-filling any half-spectrum bins the (possibly short) input did
not supply. Getting this convention wrong is the usual real-FFT bug; it is pinned by the round-trip
and packing tests.

**Input coercion & output dtype** are resolved here per transform kind: c2c/irfft read complex128,
rfft reads double; forward/complex → complex128 output, irfft/hfft → float64 output. A non-matching
input dtype is `astype`-coerced (float32/float16 → double — §7).

---

## 5. Layer-3 semantics ported verbatim (`np.fft.RawFft.cs`)

`RawFft` reproduces `_raw_fft`'s front matter and `RawFftNd`/`CookNdArgs` reproduce `_raw_fftnd`/
`_cook_nd_args`. All of the following are probed against NumPy 2.4.2:

- **Norm factor `fct`** (computed in double, the output's real precision): forward `backward/None →
  1`, `ortho → 1/√n`, `forward → 1/n`; the **inverse swaps the norm first** (`_swap_direction`), so
  `ifft` default divides by `n`.
- **`n` (1-D):** default `a.shape[axis]`; `n < 1` → `ValueError("Invalid number of FFT data points
  ({n}) specified.")`; `n < size` truncates, `n > size` zero-pads. `irfft`/`hfft` default output
  `n = 2*(m-1)`; `rfft` output length is `n//2 + 1`.
- **`s`/`axes` (N-D):** `s`-without-`axes` defaults to the **last `len(s)`** axes; `s[i] == -1` means
  "use the full input length"; `irfftn` sets the shapeless last axis to `2*(m-1)`; `len(s) !=
  len(axes)` → `ValueError("Shape and axes have different lengths.")`. (The NumPy-2.0 deprecation
  *warnings* are intentionally not modelled — NumSharp does not surface Python warnings.)

### Error taxonomy (the parts that are easy to get subtly wrong)

NumPy raises **different exceptions for the same-looking mistake depending on which line hits first**,
and the port reproduces the ordering and the verbatim text:

| Situation | NumPy / NumSharp |
|---|---|
| bad `norm`, forward transform | `ValueError("… should be \"backward\",\"ortho\" or \"forward\".")` — **no** space after the first comma |
| bad `norm`, inverse / `hfft`/`ihfft` | the `_swap_direction` `KeyError` path — the **same text with a space** after `"backward,"` |
| out-of-range `axis`, `n` omitted | `IndexError("tuple index out of range")` — the `a.shape[axis]` tuple subscript fires before axis-normalization |
| out-of-range `axis`, `n` given | `AxisError` reporting the **original** axis |
| out-of-range `axis`, N-D with `s=None` | `np.take` message: `"index {i} is out of bounds for axis 0 with size {ndim}"` |
| `rfft`/`ihfft` of a **complex** input | `TypeError("ufunc 'rfft_n_even'/'rfft_n_odd' not supported …")` — the ufunc is chosen by input **parity**, and this fires only after `n`/norm/axis/out have all validated |
| bad `out` shape | `ValueError("output array has wrong shape.")` |
| N-D over an empty axis set (0-d input / explicit empty `axes`) | `IndexError("list index out of range")` — NumPy's leaked `s[-1]`/`axes[-1]` subscript |

### `out=` — same_kind casting

`out=` joins on the loop dtype: an exact match writes in place through the out's own strides (any
layout). NumPy's ufunc `out=` additionally accepts any **same_kind** cast *from* the loop output —
e.g. `irfft`'s float64 into a complex128 (imag = 0) or float32 `out` — which the driver reproduces
(`NDIterCasting.CanCast` → compute in the loop dtype → `np.copyto` into `out`), with the verbatim
ufunc cast-rejection message otherwise. The N-D composition wrappers (`ifft2`/`irfft2`) accept `out`
for signature parity but do **not** thread it through — exactly as NumPy passes `out=None` internally.

### Helpers

`fftfreq`/`rfftfreq` are float64 generators built on `arange`/`concatenate`; their argument order is
NumPy's probed order (`1.0/(n*d)` divide-by-zero **first**, then the Array-API `device` check, then
the negative-length rule — which `rfftfreq` lacks, so `rfftfreq(-5)` is `[]`). A non-integer `n`
raises `ValueError("n should be an integer")` before any other check. `fftshift`/`ifftshift` are
dtype-preserving cyclic rolls by `±(dim//2)` per axis, validating each axis up front the way NumPy
indexes `x.shape[ax]` (so an out-of-range axis is `IndexError("tuple index out of range")`, not
`AxisError`); a 0-d input is a no-op copy (`[Misaligned]` — NumPy leaks a roll-unpack `ValueError`).

### Companion complex accessors (the M0 prerequisite, now public)

The Hermitian transforms need a public conjugate, and users expect the rest next to FFT, so this pass
also promoted/added `np.conjugate`/`np.conj` (a full ufunc with `out=`/`where=`/`dtype=`), `np.real`,
`np.imag` and `np.angle(z, deg=false)` — in `src/NumSharp.Core/Math/np.{conjugate,real,imag,angle}.cs`.

---

## 6. Scope (dtypes)

| Input | Result | Parity |
|---|---|---|
| `float64` | complex128 (fwd/complex) · float64 (irfft/hfft) | **contractual — bit-exact vs NumPy 2.4.2** |
| `complex128` | complex128 | **contractual — bit-exact** |
| `int*` / `bool` | promoted to double → complex128 / float64 | **contractual — bit-exact** (NumPy promotes identically) |
| `float32` / `float16` | complex128 / float64 (NumPy: complex64 / float32/float16) | **values bit-exact; dtype-only divergence** — see §7 |
| `Decimal` | — | pocketfft is float-only; Decimal has no NumPy FFT dtype |

---

## 7. The one deliberate divergence — no complex64 (values are bit-exact)

NumSharp has exactly one complex type, `System.Numerics.Complex` (complex128), and no complex64. So a
**float32/float16** input is returned as **complex128** (fft/rfft/ihfft and the N-D forms) or
**float64** (irfft/hfft), where NumPy 2.x returns complex64/float32/float16. **The VALUES are
bit-identical to NumPy 2.4.2** — verified 902/902 over every transform × {radix, radfg, Bluestein}
size × {backward, ortho, forward} norm × {1-D, N-D, truncate, zero-pad} — so the only divergence is
the result *dtype*.

**This required reproducing NumPy's actual per-loop precision, which is subtler than "promote to
double".** NumPy's fft ufuncs register a single (`Ff->F` / `ff->F`) and a double (`Dd->D` / `dd->D`)
loop, and the loop it picks — hence whether the transform runs in single or double — depends on the
operand shapes, NOT a blanket promotion:

- **A real float32/float16 input to fft/ifft/irfft** must be promoted to complex, and NumPy promotes it
  to **complex128** → the **double** loop → the double transform, cast (rounded) to the complex64/
  float32/float16 output. NumSharp does the same: the existing double engine, then a round-to-result-
  precision pass (`PocketFFTDriver.RoundInPlace`).
- **`rfft` of a float32 real input**, and **fft/ifft/irfft of a complex64-precision operand** (the N-D
  intermediates), already match a *single* loop without a real→complex128 promotion → the **single**
  transform — but **only when the normalization factor is a float** (ortho/forward norm). For
  backward/None the Python `fct` is the **int `1`**, which resolves NumPy's ufunc to the **double**
  loop, so those cells are double + round too. NumSharp mirrors this exactly via a genuine
  **single-precision pocketfft engine** (`PocketFFT.{Complex,Real,Bluestein}.Single.cs` +
  `PocketFFT.Plan.Single.cs`), gated in `PocketFFTDriver.Execute` on `floatPrec && !effNormUnity &&
  (complex operand || rfft-float32)`. Its butterflies run in `float`; its twiddle table is the double
  `SinCos2PiByN` narrowed at lookup (`TwiddleF.At`, exactly `sincos_2pibyn<float>`'s `Thigh==double`
  rule). Single-precision results are up-cast float→double on write, so the complex128/float64 output
  holds the complex64/float32 values.
- The **normalization factor** itself is computed in the input's `real_dtype` (float32/float16),
  matching NumPy's `reciprocal(sqrt(n, dtype=real_dtype))` — `RawFft.FftFct`.

The result is recorded in the corpus **as NumPy produces it** (complex64/float32/float16) and the
residual **dtype** mismatch is excused in `MisalignedRegistry` as **F1** — but the harness now
**value-verifies** these cells (up-casting NumPy's complex64/float32/float16 bytes and bit-comparing),
so F1 is a *values-checked* dtype-only excuse, not a skipped comparison. Full parity means adding a
`complex64` dtype ([#569](https://github.com/SciSharp/NumSharp/issues/569)); a real complex64 would
flip the dtype cell automatically, values unchanged.

> **Bonus fix found while doing this:** the pre-existing double `Cfftp`/`CfftpF` `pass7`/`pass11`
> codelets applied `special_mul` to `ca`/`cb` instead of `ca±cb` (the `PM`) in their **ido>1** branch —
> so `np.fft.fft(float64, n)` was silently WRONG for any `n` with a radix-7 or radix-11 factor at
> ido>1 (n = 49, 98, 121, 143, 259, …). Latent because the `fft.jsonl` n-sweep was only {4, 12, 13}.
> Now corrected (matches pocketfft's `PARTSTEP7`/`PARTSTEP11`) and covered by added corpus sizes.

---

## 8. Gates

- **Differential-fuzz tier** `Fuzz/corpus/fft.jsonl` — **2,000 cases** (floor 1,700 in
  `FuzzCorpusTests`), the FFT sub-registry of `OpRegistry` (all 16 transforms + 4 helpers), generated
  by `gen_oracle.gen_fft`. Swept over dtype {float64/complex128/int/bool contractual; float32/float16
  values bit-exact, dtype-only cell} × `n` {default, truncate 4, pad 12, prime 13 (Bluestein), and the
  perfect squares 9/25/49/64/121/169 that force each mixed-radix codelet's ido>1 branch — pass3/5/7/8/
  11/passg} × the four `norm` values × axis {0, 1, middle, negative} × layouts {C, F, strided, reversed,
  offset, broadcast-read, transposed, 3-D} × the N-D `s`/`axes` sweeps. Bit-compared against NumPy 2.4.2
  (complex128 exact; the float32/float16 cells are VALUE-verified via up-cast, then F1 excuses the
  dtype). Discriminating values only (no NaN/inf — a single NaN blanks a whole spectrum and stops
  discriminating the butterflies/twiddles/Bluestein chirp).
- **Unit tests** — 330 under the `Fourier` namespace, all green:

  | File | methods | covers |
  |---|---|---|
  | `np.fft.Standard.Test.cs` | 53 | fft/ifft/fft2/ifft2/fftn/ifftn |
  | `np.fft.Real.Test.cs` | 72 | rfft/irfft + N-D, even/odd packing, `2*(m-1)` default |
  | `np.fft.Hermitian.Test.cs` | 47 | hfft/ihfft compositions + norm swap |
  | `np.fft.Helper.Test.cs` | 43 | fftfreq/rfftfreq/fftshift/ifftshift |
  | `np.fft.Validation.Test.cs` | 19 | the §5 error taxonomy |
  | `np.fft.SharedCore.Test.cs` | 10 | `_raw_fft`/`_cook_nd_args` shared-core edge cases |
  | `np.fft.OutDtype.Test.cs` | 6 | `out=` same_kind casting |
  | `np.fft.RoundTrip.Test.cs` | 3 | `ifft∘fft`, `irfft∘rfft`, DC-component identity |
  | `np.conjugate.Test.cs` / `np.RealImagAngle.Test.cs` | 18 / 24 | the companion accessors |

- **Regenerate** (deterministic; needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py fft`, then
  `dotnet build` (copies the corpus to the test output). **Run:**
  `dotnet test --filter "TestCategory=FuzzMatrix&FullyQualifiedName~fft"` and
  `dotnet test --filter "FullyQualifiedName~Fourier"`.

---

## 9. Traps — do not re-break

- **The FFTPACK half-complex packing lives in the driver, not the engine.** `rfftp`/`radfg` emit
  `R0,R1,I1,…`; `PocketFFTDriver.RunR2C`/`RunC2R` do the ↔ complex128 packing (I0 = 0, the even-`n`
  Nyquist term, and zero-filling short inputs). Split exactly as NumPy splits it between the ufunc
  loop and the codelet.
- **`Cmplx` must stay bit-reinterpretable with `System.Numerics.Complex`** (`[StructLayout(Sequential)]`,
  `r` then `i`). The driver reads/writes `Complex*` through it at the boundary; a field reorder
  silently corrupts every complex result.
- **Parity rests on scalar operation order + the Windows CRT trig.** Do not "optimize" a butterfly
  into a different arithmetic order or a fused form — the port matches NumPy *because* the order is
  verbatim and `Math.Cos`/`Sin` == the MSVC `ucrtbase` NumPy links. No `FusedMultiplyAdd` is used or
  wanted here.
- **The plan cache must never affect bits.** It caches immutable deterministic plans only; NumPy's
  own build has no cache. Keep it bit-neutral (twiddle-setup savings only).
- **`n = None` and `n` given raise different axis errors.** Route the default-`n` path through
  `ShapeAt` (`IndexError`) and the explicit-`n` path through `NormalizeAxisIndex` (`AxisError`) — they
  are not interchangeable.

---

## 10. Not done

1. **Release publication.** #114 / #560's `fft` sub-item remains open until this implementation
   ships in a release.
2. **float32 parity** is the deliberate F1 divergence above, waiting on **#569** (complex64).
3. **Performance** is un-benchmarked — the plan's M5 tuning (in-place / non-buffered fast paths) is a
   correctness-neutral follow-up. No NPY/NS ratios are claimed here.

> Stale-comment note: `RawFftNd`'s XML doc still says the 1-D leaves "currently throw at the engine
> seam … `NotImplementedException`", left over from the stubbed-compute milestone (`d0081b6d`). The
> seam is wired (`RawFft` calls `PocketFFTDriver.Execute`); the comment is inaccurate and should be
> refreshed when the module is next touched.
