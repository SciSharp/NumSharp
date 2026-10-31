---
paths:
  - "src/NumSharp.Core/Fourier/**"
  - "src/NumSharp.Core/Backends/FftEngine.cs"
  - "src/NumSharp.Core/Math/np.{windows,i0}.cs"
  - "src/NumSharp.Core/Backends/Default/Math/Default.I0.cs"
  - "src/NumSharp.Core/Math/np.{conjugate,real,imag,angle}.cs"
  - "test/NumSharp.Tests/Fourier/**"
---

# FFT and window functions

## Window functions

`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser` (`numpy/lib/_function_base_impl.py`; `Math/np.windows.cs`, with
the cephes `BesselI0` port behind kaiser). Each takes a point count `M` (kaiser also `beta`) and returns a 1-D float64
taper (NumPy forces at least float64 through `np.array([0.0, M])`).
- `M` is `double`, NumPy's own stub type `_FloatLike_co`: an int caller binds through the implicit conversion, and a
  non-integer `M` behaves as NumPy's float M (the length is `len(arange(1-M, M, 2))`, so `np.hanning(5.7)` has 6 samples,
  and kaiser can carry a trailing NaN once `|(n-α)/α| > 1`).
- The cosine windows (bartlett/blackman/hamming/hanning): `M < 1` → empty `(0,)`, `M == 1` → `ones(1,)`. kaiser has no
  `M < 1` guard: `arange(0, M)` is empty for `M ≤ 0` but a single element for `0 < M < 1` (`np.kaiser(0.5, 5)` =
  `[0.0367…]`), and `M == 1` is special-cased only to avoid `alpha == 0`.
- Formulae, verbatim from NumPy, with `n = arange(1-M, M, 2)` (kaiser `arange(0, M)`): bartlett
  `where(n ≤ 0, 1+n/(M-1), 1-n/(M-1))`; hanning `0.5+0.5·cos(πn/(M-1))`; hamming `0.54+0.46·cos(πn/(M-1))`; blackman
  `0.42+0.5·cos(πn/(M-1))+0.08·cos(2πn/(M-1))`; kaiser `i0(β·√(1-((n-α)/α)²))/i0(β)` with `α = (M-1)/2` and `i0` the cephes
  Chebyshev routine (`_i0_1`/`_i0_2`/`_chbevl`, the internal `np.BesselI0` scalar helper), which the public `np.i0` runs
  too (`TensorEngine.I0`, `Backends/Default/Math/Default.I0.cs`, one cephes kernel per float width; facade
  `Math/np.i0.cs`).
- Bit-identical to NumPy 2.4.2: the transform is one `np.evaluate` fused expression in NumPy's exact operation order (so
  `(π·n)/(M-1)` rounds the same two steps), float64 add/sub/mul/div are IEEE-exact, and float64 `cos` is the `ucrtbase`
  `Math.Cos` NumPy also calls (`i0` rides `Math.Exp`/`Math.Sqrt`). Because those are host libm, the `windows` tier is
  host-pinned (`RunHostLibmCorpus`: strict on win-amd64, Inconclusive elsewhere); bartlett is pure arithmetic and
  portable. Gates: `Math/np.windows.Test.cs`, the `windows` tier (float M included).

## Fourier transforms (`np.fft.*`)

`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`,
`fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, plus the complex accessors `conjugate`/`conj`, `real`, `imag`, `angle`.
- A pure-managed C# port of pocketfft, NumPy's own FFT engine (`src/NumSharp.Core/Fourier/`, no native dependency). The
  double/complex128 path is bit-for-bit identical to NumPy 2.4.2: the port is a faithful scalar transcription, NumPy's
  win-amd64 wheel is itself scalar (`POCKETFFT_NO_VECTORS` under MSVC), and the twiddles come from the same CRT
  (`Math.Cos`/`Sin` = MSVC `ucrtbase`). No explicit FMA is needed, which is why `fft.jsonl` is a portable tier.
- arm64 NumPy differs, not NumSharp: NumPy's arm64 wheels compile pocketfft with the default FP contraction on an FMA
  baseline, so its `a*b + c*d` twiddle products (`sincos_2pibyn::operator[]`) and butterfly multiply-adds (`MULPM`,
  `radf5`) round once (clang fuses the LEFT product: `fmuladd(a, b, c*d)`), and live arm64 NumPy's `rfft` is an ULP or so
  off NumSharp's in most float64 lanes. A C# replica of `rfftp` fed macOS's libm twiddles reproduces both macOS results
  (NumSharp's when evaluated literally, NumPy's when fused; hashes in `InteropTestBase.PocketFftFusedArithmetic`). The
  live interop gate keeps these cells strict on x64 and reports them Inconclusive, with the measurement, on arm64
  (`InteropTestBase.AssertExactUnlessNumPyFuses`, `SpectrumLiveParityTests`, the Gist FFT/HPS cells).
- Twiddle trap: twiddles come from the PLATFORM libm on both sides (`Math.Cos`/`Math.Sin` ↔ `std::cos`/`std::sin`), and
  libms disagree in the last bit (at n = 1024 Apple's `sin(72·π/4096)` and `sin(216·π/4096)` are 1 ULP above correctly
  rounded, and ucrtbase misses correct rounding on some candidate angles). So NumSharp's own FFT bits are not
  host-independent for large n; the offline `fft.jsonl` tier (win-amd64 bytes) is portable because ucrtbase, glibc and
  Apple agree on its sizes' twiddles. A larger corpus size can turn host-pinned.
- The facade is the `np.random` house shape: a lowercase property `np.fft` returning `FourierModule`, so `np.fft.fft(x)`
  ports verbatim. Each `FourierModule` function but the two shifts is one call into the `TensorEngine.Fft` sub-engine
  (`Backends/FftEngine.cs`; the default `DefaultFftEngine` in `Fourier/DefaultFftEngine*.cs` holds NumPy's statements;
  hfft keeps its class boundary in the facade; `openblas.md` for the seam). Three 1-D kernels do the work:
  `DefaultFftEngine`'s `protected virtual` `C2CKernel` / `R2CKernel` / `C2RKernel`, each a call into
  `PocketFFTDriver.Execute` (`c2c`/`r2c`/`c2r`); every N-D/2-D form
  composes 1-D transforms over `_raw_fftnd`, and `hfft`/`ihfft` compose `irfft`/`rfft` with `np.conjugate` and the norm
  swap. The strided all-but-axis driver (`PocketFFTDriver.cs`, a port of `_pocketfft_umath.cpp`) does `copy_input`
  (gather `min(nin, n)` strided elements, zero-pad) → `plan.Exec(buf, fct, dir)` → `copy_output` per lane, reading each
  operand's strides/offset (every layout works), with the FFTPACK half-complex packing (`R0,R1,I1,…`, I0 = 0, the even-n
  Nyquist term) where NumPy's `rfft_impl`/`irfft_loop` apply it. Engine files: `PocketFFT.Twiddle.cs`
  (`sincos_2pibyn`, `cmplx`, prime/cost/good_size), `.Complex.cs` (`cfftp` mixed radix, codelets 2/3/4/5/7/8/11 +
  generic), `.Real.cs` (`rfftp` radf/radb 2/3/4/5 + generic), `.Bluestein.cs` (`fftblue` for large-prime lengths),
  `.Plan.cs` (mixed radix when `largest_prime_factor(n)² ≤ n`, else a cost comparison with Bluestein, and a bounded
  64-entry plan cache that is bit-neutral: NumPy's build has `POCKETFFT_CACHE_SIZE == 0`, so caching only saves twiddle
  setup).
- Layer-3 semantics (`DefaultFftEngine.RawFft.cs`, a port of `_pocketfft.py`): the norm factor `fct` (forward
  `None/backward → 1`, `ortho → 1/√n`, `forward → 1/n`; the inverse swaps direction FIRST, so `ifft` divides by n); `n < 1` →
  `ValueError("Invalid number of FFT data points ({n}) specified.")`; `n` truncates/zero-pads; `rfft` output `n//2+1`;
  `irfft`/`hfft` default `n = 2*(m-1)`; the N-D `s`/`axes` cook (`s` without `axes` → the last `len(s)` axes, the
  `s[i] == -1` sentinel, `len(s) != len(axes)` → `ValueError("Shape and axes have different lengths.")`).
- Errors in NumPy's line order: the bad-`norm` message has no space after the first comma on a FORWARD transform but a
  space on the inverse/`hfft`/`ihfft` (`_swap_direction`'s KeyError path); an out-of-range axis is `IndexError("tuple
  index out of range")` when `n` is omitted (the `a.shape[axis]` subscript fires first) but `AxisError` when `n` is
  given; `rfft`/`ihfft` of a COMPLEX input is `TypeError("ufunc 'rfft_n_even'/'rfft_n_odd' not supported …")` (the ufunc
  chosen by input parity, raised only after n/norm/axis/out validate); a bad `out` is `ValueError("output array has wrong
  shape.")`. `out=` accepts a same_kind cast from the loop dtype; the N-D wrappers accept `out` for signature parity but
  do not thread it (NumPy passes `out=None`).
- Helpers: `fftfreq`/`rfftfreq` are float64 generators with NumPy's check order (divide by zero first, then `device`,
  then `fftfreq`'s negative length; a non-integer n → `ValueError("n should be an integer")`); `fftshift`/`ifftshift` are
  dtype-preserving cyclic rolls by `±(dim//2)` (0-d → a no-op copy, `[Misaligned]`).
- `[Misaligned]` F1, dtype only: NumSharp has one complex type (complex128) and no complex64, so a float32/float16 input
  returns complex128/float64 where NumPy returns complex64/float32/float16. The VALUES are bit-identical, because
  NumSharp reproduces NumPy's per-loop precision: NumPy's fft ufuncs have a single (`Ff->F`/`ff->F`) and a double
  (`Dd->D`/`dd->D`) loop chosen by operand shape. A real float32/float16 input to fft/ifft/irfft promotes real →
  complex128 → the double loop, then rounds to the complex64/float32/float16 output (`PocketFFTDriver.RoundInPlace`);
  rfft of a float32 real input and fft/ifft/irfft of a complex64-precision N-D intermediate hit the SINGLE loop, but only
  when the norm factor is a float (ortho/forward: backward/None's int `fct = 1` resolves to the double loop). A
  single-precision pocketfft engine (`Fourier/PocketFFT.{Complex,Real,Bluestein,Plan}.Single.cs`, `CmplxF`,
  `TwiddleF.At`) serves those, gated in `PocketFFTDriver.Execute` on `floatPrec && !effNormUnity && (complex operand ||
  rfft-float32)`; the norm factor is computed in the input's real dtype (`FftFct`, `DefaultFftEngine.RawFft.cs`). The
  harness value-verifies these cells (it up-casts NumPy's complex64/float32/float16 and bit-compares:
  `FuzzCorpusTests.Kinds.FftFloatValuesMatch`).
- Gates: `Fuzz/corpus/fft.jsonl` (the n sweep includes the perfect squares 9/25/49/64/121/169, which reach every
  mixed-radix codelet's ido > 1 branch: pass3/5/7/8/11/passg), `OpRegistry`'s 16 transforms and 4 helpers,
  `gen_oracle.gen_fft`, `MisalignedRegistry` F1, and the `Fourier`-namespace unit tests (`FftSubEngineTests` for the
  sub-engine). Sources: `Fourier/np.fft.{cs,Standard,Real,Hermitian,Helper}.cs` (the facades),
  `Fourier/DefaultFftEngine{,.Standard,.Real,.Hermitian,.Helper,.RawFft,.Kernels}.cs`, `Backends/FftEngine.cs`,
  `Fourier/PocketFFT*.cs`.
