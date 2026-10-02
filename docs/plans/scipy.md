# SciPy → C# port — `ScipySharp`, a sibling repo on top of NumSharp

**Source of truth:** live `scipy==1.16.3` (enumerated on this machine, 2026-09-07) + the SciPy
source tree (to be vendored **reference-only** at `refs/scipy`, pinned to the `v1.16.3` tag). NumPy
behaviour is already NumSharp's oracle at `numpy==2.4.2`; SciPy sits on top of that same NumPy, so
`ScipySharp`'s oracle is `scipy==1.16.3` **paired with** NumSharp's existing `numpy==2.4.2` pin.

**Why a separate repo (not a NumSharp subfolder):** SciPy is to NumPy what OptunaSharp is to
NumSharp — a *consumer* library. It references NumSharp, it does not extend it. Following the
established sibling pattern (`K:/source/OptunaSharp`), `ScipySharp` is its own repo that pulls
`NumSharp` + `NumSharp.Interop.OpenBLAS` from nuget.org, vendors NumSharp reference-only at
`refs/NumSharp`, and shares the same `Open.snk` identity. Nothing here changes NumSharp itself
(the one exception is a short list of NumSharp substrate gaps in §5 that a few SciPy areas want —
those land in NumSharp as ordinary `np.*` work, not in this repo).

---

## 0. The thesis — SciPy is **NOT** "a bunch of `np.*` calls"

The intuition "SciPy is written in Python, so a port is re-expressing `np.*`" is **half true and
half a trap.** Measured against the installed 1.16.3 wheel:

- The wheel ships **114 compiled extension binaries** (`.pyd`/`.so`) and almost no source. The
  algorithms live in **C / C++ / Fortran / Cython**, behind a thin Python API layer. Per subpackage:
  `optimize` 15, `sparse` 15, `linalg` 13, `stats` 10, `special` 9, `spatial` 8, **`signal` 6**,
  `integrate`/`interpolate`/`_lib` 7 each, `io`/`ndimage` 5, `fft`/`fftpack`/`odr`/`cluster`/root 1–3.
- So a faithful port is **NumSharp's own two patterns**, in the same proportion NumSharp already
  uses them:
  - **Pattern 1 — compose existing `np.*`** (CLAUDE.md §"Implementation Patterns"): where SciPy is
    genuinely numpy-level (closed-form array math, FFT compositions, polynomial/root math).
  - **Pattern 2 — write a kernel** (`DirectILKernelGenerator` / `ILKernelGenerator`): where SciPy
    drops to a compiled loop with **no `np.*` equivalent** — sequential IIR recurrences, 2-D
    neighborhood scans, LFSRs, order statistics. These do not vectorize; that is *why* SciPy wrote
    them in C, and it is exactly the class of work NumSharp already does everywhere.
  - **Seam — LAPACK/BLAS via `NumSharp.Interop.OpenBLAS`**: where SciPy calls LAPACK
    (`scipy.linalg` extras, least-squares, eig/SVD), reuse the byte-exact OpenBLAS seam NumSharp
    already ships.

**The corollary that shapes the whole roadmap:** the split between "Pattern 1" and "Pattern 2" is
knowable per function *before* writing a line, by asking "does SciPy call a compiled kernel here?"
This document does that classification for `scipy.signal` (§7) and sketches it for the rest (§6).

---

## 1. Scope — what SciPy offers (19 subpackages)

Enumerated from `scipy==1.16.3`. NumSharp already covers the NumPy layer these sit on; every row
below is **additive** (none ship in NumSharp today).

| Subpackage | Compiled `.pyd` | What it is | NumSharp substrate it can reuse |
|--|--|--|--|
| `scipy.linalg` | 13 | Superset of `numpy.linalg` — `expm/logm/sqrtm`, Schur, Toeplitz/circulant, LU/QR/SVD/Cholesky, BLAS/LAPACK wrappers | **`np.linalg` via OpenBLAS** (most of it), poly |
| `scipy.fft` / `scipy.fftpack` | 1 / 1 | DCT/DST/DHT + the numpy-fft surface | **`np.fft` bit-exact** (real/complex); DCT/DST are new |
| **`scipy.signal`** | **6** | Signal processing — *the #349 driver* | `np.fft`, `np.convolve`/`np.correlate` (1-D), poly, linalg |
| `scipy.ndimage` | 5 | N-D image filters, morphology, interpolation, measurements | strided kernels; shares `_sigtools`-class work with signal |
| `scipy.interpolate` | 7 | `interp1d`, splines, `CubicSpline`, RBF, `griddata` | `np.interp`, linalg (spline solves), poly |
| `scipy.integrate` | 7 | `quad`, `solve_ivp`, `odeint` (QUADPACK/ODEPACK) | mostly greenfield managed ports |
| `scipy.optimize` | 15 | minimize/root/curve_fit/linprog (BFGS, L-BFGS-B, Nelder–Mead, MINPACK) | linalg; L-BFGS-B **already ported in OptunaSharp** (reusable design) |
| `scipy.stats` | 10 | distributions, hypothesis tests, descriptive stats, QMC | `np.random` (MT19937/PCG64 byte-exact), sort/reduce |
| `scipy.special` | 9 | gamma/erf/Bessel/orthogonal polys/elliptic (**Cephes**) | greenfield Cephes port (NumSharp has cephes-style bits only in its complex/float kernels) |
| `scipy.sparse` (+`.linalg`,`.csgraph`) | 15 | sparse arrays, sparse solvers, graph algorithms | new storage classes; greenfield |
| `scipy.spatial` (+`.distance`) | 8 | KD-tree, Delaunay/Voronoi/hull (Qhull), rotations, metrics | greenfield (Qhull is heavy) |
| `scipy.cluster` | 3 | k-means (`vq`), hierarchical | reduce/sort; greenfield |
| `scipy.constants` | 0 | physical/math constants + unit conversions | **pure data — trivial** |
| `scipy.io` | 5 | MATLAB `.mat`, WAV, ARFF, netCDF, Harwell-Boeing | NumSharp `.npy`/`.npz` IO patterns |
| `scipy.differentiate` | 0 | finite-difference Jacobian/Hessian | pure `np.*` |
| `scipy.odr` | 1 | orthogonal distance regression (ODRPACK) | linalg; greenfield Fortran port |
| `scipy.datasets` | 0 | sample datasets (network fetch) | out of scope / trivial |
| `scipy.misc` | 0 | deprecated legacy shim | skip |

---

## 2. Repo setup & dependency model (mirror OptunaSharp exactly)

**Name (open decision, recommendation):** repo `SciSharp/ScipySharp`; the managed port parallel to
`NumSharp`. (`Scipy.NET` would be the pythonnet-wrapper analog parallel to `Numpy.NET`, out of scope
here.) Packages are **modular, one per subpackage**, like SciPy itself and like OptunaSharp's
satellite packages:

```
ScipySharp                      # meta-package (references the commonly-used subset)
ScipySharp.Core                 # the `sp` facade root, oracle-shared helpers, common infra
ScipySharp.Signal               # scipy.signal   ← FIRST TARGET (#349)
ScipySharp.Fft                  # scipy.fft/fftpack (DCT/DST on top of np.fft)
ScipySharp.Linalg               # scipy.linalg extras (expm/…): thin over np.linalg + OpenBLAS
ScipySharp.Special              # Cephes special functions
ScipySharp.Stats                # distributions + tests
ScipySharp.Interpolate / .Integrate / .Optimize / .Ndimage / .Spatial / .Sparse / .Cluster / .Io / .Constants …
```

**Dependency wiring (verbatim from `OptunaSharp/Directory.Build.props`):**
- `SignAssembly=true`, `AssemblyOriginatorKeyFile=$(MSBuildThisFileDirectory)Open.snk` — the same
  Microsoft published open key, public-key-token **`cc7b13ffcd2ddd51`**. Keeps identity consistent
  with NumSharp so cross-repo `InternalsVisibleTo(..., PublicKey=…)` resolves.
- `<NumSharpVersion>` single-sourced (start at the current release, `0.70.0`); each package
  `PackageReference`s **`NumSharp`** and **`NumSharp.Interop.OpenBLAS`** at that version from
  nuget.org.
- NumSharp source vendored **reference-only** at `refs/NumSharp` (pinned to `v$(NumSharpVersion)`)
  for offline browsing / LLM cross-referencing — **not** part of the build. Add `refs/scipy`
  (pinned `v1.16.3`) the same way, as the porting source of truth.
- `net8.0` + `net10.0` multi-target, `LangVersion=latest`, MIT, `SciSharp STACK` metadata,
  `VersionPrefix` 0.1.0. Strong-name gate test (`verify_strong_name`) copied over.

**Non-negotiable rule inherited from NumSharp CLAUDE.md:** kernels are **single-threaded**
(`[[single-threaded-only-rule]]`); single-core parity with SciPy is the ceiling. SciPy's own default
is single-threaded per call too (its `workers=` parallelism is opt-in), so this matches.

---

## 3. Architecture — the `sp` facade and the engine split

Mirror NumSharp's `np` static-API design:

```csharp
using ScipySharp;                    // brings in `sp`
using static ScipySharp.sp;          // optional, for sp.signal.spectrogram(...) → spectrogram(...)

var (f, t, Sxx) = sp.signal.spectrogram(x, fs: 1000);   // ports scipy.signal.spectrogram verbatim
```

- **`sp`** is the static root (like `np`), with nested module classes `sp.signal`, `sp.fft`,
  `sp.linalg`, `sp.special`, … each returned by a lowercase property — the **`np.fft` / `np.random`
  house shape** (a lowercase facade property returning a PascalCase module class). This makes
  `scipy.signal.X` → `sp.signal.X` a verbatim transcription.
- **Result tuples** reuse NumSharp's established idiom: a `readonly struct` with implicit
  `NDArray[]` conversion + `Deconstruct` + indexer (the `np.meshgrid` / `np.unique` `UniqueResult`
  pattern), so `f, t, Sxx = spectrogram(...)` and `Pxx = welch(...)[1]` both read like Python.
- **Kernels** live in the consuming package, authored with the **same two generators** NumSharp
  exposes: `DirectILKernelGenerator` for whole-array kernels (the IIR/neighborhood/LFSR work) and
  `ILKernelGenerator` for NDIter per-chunk kernels. `ScipySharp` does not fork the engine; it uses
  NumSharp's public `TensorEngine`/`NDArray`/`Shape` surface and, where it needs raw pointers,
  reads operands as `(T*)a.GetData().Address + a.Shape.Offset` with `a.Shape.Strides` (the public,
  `InternalsVisibleTo`-free recipe documented in NumSharp CLAUDE.md).
- **LAPACK/BLAS** goes through `NumSharp.Interop.OpenBLAS` (`TensorEngine.Blas`) — the same seam
  `np.linalg` uses. `scipy.linalg`'s extras (`expm` via Padé+scaling, `lstsq`, `qr`, `svd`) either
  compose over `np.linalg` or add a thin LAPACK call on the existing binding.

---

## 4. Testing — SciPy as the differential-fuzz oracle (reuse NumSharp's pipeline)

Port NumSharp's oracle philosophy 1:1 (CLAUDE.md §"Differential-Fuzz Pipeline"): **SciPy is the
oracle; Python generates a committed, bytes-exact corpus; the C# harness replays operand bytes and
bit-compares — no Python at test time or in CI.**

- `test/oracle/gen_scipy_oracle.py` — deterministic op matrices over `scipy==1.16.3`, emitting
  `(dtype, shape, strides, offset, input-bytes) → (result dtype/shape/bytes | error{type,text})`
  jsonl, reusing NumSharp's `layout_catalog` layout builders (C/F/strided/reversed/offset/broadcast).
- `ScipySharp.Tests.Oracle` — the C# replay harness (a copy of `FuzzCorpus`/`OpRegistry`/`BitDiff`/
  `MisalignedRegistry`), one `[FuzzMatrix]` test per op-corpus file.
- **Two parity classes, declared per op** (SciPy is less bit-stable than NumPy):
  - **Bit-exact** where the math is closed-form / integer / FFT-composed — windows, waveforms,
    filter *design* coefficients, spectral (rides NumSharp's bit-exact `np.fft`), the `_sigtools`
    integer/order-statistic kernels. This is the target and the default.
  - **Tolerance (`allclose`) with a documented bound** where SciPy reduces through LAPACK/BLAS whose
    accumulation order a managed kernel can't reproduce bit-for-bit — same boundary NumSharp
    documents for its managed LU and OptunaSharp documents for its GP stack. Host-pin (Win/x64,
    threads=1) the byte-exact LAPACK cells the way NumSharp's `matmul_parity`/`linalg_parity` tiers
    do; go `Inconclusive` off the pinned host.
- **Metamorphic tests** (oracle-free invariants) carry weight in signal: `istft(stft(x)) ≈ x` under
  COLA, `filtfilt` zero-phase symmetry, `sosfilt ≡ lfilter` for an equivalent TF, Parseval for the
  spectral estimators.

---

## 5. NumSharp substrate — what's ready, and the short gap list

**Ready to build on (no NumSharp change):**
- `np.fft.*` — complete and **bit-exact** managed pocketfft (real+complex, N-D). The spectral and
  chirp-Z families ride this directly.
- `np.convolve` / `np.correlate` — 1-D, managed SIMD, byte-exact (and byte-identical to SciPy's
  `?dot`-reduced path with `NumSharp.Interop.OpenBLAS`). Covers `signal.convolve`/`correlate` 1-D +
  `fftconvolve`/`oaconvolve` (FFT path) directly.
- Polynomial family — `poly/roots/polyval/polyfit/vander/polyder/polyint/polyadd/sub/mul/div/poly1d`,
  byte-identical via the `eigvals`/`lstsq` seam. Filter design's TF↔ZPK↔SOS conversions and
  `freqz`/`freqs` are polynomial evaluation over the unit circle / imaginary axis.
- `np.linalg` via OpenBLAS — `solve/inv/det/slogdet/lstsq/qr/svd/eig/eigh/cholesky/pinv/
  matrix_rank/cond`, byte-exact on the pinned host. Covers filter design roots, `place_poles`
  (pole placement), LTI state-space, spline solves, `scipy.linalg`'s core.
- `np.interp`, `np.gradient`, reductions/sort/`unique`/`searchsorted`, the whole ufunc surface.

**Short NumSharp gap list (land these in NumSharp as ordinary `np.*` work, not in `ScipySharp`):**
- **`expm`/`logm`/`sqrtm`** (matrix functions) — wanted by `scipy.linalg`, `signal.cont2discrete`
  (the `expm` discretization method), LTI `lsim`. Padé-13 + scaling-and-squaring over the existing
  matmul/solve seam. Medium.
- **DCT/DST** (`scipy.fft`/`fftpack`) — not in `numpy.fft`, so not in NumSharp. Either add to
  NumSharp's `np.fft` engine (pocketfft has the `dct`/`dst` codelets) or implement in
  `ScipySharp.Fft` as a real-FFT composition. Needed by `signal` B-splines and `signal.dct`-based
  smoothing.
- **`scipy.special` primitives** (`gamma`, `erf`, `iv`/`kv` Bessel, `sinc` is in numpy) — greenfield
  Cephes port. `signal.windows.chebwin`/`kaiser`/`dpss` need `i0`/`chebyshev`/eigval routines; a
  handful of special functions unblock the windows, the rest of `scipy.special` is its own package.

None of these block the **first target** (§8): windows + spectrogram need only `np.fft` + `i0`
(one Cephes function) + `get_window` array math.

---

## 6. Roadmap across subpackages (native-substrate-first)

Ordering directive, adapted from OptunaSharp's "native before external": **do the subpackages whose
math NumSharp's substrate already supports before the ones that need a new heavy native engine.**
Difficulty = port effort; "Substrate" = how much is Pattern-1 vs new kernels/engine.

| Phase | Subpackage(s) | Why here | Substrate |
|--|--|--|--|
| **A** | `constants`, `differentiate` | pure data / pure `np.*` — warms up the repo + oracle harness | trivial |
| **B** | **`signal`** | the #349 driver; ~⅓ Pattern-1, rides `np.fft`; the 6 kernels are classic `DirectILKernelGenerator` work | see §7 |
| **C** | `fft`/`fftpack` | DCT/DST on top of `np.fft`; unblocks signal B-splines + interpolate | small kernel/composition |
| **D** | `linalg` extras | `expm/logm/sqrtm/schur/toeplitz` over the OpenBLAS seam | LAPACK seam |
| **E** | `special` | Cephes port; unblocks stats + the remaining windows | greenfield, well-bounded |
| **F** | `interpolate`, `integrate` | splines over linalg; QUADPACK/ODEPACK managed ports | medium |
| **G** | `optimize` | minimize/root/curve_fit; **L-BFGS-B already ported in OptunaSharp** (reuse the design) | medium-heavy |
| **H** | `stats` | distributions (rides `np.random` byte-exact) + tests + Cephes | large but shallow |
| **I** | `ndimage` | N-D neighborhood kernels — shares `_sigtools`-class code with signal | kernel-heavy |
| **J** | `sparse`, `spatial`, `cluster`, `io`, `odr` | new storage/geometry engines (Qhull/ODRPACK) — heaviest greenfield | greenfield |

`datasets`/`misc` — skip.

---

## 7. `scipy.signal` in detail — the first target (issue #349)

**Surface:** 158 public names + 26 windows (`scipy.signal.windows`). **Compiled kernels (the only
parts with no `np.*` equivalent):** `_sigtools` (direct convolve/correlate, `order_filter`,
`medfilt`), `_sosfilt`, `_upfirdn_apply`, `_spline`, `_peak_finding_utils`, `_max_len_seq_inner`.
Everything else — verified by reading the sources — is pure numpy/Python. Notably **`_spectral_py.py`
is pure numpy + FFT** (welch/csd/coherence/spectrogram/periodogram/stft/istft **and** `lombscargle`
all live there with no compiled kernel), so the entire spectral family is Pattern-1 on NumSharp's
bit-exact `np.fft`.

### Bucket A — Pattern-1 (compose `np.*`; bit-exact target). *~half the surface.*

- **Windows** (`sp.signal.windows`, 26 + `get_window`): `boxcar, triang, bartlett, barthann, hann,
  hamming, general_hamming, blackman, blackmanharris, nuttall, flattop, bohman, parzen, cosine,
  lanczos, tukey, kaiser, kaiser_bessel_derived, gaussian, general_gaussian, general_cosine,
  chebwin, dpss, exponential, taylor`. Closed-form array formulas; `kaiser`/`kaiser_bessel_derived`
  need `i0` (one Cephes fn), `chebwin` needs a Chebyshev-poly/real-FFT step, `dpss` needs a
  symmetric-tridiagonal eigen-solve (`np.linalg.eigh`).
- **Spectral** (rides `np.fft`): `periodogram, welch, csd, coherence, spectrogram, stft, istft,
  ShortTimeFFT, check_COLA, check_NOLA, closest_STFT_dual_window, lombscargle, vectorstrength`.
  **← `spectrogram` is here — the #349 ask, zero new kernels.**
- **Waveforms:** `chirp, gausspulse, sawtooth, square, sweep_poly, unit_impulse`. (`max_len_seq` is
  Bucket B — it's an LFSR.)
- **Chirp-Z / Zoom FFT:** `czt, zoom_fft, CZT, ZoomFFT, czt_points` — FFT compositions.
- **Filter design + conversions** (polynomial/root/complex math over poly + `np.linalg`): `butter,
  buttord, cheby1, cheb1ord, cheby2, cheb2ord, ellip, ellipord, bessel, besselap, buttap, cheb1ap,
  cheb2ap, ellipap, iirfilter, iirdesign, iirnotch, iirpeak, iircomb, bilinear, bilinear_zpk,
  freqz, freqz_zpk, freqz_sos, sosfreqz, freqs, freqs_zpk, findfreqs, group_delay, tf2zpk, zpk2tf,
  tf2sos, sos2tf, tf2ss, ss2tf, zpk2sos, sos2zpk, zpk2ss, ss2zpk, normalize, abcd_normalize,
  residue, residuez, invres, invresz, unique_roots, lp2lp/hp/bp/bs(+_zpk), place_poles, cont2discrete,
  firwin, firwin2, firls, remez, kaiserord, kaiser_atten, kaiser_beta, savgol_coeffs, gammatone,
  minimum_phase, band_stop_obj`. (`ellip*` need elliptic integrals from `special`; `remez`/`firls`
  need a small linear solve; `place_poles` needs `np.linalg`.)
- **Misc:** `detrend, savgol_filter, correlation_lags, choose_conv_method, hilbert, hilbert2,
  envelope` (analytic-signal FFT), `fftconvolve, oaconvolve, convolve, correlate` (1-D rides
  `np.convolve`/`np.correlate`).

### Bucket B — Pattern-2 (new `DirectILKernelGenerator` kernels; SciPy uses compiled C).

- **IIR / feedback recurrences** (`_sosfilt`, `lfilter`): `lfilter, lfiltic, lfilter_zi, filtfilt,
  sosfilt, sosfilt_zi, sosfiltfilt, deconvolve`. A sequential `y[n] = Σ b·x − Σ a·y[n−1..]` feedback
  loop — **inherently scalar, non-vectorizable**; this is the canonical "SciPy wrote it in C" case.
- **2-D / neighborhood scans** (`_sigtools`): `convolve2d, correlate2d, sepfir2d, order_filter,
  medfilt, medfilt2d, wiener`. (`medfilt` = order-statistic; `order_filter` = ranked window.)
- **Polyphase resampling** (`_upfirdn_apply`): `upfirdn, resample_poly, decimate`. (`resample` — the
  FFT resampler — is Bucket A.)
- **Symmetric-IIR B-splines** (`_spline`): `cspline1d, qspline1d, cspline2d, qspline2d,
  cspline1d_eval, qspline1d_eval, spline_filter, gauss_spline, symiirorder1, symiirorder2`.
- **Peak finding** (`_peak_finding_utils`): `find_peaks, peak_prominences, peak_widths,
  find_peaks_cwt, argrelmin, argrelmax, argrelextrema`. (`argrel*` are `np.*`-composable; the
  prominence/width topographic scans are the kernel.)
- **LFSR:** `max_len_seq` (`_max_len_seq_inner`).

### Bucket C — LTI system object models (control theory).

- Classes: `lti, dlti, StateSpace, TransferFunction, ZerosPolesGain`. Simulation:
  `lsim, impulse, step, freqresp, bode` (+ discrete `dlsim, dimpulse, dstep, dfreqresp, dbode`).
  These are a small class hierarchy + representation conversions (Bucket A) + an ODE/`expm` step
  (`lsim` needs `cont2discrete`→`expm`, hence the NumSharp `expm` gap in §5).

### Signal phasing (what to build, in order)

1. **`ScipySharp.Signal` skeleton + `sp.signal` facade + oracle harness** (Phase A infra reused).
2. **Windows** (Bucket A) — pure array math; the one dependency is `i0`/`eigh`, both cheap. Gate:
   window corpus bit-exact vs scipy.
3. **Spectral → `spectrogram`** (Bucket A) — **closes #349.** `get_window` + `_spectral_helper`
   (STFT framing over `np.fft`) + `welch` averaging + `detrend`. Metamorphic COLA/`istft` tests.
4. **Filter design** (Bucket A) — `butter`/`cheby`/`iirfilter`/`freqz`/`firwin` + TF/ZPK/SOS
   conversions; unblocks realistic filtering demos.
5. **Filtering kernels** (Bucket B) — `lfilter`/`sosfilt`/`filtfilt` first (most-used), then
   `medfilt`/`order_filter`/`convolve2d`, then `upfirdn`/`resample_poly`, then B-splines, then peaks.
6. **LTI** (Bucket C) — after NumSharp `expm` lands.

---

## 8. ▶ START HERE

**First deliverable = `ScipySharp.Signal` with `windows` + `spectrogram`**, which is the literal
#349 request and needs **no new NumSharp feature** (only `np.fft`, already bit-exact, plus a single
Cephes `i0` for the Kaiser windows). Concretely:

1. Scaffold `SciSharp/ScipySharp` from the OptunaSharp template (`Directory.Build.props`, `Open.snk`,
   `refs/NumSharp` + `refs/scipy`, CI, strong-name gate).
2. `ScipySharp.Core` — the `sp` facade root + the copied oracle harness (`FuzzCorpus`/`OpRegistry`/
   `BitDiff`/`MisalignedRegistry`) + `gen_scipy_oracle.py`.
3. `ScipySharp.Signal` — `windows.*` (bit-exact corpus) → `get_window` → `spectrogram`/`welch`/
   `stft` (bit-exact vs scipy over `np.fft`) → close #349 with a working C# spectrogram example.

---

## 9. Risks, non-goals, open decisions

- **Open decision — name:** `ScipySharp` (recommended, parallel to `NumSharp`) vs `SciSharp.Scipy`
  vs per-package `SciSharp.*`. Maintainer's call; the plan assumes `ScipySharp.*`.
- **Bit-exactness is not free everywhere.** Closed-form/FFT/integer parts are bit-exact; LAPACK-
  reduced parts (filter-design eig, `lstsq`-based `firls`/`remez`, LTI solves) are tolerance-parity
  unless host-pinned to the OpenBLAS binary — accept the same `[Misaligned]`/host-pin discipline
  NumSharp already uses.
- **`scipy.special` is on the critical path for a few windows and much of `stats`** — a Cephes port
  is well-bounded but real work; keep it its own package and only pull the handful of functions the
  earlier phases need.
- **Heaviest greenfield (defer):** Qhull (`spatial`), ODRPACK (`odr`), QUADPACK/ODEPACK
  (`integrate`), sparse graph/solvers. These have no NumSharp substrate and are large native
  algorithms — Phase J, after the numpy-adjacent subpackages prove the repo.
- **Non-goals:** `scipy.datasets` (network fetch), `scipy.misc` (deprecated), GPU/threaded kernels
  (single-threaded rule), and any pythonnet wrapping (that would be a separate `Scipy.NET`, not this
  managed port).
- **NumSharp changes stay in NumSharp.** The §5 gap items (`expm`/`logm`/`sqrtm`, DCT/DST, the
  Cephes primitives that arguably belong as `np.*`) are proposed as NumSharp PRs, keeping `ScipySharp`
  a pure consumer.

---

## TL;DR

- **SciPy ≠ "a bunch of `np.*` calls":** the 1.16.3 wheel is a thin Python API over **114 compiled
  C/C++/Fortran/Cython kernels**. A faithful port is **NumSharp's own two patterns** — compose
  `np.*` (Pattern 1) where SciPy is numpy-level, write a `DirectILKernelGenerator` kernel (Pattern 2)
  where SciPy dropped to C — plus the OpenBLAS/LAPACK seam NumSharp already ships.
- **Structure:** a **separate sibling repo** `ScipySharp`, wired exactly like `OptunaSharp` (refs
  `NumSharp` + `NumSharp.Interop.OpenBLAS` from nuget, vendors `refs/NumSharp` reference-only, shares
  `Open.snk`/token `cc7b13ffcd2ddd51`, `net8.0`+`net10.0`, `SciSharp STACK`), with **modular
  per-subpackage packages** and an `sp` facade mirroring `np`.
- **Oracle:** reuse NumSharp's differential-fuzz pipeline with **`scipy==1.16.3` as the oracle** —
  committed bytes-exact corpus, C# replay, no Python in CI; bit-exact where closed-form/FFT/integer,
  host-pinned tolerance where LAPACK-reduced.
- **`scipy.signal` (158 names + 26 windows, 6 compiled kernels):** ~half is Pattern-1 (windows,
  waveforms, **all of spectral incl. `lombscargle`**, filter design/conversions) riding NumSharp's
  bit-exact FFT + poly + linalg; the other half is the classic kernel work (IIR `lfilter`/`sosfilt`,
  2-D `medfilt`/`convolve2d`, `upfirdn`, B-splines, peak finding, LFSR) + a small LTI object layer.
- **Start:** `ScipySharp.Signal` **windows + `spectrogram`** — the exact #349 request, needing **no
  new NumSharp feature** (only the existing `np.fft` + one Cephes `i0`).
- **NumSharp gaps to land separately (small):** `expm`/`logm`/`sqrtm`, DCT/DST, a handful of Cephes
  special functions — as NumSharp PRs, so `ScipySharp` stays a pure consumer.
