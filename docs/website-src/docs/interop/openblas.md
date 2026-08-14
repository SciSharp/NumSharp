# OpenBLAS — NumPy's bits from NumPy's own binary

`NumSharp.Interop.OpenBLAS` is the optional native matrix-product backend. Reference it and
`np.dot` / `np.matmul` compute `float32` / `float64` products through OpenBLAS — **faster on large
matrices** (1.7–13× over NumSharp's own managed GEMM) and **bit-identical to NumPy**, because the
package bundles the very OpenBLAS
binaries NumPy itself ships and drives them through a route-for-route port of NumPy's own two
matrix-product dispatchers. NumSharp.Core stays 100 % managed C# with no native dependency: with
this package absent, every kernel — matrix products included — is NumSharp's own SIMD code. This
page is the package's reference: why it exists, the quick start, the bundled binary, the three
knobs the bits depend on, the API, where the library comes from, the build-time version override,
what is ported, extending the seam, performance and troubleshooting.

**On this page:** [Why it exists](#why-this-package-exists) ·
[From zero to identical bits](#from-zero-to-identical-bits) ·
[The bundled binary](#the-bundled-binary) ·
[Three knobs decide the bits](#three-knobs-decide-the-bits) · [The API](#the-api) ·
[Where the library comes from](#where-the-library-comes-from) ·
[Build-time override](#overriding-the-version-at-build-time) · [How it computes](#how-it-computes) ·
[Writing your own backend](#writing-your-own-backend) · [Performance](#performance) ·
[Environment](#environment) · [Troubleshooting](#troubleshooting) · [Claims](#claims-ledger)

> Verified against numpy 2.4.2 · scipy-openblas 0.3.31.22.0 (OpenBLAS 0.3.31.dev) · net8.0/net10.0.
> Every claim below is pinned by a gate: [`MatmulParityBackendTests`][gate-backend] (30 tests),
> [`OpenBlasDeliveryTests`][gate-delivery] (20), the 342-case host-pinned
> [`matmul_parity` corpus tier][gate-corpus], and the 15-step scripted build gate
> [`verify_build_override.sh`][gate-build].

---

## Why this package exists

NumSharp's own GEMM and NumPy's disagree — not because either is wrong, but because floating-point
addition is not associative and the two sum in different orders. Measured on a
`(128,784) @ (784,128)` float32 product, **94.5 % of the elements differ** (max ~976 000 ULP), and
45 % differ even at contraction depth k = 10. Each answer is a correct matrix product; for anything
that *compounds* — training a network, iterating a solver — the gap grows: 50 Adam steps spread it
to 73 % of a weight matrix's bytes.

No portable algorithm can close that, because NumPy's float32/float64 matrix products are not a
portable algorithm. NumPy **always** calls cblas for them (since gh-23588 it copies a non-blasable
operand into a contiguous temp rather than take its own C loop), and OpenBLAS' `sgemm` uses an
arch-specific multi-accumulator register scheme that matches neither a sequential `mul + add` chain
nor a sequential FMA chain. NumPy's bits *are* its BLAS's bits. Reproducing them means calling the
same binary, through the same route, with the same flags — which is what this package does.

The end-to-end result: 342/342 committed corpus cases replay bit-exact with the backend installed
(294 of them diverge without it), and a 50-step float32 MLP trained in both stacks produces
**byte-identical weights**.

<sub>See here [`FuzzCorpusTests.MatmulParity`][gate-corpus], [`MatmulParityPin`][gate-pin]</sub>

---

## From zero to identical bits

```bash
dotnet add package NumSharp.Interop.OpenBLAS
```

Referencing the package is the whole opt-in. A module initializer runs when the assembly loads: it
walks the [discovery order](#where-the-library-comes-from), loads the bundled OpenBLAS for this
machine's RID and installs the backend on the engine. If nothing can be loaded, **nothing
changes** — NumSharp keeps computing everything itself, and merely referencing the package can
never break the app (the initializer never throws).

```csharp
using NumSharp;
using NumSharp.Interop.OpenBLAS;

var c = np.dot(a, b);            // float32/float64 products now go through OpenBLAS

Console.WriteLine(Blas.Info);
// …\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 24]
// OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24

Blas.Disable();                  // back to NumSharp's own managed SIMD GEMM
```

### The byte-parity recipe

The zero-config default matches the NumPy *on the same machine*, run at the same thread count. For
a run you can byte-compare, pin the thread count on **both** sides — OpenBLAS' accumulation order
depends on it, and NumPy runs larger products multi-threaded by default while small ones stay
single-threaded:

```csharp
Blas.Enable(threads: 1);                 // the C# side
```

```bash
OPENBLAS_NUM_THREADS=1 python train.py   # the NumPy side — set before `import numpy`
```

With that, small and large products alike come out byte-identical (verified through the hardest
k = 784 case: 0 of 16 384 result elements differ). Add `coreType: Blas.ParityCoreType` on both
sides only when the two runs happen on *different machines* — see
[Three knobs](#three-knobs-decide-the-bits) for why, and for the hazard.

<sub>See here [`ThreadPin_IsReportedBack`][gate-backend], [`BundleAutoinstall_WithNoOptOut_InstallsTheBackend`][gate-delivery]</sub>

---

## The bundled binary

**The package ships the OpenBLAS binaries NumPy itself ships** — not "an" OpenBLAS. NumPy 2.x does
not build its own: its wheels bundle the prebuilt `scipy-openblas32` / `scipy-openblas64` artifacts
published on PyPI by the openblas-libs project, pinned in numpy's `requirements/ci_requirements.txt`.
This package bundles those same artifacts at the same pinned version, **verified byte-for-byte**:
the DLL here and the one in numpy 2.4.2's `numpy.libs/` have the identical SHA-256 (`74a40872…` —
which is also where NumPy's mangled file name comes from, since `delvewheel` names the file by its
content hash).

So bit-parity does not depend on having Python installed, or on which copy of a BLAS won a loader
race. Bundled version: **scipy-openblas 0.3.31.22.0** (OpenBLAS 0.3.31.dev), matching
**numpy 2.4.2** — as `runtimes/<rid>/native/` NuGet assets for 8 RIDs (62 MB packed), which NuGet
resolves per machine so a RID-specific publish carries only the one binary:

| RID | distribution | RID | distribution |
|---|---|---|---|
| `win-x64` | scipy-openblas64 | `linux-musl-x64` | scipy-openblas64 |
| `win-arm64` | scipy-openblas**32** † | `linux-musl-arm64` | scipy-openblas64 |
| `linux-x64` | scipy-openblas64 | `osx-x64` | scipy-openblas64 |
| `linux-arm64` | scipy-openblas64 | `osx-arm64` | scipy-openblas64 |

† not an inconsistency — NumPy's own build selects the LP64 distribution on win-arm64, so matching
it is what parity requires. Both scipy distributions are first-class here: **scipy-openblas64**
(ILP64, 64-bit BLAS integers, symbols `scipy_cblas_sgemm64_`) and **scipy-openblas32** (LP64,
`scipy_cblas_sgemm`), and the 32-bit distribution is the *only* build PyPI publishes for 32-bit x86.

The bundled OpenBLAS is not privileged, either: any CBLAS export set binds, probed in this order —
numpy's renamed ILP64 wheels (`scipy_cblas_*64_`), plain ILP64 (`cblas_*64_` / `cblas_*_64`),
scipy's LP64 build (`scipy_cblas_*`), stock LP64 (`cblas_*`) — so MKL, Accelerate, a system
`libblas` or a reference CBLAS all work, with the BLAS integer width marshalled per call. Ten
symbols are required (`gemm`, `gemv`, `syrk`, `dot`, `axpy`, each in s/d); OpenBLAS' thread,
core-name and build-info entry points are used when present, and a provider without them works fine.

Two maintainer details worth knowing: the binaries are **not in git** — `tools/openblas-manifest.json`
is the checked-in pin and `fetch_openblas.py` stages them, verifying **two** SHA-256 hashes per RID
(the wheel's from PyPI, then the extracted library's own) so a re-pointed URL or mutated artifact
fails loudly. And the license expression is compound (`Apache-2.0 AND BSD-3-Clause-Attribution AND
BSD-2-Clause AND (GPL-3.0-or-later WITH GCC-exception-3.1)`, full text in
`THIRD-PARTY-NOTICES.txt`): the statically linked libgfortran's GCC Runtime Library Exception is
what permits redistribution — the same basis on which NumPy and SciPy ship it in every wheel.

<sub>See here [`BundledLibrary_HashesToThePinnedManifest`][gate-backend], [`BundledLibrary_ForThisRid_ReachesTheBuildOutput`][gate-backend]</sub>

---

## Three knobs decide the bits

Bundling the binary fixes only one of the three things the result bits depend on. A parity claim
that ignores either of the other two is wrong:

1. **The library build** — pinned by the package (the bundle) or by you (a named library).
2. **The worker-thread count** — 1, 2, 4 and 24 threads give **four different answers** for the
   same product. `Blas.Enable(threads: n)` pins it; it is a correctness knob, not a performance one.
3. **The CPU** — OpenBLAS is built `DYNAMIC_ARCH` and picks a micro-kernel from the machine it
   finds itself on, and different kernels accumulate in different orders. Measured by forcing the
   choice on one host (same binary, 1 thread, `(512,784) @ (784,512)` float32):

| `OPENBLAS_CORETYPE` | dispatched | result hash |
|---|---|---|
| *(unset)* | Haswell | `b9ea5057…` |
| `Haswell` | Haswell | `b9ea5057…` — identical to unset |
| `Nehalem` | Nehalem | `b6bb0f58…` |
| `Sandybridge` | Sandybridge | `1903e357…` |
| `Core2` | Katmai | `0da21438…` |
| `SkylakeX` | — | **process killed: illegal instruction** |

`Blas.Enable(threads: 1, coreType: Blas.ParityCoreType)` pins that last variable to `"Haswell"` —
the kernel the committed corpus was generated with, runnable on any x86-64 with AVX2+FMA (Haswell
2013 and later, and every AMD Zen). It is **not the default, deliberately**: the package's main
promise is matching the NumPy on *this* machine, and that NumPy dispatches by CPU too — pinning
unilaterally would *create* a divergence on any host newer than Haswell. Pin both sides or neither;
the same `OPENBLAS_CORETYPE` environment variable drives NumPy, because it is the same library.

> **Hazard.** `coreType` *overrides* OpenBLAS' CPU detection rather than expressing a preference,
> so naming kernels the CPU cannot run terminates the process with an illegal instruction —
> measured, not theoretical. `Blas.Enable` refuses the combinations it can recognise with a
> `PlatformNotSupportedException` first. It also only takes effect on the call that actually
> **loads** the library: OpenBLAS reads the variable once while initialising and never re-reads it,
> so a later request that cannot be honoured throws instead of being silently dropped.

<sub>See here [`CoreTypeAboveThisCpu_IsRefused_RatherThanCrashingTheProcess`][gate-backend], [`ParityCoreType_MatchesTheCorpusHostPin`][gate-backend]</sub>

---

## The API

The whole public surface is the static `Blas` class (namespace `NumSharp.Interop.OpenBLAS`):

| Member | Meaning |
|---|---|
| `Blas.Enable(library = null, threads = 0, coreType = null)` | Load a CBLAS library (or auto-discover one) and install the backend. Idempotent when the same library is already loaded. |
| `Blas.TryEnable(library = null, threads = 0)` | `Enable` that reports `false` instead of throwing — the module-load path. |
| `Blas.Disable()` | Uninstall the backend; the native library stays loaded (unloading a BLAS mid-process is not worth the risk), so a later `Enable` is free. |
| `Blas.Enabled` | `true` when the backend is installed **and** a library is bound — both halves, see below. |
| `Blas.Info` | Path, symbol scheme, integer width, thread count and build string of the loaded library, or `null`. |
| `Blas.LibraryPath` | Full path of the loaded library — the machine-readable form, because "which binary answered?" is a correctness question here, not a diagnostic one. |
| `Blas.CoreName` | The `DYNAMIC_ARCH` micro-kernel OpenBLAS actually dispatched (e.g. `"Haswell"`). |
| `Blas.IsBundledLibrary` | `true` when the loaded library is the package's bundled asset — marker-aware: a build-staged override in the same folder layout reports `false`. |
| `Blas.ParityCoreType` | `"Haswell"` — the kernel the committed corpus was generated with. |

Three semantics are load-bearing, each pinned by a test:

- **A named library is binding.** An explicit `Blas.Enable(path)` (or `NUMSHARP_PARITY_BLAS`) is
  used as given and never silently replaced — not even by the bundled copy sitting in the output
  folder — because parity is a claim about one specific binary.
- **A failed `Enable` is a no-op.** It throws having changed nothing, so a mistyped path cannot
  demote a working parity setup back to the managed kernels behind your back. (An earlier design
  unloaded the working library first — the worst failure mode a parity feature can have:
  `Enabled` reporting `true` while every product silently reverted to managed kernels.)
- **`Enabled` means installed *and* bound.** An installed-but-unbound backend declines every
  operand, so every product would quietly fall back to the managed kernels while the answer still
  looks like a matrix product; reporting `true` there would claim a parity it is not delivering.

Opting out of the automatic install (`NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0`) leaves `Blas.Enable()`
fully functional — it only skips the module-load call.

<sub>See here [`NamedLibrary_IsBinding_NeverSilentlySubstituted`][gate-backend], [`FailedEnable_OverAWorkingLibrary_ChangesNothing`][gate-backend], [`Enabled_MeansInstalledAND_Bound_NotMerelyInstalled`][gate-backend]</sub>

---

## Where the library comes from

Discovery is a fixed, documented order — the first tier that yields a loadable library wins
(specified in [`OPENBLAS_DELIVERY_DESIGN.md`][delivery]):

| # | Source | Notes |
|---|---|---|
| 1 | explicit path / `NUMSHARP_PARITY_BLAS` | **binding** — never substituted, a miss is fatal |
| 2 | `NUMSHARP_OPENBLAS_PATH`, then a build-recorded `OpenBlasPath` marker | override path(s), ahead of the bundle; **non-binding** — fall through when they hold no BLAS |
| 3 | build-staged **version override** (`openblas.source.json`, `mode: "version"`) | **hard-required** — a miss throws `BlasRequiredOverrideException`, never falls through; sha-verified when the marker pins one |
| 4 | the **bundled** `runtimes/<rid>/native/` asset | the zero-config parity default; `NUMSHARP_BLAS_BUNDLED=0` drops it |
| 5 | machine-wide OpenBLAS (apt / brew / MacPorts / conda / vcpkg / source installs) | honours `OPENBLAS_HOME` / `VCPKG_ROOT` / `CONDA_PREFIX`; a different compiler and build, so **not** byte-parity — a correct, fast BLAS, not a bit-identical one |
| 6 | bare loader names | `libscipy_openblas64_`, `libscipy_openblas`, `scipy_openblas`, `libopenblas`, `libblas`, … |
| 7 | every directory on `PATH` | **last resort** — sweeps for a BLAS under a non-standard name; never runs when an earlier tier binds |

Two design decisions worth understanding:

**A pip-installed numpy's `numpy.libs/` is never scanned.** The bundled asset already *is* that
binary at the pinned version, so results stay a property of the package version rather than of
whichever python happens to be installed — a library's numeric output must not change because an
unrelated `pip install` ran. (A conda or system *OpenBLAS* is still machine tooling, tier 5; a
*numpy* is not.) To match a NumPy whose OpenBLAS differs from the bundle, name it — binding — or
point `NUMSHARP_OPENBLAS_PATH` at it to take priority while keeping the fallback:

```csharp
Blas.Enable(@"…\site-packages\numpy.libs\libscipy_openblas64_-<hash>.dll");
```

**The source marker is what tells an override from the bundle.** A build-staged version override
lands in the *same* `runtimes/<rid>/native/` layout as the bundle — and the delivery model's
"same binary" invariant can make the two content-identical — so only the `openblas.source.json`
sidecar the build writes can tell them apart. The same folder is tier 3 (hard-required) when the
marker declares a version override, and tier 4 (the soft default) when there is no marker. A
required override that cannot load **throws rather than substituting** — at module load the error
goes to stderr and the backend stays uninstalled (never silently replaced); an explicit
`Blas.Enable()` raises the full `BlasRequiredOverrideException`.

<sub>See here [`AutoDiscovery_PrefersTheBundledLibrary`][gate-delivery], [`VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle`][gate-delivery], [`NumpyLibs_DiscoveryTier_IsDeleted`][gate-delivery]</sub>

---

## Overriding the version at build time

The override is configured **on the `PackageReference`** — no second package:

```xml
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasVersion>0.3.31.22.0</OpenBlasVersion>            <!-- scipy-openblas version, enforced -->
  <!-- optional: read-from / download-to directory (relative resolves against the project) -->
  <!-- <OpenBlasPath>native/openblas</OpenBlasPath> -->
</PackageReference>
```

At build, the pinned scipy-openblas wheel is fetched from PyPI, verified twice (the wheel's sha256
from the index, then the extracted library's), the one native library is extracted into a global
per-user cache, staged over the bundle in the output, and the `openblas.source.json` marker is
written so the runtime treats the folder as a *required* override (discovery tier 3). **A version
override is a hard requirement**: a build that cannot download or verify it **fails**, and a
runtime that cannot load it **throws** — never a silent fallback to different bits, because
silently running with different bits is the one failure a pin exists to prevent.

| Metadata | Env (wins) | Meaning | Default |
|---|---|---|---|
| `OpenBlasVersion` | `NUMSHARP_OPENBLAS_VERSION` | scipy-openblas version to fetch from PyPI — **enforced** | — (the bundle) |
| `OpenBlasPath` | `NUMSHARP_OPENBLAS_PATH` | directory to **read from** — and **download to** when a version is also set; alone (no version) it is the soft "look here" form, non-binding | — |
| `OpenBlasDistribution` | `NUMSHARP_OPENBLAS_DISTRIBUTION` | `scipy-openblas64` \| `scipy-openblas32` (`64`/`32` aliases) | per-RID from the manifest |
| `OpenBlasFeed` | `NUMSHARP_OPENBLAS_FEED` | PyPI JSON base (mirror hook); a non-pypi.org feed **requires** an explicit sha | `pypi.org` |
| `OpenBlasSha256` | `NUMSHARP_OPENBLAS_SHA256` | expected extracted-library hash | the manifest pin, for the default version |
| `OpenBlasDelivery` | `NUMSHARP_OPENBLAS_DELIVERY` | `none` \| `build` \| `package` | `build` |

The moving parts, briefly:

- **Environment beats metadata** for every knob; among multiple references carrying a version, the
  highest wins (logged), and a version-vs-bare-path conflict across references is a hard error.
- **The cache** (`%LOCALAPPDATA%\NumSharp\openblas` on Windows, `$XDG_CACHE_HOME/NumSharp/openblas`
  or `~/.cache/NumSharp/openblas` elsewhere) holds **extracted libraries only** — the wheel is deleted after extraction — keyed by
  `(distribution, version, rid, sha256)`. Repeated and offline builds serve from it, and it
  **self-heals**: every hit re-verifies the entry's content hash, and a truncated or tampered entry
  is discarded and re-downloaded; writes are temp-file + rename, so a killed or racing build cannot
  strand garbage under the hash's name.
- **The default path costs nothing**: with no version/path set, the resolve step is a cheap item
  scan, the download task is never even compiled, and no network is touched. Runtime never touches
  the network at all — all fetching is a build concern.
- **Transitive by design**: the machinery ships in `buildTransitive/`, which NuGet flows to direct
  *and* transitive consumers — so a package that merely *depends* on this one gives its consumers
  the identical override experience; the targets read the `OpenBlas*` metadata off **any**
  `PackageReference`, including one pointing at the intermediate package.
- **`OpenBlasDelivery=package`** lets a package author bake a pinned version into *their own*
  nupkg: it stages **all** supported RIDs plus their markers as `runtimes/<rid>/native/` pack
  content, and NuGet's nearest-wins rule makes that copy beat this package's transitive bundle.
  `none` opts a reference out; `build` (the default) stages into the consuming app's output only.
- **`NumSharpOpenBlasEnabled=false`** (MSBuild property) switches the entire build-side machinery
  off.

The whole flow is exercised end-to-end against the real packed nupkg by
[`verify_build_override.sh`][gate-build] — 15 scripted steps including the adversarial half:
a tampered artifact fails the build with a checksum mismatch, a mirror without an explicit sha is
refused before any network, an invalid `OpenBlasDelivery` value names the valid set, a poisoned
cache entry is discarded and re-downloaded, and a two-reference version-vs-path conflict is a hard
error.

<sub>See here [`verify_build_override.sh`][gate-build], [`VersionMarker_WithMismatchedSha_RefusesTheFile`][gate-delivery], [`BundleAutoinstall_OnARequiredOverrideMiss_IsLoudButDoesNotThrow`][gate-delivery]</sub>

---

## How it computes

### One seam: a property on the engine

The package implements NumSharp's `IBlasBackend` and assigns it to the engine's settable
`TensorEngine.Blas` property from its module initializer. The engine is **not** replaced or
subclassed — deliberately: an `NDArray` binds its engine at construction and `np.dot(a, b)`
dispatches on `a.TensorEngine`, so swapping the engine would only reach arrays created afterwards,
while a property on the cached singleton takes effect for arrays that already exist.

```csharp
// The entire Core-side surface:
public interface IBlasBackend
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;  // optional
    // + 15 more optional Try* members — see "Writing your own backend"
}
public abstract partial class TensorEngine { public IBlasBackend Blas { get; set; } }
```

Every member is `Try`-shaped: the backend answers only for the operands it implements and returns
`false` for the rest, which the engine then computes itself. **Installing a backend can change
WHICH implementation runs, never WHETHER NumSharp can compute the product** — that invariant is
what makes referencing the package safe. It answers for float32/float64 only; integer and bool
products fall through and are bit-exact by construction anyway (modular integer addition is
associative, so summation order cannot change the result).

### Both NumPy dispatchers, route for route

`np.dot` and `np.matmul` are **not the same C code** in NumPy — `cblas_matrixproduct` (plus the
N-D `dotfunc` tail) versus the `@TYPE@_matmul` gufunc — and they disagree when an operand is not
blasable: a stride-2 matrix times a vector gets gemv-on-a-copy from one and the portable loop from
the other, and 278/300 elements differ. A backend claiming bit-parity must reproduce that split,
which is why the interface has both `TryDot` and `TryMatMul2D` rather than one matrix-product
method. The ported routes:

| Route | When |
|---|---|
| `cblas_?gemm` | matrix @ matrix — with NumPy's copy-into-a-temp rule for non-blasable operands and its transpose choices |
| `cblas_?syrk` | `a @ a.T` — both dispatchers detect the shared data pointer and compute one triangle, mirrored |
| `cblas_?gemv` | matrix @ vector, both directions (operands swapped for vector · matrix) |
| chunked `cblas_?dot` | row · column — summed in a **double** accumulator, exactly as NumPy's `FLOAT_dot` does "for stability" (this route is *more* accurate than gemm, and parity must reproduce that too) |
| `cblas_?axpy` | the scalar-multiply branch of `cblas_matrixproduct` |
| portable loop | zero-sized extents and dimensions above `BLAS_MAXSIZE` — NumPy excludes these from its blas routes, so the backend declines them and the engine settles them in the same place NumPy does |
| the N-D `dotfunc` tail | `np.dot` above 2-D — **not** gemm at all: a `?dot` per (row, column-plane) pair, as NumPy walks it |

Strides are ported in **elements** (NumPy's are bytes — the same logic with `itemsize == 1`), and
real strides pass through to the BLAS: for mat @ mat the layout does not change the bits, but for
the special routes it does (a `(1,500)` vector with stride 2 changes gemv's `incX` and the result
with it).

`TryMatMulBatched` exists because the seam's granularity is a performance decision: it takes a
whole stacked product at once, so the route decision and any scratch allocation are built once per
stack — the way NumPy hoists them out of its gufunc's outer loop — instead of once per element.
Measured on 2000 stacked 8×8 float32 products, per-element setup made the backend **0.80×** —
*slower* than NumSharp's own managed GEMM — against **6.47×** with the work hoisted; the two routes
are bit-identical (25/25 stacked products A/B'd through a decorator that declines the batched
path).

### What is *not* served, and why that is correct

| Dtype | What happens | Why |
|---|---|---|
| int / uint / bool | managed kernels | bit-exact by construction — summation order cannot change modular addition |
| `Half` | managed kernels | NumPy does **not** send float16 to BLAS — its portable C loop *is* the shipping implementation, which NumSharp mirrors (accumulating in float32, as NumPy does) — so the managed answer is already NumPy's |
| `Complex` | managed kernels — **the one real gap** | complex64/128 *are* cblas dtypes (`cgemm`/`zgemm`), so NumPy's complex products are zgemm's bits and the managed kernel diverges the same way float32 did. The seam already supports a complex backend (probed with a third-party one); the routes are simply not written yet |

<sub>See here [`EveryShapeRoute_RoundTripsThroughTheBackend`][gate-backend], [`StackedProducts_GoThroughTheBatchedEntryPoint_Once`][gate-backend], [`IntegerProducts_FallThroughToTheManagedKernel`][gate-backend]</sub>

---

## Writing your own backend

A backend needs **no access to NumSharp's internals** — the probe that verified this wrote a
working backend (and then a Complex128 one) in an assembly with no friend access, no interface
change, no Core change:

```csharp
public sealed class MyBackend : NumSharp.Backends.IBlasBackend
{
    public string Info => "MyBackend";

    public unsafe bool TryDot(NDArray a, NDArray b, out NDArray result)
    {
        // read operands as: base pointer + offset, with element strides
        var pa = (double*)a.GetData().Address + a.Shape.Offset;
        var strides = a.Shape.Strides;               // element units, not bytes
        // … compute, or:
        result = null; return false;                 // decline → the engine computes it
    }

    public bool TryMatMul2D(NDArray a, NDArray b, NDArray result) => false;
}

BackendFactory.GetEngine().Blas = new MyBackend();   // installed; null uninstalls
```

**The one trap:** do not read operands through `a.GetData<T>().Address`. It *densifies* a
non-contiguous view, so pairing it with `a.Shape.Strides` reads outside the buffer and silently
returns wrong numbers on exactly the sliced and transposed operands a matrix product sees most.
The correct public recipe is `(T*)a.GetData().Address + a.Shape.Offset` together with
`a.Shape.Strides`.

`TryMatMulBatched` and the fifteen linear-algebra members are **default interface methods**
returning `false`, so a backend written against the two core members compiles and runs untouched —
growing the interface without breaking an implementer is itself part of the contract, and pinned
by a test. The fifteen cover the rest of what NumPy routes through this same binary:

| Group | Members | When no backend answers |
|---|---|---|
| Products (CBLAS) | `TryInner`, `TryVdot`, `TryVecdot`, `TryMatvec`, `TryVecmat` | **managed kernels compute it** — the invariant holds |
| Factorisations (LAPACK) | `TryCholesky`, `TryDet`, `TrySlogdet`, `TryEig`, `TryEigh`, `TryInv`, `TryLstsq`, `TryQr`, `TrySolve`, `TrySvd` | **`NotSupportedException`** — Core ships no managed LU/QR/SVD/eigensolver, so there is nothing to fall back to; the message names the NumPy API, the LAPACK routine and the `Try*` member to implement |

LAPACK sits on the *same* property because it lives inside the very binary already bundled:
scipy-openblas ships `gesv` / `getrf` / `potrf` / `geev` / `syevd` / `heevd` / `gesdd` / `geqrf` /
`orgqr` / `gelsd` next to the BLAS symbols, so a backend that loaded the library for `sgemm`
already holds the handle for everything else. The NumPy-exact validation for all of `np.linalg`
runs *before* the seam and is gated today — an implementation only has to supply numerics.

<sub>See here [`ABackendThatKnowsNothingOfBatching_StillWorks`][gate-backend], [`TheBackendIsAPlainSettableProperty`][gate-backend]</sub>

---

## Performance

The parity route is also the fast route — OpenBLAS, after all. Single products (Release,
best-of-rounds, milliseconds):

| shape | dtype | managed (Core) | OpenBLAS backend | backend, 1 thread |
|---|---|---|---|---|
| 128×784×128 | f32 | 0.436 | **0.233** | 0.227 |
| 256³ | f32 | 0.604 | **0.296** | 0.291 |
| 1024³ | f32 | 38.77 | **15.97** | 16.79 |
| 2048³ | f32 | 282.2 | **118.3** | 124.3 |
| 128×784×128 | f64 | 0.966 | **0.435** | 0.426 |
| 1024³ | f64 | 134.5 | **33.8** | 32.3 |
| 2048³ | f64 | 3860 | **292.6** | 310.6 |

Stacked products go through the batched entry point (route decision and scratch hoisted per
stack, not per element):

| workload | managed ms | backend ms | ratio |
|---|---|---|---|
| 2000 × (8,8) f32 | 2.34 | **0.361** | 6.47× |
| 500 × (32,32) f32 | 1.93 | **0.850** | 2.27× |
| 8 × (256,256) f32 | 4.78 | **2.33** | 2.05× |
| 2000 × (8,8) int32 (declined) | 2.51 | 2.65 | 0.95× — the seam itself is free |

---

## Environment

| Variable | Effect |
|---|---|
| `NUMSHARP_PARITY_BLAS` | Path (or directory) of the CBLAS library to load. **Binding**, like an explicit argument — used as given, never substituted. |
| `NUMSHARP_OPENBLAS_PATH` | File(s)/dir(s) to scan for a CBLAS, path-separator delimited. Tried **first** (ahead of the bundled asset), non-binding — falls through when it holds no BLAS. Also the build-time read/download directory. |
| `NUMSHARP_BLAS_BUNDLED=0` | Drop the bundled asset from discovery; machine tooling becomes the default. |
| `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0` | Skip the module-load auto-install; `Blas.Enable(...)` still works. (The pre-rename `NUMSHARP_BLAS_AUTOINSTALL` spelling is retired and ignored.) |
| `NUMSHARP_OPENBLAS_VERSION` | **Build-time**: scipy-openblas version to download from PyPI and stage over the bundle — a hard requirement; beats `<OpenBlasVersion>` metadata. |
| `NUMSHARP_OPENBLAS_{DISTRIBUTION, FEED, SHA256, DELIVERY}` | **Build-time**: distribution pick (`64`/`32`), PyPI mirror base (needs the sha), expected extracted-lib sha256, delivery mode (`none`/`build`/`package`); each beats its `OpenBlas*` metadata twin. |
| `OPENBLAS_HOME` / `OPENBLAS_ROOT` / `VCPKG_ROOT` / `CONDA_PREFIX` | Roots consulted when scanning for a machine-wide OpenBLAS (discovery tier 5). |
| `OPENBLAS_CORETYPE` | Read by OpenBLAS itself at load. Set it in the *real* environment (before process start) to pin NumPy and NumSharp at once. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS' own thread-count variable — the way to pin the NumPy side of a parity run. |

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Blas.Enabled` is `false` after referencing the package | No CBLAS could be found or loaded — an asset-free package build, an unsupported RID, or the opt-out variables. `Blas.Enable()` throws the actual reason; `Blas.Info` / `Blas.LibraryPath` say what, if anything, is loaded. |
| `BlasRequiredOverrideException` | The build pinned an `OpenBlasVersion` and the staged override cannot be loaded (or nothing matches its pinned sha). The pin is a contract — rebuild so the staging runs, or remove the pin. At module load this is one stderr line and an uninstalled backend, never a silent substitution. |
| `PlatformNotSupportedException` from `Enable(coreType:)` | The named kernel needs an ISA this CPU lacks; running it would kill the process, so it is refused up front. |
| Process dies with *illegal instruction* | An `OPENBLAS_CORETYPE` above this CPU was set in the raw environment, bypassing `Enable`'s check. Unset it or name a kernel the CPU can run. |
| `EntryPointNotFoundException: … exports no recognizable cblas_sgemm symbol` | The named file is not a CBLAS provider (a plain BLAS without the `cblas_` interface, or an unrelated library). The working setup, if any, is untouched. |
| `Enable(coreType:)` throws although the library is already loaded | The kernel is chosen once, at load. If the request already holds it is a no-op; if not, it throws rather than silently leaving the bits un-pinned. Changing kernels needs a fresh process — a loaded BLAS cannot be safely unloaded. |
| Results differ from NumPy on the same machine | Thread counts differ — pin both sides (`Blas.Enable(threads: n)` and `OPENBLAS_NUM_THREADS=n` before `import numpy`). Or the local numpy's OpenBLAS is a different version than the bundle — name its library (binding) or point `NUMSHARP_OPENBLAS_PATH` at it. Or the dtype is complex — not served yet, see [How it computes](#how-it-computes). |
| Results differ across machines | `DYNAMIC_ARCH` dispatched different micro-kernels. Pin `coreType: Blas.ParityCoreType` (and `OPENBLAS_CORETYPE` for NumPy) on **both** sides. |
| Setting `OPENBLAS_CORETYPE` via `Environment.SetEnvironmentVariable` does nothing | .NET keeps its own environment table; a native `getenv` never sees it. Set it in the real environment before the process starts, or pass `coreType:` to `Enable` — which goes through the C runtime. |
| A build with a version override fails offline | A version override is a hard requirement by design. Prime the cache with one online build, point `OpenBlasFeed` at a mirror (with an explicit `OpenBlasSha256`), or drop the pin. |
| `NUMSHARP_BLAS_AUTOINSTALL=0` no longer suppresses the install | That spelling is retired outright (it never shipped in a release) and deliberately ignored. Use `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0`. |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | Referencing the package installs the backend at assembly load; the env opt-out suppresses it | backend present with no explicit call; absent under `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0` | [`BundleAutoinstall_WithNoOptOut_InstallsTheBackend`][gate-delivery], [`BundleAutoinstall_HonorsTheOptOut`][gate-delivery] |
| 2 | Without the package the engine has no BLAS backend and every product is managed | `engine.Blas == null` by default | [`NotInstalled_ByDefault_TheEngineHasNoBlasBackend`][gate-backend] |
| 3 | Products through the backend are bit-identical to numpy 2.4.2 | 342/342 corpus cases bit-exact; 294 divergent without the backend | [`FuzzCorpusTests.MatmulParity`][gate-corpus] |
| 4 | The corpus pin identifies the library by content hash and goes Inconclusive — never red — on a foreign host | `blas_library_sha256` compared against the loaded file | [`MatmulParityPin`][gate-pin] |
| 5 | The bundled binary hashes to the committed manifest pin | per-RID sha256 equals `openblas-manifest.json` | [`BundledLibrary_HashesToThePinnedManifest`][gate-backend] |
| 6 | Auto-discovery prefers the bundle over machine tooling; the bundle can be opted out | bundled path wins; `NUMSHARP_BLAS_BUNDLED=0` drops it | [`AutoDiscovery_PrefersTheBundledLibrary`][gate-backend], [`BundledLibrary_CanBeOptedOutOf`][gate-backend] |
| 7 | A named library is binding — a missing one does not fall back to the bundle | throw, no substitution | [`NamedLibrary_OutranksTheBundledOne_AndAMissingOneDoesNotFallBackToIt`][gate-backend] |
| 8 | A failed `Enable` changes nothing, even over a working setup | working library and backend intact after a failed call | [`FailedEnable_OverAWorkingLibrary_ChangesNothing`][gate-backend], [`FailedEnable_OnANonCBlasFile_OverAWorkingLibrary_ChangesNothing`][gate-backend] |
| 9 | `Enabled` means installed **and** bound | `false` when a backend is installed with no library | [`Enabled_MeansInstalledAND_Bound_NotMerelyInstalled`][gate-backend] |
| 10 | The thread pin takes effect and is reported back | `Info` reflects `Enable(threads:)` | [`ThreadPin_IsReportedBack`][gate-backend] |
| 11 | A `coreType` this CPU cannot run is refused instead of crashing the process | `PlatformNotSupportedException`, process alive | [`CoreTypeAboveThisCpu_IsRefused_RatherThanCrashingTheProcess`][gate-backend] |
| 12 | `ParityCoreType` is the kernel the committed corpus was generated with | equals the host pin's recorded core | [`ParityCoreType_MatchesTheCorpusHostPin`][gate-backend] |
| 13 | Only float32/float64 products change; integers fall through; every other operation is untouched | identical bytes backend on vs off | [`IntegerProducts_FallThroughToTheManagedKernel`][gate-backend], [`EveryOtherOperation_StaysTheManagedImplementation`][gate-backend] |
| 14 | Stacked products take the batched entry once per stack, bit-identical to the per-element route | call counting + 25/25 A/B | [`StackedProducts_GoThroughTheBatchedEntryPoint_Once`][gate-backend], [`BatchedEntryPoint_IsBitIdenticalToThePerElementRoute`][gate-backend] |
| 15 | A backend implementing only the two core members still works | decorator declining the batched path | [`ABackendThatKnowsNothingOfBatching_StillWorks`][gate-backend] |
| 16 | A version override is hard-required and sha-verified — never silently substituted | throws with the bundle sitting loadable beside it; a mismatched sha is refused | [`VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle`][gate-delivery], [`VersionMarker_WithMismatchedSha_RefusesTheFile`][gate-delivery] |
| 17 | A required-override miss at module load is loud on stderr, not fatal | one stderr line; backend left uninstalled | [`BundleAutoinstall_OnARequiredOverrideMiss_IsLoudButDoesNotThrow`][gate-delivery] |
| 18 | A path override is non-binding: it binds when it holds a library, falls through when it does not | both directions | [`PathMarker_BindsItsDirectory_WhenItHoldsALibrary`][gate-delivery], [`PathMarker_IsNonBinding_FallsThroughToTheBundle`][gate-delivery] |
| 19 | A pip-installed numpy's `numpy.libs/` is never scanned | the tier stays deleted | [`NumpyLibs_DiscoveryTier_IsDeleted`][gate-delivery] |
| 20 | The build pipeline — download, double verify, self-healing cache, staging, packing — works end-to-end against the real nupkg | 15 scripted steps incl. tamper, mirror-needs-sha, poisoned cache, conflicting references | [`verify_build_override.sh`][gate-build] |

---

## See also

- [Interoperability](index.md) — the memory-bridge contract behind the other `NumSharp.Interop.*`
  packages (this one shares a *binary*, not a buffer)
- [`docs/GEMM_PARITY.md`][gemm] — the full parity record: probe results, the ported routes, the
  design review, the host pin, the traps
- [`docs/OPENBLAS_DELIVERY_DESIGN.md`][delivery] — the delivery/discovery model: the two phases,
  the source marker, the transitive story, every resolved decision
- [Benchmarks](../benchmarks-dashboard.md) — NumSharp-vs-NumPy performance across the whole API

[gate-backend]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Backends/MatmulParityBackendTests.cs
[gate-delivery]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Backends/OpenBlasDeliveryTests.cs
[gate-corpus]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Fuzz/FuzzCorpusTests.cs
[gate-pin]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Fuzz/MatmulParityPin.cs
[gate-build]: https://github.com/SciSharp/NumSharp/blob/master/src/NumSharp.Interop.OpenBLAS/tools/verify_build_override.sh
[gemm]: https://github.com/SciSharp/NumSharp/blob/master/docs/GEMM_PARITY.md
[delivery]: https://github.com/SciSharp/NumSharp/blob/master/docs/OPENBLAS_DELIVERY_DESIGN.md
