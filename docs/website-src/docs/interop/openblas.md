# OpenBLAS — a native BLAS backend for `np.dot` / `np.matmul`

`NumSharp.Interop.OpenBLAS` is the optional native matrix-product backend. Reference it and
`np.dot` / `np.matmul` compute `float32` / `float64` products through OpenBLAS instead of NumSharp's
own managed SIMD GEMM — **1.7–13× faster on large matrices**. It bundles the prebuilt OpenBLAS
binaries NumPy and SciPy themselves ship, per-RID, so you get a proven, tuned BLAS with nothing to
install and no Python on the machine. NumSharp.Core stays 100 % managed C# with no native
dependency: with this package absent, every kernel — matrix products included — is NumSharp's own
code, and (as everywhere in NumSharp) 1-to-1 with NumPy. This page is the package's reference: what
it buys, the quick start, the bundled binary, how the binary is delivered, reproducibility, the API,
how it computes, extending the seam, performance and troubleshooting.

**On this page:** [What it buys](#what-it-buys) · [Quick start](#quick-start) ·
[The bundled binary](#the-bundled-binary) · [Delivering the binary](#delivering-the-binary) ·
[Where the library comes from](#where-the-library-comes-from) ·
[Reproducible results](#reproducible-results) · [The API](#the-api) ·
[How it computes](#how-it-computes) · [Writing your own backend](#writing-your-own-backend) ·
[Performance](#performance) · [Environment](#environment) · [Troubleshooting](#troubleshooting) ·
[Claims](#claims-ledger)

> Bundles scipy-openblas 0.3.31.22.0 (OpenBLAS 0.3.31.dev — the build numpy 2.4.2 ships) · net8.0/net10.0.
> The behavioural claims below are pinned by gates: [`MatmulParityBackendTests`][gate-backend] (30
> tests), [`OpenBlasDeliveryTests`][gate-delivery] (20), and the 15-step scripted build gate
> [`verify_build_override.sh`][gate-build].

---

## What it buys

NumSharp computes every matrix product itself, in managed SIMD code, out of the box — correct and
portable, with no native dependency. What a hand-written managed GEMM cannot do is keep up with a
decades-tuned native BLAS on large matrices: OpenBLAS' arch-specific, blocked, multi-threaded
micro-kernels are simply faster once the matrices grow. This package routes `float32` / `float64`
products through OpenBLAS — the same library NumPy and SciPy use — and leaves everything else exactly
as it was.

The speedup is real and grows with size: **1.7–13×** over the managed GEMM across the measured
range, up to a `2048³` float64 product dropping from 3860 ms to 293 ms
(see [Performance](#performance)).

A native BLAS is a *performance* choice, and — as with swapping any BLAS — a correct floating-point
product can differ in its last bits from another implementation, or from the same one at a different
thread count or on a different CPU, because summation order changes. That is ordinary IEEE behaviour,
not a defect; if you need results that reproduce exactly, [pin the two variables that move
them](#reproducible-results).

---

## Quick start

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

<sub>See here [`BundleAutoinstall_WithNoOptOut_InstallsTheBackend`][gate-delivery], [`Enable_SetsTheBackendOnTheEngine_WithoutReplacingIt`][gate-backend]</sub>

---

## The bundled binary

**The package ships the OpenBLAS binaries NumPy itself ships** — not "an" OpenBLAS. NumPy 2.x does
not build its own: its wheels bundle the prebuilt `scipy-openblas32` / `scipy-openblas64` artifacts
published on PyPI by the openblas-libs project, pinned in numpy's `requirements/ci_requirements.txt`.
This package bundles those same artifacts at the same pinned version — the identical PyPI wheel,
its SHA-256 checked against the manifest at pack time — so what you load is a widely-deployed,
battle-tested build, with no Python installed and no loader-race guesswork about which BLAS won.

Bundled version: **scipy-openblas 0.3.31.22.0** (OpenBLAS 0.3.31.dev), as `runtimes/<rid>/native/`
NuGet assets for 8 RIDs (62 MB packed), which NuGet resolves per machine so a RID-specific publish
carries only the one binary:

| RID | distribution | RID | distribution |
|---|---|---|---|
| `win-x64` | scipy-openblas64 | `linux-musl-x64` | scipy-openblas64 |
| `win-arm64` | scipy-openblas**32** † | `linux-musl-arm64` | scipy-openblas64 |
| `linux-x64` | scipy-openblas64 | `osx-x64` | scipy-openblas64 |
| `linux-arm64` | scipy-openblas64 | `osx-arm64` | scipy-openblas64 |

† not an inconsistency — NumPy's own build selects the LP64 distribution on win-arm64, so this
follows suit. Both scipy distributions are first-class here: **scipy-openblas64** (ILP64, 64-bit
BLAS integers, symbols `scipy_cblas_sgemm64_`) and **scipy-openblas32** (LP64, `scipy_cblas_sgemm`),
and the 32-bit distribution is the *only* build PyPI publishes for 32-bit x86.

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

## Delivering the binary

Getting a binary onto disk and choosing which one to bind are two separate phases, and keeping
them apart is the whole delivery design (specified in [`OPENBLAS_DELIVERY_DESIGN.md`][delivery]):

| Phase | Runs in | Decides | Network |
|---|---|---|---|
| **Build** (MSBuild targets shipped in the nupkg) | the consumer's `dotnet build` / `publish` / `pack` | which native library lands in the output's `runtimes/<rid>/native/` (or in a nupkg) — plus a **source marker** recording where it came from | only for a version override |
| **Runtime** (the module initializer) | the app process, at assembly load | which on-disk library to load and wire as the backend | **never** |

So every delivery method is a way of arranging what the runtime will find. There are five,
quickest first — then a subsection each.

**1 · The bundle — reference the package, done.** The pinned binary ships inside the nupkg and
auto-installs at load. Nothing to configure:

```bash
dotnet add package NumSharp.Interop.OpenBLAS
```

**2 · Pin a different version — build-time, enforced.** The build downloads that scipy-openblas
release from PyPI, verifies it twice and stages it over the bundle. The pin is a contract — a
build that cannot satisfy it fails, a runtime that cannot load it throws:

```xml
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasVersion>0.3.34.106.0</OpenBlasVersion>
</PackageReference>
```

```bash
NUMSHARP_OPENBLAS_VERSION=0.3.34.106.0 dotnet build   # same thing — env beats metadata
```

**3 · Read from a directory you manage — build-time, soft.** No download, no enforcement: "look
here first", fall through when it holds no BLAS:

```xml
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasPath>native/openblas</OpenBlasPath>   <!-- relative resolves against the project -->
</PackageReference>
```

**4 · Deliver through your own package.** A package that merely *depends* on this one passes the
whole experience through — its consumers put the same metadata on *their* reference to *your*
package — and it can bake a pinned version into its own nupkg so they download nothing:

```xml
<!-- a consumer of MyPackage (which depends on NumSharp.Interop.OpenBLAS): -->
<PackageReference Include="MyPackage" Version="1.0.0">
  <OpenBlasVersion>0.3.34.106.0</OpenBlasVersion>       <!-- honoured as if direct -->
</PackageReference>

<!-- or in MyPackage.csproj itself — bakes all 8 RIDs into MyPackage.nupkg: -->
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasVersion>0.3.34.106.0</OpenBlasVersion>
  <OpenBlasDelivery>package</OpenBlasDelivery>
</PackageReference>
```

**5 · Name or point at a library at runtime.** No build involvement at all — an explicit path is
**binding** (used as given or the call throws, never substituted), the env path is **priority
but non-binding**:

```csharp
Blas.Enable(@"C:\opt\openblas\libopenblas.dll");   // binding — exactly this file, or throw
```

```bash
NUMSHARP_PARITY_BLAS=/opt/blas/libopenblas.so ./app   # binding — the env spelling of the same
NUMSHARP_OPENBLAS_PATH=/opt/myblas ./app              # tried before the bundle, falls through
```

Which one you want:

| You want | Use | Guarantee |
|---|---|---|
| a proven, tuned BLAS with zero config | the bundle (1) | the pinned scipy-openblas (what NumPy ships), by default |
| a newer / different scipy-openblas, guaranteed | version override (2) | **hard-required** — build fails / runtime throws rather than substituting |
| your own build or a vendored copy | path override (3), or `NUMSHARP_OPENBLAS_PATH` (5) | soft — wins when present, falls through when not |
| the same knobs for consumers of *your* package | transitive metadata (4) | identical to a direct reference |
| a self-contained nupkg, no downloads for consumers | `OpenBlasDelivery=package` (4) | NuGet nearest-wins beats the transitive bundle |
| exactly this one file, or fail | `Blas.Enable(path)` / `NUMSHARP_PARITY_BLAS` (5) | **binding** — never silently replaced |

When several are present at once, [the discovery order](#where-the-library-comes-from) decides;
`NumSharpOpenBlasEnabled=false` (MSBuild property) switches the entire build-side machinery off.

### 1 · The bundle

What the binary *is* — provenance, RIDs, symbol schemes, licensing — is
[The bundled binary](#the-bundled-binary) above; this is how it travels. The nupkg carries all
8 RIDs as `runtimes/<rid>/native/` assets and NuGet resolves them per machine: a plain
`dotnet build` gets the host RID's library copied beside the output, a portable publish carries
the full `runtimes/` tree, and a RID-specific publish flattens the one matching binary next to
the app. At load, the module initializer finds it (discovery tier 4) and installs the backend.
If the asset is absent — an unsupported RID, `NUMSHARP_BLAS_BUNDLED=0`, an asset-free source
build — discovery simply continues to machine tooling and, failing that, NumSharp's managed
kernels. `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0` skips the automatic install while leaving
`Blas.Enable()` fully functional.

<sub>See here [`BundledLibrary_ForThisRid_ReachesTheBuildOutput`][gate-backend], [`AutoDiscovery_PrefersTheBundledLibrary`][gate-backend], [`BundledLibrary_CanBeOptedOutOf`][gate-backend]</sub>

### 2 · Version override (build-time, enforced)

The full knob surface — metadata on the `PackageReference`, each with an environment twin that
**beats** it:

| Metadata | Env (wins) | Meaning | Default |
|---|---|---|---|
| `OpenBlasVersion` | `NUMSHARP_OPENBLAS_VERSION` | scipy-openblas version to fetch from PyPI — **enforced** | — (the bundle) |
| `OpenBlasPath` | `NUMSHARP_OPENBLAS_PATH` | directory to **read from** — and **download to** when a version is also set | — |
| `OpenBlasDistribution` | `NUMSHARP_OPENBLAS_DISTRIBUTION` | `scipy-openblas64` \| `scipy-openblas32` (`64`/`32` aliases) | per-RID from the manifest |
| `OpenBlasFeed` | `NUMSHARP_OPENBLAS_FEED` | PyPI JSON base (mirror hook); a non-pypi.org feed **requires** an explicit sha | `pypi.org` |
| `OpenBlasSha256` | `NUMSHARP_OPENBLAS_SHA256` | expected extracted-library hash | the manifest pin, for the default version |
| `OpenBlasDelivery` | `NUMSHARP_OPENBLAS_DELIVERY` | `none` \| `build` \| `package` — see subsection 4 below | `build` |

What a version-mode build does, step by step: resolve the knobs (env > metadata) → query the
PyPI JSON index (`<feed>/pypi/<distribution>/<version>/json`) → pick the wheel by this RID's
platform tag, read from the packed manifest so the RID map is *literally* the maintainer
script's (the "same binary" invariant) → download the wheel to a temp file → verify the wheel's
sha256 against the index → extract the one native library member → verify the extracted
library's sha256 (an explicit `OpenBlasSha256`, or the manifest pin when version and
distribution equal the default) → store it in the per-user cache → **delete the wheel** → copy
it over the bundle in the output → write the `openblas.source.json` marker that makes the
folder a *required* override (discovery tier 3).

Enforcement runs in both phases. At build, any failure — network, hash mismatch, no wheel for
the RID — **fails the build**; at runtime, a marker-declared override that cannot load
**throws** `BlasRequiredOverrideException` rather than falling back (at module load: one stderr
line, backend left uninstalled — never substituted). Silently loading a *different* build than
the one pinned is the failure a pin exists to prevent.

The cache (`%LOCALAPPDATA%\NumSharp\openblas` on Windows, `$XDG_CACHE_HOME/NumSharp/openblas`
or `~/.cache/NumSharp/openblas` elsewhere) holds **extracted libraries only**, keyed
`(distribution, version, rid, sha256)`. Repeated and offline builds serve from it once primed,
and it **self-heals**: every hit re-verifies the entry's content hash, a truncated or tampered
entry is discarded and re-downloaded, and writes go through temp-file + rename so a killed or
racing build cannot strand garbage under the hash's name. Offline with an unprimed cache, a
version override fails loudly by design — prime it with one online build, or point
`OpenBlasFeed` at a mirror (which then requires an explicit `OpenBlasSha256`).

Three edge rules: among multiple references carrying a version the **highest wins** (logged); a
version on one reference against a bare path on another is a **hard error**; and the default
path costs nothing — with no override set, the resolve step is a cheap item scan, the download
task is never even compiled, and no network is touched.

The whole build side is exercised end-to-end against the real packed nupkg by
[`verify_build_override.sh`][gate-build] — 15 scripted steps including the adversarial half: a
tampered artifact fails the build with a checksum mismatch, a mirror without an explicit sha is
refused before any network, an invalid `OpenBlasDelivery` value names the valid set, a poisoned
cache entry is discarded and re-downloaded, and a two-reference version-vs-path conflict is a
hard error.

<sub>See here [`VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle`][gate-delivery], [`VersionMarker_WithMismatchedSha_RefusesTheFile`][gate-delivery], [`BundleAutoinstall_OnARequiredOverrideMiss_IsLoudButDoesNotThrow`][gate-delivery], [`verify_build_override.sh`][gate-build]</sub>

### 3 · Path override (build-time, soft)

`OpenBlasPath` alone is "read tooling from here": the build records the directory in a
`mode: "path"` source marker, and at runtime it is probed ahead of the bundle (tier 2) and
silently skipped when it holds no loadable BLAS. A path is a *location*, not a contract — no
version enforcement, no hard failure. A relative path resolves against the project directory.
Combined with a version (`OpenBlasVersion` **and** `OpenBlasPath`), the pin is downloaded
*into* that directory instead of the default output folder — the directory then doubles as
cache and read location, and the hard-required semantics of method 2 apply.

<sub>See here [`PathMarker_BindsItsDirectory_WhenItHoldsALibrary`][gate-delivery], [`PathMarker_IsNonBinding_FallsThroughToTheBundle`][gate-delivery]</sub>

### 4 · Transitive and package-author delivery

The machinery ships in the nupkg's `buildTransitive/` folder, which NuGet flows to **direct and
transitive** consumers alike — the targets run in the final app's build even when the reference
chain is `App → MyPackage → NumSharp.Interop.OpenBLAS`, and they scan **every**
`PackageReference` for the `OpenBlas*` metadata, not just this package's. That is the whole
trick behind method 4's first form: metadata on the intermediate reference delivers exactly as
a direct one would.

`OpenBlasDelivery` tunes what a reference's metadata does:

| `OpenBlasDelivery` | Effect |
|---|---|
| `none` | ignore the OpenBLAS metadata on this reference entirely |
| `build` (default) | stage into the consuming app's output only |
| `package` | ALSO pack the staged assets into **this project's** nupkg — for package authors |

`package` stages **all** manifest RIDs plus their source markers as `runtimes/<rid>/native/`
pack content, and NuGet's nearest-wins rule makes the dependent's copy beat this package's
transitive bundle. Two boundaries: it contradicts a bare `OpenBlasPath` (the combination is
rejected), and it is **not** needed merely to expose the *default* version — the bundle already
flows transitively for free. `package` is for a dependent that wants a *different, pinned*
version baked into its own nupkg — which is also what keeps every dependent nupkg from gaining
62 MB by default.

<sub>See here [`verify_build_override.sh`][gate-build] — the transitive, two-reference-conflict and `delivery=package` steps</sub>

### 5 · Runtime-only: naming and pointing

The two runtime knobs need no build participation, and they differ in exactly one property:

- **Binding** — `Blas.Enable(path)` or `NUMSHARP_PARITY_BLAS`. The named file (or directory) is
  used as given and **never substituted**, not even by the bundled copy sitting in the output;
  a file that cannot load or is not a CBLAS throws, having changed nothing. Use it to pin one
  exact build — a vendored copy, a specific MKL, a library another environment already uses.
- **Priority, non-binding** — `NUMSHARP_OPENBLAS_PATH`, one or more files/directories
  (path-separator delimited). Probed ahead of the bundle, skipped without a word when it holds
  no BLAS — the right knob for "prefer my local build where present".

Below both sits ambient machine tooling (discovery tiers 5–7: system install prefixes, bare
loader names, a PATH sweep) — less a delivery method you choose than what discovery finds when
you chose none and the bundle is absent or opted out. It is a correct, fast BLAS, though a
different build than the bundle (see [Where the library comes from](#where-the-library-comes-from)).

<sub>See here [`NamedLibrary_IsBinding_NeverSilentlySubstituted`][gate-backend], [`NamedLibrary_OutranksTheBundledOne_AndAMissingOneDoesNotFallBackToIt`][gate-backend], [`OverridePathEnv_OutranksTheVersionMarker`][gate-delivery]</sub>

---

## Where the library comes from

Whatever combination of methods delivered binaries, runtime discovery resolves them in one
fixed, documented order — the first tier that yields a loadable library wins:

| # | Source | Notes |
|---|---|---|
| 1 | explicit path / `NUMSHARP_PARITY_BLAS` | **binding** — never substituted, a miss is fatal |
| 2 | `NUMSHARP_OPENBLAS_PATH`, then a build-recorded `OpenBlasPath` marker | override path(s), ahead of the bundle; **non-binding** — fall through when they hold no BLAS |
| 3 | build-staged **version override** (`openblas.source.json`, `mode: "version"`) | **hard-required** — a miss throws `BlasRequiredOverrideException`, never falls through; sha-verified when the marker pins one |
| 4 | the **bundled** `runtimes/<rid>/native/` asset | the zero-config default; `NUMSHARP_BLAS_BUNDLED=0` drops it |
| 5 | machine-wide OpenBLAS (apt / brew / MacPorts / conda / vcpkg / source installs) | honours `OPENBLAS_HOME` / `VCPKG_ROOT` / `CONDA_PREFIX`; a different compiler and build, so it ranks below the bundle — a correct, fast BLAS, not the pinned one |
| 6 | bare loader names | `libscipy_openblas64_`, `libscipy_openblas`, `scipy_openblas`, `libopenblas`, `libblas`, … |
| 7 | every directory on `PATH` | **last resort** — sweeps for a BLAS under a non-standard name; never runs when an earlier tier binds |

Two design decisions worth understanding:

**A pip-installed numpy's `numpy.libs/` is never scanned.** The bundled asset already *is* that
same prebuilt artifact at the pinned version, so which library binds stays a property of the
package version rather than of whichever python happens to be installed — an app's chosen BLAS
should not change because an unrelated `pip install` ran. (A conda or system *OpenBLAS* is still
machine tooling, tier 5; a *numpy* is not.) To bind a NumPy install's own OpenBLAS instead of the
bundle, name it — binding — or point `NUMSHARP_OPENBLAS_PATH` at it to take priority while keeping
the fallback ([method 5](#delivering-the-binary) above).

**The source marker is what tells an override from the bundle.** A build-staged version override
lands in the *same* `runtimes/<rid>/native/` layout as the bundle — and the delivery model's
"same binary" invariant can make the two content-identical — so only the `openblas.source.json`
sidecar the build writes can tell them apart. The same folder is tier 3 (hard-required) when the
marker declares a version override, and tier 4 (the soft default) when there is no marker. A
required override that cannot load **throws rather than substituting** — at module load the error
goes to stderr and the backend stays uninstalled (never silently replaced); an explicit
`Blas.Enable()` raises the full `BlasRequiredOverrideException`.

<sub>See here [`AutoDiscovery_PrefersTheBundledLibrary`][gate-backend], [`VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle`][gate-delivery], [`NumpyLibs_DiscoveryTier_IsDeleted`][gate-delivery]</sub>

---

## Reproducible results

A tuned, threaded, arch-dispatched BLAS sums in an order that depends on the machine and the
settings, so the same correct product can come back with slightly different low bits from one run
to the next. That is normal IEEE floating-point behaviour, shared by NumPy, SciPy and every BLAS
consumer. If you need results that reproduce **exactly** — a golden test, cross-machine
determinism — pin the two variables that move them (the third, the library build, you already
pinned by choosing a specific binary):

1. **The worker-thread count.** 1, 2, 4 and 24 threads give different low bits for the same
   product, because the reduction is split differently. `Blas.Enable(threads: n)` pins it. For a
   run you want to reproduce against a NumPy process, set `OPENBLAS_NUM_THREADS=n` there too
   (before `import numpy`) — NumPy runs larger products multi-threaded by default while small ones
   stay single-threaded, so pinning is what makes small and large agree.
2. **The CPU micro-kernel.** OpenBLAS is built `DYNAMIC_ARCH` and picks a kernel from the CPU it
   runs on; different kernels accumulate differently. Measured by forcing the choice on one host
   (same binary, 1 thread, `(512,784) @ (784,512)` float32):

| `OPENBLAS_CORETYPE` | dispatched | result hash |
|---|---|---|
| *(unset)* | Haswell | `b9ea5057…` |
| `Haswell` | Haswell | `b9ea5057…` — identical to unset |
| `Nehalem` | Nehalem | `b6bb0f58…` |
| `Sandybridge` | Sandybridge | `1903e357…` |
| `Core2` | Katmai | `0da21438…` |
| `SkylakeX` | — | **process killed: illegal instruction** |

`Blas.Enable(coreType: Blas.ParityCoreType)` pins the kernel to `"Haswell"` — a fixed choice that
runs on any x86-64 with AVX2+FMA (Haswell 2013 and later, and every AMD Zen), so a result computed
on one such machine reproduces on another. It is **not the default, deliberately**: the default
lets OpenBLAS pick the fastest kernel the CPU supports (exactly as NumPy does), which is what you
want unless you are specifically chasing cross-machine reproducibility. Pin it on every machine
that must agree, or none; the same `OPENBLAS_CORETYPE` environment variable drives NumPy, since it
is the same library.

> **Hazard.** `coreType` *overrides* OpenBLAS' CPU detection rather than expressing a preference,
> so naming kernels the CPU cannot run terminates the process with an illegal instruction —
> measured, not theoretical. `Blas.Enable` refuses the combinations it can recognise with a
> `PlatformNotSupportedException` first. It also only takes effect on the call that actually
> **loads** the library: OpenBLAS reads the variable once while initialising and never re-reads it,
> so a later request that cannot be honoured throws instead of being silently dropped.

<sub>See here [`ThreadPin_IsReportedBack`][gate-backend], [`CoreTypeAboveThisCpu_IsRefused_RatherThanCrashingTheProcess`][gate-backend]</sub>

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
| `Blas.LibraryPath` | Full path of the loaded library — the machine-readable form of "which binary answered?". |
| `Blas.CoreName` | The `DYNAMIC_ARCH` micro-kernel OpenBLAS actually dispatched (e.g. `"Haswell"`). |
| `Blas.IsBundledLibrary` | `true` when the loaded library is the package's bundled asset — marker-aware: a build-staged override in the same folder layout reports `false`. |
| `Blas.ParityCoreType` | `"Haswell"` — a fixed micro-kernel to pin for cross-machine reproducibility (see [Reproducible results](#reproducible-results)). |

Three semantics are load-bearing, each pinned by a test:

- **A named library is binding.** An explicit `Blas.Enable(path)` (or `NUMSHARP_PARITY_BLAS`) is
  used as given and never silently replaced — not even by the bundled copy sitting in the output
  folder — so "pin exactly this build" means exactly that.
- **A failed `Enable` is a no-op.** It throws having changed nothing, so a mistyped path cannot
  demote a working setup back to the managed kernels behind your back. (An earlier design unloaded
  the working library first, leaving `Enabled` reporting `true` while every product had silently
  reverted to the managed kernel — the worst kind of failure, since the answer still looks fine.)
- **`Enabled` means installed *and* bound.** An installed-but-unbound backend declines every
  operand, so every product would quietly fall back to the managed kernels; reporting `true` there
  would claim a native backend that is not actually computing anything.

Opting out of the automatic install (`NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0`) leaves `Blas.Enable()`
fully functional — it only skips the module-load call.

<sub>See here [`NamedLibrary_IsBinding_NeverSilentlySubstituted`][gate-backend], [`FailedEnable_OverAWorkingLibrary_ChangesNothing`][gate-backend], [`Enabled_MeansInstalledAND_Bound_NotMerelyInstalled`][gate-backend]</sub>

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
products fall through to the managed kernel, which a native BLAS could not improve on anyway
(integer addition is associative, so summation order does not move the result).

### Both `np.dot` and `np.matmul`, route for route

`np.dot` and `np.matmul` are **not the same operation** in NumPy — `cblas_matrixproduct` (plus the
N-D `dotfunc` tail) versus the `@TYPE@_matmul` gufunc — and they handle a non-blasable operand
differently (a stride-2 matrix times a vector: one takes gemv-on-a-copy, the other a portable
loop). NumSharp funnels both APIs through this one seam, so the backend implements **both** routes
to keep `np.dot` and `np.matmul` behaving exactly as they do without it. The ported routes:

| Route | When |
|---|---|
| `cblas_?gemm` | matrix @ matrix — with NumPy's copy-into-a-temp rule for non-blasable operands and its transpose choices |
| `cblas_?syrk` | `a @ a.T` — the shared data pointer is detected and one triangle computed, mirrored |
| `cblas_?gemv` | matrix @ vector, both directions (operands swapped for vector · matrix) |
| chunked `cblas_?dot` | row · column — summed in a **double** accumulator for stability, so this route is *more* accurate than plain gemm |
| `cblas_?axpy` | the scalar-multiply branch of `cblas_matrixproduct` |
| portable loop | zero-sized extents and dimensions above `BLAS_MAXSIZE` — the backend declines these and the engine settles them itself, matching NumPy, which excludes them from its own blas routes |
| the N-D `dotfunc` tail | `np.dot` above 2-D — **not** gemm at all: a `?dot` per (row, column-plane) pair |

Strides are ported in **elements** (NumPy's are bytes — the same logic with `itemsize == 1`), and
real strides pass through to the BLAS: for mat @ mat the layout does not change the result, but for
the special routes it does (a `(1,500)` vector with stride 2 changes gemv's `incX`).

`TryMatMulBatched` exists because the seam's granularity is a performance decision: it takes a
whole stacked product at once, so the route decision and any scratch allocation are built once per
stack — the way NumPy hoists them out of its gufunc's outer loop — instead of once per element.
Measured on 2000 stacked 8×8 float32 products, per-element setup made the backend **0.80×** —
*slower* than NumSharp's own managed GEMM — against **6.47×** with the work hoisted; the batched and
per-element routes produce identical results (25/25 A/B'd through a decorator that declines the
batched path).

### What is *not* served

| Dtype | What happens | Why |
|---|---|---|
| int / uint / bool | managed kernels | a native BLAS buys nothing — integer addition is associative, so order does not move the result |
| `Half` | managed kernels | NumPy does **not** route float16 through BLAS, so there is no native path to gain from; NumSharp's managed half kernel already handles it (accumulating in float32) |
| `Complex` | managed kernels — **the one real gap** | complex64/128 *are* BLAS dtypes (`cgemm`/`zgemm`), so a native backend could accelerate them; the routes are simply not written yet. The seam already supports it (probed with a third-party Complex128 backend) |

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
already holds the handle for everything else. The validation for all of `np.linalg` runs *before*
the seam and is gated today — an implementation only has to supply numerics.

<sub>See here [`ABackendThatKnowsNothingOfBatching_StillWorks`][gate-backend], [`TheBackendIsAPlainSettableProperty`][gate-backend]</sub>

---

## Performance

The native route is the fast route — OpenBLAS, after all. Single products (Release,
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
| `OPENBLAS_CORETYPE` | Read by OpenBLAS itself at load — pins the `DYNAMIC_ARCH` micro-kernel (see [Reproducible results](#reproducible-results)). Set it in the *real* environment, before process start. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS' own worker-thread count. Affects the low bits of a result; set it to reproduce a run at a fixed thread count. |

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Blas.Enabled` is `false` after referencing the package | No CBLAS could be found or loaded — an asset-free package build, an unsupported RID, or the opt-out variables. `Blas.Enable()` throws the actual reason; `Blas.Info` / `Blas.LibraryPath` say what, if anything, is loaded. |
| `BlasRequiredOverrideException` | The build pinned an `OpenBlasVersion` and the staged override cannot be loaded (or nothing matches its pinned sha). The pin is a contract — rebuild so the staging runs, or remove the pin. At module load this is one stderr line and an uninstalled backend, never a silent substitution. |
| `PlatformNotSupportedException` from `Enable(coreType:)` | The named kernel needs an ISA this CPU lacks; running it would kill the process, so it is refused up front. |
| Process dies with *illegal instruction* | An `OPENBLAS_CORETYPE` above this CPU was set in the raw environment, bypassing `Enable`'s check. Unset it or name a kernel the CPU can run. |
| `EntryPointNotFoundException: … exports no recognizable cblas_sgemm symbol` | The named file is not a CBLAS provider (a plain BLAS without the `cblas_` interface, or an unrelated library). The working setup, if any, is untouched. |
| `Enable(coreType:)` throws although the library is already loaded | The kernel is chosen once, at load. If the request already holds it is a no-op; if not, it throws rather than silently leaving the kernel un-pinned. Changing kernels needs a fresh process — a loaded BLAS cannot be safely unloaded. |
| Results are not reproducible run-to-run or machine-to-machine | Expected of any threaded, arch-tuned BLAS — the low bits move with the thread count and CPU kernel. Pin both (`Blas.Enable(threads: n, coreType: Blas.ParityCoreType)`), and the matching `OPENBLAS_NUM_THREADS` / `OPENBLAS_CORETYPE` on any NumPy process you compare against. See [Reproducible results](#reproducible-results). |
| A complex-dtype product is slow | Complex is not served by the backend yet — it stays on the managed kernel. See [How it computes](#how-it-computes). |
| Setting `OPENBLAS_CORETYPE` via `Environment.SetEnvironmentVariable` does nothing | .NET keeps its own environment table; a native `getenv` never sees it. Set it in the real environment before the process starts, or pass `coreType:` to `Enable` — which goes through the C runtime. |
| A build with a version override fails offline | A version override is a hard requirement by design. Prime the cache with one online build, point `OpenBlasFeed` at a mirror (with an explicit `OpenBlasSha256`), or drop the pin. |
| `NUMSHARP_BLAS_AUTOINSTALL=0` no longer suppresses the install | That spelling is retired outright (it never shipped in a release) and deliberately ignored. Use `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0`. |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | Referencing the package installs the backend at assembly load; the env opt-out suppresses it | backend present with no explicit call; absent under `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0` | [`BundleAutoinstall_WithNoOptOut_InstallsTheBackend`][gate-delivery], [`BundleAutoinstall_HonorsTheOptOut`][gate-delivery] |
| 2 | Installing the backend sets a property on the engine; it does not replace or subclass the engine | same engine instance, `Blas` non-null after `Enable` | [`Enable_SetsTheBackendOnTheEngine_WithoutReplacingIt`][gate-backend], [`TheBackendIsAPlainSettableProperty`][gate-backend] |
| 3 | Without the package the engine has no BLAS backend and every product is managed | `engine.Blas == null` by default | [`NotInstalled_ByDefault_TheEngineHasNoBlasBackend`][gate-backend] |
| 4 | The bundled binary hashes to the committed manifest pin | per-RID sha256 equals `openblas-manifest.json` | [`BundledLibrary_HashesToThePinnedManifest`][gate-backend] |
| 5 | Auto-discovery prefers the bundle over machine tooling; the bundle can be opted out | bundled path wins; `NUMSHARP_BLAS_BUNDLED=0` drops it | [`AutoDiscovery_PrefersTheBundledLibrary`][gate-backend], [`BundledLibrary_CanBeOptedOutOf`][gate-backend] |
| 6 | A named library is binding — a missing one does not fall back to the bundle | throw, no substitution | [`NamedLibrary_OutranksTheBundledOne_AndAMissingOneDoesNotFallBackToIt`][gate-backend] |
| 7 | A failed `Enable` changes nothing, even over a working setup | working library and backend intact after a failed call | [`FailedEnable_OverAWorkingLibrary_ChangesNothing`][gate-backend], [`FailedEnable_OnANonCBlasFile_OverAWorkingLibrary_ChangesNothing`][gate-backend] |
| 8 | `Enabled` means installed **and** bound | `false` when a backend is installed with no library | [`Enabled_MeansInstalledAND_Bound_NotMerelyInstalled`][gate-backend] |
| 9 | The thread pin takes effect and is reported back | `Info` reflects `Enable(threads:)` | [`ThreadPin_IsReportedBack`][gate-backend] |
| 10 | A `coreType` this CPU cannot run is refused instead of crashing the process | `PlatformNotSupportedException`, process alive | [`CoreTypeAboveThisCpu_IsRefused_RatherThanCrashingTheProcess`][gate-backend] |
| 11 | Only float32/float64 products change; integers fall through; every other operation is untouched | identical results backend on vs off | [`IntegerProducts_FallThroughToTheManagedKernel`][gate-backend], [`EveryOtherOperation_StaysTheManagedImplementation`][gate-backend] |
| 12 | Every shape route round-trips through the backend | strided/transposed views accepted, not silently wrong | [`EveryShapeRoute_RoundTripsThroughTheBackend`][gate-backend], [`StridedAndTransposedViews_AreAccepted_NotSilentlyWrong`][gate-backend] |
| 13 | Stacked products take the batched entry once per stack; batched and per-element routes agree | call counting + 25/25 A/B | [`StackedProducts_GoThroughTheBatchedEntryPoint_Once`][gate-backend], [`BatchedEntryPoint_IsBitIdenticalToThePerElementRoute`][gate-backend] |
| 14 | A backend implementing only the two core members still works | decorator declining the batched path | [`ABackendThatKnowsNothingOfBatching_StillWorks`][gate-backend] |
| 15 | A version override is hard-required and sha-verified — never silently substituted | throws with the bundle sitting loadable beside it; a mismatched sha is refused | [`VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle`][gate-delivery], [`VersionMarker_WithMismatchedSha_RefusesTheFile`][gate-delivery] |
| 16 | A required-override miss at module load is loud on stderr, not fatal | one stderr line; backend left uninstalled | [`BundleAutoinstall_OnARequiredOverrideMiss_IsLoudButDoesNotThrow`][gate-delivery] |
| 17 | A path override is non-binding: it binds when it holds a library, falls through when it does not | both directions | [`PathMarker_BindsItsDirectory_WhenItHoldsALibrary`][gate-delivery], [`PathMarker_IsNonBinding_FallsThroughToTheBundle`][gate-delivery] |
| 18 | A pip-installed numpy's `numpy.libs/` is never scanned | the tier stays deleted | [`NumpyLibs_DiscoveryTier_IsDeleted`][gate-delivery] |
| 19 | The build pipeline — download, double verify, self-healing cache, staging, packing — works end-to-end against the real nupkg | 15 scripted steps incl. tamper, mirror-needs-sha, poisoned cache, conflicting references | [`verify_build_override.sh`][gate-build] |

---

## See also

- [Interoperability](index.md) — the memory-bridge contract behind the other `NumSharp.Interop.*`
  packages (this one shares a *binary*, not a buffer)
- [`docs/GEMM_PARITY.md`][gemm] — the engineering deep-dive: the ported routes, the design review,
  the delivery model, the probe results and traps
- [`docs/OPENBLAS_DELIVERY_DESIGN.md`][delivery] — the delivery/discovery model: the two phases,
  the source marker, the transitive story, every resolved decision
- [Benchmarks](../benchmarks-dashboard.md) — NumSharp performance across the whole API

[gate-backend]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Backends/MatmulParityBackendTests.cs
[gate-delivery]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/Backends/OpenBlasDeliveryTests.cs
[gate-build]: https://github.com/SciSharp/NumSharp/blob/master/src/NumSharp.Interop.OpenBLAS/tools/verify_build_override.sh
[gemm]: https://github.com/SciSharp/NumSharp/blob/master/docs/GEMM_PARITY.md
[delivery]: https://github.com/SciSharp/NumSharp/blob/master/docs/OPENBLAS_DELIVERY_DESIGN.md
