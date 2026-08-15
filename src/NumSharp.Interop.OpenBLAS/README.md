# NumSharp.Interop.OpenBLAS

Optional external-CBLAS matrix-product backend for [NumSharp](https://github.com/SciSharp/NumSharp).

**NumSharp.Core is 100 % managed C# with no native dependency.** Its own SIMD GEMM computes every
matrix product out of the box. This package is the *alternative*: reference it and `np.dot` /
`np.matmul` compute `float32` / `float64` products through OpenBLAS — both **faster on large
matrices** and **bit-identical to NumPy**.

```powershell
dotnet add package NumSharp.Interop.OpenBLAS
```

```csharp
using NumSharp;
using NumSharp.Interop.OpenBLAS;

// Referencing the package is the whole opt-in. A module initializer loads the bundled OpenBLAS
// for this machine's RID and installs the backend; if that fails, NOTHING changes and NumSharp
// keeps doing its own managed math.
var c = np.dot(a, b);

Console.WriteLine(OpenBlasEngine.Info);
// …\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 24]
// OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24

OpenBlasEngine.Disable();   // clears engine.Blas — back to NumSharp's managed SIMD GEMM
```

## The bundled binary

**The package ships the OpenBLAS binaries NumPy itself ships** — not "an" OpenBLAS. NumPy 2.x does
not build its own: its wheels bundle the prebuilt `scipy-openblas32` / `scipy-openblas64` artifacts
published on PyPI by the [openblas-libs](https://github.com/MacPython/openblas-libs) project, pinned
in numpy's `requirements/ci_requirements.txt`. This package bundles those same artifacts at the same
pinned version, **verified byte-for-byte**: the DLL here and the one in numpy 2.4.2's `numpy.libs/`
have the identical SHA-256 (`74a40872…`, which is also where NumPy's mangled file name comes from —
`delvewheel` names the file by its hash).

So bit-parity no longer depends on having Python installed, or on which copy of a BLAS won a loader
race. Bundled version: **scipy-openblas 0.3.31.22.0** (OpenBLAS 0.3.31.dev), matching **numpy 2.4.2**.

| RID | distribution | RID | distribution |
|---|---|---|---|
| `win-x64` | scipy-openblas64 | `linux-musl-x64` | scipy-openblas64 |
| `win-arm64` | scipy-openblas**32** † | `linux-musl-arm64` | scipy-openblas64 |
| `linux-x64` | scipy-openblas64 | `osx-x64` | scipy-openblas64 |
| `linux-arm64` | scipy-openblas64 | `osx-arm64` | scipy-openblas64 |

† not an inconsistency — NumPy's own build selects the LP64 distribution there, so matching it is
what parity requires. It exports `scipy_cblas_sgemm` with a 32-bit BLAS integer, and both schemes are
bound automatically. Both scipy distributions are first-class: **scipy-openblas64** (ILP64, the
default on the seven other RIDs) and **scipy-openblas32** (LP64). The 32-bit distribution is also the
*only* build PyPI publishes for 32-bit x86 (`win32` / `i686`), so any 32-bit target must use it.

NuGet resolves `runtimes/<rid>/native/` per machine, so a RID-specific publish carries only the one
asset. Licences travel with them in `THIRD-PARTY-NOTICES.txt` (OpenBLAS and its bundled LAPACK are
BSD-3-Clause-Attribution; the statically linked libgfortran carries the GCC Runtime Library
Exception, the same basis on which NumPy and SciPy ship it in every wheel).

### Discovery order

(The delivery/discovery model is specified in `docs/OPENBLAS_DELIVERY_DESIGN.md`.)

1. An explicit path, or `NUMSHARP_OPENBLAS_LIBRARY` — **binding**, never substituted.
2. Override path(s): `NUMSHARP_OPENBLAS_SEARCH_PATH` — one or more files/directories (path-separator
   delimited) — then a build-recorded `OpenBlasPath` (a `mode: "path"` entry in the
   `openblas.source.json` source marker). Non-binding: they take priority when they hold a loadable
   BLAS, and are silently skipped (fall through to the rest) when they do not.
3. A build-staged **version override** (an `openblas.source.json` marker with `mode: "version"`,
   written when the build pinned `OpenBlasVersion` / `NUMSHARP_OPENBLAS_VERSION`) —
   **hard-required**: a miss throws `OpenBlasRequiredOverrideException` rather than falling through
   (the pin is a contract, and quietly substituting the bundle or a system BLAS would break it
   invisibly). When the marker pins a sha256, only a file hashing to it is loaded.
4. An explicit OpenBLAS root — `OPENBLAS_HOME` / `OPENBLAS_ROOT` (OpenBLAS' own convention). A
   deliberate "use the OpenBLAS installed here" signal, so it **outranks the bundle**. **Not**
   byte-parity with NumPy, so setting it trades parity-by-default for the named build.
5. **The bundled asset** — the zero-config parity default.
6. A machine-wide OpenBLAS from an OS package manager or a source build — apt multiarch, a Homebrew
   keg, MacPorts, a conda tree, a vcpkg triplet, `/opt/OpenBLAS`, `/usr/lib64` … honouring the
   *ambient* `VCPKG_ROOT` / `CONDA_PREFIX`. **Not** byte-parity with NumPy (a different
   compiler and build), so it ranks below the parity sources: a correct, fast BLAS, not a bit-identical one.
7. Bare loader names (`libscipy_openblas64_`, `libscipy_openblas`, `scipy_openblas`, `libopenblas`,
   `libblas`, …).
8. Every directory on `PATH`, swept for a BLAS under a non-standard name — the broadest, **last-resort**
   scan, reached only if nothing above binds.

A pip-installed numpy's own `numpy.libs/` is deliberately **never scanned** (tier removed
2026-08-13): the bundled asset already *is* that binary at the pinned version, so results stay a
property of the package version rather than of whichever python happens to be installed — a
library's numeric output should not change because an unrelated `pip install` ran. (A conda/system
*OpenBLAS* is still machine tooling, tier 6; a *numpy* is not.) **To match a NumPy whose OpenBLAS
differs from the bundled one, name it** (binding, all-or-nothing) — **or point
`NUMSHARP_OPENBLAS_SEARCH_PATH` at it** to take priority over the bundled default while still falling back
to it if the path holds no BLAS:

```csharp
OpenBlasEngine.Enable(@"…\site-packages\numpy.libs\libscipy_openblas64_-<hash>.dll");
```

Set `NUMSHARP_OPENBLAS_USE_BUNDLED=0` to drop the bundled entry and make machine tooling the discovery
default.

## Overriding the OpenBLAS version (build-time)

The override is configured **on the `PackageReference`** — no second package:

```xml
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasVersion>0.3.31.22.0</OpenBlasVersion>              <!-- scipy-openblas version, enforced -->
  <!-- optional: read-from / download-to directory (relative resolves against the project) -->
  <!-- <OpenBlasPath>native/openblas</OpenBlasPath> -->
</PackageReference>
```

At build, the pinned scipy-openblas wheel is fetched from PyPI, verified twice (the wheel's sha256
from the index, then the extracted library's), the one native library is extracted into a global
per-user cache (`%LOCALAPPDATA%\NumSharp\openblas` / `~/.cache/NumSharp/openblas` — the wheel is
deleted; repeated builds and offline builds serve from the cache, and every cache hit re-verifies
the entry's content hash: a truncated or tampered entry is discarded and re-downloaded), staged
over the bundle in the output, and an `openblas.source.json` **source marker** is written so the
runtime treats the folder as a *required* override (discovery tier 3 above). **A version override is a hard requirement**: a
build that cannot download/verify it FAILS, and a runtime that cannot load it THROWS
(`OpenBlasRequiredOverrideException`) rather than quietly substituting the bundle or a system BLAS.
`OpenBlasPath` alone (no version) is the soft form — "read tooling from here", non-binding.

Environment variables win over the metadata: `NUMSHARP_OPENBLAS_VERSION`, `NUMSHARP_OPENBLAS_SEARCH_PATH`,
plus `NUMSHARP_OPENBLAS_{DISTRIBUTION,FEED,SHA256,DELIVERY}` (`OpenBlasDistribution` picks
`scipy-openblas64`/`scipy-openblas32` — `64`/`32` aliases; a non-pypi.org `OpenBlasFeed` mirror
requires an explicit `OpenBlasSha256`).

**Transitive:** the machinery ships in `buildTransitive/`, so a package that merely *depends* on
this one gives its consumers the identical experience — the targets read the OpenBLAS metadata off
**any** `PackageReference`, including one pointing at the intermediate package. A package author can
also bake a pinned version into *their own* nupkg with `<OpenBlasDelivery>package</OpenBlasDelivery>`
(stages **all** supported RIDs plus their markers as `runtimes/<rid>/native/` pack content; NuGet's
nearest-wins rule makes that copy beat this package's transitive bundle). `OpenBlasDelivery=none`
opts a reference out; `build` (the default) stages into the consuming app's output only.

Full model: `docs/OPENBLAS_DELIVERY_DESIGN.md`. Integration gate:
`tools/verify_build_override.sh` (scripted end-to-end nupkg-flow run; needs network once).

## What it does

`OpenBlasBackend` implements NumSharp's `IBlasBackend` — `Try`-shaped members only — and the package
assigns it to the engine's `TensorEngine.Blas` property. The engine itself is NOT replaced or
subclassed: every other operation, dtype, kernel and iterator is untouched, and anything the backend
does not serve (any dtype other than float32/float64, or a shape outside the ported surface) is
computed by the built-in managed kernels as before.

```csharp
// The entire Core-side surface:
public interface IBlasBackend {
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;  // optional

    // IBlasBackend.LinearAlgebra.cs — 15 more, every one optional (defaults to false):
    //   products       TryInner TryVdot TryVecdot TryMatvec TryVecmat
    //   factorisations TryCholesky TryDet TrySlogdet TryEig TryEigh TryInv
    //                  TryLstsq TryQr TrySolve TrySvd
}
public abstract partial class TensorEngine { public IBlasBackend Blas { get; set; } }
```

`TryMatMulBatched` takes a whole stacked product at once. It is a **default interface method**, so a
backend that only implements the first two members compiles and runs unchanged — the engine falls
back to calling `TryMatMul2D` once per batch element.

**`OpenBlasBackend` implements none of the 15 linear-algebra members**, and it compiles and behaves
exactly as before because of that same default-implementation rule. They exist so the seam for the
rest of what NumPy routes through this binary is fixed: `np.inner`/`np.vdot`/`np.vecdot`/
`np.matvec`/`np.vecmat`, and the LAPACK routines — `gesv`, `getrf`, `potrf`, `geev`, `syevd`/`heevd`,
`gesdd`, `geqrf`/`orgqr`, `gelsd` — which scipy-openblas ships **inside the very library this package
already bundles**, so a backend that has loaded it for `sgemm` already holds the handle.

The two groups differ in what happens when a backend declines. The **products** fall back to
NumSharp's managed kernels, so the invariant above still holds. The **factorisations** have no
managed fallback — Core carries no LU, QR, SVD or eigensolver — so `np.linalg.inv` and friends raise
`NotSupportedException` naming the routine and the `Try*` member to implement. Their NumPy-exact
validation runs first and is already gated, so an implementation only has to supply numerics. Implementing it lets a backend amortise
per-product setup across the stack (the trailing strides never vary within one), which is what NumPy
does in its matmul gufunc's outer loop; measured on 2000 stacked 8×8 float32 products, doing that
setup per element instead made this backend **0.80×** — *slower* than NumSharp's own managed GEMM —
against **6.5×** with the work hoisted.

Writing your own backend needs no access to NumSharp's internals: read operands as
`(T*)a.GetData().Address + a.Shape.Offset` with `a.Shape.Strides` (element units). Do **not** use
`a.GetData<T>().Address` — it densifies a non-contiguous view, so pairing it with `Shape.Strides`
reads outside the buffer.

The two entry points are a **route-for-route port of NumPy 2.4.2's two matrix-product dispatchers** —
`cblas_matrixproduct` (behind `np.dot`) and the `@TYPE@_matmul` gufunc (behind `np.matmul` and
`@`). They are genuinely different C code and disagree on some inputs, so both are mirrored:
gemm (with the `a @ a.T` **syrk** shortcut and NumPy's copy-if-not-blasable rule), both gemv
directions, the row·column `?dot` with NumPy's double accumulator, the portable no-blas loop,
stacked/batched products, and the N-D `np.dot` route that is not gemm at all.

## Why a port, and not just "call gemm"

Both NumSharp's answer and NumPy's are correct floating-point matrix products; they sum in
different orders. NumSharp's managed GEMM differed from NumPy on **94.5 %** of the elements of a
`(128,784)@(784,128)` float32 product (max ~976 000 ULP) — enough that training the same network in
both stacks diverges. NumPy's bits *are* its BLAS's bits, and OpenBLAS' multi-accumulator
micro-kernels match neither a sequential `mul + add` chain nor a sequential FMA chain. Reproducing
them means calling the same binary through the same route with the same flags.

## Three knobs decide the bits — the binary is only one of them

Bundling fixes the library build. Two variables remain, and a parity claim that ignores either is
wrong:

- **The worker-thread count is part of the answer.** 1, 2, 4 and 24 threads give four *different*
  results. `OpenBlasEngine.Enable(threads: n)` pins it; it is a correctness knob, not a performance one.
- **The CPU is part of the answer.** OpenBLAS is built `DYNAMIC_ARCH` and picks a micro-kernel from
  the machine it finds itself on, and different kernels accumulate in a different order. Measured by
  forcing the choice on one host: Haswell, Nehalem, Sandybridge and Katmai each produced different
  bytes for the same float32 product. **So "same binary" is necessary and not sufficient.**

```csharp
// Reproducible on ANY x86-64 with AVX2+FMA — pins the last free variable.
OpenBlasEngine.Enable(threads: 1, coreType: OpenBlasEngine.CoreType);   // "Haswell"
```

`coreType` is **not** the default, deliberately: the package's main promise is matching the NumPy on
*this* machine, and that NumPy dispatches by CPU too — so pinning here would *create* a divergence on
any host newer than Haswell. Pin both sides or neither (the same `OPENBLAS_CORETYPE` variable drives
NumPy, because it is the same library).

> **Hazard.** `coreType` *overrides* OpenBLAS' CPU detection rather than expressing a preference, so
> naming kernels the CPU cannot run terminates the process with an illegal instruction — measured.
> `OpenBlasEngine.Enable` refuses the combinations it recognises with a `PlatformNotSupportedException` first.
> It also only takes effect on the call that actually **loads** the library: OpenBLAS reads the
> variable while initialising and never re-reads it, so a later request that cannot be honoured
> throws rather than being silently dropped.

And unchanged from before:

- **A named library is binding** — an explicit path (or `NUMSHARP_OPENBLAS_LIBRARY`) is used as given and
  never silently replaced, not even by the bundled copy sitting in the output folder. A failed
  `OpenBlasEngine.Enable` is a **no-op**: it throws having changed nothing, so a mistyped path cannot demote a
  working setup back to the managed kernels behind your back.

## Supported providers

The bundled OpenBLAS is not privileged — any CBLAS export set is bound, in this order: numpy's
renamed ILP64 wheels (`scipy_cblas_sgemm64_`), plain ILP64 (`cblas_sgemm64_` / `cblas_sgemm_64`),
scipy's LP64 build (`scipy_cblas_sgemm`), and stock LP64 (`cblas_sgemm`) — so MKL, Accelerate, a
system `libblas` or a reference CBLAS all work. Integer widths are marshalled per call. OpenBLAS'
thread, core-name and build-info entry points are used when present; a provider without them works
fine.

## Environment

| Variable | Effect |
|---|---|
| `NUMSHARP_OPENBLAS_LIBRARY` | Path (or directory) of the CBLAS library to load. **Binding**, like an explicit argument — used as given, never substituted. |
| `NUMSHARP_OPENBLAS_SEARCH_PATH` | File(s)/dir(s) to scan for a CBLAS, path-separator delimited. Tried **first** (priority over the bundled asset), non-binding — falls through to the rest if it holds no BLAS. |
| `NUMSHARP_OPENBLAS_USE_BUNDLED=0` | Skip the bundled asset; machine tooling becomes the discovery default. |
| `NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` | Skip the module-load auto-install; `OpenBlasEngine.Enable(...)` still works. (The pre-rename `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL` and `NUMSHARP_BLAS_AUTOINSTALL` spellings are retired and ignored.) |
| `NUMSHARP_OPENBLAS_VERSION` | **Build-time**: scipy-openblas version to download from PyPI and stage over the bundle — a hard requirement; beats `<OpenBlasVersion>` metadata. |
| `NUMSHARP_OPENBLAS_DISTRIBUTION` / `NUMSHARP_OPENBLAS_PYPI_FEED_URL` / `NUMSHARP_OPENBLAS_SHA256` / `NUMSHARP_OPENBLAS_DELIVERY` | **Build-time**: distribution pick (`64`/`32`), PyPI mirror base (needs the sha), expected extracted-lib sha256, delivery mode (`none`/`build`/`package`); each beats its `OpenBlas*` metadata twin. |
| `OPENBLAS_HOME` / `OPENBLAS_ROOT` | Explicit OpenBLAS install root (OpenBLAS' own convention). A deliberate signal, so it ranks **above the bundle** (discovery tier 4) — not byte-parity with NumPy. |
| `VCPKG_ROOT` / `CONDA_PREFIX` | Ambient roots consulted when scanning for a machine-wide OpenBLAS (discovery tier 6, below the bundle). |
| `OPENBLAS_CORETYPE` | Read by OpenBLAS itself at load. Set it in the environment to pin both NumPy and NumSharp at once. |

## Maintainers

The binaries are not in git. Stage them before packing:

```bash
python src/NumSharp.Interop.OpenBLAS/tools/fetch_openblas.py          # download + verify + extract
python src/NumSharp.Interop.OpenBLAS/tools/fetch_openblas.py --check  # verify only
```

Every artifact is checked **twice** against the committed `tools/openblas-manifest.json` — the
wheel's SHA-256, then the extracted library's own SHA-256 — so a re-pointed URL or a mutated artifact
fails loudly instead of quietly changing the bits the package produces. Building without the assets
staged is fine and simply produces an asset-free package; `-p:RequireOpenBlasAssets=true` turns that
into a hard error for release builds.

Full design record, probe results and traps: [`docs/GEMM_PARITY.md`](https://github.com/SciSharp/NumSharp/blob/master/docs/GEMM_PARITY.md).
