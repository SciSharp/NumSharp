---
paths:
  - "src/NumSharp.Interop.OpenBLAS/**"
  - "src/NumSharp.Core/Backends/{IBlasBackend,ISlidingDotBackend,TensorEngine,TensorEngine.LinearAlgebra}.cs"
  - "test/NumSharp.Tests.Interop/**"
  - "examples/NumSharpOpenBlas*/**"
---

# The BLAS backend seam and `NumSharp.Interop.OpenBLAS`

## The seam

`NumSharp.Core` has no native dependency and no P/Invoke: every kernel is managed code, and that is what a user gets
by default. A native backend arrives only as a separate NuGet package, and Core knows it through one interface and one
settable property:

```csharp
public interface IBlasBackend                                    // Backends/IBlasBackend.cs
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;
    // + 15 default members returning false:
    //   products        TryInner TryVdot TryVecdot TryMatvec TryVecmat
    //   factorisations  TryCholesky TryDet TrySlogdet TryEig TryEigh TryInv
    //                   TryLstsq TryQr TrySolve TrySvd
}

public abstract partial class TensorEngine
{
    public IBlasBackend Blas { get; set; }   // null (the default) = NumSharp's own managed kernels
}
```

- Every member is Try-shaped: a backend answers for the operands it implements and returns false for the rest, so a
  backend changes which implementation computes a product, never whether one can be computed.
- The package installs itself from a `[ModuleInitializer]`; assigning null takes it out again. The initializer runs
  when the runtime first prepares a method that names a type of the package, and Core never loads the package
  assembly. A program that references the package and calls only `np.*` therefore runs the managed kernels: it must
  name the package first (`OpenBlasEngine.Enable()`, `OpenBlasEngine.Info`). The initializer runs at the entry of that
  method, before its first statement, in every JIT mode, so an in-process opt-out must be set by code that names no
  package type and inlines no method that does.
- The engine is not subclassed or replaced: an `NDArray` binds its engine at construction and `np.dot(a, b)` dispatches
  on `a.TensorEngine`, so only a property on the cached singleton reaches arrays that already exist (every
  construction, operation and IO path reaches the one `EngineCache<DefaultEngine>` instance).
- Read `Blas` into a local before calling it: test-then-call on a settable property races a concurrent `Disable()`.
- A failed `OpenBlasEngine.Enable` is a no-op: the library already loaded stays loaded.
- A backend needs no `InternalsVisibleTo`: read operands as `(T*)a.GetData().Address + a.Shape.Offset` with
  `a.Shape.Strides` (all public). `a.GetData<T>().Address` densifies a non-contiguous view, so never pair it with strides.
- `TryMatMulBatched` takes a whole stacked product, so per-product setup is hoisted out of the loop as NumPy hoists it
  out of its gufunc's outer loop; bit-identical to the per-element route.
- The factorisations reach `Blas` through numpy.linalg's sub-engine, a second settable property on the same engine:
  `TensorEngine.Linalg` (`Backends/LinalgEngine.cs`: numpy.linalg's computing functions as the abstract `LinalgEngine`;
  the default `DefaultLinalgEngine` is NumPy's `_linalg.py` statements over ten `protected virtual` kernels,
  `CholeskyKernel` … `SvdKernel`, which ask the operand engine's `Blas`, then the managed LU or
  `OpenBlasMissingBackendException`; null restores the default). A package that supplies LAPACK never touches it; it is
  for a provider that computes a factorisation another way (override a kernel of a `DefaultLinalgEngine` subclass and
  assign it) or replaces numpy.linalg's statements.
- numpy.fft has the third settable property, `TensorEngine.Fft` (`Backends/FftEngine.cs`: numpy.fft's computing functions
  as the abstract `FftEngine`; the default `DefaultFftEngine` is NumPy's `_pocketfft.py` / `_helper.py` statements over
  three `protected virtual` kernels, `C2CKernel` / `R2CKernel` / `C2RKernel`, that run the managed pocketfft port; null
  restores the default). It needs no `Blas`: an MKL / FFTW / cuFFT provider overrides the kernels of a
  `DefaultFftEngine` subclass and assigns it.

## Why the package exists

- No portable algorithm matches NumPy's float matrix products. For float32/float64 mat@mat NumPy always calls cblas (it
  copies a non-blasable operand into a temp rather than take its portable loop), and scipy-openblas' `sgemm` uses an
  arch-specific multi-accumulator scheme that matches neither a sequential multiply-add chain nor a sequential FMA
  chain. The managed GEMM therefore differs from NumPy on most elements of a large float32 product, and the
  difference compounds through iterative algorithms.
- The backend is a route-for-route port of both NumPy dispatchers, which are different C code and disagree when an
  operand is not blasable (a stride-2 matrix @ vector: `np.dot` takes gemv on a copy, `np.matmul` the portable loop).
  `Dot` is `cblas_matrixproduct` plus the N-D `dotfunc` tail (which is not gemm); `MultiplyMatrix` is
  `@TYPE@_matmul`'s five routes, including the `a @ a.T` syrk shortcut and NumPy's copy/transpose rules. That is why
  `IBlasBackend` has both `TryDot` and `TryMatMul2D`. Strides are ported in elements (NumPy's are bytes: the same logic
  with itemsize 1).
- Scope is float32/float64/complex128, the dtypes NumPy routes through cblas (`matmul.c.src`'s USEBLAS list). complex128
  uses `zgemm`/`zgemv`/`zsyrk`/`zdotu` (the unconjugated dot behind dot/matmul/inner/matvec) and `zdotc` (the
  conjugating dot behind vdot/vecdot/vecmat), with alpha/beta passed by pointer (`{1,0}`/`{0,0}`) and the dots through
  the `_sub` out-pointer form, exactly as NumPy calls them. It is served only when the loaded library exports the
  complex products: a real-only CBLAS declines back to the managed complex kernel.
- Every other dtype uses the managed kernels: integer and bool products are exact by construction (modular addition is
  associative, so summation order cannot change the result).
- The BLAS path is also faster than the managed GEMM on large products.

## The bundled binary

- The package ships the binary NumPy pins: `scipy-openblas64==0.3.31.22.0` (numpy's `requirements/ci_requirements.txt`),
  byte-identical to numpy 2.4.2's copy (both sha256 `74a40872…`; NumPy's mangled filename is that hash), as
  `runtimes/<rid>/native/` assets for 8 RIDs. win-arm64 is scipy-openblas32, because NumPy's own build selects LP64
  there; it exports `scipy_cblas_sgemm`, which `Bind` handles.
- The binaries are gitignored. `src/NumSharp.Interop.OpenBLAS/tools/openblas-manifest.json` is the checked-in pin, and
  `src/NumSharp.Interop.OpenBLAS/tools/fetch_openblas.py` verifies two hashes per RID.
- Library discovery, in order (gate `OpenBlasDeliveryTests`):
  1. an explicit path or `NUMSHARP_OPENBLAS_LIBRARY`: binding, never substituted, not even by the bundled copy;
  2. override paths: `NUMSHARP_OPENBLAS_SEARCH_PATH`, then a build-recorded `OpenBlasPath` marker (non-binding);
  3. a build-staged version override (`openblas.source.json` marker, `mode:"version"`): hard-required, a miss throws
     `OpenBlasRequiredOverrideException` and never falls through; sha-verified when pinned;
  4. an explicit OpenBLAS root (`OPENBLAS_HOME`/`OPENBLAS_ROOT`), above the bundle: a deliberate opt-in that trades
     parity for the named build;
  5. the bundled library, the parity default, above ambient machine tooling;
  6. machine-wide installs (apt, brew, MacPorts, conda, vcpkg and source install dirs; honours the ambient
     `VCPKG_ROOT`/`CONDA_PREFIX`; not parity);
  7. bare names in 64-bit and 32-bit spellings (ILP64 scipy-openblas64 and LP64 scipy-openblas32 both bind; 32 is the
     only build for 32-bit x86);
  8. a PATH sweep for non-standard names (last resort).

  `NUMSHARP_OPENBLAS_USE_BUNDLED=0` drops the bundled entry. `numpy.libs` is never scanned: the bundle already is that
  binary at the pin.
- The Linux and macOS wheels do not statically link the GCC runtime: `libgfortran`, `libquadmath` and `libgcc_s` ride
  beside the main library under content-hashed names that its own `DT_NEEDED`/`LC_LOAD_DYLIB` entries name, so a system
  libgfortran cannot stand in. `fetch_openblas.py` co-stages them and the manifest pins each (`vendored` per RID; a
  partial staging fails `--check` and is staged again). On Linux they sit beside the main in `native/`, found by its
  `$ORIGIN` RUNPATH.
- macOS: `delocate` parks the deps in a `.dylibs/` sibling of `lib/` and rewrites the load commands to
  `@loader_path/../.dylibs/<name>`, but NuGet delivers native assets only from under `runtimes/<rid>/native/`, so a
  sibling folder is dropped from every consumer (a ProjectReference build copies it through a Content item, so only a
  PackageReference consumer shows the difference). The deps are therefore staged and packed nested
  (`runtimes/osx-*/native/.dylibs/`), and `MacOsVendoredRuntime` makes the dylib loadable after a plain `dlopen` fails:
  - it parses the Mach-O load commands (`MachOImage`, header only) and plans the missing `@loader_path/` deps and their
    closure, sourced from `<dir>/.dylibs/<name>`, then `<dir>/<name>` (a RID-specific publish flattens the sub-folder);
  - it materializes them in place for the `runtimes/<rid>/native` layout (a relative directory symlink
    `runtimes/<rid>/.dylibs → native/.dylibs`, else per-file copies; never outside `runtimes/`);
  - when that folder is read-only or the layout is flattened (`../.dylibs` would be the parent of the app), it uses a
    per-user cache copy of main + deps in the same relative layout (`NUMSHARP_OPENBLAS_CACHE_DIR`, else
    `~/.cache/NumSharp/openblas/dyld/<sha256>/native/` + `.dylibs/`; keyed by content hash, every file byte-compared to
    its source on every hit, written temp+rename).

  `OpenBlasEngine.LibraryPath` names the file discovery chose (the parity pin hashes that file; the copy is verified
  identical); `OpenBlasEngine.LoadedImagePath` is what dyld mapped, and `Info` says `(mapped from …)`.
  Two approaches do not work: pre-loading the deps by absolute path (dyld matches by path/install name, and delocate's
  deps carry the placeholder `/DLC/scipy_openblas64/.dylibs/…`) and relinking with `install_name_tool` (it needs macOS
  tooling plus re-signing on a Linux staging runner, and it changes the pinned bytes).
- Gates: `OpenBlasMacOsVendoredRuntimeTests` (synthetic Mach-O images and the real staged dylibs, every OS) and the
  "Package consumer:" steps of the `interop-test` CI job, which run `tools/verify_package_consumer.sh` and
  `tools/verify_build_override.sh` (under `src/NumSharp.Interop.OpenBLAS/`) on all three OSes: the real nupkg restored
  into a scratch consumer by PackageReference and loaded in the portable, read-only and flattened layouts. They run last
  in that job, because both scripts `-t:Rebuild` Core/OpenBLAS in Release in the repo tree and evict this version from
  `~/.nuget/packages`. That job's `env:` must never set `NUMSHARP_OPENBLAS_CACHE_DIR`: `verify_build_override.sh`
  asserts the default per-user cache root on Linux/macOS.

## The build-time override

- Implemented in `buildTransitive/{props,targets}` plus an inline `RoslynCodeTaskFactory` task (netstandard2.0 surface
  only, its own `MiniJson`, because the factory cannot resolve simple-name References).
- `<OpenBlasVersion>` on any `PackageReference` (or `NUMSHARP_OPENBLAS_VERSION`; the environment wins) downloads that
  scipy-openblas from PyPI at build time, verifies both hashes, caches the extracted library per user
  (`%LOCALAPPDATA%/NumSharp/openblas`; the wheel is deleted; offline once primed), stages it over the bundle and writes
  the marker. A failure is a build error. The task co-extracts the vendored deps into its cache entry (an `.entry.json`
  sidecar; every file hash-verified on every hit; an entry without a sidecar is discarded) and stages them with the
  main library, so an override loads on Linux and macOS too.
- `<OpenBlasPath>` is a soft read-in-place directory (also the download directory when combined with a version).
  `<OpenBlasDelivery>package</OpenBlasDelivery>` bakes all 8 RIDs and their markers into a dependent's own nupkg.
- The cache heals itself: every hit re-verifies the entry's content hash (the directory name is the claim), a truncated
  or tampered entry is discarded and downloaded again, and writes go temp+rename so killed or racing builds cannot
  strand garbage under the hash name.
- MSBuild traps: a target's `Condition` evaluates before its `DependsOnTargets` run, so the has-override gate sits on the
  task, not the target; `dotnet pack` defaults to Release on .NET 8+, and this repo's incremental build can declare
  stale Release outputs up to date, so run `-t:Rebuild` before packing.
- glibc Linux: the fetch task matches the wheel by `"-" + <anchor of the manifest wheel's platform tag> + "."`
  (`manylinux2014_x86_64`). scipy-openblas wheels from 0.3.33.0.0 are tagged
  `manylinux_2_27_x86_64.manylinux_2_28_x86_64`, so pinning one fails on linux-x64 and linux-arm64 with
  `expected exactly one scipy-openblas64 … wheel for platform tag 'manylinux2014_x86_64', found 0`; 0.3.29.0.0 fails the
  same way (its compound tag lists `manylinux_2_17` first). The OpenBLAS of NumPy 2.5 therefore cannot be pinned there.
  Versions that stage on all 8 RIDs: 0.3.29.265.1 to 0.3.31.188.0.
- Gates: `tools/verify_build_override.sh` (a scripted nupkg flow including tamper, mirror-needs-sha, invalid-knob,
  delivery=none, two-reference-conflict and poisoned-cache steps; network once, then the cache) and
  `OpenBlasDeliveryTests` (priority pins and hostile markers).

## What decides the result bits

- The BLAS thread count: 1, 2, 4 and 24 threads give four different answers.
- The DYNAMIC_ARCH kernel the CPU dispatches: bundling pins the build, not the answer. `OpenBlasEngine.Enable(coreType:
  OpenBlasEngine.CoreType)` pins the kernel (Haswell, Nehalem, Sandybridge and Katmai all differ). It is opt-in because
  the default must match the local NumPy, which also dispatches by CPU. It applies when the call is the first load, is a
  no-op when the loaded kernels are the same, throws `InvalidOperationException` when the library is already loaded with
  others and `PlatformNotSupportedException` when the CPU cannot run them (forcing an ISA above the CPU kills the process
  with SIGILL, so `Enable` refuses what it can recognise). An unknown name raises nothing, because OpenBLAS ignores it:
  read `CoreName`. `OPENBLAS_CORETYPE` / `OPENBLAS_NUM_THREADS` in the process environment work with no code.
- `Environment.SetEnvironmentVariable` does not reach a native `getenv`: .NET keeps its own table, so `OPENBLAS_CORETYPE`
  set that way reads back correctly and changes nothing. Go through the CRT (`ucrtbase!_putenv_s` / `libc!setenv`);
  Windows' `SetEnvironmentVariableW` does not work either.

## API

`OpenBlasEngine.Enable(library, threads, coreType)` / `Disable()` / `Enabled` / `Info` / `LibraryPath` /
`LoadedImagePath` / `CoreName` / `CoreType` / `IsBundledLibrary`. `IsBundledLibrary` is decided by content (the loaded
file must hash to a manifest pin embedded in the assembly, `BundledAssetPin`, so a RID-specific or single-file publish
that flattens the bundle into the app root still reports true) and is marker-aware (a build-staged override folder
reports false even with identical bytes); both are evaluated live.

Environment variables:
- `NUMSHARP_OPENBLAS_CACHE_DIR` moves the per-user cache (macOS relocated dylibs and the build-override cache).
- `NUMSHARP_OPENBLAS_SEARCH_PATH` adds the highest-priority, non-binding discovery locations, tried before the bundle.
- `NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` opts out of the module-load install (`OpenBlasEngine.BundleAutoinstall`). The
  spellings `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL` and `NUMSHARP_BLAS_AUTOINSTALL` are ignored (pinned by
  `RetiredAutoinstallAlias_NoLongerSuppresses`).
- A required-override miss at module load is reported to stderr and the backend stays uninstalled, never substituted.

## Gates

- The host-pinned `matmul_parity` corpus tier: float32/float64/complex128 `dot`/`matmul` plus `inner`, `vdot`, `vecdot`,
  `matvec` and `vecmat`, bit-exact with the backend. It goes `Inconclusive`, never red, on a host that cannot load the
  pinned library. The pin matches the library by content hash, not filename: delvewheel mangles the name on one side,
  and a name match would accept a differently built library of the same name.
- The host-pinned `linalg_parity` tier (the LAPACK factorisations, see `linalg-einsum.md`) and the managed-versus-backend
  delta (`oracle.md`).
- Packaging contract: `MatmulParityBackendTests`. Sliding dot products (`np.correlate`/`np.convolve`): `float-math.md`.

## Example projects

`examples/NumSharpOpenBlasDefaultExample` and `examples/NumSharpOpenBlasAdvancedExample` are console apps over the
published `NumSharp.Interop.OpenBLAS` 0.70.0, in the regular csproj form (one plain `PackageReference`). No workflow runs
them.
- Default: the one package reference and no configuration. `OpenBlasEngine.Enable()` first, then the products, the
  LAPACK tour (it returns the largest residual of the identities it shows) and a matmul timing with `Disable()` /
  `Enable()` as the switch.
- Advanced: `OpenBlasVersion="0.3.31.188.0"` as an attribute of the `PackageReference` (the OpenBLAS of NumPy 2.4.4; the
  child-element form is equivalent), a `Program.cs` that names no package type and sets
  `NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` in process, `Enable(threads: 1, coreType: OpenBlasEngine.CoreType)` in a
  `NoInlining` method, and the SHA-256 of one float32 product for each thread count.
- A package reference alone does not install the backend (the initializer rule above): `dot`/`matmul` run on the
  managed kernels, and `cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `lstsq`, `pinv`,
  `matrix_rank`, `cond` and `norm` with `ord` 2 / `'nuc'` raise `OpenBlasMissingBackendException`, whose text says to
  reference the package that is already referenced (`solve`/`inv`/`det`/`slogdet` take the managed LU). The docs', the
  package README's and that exception's "referencing the package is the whole opt-in" holds only for code that names
  the package before it computes; the consumer of `verify_package_consumer.sh` names the type too, so no gate sees it.
- Parity: the bundled DLL is the file inside numpy 2.4.2's wheel (sha256 `74a40872…0377`) and the 0.3.31.188.0 DLL the
  one inside numpy 2.4.4's (`63c857e7…1080`). The float32 `(512,784) @ (784,512)` product of `np.random.seed(7)` inputs
  has NumPy's hash at 1, 2 and 24 threads for both pairs, and the 1- and 2-thread hashes are equal on Windows and Linux.
- NumPy's OpenBLAS pins (`requirements/ci_requirements.txt` at each tag): 2.2.6 / 2.3.0 → 0.3.29.0.0; 2.3.2–2.3.4 →
  0.3.30.0.1; 2.3.5 / 2.4.0 / 2.4.1 → 0.3.30.0.7; 2.4.2 → 0.3.31.22.0; 2.4.3 → 0.3.31.126.1; 2.4.4 → 0.3.31.188.0;
  2.5.0 / 2.5.1 → 0.3.33.112.0.
- Output size: a portable build output carries the libraries of all 8 RIDs (201 MB);
  `dotnet publish -r <rid> --self-contained false` gives 23 MB (Windows) / 29 MB (Linux), with the override and its
  marker flattened beside the app.
- The released NumSharp package ships the NDW012 analyzer, so example code without an `NDScope` warns.
