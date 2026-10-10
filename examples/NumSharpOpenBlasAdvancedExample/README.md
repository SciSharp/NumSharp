# NumSharpOpenBlasAdvancedExample

A console app that pins everything that decides the last bits of a matrix product: the OpenBLAS
build, the number of threads, and the kernels that OpenBLAS selects for the processor. The package
is [`NumSharp.Interop.OpenBLAS`](../../src/NumSharp.Interop.OpenBLAS/README.md).

Use this example when a result must be the same on each run, on each machine, or in NumSharp and in
a NumPy on the Python side. For the setup with no configuration, start with its twin,
[`NumSharpOpenBlasDefaultExample`](../NumSharpOpenBlasDefaultExample/README.md).

## Run it

You need the .NET 10 SDK. From this folder:

```bash
dotnet run -c Release
```

The first build downloads the pinned OpenBLAS library from PyPI (7 to 10 MB) and keeps it in a
cache for your user. Later builds, and builds without a network, use the cache.

```text
OpenBLAS    C:\src\NumSharpOpenBlasAdvancedExample\bin\Release\net10.0\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24
bundled     False
pin         scipy-openblas64 0.3.31.188.0, required: True
kernels     Haswell (pinned)
product     512 x 784 @ 784 x 512 float32, the SHA-256 of the result
  1 thread    d02ca290eb70d865
  1 thread    d02ca290eb70d865   the same bits again
  2 threads   67cdb5ec364fa7df
  24 threads  d16b7257b4864151
  NumSharp    b5560c0b1370343c   its own kernels: correct, and other bits again
```

Each hash line is the same product. All five results are correct. They differ in the last bits,
because float addition depends on the order of the terms, and each setting adds in another order.

## The three pins

| What decides the bits | Where it is pinned | How |
|---|---|---|
| The OpenBLAS build | [`NumSharpOpenBlasAdvancedExample.csproj`](NumSharpOpenBlasAdvancedExample.csproj) | `OpenBlasVersion="0.3.31.188.0"` on the package reference |
| The number of threads | [`Backend.cs`](Backend.cs) | `OpenBlasEngine.Enable(threads: 1, ...)` |
| The kernels | [`Backend.cs`](Backend.cs) | `coreType: OpenBlasEngine.CoreType`, which is `"Haswell"` |

[`Program.cs`](Program.cs) turns the automatic load of the package off before it calls
`Backend.Start()`. [`Reproducibility.cs`](Reproducibility.cs) computes the product and prints the
hashes.

## What was measured

On one machine (Core i9-13900K), under Windows 11 and under Ubuntu 24.04, with package 0.70.0:

- **The pinned library is the library of NumPy 2.4.4.** On Windows the file in the output and the
  file inside the NumPy 2.4.4 wheel have the same SHA-256 (`63c857e7...1080`).
- **NumPy 2.4.4 gives the same bits.** The same product in NumPy 2.4.4 has the hash of the program
  with 1 thread, with 2 threads and with 24 threads. The inputs are equal because NumSharp's random
  generator gives NumPy's numbers for the same seed.
- **Windows and Linux agree.** The 1-thread hash and the 2-thread hash are the same on both systems.
- **The hashes repeat.** Each line is the same on each run of the program.

To make the comparison yourself:

```python
import hashlib, numpy as np          # run with OPENBLAS_NUM_THREADS=1 in the environment
np.random.seed(7)
a = np.random.rand(512, 784).astype(np.float32)
b = np.random.rand(784, 512).astype(np.float32)
print(hashlib.sha256((a @ b).tobytes()).hexdigest()[:16])
```

## Which version to pin

The version is a release of [`scipy-openblas64`](https://pypi.org/project/scipy-openblas64/) on PyPI,
the package that NumPy itself takes its OpenBLAS from. To match a NumPy, pin the version that this
NumPy pins (`requirements/ci_requirements.txt` in the NumPy repository):

| NumPy | scipy-openblas | Platforms that package 0.70.0 can build it for |
|---|---|---|
| 2.2.6, 2.3.0 | 0.3.29.0.0 | all except glibc Linux and Windows arm64 |
| 2.3.2 to 2.3.4 | 0.3.30.0.1 | all eight |
| 2.3.5, 2.4.0, 2.4.1 | 0.3.30.0.7 | all eight |
| 2.4.2 | 0.3.31.22.0 | all eight: it is the library that the package bundles |
| 2.4.3 | 0.3.31.126.1 | all eight |
| 2.4.4 | 0.3.31.188.0 | all eight: this example |
| 2.5.0, 2.5.1 | 0.3.33.112.0 | all except glibc Linux |

The last column is from the names of the files on PyPI. Built and run: the version of this example
on Windows and on Linux, and 0.3.30.0.7 and 0.3.33.112.0 on Windows.

**A limit of package 0.70.0 on glibc Linux.** The package finds the download by the platform tag in
the name of the file, and the glibc files of `scipy-openblas64` got another tag with 0.3.33
(`manylinux_2_27` in place of `manylinux2014`). On Ubuntu, a build with `0.3.33.112.0` or
`0.3.34.0.0` fails with this message:

```text
NumSharp.Interop.OpenBLAS: expected exactly one scipy-openblas64 0.3.33.112.0 wheel for platform tag 'manylinux2014_x86_64', found 0.
```

Versions 0.3.29.265.1 to 0.3.31.188.0 have a file for each of the eight platforms of the package.
The later versions build on Windows (0.3.34.237.0, the newest one, was built and run there), and
their macOS and musl Linux files have the tags that the package expects.

## What the build does

1. It reads `OpenBlasVersion` from the package reference. The environment variable
   `NUMSHARP_OPENBLAS_VERSION` wins over it.
2. It asks PyPI for that release, selects the file for the platform that it builds for, downloads
   it, and compares its SHA-256 with the one that PyPI states.
3. It takes the library out of the file, and on Linux and macOS the Fortran runtime files beside
   it, and stores them in the cache: `%LOCALAPPDATA%\NumSharp\openblas` on Windows,
   `~/.cache/NumSharp/openblas` on Linux and macOS.
4. It copies the library into the output, in place of the bundled one, and writes
   `openblas.source.json` beside it. That file holds the version and the SHA-256 of the library.

When a step fails, the build fails. When the program starts, the package reads the file of step 4
and loads that library only. If the library is missing or has another hash, `Enable()` throws
`OpenBlasRequiredOverrideException`. The program never continues with another library, because a
silent change of the library is the failure that a pin exists to prevent.

## All settings of the package reference

The example sets one. Each setting is an attribute of the `PackageReference` (a child element with
the same name works too), and each has an environment variable that wins over it.

| Attribute | Environment variable | Meaning |
|---|---|---|
| `OpenBlasVersion` | `NUMSHARP_OPENBLAS_VERSION` | The `scipy-openblas` release to download and to require. |
| `OpenBlasSha256` | `NUMSHARP_OPENBLAS_SHA256` | The SHA-256 that the library must have. The value is for one platform, so give it a `Condition`. |
| `OpenBlasPath` | `NUMSHARP_OPENBLAS_SEARCH_PATH` | A directory. With a version: download to it and load from it. Alone: look there first, and use the bundled library when the directory holds none. |
| `OpenBlasDistribution` | `NUMSHARP_OPENBLAS_DISTRIBUTION` | `scipy-openblas64` or `scipy-openblas32` (32-bit integers in the library's interface). The default follows NumPy for each platform. |
| `OpenBlasFeed` | `NUMSHARP_OPENBLAS_PYPI_FEED_URL` | A mirror of PyPI. A mirror requires `OpenBlasSha256`. |
| `OpenBlasDelivery` | `NUMSHARP_OPENBLAS_DELIVERY` | `build` (the default): put the library in the output of the application. `package`: put it in your own NuGet package also. `none`: ignore the settings of this reference. |

The [OpenBLAS page of the documentation](../../docs/website-src/docs/interop/openblas.md) has the
details of each setting.

## Control the load

OpenBLAS selects its kernels one time, while it loads. So the kernel pin must be part of the first
load, and the package loads the library by itself when the program first uses one of its types.
There are three ways to be first:

- **In the program, as this example does.** Set `NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL` to `0` before
  the first method that names a type of the package runs, then call
  `OpenBlasEngine.Enable(threads, coreType)`. The file that sets the variable must not name such a
  type itself, and the methods that it calls must not be inlined into it. `Backend.Start()` reports
  when the order was wrong.
- **In the environment of the process.** Set the same variable where the process starts (a service
  definition, a container, a launch profile). The code then needs no special order.
- **With no code.** OpenBLAS reads `OPENBLAS_CORETYPE` and `OPENBLAS_NUM_THREADS` from the
  environment by itself. With `OPENBLAS_CORETYPE=Sandybridge` and `OPENBLAS_NUM_THREADS=3`,
  `OpenBlasEngine.Info` showed those kernels and 3 threads. The same two variables pin NumPy.

What the kernel request does, measured:

| Situation | Result |
|---|---|
| `Enable(coreType: ...)` is the first load | The kernels are the requested ones. |
| The library is loaded, with the same kernels | The call succeeds. |
| The library is loaded, with other kernels | `InvalidOperationException`: the kernels cannot change. |
| The processor cannot run the kernels (`"SkylakeX"` without AVX-512) | `PlatformNotSupportedException`. Without this check the process would stop with an illegal instruction. |
| The name is not a kernel of OpenBLAS | No error. OpenBLAS ignores the name, so read `OpenBlasEngine.CoreName` after the call. |

## Good to know

- **The kernel pin costs speed.** `"Haswell"` runs on every x86-64 processor with AVX2 and FMA. A
  processor with faster kernels (AVX-512) then does not use them. If a result must repeat on one
  machine only, pin the thread count and leave the kernels to OpenBLAS.
- **ARM.** An ARM processor has no Haswell kernels. `Backend.Start()` then pins the thread count
  only, and says so.
- **More threads are deterministic too.** Each thread count gives its own bits on each run. But
  another count, or another OpenBLAS build, gives other bits: with 24 threads the bundled build and
  the pinned build of this example disagree, and with 1 thread they agree.
- **Publish for one platform.** A build output holds the libraries of all eight platforms
  (201 MB). `dotnet publish -c Release -r linux-x64 --self-contained false` gives 29 MB, with the
  pinned library and `openblas.source.json` beside the application.
- **Arrays and scopes.** `Reproducibility.cs` opens one `NDScope`, which releases the native memory
  of every array that the method creates.
- **Where this example was run.** On Windows 11 and on Ubuntu 24.04 (x64), against the published
  package 0.70.0. It was not run on macOS, and not on an ARM processor.
