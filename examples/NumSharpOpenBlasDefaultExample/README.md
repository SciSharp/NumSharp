# NumSharpOpenBlasDefaultExample

A console app that runs NumSharp on OpenBLAS with no configuration. The project file has one
package reference, [`NumSharp.Interop.OpenBLAS`](../../src/NumSharp.Interop.OpenBLAS/README.md).
The package brings the OpenBLAS library for your platform, and NumSharp then computes its matrix
products and its linear algebra with that library. You install nothing else: no Python, no native
package, no compiler.

Use this example to see the default setup from start to end. Its twin,
[`NumSharpOpenBlasAdvancedExample`](../NumSharpOpenBlasAdvancedExample/README.md), pins the
OpenBLAS build, the thread count and the kernels, for results that reproduce bit for bit.

## Run it

You need the .NET 10 SDK. From this folder:

```bash
dotnet run -c Release
```

```text
OpenBLAS    C:\src\NumSharpOpenBlasDefaultExample\bin\Release\net10.0\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 24] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24
bundled     True
LAPACK      True
residual    4.3e-15   the largest error of the identities in LinearAlgebra.cs
matmul      1024 x 1024 float64, the best of 3 runs
  NumSharp      89.8 ms
  OpenBLAS       8.3 ms   10.8 times faster
  the two results differ by 1.1e-12 at most
```

The first line is `OpenBlasEngine.Info`: the library file, how NumSharp calls it, the thread count,
and the build text of OpenBLAS itself, which ends with the kernels that it selected for your
processor. The times are from one run on a desktop with 24 cores. On the same machine under Linux
the factor was 5.

## What each file shows

| File | What it shows |
|---|---|
| [`Program.cs`](Program.cs) | The one statement that starts the backend, and the status report. |
| [`Products.cs`](Products.cs) | `np.dot`, `np.matmul`, `np.inner`, `np.outer`, `np.vdot`, a stack of matrices in one call, float32 and complex128, and the functions that are built on the products: `np.einsum`, `np.tensordot`, `np.linalg.multi_dot`, `np.linalg.matrix_power`. |
| [`LinearAlgebra.cs`](LinearAlgebra.cs) | `solve`, `inv`, `det`, `slogdet`, `cholesky`, `qr`, `eigh`, `svd`, `lstsq`, `pinv`, `matrix_rank`, `cond`, `norm` and `eig`. It measures each result against the identity that defines it. |
| [`Speed.cs`](Speed.cs) | One product computed by OpenBLAS and by NumSharp's own kernels, with `OpenBlasEngine.Disable()` and `Enable()` as the switch. |

`Products.cs` and `LinearAlgebra.cs` print nothing. They are written to be read: each line does one
thing, and the comment beside it states the result.

## Name the package before you compute

This is the one thing that the default setup does not do for you.

The package installs its backend from a module initializer, and the .NET runtime runs a module
initializer when the program first uses a type of that assembly. A reference in the project file
does not load the assembly. So a program that references the package and calls only `np.*`
functions never starts it. With version 0.70.0 of the package, such a program:

- computes `np.dot` and `np.matmul` with NumSharp's own kernels, with no message, and
- gets `OpenBlasMissingBackendException` from `np.linalg.cholesky`, `qr`, `svd`, `eig`, `eigh`,
  `lstsq`, `pinv` and the other functions in the last row of the table below. The text of that
  exception tells you to reference the package, which the project already does.

Put one statement at the start of the program:

```csharp
using NumSharp.Interop.OpenBLAS;

OpenBlasEngine.Enable();        // installs the backend, or throws with the reason
```

The runtime starts the package before the first statement of a method that names one of its types.
So the statement can be anywhere in your startup method, and a status line such as
`Console.WriteLine(OpenBlasEngine.Info)` has the same effect. `Enable()` is the better choice: when
no library can be loaded, the automatic start stays silent, and `Enable()` throws with the reason.

## What goes through OpenBLAS

| Functions | With the backend | Without it |
|---|---|---|
| `np.dot`, `np.matmul`, `np.inner`, `np.vdot` of float32, float64 or complex128 arrays, and the functions that are built on them (`np.einsum`, `np.tensordot`, `np.linalg.multi_dot`, `np.linalg.matrix_power`) | OpenBLAS | NumSharp's own kernels |
| The same functions for every other element type | NumSharp's own kernels | NumSharp's own kernels |
| `np.linalg.solve`, `inv`, `det`, `slogdet` | LAPACK | NumSharp's own LU solver |
| `np.linalg.cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `lstsq`, `pinv`, `matrix_rank`, `cond`, and `norm` with `ord` 2 or `"nuc"` | LAPACK | `OpenBlasMissingBackendException` |

LAPACK is inside the OpenBLAS library, so the package brings both. Every other NumSharp function
is the same with and without the package. The
[OpenBLAS page of the documentation](../../docs/website-src/docs/interop/openblas.md) has the full
list and the reasons.

## Use it in your own project

The project file is what these two commands write, and nothing more:

```bash
dotnet new console
dotnet add package NumSharp.Interop.OpenBLAS
```

That one package brings NumSharp with it. Then add the `OpenBlasEngine.Enable()` statement.

## Good to know

- **The output folder is large.** A build puts the libraries of all eight platforms of the package
  in `runtimes/`: 201 MB. To ship one platform, publish for it. The output is then 23 MB on Windows:

  ```bash
  dotnet publish -c Release -r win-x64 --self-contained false
  ```

- **The last bits are NumPy's, not NumSharp's.** OpenBLAS and NumSharp's own kernels add the terms
  of a product in a different order, so the two results differ in the last bits (`Speed.cs` prints
  by how much). Both are correct. The package bundles the OpenBLAS build of NumPy 2.4.2: on the
  development machine the library file has the same SHA-256 as the one inside NumPy 2.4.2, and a
  float32 product gave the same bits as NumPy 2.4.2 with 1, 2, 4 and 24 threads.
- **The thread count changes the last bits too.** OpenBLAS uses all cores by default.
  `OpenBlasEngine.Enable(threads: 1)` gives a result that repeats. The advanced example shows what
  else you must pin.
- **Platforms.** The package has a library for Windows (x64, arm64), Linux (x64, arm64, glibc and
  musl) and macOS (x64, arm64). Where no library can be loaded, `Enable()` throws
  `DllNotFoundException` with the list of what it tried, and the program prints it and stops.
- **Arrays and scopes.** Each example opens one `NDScope`, which releases the native memory of
  every array that the method creates. Without a scope, the analyzer that comes with NumSharp
  reports NDW012 for an array that nothing releases.
- **Where this example was run.** On Windows 11 and on Ubuntu 24.04, against the published package
  0.70.0. It was not run on macOS.
