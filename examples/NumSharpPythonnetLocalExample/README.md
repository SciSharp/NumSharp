# NumSharpPythonnetLocalExample

A console app that embeds the Python on your machine in a .NET process and runs six data science
libraries over NumSharp arrays: **NumPy, pandas, SciPy, scikit-learn, Matplotlib and PyTorch**.
The arrays cross through [`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md),
in most cases without a copy: Python reads and writes the memory that NumSharp owns, and NumSharp reads
and writes the memory that Python owns.

Use this example to see how a local setup works from start to end: a virtual environment, an engine
that starts inside it, and code that reads like the Python it drives. Its twin,
[`NumSharpPythonnetGhActionsCicd`](../NumSharpPythonnetGhActionsCicd/README.md), runs the same six
libraries as tests on GitHub Actions.

## Run it

You need the .NET 10 SDK and a 64-bit Python 3.11, 3.12 or 3.13. From this folder:

```bash
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
dotnet run

# Linux, macOS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
dotnet run
```

The environment takes 1.1 GB on disk on Windows and 1.3 GB on Linux; PyTorch is the largest part.
The program prints what it embedded and where it saved the figure that Matplotlib rendered:

```text
Python      3.12.12  C:\src\NumSharpPythonnetLocalExample\.venv\Scripts\python.exe  (virtual environment)
library     C:\Python312\python312.dll
pythonnet   3.0.5
numpy       2.4.2
pandas      3.0.5
scipy       1.16.3
sklearn     1.9.1
matplotlib  3.11.2
torch       2.13.0+cpu
figure      C:\src\NumSharpPythonnetLocalExample\bin\Debug\net10.0\numsharp-matplotlib.png
```

The six examples print nothing. They are written to be read: each line does one thing, and the
comment beside it states what the line leaves behind. Each comment is an assertion in the twin
project, which its workflow runs on Windows, Linux and macOS.

## The six libraries

| File | Library | What crosses |
|---|---|---|
| [`Libraries/NumPy.cs`](Libraries/NumPy.cs) | NumPy 2.4.2 | NumPy views NumSharp's buffer and both sides write to it. The FFT of the shared samples comes back as an `NDArray`. `numpy.linalg.svd` returns three arrays into one C# tuple. `ToNumpy` / `ToNumpyCopy` / `AsNDArray` / `ToNDArray` choose a view or a copy. |
| [`Libraries/Pandas.cs`](Libraries/Pandas.cs) | pandas 3.0.5 | A `DataFrame` that sits on NumSharp's memory (`copy=False`). Means, a rolling window with missing values, a correlation matrix and a filtered frame come back as arrays. A `Series` is viewed read-only (pandas 3 Copy-on-Write). A group-by runs over a strided NumSharp column. |
| [`Libraries/SciPy.cs`](Libraries/SciPy.cs) | SciPy 1.16.3 | `scipy.signal` designs a filter (two coefficient arrays as one C# tuple) and runs it over NumSharp's samples. `scipy.ndimage` writes its result into a NumSharp array (`output=`). `scipy.stats` and `scipy.optimize` return objects whose fields read as C# numbers. |
| [`Libraries/ScikitLearn.cs`](Libraries/ScikitLearn.cs) | scikit-learn 1.9.1 | `KMeans`, `LinearRegression` and a `StandardScaler` + `PCA` pipeline fit on NumSharp arrays. Labels, centers, coefficients and predictions come back as arrays. |
| [`Libraries/Matplotlib.cs`](Libraries/Matplotlib.cs) | Matplotlib 3.11.2 | NumSharp arrays are plotted off-screen and saved as a PNG. The frame buffer of the figure is viewed as a `(300, 400, 4)` `uint8` array without a copy. A colormap turns an array of numbers into an array of colors. |
| [`Libraries/PyTorch.cs`](Libraries/PyTorch.cs) | PyTorch 2.13.0 | `ToTorch()` makes a tensor over NumSharp's buffer. Autograd trains a linear model on NumSharp data. Tensors come back as views, or as a detached copy when PyTorch refuses to share (`ToTorchNDArray(force: true)`). |

## How the program finds Python

[`PythonHost.cs`](PythonHost.cs) selects one interpreter. The first match wins:

1. The interpreter that the environment variable `NUMSHARP_PYTHONNET_PYTHON` names. The value is
   binding: when the interpreter cannot be used, the program stops with the reason. It does not
   continue with another Python.
2. A `.venv` folder in the current directory, or beside the project file. That is the environment
   the commands above create.
3. `python3` or `python` on `PATH` (an activated environment, or the system Python). On Windows,
   the `py` launcher is tried last.

The host then runs the interpreter once and asks it where it lives. Three facts decide the rest:

- **A virtual environment has no Python library of its own.** pythonnet must load the library of the
  base install (`python312.dll`, `libpython3.12.so.1.0`, or the macOS framework binary). The host
  finds it from `sys.base_prefix`.
- **The engine must be told which interpreter it embeds** (`PythonEngine.ProgramName`). CPython then
  computes its paths as that interpreter does, and adopts the virtual environment. Without the name,
  the library alone starts an engine in the base install, where the packages of the environment cannot
  be imported. On Windows, `sys.executable` is then your .NET program. On Linux, the engine takes its
  paths from the first `python3` on `PATH`, and the process crashes when that is another Python.
- **Each pythonnet release drives a fixed range of Python versions.** pythonnet 3.0.5, which the
  NuGet package resolves by default, drives Python 3.7 to 3.13. For Python 3.14, add
  `<PackageReference Include="pythonnet" Version="3.1.0" />` to the project file. The host compares
  the versions before it loads the library and tells you when they do not match.

After that, `NDArrayPythonInterop.RegisterCodec()` makes an `NDArray` cross every pythonnet boundary
without an explicit call.

## How the examples are written

```csharp
public static void Run(dynamic py)                 // a Python namespace; `import numpy as np` already ran
{
    using var scope = NDScope.Open();              // every NDArray built below is released when the method returns

    NDArray tone = np.sin(2 * Math.PI * 5 * np.linspace(0.0, 1.0, 64, endpoint: false));
    using (Py.GIL())                               // Python work runs under the GIL; NumSharp work needs none
    {
        py.tone = tone;                            // NumPy views NumSharp's buffer (the codec ran tone.ToNumpy())
        py.Exec("tone[0] = 3.0");                  // NumPy writes: tone[0] reads 3.0 in C#
        NDArray spectrum = py.np.abs(py.np.fft.rfft(tone));   // NumPy's result arrives as an NDArray
    }
}
```

- **`dynamic` for the Python side.** Attributes bind by name, an `NDArray` argument or result converts
  through the codec, a keyword argument is a C# named argument (`fs: 200`), and a Python tuple reads
  into a C# tuple. A Python program of several statements stays Python (`py.Exec("""...""")`).
- **One `NDScope` for each example.** No array is disposed by hand.
- **`Py.GIL()` around the Python parts.** In this file `np` is NumSharp and `py.np` is NumPy.
- **A NumPy integer is not a Python `int`.** Read one with `.item()` or `int(...)`. A NumPy `float64`
  converts to a C# `double` as it is.

The twelve tutorials in [`NumSharp.Interop.pythonnet.Examples`](../NumSharp.Interop.pythonnet.Examples/README.md)
cover every member of the package in this style: layouts, dtypes, lifetime, the codec, the GIL.

## Use it in your own project

Copy this folder anywhere. The project file is what these two commands write, and nothing more:

```bash
dotnet new console
dotnet add package NumSharp.Interop.pythonnet
```

That one package brings NumSharp and pythonnet with it.

To add Python to an existing application, run the second command in its project, copy
`PythonHost.cs`, call `PythonHost.Start()` once, and dispose the host when the process ends.

## Good to know

- **One engine for each process.** CPython and NumPy cannot be initialized a second time after a
  shutdown. Start the host once and keep it.
- **Shutdown.** pythonnet 3.0.5 runs 87 full collections of Python's heap while it shuts down. With
  the six libraries imported, that took 7.3 seconds on the development machine. `PythonHost`
  calls `gc.freeze()` first, and the shutdown then takes 0.2 seconds. Do that only at the end of the
  process.
- **PyTorch on Linux.** `requirements.txt` adds PyTorch's CPU-only index. Without that line, pip on
  Linux installs the CUDA build and 19 more packages. Remove the line when you want the GPU build.
- **Matplotlib without a display.** The example selects the `Agg` backend before it imports
  `pyplot`. No window opens, and the same code runs on a build server.
- **macOS: two OpenMP runtimes.** The macOS wheels of PyTorch 2.13.0 and scikit-learn 1.9.1 each
  contain a `libomp.dylib`. LLVM's OpenMP runtime stops the process when a second copy starts
  (`OMP: Error #15`), unless the variable `KMP_DUPLICATE_LIB_OK` is set. scikit-learn sets it when it
  is imported (`sklearn/__init__.py`), so the six libraries of this example need nothing more. If you
  replace scikit-learn with another library that brings its own `libomp` (faiss, LightGBM), set the
  variable from Python (`os.environ`) before the libraries compute. This example was run on Windows
  and Linux; the macOS run is the job of the twin project's workflow.
- **`No module named ...` on Windows for a module that is installed.** The path of the module file is
  longer than 260 characters. `python.exe` accepts long paths, but the `.exe` that .NET builds for an
  application does not. Put the environment in a shorter path and name it in
  `NUMSHARP_PYTHONNET_PYTHON`, or start the application with `dotnet YourApp.dll`.
- **Linux: the system Python.** On Debian and Ubuntu, `python3 -m venv` needs the package
  `python3-venv`, and the shared library of the system Python is in a package of its own: install
  `libpython3-dev`. A Python that was built without a shared library cannot be embedded. With pyenv,
  build with `--enable-shared`.
