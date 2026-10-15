# NumSharpPythonnetGhActionsCicd

An MSTest project and the GitHub Actions workflow that runs it. The tests embed CPython in the test
process through pythonnet and assert that six data science libraries work with NumSharp arrays:
**NumPy, pandas, SciPy, scikit-learn, Matplotlib and PyTorch**. The arrays cross through
[`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md).

Use this example when your own tests or pipeline jobs must call Python from .NET on a build server.
The workflow is [`.github/workflows/example-pythonnet.yml`](../../.github/workflows/example-pythonnet.yml).
It runs on `ubuntu-latest`, `windows-latest` and `macos-latest`.

The tests are the asserted twins of [`NumSharpPythonnetLocalExample`](../NumSharpPythonnetLocalExample/README.md):
the same code, with each comment of that example turned into an assertion.

## The workflow, step by step

| Step | What it does | Why |
|---|---|---|
| `actions/setup-dotnet` | Installs the .NET 10 SDK. | The project targets `net10.0`. |
| `actions/setup-python` | Installs Python 3.12 and restores the pip cache. | pythonnet 3.0.5 drives Python 3.7 to 3.13, and the pinned libraries need 3.11 or later. The key of the cache is the hash of `requirements.txt`. |
| Install the Python libraries | `python -m pip install -r requirements.txt` | The libraries go into the interpreter of the job. Each one is pinned to an exact version, so a run is reproducible. |
| Build | `dotnet build --configuration Release` | A compile error fails here, before Python starts. |
| Test | `dotnet test`, with `NUMSHARP_PYTHONNET_PYTHON` set to the `python-path` output of `setup-python` | The tests embed exactly the interpreter that the job installed into. Nothing searches `PATH`. |
| Upload the test results | Uploads the `TestResults` folder as an artifact. | It holds the `.trx` report and the figure that the Matplotlib test rendered. The step runs after a failed test step too. |

The test run also writes a table of the embedded versions (.NET, pythonnet, Python, the six
libraries) to the summary page of the job (`GITHUB_STEP_SUMMARY`).

## What the tests assert

| File | Tests | What they prove |
|---|---|---|
| [`NumPyTests.cs`](NumPyTests.cs) | 3 | NumPy views NumSharp's buffer and both sides write to it. NumPy's FFT returns as an `NDArray`. `numpy.linalg.svd` returns three arrays into one C# tuple. The explicit conversions give a view or a copy. |
| [`PandasTests.cs`](PandasTests.cs) | 4 | A `DataFrame` sits on NumSharp's memory. Statistics come back as arrays, with `NaN` for missing values. A `Series` is viewed read-only. A group-by runs over NumSharp columns. |
| [`SciPyTests.cs`](SciPyTests.cs) | 3 | `scipy.signal` filters NumSharp's samples. `scipy.ndimage` writes into a NumSharp array. `scipy.stats` and `scipy.optimize` return numbers and arrays. |
| [`ScikitLearnTests.cs`](ScikitLearnTests.cs) | 3 | `KMeans`, `LinearRegression` and a `StandardScaler` + `PCA` pipeline fit on NumSharp arrays, and their results agree with what NumSharp computes. |
| [`MatplotlibTests.cs`](MatplotlibTests.cs) | 2 | A figure is rendered without a display and saved as a PNG. Its pixels are viewed as a NumSharp array. A colormap maps an array of numbers to colors. |
| [`PyTorchTests.cs`](PyTorchTests.cs) | 3 | A tensor shares NumSharp's memory. Autograd trains a model on NumSharp data. Tensors come back as views or as detached copies. |

Each test also passes a leak check: see rule 5.

## Seven rules that keep this pipeline honest

1. **Name the interpreter. Do not search for it.** The workflow gives the tests the path of the
   interpreter it installed into (`steps.python.outputs.python-path`). [`PythonEnvironment.cs`](PythonEnvironment.cs)
   embeds that interpreter and no other one. A search of `PATH` can find a different Python, and
   the tests then pass or fail against packages that nobody selected.
2. **On a build server, a missing Python is a failure.** When the variable is not set and `CI` is
   `true` (GitHub Actions sets it), the run fails with a message that says what to add to the
   workflow. A test suite that skips itself reports green and proves nothing. On a developer
   machine the same condition is Inconclusive.
3. **One engine for the test run.** CPython and NumPy cannot be initialized a second time in a
   process. The engine starts in `[AssemblyInitialize]` and stops in `[AssemblyCleanup]`.
4. **No parallel tests.** The MSTest project template turns parallel test methods on. An embedded
   interpreter has one global lock, so this project sets `[assembly: DoNotParallelize]`.
5. **Check for leaks after each test.** A NumSharp array that Python views stays pinned until the
   last Python view dies. `NDArrayPythonInterop.LiveExports` and `LiveImports` count such memory.
   [`PythonTest.cs`](PythonTest.cs) fails a test when the counters do not return to their value
   before the test.
6. **Pin the Python side, and take the CPU build of PyTorch.** `requirements.txt` pins each library
   and adds PyTorch's CPU-only index. Without that line, pip on the Ubuntu runner installs the CUDA
   build and 19 more packages that a runner without a GPU cannot use.
7. **Draw without a display.** The Matplotlib tests select the `Agg` backend before they import
   `pyplot`. A runner has no display.

One fact about the macOS runner: the macOS wheels of PyTorch 2.13.0 and scikit-learn 1.9.1 each
contain a `libomp.dylib`. LLVM's OpenMP runtime stops the process when a second copy starts
(`OMP: Error #15`), unless the variable `KMP_DUPLICATE_LIB_OK` is set. scikit-learn sets it when it is
imported, and `PythonEnvironment.cs` imports all six libraries before the first test. If you replace
scikit-learn with another library that brings its own `libomp`, set the variable in the workflow.

## Run the tests on your machine

Make an environment (or use the one of `NumSharpPythonnetLocalExample`) and name its interpreter:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:NUMSHARP_PYTHONNET_PYTHON = "$PWD\.venv\Scripts\python.exe"
dotnet test

# Linux, macOS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
NUMSHARP_PYTHONNET_PYTHON="$PWD/.venv/bin/python" dotnet test
```

Without the variable, each test reports Inconclusive and says what to set.

## Copy it to your repository

1. Copy this folder. The project file is a plain MSTest project with one more line, the one that
   `dotnet add package NumSharp.Interop.pythonnet` writes. That package brings NumSharp and
   pythonnet with it.
2. Copy the workflow to `.github/workflows/` and change the paths: `working-directory`,
   `cache-dependency-path`, the `path` of the upload step, and the `paths` filters.

On a **self-hosted runner**, the interpreter stays on the machine between jobs. Install into a
virtual environment there, and name the interpreter of the environment:

```yaml
      - name: Install the Python libraries
        shell: bash
        run: |
          python -m venv .venv
          if [ "$RUNNER_OS" = "Windows" ]; then VENV_PYTHON=.venv/Scripts/python.exe; else VENV_PYTHON=.venv/bin/python; fi
          "$VENV_PYTHON" -m pip install -r requirements.txt
          echo "NUMSHARP_PYTHONNET_PYTHON=$("$VENV_PYTHON" -c 'import sys; print(sys.executable)')" >> "$GITHUB_ENV"
```

`PythonEnvironment.cs` embeds a virtual environment and a base install in the same way: it loads the
library of the base install and tells the engine the path of the interpreter.

For **Python 3.14**, add `<PackageReference Include="pythonnet" Version="3.1.0" />` to the project
file: pythonnet 3.0.5, which the package resolves by default, drives Python 3.7 to 3.13.
