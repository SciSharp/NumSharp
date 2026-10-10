# Pythonic

Python programs that use a data science library, and their C# ports. A port calls the library through a
`Python.*` tree that mirrors the Python module paths, keeps every NumPy line as a NumSharp line, and
prints what the Python program prints.

```python
X_train, X_test, y_train, y_test = train_test_split(means, y, test_size=0.25, stratify=y, random_state=0)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

```csharp
using PyObject parts = Python.sklearn.model_selection.train_test_split(means_np, y_np, test_size: 0.25, stratify: y_np, random_state: 0);
using PyObject model = Python.sklearn.linear_model.LogisticRegression.New(max_iter: 1000);
Python.sklearn.linear_model.LogisticRegression.fit(model, X_train_scaled, y_train);
```

This is the way to write a bridge to a Python library: each call is a typed C# member that the compiler
checks, each Python object has a visible lifetime, and arrays cross through
[`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md) with one explicit call and
no element loop. [`NumSharpPythonnetLocalExample`](../NumSharpPythonnetLocalExample/README.md) shows the
other way, `dynamic` and a codec, which is shorter and suits application code that owns the process.

## The examples

| Python program | C# port | Library | Feature |
|---|---|---|---|
| [`ScikitLearn/classification.py`](ScikitLearn/classification.py) | [`ScikitLearn/Classification.cs`](ScikitLearn/Classification.cs) | scikit-learn 1.9.1 | Supervised learning: split a table, scale it, fit a classifier, predict, score. |

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

The environment takes 260 MB on disk on Windows and 280 MB on Linux. `dotnet run` runs every port under
the name of its Python program:

```text
# ScikitLearn/classification.py
tumors     569 with 30 numbers each: 212 malignant, 357 benign
accuracy   0.9161
roc auc    0.9759
[[46  7]
 [ 5 85]]
missed     7 of 53 malignant tumors at the limit 0.5
limit      0.9528 finds all of them and refers 28 of 90 benign tumors
```

`dotnet run -- classification` runs one port and prints the lines of its Python program and no more, so
the two outputs can be compared:

```bash
# Windows. On Linux and macOS the interpreter is .venv/bin/python
.venv\Scripts\python ScikitLearn/classification.py      # the Python program
dotnet run -- classification                            # its port: the same seven lines
.venv\Scripts\python verify.py                          # runs each pair and compares the two outputs
```

```text
ScikitLearn/classification.py        7 lines   same output
```

[`verify.py`](verify.py) makes the ports embed the interpreter that runs it, so both sides compute with
the same libraries. Its exit code is the number of ports whose output is different.

## How a port reads

Read [`Classification.cs`](ScikitLearn/Classification.cs) beside
[`classification.py`](ScikitLearn/classification.py). The comment above a group of C# lines is the Python
that the group ports.

```csharp
// model = LogisticRegression(max_iter=1000)
// model.fit(X_train_scaled, y_train)
using PyObject model = Python.sklearn.linear_model.LogisticRegression.New(max_iter: 1000);
Python.sklearn.linear_model.LogisticRegression.fit(model, X_train_scaled, y_train);

// y_pred = model.predict(X_test_scaled)
// proba = model.predict_proba(X_test_scaled)
// p_benign = proba[:, 1]
using PyObject y_pred_np = Python.sklearn.linear_model.LogisticRegression.predict(model, X_test_scaled);
using PyObject proba_np = Python.sklearn.linear_model.LogisticRegression.predict_proba(model, X_test_scaled);
NDArray proba = proba_np.ToNDArray();            // (143, 2): the probabilities are NumSharp's now
NDArray p_benign = proba[":, 1"];                // NumSharp slices as NumPy does: a view, every second number
```

Three rules give every line of a port:

- **A library call is a member of the tree with the same name.** `train_test_split(...)` is
  `Python.sklearn.model_selection.train_test_split(...)`. A keyword argument is a C# named argument with
  Python's name (`test_size: 0.25`). A method of a Python object takes the object first, as Python itself
  can write it: `LogisticRegression.fit(model, X, y)`. A constructor is `New`.
- **A NumPy line is a NumSharp line.** The operators are the same: `y_test == 0`, `a & b`, `~mask`,
  `p[mask]`, `.max()`. Only a slice is spelled as a string: `X[:, :10]` is `X[":, :10"]`.

  ```csharp
  NDArray malignant = y_test == 0;                                // malignant = y_test == 0
  long missed = np.count_nonzero(malignant & (y_pred == 1));      // missed = np.count_nonzero(malignant & (y_pred == 1))
  double limit = (double)p_benign[malignant].max();               // limit = p_benign[malignant].max()
  NDArray referred = p_benign <= limit;                           // referred = p_benign <= limit
  ```
- **An array changes sides in one visible call.** `nd.ToNumpy()` gives NumPy a view of NumSharp's memory:
  nothing is copied, strided columns included. `pyobj.ToNDArray()` gives NumSharp one copy of a Python
  result. A name that ends in `_np` is NumPy's side of the array that NumSharp holds under the name
  without the ending (`means` and `means_np`).

Two more things are C#'s and have no line in the Python program:

- **`PythonRuntime.Run(() => { ... })` holds the GIL** for the Python part. A body returns .NET values (a
  number, an `NDArray`, a tuple of those), never a `PyObject`. NumSharp needs no GIL, so a NumSharp line
  can stand inside a body, between two library calls, or outside it.
- **`using` on each `PyObject`.** A Python object lives until the end of its block, and no call is chained
  on the result of another one, so no temporary is left for the finalizer. One `NDScope` releases the
  NumSharp arrays of a port.

At the end, [`Program.cs`](Program.cs) checks that no memory is still shared between the two sides
(`NDArrayPythonInterop.LiveExports` and `LiveImports` are 0). A port that forgets one `using` on an
exported array ends the program with exit code 2.

## How the tree is written

[`Python.cs`](Python.cs) is the root: `static partial class Python`. Each library adds its branch in a
file of its own ([`ScikitLearn/Python.sklearn.cs`](ScikitLearn/Python.sklearn.cs)): a nested class for
each module, a method for each call.

```csharp
internal static class model_selection
{
    private static PyObject Module => PythonRuntime.Import("sklearn.model_selection");   // imported once, never disposed

    internal static PyObject train_test_split(PyObject X, PyObject y, double? test_size = null, int? random_state = null, PyObject? stratify = null)
    {
        using PyObject function = Module.GetAttr("train_test_split");
        using var kwargs = new PyDict();
        if (test_size is { } size)
            using (PyObject value = new PyFloat(size))
                kwargs["test_size"] = value;
        if (random_state is { } seed)
            using (PyObject value = new PyInt(seed))
                kwargs["random_state"] = value;
        if (stratify is not null)
            kwargs["stratify"] = stratify;                   // the caller keeps its reference
        return function.Invoke(new[] { X, y }, kwargs);
    }
}
```

- **Always write `Python.` in front.** The class hides pythonnet's `Python.Runtime` namespace in this
  project. A file reaches pythonnet through `using Python.Runtime;` and the short names (`PyObject`, `Py`).
- **Python's names, Python's defaults.** A keyword parameter that is `null` is not sent, so Python's own
  default applies.
- **A member returns a `PyObject` that the caller owns.** A member whose name ends in `_t` returns a .NET
  value in its place: `accuracy_score_t` returns a `double`.
- **Only a call whose name is known at compile time is a member.** A class that a user selects at run
  time stays a plain `GetAttr` at the place that needs it.
- **No member moves array data.** Arrays cross with the calls of `NumSharp.Interop.pythonnet`, at the
  call site, where a reader can see them.

A tree holds the calls that its programs use, and no more. The scikit-learn branch has twelve members.

## The files

| File | What it is |
|---|---|
| [`Python.cs`](Python.cs) | The root of the tree and its rules. |
| [`ScikitLearn/Python.sklearn.cs`](ScikitLearn/Python.sklearn.cs) | The scikit-learn branch: `Python.sklearn.datasets`, `.model_selection`, `.preprocessing`, `.linear_model`, `.metrics`. |
| [`ScikitLearn/classification.py`](ScikitLearn/classification.py) | The Python program. |
| [`ScikitLearn/Classification.cs`](ScikitLearn/Classification.cs) | Its port. |
| [`PythonRuntime.cs`](PythonRuntime.cs) | The backend under the tree. It starts the engine once, holds the GIL for a `Run` body, and keeps each imported module. It knows no library. |
| [`PythonInstall.cs`](PythonInstall.cs) | Finds the Python to embed. |
| [`Program.cs`](Program.cs) | The command line and the lifetime check. |
| [`verify.py`](verify.py) | Compares each Python program with its port. |
| [`requirements.txt`](requirements.txt) | The pinned Python libraries. |

The project file is what these two commands write, and nothing more:

```bash
dotnet new console
dotnet add package NumSharp.Interop.pythonnet
```

That one package brings NumSharp and pythonnet with it.

## How the program finds Python

[`PythonInstall.cs`](PythonInstall.cs) selects one interpreter. The first match wins:

1. The interpreter that the environment variable `NUMSHARP_PYTHONNET_PYTHON` names. The value is binding:
   when the interpreter cannot be used, the program stops with the reason.
2. A `.venv` folder in the current directory, or beside the project file. That is the environment the
   commands above create.
3. `python3` or `python` on `PATH` (an activated environment, or the system Python). On Windows, the `py`
   launcher is tried last.

[`PythonRuntime.cs`](PythonRuntime.cs) then loads the library of the base install and tells the engine
which interpreter it embeds (`PythonEngine.ProgramName`), which is what makes CPython adopt a virtual
environment. [`NumSharpPythonnetLocalExample`](../NumSharpPythonnetLocalExample/README.md#how-the-program-finds-python)
explains these steps and what goes wrong without them.

When Python or a library is missing, the program prints the cause and the command that corrects it, and
ends with exit code 1:

```text
Python cannot import 'sklearn.datasets' (No module named 'sklearn'). The interpreter in use is C:\src\Pythonic\.venv\Scripts\python.exe.
Install the libraries of this project into it:
    "C:\src\Pythonic\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Good to know

- **The numbers of the library are the same to the last digit.** The accuracy (0.916083916083916), the
  area under the curve (0.9758909853249477) and the limit (0.9527624954311276) are equal in both programs
  with all their digits: scikit-learn computes on the same bytes.
- **Print a number that NumSharp computes with a fixed number of digits.** With NumSharp 0.70.0, `np.sum`
  of the 143 probabilities gives 87.20879206270375, and NumPy gives 87.2087920627038. Counts, comparisons,
  masks and a maximum have no such last digit, and the programs of this folder print those.
- **One engine for each process.** CPython and NumPy cannot be initialized a second time after a
  shutdown. `PythonRuntime` starts the engine with the first `Run` and `Program.cs` shuts it down at the
  end.
- **Where this was run.** Windows 11 x64 (Python 3.12.12) and Ubuntu 24.04 x64 on WSL 2 (Python 3.12.14),
  with the same output on both. It was not run on macOS.
