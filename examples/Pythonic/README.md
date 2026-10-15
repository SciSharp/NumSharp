# Pythonic

Python programs that use a data science library, and their C# ports. A port has the statements of its
Python program, in the same order, and prints the same lines.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"accuracy   {accuracy_score(y_test, y_pred):.4f}")
```

```csharp
using static Python.sklearn.linear_model;
using static Python.sklearn.preprocessing;

var scaler = new StandardScaler();
var X_train_scaled = scaler.fit_transform(X_train);
var X_test_scaled = scaler.transform(X_test);

var model = new LogisticRegression(max_iter: 1000);
model.fit(X_train_scaled, y_train);

var y_pred = model.predict(X_test_scaled);
print($"accuracy   {accuracy_score(y_test, y_pred):.4f}");
```

scikit-learn runs in the Python that the process embeds. `StandardScaler` is a C# class that holds the
Python object, and `X_train` is a NumSharp array whose memory scikit-learn reads without a copy. The C#
is typed: the compiler checks each call, and an editor completes it.

[`PYTHONIC_SPEC.md`](PYTHONIC_SPEC.md) is the specification: each rule of a port and of the tree, with
the measurement behind it. It also has the plan: six libraries and 39 pairs, chosen by how many Python
files on GitHub use each library and each feature. This page is the tour.

[`NumSharpPythonnetLocalExample`](../NumSharpPythonnetLocalExample/README.md) shows the other way to
call Python, `dynamic` and a codec. It needs no classes, and the compiler checks nothing.

## The examples

| Python program | C# port | Library | Feature |
|---|---|---|---|
| [`ScikitLearn/classification.py`](ScikitLearn/classification.py) | [`ScikitLearn/Classification.cs`](ScikitLearn/Classification.cs) | scikit-learn 1.9.1 | Supervised learning: split a table, scale it, fit a classifier, predict, score. |

This is the first pair of the plan (`SK-1`). The other 38 pairs, for scikit-learn, SciPy, Matplotlib,
Pillow, OpenCV and TensorFlow, are in [section 13 of the specification](PYTHONIC_SPEC.md#13-the-pairs).

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

Put [`classification.py`](ScikitLearn/classification.py) and
[`Classification.cs`](ScikitLearn/Classification.cs) side by side. The statements are the same, and C#
writes them in this way:

| Python | C# | |
|---|---|---|
| `from sklearn.metrics import accuracy_score` | `using static Python.sklearn.metrics;` | The directive imports the functions and the classes of the module. |
| `import numpy as np` | `using NumSharp;` | A NumPy line is a NumSharp line: `np.count_nonzero(y == 0)`, `a & b`, `~mask`, `p[mask].max()`. |
| `scaler = StandardScaler()` | `var scaler = new StandardScaler();` | `var` for a new name, `new` for a class. |
| `X, y = load_breast_cancer(return_X_y=True)` | `var (X, y) = load_breast_cancer(return_X_y: true);` | A keyword argument has Python's name, with `:` for `=`. |
| `means = X[:, :10]` | `var means = X[":, :10"];` | A slice is a string. |
| `print(f"roc auc    {auc:.4f}")` | `print($"roc auc    {auc:.4f}");` | `$` for `f`. The format spec is Python's. |

The `using` lines of a port stand below its `namespace Pythonic;` line. There, the name `Python` is the
class of this project. Above that line it is pythonnet's namespace, and the directive needs the full
name: `using static Pythonic.Python.sklearn.metrics;`. An alias imports one class, as
`from sklearn.preprocessing import StandardScaler` does:
`using StandardScaler = Pythonic.Python.sklearn.preprocessing.StandardScaler;`.

One line of a port has no line in the Python program:

```csharp
using var scope = new PythonScope(NDScope.Open());     // releases every array and every Python object of this method when it returns
```

Python frees an object when its last name goes away. C# has no such moment for a local variable, so the
scope gives the port one: the end of the method. That is why no other line has a `using`.

## Arrays

On the C# side every array is a NumSharp `NDArray`, and
[`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md) copies none of them:

- **An argument is a view.** For the length of a call, NumPy reads the memory that NumSharp holds, a
  strided slice such as `X[":, :10"]` included.
- **A result is a view.** `scaler.fit_transform(X_train)` returns a NumSharp array over the memory that
  scikit-learn allocated. The array keeps that memory alive until the scope closes.
- **NumSharp computes on it as on its own.** `y_test == 0`, `p_benign[malignant].max()` and
  `np.count_nonzero(...)` run in C#, on Python's memory, without a call into Python.

## print and f-strings

`print` comes from `using static Python.builtins;`. Write the C# string with `$` where Python has `f`,
and keep the format specs of the Python program:

```csharp
print($"accuracy   {accuracy:.4f}");            // 0.9161
print($"{n:,} rows, {share:.1%} benign");        // 1,234,567 rows, 62.7% benign
print(confusion_matrix(y_test, y_pred));        // an array, as NumPy prints it
```

A hole with a spec is written by Python's own `format`, and a `double` without a spec by Python's
`str`, so `1.0` stays `1.0`. An array is printed by NumSharp, which prints as NumPy does. A test of 26
lines with alignment, sign, grouping, percent, exponent and integer-base specs printed the same in both
languages.

This works when the `$"..."` is written directly as the argument of `print`. In any other place a C#
interpolated string is formatted by .NET, with .NET's format strings.

## How the tree is written

[`Python.cs`](Python.cs) is the root: `static partial class Python`. Each library adds its branch in a
file of its own ([`ScikitLearn/Python.sklearn.cs`](ScikitLearn/Python.sklearn.cs)). A Python module is
a static class, a function is a static method, and a Python class is a C# class.

```csharp
internal static class preprocessing                                  // sklearn.preprocessing
{
    internal sealed class StandardScaler : PythonObject
    {
        public StandardScaler() : base(Create("sklearn.preprocessing", "StandardScaler")) { }

        public NDArray fit_transform(NDArray X)
        {
            using Py.GILState gil = PythonRuntime.GIL();
            using PyObject result = Call(Self, "fit_transform", [X]);
            return AsArray(result);
        }
    }
}
```

Each member has these three lines: take the GIL, call Python, give the result back as a .NET value.
Three helpers of the root class do the work:

- **`Call(target, name, args, kwargs)`** turns the arguments into Python objects, calls
  `target.name(...)`, and releases what it made. A keyword whose value is `null` is not sent, so
  Python's own default applies.
- **`Create(module, type, kwargs)`** calls the constructor of a Python class. The reference goes to
  [`PythonObject`](PythonObject.cs), the base of every class of the tree.
- **`AsArray(result)`** gives a NumPy array back as a NumSharp array over the same memory.

The rules:

- **Python's names.** Modules, functions, classes, methods and keyword arguments are spelled as Python
  spells them. Without a `using static`, the full path works:
  `Python.sklearn.metrics.accuracy_score(y_true, y_pred)`.
- **.NET values in, .NET values out.** NumSharp arrays, numbers, tuples and objects of the tree. No
  `PyObject` leaves the tree.
- **One return type.** `load_breast_cancer(return_X_y=False)` returns another type in Python. The tree
  has the form that its programs use, and refuses the other one.
- **Only a call whose name is known at compile time is a member.** A tree holds the calls that its
  programs use, and no more. The scikit-learn branch has twelve.

## Lifetime

[`PythonScope`](PythonScope.cs) collects every `PythonObject` that its thread creates. NumSharp's own
`NDScope`, which the same line opens, collects every array. When the method returns, the scope releases
the Python objects and then the arrays. Three things make sure that nothing stays shared:

- **The analyzer of NumSharp.** A port without the scope line gets the warning NDW012 for its arrays
  when it is built.
- **Python's collector.** SciPy's optimizer leaves a reference cycle that holds the training table
  after `model.fit`. Python frees such garbage only when its collector runs. So when Python still holds
  NumSharp memory after the release, the scope runs the collector, the youngest generation first. That
  took 0.8 milliseconds here.
- **The check at the end.** [`Program.cs`](Program.cs) reads `NDArrayPythonInterop.LiveExports` and
  `LiveImports` after the ports. Both must be 0: no NumPy array views NumSharp memory, and no NumSharp
  array holds Python memory. A port without its scope line ends the program with exit code 2.

## The files

| File | What it is |
|---|---|
| [`ScikitLearn/classification.py`](ScikitLearn/classification.py) | The Python program. |
| [`ScikitLearn/Classification.cs`](ScikitLearn/Classification.cs) | Its port. |
| [`ScikitLearn/Python.sklearn.cs`](ScikitLearn/Python.sklearn.cs) | The scikit-learn branch: `Python.sklearn.datasets`, `.model_selection`, `.preprocessing`, `.linear_model`, `.metrics`. |
| [`Python.cs`](Python.cs) | The root of the tree, its rules, and the three helpers. |
| [`Python.builtins.cs`](Python.builtins.cs) | `print`, `str`, `format`, and the f-string type. |
| [`Python.gc.cs`](Python.gc.cs) | `gc.collect`, which the scope uses. |
| [`PythonObject.cs`](PythonObject.cs) | The base class of a Python object that C# holds. |
| [`PythonScope.cs`](PythonScope.cs) | The scope of a port. |
| [`PythonRuntime.cs`](PythonRuntime.cs) | The backend under the tree. It starts the engine once, hands out the GIL, and keeps each imported module. It knows no library. |
| [`PythonInstall.cs`](PythonInstall.cs) | Finds the Python to embed. |
| [`Program.cs`](Program.cs) | The command line and the check at the end. |
| [`verify.py`](verify.py) | Compares each Python program with its port. |
| [`requirements.txt`](requirements.txt) | The pinned Python libraries. |
| [`PYTHONIC_SPEC.md`](PYTHONIC_SPEC.md) | The specification, and the plan of the next pairs. |

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
- **A number that NumSharp computes can differ in the last digit.** With NumSharp 0.70.0, `np.sum` of the
  143 probabilities gives 87.20879206270375, and NumPy gives 87.2087920627038. Print such a number with a
  format spec. Counts, comparisons, masks and a maximum have no such last digit.
- **A Python error** arrives as pythonnet's `PythonException`, with Python's message and traceback. The
  tree empties the frames of that traceback first (`traceback.clear_frames`). Without this step the frames
  hold the arguments of the call that failed, and a NumSharp array that was passed in stays shared with
  Python long after the error was handled: here, still after a .NET collection and a full Python
  collection.
- **A scope belongs to one thread.** Open it and close it in one method without `await`. Each thread that
  calls Python opens its own.
- **One engine for each process.** CPython and NumPy cannot be initialized a second time after a
  shutdown. The engine starts with the first Python call, and `Program.cs` shuts it down at the end.
- **Where this was run.** Windows 11 x64 (Python 3.12.12) and Ubuntu 24.04 x64 on WSL 2 (Python 3.12.14),
  with the same output on both. It was not run on macOS.
