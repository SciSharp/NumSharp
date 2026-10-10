# Pythonic specification

This document specifies **pythonic C#**: C# code that drives a Python library through pythonnet and
NumSharp, and that reads like the Python program it ports.

```python
scaler = StandardScaler()                                 # Python
X_train_scaled = scaler.fit_transform(X_train)
print(f"accuracy   {accuracy_score(y_test, y_pred):.4f}")
```

```csharp
var scaler = new StandardScaler();                        // C#
var X_train_scaled = scaler.fit_transform(X_train);
print($"accuracy   {accuracy_score(y_test, y_pred):.4f}");
```

| | |
|---|---|
| Version | 3, 2026-10-10 |
| Reference implementation | This folder, commit `ebcc245a`. It has one pair, `SK-1`. |
| Verified with | NumSharp.Interop.pythonnet 0.70.0, pythonnet 3.0.5, Python 3.12, NumPy 2.4.2, scikit-learn 1.9.1, on Windows 11 x64 and Ubuntu 24.04 x64 |
| The plan (section 13) | Six libraries and 39 pairs, chosen with measurements of GitHub from 2026-10-10 (Appendix B). |
| The planned rules (section 14) | The C# forms that the plan needs. Six scratch ports tried them against Python programs (Appendix C), with SciPy 1.16.3, Matplotlib 3.11.2, Pillow 12.3.0, opencv-python-headless 5.0.0.93, TensorFlow 2.21.0 and Keras 3.15.1, on Windows 11 x64 and Ubuntu 24.04 x64 (the scratch port with TensorFlow: on Windows only). The reference implementation does not have them yet. |
| History | Version 1 (commit `b10ac7e5`): unbound static members behind full `Python.*` paths, and a `using PyObject` for each object. The owner's review rejected that form. Version 2 (commit `8aba70c4`): sections 1 to 12. A plan of five more pairs came with it, outside this file, and the owner's review found it not good enough and too small. Version 3 adds sections 13 and 14 and the appendices B and C. It changes no rule of version 2. |
| Not yet aligned | The skill file `.claude/skills/pythonic/SKILL.md` still describes version 1. |

A rule has an identifier, for example `TREE-7`. Each rule of sections 3 to 12 is a requirement: the
reference implementation meets it, and a review checks it. Appendix A gives the measurement behind a rule
where there is one.

Section 13 is the plan: the libraries, and the pairs of each library. Section 14 lists the planned rules.
A planned rule (`PLAN-7`) is not a requirement yet: it becomes one when a pair needs it and the reference
implementation meets it.

## 1. Goals

1. **The port reads like the Python program.** The same statements, in the same order, with the same
   names. A person who knows the Python program can read the port without a manual.
2. **The compiler checks each call.** Each Python function, class and keyword that a port uses is a
   typed C# member.
3. **Python does Python's work, and NumSharp does NumPy's work.** The C# side does not write the
   behavior of the library again. Python formats the numbers, Python words the errors, and Python's
   defaults apply. A NumPy line of the program runs in NumSharp, without a call into Python.
4. **No array is copied when it changes sides.** NumPy reads the memory of NumSharp, and NumSharp reads
   the memory of NumPy.
5. **Nothing stays shared after the program.** When a port returns, Python holds no NumSharp memory and
   NumSharp holds no Python memory. A check proves it on each run.

## 2. Terms

| Term | Meaning |
|---|---|
| Python program | The reference: a `.py` file that runs by itself and prints lines. |
| Port | The C# method that has the statements of one Python program. |
| Pair | A Python program and its port. |
| Tree | The root class `Python` and its nested classes, which follow the Python module paths. |
| Module class | A static class of the tree that stands for a Python module. |
| Tree class | A C# class of the tree that stands for a Python class. It derives from `PythonObject`. |
| Member | A function of a module class, or a constructor or a method of a tree class. |
| Backend | `PythonRuntime` and `PythonInstall`: the engine, the GIL and the imports. It knows no library. |
| Scope | A `PythonScope`: the lifetime of the arrays and the Python objects of one port. |
| Export | A NumPy array that views NumSharp memory. `NDArrayPythonInterop.LiveExports` counts them. |
| Import | A NumSharp array that views Python memory. `NDArrayPythonInterop.LiveImports` counts them. |

## 3. The port

**PORT-1. A pair is two files in the folder of the library.** `<Library>/<name>.py` is the Python
program, and `<Library>/<Name>.cs` is its port, in a static class `<Name>`. The name of the port on the
command line is the name of the `.py` file without its ending. `verify.py` depends on this rule.

**PORT-2. A port is one method.** `public static void Run()` holds the statements of the program. An
entry in the table of `Program.cs` gives its name, its Python file and the method.

**PORT-3. The namespace line stands first, and the `using` lines stand below it.** Below the line
`namespace Pythonic;` the name `Python` is the root class of the tree. Above that line it is the namespace
of pythonnet, and a directive needs the full name (`Pythonic.Python...`). A port has one `using` line
for each import line of the Python program, with the Python line as a comment.

```csharp
namespace Pythonic;

using NumSharp;                                    // import numpy as np
using static Python.builtins;                      // print needs no import in Python
using static Python.sklearn.metrics;               // from sklearn.metrics import accuracy_score, confusion_matrix
```

**PORT-4. Each Python import form has one C# form.**

| Python | C# |
|---|---|
| `from pkg.mod import f, C` | `using static Python.pkg.mod;` |
| `from pkg.mod import C` | `using C = Python.pkg.mod.C;` |
| `from pkg import mod` or `import pkg.mod as mod` | `using mod = Python.pkg.mod;` |
| `import pkg.mod` | `using static Python;` and then `pkg.mod.f(...)` |
| `import numpy as np` | `using NumSharp;` |
| `import numpy as xp` | `using xp = NumSharp.np;` |

- `using static` brings in every function and every class of the module, not only the names that the
  Python line lists. When two imported modules have a function or a class of the same name, the compiler
  reports the use of that name as ambiguous. Use an alias form for one of the modules.
- An alias in lower case (`mod`, `xp`) gives the warning CS8981. A port that has one starts with
  `#pragma warning disable CS8981`.
- Without a directive the full path works: `Python.sklearn.metrics.accuracy_score(y_true, y_pred)`.

**PORT-5. The first statement of a port is the scope line, and a port has no other `using`
statement.** See `LIFE-2`.

```csharp
using var scope = new PythonScope(NDScope.Open());
```

**PORT-6. One C# statement for each Python statement, in the same order.** A port does not merge
statements, split them or change their order.

**PORT-7. The names are Python's.** `X_train`, `y_pred` and `p_benign` stay as they are, also where the
C# naming rules say another thing.

**PORT-8. A new name is declared with `var`.** Python declares no types, and the tree gives each
expression its type.

**PORT-9. The comments are the comments of the Python program**, in the same places, with `//` for `#`.

**PORT-10. A statement is written by this table.** A construct that is not in the table is not specified
yet. Section 14 has the planned form of each construct that the plan of section 13 needs.

| Python | C# |
|---|---|
| `x = f(a)` | `var x = f(a);` |
| `a, b = f(x)` | `var (a, b) = f(x);` |
| `f(a, key=1.5)` | `f(a, key: 1.5)` |
| `obj = Class(key=1)` | `var obj = new Class(key: 1);` |
| `obj.method(a)` | `obj.method(a)` |
| `a[:, :10]` and `a[:, 1]` | `a[":, :10"]` and `a[":, 1"]` |
| `a[mask]` | `a[mask]` |
| `a == 0`, `a <= b`, `a & b`, `~a` | the same |
| `np.count_nonzero(a)`, `a.max()` | the same |
| `True`, `False` | `true`, `false` |
| `print(f"{x:.4f} of {n}")` | `print($"{x:.4f} of {n}");` |
| `print(x)` | `print(x);` |

`None` has no row. As a keyword argument, `null` means "not sent" (`TREE-5`), which is the same as `None`
only where `None` is Python's default for that keyword (`PLAN-18` is the planned form).

**PORT-11. A port must not contain these:** `PyObject`, `Py`, `PythonRuntime`, `dynamic`, a conversion
call (`ToNumpy`, `ToNDArray`, `AsNDArray`), `RegisterCodec`, `Console.Write`, or a .NET format string in
a `print`. Each of them is the work of the tree.

**PORT-12. A port prints the lines of its Python program.** A number that NumSharp computes can differ
from NumPy's number in the last digit (`A-4` in the appendix). Both programs print such a number with a
format spec.

## 4. The tree

**TREE-1. The root is `static partial class Python`** in the namespace of the project.

**TREE-2. Each Python package has one file:** `<Library>/Python.<package>.cs`. The modules of the
standard library that the infrastructure uses are beside the root: `Python.builtins.cs`, `Python.gc.cs`.
A file of the tree has `using Python.Runtime;` above its namespace line. A file that declares a module
class also has `#pragma warning disable CS8981` there, because the names of the tree are Python's and
Python writes module names in lower case.

**TREE-3. A Python module is a static class**, nested along the dotted path:
`Python.sklearn.model_selection`. A module class holds its import name one time. For its functions it has
the handle `private static PyObject Module => PythonRuntime.Import("pkg.mod");`. For the constructors of
its classes it has the text `private const string Name = "pkg.mod";`.

**TREE-4. A Python function is a static method** of its module class, with Python's name.

**TREE-5. The parameters are Python's.**

- A positional parameter has Python's name and Python's position. It is required.
- A keyword parameter has Python's name. Keyword parameters keep the order of the Python signature.
  Each one is optional with the default `null`, and **a keyword that is `null` is not sent**, so Python's
  own default applies. The tree never states a default of the library.
- A member has the keywords that its programs use, and no more.

**TREE-6. A member takes and returns .NET values.** No `PyObject` is in a signature.

| Python | C# parameter | C# result |
|---|---|---|
| NumPy array | `NDArray` | `NDArray` (a view, `ARR-3`) |
| `float`, NumPy float scalar | `double` | `double` |
| `int` | `int` | `long` |
| `bool` | `bool` | `bool` |
| `str` | `string` | `string` |
| an object of a Python class | its tree class | its tree class |
| a tuple or a list of a fixed length | | a C# tuple with Python's names |
| a keyword that is not given | `null` | |

**TREE-7. A member has one return type.** When the type of Python's result depends on an argument, the
tree has the form that its programs use. The argument stays in the signature, so that the call reads as
in Python, and the member refuses each other value with a `NotSupportedException` that names the form
it does not have. Example: `load_breast_cancer(return_X_y: true)` returns `(NDArray X, NDArray y)`.

**TREE-8. A Python class is a sealed C# class that derives from `PythonObject`.**

- Its constructor is the Python constructor, with the keyword parameters of `TREE-5`:
  `public StandardScaler() : base(Create(Name, "StandardScaler")) { }`.
- Its methods are instance methods with Python's names.
- A method for which Python returns the object itself returns `this`. Then a call can follow, as in
  Python: `new LogisticRegression().fit(X, y).predict(X)`.

**TREE-9. A member is three steps: take the GIL, call Python, give back a .NET value.**

```csharp
public NDArray fit_transform(NDArray X)
{
    using Py.GILState gil = PythonRuntime.GIL();
    using PyObject result = Call(Self, "fit_transform", [X]);
    return AsArray(result);
}
```

A member that returns several values takes each item of the result into a `using` local and converts
it. A member holds each `PyObject` that it gets in a `using`, and it does not chain a call on the result
of another call.

**TREE-10. Three helpers of the root class do the conversion, and members do not convert by
themselves.**

| Helper | What it does |
|---|---|
| `Call(target, name, args, kwargs)` | Calls `target.name(*args, **kwargs)`. It makes the Python form of each argument, releases what it made, and empties the frames of a Python error (`ERR-2`). |
| `Create(module, type, kwargs)` | Calls the constructor of a Python class for the base call of a tree class. It takes the GIL itself. |
| `AsArray(result)` | Gives a NumPy array back as a NumSharp view of the same memory. |

`Call` knows these argument types. A new type is added there, in the method `Argument`, and not in a
member.

| .NET argument | Python form |
|---|---|
| `NDArray` | a NumPy view of its memory (`ARR-2`) |
| a tree object | the Python object that it holds |
| `bool` | `bool` |
| `sbyte` to `long`, `ulong` | `int` |
| `double`, `float` | `float` |
| `string` | `str` |
| a positional `null` | refused: `ArgumentNullException` |
| any other type | refused: `NotSupportedException` |

**TREE-11. Only a call whose name is known at compile time is a member.** A tree holds the calls that
its programs use, and no more. The scikit-learn branch of this folder has twelve members.

**TREE-12. The documentation of a member starts with the Python call** in a `<c>` element, for example
`<c>scaler.fit_transform(X)</c>`. It states each parameter, the result, and each exception.

**TREE-13. The accessibility is the choice of the project:** `internal` in an application, as here, and
`public` in a library.

## 5. Arrays

**ARR-1. On the C# side each array is a NumSharp `NDArray`.** A port has no second name for the NumPy
side of an array.

**ARR-2. An argument is a view.** `Call` exports the array with `ToNumpy()` for the length of the call
and releases the export after it. A strided or a reversed array is exported as it is. When the Python
side keeps the array, the memory of NumSharp stays alive until Python lets go.

**ARR-3. A result is a view.** `AsArray` imports the array with `AsNDArray(allowReadonly: true)`. The
NumSharp array keeps the memory of Python alive until it is disposed, which the scope does. A read-only
NumPy array gives a read-only NumSharp array. A write on one side shows on the other side, as between
two names of one array in Python.

**ARR-4. A member copies a result only when a view is not possible.** NumSharp has no complex64, for
example. Such a member calls `ToNDArray()` in the place of `AsArray`, and its documentation says so.

**ARR-5. No code moves elements one at a time.** No loop fills a Python list from an array, and no loop
reads a Python list into an array.

**ARR-6. The codec is not registered.** `NDArrayPythonInterop.RegisterCodec()` changes the conversion
for the whole process. A tree converts inside its own calls and changes nothing outside them.

**ARR-7. NumSharp computes on a view as on its own memory.** `y_test == 0`, `p[mask].max()` and
`np.count_nonzero(...)` run in C#, on the memory of Python.

## 6. The GIL

**GIL-1. Each member takes the GIL itself.** A port never takes it and never sees it.

**GIL-2. The one entry is `PythonRuntime.GIL()`.** It starts the engine when the engine does not run
yet. Outside the backend, no code calls `Py.GIL()`.

**GIL-3. Each use of a `PyObject` happens under the GIL**, its `Dispose` included.

**GIL-4. The GIL is re-entrant.** A member can call another member, and the conversion calls of
NumSharp.Interop.pythonnet take the GIL again inside a member without harm.

**GIL-5. After its start the engine gives the GIL up** (`PythonEngine.BeginAllowThreads()`), so that
each thread, the first one included, takes it for the length of one member.

## 7. Lifetime

**LIFE-1. A tree object owns one Python reference.** `Dispose` releases it under the GIL. A second
`Dispose` does nothing, and a `Dispose` after the shutdown of the engine does nothing. After the release
a method throws `ObjectDisposedException`, and `ToString()` gives `<released StandardScaler>` and does
not throw. `PythonObject` has no finalizer: an object that is never disposed is released by the
finalizer of pythonnet at some later collection.

**LIFE-2. A method that calls the tree opens one scope, as its first statement.**

```csharp
using var scope = new PythonScope(NDScope.Open());
```

The `NDScope` of NumSharp collects each array that the thread creates, the views of Python memory
included. The `PythonScope` collects each tree object that the thread creates. That is why no other line
of a port has a `using`.

**LIFE-3. The `NDScope.Open()` call stands in the method itself.** The analyzer of NumSharp looks for
that call in each method. A method that creates arrays and does not open a scope gets the warning NDW012.
A scope class that opens the `NDScope` inside its own constructor hides the call from the analyzer.

**LIFE-4. A scope closes in this order:** the tree objects, the newest first; then the arrays; then the
Python garbage of `LIFE-5`.

**LIFE-5. The scope runs Python's collector when Python still holds NumSharp memory.** Python frees an
object that is part of a reference cycle only when its collector runs, and such garbage can hold an
array that the program passed in (`A-1`). The scope reads `LiveExports` when it opens. When more exports
are alive after the release, it runs `gc.collect(0)`, then `gc.collect(1)`, then `gc.collect(2)`, and
stops when the count is back.

**LIFE-6. A scope belongs to one thread, and scopes close innermost first.** A `using` declaration in a
method without `await` gives both. A scope that is closed on another thread or out of order throws
`InvalidOperationException`. Each thread that calls the tree opens its own scope.

**LIFE-7. What a scope collected does not leave it.** A port returns numbers and text. An array that
must live longer is copied and taken out: `var kept = a.copy(); NDScope.Detach(kept);`. A tree object
that must live longer is created outside each scope and disposed by hand.

**LIFE-8. After a scope closed, nothing is shared.** The number of exports is back at once. The number
of imports is back a moment later: NumSharp hands the Python memory back at once, and a thread of the
thread pool frees it under the GIL (`A-2`). Code that checks the counts waits for them.

**LIFE-9. The engine shuts down after the last scope closed.** A NumSharp array that views Python memory
must not be read after the shutdown.

## 8. Errors

**ERR-1. A Python error arrives as the `PythonException` of pythonnet,** with Python's type name
(`e.Type.Name`), Python's message and Python's traceback. A member does not catch it, translate it or
word it again.

**ERR-2. `Call` empties the frames of the traceback before the error leaves the tree,** with Python's
own `traceback.clear_frames`. The frames hold the arguments of the call that failed. Without this step a
NumSharp array that was passed in stays shared with Python long after the error was handled (`A-3`). The
error keeps its type, its message and its traceback text.

**ERR-3. A problem of the setup is a `PythonInteropException`:** no interpreter, a library that
pythonnet cannot load, or a module that is not installed. Its message states the cause and the command
that corrects it. A program prints the message without a stack trace.

**ERR-4. The tree's own refusals are .NET exceptions.**

| Exception | Cause |
|---|---|
| `NotSupportedException` | A form that the tree does not have (`TREE-7`), an argument type without a Python form (`TREE-10`), or a result that NumSharp cannot view (`ARR-4`). |
| `ArgumentNullException` | A positional argument that is `null`. |
| `ObjectDisposedException` | A tree object after its release. |
| `InvalidOperationException` | A scope that is closed on another thread or out of order, or a call after the shutdown of the engine. |

## 9. print and f-strings

**PRINT-1. `print` comes from `using static Python.builtins;`.** It writes to the console of .NET, so
its lines stay in order with each other output of the process.

**PRINT-2. `print` has three forms.**

| Form | What it writes |
|---|---|
| `print($"...")` | One line. The compiler passes each hole to the type `Python.fstring`. |
| `print(value)` | `str(value)` and a line end. |
| `print()` | An empty line. |

**PRINT-3. Python formats the holes.**

| Hole | Written by |
|---|---|
| A hole with a format spec: `{x:.4f}`, `{n:,}`, `{name:>10}`, `{share:.1%}` | Python's `format(value, spec)` |
| A `double` or a `float` without a spec | Python's `str`: `1.0` stays `1.0`, and `0.00001` is `1e-05`. A C# `float` is sent with its exact value, so `0.1f` prints `0.10000000149011612`, as `float(np.float32(0.1))` does in Python. |
| A tree object | Python's `str`: `LogisticRegression(max_iter=1000)` |
| An integer, a text, a `bool`, `null` | C#, with Python's text: `True`, `False`, `None` |
| A NumSharp array without a spec | NumSharp, which prints an array as NumPy does (`A-5`) |
| A 0-d NumSharp array with a spec | NumPy, which formats its one element |

**PRINT-4. The `$"..."` is written directly as the argument of `print`.** In each other place a C#
interpolated string is formatted by .NET, with .NET's format strings, and a Python spec gives wrong text
without an error. For the same reason `print` takes one argument: Python's `print(a, b)` is
`print($"{a} {b}")`.

**PRINT-5. The alignment syntax of C# (`{x,10}`) does not compile in a `print`** (error CS1739). Python
writes the width inside the spec: `{x:>10}`.

**PRINT-6. `str(value)` and `format(value, format_spec)` are members too,** with the rules of
`PRINT-3`.

## 10. The backend

**BACK-1. One engine for each process.** CPython and NumPy cannot be initialized a second time after a
shutdown. `PythonRuntime.Initialize()` starts the engine when it is first needed and returns at once
after that. It uses an engine that other code of the process already started, and it does not shut that
engine down.

**BACK-2. `PythonInstall.Locate()` selects one interpreter.** The first match wins.

1. The interpreter that the environment variable `NUMSHARP_PYTHONNET_PYTHON` names. The value is
   binding: when that interpreter cannot be used, the program stops with the reason.
2. A `.venv` folder in the current directory, or beside the project file.
3. `python3` or `python` on `PATH`. On Windows, the `py` launcher is tried last.

**BACK-3. The engine is told which interpreter it embeds** (`PythonEngine.ProgramName`), for each
interpreter. This is what makes CPython adopt a virtual environment. After the start the backend compares
`sys.prefix` with the selected environment and stops when they differ.

**BACK-4. `PythonRuntime.Import(name)` imports a module one time** and returns the same handle on each
later call. The handle belongs to the process, and no caller disposes it.

**BACK-5. `PythonRuntime.Shutdown()` is for the end of the process.** Before it shuts the engine down it
runs `gc.collect()` and `gc.freeze()`, because pythonnet 3.0.5 runs 87 full collections during its
shutdown. It also sets `RuntimeData.FormatterType` to `NoopFormatter`.

**BACK-6. The backend knows no library.** A library is a branch of the tree.

## 11. Verification

**VER-1. A pair prints the same lines.** `verify.py` runs each Python program and its port with the same
interpreter, and compares the two outputs line by line. Its exit code is the number of ports whose
output is different.

**VER-2. A run ends with the lifetime check.** After the ports, `Program.cs` waits up to 5 seconds for
`LiveExports` and `LiveImports` to reach 0 (`LIFE-8`). When one of them does not, the program prints both
counts.

| Exit code | Meaning |
|---|---|
| 0 | Each port ran, and nothing stayed shared. |
| 1 | An unknown port name, or a problem of the setup (`ERR-3`). |
| 2 | A port left memory shared with Python. |

A Python error that a port does not catch ends the program as an unhandled exception, with the Python
traceback and the .NET stack trace.

**VER-3. The project builds with no warning.** NDW012 is on. It is the first guard for a port that has
no scope line; the lifetime check is the second one.

**VER-4. A new conversion or a new format path is tested against Python** before a port uses it: the
same lines from a Python script and from C#, and a comparison of the two outputs.

**VER-5. The documentation of a pair states where it ran, and where it did not.**

## 12. The project

**PROJ-1. The project file is what `dotnet new console` and `dotnet add package
NumSharp.Interop.pythonnet` write, and nothing more.** So the project builds against the published
package, and its code uses the released API only.

**PROJ-2. `requirements.txt` pins each Python library** to the version that the pairs were verified
with. The environment of the project is `.venv`, beside the project file, and git ignores it.

**PROJ-3. The folder has this form.**

| Path | Content |
|---|---|
| `<Library>/<name>.py`, `<Library>/<Name>.cs` | A pair (`PORT-1`). |
| `<Library>/Python.<package>.cs` | The branch of the library (`TREE-2`). |
| `Python.cs` | The root and the helpers (`TREE-10`). |
| `Python.builtins.cs`, `Python.gc.cs` | `print`, `str`, `format`, the f-string type; `gc.collect`. |
| `PythonObject.cs`, `PythonScope.cs` | The base of a tree class; the scope. |
| `PythonRuntime.cs`, `PythonInstall.cs` | The backend. |
| `Program.cs`, `verify.py` | The table of the ports and the lifetime check; the comparison. |

## 13. The pairs

This section is the plan: the libraries, and the pairs of each library. One pair is done (`SK-1`). A
number such as 1.39 M or 924 K is a count of Python files on GitHub that hold a name (M is million, K is
thousand). Appendix B has the method, the exact counts and the limits of the counts.

### 13.1 What a pair is

**PAIR-1. A pair answers one question with real data.** The question is the first comment of the
program. The data is a table or a file that a library of the pair ships, or numbers that the program
makes with a seed. A pair downloads nothing.

**PAIR-2. A pair shows a feature that many programs use.** The measure is the number of Python files on
GitHub that hold the name of the feature (Appendix B). The first two pairs of a library show its two
most used features.

**PAIR-3. NumSharp does a part of the work.** The program has NumPy lines between the calls of the
library: they prepare the input, or they check the result. In the port those lines run in NumSharp, on
memory that Python owns.

**PAIR-4. A pair prints its proof.** Each result is a printed line. A picture or a file is proved by
numbers that the program computes from it: a size, a count of pixels, a value at a known place.

**PAIR-5. A pair runs without a person, without a display and without a network.** A chart is drawn off
screen. A file goes into the folder that `tempfile.gettempdir()` gives.

**PAIR-6. A pair prints the same lines on each run.** Each source of random numbers has a seed. The
random numbers of NumPy and of NumSharp are the same after the same seed (`C-11`), so the program can
make its own noise.

**PAIR-7. A pair is short.** Its program has fewer than 60 statements and runs in less than 10 seconds.

**PAIR-8. A pair states what it adds to the approach:** the planned rules of section 14 that it is the
first to need. When the pair is done, those rules move into sections 3 to 12.

**PAIR-9. The Python program is plain Python,** as a person writes it for Python only. Two limits come
from C#. A name keeps one type through the program. And the program uses the NumPy functions that
NumSharp has (`C-12`).

### 13.2 The libraries

| # | Library | Import name | Python files that import it | Repositories that depend on it | State |
|---|---|---|---:|---:|---|
| 1 | Matplotlib | `matplotlib` | 9,879,552 | 1,861,681 | planned |
| 2 | SciPy | `scipy` | 6,717,440 | 1,415,512 | planned |
| 3 | scikit-learn | `sklearn` | 6,078,464 | 1,345,505 | one pair done |
| 4 | Pillow | `PIL` | 3,817,472 | 2,457,170 | planned |
| 5 | OpenCV | `cv2` | 3,751,936 | not available | planned |
| 6 | TensorFlow, with Keras | `tensorflow` | 2,969,600 | 540,403 | planned |

The order is the number of Python files that import the library. That number is the measure of this
plan, because it counts programs. The number of dependent repositories is the cross-check.

- **Rows 1 to 3** are the libraries of version 2, which chose them without a measurement. The
  measurement confirms them: after NumPy, pandas and PyTorch, they are the three libraries that the most
  Python files import.
- **Rows 4 to 6 are new.** They are the next three libraries of the same list that exchange arrays.
- **Not in the plan:**
  - NumPy (47.4 M): NumSharp is the NumPy of a port.
  - pandas (18.2 M) and PyTorch (18.0 M): NumSharp.Interop.pythonnet has its own conversions for them,
    and [`NumSharpPythonnetLocalExample`](../NumSharpPythonnetLocalExample/README.md) shows them.
  - Requests (5.9 M), tqdm (2.7 M), PyYAML (2.0 M) and Beautiful Soup (1.2 M): no array goes to them or
    comes from them.
  - Transformers (2.9 M): it runs on PyTorch, and its programs download a model (`PAIR-1`).
  - Keras as its own import (0.8 M): it is a part of TensorFlow (`tf.keras`), and the TensorFlow pairs
    use it.
- **TensorFlow is the heavy one.** Its package for Windows has 351 MB, and an environment with it has
  1.9 GB. An environment with the other five libraries has 445 MB. TensorFlow loads into a .NET process
  and gives the same numbers there (`C-17`). The plan gives it its own requirements file (`PLAN-51`).
  When the owner drops it, the next libraries of the list that exchange arrays are scikit-image
  (0.38 M), statsmodels (0.31 M) and XGBoost (0.17 M). Before them the list has SymPy (1.3 M, no
  arrays), NetworkX (1.0 M, graphs of Python objects), seaborn (0.9 M, charts on top of Matplotlib,
  mostly from pandas tables) and Plotly (0.8 M, its result is a web page).

### 13.3 scikit-learn

The five modules of scikit-learn with the highest counts are `sklearn.metrics` (2.49 M),
`sklearn.model_selection` (1.63 M), `sklearn.preprocessing` (1.38 M), `sklearn.datasets` (1.12 M) and
`sklearn.linear_model` (1.07 M). The pair that is done, `SK-1`, uses all five. The sixth is
`sklearn.ensemble` (836 K), which `SK-2` adds.

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `SK-1` | Classification. `train_test_split` 1.39 M, `StandardScaler` 924 K, `confusion_matrix` 733 K, `accuracy_score` 725 K, `LogisticRegression` 614 K, `roc_auc_score` 472 K | **Which tumors are malignant?** 569 tumors (`load_breast_cancer`). Split, scale, fit, predict, score. NumSharp finds the limit that misses no malignant tumor. | Done. Its rules are sections 3 to 12. |
| `SK-2` | Regression. `sklearn.ensemble` 836 K, `mean_squared_error` 666 K, `LinearRegression` 516 K, `r2_score` 496 K, `RandomForestRegressor` 239 K | **How far does diabetes go in one year?** 442 patients (`load_diabetes`). A line, a ridge and a forest learn in one loop. For each: the error (NumSharp takes the root of the mean squared error) and R². Then the three measurements with the largest weight, by name. | One name for objects of different classes (`PLAN-40`). Attributes: `coef_`, `feature_importances_` (`PLAN-33`). A result object with a list of text (`PLAN-32`). Two forms of one function (`PLAN-26`). `for`, `zip`, `enumerate` (`PLAN-7`). |
| `SK-3` | Clustering. `KMeans` 556 K, `MinMaxScaler` 305 K, `PCA` 250 K, `silhouette_score` 90 K | **Which wines are alike?** 178 wines with 13 measurements (`load_wine`), each scaled to the range from 0 to 1. k-means for 2 to 6 groups, with the inertia and the silhouette of each. Then 3 groups against the 3 true grape kinds: a table of counts (NumSharp) and the adjusted Rand score. Two principal components, and the share of the variance that they keep. | Objects that a loop makes (`range`). A keyword that takes a text or a number: `n_init="auto"` (`PLAN-28`). `None` on purpose (`PLAN-18`). |
| `SK-4` | Model selection. `sklearn.pipeline` 552 K, `sklearn.svm` 419 K, `classification_report` 415 K, `GridSearchCV` 280 K, `cross_val_score` 246 K | **Which settings read handwritten digits best?** 1,797 pictures of 8 × 8 points (`load_digits`). A pipeline of a scaler and a support vector machine, a grid of four settings with 5 folds, the best setting, its score on pictures that it did not see, the report for each digit, and the two digits that are mixed up most (NumSharp, from the confusion matrix). | Literals: a list of tuples and a dict (`PLAN-3`, `PLAN-4`). Objects of the tree as arguments. An attribute that is an object (`best_estimator_`). A dict that comes back (`best_params_`, `cv_results_`). A result that is text of several lines. |
| `SK-5` | Trees and forests. `RandomForestClassifier` 482 K, `DecisionTreeClassifier` 258 K, `feature_importances_` 226 K, `KNeighborsClassifier` 203 K | **Which measurements decide?** The tumors of `SK-1`. Three kinds of classifier with 5 folds each. The five most important measurements of the forest, by name. The first levels of one tree as text (`export_text`). | A NumPy array of text as a `string[]` (`PLAN-32`). Classifiers behind one name (`PLAN-40`). |
| `SK-6` | Saved models. `joblib.load` 498 K, `joblib.dump` 367 K | **Train now, predict later.** A pipeline learns the digits and goes into a file. A second object comes out of the file and gives the same answers (`np.array_equal`, NumSharp). This is the C# service that uses the model of a Python team. | Files. An object whose class Python decides when the program runs (`PLAN-43`). |
| `SK-7` | Text. `TfidfVectorizer` 241 K, `CountVectorizer` 129 K, `MultinomialNB` 67 K | **To which team does a message go?** 24 short messages, each with the name of a team, in the program. The words become a sparse matrix, a naive Bayes model learns it, and three new messages get a team and a probability. The five words that speak most for each team (NumSharp sorts the weights). | A list of text as an argument and as a result. A sparse matrix that stays a Python object (`PLAN-39`). Labels that are text. |

### 13.4 SciPy

SciPy is a set of modules. The count of a pair is the count of its module.

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `SP-1` | `scipy.stats` 1.64 M. `stats.norm` 249 K, `pearsonr` 228 K, `spearmanr` 205 K, `ttest_ind` 118 K, `linregress` 95 K | **Is the difference real?** The tumors of `SK-1`, malignant against benign. A t-test and a rank test on the mean radius, the correlation of radius and area, a line through two measurements, the 95 % interval of a mean from the normal distribution, and a chi-square test on a table of 2 × 2 counts that NumSharp builds. | A result with names that also unpacks (`PLAN-32`). An object at module level: `norm.ppf(0.975)` (`PLAN-36`). A tuple of numbers and an array. |
| `SP-2` | `scipy.sparse` 1.47 M. `csr_matrix` 717 K | **A table that is almost empty.** 2,000 customers, 500 products and 20,000 ratings as three NumSharp arrays: row, column, value. The sparse matrix from them, the share of its filled cells, its memory against the full table, the product with the most ratings, and the two products that the same customers rate (`A.T.dot(A)`). Then a system of 100,000 equations with three values in each row (`diags`, `spsolve`), which NumSharp checks. | An object that the program only holds: it never becomes an array. Operators, an index and attributes on a tree object (`PLAN-39`). A tuple inside a tuple as an argument. |
| `SP-3` | `scipy.optimize` 1.31 M. `curve_fit` 214 K, `minimize` 183 K, `fsolve` 70 K | **Which curve fits the measurements?** 50 measurements of a decay with noise, which NumSharp makes. `curve_fit` gives the three parameters and their errors (NumSharp takes the root of the diagonal). `minimize` finds the same parameters from a cost function with bounds. `brentq` finds the time at which half is left. | C# functions that Python calls (`PLAN-22`, `PLAN-30`, `PLAN-49`). A result object with attributes (`PLAN-32`). A list of tuples with `None`. |
| `SP-4` | `scipy.special` 973 K | **Numbers that are too large for a float.** Scores of 1,000 and more: the plain formula of softmax overflows and gives `nan`, and `logsumexp` and `softmax` give the answer. `expit`, `gammaln` for a binomial of large numbers, and `comb(100, 50, exact=True)`, an integer of 30 digits. | A Python integer of any size (`PLAN-32`). |
| `SP-5` | `scipy.linalg` 838 K | **Where is the system after one hour?** A machine with three states and the rates between them, as a 3 × 3 matrix from NumSharp. `expm` gives the probabilities after a time, `eig` the speeds of decay as complex numbers, and `lu_factor` with `lu_solve` the shares in the long run. | Complex arrays as views. A tuple of arrays that goes back into the next call. |
| `SP-6` | `scipy.signal` 718 K. `filtfilt` 86 K, `find_peaks` 85 K, `savgol_filter` 57 K | **How fast does the heart beat?** A signal of 10 seconds at 500 samples for each second: beats, a slow drift and noise, all from NumSharp. A band-pass filter (`butter`, `sosfiltfilt`), the peaks (`find_peaks`), the rate from the distances of the peaks (`np.diff`, NumSharp), and the strongest frequency (`welch`). | A tuple with a dict of arrays (`PLAN-32`). An array of indexes from Python as an index in NumSharp. |
| `SP-7` | `scipy.spatial` 673 K. `cdist` 304 K, `cKDTree` 107 K | **Which store is nearest?** 1,000 customers and 20 stores as points from NumSharp. A k-d tree gives the two nearest stores of each customer. NumSharp counts the customers of each store. `cdist` gives the two stores that are closest to each other, and `ConvexHull` the area that the stores cover. | A tree class whose method returns two arrays. |
| `SP-8` | `scipy.io` 578 K. `loadmat` 212 K | **Data for MATLAB, and sound.** An array and a number go into a `.mat` file and come back: the array has two dimensions then, which is the rule of MATLAB. One second of a tone goes into a `.wav` file as 16-bit numbers and comes back with its rate. | A dict with arrays as an argument and as a result (`PLAN-4`, `PLAN-32`). Files in a folder that a `with` block makes and removes (`PLAN-20`). |
| `SP-9` | `scipy.interpolate` 547 K. `interp1d` 192 K | **Measurements at uneven times.** 30 measurements at uneven times become 200 values on an even grid: a straight line between the points (`np.interp`, NumSharp) against a cubic curve (`interp1d`, `CubicSpline`), each with its largest distance from the true curve. | An object that the program calls (`PLAN-35`). |
| `SP-10` | `scipy.ndimage` 436 K. `gaussian_filter` 161 K | **How many spots are in the picture?** A picture of 200 × 200 points with 12 soft spots and noise, from NumSharp. Smooth it, cut it at a level (NumSharp), `label` the regions, and print the size and the center of each. | A function that writes into the array of the caller: `output=` (`PLAN-42`). A tuple of an array and a number. |
| `SP-11` | `scipy.integrate` 357 K. `odeint` 183 K, `solve_ivp` 65 K | **When is the peak of an epidemic?** The SIR model as a C# function of three lines. `solve_ivp` runs it over 160 days. NumSharp finds the day and the height of the peak. `quad` gives one area with its error. | A C# function that Python calls about 100 times in one run: the pair measures the cost of one call. |

### 13.5 Matplotlib

Each chart is drawn off screen (`matplotlib.use("Agg")`, `PAIR-5`). `plt.show` has the highest count of
all (4.12 M), and no pair calls it: it opens a window and waits for a person.

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `MP-1` | The functions of pyplot. `plt.figure` 3.41 M, `plt.title` 2.49 M, `plt.savefig` 2.44 M, `plt.plot` 2.27 M, `plt.xlabel` 2.16 M, `plt.legend` 1.64 M, `plt.tight_layout` 1.50 M | **A year of temperatures.** 365 values with a season and noise, and their mean over 30 days (`np.convolve`, NumSharp). One chart with two lines, a title, the names of the axes, a legend and a grid, saved as a PNG file. The proof: the limits of the axes, the texts of the legend, and the size and the share of ink of the picture that `plt.imread` reads back. | A module that keeps the state: functions only, no object. Text arguments with a format spec (`PLAN-29`). A positional argument that the caller can leave out: `"r--"` (`PLAN-27`). Results that the program does not use (`PLAN-32`). |
| `MP-2` | Charts as objects. `plt.subplots` 3.22 M, `ax.set_title` 1.45 M, `ax.plot` 1.44 M, `plt.scatter` 651 K, `plt.bar` 276 K, `plt.hist` 270 K | **Four views of one table.** The iris flowers (`load_iris`) in a grid of 2 × 2 charts: a histogram, a scatter chart with one color for each kind, a bar chart of the means, and box plots. The proof: the counts and the edges that `hist` returns, the number of bars, the titles. | A function with two forms: `plt.subplots()` and `plt.subplots(2, 2)` (`PLAN-26`). A grid of Python objects: `axs[0, 1]` (`PLAN-41`). A tuple with two arrays and an object. A list of arrays as an argument. |
| `MP-3` | Pictures of numbers. `plt.imshow` 633 K, `ax.imshow` 496 K, `plt.colorbar` 442 K | **Which measurements go together?** The correlations of the 13 measurements of the wines (`np.corrcoef`, NumSharp) as a grid of colors with a color bar, the names on both axes, and the number in each cell. NumSharp names the two measurements that go together most. | An object of one call as an argument of the next: `fig.colorbar(im)`. Two loops with one text call in each cell: 169 calls, and the pair measures the cost of one. Lists of text as arguments. |
| `MP-4` | The picture as an array. `canvas.draw` 441 K, `buffer_rgba` 50 K, and `fig.savefig` into `io.BytesIO()` | **A chart for a .NET window or for a web answer.** A chart is drawn off screen. Its pixels come to C# as a NumSharp array of height × width × 4, without a copy and without a file. Its PNG file comes as a `byte[]`. This is what a desktop program or a web service in C# needs from Matplotlib. | A buffer that is not a NumPy array, as a view (`PLAN-32`). Bytes (`PLAN-32`). A file in memory (`PLAN-24`). |
| `MP-5` | Style. `plt.rcParams` 662 K, `plt.style.use` 242 K, `plt.fill_between` 97 K, `plt.annotate` 83 K, `plt.errorbar` 75 K | **A chart for a report.** Ten measurements at each of eight points, from NumSharp: the means with error bars, a band of one standard deviation, and an arrow at the highest mean, in a style that a `with` block sets for this chart only. | `with` (`PLAN-20`). An item of a module object that the program sets: `plt.rcParams["font.size"] = 11`. A dict as an argument (`PLAN-4`). |

### 13.6 Pillow

Pillow ships no picture. The pairs use the photo of Grace Hopper that Matplotlib ships
(`cbook.get_sample_data("grace_hopper.jpg")`, 512 × 600 points).

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `PIL-1` | Picture files. `Image.open` 2.61 M, `Image.LANCZOS` 210 K | **Small pictures of a photo.** The photo, its size and its mode. A copy with a quarter of the size, a square cut, and a gray copy, each saved as a JPEG file and as a PNG file. The proof: the sizes, the modes, the bytes of each file against the first, and the mean brightness (NumSharp). | A module and its class under one name: `Image.open(path)` and `img.resize(size)` (`PLAN-37`). A constant of a module: `Image.LANCZOS` (`PLAN-36`). A tuple of integers that the program indexes and unpacks: `img.size` (`PLAN-32`). |
| `PIL-2` | Pictures and arrays. `Image.fromarray` 1.01 M | **Change a photo with array math.** The pixels as an array (`np.asarray(img)`), and then NumSharp: more contrast (`np.clip`), a sepia tone (a matrix product), a mirror (`a[:, ::-1]`) and a frame. The array becomes a picture again (`Image.fromarray`). The proof: the histogram of Pillow against the counts of NumSharp. | An object that NumPy converts: `np.asarray(img)` (`PLAN-38`). NumSharp does the picture math. |
| `PIL-3` | Drawing. `Image.new` 952 K, `ImageDraw` 534 K, `ImageFont` 365 K | **A label for a part.** A new picture with a frame, two lines of text in the middle (Pillow gives the size of the text, and C# computes the place), and a bar of 12 cells of different colors. The proof: the color at five places, and the count of the points of each color (NumSharp). | A keyword that takes values of several forms: a color is a name, a number or three numbers (`PLAN-28`). Tuples as arguments. Text with a format spec as an argument (`PLAN-29`). |
| `PIL-4` | Filters. `ImageOps` 167 K, `ImageFilter` 155 K, `ImageChops` 86 K, `ImageEnhance` 64 K | **Make a dark scan readable.** The gray photo, which NumSharp makes dark and dull. Then `ImageOps.autocontrast`, `ImageFilter.SHARPEN` and `ImageEnhance.Contrast(img).enhance(1.5)`. `ImageChops.difference` shows where two pictures differ: `getbbox()` gives the place, or `None` when they are equal. | An object as an argument: `img.filter(ImageFilter.GaussianBlur(2))`. A result that can be `None` (`PLAN-32`). |
| `PIL-5` | Animations. `ImageSequence` 76 K | **An animation from arrays.** 20 frames of a wave, from NumSharp, become one GIF file. The file is opened again, and its frames come back as arrays. | A list of objects of the tree as an argument (`PLAN-41`). A loop over a Python iterator (`PLAN-34`). |

### 13.7 OpenCV

OpenCV has no picture class: a picture is a NumPy array, so in a port it is a NumSharp array. The pairs
use the photo of Grace Hopper too.

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `CV-1` | Files and colors. `cv2.cvtColor` 1.48 M, `cv2.imread` 1.16 M, `cv2.resize` 813 K, `cv2.imwrite` 727 K, `cv2.imdecode` 156 K | **A photo for a model.** The photo as an array (`cv2.imread`). Blue-green-red to red-green-blue and to gray, and a copy of 224 × 224 points. Then NumSharp: numbers from 0 to 1, minus the mean, divided by the deviation, the color axis first, and one more axis in front, for the shape 1 × 3 × 224 × 224. The JPEG file as bytes in memory, and back (`imencode`, `imdecode`). | A library of functions and arrays only: its branch has no tree class. Constants of a module that Python gives: `cv2.COLOR_BGR2GRAY` (`PLAN-36`). `h, w = img.shape[:2]` (`PLAN-9`). A tuple of a bool and an array. |
| `CV-2` | Drawing. `cv2.putText` 577 K, `cv2.rectangle` 513 K, `cv2.line` 318 K, `cv2.circle` 297 K, `cv2.addWeighted` 123 K | **Mark what was found.** A NumSharp array of 480 × 640 × 3 zeros. OpenCV draws three boxes with a name and a score each, a line, and a band that is half clear into that array. The memory is NumSharp's, and no array comes back. NumSharp counts the points of each color. | A function that writes into the array of the caller (`PLAN-42`). Text with a format spec as an argument (`PLAN-29`). |
| `CV-3` | Shapes. `cv2.GaussianBlur` 205 K, `cv2.threshold` 202 K, `cv2.findContours` 196 K, `cv2.drawContours` 124 K, `cv2.morphologyEx` 117 K, `cv2.Canny` 95 K | **Count the parts and measure them.** A picture with eight parts of different sizes, which OpenCV draws, and noise from NumSharp. Blur, a threshold that OpenCV chooses (Otsu), remove the small dots, find the outlines, and print the area and the box of the three largest. | A tuple of arrays of different lengths (`PLAN-32`). `sorted` with a function of the tree as its key (`PLAN-15`). Constants that the program adds: `cv2.THRESH_BINARY + cv2.THRESH_OTSU`. |
| `CV-4` | Video files. `cv2.VideoCapture` 711 K, `cv2.VideoWriter` 206 K. Most of these files read a camera. | **Follow a thing through a video.** 30 frames with a square that moves, from NumSharp, go into a video file. The file is read frame by frame until `read()` says that no frame is left. NumSharp finds the place of the square in each frame. | A `while` loop that ends with the result of a call. An object that the program must release: `cap.release()`. An array that can be `None` (`PLAN-32`). |
| `CV-5` | Geometry. `cv2.warpAffine` 114 K, `cv2.warpPerspective` 59 K | **Make a tilted page straight.** A page with a frame, turned by 12 degrees (`getRotationMatrix2D`, `warpAffine`). NumSharp finds the four corners of the frame. `getPerspectiveTransform` and `warpPerspective` make the page straight again. The proof: the difference from the first page. | float32 arrays from NumSharp for a function that takes no other type. A small matrix of the library in NumSharp math. |
| `CV-6` | Detection. `cv2.matchTemplate` 22 K, and `cv2.QRCodeDetector` | **Find a code and read it.** A QR code that OpenCV makes from a text (`QRCodeEncoder`). NumSharp puts the code into a gray page, with a slice whose bounds the program computes. OpenCV turns the page by 17 degrees, finds the code and reads it: the text is the proof. Then the place of one part of the photo in a blurred copy of the photo (`matchTemplate`, `minMaxLoc`). | A tuple of a text and two arrays. A class whose function makes its object: `QRCodeEncoder.create()` (`PLAN-37`). A slice with computed bounds, also on the left of `=` (`PLAN-11`). A tuple inside a tuple as a result. |

Two features with high counts have no pair:

- **A window and a camera** (`cv2.imshow` 748 K, `cv2.waitKey` 709 K): they need a person (`PAIR-5`).
- **The face detector of OpenCV 4** (`detectMultiScale` 116 K, `cv2.CascadeClassifier` 94 K): OpenCV 5.0
  does not have it in its base package (`C-21`). The last package of OpenCV 4, version 4.14.0.94, has
  it. The owner can pin that version in place of 5.0, and then `CV-6` can be "Where is the face?".

### 13.8 TensorFlow, with Keras

TensorFlow ships no data that it does not download. The pairs use the handwritten digits of
scikit-learn (`load_digits`). The names `Sequential`, `model.fit`, `model.predict`, `model.save`,
`load_model`, `EarlyStopping` and `Conv2D` have high counts, and programs of other libraries have them
too (`B-2`). This table gives a count only for a name that belongs to TensorFlow or to Keras.

| Pair | Feature, and how many files use it | The program | What it adds to the approach |
|---|---|---|---|
| `TF-1` | Keras models. `keras.layers` 969 K, `tf.keras` 727 K, `model.compile` 598 K, `layers.Dense` 461 K | **Read handwritten digits with a network.** The digits, which NumSharp scales. A network of two layers, ten rounds of training, the loss of the first round and of the last, the score on digits that the network did not see, and the count of right answers (`np.argmax`, NumSharp). | A list of objects of the tree as an argument: the layers (`PLAN-41`). A long module path: `tf.keras.layers.Dense`. A dict of lists as a result (`PLAN-32`). A tuple of one item: `shape=(64,)` (`PLAN-31`). |
| `TF-2` | Tensors and gradients. `tf.constant` 544 K, `tf.nn` 540 K, `tf.Variable` 496 K, `tf.reduce_mean` 366 K, `tf.function` 152 K, `tf.GradientTape` 107 K | **Find a line by gradient descent.** 100 points near a line, from NumSharp. Two variables, a loss, and 200 steps with `GradientTape`. Then the answer of TensorFlow beside the answer of NumSharp (`np.polyfit`). | `with`, when the program uses the object after the block (`PLAN-20`, `PLAN-21`). Operators on objects of the tree: `w * x + b - y` (`PLAN-39`). The values of a tensor as an array. |
| `TF-3` | Saved models: `model.save`, `load_model`, `model.summary`, `EarlyStopping`. No count: other libraries have these names too. | **A model from a file.** The network of `TF-1` learns until its loss does not fall any more (`EarlyStopping`) and goes into a `.keras` file. A second object comes out of the file, prints its table of layers, and gives the same answers (`np.allclose`, NumSharp). | A table that the library prints by itself (`PLAN-47`, `PLAN-48`). Files. A list of objects as a keyword (`PLAN-41`). |
| `TF-4` | Input pipelines. `tf.data` 270 K | **Data in batches.** The digits as a `tf.data.Dataset`: shuffled with a seed, in batches of 32. The first two batches as NumSharp arrays. The network of `TF-1`, trained from the dataset. | A chain of calls, each of which returns a new object. A loop over tuples of tensors (`PLAN-34`). |
| `TF-5` | Networks for pictures. `to_categorical` 213 K, `tf.image` 188 K, and `Conv2D` | **A network that looks at pictures, and tells the program how far it is.** The digits as pictures of 8 × 8 × 1 (`reshape`, NumSharp). A network with a convolution. A C# function that Keras calls after each round with the loss. | A C# function that the library calls with a dict (`PLAN-30`). Tuples as arguments. |

TensorFlow 1 names still have high counts (`tf.Session` 360 K, `tf.placeholder` 299 K). TensorFlow 2
does not have them, and no pair shows them.

### 13.9 The order of the work

1. **Done:** `SK-1`, the most used feature of scikit-learn.
2. **The two most used features of the first three libraries:** `SK-2`, `SP-1`, `SP-2`, `MP-1`, `MP-2`.
   With `SK-1`, these five pairs are the size of the plan of version 2: three libraries, two features
   each.
3. **The two most used features of the three new libraries:** `PIL-1`, `PIL-2`, `CV-1`, `CV-2`, `TF-1`,
   `TF-2`.
4. **The other pairs,** each library in the order of its table.

Against the plan of five pairs that came with version 2:

| Plan with version 2 | This plan | Why |
|---|---|---|
| scikit-learn, model selection, with the pipeline and its grid as Python text | `SK-4`, with the pipeline and the grid as C# literals (`PLAN-3`, `PLAN-4`). `SK-2` comes before it. | No Python text in a port. Regression has the higher count. |
| SciPy, `scipy.optimize` | `SP-3`. `SP-1` and `SP-2` come before it. | `scipy.sparse` has the higher count. `SP-3` is the pair with the C# functions that Python calls: the owner can move it up. |
| SciPy, `scipy.stats` | `SP-1` | The same module, with a question and real data. |
| Matplotlib, a line chart | `MP-1` | The same feature, with data that NumSharp makes and a proof that the program prints. |
| Matplotlib, a histogram and a scatter chart | `MP-2`, and `MP-3` to `MP-5` | `plt.subplots` has a count of 3.22 M: the grid of charts is the feature, and the histogram and the scatter chart are two of its four charts. |

### 13.10 What the plan needs from the project

- **The statements of 14.1** are for nearly each pair: a list, a loop, an operator. The tables above
  name a planned rule only where one pair is the reason for the rule.
- **Folders and branches** (`PORT-1`, `TREE-2`): `SciPy/Python.scipy.cs`,
  `Matplotlib/Python.matplotlib.cs`, `Pillow/Python.PIL.cs`, `OpenCV/Python.cv2.cs` and
  `TensorFlow/Python.tensorflow.cs`, each with its pairs.
- **`requirements.txt`** (`PROJ-2`) gets Matplotlib, Pillow and opencv-python-headless. The headless
  package of OpenCV has no window code, which no pair uses. TensorFlow gets a second file (`PLAN-51`).
- **`Python.builtins`** gets the names of `PLAN-7`, `PLAN-15`, `PLAN-18` and `PLAN-20`, the types `dict`
  and `pyobj` (`PLAN-4`, `PLAN-28`), and the extension methods of `PLAN-17`.
- **The helper `Argument`** (`TREE-10`) gets its new rows: a list, a tuple, a dict, `None`, a function,
  an array of tree objects.
- **The backend** gets `PLAN-47` and `PLAN-48`. Without them the first pair whose library prints a line
  by itself loses that line (`C-13`).
- **`PORT-5`, `PORT-6`, `PORT-8` and `PORT-11` change** when the first pair needs `PLAN-20`, `PLAN-21`,
  `PLAN-8` or `PLAN-23`. Section 14.6 says how.

## 14. The planned rules

The plan of section 13 needs C# forms that sections 3 to 12 do not have. This section states them. A
planned rule is a proposal: the owner did not review it, and the reference implementation does not meet
it.

| State | Meaning |
|---|---|
| tried (C-2) | A scratch port used the form. It printed the same lines as its Python program, and it ended with exit code 0, so nothing stayed shared (`VER-1`, `VER-2`). The number is the round of Appendix C. |
| open | Not tried. |

"Tried" is less than "verified". The tree members of a scratch port have no documentation and no
review, and a scratch port has one program where a pair has a question, data and a proof.

### 14.1 The statements of a port

These rows extend the table of `PORT-10`.

| Id | Python | C# | State |
|---|---|---|---|
| `PLAN-1` | `np.array([1.0, 2.0])`, `np.array([[1, 2], [3, 4]])`, `np.zeros((5, 5))`, `np.zeros((2, 3), dtype=np.uint8)` | The same text, with `dtype:` for `dtype=`. | tried (C-2, C-3, C-5) |
| `PLAN-2` | A list as an argument: `p0=[1.0, 1.0]`, `metrics=["accuracy"]`, `[10, 10, 60, 40]` | `p0: [1.0, 1.0]`, `metrics: ["accuracy"]`, `[10, 10, 60, 40]` | tried (C-1, C-3, C-6) |
| `PLAN-3` | A list of tuples: `[("scale", StandardScaler()), ("clf", SVC())]` | `[("scale", new StandardScaler()), ("clf", new SVC())]` | tried (C-1) |
| `PLAN-4` | A dict for the library: `{"clf__C": [0.1, 1, 10]}` | `new dict { ["clf__C"] = [0.1, 1, 10] }` | tried (C-1) |
| `PLAN-5` | A dict that the program reads: `limits = {"low": 0.25}`, `for name, limit in limits.items():` | `var limits = new Dictionary<string, double> { ["low"] = 0.25 };`, `foreach (var (name, limit) in limits.items())` | tried (C-5) |
| `PLAN-6` | A list that grows: `scores = []`, `scores.append(x)`, `scores[-1]` | `var scores = new List<double>();`, `scores.append(x);`, `scores[^1]` | tried (C-5) |
| `PLAN-7` | `for x in items:`, `for k in range(2, 7):`, `for i, x in enumerate(items):`, `for a, b in zip(names, values):` | `foreach (var x in items)`, `foreach (var k in range(2, 7))`, `foreach (var (i, x) in enumerate(items))`, `foreach (var (a, b) in zip(names, values))` | tried (C-2) |
| `PLAN-8` | The rows of an array: `for row in table:` and then `x, y, w, h = row.tolist()` | `foreach (NDArray row in table)` and then `var (x, y, w, h) = row.ToArray<int>();` | tried (C-5) |
| `PLAN-9` | `n, m = a.shape`, `h, w = img.shape[:2]` | `var (n, m) = a.shape;`, `var (h, w) = img.shape[..2];` | tried (C-2, C-3) |
| `PLAN-10` | A part of a list: `names[:3]`, `names[-2:]`, `names[-1]` | `names[..3]`, `names[^2..]`, `names[^1]` | tried (C-2, C-5): the first form and the last |
| `PLAN-11` | A slice with bounds that the program computes: `a[top:top + h, left:left + w]` | `a[$"{top}:{top + h}, {left}:{left + w}"]` | tried (C-5) |
| `PLAN-12` | `a[None]` | `a[np.newaxis]` | tried (C-5) |
| `PLAN-13` | `a @ b`, `a ** 2`, `x ** y` | `np.matmul(a, b)`, `np.power(a, 2)`, `pow(x, y)` | tried (C-2, C-5) |
| `PLAN-14` | `n / 2`, `n // 2` and `n % 3` with integers | `(double)n / 2`, `n / 2` and `n % 3`. The last two are Python's only when no number is negative. | tried (C-2) |
| `PLAN-15` | `len(x)`, `abs(x)`, `min(a, b)`, `max(a, b)`, `round(x, 2)`, `sorted(items, key=f, reverse=True)` | The same text, from `Python.builtins`, with `key: f, reverse: true`. | tried (C-2, C-5) |
| `PLAN-16` | A comprehension: `[f(c) for c in items]` | `items.Select(c => f(c)).ToList()` | tried (C-5) |
| `PLAN-17` | A method of one of Python's own types: `scores.append(x)`, `limits.items()`, `name.upper()`, `", ".join(names)`, `data[:8].hex()` | The same text. Each is an extension method with Python's name. | tried (C-5) |
| `PLAN-18` | `None` on purpose: `max_features=None` | `max_features: None` | tried (C-2) |
| `PLAN-19` | A name that is a C# keyword: `out`, `params`, `event` | `@out`, `@params`, `@event` | tried (C-2): `@out` |
| `PLAN-20` | `with obj:` | `using (with(obj)) { }` | tried (C-6). C-1 tried the shorter form `using (f(...)) { }`. |
| `PLAN-21` | A name that Python first assigns inside a block and reads after the block | The port declares the name before the block, with its type and without a value: `Tensor cost;` | tried (C-6) |
| `PLAN-22` | A function that the library calls: `lambda x: math.exp(-x * x)`, or a `def` | A C# lambda: `x => math.exp(-x * x)`, or a local function | tried (C-1, C-2) |
| `PLAN-23` | `try:` and `except cv2.error:` | `try { }` and `catch (PythonException e) when (e.Type.Name == "error") { }` | tried (C-5), with the limits of 14.6 |
| `PLAN-24` | `import math`, `import os`, `import tempfile`, `import io` | `using static Python;`, and then `math.exp(x)`, `os.path.join(a, b)`, `tempfile.gettempdir()`, `new io.BytesIO()` | tried (C-2, C-3, C-5) |
| `PLAN-25` | `os.environ["NAME"] = "value"` | The same text | tried (C-6) |

- **`PLAN-1`, `PLAN-2`.** A C# collection expression (`[1.0, 2.0]`) has no type of its own: it gets the
  type of the parameter. So it is a NumSharp array inside `np.array(...)`, and a Python list inside a
  member of the tree. One difference: `np.array([1, 2])` is an `int32` array in C# and an `int64` array
  in Python, because an integer of C# has 32 bits (`C-12`). The values are the same while each number
  fits in 32 bits.
- **`PLAN-9`, `PLAN-17`.** A C# array does not unpack, and a .NET type does not have Python's methods.
  The project has extension methods for both. They are in a static class that is not nested, because
  C# permits an extension method nowhere else: this class is beside the tree, and not in it.
- **`PLAN-4`, `PLAN-5`.** A dict that only the library reads is a `dict`: its values can be numbers,
  text and lists at the same time. A dict that the program reads is a `Dictionary` of .NET: its values
  have one type, and the program can compute with them.
- **`PLAN-8`.** NumSharp 0.70.0 gives each item of an array as an `object`, so the loop names the type
  `NDArray`, and the row is unpacked in a second statement. `tolist()` in Python and `ToArray<int>()` in
  C# both give plain integers, which a library such as OpenCV needs for a point.
- **`PLAN-11`.** The `$"..."` of a slice is a string of .NET. Its holes are integers, and .NET writes an
  integer as Python does.
- **`PLAN-14` is the most dangerous row.** `n / 2` with two C# integers gives an integer, without an
  error and without a warning. An integer of Python arrives as a `long` (`TREE-6`), so `correct /
  len(y)` is 0 in a port that does not write `(double)correct / len(y)`. With a negative number, `/` and
  `%` of C# are not `//` and `%` of Python (`C-20`).
- **`PLAN-15`.** `round` calls Python. Python rounds the exact binary value of a number, and
  `Math.Round` of .NET does not: `round(2.675, 2)` is 2.67, and `Math.Round(2.675, 2)` is 2.68 (`C-20`).
- **`PLAN-24`.** `System.Math` does the functions of `math`. A library calls a function of the program a
  thousand times (`PLAN-22`), and a call into Python for each `math.exp` in it costs too much. Python
  does `os`, `tempfile` and `io`: the library reads the path, so the path must be Python's path on each
  system.
- **`PLAN-25`.** A variable that .NET sets does not reach the library (`C-15`).

`PLAN-20` and `PLAN-21` together, for a block whose object the program uses after the block:

```python
tape = tf.GradientTape()                                  # Python
with tape:
    cost = tf.reduce_mean((w * x - 1.0) ** 2)
grad = tape.gradient(cost, w)
```

```csharp
var tape = new tf.GradientTape();                         // C#
Tensor cost;
using (with(tape))
{
    cost = tf.reduce_mean((w * x - 1.0).__pow__(2));
}
var grad = tape.gradient(cost, w);
```

### 14.2 The parameters of a member

These rows extend `TREE-5` and `TREE-7`.

| Id | Python | C# | State |
|---|---|---|---|
| `PLAN-26` | One function with two forms: `load_diabetes()` and `load_diabetes(return_X_y=True)`, `plt.subplots()` and `plt.subplots(2, 2)` | One overload for each form, each with its own return type. `TREE-7` stays for a form that only a value selects. | tried (C-2, C-3) |
| `PLAN-27` | A positional parameter that the caller can leave out: `ax.plot(x, y, "r--")`, `img.resize(size, Image.LANCZOS)` | An optional parameter at Python's position, which the member sends by position. One method, and not two overloads (`C-9`). | tried (C-3) |
| `PLAN-28` | A keyword that takes values of several forms: `n_init="auto"` and `n_init=10`; a color | The parameter has the type `pyobj?`. Where a library has one such idea in many places, as a color of Pillow, its branch has a value type with one conversion for each form. | tried (C-2, C-3) |
| `PLAN-29` | A parameter that takes text: `ax.set_title(f"{n} points")` | The parameter has the type `fstring`. A `$"..."` gets Python's format specs there (section 9), and a plain string converts. | tried (C-1, C-3) |
| `PLAN-30` | A parameter that takes a function | A typed delegate where Python fixes the signature: `Func<double, double>` for `quad`. The type `Delegate` where the program fixes it: the model of `curve_fit`. | tried (C-1, C-2) |
| `PLAN-31` | A tuple of one item: `shape=(64,)` | C# has no such literal. The parameter takes the number: `shape: 64`. | tried (C-6) |

`pyobj` and `dict` are types of `Python.builtins`. A number, a text, a bool, `None`, an array, an object
of the tree, and a list of these convert to a `pyobj` without a cast. A C# integer stays an `int` of
Python, and a C# `double` a `float` of Python. A `dict` has text keys and `pyobj` values, and it keeps
the order of its entries, as a dict of Python does.

### 14.3 The results of a member

**`PLAN-32`** extends the table of `TREE-6`.

| Python returns | C# result | State |
|---|---|---|
| Several values with names, which also unpack: `ttest_ind(a, b)` | A `readonly record struct` with Python's names, and a `Deconstruct` for the number of values that Python unpacks | tried (C-1) |
| An object with attributes: an `OptimizeResult`, a `Bunch` | A tree class with one property for each attribute that a program reads (`PLAN-33`) | tried (C-1, C-2) |
| A dict that the program reads by key | `pydict<T>`: a tree object with an indexer. It prints as Python prints the dict. | tried (C-1) |
| A dict of lists of numbers: `history.history` | `Dictionary<string, double[]>`, a copy | tried (C-6) |
| A list of text, or a NumPy array of text | `string[]`, a copy | tried (C-2): the list |
| A tuple of integers that the program indexes and unpacks: `img.size`, `A.shape` | `long[]`, which is the type of `NDArray.shape` | tried (C-2, C-3) |
| A list of numbers: `model.evaluate(X, y)` | `double[]` | tried (C-6) |
| `bytes` | `byte[]` | tried (C-5) |
| A tuple of arrays of different lengths: the contours of OpenCV | `NDArray[]`. Each item is a view. | tried (C-3) |
| A buffer that is not a NumPy array: `canvas.buffer_rgba()` | `NDArray`, a view | tried (C-3) |
| A tensor, or a NumPy scalar | `NDArray`, through `numpy.asarray`. One number is an array of no dimensions. | tried (C-6) |
| A result that no program uses: the lines that `ax.plot` returns | `void` | tried (C-3) |
| A result that can be `None` | A type that can be `null` | open |
| An integer that can be larger than 64 bits: `comb(100, 50, exact=True)` | `BigInteger` | open |

### 14.4 The objects of the tree

These rows extend `TREE-8`.

| Id | Python | C# | State |
|---|---|---|---|
| `PLAN-33` | An attribute: `model.coef_`, `res.x`, `A.nnz` | A property with Python's name, which the program can only read. It asks Python at each read. An array comes back as a view. | tried (C-1, C-2) |
| `PLAN-34` | A Python iterator: `for train, test in kfold.split(X):` | A C# iterator method. It takes the GIL for each step, and it releases the Python iterator in a `finally` block. | tried (C-1) |
| `PLAN-35` | An object that the program calls: `f = interp1d(x, y)` and then `f(x_new)` | The member returns a delegate, so `f(x_new)` reads as in Python. The scope releases the Python object behind it. An object that also has members gets the method `__call__`. | tried (C-2): the delegate |
| `PLAN-36` | An object or a constant at module level: `norm.ppf(0.975)`, `cv2.COLOR_BGR2GRAY`, `Image.LANCZOS` | An object: a static class with its methods. A constant: a static property that reads the value from Python, so the tree does not state it a second time. | tried (C-2, C-3) |
| `PLAN-37` | A module and its main class with one name: `PIL.Image` and `PIL.Image.Image` | One C# class. The functions of the module are its static methods (`Image.open(path)`), and the methods of the class are its instance methods (`img.resize(size)`). C# cannot nest a class in a class of the same name. | tried (C-3) |
| `PLAN-38` | An object that NumPy converts: `np.asarray(img)` | The tree class has an implicit conversion to `NDArray`. Then `np.asarray(img)` is a call of NumSharp, and the line reads as in Python. | tried (C-3) |
| `PLAN-39` | An operator or an index on an object: `A + A.T`, `A * 2.0`, `B[0, 2]`, `w * x - 1.0` | An operator or an indexer of the tree class, which calls Python's method (`__add__`, `__mul__`, `__getitem__`). C# has no operator for `**` and for `@`: the class has the methods `__pow__` and `dot`. | tried (C-2, C-6) |
| `PLAN-40` | One name for objects of different classes: `for name, model in models:` and then `model.fit(X, y)` | An interface of the tree with the methods that the program calls on that name. Each class also keeps its own `fit`, which returns its own type, so that `new LinearRegression().fit(X, y).coef_` compiles. The port names the interface one time: `(string, Regressor)[] models = [...]`. | tried (C-2) |
| `PLAN-41` | A list of objects as an argument: the layers of a Keras model. A NumPy array of objects as a result: `axs[0, 1]`. | An array of tree objects is a list of Python. A tree class with an indexer stands for the array of objects. | tried (C-3, C-6) |
| `PLAN-42` | A function that writes into an array of the caller: `gaussian_filter(a, 1.0, output=out)`, `cv2.rectangle(img, ...)` | An `NDArray` parameter, and nothing more. Python writes into the memory of NumSharp (`ARR-2`), and the port reads the new values. | tried (C-2, C-3) |
| `PLAN-43` | An object whose class Python decides when the program runs: `joblib.load(path)` | A generic member: `joblib.load<Pipeline>(path)`. It checks the class of the object, and it refuses another class. | open |
| `PLAN-44` | Python text inside the tree | Permitted for glue that C# cannot write: a closure around a C# function, or a class that derives from a class of Python. Never in a port, and never for a call of a library. | tried (C-1, C-4, C-6) |

### 14.5 Arrays, print, the backend and lifetime

| Id | Planned rule | Why | State |
|---|---|---|---|
| `PLAN-45` | A member that takes a list of Python has an overload for a C# array `T[]`. | A C# array converts to an `NDArray` by itself. Without the overload, `len(names)` is the compiler error CS0121 (`C-8`). | tried (C-2) |
| `PLAN-46` | `print` and the f-string write three more types as Python does. | A dtype, which is a `System.Type` in NumSharp 0.70.0, with NumPy's name: `uint8`. An `int[]` or a `long[]` as a tuple: `(300, 400, 4)`. A C# list as a list of Python: `[6.0, 100.0]`. | tried (C-3, C-5) |
| `PLAN-47` | The backend sends `sys.stdout` and `sys.stderr` of Python to the console of .NET. | A line that a library prints by itself (`model.summary()`, `verbose=1`) stays in its place between the lines of the port. Without the rule, the line is lost when the output goes to a file or to a pipe (`C-13`). | tried (C-4, C-6): `sys.stdout` |
| `PLAN-48` | The program sets the console of .NET to UTF-8. | The table of `model.summary()` has box characters (`C-14`). | tried (C-6) |
| `PLAN-49` | A C# function that Python calls obeys three lifetime rules. (1) Its result goes back through a list of Python, and not as a return value. (2) Each call has its own `NDScope`, without the collector step of `LIFE-5`. (3) A .NET exception does not enter Python: the wrapper keeps it, Python gets a `RuntimeError`, and `Call` throws the kept exception again. | `C-7` has the measurement behind each of the three. | tried (C-1) |
| `PLAN-50` | `with(obj)` calls `__enter__` at once, and `__exit__` in its `Dispose`. The Python object stays with its tree object. | The program can use the object after the block, as in Python. | tried (C-6) |
| `PLAN-51` | A large library has its own requirements file. `dotnet run` and `verify.py` run each pair whose library is installed, and they name each pair that they left out. | TensorFlow (13.2). | open |
| `PLAN-52` | A loop that makes large arrays in each round | All of them live until the scope closes (`LIFE-2`). An inner scope for each round frees them earlier, and it adds a line that the Python program does not have. | open |

### 14.6 Open questions

1. **TensorFlow: in the plan or not** (13.2).
2. **An error by its type.** Three facts (`C-18`):
   - The filter of `PLAN-23` compares one name. Python's `except ValueError` also catches each class
     that derives from `ValueError`, and scikit-learn raises such a class for a wrong parameter.
   - A port must not name `PythonException` (`PORT-11`). Inside the namespace of the project the name
     `Python.Runtime` does not resolve: the scratch port wrote
     `global::Python.Runtime.PythonException`.
   - NumSharp 0.70.0 has public exception classes with Python's names: `ValueError`, `TypeError`,
     `IndexError`, `KeyError`, `RuntimeError`, `AttributeError`.

   Proposal: `Call` throws the NumSharp class of the nearest base class of the Python error, with the
   `PythonException` inside it. Then `except ValueError:` is `catch (ValueError e)`, for a line of the
   library and for a NumPy line. This changes `ERR-1`. It was not tried.
3. **`with ... as name:` when `__enter__` returns another value.** `with tempfile.TemporaryDirectory()
   as folder:` gives the path of the folder as a text. `SP-8` needs a form for it.
4. **A value that the program sets on a Python object:** an attribute (`layer.trainable = False`) or an
   item (`plt.rcParams["font.size"] = 11`). Proposal: a property that the program can set, and a static
   property with an indexer, as `os.environ` of `PLAN-25`.
5. **Integers.** Python has one integer type. The tree returns a `long` (`TREE-6`), the shape of
   NumSharp is a `long[]`, and a C# literal is an `int`. Open: which type each builtin takes
   (`range(len(x))`), and where a port needs a cast.
6. **A name with a type.** Three planned forms name a type where `PORT-8` asks for `var`: the list of
   `PLAN-40`, the loop of `PLAN-8`, and the declaration of `PLAN-21`. C# needs the type in each.
7. **A keyword by position.** C# lets a caller pass an optional parameter by position. Python refuses
   that for a parameter that is keyword-only. A port passes each keyword by name (`PORT-10`), and
   nothing checks it.
8. **Other systems.** The rounds `C-1` to `C-5` ran on Windows 11 x64 and on Ubuntu 24.04 x64, with the
   same lines on both (`C-22`). The round with TensorFlow ran on Windows only, and nothing ran on macOS.
   The font of a chart and the build of a library can differ between systems, so a pair of Matplotlib,
   Pillow or OpenCV can print other numbers on another system. A pair compares its two programs on one
   machine (`VER-1`), so this does not break a pair.
9. **Unchanged from version 2:** `await` and a scope that crosses threads are not supported (`LIFE-6`).
   A tree as a library that several projects share is open (`TREE-13`).

**What changes in sections 3 to 12 when a planned rule becomes a rule:**

| Rule of today | Change |
|---|---|
| `PORT-5`: a port has no other `using` statement | A `with` block is a `using (with(obj)) { }` block (`PLAN-20`). |
| `PORT-6`: one C# statement for each Python statement | One more line where C# needs a declaration before a block (`PLAN-21`). |
| `PORT-8`: a new name is declared with `var` | Not where C# needs the type (question 6). |
| `PORT-10`: the table of statements | The rows of 14.1. |
| `PORT-11`: no `PythonException` in a port | The answer to question 2. |
| `TREE-5`: a positional parameter is required | Not where Python lets the caller leave it out (`PLAN-27`). |
| `TREE-6`: the table of types | The rows of `PLAN-28` to `PLAN-32`. |
| `TREE-7`: a member has one return type | One return type for each overload (`PLAN-26`). |
| `TREE-8`: a Python class is a sealed C# class | It can implement an interface (`PLAN-40`), and it can have the functions of its module as static methods (`PLAN-37`). |
| `TREE-10`: the argument types of `Call` | A list, a tuple, a `dict`, a `pyobj`, `None`, a function, an array of tree objects. |
| `PRINT-3`: the table of holes | The rows of `PLAN-46`. |
| `BACK-1` to `BACK-6` | `PLAN-47`, `PLAN-48`. |

## 15. Checklists

**To add a member:**

1. Add it to its module class or its tree class, with Python's name and Python's parameter names
   (`TREE-4`, `TREE-5`).
2. Write the three steps (`TREE-9`). Use `Call`, `Create` and `AsArray` only.
3. When an argument has a new type, add the type to `Argument` (`TREE-10`).
4. Write the documentation (`TREE-12`).

**To add a pair:**

1. Take the pair from section 13. For a pair that is not there, add it first, with the count of its
   feature (`PAIR-2`).
2. Write the Python program by `PAIR-1` to `PAIR-9`. Run it. It prints its result, with a format spec
   for each number that NumPy or NumSharp computes (`PORT-12`).
3. For each construct that sections 3 to 12 do not have, take the planned rule of section 14. Move it
   into its section, as section 14.6 says. Correct it where the pair shows that it must change.
4. Add the members that the pair needs.
5. Write the port, statement by statement (`PORT-3` to `PORT-10`).
6. Add the entry to the table of `Program.cs` and the row to the table of `README.md`.
7. Add each new Python library to `requirements.txt`, with its version.
8. Build: no warning (`VER-3`). Run `verify.py`: the same output (`VER-1`). Run the program: exit code 0
   (`VER-2`).
9. Run it on a second operating system, and state where it ran (`VER-5`).
10. Set the state of the pair in section 13 to "done".

**To try a planned rule before a pair needs it,** make a scratch port as Appendix C describes: a copy
of this project, the new members, a port, and a Python program with the same statements. The form is
"tried" when both print the same lines and the port ends with exit code 0.

**To review a port,** search it for the things of `PORT-11`, and compare it with the Python program
line by line.

## Appendix A. The measurements behind the rules

Each number is from this folder at the reference commit, with the versions of the table at the top.

**A-1. Python garbage holds the training table** (`LIFE-5`). After `LogisticRegression.fit`, four
objects are unreachable and refer to each other: a tuple, a function, a
`scipy.optimize._differentiable_functions._ScalarHessWrapper` and a bound method. The tuple holds `X`.
`gc.collect(0)` frees them in 0.8 milliseconds. A full collection takes 23 milliseconds. Without the
collection, one export and one import stayed alive after the scope, also after a wait of 5 seconds.

**A-2. Python memory is freed on another thread** (`LIFE-8`). NumSharp.Interop.pythonnet queues the
release of an imported buffer, and a thread of the thread pool releases it under the GIL. Right after a
scope closed, 7 to 9 of its 11 imports were still counted. 300 milliseconds later they were not.

**A-3. A caught Python error held the arguments** (`ERR-2`). After one failed
`train_test_split(X, y[":5"])`, two NumSharp arrays stayed exported: after the scope, after a .NET
collection with finalizers, after one more Python call, and after a full Python collection. Only
`Finalizer.Instance.Collect()` of pythonnet released them. To dispose the `Type`, the `Value` and the
`Traceback` of the exception did not help. `traceback.clear_frames` released them at once, and the
formatted traceback kept its 13 lines.

**A-4. A sum of NumSharp 0.70.0 can differ in the last digit** (`PORT-12`). `np.sum` of the 143
probabilities gives 87.20879206270375, and NumPy gives 87.2087920627038. The numbers that scikit-learn
computes are equal with all their digits: 0.916083916083916, 0.9758909853249477 and 0.9527624954311276.

**A-5. NumSharp 0.70.0 prints an array as NumPy does** (`PRINT-3`). 19 arrays gave the same text as
NumPy's `str`: integers, floats of different sizes, float32, bool, `nan` and `inf`, a 3-D array, an
array of 2,000 elements, a line that wraps, a 0-d array and an empty array.

**A-6. The format path** (`PRINT-3`, `VER-4`). A port of 26 lines printed the same text as its Python
program. It had the specs `.4f`, `>10.3f`, `<10.3f`, `^10.3f`, `*^12.2f`, `+.2f`, `08.3f`, `,`, `_`,
`,.2f`, `.1%`, `.3e`, `E`, `.3g`, `g`, `d`, `5d`, `x`, `X`, `#x`, `b`, `o`, `s`, `.2s` and `>12s`; holes
without a spec for the floats 1.0, 0.00001, 10**15, 10**16, infinity, NaN, -0.0 and 5e-324; integers up to
2**64 - 1; a 1-D array; a 0-d array with a spec and without one; and an estimator. This test ran on
Windows only.

**A-7. The guards work** (`VER-2`, `VER-3`). A port without its scope line got three NDW012 warnings
when it was built. When it ran, the program ended with exit code 2 and the line "A port left memory
shared with Python: 1 NumSharp buffers, 11 Python buffers."

**A-8. The views of the published package** (`ARR-2`, `ARR-3`, `ARR-4`). A reversed array (`y["::-1"]`),
a stepped array (`y["::3"]`) and one column of a table gave Python's results as arguments of
`accuracy_score`, `confusion_matrix` and `roc_auc_score`. `AsNDArray(allowReadonly: true)` gives a
read-only view of a read-only NumPy array, a 0-d view of a 0-d array, and a view of a reversed float32
array. For a complex64 array it throws `NotSupportedException`, and for an object that is not an array
(an `int`, a `str`, a `list`, a `dict`, `None`) too. `copy()` and `NDScope.Detach` take an array out of
its scope.

**A-9. The pair of this folder.** `ScikitLearn/classification.py` and its port print the same seven
lines on Windows 11 and on Ubuntu 24.04. One run takes 2.6 to 2.7 seconds on Windows. The pair did not
run on macOS.

## Appendix B. The measurements of GitHub

Each number is from 2026-10-10. The requests went through the `gh` command line, with its login.

**B-1. The method.**

| Measure | Request | What it counts |
|---|---|---|
| Files | `GET /search/code` with `q="import cv2" language:Python`. The answer has the field `total_count`. | The Python files that hold the phrase. |
| Dependents | The number "Repositories" of the page `github.com/<owner>/<repository>/network/dependents`. | The repositories whose package file names the library. |
| Stars | `GET /repos/<owner>/<repository>`, the field `stargazers_count`. | The people who marked the repository. |
| The pool of candidates | `GET /search/repositories` with `q=topic:data-science language:Python` and `sort=stars`, for ten topics. | The repositories of a topic, the one with the most stars first. |
| A feature | `GET /search/code` with `q="train_test_split" language:Python`. | The Python files that hold the name. |

The code search accepts 10 requests in one minute. The measurement has 38 libraries and 207 names of
the six libraries of the plan.

**B-2. The limits of the counts.**

- The search cuts a phrase at each point and at each bracket, and it does not tell a capital letter from
  a small letter. `plt.plot` is the word `plt` with the word `plot` after it. `Conv2D` also counts
  `Conv2d`, which is the name of PyTorch.
- A name that several libraries have counts for all of them. The lists of `B-5` mark such a name with †.
- Other libraries copy class names of scikit-learn (PySpark and cuML have a `StandardScaler` too), so
  the count of such a name has a share of their files. The count of a module (`sklearn.metrics`) does
  not.
- GitHub searches the files of its index: public repositories, the default branch, files of less than
  384 KB. A fork is there only when it has more stars than its parent. GitHub rounds the total: most
  totals are multiples of 256.
- For a library, the count is of the larger of two phrases, `import x` and `from x`. A file with the
  other phrase only is not counted, so the true number is higher.
- A count is the number of files that hold a name. It does not say how much a program needs the
  feature.
- The dependents number counts repositories with a package file, such as `requirements.txt`. A notebook
  without one is not counted. For a repository with several packages, and for the package of OpenCV,
  the page gives 0 or the number of one small package: the table says "not available" then.

**B-3. The libraries.** The order is the count of files.

| Repository | Import name | Files | Dependents | Stars | In the plan |
|---|---|---:|---:|---:|---|
| numpy/numpy | `numpy` | 47,448,064 | 3,558,413 | 33,050 | No: NumSharp is the NumPy of a port. |
| pandas-dev/pandas | `pandas` | 18,153,472 | 2,785,596 | 49,988 | No: it has its own conversion in the interop package. |
| pytorch/pytorch | `torch` | 17,956,864 | 848,912 | 104,032 | No: it has its own conversion in the interop package. |
| matplotlib/matplotlib | `matplotlib` | 9,879,552 | 1,861,681 | 23,348 | **1** |
| scipy/scipy | `scipy` | 6,717,440 | 1,415,512 | 15,101 | **2** |
| scikit-learn/scikit-learn | `sklearn` | 6,078,464 | 1,345,505 | 67,514 | **3** |
| psf/requests | `requests` | 5,947,392 | 4,241,861 | 54,561 | No: no arrays. |
| python-pillow/Pillow | `PIL` | 3,817,472 | 2,457,170 | 13,886 | **4** |
| opencv/opencv-python | `cv2` | 3,751,936 | not available | 5,420 | **5** |
| tensorflow/tensorflow | `tensorflow` | 2,969,600 | 540,403 | 200,597 | **6** |
| huggingface/transformers | `transformers` | 2,949,120 | 424,540 | 167,003 | No: it runs on PyTorch, and it downloads models. |
| tqdm/tqdm | `tqdm` | 2,654,208 | 1,194,655 | 31,353 | No: no arrays. |
| yaml/pyyaml | `yaml` | 2,019,328 | 1,816,002 | 2,954 | No: no arrays. |
| sympy/sympy | `sympy` | 1,304,576 | 309,194 | 14,995 | No: no arrays. |
| Beautiful Soup | `bs4` | 1,153,024 | 1,182,172 | not available | No: no arrays. |
| networkx/networkx | `networkx` | 995,328 | 452,558 | 17,316 | No: graphs of Python objects. |
| mwaskom/seaborn | `seaborn` | 923,648 | 706,653 | 14,059 | No: charts on top of Matplotlib. |
| plotly/plotly.py | `plotly` | 845,824 | 468,876 | 18,828 | No: its result is a web page. |
| keras-team/keras | `keras` | 796,672 | 306,025 | 64,360 | With TensorFlow. |
| apache/arrow | `pyarrow` | 724,992 | not available | 17,239 | No |
| jax-ml/jax | `jax` | 618,496 | 47,829 | 36,403 | No |
| nltk/nltk | `nltk` | 610,304 | 418,255 | 14,736 | No: text. |
| h5py/h5py | `h5py` | 456,704 | 355,435 | 2,256 | No |
| scikit-image/scikit-image | `skimage` | 380,928 | 241,624 | 6,608 | The first replacement. |
| pola-rs/polars | `polars` | 315,392 | 31,904 | 40,086 | No |
| statsmodels/statsmodels | `statsmodels` | 308,736 | 181,158 | 11,682 | The second replacement. |
| shapely/shapely | `shapely` | 270,336 | 119,684 | 4,521 | No |
| numba/numba | `numba` | 268,288 | 171,464 | 11,176 | No |
| librosa/librosa | `librosa` | 194,048 | not available | 8,664 | No |
| dmlc/xgboost | `xgboost` | 166,400 | 170,833 | 28,846 | The third replacement. |
| explosion/spaCy | `spacy` | 150,528 | 143,476 | 33,951 | No: text. |
| dask/dask | `dask` | 139,264 | 77,113 | 13,934 | No |
| microsoft/LightGBM | `lightgbm` | 102,144 | not available | 18,852 | No |
| piskvorky/gensim | `gensim` | 98,176 | 77,642 | 16,500 | No: text. |
| bokeh/bokeh | `bokeh` | 56,960 | 104,472 | 20,455 | No |
| catboost/catboost | `catboost` | 46,848 | not available | 9,137 | No |

The two measures agree on the first four libraries of the plan: Matplotlib, SciPy, scikit-learn and
Pillow have the most files and the most dependents of all libraries that exchange arrays. The stars
do not measure use: TensorFlow has the most stars and is sixth by files.

**B-4. The pool of candidates.** The repository search for ten topics, each with `language:Python` and
the most stars first. The number in brackets is the place of the library in the list of its topic.

| Topic | Repositories | Libraries of the table among the first 40 |
|---|---:|---|
| `machine-learning` | 87,934 | Transformers (1), PyTorch (2), scikit-learn (6), Keras (8), spaCy (21) |
| `deep-learning` | 46,188 | Transformers (1), PyTorch (3), Keras (7), spaCy (24) |
| `computer-vision` | 34,018 | scikit-image (34) |
| `data-science` | 19,284 | scikit-learn (3), Keras (4), pandas (5), spaCy (11), Matplotlib (20), gensim (25), seaborn (30), statsmodels (37) |
| `data-analysis` | 15,195 | scikit-learn (2), pandas (4), statsmodels (15) |
| `data-visualization` | 11,919 | Matplotlib (8), seaborn (11) |
| `image-processing` | 11,086 | Pillow (8), scikit-image (15) |
| `visualization` | 5,857 | Plotly (2) |
| `statistics` | 5,116 | scikit-learn (1), statsmodels (3) |
| `scientific-computing` | 2,232 | SciPy (2) |

The topic search is not the measure of the plan, for two reasons. Most of the first repositories of a
topic are applications and courses, not libraries. And NumPy, TensorFlow and OpenCV are in none of the
ten lists: their repositories have another main language or other topics.

The candidates of `B-3` are the libraries of these lists, and more libraries that the author of the
plan named. The count of files gave their order. A library that nobody named is not in the table: the
table is not a full list of GitHub. Two more candidates were counted after the table was made, to test
it: torchvision (581,632 files, a part of the PyTorch family) and Astropy (314,880 files). Neither is
near the sixth place.

**B-5. The features.** The count of each name, the highest first. A name with † is also in programs of
other libraries. A name with ‡ has the name of a variable in it (`ax`, `img`, `model`), so the true
number is higher. The search for these counts was slower than its limit says: after about 150
requests, GitHub answered about one request in one minute.

**scikit-learn** (45 names): `sklearn.metrics` 2,490,368; `sklearn.model_selection` 1,626,112;
`train_test_split` 1,390,592; `sklearn.preprocessing` 1,380,352; `sklearn.datasets` 1,122,304;
`sklearn.linear_model` 1,073,152; `StandardScaler` 923,648; `sklearn.ensemble` 835,584;
`confusion_matrix` 733,184; `accuracy_score` 724,992; `mean_squared_error` 665,600; `LogisticRegression`
614,400; `KMeans`† 556,032; `sklearn.pipeline` 551,936; `LinearRegression` 516,096; `joblib.load`
497,664; `r2_score` 496,128; `RandomForestClassifier` 482,304; `roc_auc_score` 472,064;
`sklearn.cluster` 463,872; `sklearn.svm` 418,816; `classification_report` 414,720; `sklearn.neighbors`
400,384; `load_iris` 398,336; `sklearn.decomposition` 382,976; `joblib.dump` 366,592;
`sklearn.feature_extraction` 361,472; `LabelEncoder` 336,896; `MinMaxScaler` 304,640; `sklearn.tree`
293,888; `TSNE`† 280,576; `GridSearchCV` 279,552; `DecisionTreeClassifier` 257,536;
`sklearn.decomposition import PCA` 249,856; `cross_val_score` 245,760; `TfidfVectorizer` 241,152;
`RandomForestRegressor` 239,104; `feature_importances_` 226,304; `OneHotEncoder` 223,232;
`KNeighborsClassifier` 202,752; `load_digits` 147,456; `CountVectorizer` 128,512; `silhouette_score`
89,600; `MultinomialNB` 66,560; `load_breast_cancer` 66,432.

**SciPy** (40 names): `scipy.stats` 1,638,400; `scipy.sparse` 1,474,560; `scipy.optimize` 1,308,672;
`scipy.special` 972,800; `scipy.linalg` 837,632; `scipy.signal` 717,824; `csr_matrix` 716,800; `expm`†
712,704; `scipy.spatial` 672,768; `scipy.io` 577,536; `scipy.interpolate` 546,816; `scipy.ndimage`
436,224; `scipy.integrate` 357,376; `cdist`† 303,567; `scipy.fft` 290,816; `logsumexp`† 270,848;
`stats.norm`‡ 248,832; `pearsonr` 228,352; `curve_fit` 214,016; `loadmat` 212,480; `spearmanr` 204,800;
`ConvexHull` 198,144; `interp1d` 192,000; `odeint`† 183,296; `scipy.optimize import minimize` 183,040;
`gaussian_filter` 161,280; `ttest_ind` 117,504; `savemat` 115,200; `cKDTree` 107,392; `spsolve` 101,888;
`mannwhitneyu` 99,840; `CubicSpline` 98,304; `linregress` 95,488; `scipy.cluster` 89,856; `filtfilt`
86,144; `find_peaks` 84,736; `fsolve` 70,144; `solve_ivp` 64,512; `savgol_filter` 57,344;
`chi2_contingency` 45,696.

**Matplotlib** (32 names): `plt.show` 4,120,576; `plt.subplot` 3,891,200; `plt.figure` 3,407,872;
`plt.subplots` 3,219,456; `plt.title` 2,494,464; `plt.savefig` 2,437,120; `plt.plot` 2,269,184;
`plt.xlabel` 2,158,592; `plt.close` 2,154,496; `plt.legend` 1,642,496; `plt.tight_layout` 1,497,088;
`ax.set_title`‡ 1,454,080; `ax.plot`‡ 1,439,744; `plt.rcParams` 661,504; `plt.scatter` 651,264;
`plt.imshow` 632,832; `ax.imshow`‡ 495,616; `plt.colorbar` 442,368; `canvas.draw`† 441,344; `plt.bar`
276,480; `plt.hist` 269,824; `plt.style.use` 241,664; `FuncAnimation` 129,280; `plot_surface` 118,784;
`plt.contourf` 99,584; `plt.fill_between` 96,512; `plt.annotate` 82,944; `plt.errorbar` 75,008;
`plt.imread` 62,080; `plt.pie` 51,072; `buffer_rgba` 50,432; `plt.boxplot` 38,784.

**Pillow** (23 names): `Image.open` 2,613,248; `thumbnail`† 1,081,344; `Image.fromarray` 1,007,616;
`Image.new` 952,320; `ImageDraw` 533,504; `img.save`‡ 510,976; `ImageFont` 364,544; `draw.text`‡
336,384; `img.resize`‡ 274,432; `img.convert`‡ 272,896; `Image.LANCZOS` 209,920; `ImageOps` 166,912;
`ImageTk` 156,672; `ImageFilter` 154,624; `getpixel`† 145,152; `alpha_composite` 105,984;
`Image.BICUBIC` 87,808; `ImageChops` 85,888; `ImageSequence` 75,776; `Image.ANTIALIAS` 64,512;
`ImageEnhance` 64,128; `ImageGrab` 48,640; `Image.merge` 48,000.

**OpenCV** (35 names): `cv2.cvtColor` 1,484,800; `cv2.imread` 1,161,216; `cv2.resize` 813,056;
`cv2.imshow` 747,520; `cv2.imwrite` 727,040; `cv2.VideoCapture` 710,656; `cv2.waitKey` 708,608;
`cv2.putText` 576,512; `cv2.rectangle` 513,024; `cv2.line` 317,952; `cv2.circle` 296,960;
`cv2.VideoWriter` 205,824; `cv2.GaussianBlur` 204,800; `cv2.threshold` 202,240; `cv2.findContours`
196,096; `cv2.imdecode` 156,416; `cv2.contourArea` 132,864; `cv2.imencode` 130,304; `cv2.drawContours`
123,648; `cv2.addWeighted` 122,880; `cv2.morphologyEx` 116,736; `detectMultiScale` 115,712;
`cv2.warpAffine` 113,664; `cv2.dilate` 110,336; `cv2.boundingRect` 104,448; `cv2.Canny` 95,232;
`cv2.CascadeClassifier` 93,824; `cv2.inRange` 92,416; `cv2.bitwise_and` 80,896; `cv2.warpPerspective`
59,136; `cv2.dnn` 49,984; `cv2.filter2D` 42,624; `cv2.equalizeHist` 38,592; `cv2.calcHist` 26,112;
`cv2.matchTemplate` 21,504.

**TensorFlow and Keras** (32 names): `Sequential`† 6,635,520; `model.predict`† 2,342,912; `load_model`†
2,174,976; `Conv2D`† 2,138,112; `model.fit`† 1,609,728; `LSTM`† 1,069,056; `keras.layers` 968,704;
`model.save`† 870,400; `tf.keras` 727,040; `model.compile`‡ 598,016; `tf.constant` 543,744; `tf.nn`
539,648; `tf.train` 513,024; `tf.Variable` 496,128; `layers.Dense`‡ 460,800; `tf.reduce_mean` 366,080;
`tf.Session` 360,448; `EarlyStopping`† 323,584; `tf.keras.layers` 309,248; `tf.placeholder` 299,008;
`tf.data` 269,824; `model.evaluate`† 244,224; `model.summary`† 241,152; `to_categorical` 212,736;
`tf.image` 188,160; `tf.function` 151,552; `tape.gradient`‡ 120,576; `tf.GradientTape` 107,392;
`from_tensor_slices` 105,984; `ImageDataGenerator` 91,648; `tf.keras.Sequential` 82,432; `tf.lite`
52,608.

**B-6. The packages.** From the package index (PyPI) on 2026-10-10, for Python 3.12.

| Package | Version | Note |
|---|---|---|
| `matplotlib` | 3.11.2 | Packages for Windows x64, Linux x64 and macOS arm64. |
| `pillow` | 12.3.0 | The same three systems. 7 MB. |
| `opencv-python-headless` | 5.0.0.93 | The same three systems. 61 MB. It needs NumPy 2. |
| `tensorflow` | 2.21.0 | 351 MB for Windows x64, 573 MB for Linux x64, 223 MB for macOS arm64. It brings Keras 3.15.1. On Windows it uses the processor only. |

## Appendix C. The scratch ports

Six scratch ports tried the forms of section 14 before this version stated them. A scratch port is a
copy of this project with more members in its tree, a port, and a Python program with the same
statements. Each one ran as `VER-1` and `VER-2` say: the two programs print the same lines, and the port
ends with exit code 0. The scratch ports are not in the repository. They ran against the published
packages: NumSharp 0.70.0, NumSharp.Interop.pythonnet 0.70.0 and pythonnet 3.0.5, with NumPy 2.4.2, on
Windows 11 x64 (Python 3.12.12) and on Ubuntu 24.04 x64 in WSL 2 (Python 3.12.14). `C-22` has the
results of Ubuntu.

| Round | Libraries | Lines | What it tried |
|---|---|---:|---|
| `C-1` | scikit-learn 1.9.1, SciPy 1.16.3 | 18 | A list of tuples, a dict, objects as arguments, attributes, a dict that comes back, a Python iterator, `with`, C# functions that `curve_fit` and `minimize` call, a result with names, a text parameter, overloads |
| `C-2` | scikit-learn, SciPy | 27 | One name for two classes, `None`, `range`, `enumerate`, `zip`, `len`, an object that the program calls, an object at module level, a lambda that `quad` calls, operators and an index on a sparse matrix, `output=`, `n, m = X.shape`, the operators of `PLAN-13` and `PLAN-14` |
| `C-3` | Matplotlib 3.11.2, Pillow 12.3.0, opencv-python-headless 5.0.0.93 | 17 | The canvas of a chart as an array, `hist`, a grid of charts, a photo to an array and back, a color, constants, OpenCV that draws into a NumSharp array, the contours |
| `C-4` | none | 4 | A line that Python prints by itself |
| `C-5` | Matplotlib, OpenCV | 20 | Random numbers, a slice with computed bounds, the rows of an array, `@` and `**`, a list that grows, a dict, `sorted`, a comprehension, methods of Python's own types, bytes, an error by its type |
| `C-6` | TensorFlow 2.21.0, Keras 3.15.1 | 17 | A Keras model, the same numbers in both programs, a `with` block, operators on tensors, the table that `model.summary()` prints |

Rounds `C-1`, `C-2`, `C-3`, `C-5` and `C-6` printed the same lines as their Python programs and ended
with exit code 0. Round `C-4` has no Python program: it is the measurement of `C-13`.

**C-7. A C# function that Python calls** (`PLAN-49`). Three findings of round `C-1`:

- The first wrapper gave the result of the C# function to Python as a return value. After `curve_fit`
  and `minimize`, 27 NumSharp arrays stayed exported: pythonnet keeps its own reference to an object
  that a .NET method returns, until the finalizer of .NET runs. With the result in a list of Python that
  the wrapper gets as an argument, none stayed.
- The first wrapper opened a `PythonScope` for each call, so the collector step of `LIFE-5` ran after
  each call, and the round took 15 seconds. With a plain `NDScope` for each call it took 3 seconds.
- A C# function that threw an exception left 1 array exported, until the wrapper kept the exception out
  of Python. The exception then arrives in the port with its own .NET type.

**C-8. A C# array converts to an `NDArray` by itself** (`PLAN-45`). `len(feature_names)` with a
`string[]` did not compile: error CS0121, the call is ambiguous between `len<T>(ICollection<T>)` and
`len(NDArray)`. An overload for `T[]` ended the error.

**C-9. Two overloads for one optional argument take the wrong parameter** (`PLAN-27`). With the
overloads `plot(x, y, string? label = null)` and `plot(x, y, string fmt, string? label = null)`, the
call `plot(x, y, "r--")` compiles and sets `label` to `r--`. The one method
`plot(x, y, string? fmt = null, string? label = null)` sets `fmt`.

**C-10. `condition ? array : number` is an array.** NumSharp converts a number to an array by itself, so
the two arms have the common type `NDArray`. The code of a tree that selects a value by its type casts
each arm to `object`.

**C-11. The random numbers are the same** (`PAIR-6`). After `np.random.seed(0)`, four calls gave the
same numbers in NumPy 2.4.2 and in NumSharp 0.70.0: `np.random.normal(0, 1, 5)` (1.764052, 0.400157 and
three more), `np.random.uniform(0, 1, 3)`, `np.random.randint(0, 100, 4)` (58, 65, 39, 87) and
`np.random.permutation(6)` (5, 2, 3, 4, 1, 0).

**C-12. What NumSharp 0.70.0 has** (`PAIR-9`, `PLAN-8`, `PLAN-11`, `PLAN-46`). By reflection of the
published package, and from the rounds:

- `a.dtype` is a `System.Type`, and `np.uint8` is one too. `a.shape` is a `long[]`.
- An array has an index for a text, for `Slice` values, for objects and for a mask. It has none for the
  range type of C# (`1..3`).
- A `foreach` over an array gives each item as an `object`.
- `np.histogram` is not there. These are there: `np.argmax`, `np.argsort`, `np.array_equal`,
  `np.allclose`, `np.bincount`, `np.clip`, `np.convolve`, `np.corrcoef`, `np.cumsum`, `np.diag`,
  `np.diff`, `np.expand_dims`, `np.fill_diagonal`, `np.interp`, `np.isin`, `np.matmul`, `np.meshgrid`,
  `np.percentile`, `np.polyfit`, `np.power`, `np.tile`, `np.transpose`, `np.unique`,
  `np.unravel_index`, and the modules `np.linalg`, `np.fft` and `np.random`.
- `255 - a` for an array of `uint8` is an array of `uint8`, as in NumPy.
- `np.array([1, 2])` is an `int32` array, and in NumPy it is an `int64` array. These have the same
  dtype in both: `np.arange(3)` (`int64`), `np.array([1, 2]) / 2` (`float64`), the sum of an `int32`
  array (`int64`), `np.argsort` (`int64`), and `np.random.randint` (`int32`, on Windows).

**C-13. A line that Python prints by itself is lost** (`PLAN-47`). The port of round `C-4` printed one
line, let Python print two lines (`print` and `pprint.pprint`), and printed a fourth line. With the
output in a file or in a pipe, the output had two lines: the first and the fourth. With `sys.stdout` set
to an object that writes to the console of .NET, it had the four lines in their order.

**C-14. The box characters** (`PLAN-48`). The table of `model.summary()` came out in code page 437 on
the C# side and in UTF-8 on the Python side. With `Console.OutputEncoding = Encoding.UTF8` the two
outputs were the same. The Python side of this machine has the variable `PYTHONIOENCODING=utf-8`.

**C-15. A variable that .NET sets does not reach the library** (`PLAN-25`). With
`Environment.SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0")` before the first call, TensorFlow
still wrote its note about oneDNN. With `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` through Python, it
did not.

**C-16. A chart does not keep the arrays of the program.** Without its two `plt.close` lines, round
`C-3` still ended with exit code 0: Matplotlib copies the data of a line. So the lifetime check
(`VER-2`) does not find a chart that a program did not close.

**C-17. TensorFlow in a .NET process** (13.2). TensorFlow 2.21.0 loaded through pythonnet in 2.2
seconds. After `tf.keras.utils.set_random_seed(0)`, both programs printed 2,410 parameters, a loss from
2.2295 to 0.3675, a test loss of 0.3700, an accuracy of 0.9267, and 417 right answers of 450. Nothing
stayed shared. One run of round `C-6` took 6 to 8 seconds.

**C-18. An error by its type** (`PLAN-23`, question 2 of 14.6). For a `cv2.error`, `e.Type.Name` is
`error`. After the caught error, whose call had an array as its argument, the port ended with exit
code 0 (`ERR-2`). The error of scikit-learn for a wrong parameter is an `InvalidParameterError`, and its
base classes are `ValueError` and `TypeError`. By reflection, NumSharp 0.70.0 has these public exception
classes with Python's names: `ValueError`, `TypeError`, `IndexError`, `KeyError`, `RuntimeError`,
`AttributeError`, `NameError`, `AxisError` and `LinAlgError`.

**C-19. Sizes and times.** On Windows, an environment with NumPy, SciPy, scikit-learn, Matplotlib,
Pillow and OpenCV has 445 MB. An environment with NumPy, SciPy, scikit-learn and TensorFlow has 1.9 GB.
Rounds `C-3` and `C-5` took 2 seconds each.

**C-20. Operators and rounding** (`PLAN-14`, `PLAN-15`). Of eight numbers that end with the digit 5,
`Math.Round(x, 2)` of .NET 10 and `round(x, 2)` of Python differ for two: 2.675 (2.68 against 2.67) and
1.115 (1.12 against 1.11). `-7 / 2` is -3 in C#, and `-7 // 2` is -4 in Python.
`-7 % 3` is -1 in C# and 2 in Python.

**C-21. The planned pairs can run** (`PAIR-1`, `PAIR-5`). Two Python scripts made the main calls of 34
planned pairs, with the data and the seeds of section 13 and without a network. The other five pairs
are `SK-1`, which is done, and `MP-4`, `CV-2`, `TF-1` and `TF-2`, whose main calls were in the rounds
`C-3`, `C-5` and `C-6`.

- 33 pairs ran at the first try: 30 pairs without TensorFlow in 10 seconds together, and `TF-3`, `TF-4`
  and `TF-5` in 17 seconds together.
- One pair did not: the first form of `CV-6` was a face detector, and opencv-python-headless 5.0.0.93
  has no `cv2.CascadeClassifier` and no `cv2.HOGDescriptor`, and its package has no detector files.
  The form of `CV-6` in section 13 ran: the QR code was read, also from a page that was turned by 17
  degrees, and the template was found at its place with a score of 0.9902.
- Some results of the scripts: a heart rate of 72.0 for each minute from the peaks of `SP-6`; the peak
  of the epidemic of `SP-11` at day 38, after 104 calls of the function of the program; the best
  setting of `SK-4` with a score of 0.9829; the same counts from the histogram of Pillow and from
  `np.bincount` in `PIL-2`; the same answers from the loaded model of `TF-3`.

These scripts are Python only. They show that the library side of each pair works. They do not show
that the port works: the rounds `C-1` to `C-6` tried the forms that the ports need.

**C-22. The rounds on Ubuntu** (question 8 of 14.6). The rounds `C-1`, `C-2`, `C-3` and `C-5` printed
the same lines as their Python programs on Ubuntu 24.04 too, each with exit code 0, in 1.4 to 2.0
seconds. Each Python program printed the same lines on Ubuntu as on Windows: also the count of the
pixels of the chart, the numbers of Pillow and the contour of OpenCV. Round `C-4` lost the same two
lines without the redirect, and it had the four lines with it. The environment of Ubuntu has 530 MB.
Round `C-6` did not run there: the package of TensorFlow for Linux has 573 MB.
