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
| Version | 2, 2026-10-10 |
| Reference implementation | This folder, commit `ebcc245a` |
| Verified with | NumSharp.Interop.pythonnet 0.70.0, pythonnet 3.0.5, Python 3.12, NumPy 2.4.2, scikit-learn 1.9.1, on Windows 11 x64 and Ubuntu 24.04 x64 |
| Replaces | Version 1 (commit `b10ac7e5`): unbound static members behind full `Python.*` paths, and a `using PyObject` for each object. The owner's review rejected that form. |
| Not yet aligned | The skill file `.claude/skills/pythonic/SKILL.md` still describes version 1. |

A rule has an identifier, for example `TREE-7`. Each rule is a requirement: the reference implementation
meets it, and a review checks it. Appendix A gives the measurement behind a rule where there is one.
Section 13 lists what this version does not specify.

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
yet (section 13).

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
only where `None` is Python's default for that keyword (section 13).

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

## 13. Not specified yet

The first pair needed none of these. Each one gets its rule with the first pair that needs it. The
second column is a proposal, not a rule.

| Construct | Proposal |
|---|---|
| An attribute of an object: `model.coef_`, `search.best_params_` | A read-only property with Python's name. An array comes back as a view, so that a write reaches the object, as in Python. |
| A `dict` or a `list` argument: `param_grid={"C": [0.1, 1, 10]}` | A C# collection, with a new row for it in `Argument`. Open: a C# array of numbers converts to an `NDArray` by itself, and Python must get a `list` there. |
| A `with` block | The tree class is `IDisposable`, and `Dispose` calls `__exit__`. The port writes `using (...) { }`. |
| A C# function that Python calls | A delegate parameter. The tree wraps it, and the wrapper takes the GIL and opens a scope for each call. |
| A text parameter with a format spec: `ax.set_title(f"{a:.2f}")` | The parameter has the type `Python.fstring`, so that the rules of section 9 hold there too. |
| A keyword that is `None` on purpose | Open. `null` means "not sent" today. |
| An operator or an index on a tree object: `axs[0, 1]` | An operator or an indexer of the tree class. |
| A second form of one function (`TREE-7`) | Open. |
| `for`, `if`, a comprehension, a lambda | Plain C#. Open: a comprehension inside an argument. |
| A catch by the Python type: `except ValueError` | `catch (PythonException e) when (e.Type.Name == "ValueError")` today. |
| Python source that runs as text | Not used. The skill of version 1 allowed it for bodies of several statements. |
| `await`, and a scope that crosses threads | Not supported (`LIFE-6`). |
| A tree as a library that several projects share | Open (`TREE-13`). |

## 14. Checklists

**To add a member:**

1. Add it to its module class or its tree class, with Python's name and Python's parameter names
   (`TREE-4`, `TREE-5`).
2. Write the three steps (`TREE-9`). Use `Call`, `Create` and `AsArray` only.
3. When an argument has a new type, add the type to `Argument` (`TREE-10`).
4. Write the documentation (`TREE-12`).

**To add a pair:**

1. Write the Python program. Run it. It prints its result, with a format spec for each number that
   NumPy or NumSharp computes (`PORT-12`).
2. Add the members that it needs.
3. Write the port, statement by statement (`PORT-3` to `PORT-10`).
4. Add the entry to the table of `Program.cs` and the row to the table of `README.md`.
5. Add each new Python library to `requirements.txt`, with its version.
6. Build: no warning (`VER-3`). Run `verify.py`: the same output (`VER-1`). Run the program: exit code 0
   (`VER-2`).
7. Run it on a second operating system, and state where it ran (`VER-5`).

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
