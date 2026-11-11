# NumSharp Exceptions

NumSharp raises NumPy's errors: an exception named after the Python or NumPy exception NumPy raises (`ValueError`, `TypeError`, `OverflowError`, `AxisError`, `LinAlgError`, ...), with NumPy's message text. Every exception class of Python 3.12 and of NumPy has a NumSharp class of the same name, with Python's constructors and attributes, so code ported from Python raises and catches the same classes.

## Design principles

1. **NumPy's message.** The message is NumPy's text for the same failure, so an error you look up in NumPy's documentation or issue tracker reads the same.
2. **A .NET twin, then Python's hierarchy.** A class directly under Python's `Exception` derives from the .NET exception with the same meaning: `ValueError` is an `ArgumentException`, `ArithmeticError` an `ArithmeticException`, `MemoryError` an `OutOfMemoryException`, `OSError` an `IOException`, `EOFError` an `EndOfStreamException`, `BaseExceptionGroup` an `AggregateException`. A class with no .NET twin derives from `NumSharpException`. Every other class derives from its Python parent: `OverflowError` from `ArithmeticError`, `FileNotFoundError` from `OSError`, `RecursionError` from `RuntimeError`.
3. **One marker.** Every NumSharp exception implements `INumSharpException`, whatever its .NET base, so one filter catches them all.
4. **The same text under every culture.** A number in a message is formatted with the invariant culture, as NumPy's texts are ASCII whatever the locale (see [Culture-independent texts](#culture-independent-texts)).
5. **Python's constructors and attributes.** Every Python-named class takes no argument, a message, or a message and its cause. A bare `new KeyError()` reads Python's `str(KeyError())`, an empty text (the `SyntaxError` family reads `None`, as CPython prints it), so a port of a bare `assert` or `raise NotImplementedError` keeps Python's text. `new KeyError(message, innerException)` is Python's `raise KeyError(message) from e`: the cause stays in `InnerException`. Classes with attributes in Python take Python's arguments and expose the attributes (`OSError.errno`, `SyntaxError.lineno`, `StopIteration.value`, see [Python's attributes](#pythons-attributes)).

---

## Exception hierarchy

```
Exception
+-- NumSharpException
|   +-- TypeError
|   |   +-- IncorrectTypeException      no loop for the requested dtype (NumPy's TypeError text)
|   |   +-- DTypePromotionError         numpy.exceptions.DTypePromotionError
|   |   +-- UFuncTypeError              a ufunc that cannot run for its operands' dtypes
|   |       +-- UFuncNoLoopError        no loop for the dtypes
|   |       |   +-- UFuncBinaryResolutionError
|   |       +-- UFuncCastingError       a cast the casting= rule forbids
|   |           +-- UFuncInputCastingError
|   |           +-- UFuncOutputCastingError
|   +-- IndexError
|   +-- KeyError
|   +-- AttributeError
|   +-- NameError
|   |   +-- UnboundLocalError
|   +-- AssertionError
|   +-- SyntaxError
|   |   +-- IndentationError
|   |       +-- TabError
|   +-- SystemError
|   +-- BufferError
|   +-- ReferenceError
|   +-- RuntimeError
|   |   +-- NotImplementedError         Python's, as NumPy raises it
|   |   +-- RecursionError
|   |   +-- TooHardError                numpy.exceptions.TooHardError
|   +-- ImportError
|   |   +-- ModuleNotFoundError
|   +-- StopIteration
|   +-- StopAsyncIteration
|   +-- LapackError                     numpy.linalg.lapack_lite.LapackError
|   +-- ConverterError                  numpy.lib._iotools.ConverterError
|   |   +-- ConverterLockError
|   +-- KeyboardInterrupt               BaseException-level in Python
|   +-- SystemExit                      BaseException-level in Python
|   +-- GeneratorExit                   BaseException-level in Python
|   +-- IncorrectShapeException         a shape that does not fit (NumPy's ValueError text)
|   +-- IncorrectSizeException          a size that does not fit (NumPy's ValueError text)
+-- ArgumentException
|   +-- ValueError
|   |   +-- UnicodeError
|   |   |   +-- UnicodeEncodeError
|   |   |   +-- UnicodeDecodeError
|   |   |   +-- UnicodeTranslateError
|   |   +-- LinAlgError                 numpy.linalg.LinAlgError
|   +-- LookupError
|   +-- ArgumentOutOfRangeException
|       +-- AxisError                   numpy.exceptions.AxisError
|           +-- AxisOutOfRangeException legacy, no longer raised
+-- ArithmeticException
|   +-- ArithmeticError
|       +-- OverflowError
|       +-- ZeroDivisionError
|       +-- FloatingPointError
+-- OutOfMemoryException
|   +-- MemoryError
|       +-- ArrayMemoryError            NumPy's _ArrayMemoryError
+-- IOException
|   +-- OSError
|   |   +-- FileNotFoundError, FileExistsError, PermissionError, IsADirectoryError, NotADirectoryError,
|   |   |   BlockingIOError, ChildProcessError, InterruptedError, ProcessLookupError, TimeoutError
|   |   +-- ConnectionError
|   |       +-- BrokenPipeError, ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError
|   +-- EndOfStreamException
|       +-- EOFError
+-- AggregateException
|   +-- BaseExceptionGroup
|       +-- ExceptionGroup
+-- NotSupportedException
|   +-- MissingBackendException         an operation that needs a native backend
|   |   +-- OpenBlasMissingBackendException
|   +-- StringComputationNotSupportedException
+-- MAError                             numpy.ma.MAError
    +-- MaskError                       numpy.ma.MaskError
```

`ValueError`, `LookupError` and `AxisError` sit under .NET's argument exceptions, so a .NET `catch (ArgumentException)` also catches them. Python's `IndexError` and `KeyError` derive from `LookupError`; in NumSharp they are `NumSharpException`s, because `LookupError` is an `ArgumentException` here.

The arithmetic, memory, I/O and group classes sit on their .NET twins, so `catch (ArithmeticException)` catches every `OverflowError` and `ZeroDivisionError`, `catch (OutOfMemoryException)` every `MemoryError`, and `catch (IOException)` every `OSError` and `EOFError` (`EOFError` is no `OSError`, as in Python).

Footguns, each the price of Python's hierarchy:

- `OverflowError` is not .NET's `OverflowException`, and `ZeroDivisionError` is not `DivideByZeroException` (they are siblings under `ArithmeticException`): a `catch (OverflowException)` misses NumPy's overflow. NumSharp keeps .NET's `OverflowException` for its own limits only (a managed array past `int.MaxValue` elements, a `Char8` conversion, `decimal` arithmetic), and a `checked` conversion inside .NET still throws it.
- `FileNotFoundError` is not .NET's `FileNotFoundException`, `PermissionError` is not `UnauthorizedAccessException`, and `TimeoutError` is not `TimeoutException`.
- `NotImplementedError` is not .NET's `NotImplementedException`: NumSharp raises `NotImplementedError` where NumPy refuses an input, and keeps `NotImplementedException` / `NotSupportedException` for what NumSharp itself does not implement.
- `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` sit beside `Exception` under Python's `BaseException`, so `except Exception` does not catch them. C# has one root: `catch (Exception)` catches them.
- An MSTest `Assert.ThrowsException<TypeError>` checks the exact type, and a ufunc refusal now throws one of the `UFuncTypeError` subclasses; `Assert.Throws<TypeError>` (or `catch (TypeError)`) accepts them.

Python's `OSError(errno, ...)` and `BaseExceptionGroup(message, excs)` return a subclass chosen by their arguments (`OSError(2, ...)` is a `FileNotFoundError`); a C# constructor cannot, so `new OSError(2, ...)` builds an `OSError`, and `OSError.Create(errno, strerror, ...)` / `BaseExceptionGroup.Create(message, excs)` are Python's calls. The errno numbers are the operating system's own, as CPython's `errno` module has them: the same number can name different subclasses on Windows, Linux and macOS.

---

## What common failures raise

| Failure | NumSharp | NumPy | Message (NumPy's text) |
|---------|----------|-------|------------------------|
| Shapes that do not broadcast (`a + b`) | `IncorrectShapeException` | `ValueError` | `operands could not be broadcast together with shapes (3,4) (3,)` |
| A shape `np.broadcast_to` cannot stretch to | `IncorrectShapeException` | `ValueError` | `operands could not be broadcast together with remapped shapes [original->remapped]: (3,)  and requested shape (4,)` |
| Arrays `np.broadcast_arrays` cannot broadcast together | `IncorrectShapeException` | `ValueError` | `shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).` |
| A reduction's `out=` of the wrong rank (`np.max(a, 0, out)`, `np.ptp`, `np.median`) | `ValueError` | `ValueError` | `output parameter for reduction operation maximum has the wrong number of dimensions: Found 2 but expected 1` |
| A reshape to another size | `IncorrectShapeException` | `ValueError` | `cannot reshape array of size 6 into shape (4,)` |
| An axis out of range | `AxisError` | `AxisError` | `axis 5 is out of bounds for array of dimension 2` |
| An index out of range | `IndexError` | `IndexError` | `index 5 is out of bounds for axis 0 with size 3` |
| A reduction of an empty array | `ValueError` | `ValueError` | `zero-size array to reduction operation maximum which has no identity` |
| A write into a read-only view (a broadcast) | `ValueError` | `ValueError` | `assignment destination is read-only` |
| A dtype with no loop (`np.sqrt` of text) | `TypeError` | `TypeError` | `ufunc 'sqrt' not supported for the input types, ...` |
| A `dtype=` request with no loop | `IncorrectTypeException` | `TypeError` | `No loop matching the specified signature and casting was found for ufunc sqrt` |
| A ufunc with no loop for its operands (`np.add` of text and numbers) | `UFuncNoLoopError` | `UFuncTypeError` (a `TypeError`) | `ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> None` |
| Two text classes added (`S` and `U`) | `UFuncBinaryResolutionError` | `UFuncTypeError` | `ufunc 'add' cannot use operands with types dtype('S1') and dtype('<U1')` |
| A ufunc cast a casting rule refuses (an `out=` the result cannot reach, a `dtype=` loop an input cannot reach, `np.around` of a bool array) | `UFuncOutputCastingError` / `UFuncInputCastingError` | `UFuncTypeError` | `Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'` |
| Two dtypes with no common dtype | `DTypePromotionError` | `DTypePromotionError` | `The DTypes ... do not have a common DType. ...` |
| An out-of-range Python-style integer | `OverflowError` | `OverflowError` | `Python integer 300 out of bounds for int8` |
| A float infinity converted to an integer | `OverflowError` | `OverflowError` | `cannot convert float infinity to integer` |
| Weights that sum to zero (`np.average`) | `ZeroDivisionError` | `ZeroDivisionError` | `Weights sum to zero, can't be normalized` |
| An array the memory cannot hold | `ArrayMemoryError` | `MemoryError` | `Unable to allocate 8.00 GiB for an array with shape (2147483657,) and data type uint32` |
| A `StringDType` cell that cannot be allocated | `MemoryError` | `MemoryError` | `Failed to allocate string in add` |
| A missing file (`np.load`, `np.fromfile`, `np.save` into a missing directory) | `FileNotFoundError` | `FileNotFoundError` | `[Errno 2] No such file or directory: 'x.npy'` |
| A file that cannot be opened (read-only, held, or a directory on Windows) | `PermissionError` | `PermissionError` | `[Errno 13] Permission denied: 'data'` |
| A directory opened as a file (Linux, macOS) | `IsADirectoryError` | `IsADirectoryError` | `[Errno 21] Is a directory: 'data'` |
| `np.load` of an empty file, or past the last array | `EOFError` | `EOFError` | `No data left in file` |
| A missing member of an `.npz` archive (`npz["x"]`) | `KeyError` | `KeyError` | `'x is not a file in the archive'` |
| A text that does not parse as a number | `ValueError` | `ValueError` | `invalid literal for int() with base 10: np.str_('x')` |
| Non-ASCII text cast to `S` | `UnicodeEncodeError` | `UnicodeEncodeError` | `'ascii' codec can't encode character '\xe9' in position 0: ...` |
| An unknown codec name | `LookupError` | `LookupError` | `unknown encoding: nope` |
| A singular matrix | `LinAlgError` | `LinAlgError` | `Singular matrix` |
| `item()` of an array with several elements | `IncorrectSizeException` | `ValueError` | `can only convert an array of size 1 to a Python scalar` |
| A C# numeric cast of a non-0-d array to a scalar | `TypeError` | `TypeError` | `only 0-dimensional arrays can be converted to Python scalars` |
| A `(bool)` cast of an array with several elements, or none | `ValueError` | `ValueError` | `The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` (empty: ``The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.``) |
| An unknown dtype string | `TypeError` | `TypeError` | `data type 'xyz' not understood` |
| A factorisation without a backend (`np.linalg.svd`) | `OpenBlasMissingBackendException` | - | names the `NumSharp.Interop.OpenBLAS` package to reference |
| An `NDExpr.Call` delegate given a text argument inside `np.evaluate` (every other node computes on text or raises NumPy's own error) | `StringComputationNotSupportedException` | - | names the delegate and the text nodes to use |

`AxisError` derives from `ArgumentOutOfRangeException`, so its `Message` ends with .NET's parameter suffix (`(Parameter 'axis')`); the first line is NumPy's text.

The other NumPy-named classes are raised where NumPy raises them: `KeyError` for a key NumPy looks up and does not find (an unknown `flags` name, a `%(name)s` formatting key, a dtype field name, an `einsum_path` path name, an `.npz` member), `AttributeError` for an attribute NumPy refuses (an `.npz` archive's `npz.f.x` for a missing member), `BufferError` when a buffer cannot be exported in the requested form, `OSError` for a file operation NumPy refuses (`tofile` of a `StringDType` array in binary mode), `SystemError` for data no Python object can represent, `SyntaxError` for text that is not a valid Python literal, `TooHardError` when `np.shares_memory`'s bounded search exhausts `max_work`, `NotImplementedError` where NumPy declines an input (a masked array's `tofile`, numpy.ma's two-dimensional functions `compress_rows`, `compress_cols`, `compress_rowcols`, `mask_rowcols` and `notmasked_contiguous` on other ranks, a seedless `SeedSequence` asked for state), and `RuntimeError` / `AssertionError` / `NameError` where NumPy's own code raises them. The classes NumSharp has no situation for (`StopIteration`, `RecursionError`, `KeyboardInterrupt`, `ConverterError`, `LapackError`, the connection errors, the groups, ...) exist so ported code raises and catches them as in Python.

NumSharp also raises .NET's own exceptions where they describe a failure of its own: `ArgumentNullException` for a `null` argument NumPy has no `None` meaning for, `InvalidCastException` for a cast that a casting rule refuses (`Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'`), `ObjectDisposedException` for a disposed array or iterator, `OverflowException` for a limit of .NET itself (an axis longer than a managed array, `decimal` arithmetic out of range), and `OutOfMemoryException` when the runtime itself runs out of memory.

---

## Python's attributes

Each class with attributes in Python takes Python's arguments and exposes them under Python's names, with CPython's text built from them:

| Class | Constructor (Python's arguments) | Attributes | Text |
|-------|----------------------------------|------------|------|
| `OSError` and its subclasses | `(errno, strerror)`, `(errno, strerror, filename)`, `(errno, strerror, filename, winerror)`, `(errno, strerror, filename, winerror, filename2)` | `errno`, `strerror`, `filename`, `filename2`, `winerror` (Windows), `characters_written` (`BlockingIOError`) | `[Errno 2] No such file or directory: 'x.npy'`; on Windows a `winerror` maps to its errno (CPython's `winerror_to_errno`) and reads `[WinError 5] Access is denied: 'f'` |
| `SyntaxError`, `IndentationError`, `TabError` | `(msg, filename, lineno, offset, text, end_lineno = null, end_offset = null)` | `msg`, `filename`, `lineno`, `offset`, `text`, `end_lineno`, `end_offset` | `invalid syntax (c.py, line 3)`, the file's base name at the platform's separator |
| `NameError`, `UnboundLocalError` | `(message, name)` | `name` | the message |
| `AttributeError` | `(message, name, obj)` | `name`, `obj` | the message |
| `StopIteration` | `(value)` | `value` | Python's `str(value)` |
| `SystemExit` | `(code)` | `code` | Python's `str(code)` |
| `UnicodeEncodeError`, `UnicodeDecodeError` | `(encoding, object, start, end, reason)` | `encoding`, `@object`, `start`, `end`, `reason` | `'ascii' codec can't encode character '\xe9' in position 1: ordinal not in range(128)` |
| `UnicodeTranslateError` | `(object, start, end, reason)` | the same, `encoding` always null | `can't translate character '\xe9' in position 1: reason` |
| `ImportError`, `ModuleNotFoundError` | `(message, name, path)` | `msg`, `name`, `path` | the message |
| `BaseExceptionGroup`, `ExceptionGroup` | `(message, exceptions)` | `message`, `exceptions`, and the methods `subgroup`, `split`, `derive` | `message (2 sub-exceptions)` |
| `UFuncNoLoopError` | `(ufunc, nin, dtypes)` | `ufunc`, `dtypes` | `ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> None` |
| `UFuncInputCastingError`, `UFuncOutputCastingError` | `(ufunc, nin or nout, casting, from_, to, i)` | `ufunc`, `casting`, `from_`, `to`, `in_i` / `out_i` | `Cannot cast ufunc 'add' input 0 from dtype('float64') to dtype('int64') with casting rule 'safe'` |
| `ArrayMemoryError` | `(shape, dtype)` | `shape`, `dtype` | `Unable to allocate 7.45 GiB for an array with shape (1000000000,) and data type float64` |

`start` and `end` of the Unicode errors count code points of the text, as Python's str indices do, or bytes of the bytes. A ufunc error's `ufunc` is the ufunc's name (`add`): NumSharp has no ufunc objects, and NumPy's texts only print the name; the ufunc's input or output count, which NumPy reads off the ufunc object, is a constructor argument. NumSharp's own raise sites fill the attributes too, so a `catch (FileNotFoundError e)` reads `e.filename` and a `catch (UFuncOutputCastingError e)` reads `e.to` without parsing the message.

---

## Catching exceptions

Catch the NumPy class you would catch in Python, or a .NET base:

```csharp
try
{
    var total = np.sum(a, axis: 5);
}
catch (AxisError e)                      // NumPy's AxisError (also an ArgumentOutOfRangeException)
{
    Console.WriteLine(e.Message);        // axis 5 is out of bounds for array of dimension 2 ...
}
catch (ValueError)                       // ValueError and its subclasses: UnicodeError, LinAlgError
{
}
catch (UFuncTypeError e)                 // a ufunc refusal: no loop, or a cast the casting rule forbids
{
    Console.WriteLine(e.ufunc);          // add
}
catch (TypeError)                        // TypeError, IncorrectTypeException, DTypePromotionError
{
}
catch (OverflowError)                    // NumPy's OverflowError (an ArithmeticException, not an OverflowException)
{
}
catch (FileNotFoundError e)              // np.load of a missing file (an OSError, an IOException)
{
    Console.WriteLine(e.filename);
}
catch (Exception e) when (e is INumSharpException)
{
    // any other NumSharp error, whatever its .NET base
}
```

---

## Culture-independent texts

.NET formats a number with the thread's current culture unless told otherwise, and many cultures change the text: de-DE writes `1,5`, sv-SE writes the minus sign as U+2212 (`−6`), ar-SA puts U+061C before it, and en-US prints an infinity as `∞`. NumPy's messages never change with the locale, so NumSharp formats every number in an exception message with `CultureInfo.InvariantCulture`:

```csharp
CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("ar-SA");
var a = np.arange(5.0);
var x = a[-6];                                  // IndexError: index -6 is out of bounds for axis 0 with size 5
var s = np.sum(np.ones(new Shape(2, 2)), -3);   // AxisError: axis -3 is out of bounds for array of dimension 2
                                                // (Parameter 'axis') Actual value was -3.
```

The same holds for `np.finfo(...)` and `np.iinfo(...)`, whose `ToString()` is NumPy's repr byte for byte: `finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)`, `iinfo(min=0, max=18446744073709551615, dtype=uint64)`. Array printing, `astype` to text, `savetxt`, `tofile` and the text parsers were culture-independent already.

NumSharp's own build keeps it that way with the analyzer rule **NDW020**: in NumSharp.Core a message that formats a number, an `object` or a type parameter without a format provider is a build error. It checks interpolated strings, `+` concatenations, `ToString()`, `string.Format`, `string.Join`, `StringBuilder` appends and an `ArgumentOutOfRangeException`'s actual value, inside the exception's arguments, the locals they read and the methods they call. The fix is `string.Create(CultureInfo.InvariantCulture, $"...")` for an interpolated string, or `CultureInfo.InvariantCulture` passed to the formatting call. The rule ships in the NumSharp package but is off by default, because your own exception texts are your business; turn it on with `dotnet_diagnostic.NDW020.severity = error` in an `.editorconfig` if you want the same guarantee.

---

## Legacy exception types

- **`AxisOutOfRangeException`** is the old axis exception, with a parenthesized text (`axis (5) is out of bounds for array of dimension (2)`) NumPy never prints. NumSharp no longer raises it: every axis check raises `AxisError`. It derives from `AxisError`, so a `catch (AxisOutOfRangeException)` still compiles but no longer catches anything; catch `AxisError`.
- **`IncorrectShapeException`**, **`IncorrectSizeException`** and **`IncorrectTypeException`** are raised today, with NumPy's message text. NumPy raises a `ValueError` (or a `TypeError`) for the same failures. `IncorrectTypeException` is a `TypeError`, so `catch (TypeError)` catches it; the shape and size exceptions are `NumSharpException`s only.
