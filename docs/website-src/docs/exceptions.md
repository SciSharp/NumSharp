# NumSharp Exceptions

NumSharp raises NumPy's errors: an exception named after the Python or NumPy exception NumPy raises (`ValueError`, `TypeError`, `AxisError`, `LinAlgError`, ...), with NumPy's message text. Where .NET already has the matching exception, NumSharp uses it (`OverflowException` for Python's `OverflowError`, `NotSupportedException` for a dtype or feature NumSharp does not implement).

## Design principles

1. **NumPy's message.** The message is NumPy's text for the same failure, so an error you look up in NumPy's documentation or issue tracker reads the same.
2. **NumPy's class, on a .NET base.** Each class follows Python's hierarchy where .NET allows it, and derives from the .NET exception a .NET caller expects: `ValueError` is an `ArgumentException`, `AxisError` an `ArgumentOutOfRangeException`, `OSError` an `IOException`.
3. **One marker.** Every NumSharp exception implements `INumSharpException`, whatever its .NET base, so one filter catches them all.

---

## Exception hierarchy

```
Exception
+-- NumSharpException
|   +-- TypeError
|   |   +-- IncorrectTypeException      no loop for the requested dtype (NumPy's UFuncTypeError text)
|   |   +-- DTypePromotionError         numpy.exceptions.DTypePromotionError
|   +-- IndexError
|   +-- KeyError
|   +-- AttributeError
|   +-- NameError
|   +-- AssertionError
|   +-- SyntaxError
|   +-- SystemError
|   +-- BufferError
|   +-- RuntimeError
|   |   +-- TooHardError                numpy.exceptions.TooHardError
|   +-- IncorrectShapeException         a shape that does not fit (NumPy's ValueError text)
|   +-- IncorrectSizeException          a size that does not fit (NumPy's ValueError text)
+-- ArgumentException
|   +-- ValueError
|   |   +-- UnicodeError
|   |   |   +-- UnicodeEncodeError
|   |   |   +-- UnicodeDecodeError
|   |   +-- LinAlgError                 numpy.linalg.LinAlgError
|   +-- LookupError
|   +-- ArgumentOutOfRangeException
|       +-- AxisError                   numpy.exceptions.AxisError
|           +-- AxisOutOfRangeException legacy, no longer raised
+-- IOException
|   +-- OSError
+-- NotSupportedException
|   +-- MissingBackendException         an operation that needs a native backend
|   |   +-- OpenBlasMissingBackendException
|   +-- StringComputationNotSupportedException
+-- MAError                             numpy.ma.MAError
    +-- MaskError                       numpy.ma.MaskError
```

`ValueError`, `LookupError` and `AxisError` sit under .NET's argument exceptions, so a .NET `catch (ArgumentException)` also catches them. Python's `IndexError` and `KeyError` derive from `LookupError`; in NumSharp they are `NumSharpException`s, because `LookupError` is an `ArgumentException` here.

---

## What common failures raise

| Failure | NumSharp | NumPy | Message (NumPy's text) |
|---------|----------|-------|------------------------|
| Shapes that do not broadcast (`a + b`) | `IncorrectShapeException` | `ValueError` | `operands could not be broadcast together with shapes (3,4) (3,)` |
| A reshape to another size | `IncorrectShapeException` | `ValueError` | `cannot reshape array of size 6 into shape (4,)` |
| An axis out of range | `AxisError` | `AxisError` | `axis 5 is out of bounds for array of dimension 2` |
| An index out of range | `IndexError` | `IndexError` | `index 5 is out of bounds for axis 0 with size 3` |
| A reduction of an empty array | `ValueError` | `ValueError` | `zero-size array to reduction operation maximum which has no identity` |
| A write into a read-only view (a broadcast) | `ValueError` | `ValueError` | `assignment destination is read-only` |
| A dtype with no loop (`np.sqrt` of text) | `TypeError` | `TypeError` | `ufunc 'sqrt' not supported for the input types, ...` |
| A `dtype=` request with no loop | `IncorrectTypeException` | `TypeError` | `No loop matching the specified signature and casting was found for ufunc sqrt` |
| Two dtypes with no common dtype | `DTypePromotionError` | `DTypePromotionError` | `The DTypes ... do not have a common DType. ...` |
| An out-of-range Python-style integer | `OverflowException` | `OverflowError` | `Python integer 300 out of bounds for int8` |
| A text that does not parse as a number | `ValueError` | `ValueError` | `invalid literal for int() with base 10: np.str_('x')` |
| Non-ASCII text cast to `S` | `UnicodeEncodeError` | `UnicodeEncodeError` | `'ascii' codec can't encode character '\xe9' in position 0: ...` |
| An unknown codec name | `LookupError` | `LookupError` | `unknown encoding: nope` |
| A singular matrix | `LinAlgError` | `LinAlgError` | `Singular matrix` |
| `item()` of an array with several elements | `IncorrectSizeException` | `ValueError` | `can only convert an array of size 1 to a Python scalar` |
| A C# numeric cast of a non-0-d array to a scalar | `TypeError` | `TypeError` | `only 0-dimensional arrays can be converted to Python scalars` |
| A `(bool)` cast of an array with several elements, or none | `ValueError` | `ValueError` | `The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` (empty: ``The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.``) |
| An unknown dtype string | `NotSupportedException` | `TypeError` | `data type 'xyz' not understood` |
| A factorisation without a backend (`np.linalg.svd`) | `OpenBlasMissingBackendException` | - | names the `NumSharp.Interop.OpenBLAS` package to reference |
| An `NDExpr.Call` delegate given a text argument inside `np.evaluate` (every other node computes on text or raises NumPy's own error) | `StringComputationNotSupportedException` | - | names the delegate and the text nodes to use |

`AxisError` derives from `ArgumentOutOfRangeException`, so its `Message` ends with .NET's parameter suffix (`(Parameter 'axis')`); the first line is NumPy's text.

The other NumPy-named classes are raised where NumPy raises them: `KeyError` for a key NumPy looks up and does not find (an unknown `flags` name, a `%(name)s` formatting key, a dtype field name, an `einsum_path` path name), `AttributeError` for an attribute NumPy refuses, `BufferError` when a buffer cannot be exported in the requested form, `OSError` for a file operation NumPy refuses (`tofile` of a `StringDType` array in binary mode), `SystemError` for data no Python object can represent, `SyntaxError` for text that is not a valid Python literal, `TooHardError` when `np.shares_memory`'s bounded search exhausts `max_work`, and `RuntimeError` / `AssertionError` / `NameError` where NumPy's own code raises them. `MAError` and `MaskError` exist so code ported from `numpy.ma` can catch them.

NumSharp also raises .NET's own exceptions where they describe the failure: `ArgumentNullException` for a `null` argument NumPy has no `None` meaning for, `InvalidCastException` for a cast that a casting rule refuses (`Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'`), `ArgumentException` for a ufunc output cast that a casting rule refuses (`Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'`), `ObjectDisposedException` for a disposed array or iterator, and `OutOfMemoryException` for NumPy's `MemoryError`.

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
catch (TypeError)                        // TypeError, IncorrectTypeException, DTypePromotionError
{
}
catch (Exception e) when (e is INumSharpException)
{
    // any other NumSharp error, whatever its .NET base
}
```

---

## Legacy exception types

- **`AxisOutOfRangeException`** is the old axis exception, with a parenthesized text (`axis (5) is out of bounds for array of dimension (2)`) NumPy never prints. NumSharp no longer raises it: every axis check raises `AxisError`. It derives from `AxisError`, so a `catch (AxisOutOfRangeException)` still compiles but no longer catches anything; catch `AxisError`.
- **`IncorrectShapeException`**, **`IncorrectSizeException`** and **`IncorrectTypeException`** are raised today, with NumPy's message text. NumPy raises a `ValueError` (or a `TypeError`) for the same failures. `IncorrectTypeException` is a `TypeError`, so `catch (TypeError)` catches it; the shape and size exceptions are `NumSharpException`s only.
