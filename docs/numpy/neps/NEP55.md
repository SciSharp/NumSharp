# NEP 55 - UTF-8 Variable-Width String DType

**Status:** Final
**NumSharp Impact:** Implemented - `np.dtypes.StringDType` / `np.dtype("T")`, byte for byte with NumPy 2.4.2

## Summary

Adds `StringDType`, a variable-length UTF-8 encoded string dtype for NumPy 2.0, replacing object arrays for string data.

## The Problem with Fixed-Width Strings

| Aspect | `bytes_` (S) | `unicode` (U) | New `StringDType` (T) |
|--------|--------------|---------------|----------------------|
| Encoding | Null-terminated | UCS-4 (32-bit) | UTF-8 |
| Width | Fixed | Fixed | **Variable** |
| Memory | Wastes space | 4x overhead for ASCII | Optimized |
| Max length | Pre-determined | Pre-determined | **Unlimited** |

### Fixed-Width Problems

1. Must determine max string length before array creation
2. Short strings padded to match longest element
3. Memory waste with mixed-length strings

## New StringDType Usage

### Basic Creation
```python
from numpy.dtypes import StringDType
import numpy as np

data = ["short", "this is a very long string"]
arr = np.array(data, dtype=StringDType())

# Or using character code
arr = np.array(data, dtype="T")
```

### Missing Data Support
```python
dt = StringDType(na_object=np.nan)
arr = np.array(["hello", np.nan, "world"], dtype=dt)
np.isnan(arr)  # array([False, True, False])
```

### String Coercion Control
```python
# Default: coerce non-strings
np.array([1, 3.4], dtype=StringDType())
# array(['1', '3.4'], dtype=StringDType())

# Strict: reject non-strings
np.array([1, 3.4], dtype=StringDType(coerce=False))
# ValueError
```

## String Operations: `np.strings` Namespace

New namespace replaces `np.char`:

```python
np.strings.upper(arr)       # Uppercase
np.strings.lower(arr)       # Lowercase
np.strings.str_len(arr)     # String length
np.strings.isalpha(arr)     # Check alphabetic
np.strings.find(arr, sub)   # Find substring
np.strings.replace(arr, old, new)
np.strings.strip(arr)       # Strip whitespace
```

### String Arithmetic
```python
arr + "!"      # Concatenation
arr * 2        # Repetition
arr == "test"  # Comparison
```

## NumSharp Implementation

NumSharp implements `StringDType` as NumPy does, alongside the fixed-width `S` and `U` dtypes:

```csharp
var dt = np.dtypes.StringDType(na_object: double.NaN);
var arr = np.array(new object[] { "hello", double.NaN, "world" }, dt);
np.isnan(arr);              // [False  True False]
np.strings.add(arr, "!");   // ['hello!' nan 'world!']: a NaN-like NA propagates
np.array(new[] { "a", "bb" }, np.dtype("T"));   // the character code
np.dtypes.StringDType(coerce: false);           // refuses non-str values with NumPy's ValueError
```

- **The descriptor.** `np.dtypes.StringDType(na_object, coerce)` and `np.dtype("T")` build NumPy's descriptor
  (kind and char `'T'`, a 16-byte cell); `dtype.na_object` / `dtype.coerce` read its settings. Every
  `StringDType` descriptor is a fresh instance, and equality follows NumPy's `na_object` comparison.
- **Storage, byte for byte.** The text lane `NPTypeCode.String` holds `S`, `U` and `T` arrays; a `T` array holds
  NumPy's 16-byte cells: short strings inside the cell, medium and long ones in a per-buffer arena, and strings
  rewritten longer on the heap (a port of NumPy's `static_string.c`).
- **Computation.** Comparisons, concatenation and repetition, `maximum` / `minimum`, the reductions (`np.sum`
  concatenates), sorting, searching, unique values, set routines and order statistics, and all 46 `np.strings`
  functions compute with NumPy's missing-value rules: a NaN-like NA propagates and sorts last, a string NA is its
  text, a `None`-like NA raises where NumPy raises.
- **Files and Python.** `np.save` / `np.load(allow_pickle: true)` write and read NumPy's pickle of a `StringDType`
  array; `np.loadtxt` / `np.savetxt` take `StringDType` columns; the pythonnet bridge copies a `StringDType` array
  to and from NumPy through NumPy's own `NpyString` API.

The user guide is [Strings, bytes and chararrays](../../website-src/docs/fundamentals/strings-and-bytes.md); the
capability matrix is [`STRING_COMPAT_MATRIX.md`](../STRING_COMPAT_MATRIX.md).

## References

- [NEP 55 Full Text](https://numpy.org/neps/nep-0055-string_dtype.html)
- [NumPy StringDType docs](https://numpy.org/doc/stable/reference/routines.strings.html)
