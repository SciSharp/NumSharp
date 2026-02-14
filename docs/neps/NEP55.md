# NEP 55 - UTF-8 Variable-Width String DType

**Status:** Final
**NumSharp Impact:** MEDIUM - New string handling in NumPy 2.0

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

## NumSharp Implications

### Current String Support

NumSharp's string handling is limited. Check current status:
- `NPTypeCode.Char` - single characters only
- No `NPTypeCode.String` or variable-length string dtype

### Implementation Options

**Option 1: Object Array Pattern**
```csharp
// Store strings as object references
var arr = np.array(new string[] { "hello", "world" }, dtype: NPTypeCode.Object);
```

**Option 2: New StringDType**
```csharp
// Add new NPTypeCode
enum NPTypeCode {
    // ... existing ...
    String,  // Variable-length UTF-8
}

// Or dedicated class
class StringDType : DType {
    public NullObject NaObject { get; }
    public bool Coerce { get; }
}
```

### Storage Considerations

NumPy's StringDType uses:
- **Small string optimization:** ≤15 bytes stored inline
- **Arena allocation:** Longer strings in heap arena
- **Thread-safe allocator:** Mutex-protected access

For NumSharp, consider:
- Using `string[]` for simplicity
- Or `Span<byte>` with UTF-8 encoding for performance

### Priority

**MEDIUM** - String handling is important but not as critical as numeric operations. Current workaround: use object arrays with strings.

## References

- [NEP 55 Full Text](https://numpy.org/neps/nep-0055-string_dtype.html)
- [NumPy StringDType docs](https://numpy.org/doc/stable/reference/arrays.strings.html)
