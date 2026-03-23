# NumPy String Types - Complete Reference

> **Source**: NumPy v2.4.2 (`src/numpy/`)
> **Purpose**: Reference for implementing string support in NumSharp

---

## Table of Contents

1. [Overview: Three String Representations](#overview-three-string-representations)
2. [Fixed-Width Byte Strings (bytes_ / 'S')](#1-fixed-width-byte-strings-bytes_--s)
3. [Fixed-Width Unicode Strings (str_ / 'U')](#2-fixed-width-unicode-strings-str_--u)
4. [Variable-Length Strings (StringDType / 'T')](#3-variable-length-strings-stringdtype--t---numpy-20)
5. [Deprecated char Type ('c')](#4-deprecated-char-type-c)
6. [Object dtype with Strings](#5-object-dtype-with-strings)
7. [Type Codes and Constants](#6-type-codes-and-constants)
8. [Memory Layout Details](#7-memory-layout-details)
9. [String Operations API](#8-string-operations-api)
10. [Splitting and Joining (Object Array Results)](#9-splitting-and-joining-object-array-results)
11. [Type Conversion and Casting](#10-type-conversion-and-casting)
12. [Structured Arrays with String Fields](#11-structured-arrays-with-string-fields)
13. [File I/O with Strings](#12-file-io-with-strings)
14. [Sorting and Searching](#13-sorting-and-searching)
15. [Boolean Conversion and Truthiness](#14-boolean-conversion-and-truthiness)
16. [chararray Legacy Class](#15-chararray-legacy-class)
17. [Implementation Internals](#16-implementation-internals)
18. [Edge Cases and Behavior](#17-edge-cases-and-behavior)
19. [Best Practices](#18-best-practices)

---

## Overview: Three String Representations

NumPy provides **three distinct string types** with fundamentally different memory layouts and use cases:

| Type | DType Code | Type Char | C Enum | Description |
|------|------------|-----------|--------|-------------|
| **bytes_** | `NPY_STRING` | `'S'` | 18 | Fixed-width byte strings (ASCII/binary) |
| **str_** | `NPY_UNICODE` | `'U'` | 19 | Fixed-width Unicode strings (UTF-32/UCS-4) |
| **StringDType** | `NPY_VSTRING` | `'T'` | 2056 | Variable-length UTF-8 strings (NumPy 2.0+) |

### Quick Comparison

| Feature | bytes_ ('S') | str_ ('U') | StringDType ('T') |
|---------|-------------|------------|-------------------|
| **Storage per char** | 1 byte | 4 bytes | 1-4 bytes (UTF-8) |
| **Width** | Fixed | Fixed | Variable |
| **Unicode support** | No (ASCII/binary) | Full (UTF-32) | Full (UTF-8) |
| **NA/missing support** | No | No | Yes |
| **Memory efficiency** | Best for ASCII | Worst | Best overall |
| **Min element size** | 1 byte | 4 bytes | 16 bytes (header) |
| **NumPy version** | All | All | 2.0+ |
| **Recommended for** | Binary data, ASCII | Fixed-width Unicode | General text |

---

## 1. Fixed-Width Byte Strings (`bytes_` / `'S'`)

### Overview

The `bytes_` dtype stores fixed-width sequences of bytes. Each element occupies exactly `itemsize` bytes, regardless of actual content length. Shorter strings are null-padded.

### Memory Layout

```
Structure: [byte0][byte1][byte2]...[byteN-1]
           |<-------- itemsize bytes ------->|

Example: dtype='S5' holding "abc"
Memory:  [0x61][0x62][0x63][0x00][0x00]
           'a'   'b'   'c'  null  null
         |<------ 5 bytes total ------>|
```

### Dtype Specification

```python
# All equivalent ways to specify bytes dtype
np.dtype('S')           # Zero-length bytes (itemsize=0)
np.dtype('S5')          # 5-byte fixed string
np.dtype('S10')         # 10-byte fixed string
np.dtype('|S10')        # Explicit byte order (| = not-applicable for bytes)
np.dtype(np.bytes_)     # Using scalar type
np.dtype((np.bytes_, 5)) # Tuple form with size

# Byte order markers (all equivalent for bytes, no byte swapping needed)
np.dtype('|S5')   # Native/not-applicable
np.dtype('=S5')   # Native
np.dtype('<S5')   # Little-endian (same as native for single bytes)
np.dtype('>S5')   # Big-endian (same as native for single bytes)
```

### Creating Arrays

```python
# From Python bytes
arr = np.array([b'hello', b'world'], dtype='S')
# dtype='|S5' (auto-sized to longest)

# From Python strings (encoded as ASCII)
arr = np.array(['hello', 'world'], dtype='S')
# dtype='|S5'

# Fixed size (truncates if too long)
arr = np.array([b'hello', b'world'], dtype='S3')
# array([b'hel', b'wor'], dtype='|S3')

# Explicit size
arr = np.array([b'hi'], dtype='S10')
# array([b'hi'], dtype='|S10')  # Null-padded to 10 bytes
```

### Key Properties

```python
arr = np.array([b'hello'], dtype='S10')

arr.dtype           # dtype('|S10')
arr.dtype.char      # 'S'
arr.dtype.kind      # 'S'
arr.dtype.type      # <class 'numpy.bytes_'>
arr.dtype.itemsize  # 10 (bytes per element)
arr.dtype.name      # 'bytes80' (bits = itemsize * 8)
arr.dtype.str       # '|S10'

# Scalar access
arr[0]              # np.bytes_(b'hello')
type(arr[0])        # <class 'numpy.bytes_'>
bytes(arr[0])       # b'hello' (Python bytes, null-stripped)
```

### Characteristics

- **Encoding**: None assumed - raw bytes. Content interpreted as ASCII when displayed.
- **Character range**: 0x00-0xFF (full byte range)
- **Safe content**: ASCII (0x00-0x7F) for text; any bytes for binary data
- **Null handling**: Stored but may be stripped on scalar access
- **Itemsize**: Exactly equals byte count per element
- **Use cases**: Binary data, ASCII text, fixed-record file formats, C struct compatibility

### Limitations

```python
# Cannot store non-ASCII Unicode directly
arr = np.array(['hello'], dtype='S')  # Works (ASCII)
arr = np.array(['caf\xe9'], dtype='S')  # Works (Latin-1 byte)

# But emoji/CJK require explicit encoding
text = 'hello \U0001F60A'  # Contains emoji
encoded = text.encode('utf-8')  # b'hello \xf0\x9f\x98\x8a'
arr = np.array([encoded], dtype='S')  # Store as raw bytes
```

---

## 2. Fixed-Width Unicode Strings (`str_` / `'U'`)

### Overview

The `str_` dtype stores fixed-width Unicode strings using **UTF-32/UCS-4 encoding**. Each character occupies exactly 4 bytes (one `npy_ucs4` code point), providing O(1) character access but with significant memory overhead.

### Memory Layout

```
Structure: [codepoint0][codepoint1][codepoint2]...[codepointN-1]
           |<------------- itemsize bytes (N * 4) ------------>|

Each codepoint: 4 bytes (32-bit unsigned integer)

Example: dtype='<U3' holding "A\U0001F60AB" (A + emoji + B)
Memory (little-endian):
  [0x41000000][0x0A F6 01 00][0x42000000]
      'A'          U+1F60A        'B'
  |<------------ 12 bytes total ----------->|

Note: Emoji U+1F60A stored as 0x0001F60A in 4 bytes
```

### Dtype Specification

```python
# All ways to specify unicode dtype
np.dtype('U')           # Zero-length unicode (itemsize=0)
np.dtype('U5')          # 5 characters = 20 bytes
np.dtype('U10')         # 10 characters = 40 bytes
np.dtype('<U5')         # Little-endian
np.dtype('>U5')         # Big-endian
np.dtype('=U5')         # Native byte order
np.dtype(np.str_)       # Using scalar type
np.dtype((np.str_, 5))  # Tuple form with character count

# Platform-dependent native order
import sys
if sys.byteorder == 'little':
    np.dtype('U5').str  # '<U5'
else:
    np.dtype('U5').str  # '>U5'
```

### Creating Arrays

```python
# From Python strings
arr = np.array(['hello', 'world'], dtype='U')
# dtype='<U5' (auto-sized to longest)

# With Unicode content
arr = np.array(['hello', 'caf\xe9', '\U0001F60A'], dtype='U')
# dtype='<U5' (emoji is 1 character)

# Fixed size (truncates if too long)
arr = np.array(['hello', 'world'], dtype='U3')
# array(['hel', 'wor'], dtype='<U3')

# Explicit size
arr = np.array(['hi'], dtype='U10')
# array(['hi'], dtype='<U10')  # Null-padded
```

### Key Properties

```python
arr = np.array(['hello \U0001F60A'], dtype='U10')

arr.dtype           # dtype('<U10')
arr.dtype.char      # 'U'
arr.dtype.kind      # 'U'
arr.dtype.type      # <class 'numpy.str_'>
arr.dtype.itemsize  # 40 (bytes = chars * 4)
arr.dtype.name      # 'str320' (bits = itemsize * 8)
arr.dtype.str       # '<U10' or '>U10'

# Character count (not byte count!)
num_chars = arr.dtype.itemsize // 4  # 10

# Scalar access
arr[0]              # np.str_('hello \U0001F60A')
type(arr[0])        # <class 'numpy.str_'>
str(arr[0])         # 'hello \U0001F60A' (Python str)
```

### The `npy_ucs4` Type

Internally, each character is stored as `npy_ucs4`:

```c
// From npy_common.h - platform-dependent definition
typedef unsigned int npy_ucs4;      // Most common (32-bit int)
typedef unsigned long npy_ucs4;     // Some platforms
typedef unsigned short npy_ucs4;    // Rare (16-bit)

// Valid range: 0x00000000 to 0x0010FFFF (Unicode code points)
```

### Byte Order

Unlike bytes, Unicode strings have meaningful byte order:

```python
# Little-endian: 'A' (U+0041) stored as 41 00 00 00
# Big-endian:    'A' (U+0041) stored as 00 00 00 41

arr_le = np.array(['A'], dtype='<U1')
arr_be = np.array(['A'], dtype='>U1')

# View raw bytes
arr_le.view(np.uint8)  # [65, 0, 0, 0]
arr_be.view(np.uint8)  # [0, 0, 0, 65]

# Byte swapping
arr_be.byteswap()  # Swaps in place
arr_le.newbyteorder('>')  # Returns view with swapped interpretation
```

### Characteristics

- **Encoding**: UTF-32/UCS-4 (fixed 4 bytes per code point)
- **Character range**: U+0000 to U+10FFFF (full Unicode)
- **Itemsize**: 4 * character_count
- **Memory**: 4x overhead for ASCII content
- **Access**: O(1) character indexing (unlike UTF-8)
- **Use cases**: Fixed-width text fields, guaranteed character alignment

### Helper Functions

```python
def get_num_chars(arr):
    """Get number of characters per element in string array."""
    if issubclass(arr.dtype.type, np.str_):
        return arr.dtype.itemsize // 4  # Unicode: 4 bytes/char
    return arr.dtype.itemsize  # Bytes: 1 byte/char

def unicode_itemsize(num_chars):
    """Calculate itemsize for given character count."""
    return num_chars * 4
```

---

## 3. Variable-Length Strings (`StringDType` / `'T'`) - NumPy 2.0+

### Overview

`StringDType` is NumPy 2.0's modern string type featuring:
- **Variable-length storage** (no fixed itemsize)
- **UTF-8 encoding** (memory-efficient for ASCII)
- **NA/missing data support** (configurable sentinel)
- **Arena allocation** with small-string optimization

### Dtype Specification

```python
from numpy.dtypes import StringDType

# Basic creation
np.dtype('T')                               # Default StringDType
np.dtypes.StringDType()                     # Constructor form

# With NA support
np.dtypes.StringDType(na_object=None)       # None as NA
np.dtypes.StringDType(na_object=np.nan)     # NaN as NA
np.dtypes.StringDType(na_object=pd.NA)      # pandas NA

# Coercion control
np.dtypes.StringDType(coerce=True)          # Convert non-strings (default)
np.dtypes.StringDType(coerce=False)         # Reject non-strings

# Combined
np.dtypes.StringDType(na_object=np.nan, coerce=False)
```

### Creating Arrays

```python
# Basic creation
arr = np.array(['hello', 'world'], dtype='T')
arr = np.array(['hello', 'world'], dtype=StringDType())

# With NA values
dt = StringDType(na_object=np.nan)
arr = np.array(['hello', np.nan, 'world'], dtype=dt)
arr[1] is np.nan  # True

# From other types (with coerce=True)
arr = np.array([1, 2, 3], dtype=StringDType())  # ['1', '2', '3']
arr = np.array([b'hello'], dtype=StringDType()) # ['hello']

# Without coercion (strict)
dt = StringDType(coerce=False)
np.array([1, 2], dtype=dt)  # Raises ValueError
```

### Key Properties

```python
arr = np.array(['hello'], dtype='T')

arr.dtype           # StringDType()
arr.dtype.char      # 'T'
arr.dtype.kind      # 'T'
arr.dtype.itemsize  # 16 (header size, not string length!)
arr.dtype.name      # 'StringDType128'
arr.dtype.str       # '|T16'

# NA-related properties
dt = StringDType(na_object=np.nan)
dt.na_object        # nan
dt.coerce           # True
hasattr(dt, 'na_object')  # True (False if no NA configured)
```

### Memory Layout (Internal)

The StringDType uses a sophisticated packed structure:

```c
// Per-element: 16 bytes on 64-bit, 8 bytes on 32-bit
union _npy_static_string_u {
    // For long strings (>15 bytes on 64-bit)
    struct _npy_static_vstring_t {
        size_t offset;          // Arena offset or heap pointer
        size_t size_and_flags;  // Size + flag byte in MSB
    } vstring;

    // For short strings (<=15 bytes on 64-bit) - Small String Optimization
    struct _short_string_buffer {
        char buf[15];           // Inline string data (7 on 32-bit)
        unsigned char size_and_flags;  // Size in low 4 bits
    } direct_buffer;
};
```

### Storage Modes

| Mode | Condition | Storage Location | Description |
|------|-----------|------------------|-------------|
| **Short String** | <= 15 bytes (64-bit) | Inline | Data stored directly in array buffer |
| **Short String** | <= 7 bytes (32-bit) | Inline | Data stored directly in array buffer |
| **Medium Arena** | <= 255 bytes, initial | Arena buffer | Contiguous allocation, 1-byte size |
| **Long Arena** | > 255 bytes, initial | Arena buffer | Contiguous allocation, size_t size |
| **Heap** | Mutation exceeds arena | malloc | Independent allocation |

### Flag Bits

```c
#define NPY_STRING_MISSING       0x80  // 1000 0000 - Null/NA value
#define NPY_STRING_INITIALIZED   0x40  // 0100 0000 - Element has been set
#define NPY_STRING_OUTSIDE_ARENA 0x20  // 0010 0000 - Heap-allocated
#define NPY_STRING_LONG          0x10  // 0001 0000 - Arena string >255 bytes

#define NPY_SHORT_STRING_MAX_SIZE (sizeof(npy_static_string) - 1)  // 15 or 7
#define NPY_MEDIUM_STRING_MAX_SIZE 0xFF  // 255
#define NPY_MAX_STRING_SIZE ((1 << 8*(sizeof(size_t)-1)) - 1)  // Very large
```

### Arena Allocator

The arena is a contiguous buffer that grows as needed:

```c
struct npy_string_arena {
    size_t cursor;      // Current write position
    size_t size;        // Total buffer size
    char *buffer;       // The buffer itself
};

struct npy_string_allocator {
    npy_string_malloc_func malloc;
    npy_string_free_func free;
    npy_string_realloc_func realloc;
    npy_string_arena arena;
    PyMutex allocator_lock;  // Thread safety
};
```

### NA/Missing Data Handling

```python
# Different NA sentinels
dt_none = StringDType(na_object=None)
dt_nan = StringDType(na_object=np.nan)
dt_custom = StringDType(na_object="__NA__")

# Creating with NA
arr = np.array(['hello', np.nan, 'world'], dtype=dt_nan)

# Checking for NA
np.isnan(arr)           # [False, True, False] (only for NaN-like NA)
arr[1] is np.nan        # True (identity check)

# NA in operations
# Sorting: NA values sorted to end
# Comparisons: May raise for non-NaN NA (e.g., None)
```

### Null Byte Handling

Unlike legacy strings, StringDType preserves embedded nulls:

```python
arr = np.array(["hello\x00world"], dtype='T')
arr[0]  # 'hello\x00world' (preserved)
len(arr[0])  # 11

# Legacy types strip at first null on access
arr_u = np.array(["hello\x00world"], dtype='U')
# Stored fully, but may behave differently in operations
```

### Thread Safety

StringDType uses mutex locks for thread-safe allocator access:

```c
// Acquiring allocator (locks mutex)
npy_string_allocator *alloc = NpyString_acquire_allocator(descr);

// ... use allocator ...

// Releasing (unlocks mutex)
NpyString_release_allocator(alloc);
```

---

## 4. Deprecated char Type (`'c'`)

### Overview

The `'c'` (NPY_CHAR) type is **deprecated** and will raise an error if used. It was historically used for single-character access but is no longer supported.

```python
# This will raise an error
np.dtype('c')  # DeprecationWarning or error

# NPY_CHAR = 24 in the enum, but deprecated
```

### Historical Usage

```python
# Legacy code that no longer works:
# arr = np.array('abcd', dtype='c')  # Would create array of 4 chars

# Modern equivalent using bytes_:
arr = np.array(list(b'abcd'), dtype='S1')  # Array of 4 single-byte strings

# Or view bytes as individual characters:
arr = np.frombuffer(b'abcd', dtype='S1')
```

### In chararray (Legacy)

```python
# The 'c' dtype can still appear when viewing bytes:
A = np.array('abc1', dtype='c').view(np.char.chararray)
A.shape  # (4,) - one element per character
A.dtype  # dtype('S1')
```

---

## 5. Object dtype with Strings

### Overview

The `object` dtype (`'O'`) stores Python object references, including strings. This provides unlimited flexibility but loses NumPy's memory efficiency and vectorized operations.

### When to Use

| Scenario | Use Object dtype? |
|----------|-------------------|
| Variable-length strings (pre-NumPy 2.0) | Yes |
| Mixed types in array | Yes |
| Need Python string methods directly | Yes |
| Performance-critical operations | No (use 'U' or 'T') |
| Large arrays | No (memory overhead) |

### Creating Object Arrays with Strings

```python
# Automatic object dtype for mixed content
arr = np.array(['hello', 123, None])
arr.dtype  # dtype('O')

# Explicit object dtype
arr = np.array(['hello', 'world'], dtype=object)
arr.dtype  # dtype('O')

# Each element is a Python str object
type(arr[0])  # <class 'str'>
```

### Memory Layout

```python
# Object arrays store pointers, not string data
arr = np.array(['hello', 'world'], dtype=object)

# Each element is 8 bytes (pointer on 64-bit)
# Actual string data is in Python heap
arr.itemsize  # 8 (pointer size)

# Compare to fixed-width:
arr_u = np.array(['hello', 'world'], dtype='U')
arr_u.itemsize  # 20 (5 chars * 4 bytes)
```

### String Operations on Object Arrays

```python
# numpy.strings functions work with object arrays
arr = np.array(['hello', 'world'], dtype=object)

np.strings.upper(arr)      # array(['HELLO', 'WORLD'], dtype=object)
np.strings.str_len(arr)    # array([5, 5])
np.strings.find(arr, 'o')  # array([4, 1])

# But some return object arrays for split operations
np.char.split(arr, 'l')    # Returns object array of lists
```

### Interoperability

```python
# Object dtype works in type promotion with StringDType
arr_obj = np.array(['hello'], dtype=object)
arr_t = np.array(['world'], dtype='T')

# Comparison works (promotes to common type)
arr_obj == arr_t  # Works

# Concatenation requires explicit conversion
np.concatenate([arr_obj.astype('T'), arr_t])
```

---

## 6. Type Codes and Constants

### NPY_TYPES Enum

```c
enum NPY_TYPES {
    NPY_BOOL = 0,
    NPY_BYTE, NPY_UBYTE,           // 1, 2
    NPY_SHORT, NPY_USHORT,         // 3, 4
    NPY_INT, NPY_UINT,             // 5, 6
    NPY_LONG, NPY_ULONG,           // 7, 8
    NPY_LONGLONG, NPY_ULONGLONG,   // 9, 10
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,  // 11, 12, 13
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,  // 14, 15, 16
    NPY_OBJECT = 17,
    NPY_STRING,                    // 18 - bytes_
    NPY_UNICODE,                   // 19 - str_
    NPY_VOID,                      // 20
    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,  // 21, 22, 23
    NPY_CHAR,                      // 24 - Deprecated
    NPY_NTYPES_LEGACY = 24,
    NPY_NOTYPE = 25,
    NPY_USERDEF = 256,
    NPY_VSTRING = 2056,            // StringDType (NumPy 2.0+)
};
```

### NPY_TYPECHAR Enum

```c
enum NPY_TYPECHAR {
    // Numeric types...
    NPY_STRINGLTR = 'S',           // bytes_
    NPY_DEPRECATED_STRINGLTR2 = 'a',  // Legacy alias for 'S'
    NPY_UNICODELTR = 'U',          // str_
    NPY_VSTRINGLTR = 'T',          // StringDType
    NPY_CHARLTR = 'c',             // Deprecated
    // ...
};
```

### Type Check Macros

```c
// Check if type is string-like
#define PyTypeNum_ISSTRING(type) \
    (((type) == NPY_STRING) || ((type) == NPY_UNICODE))

// Check if type is flexible (string, unicode, or void)
#define PyTypeNum_ISFLEXIBLE(type) \
    (((type) >= NPY_STRING) && ((type) <= NPY_VOID))

// Python checks
np.issubdtype(arr.dtype, np.bytes_)      # True for 'S'
np.issubdtype(arr.dtype, np.str_)        # True for 'U'
np.issubdtype(arr.dtype, np.character)   # True for 'S' or 'U'
np.issubdtype(arr.dtype, np.flexible)    # True for 'S', 'U', 'V'
```

### Type Hierarchy

```
numpy.generic
    numpy.flexible
        numpy.character
            numpy.bytes_   (S)
            numpy.str_     (U)
        numpy.void         (V)
    numpy.number
        ...
```

---

## 7. Memory Layout Details

### Itemsize Calculation

```python
# bytes_ (S): itemsize = byte count
np.dtype('S10').itemsize  # 10

# str_ (U): itemsize = character count * 4
np.dtype('U10').itemsize  # 40

# StringDType (T): itemsize = header size (fixed)
np.dtype('T').itemsize    # 16 (64-bit) or 8 (32-bit)
```

### Array Memory Layout

```python
# Contiguous bytes array
arr = np.array([b'abc', b'de', b'fghi'], dtype='S4')
# Memory: [a][b][c][0] [d][e][0][0] [f][g][h][i]
#         |-- 4 -----|  |-- 4 ----|  |-- 4 ---|
# Total: 12 bytes, strides=(4,)

# Contiguous unicode array
arr = np.array(['ab', 'cd'], dtype='U2')
# Memory: [a...][b...] [c...][d...]  (each [...] = 4 bytes)
#         |--- 8 ----|  |--- 8 ---|
# Total: 16 bytes, strides=(8,)

# StringDType array
arr = np.array(['hello', 'world'], dtype='T')
# Array buffer: [header1][header2]  (each header = 16 bytes)
# Arena buffer: [h][e][l][l][o][w][o][r][l][d]  (separate allocation)
```

### Stride Considerations

```python
# All string types support standard striding
arr = np.array([['a', 'b'], ['c', 'd']], dtype='U1')
arr.strides  # (8, 4) - row stride, column stride in bytes

# Slicing creates views (shared memory)
view = arr[::2, :]
view.strides  # (16, 4) - doubled row stride

# StringDType slicing
arr_t = np.array(['hello', 'world', 'test'], dtype='T')
view = arr_t[::2]  # ['hello', 'test']
# Headers are views, but string data is shared via arena
```

---

## 8. String Operations API

### Module Organization

```python
numpy.strings    # Modern API (NumPy 2.0+, preferred)
numpy.char       # Legacy API (backwards compatibility, wraps numpy.strings)
```

### Comparison Operations

All comparisons work with `S`, `U`, and `T` dtypes:

```python
# Element-wise comparison (return bool array)
np.equal(arr1, arr2)           # arr1 == arr2
np.not_equal(arr1, arr2)       # arr1 != arr2
np.less(arr1, arr2)            # arr1 < arr2 (lexicographic)
np.less_equal(arr1, arr2)      # arr1 <= arr2
np.greater(arr1, arr2)         # arr1 > arr2
np.greater_equal(arr1, arr2)   # arr1 >= arr2

# Direct operators also work
arr1 == arr2
arr1 < arr2

# compare_chararrays (low-level)
np.char.compare_chararrays(arr1, arr2, '==', rstrip=True)
# rstrip=True strips trailing whitespace before comparing
```

### Concatenation and Repetition

```python
# String concatenation
np.strings.add(a, b)
# 'hello' + 'world' = 'helloworld'
# Broadcasts: ['a', 'b'] + 'x' = ['ax', 'bx']

# String repetition
np.strings.multiply(a, n)
# 'ab' * 3 = 'ababab'
# ['a', 'b'] * [2, 3] = ['aa', 'bbb']

# Negative/zero repeat = empty string
np.strings.multiply('abc', 0)   # ''
np.strings.multiply('abc', -1)  # ''

# Overflow protection
np.strings.multiply('abc', sys.maxsize)  # Raises OverflowError
```

### Length

```python
np.strings.str_len(arr)
# Returns array of int64 with string lengths

np.strings.str_len(['hello', '', 'ab'])  # [5, 0, 2]
np.strings.str_len(['caf\xe9'])           # [4] (4 chars, not bytes)
np.strings.str_len(['\U0001F60A'])        # [1] (1 emoji = 1 char)
```

### Searching

```python
# Find first occurrence (returns -1 if not found)
np.strings.find(a, sub, start=0, end=None)
np.strings.find('hello world', 'o')        # 4
np.strings.find('hello world', 'o', 5)     # 7
np.strings.find('hello world', 'x')        # -1

# Find last occurrence
np.strings.rfind(a, sub, start=0, end=None)
np.strings.rfind('hello world', 'o')       # 7

# Like find but raises ValueError if not found
np.strings.index(a, sub, start=0, end=None)
np.strings.rindex(a, sub, start=0, end=None)

# Count occurrences
np.strings.count(a, sub, start=0, end=None)
np.strings.count('banana', 'a')            # 3
np.strings.count('banana', 'na')           # 2

# Prefix/suffix check
np.strings.startswith(a, prefix, start=0, end=None)
np.strings.endswith(a, suffix, start=0, end=None)
np.strings.startswith('hello', 'he')       # True
np.strings.endswith('hello', 'lo')         # True
```

### Case Operations

```python
np.strings.upper(arr)       # HELLO
np.strings.lower(arr)       # hello
np.strings.capitalize(arr)  # Hello (first char upper, rest lower)
np.strings.title(arr)       # Hello World (each word capitalized)
np.strings.swapcase(arr)    # hELLO wORLD (swap case)
```

### Character Classification

```python
# All return bool arrays
np.strings.isalpha(arr)     # All alphabetic?
np.strings.isdigit(arr)     # All digits?
np.strings.isalnum(arr)     # All alphanumeric?
np.strings.isspace(arr)     # All whitespace?
np.strings.islower(arr)     # All lowercase (cased chars)?
np.strings.isupper(arr)     # All uppercase (cased chars)?
np.strings.istitle(arr)     # Titlecase format?
np.strings.isnumeric(arr)   # Numeric (Unicode-aware)?
np.strings.isdecimal(arr)   # Decimal digits (Unicode-aware)?

# Empty strings return False for all
np.strings.isalpha('')      # False
np.strings.isspace('')      # False
```

### Whitespace Operations

```python
# Strip whitespace
np.strings.strip(arr, chars=None)   # Both ends
np.strings.lstrip(arr, chars=None)  # Left only
np.strings.rstrip(arr, chars=None)  # Right only

# Custom characters to strip
np.strings.strip('xxhelloxx', 'x')  # 'hello'

# Padding/alignment
np.strings.center(arr, width, fillchar=' ')
np.strings.ljust(arr, width, fillchar=' ')
np.strings.rjust(arr, width, fillchar=' ')
np.strings.zfill(arr, width)        # Numeric zero-fill: '42' -> '0042'

# Tab expansion
np.strings.expandtabs(arr, tabsize=8)
```

### Splitting and Partitioning

```python
# Partition at separator (returns 3 parts)
np.strings.partition(arr, sep)
np.strings.partition('hello-world', '-')
# ('hello', '-', 'world')

np.strings.rpartition(arr, sep)  # From right
np.strings.rpartition('a-b-c', '-')
# ('a-b', '-', 'c')

# Replacement
np.strings.replace(arr, old, new, count=-1)
np.strings.replace('hello', 'l', 'L')      # 'heLLo'
np.strings.replace('hello', 'l', 'L', 1)   # 'heLlo'

# Slicing (substring extraction)
np.strings.slice(arr, start=0, stop=None, step=1)
np.strings.slice('hello', 1, 4)   # 'ell'
np.strings.slice('hello', None, None, -1)  # 'olleh'
```

### Encoding/Decoding (bytes_ only)

```python
# Encode str -> bytes
np.strings.encode(arr, encoding='utf-8', errors='strict')

# Decode bytes -> str
np.strings.decode(arr, encoding='utf-8', errors='strict')

# Error handling modes: 'strict', 'ignore', 'replace', etc.
```

### Printf-style Formatting

```python
np.strings.mod(format_arr, values)
# Like Python's % formatting

np.strings.mod(['%s is %d'], [('answer', 42)])
# ['answer is 42']

np.strings.mod(['Value: %05.2f'], [3.14159])
# ['Value: 03.14']
```

### Character Translation

```python
np.strings.translate(arr, table, deletechars=None)
# Apply character mapping table

# table: str.maketrans result or 256-char string
table = str.maketrans('abc', 'xyz')
np.strings.translate('abcdef', table)  # 'xyzdef'
```

---

## 9. Splitting and Joining (Object Array Results)

### Overview

Split operations (`split`, `rsplit`, `splitlines`) return **object arrays** containing Python lists, not string arrays. This is because the number of splits varies per element.

### split and rsplit

```python
# split - split at separator, returns object array of lists
arr = np.array(['a,b,c', 'd,e'], dtype='U')
result = np.char.split(arr, ',')
# result.dtype = object
# result[0] = ['a', 'b', 'c']
# result[1] = ['d', 'e']

# With maxsplit
result = np.char.split(arr, ',', maxsplit=1)
# result[0] = ['a', 'b,c']

# rsplit - split from right
result = np.char.rsplit(arr, ',', maxsplit=1)
# result[0] = ['a,b', 'c']

# Works with all string dtypes
arr_t = np.array(['a,b,c'], dtype='T')
np.char.split(arr_t, ',')  # Same behavior
```

### splitlines

```python
# Split on line boundaries
arr = np.array(['line1\nline2\nline3'])
result = np.char.splitlines(arr)
# result[0] = ['line1', 'line2', 'line3']

# Keep line endings
result = np.char.splitlines(arr, keepends=True)
# result[0] = ['line1\n', 'line2\n', 'line3']
```

### join

```python
# Join sequences with separator
sep = np.array(['-', '.'])
seq = np.array([['a', 'b', 'c'], ['x', 'y', 'z']], dtype=object)

# Join each sequence with corresponding separator
result = np.char.join(sep, seq)  # Not directly available in np.strings

# For simple cases, use Python:
arr = np.array(['abc', 'def'])
joined = np.char.join(',', arr)  # 'a,b,c' and 'd,e,f'
```

### partition and rpartition

Unlike split, `partition` returns **fixed-size tuple arrays**:

```python
# partition returns 3 parts: (before, sep, after)
arr = np.array(['hello-world', 'foo-bar-baz'])
result = np.strings.partition(arr, '-')
# Returns tuple of 3 arrays:
# (['hello', 'foo'], ['-', '-'], ['world', 'bar-baz'])

# rpartition splits at last occurrence
result = np.strings.rpartition(arr, '-')
# (['hello', 'foo-bar'], ['-', '-'], ['world', 'baz'])
```

### Internal _vec_string Function

Many operations use the internal `_vec_string` function:

```python
from numpy._core.multiarray import _vec_string

# Signature: _vec_string(arr, output_dtype, method_name, args_tuple)
# Calls Python string method element-wise

# Example (internal use only):
_vec_string(['hello', 'world'], np.object_, 'split', (',',))
```

---

## 10. Type Conversion and Casting

### Between String Types

```python
# bytes_ <-> str_
arr_s = np.array([b'hello'], dtype='S')
arr_u = arr_s.astype('U')  # Decode as ASCII

arr_u = np.array(['hello'], dtype='U')
arr_s = arr_u.astype('S')  # Encode as ASCII (fails for non-ASCII)

# str_ <-> StringDType
arr_u = np.array(['hello'], dtype='U')
arr_t = arr_u.astype('T')  # Full Unicode preserved

arr_t = np.array(['hello'], dtype='T')
arr_u = arr_t.astype('U10')  # May truncate to width

# StringDType -> bytes_ (ASCII only)
arr_t = np.array(['hello'], dtype='T')
arr_s = arr_t.astype('S')  # Works for ASCII
```

### Width Truncation

```python
# Truncation when target is smaller
arr = np.array(['hello world'], dtype='U')
arr.astype('U5')   # ['hello'] - truncated

arr.astype('S3')   # [b'hel'] - truncated
```

### Numeric Conversions

```python
# Numbers to strings
np.array([1, 2, 3]).astype('U')     # ['1', '2', '3']
np.array([1.5, 2.5]).astype('U')    # ['1.5', '2.5']
np.array([1+2j]).astype('U')        # ['(1+2j)']

# Special values
np.array([np.nan, np.inf, -np.inf]).astype('U')
# ['nan', 'inf', '-inf']

# Strings to numbers
np.array(['1.5', '2.5']).astype(float)  # [1.5, 2.5]
np.array(['1', '2']).astype(int)        # [1, 2]
```

### Void (V) Conversions

```python
# StringDType <-> Void (raw bytes)
arr_t = np.array(['hello'], dtype='T')
arr_v = arr_t.astype('V5')  # Raw UTF-8 bytes

arr_v = np.array([b'hello'], dtype='V5')
arr_t = arr_v.astype('T')   # Interpret as UTF-8
```

### Casting Safety

```python
# Safe casts (no data loss possible)
arr_s = np.array([b'hi'], dtype='S')
arr_s.astype('U', casting='safe')  # OK

# Unsafe casts (may lose data)
arr_u = np.array(['\u00e9'], dtype='U')  # 'e' with acute
arr_u.astype('S', casting='safe')  # Raises TypeError
arr_u.astype('S', casting='unsafe')  # May corrupt

# StringDType NA casting
dt_na = StringDType(na_object=np.nan)
dt_no_na = StringDType()
arr_na = np.array(['hi', np.nan], dtype=dt_na)
arr_na.astype(dt_no_na, casting='safe')  # Raises (NA -> string unsafe)
```

### Datetime/Timedelta Conversions

```python
# datetime64 -> StringDType
dt_arr = np.array(['2023-01-15T10:30:00'], dtype='datetime64[s]')
str_arr = dt_arr.astype('T')
# ['2023-01-15T10:30:00']

# StringDType -> datetime64
str_arr = np.array(['2023-01-15'], dtype='T')
dt_arr = str_arr.astype('datetime64[D]')

# NaT handling
dt = StringDType(na_object=np.nan)
arr = np.array(['2023-01-15', np.nan], dtype=dt)
dt_arr = arr.astype('datetime64[D]')
# arr[1] becomes NaT

# 'nat', 'NAT', 'NaT', etc. all parse as NaT
arr = np.array(['NaT', 'nat', 'NAT', ''], dtype='T')
arr.astype('datetime64[s]')  # All become NaT

# timedelta64 similar behavior
td_arr = np.array([12358], dtype='timedelta64[s]')
str_arr = td_arr.astype('T')  # ['12358'] (seconds as string)
```

---

## 11. Structured Arrays with String Fields

### Creating Structured Arrays

```python
# Structured dtype with string fields
dt = np.dtype([('name', 'U20'), ('id', 'i4'), ('code', 'S5')])
arr = np.array([('Alice', 1, b'ABC'), ('Bob', 2, b'DEF')], dtype=dt)

# Access fields
arr['name']  # array(['Alice', 'Bob'], dtype='<U20')
arr['code']  # array([b'ABC', b'DEF'], dtype='|S5')

# Nested structured types
dt = np.dtype([('person', [('first', 'U10'), ('last', 'U10')]), ('age', 'i4')])
```

### StringDType in Structured Arrays

```python
# StringDType can be used in structured arrays (NumPy 2.0+)
from numpy.dtypes import StringDType

dt = np.dtype([('name', StringDType()), ('value', 'f8')])
arr = np.array([('hello', 1.5), ('world', 2.5)], dtype=dt)

# Variable-length strings in structured array
arr['name'][0] = 'a very long string that exceeds any fixed width'
```

### Record Arrays

```python
# recarray provides attribute access
rec = np.rec.array([('Alice', 25), ('Bob', 30)],
                   dtype=[('name', 'U10'), ('age', 'i4')])
rec.name   # array(['Alice', 'Bob'], dtype='<U10')
rec.age    # array([25, 30])
rec[0].name  # 'Alice'
```

### Memory Layout

```python
# Fixed-width strings have predictable offsets
dt = np.dtype([('a', 'U5'), ('b', 'i4'), ('c', 'S3')])
dt.itemsize   # 20 + 4 + 3 = 27 (with potential padding)
dt.fields     # {'a': (dtype('<U5'), 0), 'b': (dtype('int32'), 20), ...}

# StringDType has fixed header size
dt = np.dtype([('name', 'T'), ('value', 'f8')])
dt.itemsize   # 16 + 8 = 24
```

---

## 12. File I/O with Strings

### NPY Format (np.save/np.load)

```python
# Saving string arrays
arr = np.array(['hello', 'world'], dtype='U')
np.save('strings.npy', arr)

# Loading
loaded = np.load('strings.npy')
# dtype preserved

# StringDType (NumPy 2.0+)
arr_t = np.array(['hello', 'world'], dtype='T')
np.save('strings_t.npy', arr_t)
loaded = np.load('strings_t.npy')
# StringDType preserved (requires NumPy 2.0+ to load)
```

### NPZ Archives

```python
# Multiple arrays
arr1 = np.array(['a', 'b'], dtype='U')
arr2 = np.array([b'x', b'y'], dtype='S')

np.savez('strings.npz', unicode=arr1, bytes=arr2)

with np.load('strings.npz') as data:
    print(data['unicode'])  # ['a', 'b']
    print(data['bytes'])    # [b'x', b'y']
```

### Text Files (loadtxt/savetxt)

```python
# Load text as strings
arr = np.loadtxt('data.txt', dtype='U', delimiter=',')

# Save strings to text
arr = np.array(['hello', 'world'])
np.savetxt('out.txt', arr, fmt='%s')

# Structured with strings
dt = np.dtype([('name', 'U10'), ('value', 'f8')])
data = np.loadtxt('data.csv', dtype=dt, delimiter=',')
```

### genfromtxt

```python
# More flexible text loading
arr = np.genfromtxt('data.txt', dtype='U', delimiter=',',
                    missing_values='NA', filling_values='')

# Auto-detect dtype
arr = np.genfromtxt('data.txt', dtype=None, encoding='utf-8')
```

### Binary Files (tofile/fromfile)

```python
# Raw binary I/O (fixed-width only)
arr = np.array(['hello', 'world'], dtype='U5')
arr.tofile('strings.bin')

loaded = np.fromfile('strings.bin', dtype='U5')
# Must know exact dtype to read correctly
```

### Memory Mapping

```python
# Memory-mapped string arrays (fixed-width)
arr = np.memmap('strings.bin', dtype='U100', mode='r', shape=(1000,))
# Efficient for large files, strings loaded on demand

# StringDType cannot be memory-mapped (variable length)
```

---

## 13. Sorting and Searching

### Sorting

```python
# Sort string arrays lexicographically
arr = np.array(['banana', 'apple', 'cherry'])
np.sort(arr)  # ['apple', 'banana', 'cherry']

# Sort in place
arr.sort()

# Stable sort
np.sort(arr, kind='stable')

# Descending order
np.sort(arr)[::-1]  # ['cherry', 'banana', 'apple']
```

### argsort

```python
# Get sort indices
arr = np.array(['banana', 'apple', 'cherry'])
indices = np.argsort(arr)  # [1, 0, 2]
arr[indices]  # ['apple', 'banana', 'cherry']

# Multiple arrays
names = np.array(['Bob', 'Alice', 'Charlie'])
ages = np.array([25, 30, 20])
order = np.argsort(names)
names[order], ages[order]  # Sort both by name
```

### argmax/argmin

```python
# Find lexicographic max/min
arr = np.array(['banana', 'apple', 'cherry'], dtype='T')
np.argmax(arr)  # 2 ('cherry' is max)
np.argmin(arr)  # 1 ('apple' is min)

# Matches Python's max()/min()
assert np.argmax(arr) == arr.tolist().index(max(arr.tolist()))
```

### searchsorted

```python
# Binary search in sorted array
arr = np.array(['apple', 'banana', 'cherry', 'date'])
np.searchsorted(arr, 'blueberry')  # 2 (would insert between banana and cherry)
np.searchsorted(arr, 'apple')      # 0
np.searchsorted(arr, 'apple', side='right')  # 1
```

### StringDType Sorting with NA

```python
dt = StringDType(na_object=np.nan)
arr = np.array(['banana', np.nan, 'apple', np.nan], dtype=dt)

# NA values sort to end
np.sort(arr)  # ['apple', 'banana', nan, nan]

# Non-comparable NA (like None) raises error
dt = StringDType(na_object=None)
arr = np.array(['a', None, 'b'], dtype=dt)
np.sort(arr)  # ValueError: Cannot compare null that is not nan-like
```

### Comparison for Sorting

```python
# Lexicographic byte-by-byte comparison
# For Unicode: compares code points, not locale-aware

arr = np.array(['Z', 'a'])
np.sort(arr)  # ['Z', 'a'] (uppercase before lowercase in Unicode)

# Case-insensitive sort
arr = np.array(['Banana', 'apple', 'Cherry'])
indices = np.argsort(np.strings.lower(arr))
arr[indices]  # ['apple', 'Banana', 'Cherry']
```

---

## 14. Boolean Conversion and Truthiness

### String to Boolean

```python
# Empty string is falsy, non-empty is truthy
arr = np.array(['hello', '', 'world', ''], dtype='T')
arr.astype(bool)  # [True, False, True, False]

# In boolean context
np.any(arr)   # True (at least one non-empty)
np.all(arr)   # False (has empty strings)

# nonzero returns indices of non-empty strings
arr.nonzero()  # (array([0, 2]),)
```

### Boolean to String

```python
# Boolean converts to 'True'/'False'
arr = np.array([True, False, True])
arr.astype('T')  # ['True', 'False', 'True']
arr.astype('U')  # ['True', 'False', 'True']
```

### StringDType NA in Boolean Context

```python
dt = StringDType(na_object=np.nan)
arr = np.array(['hello', np.nan, ''], dtype=dt)

# NA (NaN-like) is truthy for nonzero
arr.nonzero()  # (array([0, 1]),) - includes NA

# But isnan identifies it
np.isnan(arr)  # [False, True, False]
```

### Comparison with Empty String

```python
# Testing for empty strings
arr = np.array(['hello', '', 'world'])
mask = arr == ''         # [False, True, False]
mask = np.strings.str_len(arr) == 0  # Same result
```

---

## 15. chararray Legacy Class

### Overview

`np.char.chararray` is a **deprecated** ndarray subclass providing:
- Automatic whitespace stripping on element access
- String methods as instance methods
- Operator overloading for string operations

**Recommendation**: Use regular arrays with `np.strings` functions instead.

### Creation

```python
# Via np.char.array (recommended if using chararray)
carr = np.char.array(['hello', 'world'])
carr = np.char.array(['hello', 'world'], itemsize=10)

# Via np.char.asarray (no copy if possible)
carr = np.char.asarray(['hello', 'world'])

# Direct constructor (not recommended)
carr = np.char.chararray((3,), itemsize=5, unicode=True)
```

### Automatic Whitespace Stripping

```python
carr = np.char.array(['hello ', 'world '])
carr[0]  # 'hello' (trailing space stripped!)

# Regular array keeps whitespace
arr = np.array(['hello ', 'world '], dtype='U')
arr[0]  # 'hello ' (preserved)
```

### Instance Methods

```python
carr = np.char.array(['hello', 'world'])

# All numpy.strings functions available as methods
carr.upper()        # chararray(['HELLO', 'WORLD'])
carr.capitalize()   # chararray(['Hello', 'World'])
carr.center(10)     # chararray(['  hello   ', '  world   '])
carr.find('o')      # array([4, 1])

# Operators
carr + ' suffix'    # chararray(['hello suffix', 'world suffix'])
carr * 2            # chararray(['hellohello', 'worldworld'])
carr % (1,)         # Printf formatting if format strings
```

### Comparison Behavior

```python
# chararray comparisons strip whitespace
carr1 = np.char.array(['hello '])
carr2 = np.char.array(['hello'])
carr1 == carr2  # array([True])  (spaces stripped before compare)

# Regular array comparison
arr1 = np.array(['hello '])
arr2 = np.array(['hello'])
arr1 == arr2  # array([False])  (exact comparison)
```

### Validation

```python
# chararray enforces string dtype
carr = np.char.chararray((3,))
carr.dtype.char  # 'S' or 'U'

# Cannot create with non-string dtype
# (raises ValueError in __array_finalize__)
```

---

## 16. Implementation Internals

### Encoding Modes (C++)

```cpp
// From string_buffer.h
enum class ENCODING {
    ASCII,  // bytes_ dtype - 1 byte per character
    UTF32,  // str_ dtype - 4 bytes per character (npy_ucs4)
    UTF8    // StringDType - 1-4 bytes per character
};
```

### Character Access Templates

```cpp
// Get character from buffer, returns code point and advances
template <ENCODING enc>
inline npy_ucs4 getchar(const unsigned char *buf, int *bytes);

// ASCII: 1 byte
template <>
inline npy_ucs4 getchar<ENCODING::ASCII>(const unsigned char *buf, int *bytes) {
    *bytes = 1;
    return (npy_ucs4) *buf;
}

// UTF-32: 4 bytes
template <>
inline npy_ucs4 getchar<ENCODING::UTF32>(const unsigned char *buf, int *bytes) {
    *bytes = 4;
    return *(npy_ucs4 *)buf;
}

// UTF-8: 1-4 bytes
template <>
inline npy_ucs4 getchar<ENCODING::UTF8>(const unsigned char *buf, int *bytes) {
    Py_UCS4 codepoint;
    *bytes = utf8_char_to_ucs4_code(buf, &codepoint);
    return (npy_ucs4)codepoint;
}
```

### Character Classification (Templates)

```cpp
// Platform-specific implementations
template<ENCODING enc>
inline bool codepoint_isalpha(npy_ucs4 code);

template<>
inline bool codepoint_isalpha<ENCODING::ASCII>(npy_ucs4 code) {
    return NumPyOS_ascii_isalpha(code);  // ASCII-only check
}

template<>
inline bool codepoint_isalpha<ENCODING::UTF32>(npy_ucs4 code) {
    return Py_UNICODE_ISALPHA(code);  // Full Unicode via Python
}

template<>
inline bool codepoint_isalpha<ENCODING::UTF8>(npy_ucs4 code) {
    return Py_UNICODE_ISALPHA(code);  // Same as UTF32 after decoding
}

// Similar templates for: isdigit, isspace, isalnum, islower, isupper, istitle
```

### StringDType API Functions

```c
// Allocator management
npy_string_allocator *NpyString_new_allocator(malloc, free, realloc);
void NpyString_free_allocator(npy_string_allocator *allocator);

// Thread-safe allocator access
npy_string_allocator *NpyString_acquire_allocator(descr);
void NpyString_release_allocator(allocator);

// String operations
int NpyString_pack(allocator, packed_string, buf, size);
int NpyString_load(allocator, packed_string, unpacked_string);
int NpyString_free(packed_string, allocator);

// Queries
int NpyString_isnull(packed_string);
size_t NpyString_size(packed_string);
int NpyString_cmp(s1, s2);

// Special values
int NpyString_pack_null(allocator, packed_string);
int NpyString_pack_empty(packed_string);
```

### Buffer Class (C++)

```cpp
template <ENCODING enc>
struct Buffer {
    char *buf;      // Start of string data
    char *after;    // One past end

    // Get number of codepoints (not bytes!)
    size_t num_codepoints();

    // Character-by-character operations
    // (implementation varies by encoding)
};
```

### UTF-8 Utilities API

```c
// From utf8_utils.h

// Convert UTF-8 byte sequence to code point
// Returns number of bytes consumed
size_t utf8_char_to_ucs4_code(const unsigned char *c, Py_UCS4 *code);

// Get number of bytes for UTF-8 character from first byte
// Uses lookup table based on high bits of first byte
static inline int num_bytes_for_utf8_character(const unsigned char *c) {
    static const char LENGTHS_LUT[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x00-0x7F: ASCII
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 4, 0   // continuation/multibyte
    };
    return LENGTHS_LUT[c[0] >> 3];
}

// Find previous UTF-8 character in buffer
const unsigned char* find_previous_utf8_character(const unsigned char *c, size_t nchar);

// How many UTF-8 bytes needed for a code point
int num_utf8_bytes_for_codepoint(uint32_t code);

// Count codepoints in UTF-8 byte sequence
int num_codepoints_for_utf8_bytes(const unsigned char *s, size_t *num_codepoints,
                                   size_t max_bytes);

// Calculate UTF-8 byte length from UCS-4 codepoints
int utf8_size(const Py_UCS4 *codepoints, long max_length,
              size_t *num_codepoints, size_t *utf8_bytes);

// Convert code point to UTF-8 bytes
// Returns number of bytes written
size_t ucs4_code_to_utf8_char(Py_UCS4 code, char *c);

// Get buffer size needed for UTF-8 string
Py_ssize_t utf8_buffer_size(const uint8_t *s, size_t max_bytes);

// Find byte positions for character indices
void find_start_end_locs(char* buf, size_t buffer_size,
                         npy_int64 start_index, npy_int64 end_index,
                         char **start_loc, char **end_loc);

// Get character index from byte offset
size_t utf8_character_index(const char* start_loc, size_t start_byte_offset,
                            size_t start_index, size_t search_byte_offset,
                            size_t buffer_size);
```

### UTF-8 Byte Length by Code Point

| Code Point Range | UTF-8 Bytes | Example |
|------------------|-------------|---------|
| U+0000 - U+007F | 1 | ASCII 'A' |
| U+0080 - U+07FF | 2 | Latin 'e' |
| U+0800 - U+FFFF | 3 | CJK, Euro |
| U+10000 - U+10FFFF | 4 | Emoji |

---

## 17. Edge Cases and Behavior

### Empty Strings

```python
# Empty string is valid
np.array([''], dtype='S')   # dtype='|S0' or '|S1' depending on context
np.array([''], dtype='U')   # dtype='<U0' or '<U1'
np.array([''], dtype='T')   # Empty StringDType element

# Empty vs zero-size dtype
np.dtype('S0').itemsize   # 0
np.dtype('U0').itemsize   # 0
# But arrays usually have minimum itemsize of 1
```

### Null Bytes

```python
# bytes_ preserves nulls
arr = np.array([b'hello\x00world'], dtype='S')
arr[0]  # b'hello' (null may terminate on access in some contexts)

# str_ preserves nulls
arr = np.array(['hello\x00world'], dtype='U')
len(arr[0])  # 11

# StringDType preserves nulls
arr = np.array(['hello\x00world'], dtype='T')
arr[0]  # 'hello\x00world'
len(arr[0])  # 11
```

### Unicode Edge Cases

```python
# Surrogate pairs (invalid in UTF-8, valid in UTF-16)
# NumPy UTF-32 can store any code point
arr = np.array(['\ud800'], dtype='U')  # Valid (isolated surrogate)
arr.astype('T')  # May fail (invalid UTF-8)

# Combining characters
arr = np.array(['e\u0301'], dtype='U')  # e + combining acute
np.strings.str_len(arr)  # [2] (2 code points, displays as 1 char)

# Normalization not automatic
arr = np.array(['\u00e9', 'e\u0301'], dtype='U')  # Same display, different
arr[0] == arr[1]  # False (different code point sequences)
```

### Very Large Strings

```python
# Size limits
# bytes_/str_: Limited by np.intc.max // itemsize_per_char
max_chars_u = np.iinfo(np.intc).max // 4  # ~536M chars on 32-bit signed

# StringDType: Limited by NPY_MAX_STRING_SIZE
# ~2^56 bytes on 64-bit (size_t minus 1 flag byte)

# Overflow detection
np.strings.multiply('a', sys.maxsize)  # Raises OverflowError
np.strings.add(very_large_1, very_large_2)  # Raises if exceeds limit
```

### Mixed dtype Operations

```python
# bytes_ and str_ cannot be compared directly
arr_s = np.array([b'hello'], dtype='S')
arr_u = np.array(['hello'], dtype='U')
arr_s == arr_u  # TypeError: no loop found

# Must cast explicitly
(arr_s.astype('U') == arr_u)  # Works

# StringDType can compare with str_ and object
arr_t = np.array(['hello'], dtype='T')
arr_u = np.array(['hello'], dtype='U')
arr_t == arr_u  # Works (promotion to common type)
```

### Broadcasting

```python
# All string operations broadcast
arr = np.array(['a', 'b', 'c'])
np.strings.add(arr, 'x')  # ['ax', 'bx', 'cx']

arr2d = np.array([['a', 'b'], ['c', 'd']])
np.strings.add(arr2d, [['1'], ['2']])
# [['a1', 'b1'], ['c2', 'd2']]
```

---

## 18. Best Practices

### Choosing a String Type

| Use Case | Recommended Type | Reason |
|----------|------------------|--------|
| Modern text processing | `StringDType ('T')` | Memory efficient, full Unicode, NA support |
| Pre-NumPy 2.0 compatibility | `str_ ('U')` | Works everywhere |
| Fixed-width records | `str_ ('U')` or `bytes_ ('S')` | Predictable memory layout |
| Binary data | `bytes_ ('S')` | No encoding assumptions |
| C struct compatibility | `bytes_ ('S')` | Direct memory mapping |
| Database integration | `StringDType ('T')` | Variable length, NA support |

### Performance Considerations

```python
# StringDType is most memory-efficient for mixed content
# ASCII-heavy: 1 byte/char (vs 4 for str_)
# With some emoji: still efficient

# str_ has O(1) character access
# StringDType has O(n) character access (UTF-8)

# For large string operations, use vectorized numpy.strings functions
# Avoid Python loops over individual strings
```

### Memory Management

```python
# StringDType uses arenas - memory released when array deleted
arr = np.array(['large' * 1000] * 10000, dtype='T')
del arr  # Arena freed

# str_ memory is contiguous - good for memory-mapped files
arr = np.memmap('data.bin', dtype='U100', mode='r', shape=(1000,))

# bytes_ for raw binary protocols
data = np.frombuffer(binary_data, dtype='S100')
```

### API Usage

```python
# Prefer numpy.strings over numpy.char
import numpy as np

# Good
result = np.strings.upper(arr)
result = np.strings.find(arr, 'pattern')

# Avoid (legacy)
result = np.char.upper(arr)

# Never use chararray in new code
# carr = np.char.chararray(...)  # Deprecated
```

### Error Handling

```python
# Check for encoding errors when converting
try:
    arr_s = arr_u.astype('S')
except UnicodeEncodeError:
    # Handle non-ASCII content
    pass

# Check for NA before operations
if hasattr(dtype, 'na_object'):
    # Handle potential NA values
    mask = np.isnan(arr) if isinstance(dtype.na_object, float) else (arr is dtype.na_object)
```

---

## Appendix: Quick Reference

### Dtype Strings

| Pattern | Type | Example | Description |
|---------|------|---------|-------------|
| `S` | bytes_ | `S10` | 10-byte string |
| `\|S` | bytes_ | `\|S10` | Same (byte order n/a) |
| `U` | str_ | `U10` | 10-char (40-byte) unicode |
| `<U` | str_ | `<U10` | Little-endian unicode |
| `>U` | str_ | `>U10` | Big-endian unicode |
| `T` | StringDType | `T` | Variable-length UTF-8 |

### Common Operations Matrix

| Operation | Function | bytes_ | str_ | StringDType |
|-----------|----------|--------|------|-------------|
| Length | `str_len` | Y | Y | Y |
| Concat | `add` | Y | Y | Y |
| Repeat | `multiply` | Y | Y | Y |
| Find | `find/rfind` | Y | Y | Y |
| Count | `count` | Y | Y | Y |
| Upper | `upper` | Y | Y | Y |
| Lower | `lower` | Y | Y | Y |
| Strip | `strip/lstrip/rstrip` | Y | Y | Y |
| Replace | `replace` | Y | Y | Y |
| Compare | `equal/less/etc` | Y | Y | Y |
| Encode | `encode` | - | Y | Y |
| Decode | `decode` | Y | - | - |
| isalpha | `isalpha` | Y (ASCII) | Y (Unicode) | Y (Unicode) |
| isnan | `isnan` | N | N | Y (with NA) |

### Memory Sizes

| Content | bytes_ | str_ | StringDType |
|---------|--------|------|-------------|
| "" (empty) | 0-1 bytes | 0-4 bytes | 16 bytes |
| "hello" (5 ASCII) | 5 bytes | 20 bytes | 16 bytes (inline) |
| "hello world" (11) | 11 bytes | 44 bytes | 16 + 11 bytes |
| Single emoji | N/A | 4 bytes | 16 + 4 bytes |
| 100 ASCII chars | 100 bytes | 400 bytes | 16 + 100 bytes |

### Array Functions with String Support

| Function | bytes_ | str_ | StringDType | Notes |
|----------|--------|------|-------------|-------|
| `np.sort` | Y | Y | Y | Lexicographic |
| `np.argsort` | Y | Y | Y | |
| `np.argmax` | Y | Y | Y | |
| `np.argmin` | Y | Y | Y | |
| `np.searchsorted` | Y | Y | Y | |
| `np.unique` | Y | Y | Y | |
| `np.concatenate` | Y | Y | Y | |
| `np.stack/vstack/hstack` | Y | Y | Y | |
| `np.where` | Y | Y | Y | |
| `np.take` | Y | Y | Y | |
| `np.nonzero` | Y | Y | Y | Empty string = False |
| `np.any` | Y | Y | Y | Empty string = False |
| `np.all` | Y | Y | Y | Empty string = False |
| `np.copy` | Y | Y | Y | |
| `np.resize` | Y | Y | Y | |
| `np.save/load` | Y | Y | Y | .npy format |

### Type Hierarchy

```
numpy.generic
    numpy.number
        ... (numeric types)
    numpy.flexible
        numpy.void (V)
        numpy.character
            numpy.bytes_ (S)
            numpy.str_ (U)
    numpy.object_ (O)

# StringDType is a new-style dtype, not in scalar hierarchy
numpy.dtypes.StringDType (T)
```

### String UFuncs

| UFunc | Signature | Notes |
|-------|-----------|-------|
| `np.add` | `(T,T)->T` | String concatenation |
| `np.multiply` | `(T,i)->T` | String repetition |
| `np.equal` | `(T,T)->?` | Element-wise == |
| `np.not_equal` | `(T,T)->?` | Element-wise != |
| `np.less` | `(T,T)->?` | Lexicographic < |
| `np.less_equal` | `(T,T)->?` | Lexicographic <= |
| `np.greater` | `(T,T)->?` | Lexicographic > |
| `np.greater_equal` | `(T,T)->?` | Lexicographic >= |
| `np.maximum` | `(T,T)->T` | Lexicographic max |
| `np.minimum` | `(T,T)->T` | Lexicographic min |

### isnumeric vs isdecimal vs isdigit

| Function | '123' | '\u00b2' (superscript 2) | '\u2460' (circled 1) | Arabic-Indic |
|----------|-------|--------------------------|---------------------|--------------|
| `isdigit` | True | True | True | True |
| `isdecimal` | True | False | False | True |
| `isnumeric` | True | True | True | True |

- `isdecimal`: Characters forming decimal numbers in base 10 (0-9)
- `isdigit`: Includes superscript/subscript digits
- `isnumeric`: Includes fractions, Roman numerals, etc.

**Note**: `isdecimal` and `isnumeric` are Unicode-only, will raise TypeError on bytes_

### StringDType Flag Summary

| Flag | Value | Meaning |
|------|-------|---------|
| `NPY_STRING_MISSING` | 0x80 | NA/null value |
| `NPY_STRING_INITIALIZED` | 0x40 | Element has been set |
| `NPY_STRING_OUTSIDE_ARENA` | 0x20 | Heap-allocated |
| `NPY_STRING_LONG` | 0x10 | Arena string >255 bytes |

### Short String Optimization Thresholds

| Architecture | Max Inline Size | Struct Size |
|--------------|-----------------|-------------|
| 64-bit | 15 bytes | 16 bytes |
| 32-bit | 7 bytes | 8 bytes |

---

*Document generated from NumPy v2.4.2 source analysis*
