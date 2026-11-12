# Dtypes in NumSharp

Every array in NumSharp has a **dtype**-a data type that determines what kind of values the array stores, how many bytes each element takes, and which operations are valid. When you write `np.zeros(10, np.int32)`, the `np.int32` is the dtype. When you call `arr.astype(np.float64)`, you're converting to a different dtype.

This page covers NumSharp's supported dtypes, how they map to NumPy's types, how to refer to them in code, and the places where NumSharp's behavior differs from NumPy.

---

## Supported dtypes

NumSharp supports the following dtypes:

| NPTypeCode | C# Type | NumPy Equivalent | Bytes | Kind | SIMD |
|------------|---------|------------------|-------|------|------|
| `Boolean`  | `bool`                     | `bool`       | 1  | `b` | Operation-dependent |
| `SByte`    | `sbyte`                    | `int8`       | 1  | `i` | Yes |
| `Byte`     | `byte`                     | `uint8`      | 1  | `u` | Yes |
| `Int16`    | `short`                    | `int16`      | 2  | `i` | Yes |
| `UInt16`   | `ushort`                   | `uint16`     | 2  | `u` | Yes |
| `Int32`    | `int`                      | `int32`      | 4  | `i` | Yes |
| `UInt32`   | `uint`                     | `uint32`     | 4  | `u` | Yes |
| `Int64`    | `long`                     | `int64`      | 8  | `i` | Yes |
| `UInt64`   | `ulong`                    | `uint64`     | 8  | `u` | Yes |
| `Half`     | `System.Half`              | `float16`    | 2  | `f` | Specialized kernels |
| `Single`   | `float`                    | `float32`    | 4  | `f` | Yes |
| `Double`   | `double`                   | `float64`    | 8  | `f` | Yes |
| `Decimal`  | `decimal`                  | *no equiv*   | 16 | `f` | Scalar arithmetic |
| `Complex`  | `System.Numerics.Complex`  | `complex128` | 16 | `c` | Operation-dependent |
| `Char`     | `char`                     | *no equiv*   | 2  | `u` | Through uint16 kernels where supported |
| `String`   | `byte[]` values            | `bytes_` (`S`) | n | `S` | Specialized kernels |
| `String`   | `string` values            | `str_` (`U`) | 4 × n | `U` | Specialized kernels |
| `String`   | `string` or NA values      | `StringDType` (`T`) | 16-byte cell | `T` | Specialized kernels |

**Bytes** and **Kind** are the descriptor's `itemsize` and `kind`. For `S` and `U`, **n** is the width in bytes or Unicode code points; shorter values are NUL-padded and longer writes truncate. `StringDType` has no fixed text width: each element has a 16-byte cell, with longer text stored separately. Its `nbytes` counts the cells, not that extra storage.

The three string dtypes share `NPTypeCode.String`. Their full descriptors distinguish the representation and carry the width, byte order or `na_object` / `coerce` settings. Use `arr.dtype` when preserving a dtype. See [Strings, bytes and chararrays](fundamentals/strings-and-bytes.md) for creation, operations, missing values and interop.

**Decimal** and **Char** are NumSharp-specific types with no NumPy counterpart; see [NumSharp-specific types](#numsharp-specific-types-decimal-and-char).

---

## Referring to Dtypes in Code

Every public `dtype` parameter takes a **`DType` descriptor**. You can name a dtype with an `np.*` alias, a C# `Type`, an `NPTypeCode` or a dtype string:

### 1. `NPTypeCode` enum

```csharp
var arr = np.zeros(new Shape(10), NPTypeCode.Int32);
var cplx = np.zeros(new Shape(2, 3), NPTypeCode.Complex);
```

Use the enum when the element type is already known. String dtypes need a descriptor such as `"U10"`, `"S5"` or `np.dtypes.StringDType()` because `NPTypeCode.String` alone carries no width or NA policy.

### 2. `np.*` class-level aliases (idiomatic)

```csharp
var arr = np.zeros(new Shape(10), np.int32);
var half = np.ones(new Shape(5), np.float16);
var cplx = np.zeros(new Shape(2, 3), np.complex128);
```

These match NumPy's Python API (`np.int32`, `np.float16`, `np.complex128`). Most NumSharp code uses this form.

### 3. Dtype strings (NumPy-compatible parsing)

```csharp
var a = np.dtype("int32");
var b = np.dtype("float16");
var c = np.dtype("complex128");
var d = np.dtype("i4");         // NumPy shorthand
var e = np.dtype("<f8");        // with byte-order prefix
```

Use strings when the dtype is dynamic (from config, JSON, a file header), or to name a fixed string width. Supported forms and explicit refusals are listed in [the parsing table](#dtype-string-parsing).

---

## Dtype String Parsing

NumSharp's `np.dtype(string)` mirrors NumPy 2.x exactly for every form NumSharp has a type for, throws NumPy's `TypeError` for a string NumPy rejects too, and throws `NotSupportedException` for valid NumPy forms it has no type for (`V`, `O`, `complex64`, sub-array and structured dtypes).

### Single-char codes

| String | Resolves to | Notes |
|--------|-------------|-------|
| `"?"` | `Boolean` | |
| `"b"` | `SByte` | NumPy: int8 (signed, C `char` convention) |
| `"B"` | `Byte` | NumPy: uint8 |
| `"h"` / `"H"` | `Int16` / `UInt16` | |
| `"i"` / `"I"` | `Int32` / `UInt32` | C `int`, always 32-bit; on Windows its own class `np.intc` / `np.uintc` (see [C-type classes](#c-type-classes)) |
| `"l"` / `"L"` | **platform-dependent** | C `long` - see [Platform-Dependent Types](#platform-dependent-types) |
| `"q"` / `"Q"` | `Int64` / `UInt64` | C `long long`, always 64-bit; on Linux/macOS its own class `np.longlong` / `np.ulonglong` |
| `"n"` / `"N"` | pointer-sized | NumPy 2.x's letters for `np.intp` / `np.uintp` |
| `"p"` / `"P"` | pointer-sized | see [Platform-Dependent Types](#platform-dependent-types) |
| `"e"` | `Half` | NumPy: float16 |
| `"f"` | `Single` | NumPy: float32 |
| `"d"` / `"g"` | `Double` | `g` is the long double: its own class `np.longdouble`, stored as float64 (8 bytes, as on NumPy's Windows build) |
| `"D"` / `"G"` | `Complex` | `G` is the long-double complex: its own class `np.clongdouble`, stored as complex128 |
| `"F"` | **throws** | complex64 - NumSharp only has complex128 |
| `"c"` | `S1` | a 1-byte bytestring (NumPy's `'c'`) |
| `"S"` / `"a"` / `"U"` | `S` / `S` / `<U` | unsized string descriptors (`itemsize` 0): a creation or a cast sizes them from the data |
| `"T"` | `StringDType()` | variable-width UTF-8; use `np.dtypes.StringDType(na_object: ..., coerce: ...)` for parameters |
| `"V"` / `"O"` | **throws** | void, object-not supported |
| `"M"` / `"m"` | datetime64 / timedelta64 descriptors | parse, promote and cast-check; arrays cannot be allocated yet |

A single control character below `"\x18"` is a type NUMBER, as in NumPy (`np.dtype("\x0c")` is float64, `np.dtype("\x00")` bool, `np.dtype("\x13")` the unsized `<U`).

### Sized variants

| String | Resolves to | Notes |
|--------|-------------|-------|
| `"b1"` | `Boolean` | |
| `"i1"` / `"u1"` | `SByte` / `Byte` | |
| `"i2"` / `"u2"` / `"f2"` | `Int16` / `UInt16` / `Half` | |
| `"i4"` / `"u4"` / `"f4"` | `Int32` / `UInt32` / `Single` | |
| `"i8"` / `"u8"` / `"f8"` / `"c16"` | `Int64` / `UInt64` / `Double` / `Complex` | |
| `"c8"` | **throws** | complex64-not supported |
| `"S5"` / `"a5"` / `"\|S5"` | `\|S5` | a 5-byte bytestring (`itemsize` 5) |
| `"U5"` / `"<U5"` / `">U5"` | `<U5` / `<U5` / `>U5` | 5 UCS-4 code points (`itemsize` 20); a `>` prefix is kept |
| `"i3"`, `"f3"`, `"f16"`, `"b4"`, ... | **throws** | invalid NumPy dtype strings |

### Named forms

| String | Resolves to |
|--------|-------------|
| `"int8"` / `"sbyte"` | `SByte` |
| `"uint8"` / `"ubyte"` | `Byte` |
| `"byte"` | **`SByte`** (NumPy convention: `byte` = int8, signed) |
| `"int16"` / `"short"` | `Int16` |
| `"uint16"` / `"ushort"` | `UInt16` |
| `"int32"` / `"intc"` | `Int32` (`"intc"` is its own class on Windows) |
| `"uint32"` / `"uintc"` | `UInt32` (`"uintc"` is its own class on Windows) |
| `"int64"` / `"longlong"` | `Int64` (`"longlong"` is its own class on Linux/macOS) |
| `"uint64"` / `"ulonglong"` | `UInt64` (`"ulonglong"` is its own class on Linux/macOS) |
| `"longdouble"` / `"clongdouble"` | `Double` / `Complex`, each its own class |
| `"int"` / `"int_"` / `"intp"` | **pointer-sized**-int64 on 64-bit |
| `"uint"` / `"uintp"` | **pointer-sized**-uint64 on 64-bit |
| `"long"` / `"ulong"` | **platform-dependent** C long/ulong |
| `"float16"` / `"half"` | `Half` |
| `"float32"` / `"single"` | `Single` |
| `"float64"` / `"double"` | `Double` |
| `"float"` | `Double` (NumPy: `float` → float64) |
| `"complex128"` / `"complex"` / `"cdouble"` | `Complex` |
| `"complex64"` / `"csingle"` | **throws** |
| `"bool"` / `"bool_"` | `Boolean` |
| `"str"` / `"str_"` / `"unicode"` | `<U` (unsized) |
| `"bytes"` / `"bytes_"` | `S` (unsized) |

Byte-order prefixes (`<`, `>`, `=`, `|`) are parsed, not discarded, before a single-character or sized code. An explicit host order normalizes to native (`np.dtype("<i4")` on a little-endian host); a non-native descriptor keeps its order (`np.dtype(">U5").byteorder == '>'`, `np.dtype(">i4").isnative` is false). Buffer and file readers convert non-native data where required. `S` has no byte order. `StringDType` also has no byte order, and its `newbyteorder` operation raises NumPy's `TypeError`. A NAME takes no prefix, as in NumPy, which looks names up by the whole string: `np.dtype("<float64")` throws `data type '<float64' not understood`, and `np.dtype("<str")` throws the same way.

An invalid string throws NumPy's `TypeError` with NumPy's text, the string quoted the way Python's `repr` quotes it
(`data type "a'b" not understood`); a removed alias says `Alias 'bool8' was removed in NumPy 2.0. Use a name without a
digit at the end.` A string that NumPy accepts but whose dtype NumSharp has no storage for (complex64, void, object,
and the 16-byte long double `f16` / `float128` where the platform has one, as 64-bit Linux does) throws
`NotSupportedException` saying `is not supported by NumSharp`, so a caller can tell "not a dtype" from "a dtype
NumSharp lacks".

A comma string - a top-level comma, a leading digit or a leading `()` - is read the way NumPy's `_commastring` reads
it: each field is `[byte order][repeat count][byte order]dtype`. A field whose NumPy result is a plain dtype parses:
`np.dtype("()f8")` is float64 (an empty repeat count is the type itself), and a count sizes an unsized string class,
so `np.dtype("5U")` is `<U5`, `np.dtype("2>U")` is `>U2` and `np.dtype("3S")` is `S3`. A sub-array (`"3f8"`,
`"(2,3)i4"`) or a structured dtype (`"i4,f8"`) throws `NotSupportedException` (`NumPy dtype '3f8' (a sub-array
dtype) is not supported by NumSharp`), and only after NumPy's own errors in the string, which keep NumPy's type and
text: a `ValueError` such as `invalid shape in fixed-type tuple: dimension does not fit into a C int.`,
`inconsistent byte-order specification < and >` or `invalid itemsize in generic type tuple` (`"()U"`), a `TypeError`
for a field NumPy does not understand (`np.dtype("F,xyz")` names `'xyz'`, although NumSharp has no `F`), and a
`SyntaxError` for a repeat count that is not a Python literal (`np.dtype("01f8")`: `leading zeros in decimal integer
literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 1)`).

---

## Class-level Aliases

The supported dtype aliases on `np` are **`DType` descriptors**, not `System.Type` objects:

```csharp
DType integers = np.int32;
DType unicode = np.str_;                         // unsized U; creation infers a width
DType bytes = np.bytes_;                         // unsized S
DType text = np.dtypes.StringDType(na_object: null);
Type clr = np.int32.type;                        // typeof(int)
```

Names that clash with C# keywords use `@`: `np.@byte`, `np.@short`, `np.@long`, `np.@bool`, `np.@decimal`. **`np.@char` is the `numpy.char` module**, with an implicit conversion to the existing Char descriptor for compatibility. Use `np.dtype(typeof(char))` when you want the descriptor itself; `np.@char.array(...)` creates a chararray of `S` / `U` strings, not a `Char` array.

Aliases and parsed descriptors compare as descriptors, and a builtin alias and the string parser resolve to the same descriptor object (this invariant is guaranteed):

```csharp
np.int32 == np.dtype("int32");     // true
np.str_ == np.dtype("U");         // true
np.dtype("U3") == np.dtype("U5"); // false: width matters
// These are always true on every platform:
ReferenceEquals(np.int32, np.dtype("int32"));
ReferenceEquals(np.@long, np.dtype("long"));
ReferenceEquals(np.float16, np.dtype("float16"));
```

A descriptor also compares with a dtype string, as NumPy's `dtype.__eq__` does: the string is parsed and compared, and a string that is not a dtype is simply unequal, never an error. A `StringDType` descriptor equals no string, and a `null` string stands for "no descriptor":

```csharp
np.int32 == "i4";                  // true
np.int32 != "bogus";               // true: not a dtype, so unequal
np.dtype("T") == "T";              // false, as in NumPy
DType none = null;
none == null;                      // true: there is no descriptor
```

Use `.type` or an explicit `(Type)` cast only when a .NET API needs the CLR element type. It is not a replacement for a text descriptor.

---

## Complex: Only 128-bit Is Supported

NumSharp uses `System.Numerics.Complex`, which is two 64-bit floats-equivalent to NumPy's **`complex128`**. There is no 32-bit variant.

```csharp
// These all resolve to Complex (complex128):
np.complex128;
np.complex_;          // NumPy 1.x's name, removed in NumPy 2.0: kept so ported code compiles (np.dtype("complex_") throws)
np.cdouble;           // NumPy alias for complex128
np.clongdouble;       // NumPy's long-double complex: its own class, stored as complex128 (equal to it)
np.dtype("complex");  // NumPy 2.x default complex is complex128
np.dtype("D");
np.dtype("c16");
np.dtype("G");

// Class objects exist, but no complex64 array storage exists:
np.complex64;           // NDScalarType, usable in type checks
np.csingle;             // the same scalar class

// These descriptor requests throw NotSupportedException:
np.dtype("complex64");
np.dtype("F");
np.dtype("c8");
```

Why throw instead of silently widening to complex128? Because quietly upgrading a user's `complex64` request to `complex128` doubles their memory use without telling them. If you need complex arrays in NumSharp, use `complex128` explicitly-the throw is there to prevent surprise.

**Converting from complex arrays to real arrays:** `(int)complexScalar` and `(float)complexScalar` throw `TypeError` with Python's text (`int() argument must be a string, a bytes-like object or a real number, not 'complex'`), matching Python's built-in behavior. NumPy 2.x emits a `ComplexWarning` and drops the imaginary part; NumSharp has no warning system, so we treat this as a hard error. If you want the real part, use `np.real(arr)` first.

```csharp
var c = NDArray.Scalar<Complex>(new Complex(3, 4));
var x = (int)c;        // throws TypeError
var r = (int)np.real(c);  // 3 - explicit, unambiguous
```

---

## NumPy Types NumSharp Doesn't Support

NumPy has several dtype families that NumSharp does not store as arrays. They are separate from the **supported** text families: `bytes_` (`S`), `str_` (`U`) and variable-width `StringDType` (`T`). See [Working with arrays of strings and bytes](fundamentals/strings-and-bytes.md).

| NumPy dtype | NumPy character | Why not in NumSharp |
|-------------|-----------------|---------------------|
| `complex64` | `F`, `c8` | NumSharp has only one complex type (`complex128`). Silently widening would double memory without asking. See [Complex: Only 128-bit Is Supported](#complex-only-128-bit-is-supported). |
| `void` / `V` | `V` | NumPy "raw bytes" scalar. No .NET equivalent; use `byte[]` or `Memory<byte>`. |
| `object` / `O` | `O` | Arbitrary Python-object arrays. Use .NET collections; `NDArray<object>` is not supported. The string split functions return the specialized `NDListArray`, not a general object dtype. |
| `datetime64` | `M`, `M8[ns]` etc. | The descriptors exist (they parse, print, promote and cast-check with NumPy's unit rules), but arrays of them cannot be allocated yet. Use `DateTime[]` directly, or `long[]` with epoch seconds. |
| `timedelta64` | `m`, `m8[us]` etc. | Same as `datetime64`: descriptors only. Use `TimeSpan[]` or `long[]`. |
| Structured / record dtypes | comma strings with several fields: `"f8,i4"`, `"f8,"` | NumPy allows composite dtypes like `np.dtype([('x', 'f4'), ('y', 'i4')])` for heterogeneous records. NumSharp throws on every comma string whose NumPy result is structured, after NumPy's own errors in it. Use a struct array or multiple parallel `NDArray`s. |
| Sub-array dtypes | `('f4', (3,))`; comma strings with a repeat count: `"3f8"`, `"(2,)f8"` | NumPy dtype-with-subshape. Not supported. A count on an unsized string class is no sub-array: `"5U"` is `<U5`, and `"()f8"` is float64. |

Every row above is tested in `test/NumSharp.Tests/Creation/DTypeStringParityTests.cs` with an `ExpectThrow` assertion. If you run into one of these in ported NumPy code, the exception message tells you which NumSharp alternative to use.

### Why Throw Instead of Silent Approximation?

A recurring temptation is to "do the nearest thing"-e.g., widen `complex64` to `complex128` or map `V7` to `byte[]`. NumSharp refuses this because:

1. **Memory surprise**: doubling precision doubles allocation; a user loading a gigabyte of `complex64` data would unexpectedly use two gigabytes.
2. **Precision surprise**: downstream computations on the "wrong" type produce results the user didn't request.
3. **Signal clarity**: a `NotSupportedException` with a clear message ("use np.complex128 instead") is actionable. Silent widening is a ticking bug.

---

## NumSharp-Specific Types (Decimal and Char)

Two types in NumSharp have no NumPy equivalent. They exist for .NET-idiomatic use cases where NumPy's dtype set is too narrow.

### `Decimal` - 128-bit fixed-point

.NET's `System.Decimal` is a 16-byte fixed-point number with 28-29 significant digits. It's the right type for **money and financial computation** where binary floating-point's representation errors are unacceptable (`0.1 + 0.2 != 0.3` is a non-starter for an accounting ledger).

```csharp
var prices = np.array([19.99m, 29.99m, 5.00m]);
prices.typecode;                // NPTypeCode.Decimal
InfoOf<decimal>.Size;           // 16 (actual memory footprint)
var total = np.sum(prices);     // exact decimal sum, no float drift
```

**Characteristics:**
- `kind == 'f'` (float-like-it's a fractional type even though internally integer-based)
- `char == 'A'`, a type character of its own, as NumPy's user-defined dtypes have one: no NumPy dtype uses it, so `np.dtype("A")` is NumPy's "not understood" rather than some other dtype
- No SIMD acceleration (decimal arithmetic is scalar-only; much slower than `double`)
- No IEEE special values: no NaN, no Infinity, no subnormals
- `np.finfo(NPTypeCode.Decimal)` works and returns limited info (bits=128, precision=28, no subnormals)
- Boundary values: `Decimal.MinValue` / `Decimal.MaxValue` (±79228162514264337593543950335)
- `np.dtype(typeof(decimal)).itemsize == 16`, matching `InfoOf<decimal>.Size`.

**When to use:**
- Financial calculations (currency, tax, interest)
- Any scenario where exact decimal representation matters more than speed

**When NOT to use:**
- Scientific computing (`double` is faster and has wider range)
- SIMD-critical paths (no vectorization)
- Interop with NumPy/Python (no round-trip-NumPy has no decimal type)

### `Char` - 16-bit UTF-16 code unit

`System.Char` is a 2-byte Unicode UTF-16 code unit. NumSharp preserves it as a dtype mostly for arrays of characters where the type system benefits from knowing "these are characters, not shorts."

```csharp
var letters = np.array(['a', 'b', 'c']);
letters.typecode;               // NPTypeCode.Char
InfoOf<char>.Size;              // 2 (actual memory footprint)
```

**Important:** NumSharp's `Char` is **not** the same as NumPy's `'c'` / `S1` (a 1-byte bytestring) or `<U1` (a 4-byte code point). They have different sizes, different encodings, different semantics. Porting NumPy text code to NumSharp `Char` will almost always be wrong-use NumSharp's `S` and `U` string arrays, which are NumPy's.

**Characteristics:**
- `kind == 'u'` (unsigned-integer semantics, not the `S` / `U` text families)
- `char == 'u'`, PEP 3118's code for a 2-byte UCS-2 character, which its memoryview reports as the format too; NumPy reads `'u'` only as the unsigned-integer kind (`'u2'`), so `np.dtype("u")` is "not understood" (NumPy's `'c'` is `S1`)
- Treated as `ushort` for many operations (same byte width)
- Boundary values: `'\0'` (0) to `char.MaxValue` (65535)
- `np.dtype(typeof(char)).itemsize == 2`, matching `InfoOf<char>.Size`.
- A cast to text formats the numeric code unit: `'a'` becomes `"97"`. To build text from characters, first build the intended .NET `string`; do not use `astype("U")` as a UTF-16 decoder.

**When to use:**
- Arrays of individual characters where type annotation matters
- Interop with APIs that treat char specifically

**When NOT to use:**
- Text data-use a `U` array (`np.array(new[] { "a", "bc" })`) or a .NET `string[]`
- Porting NumPy bytestring arrays-use an `S` array (`arr.astype("S5")`) or `byte[]`

---

## Platform-Dependent Types

Some dtype names follow C's native `long` convention, which differs between compilers:

- **Windows (MSVC, LLP64 model):** C `long` is 32 bits
- **64-bit Linux/Mac (gcc, LP64 model):** C `long` is 64 bits

NumPy inherits this from its C compiler, so `np.dtype("long")` gives **`int32`** on Windows and **`int64`** on Linux. This is a well-known NumPy quirk, tracked in [numpy/numpy#9464](https://github.com/numpy/numpy/issues/9464). NumSharp matches NumPy's platform convention exactly by detecting the OS at runtime.

### What's platform-dependent

| Spelling | Windows 64-bit | Linux/Mac 64-bit |
|----------|----------------|------------------|
| `np.@long`, `np.dtype("long")`, `"l"` | `Int32` | `Int64` |
| `np.@ulong`, `np.dtype("ulong")`, `"L"` | `UInt32` | `UInt64` |

### What's *not* platform-dependent

Everything else is fixed across platforms:

| Spelling | Always |
|----------|--------|
| `np.int_`, `np.intp`, `"int"`, `"int_"`, `"intp"`, `"p"` | pointer-sized (int64 on 64-bit platforms) |
| `np.longlong`, `"longlong"`, `"q"`, `"i8"` | `Int64` storage (the class of `np.longlong` is int64's on Windows, its own on Linux/macOS) |
| `np.int32`, `"int32"`, `"i"`, `"i4"` | `Int32` storage (the class of `"i"` / `np.intc` is int32's on Linux/macOS, its own on Windows) |
| `np.int16`, `"int16"`, `"h"`, `"i2"` | `Int16` |

### Recommendation

If you want **portable** code across Windows and Linux, avoid `long`/`ulong`/`l`/`L`. Use explicit sized names:

```csharp
// Portable - same result on every platform:
var a = np.zeros(shape, np.int32);
var b = np.zeros(shape, np.int64);
var c = np.dtype("int64");

// Platform-dependent - different result on Win vs Linux:
var d = np.zeros(shape, np.@long);
var e = np.dtype("long");
```

This is the same guidance NumPy itself gives-see the [NumPy data types page](https://numpy.org/doc/stable/user/basics.types.html).

---

## Creating Arrays with a Specific Dtype

### Explicit dtype

```csharp
var a = np.zeros(new Shape(3, 4), NPTypeCode.Single);  // float32 zeros
var b = np.ones(new Shape(5), np.float16);             // Half ones
var c = np.full(new Shape(2), (Half)3.14);             // Half filled with 3.14
var d = np.arange(0, 10, dtype: np.int8);              // int8 range
var e = np.empty(new Shape(100), np.complex128);       // uninitialized complex
```

### Inferred from the source array

`np.array(T[])` infers the dtype from the .NET array type:

```csharp
np.array([1, 2, 3]);                    // dtype=int32   (int literals)
np.array([1.0, 2.0]);                   // dtype=float64 (double literals)
np.array([(Half)1, (Half)2]);           // dtype=float16
np.array([new Complex(1,2), new Complex(3,4)]);  // dtype=complex128
np.array(new sbyte[] { -1, 0, 1 });     // dtype=int8    (explicit - see note)
```

> **Narrow integer types need the explicit `new T[]` form.** A collection-expression literal `[…]` infers its element type from the values, and `sbyte`/`byte`/`short`/`ushort` have **no C# literal suffix**, so `[-1, 0, 1]` is `int` (→ int32), not `sbyte`. Use `new sbyte[] { -1, 0, 1 }` (or cast every element) when you need one of those. Types **with** a suffix are fine inline - `[1L, 2L, 3L]` → int64, `[1u, 2u, 3u]` → uint32, `[1f, 2f]` → float32, `[1m, 2m]` → decimal.

### Converting between dtypes

Use `.astype()` for array-level conversions:

```csharp
var doubles = np.array([1.5, 2.7, 3.9]);
var ints    = doubles.astype(NPTypeCode.Int32);      // [1, 2, 3] (truncated)
var halfs   = doubles.astype(NPTypeCode.Half);       // [1.5, 2.7, 3.9] (float16)
var cplxs   = doubles.astype(NPTypeCode.Complex);    // [1.5+0j, 2.7+0j, 3.9+0j]
```

### Scalar ↔ NDArray casts

Every numeric C# type can be implicitly converted to a 0-d `NDArray`:

```csharp
NDArray s1 = (sbyte)42;        // 0-d int8 scalar
NDArray s2 = (Half)3.14;       // 0-d float16 scalar
NDArray s3 = new Complex(1, 2); // 0-d complex128 scalar
```

Explicit casts back to .NET scalars require a 0-dimensional array (`ndim == 0`):

```csharp
var scalar = np.array([42])[0];  // the element: NumPy's scalar (a read-only 0-d copy)
int x = (int)scalar;                     // works

var oneD = np.array([42]);
int y = (int)oneD;   // throws TypeError: only 0-dimensional arrays can be converted to Python scalars
bool b = (bool)oneD; // true: bool() reads the only element of a one-element array of any rank
```

This matches NumPy 2.x's strict behavior: `int(np.array([42]))` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`, while `bool(np.array([42]))` is NumPy's truth test (one element of any rank; a larger or empty array raises `ValueError`).

---

## Special Values

### NaN, Infinity (floating-point types)

`Half`, `Single`, and `Double` have IEEE 754 special values. NumSharp preserves them exactly through array storage and scalar round-trips:

```csharp
var h = NDArray.Scalar<Half>(Half.NaN);
Half.IsNaN((Half)h);   // true

var d = NDArray.Scalar<double>(double.PositiveInfinity);
double.IsPositiveInfinity((double)d);  // true
```

`Decimal` has no NaN or infinity. A `Complex` value is NaN or infinite through its parts, as in NumPy's complex128: .NET's `Complex.NaN` and `Complex.Infinity` are such values, `np.isnan` is true when either part is NaN, and `np.isinf` when either part is infinite.

### Boundary values

`np.iinfo` and `np.finfo` give you the machine limits:

```csharp
np.iinfo(np.int8).min;           // -128
np.iinfo(np.int8).max;           // 127
np.iinfo(np.uint64).max;         // long.MaxValue (clamped to long)
np.iinfo(np.uint64).maxUnsigned; // 18446744073709551615 (true ulong.MaxValue)

np.finfo(np.float16).eps;              // 2^-10 = 0.0009765625
np.finfo(np.float16).smallest_normal;  // 2^-14
np.finfo(np.float64).max;              // double.MaxValue
```

`iinfo.max` is declared as `long`-for `uint64` its value is clamped to `long.MaxValue`. Use `maxUnsigned` (a `ulong`) to get the true 64-bit-unsigned max.

`np.finfo(np.complex128)` reports the **underlying float64 precision**, matching NumPy-its `dtype` property is `Double`, `bits == 64`, `precision == 15`. This is NumPy's convention: a complex number's precision is the precision of its real and imaginary components.

---

## NumPy Scalars

A NumPy scalar (`np.float64(1.5)`, the result of `a.sum()`) is a 0-dimensional `NDArray` in NumSharp. It computes,
promotes and prints as the array it is, and it follows NumPy's scalar rules: it is **read-only**, and it is not an
explicit 0-d array.

```csharp
var s = np.float64(1.5);        // NumPy's class call: a read-only 0-d float64 scalar
var t = np.int8("12");          // 12: text is parsed by Python's int()
var c = np.complex128(1, 2);    // (1+2j)
var u = np.str_(5);             // '5', a <U1 scalar
var z = np.float64();           // 0.0: no argument is the zero
var n = np.float64(null);       // nan: null is Python's None
var arr = np.float64(new object[] { 1, 2 });   // a list gives the ndarray [1., 2.], as in NumPy

s.flags.writeable;              // False (np.float64(1.5).flags.writeable is False)
np.isscalar(s);                 // True
np.isscalar(np.array(5));       // False: np.array(x) builds a 0-d ARRAY, not a scalar
```

Scalars come back where NumPy returns them: a full reduction (`a.sum()`), every ufunc and operator whose operands are
all 0-d (`np.add(np.array(1), 1)`, `np.sqrt(np.array(4.0))`, `a0 < b0`), and `np.clip`, `np.round`, `np.dot` / `np.vdot`
/ `np.inner`, `np.sinc`, `np.nan_to_num` and `np.interp` of 0-d input. A result written into `out=` is that array, as
given. Functions that build an array keep a 0-d ndarray: `np.where`, `np.copy`, `astype` and the views (`np.squeeze`,
`.real`, `.T`, `view`).

An element is a scalar too: an integer for every axis (`a[1, 2]`, `a[0]` of a 1-D array, `z[()]` of a 0-d array, a
matrix's `m[0, 1]`) reads a COPY of the element, which a later write to the array does not reach; write an element
through the setter (`a[1, 2] = v`). Any other index stays a view, and a 0-d view (`a[1, 2, ...]`, `z[...]`) is an array
that writes through. Indexing a scalar indexes a copy of its value, as NumPy's scalar does: `s[...]` is a writeable 0-d
array, `s[np.newaxis]` a 1-element array, and a key a 0-d array refuses raises `IndexError: invalid index to scalar
variable.`

```csharp
var a = np.arange(5.0);
var x = a[2];                   // 2.0, read-only
a[2] = 99.0;                    // x is still 2.0
var s = a.sum();
var v = s[Slice.Ellipsis];      // a writeable 0-d array over a copy of s
```

Every NumPy scalar class with a NumSharp dtype has a call: `np.bool` / `np.bool_`, `np.int8` … `np.int64`, `np.uint8` …
`np.uint64`, the C-type names (`np.byte`, `np.short`, `np.intc`, `np.int_`, `np.long`, `np.longlong`, `np.intp` and their
unsigned twins `np.ubyte` … `np.uintp`), `np.float16` / `np.half`, `np.float32` / `np.single`, `np.float64` /
`np.double`, `np.longdouble`, `np.complex128` / `np.cdouble`, `np.clongdouble`, `np.str_` and `np.bytes_`. Without
parentheses each name is still the `DType` it always was (`np.zeros(3, np.float64)`).

The conversion is NumPy's constructor:

- A C# number is NumPy's Python scalar, stored with NEP 50's range check: `np.int8(300)` raises `OverflowError: Python
  integer 300 out of bounds for int8`, and a float is truncated first (`np.int8(3.9)` is 3).
- A NumPy scalar or an array is cast: `np.uint8(np.int64(-1))` is 255, `np.int8((Half)300)` is 44.
- Text is parsed by Python's `int()` / `float()` / `complex()` grammar, with their errors.
- `np.str_(x)` is Python's `str(x)` (`np.str_(new object[] { 1, 2 })` is `'[1, 2]'`); `np.bytes_(x)` is Python's
  `bytes(x)` first (`np.bytes_(5)` is five NUL bytes) and NumPy's conversion where `bytes()` refuses (`np.bytes_(1.5)` is
  `b'1.5'`). Both take Python's `encoding` / `errors` arguments.
- `np.complex128(real, imag)` takes two real numbers.

The call needs C# 14 at the call site (any target framework with the .NET 10 SDK). C# 12 code writes the class's own
spelling, `NDScalarClassCalls.float64(x)`. The call is an extension member, so three forms do not compile: a bare
`float64(x)` under `using static NumSharp.np;` (import `NDScalarClassCalls` instead), a method-group conversion
`Func<object, NDArray> f = np.float64;` (use a lambda), and a `dynamic` argument (cast it to `object`).

`NDArray.Scalar(value)` and `NDArray.Scalar(value, dtype)` build the same read-only scalar. A 0-d array you write into is
an array: create it with `np.array(x)` or `np.zeros(Shape.Scalar, dtype)`. Converting a scalar into an array gives a new
writeable array, as in NumPy: `np.asarray(s)` and `np.array(s)` copy it, and `np.asarray(s, copy: false)` raises.

```csharp
var dt = np.array(1e-4f);       // a mutable 0-d parameter (np.evaluate reads it per call)
dt.SetValue(5e-5f);             // fine: an array
NDArray.Scalar(1e-4f).SetValue(5e-5f);   // ValueError: assignment destination is read-only
```

A 0-d array that `np.array` / `np.asarray` did not build is read as a scalar (`np.isscalar(np.zeros(()))` is True, where
NumPy's is False); it stays writeable.

---

## Scalar Types and Type Checks

NumPy's type tests are written against its scalar **classes**: `np.issubdtype(a.dtype, np.integer)`,
`isinstance(x, np.floating)`, `issubclass(t, np.number)`. NumSharp has the same classes, as `NDScalarType` objects with
NumPy's names:

```
object (Python's)
 \-> generic
      +-> bool
      +-> number
      |    +-> integer
      |    |    +-> signedinteger     int8 int16 intc° int32 int64 longlong° timedelta64
      |    |    \-> unsignedinteger   uint8 uint16 uintc° uint32 uint64 ulonglong° (char)
      |    \-> inexact
      |         +-> floating          float16 float32 float64 (+ float) longdouble (decimal)
      |         \-> complexfloating   complex64* complex128 (+ complex) clongdouble
      +-> flexible
      |    +-> character            bytes_ (+ bytes) str_ (+ str)
      |    \-> void*
      +-> datetime64
      \-> object_*

* a NumPy class NumSharp has no dtype for     (+ x) a second base: Python's class x
° a class of its own on one platform only: intc / uintc on Windows, longlong / ulonglong on Linux and macOS
```

`np.generic`, `np.number`, `np.integer`, `np.signedinteger`, `np.unsignedinteger`, `np.inexact`, `np.floating`,
`np.complexfloating`, `np.flexible` and `np.character` are the abstract classes. A concrete class is spelled by its
dtype, as everywhere else in NumSharp (`np.float64` is a `DType`), and a dtype converts to its class implicitly, which
is NumPy's `dtype.type`. NumSharp's own `decimal` and `char` dtypes sit in the family of their kind.

The hierarchy is NumPy's whole one. Python's built-in classes are part of it (`NDScalarType.PyObject`, `PyInt`,
`PyBool`, `PyFloat`, `PyComplex`, `PyStr`, `PyBytes`, `PyMemoryView`): `numpy.generic` derives from `object`,
`numpy.float64` also from `float`, `numpy.complex128` from `complex`, and `numpy.bytes_` / `numpy.str_` from `bytes` /
`str`, so every `__mro__` is NumPy's tuple. The NumPy classes NumSharp has no dtype for, `np.complex64` (`np.csingle`),
`np.object_` and `np.void`, are classes too: every type test sees them, and only the conversion to a dtype refuses
(`np.dtype(np.complex64)` raises `NotSupportedException`).

```csharp
np.issubdtype(a.dtype, np.integer);          // the most common NumPy type test
np.issubdtype(np.float32, np.floating);      // True
np.issubdtype(np.bool_, np.number);          // False: bool is not a number in NumPy 2.x
np.issubdtype("m8[s]", np.signedinteger);    // True: timedelta64 is a signed integer

np.floating.__instancecheck__(a.mean());     // isinstance(a.mean(), np.floating)
np.integer.__subclasscheck__(a.dtype.type);  // issubclass(a.dtype.type, np.integer)

NDScalarType t = a.dtype;                    // a.dtype.type: <class 'numpy.float64'>
t.__mro__;                                   // [float64, floating, inexact, number, generic, float, object]
np.floating.__subclasses__();                // [float16, float32, float64, longdouble, decimal]
DType d = (DType)t;                          // np.dtype(t): back to the descriptor

NDScalarType.PyFloat.__subclasscheck__(np.float64);   // issubclass(np.float64, float): True
NDScalarType.PyFloat.__subclasscheck__(np.float32);   // False
np.issubdtype(NDScalarType.PyFloat, np.floating);     // np.issubdtype(float, np.floating): True
np.issubdtype(NDScalarType.PyBool, NDScalarType.PyInt); // np.issubdtype(bool, int): False (numpy.bool vs numpy.int64)
np.issubdtype(np.complex64, np.complexfloating);      // True
np.finfo(np.complex64).dtype;                         // float32
```

Rules worth knowing:

- **An abstract class is not a dtype.** NumPy raises `TypeError` for `np.dtype(np.integer)` and for any `dtype=` that
  receives one; in NumSharp `np.dtype(np.integer)` raises NumPy's text and passing `np.integer` where a `DType` is
  expected does not compile. The conversion from a class to a dtype is explicit for that reason.
- **A string argument of `np.issubdtype` is a dtype string**, as in NumPy: `np.issubdtype(np.float32, "float")` is False,
  because `"float"` is float64. Families are classes (`np.floating`), never strings.
- **`np.isdtype` compares classes by identity**, not by subclass: `np.isdtype(np.int64, np.integer)` is False. Use
  `np.issubdtype` for the hierarchy, or `np.isdtype`'s kind names (`"integral"`, `"real floating"`, ...).
- **A C# value is a NumPy scalar and a Python object.** NumSharp's scalars are 0-d arrays, not objects of scalar classes,
  so `__instancecheck__` treats a C# value of a dtype's element type as a scalar of that dtype (a `double` is a
  `numpy.float64`), and a 0-d array that is NumPy's scalar (`a.sum()`, an element `a[0]`, `NDArray.Scalar(...)`) as well. A C# value is also the Python object it stands for (an `int` is a Python
  `int`, a `double` a `float`, a `string` a `str`), so `NDScalarType.PyFloat.__instancecheck__(1.5)` is True. A 0-d
  array follows NumPy's classes exactly: a float64 scalar is a Python `float`, an int64 one is not an `int`. An explicit
  `np.array(5)` is a 0-d ndarray, not a scalar, exactly as in NumPy.
- **A Python built-in class is no NumPy dtype.** `np.issubdtype` reads one as NumPy does, as the class of
  `np.dtype(cls)` (`float` is numpy.float64, `int` numpy.int64, `object` numpy.object_), while `np.isdtype` refuses it
  with NumPy's `TypeError`.

The type tables are there too: `np.typecodes` (NumPy's type-code strings), `np.sctypeDict` (NumPy's type names, each
mapped to the NumSharp dtype; the names of `complex64`, `object_` and `void`, which have no dtype, are absent),
`np.ScalarType` (every scalar class, Python's seven first, in NumPy's order: 31 classes), and `np.True_` / `np.False_`
(C# `bool`s).

### C-type classes

NumPy keeps a class for each C type, and where two C types have the same size and kind, two classes share one
storage format: the long double (`np.longdouble`, `'g'`) beside float64 and its complex twin (`np.clongdouble`, `'G'`)
beside complex128 everywhere; the C `int` (`np.intc`, `'i'`) beside int32 on Windows, where int32 is the C `long`; the
C `long long` (`np.longlong`, `'q'`) beside int64 on Linux and macOS, where int64 is the C `long`. NumSharp has the same
classes, each stored like its twin:

```csharp
np.dtype("g");                                // dtype('float64'), char 'g', type longdouble
np.longdouble == np.float64;                  // True: equal descriptors (and equal hashes)
np.issubdtype(np.float64, np.longdouble);     // False: different classes, as in NumPy
np.can_cast(np.longdouble, np.float64, "no"); // True: the same bytes
np.promote_types(np.float64, np.longdouble);  // longdouble
np.zeros(3, np.longdouble).dtype.@char;       // 'g': creation, views, copies, joins and gathers keep the class
```

Two differences from NumPy remain. Computing routines (`x + x`, `np.sum`, `np.clip`, ...) return the class of the
storage format, float64 for a long double array, where NumPy keeps the long double; the values are the same, and the
result's dtype still compares equal to `np.longdouble`. And NumSharp's long double is 8 bytes on every platform, as on
NumPy's Windows build: on Linux, NumPy's long double is the 16-byte extended type (`float128`), which NumSharp does not
have.

### `np.typing`

`numpy.typing`'s aliases are annotations, and C# annotations are C# types, so `np.typing` names the type NumSharp uses
in each role: `np.typing.NDArray` is `NDArray<T>` (`npt.NDArray[np.float64]` is `NDArray<double>`), `np.typing.ArrayLike`
is `NDArray` (every array_like parameter, which accepts scalars and C# arrays implicitly) and `np.typing.DTypeLike` is
`DType` (every `dtype=` parameter). `np.typing.NBitBase` exists but is deprecated, as in NumPy 2.3.

```csharp
// def scale(x: npt.ArrayLike, dtype: npt.DTypeLike = None) -> npt.NDArray[np.float64]:
static NDArray scale(NDArray x, DType dtype = null) => np.asarray(x, dtype) * 2.0;
```

---

## Type Promotion

When you combine two dtypes (e.g., `int32 + float32`), NumSharp picks a result dtype following NumPy 2.x rules (NEP 50). The result type is the smallest type that can hold both inputs' values:

```csharp
var a = np.array([1, 2, 3]);
var b = np.array([1.5, 2.5, 3.5]);
var c = a + b;
c.dtype;  // Double - int32 + float64 promotes to float64
```

Quick reference for common pairs:

| Left | Right | Result | Why |
|------|-------|--------|-----|
| `int8` | `uint8` | `int16` | both widen to fit signed range |
| `int32` | `uint32` | `int64` | can't fit uint32 in int32 |
| `int32` | `uint64` | `float64` | no common integer type |
| `float16` | `int16` | `float32` | precision of float16 insufficient |
| `float16` | `float32` | `float32` | higher precision wins |
| numeric | `complex128` | `complex128` | complex absorbs numeric operands; text follows descriptor promotion |

For any pair, ask `np.promote_types(a, b)` or `np.result_type(...)`; the full 15×15 table is `_nptypemap_arr_arr` in `src/NumSharp.Core/Logic/np.find_common_type.cs`. Tests in `test/NumSharp.Tests/Casting/DtypeConversionMatrixTests.cs` verify every pair against NumPy 2.4.2.

`np.find_common_type` is not this rule. It is NumPy 1.26.4's older function (deprecated in 1.25, removed in 2.0), ported as it was: the arrays' type is the first entry of the list `?bBhHiIlLqQefdgFDGO` that every array type casts to safely, and a scalar type changes it only when it is of a higher kind. So `np.find_common_type(new[] { np.int8 }, new[] { np.int64 })` is int8, where `np.result_type(np.int8, np.int64)` is int64.

String promotion also accounts for the descriptor's width and missing-value policy: `S3` with `U2` promotes to `U3`; `T` with `U` promotes to `T`; `T` with `S` or a numeric dtype has no common dtype. Fixed-width text can promote with a strong numeric dtype to a wide enough string, but an individual ufunc still needs a loop for its operands. Promotion does not make `text + number` numeric addition or implicit concatenation.

```csharp
np.promote_types(np.dtype("S3"), np.dtype("U2")); // dtype('<U3')
np.promote_types(np.dtype("T"), np.dtype("U2")); // StringDType()
np.promote_types(np.dtype("T"), np.int32);       // DTypePromotionError
```

Explicitly cast numeric values to text before concatenating them. The [text guide](fundamentals/strings-and-bytes.md#dtype-integration) describes cast safety, truncation and encoding errors.

For the deeper story on how NumPy 2.x promotion differs from NumPy 1.x, see [NumPy Compliance](compliance.md).

---

## Breaking Changes

If you're upgrading from an earlier NumSharp, be aware of these dtype-related changes:

### Text and descriptor migration

- `NDArray x = "[1, 2, 3]"` now holds one text element; it no longer parses a numeric list. Use explicit numeric input or `np.fromstring` for delimited numbers.
- NumPy `<U1` files and Python arrays load/import as **U1**, not `Char`. A UCS-4 code point is four bytes; a C# char is two.
- `np.@char` is the module object. It still converts to the Char descriptor in a dtype parameter, but use `np.dtype(typeof(char))` for descriptor member access.
- `arr.dtype` and numeric `np.*` dtype aliases are `DType`, not `System.Type`. Use `.type` when a CLR type is required, and keep the descriptor when allocating/casting text.

### `np.byte` now returns `sbyte` (int8), not `byte` (uint8)

NumPy convention: `np.byte = int8` (signed, C `char`-style). NumSharp now follows NumPy.

```csharp
// Before:
Type t = np.@byte;  // typeof(byte) - uint8

// After:
DType t = np.@byte;  // int8; t.type == typeof(sbyte)
// If you meant uint8, use:
DType u = np.uint8;  // or np.ubyte
```

### `np.complex64` is a class without a dtype

Previously it was a silent alias for `np.complex128`. It is now NumPy's scalar class (an `NDScalarType`, as are
`np.csingle`, `np.object_` and `np.void`): type tests over it work, and converting it to a dtype raises
`NotSupportedException` with a message pointing users to `np.complex128`. `np.zeros(3, np.complex64)` does not compile
(a class converts to a dtype only explicitly). Same refusal for `np.dtype("complex64")` / `"F"` / `"c8"`.

### `np.intp` / `np.uintp` now return `long` / `ulong` (not `IntPtr` / `UIntPtr`)

Previously these were `typeof(nint)` / `typeof(nuint)`-which have `NPTypeCode.Empty` and broke `np.zeros(shape, np.intp.GetTypeCode())`. They now match `np.int64` / `np.uint64` on 64-bit platforms (and `np.int32` / `np.uint32` on 32-bit).

### Complex → real scalar casts now throw `TypeError`

Previously they silently dropped the imaginary part. Now they throw, matching Python's `int(complex)` / `float(complex)` semantics. Use `np.real(arr)` explicitly if that's what you want.

### `np.dtype("int")` now returns `Int64` (pointer-sized), not `Int32`

NumPy 2.x made `int` an alias for `intp` (pointer-sized). NumSharp now follows. If you want fixed 32-bit, use `np.int32` / `np.dtype("int32")` / `"i4"`.

---

## Invalid Dtype Strings

`np.dtype(s)` throws NumPy's `TypeError` (`data type 'xyz' not understood`) for any string that isn't a valid NumPy
dtype:

```csharp
np.dtype("xyz");       // throws - not a dtype
np.dtype("f32");       // throws - f is 2/4/8 bytes only (and 16 where the platform's long double is)
np.dtype("i3");        // throws - i is 1/2/4/8 bytes only
np.dtype("?1");        // throws - ? is not sized
np.dtype("   i4");     // throws - no whitespace trimming
```

It throws `NotSupportedException` instead for NumPy dtypes NumSharp doesn't implement:

```csharp
np.dtype("V7");        // throws - raw void bytes
np.dtype("object");    // throws - object dtype
np.dtype("3f8");       // throws - a sub-array of three float64
np.dtype("i4,f8");     // throws - a structured dtype
```

A comma string NumPy refuses throws NumPy's error instead:

```csharp
np.dtype("3000000000f8");  // ValueError - invalid shape in fixed-type tuple: dimension does not fit into a C int.
np.dtype("1 2f8");         // SyntaxError - invalid syntax (<unknown>, line 1)
```

`np.dtype("S10")` and `np.dtype("U32")` are NumPy's fixed-width string dtypes, which NumSharp supports, and so are
their comma-string spellings `np.dtype("10S")` and `np.dtype("32U")`. A `datetime64` descriptor (`np.dtype("M8")`) parses, but `np.zeros(3, np.dtype("M8[s]"))` throws until its storage lands.

This is strict on purpose: silently accepting "close enough" dtype strings produces hard-to-debug corruption downstream.

---

## Common Patterns

### Loading binary data with a known dtype

```csharp
byte[] raw = File.ReadAllBytes("sensor.bin");
var readings = np.frombuffer(raw, np.float16);  // interpret as float16
```

### Making arrays with matching dtype

```csharp
var template = np.array(new[] { "cat", "bird" }, "U8");
var sameType = np.ones(template.shape, template.dtype); // keeps U8, not just NPTypeCode.String
var sameShapeAndType = np.ones_like(template);
```

### Force-cast vs safe-cast

```csharp
// Force: the default casting "unsafe" converts every value, so 300.0 wraps
var forced = np.array([300.0]).astype(NPTypeCode.Byte);
// forced[0] == 44 (300 wrapped modulo 256)

// Safe: casting "safe" checks the dtype pair, not the values: float64 never casts safely to uint8
np.array([300.0]).astype(np.uint8, casting: "safe");   // NumPy's order: astype(dtype, order, casting, copy)
// InvalidCastException (NumPy's TypeError): Cannot cast array data from dtype('float64') to dtype('uint8') according to the rule 'safe'
```

Neither rule checks a value against the target's range: use `np.can_cast` to test a dtype pair, and compare against `np.iinfo` before a cast that must not wrap.

---

## API Reference

### Dtype specification (three forms, all equivalent)

| Form | Example | When to use |
|------|---------|-------------|
| `NPTypeCode` enum | `NPTypeCode.Int32` | Internal code, compile-time known |
| `DType` via `np.*` | `np.int32`, `np.complex128` | Idiomatic user code |
| String via `np.dtype()` | `np.dtype("i4")`, `np.dtype("complex128")` | Runtime / config-driven |

### Introspection

On `NDArray` itself the key properties are `.dtype` (a `DType`, NumPy's `ndarray.dtype`) and `.typecode` (an `NPTypeCode`). `np.float64` and the other `np.*` dtype names are `DType` descriptors too, so `arr.dtype == np.float64` compares descriptors.

| Expression | Returns | Notes |
|------------|---------|-------|
| `arr.dtype` | `DType` | The descriptor (`int32`, `float64`, ...) |
| `arr.dtype.type` | `System.Type` | The C# element type (e.g. `typeof(int)`) |
| `(NDScalarType)arr.dtype` | `NDScalarType` | NumPy's `dtype.type`: the scalar class (`<class 'numpy.int32'>`); see [Scalar Types and Type Checks](#scalar-types-and-type-checks) |
| `arr.typecode` | `NPTypeCode` | Enum value (`NPTypeCode.Int32`, etc.) |
| `arr.itemsize` / `arr.dtype.itemsize` | `int` | Bytes per element; for `T`, the 16-byte cell, not the text payload |
| `arr.typecode.SizeOf()` | `int` | Numeric lane size only; cannot describe the three string families |
| `arr.typecode.AsNumpyDtypeName()` | `string` | e.g. `"int32"`, `"float16"`, `"complex128"` |
| `np.dtype("int32")` | `DType` | Full descriptor object |
| `np.dtype("int32").typecode` | `NPTypeCode` | Same as `arr.typecode` would be |
| `np.dtype("int32").itemsize` | `int` | Bytes |
| `np.dtype("int32").kind` | `char` | NumPy's kind: `'b'`/`'i'`/`'u'`/`'f'`/`'c'`/`'M'`/`'m'` |
| `np.dtype("int32").@char` | `char` | NumPy type char (e.g. `'b'`, `'e'`; int32 is `'l'` on Windows, `'i'` on Linux and macOS, as in NumPy) |
| `np.dtype("int32").name` | `string` | NumPy's dtype name (`"int32"`, `"datetime64[s]"`) |
| `np.dtype("int32").byteorder` | `char` | `'='` native, `'\|'` for 1-byte dtypes, `'>'` / `'<'` when given explicitly non-native (`np.dtype(">i4")`) |
| `DType d = typeof(int);` | `DType` | A C# type, an `NPTypeCode` or a dtype string converts to `DType` implicitly |
| `np.dtype((DType)null)` | `DType` | NumPy's `np.dtype(None)`: the float64 descriptor, for every `np.dtype` overload and every `np.result_type` operand. Write the null typed: a bare `null` matches the string and `Type` overloads alike |
| `InfoOf<T>.Size` | `int` | Byte size of an unmanaged CLR element type; use `arr.itemsize` for a string dtype |
| `InfoOf<T>.NPTypeCode` | `NPTypeCode` | `NPTypeCode` for CLR type `T` |

### Machine limits

| Function | Returns | Works for |
|----------|---------|-----------|
| `np.iinfo(dtype)` | `iinfo` with `bits`, `min`, `max`, `kind` | integer dtypes + Boolean + Char |
| `np.finfo(dtype)` | `finfo` with `bits`, `eps`, `epsneg`, `min`, `max`, `precision`, `resolution`, `maxexp`, `minexp`, `nmant`, `iexp` / `nexp`, `machep`, `negep`, `smallest_normal`, `smallest_subnormal` (NumPy's values; `2**minexp == smallest_normal`) | `Half`, `Single`, `Double`, `Decimal`, `Complex` |

### Exceptions

| Exception | When |
|-----------|------|
| `TypeError` | A dtype string NumPy rejects too (`data type 'xyz' not understood`, a removed alias such as `'bool8'`) |
| `ValueError` / `SyntaxError` | A comma string NumPy refuses (`invalid shape in fixed-type tuple.`, `format number 2 of "f8,x y" is not recognized`; a repeat count that is not a Python literal) |
| `NotSupportedException` | A NumPy dtype NumSharp doesn't implement (`V`/`O`/`complex64`/…, a sub-array or structured comma string); an array of a descriptor with no storage yet (`M8`, `m8`); the conversion of `np.complex64` / `np.csingle` / `np.object_` / `np.void` to a dtype |
| `StringComputationNotSupportedException` | A delegate node of the expression engine (`NDExpr.Call` inside `np.evaluate`) given a text argument: its parameters are numbers. Every other node computes on text as NumPy does, or raises NumPy's own error |
| `DTypePromotionError` | No common dtype, such as `StringDType` with a numeric dtype or `S` |
| `UnicodeEncodeError` / `UnicodeDecodeError` | An invalid encoding conversion, such as non-ASCII `U` → `S` or invalid UTF-8 `S` → `T` |
| `TypeError` | Complex → non-complex scalar cast (`(int)complexScalar`, etc.) |
| `TypeError` | NDArray → numeric scalar cast on a non-0-d array (NumPy 2.x's strict 0-d requirement and text) |
| `ValueError` | `(bool)` of an array with more than one element or none (NumPy's truth-value texts) |

---

## Related Reading

- [NumPy Compliance & Compatibility](compliance.md) - Type promotion, NEP 50, broader NumPy 2.x parity
- [Broadcasting](broadcasting.md) - How shapes combine across operations (dtype-independent)
- [Buffering, Arrays and Unmanaged Memory](buffering.md) - How dtype affects memory layout
- [IL Kernel Generation in NumSharp](il-generation.md) - Which dtypes get SIMD acceleration and why
- [NumPy data types user guide](https://numpy.org/doc/stable/user/basics.types.html) - NumPy's own dtype reference
