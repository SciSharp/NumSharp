# Dtypes in NumSharp

Every array in NumSharp has a **dtype**—a data type that determines what kind of values the array stores, how many bytes each element takes, and which operations are valid. When you write `np.zeros(10, np.int32)`, the `np.int32` is the dtype. When you call `arr.astype(np.float64)`, you're converting to a different dtype.

This page covers the 15 dtypes NumSharp supports, how they map to NumPy's types, how to refer to them in code, and the places where NumSharp's behavior diverges from NumPy (and why).

---

## The 15 Supported Dtypes

NumSharp supports every numeric dtype NumPy defines, plus a few .NET-specific ones:

| NPTypeCode | C# Type | NumPy Equivalent | Bytes | Kind | SIMD |
|------------|---------|------------------|-------|------|------|
| `Boolean`  | `bool`                     | `bool`       | 1  | `?` † | Limited |
| `SByte`    | `sbyte`                    | `int8`       | 1  | `i` | Yes |
| `Byte`     | `byte`                     | `uint8`      | 1  | `u` | Yes |
| `Int16`    | `short`                    | `int16`      | 2  | `i` | Yes |
| `UInt16`   | `ushort`                   | `uint16`     | 2  | `u` | Yes |
| `Int32`    | `int`                      | `int32`      | 4  | `i` | Yes |
| `UInt32`   | `uint`                     | `uint32`     | 4  | `u` | Yes |
| `Int64`    | `long`                     | `int64`      | 8  | `i` | Yes |
| `UInt64`   | `ulong`                    | `uint64`     | 8  | `u` | Yes |
| `Half`     | `System.Half`              | `float16`    | 2  | `f` | None |
| `Single`   | `float`                    | `float32`    | 4  | `f` | Yes |
| `Double`   | `double`                   | `float64`    | 8  | `f` | Yes |
| `Decimal`  | `decimal`                  | *no equiv*   | 32 ‡ | `f` | None |
| `Complex`  | `System.Numerics.Complex`  | `complex128` | 16 | `c` | None |
| `Char`     | `char`                     | *no equiv*   | 1 ‡ | `S` | None |

**Bytes column reports `NPTypeCode.SizeOf()` / `DType.itemsize`** — what NumSharp actually returns to your code. Two of these diverge from both NumPy and the underlying .NET type:
- † `Boolean.kind` is `'?'` in NumSharp; NumPy uses `'b'`. (NumSharp stores the type-char in the `kind` slot for bool.)
- ‡ **`Decimal.itemsize == 32` and `Char.itemsize == 1`** are NumSharp reporting bugs. The actual .NET memory footprint is 16 bytes for `decimal` and 2 bytes for `char`. `InfoOf<decimal>.Size == 16` and `InfoOf<char>.Size == 2` give you the correct values. Storage allocation uses the correct .NET size; only the `DType.itemsize` property is wrong.

**Half**, **SByte**, and **Complex** are the newest additions—see [Breaking Changes](#breaking-changes) below.

**Decimal** and **Char** are NumSharp-specific types with no NumPy counterpart—see [NumSharp-Specific Types](#numsharp-specific-types-decimal-and-char) for how they behave and when to use them.

---

## Referring to Dtypes in Code

There are three ways to name a dtype:

### 1. `NPTypeCode` enum (fastest, internal-style)

```csharp
var arr = np.zeros(new Shape(10), NPTypeCode.Int32);
var cplx = np.zeros(new Shape(2, 3), NPTypeCode.Complex);
```

Use this when you want zero overhead and the type is known at compile time. `NPTypeCode` values are stable enum constants.

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

Use strings when the dtype is dynamic (from config, JSON, a file header). NumSharp accepts every dtype string NumPy 2.x accepts—see [the parsing table](#dtype-string-parsing) below.

---

## Dtype String Parsing

NumSharp's `np.dtype(string)` mirrors NumPy 2.x exactly for every form NumSharp has a type for, and throws `NotSupportedException` for forms it doesn't.

### Single-char codes

| String | Resolves to | Notes |
|--------|-------------|-------|
| `"?"` | `Boolean` | |
| `"b"` | `SByte` | NumPy: int8 (signed, C `char` convention) |
| `"B"` | `Byte` | NumPy: uint8 |
| `"h"` / `"H"` | `Int16` / `UInt16` | |
| `"i"` / `"I"` | `Int32` / `UInt32` | always 32-bit per NumPy spec |
| `"l"` / `"L"` | **platform-dependent** | C `long` — see [Platform-Dependent Types](#platform-dependent-types) |
| `"q"` / `"Q"` | `Int64` / `UInt64` | C `long long`—always 64-bit |
| `"p"` / `"P"` | pointer-sized | see [Platform-Dependent Types](#platform-dependent-types) |
| `"e"` | `Half` | NumPy: float16 |
| `"f"` | `Single` | NumPy: float32 |
| `"d"` / `"g"` | `Double` | `g` (long double) collapses to float64 |
| `"D"` / `"G"` | `Complex` | `G` (long-double complex) collapses to complex128 |
| `"F"` | **throws** | complex64 — NumSharp only has complex128 |
| `"c"` | **throws** | NumPy: `S1` (1-byte string)—not supported |
| `"S"` / `"U"` / `"V"` / `"O"` / `"M"` / `"m"` | **throws** | bytestring, unicode, void, object, datetime64, timedelta64—not supported |

### Sized variants

| String | Resolves to | Notes |
|--------|-------------|-------|
| `"b1"` | `Boolean` | |
| `"i1"` / `"u1"` | `SByte` / `Byte` | |
| `"i2"` / `"u2"` / `"f2"` | `Int16` / `UInt16` / `Half` | |
| `"i4"` / `"u4"` / `"f4"` | `Int32` / `UInt32` / `Single` | |
| `"i8"` / `"u8"` / `"f8"` / `"c16"` | `Int64` / `UInt64` / `Double` / `Complex` | |
| `"c8"` | **throws** | complex64—not supported |
| `"i3"`, `"f3"`, `"f16"`, `"b4"`, ... | **throws** | invalid NumPy dtype strings |

### Named forms

| String | Resolves to |
|--------|-------------|
| `"int8"` / `"sbyte"` | `SByte` |
| `"uint8"` / `"ubyte"` | `Byte` |
| `"byte"` | **`SByte`** (NumPy convention: `byte` = int8, signed) |
| `"int16"` / `"short"` | `Int16` |
| `"uint16"` / `"ushort"` | `UInt16` |
| `"int32"` / `"intc"` | `Int32` |
| `"uint32"` / `"uintc"` | `UInt32` |
| `"int64"` / `"longlong"` | `Int64` |
| `"uint64"` / `"ulonglong"` | `UInt64` |
| `"int"` / `"int_"` / `"intp"` | **pointer-sized**—int64 on 64-bit |
| `"uint"` / `"uintp"` | **pointer-sized**—uint64 on 64-bit |
| `"long"` / `"ulong"` | **platform-dependent** C long/ulong |
| `"float16"` / `"half"` | `Half` |
| `"float32"` / `"single"` | `Single` |
| `"float64"` / `"double"` | `Double` |
| `"float"` | `Double` (NumPy: `float` → float64) |
| `"complex128"` / `"complex"` | `Complex` |
| `"complex64"` | **throws** |
| `"bool"` / `"bool_"` | `Boolean` |

Byte-order prefixes (`<`, `>`, `=`, `|`) are accepted and ignored: `np.dtype("<i4")` works just like `np.dtype("i4")`. NumSharp is always host-endian.

---

## Class-level Aliases

Every dtype name from NumPy's Python API is available as a `public static readonly Type` on `np`:

```csharp
Type t1 = np.int32;        // typeof(int)
Type t2 = np.float16;      // typeof(Half)
Type t3 = np.complex128;   // typeof(Complex)
Type t4 = np.@long;        // platform-dependent C long
Type t5 = np.intp;         // pointer-sized int
```

Names that clash with C# keywords are prefixed with `@`: `np.@byte`, `np.@short`, `np.@long`, `np.@ushort`, `np.@ulong`, `np.@bool`, `np.@uint`, `np.@double`, `np.@decimal`.

The class-level aliases and the string parser always resolve to the same .NET type—this invariant is guaranteed:

```csharp
// These are always true on every platform:
np.int32 == np.dtype("int32").type
np.@long == np.dtype("long").type
np.float16 == np.dtype("float16").type
```

---

## Complex: Only 128-bit Is Supported

NumSharp uses `System.Numerics.Complex`, which is two 64-bit floats—equivalent to NumPy's **`complex128`**. There is no 32-bit variant.

```csharp
// These all resolve to Complex (complex128):
np.complex128;
np.complex_;
np.cdouble;           // NumPy alias for complex128
np.clongdouble;       // NumPy: long-double complex, collapses to complex128
np.dtype("complex");  // NumPy 2.x default complex is complex128
np.dtype("D");
np.dtype("c16");
np.dtype("G");

// These all throw NotSupportedException:
np.complex64;
np.csingle;              // NumPy alias for complex64
np.dtype("complex64");
np.dtype("F");
np.dtype("c8");
```

Why throw instead of silently widening to complex128? Because quietly upgrading a user's `complex64` request to `complex128` doubles their memory use without telling them. If you need complex arrays in NumSharp, use `complex128` explicitly—the throw is there to prevent surprise.

**Converting from complex arrays to real arrays:** `(int)complexScalar` and `(float)complexScalar` throw `TypeError` ("can't convert complex to int"), matching Python's built-in behavior. NumPy 2.x emits a `ComplexWarning` and drops the imaginary part; NumSharp has no warning system, so we treat this as a hard error. If you want the real part, use `np.real(arr)` first.

```csharp
var c = NDArray.Scalar<Complex>(new Complex(3, 4));
var x = (int)c;        // throws TypeError
var r = (int)np.real(c);  // 3 — explicit, unambiguous
```

---

## NumPy Types NumSharp Doesn't Support

NumPy has several dtype families that NumSharp deliberately does not implement. Attempting to construct or parse any of these throws `NotSupportedException` (never silent misbehavior):

| NumPy dtype | NumPy character | Why not in NumSharp |
|-------------|-----------------|---------------------|
| `complex64` | `F`, `c8` | NumSharp has only one complex type (`complex128`). Silently widening would double memory without asking. See [Complex: Only 128-bit Is Supported](#complex-only-128-bit-is-supported). |
| `bytes_` / `S` / `a` | `S`, `a`, `c` (=S1) | NumPy bytestrings are a variable-length null-terminated byte sequence type. Not a natural fit for .NET where `string` is UTF-16 and `byte[]` is a separate concept. Use .NET strings directly. |
| `str_` / `U` | `U` | NumPy unicode strings (UCS-4 fixed-width). Same reason—use `string` / `string[]`. |
| `void` / `V` | `V` | NumPy "raw bytes" scalar. No .NET equivalent; use `byte[]` or `Memory<byte>`. |
| `object` / `O` | `O` | NumPy boxed-Python-object arrays. Use `object[]` or `NDArray<object>` conceptually. |
| `datetime64` | `M`, `M8[ns]` etc. | Needs nanosecond-epoch semantics and unit metadata that NumSharp doesn't model. Use `DateTime[]` directly, or `long[]` with epoch seconds. |
| `timedelta64` | `m`, `m8[us]` etc. | Same reason as `datetime64`. Use `TimeSpan[]` or `long[]`. |
| Structured / record dtypes | `(...)` in dtype string | NumPy allows composite dtypes like `np.dtype([('x', 'f4'), ('y', 'i4')])` for heterogeneous records. NumSharp throws on any dtype string containing `(`. Use a struct array or multiple parallel `NDArray`s. |
| Sub-array dtypes | `('f4', (3,))` | NumPy dtype-with-subshape. Not supported. |

Every row above is tested in `test/NumSharp.UnitTest/Creation/DTypeStringParityTests.cs` with an `ExpectThrow` assertion. If you run into one of these in ported NumPy code, the exception message tells you which NumSharp alternative to use.

### Why Throw Instead of Silent Approximation?

A recurring temptation is to "do the nearest thing"—e.g., widen `complex64` to `complex128` or map `S10` to `string`. NumSharp refuses this because:

1. **Memory surprise**: doubling precision doubles allocation; a user loading a gigabyte of `complex64` data would unexpectedly use two gigabytes.
2. **Precision surprise**: downstream computations on the "wrong" type produce results the user didn't request.
3. **Signal clarity**: a `NotSupportedException` with a clear message ("use np.complex128 instead") is actionable. Silent widening is a ticking bug.

---

## NumSharp-Specific Types (Decimal and Char)

Two types in NumSharp have no NumPy equivalent. They exist for .NET-idiomatic use cases where NumPy's dtype set is too narrow.

### `Decimal` — 128-bit fixed-point

.NET's `System.Decimal` is a 16-byte fixed-point number with 28-29 significant digits. It's the right type for **money and financial computation** where binary floating-point's representation errors are unacceptable (`0.1 + 0.2 != 0.3` is a non-starter for an accounting ledger).

```csharp
var prices = np.array(new[] { 19.99m, 29.99m, 5.00m });
prices.typecode;                // NPTypeCode.Decimal
InfoOf<decimal>.Size;           // 16 (actual memory footprint)
var total = np.sum(prices);     // exact decimal sum, no float drift
```

**Characteristics:**
- `kind == 'f'` (float-like—it's a fractional type even though internally integer-based)
- No SIMD acceleration (decimal arithmetic is scalar-only; much slower than `double`)
- No IEEE special values: no NaN, no Infinity, no subnormals
- `np.finfo(NPTypeCode.Decimal)` works and returns limited info (bits=128, precision=28, no subnormals)
- Boundary values: `Decimal.MinValue` / `Decimal.MaxValue` (±79228162514264337593543950335)
- **Known quirk:** `NPTypeCode.Decimal.SizeOf()` and `DType.itemsize` both report `32` instead of the correct `16`. Use `InfoOf<decimal>.Size` for the true byte count.

**When to use:**
- Financial calculations (currency, tax, interest)
- Any scenario where exact decimal representation matters more than speed

**When NOT to use:**
- Scientific computing (`double` is faster and has wider range)
- SIMD-critical paths (no vectorization)
- Interop with NumPy/Python (no round-trip—NumPy has no decimal type)

### `Char` — 16-bit UTF-16 code unit

`System.Char` is a 2-byte Unicode UTF-16 code unit. NumSharp preserves it as a dtype mostly for arrays of characters where the type system benefits from knowing "these are characters, not shorts."

```csharp
var letters = np.array(new[] { 'a', 'b', 'c' });
letters.typecode;               // NPTypeCode.Char
InfoOf<char>.Size;              // 2 (actual memory footprint)
```

**Important:** NumSharp's `Char` is **not** the same as NumPy's `'c'` / `S1` (which is a 1-byte bytestring). They have different sizes, different encodings, different semantics. Porting NumPy bytestring code to NumSharp `Char` will almost always be wrong—use `byte` arrays for bytestring data and `string` for actual text.

**Characteristics:**
- `kind == 'S'` (bytestring-like category, chosen for NumPy roundtrip ergonomics despite the semantic difference)
- Treated as `ushort` for many operations (same byte width)
- Boundary values: `'\0'` (0) to `char.MaxValue` (65535)
- **Known quirk:** `NPTypeCode.Char.SizeOf()` and `DType.itemsize` both report `1` instead of the correct `2`. Use `InfoOf<char>.Size` for the true byte count. Storage allocation uses the correct 2-byte size.

**When to use:**
- Arrays of individual characters where type annotation matters
- Interop with APIs that treat char specifically

**When NOT to use:**
- Text data—use `string` or `string[]`
- Porting NumPy bytestring arrays—use `byte[]` with explicit encoding

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
| `np.longlong`, `"longlong"`, `"q"`, `"i8"` | `Int64` |
| `np.int32`, `"int32"`, `"i"`, `"i4"` | `Int32` |
| `np.int16`, `"int16"`, `"h"`, `"i2"` | `Int16` |

### Recommendation

If you want **portable** code across Windows and Linux, avoid `long`/`ulong`/`l`/`L`. Use explicit sized names:

```csharp
// Portable — same result on every platform:
var a = np.zeros(shape, np.int32);
var b = np.zeros(shape, np.int64);
var c = np.dtype("int64");

// Platform-dependent — different result on Win vs Linux:
var d = np.zeros(shape, np.@long);
var e = np.dtype("long");
```

This is the same guidance NumPy itself gives—see the [NumPy data types page](https://numpy.org/doc/stable/user/basics.types.html).

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
np.array(new[] { 1, 2, 3 });                    // dtype=int32 (from int[])
np.array(new[] { 1.0, 2.0 });                   // dtype=float64 (from double[])
np.array(new[] { (Half)1, (Half)2 });           // dtype=float16
np.array(new[] { new Complex(1,2), new Complex(3,4) });  // dtype=complex128
np.array(new sbyte[] { -1, 0, 1 });             // dtype=int8
```

### Converting between dtypes

Use `.astype()` for array-level conversions:

```csharp
var doubles = np.array(new[] { 1.5, 2.7, 3.9 });
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
var scalar = np.array(new[] { 42 })[0];  // 0-d view
int x = (int)scalar;                     // works

var oneD = np.array(new[] { 42 });
int y = (int)oneD;   // throws IncorrectShapeException (ndim == 1)
```

This matches NumPy 2.x's strict behavior: `int(np.array([42]))` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

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

`Decimal` and `Complex` have no NaN/Inf equivalents (Complex's real/imag components individually can be `double.NaN`, but there's no single `Complex.NaN`).

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

`iinfo.max` is declared as `long`—for `uint64` its value is clamped to `long.MaxValue`. Use `maxUnsigned` (a `ulong`) to get the true 64-bit-unsigned max.

`np.finfo(np.complex128)` reports the **underlying float64 precision**, matching NumPy—its `dtype` property is `Double`, `bits == 64`, `precision == 15`. This is NumPy's convention: a complex number's precision is the precision of its real and imaginary components.

---

## Type Promotion

When you combine two dtypes (e.g., `int32 + float32`), NumSharp picks a result dtype following NumPy 2.x rules (NEP 50). The result type is the smallest type that can hold both inputs' values:

```csharp
var a = np.array(new int[] { 1, 2, 3 });
var b = np.array(new[] { 1.5, 2.5, 3.5 });
var c = a + b;
c.dtype;  // Double — int32 + float64 promotes to float64
```

Quick reference for common pairs:

| Left | Right | Result | Why |
|------|-------|--------|-----|
| `int8` | `uint8` | `int16` | both widen to fit signed range |
| `int32` | `uint32` | `int64` | can't fit uint32 in int32 |
| `int32` | `uint64` | `float64` | no common integer type |
| `float16` | `int16` | `float32` | precision of float16 insufficient |
| `float16` | `float32` | `float32` | higher precision wins |
| any | `complex128` | `complex128` | complex absorbs |

For full 15×15 promotion rules see `np.find_common_type` (`src/NumSharp.Core/Logic/np.find_common_type.cs`). Tests in `test/NumSharp.UnitTest/Casting/DtypeConversionMatrixTests.cs` verify every pair against NumPy 2.4.2.

For the deeper story on how NumPy 2.x promotion differs from NumPy 1.x, see [NumPy Compliance](compliance.md).

---

## Breaking Changes

If you're upgrading from an earlier NumSharp, be aware of these dtype-related changes:

### `np.byte` now returns `sbyte` (int8), not `byte` (uint8)

NumPy convention: `np.byte = int8` (signed, C `char`-style). NumSharp now follows NumPy.

```csharp
// Before:
Type t = np.@byte;  // typeof(byte) — uint8

// After:
Type t = np.@byte;  // typeof(sbyte) — int8
// If you meant uint8, use:
Type t = np.uint8;  // or np.ubyte
```

### `np.complex64` now throws

Previously it was a silent alias for `np.complex128`. It now raises `NotSupportedException` with a message pointing users to `np.complex128`. Same for `np.dtype("complex64")` / `"F"` / `"c8"`.

### `np.intp` / `np.uintp` now return `long` / `ulong` (not `IntPtr` / `UIntPtr`)

Previously these were `typeof(nint)` / `typeof(nuint)`—which have `NPTypeCode.Empty` and broke `np.zeros(shape, np.intp.GetTypeCode())`. They now match `np.int64` / `np.uint64` on 64-bit platforms (and `np.int32` / `np.uint32` on 32-bit).

### Complex → real scalar casts now throw `TypeError`

Previously they silently dropped the imaginary part. Now they throw, matching Python's `int(complex)` / `float(complex)` semantics. Use `np.real(arr)` explicitly if that's what you want.

### `np.dtype("int")` now returns `Int64` (pointer-sized), not `Int32`

NumPy 2.x made `int` an alias for `intp` (pointer-sized). NumSharp now follows. If you want fixed 32-bit, use `np.int32` / `np.dtype("int32")` / `"i4"`.

---

## Invalid Dtype Strings

`np.dtype(s)` throws `NotSupportedException` (with a descriptive message) for any string that isn't a valid NumPy dtype:

```csharp
np.dtype("xyz");       // throws — not a dtype
np.dtype("f16");       // throws — f is 2/4/8 bytes only
np.dtype("i3");        // throws — i is 1/2/4/8 bytes only
np.dtype("?1");        // throws — ? is not sized
np.dtype("   i4");     // throws — no whitespace trimming
```

It also throws for NumPy dtypes NumSharp doesn't implement:

```csharp
np.dtype("S10");       // throws — bytestring
np.dtype("U32");       // throws — unicode string
np.dtype("M8");        // throws — datetime64
np.dtype("object");    // throws — object dtype
```

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
var template = np.zeros(shape, np.int8);
var sameType = np.ones(template.shape, template.typecode);  // template.typecode, not template.dtype.typecode
// or more concisely:
var sameType = np.ones_like(template);
```

### Force-cast vs safe-cast

```csharp
// Force: silently wraps/truncates — fastest
var forced = np.array(new[] { 300.0 }).astype(NPTypeCode.Byte);
// forced[0] == 44 (300 wrapped modulo 256)

// Safe: raise on overflow (if NumSharp had this; currently matches NumPy's behavior
// which wraps by default and requires explicit casting='safe' for stricter modes).
```

---

## API Reference

### Dtype specification (three forms, all equivalent)

| Form | Example | When to use |
|------|---------|-------------|
| `NPTypeCode` enum | `NPTypeCode.Int32` | Internal code, compile-time known |
| `Type` via `np.*` | `np.int32`, `np.complex128` | Idiomatic user code |
| String via `np.dtype()` | `np.dtype("i4")`, `np.dtype("complex128")` | Runtime / config-driven |

### Introspection

On `NDArray` itself the key properties are `.dtype` (a `System.Type`) and `.typecode` (an `NPTypeCode`). The `DType` class (with itemsize, kind, char, name, byteorder) is only returned by `np.dtype(string)`; construct it explicitly with `new DType(arr.dtype)` if you need those fields from an array.

| Expression | Returns | Notes |
|------------|---------|-------|
| `arr.dtype` | `System.Type` | The .NET type (e.g. `typeof(int)`)—NOT a `DType` object |
| `arr.typecode` | `NPTypeCode` | Enum value (`NPTypeCode.Int32`, etc.) |
| `arr.typecode.SizeOf()` | `int` | Bytes per element (see quirks table for Decimal/Char) |
| `arr.typecode.AsNumpyDtypeName()` | `string` | e.g. `"int32"`, `"float16"`, `"complex128"` |
| `np.dtype("int32")` | `DType` | Full descriptor object |
| `np.dtype("int32").type` | `System.Type` | Same as `arr.dtype` would be |
| `np.dtype("int32").typecode` | `NPTypeCode` | Same as `arr.typecode` would be |
| `np.dtype("int32").itemsize` | `int` | Bytes (via `typecode.SizeOf()`) |
| `np.dtype("int32").kind` | `char` | `'?'`/`'i'`/`'u'`/`'f'`/`'c'`/`'S'` (see ‡ below) |
| `np.dtype("int32").@char` | `char` | NumPy type char (e.g. `'i'`, `'b'`, `'e'`) |
| `np.dtype("int32").name` | `string` | .NET `Type.Name` (e.g. `"Int32"`)—NOT the NumPy dtype name |
| `np.dtype("int32").byteorder` | `char` | Always `'='` (native) in NumSharp |
| `new DType(arr.dtype)` | `DType` | Construct `DType` from an `NDArray`'s `.dtype` |
| `InfoOf<T>.Size` | `int` | Byte size of CLR type `T` (correct for all 15 types, including Decimal/Char) |
| `InfoOf<T>.NPTypeCode` | `NPTypeCode` | `NPTypeCode` for CLR type `T` |

‡ `kind` for `NPTypeCode.Boolean` returns `'?'` rather than NumPy's `'b'`; for Complex it's `'c'` (matches NumPy).

### Machine limits

| Function | Returns | Works for |
|----------|---------|-----------|
| `np.iinfo(dtype)` | `iinfo` with `bits`, `min`, `max`, `kind` | integer dtypes + Boolean + Char |
| `np.finfo(dtype)` | `finfo` with `bits`, `eps`, `min`, `max`, `precision`, `resolution`, `maxexp`, `minexp`, `smallest_normal`, `smallest_subnormal` | `Half`, `Single`, `Double`, `Decimal`, `Complex` |

### Exceptions

| Exception | When |
|-----------|------|
| `NotSupportedException` | dtype string unrecognized, or NumPy dtype NumSharp doesn't implement (`S`/`U`/`M`/`complex64`/…); access to `np.complex64` / `np.csingle` class-level aliases |
| `TypeError` | Complex → non-complex scalar cast (`(int)complexScalar`, etc.) |
| `IncorrectShapeException` | NDArray → scalar cast on non-0-d array (matches NumPy 2.x's strict 0-d requirement) |
| `ArgumentNullException` | `np.dtype(null)` |

---

## Related Reading

- [NumPy Compliance & Compatibility](compliance.md) — Type promotion, NEP 50, broader NumPy 2.x parity
- [Broadcasting](broadcasting.md) — How shapes combine across operations (dtype-independent)
- [Buffering, Arrays and Unmanaged Memory](buffering.md) — How dtype affects memory layout
- [IL Kernel Generation in NumSharp](il-generation.md) — Which dtypes get SIMD acceleration and why
- [NumPy data types user guide](https://numpy.org/doc/stable/user/basics.types.html) — NumPy's own dtype reference
