# Working with arrays of strings and bytes

NumSharp has NumPy's two fixed-width text dtypes: `str_` (`U`, Unicode) and `bytes_` (`S`, bytes). It stores, reads, writes, converts, copies, casts and prints them byte for byte like NumPy 2.4.2.

Computing on text arrives in stages. Comparisons and truth values compute now, with NumPy's results. Concatenation, sorting, searching, the `np.strings` functions and conversion between text and numbers throw `StringComputationNotSupportedException` until their stage lands. Where NumPy itself refuses a call (`np.sum` of text, `np.sqrt` of text), NumSharp raises NumPy's exception with NumPy's message.

NumPy's raw `void` (`V`), `object` (`O`) and variable-width `StringDType` dtypes stay unsupported.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsStringsBytesDocTests - every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsStringsBytesDocTests.cs. Section → method(s):
     The two string dtypes → TwoStringDtypes_WidthAndItemSize, TwoStringDtypes_RawBytes
     Creating string arrays → Creating_FromCSharpValues, Creating_BytesNeedObjectArray
     Reading and writing elements → Elements_ReadAndWrite, CharVectorApi_RefusesStringArrays
     Dtype integration → Dtypes_PromotionAndCasting
     Moving data → Moving_ViewsCopiesAndCasts, Moving_NonAsciiToBytes_Throws
     Printing → Printing_StrAndRepr
     Computing on text → Computing_Comparisons, Computing_TruthValues, Computing_NumPyRefusals, Computing_NextPhase, Computing_NeverEqualToNumbers, Computing_ChecksInNumPysOrder
     .npy files → Npy_RoundTrip, Npy_CharWritesAsU1
     The Char dtype → Char_IsTwoByteUtf16CodeUnit
     Byte streams via np.frombuffer → Bytes_ViaFrombuffer -->

---

## The two string dtypes

| NumPy dtype | Descriptor | Bytes per character | Example | `itemsize` |
|-------------|------------|---------------------|---------|------------|
| `str_` | `U`, `<U5` | 4 (one UCS-4 code point, native byte order) | `np.dtype("U5")` | 20 |
| `bytes_` | `S`, `\|S5` | 1 | `np.dtype("S5")` | 5 |

The width is part of the dtype. Every element of a `<U5` array holds up to five characters:

- A shorter value is padded with NUL characters.
- Reading an element removes the trailing NULs. NULs inside the value stay.
- A value that is too long is cut to the width when you write it, as NumPy does.

```csharp
var names = np.array(new[] { "cat", "dog", "bird" });
names.dtype;             // dtype('<U4') - the longest value sets the width
names.dtype.itemsize;    // 16 - 4 characters x 4 bytes
np.dtype("S5").str;      // "|S5"
np.dtype("U5").name;     // "str160" - NumPy's name counts bits
```

The data buffer holds the characters directly, as in NumPy. A `U` character is a 4-byte code point. An `S` character is one byte:

```csharp
np.array(new[] { "ab" }).tobytes();               // 61 00 00 00 62 00 00 00
np.array(new[] { "ab" }).astype("S3").tobytes();  // 61 62 00
```

A character outside the Basic Multilingual Plane (an emoji) is ONE `U` code point. C# stores it as two UTF-16 chars, but `np.array(new[] { "😀" })` is `<U1`.

---

## Creating string arrays

```csharp
np.array(new[] { "cat", "dog" });                    // ['cat', 'dog'], <U3
np.array(new[] { "cat", "dog" }, np.dtype("S5"));    // [b'cat', b'dog'], |S5
np.zeros(new Shape(2), np.dtype("U3"));              // ['', ''], <U3
np.full(new Shape(2), "ab");                         // ['ab', 'ab'], <U2
NDArray.Scalar("abc");                               // 'abc', a 0-d <U3 array
np.frombuffer(Encoding.ASCII.GetBytes("abcdef"), "S3");   // [b'abc', b'def'], no copy
```

A C# `string` is NumPy's `str`, and a `byte[]` is NumPy's `bytes`. Build an `S` array from byte values with an `object[]`:

```csharp
np.array(new object[] { Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("cde") });   // [b'ab', b'cde'], |S3
```

> **Footgun:** `np.array(new[] { bytes1, bytes2 })` with a `byte[][]` is a jagged NUMERIC array in C#. It becomes a 2-D `uint8` array, not an `S` array.

---

## Reading and writing elements

```csharp
names.item(1);                     // "dog" - a C# string
names.astype("S4").item(1);        // byte[] { 100, 111, 103 } - b'dog'
NDArray.AsStringArray(names);      // { "cat", "dog", "bird" }
names[1];                          // array('dog', dtype='<U4') - a 0-d view
names[1] = "zebras";               // the array is now ['cat', 'zebr', 'bird']: cut to 4 characters
```

`GetString`, `SetString`, `GetStringAt` and `SetStringAt` belong to the `Char` dtype: they read a vector of UTF-16 chars as one string. They refuse a `U` or `S` array with an `InvalidOperationException`.

---

## Dtype integration

String descriptors take part in promotion and in the casting rules exactly as in NumPy:

```csharp
np.promote_types(np.dtype("S3"), np.dtype("U2"));           // dtype('<U3')
np.result_type(new[] { "U3", "S7" });                        // dtype('<U7')
np.can_cast(np.dtype("S3"), np.dtype("U3"), "safe");         // true - ASCII bytes fit in Unicode
np.can_cast(np.dtype("U3"), np.dtype("S3"), "same_kind");    // false
np.can_cast(np.dtype("U5"), np.dtype("U3"), "safe");         // false - the cast would cut values
```

---

## Moving data

Views, copies and element movement work on string arrays as on numeric ones: slicing, transposes, reshapes, `copy`, `concatenate`, `take`, boolean masks, `where`, `pad`, `copyto`, `put` and the other indexing routines.

Casts between string dtypes follow NumPy:

- A narrower width cuts values.
- `U` to `S` encodes ASCII.
- `S` to `U` decodes ASCII.

```csharp
np.concatenate(new[] { np.array(new[] { "a" }), np.array(new[] { "long" }) });   // ['a', 'long'], <U4
names.astype("U2");                        // ['ca', 'do', 'bi']
np.array(new[] { "é" }).astype("S1");      // UnicodeEncodeError
```

The last call raises `UnicodeEncodeError` with Python's message: `'ascii' codec can't encode character '\xe9' in position 0: ordinal not in range(128)`.

---

## Printing

`ToString()` is NumPy's `str()`, and `ToString(true)` is NumPy's `repr()`, including Python's quoting rules:

```csharp
names.ToString();                                     // ['cat' 'dog' 'bird']
names.ToString(true);                                 // array(['cat', 'dog', 'bird'], dtype='<U4')
np.array2string(np.array(new[] { "a'b", "c\"d" }));   // ["a'b" 'c"d']
```

---

## Computing on text

### Comparisons

`==`, `!=`, `<`, `<=`, `>`, `>=` and the ufuncs `np.equal` … `np.greater_equal` compare two text arrays of one class (`U` with `U`, `S` with `S`), of any widths, as NumPy does. Strings compare code unit by code unit as unsigned values. Trailing NULs never count (`'a' == 'a\0'`), embedded NULs do. Broadcasting, every memory layout, `out=`, `where=` and `dtype=bool` work as for numbers. A C# `string` against a text array is NumPy's `str` scalar:

```csharp
names == names;                       // [ True  True  True]
names < np.array(new[] { "dog" });    // [ True False  True]
names > "cat";                        // [False  True False]
```

A `bytes` array against a `str` array has no comparison loop in NumPy: `==` is all `False`, `!=` all `True`, and `<` raises NumPy's `TypeError`.

### Truth values

A string is `True` when it is not empty, that is when any code unit is not NUL. A string of spaces and the text `'False'` are `True`. `astype(bool)`, `np.any`, `np.all`, `np.nonzero`, `np.count_nonzero`, `np.where`, `np.compress`, the logical ufuncs, `np.place` and `np.putmask` masks, and `np.trim_zeros` read it:

```csharp
np.array(new[] { "", "a", " " }).astype(np.@bool);   // [False  True  True]
np.count_nonzero(names);                            // 3
```

### What is not computed yet

NumPy answers a call that involves text in one of two ways, and NumSharp follows it.

**NumPy refuses the call.** NumSharp raises the same exception with the same message:

```csharp
np.sum(names);    // TypeError: the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U4'), dtype('<U4'), dtype('<U8'))
np.sqrt(names);   // TypeError: ufunc 'sqrt' not supported for the input types, ...
```

**NumPy computes on the text in a later stage.** NumSharp throws `StringComputationNotSupportedException`, a `NotSupportedException`:

```csharp
np.add(names, names);     // concatenation
np.sort(names);           // sorting
```

A comparison of text with numbers is "never equal" in NumPy, and in NumSharp: `names == np.array(new[] { 1, 2, 3 })` is `[False False False]`.

NumSharp also reports the problems of a call in NumPy's order, before it answers:

1. The `dtype=` request.
2. A read-only `out=`, then a non-bool `where=`.
3. The axis.
4. The refusal, or the next-phase exception.

```csharp
np.sum(names, 9);   // AxisError: axis 9 is out of bounds for array of dimension 1
```

An unsized string `dtype=` request (`"U"`, `"S"`) gets NumPy's answer too. `np.sqrt(x, dtype: "U")` names no loop. `np.add(x, x, dtype: "U")` waits for a later stage, because NumPy formats the numbers as text and concatenates them.

### When the shape decides the answer

Some calls that compute on text in NumPy never read a character, because the shape or the dtypes decide the result. NumSharp gives those answers now:

- **An empty array.** Truth tests, counts, index searches, sorts, `np.unique`, `np.isin`, the set routines and the binary ufuncs return NumPy's empty or identity result.
- **Lines of one element.** A sort, a partition or an arg-reduction over lines of one element compares nothing.
- **A pair with no comparison loop.** Text against numbers is never equal, so `np.isin` and `np.array_equal` return `False`.

```csharp
var empty = np.zeros(new Shape(0L, 3L), "U2");
np.any(empty);                                  // False
np.argmax(np.array(new[,] { { "b" }, { "a" } }), 1);   // [0 0]
np.isin(names, np.array(new[] { 1.5, 2.0 }));   // [False False False]
np.array_equal(names, np.zeros(new Shape(3L))); // False
```

The same calls over text that NumPy sorts or searches stay in a later stage.

`np.place` and `np.putmask` follow NumPy's two value forms. A C# array is a Python list: NumSharp converts it straight into the target's dtype. An `NDArray` must cast to the target's dtype under the `'safe'` rule, as in NumPy:

```csharp
var a = np.arange(5).astype(np.int32);
var mask = np.ones(new Shape(5L), np.@bool);
np.place(a, mask, new[] { 1.9, 2.9 });            // a list: [1 2 1 2 1]
np.place(a, mask, np.array(new[] { 1.9, 2.9 }));  // InvalidCastException (NumPy's TypeError):
                                                  // Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'
```

### Masked text arrays

`np.ma` works on text arrays as numpy.ma does. The default fill value is `'N/A'`, stored at its own width (`<U3` or `|S3`). `filled()` cuts it to the width of the array:

```csharp
var m = np.ma.array(np.array(new[] { "ab", "c", "de" }), mask: new[] { true, false, false });
m.fill_value;      // "N/A"
m.filled();        // ['N/' 'c' 'de']
m.filled("zz");    // ['zz' 'c' 'de']
np.ma.count(m);    // 2
```

The rules for computing are the same as for plain arrays. A masked reduction refuses with NumPy's message. A call that must format numbers as text waits for a later stage:

```csharp
np.ma.sum(m);      // TypeError: the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))
m.filled(5);       // StringComputationNotSupportedException: NumPy writes 5 as the text '5'
```

For what is not computed yet, convert to .NET and compute there. For example, `NDArray.AsStringArray(names).Select(s => s.ToUpper())`.

---

## .npy files

`np.save` and `np.load` read and write `U` and `S` arrays byte for byte, as NumPy writes them. A big-endian `>U3` file is byte-swapped when it is read.

> **Breaking change:** a `<U1` file now loads as a `<U1` string array, not as `Char`. A `Char` array is still written as `<U1`, because NumPy has no 2-byte character dtype.

```csharp
var stream = new MemoryStream();
np.save(stream, names);
stream.Position = 0;
np.load_npy(stream);    // array(['cat', 'dog', 'bird'], dtype='<U4')
```

---

## The `Char` dtype

`NPTypeCode.Char` wraps .NET's `System.Char`, a **2-byte UTF-16 code unit**. It stores characters as numbers, not text:

```csharp
var letters = np.array(new[] { 'a', 'b', 'c' });
letters.typecode;        // NPTypeCode.Char
InfoOf<char>.Size;       // 2  (actual memory footprint)
```

**`Char` is not NumPy's `<U1` or `S1`.** NumPy's `<U1` is a 4-byte code point, and `S1` is one byte. NumSharp's `Char` is a 2-byte UTF-16 unit that behaves like a 2-byte integer in arithmetic. For text, use a `U` array. The full details and the itemsize quirk are in [Data types → Char](../dtypes.md#numsharp-specific-types-decimal-and-char).

---

## Byte streams → `byte[]` + `np.frombuffer`

For raw bytes, use `byte[]` and reinterpret it with `np.frombuffer` when you want to view the bytes as numbers:

```csharp
byte[] raw = File.ReadAllBytes("data.bin");
var asBytes  = np.frombuffer(raw, np.uint8);     // NDArray<byte>, zero-copy view
var asFloats = np.frombuffer(raw, np.float32);   // reinterpret the same bytes as float32
```

This is NumPy's `np.frombuffer` on a bytestring, and the general interop bridge. See [Any library via np.frombuffer](../interop/np-frombuffer.md).

---

## Porting cheat-sheet

| NumPy | NumSharp |
|-------|----------|
| `np.array(["hello", "world"])` (→ `<U5`) | `np.array(new[] { "hello", "world" })` (→ `<U5`) |
| `np.array([b"hi"], dtype="S2")` | `np.array(new object[] { Encoding.ASCII.GetBytes("hi") }, np.dtype("S2"))` |
| `arr.astype("U10")` | `arr.astype("U10")` |
| `arr[0]` (a `numpy.str_`) | `arr.item(0)` (a C# `string`) |
| `arr.tolist()` | `arr.tolist()` or `NDArray.AsStringArray(arr)` |
| `np.char.upper(arr)` / `np.strings.upper(arr)` | a later stage: `NDArray.AsStringArray(arr).Select(s => s.ToUpper())` |
| `StringDType()` array | not supported: `string[]` / `List<string>` |
| `np.array(['a','b','c'])` (single characters, `<U1`) | `np.array(new[] { "a", "b", "c" })` (`<U1`); `np.array(new[] { 'a', 'b', 'c' })` is `Char` |

---

## Troubleshooting

### "`StringComputationNotSupportedException`: ... is the next phase"
NumPy computes this call on text, and NumSharp does not compute it yet (comparisons and truth values already compute). Convert the values to .NET (`NDArray.AsStringArray`, `item`) and use LINQ, or keep the computation in NumPy.

### "My `byte[][]` became a `uint8` matrix"
A jagged `byte[][]` is numeric data in C#. Pass an `object[]` of `byte[]` values to get an `S` array.

### "My value was cut"
A string array has a fixed width. Writing a longer value cuts it, as NumPy does. Use a wider dtype: `arr.astype("U10")`.

### "`UnicodeEncodeError` when I cast to `S`"
A `U` to `S` cast encodes ASCII only, as NumPy does. Keep non-ASCII text in a `U` array.

### "`GetString` threw `InvalidOperationException`"
`GetString` reads `Char` vectors. Read a string element with `item(i)` or `NDArray.AsStringArray`.

---

## API reference

| Item | Behavior |
|------|----------|
| `np.dtype("U5")` / `np.dtype("S5")` | The fixed-width descriptors: `itemsize` 20 / 5, `kind` `'U'` / `'S'` |
| `np.array(string[])` | A `U` array as wide as the longest value |
| `np.array(object[] of byte[])` | An `S` array as wide as the longest value |
| `NDArray.Scalar(string)` / `NDArray.Scalar(byte[])` | A 0-d `U` / `S` array |
| `arr.item(i)` | A C# `string` (`U`) or `byte[]` (`S`), trailing NULs removed |
| `NDArray.AsStringArray(arr)` | Every element of a `U` array as a `string[]`, in C order |
| `arr[i] = "text"` | Writes one element, cut to the width |
| `arr.astype("S3")` / `arr.astype("U8")` | Casts between string dtypes, as NumPy does |
| Comparisons (`==`, `<`, `np.equal`, ...) and truth values (`astype(bool)`, `np.any`, `np.nonzero`, ...) | NumPy's results |
| Other computing calls (`np.add`, `np.sort`, ...) | `StringComputationNotSupportedException` (a later stage) |
| NumPy's refusals (`np.sum`, `np.sqrt`, ...) | NumPy's exception and message |
| `np.dtype("V7")` / `"O"` / `StringDType` | `NotSupportedException` |
| `NPTypeCode.Char` | A 2-byte UTF-16 code unit dtype, written to `.npy` as `<U1` |
| `np.frombuffer(byte[], dtype)` | Reinterprets raw bytes as an array, string dtypes included |

---

## Related reading

- [Data types](../dtypes.md) - the full dtype set and the `Char` quirks.
- [Structured arrays](structured-arrays.md) - a NumPy dtype family NumSharp does not implement.
- [Any library via np.frombuffer](../interop/np-frombuffer.md) - the byte-buffer bridge.
- [NumPy strings-and-bytes guide](https://numpy.org/doc/stable/user/basics.strings.html) - the upstream article.
