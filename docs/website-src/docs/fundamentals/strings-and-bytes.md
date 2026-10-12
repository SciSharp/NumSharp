# Working with arrays of strings and bytes

NumSharp has NumPy's two fixed-width text dtypes: `str_` (`U`, Unicode) and `bytes_` (`S`, bytes). It stores, reads, writes, converts, copies, casts and prints them byte for byte like NumPy 2.4.2.

Conversion between text and numbers works as in NumPy: a number written into a text array is its scalar's `str()`, and a text cast to a number is parsed by Python's `int()`, `float()` or `complex()`, with NumPy's errors.

Computing on text arrives in stages. Comparisons, truth values, concatenation and repetition compute now, with NumPy's results. Sorting, searching and the other `np.strings` functions throw `StringComputationNotSupportedException` until their stage lands. Where NumPy itself refuses a call (`np.sum` of text, `np.sqrt` of text), NumSharp raises NumPy's exception with NumPy's message.

NumPy 2's variable-width `StringDType` (`T`) stores, reads, writes, copies, casts and prints as in NumPy too; computing on it is the next phase (see [Variable-width text: `StringDType`](#variable-width-text-stringdtype)). NumPy's raw `void` (`V`) and `object` (`O`) dtypes stay unsupported.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsStringsBytesDocTests - every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsStringsBytesDocTests.cs. Section → method(s):
     The two string dtypes → TwoStringDtypes_WidthAndItemSize, TwoStringDtypes_RawBytes
     Creating string arrays → Creating_FromCSharpValues, Creating_BytesNeedObjectArray
     Reading and writing elements → Elements_ReadAndWrite, CharVectorApi_RefusesStringArrays
     Dtype integration → Dtypes_PromotionAndCasting
     Moving data → Moving_ViewsCopiesAndCasts, Moving_NonAsciiToBytes_Throws
     Converting between text and numbers → Converting_NumbersToText, Converting_TextToNumbers, Converting_StoresAndUfuncs, Converting_CSharpStringIsText
     Printing → Printing_StrAndRepr
     Computing on text → Computing_Comparisons, Computing_TruthValues, Computing_ConcatenationAndRepetition, Computing_NumPyRefusals, Computing_NextPhase, Computing_NeverEqualToNumbers, Computing_ChecksInNumPysOrder
     Variable-width text: StringDType → StringDType_Creating, StringDType_NAObjects, StringDType_WritingElements, StringDType_MovingAndCasting, StringDType_Refusals
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

## Converting between text and numbers

### Numbers to text

A number becomes the text of its NumPy scalar's `str()`. An unsized `"U"` or `"S"` request takes the width NumPy reserves for the number type: 5 for `bool`, 21 for `int64`, 32 for the floats, 64 for complex. A sized request cuts the text to its width:

```csharp
np.arange(3).astype("U");                          // ['0' '1' '2'], <U21
np.array(new[] { 1.5, 1e16, 1e-5 }).astype("U");   // ['1.5' '1e+16' '1e-05'], <U32
np.array(new[] { true, false }).astype("U");       // ['True' 'False'], <U5
np.array(new[] { 0.1f }).astype("U");              // ['0.1'] - float32 prints its own shortest digits
np.array(new[] { 3.14159 }).astype("U3");          // ['3.1']
```

The same text is written wherever a number goes into a text array: an element, `fill`, `np.full`, `np.ones`, `np.pad`, `np.copyto`, and a masked array's fill value:

```csharp
var u = np.array(new[] { "ab", "cd" });
u[0] = 5;                                     // ['5' 'cd']
u[1] = 2.5;                                   // ['5' '2.'] - cut to the width
np.full(new Shape(2), 7, "U");                // ['7' '7'], <U1
np.ones(new Shape(2), "U");                   // ['1' '1'], <U1
np.pad(np.array(new[] { "ab", "c" }), 1);     // ['0' 'ab' 'c' '0']
```

### Text to numbers

A cast to a number parses each element. `int()` takes decimal digits with an optional sign, underscores between digits and surrounding whitespace. `float()` also takes `inf`, `nan` and exponents. `complex()` takes `1+2j` forms, with or without parentheses. An element that does not parse raises NumPy's `ValueError`, and an integer that does not fit raises NumPy's `OverflowError` (as `OverflowException`):

```csharp
np.array(new[] { "1", " 22 ", "-3", "1_000" }).astype(np.int64);    // [   1   22   -3 1000]
np.array(new[] { "1.5", "inf", "-nan", "1e3" }).astype(np.float64); // [   1.5    inf    nan 1000. ]
np.array(new[] { "1+2j", "3", "(4-5j)" }).astype(np.complex128);    // [1.+2.j 3.+0.j 4.-5.j]
np.array(new[] { "1", "x" }).astype(np.int64);      // ValueError: invalid literal for int() with base 10: np.str_('x')
np.array(new[] { "1", "a" }).astype(np.float64);    // ValueError: could not convert string to float: np.str_('a')
np.array(new[] { "300" }).astype(np.int8);          // OverflowException: Python integer 300 out of bounds for int8
```

A `bytes` (`S`) element parses the same way. The error names the first element that fails, in NumPy's order.

A C# `string` written into a numeric array is NumPy's Python `str`. It parses the same way, and its error quotes the string itself:

```csharp
var a = np.zeros(new Shape(3), np.int32);
a[0] = "12";        // [12  0  0]
a[0] = "hello";     // ValueError: invalid literal for int() with base 10: 'hello'
```

### Conversions inside other calls

NumPy converts between text and numbers inside other calls too, and NumSharp follows:

```csharp
np.add(np.array(new[] { 1.5, 2.0 }), np.array(new long[] { 1, 2 }), dtype: "U");   // ['1.51' '2.02'], <U53
np.equal(names, np.array(new[] { "cat", "cow", "bird" }), np.empty(new Shape(3), "U5"));   // ['True' 'False' 'True']
np.add(np.array(new[] { 1.0, 2.5 }), np.array(new[] { 1.0, 3.0 }), np.zeros(new Shape(2), "U5"));   // ['2.0' '5.5']
np.prod(np.array(new[] { "1.5", "2", "4" }), (int?)null, np.float64);   // 12.0 - the text is parsed first
```

- `np.add` with a string `dtype=` writes each number as text, then concatenates.
- A ufunc with a text `out=` writes each result as text, cut to the width of `out`.
- A reduction with a numeric `dtype=` parses the text first.

With `where=`, NumPy also parses the CURRENT text of `out` into the loop's number type before it computes, so a masked-off `'x'` in a float loop raises `ValueError`.

> **Breaking change:** a C# string converts to a 0-d `U` array, as `np.array(str)` does in NumPy. `NDArray x = "[1, 2, 3]"` is `array('[1, 2, 3]', dtype='<U9')`. It no longer parses the numbers in the text. Parse explicitly with `astype`.

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

### Concatenation and repetition

`np.add` and the `+` operator concatenate two text arrays of one class, element by element. The result is as wide as the two widths together. Trailing NULs never count, embedded NULs do. A C# `string` is NumPy's `str` scalar. A string on the left stays the first operand. `np.strings.add` is `np.add`:

```csharp
names + "s";                                      // ['cats' 'dogs' 'birds'], <U5
"a " + names;                                     // ['a cat' 'a dog' 'a bird'], <U6
np.add(names, np.array(new[] { "!", "?", "" }));  // ['cat!' 'dog?' 'bird'], <U5
```

`np.strings.multiply` repeats each string. The result is as wide as the longest repetition. A count less than 1 gives an empty string:

```csharp
np.strings.multiply(names, 2);                                  // ['catcat' 'dogdog' 'birdbird'], <U8
np.strings.multiply(names, np.array(new long[] { 1, 0, -1 }));  // ['cat' '' ''], <U3
```

The `np.multiply` ufunc cannot size its result, so it needs `out=`. It writes each repetition cut to the width of `out`. Without `out=`, `np.multiply(names, 2)` and `names * 2` raise NumPy's `TypeError`, as in NumPy:

```csharp
np.multiply(names, 2, np.zeros(new Shape(3L), "U5"));   // ['catca' 'dogdo' 'birdb']
```

Broadcasting, every memory layout, `out=` (a narrower or wider text array receives the result cut or padded) and `where=` work as for numbers. A `bytes` array with a `str` array has no loop: `np.add` raises NumPy's `TypeError`. A repetition longer than 2^63 - 1 characters raises NumPy's `OverflowError` (as `OverflowException`).

### What is not computed yet

NumPy answers a call that involves text in one of two ways, and NumSharp follows it.

**NumPy refuses the call.** NumSharp raises the same exception with the same message:

```csharp
np.sum(names);    // TypeError: the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U4'), dtype('<U4'), dtype('<U8'))
np.sqrt(names);   // TypeError: ufunc 'sqrt' not supported for the input types, ...
```

**NumPy computes on the text in a later stage.** NumSharp throws `StringComputationNotSupportedException`, a `NotSupportedException`:

```csharp
np.sort(names);           // sorting
np.unique(names);         // sorting, to find the unique values
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

An unsized string `dtype=` request (`"U"`, `"S"`) gets NumPy's answer too. `np.sqrt(x, dtype: "U")` names no loop. `np.add(x, x, dtype: "U")` writes the numbers as text and concatenates them.

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

The rules for computing are the same as for plain arrays. A masked reduction refuses with NumPy's message. A number used as a fill value is written as its text:

```csharp
np.ma.sum(m);      // TypeError: the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))
m.filled(5);       // ['5' 'c' 'de']
```

For what is not computed yet, convert to .NET and compute there. For example, `NDArray.AsStringArray(names).Select(s => s.ToUpper())`.

---

## Variable-width text: `StringDType`

NumPy 2's `StringDType` (`T`, NEP 55) has no width. Each element is a 16-byte cell: a text of up to 15 UTF-8 bytes sits in the cell itself, a longer text in an arena that belongs to the array's buffer, and a text that a later write makes longer than its slot moves to the heap. NumSharp stores `StringDType` arrays with NumPy's cell layout byte for byte (`tobytes()` returns the same bytes as NumPy), with its missing-data objects, its coercion rules and its casts.

Computing on `StringDType` text (comparisons, concatenation, `np.sum`, sorting, the `np.strings` functions) is the next phase. Until then those calls throw `StringComputationNotSupportedException`, or the `NotSupportedException` of a route that has no `StringDType` form yet.

### Creating `StringDType` arrays

```csharp
var t = np.array(new[] { "a", "a much longer text" }, np.dtype("T"));
t.dtype;                                            // StringDType()
t.dtype.itemsize;                                   // 16: the cell, whatever the length of the text
np.zeros(new Shape(2), np.dtypes.StringDType());    // ['', '']
np.ones(new Shape(2), np.dtypes.StringDType());     // ['1', '1']
```

### Missing data: the NA object

`np.dtypes.StringDType(na_object: …)` names the object that stands for a missing element: `null` (NumPy's `None`), `double.NaN`, a string, or another object. Without an NA object, `None` is just the text `'None'`:

```csharp
var withNA = np.dtypes.StringDType(na_object: null);          // NumPy's StringDType(na_object=None)
var a = np.array(new object[] { "x", null }, withNA);
a.item(1);                                                     // null: the NA object
a.ToString(true);                                              // array(['x', None], dtype=StringDType(na_object=None))
np.array(new object[] { "x", null }, np.dtype("T")).item(1);   // "None"
```

### Writing elements

A value written into a `StringDType` element is its `str()`. With `coerce: false`, only text is accepted:

```csharp
var t = np.array(new[] { "a", "b" }, np.dtypes.StringDType());
t[0] = 3.25;                       // '3.25'
var strict = np.array(new[] { "a" }, np.dtypes.StringDType(coerce: false));
strict[0] = 5;                     // ValueError: StringDType only allows string data when string coercion is disabled.
```

### Moving and casting

Views share the buffer and its arena. A copy, a gather (`np.take`, fancy indexing, a boolean mask) and a scatter duplicate every string into the destination's own buffer. The casts follow NumPy: text, bool and numbers cast to `StringDType`; `StringDType` casts to a sized `U` or `S` (the text is cut to the width) and to numbers (each text is parsed by Python's `int()`, `float()` or `complex()`; a missing element raises):

```csharp
var t = np.array(new[] { "12", "-3" }, np.dtypes.StringDType());
t.astype(np.int64);                                               // [12, -3]
t.astype("U1");                                                   // ['1', '-']
np.array(new[] { 1.5, 2.0 }).astype(np.dtypes.StringDType());     // ['1.5', '2.0']
```

A copy moves a short text's cell as one block and reserves the destination's arena once for the longer texts. Measured against NumPy 2.4.2 on the same machine, copies, gathers (`np.take`, a boolean mask), `np.concatenate`, `np.where`, `np.repeat`, `np.array` from a list and the casts between `U` and `StringDType` run 1.8 to 31 times as fast as NumPy, for 1,000 to 1,000,000 elements.

### What `StringDType` refuses

| Call | NumSharp |
|------|----------|
| `view(another dtype)`, `memoryview` (`.data`), `np.frombuffer(..., "T")` | NumPy's errors: a cell is not raw text |
| `np.ndarray(shape, "T", buffer: ...)` | `ArgumentException`. NumPy reads the bytes as cells; a cell holds an arena offset or a heap pointer, so NumSharp refuses |
| `np.nditer` that buffers or casts a `StringDType` operand | `NotSupportedException` (a later package; `refs_ok` without buffering works) |
| `.npy` files, `loadtxt` / `savetxt`, numpy.ma with `StringDType` | not yet (a later package) |
| Computation: `==`, `np.sum`, `np.sort`, `np.strings.*` | `StringComputationNotSupportedException` or `NotSupportedException` (phase 2) |

NumPy 2.4.2 has bugs on `StringDType` that NumSharp does not reproduce; NumSharp gives the correct answer:

- `np.place` over a `StringDType` array crashes the process (a segmentation fault).
- `np.choose`, `np.put` and `np.putmask` copy the cells raw: an arena string gives garbage.
- The `bytes` to `StringDType` cast does not validate UTF-8, so a later read of the element raises. NumSharp raises the `UnicodeDecodeError` at the cast.
- The `StringDType` to `complex` cast writes the real part into the imaginary part too.
- A short string copied onto itself (`a[...] = a[:1]`) becomes `''`.

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
| `np.arange(3).astype(str)` (→ `<U21`) | `np.arange(3).astype("U")` (→ `<U21`) |
| `np.array(["1", "2"]).astype(float)` | `np.array(new[] { "1", "2" }).astype(np.float64)` |
| `arr[0]` (a `numpy.str_`) | `arr.item(0)` (a C# `string`) |
| `arr.tolist()` | `arr.tolist()` or `NDArray.AsStringArray(arr)` |
| `a + b` / `np.char.add(a, b)` / `np.strings.add(a, b)` | `a + b` / `np.add(a, b)` / `np.strings.add(a, b)` |
| `np.char.multiply(a, 3)` / `np.strings.multiply(a, 3)` | `np.strings.multiply(a, 3)` |
| `np.char.upper(arr)` / `np.strings.upper(arr)` | a later stage: `NDArray.AsStringArray(arr).Select(s => s.ToUpper())` |
| `np.array(["a", "bb"], dtype=StringDType())` | `np.array(new[] { "a", "bb" }, np.dtypes.StringDType())` |
| `StringDType(na_object=None)` | `np.dtypes.StringDType(na_object: null)` |
| `np.array(['a','b','c'])` (single characters, `<U1`) | `np.array(new[] { "a", "b", "c" })` (`<U1`); `np.array(new[] { 'a', 'b', 'c' })` is `Char` |

---

## Troubleshooting

### "`StringComputationNotSupportedException`: ... is the next phase"
NumPy computes this call on text, and NumSharp does not compute it yet (comparisons, truth values, concatenation, repetition and the conversions between text and numbers already compute). Convert the values to .NET (`NDArray.AsStringArray`, `item`) and use LINQ, or keep the computation in NumPy.

### "`ValueError: could not convert string to float`"
A cast from text to a number parses every element, as NumPy does. Clean the text first, or keep it as text.

### "My `NDArray x = "1.5"` is text, not a number"
A C# string is NumPy's `str`: a 0-d `U` array. Write the number itself (`NDArray x = 1.5`), or parse the text with `astype(np.float64)`.

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
| `arr.astype("U")` of numbers | Each number as its scalar's `str()`, at NumPy's width for the number type (21 for `int64`, 32 for `float64`) |
| `arr.astype(np.float64)` of text | Each text parsed by `float()` (`int()` / `complex()` for those targets), with NumPy's errors |
| `arr[i] = 5` / `arr[i] = "12"` | A number stored as text / a C# string parsed into a numeric array |
| Comparisons (`==`, `<`, `np.equal`, ...) and truth values (`astype(bool)`, `np.any`, `np.nonzero`, ...) | NumPy's results |
| `np.add(a, b)` / `a + b` / `np.strings.add(a, b)` over two text arrays of one class | The concatenation, as wide as the two widths together |
| `np.strings.multiply(a, i)` / `np.multiply(a, i, out)` | The repetition, as wide as the longest one / cut to the width of `out` |
| Other computing calls (`np.sort`, `np.unique`, ...) | `StringComputationNotSupportedException` (a later stage) |
| NumPy's refusals (`np.sum`, `np.sqrt`, ...) | NumPy's exception and message |
| `np.dtypes.StringDType(...)` / `np.dtype("T")` | NumPy's variable-width `StringDType`: storage, element access, copies, gathers, scatters, casts and printing as NumPy's; computing on it throws `StringComputationNotSupportedException` (phase 2) |
| `np.dtype("V7")` / `"O"` | `NotSupportedException` |
| `NPTypeCode.Char` | A 2-byte UTF-16 code unit dtype, written to `.npy` as `<U1` |
| `np.frombuffer(byte[], dtype)` | Reinterprets raw bytes as an array, string dtypes included |

---

## Related reading

- [Data types](../dtypes.md) - the full dtype set and the `Char` quirks.
- [Structured arrays](structured-arrays.md) - a NumPy dtype family NumSharp does not implement.
- [Any library via np.frombuffer](../interop/np-frombuffer.md) - the byte-buffer bridge.
- [NumPy strings-and-bytes guide](https://numpy.org/doc/stable/user/basics.strings.html) - the upstream article.
