# Working with arrays of strings and bytes

NumPy has a family of text and byte dtypes — fixed-width `str_` (`U`), `bytes_` (`S`), raw `void` (`V`), and the variable-width `StringDType` added in NumPy 2.0. **NumSharp implements none of them.** It has one character-adjacent dtype, `Char`, and for everything else the answer is: use .NET's own `string`, `string[]`, and `byte[]` directly, and cross the boundary with `np.frombuffer` when you need bytes as an array.

This page explains what NumSharp offers, why the NumPy string dtypes are deliberately absent, and how to port code that used them.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsStringsBytesDocTests — every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsStringsBytesDocTests.cs. Section → method(s):
     The Char dtype → Char_IsTwoByteUtf16CodeUnit
     NumPy string/bytes dtypes throw → UnsupportedStringDtypes_Throw
     Text stays .NET (string[] + LINQ) → Text_StaysDotNet_LengthsViaLinq
     Byte streams via np.frombuffer → Bytes_ViaFrombuffer -->

---

## The `Char` dtype

`NPTypeCode.Char` wraps .NET's `System.Char` — a **2-byte UTF-16 code unit**. It exists so an array of individual characters can carry "these are characters, not `ushort`s" in the type system:

```csharp
var letters = np.array(new[] { 'a', 'b', 'c' });
letters.typecode;        // NPTypeCode.Char
InfoOf<char>.Size;       // 2  (actual memory footprint)
```

**`Char` is not NumPy's `'c'` / `S1`.** NumPy's `S1` is a *one-byte* bytestring; NumSharp's `Char` is a *two-byte* UTF-16 unit. Different size, different encoding, different semantics — porting NumPy bytestring code onto `Char` will almost always be wrong. Its `kind` is reported as `'S'` for NumPy round-trip ergonomics, and it maps to `<U1` when written to `.npy`, but it behaves like a 2-byte integer for arithmetic. Full details and the itemsize quirk are in [Data types → Char](../dtypes.md#numsharp-specific-types-decimal-and-char).

---

## NumPy string/bytes dtypes are not supported

Every NumPy text or byte dtype raises `NotSupportedException` when you try to construct or parse it:

| NumPy dtype | Character code | NumSharp |
|-------------|----------------|----------|
| `str_` (Unicode, UCS-4 fixed width) | `U`, `<U10` | **throws** |
| `bytes_` (null-terminated bytestring) | `S`, `a`, `\|S5` | **throws** |
| `void` (raw byte block) | `V`, `\|V7` | **throws** |
| `object` (boxed Python objects) | `O` | **throws** |
| `StringDType` (variable-width UTF-8, NumPy 2.0) | — | **throws** |

```csharp
np.dtype("U5");     // throws NotSupportedException
np.dtype("S10");    // throws
np.dtype("V7");     // throws
```

The throw is deliberate — see [Data types → why throw instead of silent approximation](../dtypes.md#why-throw-instead-of-silent-approximation). Silently mapping `S10` to `string` or `U` to `Char` would produce a differently-sized, differently-encoded array than the caller asked for, and corrupt any binary round-trip.

### Why these are absent

NumPy's string dtypes are *fixed-width, in-buffer* text: an `<U10` array reserves 10 UCS-4 code points per element inside the array's data buffer, and `StringDType` stores UTF-8 out-of-band with in-buffer pointers. Neither maps cleanly onto .NET, where:

- `string` is a heap object (UTF-16, variable length), not a fixed-width buffer slot;
- `byte[]` is the natural home for raw bytes;
- there is no zero-cost way to store variable-length text inside NumSharp's unmanaged element buffer.

So rather than build a half-fidelity port, NumSharp leaves text to .NET's excellent native types and keeps `NDArray` numeric.

---

## What to use instead

### Text data → `string` / `string[]`

Keep text in ordinary .NET collections. There is no `NDArray` of strings; index and process them with LINQ or plain loops:

```csharp
string[] labels = { "cat", "dog", "bird" };
var lengths = np.array(labels.Select(s => s.Length).ToArray());   // NDArray<int> of lengths
```

If you need per-character arrays, `Char` works for a single fixed-width column, but a jagged set of words is a `string[]`, not an array dtype.

### Byte streams → `byte[]` + `np.frombuffer`

For raw bytes, use `byte[]` and reinterpret with `np.frombuffer` when you want to view them as numbers:

```csharp
byte[] raw = File.ReadAllBytes("data.bin");
var asBytes  = np.frombuffer(raw, np.uint8);     // NDArray<byte>, zero-copy view
var asFloats = np.frombuffer(raw, np.float32);   // reinterpret the same bytes as float32
```

This is the NumSharp analog of NumPy's `np.frombuffer` on a bytestring, and the general interop bridge — see [Any library via np.frombuffer](../interop/np-frombuffer.md).

### Fixed-width byte records → parse yourself

NumPy code that used `S`/`V` to read a fixed-width binary field should read the `byte[]` and slice it with .NET (`Encoding.ASCII.GetString`, `BinaryReader`, `Span<byte>`), then build numeric `NDArray`s from the parsed values.

---

## Porting cheat-sheet

| NumPy | NumSharp |
|-------|----------|
| `np.array(["hello", "world"])` (→ `<U5`) | `string[] { "hello", "world" }` (plain .NET) |
| `np.array([b"hi"], dtype="S2")` | `byte[]` / `np.frombuffer(bytes, np.uint8)` |
| `arr.astype("U10")` | not supported — keep text as `string` |
| `np.char.upper(arr)` | `labels.Select(s => s.ToUpper())` (LINQ on `string[]`) |
| `StringDType()` array | `string[]` / `List<string>` |
| `np.array(['a','b','c'])` (single chars) | `np.array(new[] { 'a', 'b', 'c' })` → `Char` |

---

## Troubleshooting

### "`np.dtype("U5")` / `"S10"` threw NotSupportedException"
That's expected — NumSharp has no string/bytes dtypes. Keep the data in `string`/`string[]`/`byte[]`; the exception message names the alternative.

### "I mapped a NumPy `S1` array to `Char` and the bytes are wrong"
`Char` is 2-byte UTF-16, not NumPy's 1-byte `S1`. Use `byte` (`np.uint8`) for bytestring data.

### "`arr.astype(np.float64)` failed on my Char array"
`Char` is treated as a 2-byte integer for numeric conversions; make sure the values are what you expect (code points), not text you meant to keep as `string`.

---

## API reference

| Item | Behavior |
|------|----------|
| `NPTypeCode.Char` / `np.dtype('a'..'z' as chars)` | 2-byte UTF-16 code unit dtype; `kind == 'S'`; `<U1` on `.npy` |
| `np.dtype("U…")` / `"S…"` / `"V…"` / `"O"` | `NotSupportedException` |
| `np.frombuffer(byte[], dtype)` | reinterpret raw bytes as a numeric array |
| `InfoOf<char>.Size` | `2` (true footprint; note `DType.itemsize` reports `1`) |

---

## Related reading

- [Data types](../dtypes.md) — the full dtype set, the `Char` quirks, and why unsupported dtypes throw.
- [Structured arrays](structured-arrays.md) — the other NumPy dtype family NumSharp does not implement.
- [Any library via np.frombuffer](../interop/np-frombuffer.md) — the byte-buffer bridge.
- [NumPy strings-and-bytes guide](https://numpy.org/doc/stable/user/basics.strings.html) — the upstream article.
