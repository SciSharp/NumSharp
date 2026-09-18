# Structured arrays

In NumPy a **structured array** is an `ndarray` whose dtype is a composition of named fields — a C-struct-like record: `np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])`. **NumSharp does not implement structured or record dtypes.** Any dtype string containing a field specification (a `(`) raises `NotSupportedException`, and there is no `recarray` / `record` type.

This page explains why, and gives the .NET-idiomatic replacements for the things structured arrays are used for.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsStructuredArraysDocTests — the code examples on this page are executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsStructuredArraysDocTests.cs. Section → method(s):
     Structured dtype strings throw → StructuredDtypeStrings_Throw
     Replacement 1: parallel NDArrays → ParallelArrays_ColumnarReplacement
     Replacement 2: .NET record arrays → RecordArrays_BuildColumnFromField -->

---

## Not supported

```csharp
np.dtype("i8, f4, S3");                 // throws NotSupportedException
np.dtype("3int8, float32");             // throws
// there is no np.dtype([("x","f4"),("y","i4")]) form at all
```

The rejection is tested (`test/NumSharp.Tests/Creation/DTypeStringParityTests.cs`) with an explicit throw assertion, and it is deliberate rather than an oversight — a structured dtype pulls in field offsets, alignment/padding rules, titles, sub-array fields, unions, `recarray` attribute access, and multi-field view semantics, none of which have a natural home in NumSharp's numeric, single-dtype `NDArray`.

### Why NumSharp leaves them out

Structured arrays exist in NumPy for two jobs, and .NET already does both well:

- **Interfacing with C structs / binary blobs.** In .NET this is what `struct` layout, `System.Runtime.InteropServices`, `BinaryReader`, and `Span<byte>` are for — with compile-time field types and no dtype-string parsing.
- **Lightweight tabular data.** NumPy's own docs steer tabular work toward pandas/xarray because the C-struct memory layout has poor cache behavior for column operations. In .NET the idiomatic answers are a `class`/`record` list, a columnar set of `NDArray`s, or a dedicated dataframe library.

So NumSharp keeps `NDArray` numeric and homogeneous, and points structured use cases at the tools built for them.

---

## What to use instead

### 1. Parallel `NDArray`s (columnar / struct-of-arrays)

The closest analog, and usually the *fastest* for analytics — one array per field, indexed in lockstep:

```csharp
// NumPy: x = np.array([('Rex', 9, 81.0)], dtype=[('name','U10'),('age','i4'),('weight','f4')])
var names   = new[] { "Rex", "Fido" };            // string[]  (text stays .NET)
var age     = np.array([9, 3]);           // int32 column
var weight  = np.array([81.0f, 27.0f]);   // float32 column

// "x['age']" → the age column directly:
age;                                              // [9 3]
// row 0 → gather across columns:
(names[0], age[0], weight[0]);
```

Numeric operations vectorize per column (`age + 1`, `weight * 2`), which is exactly what structured arrays are *slow* at.

### 2. .NET `struct` / `record` arrays

When you want a genuine record type (heterogeneous fields, passed around as one value), use a .NET type — this is the natural replacement for the "C struct interop" use case:

```csharp
record Pet(string Name, int Age, float Weight);
Pet[] pets = { new("Rex", 9, 81f), new("Fido", 3, 27f) };
```

You get compile-time field names and types, and can still build numeric `NDArray`s from any field (`np.array(pets.Select(p => p.Weight).ToArray())`).

### 3. `np.frombuffer` for fixed-layout binary records

To read a packed binary file of C structs, read the bytes and slice per field with `np.frombuffer` + strided views, or with a `[StructLayout]` .NET struct and `MemoryMarshal`:

```csharp
byte[] raw = File.ReadAllBytes("records.bin");
// e.g. interpret an interleaved column with a strided view, or
// marshal to a [StructLayout(LayoutKind.Sequential)] struct[] with MemoryMarshal.Cast.
```

See [Any library via np.frombuffer](../interop/np-frombuffer.md).

---

## Porting cheat-sheet

| NumPy structured pattern | NumSharp replacement |
|--------------------------|----------------------|
| `np.dtype([('a','i4'),('b','f4')])` | one `NDArray` per field, or a `struct`/`record` |
| `x['a']` (field access) | the corresponding column `NDArray` (parallel arrays) |
| `x[0]` (structured scalar) | a tuple / `record` instance gathered across columns |
| `x[['a','b']]` (multi-field view) | select the columns you want |
| `np.rec.array(...)` / `recarray` | `record[]` with named properties |
| structured `.npy` file | **read raises** — re-export as separate arrays, or parse the bytes |

---

## Troubleshooting

### "`np.dtype("i4,f4")` threw NotSupportedException"
Structured dtypes are not supported. Use parallel `NDArray`s (one per field) or a .NET `record`.

### "Loading a `.npy` file with a structured dtype raised"
NumSharp cannot represent the structured dtype, so it rejects the file. Re-save it from NumPy as separate arrays in an `.npz` (`np.savez("out.npz", a=x['a'], b=x['b'])`), which NumSharp reads as named members.

### "I need attribute access like `rec.age`"
Use a .NET `record`/`class` — you get `pet.Age` with compile-time checking, which is stronger than `recarray`'s runtime attribute lookup.

---

## API reference

| Item | Behavior |
|------|----------|
| `np.dtype(string with fields)` | `NotSupportedException` — no structured dtype |
| `recarray` / `record` / `np.rec.*` | not present |
| field access `x['name']` | not present — use parallel arrays / records |
| `np.frombuffer(byte[], dtype)` | the supported route for fixed-layout binary records |

---

## Related reading

- [Data types](../dtypes.md) — the 15 supported dtypes and why unsupported ones throw.
- [Arrays of strings and bytes](strings-and-bytes.md) — the other unsupported NumPy dtype family.
- [Array creation](array-creation.md) — building the parallel arrays that replace records.
- [Any library via np.frombuffer](../interop/np-frombuffer.md) — reading packed binary records.
- [NumPy structured-arrays guide](https://numpy.org/doc/stable/user/basics.rec.html) — the upstream article.
