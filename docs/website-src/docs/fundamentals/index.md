# Fundamentals and usage

These pages clarify the concepts, design decisions, and behavioral rules behind NumSharp - the .NET port of NumPy that targets **1-to-1 API and behavioral compatibility with NumPy 2.x**. They are the NumSharp counterparts of NumPy's own [*NumPy fundamentals*](https://numpy.org/doc/stable/user/basics.html) guide: one article per NumPy article, rewritten for C# and honest about where a .NET runtime forces NumSharp to diverge.

If you have written NumPy before, these pages are the fastest way to map that knowledge onto `NDArray`. If you have not, they are the place to build a mental model before reaching for the [API reference](../../api/index.md).

<!-- Tests: every code example in the articles below is executed and asserted by a matching class in test/NumSharp.Tests/Documentation/ (NumSharp.Tests.Documentation namespace):
     array-creation.md → FundamentalsArrayCreationDocTests
     indexing.md → FundamentalsIndexingDocTests
     io.md → FundamentalsIoDocTests
     copies-and-views.md → FundamentalsCopiesViewsDocTests
     strings-and-bytes.md → FundamentalsStringsBytesDocTests
     structured-arrays.md → FundamentalsStructuredArraysDocTests
     ufuncs.md → FundamentalsUfuncsDocTests
     (Data types → dtypes.md and Broadcasting → broadcasting.md are the pre-existing pages, covered by NDArrayDocExamplesTests and their own suites.) -->

---

## The articles

| Article | What it covers | NumPy original |
|---------|----------------|----------------|
| [Array creation](array-creation.md) | The six ways to make an `NDArray` - from .NET sequences, intrinsic functions, joins, disk, raw bytes, and random | [basics.creation](https://numpy.org/doc/stable/user/basics.creation.html) |
| [Indexing on NDArray](indexing.md) | Basic indexing (views), advanced indexing (copies), boolean masks, the flat iterator, and assignment | [basics.indexing](https://numpy.org/doc/stable/user/basics.indexing.html) |
| [I/O with NumSharp](io.md) | The byte-exact `.npy`/`.npz` stack, text I/O (`loadtxt`/`savetxt`), and raw-binary I/O | [basics.io](https://numpy.org/doc/stable/user/basics.io.html) |
| [Data types](../dtypes.md) | The 15 supported dtypes, how they map to NumPy, dtype strings, promotion, and the two types with no NumPy analog | [basics.types](https://numpy.org/doc/stable/user/basics.types.html) |
| [Broadcasting](../broadcasting.md) | How arrays of different shapes combine in arithmetic - the shape rules and the stride-0 mechanism | [basics.broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| [Copies and views](copies-and-views.md) | Which operations share memory and which duplicate it, and how to tell | [basics.copies](https://numpy.org/doc/stable/user/basics.copies.html) |
| [Arrays of strings and bytes](strings-and-bytes.md) | The `Char` dtype, why NumPy's string/bytes dtypes are not ported, and what to use instead | [basics.strings](https://numpy.org/doc/stable/user/basics.strings.html) |
| [Structured arrays](structured-arrays.md) | Why NumSharp does not implement record/structured dtypes, and the .NET-idiomatic alternatives | [basics.rec](https://numpy.org/doc/stable/user/basics.rec.html) |
| [Universal functions (ufunc) basics](ufuncs.md) | Elementwise ops, `out=`/`where=`/`dtype=`, broadcasting, casting, and reductions | [basics.ufuncs](https://numpy.org/doc/stable/user/basics.ufuncs.html) |

> **Data types** and **Broadcasting** already have dedicated, in-depth NumSharp guides ([Dtypes](../dtypes.md), [Broadcasting](../broadcasting.md)); they *are* the NumSharp conversions of NumPy's data-types and broadcasting articles, so they are linked here rather than duplicated.

---

## What "convert each NumPy article to a NumSharp one" means

NumSharp's goal is that ported NumPy code behaves identically. So each article follows its NumPy source closely - same concepts, same section order, the same worked examples - but:

- **Code is C#, not Python.** `np.zeros((2, 3))`, `arr["1:3, :2"]`, `arr.astype(np.float64)`. Python slice literals (`a[1:5:2]`) become **strings** in C# (`a["1:5:2"]`) because C# has no slice syntax - see [Indexing](indexing.md).
- **The 15 NumSharp dtypes stand in for NumPy's set.** NumSharp adds `Decimal` and `Char` (no NumPy analog) and omits `complex64`, the string/bytes/void/object dtypes, `datetime64`/`timedelta64`, and structured dtypes. Each article says plainly where a NumPy feature has no NumSharp counterpart and what to use instead.
- **Divergences are documented, never silent.** Where NumSharp cannot match NumPy exactly (e.g. a C# `int[]` is a *strong* array that wraps on downcast, where a Python list is *weak* and raises), the article flags it. Behavioral differences are gated in the test suite under the `Misaligned` category.

---

## Where these fit among the other guides

The fundamentals articles are conceptual. When you want the exhaustive how-to, the neighbouring guides go deeper:

- [NDArray](../NDArray.md) - the anatomy of the array (storage, shape, strides, the view/copy rule).
- [Getting & Setting Values](../getting-and-setting-values.md) - every accessor for reading and writing, with the conversion and write-through rules.
- [Iterating & Enumerating](../iterating-and-enumerating.md) - `flat`, `nditer`, typed/unboxed iteration, `.NET` interop.
- [NumPy Compliance & Compatibility](../compliance.md) - type promotion (NEP 50) and the broader 2.x parity story.
- [Array API Standard](../array-api-standard.md) - the array-API surface NumSharp implements.

---

## Related reading

- [NumPy fundamentals](https://numpy.org/doc/stable/user/basics.html) - the upstream guide these pages track.
- [NumPy API Coverage & Support](../coverage-support-dashboard.md) - which `np.*` functions are implemented.
