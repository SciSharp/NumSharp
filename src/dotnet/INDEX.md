# .NET Runtime Source Files (Span + DateTime + Char)

Downloaded from [dotnet/runtime](https://github.com/dotnet/runtime) `main` branch (.NET 10).

**Purpose:**
1. Source of truth for converting `Span<T>` to `UnmanagedSpan<T>` with `long` indexing support.
2. Reference/template for `DateTime64` struct (NumPy-parity datetime64 with full `long` range) in `src/NumSharp.Core/DateTime64.cs` — forked from `DateTime.cs` with `ulong _dateData` replaced by `long _ticks`, `DateTimeKind` bits removed, range expanded to the full `long` space, and `NaT == long.MinValue` sentinel added.
3. Source of truth for porting `Char` to `Char8` — a NumPy-compliant 1-byte character type that interops with C# `char` and `string`.

**Total:** 76 files | ~73,000 lines of code

---

## Directory Structure

```
src/dotnet/
├── src/
│   ├── coreclr/System.Private.CoreLib/src/System/Runtime/InteropServices/
│   │   └── MemoryMarshal.CoreCLR.cs
│   └── libraries/
│       ├── Common/src/System/
│       │   ├── HexConverter.cs
│       │   └── Runtime/Versioning/
│       │       └── NonVersionableAttribute.cs
│       ├── System.Memory/
│       │   ├── ref/
│       │   │   └── System.Memory.cs
│       │   └── src/System/Buffers/
│       │       ├── ArrayMemoryPool.cs
│       │       ├── BuffersExtensions.cs
│       │       ├── IBufferWriter.cs
│       │       ├── MemoryPool.cs
│       │       ├── ReadOnlySequence.cs
│       │       ├── ReadOnlySequence.Helpers.cs
│       │       ├── SequenceReader.cs
│       │       └── SequenceReader.Search.cs
│       ├── System.Private.CoreLib/src/System/
│       │   ├── Buffer.cs
│       │   ├── ByReference.cs
│       │   ├── Char.cs
│       │   ├── CharEnumerator.cs
│       │   ├── DateTime.cs
│       │   ├── DateTimeOffset.cs
│       │   ├── Index.cs
│       │   ├── IUtfChar.cs
│       │   ├── Marvin.cs
│       │   ├── Memory.cs
│       │   ├── MemoryDebugView.cs
│       │   ├── MemoryExtensions.cs
│       │   ├── MemoryExtensions.Globalization.cs
│       │   ├── MemoryExtensions.Globalization.Utf8.cs
│       │   ├── MemoryExtensions.Trim.cs
│       │   ├── MemoryExtensions.Trim.Utf8.cs
│       │   ├── Number.Parsing.cs
│       │   ├── Range.cs
│       │   ├── ReadOnlyMemory.cs
│       │   ├── ReadOnlySpan.cs
│       │   ├── Span.cs
│       │   ├── SpanDebugView.cs
│       │   ├── SpanHelpers.BinarySearch.cs
│       │   ├── SpanHelpers.Byte.cs
│       │   ├── SpanHelpers.ByteMemOps.cs
│       │   ├── SpanHelpers.Char.cs
│       │   ├── SpanHelpers.cs
│       │   ├── SpanHelpers.Packed.cs
│       │   ├── SpanHelpers.T.cs
│       │   ├── ThrowHelper.cs
│       │   ├── Buffers/
│       │   │   ├── MemoryHandle.cs
│       │   │   └── MemoryManager.cs
│       │   ├── Globalization/
│       │   │   ├── CharUnicodeInfo.cs
│       │   │   ├── GlobalizationMode.cs
│       │   │   ├── TextInfo.cs
│       │   │   └── UnicodeCategory.cs
│       │   ├── Numerics/
│       │   │   ├── BitOperations.cs
│       │   │   ├── Vector.cs
│       │   │   └── Vector_1.cs
│       │   ├── Runtime/
│       │   │   ├── CompilerServices/
│       │   │   │   ├── IntrinsicAttribute.cs
│       │   │   │   ├── RuntimeHelpers.cs
│       │   │   │   └── Unsafe.cs
│       │   │   ├── InteropServices/
│       │   │   │   ├── MemoryMarshal.cs
│       │   │   │   ├── NativeMemory.cs
│       │   │   │   └── Marshalling/
│       │   │   │       ├── ReadOnlySpanMarshaller.cs
│       │   │   │       └── SpanMarshaller.cs
│       │   │   └── Intrinsics/
│       │   │       ├── Vector128.cs
│       │   │       └── Vector256.cs
│       │   ├── SearchValues/
│       │   │   └── SearchValues.cs
│       │   └── Text/
│       │       ├── Ascii.cs
│       │       ├── Ascii.CaseConversion.cs
│       │       ├── Ascii.Equality.cs
│       │       ├── Ascii.Transcoding.cs
│       │       ├── Ascii.Trimming.cs
│       │       ├── Ascii.Utility.cs
│       │       ├── Ascii.Utility.Helpers.cs
│       │       ├── Latin1Utility.cs
│       │       ├── Latin1Utility.Helpers.cs
│       │       ├── Rune.cs
│       │       ├── SpanLineEnumerator.cs
│       │       ├── SpanRuneEnumerator.cs
│       │       ├── UnicodeDebug.cs
│       │       ├── UnicodeUtility.cs
│       │       └── Unicode/
│       │           ├── Utf8Utility.cs
│       │           └── Utf16Utility.cs
│       └── System.Runtime/ref/
│           └── System.Runtime.cs
└── INDEX.md (this file)
```

---

## File Inventory

### DateTime Types (source for DateTime64)
| File | Lines | Description |
|------|-------|-------------|
| `System/DateTime.cs` | 2061 | `DateTime` struct - 100-ns ticks in `ulong _dateData` (top 2 bits = `DateTimeKind`, low 62 = `Ticks`). Range `[0, 3,155,378,975,999,999,999]`. Template for `DateTime64`. |
| `System/DateTimeOffset.cs` | 1046 | `DateTimeOffset` struct - `DateTime` + offset in minutes. Used for `DateTime64` ↔ `DateTimeOffset` interop. |

### Primitive Types (Char family)
| File | Lines | Description |
|------|-------|-------------|
| `System/Char.cs` | 2,066 | `char` struct (UTF-16 code unit). Source of truth for `Char8` port — Unicode category/numeric lookups, IsDigit/IsLetter/IsWhiteSpace, ToUpper/ToLower, UTF-16 surrogate helpers, parsing, formatting, `IUtfChar<char>` implementation, operator overloads. |
| `System/CharEnumerator.cs` | 55 | `CharEnumerator` - foreach iteration over chars in a string. |
| `System/IUtfChar.cs` | 35 | `IUtfChar<TSelf>` interface - abstracts UTF-8 / UTF-16 code units for generic UTF algorithms. Char implements it with 16-bit semantics; Char8 will implement it with 8-bit semantics. |
| `System/Globalization/CharUnicodeInfo.cs` | 542 | Unicode category lookups, numeric value lookups, surrogate constants (HIGH_SURROGATE_START, LOW_SURROGATE_END, etc.). Used by `Char.IsLetter` / `Char.IsDigit` for non-Latin-1 chars. |
| `System/Globalization/UnicodeCategory.cs` | 39 | `UnicodeCategory` enum (UppercaseLetter, DecimalDigitNumber, SpaceSeparator, etc.). |
| `System/Globalization/GlobalizationMode.cs` | 99 | Invariant / ICU / NLS mode flags. Referenced by Char.cs for culture-aware paths. |
| `System/Globalization/TextInfo.cs` | 844 | Culture-aware `ToUpper`/`ToLower` for chars/strings. Char.cs delegates to this for non-Latin-1 chars. **Not needed for Char8** (ASCII bit-flip suffices), but kept for reference. |

### Text: ASCII / Latin-1 / Unicode / Rune
| File | Lines | Description |
|------|-------|-------------|
| `System/Text/Ascii.cs` | 230 | `Ascii` static class — `IsValid`, `Equals`, `EqualsIgnoreCase`, `ToUpper`, `ToLower`, `Trim*`, `FromUtf16`, `ToUtf16`, transcoding. **Core API template for Char8.** |
| `System/Text/Ascii.CaseConversion.cs` | 527 | SIMD-vectorized ASCII case conversion — bit-flip upper/lower, cross-UTF-8/UTF-16 transcoding. |
| `System/Text/Ascii.Equality.cs` | 593 | ASCII equality checks, case-insensitive comparisons, ordinal equality with SIMD. |
| `System/Text/Ascii.Transcoding.cs` | 82 | Transcoding between ASCII byte representation and UTF-8/UTF-16 (entry points). |
| `System/Text/Ascii.Trimming.cs` | 83 | ASCII whitespace trimming helpers. |
| `System/Text/Ascii.Utility.cs` | 2,333 | Low-level SIMD-accelerated ASCII validation/scanning (`GetIndexOfFirstNonAsciiByte`, widening, narrowing). |
| `System/Text/Ascii.Utility.Helpers.cs` | 87 | SIMD vector helpers for Ascii.Utility. |
| `System/Text/Latin1Utility.cs` | 1,119 | Latin-1 (ISO-8859-1, 0x00–0xFF) validation, narrow/widen between `byte` (Char8) and `char` — **directly applicable to Char8 ↔ char interop**. |
| `System/Text/Latin1Utility.Helpers.cs` | 109 | Latin-1 SIMD helpers. |
| `System/Text/Rune.cs` | 1,564 | `Rune` struct — a full Unicode scalar value (21 bits). UTF-8/UTF-16 decoding, classification. Useful for Char8 → Unicode round-trip scenarios. |
| `System/Text/UnicodeUtility.cs` | 185 | `IsValidUnicodeScalar`, `IsSurrogateCodePoint`, ASCII/BMP range checks. |
| `System/Text/UnicodeDebug.cs` | 75 | Debug helpers for Unicode (`AssertIsValidCodePoint`, etc.). |
| `System/Text/Unicode/Utf8Utility.cs` | 296 | UTF-8 encoding/decoding helpers. |
| `System/Text/Unicode/Utf16Utility.cs` | 314 | UTF-16 encoding/decoding helpers, surrogate pair handling. |

### Parsing & Conversion Helpers
| File | Lines | Description |
|------|-------|-------------|
| `Common/src/System/HexConverter.cs` | 616 | Hex digit parsing (`IsHexChar`, `IsHexUpperChar`, `IsHexLowerChar`, FromChar, ToCharUpper, ToCharLower). Used by `Char.IsAsciiHexDigit`. |
| `System/Number.Parsing.cs` | 1,505 | Number parsing infrastructure — `ThrowOverflowException<T>` referenced by `Char.TryParse`. Heavyweight (pulls in full number parsing); likely stubbed for Char8. |


### Core Span Types
| File | Lines | Description |
|------|-------|-------------|
| `System/Span.cs` | 451 | `Span<T>` ref struct - contiguous memory region |
| `System/ReadOnlySpan.cs` | 421 | `ReadOnlySpan<T>` ref struct - read-only view |

### Memory Types
| File | Lines | Description |
|------|-------|-------------|
| `System/Memory.cs` | 486 | `Memory<T>` struct - heap-storable span wrapper |
| `System/ReadOnlyMemory.cs` | 408 | `ReadOnlyMemory<T>` struct |
| `System/Buffers/MemoryHandle.cs` | 56 | Pinned memory handle |
| `System/Buffers/MemoryManager.cs` | 73 | Abstract memory owner |
| `System/Buffers/MemoryPool.cs` | 51 | Memory pool abstraction |
| `System/Buffers/ArrayMemoryPool.cs` | 24 | Array-backed memory pool |

### Extension Methods (MemoryExtensions)
| File | Lines | Description |
|------|-------|-------------|
| `System/MemoryExtensions.cs` | 6,473 | **Main extensions** - Contains, IndexOf, IndexOfAny, LastIndexOf, SequenceEqual, StartsWith, EndsWith, Trim, Sort, BinarySearch, CopyTo, ToArray, etc. |
| `System/MemoryExtensions.Trim.cs` | 877 | Trim, TrimStart, TrimEnd |
| `System/MemoryExtensions.Globalization.cs` | 414 | Culture-aware comparisons |
| `System/MemoryExtensions.Globalization.Utf8.cs` | 78 | UTF-8 globalization |
| `System/MemoryExtensions.Trim.Utf8.cs` | 76 | UTF-8 trimming |

### SpanHelpers (Internal Implementation)
| File | Lines | Description |
|------|-------|-------------|
| `System/SpanHelpers.cs` | 347 | Base helpers, Fill, ClearWithReferences |
| `System/SpanHelpers.T.cs` | 4,235 | **Generic helpers** - IndexOf, LastIndexOf, SequenceEqual, etc. |
| `System/SpanHelpers.Byte.cs` | 1,469 | Byte-optimized operations |
| `System/SpanHelpers.Char.cs` | 1,014 | Char-optimized operations |
| `System/SpanHelpers.Packed.cs` | 1,345 | Packed/SIMD search implementations |
| `System/SpanHelpers.ByteMemOps.cs` | 591 | Byte memory operations |
| `System/SpanHelpers.BinarySearch.cs` | 78 | Binary search implementation |

### Memory Marshalling
| File | Lines | Description |
|------|-------|-------------|
| `System/Runtime/InteropServices/MemoryMarshal.cs` | 621 | Low-level memory operations |
| `MemoryMarshal.CoreCLR.cs` | 47 | CoreCLR-specific implementations |
| `Marshalling/SpanMarshaller.cs` | 214 | P/Invoke span marshalling |
| `Marshalling/ReadOnlySpanMarshaller.cs` | 240 | P/Invoke read-only span marshalling |

### Buffer & Sequences
| File | Lines | Description |
|------|-------|-------------|
| `System/Buffer.cs` | 214 | Buffer operations, Memmove |
| `System/Buffers/IBufferWriter.cs` | 51 | Buffer writer interface |
| `System/Buffers/BuffersExtensions.cs` | 157 | Buffer extension methods |
| `System/Buffers/ReadOnlySequence.cs` | 687 | Discontiguous memory sequence |
| `System/Buffers/ReadOnlySequence.Helpers.cs` | 698 | Sequence helper methods |
| `System/Buffers/SequenceReader.cs` | 457 | Sequence reader |
| `System/Buffers/SequenceReader.Search.cs` | 851 | Sequence search operations |

### SIMD / Vectors
| File | Lines | Description |
|------|-------|-------------|
| `System/Runtime/Intrinsics/Vector128.cs` | 4,562 | 128-bit SIMD vector |
| `System/Runtime/Intrinsics/Vector256.cs` | 4,475 | 256-bit SIMD vector |
| `System/Numerics/Vector.cs` | 3,582 | Platform-agnostic vector |
| `System/Numerics/Vector_1.cs` | 1,235 | `Vector<T>` generic |
| `System/Numerics/BitOperations.cs` | 958 | Bit manipulation (PopCount, LeadingZeroCount, etc.) |

### Utilities & Helpers
| File | Lines | Description |
|------|-------|-------------|
| `System/ThrowHelper.cs` | 1,457 | Exception throwing helpers |
| `System/Runtime/CompilerServices/Unsafe.cs` | 1,028 | Unsafe memory operations |
| `System/Runtime/CompilerServices/RuntimeHelpers.cs` | 193 | Runtime helper methods |
| `System/Index.cs` | 168 | `Index` struct (^ operator) |
| `System/Range.cs` | 134 | `Range` struct (.. operator) |
| `System/Marvin.cs` | 276 | Marvin hash algorithm |
| `System/ByReference.cs` | 19 | Internal ref helper |
| `System/SearchValues/SearchValues.cs` | 315 | Search value optimizations |
| `System/Runtime/InteropServices/NativeMemory.cs` | 96 | Native memory allocation |

### Attributes
| File | Lines | Description |
|------|-------|-------------|
| `Runtime/CompilerServices/IntrinsicAttribute.cs` | 13 | JIT intrinsic marker |
| `Runtime/Versioning/NonVersionableAttribute.cs` | 32 | Version stability marker |

### Debug Views
| File | Lines | Description |
|------|-------|-------------|
| `System/SpanDebugView.cs` | 25 | Debugger visualization for Span |
| `System/MemoryDebugView.cs` | 25 | Debugger visualization for Memory |

### Enumerators
| File | Lines | Description |
|------|-------|-------------|
| `System/Text/SpanLineEnumerator.cs` | 91 | Line-by-line enumeration |
| `System/Text/SpanRuneEnumerator.cs` | 63 | Unicode rune enumeration |

### Reference Assemblies (API Surface)
| File | Lines | Description |
|------|-------|-------------|
| `System.Memory/ref/System.Memory.cs` | 837 | System.Memory public API |
| `System.Runtime/ref/System.Runtime.cs` | 17,366 | System.Runtime public API (includes Span) |

---

## Key APIs to Convert for UnmanagedSpan<T>

### From Span.cs
- `Span(T[]? array)` - array constructor
- `Span(T[]? array, int start, int length)` - array slice constructor
- `Span(void* pointer, int length)` - **KEY: change to long**
- `Span(ref T reference)` - single element
- `this[int index]` - **KEY: change to long**
- `int Length` - **KEY: change to long**
- `bool IsEmpty`
- `Enumerator GetEnumerator()`
- `ref T GetPinnableReference()`
- `void Clear()`
- `void Fill(T value)`
- `void CopyTo(Span<T> destination)`
- `bool TryCopyTo(Span<T> destination)`
- `Span<T> Slice(int start)` - **KEY: change to long**
- `Span<T> Slice(int start, int length)` - **KEY: change to long**
- `T[] ToArray()`
- Operators: `==`, `!=`, implicit conversions

### From MemoryExtensions.cs (6,473 lines)
Critical extension methods to port:
- `Contains<T>(this Span<T>, T)`
- `IndexOf<T>(this Span<T>, T)`
- `IndexOf<T>(this Span<T>, ReadOnlySpan<T>)`
- `IndexOfAny<T>(this Span<T>, ...)`
- `LastIndexOf<T>(...)`
- `LastIndexOfAny<T>(...)`
- `SequenceEqual<T>(...)`
- `StartsWith<T>(...)`
- `EndsWith<T>(...)`
- `Reverse<T>(...)`
- `Sort<T>(...)`
- `BinarySearch<T>(...)`
- `CopyTo<T>(...)`
- `ToArray<T>(...)`
- `Trim(...)`, `TrimStart(...)`, `TrimEnd(...)`

### From SpanHelpers.T.cs (4,235 lines)
Internal implementations using SIMD:
- `IndexOf<T>` / `IndexOfValueType<T>`
- `LastIndexOf<T>`
- `IndexOfAny<T>` (2, 3, 4, 5 values)
- `SequenceEqual<T>`
- `SequenceCompareTo<T>`
- `Fill<T>`
- `CopyTo<T>` / `Memmove<T>`
- `Reverse<T>`

---

## Conversion Strategy

1. **Phase 1:** Core `UnmanagedSpan<T>` struct
   - Change `int _length` to `long _length`
   - Change all index parameters from `int` to `long`
   - Keep `ref T _reference` pattern (or use `T*` for unmanaged)

2. **Phase 2:** `ReadOnlyUnmanagedSpan<T>`
   - Read-only variant

3. **Phase 3:** Extension methods
   - Port critical MemoryExtensions methods
   - Adapt SIMD helpers for long indexing

4. **Phase 4:** Helper methods
   - SpanHelpers with long support
   - ThrowHelper adaptations

---

## Key APIs to Port for Char8

`Char8` is a 1-byte character type that maps to NumPy's `"S1"` / `"c"` dtype and interops with C#'s `char` (UTF-16) and `string`. Each method is adapted from `Char.cs` but operates on a single byte (0–255) rather than a UTF-16 code unit.

### Core struct layout
- `[StructLayout(LayoutKind.Sequential)]` with a single `byte _value`
- Implements `IComparable`, `IComparable<Char8>`, `IEquatable<Char8>`, `IConvertible`, `ISpanFormattable`, `IUtfChar<Char8>` (1-byte variant)
- Implicit conversions: `Char8 ↔ byte`, `Char8 → char` (when ≤ 0x7F or via ISO-8859-1), `Char8 → int`
- Explicit conversions: `char → Char8` (truncation or throw on non-ASCII)

### Classification predicates (ASCII fast path)
- `IsDigit(Char8)` — `'0'..'9'`
- `IsLetter(Char8)` — ASCII letters only (no Unicode category lookup)
- `IsLetterOrDigit(Char8)`
- `IsWhiteSpace(Char8)` — `' '`, `'\t'`, `'\n'`, `'\r'`, `'\v'`, `'\f'`
- `IsUpper(Char8)`, `IsLower(Char8)`
- `IsPunctuation(Char8)`, `IsSymbol(Char8)`, `IsControl(Char8)`
- `IsAscii(Char8)` — always `value <= 0x7F`
- `IsAsciiDigit/Letter/LetterOrDigit/HexDigit` — fast ASCII-only checks

### Case conversion
- `ToUpper(Char8)` / `ToLower(Char8)` — ASCII-only (bit flip), throws or no-op for non-ASCII
- `ToUpperInvariant(Char8)` / `ToLowerInvariant(Char8)` — identical to ASCII versions

### Parsing & formatting
- `Parse(string)` — single character or throws
- `TryParse(string, out Char8)`
- `ToString()` — returns `string` of length 1 (ASCII interop via default encoding)
- `TryFormat(Span<char>, out int written, ...)` — writes 1 char

### Numeric lookups
- `GetNumericValue(Char8)` — returns double (0.0–9.0 for digits, -1.0 otherwise)

### Operators
- `==`, `!=`, `<`, `>`, `<=`, `>=`
- Implements `IEqualityOperators<Char8, Char8, bool>`, `IComparisonOperators<Char8, Char8, bool>`
- Implements `IIncrementOperators<Char8>`, `IDecrementOperators<Char8>`
- Implements `IAdditionOperators<Char8, Char8, Char8>`, etc. (modular arithmetic on byte)

### String / ASCII interop
- `Char8[] FromString(string)` — encodes string as ASCII bytes (throws on non-ASCII)
- `string ToString(Char8[])` — decodes ASCII bytes to string
- `FromAsciiString(ReadOnlySpan<char>) → Char8[]`
- Implicit `ReadOnlySpan<Char8> → ReadOnlySpan<byte>` for interop with UTF-8 APIs

### IUtfChar<Char8> implementation
- `CastFrom(byte value)` → `(Char8)value`
- `CastFrom(char value)` → throws or truncates if > 0xFF
- `CastFrom(int value)` → `(Char8)(byte)value`
- `CastToUInt32(Char8 value)` → `value` (byte → uint)

---

## Char8 Port Strategy

1. **Phase 1:** `Char8` struct in NumSharp
   - Define `public readonly struct Char8 : IComparable<Char8>, IEquatable<Char8>, IConvertible, IUtfChar<Char8>`
   - Single `byte _value` field — 1-byte layout
   - Conversion operators to/from `byte`, `char`, `int`

2. **Phase 2:** ASCII classification & case conversion
   - Port `IsDigit`, `IsLetter`, `IsUpper`, `IsLower`, `IsWhiteSpace`, `IsControl`, etc. as ASCII-only
   - Port `ToUpper`, `ToLower` via bit manipulation (no locale)

3. **Phase 3:** String/ASCII round-trip
   - `Char8[] FromString(string)` / `string ToString(Char8[])`
   - `FromUtf8`, `ToUtf8` span helpers

4. **Phase 4:** NumSharp integration
   - Add `NPTypeCode.Char8` enum value (= 1 byte)
   - `InfoOf<Char8>.Size = 1`
   - `np.dtype("S1")` / `np.dtype("c")` → `NPTypeCode.Char8`
   - `NDArray<Char8>` indexing, `SetChar8`/`GetChar8`
   - Wire into `np.frombuffer`, `np.array`, cast table, IL kernels

5. **Phase 5:** Formatting & parsing
   - `TryFormat`, `TryParse`
   - `IUtfChar<Char8>` members

---

## Transitive Dependencies NOT Fetched

The following are referenced by the fetched files but intentionally **not pulled in**, because they lead deep into runtime internals that are not needed for Char8 (byte-sized, ASCII/Latin-1 only) and would balloon the surface area:

| Missing type | Referenced by | Why skipped |
|--------------|--------------|-------------|
| `PackedSpanHelpers` | `Ascii.Utility.cs` | SIMD search shortcut — can substitute `SpanHelpers.Packed.cs` (already present) or stub. |
| `AppContextConfigHelper` | `GlobalizationMode.cs` | Runtime config switches — for Char8 we assume invariant mode, so this is irrelevant. Stub to `false` / defaults. |
| `LocalAppContextSwitches` | (various) | Same as above. |
| `CultureData` | `TextInfo.cs` | Full ICU/NLS culture data. Char8 case conversion is ASCII bit-flip — delete all culture paths in the ported TextInfo. |
| `CompareInfo` | `TextInfo.cs` | Culture-aware comparison. Not needed for byte comparison. |
| `NumberBuffer` / `NumberFormatInfo` | `Number.Parsing.cs` | Full numeric parsing infrastructure. For `Char8.TryParse` we only need single-character parsing — reimplement locally instead of dragging in `BigInteger`, `Grisu3`, `Dragon4`. |
| `SR.*` resource strings | many | Localized error messages. Substitute with hardcoded English strings or `nameof(...)`. |
| `ThrowHelper` resource-based members | many | NumSharp has its own `ThrowHelper`. Wire ported code to it. |
| `Utf8Utility.*` partials beyond core | `Ascii.CaseConversion.cs` | The fetched `Utf8Utility.cs` is the entry point; the massive partial classes (`Utf8Utility.Transcoding.cs`, etc.) that it forwards to are omitted — add on demand. |
| `Utf16Utility.*` partials | same | same |

**Rule of thumb:** when porting, if a Char.cs member depends on any of these transitively, either:
1. Rewrite using Char8's simpler (ASCII/Latin-1) semantics, or
2. Stub the call and throw `NotSupportedException` until needed.

---

## License

All files are from the .NET Runtime repository and are licensed under the MIT License.
See: https://github.com/dotnet/runtime/blob/main/LICENSE.TXT
