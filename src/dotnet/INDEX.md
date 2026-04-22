# .NET Runtime Source Files

Downloaded from [dotnet/runtime](https://github.com/dotnet/runtime) `main` branch (.NET 10).

**Purpose:**
1. Source of truth for converting `Span<T>` to `UnmanagedSpan<T>` with `long` indexing support.
2. Reference/template for `DateTime64` struct (NumPy-parity datetime64 with full `long` range) in `src/NumSharp.Core/DateTime64.cs` — forked from `DateTime.cs` with `ulong _dateData` replaced by `long _ticks`, `DateTimeKind` bits removed, range expanded to the full `long` space, and `NaT == long.MinValue` sentinel added.

**Total:** 55 files | ~63,000 lines of code

---

## Directory Structure

```
src/dotnet/
├── src/
│   ├── coreclr/System.Private.CoreLib/src/System/Runtime/InteropServices/
│   │   └── MemoryMarshal.CoreCLR.cs
│   └── libraries/
│       ├── Common/src/System/Runtime/Versioning/
│       │   └── NonVersionableAttribute.cs
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
│       │   ├── DateTime.cs
│       │   ├── DateTimeOffset.cs
│       │   ├── Index.cs
│       │   ├── Marvin.cs
│       │   ├── Memory.cs
│       │   ├── MemoryDebugView.cs
│       │   ├── MemoryExtensions.cs
│       │   ├── MemoryExtensions.Globalization.cs
│       │   ├── MemoryExtensions.Globalization.Utf8.cs
│       │   ├── MemoryExtensions.Trim.cs
│       │   ├── MemoryExtensions.Trim.Utf8.cs
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
│       │       ├── SpanLineEnumerator.cs
│       │       └── SpanRuneEnumerator.cs
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

## License

All files are from the .NET Runtime repository and are licensed under the MIT License.
See: https://github.com/dotnet/runtime/blob/main/LICENSE.TXT
