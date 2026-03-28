# [Rewrite] np.save/load - NPY/NPZ Format (NEP-01)

## Overview

NumSharp's implementation of `.npy` and `.npz` file I/O needs a complete rewrite. The current code was written as a quick solution years ago and has accumulated significant technical debt. It diverges from NumPy's behavior in ways that break interoperability, throws exceptions on valid files that NumPy handles gracefully, and exposes an API that doesn't match NumPy's signatures. This issue proposes bringing the implementation into full compliance with NEP-01 (the NumPy Enhancement Proposal that defines the binary format) and achieving API parity with NumPy 2.x.

## The Problem

### We Only Support a Fraction of the Format

The NPY format has evolved through three versions since its introduction in NumPy 1.0.5. Version 1.0 uses a 2-byte header length field, limiting headers to 65KB - plenty for simple arrays but insufficient for structured arrays with many fields. Version 2.0, introduced in NumPy 1.9, extended this to a 4-byte field supporting headers up to 4GB. Version 3.0, added in NumPy 1.17, switched the header encoding from Latin-1 to UTF-8 to support Unicode field names in structured dtypes.

NumSharp currently only recognizes version 1.0 and throws `NotSupportedException` when encountering version 2.0 or 3.0 files. This means any `.npy` file saved by modern NumPy with a large structured dtype or Unicode field names simply cannot be loaded. Users hitting this wall have no workaround other than re-saving their data in NumPy with simpler dtypes.

### Header Alignment is Wrong

One of the cleverest aspects of the NPY format is that the header is padded so the data section begins at a 64-byte aligned offset. This alignment enables memory-mapped access to the array data without copying - the operating system can map the file directly into virtual memory and the CPU can access the data with aligned SIMD instructions.

NumSharp's implementation uses 16-byte alignment instead of 64. Files written by NumSharp technically conform to the format specification (alignment is not strictly required), but they lose the performance benefits of memory mapping. More importantly, this signals a fundamental misunderstanding of the format that likely indicates other subtle bugs.

### We Reject Valid Data Layouts

NumPy arrays can be stored in either C-order (row-major) or Fortran-order (column-major) layout. The format captures this in the `fortran_order` header field. When NumPy encounters a Fortran-order file, it reads the data, reshapes with reversed dimensions, then transposes to produce the correct array.

NumSharp throws an exception when `fortran_order` is `True`. This is particularly frustrating because Fortran-order arrays are common in scientific computing - they're the native layout for MATLAB, R, and many numerical libraries. A user trying to load data exported from these tools will simply get an error with no path forward.

The same issue applies to byte order. NumPy files can contain big-endian data (indicated by `>` in the dtype descriptor). NumPy handles this transparently by byte-swapping on read when the host system uses a different byte order. NumSharp throws an exception. While big-endian systems are rare today, big-endian files still exist - especially in legacy scientific datasets and network protocols.

### The API Doesn't Match NumPy

When users learn NumPy, they learn to call `np.save()`, `np.load()`, `np.savez()`, and `np.savez_compressed()`. These function signatures have specific parameters with specific defaults that users come to expect.

NumSharp's API diverges in several ways. The methods are named `Save_Npz` and `Load_Npz` instead of `savez` - mixing PascalCase with underscores in a way that matches neither .NET conventions nor NumPy conventions. The `load` function is missing critical parameters like `allow_pickle` (which controls whether object arrays can be loaded), `mmap_mode` (for memory-mapped access), and `max_header_size` (a security feature that limits header parsing to prevent denial-of-service attacks from malicious files).

The `Load_Npz<T>` method requires a generic type parameter with a complex constraint (`where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable`). This constraint exists because the implementation returns .NET arrays rather than NDArrays, but it creates an awkward API that doesn't exist in NumPy. Users shouldn't need to specify type parameters to load a file - the type information is in the file itself.

### NPZ Handling Has Subtle Bugs

The `.npz` format is simply a ZIP archive containing multiple `.npy` files. NumPy's `NpzFile` class provides lazy loading - arrays aren't actually read from disk until accessed. It also provides a convenient interface where both `npz['arr_0']` and `npz['arr_0.npy']` work as keys, with the `.files` property returning the stripped names.

NumSharp's `NpzDictionary` class doesn't strip the `.npy` extension, so code written for NumPy that accesses arrays by their logical names will fail. The class also doesn't implement the context manager protocol properly for deterministic cleanup, potentially leaking file handles.

### Security Features Are Missing

NumPy added the `max_header_size` parameter after discovering that `ast.literal_eval()` (used to parse the header dictionary) can be slow or even crash on extremely large inputs. A malicious actor could craft a `.npy` file with a header designed to cause denial of service. The default limit of 10,000 bytes is sufficient for any legitimate array while protecting against attacks.

NumPy also changed `allow_pickle` to default to `False` in version 1.16.3 after security researchers demonstrated that pickle deserialization could execute arbitrary code. Object arrays in `.npy` files are serialized using pickle, so loading an untrusted file with `allow_pickle=True` is equivalent to running untrusted code.

NumSharp has neither protection. There's no header size limit, and there's no pickle support at all (which means object arrays simply can't be loaded, but also means there's no parameter to control this behavior).

### Dtype Coverage is Incomplete

The dtype descriptor in an NPY header maps directly to NumPy's dtype system. NumSharp's parser handles the common cases but has gaps and bugs.

The unsigned byte type `|u1` is mapped to signed byte, which is simply wrong. Complex number types (`<c8` for complex64, `<c16` for complex128) aren't recognized at all, even though .NET has `System.Numerics.Complex`. Unicode strings (`<U` prefix) aren't supported - only ASCII byte strings (`|S` prefix). The datetime64 and timedelta64 types, which are heavily used in pandas, aren't supported.

Structured dtypes (record arrays) aren't supported either, though this is a larger undertaking that might reasonably be deferred.

## What We Need to Do

### Rewrite the Core Format Handling

The implementation should follow NumPy's `_format_impl.py` closely. This file is well-documented and has been battle-tested over nearly two decades. Key elements include:

The magic number validation should properly check for `\x93NUMPY` (where `\x93` is byte value 147, not the character '?'). The current code happens to work because `BinaryReader.ReadChar()` with ASCII encoding returns 63 for non-ASCII bytes, but this is accidental.

Header parsing should use a proper tokenizer or at minimum careful string parsing that handles edge cases like trailing commas, tuple shapes `(3,)` for 1D arrays, and the exact set of allowed keys. The current implementation uses `IndexOf` and `Substring` which is fragile.

Writing should automatically select the minimum version that can represent the data - version 1.0 for small headers with ASCII field names, version 2.0 for large headers, version 3.0 for Unicode field names.

### Handle All Valid Data Layouts

Fortran-order arrays should be readable and writable. On read, the data should be read linearly, reshaped with reversed dimensions, then transposed. On write, Fortran-contiguous arrays should be detected and written with `fortran_order: True`.

Big-endian data should be byte-swapped to native order on read. NumSharp always uses native byte order internally, so this is a read-time conversion. Writing always uses native order (little-endian on all common platforms today).

### Implement Proper NPZ Support

Replace `NpzDictionary<T>` with an `NpzFile` class that matches NumPy's interface. This class should:

- Provide lazy loading with internal caching (arrays loaded on first access, then cached)
- Accept both `"arr_0"` and `"arr_0.npy"` as valid keys
- Expose a `files` property with stripped names
- Implement `IDisposable` properly for the underlying ZIP archive
- Support the `f` attribute for dot-notation access (`npz.f.weights` instead of `npz["weights"]`)

The `savez` and `savez_compressed` functions should accept both positional arrays (named `arr_0`, `arr_1`, etc.) and keyword arguments. They should always enable Zip64 extensions to support archives larger than 4GB.

### Match the NumPy API

The public API should match NumPy's signatures:

```python
np.save(file, arr, allow_pickle=True)
np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
        encoding='ASCII', *, max_header_size=10000)
np.savez(file, *args, allow_pickle=True, **kwds)
np.savez_compressed(file, *args, allow_pickle=True, **kwds)
```

The `allow_pickle` parameter should be implemented even though we don't support pickle - it should raise a clear error when `allow_pickle=False` and an object array is encountered, matching NumPy's behavior.

The `mmap_mode` parameter enables memory-mapped access. While full implementation is complex, at minimum we should accept the parameter and either implement basic read-only mapping or raise `NotImplementedException` with a clear message.

### Add Security Protections

Implement the `max_header_size` check. Before parsing the header with any string operations, verify its length is within the limit. If `allow_pickle=True`, the limit can be bypassed (the user has already indicated they trust the file).

Validate the header dictionary strictly: exactly three keys (`descr`, `fortran_order`, `shape`), `shape` must be a tuple of non-negative integers, `fortran_order` must be a boolean, `descr` must produce a valid dtype.

### Improve Error Messages

Replace generic exceptions with descriptive messages that help users understand what went wrong. NumPy's error messages are a good template:

- `"the magic string is not correct; expected b'\\x93NUMPY', got {actual!r}"`
- `"we only support format version (1,0), (2,0), and (3,0), not {version}"`
- `"Header does not contain the correct keys: {keys!r}"`
- `"Object arrays cannot be loaded when allow_pickle=False"`

## Breaking Changes

This rewrite will change the public API. The `Save_Npz` and `Load_Npz` methods will be deprecated in favor of `savez` and `load`. The `NpzDictionary<T>` class will be replaced by `NpzFile`. Code that depends on these will need to be updated.

The default value of `allow_pickle` on `load` will be `False` to match NumPy's security-conscious default. Code that loads object arrays will need to explicitly pass `allow_pickle=True`.

Files written by the old implementation with 16-byte alignment will still be readable, but memory mapping performance characteristics may differ.

## What We're Not Doing (Yet)

Some features are out of scope for this rewrite:

**Object arrays** require implementing Python's pickle protocol, which is a substantial undertaking with security implications. For now, we'll raise a clear error when object arrays are encountered.

**Structured dtypes** (record arrays) are complex to map to .NET types. A future enhancement could support them via dynamically generated types or a generic record class.

**datetime64 and timedelta64** need design decisions about how to map to .NET types (`DateTime`, `DateTimeOffset`, `TimeSpan`, or custom structs).

**Memory-mapped write mode** is complex because it requires pre-allocating the file to the correct size and carefully managing the mapping lifetime.

## References

The authoritative source for the NPY format is NEP-01, available at https://numpy.org/neps/nep-0001-npy-format.html. The implementation lives in `numpy/lib/_format_impl.py` (low-level format handling) and `numpy/lib/_npyio_impl.py` (high-level `save`/`load` functions).

I've prepared comprehensive documentation of the format in `docs/numpy/NUMPY_NPZ_SAVE_LOAD.md` based on reading the NumPy source code. This covers every constant, every function, every edge case, and every error message. It should serve as the specification for our implementation.
