# NumPy .npy/.npz Format Implementation

This document describes how NumPy implements `np.save`, `np.load`, and the underlying `.npy`/`.npz` binary formats. Based on analysis of `numpy/lib/_format_impl.py` (NumPy 2.x).

## Table of Contents

1. [Binary File Structure](#binary-file-structure)
2. [Format Versions](#format-versions)
3. [Header Format](#header-format)
4. [Data Layout](#data-layout)
5. [Write Implementation](#write-implementation)
6. [Read Implementation](#read-implementation)
7. [NPZ Archive Format](#npz-archive-format)
8. [Memory Mapping](#memory-mapping)
9. [Edge Cases](#edge-cases)
10. [Security Considerations](#security-considerations)
11. [Constants Reference](#constants-reference)
12. [Implicit Behaviors and Gotchas](#implicit-behaviors-and-gotchas)
13. [Limitations and Warnings](#limitations-and-warnings)
14. [Error Messages](#error-messages)
15. [Public API Reference](#public-api-reference)

---

## Binary File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ MAGIC STRING (6 bytes)                                          │
│   Fixed value: \x93NUMPY                                        │
├─────────────────────────────────────────────────────────────────┤
│ VERSION (2 bytes)                                               │
│   major (1 byte), minor (1 byte)                                │
├─────────────────────────────────────────────────────────────────┤
│ HEADER LENGTH (2 or 4 bytes, little-endian)                     │
│   v1.0: uint16 (<H) - max 65,535 bytes                          │
│   v2.0/v3.0: uint32 (<I) - max 4 GiB                            │
├─────────────────────────────────────────────────────────────────┤
│ HEADER (variable length, padded)                                │
│   Python dict literal as ASCII/UTF-8 string                     │
│   Padded with spaces, terminated with \n                        │
│   Total (magic + version + length + header) aligned to 64 bytes │
├─────────────────────────────────────────────────────────────────┤
│ DATA (variable length)                                          │
│   Raw bytes in C-order or Fortran-order                         │
│   OR pickle stream if dtype contains Python objects             │
└─────────────────────────────────────────────────────────────────┘
```

### Byte-Level Example (version 1.0, int32 array)

```
Offset  Bytes           Description
------  --------------  ------------------------------------
0x00    93 4E 55 4D     Magic: \x93NUM
0x04    50 59           Magic: PY
0x06    01 00           Version: 1.0
0x08    76 00           Header length: 118 (little-endian)
0x0A    7B 27 64 65...  Header: {'descr': '<i4', ...}\n
...     20 20 20 0A     Padding spaces + newline
0x80    01 00 00 00...  Data starts at offset 128 (64-aligned)
```

---

## Format Versions

| Version | Header Length Field | Encoding | Introduced | Notes |
|---------|---------------------|----------|------------|-------|
| (1, 0)  | 2 bytes (`<H`)      | latin1   | NumPy 1.0.5 | Default, most compatible |
| (2, 0)  | 4 bytes (`<I`)      | latin1   | NumPy 1.9.0 | For headers > 65,535 bytes |
| (3, 0)  | 4 bytes (`<I`)      | utf-8    | NumPy 1.17  | Unicode field names |

### Version Auto-Selection Logic

```python
def _wrap_header_guess_version(header):
    # 1. Try version 1.0 first (most compatible)
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:  # Header too large
        pass

    # 2. Try version 2.0 (larger headers)
    try:
        ret = _wrap_header(header, (2, 0))
        warnings.warn("Stored array in format 2.0...")
        return ret
    except UnicodeEncodeError:  # Unicode field names
        pass

    # 3. Fall back to version 3.0 (UTF-8)
    warnings.warn("Stored array in format 3.0...")
    return _wrap_header(header, (3, 0))
```

---

## Header Format

### Header Dictionary

The header is a Python dictionary literal with exactly three keys:

```python
{
    'descr': '<i4',           # dtype descriptor
    'fortran_order': False,   # memory layout
    'shape': (3, 4),          # array dimensions
}
```

### Key Descriptions

| Key | Type | Description |
|-----|------|-------------|
| `descr` | str or list | dtype descriptor (see below) |
| `fortran_order` | bool | `True` if F-contiguous data layout |
| `shape` | tuple of int | Array dimensions; `()` for scalar |

### dtype Descriptor Encoding (`descr`)

#### Simple Types

Format: `<endian><type><size>`

| Endian | Meaning |
|--------|---------|
| `<` | Little-endian |
| `>` | Big-endian |
| `\|` | Not applicable (single-byte or opaque) |
| `=` | Native byte order |

| Type Code | C Type | Examples |
|-----------|--------|----------|
| `b` | bool | `\|b1` |
| `i` | signed int | `\|i1`, `<i2`, `<i4`, `<i8` |
| `u` | unsigned int | `\|u1`, `<u2`, `<u4`, `<u8` |
| `f` | float | `<f2`, `<f4`, `<f8`, `<f16` |
| `c` | complex | `<c8`, `<c16`, `<c32` |
| `M` | datetime64 | `<M8`, `<M8[ns]`, `<M8[D]` |
| `m` | timedelta64 | `<m8`, `<m8[ns]`, `<m8[s]` |
| `S` | byte string | `\|S10` (10-byte string) |
| `U` | unicode string | `<U5` (5-char unicode) |
| `V` | void/opaque | `\|V16` (16-byte buffer) |
| `O` | object | `\|O` (Python object, uses pickle) |
| `M` | datetime64 | `<M8[ns]`, `<M8[D]` (with unit) |
| `m` | timedelta64 | `<m8[us]`, `<m8[s]` (with unit) |

**datetime64/timedelta64 notes:**
- Unit is preserved in descriptor: `<M8[ns]` for nanoseconds
- Round-trips correctly with full unit preservation
- Units: `Y`, `M`, `W`, `D`, `h`, `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, `as`

**Note**: datetime64/timedelta64 preserve their units (ns, us, ms, s, m, h, D, W, M, Y).

#### Structured Types (Record Arrays)

List of tuples: `[(name, dtype), ...]` or `[(name, dtype, shape), ...]`

```python
# Simple record
[('x', '<i4'), ('y', '<f8')]

# With subarray shape
[('matrix', '<f8', (2, 3))]

# Nested structure
[('outer', [('inner1', '<i4'), ('inner2', '<f8')]), ('value', '|u1')]
```

### dtype_to_descr Implementation

```python
def dtype_to_descr(dtype):
    # Strip metadata (emits warning if present)
    dtype = drop_metadata(dtype)

    if dtype.names is not None:
        # Record array: use .descr
        return dtype.descr
    elif not type(dtype)._legacy:
        # Non-legacy user dtype: use pickle
        warnings.warn("Custom dtypes saved as pickle...")
        return "|O"
    else:
        # Simple type: use .str
        return dtype.str
```

### descr_to_dtype Implementation

```python
def descr_to_dtype(descr):
    if isinstance(descr, str):
        # Simple type: "<i4", "|b1", etc.
        return numpy.dtype(descr)

    elif isinstance(descr, tuple):
        # Subarray: (dtype_descr, shape)
        # Example: ('<f8', (2, 3)) -> 2x3 array of float64
        dt = descr_to_dtype(descr[0])
        return numpy.dtype((dt, descr[1]))

    else:
        # Record array: list of field tuples
        # Each field: (name, descr) or (name, descr, shape)
        # Name can be string or (title, name) tuple
        names, formats, titles, offsets = [], [], [], []
        offset = 0

        for field in descr:
            if len(field) == 2:
                name, descr_str = field
                dt = descr_to_dtype(descr_str)
            else:
                name, descr_str, shape = field
                dt = numpy.dtype((descr_to_dtype(descr_str), shape))

            # Skip padding bytes (aligned dtypes insert void padding)
            # Condition: empty name AND void type AND no subfields
            is_pad = (name == '' and dt.type is numpy.void and dt.names is None)

            if not is_pad:
                # Handle titled fields: name can be (title, actual_name)
                title, name = name if isinstance(name, tuple) else (None, name)
                titles.append(title)
                names.append(name)
                formats.append(dt)
                offsets.append(offset)

            # Always advance offset (padding consumes space)
            offset += dt.itemsize

        return numpy.dtype({
            'names': names,
            'formats': formats,
            'titles': titles,
            'offsets': offsets,
            'itemsize': offset  # Total size including padding
        })
```

**Field tuple formats:**
- `(name, dtype_str)` - Simple field
- `(name, dtype_str, shape)` - Subarray field
- `((title, name), dtype_str)` - Titled field
- `('', 'V8')` - Padding (8 bytes of void, skipped)

### Header Alignment

The total header (magic + version + length field + header string) is padded to a multiple of `ARRAY_ALIGN` (64 bytes) for memory-mapping compatibility.

```python
ARRAY_ALIGN = 64

def _wrap_header(header, version):
    header_bytes = header.encode(encoding)  # latin1 or utf8
    hlen = len(header_bytes) + 1  # +1 for trailing newline

    # Calculate padding needed
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)

    # Build: magic + length + header + padding + newline
    return magic(*version) + struct.pack(fmt, hlen + padlen) + header_bytes + b' ' * padlen + b'\n'
```

### Growth Axis Padding

Extra spaces are added after the dictionary to allow in-place header modification when appending data:

```python
GROWTH_AXIS_MAX_DIGITS = 21  # len(str(8 * 2**64 - 1))

# Add padding for the axis that can grow
# C-order: first axis; F-order: last axis
growth_axis = -1 if fortran_order else 0
current_digits = len(repr(shape[growth_axis])) if shape else 0
padding = GROWTH_AXIS_MAX_DIGITS - current_digits
header += " " * padding
```

---

## Data Layout

### C-Order (Row-Major)

Default layout. Elements are contiguous along the last axis.

```
Array: [[1, 2, 3],
        [4, 5, 6]]

Memory: [1, 2, 3, 4, 5, 6]
```

### Fortran-Order (Column-Major)

Elements are contiguous along the first axis.

```
Array: [[1, 2, 3],
        [4, 5, 6]]

Memory: [1, 4, 2, 5, 3, 6]
```

### Contiguity Detection

```python
def header_data_from_array_1_0(array):
    d = {'shape': array.shape}

    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        # Non-contiguous: will be copied as C-order
        d['fortran_order'] = False

    d['descr'] = dtype_to_descr(array.dtype)
    return d
```

**Important**: 1-D arrays are both C-contiguous and F-contiguous. The check order (C first) ensures 1-D arrays use C-order.

### Contiguity Decision Table

| Array Type | C-contiguous | F-contiguous | fortran_order |
|------------|--------------|--------------|---------------|
| 1D array | True | True | False |
| 2D C-order | True | False | False |
| 2D transposed | False | True | True |
| 2D F-order | False | True | True |
| 1D strided | False | False | False |
| 2D strided | False | False | False |
| Scalar | True | True | False |
| Empty | True | True | False |

**Key insight**: Both C and F are true for 1D/scalar/empty arrays. The check order matters!

**Contiguity edge cases:**
| Array Type | C_CONTIGUOUS | F_CONTIGUOUS | fortran_order |
|------------|--------------|--------------|---------------|
| 1D array | True | True | False |
| Scalar | True | True | False |
| Empty (0,) | True | True | False |
| Empty (0,0,0) | True | True | False |
| 2D C-order | True | False | False |
| 2D F-order | False | True | True |
| Transposed | False | True | True |
| Strided slice | False | False | False |

---

## Write Implementation

### write_array Function

```python
def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)

    # Calculate buffer size (16 MiB worth of elements)
    if array.itemsize == 0:
        buffersize = 0
    else:
        buffersize = max(16 * 1024**2 // array.itemsize, 1)

    # Write data based on dtype and contiguity
    if array.dtype.hasobject or not type(array.dtype)._legacy:
        # Object arrays or custom dtypes: use pickle
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when allow_pickle=False")
        pickle.dump(array, fp, protocol=4, **pickle_kwargs or {})

    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        # F-contiguous: write in Fortran order
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(array,
                    flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tobytes('C'))

    elif isfileobj(fp):
        # C-contiguous with real file: direct write
        array.tofile(fp)

    else:
        # C-contiguous with stream: chunked write
        for chunk in numpy.nditer(array,
                flags=['external_loop', 'buffered', 'zerosize_ok'],
                buffersize=buffersize, order='C'):
            fp.write(chunk.tobytes('C'))
```

### Write Buffer Size Calculation

```python
if array.itemsize == 0:
    buffersize = 0
else:
    # 16 MiB worth of elements, minimum 1
    buffersize = max(16 * 1024**2 // array.itemsize, 1)
```

**Buffer size examples:**
| dtype | itemsize | buffersize (elements) |
|-------|----------|----------------------|
| int8 | 1 | 16,777,216 |
| int32 | 4 | 4,194,304 |
| float64 | 8 | 2,097,152 |
| complex128 | 16 | 1,048,576 |

### isfileobj Detection

```python
def isfileobj(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        f.fileno()  # May raise for wrapped BytesIO
        return True
    except OSError:
        return False
```

**Why it matters**:
- `True`: Use fast `tofile()`/`fromfile()` methods (kernel-level I/O)
- `False`: Use chunked `tobytes()`/`frombuffer()` for streams (Python-level)

**Accepted types:**
- `io.FileIO` - raw file I/O
- `io.BufferedReader` - buffered read (from `open(file, 'rb')`)
- `io.BufferedWriter` - buffered write (from `open(file, 'wb')`)

**Rejected types:**
- `io.BytesIO` - in-memory stream
- `gzip.GzipFile` - compressed stream
- `zipfile.ZipExtFile` - file within ZIP

---

## Read Implementation

### read_array Function

```python
def read_array(fp, allow_pickle=False, pickle_kwargs=None, *, max_header_size=10000):
    if allow_pickle:
        max_header_size = 2**64  # Trusted file, no limit

    # 1. Read and validate magic/version
    version = read_magic(fp)
    _check_version(version)

    # 2. Parse header
    shape, fortran_order, dtype = _read_array_header(fp, version, max_header_size)

    # 3. Calculate element count
    if len(shape) == 0:
        count = 1  # Scalar
    else:
        count = numpy.multiply.reduce(shape, dtype=numpy.int64)

    # 4. Read data
    if dtype.hasobject:
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
        array = pickle.load(fp, **pickle_kwargs or {})

    elif isfileobj(fp):
        # Real file: fast path
        array = numpy.fromfile(fp, dtype=dtype, count=count)

    else:
        # Stream: chunked read
        array = numpy.ndarray(count, dtype=dtype)
        if dtype.itemsize > 0:
            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * dtype.itemsize)
                data = _read_bytes(fp, read_size, "array data")
                array[i:i+read_count] = numpy.frombuffer(data, dtype=dtype, count=read_count)

    # 5. Validate and reshape
    if array.size != count:
        raise ValueError(f"Failed to read all data. Expected {count}, got {array.size}")

    if fortran_order:
        array = array.reshape(shape[::-1]).transpose()
    else:
        array = array.reshape(shape)

    return array
```

### Read Buffer Size (BUFFER_SIZE)

```python
BUFFER_SIZE = 2**18  # 262,144 bytes (256 KB)

# Max elements per chunk
max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)
```

**Read chunk examples:**
| dtype | itemsize | max_read_count |
|-------|----------|----------------|
| int8 | 1 | 262,144 |
| int32 | 4 | 65,536 |
| float64 | 8 | 32,768 |
| S1000 | 1000 | 262 |

The `min(BUFFER_SIZE, dtype.itemsize)` prevents division by zero when itemsize > BUFFER_SIZE.

### _read_bytes - Robust Stream Reading

```python
def _read_bytes(fp, size, error_template="ran out of data"):
    """Read exactly `size` bytes, handling partial reads."""
    data = b""
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except BlockingIOError:
            pass

    if len(data) != size:
        raise ValueError(f"EOF: reading {error_template}, expected {size} bytes got {len(data)}")
    return data
```

**Why needed**:
- ZipExtFile may return fewer bytes than requested
- Network streams may have partial reads
- Non-blocking I/O may return early
- gzip streams have crc32 limitations at 2**32 bytes

### _read_array_header

```python
def _read_array_header(fp, version, max_header_size=10000):
    # Get format info for version
    hlength_type, encoding = {
        (1, 0): ('<H', 'latin1'),
        (2, 0): ('<I', 'latin1'),
        (3, 0): ('<I', 'utf8'),
    }[version]

    # Read header length
    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    header_length = struct.unpack(hlength_type, hlength_str)[0]

    # Read and decode header
    header = _read_bytes(fp, header_length, "array header")
    header = header.decode(encoding)

    # Security check
    if len(header) > max_header_size:
        raise ValueError(f"Header info length ({len(header)}) is large and may not be safe")

    # Parse Python literal
    try:
        d = ast.literal_eval(header)
    except SyntaxError:
        if version <= (2, 0):
            # Try filtering Python 2 'L' suffixes
            header = _filter_header(header)
            d = ast.literal_eval(header)
        else:
            raise

    # Validate structure
    if not isinstance(d, dict):
        raise ValueError("Header is not a dictionary")
    if {'descr', 'fortran_order', 'shape'} != d.keys():
        raise ValueError("Header does not contain correct keys")

    # Validate values
    if not isinstance(d['shape'], tuple) or not all(isinstance(x, int) for x in d['shape']):
        raise ValueError(f"shape is not valid: {d['shape']}")
    if not isinstance(d['fortran_order'], bool):
        raise ValueError(f"fortran_order is not valid: {d['fortran_order']}")

    dtype = descr_to_dtype(d['descr'])
    return d['shape'], d['fortran_order'], dtype
```

### Python 2 Compatibility Filter

```python
def _filter_header(s):
    """Remove 'L' suffix from integers (Python 2 legacy)."""
    import tokenize
    from io import StringIO

    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(s).readline):
        token_type, token_string = token[0], token[1]
        if last_token_was_number and token_type == tokenize.NAME and token_string == "L":
            continue  # Skip the 'L'
        tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)

    return tokenize.untokenize(tokens)
```

---

## NPZ Archive Format

NPZ is a ZIP archive containing multiple `.npy` files.

### Structure

```
archive.npz (ZIP file)
├── arr_0.npy     # Positional array 0
├── arr_1.npy     # Positional array 1
├── weights.npy   # Named array "weights"
└── biases.npy    # Named array "biases"
```

### _savez Implementation (Internal)

```python
def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    import zipfile

    if not hasattr(file, 'write'):
        file = os.fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'

    # Merge positional args as arr_0, arr_1, ...
    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
        namedict[key] = val

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    # Always enable Zip64 for large files (gh-10776)
    zipf = zipfile_factory(file, mode="w", compression=compression)
    try:
        for key, val in namedict.items():
            fname = key + '.npy'
            val = np.asanyarray(val)
            # force_zip64=True always, even for small files
            with zipf.open(fname, 'w', force_zip64=True) as fid:
                format.write_array(fid, val, allow_pickle=allow_pickle,
                                   pickle_kwargs=pickle_kwargs)
    finally:
        zipf.close()
```

### zipfile_factory

```python
def zipfile_factory(file, *args, **kwargs):
    """Create ZipFile with Zip64 support and path-like handling."""
    if not hasattr(file, 'read'):
        file = os.fspath(file)  # Handle pathlib.Path
    kwargs['allowZip64'] = True  # Always enable Zip64
    return zipfile.ZipFile(file, *args, **kwargs)
```

### savez_compressed

Same as `savez` but uses `compression=zipfile.ZIP_DEFLATED`.

### NpzFile Class (Lazy Loading)

```python
class NpzFile(Mapping):
    """Lazy-loading NPZ archive with dict-like interface."""

    zip = None
    fid = None
    _MAX_REPR_ARRAY_COUNT = 5

    def __init__(self, fid, own_fid=False, allow_pickle=False,
                 pickle_kwargs=None, *, max_header_size=10000):
        _zip = zipfile_factory(fid)
        _files = _zip.namelist()

        # Strip .npy extension for user-facing keys
        self.files = [name.removesuffix(".npy") for name in _files]

        # Map both stripped and unstripped names
        self._files = dict(zip(self.files, _files))
        self._files.update(zip(_files, _files))

        self.allow_pickle = allow_pickle
        self.max_header_size = max_header_size
        self.pickle_kwargs = pickle_kwargs
        self.zip = _zip
        self.f = BagObj(self)  # Attribute access: npz.f.array_name

        if own_fid:
            self.fid = fid

    def __getitem__(self, key):
        try:
            key = self._files[key]
        except KeyError:
            raise KeyError(f"{key} is not a file in the archive") from None

        with self.zip.open(key) as bytes:
            # Check magic to distinguish .npy from other files
            magic = bytes.read(len(format.MAGIC_PREFIX))
            bytes.seek(0)
            if magic == format.MAGIC_PREFIX:
                return format.read_array(bytes,
                    allow_pickle=self.allow_pickle,
                    pickle_kwargs=self.pickle_kwargs,
                    max_header_size=self.max_header_size)
            else:
                # Non-.npy files returned as raw bytes
                return bytes.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None  # Break reference cycle
```

### BagObj - Attribute Access Helper

```python
class BagObj:
    """Convert attribute lookups to getitem on wrapped object."""

    def __init__(self, obj):
        # Use weakref to make NpzFile collectable by refcount
        self._obj = weakref.proxy(obj)

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        # Enable tab-completion in IPython
        return list(object.__getattribute__(self, '_obj').keys())
```

### NPZ Key Naming Rules

- Positional arrays: `arr_0`, `arr_1`, `arr_2`, ...
- Keyword arrays: use the keyword name directly
- Collision check: cannot use both positional and keyword with same name
- Keys should be valid filenames: avoid `/` or `.` characters
- Cannot use `file` as keyword (conflicts with function parameter)

### NPZ Key Access

Both stripped and unstripped names work:

```python
with np.load('data.npz') as npz:
    npz['arr_0']      # Works
    npz['arr_0.npy']  # Also works (maps to same file)
    npz.files         # Returns ['arr_0'] (stripped)
```

### Non-.npy Files in NPZ

NpzFile can contain non-numpy files. They're returned as raw bytes:

```python
with np.load('mixed.npz') as npz:
    arr = npz['data']       # Returns ndarray (was data.npy)
    txt = npz['readme.txt'] # Returns bytes
    raw = npz['config.bin'] # Returns bytes
```

Magic check in `__getitem__`:
```python
magic = bytes.read(len(format.MAGIC_PREFIX))
bytes.seek(0)
if magic == format.MAGIC_PREFIX:
    return format.read_array(bytes, ...)
else:
    return bytes.read()  # Raw bytes for non-.npy
```

---

## Memory Mapping

### open_memmap Function

```python
def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None, *, max_header_size=10000):

    if isfileobj(filename):
        raise ValueError("Memmap requires file path, not file object")

    if 'w' in mode:
        # Create new file
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            raise ValueError("Cannot memmap object arrays")

        d = {'descr': dtype_to_descr(dtype), 'fortran_order': fortran_order, 'shape': shape}

        with open(filename, mode + 'b') as fp:
            _write_array_header(fp, d, version)
            offset = fp.tell()
    else:
        # Read existing file
        with open(filename, 'rb') as fp:
            version = read_magic(fp)
            shape, fortran_order, dtype = _read_array_header(fp, version, max_header_size)
            if dtype.hasobject:
                raise ValueError("Cannot memmap object arrays")
            offset = fp.tell()

    order = 'F' if fortran_order else 'C'
    if mode == 'w+':
        mode = 'r+'  # Already wrote header

    return numpy.memmap(filename, dtype=dtype, shape=shape, order=order, mode=mode, offset=offset)
```

---

## Edge Cases

### Scalar Arrays (0-dimensional)

```python
arr = np.array(42)
# shape: ()
# count: 1 (special case)
# Data: single element
```

### Empty Arrays

```python
arr = np.array([], dtype='<f8')
# shape: (0,)
# count: 0
# Data: zero bytes
```

### 1-D Array Shape

Shape tuple has trailing comma in Python repr:

```python
arr = np.array([1, 2, 3])
# Header: {'shape': (3,), ...}  # Note the comma
```

### Non-Contiguous Arrays

Strided/sliced arrays are copied to C-order during write:

```python
arr = np.arange(12)[::2]  # Non-contiguous slice
# Written as contiguous C-order copy
# fortran_order: False
```

### Zero-Size dtype

```python
dt = np.dtype({'names': [], 'formats': [], 'itemsize': 8})
# itemsize > 0 but no actual data
# buffersize = 0 for writes
# Skip data read loop
```

### Truncated Files

```python
# If file ends before expected data:
raise ValueError(
    f"Failed to read all data for array. "
    f"Expected {shape} = {count} elements, "
    f"could only read {array.size} elements. "
    f"(file seems not fully written?)"
)
```

---

## Security Considerations

### max_header_size

Default: 10,000 bytes

`ast.literal_eval()` can be slow or crash on very large inputs. This limit prevents DoS attacks with malicious headers.

```python
# Bypass for trusted files
np.load(file, allow_pickle=True)  # Sets max_header_size = 2**64
np.load(file, max_header_size=200000)  # Explicit override
```

### allow_pickle

Default: `False` (since NumPy 1.16.3)

Object arrays use Python pickle, which can execute arbitrary code. Never load untrusted `.npy` files with `allow_pickle=True`.

```python
# This is DANGEROUS with untrusted files:
np.load(untrusted_file, allow_pickle=True)
```

### Validation

Headers are strictly validated:
- Exactly 3 keys required
- `shape` must be tuple of ints
- `fortran_order` must be bool
- `descr` must produce valid dtype

### Error Messages

| Error | Message |
|-------|---------|
| Invalid version | `we only support format version (1,0), (2,0), and (3,0), not {version}` |
| Bad magic | `the magic string is not correct; expected b'\x93NUMPY', got {actual}` |
| Object w/o pickle (write) | `Object arrays cannot be saved when allow_pickle=False` |
| Object w/o pickle (read) | `Object arrays cannot be loaded when allow_pickle=False` |
| Header too large | `Header info length ({len}) is large and may not be safe to load securely` |
| Truncated header | `EOF: reading array header length, expected {n} bytes got {m}` |
| Non-dict header | `Header is not a dictionary: {value}` |
| Wrong keys | `Header does not contain the correct keys: {keys}` |
| Invalid shape | `shape is not valid: {value}` |
| Invalid fortran_order | `fortran_order is not a valid bool: {value}` |
| Invalid dtype | `descr is not a valid dtype descriptor: {value}` |
| Truncated data | `Failed to read all data for array. Expected {shape} = {count} elements, could only read {actual}` |
| No data (np.load) | `EOFError: No data left in file` |
| Pickle file w/o allow | `This file contains pickled data. Use allow_pickle=True if you trust the file.` |
| User dtype w/o pickle | `User-defined dtypes cannot be saved when allow_pickle=False` |
| Memmap w/ file obj | `Filename must be a string or a path-like object. Memmap cannot use existing file handles.` |
| Memmap w/ object dtype | `Array can't be memory-mapped: Python objects in dtype.` |

---

## Constants Reference

### File Type Detection (np.load)

```python
_ZIP_PREFIX = b'PK\x03\x04'  # Standard ZIP files
_ZIP_SUFFIX = b'PK\x05\x06'  # Empty ZIP files

# Detection order:
# 1. magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX) -> NPZ
# 2. magic == MAGIC_PREFIX -> NPY
# 3. Otherwise -> Pickle file
```

### Format Constants

```python
# Magic string
MAGIC_PREFIX = b'\x93NUMPY'
MAGIC_LEN = 8  # len(MAGIC_PREFIX) + 2 version bytes

# Alignment
ARRAY_ALIGN = 64  # Header aligned to this boundary (for mmap)

# Buffer sizes
BUFFER_SIZE = 2**18  # 262,144 bytes (256 KB) for chunked I/O

# Header growth
GROWTH_AXIS_MAX_DIGITS = 21  # len(str(8 * 2**64 - 1))

# Security
_MAX_HEADER_SIZE = 10000  # Default limit for ast.literal_eval

# Version info
_header_size_info = {
    (1, 0): ('<H', 'latin1'),  # 2-byte length, latin1 encoding
    (2, 0): ('<I', 'latin1'),  # 4-byte length, latin1 encoding
    (3, 0): ('<I', 'utf8'),    # 4-byte length, UTF-8 encoding
}

# Required header keys
EXPECTED_KEYS = {'descr', 'fortran_order', 'shape'}
```

---

## High-Level API (np.save / np.load)

### np.save

```python
def save(file, arr, allow_pickle=True):
    """
    Save array to .npy file.

    Parameters
    ----------
    file : file-like, str, or pathlib.Path
        File path or open file object.
        '.npy' extension added to paths if not present.
    arr : array_like
        Array to save. Converted via np.asanyarray().
    allow_pickle : bool
        Allow pickling object arrays. Default True.

    Notes
    -----
    - Data is APPENDED if file-like object passed (can write multiple arrays)
    - Extension only added for string/path, not file objects
    - Uses np.asanyarray() to preserve subclass data (but subclass type lost on load)
    """
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle)
```

### np.load

```python
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII', *, max_header_size=10000):
    """
    Load array from .npy, .npz, or pickle file.

    Parameters
    ----------
    file : file-like, str, or pathlib.Path
        File path or object. Must support seek() and read().
        Pickled files also need readline().
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}
        Memory-map mode. None for regular load.
    allow_pickle : bool
        Allow loading pickled objects. Default False (security).
    fix_imports : bool
        Fix Python 2 vs 3 import differences in pickle. Default True.
    encoding : {'ASCII', 'latin1', 'bytes'}
        Encoding for Python 2 strings in pickle.
        ONLY these three values allowed (others corrupt numerical data).
    max_header_size : int
        Maximum header size to parse. Default 10000.
        Ignored when allow_pickle=True (trusted file).

    Returns
    -------
    ndarray, NpzFile, or unpickled object

    Raises
    ------
    EOFError
        When reading from file handle with no data left.
    ValueError
        Object array with allow_pickle=False.
    """
    # Encoding validation - critical for data integrity
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")

    pickle_kwargs = {'encoding': encoding, 'fix_imports': fix_imports}

    with contextlib.ExitStack() as stack:
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            fid = stack.enter_context(open(os.fspath(file), "rb"))
            own_fid = True

        # Detect file type by magic bytes
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'  # Empty zip files

        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)

        if not magic:
            raise EOFError("No data left in file")

        # Seek back (handle files smaller than N bytes)
        fid.seek(-min(N, len(magic)), 1)

        if magic.startswith((_ZIP_PREFIX, _ZIP_SUFFIX)):
            # NPZ archive
            stack.pop_all()  # Transfer ownership to NpzFile
            return NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                          pickle_kwargs=pickle_kwargs,
                          max_header_size=max_header_size)

        elif magic == format.MAGIC_PREFIX:
            # NPY file
            if mmap_mode:
                if allow_pickle:
                    max_header_size = 2**64
                return format.open_memmap(file, mode=mmap_mode,
                                          max_header_size=max_header_size)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs,
                                         max_header_size=max_header_size)
        else:
            # Pure pickle file (legacy or non-numpy)
            if not allow_pickle:
                raise ValueError(
                    "This file contains pickled data. Use allow_pickle=True "
                    "if you trust the file.")
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise pickle.UnpicklingError(
                    f"Failed to interpret file {file!r} as a pickle") from e
```

### Multiple Arrays in Single File

```python
# Writing multiple arrays to one file
with open('multi.npy', 'wb') as f:
    np.save(f, arr1)
    np.save(f, arr2)
    np.save(f, arr3)

# Reading them back (in order)
with open('multi.npy', 'rb') as f:
    arr1 = np.load(f)
    arr2 = np.load(f)
    arr3 = np.load(f)
    # np.load(f) would raise EOFError here
```

---

## Implicit Behaviors and Gotchas

### Pickle Protocol

Object arrays always use **pickle protocol 4** (not configurable):

```python
pickle.dump(array, fp, protocol=4, **pickle_kwargs)
```

### Count Calculation Uses int64

To prevent overflow with large arrays:

```python
count = numpy.multiply.reduce(shape, dtype=numpy.int64)
```

### Fortran Write Uses Transpose

F-contiguous arrays are written via `array.T.tofile(fp)`:

```python
if array.flags.f_contiguous and not array.flags.c_contiguous:
    if isfileobj(fp):
        array.T.tofile(fp)  # Note: transpose!
```

### Fortran Read Uses Reversed Shape

```python
if fortran_order:
    array = array.reshape(shape[::-1])  # Reversed!
    array = array.transpose()
```

### nditer Flags

Write operations use specific nditer flags:

```python
numpy.nditer(array,
    flags=['external_loop', 'buffered', 'zerosize_ok'],
    buffersize=buffersize,
    order='C' or 'F')
```

- `external_loop`: Return 1-D arrays for efficiency
- `buffered`: Allow buffering for non-contiguous
- `zerosize_ok`: Handle empty arrays without error

### Array Allocation for Streams

Uses `numpy.ndarray()` not `numpy.empty()` for zero-width string dtype compatibility (gh-6430):

```python
array = numpy.ndarray(count, dtype=dtype)  # NOT numpy.empty()
```

### Chunked Read Size Calculation

Prevents division by zero when itemsize > BUFFER_SIZE:

```python
max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)
```

### crc32 Limitation

Comment in source: "crc32 module fails on reads greater than 2**32 bytes, breaking large reads from gzip streams."

This is why chunked reading is used even for non-file streams.

### Python 2 Header Filtering

Only applied for versions (1, 0) and (2, 0), not (3, 0):

```python
if version <= (2, 0):
    header = _filter_header(header)  # Remove 'L' suffixes
```

### Header Key Sorting

Keys are sorted alphabetically in output ("Writer SHOULD implement"), but readers "MUST NOT depend on this":

```python
for key, value in sorted(d.items()):
    header.append(f"'{key}': {repr(value)}, ")
```

### Path-like Object Handling

`os.fspath()` used throughout for pathlib.Path support:

```python
file = os.fspath(file)
```

### UnicodeError in Pickle Load

Special handling with helpful message:

```python
except UnicodeError as err:
    raise UnicodeError("Unpickling a python object failed: %r\n"
                       "You may need to pass the encoding= option "
                       "to numpy.load" % (err,)) from err
```

### open_memmap Mode Conversion

After writing header, `w+` mode becomes `r+`:

```python
if mode == 'w+':
    mode = 'r+'  # Already wrote header
```

### Structured dtype Padding Detection

Specific condition for identifying padding bytes:

```python
is_pad = (name == '' and dt.type is numpy.void and dt.names is None)
```

### Titled Fields in Structured dtypes

Field names can be tuples of `(title, name)`:

```python
title, name = name if isinstance(name, tuple) else (None, name)
```

### Non-legacy dtype Check

User-defined dtypes detected via internal `_legacy` attribute:

```python
if not type(dtype)._legacy:
    # Use pickle, emit warning
    return "|O"
```

### drop_metadata Behavior

```python
dt_meta = np.dtype('i4', metadata={'key': 'value'})
dropped = format.drop_metadata(dt_meta)
# dropped is a NEW dtype without metadata
# Original dtype unchanged
# Returns same object if no metadata present
```

### File Position After Header Read

After `read_array_header_*()`, file position is exactly at data start:

```python
shape, fort, dt = format.read_array_header_1_0(fp)
data_offset = fp.tell()  # This is the mmap offset!
# data_offset is always 64-byte aligned
```

### Header Dictionary repr() Usage

All values use Python `repr()` for serialization:

```python
header.append(f"'{key}': {repr(value)}, ")
```

This ensures:
- Tuples have trailing comma: `(3,)` not `(3)`
- Strings are properly quoted
- Numbers are exact

### Trailing Comma in Header

Header dict always has trailing comma before closing brace:

```python
"{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }"
#                                                        ^ trailing comma
```

### NaN/Inf/Special Values

IEEE 754 special values round-trip correctly:

| Value | Preserved |
|-------|-----------|
| `nan` | Yes |
| `inf` | Yes |
| `-inf` | Yes |
| `0.0` | Yes |
| `-0.0` | Yes (sign preserved!) |
| `float64.max` | Yes |
| `float64.tiny` | Yes |

### Empty Array File Size

Empty arrays still have full aligned header:

```python
arr = np.array([], dtype='<f8')
# Header: 128 bytes (64-aligned)
# Data: 0 bytes
# Total: 128 bytes
```

### Header Uses repr() for Values

All header values serialized via Python's `repr()`:

```python
header.append(f"'{key}': {repr(value)}, ")
```

This ensures `ast.literal_eval()` can parse them. Only Python literal types supported.

### Trailing Comma in Header Dict

Header dict always has trailing comma: `{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }`

### NpzFile Allows Both Key Formats

Both stripped and unstripped names work:
```python
npz['arr_0']      # Works
npz['arr_0.npy']  # Also works
```

### File Position After Header Read

After `read_array_header_*()`, file position is exactly at data start (64-byte aligned). This enables memory-mapping with correct offset.

### drop_metadata() Function

Public utility that strips dtype metadata:
```python
dt_stripped = format.drop_metadata(dt_with_metadata)
# Returns same object if no metadata
# Returns new dtype without metadata otherwise
```

### Special Float Values Preserved

NaN, +inf, -inf are preserved in binary format (IEEE 754 representation).

---

## Limitations and Warnings

### Subclass Preservation

From docstring: "Arbitrary subclasses of numpy.ndarray are not completely preserved. Subclasses will be accepted for writing, but only the array data will be written out. A regular numpy.ndarray object will be created upon reading the file."

### Empty Field Names

From docstring: "Due to limitations in the interpretation of structured dtypes, dtypes with fields with empty names will have the names replaced by 'f0', 'f1', etc. Such arrays will not round-trip through the format entirely accurately. The data is intact; only the field names will differ."

Workaround: `loadedarray.view(correct_dtype)`

### Metadata Not Saved

dtype metadata is stripped with warning:

```python
if new_dtype is not dtype:
    warnings.warn("metadata on a dtype is not saved to an npy/npz. "
                  "Use another format (such as pickle) to store it.",
                  UserWarning, stacklevel=2)
```

### Endianness Preservation

Arrays maintain their byte order. A little-endian file yields a little-endian array on ANY machine:

> "a file with little-endian numbers will yield a little-endian array on any machine reading the file"

### Object Array Memory Mapping

Cannot memory-map object arrays:

```python
if dtype.hasobject:
    raise ValueError("Array can't be memory-mapped: Python objects in dtype.")
```

### Size Validation After Read

The size check happens OUTSIDE the itemsize > 0 block, so it always validates:

```python
# After chunked read loop
if array.size != count:
    raise ValueError("Failed to read all data for array...")
```

### Warning Messages

Version 2.0 warning has a typo in original source: "It can only beread" (missing space).

---

## Public API Reference

### numpy.lib.format Module

| Function | Description |
|----------|-------------|
| `magic(major, minor)` | Create magic bytes for version |
| `read_magic(fp)` | Read version from file, returns (major, minor) |
| `write_array(fp, array, version, allow_pickle, pickle_kwargs)` | Write array to file |
| `read_array(fp, allow_pickle, pickle_kwargs, max_header_size)` | Read array from file |
| `write_array_header_1_0(fp, d)` | Write v1.0 header |
| `write_array_header_2_0(fp, d)` | Write v2.0 header |
| `read_array_header_1_0(fp, max_header_size)` | Read v1.0 header |
| `read_array_header_2_0(fp, max_header_size)` | Read v2.0 header |
| `header_data_from_array_1_0(array)` | Extract header dict from array |
| `dtype_to_descr(dtype)` | Convert dtype to serializable descr |
| `descr_to_dtype(descr)` | Convert descr back to dtype |
| `open_memmap(filename, mode, dtype, shape, fortran_order, version, max_header_size)` | Memory-map a .npy file |
| `isfileobj(f)` | Check if f is a real file object |
| `drop_metadata(dtype)` | Strip metadata from dtype |

### numpy Module (High-Level)

| Function | Description |
|----------|-------------|
| `np.save(file, arr, allow_pickle)` | Save single array |
| `np.load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)` | Load .npy/.npz/pickle |
| `np.savez(file, *args, allow_pickle, **kwds)` | Save multiple arrays (uncompressed) |
| `np.savez_compressed(file, *args, allow_pickle, **kwds)` | Save multiple arrays (compressed) |

---

---

## Error Messages

### Format Errors

| Error | Message Pattern |
|-------|-----------------|
| Bad magic | `the magic string is not correct; expected b'\x93NUMPY', got {got!r}` |
| Bad version | `we only support format version (1,0), (2,0), and (3,0), not {version}` |
| Invalid version | `Invalid version {version!r}` |
| Header too large | `Header length {hlen} too big for version={version}` |
| Header security | `Header info length ({len}) is large and may not be safe to load securely` |
| Truncated header | `EOF: reading array header, expected {size} bytes got {len}` |
| Truncated data | `EOF: reading array data, expected {size} bytes got {len}` |
| Invalid header | `Cannot parse header: {header!r}` |
| Not dict | `Header is not a dictionary: {d!r}` |
| Wrong keys | `Header does not contain the correct keys: {keys!r}` |
| Invalid shape | `shape is not valid: {shape!r}` |
| Invalid fortran | `fortran_order is not a valid bool: {value!r}` |
| Invalid descr | `descr is not a valid dtype descriptor: {descr!r}` |
| Data mismatch | `Failed to read all data for array. Expected {shape} = {count} elements, could only read {size} elements.` |
| Object no pickle | `Object arrays cannot be saved/loaded when allow_pickle=False` |
| Custom dtype | `User-defined dtypes cannot be saved when allow_pickle=False` |

### np.load Errors

| Error | Condition |
|-------|-----------|
| `EOFError("No data left in file")` | Empty file or past end |
| `ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")` | Bad encoding value |
| `pickle.UnpicklingError(f"Failed to interpret file {file!r}")` | Not npy/npz/pickle |
| `KeyError(f"{key} is not a file in the archive")` | NPZ key not found |

### open_memmap Errors

| Error | Condition |
|-------|-----------|
| `ValueError("Filename must be a string or path-like object...")` | File object passed |
| `ValueError("Array can't be memory-mapped: Python objects in dtype.")` | Object dtype |

---

## Public API Reference

### numpy.lib.format Functions

| Function | Description |
|----------|-------------|
| `magic(major, minor)` | Create magic bytes for version |
| `read_magic(fp)` | Read magic, return (major, minor) |
| `dtype_to_descr(dtype)` | Convert dtype to serializable descriptor |
| `descr_to_dtype(descr)` | Convert descriptor back to dtype |
| `header_data_from_array_1_0(array)` | Get header dict from array |
| `write_array_header_1_0(fp, d)` | Write v1.0 header |
| `write_array_header_2_0(fp, d)` | Write v2.0 header |
| `read_array_header_1_0(fp, max_header_size)` | Read v1.0 header |
| `read_array_header_2_0(fp, max_header_size)` | Read v2.0 header |
| `write_array(fp, array, version, allow_pickle)` | Write array with header |
| `read_array(fp, allow_pickle, max_header_size)` | Read array from file |
| `open_memmap(filename, mode, dtype, shape, ...)` | Open as memory-mapped |
| `isfileobj(f)` | Check if real file (vs stream) |
| `drop_metadata(dtype)` | Remove metadata from dtype |

### numpy.lib.format Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAGIC_PREFIX` | `b'\x93NUMPY'` | File magic (6 bytes) |
| `MAGIC_LEN` | `8` | Magic + version bytes |
| `ARRAY_ALIGN` | `64` | Header alignment |
| `BUFFER_SIZE` | `262144` | Chunk size (256KB) |
| `GROWTH_AXIS_MAX_DIGITS` | `21` | Space for axis growth |
| `EXPECTED_KEYS` | `{'descr', 'fortran_order', 'shape'}` | Required header keys |

### numpy Functions

| Function | Description |
|----------|-------------|
| `np.save(file, arr, allow_pickle=True)` | Save to .npy |
| `np.load(file, mmap_mode, allow_pickle, ...)` | Load .npy/.npz/pickle |
| `np.savez(file, *args, **kwds)` | Save to uncompressed .npz |
| `np.savez_compressed(file, *args, **kwds)` | Save to compressed .npz |

### File Type Detection Order

np.load detects file type by magic bytes:

| Magic Bytes | File Type | Action |
|-------------|-----------|--------|
| `b'PK\x03\x04'` | ZIP (NPZ) | Return NpzFile |
| `b'PK\x05\x06'` | Empty ZIP | Return NpzFile |
| `b'\x93NUMPY'` | NPY | read_array or open_memmap |
| Other | Pickle | pickle.load |

---

## References

- Source: `numpy/lib/_format_impl.py` (NumPy 2.x)
- Source: `numpy/lib/_npyio_impl.py` (np.save/load wrappers)
- NEP 1: [A Simple File Format for NumPy Arrays](https://numpy.org/neps/nep-0001-npy-format.html)
- NumPy Documentation: [numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html)
