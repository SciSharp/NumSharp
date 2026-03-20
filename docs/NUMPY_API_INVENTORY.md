# NumPy 2.4.2 Complete API Inventory

This document provides an exhaustive inventory of all public APIs exposed by NumPy 2.4.2 as `np.*`.

**Source:** `numpy/__init__.py`, `numpy/__init__.pyi`, and submodule `.pyi` stub files from NumPy v2.4.2

**Last Updated:** Cross-verified against actual source files

---

## Table of Contents

1. [Constants](#constants)
2. [Data Types (Scalars)](#data-types-scalars)
3. [DType Classes](#dtype-classes)
4. [Array Creation](#array-creation)
5. [Array Manipulation](#array-manipulation)
6. [Mathematical Functions](#mathematical-functions)
7. [Universal Functions (ufuncs)](#universal-functions-ufuncs)
8. [Trigonometric Functions](#trigonometric-functions)
9. [Hyperbolic Functions](#hyperbolic-functions)
10. [Exponential and Logarithmic](#exponential-and-logarithmic)
11. [Arithmetic Operations](#arithmetic-operations)
12. [Comparison Functions](#comparison-functions)
13. [Logical Functions](#logical-functions)
14. [Bitwise Operations](#bitwise-operations)
15. [Statistical Functions](#statistical-functions)
16. [Sorting and Searching](#sorting-and-searching)
17. [Set Operations](#set-operations)
18. [Window Functions](#window-functions)
19. [Linear Algebra (np.linalg)](#linear-algebra-nplinalg)
20. [FFT (np.fft)](#fft-npfft)
21. [Random Sampling (np.random)](#random-sampling-nprandom)
22. [Polynomial (np.polynomial)](#polynomial-nppolynomial)
23. [Masked Arrays (np.ma)](#masked-arrays-npma)
24. [String Operations (np.char)](#string-operations-npchar)
25. [String Operations (np.strings)](#string-operations-npstrings)
26. [Record Arrays (np.rec)](#record-arrays-nprec)
27. [Ctypes Interop (np.ctypeslib)](#ctypes-interop-npctypeslib)
28. [File I/O](#file-io)
29. [Memory and Buffer](#memory-and-buffer)
30. [Indexing Routines](#indexing-routines)
31. [Broadcasting](#broadcasting)
32. [Stride Tricks](#stride-tricks)
33. [Array Printing](#array-printing)
34. [Error Handling](#error-handling)
35. [Type Information](#type-information)
36. [Typing (np.typing)](#typing-nptyping)
37. [Testing (np.testing)](#testing-nptesting)
38. [Exceptions (np.exceptions)](#exceptions-npexceptions)
39. [Array API Aliases](#array-api-aliases)
40. [Submodules](#submodules)
41. [Classes](#classes)
42. [Deprecated APIs](#deprecated-apis)
43. [Removed APIs (NumPy 2.0)](#removed-apis-numpy-20)

---

## Constants

| Name | Type | Description |
|------|------|-------------|
| `np.e` | `float` | Euler's number (2.718281828...) |
| `np.pi` | `float` | Pi (3.141592653...) |
| `np.euler_gamma` | `float` | Euler-Mascheroni constant (0.5772156649...) |
| `np.inf` | `float` | Positive infinity |
| `np.nan` | `float` | Not a Number |
| `np.newaxis` | `None` | Alias for None, used to expand dimensions |
| `np.little_endian` | `bool` | True if system is little-endian |
| `np.True_` | `np.bool` | NumPy True constant |
| `np.False_` | `np.bool` | NumPy False constant |
| `np.__version__` | `str` | NumPy version string |
| `np.__array_api_version__` | `str` | Array API version ("2024.12") |

---

## Data Types (Scalars)

### Boolean
| Name | Aliases | Description |
|------|---------|-------------|
| `np.bool` | `np.bool_` | Boolean (True or False) |

### Signed Integers
| Name | Aliases | Bits | Description |
|------|---------|------|-------------|
| `np.int8` | `np.byte` | 8 | Signed 8-bit integer |
| `np.int16` | `np.short` | 16 | Signed 16-bit integer |
| `np.int32` | `np.intc` | 32 | Signed 32-bit integer |
| `np.int64` | `np.long` | 64 | Signed 64-bit integer |
| `np.intp` | `np.int_` | platform | Signed pointer-sized integer |
| `np.longlong` | - | platform | Signed long long |

### Unsigned Integers
| Name | Aliases | Bits | Description |
|------|---------|------|-------------|
| `np.uint8` | `np.ubyte` | 8 | Unsigned 8-bit integer |
| `np.uint16` | `np.ushort` | 16 | Unsigned 16-bit integer |
| `np.uint32` | `np.uintc` | 32 | Unsigned 32-bit integer |
| `np.uint64` | `np.ulong` | 64 | Unsigned 64-bit integer |
| `np.uintp` | `np.uint` | platform | Unsigned pointer-sized integer |
| `np.ulonglong` | - | platform | Unsigned long long |

### Floating Point
| Name | Aliases | Bits | Description |
|------|---------|------|-------------|
| `np.float16` | `np.half` | 16 | Half precision float |
| `np.float32` | `np.single` | 32 | Single precision float |
| `np.float64` | `np.double` | 64 | Double precision float |
| `np.longdouble` | - | platform | Extended precision float |
| `np.float96` | - | 96 | Platform-specific (x86 only) |
| `np.float128` | - | 128 | Platform-specific |

### Complex
| Name | Aliases | Bits | Description |
|------|---------|------|-------------|
| `np.complex64` | `np.csingle` | 64 | Single precision complex |
| `np.complex128` | `np.cdouble` | 128 | Double precision complex |
| `np.clongdouble` | - | platform | Extended precision complex |
| `np.complex192` | - | 192 | Platform-specific |
| `np.complex256` | - | 256 | Platform-specific |

### Other
| Name | Description |
|------|-------------|
| `np.object_` | Python object |
| `np.bytes_` | Byte string |
| `np.str_` | Unicode string |
| `np.void` | Void (flexible) |
| `np.datetime64` | Date and time |
| `np.timedelta64` | Time delta |

### Abstract Types
| Name | Description |
|------|-------------|
| `np.generic` | Base class for all scalar types |
| `np.number` | Base class for numeric types |
| `np.integer` | Base class for integer types |
| `np.signedinteger` | Base class for signed integers |
| `np.unsignedinteger` | Base class for unsigned integers |
| `np.inexact` | Base class for inexact types |
| `np.floating` | Base class for floating types |
| `np.complexfloating` | Base class for complex types |
| `np.flexible` | Base class for flexible types |
| `np.character` | Base class for character types |

---

## DType Classes

Located in `np.dtypes`:

| Class | Description |
|-------|-------------|
| `BoolDType` | Boolean dtype |
| `Int8DType` / `ByteDType` | 8-bit signed integer dtype |
| `UInt8DType` / `UByteDType` | 8-bit unsigned integer dtype |
| `Int16DType` / `ShortDType` | 16-bit signed integer dtype |
| `UInt16DType` / `UShortDType` | 16-bit unsigned integer dtype |
| `Int32DType` / `IntDType` | 32-bit signed integer dtype |
| `UInt32DType` / `UIntDType` | 32-bit unsigned integer dtype |
| `Int64DType` / `LongDType` | 64-bit signed integer dtype |
| `UInt64DType` / `ULongDType` | 64-bit unsigned integer dtype |
| `LongLongDType` | Long long signed integer dtype |
| `ULongLongDType` | Long long unsigned integer dtype |
| `Float16DType` | 16-bit float dtype |
| `Float32DType` | 32-bit float dtype |
| `Float64DType` | 64-bit float dtype |
| `LongDoubleDType` | Long double float dtype |
| `Complex64DType` | 64-bit complex dtype |
| `Complex128DType` | 128-bit complex dtype |
| `CLongDoubleDType` | Long double complex dtype |
| `ObjectDType` | Python object dtype |
| `BytesDType` | Byte string dtype |
| `StrDType` | Unicode string dtype |
| `VoidDType` | Void dtype |
| `DateTime64DType` | Datetime64 dtype |
| `TimeDelta64DType` | Timedelta64 dtype |
| `StringDType` | Variable-length string dtype (new in 2.0) |

---

## Array Creation

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.array` | `array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)` | Create an array |
| `np.asarray` | `asarray(a, dtype=None, order=None, *, copy=None, device=None, like=None)` | Convert to array |
| `np.asanyarray` | `asanyarray(a, dtype=None, order=None, *, like=None)` | Convert to array, pass through subclasses |
| `np.ascontiguousarray` | `ascontiguousarray(a, dtype=None, *, like=None)` | Return contiguous array in memory (C order) |
| `np.asfortranarray` | `asfortranarray(a, dtype=None, *, like=None)` | Return array in Fortran order |
| `np.asarray_chkfinite` | `asarray_chkfinite(a, dtype=None, order=None)` | Convert to array, checking for NaN/inf |
| `np.zeros` | `zeros(shape, dtype=float, order='C', *, device=None, like=None)` | Return new array of zeros |
| `np.zeros_like` | `zeros_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)` | Return array of zeros with same shape/type |
| `np.ones` | `ones(shape, dtype=float, order='C', *, device=None, like=None)` | Return new array of ones |
| `np.ones_like` | `ones_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)` | Return array of ones with same shape/type |
| `np.empty` | `empty(shape, dtype=float, order='C', *, device=None, like=None)` | Return new uninitialized array |
| `np.empty_like` | `empty_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)` | Return uninitialized array with same shape/type |
| `np.full` | `full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)` | Return new array filled with fill_value |
| `np.full_like` | `full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None, *, device=None)` | Return full array with same shape/type |
| `np.arange` | `arange([start,] stop[, step,], dtype=None, *, device=None, like=None)` | Return evenly spaced values within interval |
| `np.linspace` | `linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)` | Return evenly spaced numbers over interval |
| `np.logspace` | `logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)` | Return numbers spaced evenly on log scale |
| `np.geomspace` | `geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)` | Return numbers spaced evenly on geometric scale |
| `np.eye` | `eye(N, M=None, k=0, dtype=float, order='C', *, device=None, like=None)` | Return 2-D array with ones on diagonal |
| `np.identity` | `identity(n, dtype=float, *, like=None)` | Return identity matrix |
| `np.diag` | `diag(v, k=0)` | Extract diagonal or construct diagonal array |
| `np.diagflat` | `diagflat(v, k=0)` | Create 2-D array with flattened input as diagonal |
| `np.tri` | `tri(N, M=None, k=0, dtype=float, *, like=None)` | Array with ones at and below diagonal |
| `np.tril` | `tril(m, k=0)` | Lower triangle of array |
| `np.triu` | `triu(m, k=0)` | Upper triangle of array |
| `np.vander` | `vander(x, N=None, increasing=False)` | Generate Vandermonde matrix |
| `np.fromfunction` | `fromfunction(function, shape, *, dtype=float, like=None, **kwargs)` | Construct array by executing function |
| `np.fromiter` | `fromiter(iter, dtype, count=-1, *, like=None)` | Create array from iterable |
| `np.fromstring` | `fromstring(string, dtype=float, count=-1, *, sep, like=None)` | Create array from string data |
| `np.frombuffer` | `frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None)` | Interpret buffer as 1-D array |
| `np.from_dlpack` | `from_dlpack(x, /, *, device=None, copy=None)` | Create array from DLPack capsule |
| `np.copy` | `copy(a, order='K', subok=False)` | Return array copy |
| `np.meshgrid` | `meshgrid(*xi, copy=True, sparse=False, indexing='xy')` | Return coordinate matrices |
| `np.mgrid` | `mgrid[...]` | Dense multi-dimensional meshgrid (indexing object) |
| `np.ogrid` | `ogrid[...]` | Open multi-dimensional meshgrid (indexing object) |
| `np.indices` | `indices(dimensions, dtype=int, sparse=False)` | Return array representing grid indices |

---

## Array Manipulation

### Shape Operations
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.reshape` | `reshape(a, /, shape=None, *, newshape=None, order='C', copy=None)` | Give new shape to array |
| `np.ravel` | `ravel(a, order='C')` | Return flattened array |
| `np.ndim` | `ndim(a)` | Return number of dimensions |
| `np.shape` | `shape(a)` | Return shape of array |
| `np.size` | `size(a, axis=None)` | Return number of elements |
| `np.transpose` | `transpose(a, axes=None)` | Permute array dimensions |
| `np.matrix_transpose` | `matrix_transpose(x, /)` | Transpose last two dimensions |
| `np.moveaxis` | `moveaxis(a, source, destination)` | Move axes to new positions |
| `np.rollaxis` | `rollaxis(a, axis, start=0)` | Roll axis backwards |
| `np.swapaxes` | `swapaxes(a, axis1, axis2)` | Interchange two axes |
| `np.squeeze` | `squeeze(a, axis=None)` | Remove axes of length one |
| `np.expand_dims` | `expand_dims(a, axis)` | Expand array shape |
| `np.atleast_1d` | `atleast_1d(*arys)` | View inputs as arrays with at least one dimension |
| `np.atleast_2d` | `atleast_2d(*arys)` | View inputs as arrays with at least two dimensions |
| `np.atleast_3d` | `atleast_3d(*arys)` | View inputs as arrays with at least three dimensions |

### Joining Arrays
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.concatenate` | `concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')` | Join arrays along axis |
| `np.stack` | `stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')` | Join arrays along new axis |
| `np.vstack` | `vstack(tup, *, dtype=None, casting='same_kind')` | Stack arrays vertically (row-wise) |
| `np.hstack` | `hstack(tup, *, dtype=None, casting='same_kind')` | Stack arrays horizontally (column-wise) |
| `np.dstack` | `dstack(tup)` | Stack arrays depth-wise (along third axis) |
| `np.column_stack` | `column_stack(tup)` | Stack 1-D arrays as columns |
| `np.row_stack` | `row_stack(tup)` | Stack arrays as rows (deprecated, use vstack) |
| `np.block` | `block(arrays)` | Assemble nd-array from nested lists of blocks |

### Splitting Arrays
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.split` | `split(ary, indices_or_sections, axis=0)` | Split array into sub-arrays |
| `np.array_split` | `array_split(ary, indices_or_sections, axis=0)` | Split array into sub-arrays (allows unequal division) |
| `np.hsplit` | `hsplit(ary, indices_or_sections)` | Split array horizontally |
| `np.vsplit` | `vsplit(ary, indices_or_sections)` | Split array vertically |
| `np.dsplit` | `dsplit(ary, indices_or_sections)` | Split array along third axis |
| `np.unstack` | `unstack(array, /, *, axis=0)` | Split array into tuple of arrays along axis |

### Tiling and Repeating
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.tile` | `tile(A, reps)` | Construct array by repeating A |
| `np.repeat` | `repeat(a, repeats, axis=None)` | Repeat elements of array |

### Flipping and Rotating
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.flip` | `flip(m, axis=None)` | Reverse order of elements |
| `np.fliplr` | `fliplr(m)` | Flip array left to right |
| `np.flipud` | `flipud(m)` | Flip array up to down |
| `np.rot90` | `rot90(m, k=1, axes=(0, 1))` | Rotate array 90 degrees |
| `np.roll` | `roll(a, shift, axis=None)` | Roll array elements |

### Other Manipulation
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.resize` | `resize(a, new_shape)` | Return new array with given shape |
| `np.append` | `append(arr, values, axis=None)` | Append values to end of array |
| `np.insert` | `insert(arr, obj, values, axis=None)` | Insert values along axis |
| `np.delete` | `delete(arr, obj, axis=None)` | Delete elements from array |
| `np.trim_zeros` | `trim_zeros(filt, trim='fb')` | Trim leading/trailing zeros |
| `np.unique` | `unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)` | Find unique elements |
| `np.unique_all` | `unique_all(x)` | Return unique values, indices, inverse, and counts |
| `np.unique_counts` | `unique_counts(x)` | Return unique values and counts |
| `np.unique_inverse` | `unique_inverse(x)` | Return unique values and inverse indices |
| `np.unique_values` | `unique_values(x)` | Return unique values |
| `np.pad` | `pad(array, pad_width, mode='constant', **kwargs)` | Pad array |
| `np.require` | `require(a, dtype=None, requirements=None, *, like=None)` | Return array satisfying requirements |

---

## Mathematical Functions

### Basic Math
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.abs` | ufunc | Absolute value (alias for absolute) |
| `np.absolute` | ufunc | Absolute value |
| `np.fabs` | ufunc | Absolute value (float) |
| `np.sign` | ufunc | Sign of elements |
| `np.positive` | ufunc | Numerical positive (+x) |
| `np.negative` | ufunc | Numerical negative (-x) |
| `np.reciprocal` | ufunc | Reciprocal (1/x) |
| `np.sqrt` | ufunc | Square root |
| `np.cbrt` | ufunc | Cube root |
| `np.square` | ufunc | Square (x**2) |
| `np.power` | ufunc | First array elements raised to powers |
| `np.float_power` | ufunc | Float power (always returns float) |

### Rounding
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.round` | `round(a, decimals=0, out=None)` | Round to given decimals |
| `np.around` | `around(a, decimals=0, out=None)` | Round to given decimals (alias) |
| `np.rint` | ufunc | Round to nearest integer |
| `np.fix` | `fix(x, out=None)` | Round towards zero (pending deprecation) |
| `np.floor` | ufunc | Floor of elements |
| `np.ceil` | ufunc | Ceiling of elements |
| `np.trunc` | ufunc | Truncate elements |

### Sums and Products
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.sum` | `sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)` | Sum of array elements |
| `np.prod` | `prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)` | Product of array elements |
| `np.cumsum` | `cumsum(a, axis=None, dtype=None, out=None)` | Cumulative sum |
| `np.cumprod` | `cumprod(a, axis=None, dtype=None, out=None)` | Cumulative product |
| `np.cumulative_sum` | `cumulative_sum(x, /, *, axis=None, dtype=None, out=None, include_initial=False)` | Cumulative sum (Array API) |
| `np.cumulative_prod` | `cumulative_prod(x, /, *, axis=None, dtype=None, out=None, include_initial=False)` | Cumulative product (Array API) |
| `np.diff` | `diff(a, n=1, axis=-1, prepend=None, append=None)` | Discrete difference |
| `np.ediff1d` | `ediff1d(ary, to_end=None, to_begin=None)` | Differences between consecutive elements |
| `np.gradient` | `gradient(f, *varargs, axis=None, edge_order=1)` | Gradient of N-dimensional array |
| `np.trapezoid` | `trapezoid(y, x=None, dx=1.0, axis=-1)` | Trapezoidal integration |

### Special Values
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.clip` | `clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None)` | Clip values to range |
| `np.maximum` | ufunc | Element-wise maximum |
| `np.minimum` | ufunc | Element-wise minimum |
| `np.fmax` | ufunc | Element-wise maximum (ignores NaN) |
| `np.fmin` | ufunc | Element-wise minimum (ignores NaN) |
| `np.nan_to_num` | `nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None)` | Replace NaN/inf |
| `np.real_if_close` | `real_if_close(a, tol=100)` | Return real if imaginary close to zero |

### Miscellaneous Math
| Function | Signature | Description |
|----------|-----------|-------------|
| `np.convolve` | `convolve(a, v, mode='full')` | Discrete linear convolution |
| `np.correlate` | `correlate(a, v, mode='valid')` | Cross-correlation |
| `np.outer` | `outer(a, b, out=None)` | Outer product |
| `np.inner` | `inner(a, b)` | Inner product |
| `np.cross` | `cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)` | Cross product |
| `np.tensordot` | `tensordot(a, b, axes=2)` | Tensor dot product |
| `np.kron` | `kron(a, b)` | Kronecker product |
| `np.dot` | `dot(a, b, out=None)` | Dot product |
| `np.vdot` | `vdot(a, b)` | Vector dot product |
| `np.matmul` | `matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True)` | Matrix product |
| `np.einsum` | `einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)` | Einstein summation |
| `np.einsum_path` | `einsum_path(subscripts, *operands, optimize='greedy')` | Optimal contraction path |
| `np.modf` | ufunc | Return fractional and integral parts |
| `np.frexp` | ufunc | Decompose into mantissa and exponent |
| `np.ldexp` | ufunc | Compute x * 2**exp |
| `np.copysign` | ufunc | Copy sign of one array to another |
| `np.nextafter` | ufunc | Next floating-point value |
| `np.spacing` | ufunc | Distance to nearest float |
| `np.heaviside` | ufunc | Heaviside step function |
| `np.gcd` | ufunc | Greatest common divisor |
| `np.lcm` | ufunc | Least common multiple |
| `np.i0` | `i0(x)` | Modified Bessel function of first kind, order 0 |
| `np.sinc` | `sinc(x)` | Sinc function |
| `np.angle` | `angle(z, deg=False)` | Return angle of complex argument |
| `np.real` | `real(val)` | Return real part |
| `np.imag` | `imag(val)` | Return imaginary part |
| `np.conj` | ufunc | Complex conjugate (alias) |
| `np.conjugate` | ufunc | Complex conjugate |
| `np.interp` | `interp(x, xp, fp, left=None, right=None, period=None)` | 1-D linear interpolation |

---

## Universal Functions (ufuncs)

All ufuncs support common parameters: `out`, `where`, `casting`, `order`, `dtype`, `subok`, `signature`.

### Complete ufunc List:
`absolute`, `add`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`, `arctanh`, `bitwise_and`, `bitwise_count`, `bitwise_or`, `bitwise_xor`, `cbrt`, `ceil`, `conj`, `conjugate`, `copysign`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `divmod`, `equal`, `exp`, `exp2`, `expm1`, `fabs`, `float_power`, `floor`, `floor_divide`, `fmax`, `fmin`, `fmod`, `frexp`, `gcd`, `greater`, `greater_equal`, `heaviside`, `hypot`, `invert`, `isfinite`, `isinf`, `isnan`, `isnat`, `lcm`, `ldexp`, `left_shift`, `less`, `less_equal`, `log`, `log10`, `log1p`, `log2`, `logaddexp`, `logaddexp2`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `matmul`, `matvec`, `maximum`, `minimum`, `mod`, `modf`, `multiply`, `negative`, `nextafter`, `not_equal`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `remainder`, `right_shift`, `rint`, `sign`, `signbit`, `sin`, `sinh`, `spacing`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`, `vecdot`, `vecmat`

---

## Trigonometric Functions

| Function | Description |
|----------|-------------|
| `np.sin` | Sine |
| `np.cos` | Cosine |
| `np.tan` | Tangent |
| `np.arcsin` | Inverse sine |
| `np.arccos` | Inverse cosine |
| `np.arctan` | Inverse tangent |
| `np.arctan2` | Element-wise arc tangent of x1/x2 |
| `np.hypot` | Hypotenuse (sqrt(x1**2 + x2**2)) |
| `np.degrees` | Convert radians to degrees |
| `np.radians` | Convert degrees to radians |
| `np.deg2rad` | Convert degrees to radians |
| `np.rad2deg` | Convert radians to degrees |
| `np.unwrap` | Unwrap by changing deltas to complement |

---

## Hyperbolic Functions

| Function | Description |
|----------|-------------|
| `np.sinh` | Hyperbolic sine |
| `np.cosh` | Hyperbolic cosine |
| `np.tanh` | Hyperbolic tangent |
| `np.arcsinh` | Inverse hyperbolic sine |
| `np.arccosh` | Inverse hyperbolic cosine |
| `np.arctanh` | Inverse hyperbolic tangent |

---

## Exponential and Logarithmic

| Function | Description |
|----------|-------------|
| `np.exp` | Exponential (e**x) |
| `np.exp2` | 2**x |
| `np.expm1` | exp(x) - 1 |
| `np.log` | Natural logarithm |
| `np.log2` | Base-2 logarithm |
| `np.log10` | Base-10 logarithm |
| `np.log1p` | log(1 + x) |
| `np.logaddexp` | Log of sum of exponentials |
| `np.logaddexp2` | Log base 2 of sum of exponentials |

---

## Arithmetic Operations

| Function | Description |
|----------|-------------|
| `np.add` | Element-wise addition |
| `np.subtract` | Element-wise subtraction |
| `np.multiply` | Element-wise multiplication |
| `np.divide` | Element-wise division |
| `np.true_divide` | True division |
| `np.floor_divide` | Floor division |
| `np.mod` | Element-wise modulo |
| `np.remainder` | Element-wise remainder (same as mod) |
| `np.fmod` | Element-wise remainder (C-style) |
| `np.divmod` | Return quotient and remainder |

---

## Comparison Functions

| Function | Description |
|----------|-------------|
| `np.equal` | Element-wise equality |
| `np.not_equal` | Element-wise inequality |
| `np.less` | Element-wise less than |
| `np.less_equal` | Element-wise less than or equal |
| `np.greater` | Element-wise greater than |
| `np.greater_equal` | Element-wise greater than or equal |
| `np.array_equal` | True if arrays have same shape and elements |
| `np.array_equiv` | True if arrays are broadcastable and equal |
| `np.allclose` | True if all elements close within tolerance |
| `np.isclose` | Element-wise close within tolerance |

---

## Logical Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.all` | `all(a, axis=None, out=None, keepdims=False, *, where=True)` | Test if all elements are true |
| `np.any` | `any(a, axis=None, out=None, keepdims=False, *, where=True)` | Test if any element is true |
| `np.logical_and` | ufunc | Element-wise logical AND |
| `np.logical_or` | ufunc | Element-wise logical OR |
| `np.logical_not` | ufunc | Element-wise logical NOT |
| `np.logical_xor` | ufunc | Element-wise logical XOR |
| `np.isnan` | ufunc | Test for NaN |
| `np.isinf` | ufunc | Test for infinity |
| `np.isfinite` | ufunc | Test for finite |
| `np.isnat` | ufunc | Test for NaT (Not a Time) |
| `np.isneginf` | `isneginf(x, out=None)` | Test for negative infinity |
| `np.isposinf` | `isposinf(x, out=None)` | Test for positive infinity |
| `np.isreal` | `isreal(x)` | Test if element is real |
| `np.iscomplex` | `iscomplex(x)` | Test if element is complex |
| `np.isrealobj` | `isrealobj(x)` | Test if array is real type |
| `np.iscomplexobj` | `iscomplexobj(x)` | Test if array is complex type |
| `np.isscalar` | `isscalar(element)` | Test if element is scalar |
| `np.isfortran` | `isfortran(a)` | Test if array is Fortran contiguous |
| `np.iterable` | `iterable(y)` | Test if object is iterable |

---

## Bitwise Operations

| Function | Description |
|----------|-------------|
| `np.bitwise_and` | Element-wise AND |
| `np.bitwise_or` | Element-wise OR |
| `np.bitwise_xor` | Element-wise XOR |
| `np.bitwise_not` | Element-wise NOT (alias for invert) |
| `np.bitwise_invert` | Element-wise invert (Array API alias) |
| `np.bitwise_left_shift` | Shift bits left (Array API alias) |
| `np.bitwise_right_shift` | Shift bits right (Array API alias) |
| `np.invert` | Element-wise bit inversion |
| `np.left_shift` | Shift bits left |
| `np.right_shift` | Shift bits right |
| `np.bitwise_count` | Count number of 1-bits |
| `np.packbits` | Pack binary values into uint8 |
| `np.unpackbits` | Unpack uint8 into binary values |
| `np.binary_repr` | Return binary representation as string |
| `np.base_repr` | Return representation in given base |

---

## Statistical Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.mean` | `mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True)` | Arithmetic mean |
| `np.std` | `std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)` | Standard deviation |
| `np.var` | `var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)` | Variance |
| `np.median` | `median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` | Median |
| `np.average` | `average(a, axis=None, weights=None, returned=False, *, keepdims=False)` | Weighted average |
| `np.percentile` | `percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)` | Percentile |
| `np.quantile` | `quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)` | Quantile |
| `np.histogram` | `histogram(a, bins=10, range=None, density=None, weights=None)` | Compute histogram |
| `np.histogram2d` | `histogram2d(x, y, bins=10, range=None, density=None, weights=None)` | 2D histogram |
| `np.histogramdd` | `histogramdd(sample, bins=10, range=None, density=None, weights=None)` | Multidimensional histogram |
| `np.histogram_bin_edges` | `histogram_bin_edges(a, bins=10, range=None, weights=None)` | Compute histogram bin edges |
| `np.bincount` | `bincount(x, weights=None, minlength=0)` | Count occurrences |
| `np.digitize` | `digitize(x, bins, right=False)` | Return bin indices |
| `np.cov` | `cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None)` | Covariance matrix |
| `np.corrcoef` | `corrcoef(x, y=None, rowvar=True, bias=<no value>, ddof=<no value>, *, dtype=None)` | Correlation coefficients |
| `np.ptp` | `ptp(a, axis=None, out=None, keepdims=False)` | Peak to peak (max - min) |
| `np.count_nonzero` | `count_nonzero(a, axis=None, *, keepdims=False)` | Count non-zero elements |

### NaN-aware Functions
| Function | Description |
|----------|-------------|
| `np.nansum` | Sum ignoring NaN |
| `np.nanprod` | Product ignoring NaN |
| `np.nanmean` | Mean ignoring NaN |
| `np.nanstd` | Standard deviation ignoring NaN |
| `np.nanvar` | Variance ignoring NaN |
| `np.nanmedian` | Median ignoring NaN |
| `np.nanmin` | Minimum ignoring NaN |
| `np.nanmax` | Maximum ignoring NaN |
| `np.nanargmin` | Argmin ignoring NaN |
| `np.nanargmax` | Argmax ignoring NaN |
| `np.nancumsum` | Cumulative sum ignoring NaN |
| `np.nancumprod` | Cumulative product ignoring NaN |
| `np.nanpercentile` | Percentile ignoring NaN |
| `np.nanquantile` | Quantile ignoring NaN |

---

## Sorting and Searching

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.sort` | `sort(a, axis=-1, kind=None, order=None, *, stable=None)` | Return sorted copy |
| `np.sort_complex` | `sort_complex(a)` | Sort complex array by real, then imaginary |
| `np.argsort` | `argsort(a, axis=-1, kind=None, order=None, *, stable=None)` | Indices that would sort |
| `np.lexsort` | `lexsort(keys, axis=-1)` | Indirect stable sort using sequence of keys |
| `np.partition` | `partition(a, kth, axis=-1, kind='introselect', order=None)` | Partial sort |
| `np.argpartition` | `argpartition(a, kth, axis=-1, kind='introselect', order=None)` | Indices for partial sort |
| `np.searchsorted` | `searchsorted(a, v, side='left', sorter=None)` | Find indices for sorted array |
| `np.argmax` | `argmax(a, axis=None, out=None, *, keepdims=False)` | Indices of maximum |
| `np.argmin` | `argmin(a, axis=None, out=None, *, keepdims=False)` | Indices of minimum |
| `np.max` | `max(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` | Maximum (alias for amax) |
| `np.min` | `min(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` | Minimum (alias for amin) |
| `np.amax` | `amax(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` | Maximum |
| `np.amin` | `amin(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` | Minimum |
| `np.argwhere` | `argwhere(a)` | Find indices of non-zero elements |
| `np.nonzero` | `nonzero(a)` | Return indices of non-zero elements |
| `np.flatnonzero` | `flatnonzero(a)` | Indices of non-zero in flattened array |
| `np.where` | `where(condition, [x, y], /)` | Return elements based on condition |
| `np.extract` | `extract(condition, arr)` | Return elements satisfying condition |
| `np.place` | `place(arr, mask, vals)` | Change elements based on condition |
| `np.select` | `select(condlist, choicelist, default=0)` | Return elements from choicelist based on conditions |
| `np.piecewise` | `piecewise(x, condlist, funclist, *args, **kw)` | Evaluate piecewise function |

---

## Set Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.unique` | See above | Find unique elements |
| `np.intersect1d` | `intersect1d(ar1, ar2, assume_unique=False, return_indices=False)` | Intersection of two arrays |
| `np.union1d` | `union1d(ar1, ar2)` | Union of two arrays |
| `np.setdiff1d` | `setdiff1d(ar1, ar2, assume_unique=False)` | Set difference |
| `np.setxor1d` | `setxor1d(ar1, ar2, assume_unique=False)` | Set exclusive-or |
| `np.isin` | `isin(element, test_elements, assume_unique=False, invert=False, *, kind=None)` | Test membership |

---

## Window Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.hamming` | `hamming(M)` | Hamming window |
| `np.hanning` | `hanning(M)` | Hanning window |
| `np.bartlett` | `bartlett(M)` | Bartlett window |
| `np.blackman` | `blackman(M)` | Blackman window |
| `np.kaiser` | `kaiser(M, beta)` | Kaiser window |

---

## Linear Algebra (np.linalg)

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.linalg.norm` | `norm(x, ord=None, axis=None, keepdims=False)` | Matrix or vector norm |
| `np.linalg.matrix_norm` | `matrix_norm(x, /, *, ord='fro', keepdims=False)` | Matrix norm (Array API) |
| `np.linalg.vector_norm` | `vector_norm(x, /, *, axis=None, ord=2, keepdims=False)` | Vector norm (Array API) |
| `np.linalg.cond` | `cond(x, p=None)` | Condition number |
| `np.linalg.det` | `det(a)` | Determinant |
| `np.linalg.slogdet` | `slogdet(a)` | Sign and log of determinant |
| `np.linalg.matrix_rank` | `matrix_rank(A, tol=None, hermitian=False, *, rtol=None)` | Matrix rank |
| `np.linalg.trace` | `trace(x, /, *, offset=0, dtype=None)` | Sum along diagonal |
| `np.linalg.diagonal` | `diagonal(x, /, *, offset=0)` | Return diagonal |
| `np.linalg.solve` | `solve(a, b)` | Solve linear equations |
| `np.linalg.tensorsolve` | `tensorsolve(a, b, axes=None)` | Solve tensor equation |
| `np.linalg.lstsq` | `lstsq(a, b, rcond=None)` | Least-squares solution |
| `np.linalg.inv` | `inv(a)` | Matrix inverse |
| `np.linalg.pinv` | `pinv(a, rcond=None, hermitian=False, *, rtol=<no value>)` | Pseudo-inverse |
| `np.linalg.tensorinv` | `tensorinv(a, ind=2)` | Tensor inverse |
| `np.linalg.matrix_power` | `matrix_power(a, n)` | Matrix power |
| `np.linalg.cholesky` | `cholesky(a, /, *, upper=False)` | Cholesky decomposition |
| `np.linalg.qr` | `qr(a, mode='reduced')` | QR decomposition |
| `np.linalg.svd` | `svd(a, full_matrices=True, compute_uv=True, hermitian=False)` | Singular value decomposition |
| `np.linalg.svdvals` | `svdvals(x, /)` | Singular values |
| `np.linalg.eig` | `eig(a)` | Eigenvalues and eigenvectors |
| `np.linalg.eigh` | `eigh(a, UPLO='L')` | Eigenvalues and eigenvectors (Hermitian) |
| `np.linalg.eigvals` | `eigvals(a)` | Eigenvalues |
| `np.linalg.eigvalsh` | `eigvalsh(a, UPLO='L')` | Eigenvalues (Hermitian) |
| `np.linalg.multi_dot` | `multi_dot(arrays, *, out=None)` | Dot product of multiple arrays |
| `np.linalg.cross` | `cross(x1, x2, /, *, axis=-1)` | Cross product (Array API) |
| `np.linalg.outer` | `outer(x1, x2, /)` | Outer product (Array API) |
| `np.linalg.matmul` | `matmul(x1, x2, /)` | Matrix product (Array API) |
| `np.linalg.matrix_transpose` | `matrix_transpose(x, /)` | Transpose last two axes |
| `np.linalg.tensordot` | `tensordot(a, b, /, *, axes=2)` | Tensor dot product (Array API) |
| `np.linalg.vecdot` | `vecdot(x1, x2, /, *, axis=-1)` | Vector dot product |
| `np.linalg.LinAlgError` | Exception | Linear algebra error |

---

## FFT (np.fft)

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.fft.fft` | `fft(a, n=None, axis=-1, norm=None, out=None)` | 1-D FFT |
| `np.fft.ifft` | `ifft(a, n=None, axis=-1, norm=None, out=None)` | 1-D inverse FFT |
| `np.fft.fft2` | `fft2(a, s=None, axes=(-2, -1), norm=None, out=None)` | 2-D FFT |
| `np.fft.ifft2` | `ifft2(a, s=None, axes=(-2, -1), norm=None, out=None)` | 2-D inverse FFT |
| `np.fft.fftn` | `fftn(a, s=None, axes=None, norm=None, out=None)` | N-D FFT |
| `np.fft.ifftn` | `ifftn(a, s=None, axes=None, norm=None, out=None)` | N-D inverse FFT |
| `np.fft.rfft` | `rfft(a, n=None, axis=-1, norm=None, out=None)` | 1-D FFT of real input |
| `np.fft.irfft` | `irfft(a, n=None, axis=-1, norm=None, out=None)` | 1-D inverse FFT of real input |
| `np.fft.rfft2` | `rfft2(a, s=None, axes=(-2, -1), norm=None, out=None)` | 2-D FFT of real input |
| `np.fft.irfft2` | `irfft2(a, s=None, axes=(-2, -1), norm=None, out=None)` | 2-D inverse FFT of real input |
| `np.fft.rfftn` | `rfftn(a, s=None, axes=None, norm=None, out=None)` | N-D FFT of real input |
| `np.fft.irfftn` | `irfftn(a, s=None, axes=None, norm=None, out=None)` | N-D inverse FFT of real input |
| `np.fft.hfft` | `hfft(a, n=None, axis=-1, norm=None, out=None)` | FFT of Hermitian-symmetric signal |
| `np.fft.ihfft` | `ihfft(a, n=None, axis=-1, norm=None, out=None)` | Inverse FFT of Hermitian-symmetric signal |
| `np.fft.fftfreq` | `fftfreq(n, d=1.0, *, device=None)` | FFT sample frequencies |
| `np.fft.rfftfreq` | `rfftfreq(n, d=1.0, *, device=None)` | FFT sample frequencies (real) |
| `np.fft.fftshift` | `fftshift(x, axes=None)` | Shift zero-frequency to center |
| `np.fft.ifftshift` | `ifftshift(x, axes=None)` | Inverse of fftshift |

---

## Random Sampling (np.random)

### Legacy Functions (module-level)
| Function | Description |
|----------|-------------|
| `np.random.seed` | Seed the generator |
| `np.random.get_state` | Get generator state |
| `np.random.set_state` | Set generator state |
| `np.random.rand` | Random values in [0, 1) |
| `np.random.randn` | Standard normal distribution |
| `np.random.randint` | Random integers |
| `np.random.random` | Random floats in [0, 1) |
| `np.random.random_sample` | Random floats in [0, 1) |
| `np.random.ranf` | Random floats in [0, 1) (alias) |
| `np.random.sample` | Random floats in [0, 1) (alias) |
| `np.random.random_integers` | Random integers (deprecated) |
| `np.random.choice` | Random sample from array |
| `np.random.bytes` | Random bytes |
| `np.random.shuffle` | Shuffle array in-place |
| `np.random.permutation` | Random permutation |

### Distributions
| Function | Description |
|----------|-------------|
| `np.random.beta` | Beta distribution |
| `np.random.binomial` | Binomial distribution |
| `np.random.chisquare` | Chi-square distribution |
| `np.random.dirichlet` | Dirichlet distribution |
| `np.random.exponential` | Exponential distribution |
| `np.random.f` | F distribution |
| `np.random.gamma` | Gamma distribution |
| `np.random.geometric` | Geometric distribution |
| `np.random.gumbel` | Gumbel distribution |
| `np.random.hypergeometric` | Hypergeometric distribution |
| `np.random.laplace` | Laplace distribution |
| `np.random.logistic` | Logistic distribution |
| `np.random.lognormal` | Log-normal distribution |
| `np.random.logseries` | Logarithmic series distribution |
| `np.random.multinomial` | Multinomial distribution |
| `np.random.multivariate_normal` | Multivariate normal distribution |
| `np.random.negative_binomial` | Negative binomial distribution |
| `np.random.noncentral_chisquare` | Non-central chi-square distribution |
| `np.random.noncentral_f` | Non-central F distribution |
| `np.random.normal` | Normal distribution |
| `np.random.pareto` | Pareto distribution |
| `np.random.poisson` | Poisson distribution |
| `np.random.power` | Power distribution |
| `np.random.rayleigh` | Rayleigh distribution |
| `np.random.standard_cauchy` | Standard Cauchy distribution |
| `np.random.standard_exponential` | Standard exponential distribution |
| `np.random.standard_gamma` | Standard gamma distribution |
| `np.random.standard_normal` | Standard normal distribution |
| `np.random.standard_t` | Standard Student's t distribution |
| `np.random.triangular` | Triangular distribution |
| `np.random.uniform` | Uniform distribution |
| `np.random.vonmises` | Von Mises distribution |
| `np.random.wald` | Wald distribution |
| `np.random.weibull` | Weibull distribution |
| `np.random.zipf` | Zipf distribution |

### Classes
| Class | Description |
|-------|-------------|
| `np.random.Generator` | Container for BitGenerators |
| `np.random.RandomState` | Legacy random number generator |
| `np.random.SeedSequence` | Seed sequence for entropy |
| `np.random.BitGenerator` | Base class for bit generators |
| `np.random.MT19937` | Mersenne Twister generator |
| `np.random.PCG64` | PCG-64 generator |
| `np.random.PCG64DXSM` | PCG-64 DXSM generator |
| `np.random.Philox` | Philox counter-based generator |
| `np.random.SFC64` | SFC64 generator |
| `np.random.default_rng` | Construct default Generator |

---

## Polynomial (np.polynomial)

| Class/Function | Description |
|----------------|-------------|
| `np.polynomial.Polynomial` | Power series polynomial |
| `np.polynomial.Chebyshev` | Chebyshev polynomial |
| `np.polynomial.Legendre` | Legendre polynomial |
| `np.polynomial.Hermite` | Hermite polynomial |
| `np.polynomial.HermiteE` | Hermite E polynomial |
| `np.polynomial.Laguerre` | Laguerre polynomial |
| `np.polynomial.set_default_printstyle` | Set default print style |

### Legacy Polynomial Functions (np.*)
| Function | Description |
|----------|-------------|
| `np.poly` | Find coefficients from roots |
| `np.roots` | Find roots of polynomial |
| `np.polyfit` | Least squares polynomial fit |
| `np.polyval` | Evaluate polynomial |
| `np.polyadd` | Add polynomials |
| `np.polysub` | Subtract polynomials |
| `np.polymul` | Multiply polynomials |
| `np.polydiv` | Divide polynomials |
| `np.polyint` | Integrate polynomial |
| `np.polyder` | Differentiate polynomial |
| `np.poly1d` | 1-D polynomial class |

---

## Masked Arrays (np.ma)

| Item | Description |
|------|-------------|
| `np.ma.MaskedArray` | Array with masked values |
| `np.ma.masked` | Masked constant |
| `np.ma.nomask` | No mask constant |
| `np.ma.masked_array` | Alias for MaskedArray |
| `np.ma.array` | Create masked array |
| `np.ma.is_masked` | Test if masked |
| `np.ma.is_mask` | Test if valid mask |
| `np.ma.getmask` | Get mask |
| `np.ma.getdata` | Get data |
| `np.ma.getmaskarray` | Get mask as array |
| `np.ma.make_mask` | Create mask |
| `np.ma.make_mask_none` | Create mask of False |
| `np.ma.make_mask_descr` | Create mask dtype |
| `np.ma.mask_or` | Combine masks with OR |
| `np.ma.masked_where` | Mask where condition |
| `np.ma.masked_equal` | Mask equal values |
| `np.ma.masked_not_equal` | Mask not equal values |
| `np.ma.masked_less` | Mask less than |
| `np.ma.masked_greater` | Mask greater than |
| `np.ma.masked_less_equal` | Mask less than or equal |
| `np.ma.masked_greater_equal` | Mask greater than or equal |
| `np.ma.masked_inside` | Mask inside interval |
| `np.ma.masked_outside` | Mask outside interval |
| `np.ma.masked_invalid` | Mask invalid values |
| `np.ma.masked_object` | Mask object values |
| `np.ma.masked_values` | Mask given values |
| `np.ma.fix_invalid` | Replace invalid with fill value |
| `np.ma.filled` | Return array with masked values filled |
| `np.ma.compressed` | Return non-masked data as 1-D |
| `np.ma.harden_mask` | Force mask to be unchangeable |
| `np.ma.soften_mask` | Allow mask to be changeable |
| `np.ma.set_fill_value` | Set fill value |
| `np.ma.default_fill_value` | Return default fill value |
| `np.ma.common_fill_value` | Return common fill value |
| `np.ma.maximum_fill_value` | Return maximum fill value |
| `np.ma.minimum_fill_value` | Return minimum fill value |

Plus all standard array functions with masked-aware behavior.

---

## String Operations (np.char)

`np.char` provides character/string array operations (legacy module):

| Function | Description |
|----------|-------------|
| `np.char.add` | Concatenate strings |
| `np.char.multiply` | Multiple concatenation |
| `np.char.mod` | String formatting |
| `np.char.capitalize` | Capitalize first character |
| `np.char.center` | Center in string of length |
| `np.char.decode` | Decode bytes to string |
| `np.char.encode` | Encode string to bytes |
| `np.char.expandtabs` | Replace tabs with spaces |
| `np.char.join` | Join strings |
| `np.char.ljust` | Left-justify |
| `np.char.lower` | Convert to lowercase |
| `np.char.lstrip` | Strip leading characters |
| `np.char.partition` | Partition around separator |
| `np.char.replace` | Replace substring |
| `np.char.rjust` | Right-justify |
| `np.char.rpartition` | Partition around last separator |
| `np.char.rsplit` | Split from right |
| `np.char.rstrip` | Strip trailing characters |
| `np.char.split` | Split string |
| `np.char.splitlines` | Split by lines |
| `np.char.strip` | Strip leading/trailing |
| `np.char.swapcase` | Swap case |
| `np.char.title` | Title case |
| `np.char.translate` | Translate characters |
| `np.char.upper` | Convert to uppercase |
| `np.char.zfill` | Pad with zeros |
| `np.char.count` | Count occurrences |
| `np.char.endswith` | Test suffix |
| `np.char.find` | Find substring |
| `np.char.index` | Find substring (raise) |
| `np.char.isalnum` | Test alphanumeric |
| `np.char.isalpha` | Test alphabetic |
| `np.char.isdecimal` | Test decimal |
| `np.char.isdigit` | Test digit |
| `np.char.islower` | Test lowercase |
| `np.char.isnumeric` | Test numeric |
| `np.char.isspace` | Test whitespace |
| `np.char.istitle` | Test title case |
| `np.char.isupper` | Test uppercase |
| `np.char.rfind` | Find from right |
| `np.char.rindex` | Find from right (raise) |
| `np.char.startswith` | Test prefix |
| `np.char.str_len` | String length |
| `np.char.equal` | Element-wise equality |
| `np.char.not_equal` | Element-wise inequality |
| `np.char.greater` | Element-wise greater |
| `np.char.greater_equal` | Element-wise greater or equal |
| `np.char.less` | Element-wise less |
| `np.char.less_equal` | Element-wise less or equal |
| `np.char.compare_chararrays` | Compare character arrays |
| `np.char.array` | Create character array |
| `np.char.asarray` | Convert to character array |
| `np.char.chararray` | Character array class |

---

## String Operations (np.strings)

`np.strings` is the new string operations module (NumPy 2.x):

| Function | Description |
|----------|-------------|
| `np.strings.add` | Concatenate strings |
| `np.strings.multiply` | Multiple concatenation |
| `np.strings.mod` | String formatting |
| `np.strings.capitalize` | Capitalize first character |
| `np.strings.center` | Center in string of length |
| `np.strings.decode` | Decode bytes to string |
| `np.strings.encode` | Encode string to bytes |
| `np.strings.expandtabs` | Replace tabs with spaces |
| `np.strings.ljust` | Left-justify |
| `np.strings.lower` | Convert to lowercase |
| `np.strings.lstrip` | Strip leading characters |
| `np.strings.partition` | Partition around separator |
| `np.strings.replace` | Replace substring |
| `np.strings.rjust` | Right-justify |
| `np.strings.rpartition` | Partition around last separator |
| `np.strings.rstrip` | Strip trailing characters |
| `np.strings.strip` | Strip leading/trailing |
| `np.strings.swapcase` | Swap case |
| `np.strings.title` | Title case |
| `np.strings.translate` | Translate characters |
| `np.strings.upper` | Convert to uppercase |
| `np.strings.zfill` | Pad with zeros |
| `np.strings.count` | Count occurrences |
| `np.strings.endswith` | Test suffix |
| `np.strings.find` | Find substring |
| `np.strings.rfind` | Find from right |
| `np.strings.index` | Find substring (raise) |
| `np.strings.rindex` | Find from right (raise) |
| `np.strings.isalnum` | Test alphanumeric |
| `np.strings.isalpha` | Test alphabetic |
| `np.strings.isdecimal` | Test decimal |
| `np.strings.isdigit` | Test digit |
| `np.strings.islower` | Test lowercase |
| `np.strings.isnumeric` | Test numeric |
| `np.strings.isspace` | Test whitespace |
| `np.strings.istitle` | Test title case |
| `np.strings.isupper` | Test uppercase |
| `np.strings.startswith` | Test prefix |
| `np.strings.str_len` | String length |
| `np.strings.equal` | Element-wise equality |
| `np.strings.not_equal` | Element-wise inequality |
| `np.strings.greater` | Element-wise greater |
| `np.strings.greater_equal` | Element-wise greater or equal |
| `np.strings.less` | Element-wise less |
| `np.strings.less_equal` | Element-wise less or equal |
| `np.strings.slice` | Slice strings (new in 2.x) |

---

## Record Arrays (np.rec)

| Function | Description |
|----------|-------------|
| `np.rec.array` | Create record array |
| `np.rec.fromarrays` | Create record array from arrays |
| `np.rec.fromrecords` | Create record array from records |
| `np.rec.fromstring` | Create record array from string |
| `np.rec.fromfile` | Create record array from file |
| `np.rec.format_parser` | Parse format string |
| `np.rec.find_duplicate` | Find duplicate field names |
| `np.rec.recarray` | Record array class |
| `np.rec.record` | Record scalar type |

---

## Ctypes Interop (np.ctypeslib)

| Function | Description |
|----------|-------------|
| `np.ctypeslib.load_library` | Load shared library |
| `np.ctypeslib.ndpointer` | Create ndarray pointer type |
| `np.ctypeslib.c_intp` | ctypes type for numpy intp |
| `np.ctypeslib.as_ctypes` | Create ctypes from ndarray |
| `np.ctypeslib.as_array` | Create ndarray from ctypes |
| `np.ctypeslib.as_ctypes_type` | Convert dtype to ctypes type |

---

## File I/O

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.save` | `save(file, arr, allow_pickle=True)` | Save array to .npy file |
| `np.savez` | `savez(file, *args, allow_pickle=True, **kwds)` | Save arrays to .npz file |
| `np.savez_compressed` | `savez_compressed(file, *args, allow_pickle=True, **kwds)` | Save arrays to compressed .npz |
| `np.load` | `load(file, mmap_mode=None, allow_pickle=False, ...)` | Load array from .npy/.npz file |
| `np.loadtxt` | `loadtxt(fname, dtype=float, comments='#', delimiter=None, ...)` | Load from text file |
| `np.savetxt` | `savetxt(fname, X, fmt='%.18e', delimiter=' ', ...)` | Save to text file |
| `np.genfromtxt` | `genfromtxt(fname, dtype=float, comments='#', ...)` | Load from text with missing values |
| `np.fromfile` | `fromfile(file, dtype=float, count=-1, sep='', ...)` | Read from binary file |
| `np.fromregex` | `fromregex(file, regexp, dtype, encoding=None)` | Load using regex |
| `np.tofile` | Method on ndarray | Write to binary file |

---

## Memory and Buffer

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.shares_memory` | `shares_memory(a, b, max_work=None)` | Test if arrays share memory |
| `np.may_share_memory` | `may_share_memory(a, b, max_work=None)` | Test if arrays might share memory |
| `np.copyto` | `copyto(dst, src, casting='same_kind', where=True)` | Copy values |
| `np.putmask` | `putmask(a, mask, values)` | Set values based on mask |
| `np.put` | `put(a, ind, v, mode='raise')` | Set values at indices |
| `np.take` | `take(a, indices, axis=None, out=None, mode='raise')` | Take elements |
| `np.take_along_axis` | `take_along_axis(arr, indices, axis)` | Take along axis |
| `np.put_along_axis` | `put_along_axis(arr, indices, values, axis)` | Put along axis |
| `np.choose` | `choose(a, choices, out=None, mode='raise')` | Construct array from index array |
| `np.compress` | `compress(condition, a, axis=None, out=None)` | Select slices |
| `np.diagonal` | `diagonal(a, offset=0, axis1=0, axis2=1)` | Return diagonal |
| `np.trace` | `trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)` | Sum along diagonal |
| `np.fill_diagonal` | `fill_diagonal(a, val, wrap=False)` | Fill diagonal |
| `np.diag_indices` | `diag_indices(n, ndim=2)` | Return diagonal indices |
| `np.diag_indices_from` | `diag_indices_from(arr)` | Return diagonal indices from array |
| `np.mask_indices` | `mask_indices(n, mask_func, k=0)` | Return indices for mask |
| `np.tril_indices` | `tril_indices(n, k=0, m=None)` | Return lower triangle indices |
| `np.triu_indices` | `triu_indices(n, k=0, m=None)` | Return upper triangle indices |
| `np.tril_indices_from` | `tril_indices_from(arr, k=0)` | Return lower triangle indices from array |
| `np.triu_indices_from` | `triu_indices_from(arr, k=0)` | Return upper triangle indices from array |

---

## Indexing Routines

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.ravel_multi_index` | `ravel_multi_index(multi_index, dims, mode='raise', order='C')` | Convert multi-index to flat index |
| `np.unravel_index` | `unravel_index(indices, shape, order='C')` | Convert flat index to multi-index |
| `np.ix_` | `ix_(*args)` | Open mesh from sequences |
| `np.r_` | Indexing object | Row-wise stacking |
| `np.c_` | Indexing object | Column-wise stacking |
| `np.s_` | Indexing object | Build index tuple |
| `np.index_exp` | Indexing object | Build index expression |
| `np.ndenumerate` | `ndenumerate(arr)` | Multidimensional index iterator |
| `np.ndindex` | `ndindex(*shape)` | Iterator over array indices |
| `np.apply_along_axis` | `apply_along_axis(func1d, axis, arr, *args, **kwargs)` | Apply function along axis |
| `np.apply_over_axes` | `apply_over_axes(func, a, axes)` | Apply function over multiple axes |
| `np.vectorize` | `vectorize(pyfunc, otypes=None, ...)` | Generalized function |
| `np.frompyfunc` | `frompyfunc(func, nin, nout, *, identity)` | Create ufunc from Python function |

---

## Broadcasting

| Function | Signature | Description |
|----------|-----------|-------------|
| `np.broadcast` | `broadcast(*args)` | Produce broadcast iterator |
| `np.broadcast_to` | `broadcast_to(array, shape, subok=False)` | Broadcast array to shape |
| `np.broadcast_arrays` | `broadcast_arrays(*args, subok=False)` | Broadcast arrays against each other |
| `np.broadcast_shapes` | `broadcast_shapes(*shapes)` | Broadcast shape calculation |

---

## Stride Tricks

Located in `np.lib.stride_tricks`:

| Function | Description |
|----------|-------------|
| `as_strided` | Create view with given shape and strides |
| `sliding_window_view` | Create sliding window view of array |
| `broadcast_to` | Broadcast array to shape |
| `broadcast_arrays` | Broadcast arrays against each other |
| `broadcast_shapes` | Broadcast shape calculation |

---

## Array Printing

| Function | Description |
|----------|-------------|
| `np.set_printoptions` | Set printing options |
| `np.get_printoptions` | Get printing options |
| `np.printoptions` | Context manager for print options |
| `np.array2string` | Return string representation |
| `np.array_str` | Return string for array |
| `np.array_repr` | Return repr for array |
| `np.format_float_positional` | Format float positionally |
| `np.format_float_scientific` | Format float scientifically |

---

## Error Handling

| Function | Description |
|----------|-------------|
| `np.seterr` | Set error handling |
| `np.geterr` | Get error handling settings |
| `np.seterrcall` | Set error callback |
| `np.geterrcall` | Get error callback |
| `np.errstate` | Context manager for error handling |
| `np.setbufsize` | Set buffer size |
| `np.getbufsize` | Get buffer size |

---

## Type Information

| Function | Description |
|----------|-------------|
| `np.dtype` | Data type object |
| `np.finfo` | Machine limits for float types |
| `np.iinfo` | Machine limits for integer types |
| `np.can_cast` | Returns whether cast can occur |
| `np.promote_types` | Returns common type |
| `np.min_scalar_type` | Returns minimum scalar type |
| `np.result_type` | Returns result type |
| `np.common_type` | Returns common type |
| `np.issubdtype` | Returns if dtype is subtype |
| `np.isdtype` | Returns if object is dtype |
| `np.typename` | Return type name |
| `np.mintypecode` | Return minimum character code |
| `np.ScalarType` | Tuple of all scalar types |
| `np.typecodes` | Dict of type codes |
| `np.sctypeDict` | Dict mapping names to types |
| `np.astype` | Cast array to dtype |

---

## Typing (np.typing)

| Item | Description |
|------|-------------|
| `ArrayLike` | Type hint for array-like objects |
| `DTypeLike` | Type hint for dtype-like objects |
| `NDArray` | Type hint for ndarray |
| `NBitBase` | Base class for bit-precision types |

---

## Testing (np.testing)

| Item | Description |
|------|-------------|
| `assert_` | Assert with error message |
| `assert_equal` | Assert equal |
| `assert_almost_equal` | Assert almost equal |
| `assert_approx_equal` | Assert approximately equal |
| `assert_array_equal` | Assert arrays equal |
| `assert_array_almost_equal` | Assert arrays almost equal |
| `assert_array_almost_equal_nulp` | Assert almost equal to ULP |
| `assert_array_less` | Assert array less than |
| `assert_array_max_ulp` | Assert within ULP |
| `assert_array_compare` | Compare arrays |
| `assert_string_equal` | Assert strings equal |
| `assert_allclose` | Assert all close |
| `assert_raises` | Assert raises exception |
| `assert_raises_regex` | Assert raises with regex |
| `assert_warns` | Assert warning raised |
| `assert_no_warnings` | Assert no warnings |
| `assert_no_gc_cycles` | Assert no gc cycles |
| `TestCase` | Unit test case class |
| `SkipTest` | Skip test exception |
| `KnownFailureException` | Known failure exception |
| `IgnoreException` | Ignore exception |
| `suppress_warnings` | Context manager for warnings |
| `clear_and_catch_warnings` | Clear and catch warnings |
| `verbose` | Verbose flag |
| `rundocs` | Run doctests |
| `runstring` | Run string as test |
| `run_threaded` | Run test threaded |
| `tempdir` | Context manager for temp directory |
| `temppath` | Context manager for temp file |
| `decorate_methods` | Decorate test methods |
| `measure` | Measure function execution |
| `memusage` | Memory usage |
| `jiffies` | CPU time measurement |
| `build_err_msg` | Build error message |
| `print_assert_equal` | Print assertion equality |
| `break_cycles` | Break reference cycles |

---

## Exceptions (np.exceptions)

| Exception | Description |
|-----------|-------------|
| `AxisError` | Invalid axis error |
| `ComplexWarning` | Complex cast warning |
| `DTypePromotionError` | DType promotion error |
| `ModuleDeprecationWarning` | Module deprecation warning |
| `RankWarning` | Polyfit rank warning |
| `TooHardError` | Problem too hard to solve |
| `VisibleDeprecationWarning` | Visible deprecation warning |

---

## Array API Aliases

NumPy 2.x provides Array API (2024.12) compatible aliases:

| Alias | Original Function |
|-------|-------------------|
| `np.acos` | `np.arccos` |
| `np.acosh` | `np.arccosh` |
| `np.asin` | `np.arcsin` |
| `np.asinh` | `np.arcsinh` |
| `np.atan` | `np.arctan` |
| `np.atan2` | `np.arctan2` |
| `np.atanh` | `np.arctanh` |
| `np.concat` | `np.concatenate` |
| `np.permute_dims` | `np.transpose` |
| `np.pow` | `np.power` |
| `np.bitwise_invert` | `np.invert` |
| `np.bitwise_left_shift` | `np.left_shift` |
| `np.bitwise_right_shift` | `np.right_shift` |

---

## Submodules

| Submodule | Description |
|-----------|-------------|
| `np.char` | Character/string operations (legacy) |
| `np.core` | Core array functionality (legacy) |
| `np.ctypeslib` | Ctypes interoperability |
| `np.dtypes` | DType classes |
| `np.exceptions` | Exceptions |
| `np.f2py` | Fortran to Python interface |
| `np.fft` | Discrete Fourier transforms |
| `np.lib` | Library functions |
| `np.linalg` | Linear algebra |
| `np.ma` | Masked arrays |
| `np.polynomial` | Polynomial functions |
| `np.random` | Random sampling |
| `np.rec` | Record arrays |
| `np.strings` | String operations (new in 2.x) |
| `np.testing` | Testing utilities |
| `np.typing` | Type annotations |
| `np.emath` | Extended math (handles complex) |

---

## Classes

| Class | Description |
|-------|-------------|
| `np.ndarray` | N-dimensional array |
| `np.nditer` | Efficient multi-dimensional iterator |
| `np.nested_iters` | Nested nditer |
| `np.flatiter` | Flat iterator |
| `np.ndenumerate` | Multi-dimensional enumerate |
| `np.ndindex` | Iterator over indices |
| `np.broadcast` | Broadcast object |
| `np.dtype` | Data type object |
| `np.ufunc` | Universal function class |
| `np.matrix` | Matrix class (legacy) |
| `np.memmap` | Memory-mapped array |
| `np.record` | Record in record array |
| `np.recarray` | Record array |
| `np.busdaycalendar` | Business day calendar |
| `np.poly1d` | 1-D polynomial |
| `np.vectorize` | Generalized function class |
| `np.errstate` | Error state context manager |
| `np.printoptions` | Print options context manager |
| `np.chararray` | Character array (deprecated) |

---

## Deprecated APIs

| Item | Status | Replacement |
|------|--------|-------------|
| `np.row_stack` | Deprecated | `np.vstack` |
| `np.fix` | Pending deprecation | `np.trunc` |
| `np.chararray` | Deprecated | Use string dtype arrays |
| `np.random.random_integers` | Deprecated | `np.random.integers` |

---

## Removed APIs (NumPy 2.0)

These were removed in NumPy 2.0:

| Item | Migration |
|------|-----------|
| `np.geterrobj` | Use `np.errstate` context manager |
| `np.seterrobj` | Use `np.errstate` context manager |
| `np.cast` | Use `np.asarray(arr, dtype=dtype)` |
| `np.source` | Use `inspect.getsource` |
| `np.lookfor` | Search NumPy's documentation directly |
| `np.who` | Use IDE variable explorer or `locals()` |
| `np.fastCopyAndTranspose` | Use `arr.T.copy()` |
| `np.set_numeric_ops` | Use `PyUFunc_ReplaceLoopBySignature` |
| `np.NINF` | Use `-np.inf` |
| `np.PINF` | Use `np.inf` |
| `np.NZERO` | Use `-0.0` |
| `np.PZERO` | Use `0.0` |
| `np.add_newdoc` | Available as `np.lib.add_newdoc` |
| `np.add_docstring` | Available as `np.lib.add_docstring` |
| `np.safe_eval` | Use `ast.literal_eval` |
| `np.float_` | Use `np.float64` |
| `np.complex_` | Use `np.complex128` |
| `np.longfloat` | Use `np.longdouble` |
| `np.singlecomplex` | Use `np.complex64` |
| `np.cfloat` | Use `np.complex128` |
| `np.longcomplex` | Use `np.clongdouble` |
| `np.clongfloat` | Use `np.clongdouble` |
| `np.string_` | Use `np.bytes_` |
| `np.unicode_` | Use `np.str_` |
| `np.Inf` | Use `np.inf` |
| `np.Infinity` | Use `np.inf` |
| `np.NaN` | Use `np.nan` |
| `np.infty` | Use `np.inf` |
| `np.issctype` | Use `issubclass(rep, np.generic)` |
| `np.maximum_sctype` | Use specific dtype explicitly |
| `np.obj2sctype` | Use `np.dtype(obj).type` |
| `np.sctype2char` | Use `np.dtype(obj).char` |
| `np.sctypes` | Access dtypes explicitly |
| `np.issubsctype` | Use `np.issubdtype` |
| `np.set_string_function` | Use `np.set_printoptions` |
| `np.asfarray` | Use `np.asarray` with dtype |
| `np.issubclass_` | Use `issubclass` builtin |
| `np.tracemalloc_domain` | Available from `np.lib` |
| `np.mat` | Use `np.asmatrix` |
| `np.recfromcsv` | Use `np.genfromtxt` with comma delimiter |
| `np.recfromtxt` | Use `np.genfromtxt` |
| `np.deprecate` | Use `warnings.warn` with `DeprecationWarning` |
| `np.deprecate_with_doc` | Use `warnings.warn` with `DeprecationWarning` |
| `np.find_common_type` | Use `np.promote_types` or `np.result_type` |
| `np.round_` | Use `np.round` |
| `np.get_array_wrap` | No replacement |
| `np.DataSource` | Available as `np.lib.npyio.DataSource` |
| `np.nbytes` | Use `np.dtype(<dtype>).itemsize` |
| `np.byte_bounds` | Available under `np.lib.array_utils.byte_bounds` |
| `np.compare_chararrays` | Available as `np.char.compare_chararrays` |
| `np.format_parser` | Available as `np.rec.format_parser` |
| `np.alltrue` | Use `np.all` |
| `np.sometrue` | Use `np.any` |
| `np.trapz` | Use `np.trapezoid` |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Constants | 11 |
| Scalar Types | ~45 |
| DType Classes | 27 |
| Array Creation | ~40 |
| Array Manipulation | ~60 |
| Mathematical Functions | ~80 |
| Universal Functions | ~95 |
| Statistical Functions | ~50 |
| Window Functions | 5 |
| Linear Algebra (linalg) | ~35 |
| FFT | 18 |
| Random (distributions) | ~50 |
| Sorting/Searching | ~25 |
| Set Operations | 6 |
| String Operations (char) | ~55 |
| String Operations (strings) | ~45 |
| Record Arrays (rec) | 9 |
| Ctypes Interop | 6 |
| File I/O | ~10 |
| Testing | ~35 |
| Array API Aliases | 12 |
| **TOTAL PUBLIC APIs** | **~700+** |

---

*Document generated from NumPy 2.4.2 source code and type stubs.*
*Cross-verified against `numpy/__init__.py` and all submodule `__init__.pyi` files.*
