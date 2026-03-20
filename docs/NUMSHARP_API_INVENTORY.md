# NumSharp Current API Inventory

**Generated:** 2026-03-20 (Updated)
**Source:** `src/NumSharp.Core/`
**NumSharp Version:** 0.41.x (npalign branch)

## Summary

| Category | Count |
|----------|-------|
| **Total np.* APIs** | 142 |
| **Working** | 118 |
| **Partial** | 12 |
| **Broken/Stub** | 12 |

---

## Array Creation (`Creation/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.array` | `np.array.cs` | Working | Multiple overloads for 1D-16D arrays, jagged arrays, IEnumerable |
| `np.ndarray` | `np.array_manipulation.cs` | Working | Low-level array creation with optional buffer |
| `np.zeros` | `np.zeros.cs` | Working | All dtypes supported |
| `np.zeros_like` | `np.zeros_like.cs` | Working | |
| `np.ones` | `np.ones.cs` | Working | All dtypes supported |
| `np.ones_like` | `np.ones_like.cs` | Working | |
| `np.empty` | `np.empty.cs` | Working | Uninitialized memory |
| `np.empty_like` | `np.empty_like.cs` | Working | |
| `np.full` | `np.full.cs` | Working | All dtypes supported; TODO: NEP50 int promotion |
| `np.full_like` | `np.full_like.cs` | Working | |
| `np.arange` | `np.arange.cs` | Partial | int returns int32 (NumPy 2.x returns int64 - BUG-21) |
| `np.linspace` | `np.linspace.cs` | Working | Returns float64 by default (NumPy-aligned) |
| `np.eye` | `np.eye.cs` | Working | Supports k offset |
| `np.identity` | `np.eye.cs` | Working | Calls eye(n) |
| `np.meshgrid` | `np.meshgrid.cs` | Partial | Only 2D, missing N-D support |
| `np.mgrid` | `np.mgrid.cs` | Partial | TODO: implement mgrid overloads |
| `np.copy` | `np.copy.cs` | Partial | TODO: order support |
| `np.asarray` | `np.asarray.cs` | Working | |
| `np.asanyarray` | `np.asanyarray.cs` | Working | |
| `np.frombuffer` | `np.frombuffer.cs` | Partial | TODO: all types (limited dtype support) |
| `np.dtype` | `np.dtype.cs` | Partial | TODO: parse dtype strings |

## Stacking & Joining (`Creation/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.concatenate` | `np.concatenate.cs` | Working | Tuple overloads for 2-9 arrays |
| `np.stack` | `np.stack.cs` | Working | |
| `np.hstack` | `np.hstack.cs` | Working | |
| `np.vstack` | `np.vstack.cs` | Working | |
| `np.dstack` | `np.dstack.cs` | Working | |

## Splitting (`Manipulation/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.split` | `np.split.cs` | Working | Integer and indices overloads |
| `np.array_split` | `np.split.cs` | Working | Allows unequal division |
| `np.hsplit` | `np.hsplit.cs` | Working | Splits along axis 1 (or 0 for 1D) |
| `np.vsplit` | `np.vsplit.cs` | Working | Splits along axis 0 |
| `np.dsplit` | `np.dsplit.cs` | Working | Splits along axis 2 |

## Broadcasting (`Creation/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.broadcast` | `np.broadcast.cs` | Working | |
| `np.broadcast_to` | `np.broadcast_to.cs` | Working | Multiple overloads |
| `np.broadcast_arrays` | `np.broadcast_arrays.cs` | Working | |
| `np.are_broadcastable` | `np.are_broadcastable.cs` | Working | |

---

## Mathematical Functions (`Math/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.add` | `np.math.cs` | Working | Via TensorEngine |
| `np.subtract` | `np.math.cs` | Working | Via TensorEngine |
| `np.multiply` | `np.math.cs` | Working | Via TensorEngine |
| `np.divide` | `np.math.cs` | Working | Via TensorEngine |
| `np.true_divide` | `np.math.cs` | Working | Same as divide |
| `np.mod` | `np.math.cs` | Working | |
| `np.sum` | `np.sum.cs` | Working | Multiple overloads with axis/keepdims/dtype |
| `np.prod` | `np.math.cs` | Working | |
| `np.cumsum` | `np.cumsum.cs` | Working | Via TensorEngine |
| `np.cumprod` | `np.cumprod.cs` | Working | Via TensorEngine |
| `np.power` | `np.power.cs` | Working | Scalar and array exponents |
| `np.square` | `np.power.cs` | Working | |
| `np.sqrt` | `np.sqrt.cs` | Working | |
| `np.cbrt` | `np.cbrt.cs` | Working | |
| `np.abs` / `np.absolute` | `np.absolute.cs` | Working | Preserves int dtype |
| `np.sign` | `np.sign.cs` | Working | |
| `np.floor` | `np.floor.cs` | Working | |
| `np.ceil` | `np.ceil.cs` | Working | |
| `np.trunc` | `np.trunc.cs` | Working | |
| `np.around` / `np.round` | `np.round.cs` | Working | |
| `np.clip` | `np.clip.cs` | Working | NDArray min/max |
| `np.modf` | `np.modf.cs` | Working | |
| `np.maximum` | `np.maximum.cs` | Working | Element-wise |
| `np.minimum` | `np.minimum.cs` | Working | Element-wise |
| `np.floor_divide` | `np.floor_divide.cs` | Working | |
| `np.positive` | `np.math.cs` | Working | Identity function |
| `np.negative` | `np.math.cs` | Working | |
| `np.convolve` | `np.math.cs` | Working | |
| `np.reciprocal` | `np.reciprocal.cs` | Working | |
| `np.invert` | `np.invert.cs` | Working | Bitwise NOT |
| `np.bitwise_not` | `np.invert.cs` | Working | Alias for invert |
| `np.left_shift` | `np.left_shift.cs` | Working | |
| `np.right_shift` | `np.right_shift.cs` | Working | |
| `np.deg2rad` | `np.deg2rad.cs` | Working | |
| `np.rad2deg` | `np.rad2deg.cs` | Working | |
| `np.nansum` | `np.nansum.cs` | Working | |
| `np.nanprod` | `np.nanprod.cs` | Working | |

## Trigonometric Functions (`Math/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.sin` | `np.sin.cs` | Working | |
| `np.cos` | `np.cos.cs` | Working | |
| `np.tan` | `np.tan.cs` | Working | |

## Exponential & Logarithmic (`Math/`, `Statistics/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.exp` | `Statistics/np.exp.cs` | Working | |
| `np.exp2` | `Statistics/np.exp.cs` | Working | |
| `np.expm1` | `Statistics/np.exp.cs` | Working | |
| `np.log` | `np.log.cs` | Working | |
| `np.log2` | `np.log.cs` | Working | |
| `np.log10` | `np.log.cs` | Working | |
| `np.log1p` | `np.log.cs` | Working | |

---

## Statistics (`Statistics/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.mean` | `np.mean.cs` | Working | Multiple overloads with axis/dtype/keepdims |
| `np.std` | `np.std.cs` | Working | Supports ddof |
| `np.var` | `np.var.cs` | Working | Supports ddof |
| `np.nanmean` | `np.nanmean.cs` | Working | |
| `np.nanstd` | `np.nanstd.cs` | Working | |
| `np.nanvar` | `np.nanvar.cs` | Working | |

---

## Sorting, Searching & Counting (`Sorting_Searching_Counting/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.amax` / `np.max` | `np.amax.cs` | Working | With axis/keepdims support |
| `np.amin` / `np.min` | `np.min.cs` | Working | With axis/keepdims support |
| `np.argmax` | `np.argmax.cs` | Working | Scalar or axis-based |
| `np.argmin` | `np.argmax.cs` | Working | Scalar or axis-based |
| `np.argsort` | `np.argsort.cs` | Working | |
| `np.searchsorted` | `np.searchsorted.cs` | Partial | TODO: no multidimensional a support |
| `np.nanmax` | `np.nanmax.cs` | Working | |
| `np.nanmin` | `np.nanmin.cs` | Working | |

---

## Logic Functions (`Logic/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.all` | `np.all.cs` | Working | Global and axis-based |
| `np.any` | `np.any.cs` | Working | Global and axis-based |
| `np.allclose` | `np.allclose.cs` | **Broken** | Depends on `isclose` which returns null |
| `np.array_equal` | `np.array_equal.cs` | Working | |
| `np.isscalar` | `np.is.cs` | Working | |
| `np.isnan` | `np.is.cs` | **Broken** | `TensorEngine.IsNan` returns null |
| `np.isfinite` | `np.is.cs` | **Broken** | `TensorEngine.IsFinite` returns null |
| `np.isinf` | `np.is.cs` | Working | |
| `np.isclose` | `np.is.cs` | **Broken** | `TensorEngine.IsClose` returns null |
| `np.find_common_type` | `np.find_common_type.cs` | Working | |

## Comparison Functions (`Logic/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.equal` | `np.comparison.cs` | Working | Via operators |
| `np.not_equal` | `np.comparison.cs` | Working | Via operators |
| `np.greater` | `np.comparison.cs` | Working | Via operators |
| `np.greater_equal` | `np.comparison.cs` | Working | Via operators |
| `np.less` | `np.comparison.cs` | Working | Via operators |
| `np.less_equal` | `np.comparison.cs` | Working | Via operators |

## Logical Operations (`Logic/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.logical_and` | `np.logical.cs` | Working | |
| `np.logical_or` | `np.logical.cs` | Working | |
| `np.logical_not` | `np.logical.cs` | Working | |
| `np.logical_xor` | `np.logical.cs` | Working | |

---

## Shape Manipulation (`Manipulation/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.reshape` | `np.reshape.cs` | Working | |
| `np.transpose` | `np.transpose.cs` | Working | |
| `np.ravel` | `np.ravel.cs` | Working | |
| `np.squeeze` | `np.squeeze.cs` | Partial | TODO: what happens if slice? |
| `np.expand_dims` | `np.expand_dims.cs` | Working | |
| `np.swapaxes` | `np.swapaxes.cs` | Working | |
| `np.moveaxis` | `np.moveaxis.cs` | Working | |
| `np.rollaxis` | `np.rollaxis.cs` | Working | |
| `np.roll` | `np.roll.cs` | Working | All dtypes, with/without axis |
| `np.atleast_1d` | `np.atleastd.cs` | Working | |
| `np.atleast_2d` | `np.atleastd.cs` | Working | |
| `np.atleast_3d` | `np.atleastd.cs` | Working | |
| `np.unique` | `np.unique.cs` | Working | |
| `np.repeat` | `np.repeat.cs` | Working | |
| `np.copyto` | `np.copyto.cs` | Working | |
| `np.asscalar` | `np.asscalar.cs` | Partial | Deprecated in NumPy |

---

## Linear Algebra (`LinearAlgebra/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.dot` | `np.dot.cs` | Working | Via TensorEngine |
| `np.matmul` | `np.matmul.cs` | Working | Via TensorEngine |
| `np.outer` | `np.outer.cs` | Working | |
| `np.linalg.norm` | `np.linalg.norm.cs` | **Broken** | Declared `private static` - not accessible |
| `nd.inv()` | `NdArray.Inv.cs` | **Stub** | Returns `null` |
| `nd.qr()` | `NdArray.QR.cs` | **Stub** | Returns `default` |
| `nd.svd()` | `NdArray.SVD.cs` | **Stub** | Returns `default` |
| `nd.lstsq()` | `NdArray.LstSq.cs` | **Stub** | Named `lstqr`, returns `null` |
| `nd.multi_dot()` | `NdArray.multi_dot.cs` | **Stub** | Returns `null` |
| `nd.matrix_power()` | `NDArray.matrix_power.cs` | Working | |

---

## Indexing (`Indexing/`, `Selection/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.nonzero` | `np.nonzero.cs` | Working | Via TensorEngine |
| Integer/slice indexing | `NDArray.Indexing.cs` | Working | |
| Boolean masking (get) | `NDArray.Indexing.Masking.cs` | Working | |
| Boolean masking (set) | `NDArray.Indexing.Masking.cs` | **Broken** | Setter throws `NotImplementedException` |
| Fancy indexing | `NDArray.Indexing.Selection.cs` | Working | NDArray<int> indices |

---

## Random Sampling (`RandomSampling/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.random.seed` | `np.random.cs` | Working | |
| `np.random.RandomState` | `np.random.cs` | Working | |
| `np.random.get_state` | `np.random.cs` | Working | |
| `np.random.set_state` | `np.random.cs` | Working | |
| `np.random.rand` | `np.random.rand.cs` | Working | |
| `np.random.randn` | `np.random.randn.cs` | Working | |
| `np.random.randint` | `np.random.randint.cs` | Working | |
| `np.random.uniform` | `np.random.uniform.cs` | Working | |
| `np.random.choice` | `np.random.choice.cs` | Working | |
| `np.random.shuffle` | `np.random.shuffle.cs` | Working | |
| `np.random.permutation` | `np.random.permutation.cs` | Working | |
| `np.random.beta` | `np.random.beta.cs` | Working | |
| `np.random.binomial` | `np.random.binomial.cs` | Working | |
| `np.random.gamma` | `np.random.gamma.cs` | Working | |
| `np.random.poisson` | `np.random.poisson.cs` | Working | |
| `np.random.exponential` | `np.random.exponential.cs` | Working | |
| `np.random.geometric` | `np.random.geometric.cs` | Working | |
| `np.random.lognormal` | `np.random.lognormal.cs` | Working | |
| `np.random.chisquare` | `np.random.chisquare.cs` | Working | |
| `np.random.bernoulli` | `np.random.bernoulli.cs` | Working | |
| `np.random.laplace` | `np.random.laplace.cs` | Working | Newly implemented |
| `np.random.triangular` | `np.random.triangular.cs` | Working | Newly implemented |

---

## File I/O (`APIs/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.save` | `np.save.cs` | Working | .npy format |
| `np.load` | `np.load.cs` | Working | .npy and .npz formats |
| `np.fromfile` | `np.fromfile.cs` | Working | Binary file reading |
| `nd.tofile()` | `np.tofile.cs` | Partial | TODO: sliced data support |
| `np.Save_Npz` | `np.save.cs` | Working | .npz format |
| `np.Load_Npz` | `np.load.cs` | Working | .npz format |

---

## Other APIs (`APIs/`)

| Function | File | Status | Notes |
|----------|------|--------|-------|
| `np.size` | `np.size.cs` | Working | |
| `np.count_nonzero` | `np.count_nonzero.cs` | Working | Global and axis-based |

---

## Operators (`Operations/Elementwise/`)

### Arithmetic Operators

| Operator | File | Status | Notes |
|----------|------|--------|-------|
| `+` (add) | `NDArray.Primitive.cs` | Working | All 12 dtypes, NDArray-NDArray and NDArray-scalar |
| `-` (subtract) | `NDArray.Primitive.cs` | Working | All 12 dtypes |
| `*` (multiply) | `NDArray.Primitive.cs` | Working | All 12 dtypes |
| `/` (divide) | `NDArray.Primitive.cs` | Working | All 12 dtypes |
| `%` (mod) | `NDArray.Primitive.cs` | Working | All 12 dtypes |
| unary `-` (negate) | `NDArray.Primitive.cs` | Working | |
| unary `+` | `NDArray.Primitive.cs` | Working | Returns copy |

### Comparison Operators

| Operator | File | Status | Notes |
|----------|------|--------|-------|
| `==` | `NDArray.Equals.cs` | Working | Returns NDArray<bool>, broadcasting |
| `!=` | `NDArray.NotEquals.cs` | Working | Returns NDArray<bool>, broadcasting |
| `>` | `NDArray.Greater.cs` | Working | Returns NDArray<bool>, broadcasting |
| `>=` | `NDArray.Greater.cs` | Working | Returns NDArray<bool>, broadcasting |
| `<` | `NDArray.Lower.cs` | Working | Returns NDArray<bool>, broadcasting |
| `<=` | `NDArray.Lower.cs` | Working | Returns NDArray<bool>, broadcasting |

### Bitwise Operators

| Operator | File | Status | Notes |
|----------|------|--------|-------|
| `&` (AND) | `NDArray.AND.cs` | Working | Boolean and integer types |
| `\|` (OR) | `NDArray.OR.cs` | Working | Boolean and integer types |

---

## Constants & Types (`APIs/np.cs`)

| Constant | Value | Notes |
|----------|-------|-------|
| `np.nan` | `double.NaN` | Also `np.NaN`, `np.NAN` |
| `np.pi` | `Math.PI` | |
| `np.e` | `Math.E` | |
| `np.euler_gamma` | `0.5772...` | Euler-Mascheroni constant |
| `np.inf` | `double.PositiveInfinity` | Also `np.Inf`, `np.infty`, `np.Infinity` |
| `np.NINF` | `double.NegativeInfinity` | |
| `np.PINF` | `double.PositiveInfinity` | |
| `np.newaxis` | `Slice` | For dimension expansion |

| Type Alias | C# Type |
|------------|---------|
| `np.bool_` / `np.bool8` | `bool` |
| `np.byte` / `np.uint8` / `np.ubyte` | `byte` |
| `np.int16` | `short` |
| `np.uint16` | `ushort` |
| `np.int32` | `int` |
| `np.uint32` | `uint` |
| `np.int_` / `np.int64` / `np.int0` | `long` |
| `np.uint64` / `np.uint0` / `np.uint` | `ulong` |
| `np.intp` | `nint` (native int) |
| `np.uintp` | `nuint` (native uint) |
| `np.float32` | `float` |
| `np.float_` / `np.float64` / `np.double` | `double` |
| `np.complex_` / `np.complex128` / `np.complex64` | `Complex` |
| `np.decimal` | `decimal` |
| `np.char` | `char` |

---

## NDArray Instance Methods

### Working

| Method | File | Notes |
|--------|------|-------|
| `nd.reshape()` | `Creation/NdArray.ReShape.cs` | |
| `nd.ravel()` | `Manipulation/NDArray.ravel.cs` | |
| `nd.flatten()` | `Manipulation/NDArray.flatten.cs` | |
| `nd.T` (transpose) | `Manipulation/NdArray.Transpose.cs` | |
| `nd.swapaxes()` | `Manipulation/NdArray.swapaxes.cs` | |
| `nd.sum()` | `Math/NDArray.sum.cs` | |
| `nd.prod()` | `Math/NDArray.prod.cs` | |
| `nd.cumsum()` | `Math/NDArray.cumsum.cs` | |
| `nd.mean()` | `Statistics/NDArray.mean.cs` | |
| `nd.std()` | `Statistics/NDArray.std.cs` | |
| `nd.var()` | `Statistics/NDArray.var.cs` | |
| `nd.amax()` | `Statistics/NDArray.amax.cs` | |
| `nd.amin()` | `Statistics/NDArray.amin.cs` | |
| `nd.argmax()` | `Statistics/NDArray.argmax.cs` | |
| `nd.argmin()` | `Statistics/NDArray.argmin.cs` | |
| `nd.argsort()` | `Sorting_Searching_Counting/ndarray.argsort.cs` | |
| `nd.dot()` | `LinearAlgebra/NDArray.dot.cs` | |
| `nd.unique()` | `Manipulation/NDArray.unique.cs` | |
| `nd.roll()` | `Manipulation/NDArray.roll.cs` | |
| `nd.copy()` | `Creation/NDArray.Copy.cs` | |
| `nd.Clone()` | `Backends/NDArray.cs` | ICloneable implementation |
| `nd.negative()` | `Math/NDArray.negative.cs` | |
| `nd.positive()` | `Math/NDArray.positive.cs` | |
| `nd.convolve()` | `Math/NdArray.Convolve.cs` | |
| `nd.tofile()` | `APIs/np.tofile.cs` | Partial (TODO: sliced data) |
| `nd.astype()` | `Backends/NDArray.cs` | Type/NPTypeCode overloads |
| `nd.view()` | `Backends/NDArray.cs` | TODO: unsafe reinterpret for dtype change |
| `nd.array_equal()` | `Operations/Elementwise/NDArray.Equals.cs` | |
| `nd.itemset()` | `Manipulation/NDArray.itemset.cs` | |

### Stub/Broken

| Method | File | Issue |
|--------|------|-------|
| `nd.delete()` | `Manipulation/NdArray.delete.cs` | Returns `null` |
| `nd.inv()` | `LinearAlgebra/NdArray.Inv.cs` | Returns `null` |
| `nd.qr()` | `LinearAlgebra/NdArray.QR.cs` | Returns `default` |
| `nd.svd()` | `LinearAlgebra/NdArray.SVD.cs` | Returns `default` |
| `nd.lstsq()` | `LinearAlgebra/NdArray.LstSq.cs` | Returns `null` |
| `nd.multi_dot()` | `LinearAlgebra/NdArray.multi_dot.cs` | Returns `null` |

---

## Missing Functions (Not Implemented)

These NumPy functions are commonly used but **not implemented** in NumSharp:

| Category | Functions |
|----------|-----------|
| Sorting | `np.sort`, `np.partition`, `np.argpartition` |
| Selection | `np.where`, `np.select`, `np.choose` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.tile`, `np.pad` |
| Diagonal | `np.diag`, `np.diagonal`, `np.trace`, `np.tril`, `np.triu` |
| Cumulative | `np.diff`, `np.gradient`, `np.ediff1d` |
| Set Operations | `np.intersect1d`, `np.union1d`, `np.setdiff1d`, `np.setxor1d`, `np.in1d` |
| Bitwise Functions | `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor` (operators work, functions missing) |
| Random | `np.random.normal` (use randn instead) |
| String Operations | All `np.char.*` functions |
| Structured Arrays | `np.dtype` with field names |
| FFT | All `np.fft.*` functions |
| Polynomials | All `np.poly*` functions |

---

## Known Behavioral Differences from NumPy 2.x

| Issue | NumSharp Behavior | NumPy 2.x Behavior |
|-------|-------------------|-------------------|
| `np.arange(int)` dtype | Returns `int32` | Returns `int64` (NEP50) |
| `np.full(int)` dtype | Preserves int32 | Promotes to int64 (NEP50) |
| `np.sum(int32)` dtype | Returns `int64` | Returns `int64` (aligned) |
| Boolean mask setter | Throws `NotImplementedException` | Works |
| `np.meshgrid` | Only 2D | N-D supported |
| `np.frombuffer` | Limited dtypes | All dtypes |
| `nd.view()` with dtype | Casts (copies) | Reinterprets memory (no copy) |
| F-order | Accepted but ignored | Fully supported |

---

## TODO Comments Found (Partial Implementations)

| File | Issue |
|------|-------|
| `np.arange.cs:309` | NumPy 2.x returns int64 for integer arange (BUG-21) |
| `np.full.cs:48,62` | NumPy 2.x promotes int32 to int64 (NEP50) |
| `np.tofile.cs:16` | Support for sliced data |
| `np.dtype.cs:178` | Parse dtype strings |
| `np.mgrid.cs:8` | Implement mgrid overloads |
| `np.copy.cs:12` | Order support |
| `np.searchsorted.cs:42` | No multidimensional a support |
| `np.frombuffer.cs:10` | All types |
| `np.squeeze.cs:51` | What happens if slice? |
| `NDArray.cs:521` | view() should reinterpret, not cast |

---

## Summary by Status

### Broken/Stub APIs (12)

1. `np.allclose` - Depends on broken `isclose`
2. `np.isnan` - TensorEngine returns null
3. `np.isfinite` - TensorEngine returns null
4. `np.isclose` - TensorEngine returns null
5. `np.linalg.norm` - Private method, inaccessible
6. `nd.inv()` - Returns null
7. `nd.qr()` - Returns default
8. `nd.svd()` - Returns default
9. `nd.lstsq()` - Returns null
10. `nd.multi_dot()` - Returns null
11. `nd.delete()` - Returns null
12. Boolean mask setter - Throws NotImplementedException

### Partial APIs (12)

1. `np.arange(int)` - Returns int32 (NumPy returns int64)
2. `np.full(int)` - Preserves int32 (NumPy promotes to int64)
3. `np.meshgrid` - Only 2D
4. `np.mgrid` - TODO: implement overloads
5. `np.copy` - TODO: order support
6. `np.frombuffer` - Limited dtypes
7. `np.dtype` - TODO: parse strings
8. `np.searchsorted` - No multidimensional support
9. `np.squeeze` - TODO: slice handling
10. `np.asscalar` - Deprecated in NumPy
11. `nd.tofile()` - TODO: sliced data
12. `nd.view()` - Casts instead of reinterprets

### Working APIs (118)

All other APIs listed in this document are working as expected.

---

## Revision History

- **2026-03-20 (Updated)**: Added 15 APIs missed in initial audit:
  - Split functions: `np.split`, `np.array_split`, `np.hsplit`, `np.vsplit`, `np.dsplit`
  - Random functions: `np.random.laplace`, `np.random.triangular`
  - Bitwise: `np.bitwise_not`
  - NDArray methods: `nd.astype()`, `nd.view()`, `nd.Clone()`, `nd.array_equal()`, `nd.itemset()`
  - Operators section with all arithmetic, comparison, and bitwise operators
  - TODO comments section documenting partial implementations
  - Updated summary counts and status corrections
