# Reduction Functions API Documentation

> **Draft prepared by Documentation Engineer**
> For inclusion in DocFX website and XML doc comments

---

## np.argmax

Returns the indices of the maximum values along an axis.

### Signatures

```csharp
// Flattened array - returns single index
public static int argmax(NDArray a)

// Along axis - returns array of indices
public static NDArray argmax(NDArray a, int axis)

// Instance methods
public int NDArray.argmax()
public NDArray NDArray.argmax(int axis)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | NDArray | Input array |
| `axis` | int | Axis along which to operate. By default (no axis), operates on flattened array. |

### Returns

- **Flattened**: `int` - Index of maximum value in the flattened array
- **With axis**: `NDArray` (dtype: int32) - Array of indices with shape equal to input shape with the specified axis removed

### NumPy Compatibility

| Behavior | NumPy | NumSharp |
|----------|-------|----------|
| NaN handling | First NaN wins (returns its index) | First NaN wins |
| Empty array | Raises ValueError | Throws ArgumentException |
| Multiple maxima | Returns index of first occurrence | Returns index of first occurrence |

### Edge Cases

- **Empty array**: Throws `ArgumentException("attempt to get argmax of an empty sequence")`
- **NaN values**: For float/double types, NaN is considered greater than any other value (NumPy behavior)
- **Boolean**: True > False, returns index of first True (or 0 if all False)

### Examples

```csharp
// Basic usage
var a = np.array(new[] { 3, 1, 4, 1, 5, 9, 2, 6 });
int idx = np.argmax(a);  // 5 (index of value 9)

// 2D array, flattened
var b = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
int idx2 = np.argmax(b);  // 3 (index of value 4 in flattened array)

// Along axis
var c = np.array(new int[,] { { 1, 5 }, { 3, 2 } });
var result0 = np.argmax(c, axis: 0);  // [1, 0] - indices of max in each column
var result1 = np.argmax(c, axis: 1);  // [1, 0] - indices of max in each row

// NaN handling
var d = np.array(new[] { 1.0, double.NaN, 3.0 });
int nanIdx = np.argmax(d);  // 1 (NaN wins)
```

### See Also

- `np.argmin` - Indices of minimum values
- `np.amax` - Maximum values
- `np.nanargmax` - Ignoring NaN values (not yet implemented)

### NumPy Reference

https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

---

## np.argmin

Returns the indices of the minimum values along an axis.

### Signatures

```csharp
// Flattened array - returns single index
public static int argmin(NDArray a)

// Along axis - returns array of indices
public static NDArray argmin(NDArray a, int axis)

// Instance methods
public int NDArray.argmin()
public NDArray NDArray.argmin(int axis)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | NDArray | Input array |
| `axis` | int | Axis along which to operate. By default (no axis), operates on flattened array. |

### Returns

- **Flattened**: `int` - Index of minimum value in the flattened array
- **With axis**: `NDArray` (dtype: int32) - Array of indices with shape equal to input shape with the specified axis removed

### NumPy Compatibility

| Behavior | NumPy | NumSharp |
|----------|-------|----------|
| NaN handling | First NaN wins (returns its index) | First NaN wins |
| Empty array | Raises ValueError | Throws ArgumentException |
| Multiple minima | Returns index of first occurrence | Returns index of first occurrence |

### Edge Cases

- **Empty array**: Throws `ArgumentException("attempt to get argmin of an empty sequence")`
- **NaN values**: For float/double types, NaN is considered less than any other value (NumPy behavior)
- **Boolean**: False < True, returns index of first False (or 0 if all True)

### Examples

```csharp
// Basic usage
var a = np.array(new[] { 3, 1, 4, 1, 5, 9, 2, 6 });
int idx = np.argmin(a);  // 1 (index of first occurrence of value 1)

// Along axis
var b = np.array(new int[,] { { 5, 2 }, { 1, 8 } });
var result0 = np.argmin(b, axis: 0);  // [1, 0] - indices of min in each column
var result1 = np.argmin(b, axis: 1);  // [1, 0] - indices of min in each row

// NaN handling
var c = np.array(new[] { 1.0, double.NaN, 0.5 });
int nanIdx = np.argmin(c);  // 1 (NaN wins)
```

### See Also

- `np.argmax` - Indices of maximum values
- `np.amin` - Minimum values
- `np.nanargmin` - Ignoring NaN values (not yet implemented)

### NumPy Reference

https://numpy.org/doc/stable/reference/generated/numpy.argmin.html

---

## np.all

Test whether all array elements along a given axis evaluate to True.

### Signatures

```csharp
// Entire array - returns bool
public static bool all(NDArray a)

// Along axis - returns NDArray<bool>
public static NDArray<bool> all(NDArray nd, int axis, bool keepdims = false)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` / `nd` | NDArray | Input array |
| `axis` | int | Axis along which to perform logical AND reduction |
| `keepdims` | bool | If True, reduced axes remain with size 1 |

### Returns

- **No axis**: `bool` - True if all elements are non-zero/True
- **With axis**: `NDArray<bool>` - Boolean array with reduced shape

### Truth Evaluation

Values are considered "True" (non-zero) based on type:
- **Boolean**: True = truthy, False = falsy
- **Numeric**: 0 = falsy, any non-zero = truthy
- **Floating-point**: 0.0 = falsy, any other value (including NaN) = truthy

### Examples

```csharp
// Basic usage
var a = np.array(new[] { true, true, true });
bool result1 = np.all(a);  // true

var b = np.array(new[] { 1, 2, 3, 0 });
bool result2 = np.all(b);  // false (0 is falsy)

// Along axis
var c = np.array(new int[,] { { 1, 0 }, { 1, 1 } });
var result3 = np.all(c, axis: 0);  // [true, false]
var result4 = np.all(c, axis: 1);  // [false, true]

// With keepdims
var d = np.array(new int[,] { { 1, 1 }, { 1, 1 } });
var result5 = np.all(d, axis: 0, keepdims: true);  // [[true, true]] shape (1, 2)
```

### Performance Notes

- Uses SIMD acceleration for contiguous arrays via `ILKernelGenerator.AllSimdHelper<T>()`
- Implements early-exit optimization: returns False immediately upon finding first zero

### See Also

- `np.any` - Test if any element is True
- `np.allclose` - Element-wise comparison with tolerance (currently dead code)

### NumPy Reference

https://numpy.org/doc/stable/reference/generated/numpy.all.html

---

## np.any

Test whether any array element along a given axis evaluates to True.

### Signatures

```csharp
// Entire array - returns bool
public static bool any(NDArray a)

// Along axis - returns NDArray<bool>
public static NDArray<bool> any(NDArray nd, int axis, bool keepdims = false)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` / `nd` | NDArray | Input array |
| `axis` | int | Axis along which to perform logical OR reduction |
| `keepdims` | bool | If True, reduced axes remain with size 1 |

### Returns

- **No axis**: `bool` - True if any element is non-zero/True
- **With axis**: `NDArray<bool>` - Boolean array with reduced shape

### Truth Evaluation

Same as `np.all`: non-zero values are truthy, zero/False is falsy.

### Examples

```csharp
// Basic usage
var a = np.array(new[] { false, false, true });
bool result1 = np.any(a);  // true

var b = np.array(new[] { 0, 0, 0 });
bool result2 = np.any(b);  // false

// Along axis
var c = np.array(new int[,] { { 0, 1 }, { 0, 0 } });
var result3 = np.any(c, axis: 0);  // [false, true]
var result4 = np.any(c, axis: 1);  // [true, false]

// With keepdims
var result5 = np.any(c, axis: 1, keepdims: true);  // [[true], [false]] shape (2, 1)
```

### Performance Notes

- Uses SIMD acceleration for contiguous arrays via `ILKernelGenerator.AnySimdHelper<T>()`
- Implements early-exit optimization: returns True immediately upon finding first non-zero

### See Also

- `np.all` - Test if all elements are True
- `np.nonzero` - Find indices of non-zero elements

### NumPy Reference

https://numpy.org/doc/stable/reference/generated/numpy.any.html

---

## np.prod

Return the product of array elements over a given axis.

### Signatures

```csharp
// Static method
public static NDArray prod(in NDArray a, int? axis = null, Type dtype = null, bool keepdims = false)

// Instance method
public NDArray NDArray.prod(int? axis = null, Type dtype = null, bool keepdims = false)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | NDArray | Input array |
| `axis` | int? | Axis along which product is computed. Default (null) computes product of all elements. |
| `dtype` | Type | Return type. If null, uses input dtype (with integer promotion rules). |
| `keepdims` | bool | If True, reduced axes remain with size 1 |

### Returns

`NDArray` - Product of elements. Shape depends on axis and keepdims parameters.

### Type Promotion

When `dtype` is null:
- Integer types smaller than platform int are promoted to int
- Signed integers stay signed, unsigned stay unsigned
- Float/double remain unchanged

### Examples

```csharp
// Basic usage
var a = np.array(new[] { 1, 2, 3, 4 });
var result1 = np.prod(a);  // 24

// 2D array
var b = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
var result2 = np.prod(b);           // 24 (all elements)
var result3 = np.prod(b, axis: 0);  // [3, 8] (column products)
var result4 = np.prod(b, axis: 1);  // [2, 12] (row products)

// With keepdims
var result5 = np.prod(b, axis: 0, keepdims: true);  // [[3, 8]] shape (1, 2)

// With dtype
var c = np.array(new byte[] { 100, 100 });
var result6 = np.prod(c, dtype: typeof(long));  // 10000L (avoids overflow)
```

### Overflow Considerations

Integer multiplication can overflow without warning. For large products, specify a wider dtype:

```csharp
var a = np.array(new int[] { 1000, 1000, 1000 });
var result = np.prod(a, dtype: typeof(long));  // Use long to avoid int32 overflow
```

### See Also

- `np.sum` - Sum of elements
- `np.cumprod` - Cumulative product (not yet implemented)

### NumPy Reference

https://numpy.org/doc/stable/reference/generated/numpy.prod.html

---

## Supported Data Types

All reduction functions support the 12 NumSharp data types:

| Type | Notes |
|------|-------|
| bool | True=1, False=0 for comparisons |
| byte | Unsigned 8-bit |
| short, ushort | 16-bit signed/unsigned |
| int, uint | 32-bit signed/unsigned |
| long, ulong | 64-bit signed/unsigned |
| char | Treated as ushort for comparisons |
| float | 32-bit IEEE 754, NaN handling for argmax/argmin |
| double | 64-bit IEEE 754, NaN handling for argmax/argmin |
| decimal | 128-bit, no SIMD acceleration |

---
