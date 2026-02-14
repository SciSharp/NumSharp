# Broadcasting

Broadcasting allows arithmetic operations between arrays of different shapes. When you add a `(3, 4)` matrix to a `(4,)` vector, NumSharp automatically "broadcasts" the vector across each row—no explicit loops or copying required.

---

## How Broadcasting Works

NumSharp follows NumPy's broadcasting rules exactly:

1. **Shapes align from the right.** If arrays have different numbers of dimensions, prepend 1s to the shorter shape.

2. **Dimensions must be equal or 1.** For each dimension, sizes must match OR one must be 1.

3. **Size-1 dimensions stretch.** A dimension of size 1 expands to match the other array's size in that dimension.

```csharp
// (3, 4) + (4,) → (3, 4) + (1, 4) → (3, 4)
var matrix = np.ones((3, 4));
var row = np.array(new[] {1, 2, 3, 4});
var result = matrix + row;  // Shape: (3, 4)
```

Broadcasting creates **views, not copies**. The stretched array doesn't allocate new memory—it uses stride tricks to repeat values virtually.

---

## Shape Compatibility

### Compatible Shapes

| Shape A | Shape B | Result | Notes |
|---------|---------|--------|-------|
| `(5,)` | `(5,)` | `(5,)` | Same shape |
| `(5,)` | `()` | `(5,)` | Scalar broadcasts to any shape |
| `(3, 4)` | `(4,)` | `(3, 4)` | Row vector broadcasts across rows |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | Column vector broadcasts across columns |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | Both arrays stretch |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | Lower-dimensional array broadcasts |
| `(8, 1, 6, 1)` | `(7, 1, 5)` | `(8, 7, 6, 5)` | Complex case with multiple stretch dimensions |

### Incompatible Shapes

| Shape A | Shape B | Error |
|---------|---------|-------|
| `(3,)` | `(4,)` | 3 ≠ 4, neither is 1 |
| `(3, 4)` | `(3,)` | Trailing dimensions 4 ≠ 3 |
| `(2, 3)` | `(3, 2)` | No valid alignment |

Incompatible shapes throw `IncorrectShapeException`.

---

## Broadcasting Functions

### `np.broadcast_to(array, shape)`

Broadcasts an array to a specific shape. Returns a read-only view.

```csharp
var a = np.array(new[] {1, 2, 3});
var b = np.broadcast_to(a, (4, 3));
// b.shape: (4, 3)
// b[0]: [1, 2, 3]
// b[1]: [1, 2, 3]  (same data, not copied)
```

**Constraints:** The source shape must be unilaterally broadcastable to the target. You can only stretch dimensions that are size 1:

```csharp
np.broadcast_to(np.ones((2,)), (3, 3));   // Error: can't stretch 2 to 3
np.broadcast_to(np.ones((1, 3)), (4, 3)); // OK: stretches 1 to 4
```

### `np.broadcast_arrays(array1, array2, ...)`

Broadcasts multiple arrays against each other, returning views with a common shape.

```csharp
var a = np.array(new[] {1, 2, 3});        // (3,)
var b = np.array(new[,] {{1}, {2}});      // (2, 1)

var (a_bc, b_bc) = np.broadcast_arrays(a, b);
// Both now (2, 3)
```

Also available as:
```csharp
NDArray[] results = np.broadcast_arrays(arr1, arr2, arr3);
```

### Implicit Broadcasting

All arithmetic operators broadcast automatically:

```csharp
var a = np.ones((3, 4));
var b = np.array(new[] {1, 2, 3, 4});

a + b;   // (3, 4)
a - b;   // (3, 4)
a * b;   // (3, 4)
a / b;   // (3, 4)
```

---

## Memory Behavior

Broadcasted arrays are **views** that share memory with the original:

```csharp
var small = np.array(new[] {1, 2, 3});           // 3 elements
var big = np.broadcast_to(small, (1000000, 3));  // Appears as 3M elements

// big.size == 3_000_000
// Actual memory: still just 3 elements
// big.Shape.IsBroadcasted == true
```

**Important:** Broadcasted arrays should be treated as read-only. Writing to a broadcasted position affects all positions that share that memory. If you need to modify a broadcasted array, copy it first:

```csharp
var writable = big.copy();  // Allocates full 3M elements
```

---

## Implementation Details

NumSharp implements broadcasting through stride manipulation. When a dimension is broadcast:

- The **shape** shows the expanded size
- The **stride** for that dimension is set to 0

A stride of 0 means the index doesn't advance in memory—the same element is read repeatedly.

```csharp
var a = np.array(new[] {1, 2, 3});
var b = np.broadcast_to(a, (4, 3));

// b's internal representation:
// Shape:   (4, 3)
// Strides: (0, 1)  ← stride 0 in first dimension
```

This is tracked via `Shape.IsBroadcasted` and `BroadcastInfo`.

---

## Common Patterns

### Centering Data (subtract mean)

```csharp
var data = np.random.rand(100, 5);           // 100 samples, 5 features
var mean = np.mean(data, axis: 0);           // (5,)
var centered = data - mean;                  // (100, 5) - broadcasts
```

### Normalizing (divide by std)

```csharp
var std = np.std(data, axis: 0);             // (5,)
var normalized = centered / std;             // (100, 5)
```

### Outer Product

```csharp
var row = np.array(new[] {1, 2, 3});         // (3,)
var col = np.array(new[,] {{10}, {20}});     // (2, 1)
var outer = row * col;                       // (2, 3)
```

### Batch Operations

```csharp
var batch = np.random.rand(32, 28, 28);      // 32 images
var mean_image = np.mean(batch, axis: 0);    // (28, 28)
var normalized = batch - mean_image;         // (32, 28, 28)
```

---

## Troubleshooting

### "shape mismatch: objects cannot be broadcast"

Shapes don't follow broadcasting rules. Check alignment:

```csharp
// Wrong
var a = np.ones((3, 4));
var b = np.ones((3,));    // Trailing dim 4 ≠ 3
var c = a + b;            // Error

// Fix: reshape to column vector
var c = a + b.reshape(3, 1);  // Now (3, 4) + (3, 1) works
```

### Unexpected Output Shape

If you get a larger shape than expected, you may have accidentally broadcast:

```csharp
var a = np.ones((10, 1));
var b = np.ones((1, 10));
var c = a + b;  // (10, 10) — both stretched!
```

### Row vs Column Vector

A 1-D array `(n,)` broadcasts as a **row** `(1, n)`, not a column:

```csharp
var vec = np.array(new[] {1, 2, 3});  // (3,) — not (1, 3) or (3, 1)

// To broadcast as column:
var col = vec.reshape(3, 1);          // (3, 1)
// or
var col = vec[np.newaxis].T;          // (3, 1)
```

---

## API Reference

| Function | Description |
|----------|-------------|
| `np.broadcast_to(arr, shape)` | Broadcast array to specific shape (returns view) |
| `np.broadcast_arrays(a, b)` | Broadcast two arrays to common shape (returns tuple) |
| `np.broadcast_arrays(params NDArray[])` | Broadcast multiple arrays (returns array) |

| Property | Description |
|----------|-------------|
| `Shape.IsBroadcasted` | True if shape has broadcast strides (stride 0) |
| `BroadcastInfo` | Internal metadata for broadcast tracking |

| Exception | When |
|-----------|------|
| `IncorrectShapeException` | Shapes cannot be broadcast together |
