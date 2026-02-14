# Broadcasting

If you've ever wanted to add a single number to every element of an array, or subtract a row from every row of a matrix, you've wanted broadcasting. It's one of NumPy's most powerful features, and understanding it will change how you think about array operations.

---

## The Problem Broadcasting Solves

Let's say you have a matrix of test scores for 100 students across 5 subjects, and you want to subtract the class average from each student's scores. In a traditional programming language, you'd write nested loops:

```csharp
// The painful way (don't do this)
for (int student = 0; student < 100; student++)
{
    for (int subject = 0; subject < 5; subject++)
    {
        scores[student, subject] -= average[subject];
    }
}
```

With broadcasting, it's one line:

```csharp
var centered = scores - average;  // Just works!
```

But wait—`scores` is shape `(100, 5)` and `average` is shape `(5,)`. How does NumSharp know what to do? That's broadcasting.

---

## The Core Idea

Broadcasting is NumPy's way of making arrays with different shapes work together in arithmetic operations. Instead of requiring you to explicitly copy data to make shapes match, it "stretches" the smaller array to fit the larger one—without actually copying the data.

Think of it like this: when you add a scalar to an array, the scalar conceptually becomes an array of the same shape, filled with that value:

```csharp
var a = np.array(new[] {1, 2, 3});
var b = a + 10;
// Conceptually: [1, 2, 3] + [10, 10, 10] = [11, 12, 13]
// But no [10, 10, 10] array is actually created!
```

Broadcasting generalizes this to any dimension where stretching makes sense.

---

## The Three Rules

Broadcasting follows three simple rules. If these rules can be satisfied, the arrays are "broadcastable." If not, you get a shape error.

### Rule 1: Align from the Right

When comparing shapes, start from the rightmost dimension and work left. If one array has fewer dimensions, prepend 1s to its shape until they match.

```
Array A: (    3, 4)    → treated as (1, 3, 4)
Array B: (2, 3, 4)
```

### Rule 2: Dimensions Must Match OR Be 1

For each dimension, the sizes must either:
- Be equal, OR
- One of them must be 1

```
(5, 4) and (1, 4)  ✓  (5 vs 1, 4 vs 4)
(5, 4) and (   4)  ✓  (treated as (1, 4): 5 vs 1, 4 vs 4)
(5, 4) and (5, 1)  ✓  (5 vs 5, 4 vs 1)
(5, 4) and (3, 4)  ✗  (5 vs 3 — neither is 1!)
```

### Rule 3: Stretch the 1s

Dimensions of size 1 are "stretched" to match the larger size. The element at that position is repeated as many times as needed.

```csharp
var a = np.array(new[,] {{1}, {2}, {3}});  // Shape: (3, 1)
var b = np.array(new[] {10, 20, 30, 40});  // Shape: (4,)

var c = a + b;  // Result shape: (3, 4)

// a is stretched horizontally:     b is stretched vertically:
// [[1, 1, 1, 1],                   [[10, 20, 30, 40],
//  [2, 2, 2, 2],        +           [10, 20, 30, 40],
//  [3, 3, 3, 3]]                    [10, 20, 30, 40]]
//
// = [[11, 21, 31, 41],
//    [12, 22, 32, 42],
//    [13, 23, 33, 43]]
```

---

## Visual Examples

Let's work through some common cases.

### Scalar + Array

The simplest broadcast. A scalar (0-dimensional) broadcasts to any shape.

```csharp
var a = np.array(new[] {1, 2, 3, 4, 5});
var b = a * 2;  // [2, 4, 6, 8, 10]
```

```
Shape of a:     (5,)
Shape of 2:     ()     → treated as (1,) → stretched to (5,)
Result shape:   (5,)
```

### Row Vector + Column Vector

This is the classic outer product pattern.

```csharp
var row = np.array(new[] {1, 2, 3});           // Shape: (3,)
var col = np.array(new[,] {{10}, {20}, {30}}); // Shape: (3, 1)

var result = row + col;  // Shape: (3, 3)
// [[11, 12, 13],
//  [21, 22, 23],
//  [31, 32, 33]]
```

```
Shape of row:  (   3)  → treated as (1, 3)
Shape of col:  (3, 1)
               ------
Result shape:  (3, 3)  ← max of each dimension
```

### Matrix + Row Vector

This is your "subtract the mean from each row" pattern.

```csharp
var matrix = np.arange(12).reshape(3, 4);
// [[ 0,  1,  2,  3],
//  [ 4,  5,  6,  7],
//  [ 8,  9, 10, 11]]

var row_means = np.array(new[] {1, 2, 3, 4});  // Shape: (4,)

var centered = matrix - row_means;
// [[-1, -1, -1, -1],
//  [ 3,  3,  3,  3],
//  [ 7,  7,  7,  7]]
```

```
Shape of matrix:     (3, 4)
Shape of row_means:  (   4) → treated as (1, 4)
                     ------
Result shape:        (3, 4)
```

The row vector is broadcast across all rows of the matrix.

### Matrix + Column Vector

Want to subtract from columns instead? Use a column vector.

```csharp
var matrix = np.arange(12).reshape(3, 4);
var col_offsets = np.array(new[,] {{0}, {4}, {8}});  // Shape: (3, 1)

var result = matrix - col_offsets;
// [[0, 1, 2, 3],
//  [0, 1, 2, 3],
//  [0, 1, 2, 3]]
```

```
Shape of matrix:      (3, 4)
Shape of col_offsets: (3, 1)
                      ------
Result shape:         (3, 4)
```

### Higher Dimensions: Batched Operations

Broadcasting really shines with batched data. Say you have a batch of 32 images, each 28x28 pixels:

```csharp
var images = np.random.rand(32, 28, 28);      // Shape: (32, 28, 28)
var pixel_mean = np.mean(images, axis: 0);    // Shape: (28, 28)

var normalized = images - pixel_mean;         // Shape: (32, 28, 28)
```

The `(28, 28)` mean is broadcast across all 32 images.

---

## NumSharp Broadcasting Functions

### `np.broadcast_to(array, shape)`

Explicitly broadcast an array to a specific shape. Returns a view (not a copy).

```csharp
var a = np.array(new[] {1, 2, 3});
var b = np.broadcast_to(a, (4, 3));
// [[1, 2, 3],
//  [1, 2, 3],
//  [1, 2, 3],
//  [1, 2, 3]]

// This is a VIEW - no data is copied!
// b.Shape.IsBroadcasted == true
```

**Important:** The source shape must be compatible with the target. You can only stretch dimensions of size 1:

```csharp
np.broadcast_to(np.array(new[] {1, 2}), (3, 3));  // ERROR! (2,) can't become (3,3)
np.broadcast_to(np.array(new[] {1, 2}), (3, 2));  // OK! (2,) → (1,2) → (3,2)
```

### `np.broadcast_arrays(array1, array2, ...)`

Broadcast multiple arrays against each other, returning views with matching shapes.

```csharp
var a = np.array(new[] {1, 2, 3});              // Shape: (3,)
var b = np.array(new[,] {{1}, {2}});            // Shape: (2, 1)

var (a_bc, b_bc) = np.broadcast_arrays(a, b);
// Both now have shape (2, 3)

// a_bc:          b_bc:
// [[1, 2, 3],    [[1, 1, 1],
//  [1, 2, 3]]     [2, 2, 2]]
```

This is useful when you need to iterate over aligned arrays or pass them to functions that don't broadcast internally.

### Implicit Broadcasting

Most arithmetic operations broadcast automatically:

```csharp
var a = np.ones((3, 4));
var b = np.array(new[] {1, 2, 3, 4});

var c = a + b;  // Broadcasting happens automatically
var d = a * b;  // Works with all operators
var e = a - b;
var f = a / b;
```

---

## Why Views Matter

Here's something crucial: **broadcasting creates views, not copies**.

```csharp
var small = np.array(new[] {1, 2, 3});
var big = np.broadcast_to(small, (1000000, 3));

// big.Shape: (1000000, 3)
// big.size: 3,000,000 elements
// BUT: Only 3 values are actually stored!
```

The broadcasted array `big` looks like it has 3 million elements, but it's just pointing to the same 3 values with clever stride tricks. This is why broadcasting is so memory-efficient.

But there's a catch: **broadcasted arrays are read-only in NumPy**. If you try to modify them, you'll affect multiple "virtual" positions. NumSharp follows the same principle—if you need to modify a broadcasted array, make a copy first:

```csharp
var writable = big.copy();  // Now you have a real 3M element array
```

---

## Common Pitfalls

### Pitfall 1: Shape Mismatch Errors

The most common error is shapes that don't align:

```csharp
var a = np.ones((3, 4));
var b = np.ones((3,));

var c = a + b;  // ERROR! (3, 4) and (3,) don't broadcast
```

Why? Let's check the rules:
```
a: (3, 4)
b: (   3) → (1, 3)
   ------
   (3, 4) vs (1, 3)
   4 ≠ 3 and neither is 1 → FAIL
```

**Fix:** Reshape `b` to be a column vector:
```csharp
var c = a + b.reshape(3, 1);  // Now it works!
```

### Pitfall 2: Row vs Column Confusion

This bites everyone at some point:

```csharp
var a = np.array(new[] {1, 2, 3});  // Shape: (3,) - NOT (1, 3)!
```

A 1-D array of shape `(3,)` is neither a row nor column vector—it's just a 1-D array. When broadcast, it becomes `(1, 3)` (a row). If you want column behavior, explicitly reshape:

```csharp
var col = a.reshape(3, 1);  // Now shape (3, 1)
// or
var col = a[np.newaxis].T;  // Another way
```

### Pitfall 3: Accidentally Broadcasting

Sometimes broadcasting does something you didn't intend:

```csharp
var a = np.ones((10, 1));
var b = np.ones((1, 10));

var c = a + b;  // Shape: (10, 10) — probably not what you wanted!
```

You might have expected element-wise addition of two 10-element arrays, but broadcasting created a 10x10 matrix. Always check your shapes.

### Pitfall 4: Memory Explosion

Broadcasting is memory-efficient for views, but operations that materialize the broadcast can explode:

```csharp
var a = np.ones((1000, 1));
var b = np.ones((1, 1000));

// This is fine - just views
var (a_bc, b_bc) = np.broadcast_arrays(a, b);

// This allocates 1,000,000 elements!
var result = a + b;  // The result must be materialized
```

---

## Performance Considerations

### When Broadcasting is Fast

Broadcasting is fastest when:
- One operand is contiguous and the other has stride-0 dimensions
- The operation can use SIMD on the contiguous parts
- You're avoiding the allocation of intermediate copies

### When Broadcasting Can Be Slower

Broadcasting can have performance costs when:
- Both arrays are non-contiguous after broadcasting (cache-unfriendly access)
- The stride pattern prevents vectorization
- You're doing many small broadcasts instead of one large operation

For performance-critical code, consider whether explicit `np.tile()` or pre-allocated loops might be faster (profile first!).

---

## How NumSharp Implements Broadcasting

Internally, NumSharp uses stride manipulation to achieve broadcasting without copying data.

When you broadcast a shape like `(3,)` to `(4, 3)`:
1. The shape becomes `(4, 3)`
2. The strides become `(0, 1)` — the first dimension has stride 0!

A stride of 0 means "don't move in memory when you increment this index." The same 3 elements are accessed for every row.

```csharp
var a = np.array(new[] {1, 2, 3});
var b = np.broadcast_to(a, (4, 3));

// b.Shape.dimensions: [4, 3]
// b.Shape.strides:    [0, 1]  ← First stride is 0!
```

This is tracked internally via `Shape.IsBroadcasted` and `BroadcastInfo`.

---

## Quick Reference

### Compatible Shapes

| Shape A | Shape B | Result | Why |
|---------|---------|--------|-----|
| `(5,)` | `(5,)` | `(5,)` | Same shape |
| `(5,)` | `(1,)` | `(5,)` | 1 stretches to 5 |
| `(5,)` | `()` | `(5,)` | Scalar broadcasts |
| `(3, 4)` | `(4,)` | `(3, 4)` | (4,) → (1, 4) → (3, 4) |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | 1 stretches to 4 |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | Both stretch |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | Prepend 1, then stretch |

### Incompatible Shapes

| Shape A | Shape B | Why It Fails |
|---------|---------|--------------|
| `(3,)` | `(4,)` | 3 ≠ 4, neither is 1 |
| `(3, 4)` | `(3,)` | 4 ≠ 3, neither is 1 |
| `(2, 3)` | `(3, 2)` | No alignment works |

### Functions

| Function | Description |
|----------|-------------|
| `np.broadcast_to(a, shape)` | Broadcast `a` to specific shape (returns view) |
| `np.broadcast_arrays(a, b, ...)` | Broadcast multiple arrays to common shape |
| `a + b`, `a * b`, etc. | Implicit broadcasting in operations |

---

## Summary

Broadcasting is about making array operations work seamlessly across different shapes:

1. **Shapes align from the right**
2. **Dimensions must match or be 1**
3. **1s are stretched to match**
4. **It's memory-efficient** — views, not copies
5. **Watch your shapes** — unexpected broadcasts happen silently

Once you internalize these rules, you'll start seeing opportunities to eliminate loops and express operations more naturally. It's one of those features that, once learned, you can't imagine working without.
