# NumPy Buffered Reduction Double-Loop Analysis

**Purpose**: Understanding NumPy's optimization for buffered reduction iteration.

---

## The Problem

When reducing an array with buffering enabled, a naive approach would:

```
For each input element:
  1. Copy input to buffer
  2. Process element (accumulate into output)
  3. Copy output back to array
  4. Move to next position
```

This is **inefficient** because:
- Output element is copied back/forth for every input element
- Buffer is refilled for each step even when input is contiguous

---

## NumPy's Solution: Double-Loop

NumPy uses a **double-loop pattern** that separates iteration into:
- **Inner loop**: Iterates through the "core" (non-reduce dimensions)
- **Outer loop**: Iterates through the reduce dimension

```
Fill buffer once with coresize * outersize elements

For reduce_pos = 0 to outersize-1:     # Outer loop
  For core_idx = 0 to coresize-1:      # Inner loop
    Process element
    Advance pointers by inner strides

  Advance pointers by outer strides    # Resets inner, advances outer

Write back buffers
Move to next buffer position
```

**Key insight**: The output operand has `reduce_outer_stride = 0`, so its pointer stays at the same location during the outer loop, accumulating values.

---

## Buffer Data Structure

```c
// nditer_impl.h lines 270-293
struct NpyIter_BufferData_tag {
    npy_intp buffersize;     // Total buffer allocation size
    npy_intp size;           // Current iteration size (= coresize when reducing)
    npy_intp bufiterend;     // End of current buffer iteration
    npy_intp reduce_pos;     // Position in outer reduce loop [0, outersize)
    npy_intp coresize;       // Inner loop size (product of non-reduce dims)
    npy_intp outersize;      // Outer loop size (reduce dimension size)
    npy_intp coreoffset;     // Offset into core
    npy_intp outerdim;       // Which dimension is the reduce outer dim

    // Flexible data (stored inline):
    // npy_intp strides[nop]              - Inner strides (for core iteration)
    // npy_intp reduce_outerstrides[nop]  - Outer strides (0 for reduce operands)
    // char* reduce_outerptrs[nop]        - Reset pointers for outer loop start
    // char* buffers[nop]                 - Actual buffer allocations
    // NpyIter_TransferInfo [nop]         - Casting info
};
```

---

## How It Works

### 1. Setup (`npyiter_compute_strides_and_offsets`)

From `nditer_constr.c` lines 2150-2290:

```c
// Find best dimension for buffering (considering reduce axes)
NIT_BUFFERDATA(iter)->coresize = best_coresize;
NIT_BUFFERDATA(iter)->outerdim = best_dim;

for (int iop = 0; iop < nop; iop++) {
    npy_intp inner_stride, reduce_outer_stride;

    if (is_reduce_op) {
        if (NAD_STRIDES(reduce_axisdata)[iop] == 0) {
            // Reduce operand: iterate core normally, outer stays same
            inner_stride = itemsize;
            reduce_outer_stride = 0;  // <-- Key: output doesn't advance
        } else {
            // Broadcast operand: inner is constant, outer advances
            inner_stride = 0;
            reduce_outer_stride = itemsize;
        }
    } else {
        // Normal op: both advance
        inner_stride = itemsize;
        reduce_outer_stride = itemsize * best_coresize;
    }

    NBF_STRIDES(bufferdata)[iop] = inner_stride;
    NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop] = reduce_outer_stride;
}
```

### 2. Buffer Fill (`npyiter_copy_to_buffers`)

From `nditer_api.c` lines 2142-2149:

```c
if (itflags & NPY_ITFLAG_REDUCE) {
    // outersize = how many times we iterate the reduce dimension
    NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize / bufferdata->coresize;

    if (NBF_REDUCE_OUTERSIZE(bufferdata) > 1) {
        // Only iterate core at a time
        bufferdata->size = bufferdata->coresize;
        NBF_BUFITEREND(bufferdata) = iterindex + bufferdata->coresize;
    }
    NBF_REDUCE_POS(bufferdata) = 0;  // Reset outer position
}
```

### 3. The Double-Loop Iteration

From `nditer_templ.c.src` lines 131-210:

```c
static int npyiter_buffered_reduce_iternext(NpyIter *iter) {
    // === INNER LOOP INCREMENT ===
    if (!(itflags & NPY_ITFLAG_EXLOOP)) {
        if (++NIT_ITERINDEX(iter) < NBF_BUFITEREND(bufferdata)) {
            // Still within core - advance by inner strides
            for (iop = 0; iop < nop; ++iop) {
                ptrs[iop] += strides[iop];  // inner_stride
            }
            return 1;  // More elements
        }
    }

    // === OUTER LOOP INCREMENT (the magic!) ===
    if (++NBF_REDUCE_POS(bufferdata) < NBF_REDUCE_OUTERSIZE(bufferdata)) {
        // Move to next reduce position without re-buffering
        for (iop = 0; iop < nop; ++iop) {
            char *ptr = reduce_outerptrs[iop] + reduce_outerstrides[iop];
            ptrs[iop] = ptr;            // Current pointer
            reduce_outerptrs[iop] = ptr; // Save for next outer iteration
        }
        // Reset inner loop bounds
        NBF_BUFITEREND(bufferdata) = NIT_ITERINDEX(iter) + NBF_SIZE(bufferdata);
        return 1;  // More elements (restart inner loop)
    }

    // === BUFFER EXHAUSTED ===
    // Write back results
    npyiter_copy_from_buffers(iter);

    // Check if completely done
    if (NIT_ITERINDEX(iter) >= NIT_ITEREND(iter)) {
        return 0;  // Iteration complete
    }

    // Move to next buffer position and refill
    npyiter_goto_iterindex(iter, NIT_ITERINDEX(iter));
    npyiter_copy_to_buffers(iter, ptrs);

    return 1;
}
```

---

## Visual Example

Reducing `[0, 1, 2, 3, 4, 5]` to scalar (sum):

```
Setup:
  coresize = 1 (no inner dimensions)
  outersize = 6 (reduce dimension)

  Input op:  inner_stride = 8, reduce_outer_stride = 8
  Output op: inner_stride = 8, reduce_outer_stride = 0  <-- KEY!

Buffer fill:
  Copy input: [0, 1, 2, 3, 4, 5] to buffer
  Copy output: [0] to buffer
  Set reduce_pos = 0

Iteration:
  reduce_pos=0: inner loop (size=1)
    output[0] += input[0]  → output = 0
    inner exhausted, advance outer

  reduce_pos=1: input advances, output stays (stride=0!)
    output[0] += input[1]  → output = 1
    inner exhausted, advance outer

  reduce_pos=2:
    output[0] += input[2]  → output = 3
    ...

  reduce_pos=5:
    output[0] += input[5]  → output = 15
    outer exhausted

Write back:
  Copy output buffer [15] back to array

Result: 15
```

---

## IsFirstVisit and Double-Loop

From `nditer_api.c` lines 781-825:

```c
npy_bool NpyIter_IsFirstVisit(NpyIter *iter, int iop) {
    // Part 1: Check coordinates (non-buffered check)
    for (idim = 0; idim < ndim; ++idim) {
        if (stride == 0 && coord != 0) {
            return 0;  // Already visited
        }
    }

    // Part 2: Check buffer reduce_pos (buffered check)
    if (itflags & NPY_ITFLAG_BUFFER) {
        if (NBF_REDUCE_POS(bufferdata) != 0 &&
                NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop] == 0) {
            return 0;  // Already visited via outer loop
        }
    }

    return 1;  // First visit
}
```

---

## What NumSharp Has vs Needs

### Already Implemented ✓

| Field | Description |
|-------|-------------|
| `ReducePos` | Current position in outer loop |
| `ReduceOuterSize` | Size of outer loop |
| `ReduceOuterStrides[8]` | Per-operand outer strides |
| `GetReduceOuterStride()` | Accessor method |
| `IsFirstVisit()` | Checks both coords AND reduce_pos |

### Missing for Full Double-Loop

| Field/Feature | Description |
|---------------|-------------|
| `ReduceOuterPtrs[8]` | Reset pointers for outer loop iteration |
| `CoreSize` | Inner loop size (non-reduce dims product) |
| `OuterDim` | Which dimension is the reduce outer dim |
| `CoreOffset` | Offset into core |
| Double-loop in `Advance()` | The actual iteration pattern when BUFFERED + REDUCE |
| Outer stride calculation | Setup during buffer initialization |

---

## Should NumSharp Implement This?

**Current situation:**
1. ILKernelGenerator handles contiguous arrays with SIMD (fast path)
2. NpyIter handles non-contiguous arrays without buffering
3. Buffered reduction is rare in practice

**The double-loop is a performance optimization** for when:
- Buffering is required (type casting, non-contiguous with copy needed)
- AND reduction is occurring
- AND input data can fit in buffer to avoid re-copying

**Recommendation**: The current implementation is functionally correct. The double-loop optimization can be added later if buffered reduction performance becomes a bottleneck. The infrastructure (ReducePos, ReduceOuterSize, ReduceOuterStrides) is already in place.

---

## Priority

**Low** - This is a performance optimization, not a correctness issue. The basic reduction via op_axes and IsFirstVisit works correctly. Add this only if:
1. Buffered reduction becomes common in NumSharp usage
2. Performance profiling shows re-buffering as a bottleneck
