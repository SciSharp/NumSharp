# np.pad — Implementation Plan

> Target: NumPy 2.4.2 parity for `numpy.pad`.
> Reference: `src/numpy/numpy/lib/_arraypad_impl.py` (926 lines).

## 1. Scope & API surface

NumPy 2.4.2's `np.pad(array, pad_width, mode='constant', **kwargs)` produces a copy of `array` enlarged by zero or more "padding" elements before and after each axis. Eleven `mode` values + an arbitrary user-supplied callable. Five mode-specific kwargs (`constant_values`, `end_values`, `stat_length`, `reflect_type`, plus `**kwargs` forwarded to the callable). `pad_width` itself accepts five shapes (scalar int / `(b,a)` / `((b,a),)` / `((b_i,a_i),...)` / `dict`).

Implementation will live in `src/NumSharp.Core/Manipulation/np.pad.cs` with no DefaultEngine/ILKernel surgery — every internal operation reduces to existing primitives (`np.empty`, slicing-assignment, `np.copyto`, `np.linspace`, axis-reductions, `np.moveaxis`).

### Public API

```csharp
public static NDArray pad(NDArray array, int pad_width, string mode = "constant",
    object constant_values = null, object end_values = null,
    object stat_length = null, string reflect_type = "even");

public static NDArray pad(NDArray array, int[] pad_width, string mode = "constant", ...);
public static NDArray pad(NDArray array, int[,] pad_width, string mode = "constant", ...);
public static NDArray pad(NDArray array, (int before, int after) pad_width, string mode = "constant", ...);
public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, string mode = "constant", ...);

// Callable mode:
public delegate void PadFunc(NDArray vector1d, (int before, int after) pad_width, int axis, object kwargs);
public static NDArray pad(NDArray array, int pad_width, PadFunc mode, object kwargs = null);
public static NDArray pad(NDArray array, int[] pad_width, PadFunc mode, object kwargs = null);
// (+ overloads matching the string-mode shapes)
```

We use `object` for value kwargs (NumPy parity needs scalar, `(b,a)`, `((b,a),)`, full `((b,a),...)` broadcast). Internally normalized via `_AsPairs<T>` (mirrors NumPy's `_as_pairs`).

## 2. Internal architecture

```
np.pad(arr, pad_width, mode, **kw)
   │
   ├── NormalizePadWidth(arr.ndim, pad_width)   → long[ndim,2]
   ├── ValidatePadWidthNonNegative(...)
   ├── If callable mode: callable branch (loop over inds via moveaxis-to-last)
   ├── Else dispatch on mode-string:
   │      ├── "constant" → PadSimple(fill=cv pair) + per-axis SetPadArea
   │      ├── "edge"     → PadSimple + per-axis SetPadArea(left_edge, right_edge)
   │      ├── "linear_ramp" → PadSimple + per-axis LinearRamp + SetPadArea
   │      ├── "maximum"|"minimum"|"mean"|"median" → PadSimple + GetStats
   │      ├── "reflect"|"symmetric" → PadSimple + loop SetReflectBoth
   │      ├── "wrap"    → PadSimple + loop SetWrapBoth
   │      └── "empty"   → PadSimple (return as-is, undefined pad area)
   └── return padded
```

### Helper primitives

| Helper | Purpose | Maps to NumPy |
|--------|---------|---------------|
| `_AsPairs<T>(object, ndim, asIndex)` | normalize scalar / `(b,a)` / `((b,a),)` / `((b_i,a_i),...)` / `dict` into `T[ndim,2]` | `_as_pairs` |
| `_PadSimple(arr, padWidth, fillValue?)` | allocate `np.empty(newShape, arr.dtype, order)`, optionally fill, copy original into center, return `(padded, originalAreaSlice)` | `_pad_simple` |
| `_SliceAtAxis(start, stop, axis, ndim)` → `Slice[]` | construct full slice array reading at one axis | `_slice_at_axis` |
| `_ViewRoi(padded, originalAreaSlice, axis)` | for axis k, return `padded[Slice.All]*ndim` with axes `[k+1..]` clamped to the original slot (avoids re-overwriting corners) | `_view_roi` |
| `_SetPadArea(padded, axis, widthPair, valuePair)` | write `valuePair.left` into the left pad band, `valuePair.right` into the right pad band along axis | `_set_pad_area` |
| `_GetEdges(padded, axis, widthPair)` | extract the 1-thick left/right edge slices of the valid region | `_get_edges` |
| `_GetLinearRamps(padded, axis, widthPair, endPair)` | construct two linspace ramps from `endValue → edge` along axis | `_get_linear_ramps` |
| `_GetStats(padded, axis, widthPair, lengthPair, statFunc)` | reduce the edge `stat_length` strip along axis with keepdims=true | `_get_stats` |
| `_SetReflectBoth(roi, axis, widthPair, method, period, includeEdge)` | one pass of reflect padding, returns the residual `(leftPad, rightPad)` for the next iteration when pad > period | `_set_reflect_both` |
| `_SetWrapBoth(roi, axis, widthPair, period)` | one pass of wrap, returns residual | `_set_wrap_both` |

## 3. Per-mode implementation notes

### 3.1. `constant` (default)
* Cast `constant_values` to `arr.dtype` (NumPy: integer dtype + float constant → round).
* `_PadSimple` with `fillValue=cv[0][0]` covers the all-equal case in one `Fill()`.
* Otherwise allocate uninitialized then loop axes calling `_SetPadArea` — each axis's `_SetPadArea` writes both edges of that axis's pad band; the `_ViewRoi` view restricts the region so corners are written by axis 0's pass and never re-overwritten.

### 3.2. `edge`
* Per axis, read the 1-thick edge slice (`padded[..., left:left+1, ...]` and `padded[..., right-1:right, ...]`), assign across the pad band. The assignment broadcasts the 1-thick slice over the pad-width-long region — this needs slice-assignment with broadcast, which NumSharp's `dst[slices] = src` does.

### 3.3. `linear_ramp`
* For each axis: build `np.linspace(end_value, edge_value, num=width, endpoint=False, dtype=arr.dtype, axis=axis)`. Edge value is a slice of shape `arr.shape` with axis dim 1, end_value is a scalar.
* For the right side, reverse the linspace along axis.
* Round to integer if dtype is integer.
* Implementation note: NumSharp's `np.linspace` is scalar→scalar; we'll need either a vectorized linspace that takes per-element start/stop arrays, OR generate the ramp manually by computing `end + (edge - end) * (k / num)` for k=0..num-1 broadcast across non-axis dims. The latter is simpler and uses existing arithmetic.

### 3.4. `maximum` / `minimum` / `mean` / `median`
* `stat_length` per axis (default = full valid axis size).
* Compute `padded[edge_strip].reduce(axis=axis, keepdims=true)` for left, then right.
* Round to integer if dtype is integer (matches `_round_if_needed`).
* For `maximum`/`minimum` with `stat_length=0` → `ValueError`.
* `mean`/`median` with `stat_length=0` → NumPy emits a runtime warning + writes NaN-cast garbage; we'll match by letting the existing nan-warning fall through but document the divergence.
* Reductions exist: `np.max(arr, axis=, keepdims=)`, `np.min`, `np.mean`, `np.median`.

### 3.5. `reflect` / `symmetric`
* Per axis, while `left_pad > 0 || right_pad > 0`: pull a `min(period, pad)`-long mirrored chunk from the valid region, write into the pad band, decrement pad by that amount.
* `reflect`: mirror axis is the edge (edge not included in mirror). Period for repetition = `original_length`.
* `symmetric`: edge included; period = `original_length` (different boundary).
* `reflect_type='odd'`: pulled chunk replaced by `2*edge - chunk`.
* Edge case `axis_size == 1 && pad > 0`: NumPy falls back to `edge` for that axis ("legacy behavior").
* When `pad_width > period`, iterate — each iteration grows the "valid" region by what was just written, so the next iteration's mirror reaches into newly-written material.

### 3.6. `wrap`
* Period = `padded.shape[axis] - left_pad - right_pad` (= original axis size).
* Left pad pulled from the right end of the valid region (last `min(period, left_pad)` elements).
* Right pad pulled from the left end of the valid region.
* Iterate when pad > period (same shrink-residual pattern as reflect).

### 3.7. `empty`
* `_PadSimple` with no fill, return as-is. The pad area is uninitialized memory; NumPy doesn't guarantee anything.

### 3.8. Callable mode
* Per NumPy: allocate zero-filled padded, then for each axis `axis`:
  * `view = np.moveaxis(padded, axis, -1)` — bring the pad axis to the last position.
  * For each multi-index `ind` ranging over `view.shape[:-1]`: extract `view[ind, ...]` as a 1-D NDArray (already padded with zeros), pass `(vec, padWidth[axis], axis, kwargs)` to the user function. The function mutates the vector in place.
* Trade-off: this is slow but matches NumPy's documented contract. We won't optimize the callable path.

## 4. Dtype handling

* Output `dtype = arr.dtype`.
* For modes that compute values from existing data (edge/reflect/wrap/stat-with-integer): no conversion needed.
* For `constant` and `linear_ramp`: if the supplied scalar can't represent in `arr.dtype` exactly, we cast (NumPy rounds when target is integer; otherwise float-cast).
* All 15 NumSharp dtypes supported — modes that use only existing-data slice copies (constant fill, edge, reflect, wrap) work for every dtype because they delegate to `np.copyto` / IL copy kernels. Modes that need arithmetic (mean, median, linear_ramp, odd reflect) lean on existing math primitives — which already cover all dtypes including Half/Decimal/Complex.

## 5. Edge cases & error semantics

| Case | NumPy behavior | Plan |
|------|----------------|------|
| Negative `pad_width` element | `ValueError("index can't contain negative values")` | `ArgumentException` with same message |
| Non-integer `pad_width` | `TypeError("'pad_width' must be of integral type")` | `ArgumentException` |
| `mode='unknown'` | `ValueError("mode 'unknown' is not supported")` | `ArgumentException` |
| Unsupported kwarg for mode | `ValueError` | `ArgumentException` |
| Empty axis + non-constant mode + nonzero pad | `ValueError("can't extend empty axis ... using modes other than 'constant' or 'empty'")` | Same |
| `axis_size == 1`, reflect/symmetric, pad > 0 | Falls back to edge | Same fallback |
| `stat_length=0` + maximum/minimum | `ValueError("stat_length of 0 yields no value for padding")` | Same |
| All-zero `pad_width` | Returns a **copy** of the input (NumPy: still allocates new) | Same — `np.copy(arr)` |
| Scalar 0-D array | NumPy raises in `_as_pairs` (no axis to pad) — actually returns the 0-D as is when pad_width broadcasts to (0,) | Match by skipping (no-op) |
| F-contig source | NumPy outputs F-contig result (`order = 'F' if array.flags.fnc else 'C'`) | Match via `np.empty(shape, dtype, order)` |

## 6. Performance notes

Most modes are bandwidth-bound: one allocation + a few slice-assignments per axis. Where existing slice-assignment is fast (contig + matching dtype), we'll hit memcpy throughput. No new IL kernels are required.

Hot paths to instrument:
* `constant` with scalar values → single `np.empty + Fill` (saves N writes).
* `edge` → 2 broadcast-assignments per axis (each is a strided memcpy).
* `reflect`/`wrap` → one slice-read + slice-write per pad iteration per axis; pad iterations are bounded by `ceil(pad/period)`.

Target: ≥ 1.0× NumPy on 1M-element 1-D, ≥ 0.8× on stat modes (NumPy has highly tuned axis reductions; we want to come close).

## 7. Test plan

**Parity script** (`/tmp/pad_parity.py` + `/tmp/pad_parity.cs`):
* Generate ~80 cases combining mode × pad_width shape × dtype × ndim (1D/2D/3D).
* Each case dumps `(shape, dtype, data)` JSON; C# loads, runs, compares.
* Covers all eleven modes + callable.

**MSTest unit tests** (`test/NumSharp.UnitTest/Manipulation/np.pad.Test.cs`):
* `Constant_*` (scalar, per-side, per-axis, default-zero, dtype cast/round)
* `Edge_*` (1D/2D/3D, corner propagation)
* `LinearRamp_*` (default, end_values, integer rounding)
* `Maximum_*`, `Minimum_*`, `Mean_*`, `Median_*` (with/without stat_length)
* `Reflect_*` (even/odd, big pad with iteration, axis_size=1 fallback)
* `Symmetric_*` (even/odd)
* `Wrap_*` (small + big pad)
* `Empty_*` (verifies center matches)
* `Callable_*` (with/without kwargs, multi-axis)
* `PadWidth_*` (broadcasting variants)
* `Errors_*` (negative pad, unknown mode, bad kwargs, empty axis)
* Dtype coverage (`Dtype_Double`, `Dtype_Byte`, `Dtype_Int64`, `Dtype_Float`)

**Benchmark** (1D 1M, 2D 1000×1000, 3D 100×100×100): time vs NumPy 2.4.2 for each mode.

## 8. Deliverables

1. `src/NumSharp.Core/Manipulation/np.pad.cs` — single file, ~600-800 lines, organized: dispatcher → helpers → per-mode methods.
2. `test/NumSharp.UnitTest/Manipulation/np.pad.Test.cs` — ~40-60 unit tests.
3. `.claude/CLAUDE.md` update — move `np.pad` from Missing → Shape Manipulation.
4. One commit with extensive message in the repo's style.

## 9. Implementation order

1. Skeleton + `_AsPairs` + `_PadSimple` + dispatcher.
2. `constant` (validates end-to-end pipeline).
3. `edge` (validates `_ViewRoi` corner-propagation logic).
4. `empty`.
5. `wrap`, then `symmetric`, then `reflect` (similar structure, easier→harder).
6. Stat modes (maximum/minimum first, then mean/median).
7. `linear_ramp`.
8. Callable mode.
9. Tests, parity validation, benchmarks.
10. CLAUDE.md update, commit.
