using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.delete ==============================
        // Return a copy of an array with sub-arrays along an axis deleted.
        //
        // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::delete
        // Three obj shapes drive three execution paths:
        //   1. scalar int       — drop one position along axis
        //   2. slice            — drop a stride-defined range along axis
        //   3. integer ndarray  — drop multiple (possibly duplicate / unordered) positions
        //   4. boolean ndarray  — keep where mask is False (length must match axis dim)
        //
        // axis=None ravels the input to 1-D before deletion.
        //
        // For 1D contiguous arrays, the scalar/slice paths reuse np.concatenate of
        // pre/post views (zero-copy slicing + Buffer.MemoryCopy on the dst). The
        // array/bool paths build a keep-mask and delegate to np.compress, which
        // owns the fused popcount + axis-gather kernel.

        /// <summary>
        ///     Return a new array with the element at <paramref name="obj"/>
        ///     along <paramref name="axis"/> removed.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">Integer index of the position to remove. Accepts
        ///     negative indices (counted from the end). Raises
        ///     <see cref="IndexOutOfRangeException"/> when out of bounds for the
        ///     selected axis.</param>
        /// <param name="axis">Axis along which to delete. <c>null</c> (default)
        ///     flattens <paramref name="arr"/> first and returns a 1-D result.</param>
        /// <returns>A C-contiguous copy of <paramref name="arr"/> with one
        ///     sub-array removed along <paramref name="axis"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.delete.html</remarks>
        public static NDArray delete(NDArray arr, int obj, int? axis = null)
            => delete(arr, (long)obj, axis);

        /// <summary>
        ///     Long-index overload of <see cref="delete(NDArray, int, int?)"/>.
        /// </summary>
        public static NDArray delete(NDArray arr, long obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            var ctx = PrepareAxisContext(arr, axis);
            return DeleteSingleIndex(ctx.work, obj, ctx.axis);
        }

        /// <summary>
        ///     Slice-index overload. <paramref name="obj"/> is interpreted via
        ///     Python <c>slice.indices(N)</c> against the axis length.
        /// </summary>
        public static NDArray delete(NDArray arr, Slice obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            var ctx = PrepareAxisContext(arr, axis);
            return DeleteSlice(ctx.work, obj, ctx.axis);
        }

        /// <summary>
        ///     Array-of-indices overload. Negative indices are normalized; duplicates
        ///     are silently collapsed (each axis position is removed at most once).
        /// </summary>
        public static NDArray delete(NDArray arr, int[] obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];

            var ctx = PrepareAxisContext(arr, axis);
            return DeleteIndexArray(ctx.work, longs, ctx.axis);
        }

        /// <summary>
        ///     Long-array-of-indices overload.
        /// </summary>
        public static NDArray delete(NDArray arr, long[] obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            var ctx = PrepareAxisContext(arr, axis);
            return DeleteIndexArray(ctx.work, (long[])obj.Clone(), ctx.axis);
        }

        /// <summary>
        ///     Bool-array overload — values are interpreted as a keep-mask
        ///     inversion. Length must match the targeted axis size (NumPy raises
        ///     <c>ValueError</c> otherwise).
        /// </summary>
        public static NDArray delete(NDArray arr, bool[] obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            var ctx = PrepareAxisContext(arr, axis);
            long N = ctx.work.shape[ctx.axis];
            if (obj.Length != N)
                throw new ArgumentException(
                    "boolean array argument obj to delete must be one dimensional " +
                    $"and match the axis length of {N}",
                    nameof(obj));

            // keep = ~obj; route through np.compress which has the fused gather kernel.
            var keep = new bool[obj.Length];
            for (int i = 0; i < obj.Length; i++) keep[i] = !obj[i];

            var keepArr = np.array(keep);
            try { return np.compress(keepArr, ctx.work, ctx.axis); }
            finally { keepArr.Dispose(); }
        }

        /// <summary>
        ///     NDArray-typed obj dispatch — selects the integer-array path or the
        ///     boolean-mask path based on <c>obj.GetTypeCode</c>. 0-D and 1-element
        ///     integer arrays collapse to the scalar-index fast path (matching
        ///     NumPy's <c>obj.size == 1 and obj.dtype.kind in "ui": obj = obj.item()</c>).
        /// </summary>
        public static NDArray delete(NDArray arr, NDArray obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            var ctx = PrepareAxisContext(arr, axis);

            if (obj.GetTypeCode == NPTypeCode.Boolean)
            {
                long N = ctx.work.shape[ctx.axis];
                if (obj.size != N || obj.ndim != 1)
                    throw new ArgumentException(
                        "boolean array argument obj to delete must be one dimensional " +
                        $"and match the axis length of {N}",
                        nameof(obj));

                // keep = ~obj. Build a contig bool keep-mask without leaning on
                // bitwise-not over a possibly-strided NDArray.
                bool[] keep = new bool[N];
                for (long i = 0; i < N; i++) keep[i] = !obj.GetBoolean((int)i);
                var keepArr = np.array(keep);
                try { return np.compress(keepArr, ctx.work, ctx.axis); }
                finally { keepArr.Dispose(); }
            }

            // Integer obj. Size-1 ⇒ collapse to scalar path (NumPy parity).
            if (obj.size == 1)
            {
                long idx = ToInt64Scalar(obj, "delete");
                return DeleteSingleIndex(ctx.work, idx, ctx.axis);
            }

            // Materialise indices into a managed long[] for the multi-index path.
            long[] indices = ToInt64Vector(obj, "delete");
            return DeleteIndexArray(ctx.work, indices, ctx.axis);
        }

        // ---------------------------- helpers ----------------------------

        /// <summary>
        ///     Resolves <c>axis=None</c> to "ravel + axis=0" and normalises a
        ///     non-null axis against <c>arr.ndim</c>. The returned <c>work</c>
        ///     array is either <c>arr</c> itself or a freshly-allocated ravel
        ///     (which the caller can treat as owning since insert/delete always
        ///     produces a new array).
        /// </summary>
        private static (NDArray work, int axis) PrepareAxisContext(NDArray arr, int? axis)
        {
            if (axis is null)
            {
                NDArray work = arr.ndim == 1 ? arr : np.ravel(arr);
                return (work, 0);
            }

            int ax = axis.Value;
            if (ax < 0) ax += arr.ndim;
            if (ax < 0 || ax >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis.Value} is out of bounds for array of dimension {arr.ndim}");
            return (arr, ax);
        }

        /// <summary>
        ///     Casts <paramref name="obj"/> to <see cref="long"/> with NumPy-style
        ///     bounds error on out-of-range. Used by the scalar-index fast paths.
        /// </summary>
        private static long ToInt64Scalar(NDArray obj, string fn)
        {
            // Accept any integer dtype (NumPy: kind in "ui"). Bool is handled by
            // the caller before this point.
            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:   return obj.GetByte(0);
                case NPTypeCode.Int16:  return obj.GetInt16(0);
                case NPTypeCode.UInt16: return obj.GetUInt16(0);
                case NPTypeCode.Int32:  return obj.GetInt32(0);
                case NPTypeCode.UInt32: return obj.GetUInt32(0);
                case NPTypeCode.Int64:  return obj.GetInt64(0);
                case NPTypeCode.UInt64: return (long)obj.GetUInt64(0);
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.Name}",
                        "obj");
            }
        }

        /// <summary>
        ///     Materialises <paramref name="obj"/> as a managed <c>long[]</c>.
        ///     Any integer dtype is accepted; size-0 input returns an empty array
        ///     so the caller's "no-op" branch kicks in.
        /// </summary>
        private static long[] ToInt64Vector(NDArray obj, string fn)
        {
            if (obj.ndim > 1)
                throw new ArgumentException(
                    $"np.{fn}: index array argument obj must be one dimensional or scalar",
                    "obj");

            long n = obj.size;
            var result = new long[n];

            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:
                    for (long i = 0; i < n; i++) result[i] = obj.GetByte((int)i);
                    break;
                case NPTypeCode.Int16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt16((int)i);
                    break;
                case NPTypeCode.UInt16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt16((int)i);
                    break;
                case NPTypeCode.Int32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt32((int)i);
                    break;
                case NPTypeCode.UInt32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt32((int)i);
                    break;
                case NPTypeCode.Int64:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt64((int)i);
                    break;
                case NPTypeCode.UInt64:
                    for (long i = 0; i < n; i++) result[i] = (long)obj.GetUInt64((int)i);
                    break;
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.Name}",
                        "obj");
            }

            return result;
        }

        /// <summary>
        ///     Scalar-index path: two concatenated views (pre + post) along
        ///     <paramref name="axis"/>. Bounds-checks first; -N…N-1 are valid.
        ///     Single-chunk fall-throughs route through <see cref="np.copy"/>
        ///     to produce an owning C-contig duplicate.
        /// </summary>
        private static NDArray DeleteSingleIndex(NDArray arr, long obj, int axis)
        {
            long N = arr.shape[axis];
            if (obj < -N || obj >= N)
                throw new IndexOutOfRangeException(
                    $"index {obj} is out of bounds for axis {axis} with size {N}");
            if (obj < 0) obj += N;

            var pre = SliceAlongAxis(arr, axis, 0, obj);
            var post = SliceAlongAxis(arr, axis, obj + 1, N);

            try
            {
                if (pre.shape[axis] == 0) return np.copy(post);
                if (post.shape[axis] == 0) return np.copy(pre);
                return np.concatenate(new[] { pre, post }, axis);
            }
            finally { pre.Dispose(); post.Dispose(); }
        }

        /// <summary>
        ///     Slice-obj path. Mirrors NumPy's implementation: positive-step slice
        ///     concatenates pre + middle-kept + post; negative step is normalised
        ///     by inverting the slice direction. step!=1 falls back to the
        ///     keep-mask path which is identical to the array-obj path with the
        ///     slice expanded.
        /// </summary>
        private static NDArray DeleteSlice(NDArray arr, Slice obj, int axis)
        {
            long N = arr.shape[axis];
            var (start, stop, step) = PythonSliceIndices(obj, N);

            // Count elements that will actually be removed (matches NumPy's `xr`).
            long numToDelete;
            if (step > 0)
            {
                if (stop <= start) numToDelete = 0;
                else numToDelete = (stop - start + step - 1) / step;
            }
            else
            {
                if (stop >= start) numToDelete = 0;
                else
                {
                    long mag = -step;
                    numToDelete = (start - stop + mag - 1) / mag;
                }
            }

            if (numToDelete <= 0)
                return DuplicateArray(arr);

            // Normalise negative-step slices to forward iteration over the same
            // covered range (NumPy: start = xr[-1]; stop = xr[0] + 1; step = -step).
            if (step < 0)
            {
                long absStep = -step;
                long lastIncluded = start + (numToDelete - 1) * step; // smallest index removed
                long firstIncluded = start; // largest index removed
                start = lastIncluded;
                stop = firstIncluded + 1;
                step = absStep;
            }

            // step == 1 is a single contiguous block — pre + post concatenate.
            if (step == 1)
            {
                var pre = SliceAlongAxis(arr, axis, 0, start);
                var post = SliceAlongAxis(arr, axis, stop, N);
                try
                {
                    if (pre.shape[axis] == 0) return np.copy(post);
                    if (post.shape[axis] == 0) return np.copy(pre);
                    return np.concatenate(new[] { pre, post }, axis);
                }
                finally { pre.Dispose(); post.Dispose(); }
            }

            // Strided slice: expand to indices array and route through the mask path.
            var indices = new long[numToDelete];
            for (long i = 0; i < numToDelete; i++)
                indices[i] = start + i * step;
            return DeleteIndexArray(arr, indices, axis);
        }

        /// <summary>
        ///     Multi-index path with two execution branches based on the ratio
        ///     of deletions to axis length:
        ///     <list type="bullet">
        ///       <item><b>Sparse</b> (numDelete &lt; <see cref="DeleteChunkConcatThreshold"/>):
        ///             sort + dedupe, then concat the kept chunks between deleted
        ///             positions. Skips the O(N) bool-mask scan that compress's
        ///             popcount+gather kernel needs. Wins ~2-3x on workloads like
        ///             "delete 5 of 1M".</item>
        ///       <item><b>Dense</b> (everything else): bool keep-mask + compress
        ///             — popcount + fused-gather kernel reads each axis position
        ///             exactly once, beats chunk-concat when chunks would be tiny.</item>
        ///     </list>
        /// </summary>
        private static NDArray DeleteIndexArray(NDArray arr, long[] indices, int axis)
        {
            long N = arr.shape[axis];

            if (indices.Length == 0)
                return DuplicateArray(arr);

            // Normalize + bounds-check.
            for (int k = 0; k < indices.Length; k++)
            {
                long idx = indices[k];
                if (idx < -N || idx >= N)
                    throw new IndexOutOfRangeException(
                        $"index {indices[k]} is out of bounds for axis {axis} with size {N}");
                if (idx < 0) idx += N;
                indices[k] = idx;
            }

            // Sparse fast path: a small handful of deletions => chunk-concat is
            // far cheaper than scanning the whole bool-mask. The constant is
            // tuned against the popcount+gather kernel's per-byte throughput.
            if (indices.Length < DeleteChunkConcatThreshold)
                return DeleteChunkConcat(arr, indices, axis);

            // Dense path: compress over a bool keep mask.
            var keep = new bool[N];
            for (long i = 0; i < N; i++) keep[i] = true;
            for (int k = 0; k < indices.Length; k++) keep[indices[k]] = false;

            var keepArr = np.array(keep);
            try { return np.compress(keepArr, arr, axis); }
            finally { keepArr.Dispose(); }
        }

        /// <summary>
        ///     Number of axis-deletions below which <see cref="DeleteIndexArray"/>
        ///     prefers chunk-concat over compress. 256 is the rough crossover
        ///     point on a 1M-element axis where compress's amortized cost catches
        ///     up with the chunk-concat's per-chunk overhead.
        /// </summary>
        private const int DeleteChunkConcatThreshold = 256;

        /// <summary>
        ///     Sparse-delete fast path. Sorts and dedupes the (already-normalized)
        ///     <paramref name="indices"/>, then concatenates the kept chunks
        ///     between deleted positions. Equivalent to the bool-mask + compress
        ///     path but skips the O(N) mask scan.
        /// </summary>
        private static NDArray DeleteChunkConcat(NDArray arr, long[] indices, int axis)
        {
            long N = arr.shape[axis];

            // Sort + dedupe in place. The chunks must be derived from monotonic
            // axis positions; duplicates count as a single deletion.
            Array.Sort(indices);
            int unique = 0;
            for (int k = 0; k < indices.Length; k++)
            {
                if (k == 0 || indices[k] != indices[k - 1])
                    indices[unique++] = indices[k];
            }

            // Build chunks: arr[prev:idx_0], arr[idx_0+1:idx_1], ..., arr[idx_{n-1}+1:N].
            // The first chunk uses prev=0; subsequent chunks use prev = idx_{k-1}+1.
            // Skip zero-length chunks so concatenate sees only real data.
            var chunks = new System.Collections.Generic.List<NDArray>(unique + 1);
            var disposable = new System.Collections.Generic.List<NDArray>(unique + 1);
            try
            {
                long prev = 0;
                for (int k = 0; k < unique; k++)
                {
                    long idx = indices[k];
                    if (idx > prev)
                    {
                        var chunk = SliceAlongAxis(arr, axis, prev, idx);
                        chunks.Add(chunk);
                        disposable.Add(chunk);
                    }
                    prev = idx + 1;
                }
                if (prev < N)
                {
                    var tail = SliceAlongAxis(arr, axis, prev, N);
                    chunks.Add(tail);
                    disposable.Add(tail);
                }

                if (chunks.Count == 0)
                {
                    // Every axis position was deleted ⇒ allocate an empty
                    // result with axis dim 0 (NumPy parity).
                    var emptyDims = new long[arr.ndim];
                    for (int d = 0; d < arr.ndim; d++) emptyDims[d] = arr.shape[d];
                    emptyDims[axis] = 0;
                    return new NDArray(arr.GetTypeCode, new Shape(emptyDims), false);
                }

                return np.concatenate(chunks.ToArray(), axis);
            }
            finally
            {
                foreach (var nd in disposable) nd.Dispose();
            }
        }

        /// <summary>
        ///     Allocates a fresh, owning copy of <paramref name="arr"/>. Used by
        ///     delete/insert no-op paths to honour NumPy's "delete does not occur
        ///     in-place" contract.
        /// </summary>
        internal static NDArray DuplicateArray(NDArray arr) => np.copy(arr);

        /// <summary>
        ///     Returns an axis-aligned slice view <c>arr[..., start:stop, ...]</c>
        ///     by building a <see cref="Slice"/> array. The result shares storage
        ///     with <paramref name="arr"/> when <c>arr.Shape.IsContiguous</c> /
        ///     stride-friendly; the caller is responsible for disposing the wrapper.
        /// </summary>
        internal static NDArray SliceAlongAxis(NDArray arr, int axis, long start, long stop)
        {
            int ndim = arr.ndim;
            var slices = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                slices[i] = i == axis ? new Slice(start, stop) : Slice.All;
            return arr[slices];
        }

        /// <summary>
        ///     Python-compatible <c>slice.indices(N)</c> implementation. Resolves
        ///     None/negative bounds to absolute positions and clamps to a legal
        ///     iteration range. Used by both <see cref="DeleteSlice"/> and
        ///     <see cref="InsertExpandSliceObj"/>.
        /// </summary>
        internal static (long start, long stop, long step) PythonSliceIndices(Slice s, long N)
        {
            long step = s.Step == 0 ? 1 : s.Step;
            if (step == 0)
                throw new ArgumentException("slice step cannot be zero");

            long start, stop;
            long defaultStart = step > 0 ? 0 : N - 1;
            long defaultStop = step > 0 ? N : -1; // exclusive; -1 sentinel for backwards

            if (s.Start.HasValue)
            {
                start = s.Start.Value;
                if (start < 0) start += N;
                if (step > 0)
                {
                    if (start < 0) start = 0;
                    if (start > N) start = N;
                }
                else
                {
                    if (start < 0) start = -1;
                    if (start >= N) start = N - 1;
                }
            }
            else start = defaultStart;

            if (s.Stop.HasValue)
            {
                stop = s.Stop.Value;
                if (stop < 0) stop += N;
                if (step > 0)
                {
                    if (stop < 0) stop = 0;
                    if (stop > N) stop = N;
                }
                else
                {
                    if (stop < 0) stop = -1;
                    if (stop >= N) stop = N - 1;
                }
            }
            else stop = defaultStop;

            return (start, stop, step);
        }
    }
}
