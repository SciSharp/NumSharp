using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take elements from an array along an axis. Equivalent to fancy
        ///     indexing along the specified axis.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="indices">Integer array of indices to take.</param>
        /// <param name="axis">
        ///     Axis along which to take. <c>null</c> (default) flattens <paramref name="a"/>
        ///     and treats <paramref name="indices"/> as flat indices.
        /// </param>
        /// <param name="out">
        ///     Optional destination array. When supplied, its shape must match the
        ///     natural take output; values are cast to <paramref name="out"/>'s dtype
        ///     via <see cref="np.copyto"/> with unsafe casting and the method returns
        ///     <paramref name="out"/> itself. When <c>null</c> (default), a fresh
        ///     array is allocated with <paramref name="a"/>'s dtype.
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default — throw on OOB), <c>"wrap"</c>
        ///     (modulo with sign correction), or <c>"clip"</c> (saturate).
        /// </param>
        /// <returns>
        ///     New array with shape:
        ///     <list type="bullet">
        ///       <item><c>axis=None</c>: same as <paramref name="indices"/>.</item>
        ///       <item><c>axis=k</c>: <c>a.shape[:k] + indices.shape + a.shape[k+1:]</c>.</item>
        ///     </list>
        ///     Dtype matches <paramref name="a"/> (or <paramref name="out"/>'s dtype
        ///     when <paramref name="out"/> is supplied).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.take.html</remarks>
        [NDScoped] // reclaims the 0-d reshape alias and the natural-result temp on the out= path
        public static NDArray take(NDArray a, NDArray indices, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int modeInt = ParseMode(mode, nameof(mode));

            // 0-d source — treat as 1-element 1-D for the gather. Output shape =
            // indices.shape regardless of axis (NumPy parity).
            NDArray result;
            if (a.ndim == 0)
            {
                var flat0d = a.reshape(new Shape(1));
                result = TakeFlat(flat0d, indices, modeInt);
            }
            else if (axis == null)
            {
                result = TakeFlat(a, indices, modeInt);
            }
            else
            {
                int ax = axis.Value;
                if (ax < 0) ax += a.ndim;
                if (ax < 0 || ax >= a.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis),
                        $"axis {axis.Value} is out of bounds for array of dimension {a.ndim}");
                result = TakeAxis(a, indices, ax, modeInt);
            }

            // out= dispatch: validate shape, validate safe-cast direction
            // (out.dtype → a.dtype), then writeback-if-copy via copyto with unsafe
            // casting. NumPy's PyArray_TakeFrom invokes PyArray_FromArray(out,
            // src_dtype, WRITEBACKIFCOPY) which checks out.dtype can be SAFELY
            // cast to src.dtype — i.e. the user-supplied out must be at least as
            // narrow as src so the kernel's writes don't lose precision when later
            // written back. Matching error message:
            //   "Cannot cast array data from dtype('{out}') to dtype('{src}')
            //    according to the rule 'safe'".
            if (@out is null)
                return result;

            if (!@out.Shape.Equals(result.Shape))
                throw new ArgumentException(
                    $"output array does not match result of ndarray.take: expected shape {result.Shape}, got {@out.Shape}",
                    nameof(@out));

            if (@out.GetTypeCode != a.GetTypeCode &&
                !np.can_cast(@out.GetTypeCode, a.GetTypeCode, "safe"))
            {
                throw new TypeError(
                    $"Cannot cast array data from dtype('{@out.GetTypeCode.AsNumpyDtypeName()}') to dtype('{a.GetTypeCode.AsNumpyDtypeName()}') according to the rule 'safe'");
            }

            // A read-only out refuses LAST of the three out validations (probed 2.4.2: shape, then
            // castability, then this) — and with take's own quirky text: PyArray_TakeFrom wraps out
            // in PyArray_FromArray(..., NPY_ARRAY_WRITEBACKIFCOPY), whose writeability check names
            // the writeback base, not the "output array" the elementwise ufuncs name.
            NumSharpException.ThrowIfNotWriteable(@out.Shape, "WRITEBACKIFCOPY base");

            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>
        ///     Scalar convenience overload — take a single element by flat index.
        /// </summary>
        [NDScoped] // the 0-d index wrapper is built in THIS frame, before the array overload's scope opens
        public static NDArray take(NDArray a, long index, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            // Wrap the scalar in a 0-d NDArray so the array overload's shape-preserving
            // logic emits a 0-d result (NumPy semantic for scalar input).
            var idxArr = NDArray.Scalar(index);
            return take(a, idxArr, axis, @out, mode);
        }

        private static unsafe NDArray TakeFlat(NDArray a, NDArray indices, int mode)
        {
            // Materialise non-contig source to C-contig so the kernel can walk linearly.
            NDArray sourceOwned = null;
            NDArray source = a;
            if (!a.Shape.IsContiguous)
            {
                sourceOwned = np.ascontiguousarray(a);
                source = sourceOwned;
            }

            // Cast indices to contig int64 (a contig int32 array is read in place).
            var idx64 = CastIndicesToInt64(indices, "same_kind", out bool ownIdx, out bool idx32);

            // Output shape = indices.shape, dtype = a.dtype, C-contig allocated.
            var outShape = new Shape((long[])indices.Shape.dimensions.Clone());
            var result = new NDArray(a.typecode, outShape, false);

            long indicesCount = indices.size;
            long maxItem = a.size;
            long elemBytes = a.dtypesize;

            if (indicesCount == 0)
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
                return result;
            }

            if (maxItem == 0)
                throw new ArgumentException("cannot do a non-empty take from an empty array.", nameof(a));

            try
            {
                ExecuteTakeKernel(source, idx64, result,
                    outerSize: 1, indicesCount: indicesCount,
                    maxItem: maxItem, innerSize: elemBytes, mode: mode, idx32: idx32);
            }
            finally
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
            }

            return result;
        }

        private static unsafe NDArray TakeAxis(NDArray a, NDArray indices, int axis, int mode)
        {
            NDArray sourceOwned = null;
            NDArray source = a;
            if (!a.Shape.IsContiguous)
            {
                sourceOwned = np.ascontiguousarray(a);
                source = sourceOwned;
            }

            var idx64 = CastIndicesToInt64(indices, "same_kind", out bool ownIdx, out bool idx32);

            // Compute outer × inner factorisation around `axis`.
            long outerSize = 1;
            for (int d = 0; d < axis; d++) outerSize *= a.Shape.dimensions[d];
            long maxItem = a.Shape.dimensions[axis];
            long innerCount = 1;
            for (int d = axis + 1; d < a.ndim; d++) innerCount *= a.Shape.dimensions[d];
            long elemBytes = a.dtypesize;
            long innerBytes = innerCount * elemBytes;

            // Output shape = a.shape[:axis] + indices.shape + a.shape[axis+1:].
            int outNdim = a.ndim + indices.ndim - 1;
            var outDims = new long[outNdim];
            int outIdx = 0;
            for (int d = 0; d < axis; d++) outDims[outIdx++] = a.Shape.dimensions[d];
            for (int d = 0; d < indices.ndim; d++) outDims[outIdx++] = indices.Shape.dimensions[d];
            for (int d = axis + 1; d < a.ndim; d++) outDims[outIdx++] = a.Shape.dimensions[d];

            var result = new NDArray(a.typecode, new Shape(outDims), false);

            long indicesCount = indices.size;
            if (indicesCount == 0 || outerSize == 0)
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
                return result;
            }

            if (maxItem == 0)
                throw new ArgumentException("cannot do a non-empty take from an empty axis.", nameof(a));

            try
            {
                ExecuteTakeKernel(source, idx64, result,
                    outerSize: outerSize, indicesCount: indicesCount,
                    maxItem: maxItem, innerSize: innerBytes, mode: mode, idx32: idx32,
                    axis: axis);   // axis form: the raise-mode IndexError names this axis (NumPy wording)
            }
            finally
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
            }

            return result;
        }

        /// <summary>
        ///     Gathered-region footprint above which the take kernel emits software prefetch. Sized just
        ///     over typical per-core L2 (a 400 KB / 100 K-element source stays cache-resident and must
        ///     take the lean, prefetch-free kernel); a 40 MB / 10 M source clears it and gets prefetch.
        /// </summary>
        private const long IndexPrefetchThresholdBytes = 2L * 1024 * 1024;

        /// <summary>
        ///     Run the compiled take gather over <paramref name="source"/> into
        ///     <paramref name="result"/>, choosing between the lean flat kernel and the general
        ///     outer×inner kernel, and — under mode 'raise' — converting a kernel-reported
        ///     out-of-bounds index into NumPy's exact IndexError wording.
        /// </summary>
        /// <param name="source">The gathered array (read through raw pointer + offset).</param>
        /// <param name="idx64">The contiguous index array (int64 storage; reinterpreted as int32
        ///     values when <paramref name="idx32"/> is true).</param>
        /// <param name="result">Preallocated C-contiguous destination the kernel fills.</param>
        /// <param name="outerSize">Product of the dims before the take axis (1 for a flat take).</param>
        /// <param name="indicesCount">Number of indices gathered per outer position.</param>
        /// <param name="maxItem">The take axis's extent — the bound a raise-mode index must satisfy
        ///     after one negative wrap.</param>
        /// <param name="innerSize">Bytes per gathered slab (element bytes × trailing dims).</param>
        /// <param name="mode">0 = raise, 1 = wrap, 2 = clip (NumPy's mode order).</param>
        /// <param name="idx32">True when the index buffer holds int32 values read in place.</param>
        /// <param name="axis">The take axis as the CALLER spelled it, or null for a flat
        ///     (axis=None) take — load-bearing for the error text only: NumPy's raise-mode message
        ///     names the axis ("for axis 0 with size 8") on an axis take but drops the clause
        ///     entirely ("for size 8") on a flat one.</param>
        /// <exception cref="NotSupportedException">No IL kernel exists for this element width
        ///     (dynamic codegen unavailable).</exception>
        /// <exception cref="ArgumentOutOfRangeException">Mode 'raise' met an index still out of
        ///     [0, <paramref name="maxItem"/>) after the single negative wrap — message per
        ///     <paramref name="axis"/> above, and nothing has been partially written that the
        ///     caller exposes (the result buffer is discarded on throw).</exception>
        private static unsafe void ExecuteTakeKernel(
            NDArray source, NDArray idx64, NDArray result,
            long outerSize, long indicesCount, long maxItem, long innerSize, int mode, bool idx32 = false,
            int? axis = null)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(innerSize);
            // Software prefetch pays off only when the randomly-gathered region misses cache; below that
            // threshold it is pure per-element overhead (measured ~1.6x slower at 100K). Gate on the
            // gathered footprint = maxItem slabs of innerSize bytes.
            bool prefetch = maxItem * innerSize > IndexPrefetchThresholdBytes;

            byte* srcPtr = (byte*)source.Storage.Address + source.Shape.offset * source.dtypesize;
            // idx32: the index buffer holds int32 values; the kernels' `long*` parameter is a reinterpret.
            long* idxPtr = idx32
                ? (long*)((int*)idx64.Storage.Address + idx64.Shape.offset)
                : (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* dstPtr = (byte*)result.Storage.Address;

            long status;
            long expected = outerSize * indicesCount;
            // The lean flat gather (DirectILKernelGenerator.GatherFlat.cs) serves the single-slab-per-
            // index RAISE case — axis=None, or axis=0 with a primitive-width trailing slab — with a
            // compile-time element width and no per-element mode dispatch; wrap/clip and multi-outer
            // gathers keep the general kernel.
            var flat = (mode == 0 && outerSize == 1 && copyKind != 0)
                ? DirectILKernelGenerator.GetTakeFlatKernel(copyKind, idx32, prefetch)
                : null;
            if (flat != null)
            {
                status = flat(srcPtr, idxPtr, indicesCount, maxItem, dstPtr);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetTakeKernel(copyKind, prefetch, idx32);
                if (kernel == null)
                    throw new NotSupportedException("np.take: IL kernel unavailable");
                status = kernel(srcPtr, idxPtr, indicesCount, outerSize,
                                maxItem, innerSize, mode, dstPtr);
            }
            if (status < expected)
            {
                long failPair = status;
                long badJ = failPair % indicesCount;
                long badVal = idx32 ? ((int*)idxPtr)[badJ] : idxPtr[badJ];
                // NumPy's two IndexError spellings, verbatim (probed 2.4.2, gated by errors_full):
                // an axis take says "for axis {n} with size", the flat (axis=None) form drops the
                // axis clause entirely ("for size {n}") — the old single wording ("for axis with
                // size") matched neither.
                throw new ArgumentOutOfRangeException(
                    nameof(idx64),
                    axis.HasValue
                        ? $"index {badVal} is out of bounds for axis {axis.Value} with size {maxItem}"
                        : $"index {badVal} is out of bounds for size {maxItem}");
            }
        }

        /// <summary>
        ///     Casts <paramref name="indices"/> to contig int64 if needed. Sets
        ///     <paramref name="owned"/> to true when the returned array is a fresh
        ///     allocation that the caller must Dispose. The result shape matches
        ///     the input shape but always has contig strides.
        ///     <para>
        ///     NumPy converts an index array to <c>intp</c> under a fixed casting rule and raises
        ///     <see cref="TypeError"/> when it cannot — an integer or boolean index array is fine, but a
        ///     float/complex one is rejected rather than silently truncated. The rule differs by caller:
        ///     <c>take</c> uses <c>"same_kind"</c> (which also permits <c>uint64</c>) while <c>put</c>
        ///     uses <c>"safe"</c> (which rejects <c>uint64</c> too), and each leaks its own rule name in
        ///     the message — so <paramref name="castingRule"/> is threaded through verbatim.
        ///     </para>
        /// </summary>
        private static NDArray CastIndicesToInt64(NDArray indices, string castingRule, out bool owned)
            => CastIndicesToInt64(indices, castingRule, out owned, out _);

        /// <summary>
        ///     As <see cref="CastIndicesToInt64(NDArray, string, out bool)"/>, but a C-contiguous
        ///     <b>int32</b> index array is returned AS-IS with <paramref name="idx32"/> set: the take/put
        ///     kernels read int32 indices in place (their <c>idx32</c> variants), so the common C# index
        ///     (<c>np.array(new int[] {…})</c> is int32) no longer pays a widening copy — measured 27 % of
        ///     <c>np.take(a, idx32)</c> at 100K.
        /// </summary>
        private static NDArray CastIndicesToInt64(NDArray indices, string castingRule, out bool owned, out bool idx32)
        {
            var tc = indices.GetTypeCode;
            idx32 = false;

            if (tc == NPTypeCode.Int64 && indices.Shape.IsContiguous)
            {
                owned = false;
                return indices;
            }

            if (tc == NPTypeCode.Int32 && indices.Shape.IsContiguous)
            {
                owned = false;
                idx32 = true;
                return indices;
            }

            if (tc == NPTypeCode.Int64)
            {
                var c = np.ascontiguousarray(indices);
                owned = !ReferenceEquals(c, indices);
                return c;
            }

            if (!np.can_cast(tc, NPTypeCode.Int64, castingRule))
                throw new TypeError(
                    $"Cannot cast array data from dtype('{tc.AsNumpyDtypeName()}') to dtype('int64') according to the rule '{castingRule}'");

            owned = true;
            return indices.astype(NPTypeCode.Int64);
        }
    }
}
