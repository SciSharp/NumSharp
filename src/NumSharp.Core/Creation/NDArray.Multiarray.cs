using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The C functions of NumPy's multiarray module over arrays (Appendix D's NDArray.Multiarray.cs): np.concatenate's
    // PyArray_ConcatenateInto with its copy paths and its fixed-width string route. They are the array object's own members
    // (the np-layer dtype plan's D1: a function that joins arrays), and the np.* functions are facades over them.
    //
    // The file sits beside its facades (Creation/) rather than under Backends/ because its statements still call np.*
    // functions (np.result_type, np.can_cast), which ratchet R2 forbids under Backends/ (plan C3).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>PyArray_ConcatenateInto</c> (<c>multiarraymodule.c</c>) with <c>PyArray_ConcatenateArrays</c> and, for
        ///     <c>axis=None</c>, <c>PyArray_ConcatenateFlattenedArrays</c>: the array count, the casting name, the
        ///     <c>out</c> / <c>dtype</c> exclusion, the 0-d check, the ranks and extents, the result descriptor
        ///     (<paramref name="dtype"/>, else <paramref name="out"/>'s, else the inputs' <c>result_type</c>), each input's
        ///     cast check, the result's allocation in NumPy's layout, and the copy of every input into its slab (a memcpy, a
        ///     cast kernel, or the general copy). A fixed-width string anywhere takes the descriptor route
        ///     (<see cref="ConcatenateStrings"/>). The body of <see cref="np.concatenate(NDArray[], int?, NDArray, DType, string)"/>
        ///     once its class boundary and null check have run.
        /// </summary>
        /// <param name="arrays">The inputs, not null (an element may be: it is reported).</param>
        /// <param name="axis">The axis to join along, or null to join the flattened inputs.</param>
        /// <param name="out">The destination, or null to allocate the result.</param>
        /// <param name="dtype">The result dtype, or null.</param>
        /// <param name="casting">The casting rule's name.</param>
        /// <returns>The joined array: <paramref name="out"/> itself when given.</returns>
        /// <exception cref="ArgumentNullException">An element of <paramref name="arrays"/> is null.</exception>
        /// <exception cref="ArgumentException">An unknown <paramref name="casting"/> name; both <paramref name="out"/> and
        ///     <paramref name="dtype"/>.</exception>
        /// <exception cref="ValueError">No arrays, or a 0-d array with an axis.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for the inputs' rank.</exception>
        /// <exception cref="IncorrectShapeException">The inputs' ranks or non-axis extents differ, or <paramref name="out"/>
        ///     has the wrong rank or shape.</exception>
        /// <exception cref="InvalidCastException">An input does not cast to the result dtype under
        ///     <paramref name="casting"/>.</exception>
        /// <remarks>
        ///     The <c>axis=None</c> ravels are disposed here, exceptions included; a result this call allocated is returned or,
        ///     on a string input's error, disposed by the string route.
        /// </remarks>
        internal static NDArray ConcatenateInto(NDArray[] arrays, int? axis, NDArray @out, DType dtype, string casting)
        {
            // PyArray_ConcatenateInto: "need at least one array to concatenate", NumPy's ValueError.
            if (arrays.Length == 0)
                throw new ValueError("need at least one array to concatenate");

            // Validate casting name up-front. NumPy raises ValueError on invalid
            // casting strings regardless of whether any cast actually occurs.
            switch ((casting ?? string.Empty).ToLowerInvariant())
            {
                case "no":
                case "equiv":
                case "safe":
                case "same_kind":
                case "unsafe":
                    break;
                default:
                    throw new ArgumentException(
                        $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe', got '{casting}'",
                        nameof(casting));
            }

            // NumPy: out and dtype are mutually exclusive (raises TypeError).
            if (@out is not null && dtype is not null)
                throw new ArgumentException(
                    "concatenate() only takes `out` or `dtype` as an argument, but both were provided.");

            // 0-D arrays cannot be concatenated along an axis (NumPy: "zero-dimensional arrays cannot be concatenated");
            // axis=None ravels every input first (PyArray_ConcatenateFlattenedArrays), so a 0-d input is one element there
            // (np.concatenate((np.array(1), np.array(2)), axis=None) is [1, 2]).
            for (int k = 0; k < arrays.Length; k++)
            {
                if (arrays[k] is null)
                    throw new ArgumentNullException($"{nameof(arrays)}[{k}]");
                if (arrays[k].ndim == 0 && axis is not null)
                    throw new ValueError(
                        "zero-dimensional arrays cannot be concatenated");
            }

            // axis=None: flatten every input to 1-D, then concatenate along axis 0.
            // ravel() allocates a fresh NDArray wrapper per input (even when it returns a
            // view of the caller's storage). Without explicit cleanup those wrappers sit
            // on the finalizer queue, each holding an ARC ref to the caller's buffer.
            // We track the owning array in `disposableWorkArrays` and release in finally.
            NDArray[] workArrays = arrays;
            NDArray[] disposableWorkArrays = null;
            int effectiveAxis;
            if (axis is null)
            {
                disposableWorkArrays = new NDArray[arrays.Length];
                for (int k = 0; k < arrays.Length; k++)
                    disposableWorkArrays[k] = arrays[k].ravel();
                workArrays = disposableWorkArrays;
                effectiveAxis = 0;
            }
            else
            {
                effectiveAxis = axis.Value;
            }

            try
            {
            var first = workArrays[0];
            int ndim = first.ndim;

            // check_and_adjust_axis(&axis, ndim) of PyArray_ConcatenateArrays: NumPy's AxisError for an axis out of range,
            // the axis as given (axis=None ravels first, so its effective axis 0 always fits).
            effectiveAxis = DefaultEngine.check_and_adjust_axis(ndim, effectiveAxis);

            // Shape validation: all inputs must have same ndim and match on
            // non-axis dimensions. Accumulate axis size for the result.
            long axisSize = 0;
            var firstShape = (long[])first.Shape.dimensions.Clone();
            for (int k = 0; k < workArrays.Length; k++)
            {
                var src = workArrays[k];
                if (src.ndim != ndim)
                    throw new IncorrectShapeException(
                        $"all the input arrays must have same number of dimensions, " +
                        $"but the array at index 0 has {ndim} dimension(s) and the array at index {k} has {src.ndim} dimension(s)");

                var srcShape = src.Shape.dimensions;
                for (int j = 0; j < ndim; j++)
                {
                    if (j == effectiveAxis) continue;
                    if (srcShape[j] != firstShape[j])
                        throw new IncorrectShapeException(
                            $"all the input array dimensions except for the concatenation axis must match exactly, " +
                            $"but along dimension {j}, the array at index 0 has size {firstShape[j]} and the array at index {k} has size {srcShape[j]}");
                }

                axisSize += srcShape[effectiveAxis];
            }

            // Fixed-width strings resolve a DESCRIPTOR (the result's width is the widest input's) and copy through the
            // string route: np.concatenate.Strings.cs. The type-code fast paths below would treat a <U1 and a <U5 input as
            // one dtype.
            if (ConcatenateTouchesStrings(workArrays, @out, dtype))
            {
                firstShape[effectiveAxis] = axisSize;
                return ConcatenateStrings(workArrays, effectiveAxis, ndim, firstShape, @out, dtype, casting);
            }

            // Resolve result dtype: dtype= wins, then out=, then NEP50 promotion. The descriptor is kept beside the lane: a
            // lane-sharing C-type class (np.longdouble, Windows' np.intc) is not a storage code, and NumPy's
            // PyArray_ResultType names it for a fresh output (concatenate of longdouble arrays is longdouble).
            NPTypeCode resultType;
            DType resultDescr = null;
            if (dtype is not null)
            {
                resultType = dtype.GetTypeCode();
                resultDescr = dtype;
            }
            else if (@out is not null)
            {
                resultType = @out.GetTypeCode;
            }
            else
            {
                resultDescr = np.result_type(workArrays);
                resultType = resultDescr.GetTypeCode();
            }

            // Casting rule check: each input must cast to resultType under
            // the requested casting mode. NumPy raises TypeError; we raise
            // InvalidCastException with the NumPy-style message.
            for (int k = 0; k < workArrays.Length; k++)
            {
                var srcType = workArrays[k].GetTypeCode;
                if (srcType == resultType) continue;
                if (!np.can_cast(srcType, resultType, casting))
                    throw new InvalidCastException(
                        $"Cannot cast array data from dtype('{srcType.AsNumpyDtypeName()}') " +
                        $"to dtype('{resultType.AsNumpyDtypeName()}') according to the rule '{casting}'");
            }

            // Compute the output shape.
            firstShape[effectiveAxis] = axisSize;

            // Allocate or validate output.
            NDArray dst;
            if (@out is not null)
            {
                if (@out.ndim != ndim)
                    throw new IncorrectShapeException(
                        "Output array has wrong dimensionality");
                for (int j = 0; j < ndim; j++)
                {
                    if (@out.Shape.dimensions[j] != firstShape[j])
                        throw new IncorrectShapeException(
                            "Output array is the wrong shape");
                }
                dst = @out;
            }
            else
            {
                // NumPy lays the fresh output out along the multi-operand KEEPORDER stride
                // vote (PyArray_CreateMultiSortedStridePerm, ported on Shape): all-C
                // inputs give C; all-F inputs that are not also C give F (both-C&F operands,
                // e.g. (N,1)/(1,N) views, never vote through their size-1 axis and resolve to
                // C); and voting through size-1 axes is what makes np.stack of F-contiguous
                // (3,4) operands allocate the NEITHER-contiguous byte strides (96,8,24) —
                // probed NumPy 2.4.2. The general copy path below writes through dst slice
                // views, so any perm layout fills correctly; the memcpy/cast fast paths gate
                // on contiguity and simply decline the exotic ones.
                var inputShapes = new Shape[workArrays.Length];
                for (int k = 0; k < workArrays.Length; k++)
                    inputShapes[k] = workArrays[k].Shape;
                var retShape = Shape.ConcatOutputShape(firstShape, inputShapes);
                // fillZeros: false — every byte is overwritten below.
                dst = new NDArray(resultType, retShape, fillZeros: false);
                // The fresh output takes the result's lane-sharing C-type class, if it is one (a no-op otherwise).
                dst.Storage.AdoptLaneSharingDescr(resultDescr);
            }

            // Layered fast paths:
            // 1. TryDirectMemcpyConcat -- all sources same dtype as dst and
            //    matching layout (C/F): direct Buffer.MemoryCopy per outer
            //    slab, no slice/state construction.
            // 2. TryDirectCastConcat -- all sources contig, dst contig,
            //    mixed dtypes: drive the IL contig cast kernel per source
            //    with computed offsets.
            // 3. General path via NDIter.Copy -- broadcasted sources,
            //    exotic dtype pairs, mixed C/F layouts. NDIter's K-order
            //    axis permutation (added to CreateCopyState) ensures the
            //    unit-stride axis ends up innermost so the IL strided
            //    cast kernel's inner-contig branch fires even for F-contig
            //    sliced dsts. Without that, the strided path took ~17x
            //    longer than the fast paths on 1M F-contig inputs.
            //
            // The fast paths still win ~50-90% on workloads they cover
            // because they skip the dst[axis_range] slice creation and
            // the per-source NDIter state construction -- savings that
            // amortize over many small calls (count_1024) or compound
            // across many copy operations.
            if (TryDirectMemcpyConcat(dst, workArrays, effectiveAxis, ndim, resultType))
                return dst;

            if (TryDirectCastConcat(dst, workArrays, effectiveAxis, ndim, resultType))
                return dst;

            // General path: NDIter.Copy for anything the fast paths skip.
            var dstAccessor = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                dstAccessor[i] = Slice.All;

            long dstAxisPos = 0;
            for (int k = 0; k < workArrays.Length; k++)
            {
                var src = workArrays[k];
                var len = src.Shape.dimensions[effectiveAxis];
                if (len == 0) continue; // skip empty inputs along the concat axis

                dstAccessor[effectiveAxis] = new Slice(dstAxisPos, dstAxisPos + len);
                // dstSlice is an owning intermediate (sliced view of dst — a fresh NDArray
                // wrapper sharing storage). Releasing each iteration's wrapper atomically
                // keeps N-source loops from queueing N dead wrappers per concatenate call.
                using var dstSlice = dst[dstAccessor];
                NDIter.Copy(dstSlice, src);

                dstAxisPos += len;
            }

            return dst;
            }
            finally
            {
                if (disposableWorkArrays != null)
                {
                    for (int k = 0; k < disposableWorkArrays.Length; k++)
                        disposableWorkArrays[k]?.Dispose();
                }
            }
        }

        /// <summary>
        ///     Same-dtype fast path: when all sources match the destination
        ///     dtype and every operand (sources + dst) is C-contiguous (or
        ///     all F-contiguous), perform a direct <see cref="Buffer.MemoryCopy"/>
        ///     per outer block. Skips the <c>dst[axis_range]</c> slice which
        ///     would produce a non-contig view for F-contig dst (see the
        ///     justification block in <see cref="ConcatenateInto"/>).
        /// </summary>
        /// <returns>True if the fast path applied and the copy was performed.</returns>
        private static unsafe bool TryDirectMemcpyConcat(
            NDArray dst, NDArray[] sources, int axis, int ndim, NPTypeCode resultType)
        {
            if (!dst.Shape.IsWriteable)
                return false;

            // Try matching layouts: C-contig sources + C-contig dst, or
            // F-contig sources + F-contig dst. Mixed layouts go through the
            // general path.
            bool cPath = dst.Shape.IsContiguous;
            bool fPath = !cPath && dst.Shape.IsFContiguous;

            if (!cPath && !fPath)
                return false;

            for (int k = 0; k < sources.Length; k++)
            {
                var s = sources[k];
                if (s.GetTypeCode != resultType) return false;
                if (s.Shape.IsBroadcasted) return false;
                if (cPath)
                {
                    if (!s.Shape.IsContiguous) return false;
                }
                else // fPath
                {
                    if (!s.Shape.IsFContiguous) return false;
                }
            }

            int elemSize = resultType.SizeOf();

            // For C-contig:
            //   outerCount = product of dims [0..axis-1]   (slower-varying axes)
            //   innerStride = product of dims [axis+1..ndim-1] (faster-varying axes)
            // For F-contig (mirror image):
            //   outerCount = product of dims [axis+1..ndim-1]  (slower-varying axes in F-order)
            //   innerStride = product of dims [0..axis-1]      (faster-varying axes in F-order)
            long outerCount = 1, innerStride = 1;
            if (cPath)
            {
                for (int i = 0; i < axis; i++) outerCount *= dst.Shape.dimensions[i];
                for (int i = axis + 1; i < ndim; i++) innerStride *= dst.Shape.dimensions[i];
            }
            else
            {
                for (int i = axis + 1; i < ndim; i++) outerCount *= dst.Shape.dimensions[i];
                for (int i = 0; i < axis; i++) innerStride *= dst.Shape.dimensions[i];
            }

            // Per-outer dst row size in elements = sum of src.shape[axis] * innerStride.
            // Equivalent to dst.shape[axis] * innerStride.
            long dstRowSize = dst.Shape.dimensions[axis] * innerStride;
            long dstRowBytes = dstRowSize * elemSize;

            // Degenerate-slab specialization: (N,1)-style concats (column_stack
            // of 1-D inputs) shrink each per-outer slab to a handful of bytes —
            // the generic loop below would issue outerCount*nsrc dynamic-size
            // Buffer.MemoryCopy calls (~4.5ns each through Memmove dispatch;
            // 2M of them for two 1M columns). When every slab is the same tiny
            // size, an interleave loop with CONSTANT-size stores runs ~10-20x
            // faster (~0.2ns/copy) because the JIT reduces each copy to a
            // single fixed-width load/store pair. No setup cost beyond the
            // O(nsrc) uniformity check, so it applies at every outerCount.
            if (TryUniformTinySlabInterleave(
                    dst, sources, axis, innerStride, elemSize, outerCount, dstRowBytes))
                return true;

            byte* dstBase = dst.Storage.Address;
            // Account for sliced/aliased output via Shape.offset (offset is in elements for unmanaged storage).
            long dstOffsetBytes = dst.Shape.Offset * elemSize;

            // Pre-compute each source's per-outer slab size in bytes.
            // For axis=0, outerCount==1 so this is the entire source.
            // For axis=last, slabBytes is one element-row worth.
            long[] slabBytes = new long[sources.Length];
            byte*[] srcBases = new byte*[sources.Length];
            long[] srcOffsetBytes = new long[sources.Length];
            for (int k = 0; k < sources.Length; k++)
            {
                slabBytes[k] = sources[k].Shape.dimensions[axis] * innerStride * elemSize;
                srcBases[k] = sources[k].Storage.Address;
                srcOffsetBytes[k] = sources[k].Shape.Offset * elemSize;
            }

            // Walk outer iterations, copying each source's slab into the dst row.
            for (long outer = 0; outer < outerCount; outer++)
            {
                long dstRowStart = dstOffsetBytes + outer * dstRowBytes;
                long dstWritePos = 0;
                for (int k = 0; k < sources.Length; k++)
                {
                    long bytes = slabBytes[k];
                    if (bytes == 0) continue;
                    Buffer.MemoryCopy(
                        source: srcBases[k] + srcOffsetBytes[k] + outer * bytes,
                        destination: dstBase + dstRowStart + dstWritePos,
                        destinationSizeInBytes: bytes,
                        sourceBytesToCopy: bytes);
                    dstWritePos += bytes;
                }
            }

            return true;
        }

        /// <summary>
        ///     Interleave specialization for <see cref="TryDirectMemcpyConcat"/>:
        ///     when every source contributes the SAME tiny per-outer slab
        ///     (1/2/4/8/16 bytes — the (N,1) column pattern), copy via
        ///     constant-size loads/stores instead of dynamic-size
        ///     <see cref="Buffer.MemoryCopy"/>. The o-outer order keeps dst
        ///     writes sequential (write-combining friendly).
        /// </summary>
        /// <returns>True if the specialization applied and copied everything.</returns>
        private static unsafe bool TryUniformTinySlabInterleave(
            NDArray dst, NDArray[] sources, int axis, long innerStride,
            int elemSize, long outerCount, long dstRowBytes)
        {
            int nsrc = sources.Length;

            long slab = sources[0].Shape.dimensions[axis] * innerStride * elemSize;
            if (slab != 1 && slab != 2 && slab != 4 && slab != 8 && slab != 16)
                return false;
            for (int k = 1; k < nsrc; k++)
                if (sources[k].Shape.dimensions[axis] * innerStride * elemSize != slab)
                    return false;

            byte* dRow = dst.Storage.Address + dst.Shape.Offset * elemSize;
            byte** srcPtrs = stackalloc byte*[nsrc];
            for (int k = 0; k < nsrc; k++)
                srcPtrs[k] = sources[k].Storage.Address + sources[k].Shape.Offset * elemSize;

            switch (slab)
            {
                case 1:
                    for (long o = 0; o < outerCount; o++)
                    {
                        byte* d = dRow + o * dstRowBytes;
                        for (int k = 0; k < nsrc; k++)
                            d[k] = srcPtrs[k][o];
                    }

                    break;
                case 2:
                    for (long o = 0; o < outerCount; o++)
                    {
                        byte* d = dRow + o * dstRowBytes;
                        for (int k = 0; k < nsrc; k++)
                            *(ushort*)(d + k * 2) = *(ushort*)(srcPtrs[k] + o * 2);
                    }

                    break;
                case 4:
                    for (long o = 0; o < outerCount; o++)
                    {
                        byte* d = dRow + o * dstRowBytes;
                        for (int k = 0; k < nsrc; k++)
                            *(uint*)(d + k * 4) = *(uint*)(srcPtrs[k] + o * 4);
                    }

                    break;
                case 8:
                    for (long o = 0; o < outerCount; o++)
                    {
                        byte* d = dRow + o * dstRowBytes;
                        for (int k = 0; k < nsrc; k++)
                            *(ulong*)(d + k * 8) = *(ulong*)(srcPtrs[k] + o * 8);
                    }

                    break;
                case 16:
                    for (long o = 0; o < outerCount; o++)
                    {
                        byte* d = dRow + o * dstRowBytes;
                        for (int k = 0; k < nsrc; k++)
                        {
                            var s = (ulong*)(srcPtrs[k] + o * 16);
                            var t = (ulong*)(d + k * 16);
                            t[0] = s[0];
                            t[1] = s[1];
                        }
                    }

                    break;
            }

            return true;
        }

        /// <summary>
        ///     Cross-dtype fast path: same shape/layout assumptions as
        ///     <see cref="TryDirectMemcpyConcat"/>, but drives the IL-generated
        ///     contig cast kernel per source instead of <see cref="Buffer.MemoryCopy"/>.
        ///     Avoids the slice-then-NDIter detour for the same F-contig
        ///     reason (slice produces non-contig view, NDIter falls into
        ///     strided cast path, ~17x slower for F-contig).
        /// </summary>
        private static unsafe bool TryDirectCastConcat(
            NDArray dst, NDArray[] sources, int axis, int ndim, NPTypeCode resultType)
        {
            if (!dst.Shape.IsWriteable)
                return false;

            bool cPath = dst.Shape.IsContiguous;
            bool fPath = !cPath && dst.Shape.IsFContiguous;
            if (!cPath && !fPath)
                return false;

            for (int k = 0; k < sources.Length; k++)
            {
                if (sources[k].Shape.IsBroadcasted) return false;
                if (cPath)
                {
                    if (!sources[k].Shape.IsContiguous) return false;
                }
                else
                {
                    if (!sources[k].Shape.IsFContiguous) return false;
                }
            }

            // Resolve a cast kernel for every distinct (src dtype → resultType).
            // Bail to the general path if any pair is unsupported.
            var kernels = new Backends.Kernels.DirectILKernelGenerator.CastKernel[sources.Length];
            for (int k = 0; k < sources.Length; k++)
            {
                if (sources[k].GetTypeCode == resultType)
                {
                    kernels[k] = null; // sentinel for "use memcpy"
                }
                else
                {
                    var kk = Backends.Kernels.DirectILKernelGenerator
                        .TryGetCastKernel(sources[k].GetTypeCode, resultType);
                    if (kk is null) return false;
                    kernels[k] = kk;
                }
            }

            int dstElemSize = resultType.SizeOf();

            long outerCount = 1, innerStride = 1;
            if (cPath)
            {
                for (int i = 0; i < axis; i++) outerCount *= dst.Shape.dimensions[i];
                for (int i = axis + 1; i < ndim; i++) innerStride *= dst.Shape.dimensions[i];
            }
            else
            {
                for (int i = axis + 1; i < ndim; i++) outerCount *= dst.Shape.dimensions[i];
                for (int i = 0; i < axis; i++) innerStride *= dst.Shape.dimensions[i];
            }

            // Same degenerate-slab guard as TryDirectMemcpyConcat: tiny per-outer
            // slabs across many rows lose to the general strided-kernel path.
            if (outerCount >= 1024)
            {
                long maxSlabBytes = 0;
                for (int k = 0; k < sources.Length; k++)
                {
                    long sb = sources[k].Shape.dimensions[axis] * innerStride * dstElemSize;
                    if (sb > maxSlabBytes)
                        maxSlabBytes = sb;
                }

                if (maxSlabBytes <= 16)
                    return false;
            }

            long dstRowElems = dst.Shape.dimensions[axis] * innerStride;
            long dstRowBytes = dstRowElems * dstElemSize;

            byte* dstBase = dst.Storage.Address;
            long dstOffsetBytes = dst.Shape.Offset * dstElemSize;

            // Per-source: element count per outer slab + base/byte offsets.
            long[] slabElems = new long[sources.Length];
            long[] slabBytesDst = new long[sources.Length];
            int[] srcElemSize = new int[sources.Length];
            byte*[] srcBases = new byte*[sources.Length];
            long[] srcOffsetBytes = new long[sources.Length];
            for (int k = 0; k < sources.Length; k++)
            {
                slabElems[k] = sources[k].Shape.dimensions[axis] * innerStride;
                srcElemSize[k] = sources[k].GetTypeCode.SizeOf();
                slabBytesDst[k] = slabElems[k] * dstElemSize;
                srcBases[k] = sources[k].Storage.Address;
                srcOffsetBytes[k] = sources[k].Shape.Offset * srcElemSize[k];
            }

            for (long outer = 0; outer < outerCount; outer++)
            {
                long dstRowStart = dstOffsetBytes + outer * dstRowBytes;
                long dstWritePos = 0;
                for (int k = 0; k < sources.Length; k++)
                {
                    long elems = slabElems[k];
                    if (elems == 0) continue;

                    byte* srcPtr = srcBases[k] + srcOffsetBytes[k] + outer * elems * srcElemSize[k];
                    byte* dstPtr = dstBase + dstRowStart + dstWritePos;

                    var kernel = kernels[k];
                    if (kernel is null)
                    {
                        Buffer.MemoryCopy(
                            source: srcPtr,
                            destination: dstPtr,
                            destinationSizeInBytes: slabBytesDst[k],
                            sourceBytesToCopy: slabBytesDst[k]);
                    }
                    else
                    {
                        kernel(srcPtr, dstPtr, elems);
                    }

                    dstWritePos += slabBytesDst[k];
                }
            }

            return true;
        }

        /// <summary>
        ///     Whether a concatenation involves a fixed-width string dtype — an input, the <c>out</c> array or the requested
        ///     <c>dtype</c> — and so must resolve its result by DESCRIPTOR.
        /// </summary>
        /// <param name="arrays">The (validated) inputs.</param>
        /// <param name="out">The output array, or null.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>True when the string route applies.</returns>
        private static bool ConcatenateTouchesStrings(NDArray[] arrays, NDArray @out, DType dtype)
        {
            if (dtype?.Meta is StringDTypeMeta || @out?.typecode == NPTypeCode.String)
                return true;
            foreach (var a in arrays)
                if (a.typecode == NPTypeCode.String)
                    return true;
            return false;
        }

        /// <summary>
        ///     <c>np.concatenate</c> when a string dtype is involved — NumPy's <c>PyArray_ConcatenateInto</c> over
        ///     descriptors: the result descriptor is <paramref name="dtype"/> adapted to the inputs
        ///     (<c>PyArray_FindConcatenationDescriptor</c>: an unsized <c>'U'</c> takes the widest input's length), else
        ///     <paramref name="out"/>'s, else the inputs' <c>result_type</c> (<c>U3 + S5 → &lt;U5</c>); every input must cast
        ///     to it under <paramref name="casting"/>; each input is then copied into its slab of the result through the string
        ///     copy route (truncation, NUL padding, <c>S ↔ U</c> ASCII checks).
        /// </summary>
        /// <param name="arrays">The inputs (same rank, matching non-axis extents — already validated).</param>
        /// <param name="axis">The normalized concatenation axis.</param>
        /// <param name="ndim">The inputs' rank.</param>
        /// <param name="resultDims">The result dimensions.</param>
        /// <param name="out">The output array, or null to allocate.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <param name="casting">The casting rule name (validated by the caller).</param>
        /// <returns>The concatenated array (<paramref name="out"/> when given).</returns>
        /// <exception cref="InvalidCastException">An input cannot cast to the result descriptor under <paramref name="casting"/>
        ///     (NumPy's TypeError text: <c>Cannot cast array data from dtype('&lt;U1') to dtype('S3') according to the rule
        ///     'same_kind'</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> has the wrong rank or shape.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> input is not ASCII and the result is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> input is not ASCII and the result is <c>U</c>.</exception>
        /// <exception cref="ValueError">A text input does not parse into a numeric result (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts).</exception>
        /// <exception cref="OverflowException">An integer text input does not fit a numeric result.</exception>
        /// <remarks>
        ///     <b>Order (NumPy's <c>PyArray_ConcatenateArrays</c>).</b> The <paramref name="out"/> shape first; then each input
        ///     in turn through <c>PyArray_AssignArray</c> — its casting rule, then its conversion — so the FIRST input's rule or
        ///     conversion error wins over a later input's (a non-ASCII <c>S</c> input before a <c>U5</c> input that fails
        ///     <c>'safe'</c> into <c>&lt;U3</c> reports the UnicodeDecodeError). A numeric input into a text result is written
        ///     as its numbers' <c>str()</c>; a text input into a numeric result is parsed. An EMPTY input converts nothing.
        /// </remarks>
        private static NDArray ConcatenateStrings(NDArray[] arrays, int axis, int ndim, long[] resultDims, NDArray @out, DType dtype, string casting)
        {
            var rule = DTypeCasting.ParseCasting((casting ?? string.Empty).ToLowerInvariant());

            DType result;
            if (dtype is not null)
                result = FindConcatenationDescriptor(arrays, dtype);
            else if (@out is not null)
                result = @out.dtype;
            else
                result = DTypePromotion.ResultType(arrays.Cast<object>().ToArray());

            NDArray dst;
            if (@out is not null)
            {
                if (@out.ndim != ndim)
                    throw new IncorrectShapeException("Output array has wrong dimensionality");
                for (int j = 0; j < ndim; j++)
                {
                    if (@out.Shape.dimensions[j] != resultDims[j])
                        throw new IncorrectShapeException(
                            "Output array is the wrong shape");
                }
                dst = @out;
            }
            else
            {
                // NumPy's multi-operand KEEPORDER layout vote, as for every dtype.
                var inputShapes = new Shape[arrays.Length];
                for (int k = 0; k < arrays.Length; k++)
                    inputShapes[k] = arrays[k].Shape;
                // The pieces cover the result exactly, so a U result skips the NEEDS_INIT zero fill. (A deferred numeric
                // input leaves its slab unwritten, but then the result is discarded below.)
                dst = NDArray.ForOverwrite(result, Shape.ConcatOutputShape(resultDims, inputShapes));
            }

            try
            {
                var accessor = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    accessor[i] = Slice.All;

                long position = 0;
                foreach (var src in arrays)
                {
                    // PyArray_AssignArray, input by input: the casting rule (a dtype check, so it runs for an empty input
                    // too), then the conversion — a number written as its scalar's str(), a text parsed (the copy route's
                    // NDTextConversion), so a parse error is reported at its input, after the inputs before it were written.
                    if (!DTypeCasting.CanCastTypeTo(src.dtype, result, rule))
                        throw NDStringCasting.InvalidCast(src.dtype, result, rule, scalar: false);

                    long len = src.Shape.dimensions[axis];
                    if (len == 0)
                        continue;
                    accessor[axis] = new Slice(position, position + len);
                    // The slab view is a fresh wrapper over dst's storage: release it each iteration.
                    using var slab = dst[accessor];
                    NDIter.Copy(slab.Storage, src.Storage);
                    position += len;
                }
                return dst;
            }
            catch
            {
                // A failing input discards the result NumSharp allocated (a caller's out keeps what was written, as in NumPy).
                if (@out is null)
                    dst.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FindConcatenationDescriptor</c>: a SIZED requested descriptor is used as given; an unsized
        ///     string class (<c>'U'</c>, <c>'S'</c>) becomes the common instance of every input cast to it (the widest).
        /// </summary>
        /// <param name="arrays">The inputs.</param>
        /// <param name="requested">The requested dtype.</param>
        /// <returns>The result descriptor.</returns>
        /// <exception cref="TypeError">An input has no cast to the requested class.</exception>
        private static DType FindConcatenationDescriptor(NDArray[] arrays, DType requested)
        {
            if (!(requested.itemsize == 0 && requested.Meta is StringDTypeMeta))
                return requested;
            return DTypePromotion.CastToDTypeAndPromoteDescriptors(arrays.Select(a => a.dtype).ToArray(), requested.Meta);
        }
    }
}
