using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The C functions of NumPy's multiarray module over arrays (Appendix D's NDArray.Multiarray.cs): np.concatenate's
    // PyArray_ConcatenateInto with its copy paths and its fixed-width string route, and _array_fromobject_generic, the body
    // of np.array / np.asarray / np.asanyarray, by operand kind (an array, a C# array, any C# value). They are the array
    // object's own members (the np-layer dtype plan's D1: functions that create and join arrays), and the np.* functions
    // are facades over them.
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

        /// <summary>
        ///     NumPy's <c>_array_fromobject_generic</c> (<c>multiarraymodule.c</c>) for an ARRAY operand, the body of
        ///     <see cref="np.asarray(NDArray, DType, char, bool?, NDArray, string)"/> once the class boundaries, the device and
        ///     <c>like=</c> were handled: Python's <c>None</c> (a new 0-d array of the dtype's setitem value), a C# literal
        ///     packed as the Python scalar it stands for (<c>PyArray_Pack</c>, with its range checks), then the request - the
        ///     array itself when its dtype and layout satisfy it, a view when the requested descriptor is an equal but
        ///     distinct object, else a copy or a cast in the requested order, which <c>copy=false</c> refuses.
        /// </summary>
        /// <param name="a">The operand; null is Python's <c>None</c>, and a C# literal's 0-d array
        ///     (<see cref="NDArray.WasPythonLiteral"/>) is NumPy's Python scalar.</param>
        /// <param name="dtype">The requested dtype, or null to keep the operand's.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>: only C and F can force a copy.</param>
        /// <param name="copy">NumPy's tri-state: null copies only when needed, true always, false never (raising when
        ///     needed).</param>
        /// <returns><paramref name="a"/> itself, a view of it, or a new array.</returns>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null and <paramref name="dtype"/> is null (NumPy
        ///     builds an object array, which NumSharp has no storage for).</exception>
        /// <exception cref="ArgumentException"><paramref name="copy"/> is false and a copy, a cast or a literal's / None's
        ///     conversion is needed (NumPy's "Unable to avoid copy" ValueError).</exception>
        /// <exception cref="OverflowException">A C# literal out of an integer <paramref name="dtype"/>'s range.</exception>
        /// <exception cref="ValueError">A NaN literal into an integer dtype, or an unknown <paramref name="order"/>.</exception>
        /// <exception cref="TypeError">A complex literal into a real dtype, or None into an integer (or decimal) one.</exception>
        /// <remarks>An explicit 0-d array (<see cref="NDArray.IsExplicitZeroDimArray"/>) stays one through the copy, the cast
        ///     and the view: the mark is per object, so every new object receives it.</remarks>
        internal static NDArray ArrayFromObject(NDArray a, DType dtype, char order, bool? copy)
        {
            // Python's None: a Python object, so np.asarray(None, dtype) builds a NEW 0-d array of its setitem value (NaN,
            // nan+nanj, False, 'None', a StringDType NA object; int(None)'s TypeError) — which copy=false refuses before
            // anything is converted (PyArray_FromAny's NPY_ARRAY_ENSURENOCOPY check precedes the pack). Without a dtype
            // NumPy's discovery gives the object dtype.
            if (a is null)
            {
                if (dtype is null)
                    throw NDArray.NoneWithoutDtype("np.asarray");
                if (copy == false)
                    throw new ArgumentException(UnableToAvoidCopyText, nameof(copy));
                return NDArray.NoneForStore(dtype);
            }

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar, which np.asarray(value, dtype=...) converts
            // with PyArray_Pack into a NEW array: np.asarray(1000, dtype=np.int8) raises "Python integer 1000 out of bounds
            // for int8" and np.asarray(1.5, dtype=np.int8) is 1 — where a 0-d ndarray would be cast. A fresh array can never
            // satisfy copy=false (NumPy's "Unable to avoid copy" for any Python scalar).
            if (a.WasPythonLiteral && dtype is not null && !Backends.Strings.NDStringStorage.IsString(dtype))
            {
                var packed = NDArray.PackForStore(a, dtype.GetTypeCode(), pyArrayPackRoute: false);
                if (!ReferenceEquals(packed, a))
                {
                    if (copy == false)
                    {
                        packed.Dispose();
                        throw new ArgumentException(UnableToAvoidCopyText, nameof(copy));
                    }

                    return packed;
                }
            }

            // A string request is satisfied only by a string array of that class AND width (an unsized 'U' adapts to the
            // array first); the storage lane alone would let a <U4 array pass for 'S' or 'U2'.
            bool typeMatches = dtype == null || Backends.UnmanagedStorage.SatisfiesRequest(a, dtype);
            bool layoutMatches = LayoutAlreadyOK(a.Shape, order);
            bool noCopyNeeded = typeMatches && layoutMatches;

            // A copy or a cast of an explicit 0-d ARRAY is an explicit 0-d array again (NumPy returns a 0-d ndarray
            // from np.asarray(np.array(2), dtype=float)); the mark is per object, so it is carried over by hand.
            bool explicitZeroDim = a.IsExplicitZeroDimArray;

            // The copies keep NumPy's order code as given: astype / copy resolve 'K' to the source's own axis order
            // (a transposed or broadcast source keeps its layout), which resolving to C-or-F first would lose.
            if (copy == true)
            {
                NDArray copied = !typeMatches ? a.astype(dtype, copy: true, order: order) : a.copy(order);
                return explicitZeroDim ? copied.MarkExplicitZeroDimArray() : copied;
            }

            if (copy == false && !noCopyNeeded)
                throw new ArgumentException(UnableToAvoidCopyText, nameof(copy));

            // NumPy (_array_fromobject_generic) returns the array itself when no dtype was given or the requested descriptor
            // IS the array's — a builtin singleton, or an unsized string class, which adapts to the array's own descriptor;
            // an EQUAL descriptor that is another object (every sized string descriptor is a fresh object) gets a view of
            // the array with that descriptor: np.asarray(u, dtype='<U2') shares u's memory but is not u (owndata false).
            if (noCopyNeeded)
            {
                if (dtype is null || ReferenceEquals(dtype, a.dtype) || dtype.itemsize == 0)
                    return a;
                // The view is a new object, and the explicit-0-d mark is per object: NumPy's view of an explicit 0-d array
                // is an explicit 0-d array too, so the mark is carried over like on the copy paths.
                var view = new NDArray(a.Storage.Alias(a.Shape)) { TensorEngine = a.TensorEngine };
                return explicitZeroDim ? view.MarkExplicitZeroDimArray() : view;
            }

            // copy == null && !noCopyNeeded: copy / cast as required.
            NDArray converted = !typeMatches ? a.astype(dtype, copy: true, order: order) : a.copy(order);
            return explicitZeroDim ? converted.MarkExplicitZeroDimArray() : converted;
        }

        /// <summary>
        ///     NumPy's <c>_array_fromobject_generic</c> for a C# ARRAY operand, the body of
        ///     <see cref="np.array(Array, DType, int, bool, char)"/> once its null check ran: text values (or a string dtype
        ///     over an object or bytes array, or None among an object array's values under a dtype) take the text array
        ///     builder (the class and width discovered, every element stored by its setitem); any other object array is a
        ///     Python list (<see cref="ArrayFromPythonList"/>: NumPy's coercion of a sequence); a string dtype over a typed
        ///     numeric array casts the numbers to text (NumPy's <c>PyArray_AdaptDescriptorToArray</c>); any other array gives
        ///     its shape and element type, is flattened and converted to the requested dtype, laid out in C order (or F on
        ///     request) and raised to <paramref name="ndmin"/> dimensions.
        /// </summary>
        /// <param name="array">The C# array (any rank, rectangular or jagged); not null.</param>
        /// <param name="dtype">The requested dtype, or null to discover it.</param>
        /// <param name="ndmin">The minimum number of dimensions: leading ones are prepended.</param>
        /// <param name="copy">False lets a rank-1 array of the requested element type be wrapped instead of copied.</param>
        /// <param name="order">The memory layout: <c>'C'</c>, <c>'F'</c>, or <c>'A'</c> / <c>'K'</c>.</param>
        /// <returns>A new array (a view of one when <paramref name="ndmin"/> prepends axes to a text result, as NumPy's).</returns>
        /// <exception cref="ValueError">A ragged jagged array or object list (NumPy's inhomogeneous-shape text), or a value
        ///     the requested dtype's setitem refuses.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="OverflowException">A Python int (a C# integer in an object list) outside the requested integer
        ///     dtype.</exception>
        /// <exception cref="TypeError">A value of an object list the requested dtype cannot convert (None into an
        ///     integer dtype, a complex number into a real one).</exception>
        /// <exception cref="NotSupportedException">An element type NumSharp has no dtype for, or None (or a Python int past
        ///     64 bits) in an object list without a dtype: NumPy's object array.</exception>
        /// <exception cref="InvalidCastException">An element that does not convert to <paramref name="dtype"/>.</exception>
        internal static NDArray ArrayFromManaged(Array array, DType dtype, int ndmin, bool copy, char order)
        {
            // PyArray_OrderConverter runs while NumPy parses the arguments: an invalid order raises before any element is
            // converted (a value the dtype refuses would otherwise be reported first).
            OrderResolver.ResolveBufferOrder(order);
            // Text values (string elements, an object array holding text, or a string dtype over object / byte[][] data)
            // build a fixed-width S/U array: NumPy's discovery of the class and the longest value, then setitem per element.
            if (Backends.UnmanagedStorage.IsTextInput(array, dtype))
                return Backends.UnmanagedStorage.FromManagedText(array, dtype, ndmin, order);
            // A C# object array (object[], a jagged object[][], an object grid) is NumPy's Python list in the boundary map:
            // the coercion of a sequence discovers its shape and dtype (a C# int is a Python int: int64), or stores every item
            // through the requested descriptor (NumPy's PyArray_Pack: 300 into int8 raises).
            if (HoldsPythonObjects(array))
                return ArrayFromPythonList(array, dtype, ndmin, order);
            // A string dtype over a TYPED numeric C# array: the array is an ndarray of its element dtype (the house rule for
            // typed arrays), so NumPy's np.array(arr, dtype='U') casts it — the unsized class adapted to the array
            // (PyArray_AdaptDescriptorToArray: int32 -> <U11), each number written as its scalar's str().
            if (Backends.Strings.NDStringStorage.IsString(dtype))
            {
                using var numbers = np.array(array, null, 0, true, order);
                var text = numbers.astype(dtype, copy: true, order: 'K', casting: "unsafe");
                if (ndmin <= text.ndim)
                    return text;
                var raised = np.AtLeastNdView(text, ndmin);
                text.Dispose();
                return raised;
            }

            var arrType = array.ResolveElementType();

            //handle dim expansion and extract shape
            Shape shape;
            var dims = array.ResolveRank();
            var missing = dims - ndmin;

            if (missing < 0)
            {
                shape = Arrays.Concat(Enumerable.Repeat(1, Math.Abs(missing)).ToArray(), Shape.ExtractShape(array));
            }
            else
            {
                shape = Shape.ExtractShape(array);
            }

            //flatten
            if (shape.NDim > 1)
            {
                array = Arrays.Flatten(array);
                copy = false;
            }

            if (dtype != null && !dtype.Equals(arrType))
            {
                array = ArrayConvert.To(array, (Type)dtype);
                copy = false;
            }

            // C-contiguous materialization from the managed array.
            var result = new NDArray(copy ? (Array)array.Clone() : array, shape, 'C');

            // Honor F-order request: relay out into F-contig layout.
            char physical = OrderResolver.Resolve(order, result.Shape);
            if (physical == 'F' && result.Shape.NDim > 1 && !result.Shape.IsFContiguous)
                result = result.copy('F');
            return result;
        }

        /// <summary>
        ///     Whether a C# array is NumPy's Python list in the boundary map: its innermost element type is
        ///     <see cref="object"/> (an <c>object[]</c>, a jagged <c>object[][]</c>, an <c>object[,]</c> grid of rows), so its
        ///     items are Python objects - C# numbers Python scalars, nested arrays and lists sequences, an
        ///     <see cref="NDArray"/> or a typed C# array an ndarray.
        /// </summary>
        /// <param name="array">The array; not null.</param>
        /// <returns>True for an array of objects at its innermost level.</returns>
        private static bool HoldsPythonObjects(Array array)
        {
            var elementType = array.GetType().GetElementType();
            while (elementType is { IsArray: true })
                elementType = elementType.GetElementType();
            return elementType == typeof(object);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny</c> of a Python list (a C# object array, <see cref="HoldsPythonObjects"/>) whose items
        ///     hold no text: the coercion of a sequence (<see cref="UnmanagedStorage.FromSequence(object, DType)"/>) discovers
        ///     the shape and, without a request, the dtype NumPy discovers (a C# int is a Python int, int64; a C# float or
        ///     double a Python float, float64; promoted strongly), or stores every item through the requested descriptor's
        ///     setitem. Then the requested layout and the minimum rank.
        /// </summary>
        /// <param name="array">The object array; its text items, if any, were routed to the text builder before.</param>
        /// <param name="dtype">The requested dtype (not a text class), or null to discover it.</param>
        /// <param name="ndmin">The minimum number of dimensions: leading ones are prepended as a VIEW of the built array (the
        ///     built array's own reference goes, as NumPy's result does not own its data then).</param>
        /// <param name="order">The memory layout: <c>'F'</c> lays a result of two or more dimensions out column-major (a copy
        ///     of the C-ordered fill); any other is C (the result is fresh).</param>
        /// <returns>The new array.</returns>
        /// <exception cref="ValueError">A ragged list (NumPy's inhomogeneous-shape text), or a value the requested dtype's
        ///     setitem refuses.</exception>
        /// <exception cref="OverflowException">A Python int outside the requested integer dtype.</exception>
        /// <exception cref="TypeError">A value the requested dtype cannot convert.</exception>
        /// <exception cref="NotSupportedException">None or a Python int past 64 bits without a request (NumPy's object
        ///     array), or a value of no NumPy type.</exception>
        private static NDArray ArrayFromPythonList(Array array, DType dtype, int ndmin, char order)
        {
            var built = array.Rank > 1 && array.Length == 0
                ? EmptyObjectGrid(array, dtype)
                : Backends.UnmanagedStorage.FromSequence(array, dtype);
            if (built.ndim > 1 && OrderResolver.Resolve(order, built.Shape) == 'F')
            {
                var f = built.copy('F');
                built.Dispose();
                built = f;
            }
            if (ndmin <= built.ndim)
                return built;
            // NumPy builds the array at its discovered rank and prepends the ones as a view of it.
            var view = np.AtLeastNdView(built, ndmin);
            built.Dispose();
            return view;
        }

        /// <summary>
        ///     An EMPTY rectangular object grid (<c>object[0, 3]</c>, <c>object[2, 0, 4]</c>): an empty array of the grid's
        ///     own dimensions. NumPy cannot spell it - its nested lists end the dimensions at the first empty level
        ///     (<c>[]</c> is <c>(0,)</c>) -, and a C# grid states every dimension, as a typed grid's array keeps them
        ///     (<c>new int[0, 3]</c> is <c>(0, 3)</c>), so the grid's are kept instead of the list reading's.
        /// </summary>
        /// <param name="grid">The rectangular object array (rank 2 or more, no elements).</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's float64 default of an empty list.</param>
        /// <returns>A new empty array of the grid's dimensions.</returns>
        private static NDArray EmptyObjectGrid(Array grid, DType dtype)
        {
            var dims = new long[grid.Rank];
            for (int d = 0; d < dims.Length; d++)
                dims[d] = grid.GetLength(d);
            return np.zeros(new Shape(dims), dtype ?? NumSharp.DType.Double);
        }

        /// <summary>
        ///     NumPy's <c>_array_fromobject_generic</c> with <c>subok</c> for any C# value, the body of
        ///     <see cref="np.asanyarray(in object, DType, char, string)"/> once the device and the class boundaries were
        ///     checked: the order first (NumPy parses it with the arguments), then Python's <c>None</c>, an array
        ///     (<see cref="np.asarray(NDArray, DType, char, bool?, NDArray, string)"/>), text arrays and sequences (the text
        ///     array builder), Python lists - object arrays and grids, tuples and any other enumerable of objects - through
        ///     the coercion of a sequence (<see cref="UnmanagedStorage.FromSequence(object, DType)"/>: a C# int is a Python
        ///     int, int64; with a dtype every item is stored by its setitem), typed C# arrays, a str, typed sequences,
        ///     scalars and <c>Memory&lt;T&gt;</c>; then the dtype request and the memory layout.
        /// </summary>
        /// <param name="a">The value: null is Python's None.</param>
        /// <param name="dtype">The requested dtype, or null to discover it.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>An array: the operand itself when it is an array that satisfies the request.</returns>
        /// <exception cref="ValueError">An invalid order (NumPy's <c>PyArray_OrderConverter</c> text), a ragged list, or a
        ///     value the requested dtype's setitem refuses.</exception>
        /// <exception cref="NotSupportedException">None without a dtype, None or a Python int past 64 bits in a list without
        ///     a dtype (NumPy's object array), or a value of a type this conversion does not read.</exception>
        /// <exception cref="TypeError">None into an integer (or decimal) dtype, or a list item the requested dtype cannot
        ///     convert.</exception>
        /// <exception cref="OverflowException">A Python int in a list outside the requested integer dtype.</exception>
        internal static NDArray AnyArrayFromObject(object a, DType dtype, char order)
        {
            // PyArray_OrderConverter runs while NumPy parses the arguments: an invalid order raises before the operand is
            // converted (a value the dtype refuses, None without a dtype, would otherwise be reported first).
            OrderResolver.ResolveBufferOrder(order);
            NDArray ret;
            switch (a) {
                case null:
                    // Python's None: np.array(None, dtype) packs it with the dtype's setitem; without a dtype NumPy's
                    // discovery gives the object dtype.
                    if (dtype is null)
                        throw NDArray.NoneWithoutDtype("np.asanyarray");
                    return NDArray.NoneForStore(dtype);
                case NDArray nd:
                    return np.asarray(nd, dtype, order);
                case Array textArr when Backends.UnmanagedStorage.IsTextInput(textArr, dtype):
                    // Text (string[], an object[] holding str/bytes, or any array under a string dtype): NumPy's
                    // np.asanyarray of a list of str — a fixed-width string array.
                    ret = Backends.UnmanagedStorage.FromManagedText(textArr, dtype, 0, 'C');
                    break;
                case IEnumerable<string> texts:
                    ret = Backends.UnmanagedStorage.FromManagedText(texts.ToArray(), dtype, 0, 'C');
                    break;
                case object[] objArr:
                    // object[] (and by covariance every array of references: a jagged object[][], NDArray[]) is NumPy's
                    // Python list: the coercion of a sequence, which stores through the request's setitem when there is one
                    // (np.asanyarray(list, dtype) is np.array(list, dtype, copy=False): 300 into int8 raises).
                    ret = Backends.UnmanagedStorage.FromSequence(objArr, dtype);
                    break;
                case Array grid when grid.Rank > 1 && grid.GetType().GetElementType() == typeof(object):
                    // A rectangular object array (object[,], object[,,]) is NumPy's nested list of its rows,
                    // np.asarray([[1, 2], [3, 4]]): the same coercion walks it row by row; an empty grid keeps its dims.
                    ret = grid.Length == 0 ? EmptyObjectGrid(grid, dtype) : Backends.UnmanagedStorage.FromSequence(grid, dtype);
                    break;
                case Array array:
                    ret = new NDArray(array);
                    break;
                case string str:
                    ret = str; //implicit cast located in NDArray.Implicit.Array
                    break;

                case IEnumerable<bool> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<byte> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<sbyte> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<short> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<ushort> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<int> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<uint> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<long> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<ulong> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<char> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<Half> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<float> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<double> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<decimal> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;
                case IEnumerable<Complex> e: ret = np.array(UnmanagedStorage.ToArrayFast(e)); break;

                default:
                    var type = a.GetType();
                    //is it a scalar
                    if (type.IsPrimitive || type == typeof(decimal) || type == typeof(Half) || type == typeof(System.Numerics.Complex))
                    {
                        ret = NDArray.Scalar(a);
                        break;
                    }

                    // Memory<T>/ReadOnlyMemory<T> do not implement IEnumerable<T>.
                    if (type.IsGenericType)
                    {
                        var genericDef = type.GetGenericTypeDefinition();
                        if (genericDef == typeof(Memory<>) || genericDef == typeof(ReadOnlyMemory<>))
                        {
                            ret = UnmanagedStorage.ConvertMemory(a, type);
                            if (ret is not null)
                                break;
                        }
                    }

                    if (a is ITuple || a is IEnumerable)
                    {
                        // A tuple, or any other enumerable (List<object>, ArrayList: the typed collections took their typed
                        // cases above), is NumPy's Python tuple / list: its items, text through the text builder as an
                        // object list's, everything else the coercion of a sequence (C# ints are Python ints: int64).
                        var items = Backends.UnmanagedStorage.SequenceItems(a);
                        ret = Backends.UnmanagedStorage.IsTextInput(items, dtype)
                            ? Backends.UnmanagedStorage.FromManagedText(items, dtype, 0, 'C')
                            : Backends.UnmanagedStorage.FromSequence(items, dtype);
                        break;
                    }

                    if (a is IEnumerator enumerator)
                    {
                        ret = UnmanagedStorage.ConvertEnumerator(enumerator);
                        if (ret is not null)
                            break;
                    }

                    throw new NotSupportedException($"Unable to resolve asanyarray for type {type.Name}");
            }

            if (dtype != null && !Backends.UnmanagedStorage.SatisfiesRequest(ret, dtype))
                ret = ret.astype(dtype, true);

            // Apply requested order (no-op for scalars / 1-D / already-matching layouts).
            char physical = OrderResolver.Resolve(order, ret.Shape);
            if (ret.Shape.NDim > 1 && ret.size > 1)
            {
                bool layoutMatches = physical == 'C' ? ret.Shape.IsContiguous : ret.Shape.IsFContiguous;
                if (!layoutMatches)
                    ret = ret.copy(physical);
            }
            return ret;
        }

        /// <summary>
        ///     <see cref="np.array{T}(T[], DType)"/>'s body: a new 1-D array holding a copy of <paramref name="data"/>, then the
        ///     dtype request NumPy's <c>_array_fromobject_generic</c> makes (<c>if dtype is not None and arr.dtype !=
        ///     dtype</c>): the array as built when no dtype or its own was requested, else its cast.
        /// </summary>
        /// <typeparam name="T">The element type, one of NumSharp's dtypes.</typeparam>
        /// <param name="data">The values; not null.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>The new array, or its cast to <paramref name="dtype"/>.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no cast from <typeparamref name="T"/>'s
        ///     dtype.</exception>
        /// <remarks>The array built from <paramref name="data"/> is left to the garbage collector when it is cast.</remarks>
        internal static NDArray ArrayFromTyped<T>(T[] data, DType dtype) where T : unmanaged
        {
            var arr = new NDArray(ArraySlice.FromArray(data, true), Shape.Vector(data.Length));
            if (dtype != null && dtype.GetTypeCode() != InfoOf<T>.NPTypeCode)
                return arr.astype(dtype);
            return arr;
        }

        /// <summary>
        ///     <see cref="np.array{T}(T[,], DType)"/>'s body: a new 2-D array holding a copy of <paramref name="data"/>, then
        ///     the dtype request of <see cref="ArrayFromTyped{T}(T[], DType)"/>.
        /// </summary>
        /// <typeparam name="T">The element type, one of NumSharp's dtypes.</typeparam>
        /// <param name="data">The values; not null.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>The new array, or its cast to <paramref name="dtype"/>.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no cast from <typeparamref name="T"/>'s
        ///     dtype.</exception>
        /// <remarks>The array built from <paramref name="data"/> is left to the garbage collector when it is cast.</remarks>
        internal static NDArray ArrayFromTyped<T>(T[,] data, DType dtype) where T : unmanaged
        {
            var arr = new NDArray(ArraySlice.FromArray(data, true), new Shape(data.GetLength(0), data.GetLength(1)));
            if (dtype != null && dtype.GetTypeCode() != InfoOf<T>.NPTypeCode)
                return arr.astype(dtype);
            return arr;
        }

        /// <summary>
        ///     NumPy's <c>ValueError</c> text (<c>npy_no_copy_err_msg</c>) for <c>copy=False</c> when the conversion has to
        ///     make a new array — a cast, a relayout, a C# literal's pack or <c>None</c>'s; NumSharp raises it as an
        ///     <see cref="ArgumentException"/> on the <c>copy</c> parameter.
        /// </summary>
        private const string UnableToAvoidCopyText =
            "Unable to avoid copy while creating an array as requested.\n" +
            "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` " +
            "to allow a copy when needed (no behavior change in NumPy 1.x).\n" +
            "For more details, see " +
            "https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.";

        /// <summary>
        ///     Returns true when <paramref name="shape"/>'s memory layout already satisfies the requested
        ///     order ('C', 'F', 'A', 'K'). Mirrors NumPy's <c>STRIDING_OK</c> macro:
        ///     'A' and 'K' impose no layout constraint, so a copy is never forced for layout reasons.
        /// </summary>
        /// <param name="shape">The array's shape (its layout flags are read).</param>
        /// <param name="order">The requested order, either case.</param>
        /// <returns>True when no copy is needed for the layout.</returns>
        /// <exception cref="ValueError">An order other than C, F, A or K (NumPy's <c>PyArray_OrderConverter</c>
        ///     text).</exception>
        private static bool LayoutAlreadyOK(Shape shape, char order)
        {
            switch (order)
            {
                case 'C':
                case 'c':
                    return shape.IsContiguous;
                case 'F':
                case 'f':
                    return shape.IsFContiguous;
                case 'A':
                case 'a':
                case 'K':
                case 'k':
                    return true;
                default:
                    // NumPy's PyArray_OrderConverter text, verbatim.
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }

        // ----------------------------------------------------------------------------------------------------------------
        // np.where(condition, x, y): NumPy's PyArray_Where (multiarraymodule.c). Appended by P3's batch 7; its facades are
        // APIs/np.where.cs. The kernel namespace is named in full: this file does not import it.
        // ----------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_Where</c> (<c>multiarraymodule.c</c>), the body of <c>np.where(condition, x, y)</c>: the
        ///     elements of <paramref name="x"/> where <paramref name="condition"/> is true and of <paramref name="y"/> elsewhere,
        ///     the three broadcast together, in <c>result_type(x, y)</c> (NEP 50: a weak Python scalar adopts the other
        ///     operand's dtype). Fixed-width text selects on unit views (<c>NDStringArrays.Where</c>), whose result dtype is the
        ///     wider descriptor; the numbers take the IL kernels (a scalar operand hoisted once, contiguous operands in one
        ///     SIMD pass) or a four-operand NDIter pass.
        /// </summary>
        /// <param name="condition">The selector; a non-bool array is read by its truth (NumPy's cast to bool: 0 and 0.0
        ///     are false, NaN and the infinities true).</param>
        /// <param name="x">The values where <paramref name="condition"/> is true, an array (a weak 0-d array for a Python
        ///     scalar: the facades convert C# values with <see cref="FromPythonOperand"/>).</param>
        /// <param name="y">The values where <paramref name="condition"/> is false.</param>
        /// <returns>A new array of the broadcast shape in NumPy's iterator layout (F order when every full-size operand is
        ///     F-contiguous, C otherwise).</returns>
        /// <exception cref="IncorrectShapeException">The three operands do not broadcast together.</exception>
        /// <exception cref="TypeError">x and y have no common dtype (a number with text, NumPy's promotion error).</exception>
        /// <remarks>The array object's member (the np-layer dtype plan's D1: np.where(cond, x, y) selects by the caller's
        ///     mask, plan section 1); np.where's single-argument form is np.nonzero, the engine's. The engine's own np.where
        ///     calls (the NaN replacement of the nan reductions and scans, StringDType's cell selection) call this member.
        ///     The scope reclaims the broadcast views, the bool and dtype conversions and the scalar promotions; only the
        ///     result is yielded.</remarks>
        [NDScoped]
        internal static NDArray Where(NDArray condition, NDArray x, NDArray y)
        {
            // Boundary scope: reclaims the broadcast_arrays wrappers, the bool/dtype astype
            // conversions and the scalar-promotion temps at exit; only `result` is yielded.

            // Fixed-width strings: the result dtype is a DESCRIPTOR (result_type(x, y): the wider width) and the
            // selection runs on unit views (NDStringArrays.Where). The broadcast is validated first, as NumPy's iterator
            // does, and NumPy's result layout comes from the broadcast operands.
            if (x.typecode == NPTypeCode.String || y.typecode == NPTypeCode.String)
            {
                var operands = np.broadcast_arrays(condition, x, y);
                return Backends.Strings.NDStringArrays.Where(condition, x, y, (long[])operands[0].Shape.dimensions.Clone());
            }

            // Detect "originally scalar" on the user-supplied operands BEFORE broadcasting
            // expands them into stride-0 views. The scalar fast path below dispatches
            // specialised IL kernels that hoist the scalar into Vector.Create<T>(value) once
            // outside the loop — avoiding the per-element broadcast view dereference that the
            // NDIter expression kernel would otherwise perform.
            bool xIsScalar = x.size == 1;
            bool yIsScalar = y.size == 1;

            // Skip broadcast_arrays (which allocates 3 NDArrays + helper arrays) when all three
            // already share a shape — the frequent case of np.where(mask, arr, other_arr).
            NDArray cond, xArr, yArr;
            if (condition.Shape == x.Shape && x.Shape == y.Shape)
            {
                cond = condition;
                xArr = x;
                yArr = y;
            }
            else
            {
                var broadcasted = np.broadcast_arrays(condition, x, y);
                cond = broadcasted[0];
                xArr = broadcasted[1];
                yArr = broadcasted[2];
            }

            // Resolve output layout before dtype casts. Casts of broadcasted scalars can
            // materialize C-contiguous temporaries, but NumPy's iterator ignores those for
            // output-order selection.
            char resultOrder = ResolveWhereOrder(cond, xArr, yArr);

            // Coerce the condition to boolean using NumPy's truthiness rules
            // (0/0.0 → False, everything else including NaN/±Inf → True). The
            // iterator-driven expression kernel requires a bool condition dtype.
            if (cond.GetTypeCode != NPTypeCode.Boolean)
                cond = cond.astype(NPTypeCode.Boolean, copy: false);

            // When x and y already agree, skip the NEP50 promotion lookup. Otherwise defer to
            // _FindCommonType which handles the scalar+array NEP50 rules.
            var outType = x.GetTypeCode == y.GetTypeCode
                ? x.GetTypeCode
                : np._FindCommonType(x, y);

            if (xArr.GetTypeCode != outType)
                xArr = xArr.astype(outType, copy: false);
            if (yArr.GetTypeCode != outType)
                yArr = yArr.astype(outType, copy: false);

            // Use cond.shape (dimensions only) not cond.Shape (which may have broadcast strides).
            // NumPy's iterator allocation preserves F-order when all full-size operands agree
            // on F layout; any full C-contiguous or non-contiguous operand falls back to C.
            var result = np.empty(Shape.Adopt((long[])cond.Shape.dimensions.Clone(), resultOrder), outType);

            // Handle empty arrays - nothing to iterate
            if (result.size == 0)
                return result;

            // -----------------------------------------------------------------
            // Scalar-broadcast IL fast path
            // -----------------------------------------------------------------
            // When x or y was a Python literal / 0-d / size-1 array, broadcast_arrays expanded
            // it into a stride-0 view that fails the IsContiguous gate below. Instead of
            // materializing that view into a full contig copy (NDIter's behaviour) we read
            // the single value, cast it to outType, and dispatch a kernel that broadcasts via
            // V<T>.Create(value) once outside the SIMD loop.
            //
            // The non-scalar operand must be contig (its shape already matches the result
            // because of the broadcast above). Two scalars + contig cond is also covered.
            if (Backends.Kernels.DirectILKernelGenerator.Enabled &&
                cond.typecode == NPTypeCode.Boolean &&
                cond.Shape.IsContiguous &&
                (xIsScalar || yIsScalar))
            {
                // Promote scalars to outType once (cheap — these are 1-element NDArrays).
                // For each, use the ORIGINAL operand (x/y) so we don't rely on the broadcast
                // view; the cast yields a fresh 1-element NDArray of outType.
                NDArray xScalarSrc = xIsScalar
                    ? (x.GetTypeCode != outType ? x.astype(outType) : x)
                    : null;
                NDArray yScalarSrc = yIsScalar
                    ? (y.GetTypeCode != outType ? y.astype(outType) : y)
                    : null;

                if (xIsScalar && yIsScalar)
                {
                    WhereScalarXYDispatch(cond, xScalarSrc, yScalarSrc, result, outType);
                    return result;
                }
                if (xIsScalar && yArr.Shape.IsContiguous)
                {
                    WhereScalarXDispatch(cond, xScalarSrc, yArr, result, outType);
                    return result;
                }
                if (yIsScalar && xArr.Shape.IsContiguous)
                {
                    WhereScalarYDispatch(cond, xArr, yScalarSrc, result, outType);
                    return result;
                }
            }

            // IL Kernel fast path: all arrays contiguous, bool condition, SIMD enabled
            // Broadcasted arrays (stride=0) are NOT contiguous, so they use iterator path.
            bool canUseKernel = Backends.Kernels.DirectILKernelGenerator.Enabled &&
                                cond.typecode == NPTypeCode.Boolean &&
                                cond.Shape.IsContiguous &&
                                xArr.Shape.IsContiguous &&
                                yArr.Shape.IsContiguous;

            if (canUseKernel)
            {
                WhereKernelDispatch(cond, xArr, yArr, result, outType);
                return result;
            }

            // Iterator fallback for non-contiguous/broadcasted arrays.
            WhereImpl(cond, xArr, yArr, result);

            return result;
        }

        /// <summary>
        ///     The general route of <see cref="Where"/>: a four-operand NDIter (condition, x, y, result) driving the per-chunk
        ///     where kernel, for broadcast or non-contiguous operands.
        /// </summary>
        /// <param name="cond">The bool condition, broadcast to the result's shape.</param>
        /// <param name="x">The true values, broadcast, in the result's dtype.</param>
        /// <param name="y">The false values, broadcast, in the result's dtype.</param>
        /// <param name="result">The fresh result, written in place.</param>
        private static unsafe void WhereImpl(NDArray cond, NDArray x, NDArray y, NDArray result)
        {
            // Drive cond + x + y + result in lockstep via a 4-operand NDIter and a
            // dedicated per-chunk multi-operand kernel (Backends.Kernels.ILKernelGenerator.Where).
            // C-order traversal matches NumPy element semantics; WRITEONLY on the
            // output. Operands already share dtypes by here (cond is Boolean, x/y/
            // result are the resolved output dtype), so no casting/buffering occurs.
            //
            // The kernel SIMD-selects (Vector.ConditionalSelect over an expanded
            // bool mask) whenever the inner loop is contiguous for all four operands
            // — e.g. a row-mask broadcast over a matrix — and scalar-walks per byte
            // stride otherwise. Both beat the previous NDExpr.Where path, which was
            // scalar-only AND cast cond to the output dtype on every element.
            var dtype = result.GetTypeCode;
            using var iter = NDIterRef.MultiNew(
                4, new[] { cond, x, y, result },
                NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.WRITEONLY,
                });

            iter.ForEach(Backends.Kernels.ILKernelGenerator.GetWhereInnerLoop(dtype));
        }

        /// <summary>
        ///     The layout NumPy's iterator gives <see cref="Where"/>'s result: F when every full-size operand that has a
        ///     preference is F-contiguous only, C when any is C-contiguous only or neither; scalars, 1-D and broadcast
        ///     operands do not vote.
        /// </summary>
        /// <param name="operands">The broadcast condition, x and y.</param>
        /// <returns><c>'F'</c> or <c>'C'</c>.</returns>
        private static char ResolveWhereOrder(params NDArray[] operands)
        {
            bool sawStrictF = false;

            foreach (var operand in operands)
            {
                var shape = operand.Shape;

                // Scalar, 1-D, and broadcasted operands don't force the output layout.
                if (shape.IsScalar || shape.NDim <= 1 || shape.IsBroadcasted)
                    continue;

                bool isC = shape.IsContiguous;
                bool isF = shape.IsFContiguous;

                if (isC && !isF)
                    return 'C';

                if (isF && !isC)
                {
                    sawStrictF = true;
                    continue;
                }

                if (!isC && !isF)
                    return 'C';
            }

            return sawStrictF ? 'F' : 'C';
        }

        /// <summary>
        /// IL Kernel dispatch for contiguous arrays.
        /// Uses IL-generated kernels with SIMD optimization.
        /// </summary>
        /// <param name="cond">The contiguous bool condition.</param>
        /// <param name="x">The contiguous true values, in the result's dtype.</param>
        /// <param name="y">The contiguous false values, in the result's dtype.</param>
        /// <param name="result">The fresh result.</param>
        /// <param name="outType">The result's storage lane.</param>
        private static unsafe void WhereKernelDispatch(NDArray cond, NDArray x, NDArray y, NDArray result, NPTypeCode outType)
        {
            var condPtr = (nint)cond.Address;
            var count = result.size;

            NpFunc.Invoke(outType, WhereKernelExecute<int>, condPtr, (nint)x.Address, (nint)y.Address, (nint)result.Address, count);
        }

        /// <summary>The typed body of <see cref="WhereKernelDispatch"/>: the contiguous IL where kernel of <typeparamref name="T"/>.</summary>
        /// <typeparam name="T">The element type of x, y and the result.</typeparam>
        /// <param name="condPtr">The bool condition's first element.</param>
        /// <param name="xAddr">x's first element.</param>
        /// <param name="yAddr">y's first element.</param>
        /// <param name="resultAddr">The result's first element.</param>
        /// <param name="count">The element count.</param>
        private static unsafe void WhereKernelExecute<T>(nint condPtr, nint xAddr, nint yAddr, nint resultAddr, long count) where T : unmanaged
            => Backends.Kernels.DirectILKernelGenerator.WhereExecute((bool*)condPtr, (T*)xAddr, (T*)yAddr, (T*)resultAddr, count);

        // -----------------------------------------------------------------
        // Scalar-broadcast dispatch
        // -----------------------------------------------------------------
        // Reads the scalar value from the 1-element NDArray (already promoted to outType)
        // and invokes the appropriate IL kernel. Returns true on success; false if no IL
        // kernel was available (caller falls back to the NDIter path).

        /// <summary>The scalar-x route of <see cref="Where"/>: x hoisted once into the IL kernel, or broadcast for the NDIter
        ///     route when the dtype has no kernel.</summary>
        /// <param name="cond">The contiguous bool condition.</param>
        /// <param name="xScalar">x's one element, in the result's dtype.</param>
        /// <param name="y">The contiguous false values.</param>
        /// <param name="result">The fresh result.</param>
        /// <param name="outType">The result's storage lane.</param>
        private static unsafe void WhereScalarXDispatch(NDArray cond, NDArray xScalar, NDArray y, NDArray result, NPTypeCode outType)
        {
            // Attempt the IL kernel; if it returns false (no SIMD / unsupported dtype),
            // materialize the scalar to a broadcast view of cond's shape and fall back
            // to the existing NDIter expression path.
            bool ok = NpFunc.Invoke(outType, TryWhereScalarXExecute<int>,
                (nint)cond.Address, (nint)xScalar.Address, (nint)y.Address, (nint)result.Address, result.size);
            if (ok) return;

            var xBroadcast = np.broadcast_to(xScalar, cond.Shape);
            WhereImpl(cond, xBroadcast, y, result);
        }

        /// <summary>The scalar-y route of <see cref="Where"/> (see <see cref="WhereScalarXDispatch"/>).</summary>
        /// <param name="cond">The contiguous bool condition.</param>
        /// <param name="x">The contiguous true values.</param>
        /// <param name="yScalar">y's one element, in the result's dtype.</param>
        /// <param name="result">The fresh result.</param>
        /// <param name="outType">The result's storage lane.</param>
        private static unsafe void WhereScalarYDispatch(NDArray cond, NDArray x, NDArray yScalar, NDArray result, NPTypeCode outType)
        {
            bool ok = NpFunc.Invoke(outType, TryWhereScalarYExecute<int>,
                (nint)cond.Address, (nint)x.Address, (nint)yScalar.Address, (nint)result.Address, result.size);
            if (ok) return;

            var yBroadcast = np.broadcast_to(yScalar, cond.Shape);
            WhereImpl(cond, x, yBroadcast, result);
        }

        /// <summary>The two-scalar route of <see cref="Where"/> (see <see cref="WhereScalarXDispatch"/>).</summary>
        /// <param name="cond">The contiguous bool condition.</param>
        /// <param name="xScalar">x's one element, in the result's dtype.</param>
        /// <param name="yScalar">y's one element, in the result's dtype.</param>
        /// <param name="result">The fresh result.</param>
        /// <param name="outType">The result's storage lane.</param>
        private static unsafe void WhereScalarXYDispatch(NDArray cond, NDArray xScalar, NDArray yScalar, NDArray result, NPTypeCode outType)
        {
            bool ok = NpFunc.Invoke(outType, TryWhereScalarXYExecute<int>,
                (nint)cond.Address, (nint)xScalar.Address, (nint)yScalar.Address, (nint)result.Address, result.size);
            if (ok) return;

            var xBroadcast = np.broadcast_to(xScalar, cond.Shape);
            var yBroadcast = np.broadcast_to(yScalar, cond.Shape);
            WhereImpl(cond, xBroadcast, yBroadcast, result);
        }

        /// <summary>The typed scalar-x kernel call: false when <typeparamref name="T"/> has no kernel.</summary>
        /// <typeparam name="T">The result's element type.</typeparam>
        /// <param name="condPtr">The bool condition's first element.</param>
        /// <param name="xScalarPtr">x's one element.</param>
        /// <param name="yPtr">y's first element.</param>
        /// <param name="resPtr">The result's first element.</param>
        /// <param name="count">The element count.</param>
        /// <returns>Whether the kernel ran.</returns>
        private static unsafe bool TryWhereScalarXExecute<T>(nint condPtr, nint xScalarPtr, nint yPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = Backends.Kernels.DirectILKernelGenerator.GetWhereScalarXKernel<T>();
            if (kernel == null) return false;

            T scalarX = *(T*)xScalarPtr;
            kernel((bool*)condPtr, scalarX, (T*)yPtr, (T*)resPtr, count);
            return true;
        }

        /// <summary>The typed scalar-y kernel call: false when <typeparamref name="T"/> has no kernel.</summary>
        /// <typeparam name="T">The result's element type.</typeparam>
        /// <param name="condPtr">The bool condition's first element.</param>
        /// <param name="xPtr">x's first element.</param>
        /// <param name="yScalarPtr">y's one element.</param>
        /// <param name="resPtr">The result's first element.</param>
        /// <param name="count">The element count.</param>
        /// <returns>Whether the kernel ran.</returns>
        private static unsafe bool TryWhereScalarYExecute<T>(nint condPtr, nint xPtr, nint yScalarPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = Backends.Kernels.DirectILKernelGenerator.GetWhereScalarYKernel<T>();
            if (kernel == null) return false;

            T scalarY = *(T*)yScalarPtr;
            kernel((bool*)condPtr, (T*)xPtr, scalarY, (T*)resPtr, count);
            return true;
        }

        /// <summary>The typed two-scalar kernel call: false when <typeparamref name="T"/> has no kernel.</summary>
        /// <typeparam name="T">The result's element type.</typeparam>
        /// <param name="condPtr">The bool condition's first element.</param>
        /// <param name="xScalarPtr">x's one element.</param>
        /// <param name="yScalarPtr">y's one element.</param>
        /// <param name="resPtr">The result's first element.</param>
        /// <param name="count">The element count.</param>
        /// <returns>Whether the kernel ran.</returns>
        private static unsafe bool TryWhereScalarXYExecute<T>(nint condPtr, nint xScalarPtr, nint yScalarPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = Backends.Kernels.DirectILKernelGenerator.GetWhereScalarXYKernel<T>();
            if (kernel == null) return false;

            T scalarX = *(T*)xScalarPtr;
            T scalarY = *(T*)yScalarPtr;
            kernel((bool*)condPtr, scalarX, scalarY, (T*)resPtr, count);
            return true;
        }
    }
}
