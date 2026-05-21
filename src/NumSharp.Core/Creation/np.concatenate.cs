using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="arrays">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to <paramref name="axis"/> (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined. If
        ///     <c>null</c>, arrays are flattened before use. Default is 0.
        ///     Negative axes are normalized against the input ndim.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result. The shape must
        ///     be correct, matching what would have been returned with no
        ///     <c>out</c> argument. Cannot be used together with <paramref name="dtype"/>.
        /// </param>
        /// <param name="dtype">
        ///     If provided, the result array will have this dtype. Cannot be
        ///     used together with <paramref name="out"/>.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur. One of
        ///     <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c>
        ///     (default), or <c>"unsafe"</c>.
        /// </param>
        /// <returns>The concatenated array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html</remarks>
        public static NDArray concatenate(
            NDArray[] arrays,
            int? axis = 0,
            NDArray @out = null,
            NPTypeCode? dtype = null,
            string casting = "same_kind")
        {
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));
            if (arrays.Length == 0)
                throw new ArgumentException(
                    "need at least one array to concatenate", nameof(arrays));

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

            // 0-D arrays cannot be concatenated (NumPy: "zero-dimensional arrays cannot be concatenated").
            for (int k = 0; k < arrays.Length; k++)
            {
                if (arrays[k] is null)
                    throw new ArgumentNullException($"{nameof(arrays)}[{k}]");
                if (arrays[k].ndim == 0)
                    throw new ArgumentException(
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

            // Normalize negative axis against input ndim (NumPy AxisError on out-of-range).
            if (effectiveAxis < 0)
                effectiveAxis += ndim;
            if (effectiveAxis < 0 || effectiveAxis >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");

            // Shape validation: all inputs must have same ndim and match on
            // non-axis dimensions. Accumulate axis size for the result.
            long axisSize = 0;
            var firstShape = (long[])first.shape.Clone();
            for (int k = 0; k < workArrays.Length; k++)
            {
                var src = workArrays[k];
                if (src.ndim != ndim)
                    throw new IncorrectShapeException(
                        $"all the input arrays must have same number of dimensions, " +
                        $"but the array at index 0 has {ndim} dimension(s) and the array at index {k} has {src.ndim} dimension(s)");

                var srcShape = src.shape;
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

            // Resolve result dtype: dtype= wins, then out=, then NEP50 promotion.
            NPTypeCode resultType;
            if (dtype is not null)
                resultType = dtype.Value;
            else if (@out is not null)
                resultType = @out.GetTypeCode;
            else if (workArrays.Length == 1)
                resultType = workArrays[0].GetTypeCode;
            else
                resultType = np.result_type(workArrays);

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
                        $"Output array has wrong dimensionality: expected {ndim}, got {@out.ndim}");
                for (int j = 0; j < ndim; j++)
                {
                    if (@out.shape[j] != firstShape[j])
                        throw new IncorrectShapeException(
                            $"Output array has wrong shape: expected " +
                            $"({string.Join(", ", firstShape)}), got ({string.Join(", ", @out.shape)})");
                }
                dst = @out;
            }
            else
            {
                // NumPy-aligned: when every input is F-contiguous, produce an
                // F-contiguous destination; otherwise default to C. A (1,N)
                // input that is BOTH C- and F-contig (ambiguous layout) still
                // counts toward the F-contig vote.
                bool allF = true;
                for (int k = 0; k < workArrays.Length; k++)
                {
                    if (!workArrays[k].Shape.IsFContiguous)
                    {
                        allF = false;
                        break;
                    }
                }
                var retShape = allF ? new Shape(firstShape, 'F') : new Shape(firstShape);
                // fillZeros: false — every byte is overwritten below.
                dst = new NDArray(resultType, retShape, fillZeros: false);
            }

            // Layered fast paths:
            // 1. TryDirectMemcpyConcat -- all sources same dtype as dst and
            //    matching layout (C/F): direct Buffer.MemoryCopy per outer
            //    slab, no slice/state construction.
            // 2. TryDirectCastConcat -- all sources contig, dst contig,
            //    mixed dtypes: drive the IL contig cast kernel per source
            //    with computed offsets.
            // 3. General path via NpyIter.Copy -- broadcasted sources,
            //    exotic dtype pairs, mixed C/F layouts. NpyIter's K-order
            //    axis permutation (added to CreateCopyState) ensures the
            //    unit-stride axis ends up innermost so the IL strided
            //    cast kernel's inner-contig branch fires even for F-contig
            //    sliced dsts. Without that, the strided path took ~17x
            //    longer than the fast paths on 1M F-contig inputs.
            //
            // The fast paths still win ~50-90% on workloads they cover
            // because they skip the dst[axis_range] slice creation and
            // the per-source NpyIter state construction -- savings that
            // amortize over many small calls (count_1024) or compound
            // across many copy operations.
            if (TryDirectMemcpyConcat(dst, workArrays, effectiveAxis, ndim, resultType))
                return dst;

            if (TryDirectCastConcat(dst, workArrays, effectiveAxis, ndim, resultType))
                return dst;

            // General path: NpyIter.Copy for anything the fast paths skip.
            var dstAccessor = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                dstAccessor[i] = Slice.All;

            long dstAxisPos = 0;
            for (int k = 0; k < workArrays.Length; k++)
            {
                var src = workArrays[k];
                var len = src.shape[effectiveAxis];
                if (len == 0) continue; // skip empty inputs along the concat axis

                dstAccessor[effectiveAxis] = new Slice(dstAxisPos, dstAxisPos + len);
                // dstSlice is an owning intermediate (sliced view of dst — a fresh NDArray
                // wrapper sharing storage). Releasing each iteration's wrapper atomically
                // keeps N-source loops from queueing N dead wrappers per concatenate call.
                using var dstSlice = dst[dstAccessor];
                NpyIter.Copy(dstSlice, src);

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
        ///     justification block in <see cref="concatenate"/>).
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
                for (int i = 0; i < axis; i++) outerCount *= dst.shape[i];
                for (int i = axis + 1; i < ndim; i++) innerStride *= dst.shape[i];
            }
            else
            {
                for (int i = axis + 1; i < ndim; i++) outerCount *= dst.shape[i];
                for (int i = 0; i < axis; i++) innerStride *= dst.shape[i];
            }

            // Per-outer dst row size in elements = sum of src.shape[axis] * innerStride.
            // Equivalent to dst.shape[axis] * innerStride.
            long dstRowSize = dst.shape[axis] * innerStride;
            long dstRowBytes = dstRowSize * elemSize;

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
                slabBytes[k] = sources[k].shape[axis] * innerStride * elemSize;
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
        ///     Cross-dtype fast path: same shape/layout assumptions as
        ///     <see cref="TryDirectMemcpyConcat"/>, but drives the IL-generated
        ///     contig cast kernel per source instead of <see cref="Buffer.MemoryCopy"/>.
        ///     Avoids the slice-then-NpyIter detour for the same F-contig
        ///     reason (slice produces non-contig view, NpyIter falls into
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
            var kernels = new Backends.Kernels.ILKernelGenerator.CastKernel[sources.Length];
            for (int k = 0; k < sources.Length; k++)
            {
                if (sources[k].GetTypeCode == resultType)
                {
                    kernels[k] = null; // sentinel for "use memcpy"
                }
                else
                {
                    var kk = Backends.Kernels.ILKernelGenerator
                        .TryGetCastKernel(sources[k].GetTypeCode, resultType);
                    if (kk is null) return false;
                    kernels[k] = kk;
                }
            }

            int dstElemSize = resultType.SizeOf();

            long outerCount = 1, innerStride = 1;
            if (cPath)
            {
                for (int i = 0; i < axis; i++) outerCount *= dst.shape[i];
                for (int i = axis + 1; i < ndim; i++) innerStride *= dst.shape[i];
            }
            else
            {
                for (int i = axis + 1; i < ndim; i++) outerCount *= dst.shape[i];
                for (int i = 0; i < axis; i++) innerStride *= dst.shape[i];
            }

            long dstRowElems = dst.shape[axis] * innerStride;
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
                slabElems[k] = sources[k].shape[axis] * innerStride;
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

        // ---------------- Tuple-arity convenience overloads ----------------
        // NumPy permits `np.concatenate((a, b, c))` as a tuple. These mirror
        // that ergonomic with the array-form's default keyword params.

        public static NDArray concatenate((NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8, arrays.Item9 }, axis);
    }
}
