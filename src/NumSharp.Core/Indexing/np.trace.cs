using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the sum along diagonals of the array. For a 2-D array,
        ///     <c>trace(a) == sum(a.diagonal())</c>. For an N-D array, traces
        ///     the diagonals along the 2-D sub-arrays defined by
        ///     <paramref name="axis1"/> / <paramref name="axis2"/> and reduces
        ///     them, leaving an array with those two axes removed.
        /// </summary>
        /// <param name="a">Source array. Must have at least 2 dimensions.</param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal.
        ///     See <see cref="diagonal"/> for details.
        /// </param>
        /// <param name="axis1">First axis of the 2-D sub-array. Default 0.</param>
        /// <param name="axis2">Second axis of the 2-D sub-array. Default 1.</param>
        /// <param name="dtype">
        ///     Output dtype. <c>null</c> (default) preserves <c>a.dtype</c>,
        ///     except integer dtypes narrower than <see cref="long"/> promote
        ///     to <see cref="long"/> (NEP50 / matches NumPy's "default platform
        ///     integer" rule). Bool input promotes to <see cref="long"/>.
        /// </param>
        /// <param name="out">
        ///     Optional output array. Shape must equal the natural reduction
        ///     output; values are copied with unsafe casting and the method
        ///     returns <paramref name="out"/> itself.
        /// </param>
        /// <returns>
        ///     Sum along the diagonal. 2-D input → 0-d scalar. N-D input →
        ///     array with <c>a.shape</c> minus <paramref name="axis1"/> and
        ///     <paramref name="axis2"/>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        ///     <para>
        ///     Mirrors NumPy's <c>PyArray_Trace</c>: <c>sum(diagonal(a,
        ///     offset, axis1, axis2), axis=-1, dtype=dtype, out=out)</c>. The
        ///     diagonal view is the heavy lifting; the sum is a regular
        ///     reduction that goes through NumSharp's IL reduction kernels.
        ///     </para>
        /// </remarks>
        public static NDArray trace(
            NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1,
            Type dtype = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // Fast path: 2-D or 3-D source + no explicit dtype + kernel-eligible
            // src dtype. Runs a single strided diagonal walk with inline
            // accumulation into the promoted dtype — no NDArray intermediates
            // between input and the result. Typical 5x5..100x100 cases:
            // 3-4x faster than the generic diagonal+ascontig+sum chain.
            if ((a.ndim == 2 || a.ndim == 3) && dtype is null && @out is null &&
                TryTraceFast(a, offset, axis1, axis2, out var fast))
                return fast;

            // diagonal() validates ndim, axis values, and axis1!=axis2 and emits
            // a 1-D-extended view (the diagonal axis is the last dim of the result).
            var diag = np.diagonal(a, offset, axis1, axis2);

            // dtype rule: when not specified, promote bool/int<int64 to int64;
            // otherwise preserve. np.sum (via the TensorEngine reduction) already
            // does NEP50 promotion when typeCode is null.
            NPTypeCode? sumDtype = dtype?.GetTypeCode();

            // The diagonal is a strided view (stride[axis1] + stride[axis2]); its
            // size is at most min(dim[axis1], dim[axis2]) along the appended
            // last axis. NumSharp's reduction on a strided input with dtype
            // promotion (e.g. int32→int64) falls off SIMD and is 30-100x slower
            // than the contig path. Materialise the diagonal to a contig copy
            // before summing — the extra memcpy is dwarfed by the SIMD win
            // (4.4μs vs 22μs for 100×100, 5.8μs vs 198μs for 1000×1000).
            var diagContig = np.ascontiguousarray(diag);
            try
            {
                // Pick the axis arg that hits NumSharp's fastest reduction path:
                // - 1-D diagonal (2-D source) → axis=null hits the scalar reduction
                //   kernel (~1μs), while axis=-1 unnecessarily routes through the
                //   axis-reduction machinery (~15μs) to produce the same 0-d result.
                // - N-D diagonal (3+D source) → axis=-1 reduces along the appended
                //   diag axis, leaving the source's non-axis1/axis2 axes intact.
                int? sumAxis = diagContig.ndim == 1 ? (int?)null : -1;
                var result = a.TensorEngine.Sum(diagContig, axis: sumAxis, typeCode: sumDtype, keepdims: false);
                return DispatchOut(result, @out);
            }
            finally
            {
                if (!ReferenceEquals(diagContig, diag))
                    diagContig.Dispose();
            }
        }

        /// <summary>
        ///     Fused 2-D / 3-D trace fast path. Computes the diagonal start
        ///     offset and stride exactly like <see cref="diagonal"/> does, then
        ///     calls the per-src-dtype IL kernel to walk and accumulate in one
        ///     pass. For 3-D source: iterates the single non-axis1/axis2 axis
        ///     in the kernel's outer loop, writing one accum-typed result per
        ///     outer position. Returns <c>false</c> when the dtype has no
        ///     kernel (Half / Decimal / Complex) so the caller falls through
        ///     to the generic diagonal+ascontig+sum chain.
        /// </summary>
        private static unsafe bool TryTraceFast(
            NDArray a, int offset, int axis1, int axis2, out NDArray result)
        {
            result = null;

            int ndim = a.ndim;
            int ax1 = NormalizeAxis(axis1, ndim, nameof(axis1));
            int ax2 = NormalizeAxis(axis2, ndim, nameof(axis2));
            if (ax1 == ax2)
                throw new ArgumentException(
                    "axis1 and axis2 cannot be the same",
                    nameof(axis2));

            var (accumCode, supported) = ILKernelGenerator.GetTraceAccumTypeCode(a.GetTypeCode);
            if (!supported) return false;

            var kernel = ILKernelGenerator.GetTraceKernel(a.dtype);
            if (kernel == null) return false;

            var shape = a.Shape;
            long dim1 = shape.dimensions[ax1];
            long dim2 = shape.dimensions[ax2];
            long stride1 = shape.strides[ax1];
            long stride2 = shape.strides[ax2];

            // Same offset-stride formula diagonal() uses; offset>=0 shifts along
            // axis2, offset<0 shifts along axis1.
            long offsetStride;
            long offAbs;
            if (offset >= 0)
            {
                offsetStride = stride2;
                dim2 -= offset;
                offAbs = offset;
            }
            else
            {
                offsetStride = stride1;
                dim1 -= -(long)offset;
                offAbs = -(long)offset;
            }

            long diagSize = Math.Min(dim1, dim2);
            if (diagSize < 0) diagSize = 0;

            long elemBytes = a.dtypesize;
            long startElemOffset = shape.offset + (diagSize > 0 ? offAbs * offsetStride : 0);
            long startByteOffset = startElemOffset * elemBytes;
            long byteStride = (stride1 + stride2) * elemBytes;

            // Outer factorisation: for 2-D source there's no outer axis, so
            // outerSize=1 / outerSrcStride=0. For 3-D source the one remaining
            // axis becomes the outer iteration; outerSrcStride is that axis's
            // stride in bytes.
            long outerSize = 1;
            long outerSrcStride = 0;
            int outerAxis = -1;
            if (ndim == 3)
            {
                for (int d = 0; d < ndim; d++)
                {
                    if (d != ax1 && d != ax2) { outerAxis = d; break; }
                }
                outerSize = shape.dimensions[outerAxis];
                outerSrcStride = shape.strides[outerAxis] * elemBytes;
            }

            // Result shape: 2-D source → 0-d scalar; 3-D source → 1-D of size
            // outerSize (the non-axis1/axis2 dim).
            int accumBytes = NPTypeCodeBytes(accumCode);
            long outerDstStride = accumBytes;
            if (ndim == 2)
            {
                result = new NDArray(accumCode, Shape.NewScalar());
            }
            else
            {
                result = new NDArray(accumCode, new Shape(outerSize), false);
            }

            byte* srcPtr = (byte*)a.Storage.Address;
            byte* dstPtr = (byte*)result.Storage.Address;

            kernel(srcPtr, startByteOffset, diagSize, byteStride,
                   outerSize, outerSrcStride, outerDstStride, dstPtr);
            return true;
        }

        /// <summary>
        /// Byte size for the trace result dtypes.
        /// </summary>
        private static int NPTypeCodeBytes(NPTypeCode code) => code switch
        {
            NPTypeCode.Int64 => 8,
            NPTypeCode.UInt64 => 8,
            NPTypeCode.Single => 4,
            NPTypeCode.Double => 8,
            NPTypeCode.Half => 2,
            NPTypeCode.Decimal => 16,
            NPTypeCode.Complex => 16,
            _ => throw new NotSupportedException($"Trace result dtype {code} has no size mapping")
        };

        /// <summary>
        ///     Out= writeback: validate shape, copy with unsafe casting, return
        ///     out. Matches NumPy's reduction-out= contract.
        /// </summary>
        private static NDArray DispatchOut(NDArray result, NDArray @out)
        {

            if (@out is null)
                return result;

            // out= dispatch: shape check + unsafe-cast writeback. NumPy's
            // PyArray_GenericReduceFunction enforces this too.
            if (!@out.Shape.Equals(result.Shape))
                throw new ArgumentException(
                    $"output array does not match result of trace: expected shape {result.Shape}, got {@out.Shape}",
                    nameof(@out));

            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }
    }
}
