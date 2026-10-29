using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_Trace</c>: <c>add.reduce(diagonal(a, offset, axis1, axis2), axis=-1, dtype=dtype,
        ///     out=out)</c>. A 2-D or 3-D operand without <paramref name="dtype"/> or <paramref name="out"/> takes the fused
        ///     kernel (<see cref="TryTraceFast"/>, one strided walk of the diagonal accumulating in the promoted dtype); every
        ///     other call reads the diagonal view (<see cref="NDArray.Diagonal(NDArray, int, int, int)"/>, the array
        ///     object's: a view is not an engine member) and runs the text routes; then an
        ///     <paramref name="out"/> takes the add reduction with an out operand (<see cref="ReduceAdd"/>: the reduce
        ///     prologue's checks and the loop <c>dtype ?? promote_types(out, diagonal)</c>), and a call without one sums a
        ///     contiguous copy of the diagonal. Moved from the public <c>np.trace</c> by the np-layer dtype plan's P1b,
        ///     statement for statement; the public function keeps the matrix boundary's plain region. The out= route is
        ///     NumPy's since P1d (it summed in NEP 50's accumulator and raised its own texts for a misshapen out).
        /// </remarks>
        [NDScoped] // reclaims the diagonal view and the contig copy (the out= route's own intermediates are ReduceAdd's)
        public override NDArray Trace(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null,
            NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // Fast path: 2-D or 3-D source + no explicit dtype + kernel-eligible
            // src dtype. Runs a single strided diagonal walk with inline
            // accumulation into the promoted dtype — no NDArray intermediates
            // between input and the result. Typical 5x5..100x100 cases:
            // 3-4x faster than the generic diagonal+ascontig+sum chain.
            if ((a.ndim == 2 || a.ndim == 3) && dtype is null && @out is null &&
                TryTraceFast(a, offset, axis1, axis2, out var fast))
                // The 2-D fast result is 0-d — a numpy SCALAR (read-only) at the boundary;
                // the general path below inherits the same from the engine Sum's exit.
                return fast.MarkReductionScalar();

            // The diagonal validates ndim, axis values, and axis1!=axis2 and emits
            // a 1-D-extended view (the diagonal axis is the last dim of the result).
            var diag = NDArray.Diagonal(a, offset, axis1, axis2);
            // A StringDType input: the diagonal's add.reduce along its LAST axis (PyArray_Trace), out= included — the
            // StringDType loop concatenates (a dtype= request has no loop, an empty diagonal has no identity).
            if (Strings.NDStringUfuncs.VStringReduces("trace", diag, diag.ndim - 1, dtype, @out, loopOnly: true))
                return Strings.NDVStringReduction.SumNumPy("trace", diag, diag.ndim - 1, false, @out);
            // NumPy's trace is the diagonal's add.reduce(diag, -1, dtype=dtype, out=out): the diagonal's own checks above
            // come first, then the loop resolution on (out, diag). A U diagonal into a StringDType out concatenates (the
            // StringDType add promoter pairs StringDType with StrDType; the diagonal is cast to the default StringDType());
            // text with any other out, a text out for numbers, or a string dtype= request is NumPy's refusal (a numeric
            // input then reads as text of its REQUIRED_STR_LEN).
            switch (Strings.NDStringUfuncs.ReduceIntoOutRoute("trace", diag, diag.ndim - 1, @out, dtype, out var resolution))
            {
                case Strings.NDStringUfuncs.TextOutReduce.Concatenate:
                    return Strings.NDVStringReduction.SumNumPyOfCast(diag, diag.ndim - 1, false, @out);
                case Strings.NDStringUfuncs.TextOutReduce.NumericCast:
                    // A numeric loop over a text operand (a text diagonal parsed, a text out written): add.reduce's buffered
                    // casts, the identity first (NDTextNumericReduction).
                    return Strings.NDTextNumericReduction.Reduce("add", diag,
                        ReductionArguments.AxisFlags(diag.ndim, diag.ndim - 1), false, @out, resolution.Descriptors[0]);
            }

            // out=: the diagonal's add.reduce WITH an out operand (ReduceAdd -> ReduceIntoOut): the reduce prologue checks
            // out in NumPy's order and texts (read-only first, then the rank, then the length), and the loop is
            // dtype ?? promote_types(out, diag) without NEP 50's widening, stored with an unsafe cast (an int8 diagonal
            // [100, 100, 56] into a bool out wraps to 0 and stores False). The text routes above already answered a text
            // operand, so only numbers reach it.
            if (@out is not null)
                return ReduceAdd(diag, diag.ndim - 1, keepdims: false, dtype, @out);

            // dtype rule: when not specified, the engine Sum's NEP50 promotion applies
            // (signed/bool -> int64, unsigned -> uint64, floats preserved) — the same
            // split the 2-D/3-D fast path's TraceTypeInfo encodes.
            NPTypeCode? sumDtype = dtype?.GetTypeCode();

            // The diagonal is a strided view (stride[axis1] + stride[axis2]); its
            // size is at most min(dim[axis1], dim[axis2]) along the appended
            // last axis. NumSharp's reduction on a strided input with dtype
            // promotion (e.g. int32→int64) falls off SIMD and is 30-100x slower
            // than the contig path. Materialise the diagonal to a contig copy
            // before summing — the extra memcpy is dwarfed by the SIMD win
            // (4.4μs vs 22μs for 100×100, 5.8μs vs 198μs for 1000×1000).
            var diagContig = NDArray.AsContiguousArray(diag, null);
            try
            {
                // Pick the axis arg that hits NumSharp's fastest reduction path:
                // - 1-D diagonal (2-D source) → axis=null hits the scalar reduction
                //   kernel (~1μs), while axis=-1 unnecessarily routes through the
                //   axis-reduction machinery (~15μs) to produce the same 0-d result.
                // - N-D diagonal (3+D source) → axis=-1 reduces along the appended
                //   diag axis, leaving the source's non-axis1/axis2 axes intact.
                int? sumAxis = diagContig.ndim == 1 ? (int?)null : -1;
                return a.TensorEngine.Sum(diagContig, axis: sumAxis, dtype: sumDtype.AsType(), keepdims: false);
            }
            finally
            {
                if (!ReferenceEquals(diagContig, diag))
                    diagContig.Dispose();
            }
        }

        /// <summary>
        ///     Fused 2-D / 3-D trace fast path. Computes the diagonal start
        ///     offset and stride exactly like <see cref="Shape.Diagonal(int, int, int)"/> does, then
        ///     calls the per-src-dtype IL kernel to walk and accumulate in one
        ///     pass. For 3-D source: iterates the single non-axis1/axis2 axis
        ///     in the kernel's outer loop, writing one accum-typed result per
        ///     outer position. Returns <c>false</c> when the dtype has no
        ///     kernel (Half / Decimal / Complex) so the caller falls through
        ///     to the generic diagonal+ascontig+sum chain.
        /// </summary>
        /// <param name="a">The 2-D or 3-D source array.</param>
        /// <param name="offset">The diagonal's offset (positive along <paramref name="axis2"/>, negative along
        ///     <paramref name="axis1"/>).</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays.</param>
        /// <param name="axis2">The second axis of the 2-D sub-arrays.</param>
        /// <param name="result">The trace in the promoted dtype (0-d for 2-D, 1-D for 3-D) when the call is taken;
        ///     null otherwise.</param>
        /// <returns>True when the fused kernel computed <paramref name="result"/>; false when the dtype has no kernel.</returns>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …" (axis1 checked
        ///     first).</exception>
        /// <exception cref="ValueError">The two axes are the same axis ("axis1 and axis2 cannot be the same").</exception>
        private static unsafe bool TryTraceFast(
            NDArray a, int offset, int axis1, int axis2, out NDArray result)
        {
            result = null;

            int ndim = a.ndim;
            int ax1 = check_and_adjust_axis(ndim, axis1, nameof(axis1));
            int ax2 = check_and_adjust_axis(ndim, axis2, nameof(axis2));
            // PyArray_Diagonal's ValueError, after both axes are in range.
            if (ax1 == ax2)
                throw new ValueError(
                    "axis1 and axis2 cannot be the same",
                    nameof(axis2));

            var (accumCode, supported) = DirectILKernelGenerator.GetTraceAccumTypeCode(a.GetTypeCode);
            if (!supported) return false;

            var kernel = DirectILKernelGenerator.GetTraceKernel(a.dtype.type);
            if (kernel == null) return false;

            var shape = a.Shape;
            long dim1 = shape.dimensions[ax1];
            long dim2 = shape.dimensions[ax2];
            long stride1 = shape.strides[ax1];
            long stride2 = shape.strides[ax2];

            // Same offset-stride formula Shape.Diagonal uses; offset>=0 shifts along
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
            int accumBytes = TraceAccumBytes(accumCode);
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
        ///     The byte size of a trace accumulator dtype (the fused kernel's result element).
        /// </summary>
        /// <param name="code">One of the accumulator dtypes <see cref="DirectILKernelGenerator.GetTraceAccumTypeCode"/>
        ///     returns.</param>
        /// <returns>Its size in bytes.</returns>
        /// <exception cref="NotSupportedException">Another dtype (the kernel table and this map disagree: a bug).</exception>
        private static int TraceAccumBytes(NPTypeCode code) => code switch
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
    }
}
