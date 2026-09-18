using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            var shape = arr.Shape;

            if (shape.IsEmpty)
            {
                // NumPy parity: sum of empty array uses accumulating type (int/bool -> int64/uint64, floats preserved).
                var defaultType = typeCode ?? arr.typecode.GetAccumulatingType();
                var defaultVal = defaultType.GetDefaultValue();
                if (@out is not null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                return NDArray.Scalar(defaultVal);
            }

            if (shape.size == 0)
                return HandleEmptyArrayReduction(arr, axis_, keepdims, typeCode, @out, ReductionOp.Sum);

            // Degenerate flat reduction (0-d scalar or single-element 1-D). NumPy still applies the
            // NEP50 sum accumulator here — sum(int32)->int64, sum(bool)->int64 — REGARDLESS of size,
            // so resolve the accumulating dtype at the call site (the n>=2 path below does the same via
            // GetAccumulatingType). HandleScalarReduction is shared with amin/amax, which must PRESERVE
            // the input dtype and therefore keep passing the raw (possibly-null) typeCode; the widening
            // is a property of sum/prod, not of the helper.
            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode ?? arr.GetTypeCode.GetAccumulatingType(), @out);

            if (axis_ == null)
                return HandleElementWiseSum(arr, keepdims, typeCode, @out);

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            if (shape[axis] == 1)
                return HandleTrivialAxisReduction(arr, axis, keepdims, outputType, @out);

            // Half AXIS sum: reproduce NumPy's HALF_add reduce (float32 intermediate, narrowed per
            // inner-loop call) in a FLOAT32 shadow accumulator, then narrow to Half ONCE. The NDIter
            // HalfSumKernel rounds the shadow to float16 precision (RoundToF16) per call — PINNED
            // (contiguous reduced axis) folds the stripe in float32 pairwise (ones(4096)->4096); SLAB
            // (reduced axis outer) rounds per step and SATURATES (ones((4096,3),axis=0)->2048). The
            // float32 shadow (vs re-widening the half accumulator from memory every step) + the
            // in-place RoundToF16 is what makes the large-kept SLAB beat NumPy instead of losing to
            // its hardware F16C. The OLD Double-accumulator path returned 4096 for that SLAB case — a
            // real divergence this fixes. An explicit dtype request is honored by the normal path.
            if (arr.typecode == NPTypeCode.Half && typeCode == null)
            {
                // Both temporaries are this branch's own intermediates: `wide` is consumed by the
                // narrowing astype COPY, and `halfResult` is element-copied into @out when one is
                // given — leaving either undisposed stranded a pooled buffer per f16 axis sum
                // (caught by UndisposedIntermediateTests via the instance oracle tier, 2026-09-18).
                var wide = ExecuteAxisReduction(arr, axis, keepdims, NPTypeCode.Single, null, ReductionOp.Sum);
                NDArray halfResult;
                try
                {
                    halfResult = wide.astype(NPTypeCode.Half);
                }
                finally
                {
                    wide.Dispose();
                }
                if (@out is null) return halfResult;
                try
                {
                    for (long i = 0; i < halfResult.size; i++) @out.SetAtIndex(halfResult.GetAtIndex(i), i);
                }
                finally
                {
                    halfResult.Dispose();
                }
                return @out;
            }

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, @out, ReductionOp.Sum);
        }

        private NDArray HandleElementWiseSum(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var result = sum_elementwise_il(arr, typeCode);
            if (@out is not null) { @out.SetAtIndex(result, 0); return @out; }
            var r = NDArray.Scalar(result);
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r.MarkReductionScalar();
        }

        private unsafe NDArray ExecuteAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // NDIter-driven per-chunk path (the migration target). Currently serves the
            // (dtype, op) combinations that the legacy DirectILKernelGenerator path covers
            // only with a slow scalar kernel — Complex sum/prod/min/max. Returns null when
            // no per-chunk kernel exists yet, so we fall through to the Direct path below.
            bool useNDIter = UseNDIterReduce(inputType, outputType, op);
            // f64/f32 Min/Max only WIN on the stride-ordered NDIter reduce path where the
            // Direct whole-array kernel collapses to a cache-hostile coordinate walk — i.e.
            // broadcast (stride 0) or negative strides. For C/F-contiguous and positive-strided
            // inputs the Direct kernel's per-array (not per-stripe) traversal is faster, so keep
            // it there. (Sum/Mean always prefer NDIter — their pairwise/streaming kernels beat
            // Direct on every layout. Complex/Decimal Min/Max are NOT gated: their NDIter path
            // is a strict win over the legacy scalar Direct kernel on all layouts.)
            if (useNDIter && (op == ReductionOp.Min || op == ReductionOp.Max)
                && (inputType == NPTypeCode.Double || inputType == NPTypeCode.Single)
                && !MinMaxLayoutFavorsNDIter(shape))
                useNDIter = false;
            if (useNDIter)
            {
                var npyIterResult = ExecuteAxisReductionNDIter(arr, axis, keepdims, outputType, @out, op);
                if (npyIterResult is not null) return npyIterResult;
            }

            var key = new AxisReductionKernelKey(inputType, outputType, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetAxisReductionKernel(key);
            if (kernel == null)
                throw new NotSupportedException($"Axis reduction kernel not available for {op}({inputType}) -> {outputType}.");

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++) if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            NDArray result;
            if (@out is not null) { if (@out.Shape != outputShape) throw new IncorrectShapeException($"Output shape mismatch"); result = @out; }
            else result = AllocateReductionResult(outputType, outputDims, shape);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
                result.Storage.ExpandDimension(axis);
            // A fresh 0-d result (1-D input reduced over its only axis) is a numpy SCALAR at the
            // boundary — read-only; an out= operand returns writeable (PyArray_Return semantics).
            return @out is not null ? result : result.MarkReductionScalar();
        }

        /// <summary>
        ///     Allocates a reduction's output in the memory order NumPy's reduce iterator would pick
        ///     with KEEPORDER: an F-contiguous input yields an F-contiguous result, a C-contiguous or
        ///     general-strided input yields a C-contiguous result (issue #610). The reduction kernels
        ///     write each output element through <c>outputStrides</c> (the general/slab paths use
        ///     them directly; the C-contiguous fast paths are gated on a C-contiguous INPUT, so they
        ///     are never reached for an F-contiguous input), so filling an F-strided buffer costs no
        ///     extra copy — this is exactly how NumPy allocates the output operand and writes into it,
        ///     rather than reordering after the fact. <paramref name="outputDims"/> is the input
        ///     shape with the reduced axis removed; a 0-D or 1-D result is intrinsically both C- and
        ///     F-contiguous, so only rank &gt;= 2 results consult the order. keepdims re-inserts the
        ///     reduced axis with <see cref="UnmanagedStorage.ExpandDimension"/>, which preserves the
        ///     order (unlike a Reshape to a fresh C-shape, which would reset it). ArgMax/ArgMin do NOT
        ///     use this: NumPy allocates their index output in C-order regardless of input (probed 2.4.2).
        /// </summary>
        internal static NDArray AllocateReductionResult(NPTypeCode outputType, long[] outputDims, Shape inputShape)
        {
            if (outputDims.Length == 0)
                return new NDArray(outputType, Shape.Scalar, false);
            var order = outputDims.Length >= 2 ? OrderResolver.Resolve('K', inputShape) : 'C';
            var outputShape = order == 'F' ? new Shape(outputDims, 'F') : new Shape(outputDims);
            return new NDArray(outputType, outputShape, false);
        }

        /// <summary>
        ///     Zero-filled sibling of <see cref="AllocateReductionResult"/>, for the degenerate
        ///     size-1-axis std/var paths whose result is all zeros: the layout still follows the
        ///     input's memory order (F for an F-contig input), matching NumPy (issue #610). Unlike
        ///     the executors above, <paramref name="dims"/> is already the FINAL result shape (the
        ///     caller having applied keepdims), so no ExpandDimension follows.
        /// </summary>
        internal static NDArray AllocateReductionZeros(NPTypeCode outputType, long[] dims, Shape inputShape)
        {
            var order = dims.Length >= 2 ? OrderResolver.Resolve('K', inputShape) : 'C';
            var zeroShape = order == 'F' ? new Shape(dims, 'F') : new Shape(dims);
            return np.zeros(zeroShape, outputType);
        }

        /// <summary>
        ///     Gate for the NDIter-driven per-chunk reduction path. Returns true only for
        ///     (dtype, op) combinations that have a kernel in
        ///     <see cref="Kernels.ILKernelGenerator.GetReduceInnerLoop"/>; everything else
        ///     stays on the legacy <see cref="Kernels.DirectILKernelGenerator"/> path.
        ///     Acts as the per-dtype rollback switch (Plan §6).
        /// </summary>
        private static bool UseNDIterReduce(NPTypeCode inputType, NPTypeCode outputType, ReductionOp op)
        {
            // Complex same-type sum/prod/min/max/mean. The legacy complex axis paths were:
            // a scalar axis kernel (sum/prod/min/max — already ~parity under -c Release) and,
            // for the DEFAULT complex mean, the per-output-row-allocating MeanAxisComplex
            // (15–45× slower than NumPy — the genuine bottleneck). The NDIter double-pair
            // path puts all five on the migration-target architecture at parity-or-better and
            // collapses mean to a one-pass sum kernel + scalar divide.
            if (inputType == NPTypeCode.Complex && outputType == NPTypeCode.Complex)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Half SUM routes to the float32-shadow HalfSumKernel (outputType==Single; ReduceAdd
            // narrows to Half) — NumPy's exact per-orientation reduce via per-call RoundToF16 (PINNED
            // pairwise, SLAB per-step-round saturation). Half MEAN still accumulates in Double then
            // divides and casts back (outputType==Double). Half PROD/MIN/MAX stay on the Direct path.
            if (inputType == NPTypeCode.Half)
                return (op == ReductionOp.Sum && outputType == NPTypeCode.Single)
                    || (op == ReductionOp.Mean && outputType == NPTypeCode.Double);

            // Decimal: the legacy path is both cache-hostile AND lossy (it accumulates through
            // a double bridge). The NDIter kernels are full-precision Decimal on contiguous
            // stripes — 7–12× faster everywhere AND more accurate. No NumPy reference type.
            if (inputType == NPTypeCode.Decimal && outputType == NPTypeCode.Decimal)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Phase 6 — numeric migration onto the per-chunk target architecture.
            // Double AND float32 SUM, MEAN (mean = Sum kernel + MeanDivideByCount), and
            // MIN/MAX. SUM's PINNED path uses PairwiseFold (ported 1:1 from NumPy's
            // pairwise_sum) so it is BIT-FOR-BIT identical to NumPy for both dtypes — which
            // is what makes float32 safe to route (its earlier exclusion was a flat-
            // accumulator divergence the pairwise leaf removes); SLAB stays the streaming
            // Vector256 add. MIN/MAX route to SimdMinMaxSameType (NaN-propagating, dual-mode)
            // so f64/f32 min/max are SIMD on every layout instead of collapsing on the
            // C-contiguity-gated Direct kernel for negcol/broadcast/strided inputs.
            // Integer Sum (NEP50 widening), integer Min/Max, and Prod stay on the Direct
            // path (GetReduceInnerLoop returns null for those → caller falls back).
            if ((inputType == NPTypeCode.Double || inputType == NPTypeCode.Single) && outputType == inputType)
                return op == ReductionOp.Sum || op == ReductionOp.Mean ||
                       op == ReductionOp.Min || op == ReductionOp.Max;

            return false;
        }

        /// <summary>
        ///     Layout gate for the f64/f32 Min/Max NDIter routing (see ExecuteAxisReduction).
        ///     Returns true only where the Direct whole-array axis kernel collapses to a
        ///     cache-hostile coordinate walk — broadcast (stride 0, repeated reads) or any
        ///     negative stride (backward traversal). Contiguous and positive-strided inputs
        ///     stay on the faster Direct kernel.
        /// </summary>
        private static bool MinMaxLayoutFavorsNDIter(Shape shape)
        {
            if (shape.IsBroadcasted) return true;
            var strides = shape.strides;
            for (int i = 0; i < strides.Length; i++)
                if (strides[i] < 0) return true;
            return false;
        }

        /// <summary>
        ///     Axis reduction via the 2-operand REDUCE iterator + a per-chunk
        ///     <see cref="Kernels.ILKernelGenerator"/> kernel. Mirrors
        ///     <see cref="ExecuteAxisReduction"/>'s output-shape / keepdims / out= handling,
        ///     but seeds the reduction identity and lets the iterator drive the inner loop.
        ///     Returns null when no per-chunk kernel exists for the key (caller falls back).
        /// </summary>
        private unsafe NDArray ExecuteAxisReductionNDIter(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            // Mean is computed as a one-pass Sum kernel followed by a scalar divide by the
            // reduced-axis length — there is no separate "mean" inner loop.
            var kernelOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
            var key = new ILKernelGenerator.ReduceKernelKey(kernelOp, arr.GetTypeCode, outputType);
            var kernel = ILKernelGenerator.GetReduceInnerLoop(key);
            if (kernel is null) return null;

            var shape = arr.Shape;
            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++) if (d != axis) outputDims[od++] = shape.dimensions[d];
            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;

            NDArray result;
            if (@out is not null)
            {
                if (@out.Shape != outputShape) throw new IncorrectShapeException($"Output shape mismatch");
                result = @out;
            }
            else
            {
                result = AllocateReductionResult(outputType, outputDims, shape);
            }

            // The per-chunk kernel folds into the existing output slot(s), so the output
            // must carry the reduction identity (0/1/±inf) before the iterator runs.
            ILKernelGenerator.SeedReduceIdentity(result, kernelOp);

            // COPY_IF_OVERLAP only matters when a user-supplied out= may alias the input; a
            // fresh allocation can never overlap, so the hot path skips the overlap probe.
            var extraFlags = @out is not null ? NDIterGlobalFlags.COPY_IF_OVERLAP : NDIterGlobalFlags.None;
            using (var iter = NDIterRef.NewReduce(arr, result, axis, extraFlags))
                iter.ForEach(kernel);

            // Mean: divide the accumulated sums by the reduced-axis length (NumPy parity).
            if (op == ReductionOp.Mean)
                ILKernelGenerator.MeanDivideByCount(result, shape.dimensions[axis]);

            if (keepdims)
                result.Storage.ExpandDimension(axis);
            // Same PyArray_Return rule as ExecuteAxisReduction's exit; the output was already
            // allocated in KEEPORDER (F for an F-contig input), so there is nothing to reorder.
            return @out is not null ? result : result.MarkReductionScalar();
        }

        /// <summary>
        /// Handle empty array min/max reductions.
        /// NumPy behavior:
        /// - np.min([]) raises ValueError (no identity for min/max)
        /// - np.min(zeros((0,3)), axis=0) raises ValueError (reducing along empty dimension)
        /// - np.min(zeros((0,3)), axis=1) returns [] with shape (0,) (output is also empty)
        /// </summary>
        private NDArray HandleEmptyArrayMinMaxReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, string opName)
        {
            var shape = arr.Shape;

            // No axis specified - always throw for empty arrays (no identity element for min/max)
            if (axis_ == null)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            var axis = NormalizeAxis(axis_.Value, arr.ndim);

            // If the axis being reduced has size 0, we're reducing over an empty dimension
            // which results in an error (no values to take min/max of)
            if (shape.dimensions[axis] == 0)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            // If the axis being reduced has size > 0, but the result would be empty,
            // return an empty array of the correct shape
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode;
            var result = new NDArray(outputType, new Shape(resultShape), false);

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        private NDArray HandleEmptyArrayReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            if (axis_ == null)
            {
                // NumPy parity: empty reduction uses accumulating type (int/bool -> int64/uint64, floats preserved).
                var defaultType = typeCode ?? arr.typecode.GetAccumulatingType();
                var defaultVal = defaultType.GetDefaultValue();
                if (@out is not null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                var r = NDArray.Scalar(defaultVal);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                return r.MarkReductionScalar();
            }
            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();
            var result = np.zeros(new Shape(resultShape), outputType);
            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            if (@out is not null) { np.copyto(@out, result); return @out; }
            return result;
        }

        /// <summary>
        ///     Degenerate flat reduction of a 0-d scalar or single-element 1-D array: the result is the
        ///     lone element, reshaped to a numpy scalar (0-d) or, under <paramref name="keepdims"/>, to
        ///     all-ones. Shared by sum/prod (which pass their RESOLVED accumulating dtype, so the element
        ///     is cast/widened) and amin/amax (which pass <c>null</c> to PRESERVE the input dtype) — the
        ///     helper does not decide widening, the caller does. A null <paramref name="typeCode"/>
        ///     therefore clones the input verbatim; a non-null one casts to it.
        /// </summary>
        /// <param name="arr">The size-&#8804;1 operand.</param>
        /// <param name="keepdims">When true, the result keeps the input rank with every axis length 1.</param>
        /// <param name="typeCode">The resolved output dtype (already widened for sum/prod), or null to preserve <paramref name="arr"/>'s dtype (amin/amax).</param>
        /// <param name="out">Optional pre-allocated output; when non-null the single value is written into slot 0 and it is returned.</param>
        /// <returns>The reduced scalar as an <see cref="NDArray"/> — a read-only numpy scalar on the 0-d exit, or the writeable <paramref name="out"/> when supplied.</returns>
        private NDArray HandleScalarReduction(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
            if (@out is not null) { @out.SetAtIndex(r.GetAtIndex(0), 0); return @out; }
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            // A 0-d exit — including keepdims over a 0-d input, whose (1,)*0 reshape is still
            // 0-d — is a numpy SCALAR (read-only); (1,) keepdims results pass through writeable.
            return r.MarkReductionScalar();
        }

        private NDArray HandleTrivialAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out)
        {
            if (@out is not null) return null;
            var shape = arr.Shape;
            long[] resultDims;
            if (keepdims) { resultDims = (long[])shape.dimensions.Clone(); resultDims[axis] = 1; }
            else { resultDims = new long[arr.ndim - 1]; for (int d = 0, rd = 0; d < arr.ndim; d++) if (d != axis) resultDims[rd++] = shape.dimensions[d]; }
            if (resultDims.Length == 0)
            {
                var v = arr.GetAtIndex(0);
                if (outputType != arr.GetTypeCode) v = Converts.ChangeType(v, outputType);
                return NDArray.Scalar(v).MarkReductionScalar();
            }
            // KEEPORDER: reducing a size-1 axis of an F-contiguous input keeps F-contig; SetAtIndex
            // writes each element through the result's strides, so an F-order buffer fills correctly
            // with no copy (resultDims already carries the final keepdims shape) (issue #610).
            var result = AllocateReductionResult(outputType, resultDims, shape);
            if (outputType == arr.GetTypeCode) for (long i = 0; i < result.size; i++) result.SetAtIndex(arr.GetAtIndex(i), i);
            else for (long i = 0; i < result.size; i++) result.SetAtIndex(Converts.ChangeType(arr.GetAtIndex(i), outputType), i);
            return result;
        }

        /// <summary>
        ///     Normalizes a possibly-negative axis to a non-negative index and validates bounds.
        ///     Matches NumPy's axis normalization exactly.
        /// </summary>
        /// <param name="axis">The axis value (can be negative).</param>
        /// <param name="ndim">The number of dimensions in the array.</param>
        /// <returns>The normalized non-negative axis.</returns>
        /// <exception cref="AxisError">If the axis is out of bounds after normalization.</exception>
        internal static int NormalizeAxis(int axis, int ndim)
        {
            int originalAxis = axis;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new AxisError(originalAxis, ndim);
            return axis;
        }
    }
}
