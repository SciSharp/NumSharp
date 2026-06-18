using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null, NDArray @out = null)
        {
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

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, @out);

            if (axis_ == null)
                return HandleElementWiseSum(arr, keepdims, typeCode, @out);

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            if (shape[axis] == 1)
                return HandleTrivialAxisReduction(arr, axis, keepdims, outputType, @out);

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, @out, ReductionOp.Sum);
        }

        private NDArray HandleElementWiseSum(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var result = sum_elementwise_il(arr, typeCode);
            if (@out is not null) { @out.SetAtIndex(result, 0); return @out; }
            var r = NDArray.Scalar(result);
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r;
        }

        private unsafe NDArray ExecuteAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // NpyIter-driven per-chunk path (the migration target). Currently serves the
            // (dtype, op) combinations that the legacy DirectILKernelGenerator path covers
            // only with a slow scalar kernel — Complex sum/prod/min/max. Returns null when
            // no per-chunk kernel exists yet, so we fall through to the Direct path below.
            if (UseNpyIterReduce(inputType, outputType, op))
            {
                var npyIterResult = ExecuteAxisReductionNpyIter(arr, axis, keepdims, outputType, @out, op);
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
            else result = new NDArray(outputType, outputShape, false);

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
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        /// <summary>
        ///     Gate for the NpyIter-driven per-chunk reduction path. Returns true only for
        ///     (dtype, op) combinations that have a kernel in
        ///     <see cref="Kernels.ILKernelGenerator.GetReduceInnerLoop"/>; everything else
        ///     stays on the legacy <see cref="Kernels.DirectILKernelGenerator"/> path.
        ///     Acts as the per-dtype rollback switch (Plan §6).
        /// </summary>
        private static bool UseNpyIterReduce(NPTypeCode inputType, NPTypeCode outputType, ReductionOp op)
        {
            // Complex same-type sum/prod/min/max/mean. The legacy complex axis paths were:
            // a scalar axis kernel (sum/prod/min/max — already ~parity under -c Release) and,
            // for the DEFAULT complex mean, the per-output-row-allocating MeanAxisComplex
            // (15–45× slower than NumPy — the genuine bottleneck). The NpyIter double-pair
            // path puts all five on the migration-target architecture at parity-or-better and
            // collapses mean to a one-pass sum kernel + scalar divide.
            if (inputType == NPTypeCode.Complex && outputType == NPTypeCode.Complex)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Half MEAN only: it accumulates in Double (hardware-fast), a clean win on both
            // axes (10M: 57→15 ms axis0, 24→13 ms axis1) and the analog of the complex-mean
            // fix. Half SUM/PROD must accumulate in Half to reproduce NumPy's f16 SEQUENTIAL
            // rounding (the 2048-saturation on 4096 ones) — a serial software-arithmetic chain
            // .NET cannot beat the legacy path on for the pinned (last-axis) case (~+30%), so
            // those (and min/max) stay on the Direct path. outputType==Double for Half mean
            // (ReduceMean casts the result back to Half).
            if (inputType == NPTypeCode.Half)
                return op == ReductionOp.Mean && outputType == NPTypeCode.Double;

            // Decimal: the legacy path is both cache-hostile AND lossy (it accumulates through
            // a double bridge). The NpyIter kernels are full-precision Decimal on contiguous
            // stripes — 7–12× faster everywhere AND more accurate. No NumPy reference type.
            if (inputType == NPTypeCode.Decimal && outputType == NPTypeCode.Decimal)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Phase 6 — numeric migration onto the per-chunk target architecture.
            // Double AND float32 SUM and MEAN (mean = Sum kernel + MeanDivideByCount).
            // The PINNED path uses PairwiseFold (ported 1:1 from NumPy's pairwise_sum),
            // so results are BIT-FOR-BIT identical to NumPy for both dtypes — which is
            // what makes float32 safe to route (its earlier exclusion was a flat-
            // accumulator divergence the pairwise leaf removes). SLAB stays the
            // streaming Vector256 add (already bit-matches NumPy on that orientation).
            // Integer Sum (NEP50 widening), Prod, and Min/Max stay on the Direct path
            // (GetReduceInnerLoop returns null for those → caller falls back).
            if ((inputType == NPTypeCode.Double || inputType == NPTypeCode.Single) && outputType == inputType)
                return op == ReductionOp.Sum || op == ReductionOp.Mean;

            return false;
        }

        /// <summary>
        ///     Axis reduction via the 2-operand REDUCE iterator + a per-chunk
        ///     <see cref="Kernels.ILKernelGenerator"/> kernel. Mirrors
        ///     <see cref="ExecuteAxisReduction"/>'s output-shape / keepdims / out= handling,
        ///     but seeds the reduction identity and lets the iterator drive the inner loop.
        ///     Returns null when no per-chunk kernel exists for the key (caller falls back).
        /// </summary>
        private unsafe NDArray ExecuteAxisReductionNpyIter(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
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
                result = new NDArray(outputType, outputShape, false);
            }

            // The per-chunk kernel folds into the existing output slot(s), so the output
            // must carry the reduction identity (0/1/±inf) before the iterator runs.
            ILKernelGenerator.SeedReduceIdentity(result, kernelOp);

            // COPY_IF_OVERLAP only matters when a user-supplied out= may alias the input; a
            // fresh allocation can never overlap, so the hot path skips the overlap probe.
            var extraFlags = @out is not null ? NpyIterGlobalFlags.COPY_IF_OVERLAP : NpyIterGlobalFlags.None;
            using (var iter = NpyIterRef.NewReduce(arr, result, axis, extraFlags))
                iter.ForEach(kernel);

            // Mean: divide the accumulated sums by the reduced-axis length (NumPy parity).
            if (op == ReductionOp.Mean)
                ILKernelGenerator.MeanDivideByCount(result, shape.dimensions[axis]);

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
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
                return r;
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

        private NDArray HandleScalarReduction(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
            if (@out is not null) { @out.SetAtIndex(r.GetAtIndex(0), 0); return @out; }
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r;
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
                return NDArray.Scalar(v);
            }
            var result = new NDArray(outputType, new Shape(resultDims), false);
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
