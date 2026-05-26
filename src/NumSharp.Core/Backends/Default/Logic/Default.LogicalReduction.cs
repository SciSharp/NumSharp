using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray<bool> All(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: true);

        public NDArray<bool> Any(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: false);

        public NDArray<bool> All(NDArray nd, int[] axis, bool keepdims)
            => ReduceLogicalMultiAxis(nd, axis, keepdims, reduceAll: true);

        public NDArray<bool> Any(NDArray nd, int[] axis, bool keepdims)
            => ReduceLogicalMultiAxis(nd, axis, keepdims, reduceAll: false);

        private NDArray<bool> ReduceLogicalAxis(NDArray nd, int axis, bool keepdims, bool reduceAll)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            if (nd.ndim == 0)
            {
                if (axis == 0 || axis == -1)
                    return np.array(reduceAll ? All(nd) : Any(nd)).MakeGeneric<bool>();

                throw new AxisError(axis, 0);
            }

            axis = NormalizeAxis(axis, nd.ndim);

            // Allocate the result in the *reduced* shape (axis dropped). keepdims is applied
            // as a reshape at the end — matches Default.Reduction.Add.ExecuteAxisReduction and
            // lets the axis kernels assume one fewer output dim than input.
            var reducedDims = Shape.GetAxis(nd.Shape, axis);
            Shape reducedShape = reducedDims.Length == 0 ? Shape.Scalar : new Shape(reducedDims);
            NDArray<bool> result = CreateLogicalResult(reducedShape, reduceAll && nd.Shape.dimensions[axis] == 0);

            if (result.size == 0 || nd.Shape.dimensions[axis] == 0)
            {
                if (keepdims)
                    result.Storage.Reshape(BuildKeepdimsShape(nd.Shape, axis));
                return result;
            }

            // Fast path: IL-emitted axis kernel. Inner-axis (stride==1) routes through the
            // SIMD all/any helpers; non-inner uses AVX2 gather (float/double) or a scalar
            // early-exit loop. Returns null for unsupported dtypes (Half / Complex / Decimal),
            // which fall through to the NpyAxisIter scalar kernel below.
            ReductionOp op = reduceAll ? ReductionOp.All : ReductionOp.Any;
            var key = new AxisReductionKernelKey(nd.GetTypeCode, NPTypeCode.Boolean, op, InnerAxisContiguous: axis == nd.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetBooleanAxisReductionKernel(key);
            if (kernel != null && nd.Shape.IsContiguous)
            {
                unsafe
                {
                    fixed (long* inStrides = nd.Shape.strides)
                    fixed (long* inDims = nd.Shape.dimensions)
                    fixed (long* outStrides = result.Shape.strides)
                    {
                        byte* inBase = (byte*)nd.Storage.Address + nd.Shape.offset * nd.dtypesize;
                        long outSize = result.size > 0 ? result.size : 1;
                        kernel(inBase, (void*)result.Address, inStrides, inDims, outStrides,
                               axis, nd.Shape.dimensions[axis], nd.ndim, outSize);
                    }
                }
            }
            else
            {
                NpFunc.Invoke(nd.GetTypeCode, ExecuteLogicalAxis<int>, nd, result, axis, reduceAll);
            }

            if (keepdims)
                result.Storage.Reshape(BuildKeepdimsShape(nd.Shape, axis));

            return result;
        }

        // Build the "keepdims" shape: input shape with the reduction axis set to size 1.
        private static Shape BuildKeepdimsShape(Shape inputShape, int axis)
        {
            var dims = (long[])inputShape.dimensions.Clone();
            dims[axis] = 1;
            return new Shape(dims);
        }

        // Multi-axis reduction. Matches NumPy: reduces along all listed axes.
        // Fast paths:
        //   1. All axes reduced → 1-D SIMD `all`/`any` over the whole array.
        //   2. Adjacent axes on a C-contig input → reshape to fuse them, then a single
        //      axis reduction. Saves N-1 redundant kernel invocations.
        //   3. Otherwise → chain single-axis reductions with keepdims=true.
        private NDArray<bool> ReduceLogicalMultiAxis(NDArray nd, int[] axis, bool keepdims, bool reduceAll)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            // Empty tuple: NumPy returns input cast to bool (no reduction).
            if (axis.Length == 0)
                return CastToBoolPreservingShape(nd);

            // 0-d input
            if (nd.ndim == 0)
            {
                // Only valid axis values are 0 / -1 (and only one of them, per NumPy)
                if (axis.Length != 1 || (axis[0] != 0 && axis[0] != -1))
                    throw new AxisError(axis.Length > 0 ? axis[0] : 0, 0);

                return np.array(reduceAll ? All(nd) : Any(nd)).MakeGeneric<bool>();
            }

            int ndim = nd.ndim;
            int[] normalized = NormalizeAndValidateAxes(axis, ndim);
            Array.Sort(normalized);

            // Fast path 1: every axis reduced → 1-D path handles the whole array in one
            // SIMD pass instead of N chained axis reductions through a scalar kernel.
            if (normalized.Length == ndim)
            {
                bool scalar = reduceAll ? All(nd) : Any(nd);
                var result = np.array(scalar).MakeGeneric<bool>();
                if (keepdims)
                {
                    var dims = new long[ndim];
                    for (int i = 0; i < ndim; i++) dims[i] = 1;
                    result.Storage.Reshape(new Shape(dims));
                }
                return result;
            }

            // Fast path 2: contiguous run of axes on a C-contig input can be fused
            // (the reshape is free, no copy). E.g. axis=(0,1) of (128, 64, 64) C-contig
            // becomes a single axis-0 reduction on (8192, 64).
            if (nd.Shape.IsContiguous && AreContiguousRun(normalized))
            {
                NDArray<bool> result = ReduceContiguousAxisRun(nd, normalized, keepdims, reduceAll);
                return result;
            }

            // Fall back: chain single-axis reductions, highest axis first so lower
            // indices remain valid across passes.
            NDArray<bool> chained = null;
            NDArray current = nd;
            for (int i = normalized.Length - 1; i >= 0; i--)
            {
                chained = ReduceLogicalAxis(current, normalized[i], keepdims: true, reduceAll: reduceAll);
                current = chained;
            }

            if (!keepdims)
            {
                var newDims = new List<long>(ndim - normalized.Length);
                var axesSet = new HashSet<int>(normalized);
                long[] resultShape = chained.shape;
                for (int d = 0; d < ndim; d++)
                {
                    if (!axesSet.Contains(d))
                        newDims.Add(resultShape[d]);
                }

                Shape newShape = newDims.Count == 0 ? Shape.Scalar : new Shape(newDims.ToArray());
                chained.Storage.Reshape(newShape);
            }

            return chained;
        }

        // True iff `sorted` (already sorted) is a strict +1 progression: e.g. (1, 2, 3).
        private static bool AreContiguousRun(int[] sorted)
        {
            for (int i = 1; i < sorted.Length; i++)
            {
                if (sorted[i] != sorted[i - 1] + 1)
                    return false;
            }
            return true;
        }

        // Fuse a contiguous run of reduced axes into a single axis via reshape, then
        // run one single-axis reduction. Caller has verified C-contiguous storage.
        private NDArray<bool> ReduceContiguousAxisRun(NDArray nd, int[] sortedAxes, bool keepdims, bool reduceAll)
        {
            int ndim = nd.ndim;
            long[] origDims = nd.shape;
            int firstAxis = sortedAxes[0];
            int lastAxis = sortedAxes[sortedAxes.Length - 1];

            // Build a reshape that collapses axes [firstAxis..lastAxis] into one axis.
            var newDims = new long[ndim - sortedAxes.Length + 1];
            int w = 0;
            for (int i = 0; i < firstAxis; i++) newDims[w++] = origDims[i];

            long fusedSize = 1;
            for (int i = firstAxis; i <= lastAxis; i++) fusedSize *= origDims[i];
            newDims[w++] = fusedSize;

            for (int i = lastAxis + 1; i < ndim; i++) newDims[w++] = origDims[i];

            // Reshape is a view on C-contig data (no copy).
            NDArray reshaped = nd.reshape(newDims);
            NDArray<bool> reduced = ReduceLogicalAxis(reshaped, firstAxis, keepdims: false, reduceAll: reduceAll);

            if (keepdims)
            {
                var dimsKD = (long[])origDims.Clone();
                foreach (int a in sortedAxes) dimsKD[a] = 1;
                reduced.Storage.Reshape(new Shape(dimsKD));
            }

            return reduced;
        }

        // Cast every element to bool (truthy = non-zero), preserving the input shape.
        // Used for the np.all(a, axis=()) / np.any(a, axis=()) case where NumPy
        // returns the array reinterpreted as bool without performing any reduction.
        internal static NDArray<bool> CastToBoolPreservingShape(NDArray nd)
        {
            if (nd.GetTypeCode == NPTypeCode.Boolean)
            {
                // Already bool — return a copy so callers can't mutate the input.
                return nd.copy().MakeGeneric<bool>();
            }

            // (nd != 0) returns an NDArray<bool> with element-wise non-zero check,
            // matching NumPy's truthiness semantics (including NaN→True, inf→True).
            return nd != 0;
        }

        // Resolve negative axis values, bounds-check, and reject duplicates (NumPy raises
        // ValueError: "duplicate value in 'axis'").
        internal static int[] NormalizeAndValidateAxes(int[] axes, int ndim)
        {
            var result = new int[axes.Length];
            var seen = new HashSet<int>();
            for (int i = 0; i < axes.Length; i++)
            {
                int a = NormalizeAxis(axes[i], ndim);
                if (!seen.Add(a))
                    throw new ArgumentException("duplicate value in 'axis'");
                result[i] = a;
            }
            return result;
        }

        private static Shape CreateLogicalResultShape(Shape inputShape, int axis, bool keepdims)
        {
            if (keepdims)
            {
                var dims = (long[])inputShape.dimensions.Clone();
                dims[axis] = 1;
                return new Shape(dims);
            }

            var reducedDims = Shape.GetAxis(inputShape, axis);
            return reducedDims.Length == 0 ? Shape.Scalar : new Shape(reducedDims);
        }

        private static NDArray<bool> CreateLogicalResult(Shape resultShape, bool fillTrue)
        {
            var result = fillTrue
                ? np.ones(resultShape, NPTypeCode.Boolean)
                : np.zeros(resultShape, NPTypeCode.Boolean);

            return result.MakeGeneric<bool>();
        }

        private static void ExecuteLogicalAxis<T>(NDArray nd, NDArray<bool> result, int axis, bool reduceAll)
            where T : unmanaged
        {
            if (reduceAll)
                NpyAxisIter.ReduceBool<T, NpyAllKernel<T>>(nd.Storage, result.Storage, axis);
            else
                NpyAxisIter.ReduceBool<T, NpyAnyKernel<T>>(nd.Storage, result.Storage, axis);
        }
    }
}
