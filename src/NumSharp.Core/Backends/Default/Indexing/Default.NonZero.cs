using System;
using NumSharp.Generic;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static NDArray<long>[] NonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => nonzeros<T>(nd.MakeGeneric<T>());

        private static long CountNonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => count_nonzero<T>(nd.MakeGeneric<T>());

        private static void CountNonZeroAxisDispatch<T>(NDArray nd, NDArray result, int axis) where T : unmanaged
            => count_nonzero_axis<T>(nd.MakeGeneric<T>(), result, axis);

        public override NDArray<long>[] NonZero(NDArray nd)
        {
            return NpFunc.Invoke(nd.typecode, NonZeroDispatch<int>, nd);
        }

        /// <summary>
        /// Generic implementation of nonzero using ILKernelGenerator.
        /// Uses coordinate-based iteration via ILKernelGenerator for all cases.
        /// </summary>
        private static unsafe NDArray<long>[] nonzeros<T>(NDArray<T> x) where T : unmanaged
        {
            // Ensure at least 1D (NumPy behavior)
            x = np.atleast_1d(x).MakeGeneric<T>();
            var shape = x.Shape;
            var size = x.size;
            var ndim = x.ndim;

            // Handle empty arrays: return tuple of empty arrays (one per dimension)
            // NumPy: np.nonzero(np.array([])) -> (array([], dtype=int64),)
            if (size == 0)
            {
                var emptyResult = new NDArray<long>[ndim];
                for (int i = 0; i < ndim; i++)
                    emptyResult[i] = new NDArray<long>(0);
                return emptyResult;
            }

            // Use strided helper for all cases (handles both contiguous and non-contiguous)
            // The ILKernelGenerator.FindNonZeroStridedHelper uses coordinate-based iteration
            return ILKernelGenerator.FindNonZeroStridedHelper((T*)x.Address, shape.dimensions, shape.strides, shape.offset);
        }

        /// <summary>
        /// Count the number of non-zero elements in the array.
        /// </summary>
        /// <remarks>
        /// NumPy-aligned: np.count_nonzero([0, 1, 0, 2]) = 2
        /// </remarks>
        public override long CountNonZero(NDArray nd)
        {
            if (nd.size == 0)
                return 0;

            return NpFunc.Invoke(nd.typecode, CountNonZeroDispatch<int>, nd);
        }

        /// <summary>
        /// Count non-zero elements along a specific axis.
        /// </summary>
        public override NDArray CountNonZero(NDArray nd, int axis, bool keepdims = false)
        {
            var shape = nd.Shape;

            // Normalize axis
            while (axis < 0)
                axis = nd.ndim + axis;
            if (axis >= nd.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            // Compute output shape
            var outputDims = new long[nd.ndim - 1];
            for (int d = 0, od = 0; d < nd.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Int64, outputShape, false);

            if (nd.size == 0)
            {
                // Already zeros from allocation
                if (keepdims)
                {
                    var ks = new long[nd.ndim];
                    for (int d = 0, sd = 0; d < nd.ndim; d++)
                        ks[d] = (d == axis) ? 1 : outputDims[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            NpFunc.Invoke(nd.typecode, CountNonZeroAxisDispatch<int>, nd, result, axis);

            if (keepdims)
            {
                var ks = new long[nd.ndim];
                for (int d = 0, sd = 0; d < nd.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }

        /// <summary>
        /// Generic implementation of count_nonzero (element-wise).
        /// </summary>
        private static unsafe long count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;

            if (shape.IsContiguous)
            {
                // Fast path for contiguous arrays
                T* ptr = (T*)x.Address;
                T zero = default;
                long count = 0;
                for (long i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
                return count;
            }

            // Strided path: use NpyIter for layout-aware traversal.
            using var iter = NpyIterRef.New(x, NpyIterGlobalFlags.EXTERNAL_LOOP);
            return iter.ExecuteReducing<CountNonZeroKernel<T>, long>(default, 0L);
        }

        /// <summary>
        /// Count non-zero elements along an axis.
        /// </summary>
        private static unsafe void count_nonzero_axis<T>(NDArray<T> x, NDArray result, int axis) where T : unmanaged
        {
            var shape = x.Shape;
            long axisSize = shape.dimensions[axis];
            var outputSize = result.size;
            T zero = default;

            // Compute output dimension strides for coordinate calculation
            int outputNdim = x.ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * shape.dimensions[nextInputDim];
                }
            }

            long axisStride = shape.strides[axis];

            // Use direct pointer access to result array (result is contiguous Int64)
            long* resultPtr = (long*)result.Address;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to input coordinates
                long remaining = outIdx;
                long inputBaseOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                // Count non-zeros along axis
                long count = 0;
                T* basePtr = (T*)x.Address + shape.offset + inputBaseOffset;
                for (long i = 0; i < axisSize; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(basePtr[i * axisStride], zero))
                        count++;
                }

                // Write directly to result buffer using linear index
                resultPtr[outIdx] = count;
            }
        }

    }
}
