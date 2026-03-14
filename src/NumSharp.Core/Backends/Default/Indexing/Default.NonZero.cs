using System;
using NumSharp.Generic;
using System.Collections.Generic;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Return the indices of non-zero elements.
        /// </summary>
        /// <remarks>
        /// NumPy-aligned behavior:
        /// - Returns tuple of arrays, one per dimension
        /// - For empty arrays, returns empty arrays with correct dtype (int)
        /// - Iterates in C-order (row-major)
        /// - Handles contiguous and strided arrays efficiently
        /// </remarks>
        /// <param name="nd">Input array</param>
        /// <returns>Array of NDArray&lt;int&gt;, one per dimension containing indices of non-zero elements</returns>
        public override NDArray<int>[] NonZero(NDArray nd)
        {
            // Type dispatch to generic implementation
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: return nonzeros<bool>(nd.MakeGeneric<bool>());
                case NPTypeCode.Byte: return nonzeros<byte>(nd.MakeGeneric<byte>());
                case NPTypeCode.Int16: return nonzeros<short>(nd.MakeGeneric<short>());
                case NPTypeCode.UInt16: return nonzeros<ushort>(nd.MakeGeneric<ushort>());
                case NPTypeCode.Int32: return nonzeros<int>(nd.MakeGeneric<int>());
                case NPTypeCode.UInt32: return nonzeros<uint>(nd.MakeGeneric<uint>());
                case NPTypeCode.Int64: return nonzeros<long>(nd.MakeGeneric<long>());
                case NPTypeCode.UInt64: return nonzeros<ulong>(nd.MakeGeneric<ulong>());
                case NPTypeCode.Char: return nonzeros<char>(nd.MakeGeneric<char>());
                case NPTypeCode.Double: return nonzeros<double>(nd.MakeGeneric<double>());
                case NPTypeCode.Single: return nonzeros<float>(nd.MakeGeneric<float>());
                case NPTypeCode.Decimal: return nonzeros<decimal>(nd.MakeGeneric<decimal>());
                default:
                    throw new NotSupportedException($"NonZero not supported for type {nd.typecode}");
            }
        }

        /// <summary>
        /// Generic implementation of nonzero using ILKernelGenerator.
        /// Both contiguous and strided paths now use the unified IL-based approach.
        /// </summary>
        private static unsafe NDArray<int>[] nonzeros<T>(NDArray<T> x) where T : unmanaged
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
                var emptyResult = new NDArray<int>[ndim];
                for (int i = 0; i < ndim; i++)
                    emptyResult[i] = new NDArray<int>(0);
                return emptyResult;
            }

            // SIMD fast path for contiguous arrays
            if (shape.IsContiguous && ILKernelGenerator.Enabled && ILKernelGenerator.VectorBits > 0)
            {
                var flatIndices = new List<int>(Math.Max(16, size / 4));
                kp.FindNonZero((T*)x.Address, size, flatIndices);
                return kp.ConvertFlatToCoordinates(flatIndices, x.shape);
            }

            // Strided path for non-contiguous arrays (transposed, sliced, etc.)
            // Uses coordinate-based iteration via ILKernelGenerator
            return ILKernelGenerator.FindNonZeroStridedHelper((T*)x.Address, shape.dimensions, shape.strides, shape.offset);
        }

        /// <summary>
        /// Count the number of non-zero elements in the array.
        /// </summary>
        /// <remarks>
        /// NumPy-aligned: np.count_nonzero([0, 1, 0, 2]) = 2
        /// </remarks>
        public override int CountNonZero(NDArray nd)
        {
            if (nd.size == 0)
                return 0;

            // Type dispatch to generic implementation
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: return count_nonzero<bool>(nd.MakeGeneric<bool>());
                case NPTypeCode.Byte: return count_nonzero<byte>(nd.MakeGeneric<byte>());
                case NPTypeCode.Int16: return count_nonzero<short>(nd.MakeGeneric<short>());
                case NPTypeCode.UInt16: return count_nonzero<ushort>(nd.MakeGeneric<ushort>());
                case NPTypeCode.Int32: return count_nonzero<int>(nd.MakeGeneric<int>());
                case NPTypeCode.UInt32: return count_nonzero<uint>(nd.MakeGeneric<uint>());
                case NPTypeCode.Int64: return count_nonzero<long>(nd.MakeGeneric<long>());
                case NPTypeCode.UInt64: return count_nonzero<ulong>(nd.MakeGeneric<ulong>());
                case NPTypeCode.Char: return count_nonzero<char>(nd.MakeGeneric<char>());
                case NPTypeCode.Double: return count_nonzero<double>(nd.MakeGeneric<double>());
                case NPTypeCode.Single: return count_nonzero<float>(nd.MakeGeneric<float>());
                case NPTypeCode.Decimal: return count_nonzero<decimal>(nd.MakeGeneric<decimal>());
                default:
                    throw new NotSupportedException($"CountNonZero not supported for type {nd.typecode}");
            }
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
            var outputDims = new int[nd.ndim - 1];
            for (int d = 0, od = 0; d < nd.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Int64, outputShape, false);

            if (nd.size == 0)
            {
                // Already zeros from allocation
                if (keepdims)
                {
                    var ks = new int[nd.ndim];
                    for (int d = 0, sd = 0; d < nd.ndim; d++)
                        ks[d] = (d == axis) ? 1 : outputDims[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            // Type dispatch
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: count_nonzero_axis<bool>(nd.MakeGeneric<bool>(), result, axis); break;
                case NPTypeCode.Byte: count_nonzero_axis<byte>(nd.MakeGeneric<byte>(), result, axis); break;
                case NPTypeCode.Int16: count_nonzero_axis<short>(nd.MakeGeneric<short>(), result, axis); break;
                case NPTypeCode.UInt16: count_nonzero_axis<ushort>(nd.MakeGeneric<ushort>(), result, axis); break;
                case NPTypeCode.Int32: count_nonzero_axis<int>(nd.MakeGeneric<int>(), result, axis); break;
                case NPTypeCode.UInt32: count_nonzero_axis<uint>(nd.MakeGeneric<uint>(), result, axis); break;
                case NPTypeCode.Int64: count_nonzero_axis<long>(nd.MakeGeneric<long>(), result, axis); break;
                case NPTypeCode.UInt64: count_nonzero_axis<ulong>(nd.MakeGeneric<ulong>(), result, axis); break;
                case NPTypeCode.Char: count_nonzero_axis<char>(nd.MakeGeneric<char>(), result, axis); break;
                case NPTypeCode.Double: count_nonzero_axis<double>(nd.MakeGeneric<double>(), result, axis); break;
                case NPTypeCode.Single: count_nonzero_axis<float>(nd.MakeGeneric<float>(), result, axis); break;
                case NPTypeCode.Decimal: count_nonzero_axis<decimal>(nd.MakeGeneric<decimal>(), result, axis); break;
                default:
                    throw new NotSupportedException($"CountNonZero not supported for type {nd.typecode}");
            }

            if (keepdims)
            {
                var ks = new int[nd.ndim];
                for (int d = 0, sd = 0; d < nd.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }

        /// <summary>
        /// Generic implementation of count_nonzero (element-wise).
        /// </summary>
        private static unsafe int count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;
            int count = 0;

            if (shape.IsContiguous)
            {
                // Fast path for contiguous arrays
                T* ptr = (T*)x.Address;
                T zero = default;
                for (int i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
            }
            else
            {
                // Strided path
                var iter = x.AsIterator<T>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                T zero = default;
                while (hasNext())
                {
                    if (!EqualityComparer<T>.Default.Equals(moveNext(), zero))
                        count++;
                }
            }

            return count;
        }

        /// <summary>
        /// Count non-zero elements along an axis.
        /// </summary>
        private static unsafe void count_nonzero_axis<T>(NDArray<T> x, NDArray result, int axis) where T : unmanaged
        {
            var shape = x.Shape;
            var axisSize = shape.dimensions[axis];
            var outputSize = result.size;
            T zero = default;

            // Compute output dimension strides for coordinate calculation
            int outputNdim = x.ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
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

            int axisStride = shape.strides[axis];

            // Use direct pointer access to result array (result is contiguous Int64)
            long* resultPtr = (long*)result.Address;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to input coordinates
                int remaining = outIdx;
                int inputBaseOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                // Count non-zeros along axis
                long count = 0;
                T* basePtr = (T*)x.Address + shape.offset + inputBaseOffset;
                for (int i = 0; i < axisSize; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(basePtr[i * axisStride], zero))
                        count++;
                }

                // Write directly to result buffer using linear index
                resultPtr[outIdx] = count;
            }
        }

        /// <summary>
        /// Count the number of non-zero elements in the array.
        /// </summary>
        /// <remarks>
        /// NumPy-aligned: np.count_nonzero([0, 1, 0, 2]) = 2
        /// </remarks>
        public override int CountNonZero(in NDArray nd)
        {
            if (nd.size == 0)
                return 0;

            // Type dispatch to generic implementation
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: return count_nonzero<bool>(nd.MakeGeneric<bool>());
                case NPTypeCode.Byte: return count_nonzero<byte>(nd.MakeGeneric<byte>());
                case NPTypeCode.Int16: return count_nonzero<short>(nd.MakeGeneric<short>());
                case NPTypeCode.UInt16: return count_nonzero<ushort>(nd.MakeGeneric<ushort>());
                case NPTypeCode.Int32: return count_nonzero<int>(nd.MakeGeneric<int>());
                case NPTypeCode.UInt32: return count_nonzero<uint>(nd.MakeGeneric<uint>());
                case NPTypeCode.Int64: return count_nonzero<long>(nd.MakeGeneric<long>());
                case NPTypeCode.UInt64: return count_nonzero<ulong>(nd.MakeGeneric<ulong>());
                case NPTypeCode.Char: return count_nonzero<char>(nd.MakeGeneric<char>());
                case NPTypeCode.Double: return count_nonzero<double>(nd.MakeGeneric<double>());
                case NPTypeCode.Single: return count_nonzero<float>(nd.MakeGeneric<float>());
                case NPTypeCode.Decimal: return count_nonzero<decimal>(nd.MakeGeneric<decimal>());
                default:
                    throw new NotSupportedException($"CountNonZero not supported for type {nd.typecode}");
            }
        }

        /// <summary>
        /// Count non-zero elements along a specific axis.
        /// </summary>
        public override NDArray CountNonZero(in NDArray nd, int axis, bool keepdims = false)
        {
            var shape = nd.Shape;

            // Normalize axis
            while (axis < 0)
                axis = nd.ndim + axis;
            if (axis >= nd.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            // Compute output shape
            var outputDims = new int[nd.ndim - 1];
            for (int d = 0, od = 0; d < nd.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Int64, outputShape, false);

            if (nd.size == 0)
            {
                // Already zeros from allocation
                if (keepdims)
                {
                    var ks = new int[nd.ndim];
                    for (int d = 0, sd = 0; d < nd.ndim; d++)
                        ks[d] = (d == axis) ? 1 : outputDims[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            // Type dispatch
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: count_nonzero_axis<bool>(nd.MakeGeneric<bool>(), result, axis); break;
                case NPTypeCode.Byte: count_nonzero_axis<byte>(nd.MakeGeneric<byte>(), result, axis); break;
                case NPTypeCode.Int16: count_nonzero_axis<short>(nd.MakeGeneric<short>(), result, axis); break;
                case NPTypeCode.UInt16: count_nonzero_axis<ushort>(nd.MakeGeneric<ushort>(), result, axis); break;
                case NPTypeCode.Int32: count_nonzero_axis<int>(nd.MakeGeneric<int>(), result, axis); break;
                case NPTypeCode.UInt32: count_nonzero_axis<uint>(nd.MakeGeneric<uint>(), result, axis); break;
                case NPTypeCode.Int64: count_nonzero_axis<long>(nd.MakeGeneric<long>(), result, axis); break;
                case NPTypeCode.UInt64: count_nonzero_axis<ulong>(nd.MakeGeneric<ulong>(), result, axis); break;
                case NPTypeCode.Char: count_nonzero_axis<char>(nd.MakeGeneric<char>(), result, axis); break;
                case NPTypeCode.Double: count_nonzero_axis<double>(nd.MakeGeneric<double>(), result, axis); break;
                case NPTypeCode.Single: count_nonzero_axis<float>(nd.MakeGeneric<float>(), result, axis); break;
                case NPTypeCode.Decimal: count_nonzero_axis<decimal>(nd.MakeGeneric<decimal>(), result, axis); break;
                default:
                    throw new NotSupportedException($"CountNonZero not supported for type {nd.typecode}");
            }

            if (keepdims)
            {
                var ks = new int[nd.ndim];
                for (int d = 0, sd = 0; d < nd.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }

        /// <summary>
        /// Generic implementation of count_nonzero (element-wise).
        /// </summary>
        private static unsafe int count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;
            int count = 0;

            if (shape.IsContiguous)
            {
                // Fast path for contiguous arrays
                T* ptr = (T*)x.Address;
                T zero = default;
                for (int i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
            }
            else
            {
                // Strided path
                var iter = x.AsIterator<T>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                T zero = default;
                while (hasNext())
                {
                    if (!EqualityComparer<T>.Default.Equals(moveNext(), zero))
                        count++;
                }
            }

            return count;
        }

        /// <summary>
        /// Count non-zero elements along an axis.
        /// </summary>
        private static unsafe void count_nonzero_axis<T>(NDArray<T> x, NDArray result, int axis) where T : unmanaged
        {
            var shape = x.Shape;
            var axisSize = shape.dimensions[axis];
            var outputSize = result.size;
            T zero = default;

            // Compute output dimension strides for coordinate calculation
            int outputNdim = x.ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
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

            int axisStride = shape.strides[axis];

            // Use direct pointer access to result array (result is contiguous Int64)
            long* resultPtr = (long*)result.Address;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to input coordinates
                int remaining = outIdx;
                int inputBaseOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                // Count non-zeros along axis
                long count = 0;
                T* basePtr = (T*)x.Address + shape.offset + inputBaseOffset;
                for (int i = 0; i < axisSize; i++)
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
