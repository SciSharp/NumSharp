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
        public override NDArray<int>[] NonZero(in NDArray nd)
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

            var kp = DefaultKernelProvider;

            // SIMD fast path for contiguous arrays
            if (shape.IsContiguous && kp.Enabled && kp.VectorBits > 0)
            {
                var flatIndices = new List<int>(Math.Max(16, size / 4));
                kp.FindNonZero((T*)x.Address, size, flatIndices);
                return kp.ConvertFlatToCoordinates(flatIndices, x.shape);
            }

            // Strided path for non-contiguous arrays (transposed, sliced, etc.)
            // Uses coordinate-based iteration via ILKernelGenerator
            return kp.FindNonZeroStrided((T*)x.Address, shape.dimensions, shape.strides, shape.offset);
        }
    }
}
