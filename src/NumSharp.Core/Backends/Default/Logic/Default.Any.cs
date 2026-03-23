using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test whether any array element evaluates to True (non-zero).
        /// Supports all 12 dtypes with SIMD optimization for contiguous arrays.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <returns>True if any element is non-zero</returns>
        public override bool Any(NDArray nd)
        {
            if (nd.size == 0)
                return false; // NumPy: any([]) == False

            // Dispatch by type
            return nd.GetTypeCode switch
            {
                NPTypeCode.Boolean => AnyImpl<bool>(nd),
                NPTypeCode.Byte => AnyImpl<byte>(nd),
                NPTypeCode.Int16 => AnyImpl<short>(nd),
                NPTypeCode.UInt16 => AnyImpl<ushort>(nd),
                NPTypeCode.Int32 => AnyImpl<int>(nd),
                NPTypeCode.UInt32 => AnyImpl<uint>(nd),
                NPTypeCode.Int64 => AnyImpl<long>(nd),
                NPTypeCode.UInt64 => AnyImpl<ulong>(nd),
                NPTypeCode.Char => AnyImpl<char>(nd),
                NPTypeCode.Single => AnyImpl<float>(nd),
                NPTypeCode.Double => AnyImpl<double>(nd),
                NPTypeCode.Decimal => AnyImplDecimal(nd),
                _ => throw new NotSupportedException($"Type {nd.GetTypeCode} not supported for np.any")
            };
        }

        /// <summary>
        /// Generic implementation of Any for unmanaged types.
        /// Uses SIMD for contiguous arrays, falls back to iteration for strided arrays.
        /// </summary>
        private static unsafe bool AnyImpl<T>(NDArray nd) where T : unmanaged
        {
            var shape = nd.Shape;

            if (shape.IsContiguous)
            {
                // SIMD fast path for contiguous arrays
                if (ILKernelGenerator.Enabled)
                {
                    return ILKernelGenerator.AnySimdHelper<T>((void*)nd.Address, nd.size);
                }

                // Scalar fallback for contiguous arrays
                var addr = (T*)nd.Address;
                var len = nd.size;
                for (int i = 0; i < len; i++)
                {
                    if (!addr[i].Equals(default(T)))
                        return true;
                }
                return false;
            }
            else
            {
                // Iterator fallback for non-contiguous (strided/sliced) arrays
                using var iter = nd.AsIterator<T>();
                while (iter.HasNext())
                {
                    if (!iter.MoveNext().Equals(default(T)))
                        return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Special implementation for Decimal (not supported by SIMD).
        /// </summary>
        private static bool AnyImplDecimal(NDArray nd)
        {
            using var iter = nd.AsIterator<decimal>();
            while (iter.HasNext())
            {
                if (iter.MoveNext() != 0m)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="axis">Axis along which to reduce</param>
        /// <returns>Array of bools with the axis dimension removed</returns>
        public override NDArray<bool> Any(NDArray nd, int axis)
        {
            // TODO: Implement axis reduction for Any
            throw new NotImplementedException($"DefaultEngine.Any with axis={axis} not yet implemented. Use np.any(arr, axis) directly.");
        }
    }
}
