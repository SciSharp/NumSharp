using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test whether all array elements evaluate to True (non-zero).
        /// Supports all 12 dtypes with SIMD optimization for contiguous arrays.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <returns>True if all elements are non-zero</returns>
        public override bool All(NDArray nd)
        {
            if (nd.size == 0)
                return true; // NumPy: all([]) == True (vacuous truth)

            // Dispatch by type
            return nd.GetTypeCode switch
            {
                NPTypeCode.Boolean => AllImpl<bool>(nd),
                NPTypeCode.Byte => AllImpl<byte>(nd),
                NPTypeCode.Int16 => AllImpl<short>(nd),
                NPTypeCode.UInt16 => AllImpl<ushort>(nd),
                NPTypeCode.Int32 => AllImpl<int>(nd),
                NPTypeCode.UInt32 => AllImpl<uint>(nd),
                NPTypeCode.Int64 => AllImpl<long>(nd),
                NPTypeCode.UInt64 => AllImpl<ulong>(nd),
                NPTypeCode.Char => AllImpl<char>(nd),
                NPTypeCode.Single => AllImpl<float>(nd),
                NPTypeCode.Double => AllImpl<double>(nd),
                NPTypeCode.Decimal => AllImplDecimal(nd),
                _ => throw new NotSupportedException($"Type {nd.GetTypeCode} not supported for np.all")
            };
        }

        /// <summary>
        /// Generic implementation of All for unmanaged types.
        /// Uses SIMD for contiguous arrays, falls back to iteration for strided arrays.
        /// </summary>
        private static unsafe bool AllImpl<T>(NDArray nd) where T : unmanaged
        {
            var shape = nd.Shape;

            if (shape.IsContiguous)
            {
                // SIMD fast path for contiguous arrays
                if (ILKernelGenerator.Enabled)
                {
                    return ILKernelGenerator.AllSimdHelper<T>((void*)nd.Address, nd.size);
                }

                // Scalar fallback for contiguous arrays
                var addr = (T*)nd.Address;
                var len = nd.size;
                for (int i = 0; i < len; i++)
                {
                    if (addr[i].Equals(default(T)))
                        return false;
                }
                return true;
            }
            else
            {
                // Iterator fallback for non-contiguous (strided/sliced) arrays
                using var iter = nd.AsIterator<T>();
                while (iter.HasNext())
                {
                    if (iter.MoveNext().Equals(default(T)))
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Special implementation for Decimal (not supported by SIMD).
        /// </summary>
        private static bool AllImplDecimal(NDArray nd)
        {
            using var iter = nd.AsIterator<decimal>();
            while (iter.HasNext())
            {
                if (iter.MoveNext() == 0m)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="axis">Axis along which to reduce</param>
        /// <returns>Array of bools with the axis dimension removed</returns>
        public override NDArray<bool> All(NDArray nd, int axis)
        {
            // TODO: Implement axis reduction for All
            // For now, delegate to the np.all implementation which has this logic
            throw new NotImplementedException($"DefaultEngine.All with axis={axis} not yet implemented. Use np.all(arr, axis) directly.");
        }
    }
}
