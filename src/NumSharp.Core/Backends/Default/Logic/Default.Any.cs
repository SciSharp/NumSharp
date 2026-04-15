using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;
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
            return nd.GetTypeCode switch
            {
                NPTypeCode.Boolean => AnyImpl<bool>(nd),
                NPTypeCode.Byte => AnyImpl<byte>(nd),
                NPTypeCode.SByte => AnyImpl<sbyte>(nd),
                NPTypeCode.Int16 => AnyImpl<short>(nd),
                NPTypeCode.UInt16 => AnyImpl<ushort>(nd),
                NPTypeCode.Int32 => AnyImpl<int>(nd),
                NPTypeCode.UInt32 => AnyImpl<uint>(nd),
                NPTypeCode.Int64 => AnyImpl<long>(nd),
                NPTypeCode.UInt64 => AnyImpl<ulong>(nd),
                NPTypeCode.Char => AnyImpl<char>(nd),
                NPTypeCode.Half => AnyImplHalf(nd),
                NPTypeCode.Single => AnyImpl<float>(nd),
                NPTypeCode.Double => AnyImpl<double>(nd),
                NPTypeCode.Complex => AnyImplComplex(nd),
                NPTypeCode.Decimal => AnyImplDecimal(nd),
                _ => throw new NotSupportedException($"Type {nd.GetTypeCode} not supported for np.any")
            };
        }

        /// <summary>
        /// Generic implementation of Any for unmanaged types.
        /// Uses the new iterator core for both contiguous and strided layouts.
        /// </summary>
        private static bool AnyImpl<T>(NDArray nd) where T : unmanaged
            => NpyIter.ReduceBool<T, NpyAnyKernel<T>>(nd.Storage);

        private static bool AnyImplDecimal(NDArray nd) => NpyIter.ReduceBool<decimal, NpyAnyKernel<decimal>>(nd.Storage);

        /// <summary>
        /// Special implementation for Half (float16).
        /// </summary>
        private static unsafe bool AnyImplHalf(NDArray nd)
        {
            var shape = nd.Shape;
            if (shape.IsContiguous)
            {
                var addr = (Half*)nd.Address;
                long len = nd.size;
                for (long i = 0; i < len; i++)
                {
                    if (addr[i] != Half.Zero)
                        return true;
                }
                return false;
            }
            else
            {
                using var iter = nd.AsIterator<Half>();
                while (iter.HasNext())
                {
                    if (iter.MoveNext() != Half.Zero)
                        return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Special implementation for Complex (complex128).
        /// </summary>
        private static unsafe bool AnyImplComplex(NDArray nd)
        {
            var shape = nd.Shape;
            if (shape.IsContiguous)
            {
                var addr = (System.Numerics.Complex*)nd.Address;
                long len = nd.size;
                for (long i = 0; i < len; i++)
                {
                    if (addr[i] != System.Numerics.Complex.Zero)
                        return true;
                }
                return false;
            }
            else
            {
                using var iter = nd.AsIterator<System.Numerics.Complex>();
                while (iter.HasNext())
                {
                    if (iter.MoveNext() != System.Numerics.Complex.Zero)
                        return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="axis">Axis along which to reduce</param>
        /// <returns>Array of bools with the axis dimension removed</returns>
        public override NDArray<bool> Any(NDArray nd, int axis)
        {
            return Any(nd, axis, keepdims: false);
        }
    }
}
