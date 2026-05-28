using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;
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
            return nd.GetTypeCode switch
            {
                NPTypeCode.Boolean => AllImpl<bool>(nd),
                NPTypeCode.Byte => AllImpl<byte>(nd),
                NPTypeCode.SByte => AllImpl<sbyte>(nd),
                NPTypeCode.Int16 => AllImpl<short>(nd),
                NPTypeCode.UInt16 => AllImpl<ushort>(nd),
                NPTypeCode.Int32 => AllImpl<int>(nd),
                NPTypeCode.UInt32 => AllImpl<uint>(nd),
                NPTypeCode.Int64 => AllImpl<long>(nd),
                NPTypeCode.UInt64 => AllImpl<ulong>(nd),
                NPTypeCode.Char => AllImpl<char>(nd),
                NPTypeCode.Half => AllImplHalf(nd),
                NPTypeCode.Single => AllImpl<float>(nd),
                NPTypeCode.Double => AllImpl<double>(nd),
                NPTypeCode.Complex => AllImplComplex(nd),
                NPTypeCode.Decimal => AllImplDecimal(nd),
                _ => throw new NotSupportedException($"Type {nd.GetTypeCode} not supported for np.all")
            };
        }

        /// <summary>
        /// Generic implementation of All for unmanaged types.
        /// Uses the new iterator core for both contiguous and strided layouts.
        /// </summary>
        private static bool AllImpl<T>(NDArray nd) where T : unmanaged
            => NpyIter.ReduceBool<T, NpyAllKernel<T>>(nd);

        private static bool AllImplDecimal(NDArray nd) => NpyIter.ReduceBool<decimal, NpyAllKernel<decimal>>(nd);

        /// <summary>
        /// Special implementation for Half (float16).
        /// Zero is falsy, NaN is truthy.
        /// </summary>
        private static unsafe bool AllImplHalf(NDArray nd)
        {
            var shape = nd.Shape;
            if (shape.IsContiguous)
            {
                var addr = (Half*)nd.Address;
                long len = nd.size;
                for (long i = 0; i < len; i++)
                {
                    if (addr[i] == Half.Zero)
                        return false;
                }
                return true;
            }
            else
            {
                using var iter = nd.AsIterator<Half>();
                while (iter.HasNext())
                {
                    if (iter.MoveNext() == Half.Zero)
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Special implementation for Complex (complex128).
        /// Zero is falsy (both real and imaginary are 0).
        /// </summary>
        private static unsafe bool AllImplComplex(NDArray nd)
        {
            var shape = nd.Shape;
            if (shape.IsContiguous)
            {
                var addr = (System.Numerics.Complex*)nd.Address;
                long len = nd.size;
                for (long i = 0; i < len; i++)
                {
                    if (addr[i] == System.Numerics.Complex.Zero)
                        return false;
                }
                return true;
            }
            else
            {
                using var iter = nd.AsIterator<System.Numerics.Complex>();
                while (iter.HasNext())
                {
                    if (iter.MoveNext() == System.Numerics.Complex.Zero)
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="axis">Axis along which to reduce</param>
        /// <returns>Array of bools with the axis dimension removed</returns>
        public override NDArray<bool> All(NDArray nd, int axis)
        {
            return All(nd, axis, keepdims: false);
        }
    }
}
