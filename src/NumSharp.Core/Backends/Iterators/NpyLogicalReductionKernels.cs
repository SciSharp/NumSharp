using System;
using System.Collections.Generic;
using System.Numerics;

namespace NumSharp.Backends.Iteration
{
    // =========================================================================
    // Boolean Reduction Kernels (all/any)
    // =========================================================================

    internal interface INpyBooleanReductionKernel<T>
        where T : unmanaged
    {
        static abstract bool Identity { get; }
        static abstract bool Accumulate(bool accumulator, T value);
        static abstract bool ShouldExit(bool accumulator);
    }

    internal readonly struct NpyAllKernel<T> : INpyBooleanReductionKernel<T>
        where T : unmanaged
    {
        public static bool Identity => true;

        public static bool Accumulate(bool accumulator, T value)
            => accumulator && !EqualityComparer<T>.Default.Equals(value, default);

        public static bool ShouldExit(bool accumulator) => !accumulator;
    }

    internal readonly struct NpyAnyKernel<T> : INpyBooleanReductionKernel<T>
        where T : unmanaged
    {
        public static bool Identity => false;

        public static bool Accumulate(bool accumulator, T value)
            => accumulator || !EqualityComparer<T>.Default.Equals(value, default);

        public static bool ShouldExit(bool accumulator) => accumulator;
    }

    // =========================================================================
    // Numeric Axis Reduction Kernels (sum/prod/min/max along axis)
    // =========================================================================

    /// <summary>
    /// Generic numeric axis reduction kernel interface.
    /// Used by NpyAxisIter for sum, prod, min, max along an axis.
    /// </summary>
    internal unsafe interface INpyAxisNumericReductionKernel<T>
        where T : unmanaged
    {
        /// <summary>
        /// Execute the reduction along the axis.
        /// </summary>
        /// <param name="src">Source pointer at base position</param>
        /// <param name="srcStride">Stride along the reduction axis</param>
        /// <param name="length">Length of the reduction axis</param>
        /// <returns>Reduced value</returns>
        static abstract T Execute(T* src, long srcStride, long length);
    }

    /// <summary>Sum reduction kernel for axis operations.</summary>
    internal readonly struct NpySumAxisKernel<T> : INpyAxisNumericReductionKernel<T>
        where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
    {
        public static unsafe T Execute(T* src, long srcStride, long length)
        {
            T sum = T.AdditiveIdentity;
            for (long i = 0; i < length; i++)
                sum += src[i * srcStride];
            return sum;
        }
    }

    /// <summary>Product reduction kernel for axis operations.</summary>
    internal readonly struct NpyProdAxisKernel<T> : INpyAxisNumericReductionKernel<T>
        where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
    {
        public static unsafe T Execute(T* src, long srcStride, long length)
        {
            T product = T.MultiplicativeIdentity;
            for (long i = 0; i < length; i++)
                product *= src[i * srcStride];
            return product;
        }
    }

    /// <summary>Max reduction kernel for axis operations.</summary>
    internal readonly struct NpyMaxAxisKernel<T> : INpyAxisNumericReductionKernel<T>
        where T : unmanaged, IComparisonOperators<T, T, bool>, IMinMaxValue<T>
    {
        public static unsafe T Execute(T* src, long srcStride, long length)
        {
            if (length == 0)
                return T.MinValue;

            T max = src[0];
            for (long i = 1; i < length; i++)
            {
                T value = src[i * srcStride];
                if (value > max)
                    max = value;
            }
            return max;
        }
    }

    /// <summary>Min reduction kernel for axis operations.</summary>
    internal readonly struct NpyMinAxisKernel<T> : INpyAxisNumericReductionKernel<T>
        where T : unmanaged, IComparisonOperators<T, T, bool>, IMinMaxValue<T>
    {
        public static unsafe T Execute(T* src, long srcStride, long length)
        {
            if (length == 0)
                return T.MaxValue;

            T min = src[0];
            for (long i = 1; i < length; i++)
            {
                T value = src[i * srcStride];
                if (value < min)
                    min = value;
            }
            return min;
        }
    }
}
