using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise.
        ///     The fractional and integral parts are negative if the given number is negative.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Fractional part of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.modf.html</remarks>
        public static (NDArray Fractional, NDArray Intergral) modf(NDArray x, NPTypeCode? dtype = null)
        {
            var (frac, whole) = x.TensorEngine.ModF(x, dtype);
            if (x.Shape.NDim > 1 && x.size > 1
                && x.Shape.IsFContiguous && !x.Shape.IsContiguous)
            {
                if (!ReferenceEquals(frac, null) && frac.Shape.NDim > 1 && !frac.Shape.IsFContiguous)
                    frac = frac.copy('F');
                if (!ReferenceEquals(whole, null) && whole.Shape.NDim > 1 && !whole.Shape.IsFContiguous)
                    whole = whole.copy('F');
            }
            return (frac, whole);
        }

        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise.
        ///     The fractional and integral parts are negative if the given number is negative.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Fractional part of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.modf.html</remarks>
        public static (NDArray Fractional, NDArray Intergral) modf(NDArray x, Type dtype) 
            => x.TensorEngine.ModF(x, dtype);
    }
}
