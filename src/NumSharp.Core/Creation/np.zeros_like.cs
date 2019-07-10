using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        ///  <summary>
        ///  Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        ///  <param name="nd">The shape and data-type of `nd` define these same attributes of
        ///  the returned array.</param>
        /// <param name="dtype">data-type, optional
        /// Overrides the data type of the result.</param>
        /// <param name="order">{'C', 'F', 'A', or 'K'}, optional
        /// Overrides the memory layout of the result. 'C' means C-order,
        /// 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        /// 'C' otherwise. 'K' means match the layout of `a` as closely
        /// as possible.
        /// </param>
        /// 
        /// See also:
        /// <seealso cref="empty_like"/>: Return an empty array with shape and type of input.
        /// <seealso cref="ones_like"/>: Return an array of ones with shape and type of input.
        /// <seealso cref="full_like"/>: Return a new array with shape of input filled with value.
        /// <seealso cref="zeros"/>: Return a new array setting values to zero.
        ///  <returns>Array of zeros with the same shape and type as `nd`.</returns>
        public static NDArray zeros_like(NDArray nd, Type dtype = null, string order = "K")
        {
            return np.zeros(new Shape(nd.shape), dtype != null ? dtype : nd.dtype);
        }
    }
}
