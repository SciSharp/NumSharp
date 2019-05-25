using System;
using System.Collections.Generic;
using System.Text;
using Numpy.Models;

namespace Numpy
{

    public static partial class np
    {

        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <param name="shape">
        /// Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array, e.g., numpy.int8.  Default is
        /// numpy.float64.
        /// </param>
        /// <param name="order">
        /// Whether to store multi-dimensional data in row-major
        /// (C-style) or column-major (Fortran-style) order in
        /// memory.
        /// </param>
        /// <returns>
        /// Array of zeros with the given shape, dtype, and order.
        /// </returns>
        public static NDarray zeros(params int[] shape)
            => NumPy.Instance.zeros(new Shape(shape));
    }
}
