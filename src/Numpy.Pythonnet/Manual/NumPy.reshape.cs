using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Numpy;
using Numpy.Models;
using Python.Runtime;

namespace Numpy
{
    /// <summary>
    /// Manual type conversions
    /// </summary>
    public static partial class np
    {
        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// 
        /// Notes
        /// 
        /// It is not always possible to change the shape of an array without
        /// copying the data. If you want an error to be raised when the data is copied,
        /// you should assign the new shape to the shape attribute of the array:
        /// 
        /// The order keyword gives the index ordering both for fetching the values
        /// from a, and then placing the values into the output array.
        /// For example, let’s say you have an array:
        /// 
        /// You can think of reshaping as first raveling the array (using the given
        /// index order), then inserting the elements from the raveled array into the
        /// new array using the same kind of index ordering as was used for the
        /// raveling.
        /// </summary>
        /// <param name="a">The array to reshape</param>
        /// <param name="newshape">
        /// The new shape should be compatible with the original shape. If
        /// an integer, then the result will be a 1-D array of that length.
        /// One shape dimension can be -1. In this case, the value is
        /// inferred from the length of the array and remaining dimensions.
        /// </param>
        /// <returns>
        /// This will be a new view object if possible; otherwise, it will
        /// be a copy.  Note there is no guarantee of the memory layout (C- or
        /// Fortran- contiguous) of the returned array.
        /// </returns>
        public static NDarray reshape(NDarray a, params int[] newshape)
        {
            return NumPy.Instance.reshape(a, new Shape(newshape));
        }
    }
}
