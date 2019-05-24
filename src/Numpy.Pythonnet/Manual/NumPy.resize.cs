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
    public partial class NumPy
    {

        /// <summary>
        /// Return a new array with the specified shape.
        /// 
        /// If the new array is larger than the original array, then the new
        /// array is filled with repeated copies of a.  Note that this behavior
        /// is different from a.resize(new_shape) which fills with zeros instead
        /// of repeated copies of a.
        /// 
        /// Notes
        /// 
        /// Warning: This functionality does not consider axes separately,
        /// i.e. it does not apply interpolation/extrapolation.
        /// It fills the return array with the required number of elements, taken
        /// from a as they are laid out in memory, disregarding strides and axes.
        /// (This is in case the new shape is smaller. For larger, see above.)
        /// This functionality is therefore not suitable to resize images,
        /// or data where each axis represents a separate and distinct entity.
        /// </summary>
        /// <param name="a">
        /// Array to be resized.
        /// </param>
        /// <param name="new_shape">
        /// Shape of resized array.
        /// </param>
        /// <returns>
        /// The new array is formed from the data in the old array, repeated
        /// if necessary to fill out the required number of elements.  The
        /// data are repeated in the order that they are stored in memory.
        /// </returns>
        public NDarray resize(NDarray a, Shape new_shape)
        {
            var pyargs = ToTuple(new object[]
            {
                a,
                new_shape,
            });
            dynamic py = self.InvokeMethod("resize", pyargs);
            return ToCsharp<NDarray>(py);
        }

    }
}
