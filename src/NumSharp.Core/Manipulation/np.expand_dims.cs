using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        private static readonly int[] __one = new int[] {1};

        /// <summary>
        ///     Expand the shape of an array.
        ///     Insert a new axis that will appear at the axis position in the expanded array shape.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Position in the expanded axes where the new axis is placed. In a similar manner like <see cref="List{T}.Insert"/></param>
        /// <returns>View of a with the number of dimensions increased by one.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html</remarks>
        public static NDArray expand_dims(NDArray a, int axis)
        {
            //test if the ndarray is empty.
            if (a.size == 0)
                return a;

            //then we append a 1 dim between the slice/split.
            return new NDArray(a.Storage.Alias(expand_dims(a.Shape, axis)));
        }        
        
        public static Shape expand_dims(Shape a, int axis)
        {
            //test if the ndarray is empty.
            if (a.size == 0)
                return a;

            //handle negative axis
            if (axis < 0)
                axis = a.NDim + axis;

            //we create an nd-array of the shape and then slice/split it on axis index.
            var shape = a.dimensions.ToList();
            shape.Insert(axis, 1);

            return shape.ToArray();
        }
    }
}
