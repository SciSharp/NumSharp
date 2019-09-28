using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return the number of elements along a given axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which the elements are counted. By default, give the total number of elements.</param>
        /// <returns>Number of elements along the specified axis.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html</remarks>
        public static int size(NDArray a, int? axis = null)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            if (!axis.HasValue)
                return a.size;

            if (axis.Value <= -1) 
                axis = a.ndim + axis.Value;

            return a.Shape.dimensions[axis.Value];
        }

        //based on:
        //a=np.random.randint(0,10,(4,2,3))
        //print(np.size(a))
        //print(np.size(a, 0))
        //print(np.size(a, 1))
        //print(np.size(a, 2))
        //print(np.size(a, -1))
    }
}
