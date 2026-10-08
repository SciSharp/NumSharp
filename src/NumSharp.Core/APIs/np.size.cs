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
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.size.html
        ///     <para>
        ///     NumPy 2.4's statements: <c>axis = normalize_axis_tuple(axis, len(shape))</c>, then the product of those
        ///     lengths — so a 0-d input has no axis at all.
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range ("axis … is out of bounds for array of
        ///     dimension …").</exception>
        public static long size(NDArray a, int? axis = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            if (!axis.HasValue)
                return a.size;

            // normalize_axis_tuple's AxisError, on the axis as given: the old wrap-then-index read the dimension array
            // out of its bounds (IndexOutOfRangeException) for an axis past either end, and a 0-d input.
            return a.Shape.dimensions[Backends.DefaultEngine.check_and_adjust_axis(a.ndim, axis.Value)];
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
