using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="ndArrays">The arrays to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params NDArray[] ndArrays)
        {
            if (ndArrays.Length <= 1)
                return true;
            return Shape.AreBroadcastable(ndArrays.Select(a => a.Shape).ToArray());
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="shapes">The shapes to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params Shape[] shapes)
        {
            return Shape.AreBroadcastable(shapes);
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="shapes">The shapes to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params int[][] shapes)
        {
            return Shape.AreBroadcastable(shapes);
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="lhs">An array to test for broadcasting.</param>
        /// <param name="rhs">An array to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(NDArray lhs, NDArray rhs)
        {
            return Shape.AreBroadcastable(lhs.Shape, rhs.Shape);
        }

        /// <summary>
        ///     Tests if these two shapes are broadcastable against each other.
        /// </summary>
        /// <param name="shape1">First shape to test.</param>
        /// <param name="shape2">Second shape to test.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(long[] shape1, long[] shape2)
        {
            return DefaultEngine.AreBroadcastable(new[] { new Shape(shape1), new Shape(shape2) });
        }
    }
}
