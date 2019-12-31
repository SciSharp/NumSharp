
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="ndArrays">The arrays to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params NDArray[] ndArrays)
        {
            return DefaultEngine.AreBroadcastable(ndArrays);
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="shapes">The shapes to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params Shape[] shapes)
        {
            return DefaultEngine.AreBroadcastable(shapes);
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="shapes">The shapes to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(params int[][] shapes)
        {
            return DefaultEngine.AreBroadcastable(shapes);
        }

        /// <summary>
        ///     Tests if these two two arrays are broadcastable against each other.
        /// </summary>
        /// <param name="lhs">An array to test for broadcasting.</param>
        /// <param name="rhs">An array to test for broadcasting.</param>
        /// <returns>True if these can be broadcasted against each other.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static bool are_broadcastable(NDArray lhs, NDArray rhs)
        {
            return DefaultEngine.AreBroadcastable(lhs, rhs);
        }
    }
}
