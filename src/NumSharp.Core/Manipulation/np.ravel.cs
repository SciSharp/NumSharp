using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html</remarks>
        /// <param name="a">Input array. The elements in a are read in the order specified by order, and packed as a 1-D array.</param>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a sliced or broadcasted, the a copy will be made.</remarks>
        public static NDArray ravel(NDArray a)
        {
            // ReSharper disable once ConvertIfStatementToReturnStatement
            if (!a.Shape.IsContiguous)
                return new NDArray(new UnmanagedStorage(a.Storage.CloneData(), Shape.Vector(a.size)));

            return a.reshape(Shape.Vector(a.size));
        }
    }
}
