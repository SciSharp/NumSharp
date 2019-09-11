namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="clone">Should the data be cloned, true by default.</param>
        /// <returns>A copy of the input array, flattened to one dimension.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html</remarks>
        public NDArray flatten(bool clone) => clone ? new NDArray(CloneData(), Shape.Vector(size)) : new NDArray(Storage, Shape.Vector(size));
    }
}
