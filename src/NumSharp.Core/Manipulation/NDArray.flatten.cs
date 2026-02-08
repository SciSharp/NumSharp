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
        public NDArray flatten(bool clone)
        {
            // Broadcast arrays: the non-clone path wraps the small backing buffer in a
            // vector shape of broadcast size, causing out-of-bounds reads. The clone path
            // uses CloneData which handles broadcast correctly, but ravel() is cleaner.
            if (Shape.IsBroadcasted)
                return np.ravel(this);
            return clone ? new NDArray(CloneData(), Shape.Vector(size)) : new NDArray(Storage, Shape.Vector(size));
        }
    }
}
