using System.Diagnostics.CodeAnalysis;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order"></param>
        /// <returns>A copy of the input array, flattened to one dimension.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html</remarks>
        public NDArray flatten(char order = 'C')
        {
            // Broadcast arrays: the flat property calls reshape() which goes through
            // _reshapeBroadcast with modular arithmetic that produces wrong element order.
            // ravel() correctly materializes broadcast via CloneData/MultiIterator.Assign.
            if (Shape.IsBroadcasted)
                return np.ravel(this);
            return flat.copy(order);
        }
    }
}
