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
        public NDArray flatten(char order = 'C') => flat.copy(order);
    }
}
