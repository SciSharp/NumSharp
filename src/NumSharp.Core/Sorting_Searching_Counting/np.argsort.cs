namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Returns the indices that would sort an array.
        ///
        /// Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static NDArray argsort<T>(NDArray nd, int axis = -1)
            => nd.argsort<T>(axis);
    }
}
