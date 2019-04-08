using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Returns the index of the maximum value of the array.
        /// </summary>
        public static int argmax(NDArray nd)
            => nd.argmax();

        /// <summary>
        /// Returns the index of the maximum value of the array.
        /// </summary>
        public static int argmax<T>(NDArray nd)
            => nd.argmax<T>();

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
