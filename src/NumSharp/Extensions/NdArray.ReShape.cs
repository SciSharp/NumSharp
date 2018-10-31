using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public static NDArray<T> ReShape<T>(this NDArray<T> np, params int[] shape)
        {
            np.Shape = shape;

            return np;
        }
    }
}
