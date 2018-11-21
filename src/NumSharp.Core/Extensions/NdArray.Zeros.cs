using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public partial class NDArrayExtensions
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static NDArray<int> Zeros(this NDArray<int> np, params int[] shape)
        {
            int length = 1;

            for(int i = 0; i< shape.Length; i++)
            {
                length *= shape[i];
            }

            np.Data = Enumerable.Range(0, length).Select(x => 0).ToArray();
            np.reshape(shape);

            return np;
        }

        public static NDArray<double> Zeros(this NDArray<double> np, params int[] shape)
        {
            int length = 1;

            for (int i = 0; i < shape.Length; i++)
            {
                length *= shape[i];
            }

            np.Data = Enumerable.Range(0, length).Select(x => 0.0).ToArray();
            np.reshape(shape);

            return np;
        }
    }
}
