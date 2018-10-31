using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public partial class NDArrayExtensions
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <param name="np"></param>
        /// <param name="dimenstions"></param>
        /// <returns></returns>
        public static NDArray<int> Zeros(this NDArray<int> np, params int[] select)
        {
            int length = 1;

            for(int i = 0; i< select.Length; i++)
            {
                length *= select[i];
            }

            np.ARange(length).ReShape(select);

            return np;
        }

        public static NDArray<double> Zeros(this NDArray<double> np, params int[] select)
        {
            int length = 1;

            for (int i = 0; i < select.Length; i++)
            {
                length *= select[i];
            }

            np.ARange(length).ReShape(select);

            return np;
        }
    }
}
