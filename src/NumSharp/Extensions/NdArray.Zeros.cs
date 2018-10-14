using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <param name="np"></param>
        /// <param name="rows"></param>
        /// <param name="dimenstions"></param>
        /// <returns></returns>
        public static NdArray<List<TData>> Zeros<TData>(this NdArray<TData> np, int rows, int dimenstions)
        {
            np.Data = new List<TData>();

            for (int i = 0; i < rows * dimenstions; i++)
            {
                np.Data.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom("0"));
            }

            np.NDim = dimenstions * rows;

            return np.ReShape(rows, dimenstions);
        }

        public static NdArray<TData> Zeros<TData>(this NdArray<TData> np, int rows)
        {
            np.Data = new List<TData>();

            for (int i = 0; i < rows; i++)
            {
                np.Data.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom("0"));
            }

            np.NDim = 1;

            return np;
        }
    }
}
