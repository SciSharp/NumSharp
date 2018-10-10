using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<TData> ARange<TData>(this NdArray<TData> np, double stop, double start = 0, double step = 1)
        {
            np.Data = new List<TData>();

            int size = 0;

            for (double i = start; i < stop; i = i + step)
            {
                np.Data.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom(i.ToString()));

                size++;
            }

            np.NDim = 1;

            return np;
        }
    }
}
