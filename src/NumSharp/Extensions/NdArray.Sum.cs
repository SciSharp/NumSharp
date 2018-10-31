using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static int Sum<TData>(this NDArray_Legacy<TData> np, NDArray_Legacy<TData> np2)
        {
            int result = 0;
            for(int i = 0; i < np.Length; i++)
            {
                if(np[i].Equals(np2[i]))
                {
                    result++;
                }
            }

            return result;
        }
    }
}
