using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> HStack(this NDArray<double> np, NDArray<double> np2 )
        {
            var puffer = np.Data.ToList();

            puffer.AddRange(np2.Data);

            var returnValue = new NDArray<double>().Zeros(np.Length + np2.Length);
            returnValue.Data = puffer;

            return returnValue;
        }
    }
}
