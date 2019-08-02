using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// From https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
        /// This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArray hstack<T>(params NDArray[] nps)
        {
            if (nps[0].shape.Length == 1)
                return np.concatenate(nps, 0);
            else
                return np.concatenate(nps, 1);
        }
    }
}
