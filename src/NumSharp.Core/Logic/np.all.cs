using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <returns></returns>
        public static bool all(NDArray nd)
        {
            var data = nd.Storage.GetData<bool>();
            for (int i = 0; i < data.Length; i++)
            {
                if (!data[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns>Returns an array of bools</returns>
        public static NDArray<bool> all(NDArray nd, int axis)
        {
            throw new NotImplementedException($"np.all axis {axis}");
        }
    }
}
