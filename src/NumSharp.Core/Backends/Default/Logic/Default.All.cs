using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <returns></returns>
        public bool All(NDArray nd)
        {
            var data = nd.Data<bool>();
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
        public NDArray<bool> All(NDArray nd, int axis)
        {
            throw new NotImplementedException($"np.all axis {axis}");
        }
    }
}
