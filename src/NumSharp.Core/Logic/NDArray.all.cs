using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{

    public partial class NDArray
    {

        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        public bool all()
        {
            var nd = this;
            // TODO: support boolena evaluation of other types like int
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
        public NDArray<bool> all( int axis)
        {
            throw new NotImplementedException($"np.all axis {axis}");
        }
    }

}
