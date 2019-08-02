using NumSharp.Generic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Stack arrays in sequence horizontally
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public NDArray hstack(params NDArray[] nps)
        {
            NDArray[] list = new NDArray[1 + nps.Length];
            list[0] = this;
            nps.CopyTo(list, 1);
            return np.hstack(list);
        }
    }
}
