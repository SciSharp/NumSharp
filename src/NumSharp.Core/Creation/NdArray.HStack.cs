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
        public NDArray hstack<T>(params NDArray[] nps)
        {
            List<NDArray> list = new List<NDArray>();
            list.Add(this);
            list.AddRange(nps);
            return np.hstack<T>(list.ToArray());
        }
    }
}
