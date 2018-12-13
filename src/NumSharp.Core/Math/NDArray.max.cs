using System;
using System.Collections.Generic;
using System.Text;
using np = NumSharp.Core.NumPy;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray max(int? axis = null)
        {
            return np.amax(this, axis);
        }
    }
}
