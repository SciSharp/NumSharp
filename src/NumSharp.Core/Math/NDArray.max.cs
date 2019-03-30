using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray max(int? axis = null)
        {
            return np.amax(this, axis);
        }
    }
}
