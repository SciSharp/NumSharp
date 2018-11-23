using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray max(int? axis = null)
        {
            return new NumPy().amax(this, axis);
        }
    }
}
