using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray amin(NDArray nd, int? axis = null)
        {
            return nd.amin(axis);
        }
    }
}
