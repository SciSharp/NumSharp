using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray reshape(NDArray nd, params int[] shape)
        {
            nd.reshape(shape);
            return nd;
        }
    }
}
