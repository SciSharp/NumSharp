using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPyExtensions
    {
        public static NDArray reshape(this NDArray nd, params int[] shape)
        {
            nd.Shape = new Shape(shape);
            return nd;
        }
    }
}
