using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public Type int16 => typeof(short);
        public Type int32 => typeof(int);
        public Type double8 => typeof(double);
        public Type decimal16 => typeof(decimal);

        public NDArray reshape(NDArray nd, params int[] shape)
        {
            nd.Shape = new Shape(shape);

            return nd;
        }
    }
}
