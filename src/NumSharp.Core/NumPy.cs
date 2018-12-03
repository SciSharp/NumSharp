using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy  
    /// </summary>
    public partial class NumPy
    {
        // https://docs.scipy.org/doc/numpy-1.15.0/user/basics.types.html
        public Type int16 => typeof(short);
        public Type int32 => typeof(int);
        public Type int64 => typeof(Int64);
        public Type float32 => typeof(float);
        public Type float64 => typeof(double);

        public NumPyRandom random => new NumPyRandom();
    }
}
