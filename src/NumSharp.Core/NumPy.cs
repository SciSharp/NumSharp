using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy  
    /// </summary>
    public static partial class NumPy
    {
        // https://docs.scipy.org/doc/numpy-1.15.0/user/basics.types.html
        public static Type int16 => typeof(short);
        public static Type int32 => typeof(int);
        public static Type int64 => typeof(Int64);
        public static Type float32 => typeof(float);
        public static Type float64 => typeof(double);
        public static Type chars => typeof(string);

        public static NumPyRandom random => new NumPyRandom();
    }
}
