using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy
    /// </summary>
    public static partial class np
    {
        public static BackendType BackendEngine { get; set; }

        // https://docs.scipy.org/doc/numpy-1.15.0/user/basics.types.html
        public static readonly Type bool_ = typeof(bool);
        public static readonly Type int8 = typeof(char);
        public static readonly Type uint8 = typeof(byte);
        public static readonly Type int16 = typeof(short);
        public static readonly Type uint16 = typeof(ushort);
        public static readonly Type int32 = typeof(int);
        public static readonly Type uint32 = typeof(uint);
        public static readonly Type int64 = typeof(long);
        public static readonly Type uint64 = typeof(ulong);
        public static readonly Type float32 = typeof(float);
        public static readonly Type float64 = typeof(double);
        public static readonly Type float_ = float64;
        public static readonly Type complex128 = typeof(Complex);
        public static readonly Type complex64 = complex128;
        public static readonly Type complex_ = complex128;
        public static readonly Type chars = typeof(string);
        public static NumPyRandom random { get; } = new NumPyRandom();

        // np.nan
        public static double nan => double.NaN;
    }
}
