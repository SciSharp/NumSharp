using NumSharp.Backends;
using System;
using System.Collections.Generic;
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
        public static Type uint8 => typeof(byte);
        public static Type int16 => typeof(short);
        public static Type uint16 => typeof(ushort);
        public static Type int32 => typeof(int); // Int32
        public static Type uint32 => typeof(uint); // UInt32
        public static Type int64 => typeof(long); // Int64
        public static Type float32 => typeof(float); // Single
        public static Type float64 => typeof(double);
        public static Type chars => typeof(string);
        public static NumPyRandom random { get; } = new NumPyRandom();

        // np.nan
        public static double nan => double.NaN;
    }
}
