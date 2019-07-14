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

        // https://docs.scipy.org/doc/numpy-1.16.0/user/basics.types.html
        public static readonly Type bool_ = typeof(bool);
        public static readonly Type bool8 = bool_;

        public static readonly Type @char = typeof(char);

        public static readonly Type @byte = typeof(byte);
        public static readonly Type uint8 = typeof(byte);
        public static readonly Type ubyte = uint8;


        public static readonly Type int16 = typeof(short);

        public static readonly Type uint16 = typeof(ushort);

        public static readonly Type int32 = typeof(int);

        public static readonly Type uint32 = typeof(uint);

        public static readonly Type int_ = typeof(long);
        public static readonly Type int64 = int_;
        public static readonly Type intp = int_; //TODO! IntPtr?
        public static readonly Type int0 = int_;

        public static readonly Type uint64 = typeof(ulong);
        public static readonly Type uint0 = uint64;
        public static readonly Type @uint = uint64;

        public static readonly Type float32 = typeof(float);

        public static readonly Type float_ = typeof(double);
        public static readonly Type float64 = float_;
        public static readonly Type @double = float_;

        public static readonly Type complex_ = typeof(Complex);
        public static readonly Type complex128 = complex_;
        public static readonly Type complex64 = complex_;
        public static readonly Type @decimal = typeof(decimal);

        public static Type chars => throw new NotSupportedException("Please use char with extra dimension.");

        public static NumPyRandom random { get; } = new NumPyRandom();

        // np.nan
        public static double nan => double.NaN;
    }
}
