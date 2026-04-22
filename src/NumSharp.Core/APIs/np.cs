using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy
    /// </summary>
    public static partial class np
    {
        public static BackendType BackendEngine { get; set; }

        /// <summary>
        ///     A convenient alias for None, useful for indexing arrays.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html<br></br><br></br>https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number</remarks>
        public static readonly Slice newaxis = new Slice(null, null, 1) {IsNewAxis = true};

        // Platform-detected C-type sizes. See np.dtype.cs for the same detection logic
        // used by string parsing ('l', 'L', 'long', 'ulong' follow C long, which is
        // 32-bit on Windows/MSVC (LLP64) and 64-bit on 64-bit Linux/Mac (LP64)).
        private static readonly Type _np_cLong =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? typeof(int)
                : (IntPtr.Size == 8 ? typeof(long) : typeof(int));
        private static readonly Type _np_cULong =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? typeof(uint)
                : (IntPtr.Size == 8 ? typeof(ulong) : typeof(uint));

        // https://numpy.org/doc/stable/user/basics.types.html
        public static readonly Type bool_ = typeof(bool);
        public static readonly Type bool8 = bool_;
        public static readonly Type @bool = bool_;

        public static readonly Type @char = typeof(char);

        // NumPy: np.byte = int8 (signed, C char convention). NumSharp follows NumPy.
        // For .NET-style uint8 use np.uint8 / np.ubyte.
        public static readonly Type @byte = typeof(sbyte);
        public static readonly Type int8 = typeof(sbyte);
        public static readonly Type @sbyte = typeof(sbyte);

        public static readonly Type uint8 = typeof(byte);
        public static readonly Type ubyte = uint8;

        public static readonly Type @short = typeof(short);
        public static readonly Type int16 = typeof(short);

        public static readonly Type @ushort = typeof(ushort);
        public static readonly Type uint16 = typeof(ushort);

        // 'intc' / 'uintc' are NumPy's aliases for C 'int' / 'unsigned int' (always 32-bit in practice).
        public static readonly Type intc = typeof(int);
        public static readonly Type uintc = typeof(uint);
        public static readonly Type int32 = typeof(int);
        public static readonly Type uint32 = typeof(uint);

        // 'long' / 'ulong' follow C long convention — platform-dependent (32-bit on Windows).
        // Access as np.@long / np.@ulong because `long` / `ulong` are C# keywords.
        public static readonly Type @long = _np_cLong;
        public static readonly Type @ulong = _np_cULong;

        // 'longlong' / 'ulonglong' are C 'long long' / 'unsigned long long' — always 64-bit.
        public static readonly Type longlong = typeof(long);
        public static readonly Type ulonglong = typeof(ulong);

        // NumPy 2.x: int_ and intp are pointer-sized (int64 on 64-bit platforms).
        // On 64-bit OS typeof(long) is the correct choice — NOT typeof(nint), which is
        // System.IntPtr and has NPTypeCode.Empty (breaks np.zeros/np.empty dispatch).
        public static readonly Type int_ = typeof(long);
        public static readonly Type int64 = int_;
        public static readonly Type intp = IntPtr.Size == 8 ? typeof(long) : typeof(int);
        public static readonly Type uintp = IntPtr.Size == 8 ? typeof(ulong) : typeof(uint);
        public static readonly Type int0 = int_;

        public static readonly Type uint64 = typeof(ulong);
        public static readonly Type uint0 = uint64;
        public static readonly Type @uint = uintp;  // NumPy 2.x: np.uint == np.uintp (pointer-sized)

        public static readonly Type float16 = typeof(Half);
        public static readonly Type half = float16;

        public static readonly Type float32 = typeof(float);
        public static readonly Type single = float32;

        public static readonly Type float_ = typeof(double);
        public static readonly Type float64 = float_;
        public static readonly Type @double = float_;

        // ---- Complex ----
        // NumSharp's Complex = System.Numerics.Complex = two 64-bit floats (complex128).
        // There is NO complex64 in NumSharp — any attempt to use it throws.
        public static readonly Type complex_ = typeof(Complex);
        public static readonly Type complex128 = complex_;
        public static readonly Type cdouble = complex_;      // NumPy alias for complex128
        public static readonly Type clongdouble = complex_;  // NumPy: long-double complex collapses to complex128

        /// <summary>
        ///     NumSharp does not support <c>complex64</c> (two 32-bit floats). The only complex
        ///     type available is <see cref="complex128"/> (two 64-bit floats, backed by
        ///     <see cref="System.Numerics.Complex"/>). Accessing this property throws
        ///     <see cref="NotSupportedException"/>; use <see cref="complex128"/> or
        ///     <see cref="complex_"/> instead.
        /// </summary>
        public static Type complex64 => throw new NotSupportedException(
            "NumSharp does not support complex64 (two 32-bit floats). " +
            "Use np.complex128 (System.Numerics.Complex, two 64-bit floats) instead.");

        /// <summary>
        ///     NumPy alias for complex64. Same as <see cref="complex64"/> — throws
        ///     because NumSharp does not support complex64.
        /// </summary>
        public static Type csingle => throw new NotSupportedException(
            "NumSharp does not support csingle (= complex64, two 32-bit floats). " +
            "Use np.complex128 / np.cdouble instead.");

        public static readonly Type @decimal = typeof(decimal);

        public static Type chars => throw new NotSupportedException("Please use char with extra dimension.");

        public static NumPyRandom random { get; } = new NumPyRandom();

        #region Constants

        public static double nan => double.NaN;
        public static double NAN => double.NaN;
        public static double NaN => double.NaN;
        public static double pi => Math.PI;
        public static double e => Math.E;
        public static double euler_gamma => 0.57721566490153286060651209008240243d;
        public static double inf => double.PositiveInfinity;
        public static double infty => double.PositiveInfinity;
        public static double Inf => double.PositiveInfinity;
        public static double NINF => double.NegativeInfinity;
        public static double PINF => double.PositiveInfinity;
        public static double Infinity => double.PositiveInfinity;
        public static double infinity => double.PositiveInfinity;

        #endregion
    }
}
