using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy
    /// </summary>
    [ModuleName("np")]
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
        //
        // The dtype spellings are DType DESCRIPTORS — the canonical instance of each dtype class (the same object
        // `np.dtype("float64")` returns, so `np.float64 == a.dtype` is a cheap structural compare and `np.float64.itemsize`
        // / `.name` / `.kind` read like NumPy). In NumPy these names are the scalar TYPE classes and `np.dtype(np.float64)`
        // is the descriptor; NumSharp has no scalar-type objects (a C# `double` IS the scalar), so the descriptor is the
        // one dtype currency: every `dtype=` parameter takes a DType, and a C# Type / NPTypeCode / dtype string converts
        // to it implicitly. `Type t = np.float64` still works (implicit DType → Type), as does `typeof(double) == np.float64`.
        public static readonly DType bool_ = typeof(bool);

        /// <summary>
        ///     NumPy 1.x's <c>np.bool8</c>: the <see cref="bool_"/> descriptor. Kept on purpose, so code written against
        ///     NumPy 1.x compiles and runs unchanged, although NumPy 1.24 deprecated the name and NumPy 2.0 removed it
        ///     (<c>np.bool8</c> raises <c>AttributeError: module 'numpy' has no attribute 'bool8'</c>). New code writes
        ///     <see cref="bool_"/>.
        /// </summary>
        /// <remarks>
        ///     Only the member is kept: the dtype STRING follows NumPy 2.x, so <c>np.dtype("bool8")</c> raises NumPy's
        ///     <c>Alias 'bool8' was removed in NumPy 2.0. Use a name without a digit at the end.</c>
        /// </remarks>
        public static readonly DType bool8 = bool_;
        public static readonly DType @bool = bool_;

        // np.@char is the numpy.char MODULE (APIs/np.char.cs); it converts implicitly to the Char dtype it used to hold
        // here, so `astype(np.@char)` and `a.dtype == np.@char` keep their meaning.

        // NumPy: np.byte = int8 (signed, C char convention). NumSharp follows NumPy.
        // For .NET-style uint8 use np.uint8 / np.ubyte.
        public static readonly DType @byte = typeof(sbyte);
        public static readonly DType int8 = typeof(sbyte);
        public static readonly DType @sbyte = typeof(sbyte);

        public static readonly DType uint8 = typeof(byte);
        public static readonly DType ubyte = uint8;

        public static readonly DType @short = typeof(short);
        public static readonly DType int16 = typeof(short);

        public static readonly DType @ushort = typeof(ushort);
        public static readonly DType uint16 = typeof(ushort);

        /// <summary>
        ///     <c>numpy.intc</c> (C <c>int</c>, char <c>'i'</c>, type number 5): on Windows its own class beside
        ///     <see cref="int32"/> (which is the C <c>long</c> there) — equal to it (<c>np.dtype(np.intc) ==
        ///     np.dtype(np.int32)</c>, the same 32-bit lane) but not the same class (<c>np.issubdtype(np.int32,
        ///     np.intc)</c> is False); on Linux and macOS the <see cref="int32"/> descriptor itself.
        /// </summary>
        public static readonly DType intc = DTypeRegistry.CInt.singleton;

        /// <summary><c>numpy.uintc</c> (C <c>unsigned int</c>, char <c>'I'</c>): its own class on Windows, <see cref="uint32"/> elsewhere (see <see cref="intc"/>).</summary>
        public static readonly DType uintc = DTypeRegistry.CUInt.singleton;
        public static readonly DType int32 = typeof(int);
        public static readonly DType uint32 = typeof(uint);

        // 'long' / 'ulong' follow C long convention — platform-dependent (32-bit on Windows).
        // Access as np.@long / np.@ulong because `long` / `ulong` are C# keywords.
        public static readonly DType @long = _np_cLong;
        public static readonly DType @ulong = _np_cULong;

        /// <summary>
        ///     <c>numpy.longlong</c> (C <c>long long</c>, char <c>'q'</c>, type number 9): on Linux and macOS its own class
        ///     beside <see cref="int64"/> (which is the C <c>long</c> there) — equal to it, the same 64-bit lane, but not the
        ///     same class; on Windows the <see cref="int64"/> descriptor itself.
        /// </summary>
        public static readonly DType longlong = DTypeRegistry.CLongLong.singleton;

        /// <summary><c>numpy.ulonglong</c> (C <c>unsigned long long</c>, char <c>'Q'</c>): its own class on Linux and macOS, <see cref="uint64"/> on Windows (see <see cref="longlong"/>).</summary>
        public static readonly DType ulonglong = DTypeRegistry.CULongLong.singleton;

        // NumPy 2.x: int_ and intp are pointer-sized (int64 on 64-bit platforms).
        // On 64-bit OS typeof(long) is the correct choice — NOT typeof(nint), which is
        // System.IntPtr and has NPTypeCode.Empty (breaks np.zeros/np.empty dispatch).
        public static readonly DType int_ = typeof(long);
        public static readonly DType int64 = int_;
        public static readonly DType intp = IntPtr.Size == 8 ? typeof(long) : typeof(int);
        public static readonly DType uintp = IntPtr.Size == 8 ? typeof(ulong) : typeof(uint);

        /// <summary>
        ///     NumPy 1.x's <c>np.int0</c> (an alias of <c>np.intp</c>): the int64 descriptor, <see cref="intp"/>'s width on a
        ///     64-bit host. Kept on purpose, so code written against NumPy 1.x compiles and runs unchanged, although NumPy
        ///     1.24 deprecated the name and NumPy 2.0 removed it (<c>np.int0</c> raises
        ///     <c>AttributeError: module 'numpy' has no attribute 'int0'</c>). New code writes <see cref="intp"/> or
        ///     <see cref="int64"/>.
        /// </summary>
        /// <remarks>
        ///     Only the member is kept: the dtype STRING follows NumPy 2.x, so <c>np.dtype("int0")</c> raises NumPy's
        ///     <c>Alias 'int0' was removed in NumPy 2.0. Use a name without a digit at the end.</c>
        /// </remarks>
        public static readonly DType int0 = int_;

        public static readonly DType uint64 = typeof(ulong);

        /// <summary>
        ///     NumPy 1.x's <c>np.uint0</c> (an alias of <c>np.uintp</c>): the uint64 descriptor, <see cref="uintp"/>'s width
        ///     on a 64-bit host. Kept on purpose, so code written against NumPy 1.x compiles and runs unchanged, although
        ///     NumPy 1.24 deprecated the name and NumPy 2.0 removed it (<c>np.uint0</c> raises
        ///     <c>AttributeError: module 'numpy' has no attribute 'uint0'</c>). New code writes <see cref="uintp"/> or
        ///     <see cref="uint64"/>.
        /// </summary>
        /// <remarks>
        ///     Only the member is kept: the dtype STRING follows NumPy 2.x, so <c>np.dtype("uint0")</c> raises NumPy's
        ///     <c>Alias 'uint0' was removed in NumPy 2.0. Use a name without a digit at the end.</c>
        /// </remarks>
        public static readonly DType uint0 = uint64;
        public static readonly DType @uint = uintp;  // NumPy 2.x: np.uint == np.uintp (pointer-sized)

        public static readonly DType float16 = typeof(Half);
        public static readonly DType half = float16;

        public static readonly DType float32 = typeof(float);
        public static readonly DType single = float32;

        /// <summary>
        ///     NumPy 1.x's <c>np.float_</c>: the <see cref="float64"/> descriptor. Kept on purpose, so code written against
        ///     NumPy 1.x compiles and runs unchanged, although NumPy 2.0 removed it (<c>np.float_</c> raises
        ///     <c>AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.</c>). New code
        ///     writes <see cref="float64"/>.
        /// </summary>
        /// <remarks>
        ///     Only the member is kept: the dtype STRING follows NumPy 2.x, which has no such name, so
        ///     <c>np.dtype("float_")</c> is NumPy's <c>data type 'float_' not understood</c>.
        /// </remarks>
        public static readonly DType float_ = typeof(double);
        public static readonly DType float64 = float_;
        public static readonly DType @double = float_;

        /// <summary>
        ///     <c>numpy.longdouble</c> (C <c>long double</c>, char <c>'g'</c>, type number 13) — its own class
        ///     (<c>np.dtypes.LongDoubleDType</c>, scalar class <c>numpy.longdouble</c>) over the float64 storage lane. NumSharp
        ///     has no extended precision, so it is the 8-byte double of NumPy's win-amd64 build on every platform:
        ///     <c>np.dtype(np.longdouble) == np.dtype('d')</c> and a cast between them is a no-op, while
        ///     <c>np.issubdtype(np.float64, np.longdouble)</c> is False, as in NumPy.
        /// </summary>
        /// <remarks>
        ///     An array created with this dtype keeps it (<c>np.zeros(3, np.longdouble).dtype.@char</c> is <c>'g'</c>), and
        ///     so do its views, copies and casts to it; an operation computes in the float64 lane and returns float64 where
        ///     NumPy keeps <c>longdouble</c> (equal descriptors, so <c>result.dtype == np.longdouble</c> still holds). Linux
        ///     NumPy's <c>longdouble</c> is an 80-bit type in 16 bytes (<c>float128</c>); NumSharp's is not.
        /// </remarks>
        public static readonly DType longdouble = DTypeRegistry.LongDouble.singleton;

        // ---- Complex ----
        // NumSharp's Complex = System.Numerics.Complex = two 64-bit floats (complex128). NumPy's complex64 is a scalar CLASS
        // here (np.complex64, DTypes/np.scalar_types.cs) with no dtype behind it: converting it to a dtype refuses.

        /// <summary>
        ///     NumPy 1.x's <c>np.complex_</c>: the <see cref="complex128"/> descriptor. Kept on purpose, so code written
        ///     against NumPy 1.x compiles and runs unchanged, although NumPy 2.0 removed it (<c>np.complex_</c> raises
        ///     <c>AttributeError: `np.complex_` was removed in the NumPy 2.0 release. Use `np.complex128` instead.</c>). New
        ///     code writes <see cref="complex128"/>.
        /// </summary>
        /// <remarks>
        ///     Only the member is kept: the dtype STRING follows NumPy 2.x, which has no such name, so
        ///     <c>np.dtype("complex_")</c> is NumPy's <c>data type 'complex_' not understood</c>.
        /// </remarks>
        public static readonly DType complex_ = typeof(Complex);
        public static readonly DType complex128 = complex_;
        public static readonly DType cdouble = complex_;      // NumPy alias for complex128
        /// <summary>
        ///     <c>numpy.clongdouble</c> (complex <c>long double</c>, char <c>'G'</c>, type number 16) — its own class over the
        ///     complex128 lane, two 8-byte parts (see <see cref="longdouble"/>): equal to <see cref="complex128"/>, not the
        ///     same class.
        /// </summary>
        public static readonly DType clongdouble = DTypeRegistry.CLongDouble.singleton;

        /// <summary>NumSharp-only: the <see cref="System.Decimal"/> descriptor (no NumPy counterpart).</summary>
        public static readonly DType @decimal = typeof(decimal);

        /// <summary>
        ///     NumPy's <c>np.str_</c> as a dtype spelling: the UNSIZED fixed-width Unicode descriptor <c>dtype('&lt;U')</c>
        ///     (the same object <c>np.dtype("U")</c> and <c>np.dtype(typeof(string))</c> return). As a <c>dtype=</c> argument
        ///     it means "Unicode, length chosen from the data" — <c>np.array(new[] {"ab", "c"}, np.str_)</c> is
        ///     <c>&lt;U2</c>, and an allocation with no data to measure (<c>np.zeros(3, np.str_)</c>) is <c>&lt;U1</c>.
        /// </summary>
        public static readonly DType str_ = typeof(string);

        /// <summary>
        ///     NumPy's <c>np.bytes_</c> as a dtype spelling: the UNSIZED fixed-width bytes descriptor <c>dtype('S')</c>. It has
        ///     no C# <see cref="Type"/> spelling — <c>typeof(byte[])</c> is a <c>uint8</c> array, as for every C# array type — so
        ///     this field, <c>"S"</c>/<c>"S5"</c> or <see cref="dtypes.BytesDType"/> are the ways to request it.
        /// </summary>
        public static readonly DType bytes_ = DTypeRegistry.Bytes.singleton;

        public static DType chars => throw new NotSupportedException("Please use char with extra dimension.");

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
