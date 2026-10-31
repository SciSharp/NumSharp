using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

// ReSharper disable once CheckNamespace
namespace NumSharp
{
    /// <summary>
    ///     Represents all available types in numpy.
    /// </summary>
    /// <remarks>
    ///     <para>The int values of the enum are a copy of <see cref="TypeCode"/> excluding types not available in numpy.</para>
    ///     <para>
    ///         Two values are NumSharp's own: <see cref="NPTypeCode.Half"/> takes 16, which is <see cref="TypeCode.DateTime"/>
    ///         (so <see cref="NPTypeCodeExtensions.GetTypeCode(Type)"/> returns <see cref="NPTypeCode.Empty"/> for
    ///         <see cref="System.DateTime"/>), and <see cref="NPTypeCode.Complex"/> takes 128. The values are not NumPy's type
    ///         numbers, though a few coincide: a descriptor reports its type number through <see cref="DType.type_num"/>
    ///         (<see cref="NPY_TYPES"/>).
    ///     </para>
    ///     <para>
    ///         NumPy's other dtypes are listed after <see cref="NPTypeCode.Complex"/>, commented out, in two groups: the ones
    ///         that need a storage lane of their own, each with the value it would take and what has to change before it is
    ///         uncommented, and the NumPy classes that store in one of the lanes above, which stay commented.
    ///     </para>
    /// </remarks>
    public enum NPTypeCode
    {
        /// <summary>A null reference.</summary>
        Empty = 0,

        /// <summary>A simple type representing Boolean values of true or false.</summary>
        Boolean = 3,

        /// <summary>An integral type representing unsigned 16-bit integers with values between 0 and 65535. The set of possible values for the <see cref="F:System.TypeCode.Char"></see> type corresponds to the Unicode character set.</summary>
        Char = 4,

        /// <summary>An integral type representing signed 8-bit integers with values between -128 and 127.</summary>
        SByte = 5,

        /// <summary>An integral type representing unsigned 8-bit integers with values between 0 and 255.</summary>
        Byte = 6,

        /// <summary>An integral type representing signed 16-bit integers with values between -32768 and 32767.</summary>
        Int16 = 7,

        /// <summary>An integral type representing unsigned 16-bit integers with values between 0 and 65535.</summary>
        UInt16 = 8,

        /// <summary>An integral type representing signed 32-bit integers with values between -2147483648 and 2147483647.</summary>
        Int32 = 9,

        /// <summary>An integral type representing unsigned 32-bit integers with values between 0 and 4294967295.</summary>
        UInt32 = 10, // 0x0000000A

        /// <summary>An integral type representing signed 64-bit integers with values between -9223372036854775808 and 9223372036854775807.</summary>
        Int64 = 11, // 0x0000000B

        /// <summary>An integral type representing unsigned 64-bit integers with values between 0 and 18446744073709551615.</summary>
        UInt64 = 12, // 0x0000000C

        /// <summary>A floating point type representing values ranging from approximately 1.5 x 10 -45 to 3.4 x 10 38 with a precision of 7 digits.</summary>
        Single = 13, // 0x0000000D

        /// <summary>
        ///     An alias of <see cref="NPTypeCode.Single"/> (the same value, 13), not a type of its own: code that switches on
        ///     <see cref="NPTypeCode"/> cannot tell the two apart. .NET does not specify which name <c>ToString()</c> returns
        ///     for a shared value; on .NET 10 both print <c>"Single"</c>.
        /// </summary>
        Float = 13, // 0x0000000D

        /// <summary>A floating point type representing values ranging from approximately 5.0 x 10 -324 to 1.7 x 10 308 with a precision of 15-16 digits.</summary>
        Double = 14, // 0x0000000E

        /// <summary>A simple type representing values ranging from 1.0 x 10 -28 to approximately 7.9 x 10 28 with 28-29 significant digits.</summary>
        Decimal = 15, // 0x0000000F

        /// <summary>A 16-bit floating point type (IEEE 754 half-precision).</summary>
        Half = 16, // 0x00000010

        /// <summary>A sealed class type representing Unicode character strings.</summary>
        String = 18, // 0x00000012

        /// <summary>A complex number type with two 64-bit floating point components (real and imaginary).</summary>
        Complex = 128, //0x00000080

        // ---------------------------------------------------------------------------------------------------------------
        // NumPy's other type codes, commented out.
        //
        // NumPy 2.4.2 numbers its dtypes with NPY_TYPES (numpy/_core/include/numpy/ndarraytypes.h), which
        // Creation/np.dtype.cs mirrors byte for byte. The members above cover NumPy's bool, the sized integers, float16,
        // float32, float64, complex128 and the three string classes (String is the one lane of bytes_, str_ and
        // StringDType: NPY_STRING, NPY_UNICODE and NPY_VSTRING); Char and Decimal are NumSharp's own. The rest follow in
        // two groups.
        //
        // Group 1 needs a storage lane of its own. Before uncommenting one of its members:
        //   - Keep the value at 17 or in 19..127. DTypeRegistry._byCode and the pythonnet package's _dtypeStrings cache
        //     are 129-slot arrays indexed by the value, and NpFunc sizes every dispatch table to the largest value + 1.
        //     The free values below 17 (1 and 2) misbehave: GetComputingType, ResolveUnaryReturnType and the divide and
        //     arctan2 dtype checks read every value below Single as bool or an integer, and GetTypeCode(Type) casts
        //     System.TypeCode.DBNull (2) straight to NPTypeCode. The value also ranks the code where two codes are
        //     compared as numbers: np.find_common_type's _find_common_coerce returns the larger one.
        //   - Give it a SizeOf case, or exclude it in StackedMemoryPool: that class's type initializer calls SizeOf on
        //     every member (Enum.GetValues), so a member without a case makes every ArraySlice.Scalar and
        //     UnmanagedStorage.Scalar call throw TypeInitializationException for the rest of the process.
        //   - Extend IsNumerical's hard-coded ranges (3..16 and 128) if the type is a number.
        //   - Register its DType class with this code in DTypeRegistry, and teach every switch over NPTypeCode: the
        //     extension methods below, InfoOf, Converts, the promotion tables of Logic/np.find_common_type.cs and the
        //     kernels. Each of them throws for a code it does not know (most with NotSupportedException), so a missed
        //     switch fails only at run time.
        //
        // Group 2 lists NumPy classes that store in one of the lanes above, written as aliases to name that lane. They
        // stay commented: an alias has its lane's value (LongDouble == Double), so the code cannot carry the class (the
        // DType descriptor does), and .NET does not specify which of two names Enum.ToString returns, while type-code
        // names reach error texts and kernel cache keys.
        //
        // Not listed: NPY_CHAR (24), NumPy's deprecated one-byte string type number, which raises when used (the
        // character 'c' still spells S1; it is not NumSharp's Char); the sentinels NPY_NTYPES_LEGACY, NPY_NOTYPE (whose
        // role Empty plays), NPY_USERDEF and NPY_NTYPES_ABI_COMPATIBLE; and the macros and characters that name one of
        // these numbers (NPY_INT8 .. NPY_COMPLEX256, NPY_INTP, NPY_DEFAULT_INT, 'n' / 'N' and 'p' / 'P').
        // ---------------------------------------------------------------------------------------------------------------

        // ---- Group 1: dtypes that need a storage lane of their own ----

        // /// <summary>
        // ///     NumPy's <c>complex64</c> (<c>NPY_CFLOAT</c> = 14, character <c>'F'</c>): a float32 real part and a float32
        // ///     imaginary part, 8 bytes per element.
        // /// </summary>
        // /// <remarks>
        // ///     Needs an element struct first: .NET has no single-precision complex type. Until then NumSharp has one
        // ///     complex width, so a result NumPy types complex64 comes back as complex128, and <c>np.complex64</c> /
        // ///     <c>np.csingle</c> throw (issue #569). 17 is the one free value below <see cref="NPTypeCode.String"/>; it
        // ///     keeps the numeric codes in one block (3..17, plus <see cref="NPTypeCode.Complex"/>), so
        // ///     <see cref="NPTypeCodeExtensions.IsNumerical"/> only widens its first range.
        // /// </remarks>
        // Complex64 = 17, // 0x00000011

        // /// <summary>
        // ///     NumPy's <c>datetime64</c> (<c>NPY_DATETIME</c> = 21, character <c>'M'</c>): an int64 count of ticks since
        // ///     1970-01-01T00:00, the tick unit and its multiple held by the descriptor (<c>M8[ns]</c>, <c>M8[5s]</c>);
        // ///     <see cref="long.MinValue"/> is NaT.
        // /// </summary>
        // /// <remarks>
        // ///     Stage C of docs/plans/dtype-system.md (NumPy's semantics: docs/plans/datetime64.md). Until then the
        // ///     <c>DateTime64DType</c> class has no storage (its TypeCode is <see cref="NPTypeCode.Empty"/>): its
        // ///     descriptors parse, print, promote and cast-check, and no array holds one. The value is not
        // ///     <see cref="TypeCode.DateTime"/> (16, taken by <see cref="NPTypeCode.Half"/>): <see cref="System.DateTime"/>
        // ///     keeps mapping to <see cref="NPTypeCode.Empty"/>, and <see cref="NumSharp.DateTime64"/> counts .NET's
        // ///     100 ns ticks since 0001-01-01, not NumPy's units.
        // /// </remarks>
        // DateTime64 = 19, // 0x00000013

        // /// <summary>
        // ///     NumPy's <c>timedelta64</c> (<c>NPY_TIMEDELTA</c> = 22, character <c>'m'</c>): an int64 count of ticks of
        // ///     the descriptor's unit (<c>m8[ms]</c>); <see cref="long.MinValue"/> is NaT.
        // /// </summary>
        // /// <remarks>
        // ///     Stage C, with DateTime64: one family for dispatch (NumPy's <c>PyTypeNum_ISDATETIME</c>), two dtypes for
        // ///     arithmetic (a datetime minus a datetime is a timedelta). NumPy's scalar hierarchy puts
        // ///     <c>timedelta64</c> under <c>signedinteger</c>, so it is an <c>np.number</c>, and <c>datetime64</c>
        // ///     directly under <c>generic</c>: <see cref="NPTypeCodeExtensions.IsNumerical"/> has to answer for each.
        // /// </remarks>
        // TimeDelta64 = 20, // 0x00000014

        // /// <summary>
        // ///     NumPy's <c>object_</c> (<c>NPY_OBJECT</c> = 17, character <c>'O'</c>): each element is a reference to an
        // ///     arbitrary object (a <c>PyObject*</c>, 8 bytes on a 64-bit host), and operations call the elements' own
        // ///     operators.
        // /// </summary>
        // /// <remarks>
        // ///     Needs storage NumSharp does not have: its buffers are unmanaged memory, which cannot hold managed
        // ///     references, so an object lane needs GC handles or a managed side array, and every kernel needs a loop that
        // ///     dispatches per element.
        // /// </remarks>
        // Object = 21, // 0x00000015

        // /// <summary>
        // ///     NumPy's <c>void</c> (<c>NPY_VOID</c> = 20, character <c>'V'</c>): N raw bytes per element, either opaque
        // ///     (<c>V8</c>) or a structured record whose named, typed fields the descriptor lays out.
        // /// </summary>
        // /// <remarks>
        // ///     The element size is the descriptor's itemsize, so <see cref="NPTypeCodeExtensions.SizeOf"/> has no answer
        // ///     for the code and StackedMemoryPool has to skip it. NumSharp has no structured dtypes yet (np.load refuses
        // ///     .npy files with fields).
        // /// </remarks>
        // Void = 22, // 0x00000016

        // ---- Group 2: NumPy classes stored in one of the lanes above (they stay commented, see the header) ----

        // /// <summary>
        // ///     NumPy's <c>longdouble</c> (<c>NPY_LONGDOUBLE</c> = 13, character <c>'g'</c>): C <c>long double</c>, 8 bytes
        // ///     where it is <c>double</c> (MSVC, Apple arm64) and 16 bytes elsewhere (x87 80-bit extended on x86-64 Linux
        // ///     and macOS, IEEE binary128 on aarch64 Linux). NumSharp gives every host the 8-byte form and stores it in
        // ///     the Double lane; a lane of its own would need an extended-precision struct, which .NET lacks.
        // /// </summary>
        // LongDouble = Double,

        // /// <summary>
        // ///     NumPy's <c>clongdouble</c> (<c>NPY_CLONGDOUBLE</c> = 16, character <c>'G'</c>): a complex of two long
        // ///     doubles, 16 or 32 bytes by platform; NumSharp stores it as complex128, in the Complex lane.
        // /// </summary>
        // CLongDouble = Complex,

        // /// <summary>
        // ///     NumPy's <c>bytes_</c> (<c>NPY_STRING</c> = 18, character <c>'S'</c>): fixed-width byte strings, the width
        // ///     being the itemsize. <c>bytes_</c> and <c>str_</c> (<c>'U'</c>, <c>NPY_UNICODE</c> = 19) store in the
        // ///     String lane as two DType classes, with StringDType as the third: the descriptor carries the class and the
        // ///     width, which the lane alone cannot.
        // /// </summary>
        // Bytes = String,

        // /// <summary>
        // ///     NumPy's <c>StringDType</c> (<c>NPY_VSTRING</c> = 2056, character <c>'T'</c>, NEP 55): variable-width UTF-8
        // ///     text, one 16-byte cell per element that holds a short text inline and otherwise refers to text stored
        // ///     elsewhere (the array's arena or the heap).
        // /// </summary>
        // StringDType = String,

        // NumPy's C integer classes. On each platform two of the three signed C ints are int32 and int64 themselves and
        // the third is a class of its own over one of those widths: intc on Windows (where C long is 32-bit), longlong on
        // Linux and macOS. DType.type_num reports the host's numbers.

        // /// <summary>NumPy's <c>intc</c> (<c>NPY_INT</c> = 5, character <c>'i'</c>): C <c>int</c>, 32-bit everywhere.</summary>
        // IntC = Int32,

        // /// <summary>NumPy's <c>uintc</c> (<c>NPY_UINT</c> = 6, character <c>'I'</c>): C <c>unsigned int</c>.</summary>
        // UIntC = UInt32,

        // /// <summary>
        // ///     NumPy's <c>long</c> (<c>NPY_LONG</c> = 7, character <c>'l'</c>): C <c>long</c>, 64-bit on Linux and macOS,
        // ///     32-bit on Windows, where the alias would be Int32.
        // /// </summary>
        // Long = Int64,

        // /// <summary>NumPy's <c>ulong</c> (<c>NPY_ULONG</c> = 8, character <c>'L'</c>): C <c>unsigned long</c>, UInt32 on Windows.</summary>
        // ULong = UInt64,

        // /// <summary>NumPy's <c>longlong</c> (<c>NPY_LONGLONG</c> = 9, character <c>'q'</c>): C <c>long long</c>, 64-bit everywhere.</summary>
        // LongLong = Int64,

        // /// <summary>NumPy's <c>ulonglong</c> (<c>NPY_ULONGLONG</c> = 10, character <c>'Q'</c>): C <c>unsigned long long</c>.</summary>
        // ULongLong = UInt64,
    }

    public static class NPTypeCodeExtensions
    {
        /// <summary>
        ///     Returns true if typecode is a number (incl. <see cref="bool"/>, <see cref="char"/>, <see cref="Half"/> and <see cref="Complex"/>).
        /// </summary>
        [DebuggerNonUserCode]
        public static bool IsNumerical(this NPTypeCode typeCode)
        {
            var val = (int)typeCode;
            // 3-16 covers Boolean through Half, 128 is Complex
            return (val >= 3 && val <= 16) || val == 128;
        }

        /// <summary>
        ///     Extracts <see cref="NPTypeCode"/> from given <see cref="Type"/>.
        /// </summary>
        /// <remarks>In case there was no successful cast to <see cref="NPTypeCode"/>, return will be <see cref="NPTypeCode.Empty"/></remarks>
        [DebuggerNonUserCode]
        public static NPTypeCode GetTypeCode(this Type type)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();

            var tc = Type.GetTypeCode(type);
            if (tc == TypeCode.Object)
            {
                if (type == typeof(Complex))
                    return NPTypeCode.Complex;
                if (type == typeof(Half))
                    return NPTypeCode.Half;

                return NPTypeCode.Empty;
            }

            // TypeCode.DateTime (16) collides with NPTypeCode.Half (16). DateTime/DateTimeOffset
            // are not NumPy dtypes; return Empty so callers know to handle them via the
            // dedicated Converts overloads (or NumSharp's DateTime64 wrapper struct).
            if (tc == TypeCode.DateTime)
                return NPTypeCode.Empty;

            try
            {
                return (NPTypeCode)(int)tc;
            }
            catch (InvalidCastException)
            {
                return NPTypeCode.Empty;
            }
        }

        /// <summary>
        ///     Extracts <see cref="NPTypeCode"/> from given <typeparamref name="T"/>.
        /// </summary>
        /// <remarks>In case there was no successful cast to <see cref="NPTypeCode"/>, return will be <see cref="NPTypeCode.Empty"/></remarks>
        [DebuggerNonUserCode]
        public static NPTypeCode GetTypeCode<T>()
        {
            return InfoOf<T>.NPTypeCode;
        }

        /// <summary>
        ///     Convert <see cref="NPTypeCode"/> into its <see cref="Type"/>
        /// </summary>
        /// <param name="typeCode"></param>
        /// <returns></returns>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static Type AsType(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            // %foreach all_dtypes,all_dtypes_lowercase%
	            // case NPTypeCode.#1: return typeof(#2);
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Complex: return typeof(Complex);
                case NPTypeCode.Boolean: return typeof(bool);
                case NPTypeCode.SByte: return typeof(sbyte);
                case NPTypeCode.Byte: return typeof(byte);
                case NPTypeCode.Int16: return typeof(short);
                case NPTypeCode.UInt16: return typeof(ushort);
                case NPTypeCode.Int32: return typeof(int);
                case NPTypeCode.UInt32: return typeof(uint);
                case NPTypeCode.Int64: return typeof(long);
                case NPTypeCode.UInt64: return typeof(ulong);
                case NPTypeCode.Char: return typeof(char);
                case NPTypeCode.Half: return typeof(Half);
                case NPTypeCode.Double: return typeof(double);
                case NPTypeCode.Single: return typeof(float);
                case NPTypeCode.Decimal: return typeof(decimal);
                case NPTypeCode.String: return typeof(string);
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Convert a nullable <see cref="NPTypeCode"/> into its <see cref="Type"/>, or
        ///     <c>null</c> when the code is null — the "infer / preserve dtype" sentinel the
        ///     dtype-carrying APIs use. Lets a call site pass either an <see cref="NPTypeCode"/>
        ///     or an <see cref="NPTypeCode"/>? to a <see cref="Type"/>-typed dtype parameter
        ///     uniformly via <c>x.AsType()</c>.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static Type AsType(this NPTypeCode? typeCode)
            => typeCode.HasValue ? typeCode.Value.AsType() : null;

        /// <summary>
        ///     Checks if given <see cref="Type"/> has a match in <see cref="NPTypeCode"/>.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsValidNPType(this Type type)
        {
            return type.GetTypeCode() != NPTypeCode.Empty;
        }

        /// <summary>
        ///     Gets the size of given <paramref name="typeCode"/>
        /// </summary>
        /// <param name="typeCode"></param>
        /// <returns></returns>
        /// <remarks>The size is computed by <see cref="Marshal.SizeOf{T}()"/></remarks>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static int SizeOf(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            // %foreach all_dtypes,all_dtypes_lowercase%
	            // case NPTypeCode.#1: return InfoOf<#2>.Size;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Complex: return InfoOf<Complex>.Size;
                case NPTypeCode.Boolean: return 1;
                case NPTypeCode.SByte: return 1;
                case NPTypeCode.Byte: return 1;
                case NPTypeCode.Int16: return 2;
                case NPTypeCode.UInt16: return 2;
                case NPTypeCode.Int32: return 4;
                case NPTypeCode.UInt32: return 4;
                case NPTypeCode.Int64: return 8;
                case NPTypeCode.UInt64: return 8;
                case NPTypeCode.Char: return 2;
                case NPTypeCode.Half: return 2;
                case NPTypeCode.Double: return 8;
                case NPTypeCode.Single: return 4;
                case NPTypeCode.Decimal: return 16;
                // A fixed-width string's element size is its DESCRIPTOR's (S5 = 5 bytes, U5 = 20): the lane has none.
                // Any code that sizes a string array by its type code instead of `storage.DTypeSize` /
                // `dtype.itemsize` is a path that has not been routed for strings — fail loudly, never read 1 byte.
                case NPTypeCode.String:
                    throw new NotSupportedException("NPTypeCode.String has no fixed element size: a fixed-width string array's item size is its descriptor's (dtype.itemsize).");
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsRealNumber(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            //true was done manually.
	            // %foreach all_dtypes%
	            // case NPTypeCode.#1: return false;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Complex: return true;
                case NPTypeCode.Boolean: return false;
                case NPTypeCode.SByte: return false;
                case NPTypeCode.Byte: return false;
                case NPTypeCode.Int16: return false;
                case NPTypeCode.UInt16: return false;
                case NPTypeCode.Int32: return false;
                case NPTypeCode.UInt32: return false;
                case NPTypeCode.Int64: return false;
                case NPTypeCode.UInt64: return false;
                case NPTypeCode.Char: return false;
                case NPTypeCode.Half: return true;
                case NPTypeCode.Double: return true;
                case NPTypeCode.Single: return true;
                case NPTypeCode.Decimal: return true;
                case NPTypeCode.String: return false;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a uint, byte, ulong and so on.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsUnsigned(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            //true was done manually.
	            // %foreach all_dtypes%
	            // case NPTypeCode.#1: return false;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Complex: return false;
                case NPTypeCode.Boolean: return true;
                case NPTypeCode.SByte: return false;
                case NPTypeCode.Byte: return true;
                case NPTypeCode.Int16: return false;
                case NPTypeCode.UInt16: return true;
                case NPTypeCode.Int32: return false;
                case NPTypeCode.UInt32: return true;
                case NPTypeCode.Int64: return false;
                case NPTypeCode.UInt64: return true;
                case NPTypeCode.Char: return true;
                case NPTypeCode.Half: return false;
                case NPTypeCode.Double: return false;
                case NPTypeCode.Single: return false;
                case NPTypeCode.Decimal: return false;
                case NPTypeCode.String: return true;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsSigned(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            //true was done manually.
	            // %foreach all_dtypes%
	            // case NPTypeCode.#1: return false;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Complex: return false;
                case NPTypeCode.Boolean: return false;
                case NPTypeCode.SByte: return true;
                case NPTypeCode.Byte: return false;
                case NPTypeCode.Int16: return true;
                case NPTypeCode.UInt16: return false;
                case NPTypeCode.Int32: return true;
                case NPTypeCode.UInt32: return false;
                case NPTypeCode.Int64: return true;
                case NPTypeCode.UInt64: return false;
                case NPTypeCode.Char: return false;
                case NPTypeCode.Half: return true;
                case NPTypeCode.Double: return true;
                case NPTypeCode.Single: return true;
                case NPTypeCode.Decimal: return true;
                case NPTypeCode.String: return false;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        internal static int GetGroup(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            //true was done manually.
	            // %foreach all_dtypes%
	            // case NPTypeCode.#1: return false;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Boolean: return -1;

                case NPTypeCode.String: return 0;
                case NPTypeCode.Byte: return 0;
                case NPTypeCode.Char: return 0;

                case NPTypeCode.SByte: return 1;
                case NPTypeCode.Int16: return 1;
                case NPTypeCode.Int32: return 1;
                case NPTypeCode.Int64: return 1;

                case NPTypeCode.UInt16: return 2;
                case NPTypeCode.UInt32: return 2;
                case NPTypeCode.UInt64: return 2;

                case NPTypeCode.Half: return 3;
                case NPTypeCode.Single: return 3;
                case NPTypeCode.Double: return 3;

                case NPTypeCode.Decimal: return 4;

                case NPTypeCode.Complex: return 10;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        internal static int GetPriority(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
	            //true was done manually.
	            // %foreach all_dtypes%
	            // case NPTypeCode.#1: return false;
	            // %
	            // default:
		            // throw new NotSupportedException();
                case NPTypeCode.Boolean: return 0;
                case NPTypeCode.String: return 0;
                case NPTypeCode.Byte: return 0;
                case NPTypeCode.Char: return 0;

                case NPTypeCode.SByte: return 1 * 10 * 1;
                case NPTypeCode.Int16: return 1 * 10 * 2;
                case NPTypeCode.Int32: return 1 * 10 * 4;
                case NPTypeCode.Int64: return 1 * 10 * 8;

                case NPTypeCode.UInt16: return 2 * 10 * 2;
                case NPTypeCode.UInt32: return 2 * 10 * 4;
                case NPTypeCode.UInt64: return 2 * 10 * 8;

                case NPTypeCode.Half: return 5 * 10 * 2;
                case NPTypeCode.Single: return 5 * 10 * 4;
                case NPTypeCode.Double: return 5 * 10 * 8;
                case NPTypeCode.Decimal: return 5 * 10 * 16;

                case NPTypeCode.Complex: return 5000;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     The storage code of the dtype a NumPy type character names: NumPy's <c>PyArray_DescrFromType(c)</c>, then that
        ///     descriptor's lane. It IS <c>np.dtype(c.ToString()).GetTypeCode()</c>, so the character table cannot drift away
        ///     from the parser (a hand-written switch here once read <c>'D'</c> as float64 and Windows' <c>'l'</c> as int64).
        /// </summary>
        /// <param name="typeCode">
        ///     A NumPy type character: an <see cref="NPY_TYPECHAR"/> member, or a valid character that has none, cast to the
        ///     enum (<c>(NPY_TYPECHAR)'n'</c>, <c>(NPY_TYPECHAR)'p'</c>).
        /// </param>
        /// <returns>
        ///     The <see cref="NPTypeCode"/> NumSharp stores the dtype in, resolved for THIS host like C itself:
        ///     <see cref="NPY_TYPECHAR.NPY_LONGLTR"/> is <see cref="NPTypeCode.Int32"/> on Windows and
        ///     <see cref="NPTypeCode.Int64"/> on Linux and macOS. <c>'g'</c> and <c>'G'</c> come back as
        ///     <see cref="NPTypeCode.Double"/> and <see cref="NPTypeCode.Complex"/>, the lanes <c>np.longdouble</c> and
        ///     <c>np.clongdouble</c> use. The string characters (bytes <c>'S'</c>/<c>'a'</c>/<c>'c'</c>, str <c>'U'</c>,
        ///     StringDType <c>'T'</c>) all come back as <see cref="NPTypeCode.String"/>, the one lane of the three string
        ///     classes: the class and the width are gone, so keep the descriptor (<c>np.dtype(...)</c>) when either matters.
        /// </returns>
        /// <remarks>
        /// <para>
        ///     A KIND letter is not a type character. <see cref="NPY_TYPECHAR.NPY_UNSIGNEDLTR"/> (<c>'u'</c>) is refused, as
        ///     <c>np.dtype('u')</c> is in NumPy, and <see cref="NPY_TYPECHAR.NPY_COMPLEXLTR"/> has the value of
        ///     <see cref="NPY_TYPECHAR.NPY_CHARLTR"/>, so it is read as NumPy's one-byte string <c>S1</c> (the
        ///     <see cref="NPTypeCode.String"/> lane), never as complex. Use <see cref="NPY_TYPECHAR.NPY_CDOUBLELTR"/> for
        ///     complex128.
        /// </para>
        /// <para>
        ///     Being the parser's answer, it reads every value the way <c>np.dtype</c> reads that one character, including
        ///     values that are no type character: a control character below <c>'\x18'</c> is a TYPE NUMBER to NumPy
        ///     (<c>np.dtype('\x0c')</c> is float64), so <c>((NPY_TYPECHAR)0).ToTypeCode()</c> is
        ///     <see cref="NPTypeCode.Boolean"/>, not a refusal. Validate the value first when it comes from outside.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="typeCode"/> is outside 0..0xFFFF, so it is no UTF-16 character.</exception>
        /// <exception cref="NotSupportedException">
        ///     NumPy does not know the character ("data type 'u' not understood"); NumSharp has no such dtype (complex64
        ///     <c>'F'</c>, void <c>'V'</c>, object <c>'O'</c>); or the dtype has no storage lane yet (datetime64 <c>'M'</c>,
        ///     timedelta64 <c>'m'</c>).
        /// </exception>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static NPTypeCode ToTypeCode(this NPY_TYPECHAR typeCode)
        {
            int c = (int)typeCode;
            // A value past char.MaxValue would wrap in the (char) cast to an unrelated character and resolve silently.
            if ((uint)c > char.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, "A NumPy type character is one UTF-16 code unit (0..0xFFFF).");
            // GetTypeCode, not the typecode property: a datetime descriptor parses but has no storage lane, and the property
            // would answer Empty (NumSharp's "infer" sentinel) where the method refuses.
            return np.dtype(((char)c).ToString()).GetTypeCode();
        }

        /// <summary>
        ///     NumPy's type character for the dtype stored in <paramref name="typeCode"/>: that lane's descriptor character,
        ///     exactly the <c>np.dtype(...).char</c> NumPy reports on this host.
        /// </summary>
        /// <param name="typeCode">The storage code.</param>
        /// <returns>
        ///     The character as an <see cref="NPY_TYPECHAR"/>. Platform-dependent for the 32- and 64-bit integers: int32 is
        ///     <see cref="NPY_TYPECHAR.NPY_LONGLTR"/> on Windows and <see cref="NPY_TYPECHAR.NPY_INTLTR"/> elsewhere, int64
        ///     <see cref="NPY_TYPECHAR.NPY_LONGLONGLTR"/> on Windows and <see cref="NPY_TYPECHAR.NPY_LONGLTR"/> elsewhere.
        ///     <see cref="NPTypeCode.Complex"/> is <see cref="NPY_TYPECHAR.NPY_CDOUBLELTR"/> (not the kind letter
        ///     <see cref="NPY_TYPECHAR.NPY_COMPLEXLTR"/>). For every dtype <c>np.dtype</c> can parse outside the string lane,
        ///     <see cref="ToTypeCode(NPY_TYPECHAR)"/> of the result is <paramref name="typeCode"/> again.
        /// </returns>
        /// <exception cref="ArgumentException"><paramref name="typeCode"/> is <see cref="NPTypeCode.Empty"/>: no dtype, so no character.</exception>
        /// <exception cref="NotSupportedException">
        ///     <paramref name="typeCode"/> is one of NumSharp's own dtypes (<see cref="NPTypeCode.Decimal"/>,
        ///     <see cref="NPTypeCode.Char"/>): NumPy has no such dtype, so no NumPy character names it. Do not substitute their
        ///     descriptors' own <see cref="DType.@char"/>: it borrows NumPy characters of OTHER dtypes (decimal reports <c>'q'</c>,
        ///     C <c>long long</c>; char reports <c>'c'</c>, NumPy's <c>S1</c>), so NumPy-facing code would read a decimal as an
        ///     int64. Or <paramref name="typeCode"/> is <see cref="NPTypeCode.String"/>: the lane of the string classes
        ///     (<c>S</c>, <c>U</c> and <c>StringDType</c>) names no one dtype, so ask the array's own descriptor
        ///     (<c>a.dtype.@char</c>) instead.
        /// </exception>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static NPY_TYPECHAR ToTYPECHAR(this NPTypeCode typeCode)
        {
            var descr = DType.From(typeCode);
            // NumSharp numbers its own dtypes in NumPy's user-defined range (NPY_USERDEF and up), as NumPy numbers a registered
            // user dtype; that range is exactly the set no NumPy character can name.
            if (descr.num >= (int)NPY_TYPES.NPY_USERDEF)
                throw new NotSupportedException($"NumSharp's {descr.name} dtype has no NumPy type character: NumPy has no {descr.name} dtype.");
            return (NPY_TYPECHAR)descr.@char;
        }

        public static int CompareTo(this NPTypeCode left, NPTypeCode right)
        {
            if (left == right)
                return 0;
            
            var sizeLeft = left.SizeOf();
            var sizeRight = right.SizeOf();
            var groupLeft = left.GetGroup();
            var groupRight = right.GetGroup();

            if (groupLeft == groupRight)
            {
                return sizeLeft.CompareTo(sizeRight);
            }

            if (groupLeft > groupRight)
            {
                return 1;
            }

            return -1;
        }

        /// <summary>
        ///     Returns the equivalent numpy's name, e.g. <see cref="NPTypeCode.Int32"/> is np.int32, therefore the return is "int32".
        ///     Error texts quote it as <c>dtype('…')</c>, so it is the NumPy name a NumPy user would see for that data.
        /// </summary>
        /// <param name="typeCode">The storage code.</param>
        /// <returns>
        ///     NumPy's dtype name; <c>"str"</c> for <see cref="NPTypeCode.String"/> (NumPy's <c>np.dtype('U').name</c>; NumPy
        ///     has no dtype named <c>"string"</c>); <c>""</c> for <see cref="NPTypeCode.Empty"/>. NumSharp's own dtypes answer
        ///     with the NumPy dtype they stand in for: <see cref="NPTypeCode.Char"/> is <c>"uint16"</c> (it promotes and computes
        ///     as uint16) and <see cref="NPTypeCode.Decimal"/> is <c>"float64"</c>, so a decimal array reads as float64 in a
        ///     NumPy-worded error.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="typeCode"/> is not a defined storage code.</exception>
        public static string AsNumpyDtypeName(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Empty:
                    return "";
                case NPTypeCode.Boolean:
                    return "bool";
                case NPTypeCode.Char:
                    // Char is a 16-bit unsigned integer (System.Char / UTF-16 code unit).
                    // NumSharp's closest NumPy analogue is uint16 — NOT uint8 (which is Byte).
                    return "uint16";
                case NPTypeCode.SByte:
                    return "int8";
                case NPTypeCode.Byte:
                    return "uint8";
                case NPTypeCode.Int16:
                    return "int16";
                case NPTypeCode.UInt16:
                    return "uint16";
                case NPTypeCode.Int32:
                    return "int32";
                case NPTypeCode.UInt32:
                    return "uint32";
                case NPTypeCode.Int64:
                    return "int64";
                case NPTypeCode.UInt64:
                    return "uint64";
                case NPTypeCode.Half:
                    return "float16";
                case NPTypeCode.Single:
                    return "float32";
                case NPTypeCode.Double:
                    return "float64";
                case NPTypeCode.Decimal:
                    return "float64";
                case NPTypeCode.String:
                    // NumPy's str_ dtype is named "str" (np.dtype('U').name); NumPy has no dtype called "string".
                    return "str";
                case NPTypeCode.Complex:
                    return "complex128";
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
        }

        /// <summary>
        ///     Gets the dtype that is used as return type in case when statistics are computed with high decimal precision like <see cref="np.sin"/>
        /// </summary>
        /// <returns>dtype in case when statistics are computed like <see cref="np.mean"/></returns>
        public static NPTypeCode GetComputingType(this NPTypeCode typeCode)
        {
            if (typeCode < NPTypeCode.Single)
                return NPTypeCode.Double;

            return typeCode;
        }
        
        /// <summary>
        ///     Gets the dtype that is used as accumulation in case when statistics are computed like <see cref="np.sum"/>
        /// </summary>
        /// <returns>dtype in case when statistics are computed like <see cref="np.sum"/></returns>
        public static NPTypeCode GetAccumulatingType(this NPTypeCode typeCode)
        {
            // #region Compute
		    // switch (typeCode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_accumulatingType%
			    // case NPTypeCode.#1: return NPTypeCode.#2;
			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }
            // #endregion
            #region Compute
		    // NumPy 2.x uses int64/uint64 for accumulating integer types to prevent overflow
		    switch (typeCode)
		    {
			    case NPTypeCode.Boolean: return NPTypeCode.Int64;
			    case NPTypeCode.SByte: return NPTypeCode.Int64;
			    case NPTypeCode.Byte: return NPTypeCode.UInt64;
			    case NPTypeCode.Int16: return NPTypeCode.Int64;
			    case NPTypeCode.UInt16: return NPTypeCode.UInt64;
			    case NPTypeCode.Int32: return NPTypeCode.Int64;
			    case NPTypeCode.UInt32: return NPTypeCode.UInt64;
			    case NPTypeCode.Int64: return NPTypeCode.Int64;
			    case NPTypeCode.UInt64: return NPTypeCode.UInt64;
			    case NPTypeCode.Char: return NPTypeCode.UInt64;
			    case NPTypeCode.Half: return NPTypeCode.Half;
			    case NPTypeCode.Double: return NPTypeCode.Double;
			    case NPTypeCode.Single: return NPTypeCode.Single;
			    case NPTypeCode.Decimal: return NPTypeCode.Decimal;
			    case NPTypeCode.Complex: return NPTypeCode.Complex;
			    default:
				    throw new NotSupportedException();
		    }
            #endregion

            return typeCode;
        }
        
        /// <summary>
        ///     Gets the default value of <paramref name="typeCode"/>.
        /// </summary>
        public static object GetDefaultValue(this NPTypeCode typeCode)
        {
            // #region Compute
		    // switch (typeCode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1: return default(#2);
			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }
            // #endregion
            #region Compute
		    switch (typeCode)
		    {
			    case NPTypeCode.Boolean: return default(bool);
			    case NPTypeCode.SByte: return default(sbyte);
			    case NPTypeCode.Byte: return default(byte);
			    case NPTypeCode.Int16: return default(short);
			    case NPTypeCode.UInt16: return default(ushort);
			    case NPTypeCode.Int32: return default(int);
			    case NPTypeCode.UInt32: return default(uint);
			    case NPTypeCode.Int64: return default(long);
			    case NPTypeCode.UInt64: return default(ulong);
			    case NPTypeCode.Char: return default(char);
			    case NPTypeCode.Half: return default(Half);
			    case NPTypeCode.Double: return default(double);
			    case NPTypeCode.Single: return default(float);
			    case NPTypeCode.Decimal: return default(decimal);
			    case NPTypeCode.Complex: return default(Complex);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
        }

        /// <summary>
        /// Gets the multiplicative identity (1) for a type.
        /// Used for reduction identity in Product operations.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static object GetOneValue(this NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Boolean => true,
                NPTypeCode.SByte => (sbyte)1,
                NPTypeCode.Byte => (byte)1,
                NPTypeCode.Int16 => (short)1,
                NPTypeCode.UInt16 => (ushort)1,
                NPTypeCode.Int32 => 1,
                NPTypeCode.UInt32 => 1u,
                NPTypeCode.Int64 => 1L,
                NPTypeCode.UInt64 => 1UL,
                NPTypeCode.Char => (char)1,
                NPTypeCode.Half => (Half)1,
                NPTypeCode.Single => 1f,
                NPTypeCode.Double => 1d,
                NPTypeCode.Decimal => 1m,
                NPTypeCode.Complex => Complex.One,
                _ => throw new NotSupportedException($"Type {typeCode} not supported")
            };
        }

        /// <summary>
        /// Is <paramref name="typeCode"/> a floating point type (float, double, or decimal)?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsFloatingPoint(this NPTypeCode typeCode)
        {
            return typeCode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal;
        }

        /// <summary>
        /// Is <paramref name="typeCode"/> an integer type (signed or unsigned)?
        /// </summary>
        /// <remarks>Excludes Boolean and Char.</remarks>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsInteger(this NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.SByte or NPTypeCode.Byte => true,
                NPTypeCode.Int16 or NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                _ => false
            };
        }

        /// <summary>
        /// Is <paramref name="typeCode"/> capable of SIMD operations?
        /// </summary>
        /// <remarks>
        /// Returns true if the type can use SIMD when hardware support is available.
        /// Does not check if SIMD hardware is actually available.
        /// Boolean, Char, and Decimal cannot use SIMD.
        /// </remarks>
        [DebuggerNonUserCode]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsSimdCapable(this NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.SByte or NPTypeCode.Byte => true,
                NPTypeCode.Int16 or NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                NPTypeCode.Single or NPTypeCode.Double => true,
                // Half has limited SIMD support through conversion, Complex is not SIMD capable
                _ => false  // Boolean, Char, Half, Decimal, Complex, String
            };
        }
    }
}
