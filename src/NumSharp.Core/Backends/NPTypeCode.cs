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
    /// <remarks>The int values of the enum are a copy of <see cref="TypeCode"/> excluding types not available in numpy.</remarks>
    public enum NPTypeCode
    {
        /// <summary>A null reference.</summary>
        Empty = 0,

        /// <summary>A simple type representing Boolean values of true or false.</summary>
        Boolean = 3,

        /// <summary>An integral type representing unsigned 16-bit integers with values between 0 and 65535. The set of possible values for the <see cref="F:System.TypeCode.Char"></see> type corresponds to the Unicode character set.</summary>
        Char = 4,

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
        Float = 13, // 0x0000000D

        /// <summary>A floating point type representing values ranging from approximately 5.0 x 10 -324 to 1.7 x 10 308 with a precision of 15-16 digits.</summary>
        Double = 14, // 0x0000000E

        /// <summary>A simple type representing values ranging from 1.0 x 10 -28 to approximately 7.9 x 10 28 with 28-29 significant digits.</summary>
        Decimal = 15, // 0x0000000F

        /// <summary>A sealed class type representing Unicode character strings.</summary>
        String = 18, // 0x00000012

        Complex = 128, //0x00000080
    }

    public static class NPTypeCodeExtensions
    {
        /// <summary>
        ///     Returns true if typecode is a number (incl. <see cref="bool"/>, <see cref="char"/> and <see cref="Complex"/>).
        /// </summary>
        [DebuggerNonUserCode]
        public static bool IsNumerical(this NPTypeCode typeCode)
        {
            var val = (int)typeCode;
            return val >= 3 && val <= 15 || val == 129;
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
                {
                    return NPTypeCode.Complex;
                }

                return NPTypeCode.Empty;
            }

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
        [MethodImpl((MethodImplOptions)768)]
        public static Type AsType(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if _REGEN1
	            %foreach all_dtypes,all_dtypes_lowercase%
	            case NPTypeCode.#1: return typeof(#2);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return typeof(bool);
	            case NPTypeCode.Byte: return typeof(byte);
                case NPTypeCode.Char: return typeof(char);
                case NPTypeCode.Int32: return typeof(int);
	            case NPTypeCode.Int64: return typeof(long);
	            case NPTypeCode.Single: return typeof(float);
	            case NPTypeCode.Double: return typeof(double);
	            case NPTypeCode.String: return typeof(string);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Checks if given <see cref="Type"/> has a match in <see cref="NPTypeCode"/>.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
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
        [MethodImpl((MethodImplOptions)768)]
        public static int SizeOf(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN
	            %foreach all_dtypes,all_dtypes_lowercase%
	            case NPTypeCode.#1: return InfoOf<#2>.Size;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Complex: return InfoOf<Complex>.Size;
                case NPTypeCode.Boolean: return 1;
                case NPTypeCode.Byte: return 1;
                case NPTypeCode.Int16: return 2;
                case NPTypeCode.UInt16: return 2;
                case NPTypeCode.Int32: return 4;
                case NPTypeCode.UInt32: return 4;
                case NPTypeCode.Int64: return 8;
                case NPTypeCode.UInt64: return 8;
                case NPTypeCode.Char: return 1;
                case NPTypeCode.Double: return 8;
                case NPTypeCode.Single: return 4;
                case NPTypeCode.Decimal: return 32;
                case NPTypeCode.String: return 1; //because it is a char basically.
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        public static bool IsRealNumber(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN //true was done manually.
	            %foreach all_dtypes%
	            case NPTypeCode.#1: return false;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Complex: return true;
                case NPTypeCode.Boolean: return false;
                case NPTypeCode.Byte: return false;
                case NPTypeCode.Int16: return false;
                case NPTypeCode.UInt16: return false;
                case NPTypeCode.Int32: return false;
                case NPTypeCode.UInt32: return false;
                case NPTypeCode.Int64: return false;
                case NPTypeCode.UInt64: return false;
                case NPTypeCode.Char: return false;
                case NPTypeCode.Double: return true;
                case NPTypeCode.Single: return true;
                case NPTypeCode.Decimal: return true;
                case NPTypeCode.String: return false;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a uint, byte, ulong and so on.
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        public static bool IsUnsigned(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN //true was done manually.
	            %foreach all_dtypes%
	            case NPTypeCode.#1: return false;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Complex: return false;
                case NPTypeCode.Boolean: return true;
                case NPTypeCode.Byte: return true;
                case NPTypeCode.Int16: return false;
                case NPTypeCode.UInt16: return true;
                case NPTypeCode.Int32: return false;
                case NPTypeCode.UInt32: return true;
                case NPTypeCode.Int64: return false;
                case NPTypeCode.UInt64: return true;
                case NPTypeCode.Char: return true;
                case NPTypeCode.Double: return false;
                case NPTypeCode.Single: return false;
                case NPTypeCode.Decimal: return false;
                case NPTypeCode.String: return true;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        public static bool IsSigned(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN //true was done manually.
	            %foreach all_dtypes%
	            case NPTypeCode.#1: return false;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Complex: return false;
                case NPTypeCode.Boolean: return false;
                case NPTypeCode.Byte: return false;
                case NPTypeCode.Int16: return true;
                case NPTypeCode.UInt16: return false;
                case NPTypeCode.Int32: return true;
                case NPTypeCode.UInt32: return false;
                case NPTypeCode.Int64: return true;
                case NPTypeCode.UInt64: return false;
                case NPTypeCode.Char: return false;
                case NPTypeCode.Double: return true;
                case NPTypeCode.Single: return true;
                case NPTypeCode.Decimal: return true;
                case NPTypeCode.String: return false;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        internal static int GetGroup(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN //true was done manually.
	            %foreach all_dtypes%
	            case NPTypeCode.#1: return false;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return -1;
                
                case NPTypeCode.String: return 0;
                case NPTypeCode.Byte: return 0;
                case NPTypeCode.Char: return 0;

                case NPTypeCode.Int16: return 1;
                case NPTypeCode.Int32: return 1;
                case NPTypeCode.Int64: return 1;

                case NPTypeCode.UInt16: return 2;
                case NPTypeCode.UInt32: return 2;
                case NPTypeCode.UInt64: return 2;

                case NPTypeCode.Single: return 3;
                case NPTypeCode.Double: return 3;

                case NPTypeCode.Decimal: return 4;

                case NPTypeCode.Complex: return 10;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Is <paramref name="typeCode"/> a float, double, complex or decimal?
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        internal static int GetPriority(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if __REGEN //true was done manually.
	            %foreach all_dtypes%
	            case NPTypeCode.#1: return false;
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return 0;
                case NPTypeCode.String: return 0;
                case NPTypeCode.Byte: return 0;
                case NPTypeCode.Char: return 0;

                case NPTypeCode.Int16: return 1 * 10 * 2;
                case NPTypeCode.Int32: return 1 * 10 * 4;
                case NPTypeCode.Int64: return 1 * 10 * 8;

                case NPTypeCode.UInt16: return 2 * 10 * 2;
                case NPTypeCode.UInt32: return 2 * 10 * 4;
                case NPTypeCode.UInt64: return 2 * 10 * 8;

                case NPTypeCode.Single: return 5 * 10 * 4;
                case NPTypeCode.Double: return 5 * 10 * 8;
                case NPTypeCode.Decimal: return 5 * 10 * 32;

                case NPTypeCode.Complex: return 5000;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Gets NumSharp's <see cref="NPTypeCode"/> equivalent of <paramref name="typeCode"/>
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        internal static NPTypeCode ToTypeCode(this NPY_TYPECHAR typeCode)
        {
            switch (typeCode)
            {
                case NPY_TYPECHAR.NPY_BOOLLTR:
                    return NPTypeCode.Boolean;

                case NPY_TYPECHAR.NPY_BYTELTR:
                    return NPTypeCode.Byte;

                case NPY_TYPECHAR.NPY_UBYTELTR:
                    //case NPY_TYPECHAR.NPY_CHARLTR: //char has been deprecated in favor of string.
                    return NPTypeCode.Char;

                case NPY_TYPECHAR.NPY_SHORTLTR:
                    return NPTypeCode.Int16;

                case NPY_TYPECHAR.NPY_USHORTLTR:
                    return NPTypeCode.UInt16;

                case NPY_TYPECHAR.NPY_INTLTR:
                    return NPTypeCode.Int32;

                case NPY_TYPECHAR.NPY_UINTLTR:
                    return NPTypeCode.UInt32;

                case NPY_TYPECHAR.NPY_LONGLTR:
                case NPY_TYPECHAR.NPY_INTPLTR:
                case NPY_TYPECHAR.NPY_LONGLONGLTR:
                    return NPTypeCode.Int64;

                case NPY_TYPECHAR.NPY_UINTPLTR:
                case NPY_TYPECHAR.NPY_ULONGLTR:
                case NPY_TYPECHAR.NPY_UNSIGNEDLTR:
                case NPY_TYPECHAR.NPY_ULONGLONGLTR:
                    return NPTypeCode.UInt64;

                case NPY_TYPECHAR.NPY_HALFLTR:
                case NPY_TYPECHAR.NPY_FLOATLTR:
                case NPY_TYPECHAR.NPY_CFLOATLTR:
                    return NPTypeCode.Single;

                case NPY_TYPECHAR.NPY_DOUBLELTR:
                case NPY_TYPECHAR.NPY_CDOUBLELTR:
                case NPY_TYPECHAR.NPY_LONGDOUBLELTR:
                    return NPTypeCode.Double;

                case NPY_TYPECHAR.NPY_STRINGLTR:
                case NPY_TYPECHAR.NPY_STRINGLTR2:
                case NPY_TYPECHAR.NPY_UNICODELTR:
                    return NPTypeCode.Char;

                case NPY_TYPECHAR.NPY_VOIDLTR:
                    return NPTypeCode.Empty;

                case NPY_TYPECHAR.NPY_COMPLEXLTR:
                    return NPTypeCode.Complex;

                    return NPTypeCode.Decimal;

                default:
                    throw new NotSupportedException($"NPY_TYPECHAR of type {typeCode} is not supported.");
            }
        }

        /// <summary>
        ///     Gets NumSharp's <see cref="NPTypeCode"/> equivalent of <paramref name="typeCode"/>
        /// </summary>
        [DebuggerNonUserCode]
        [MethodImpl((MethodImplOptions)768)]
        internal static NPY_TYPECHAR ToTYPECHAR(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Empty:
                    return NPY_TYPECHAR.NPY_VOIDLTR;
                case NPTypeCode.Boolean:
                    return NPY_TYPECHAR.NPY_BOOLLTR;
                case NPTypeCode.Char:
                    return NPY_TYPECHAR.NPY_CHARLTR;
                case NPTypeCode.Byte:
                    return NPY_TYPECHAR.NPY_BYTELTR;
                case NPTypeCode.Int16:
                    return NPY_TYPECHAR.NPY_SHORTLTR;
                case NPTypeCode.UInt16:
                    return NPY_TYPECHAR.NPY_USHORTLTR;
                case NPTypeCode.Int32:
                    return NPY_TYPECHAR.NPY_INTLTR;
                case NPTypeCode.UInt32:
                    return NPY_TYPECHAR.NPY_UINTLTR;
                case NPTypeCode.Int64:
                    return NPY_TYPECHAR.NPY_LONGLTR;
                case NPTypeCode.UInt64:
                    return NPY_TYPECHAR.NPY_ULONGLTR; //todo! is that longlong or long?
                case NPTypeCode.Single:
                    return NPY_TYPECHAR.NPY_FLOATLTR;
                case NPTypeCode.Double:
                    return NPY_TYPECHAR.NPY_DOUBLELTR;
                case NPTypeCode.Decimal:
                    return NPY_TYPECHAR.NPY_LONGLONGLTR;
                case NPTypeCode.String:
                    return NPY_TYPECHAR.NPY_STRINGLTR;
                case NPTypeCode.Complex:
                    return NPY_TYPECHAR.NPY_COMPLEXLTR;
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
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
        /// </summary>
        internal static string AsNumpyDtypeName(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Empty:
                    return "";
                case NPTypeCode.Boolean:
                    return "bool";
                case NPTypeCode.Char:
                    return "uint8";
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
                case NPTypeCode.Single:
                    return "float32";
                case NPTypeCode.Double:
                    return "float64";
                case NPTypeCode.Decimal:
                    return "float64";
                case NPTypeCode.String:
                    return "string";
                case NPTypeCode.Complex:
                    return "complex64";
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
#if _REGEN1
            #region Compute
		    switch (typeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_accumulatingType%
			    case NPTypeCode.#1: return NPTypeCode.#2;
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else
            #region Compute
		    switch (typeCode)
		    {
			    case NPTypeCode.Boolean: return NPTypeCode.Byte;
			    case NPTypeCode.Byte: return NPTypeCode.Int32;
			    case NPTypeCode.Int32: return NPTypeCode.Int64;
			    case NPTypeCode.Int64: return NPTypeCode.Single;
			    case NPTypeCode.Single: return NPTypeCode.Double;
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif

            return typeCode;
        }
        
        /// <summary>
        ///     Gets the default value of <paramref name="typeCode"/>.
        /// </summary>
        public static ValueType GetDefaultValue(this NPTypeCode typeCode)
        {
#if _REGEN1
            #region Compute
		    switch (typeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return default(#2);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else
            #region Compute
		    switch (typeCode)
		    {
			    case NPTypeCode.Boolean: return default(bool);
			    case NPTypeCode.Byte: return default(byte);
			    case NPTypeCode.Int32: return default(int);
			    case NPTypeCode.Int64: return default(long);
			    case NPTypeCode.Single: return default(float);
			    case NPTypeCode.Double: return default(double);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }
    }
}
