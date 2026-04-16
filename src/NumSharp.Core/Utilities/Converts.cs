using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides various methods related to <see cref="System.Convert"/>.
    /// </summary>
    public static partial class Converts
    {
        /// <summary>
        /// Creates a converter function that handles all types including Half, Complex, and SByte.
        /// Used as fallback when explicit type pair not found in FindConverter.
        /// Uses NumPy-compatible wrapping behavior for integer overflow (no exceptions).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Func<TIn, TOut> CreateFallbackConverter<TIn, TOut>()
        {
            var toutCode = InfoOf<TOut>.NPTypeCode;
            var tinCode = InfoOf<TIn>.NPTypeCode;

            // Special handling for Half output (doesn't implement IConvertible)
            if (toutCode == NPTypeCode.Half)
            {
                return @in => {
                    double d;
                    if (@in is Half h) d = (double)h;
                    else if (@in is Complex c) d = c.Real;
                    else if (@in is IConvertible ic) d = ic.ToDouble(null);
                    else d = Convert.ToDouble(@in);
                    return (TOut)(object)(Half)d;
                };
            }

            // Special handling for Complex output (doesn't implement IConvertible)
            if (toutCode == NPTypeCode.Complex)
            {
                return @in => {
                    double d;
                    if (@in is Half h) d = (double)h;
                    else if (@in is IConvertible ic) d = ic.ToDouble(null);
                    else d = Convert.ToDouble(@in);
                    return (TOut)(object)new Complex(d, 0);
                };
            }

            // For integer output types, use Converts.ToXxx with unchecked wrapping (NumPy parity)
            // This handles SByte, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char
            return toutCode switch
            {
                NPTypeCode.SByte => CreateIntegerConverter<TIn, TOut, sbyte>(tinCode, Converts.ToSByte, Converts.ToSByte, Converts.ToSByte),
                NPTypeCode.Byte => CreateIntegerConverter<TIn, TOut, byte>(tinCode, Converts.ToByte, Converts.ToByte, Converts.ToByte),
                NPTypeCode.Int16 => CreateIntegerConverter<TIn, TOut, short>(tinCode, Converts.ToInt16, Converts.ToInt16, Converts.ToInt16),
                NPTypeCode.UInt16 => CreateIntegerConverter<TIn, TOut, ushort>(tinCode, Converts.ToUInt16, Converts.ToUInt16, Converts.ToUInt16),
                NPTypeCode.Int32 => CreateIntegerConverter<TIn, TOut, int>(tinCode, Converts.ToInt32, Converts.ToInt32, Converts.ToInt32),
                NPTypeCode.UInt32 => CreateIntegerConverter<TIn, TOut, uint>(tinCode, Converts.ToUInt32, Converts.ToUInt32, Converts.ToUInt32),
                NPTypeCode.Int64 => CreateIntegerConverter<TIn, TOut, long>(tinCode, Converts.ToInt64, Converts.ToInt64, Converts.ToInt64),
                NPTypeCode.UInt64 => CreateIntegerConverter<TIn, TOut, ulong>(tinCode, Converts.ToUInt64, Converts.ToUInt64, Converts.ToUInt64),
                NPTypeCode.Char => CreateIntegerConverter<TIn, TOut, char>(tinCode, Converts.ToChar, Converts.ToChar, Converts.ToChar),
                _ => CreateDefaultConverter<TIn, TOut>()
            };
        }

        /// <summary>
        /// Creates a converter for integer types using Converts.ToXxx methods with unchecked wrapping.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Func<TIn, TOut> CreateIntegerConverter<TIn, TOut, TIntermediate>(
            NPTypeCode tinCode,
            Func<long, TIntermediate> fromLong,
            Func<double, TIntermediate> fromDouble,
            Func<Half, TIntermediate> fromHalf)
        {
            return @in =>
            {
                TIntermediate result;
                if (@in is Half h)
                    result = fromHalf(h);
                else if (@in is Complex c)
                    result = fromDouble(c.Real);
                else if (@in is IConvertible ic)
                    // Use ToInt64 for integer sources, ToDouble for float sources
                    result = IsIntegerType(tinCode) ? fromLong(ic.ToInt64(null)) : fromDouble(ic.ToDouble(null));
                else
                    result = fromDouble(Convert.ToDouble(@in));
                return (TOut)(object)result!;
            };
        }

        /// <summary>
        /// Returns true if the type code represents an integer type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsIntegerType(NPTypeCode code) => code switch
        {
            NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or
            NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or
            NPTypeCode.Char => true,
            _ => false
        };

        /// <summary>
        /// Creates a default converter for non-integer types (Single, Double, Decimal, Boolean).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Func<TIn, TOut> CreateDefaultConverter<TIn, TOut>()
        {
            var tout = typeof(TOut);
            return @in =>
            {
                if (@in is Half h) return (TOut)Convert.ChangeType((double)h, tout);
                if (@in is Complex c) return (TOut)Convert.ChangeType(c.Real, tout);
                return (TOut)Convert.ChangeType(@in, tout);
            };
        }

        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl(Optimize)]
        public static TOut ChangeType<TOut>(Object value)
        {
            if (value == null)
                return default;

            // NumPy-compatible conversion using Converts.ToXxx methods
            // These methods handle NaN/Inf, overflow/wrapping, and truncation correctly
            switch (InfoOf<TOut>.NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    return (TOut)(object)ToBoolean_NumPy(value);
                case NPTypeCode.Char:
                    return (TOut)(object)Converts.ToChar(ToLong_NumPy(value));
                case NPTypeCode.Byte:
                    return (TOut)(object)ToByte_NumPy(value);
                case NPTypeCode.SByte:
                    return (TOut)(object)ToSByte_NumPy(value);
                case NPTypeCode.Int16:
                    return (TOut)(object)ToInt16_NumPy(value);
                case NPTypeCode.UInt16:
                    return (TOut)(object)ToUInt16_NumPy(value);
                case NPTypeCode.Int32:
                    return (TOut)(object)ToInt32_NumPy(value);
                case NPTypeCode.UInt32:
                    return (TOut)(object)ToUInt32_NumPy(value);
                case NPTypeCode.Int64:
                    return (TOut)(object)ToInt64_NumPy(value);
                case NPTypeCode.UInt64:
                    return (TOut)(object)ToUInt64_NumPy(value);
                case NPTypeCode.Single:
                    return (TOut)(object)ToSingle_NumPy(value);
                case NPTypeCode.Double:
                    return (TOut)(object)ToDouble_NumPy(value);
                case NPTypeCode.Decimal:
                    return (TOut)(object)ToDecimal_NumPy(value);
                case NPTypeCode.Half:
                    return (TOut)(object)ToHalf_NumPy(value);
                case NPTypeCode.Complex:
                    return (TOut)(object)ToComplex_NumPy(value);
                case NPTypeCode.String:
                    return (TOut)(object)((IConvertible)value).ToString(CultureInfo.InvariantCulture);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }


        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl(Optimize)]
        public static Object ChangeType(Object value, NPTypeCode typeCode)
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // NumPy-compatible conversion using Converts.ToXxx methods
            // These methods handle NaN/Inf, overflow/wrapping, and truncation correctly
            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    return ToBoolean_NumPy(value);
                case NPTypeCode.Char:
                    return Converts.ToChar(ToLong_NumPy(value));
                case NPTypeCode.Byte:
                    return ToByte_NumPy(value);
                case NPTypeCode.SByte:
                    return ToSByte_NumPy(value);
                case NPTypeCode.Int16:
                    return ToInt16_NumPy(value);
                case NPTypeCode.UInt16:
                    return ToUInt16_NumPy(value);
                case NPTypeCode.Int32:
                    return ToInt32_NumPy(value);
                case NPTypeCode.UInt32:
                    return ToUInt32_NumPy(value);
                case NPTypeCode.Int64:
                    return ToInt64_NumPy(value);
                case NPTypeCode.UInt64:
                    return ToUInt64_NumPy(value);
                case NPTypeCode.Single:
                    return ToSingle_NumPy(value);
                case NPTypeCode.Double:
                    return ToDouble_NumPy(value);
                case NPTypeCode.Decimal:
                    return ToDecimal_NumPy(value);
                case NPTypeCode.Half:
                    return ToHalf_NumPy(value);
                case NPTypeCode.Complex:
                    return ToComplex_NumPy(value);
                case NPTypeCode.String:
                    return ((IConvertible)value).ToString(CultureInfo.InvariantCulture);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }

        // NumPy-compatible conversion helper methods
        // These route to our Converts.ToXxx methods which handle NaN/Inf/overflow correctly

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ToBoolean_NumPy(object value) => value switch
        {
            bool b => b,
            double d => Converts.ToBoolean(d),
            float f => Converts.ToBoolean(f),
            Half h => Converts.ToBoolean(h),
            Complex c => Converts.ToBoolean(c),
            decimal m => Converts.ToBoolean(m),
            long l => Converts.ToBoolean(l),
            ulong ul => Converts.ToBoolean(ul),
            int i => Converts.ToBoolean(i),
            uint ui => Converts.ToBoolean(ui),
            short s => Converts.ToBoolean(s),
            ushort us => Converts.ToBoolean(us),
            byte by => Converts.ToBoolean(by),
            sbyte sb => Converts.ToBoolean(sb),
            char c => Converts.ToBoolean(c),
            _ => Converts.ToBoolean(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte ToByte_NumPy(object value) => value switch
        {
            byte b => b,
            double d => Converts.ToByte(d),
            float f => Converts.ToByte(f),
            Half h => Converts.ToByte(h),
            Complex c => Converts.ToByte(c),
            decimal m => Converts.ToByte(m),
            long l => Converts.ToByte(l),
            ulong ul => Converts.ToByte(ul),
            int i => Converts.ToByte(i),
            uint ui => Converts.ToByte(ui),
            short s => Converts.ToByte(s),
            ushort us => Converts.ToByte(us),
            sbyte sb => Converts.ToByte(sb),
            char c => Converts.ToByte(c),
            bool b => Converts.ToByte(b),
            _ => Converts.ToByte(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ToSByte_NumPy(object value) => value switch
        {
            sbyte sb => sb,
            double d => Converts.ToSByte(d),
            float f => Converts.ToSByte(f),
            Half h => Converts.ToSByte(h),
            Complex c => Converts.ToSByte(c),
            decimal m => Converts.ToSByte(m),
            long l => Converts.ToSByte(l),
            ulong ul => Converts.ToSByte(ul),
            int i => Converts.ToSByte(i),
            uint ui => Converts.ToSByte(ui),
            short s => Converts.ToSByte(s),
            ushort us => Converts.ToSByte(us),
            byte b => Converts.ToSByte(b),
            char c => Converts.ToSByte(c),
            bool b => Converts.ToSByte(b),
            _ => Converts.ToSByte(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static short ToInt16_NumPy(object value) => value switch
        {
            short s => s,
            double d => Converts.ToInt16(d),
            float f => Converts.ToInt16(f),
            Half h => Converts.ToInt16(h),
            Complex c => Converts.ToInt16(c),
            decimal m => Converts.ToInt16(m),
            long l => Converts.ToInt16(l),
            ulong ul => Converts.ToInt16(ul),
            int i => Converts.ToInt16(i),
            uint ui => Converts.ToInt16(ui),
            ushort us => Converts.ToInt16(us),
            byte b => Converts.ToInt16(b),
            sbyte sb => Converts.ToInt16(sb),
            char c => Converts.ToInt16(c),
            bool b => Converts.ToInt16(b),
            _ => Converts.ToInt16(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort ToUInt16_NumPy(object value) => value switch
        {
            ushort us => us,
            double d => Converts.ToUInt16(d),
            float f => Converts.ToUInt16(f),
            Half h => Converts.ToUInt16(h),
            Complex c => Converts.ToUInt16(c),
            decimal m => Converts.ToUInt16(m),
            long l => Converts.ToUInt16(l),
            ulong ul => Converts.ToUInt16(ul),
            int i => Converts.ToUInt16(i),
            uint ui => Converts.ToUInt16(ui),
            short s => Converts.ToUInt16(s),
            byte b => Converts.ToUInt16(b),
            sbyte sb => Converts.ToUInt16(sb),
            char c => Converts.ToUInt16(c),
            bool b => Converts.ToUInt16(b),
            _ => Converts.ToUInt16(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToInt32_NumPy(object value) => value switch
        {
            int i => i,
            double d => Converts.ToInt32(d),
            float f => Converts.ToInt32(f),
            Half h => Converts.ToInt32(h),
            Complex c => Converts.ToInt32(c),
            decimal m => Converts.ToInt32(m),
            long l => Converts.ToInt32(l),
            ulong ul => Converts.ToInt32(ul),
            uint ui => Converts.ToInt32(ui),
            short s => Converts.ToInt32(s),
            ushort us => Converts.ToInt32(us),
            byte b => Converts.ToInt32(b),
            sbyte sb => Converts.ToInt32(sb),
            char c => Converts.ToInt32(c),
            bool b => Converts.ToInt32(b),
            _ => Converts.ToInt32(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint ToUInt32_NumPy(object value) => value switch
        {
            uint ui => ui,
            double d => Converts.ToUInt32(d),
            float f => Converts.ToUInt32(f),
            Half h => Converts.ToUInt32(h),
            Complex c => Converts.ToUInt32(c),
            decimal m => Converts.ToUInt32(m),
            long l => Converts.ToUInt32(l),
            ulong ul => Converts.ToUInt32(ul),
            int i => Converts.ToUInt32(i),
            short s => Converts.ToUInt32(s),
            ushort us => Converts.ToUInt32(us),
            byte b => Converts.ToUInt32(b),
            sbyte sb => Converts.ToUInt32(sb),
            char c => Converts.ToUInt32(c),
            bool b => Converts.ToUInt32(b),
            _ => Converts.ToUInt32(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ToInt64_NumPy(object value) => value switch
        {
            long l => l,
            double d => Converts.ToInt64(d),
            float f => Converts.ToInt64(f),
            Half h => Converts.ToInt64(h),
            Complex c => Converts.ToInt64(c),
            decimal m => Converts.ToInt64(m),
            ulong ul => Converts.ToInt64(ul),
            int i => Converts.ToInt64(i),
            uint ui => Converts.ToInt64(ui),
            short s => Converts.ToInt64(s),
            ushort us => Converts.ToInt64(us),
            byte b => Converts.ToInt64(b),
            sbyte sb => Converts.ToInt64(sb),
            char c => Converts.ToInt64(c),
            bool b => Converts.ToInt64(b),
            _ => Converts.ToInt64(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ToUInt64_NumPy(object value) => value switch
        {
            ulong ul => ul,
            double d => Converts.ToUInt64(d),
            float f => Converts.ToUInt64(f),
            Half h => Converts.ToUInt64(h),
            Complex c => Converts.ToUInt64(c),
            decimal m => Converts.ToUInt64(m),
            long l => Converts.ToUInt64(l),
            int i => Converts.ToUInt64(i),
            uint ui => Converts.ToUInt64(ui),
            short s => Converts.ToUInt64(s),
            ushort us => Converts.ToUInt64(us),
            byte b => Converts.ToUInt64(b),
            sbyte sb => Converts.ToUInt64(sb),
            char c => Converts.ToUInt64(c),
            bool b => Converts.ToUInt64(b),
            _ => Converts.ToUInt64(((IConvertible)value).ToDouble(null))
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToSingle_NumPy(object value) => value switch
        {
            float f => f,
            double d => Converts.ToSingle(d),
            Half h => Converts.ToSingle(h),
            Complex c => Converts.ToSingle(c),
            decimal m => Converts.ToSingle(m),
            long l => Converts.ToSingle(l),
            ulong ul => Converts.ToSingle(ul),
            int i => Converts.ToSingle(i),
            uint ui => Converts.ToSingle(ui),
            short s => Converts.ToSingle(s),
            ushort us => Converts.ToSingle(us),
            byte b => Converts.ToSingle(b),
            sbyte sb => Converts.ToSingle(sb),
            char c => Converts.ToSingle(c),
            bool b => Converts.ToSingle(b),
            _ => ((IConvertible)value).ToSingle(null)
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble_NumPy(object value) => value switch
        {
            double d => d,
            float f => Converts.ToDouble(f),
            Half h => Converts.ToDouble(h),
            Complex c => c.Real,  // NumPy: discard imaginary
            decimal m => Converts.ToDouble(m),
            long l => Converts.ToDouble(l),
            ulong ul => Converts.ToDouble(ul),
            int i => Converts.ToDouble(i),
            uint ui => Converts.ToDouble(ui),
            short s => Converts.ToDouble(s),
            ushort us => Converts.ToDouble(us),
            byte b => Converts.ToDouble(b),
            sbyte sb => Converts.ToDouble(sb),
            char c => Converts.ToDouble(c),
            bool b => Converts.ToDouble(b),
            _ => ((IConvertible)value).ToDouble(null)
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static decimal ToDecimal_NumPy(object value) => value switch
        {
            decimal m => m,
            double d => Converts.ToDecimal(d),
            float f => Converts.ToDecimal(f),
            Half h => Converts.ToDecimal(h),
            long l => Converts.ToDecimal(l),
            ulong ul => Converts.ToDecimal(ul),
            int i => Converts.ToDecimal(i),
            uint ui => Converts.ToDecimal(ui),
            short s => Converts.ToDecimal(s),
            ushort us => Converts.ToDecimal(us),
            byte b => Converts.ToDecimal(b),
            sbyte sb => Converts.ToDecimal(sb),
            char c => Converts.ToDecimal(c),
            bool b => Converts.ToDecimal(b),
            _ => ((IConvertible)value).ToDecimal(null)
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Half ToHalf_NumPy(object value) => value switch
        {
            Half h => h,
            double d => Converts.ToHalf(d),
            float f => Converts.ToHalf(f),
            decimal m => (Half)(double)m,
            long l => Converts.ToHalf(l),
            ulong ul => Converts.ToHalf(ul),
            int i => Converts.ToHalf(i),
            uint ui => Converts.ToHalf(ui),
            short s => Converts.ToHalf(s),
            ushort us => Converts.ToHalf(us),
            byte b => Converts.ToHalf(b),
            sbyte sb => Converts.ToHalf(sb),
            char c => (Half)c,
            bool b => b ? (Half)1 : (Half)0,
            _ => (Half)((IConvertible)value).ToDouble(null)
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Complex ToComplex_NumPy(object value) => value switch
        {
            Complex c => c,
            double d => new Complex(d, 0),
            float f => new Complex(f, 0),
            Half h => new Complex((double)h, 0),
            decimal m => new Complex((double)m, 0),
            long l => new Complex(l, 0),
            ulong ul => new Complex(ul, 0),
            int i => new Complex(i, 0),
            uint ui => new Complex(ui, 0),
            short s => new Complex(s, 0),
            ushort us => new Complex(us, 0),
            byte b => new Complex(b, 0),
            sbyte sb => new Complex(sb, 0),
            char c => new Complex(c, 0),
            bool b => new Complex(b ? 1 : 0, 0),
            _ => new Complex(((IConvertible)value).ToDouble(null), 0)
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ToLong_NumPy(object value) => value switch
        {
            long l => l,
            int i => i,
            short s => s,
            sbyte sb => sb,
            ulong ul => (long)ul,
            uint ui => ui,
            ushort us => us,
            byte b => b,
            char c => c,
            double d => Converts.ToInt64(d),
            float f => Converts.ToInt64(f),
            Half h => Converts.ToInt64(h),
            decimal m => Converts.ToInt64(m),
            bool b => b ? 1L : 0L,
            _ => Converts.ToInt64(((IConvertible)value).ToDouble(null))
        };

        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl(Optimize)]
        public static Object ChangeType<T>(T value, NPTypeCode typeCode) where T : IConvertible
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;

#if _REGEN
            switch (typeCode)
            {
	            %foreach supported_dtypes, supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                	    %foreach supported_dtypes, supported_dtypes_lowercase%
	                    case NPTypeCode.#101: return Converts.To#1(Unsafe.As<T, #102>(ref value));
	                    %
                        default: throw new NotSupportedException();
                    }
	            %
	            default:
		            throw new NotSupportedException();
            }
#else

            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToBoolean(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToBoolean(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToBoolean(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToBoolean(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToBoolean(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToBoolean(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToBoolean(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToBoolean(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToBoolean(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToBoolean(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToBoolean(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToBoolean(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Byte:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToByte(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToByte(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToByte(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToByte(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToByte(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToByte(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToByte(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToByte(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToByte(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToByte(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToByte(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToByte(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int16:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt16(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt16(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt16(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt16(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt16(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt16(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt16(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt16(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt16(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt16(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt16(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt16(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt16:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt16(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt16(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt16(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt16(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt16(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt16(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt16(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt16(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt16(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt16(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt16(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt16(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int32:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt32(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt32(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt32(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt32(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt32(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt32(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt32(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt32(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt32(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt32(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt32(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt32(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt32:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt32(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt32(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt32(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt32(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt32(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt32(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt32(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt32(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt32(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt32(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt32(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt32(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int64:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt64(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt64(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt64(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt64(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt64(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt64(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt64(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt64(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt64(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt64(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt64(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt64(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt64:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt64(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt64(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt64(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt64(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt64(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt64(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt64(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt64(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt64(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt64(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt64(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt64(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Char:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToChar(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToChar(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToChar(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToChar(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToChar(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToChar(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToChar(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToChar(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToChar(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToChar(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToChar(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToChar(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Double:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToDouble(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToDouble(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToDouble(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToDouble(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToDouble(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToDouble(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToDouble(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToDouble(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToDouble(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToDouble(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToDouble(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToDouble(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Single:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToSingle(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToSingle(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToSingle(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToSingle(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToSingle(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToSingle(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToSingle(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToSingle(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToSingle(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToSingle(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToSingle(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToSingle(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Decimal:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToDecimal(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToDecimal(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToDecimal(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToDecimal(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToDecimal(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToDecimal(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToDecimal(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToDecimal(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToDecimal(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToDecimal(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToDecimal(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToDecimal(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Half:
                    // Half target type - C# Half has direct casts from all numeric types except decimal
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return (Half)(Unsafe.As<T, bool>(ref value) ? 1 : 0);
                        case NPTypeCode.Byte:    return (Half)Unsafe.As<T, byte>(ref value);
                        case NPTypeCode.SByte:   return (Half)Unsafe.As<T, sbyte>(ref value);
                        case NPTypeCode.Int16:   return (Half)Unsafe.As<T, short>(ref value);
                        case NPTypeCode.UInt16:  return (Half)Unsafe.As<T, ushort>(ref value);
                        case NPTypeCode.Int32:   return (Half)Unsafe.As<T, int>(ref value);
                        case NPTypeCode.UInt32:  return (Half)Unsafe.As<T, uint>(ref value);
                        case NPTypeCode.Int64:   return (Half)Unsafe.As<T, long>(ref value);
                        case NPTypeCode.UInt64:  return (Half)Unsafe.As<T, ulong>(ref value);
                        case NPTypeCode.Char:    return (Half)Unsafe.As<T, char>(ref value);
                        case NPTypeCode.Double:  return (Half)Unsafe.As<T, double>(ref value);
                        case NPTypeCode.Single:  return (Half)Unsafe.As<T, float>(ref value);
                        case NPTypeCode.Decimal: return (Half)Unsafe.As<T, decimal>(ref value);
                        default:
                            throw new NotSupportedException();
                    }
                default:
                    throw new NotSupportedException();
            }
#endif
        }


        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl(Optimize)]
        public static TOut ChangeType<TIn, TOut>(TIn value) where TIn : IConvertible where TOut : IConvertible
        {
            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;

#if _REGEN
            switch (InfoOf<TOut>.NPTypeCode)
            {
	            %foreach supported_dtypes, supported_dtypes_lowercase%
	            case NPTypeCode.#1: {
                    |#2 res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
                	    %foreach supported_dtypes, supported_dtypes_lowercase%
	                    case NPTypeCode.#101: res = Converts.To#1(Unsafe.As<TIn, #102>(ref value)); return Unsafe.As<#2, TOut>(ref res);
	                    %
                        default: throw new NotSupportedException();
                    }
                }
	            %
	            default:
		            throw new NotSupportedException();
            }
#else

            switch (InfoOf<TOut>.NPTypeCode)
            {
	            case NPTypeCode.Boolean: {
                    bool res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToBoolean(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToBoolean(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToBoolean(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToBoolean(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToBoolean(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToBoolean(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToBoolean(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToBoolean(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToBoolean(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToBoolean(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToBoolean(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<bool, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToBoolean(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<bool, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Byte: {
                    byte res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToByte(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToByte(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToByte(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToByte(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToByte(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToByte(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToByte(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToByte(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToByte(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToByte(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToByte(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<byte, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToByte(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<byte, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Int16: {
                    short res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToInt16(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToInt16(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToInt16(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToInt16(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToInt16(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToInt16(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToInt16(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToInt16(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToInt16(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToInt16(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToInt16(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<short, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToInt16(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<short, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.UInt16: {
                    ushort res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToUInt16(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToUInt16(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToUInt16(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToUInt16(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToUInt16(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToUInt16(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToUInt16(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToUInt16(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToUInt16(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToUInt16(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToUInt16(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToUInt16(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<ushort, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Int32: {
                    int res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToInt32(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToInt32(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToInt32(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToInt32(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToInt32(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToInt32(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToInt32(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToInt32(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToInt32(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToInt32(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToInt32(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<int, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToInt32(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<int, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.UInt32: {
                    uint res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToUInt32(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToUInt32(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToUInt32(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToUInt32(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToUInt32(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToUInt32(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToUInt32(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToUInt32(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToUInt32(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToUInt32(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToUInt32(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<uint, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToUInt32(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<uint, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Int64: {
                    long res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToInt64(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToInt64(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToInt64(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToInt64(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToInt64(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToInt64(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToInt64(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToInt64(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToInt64(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToInt64(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToInt64(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<long, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToInt64(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<long, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.UInt64: {
                    ulong res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToUInt64(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToUInt64(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToUInt64(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToUInt64(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToUInt64(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToUInt64(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToUInt64(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToUInt64(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToUInt64(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToUInt64(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToUInt64(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToUInt64(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<ulong, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Char: {
                    char res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToChar(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToChar(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToChar(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToChar(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToChar(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToChar(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToChar(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToChar(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToChar(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToChar(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToChar(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<char, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToChar(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<char, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Double: {
                    double res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToDouble(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToDouble(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToDouble(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToDouble(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToDouble(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToDouble(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToDouble(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToDouble(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToDouble(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToDouble(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToDouble(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<double, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToDouble(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<double, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Single: {
                    float res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToSingle(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToSingle(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToSingle(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToSingle(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToSingle(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToSingle(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToSingle(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToSingle(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToSingle(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToSingle(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToSingle(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<float, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToSingle(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<float, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            case NPTypeCode.Decimal: {
                    decimal res;
                    switch (InfoOf<TIn>.NPTypeCode)
                    {
	                    case NPTypeCode.Boolean: res = Converts.ToDecimal(Unsafe.As<TIn, bool>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Byte: res = Converts.ToDecimal(Unsafe.As<TIn, byte>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Int16: res = Converts.ToDecimal(Unsafe.As<TIn, short>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.UInt16: res = Converts.ToDecimal(Unsafe.As<TIn, ushort>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Int32: res = Converts.ToDecimal(Unsafe.As<TIn, int>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.UInt32: res = Converts.ToDecimal(Unsafe.As<TIn, uint>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Int64: res = Converts.ToDecimal(Unsafe.As<TIn, long>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.UInt64: res = Converts.ToDecimal(Unsafe.As<TIn, ulong>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Char: res = Converts.ToDecimal(Unsafe.As<TIn, char>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Double: res = Converts.ToDecimal(Unsafe.As<TIn, double>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Single: res = Converts.ToDecimal(Unsafe.As<TIn, float>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
	                    case NPTypeCode.Decimal: res = Converts.ToDecimal(Unsafe.As<TIn, decimal>(ref value)); return Unsafe.As<decimal, TOut>(ref res);
                        default: throw new NotSupportedException();
                    }
                }
	            default:
		            throw new NotSupportedException();
            }
#endif
        }

        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl(Optimize)]
        public static Object ChangeType(Object value, NPTypeCode typeCode, IFormatProvider provider)
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    return ((IConvertible)value).ToBoolean(provider);
                case NPTypeCode.Char:
                    return ((IConvertible)value).ToChar(provider);
                case NPTypeCode.Byte:
                    return ((IConvertible)value).ToByte(provider);
                case NPTypeCode.Int16:
                    return ((IConvertible)value).ToInt16(provider);
                case NPTypeCode.UInt16:
                    return ((IConvertible)value).ToUInt16(provider);
                case NPTypeCode.Int32:
                    return ((IConvertible)value).ToInt32(provider);
                case NPTypeCode.UInt32:
                    return ((IConvertible)value).ToUInt32(provider);
                case NPTypeCode.Int64:
                    return ((IConvertible)value).ToInt64(provider);
                case NPTypeCode.UInt64:
                    return ((IConvertible)value).ToUInt64(provider);
                case NPTypeCode.Single:
                    return ((IConvertible)value).ToSingle(provider);
                case NPTypeCode.Double:
                    return ((IConvertible)value).ToDouble(provider);
                case NPTypeCode.Decimal:
                    return ((IConvertible)value).ToDecimal(provider);
                case NPTypeCode.String:
                    return ((IConvertible)value).ToString(provider);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }


        /// <summary>
        ///     Finds the conversion method from <see cref="Convert"/> based on <typeparamref name="TIn"/> and <typeparamref name="TOut"/>.
        /// </summary>
        /// <typeparam name="TIn">The type that is expected as input and to be converted from</typeparam>
        /// <typeparam name="TOut">The type we expect to convert to.</typeparam>
        public static Func<TIn, TOut> FindConverter<TIn, TOut>()
        {
#if _REGEN
#region Compute
            //#n is input, #10n is output
		    switch (InfoOf<TIn>.NPTypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            %foreach supported_dtypes,supported_dtypes_lowercase%
			            case NPTypeCode.#101:
			            {
				            Func<#2, #102> ret = Converts.55To#101;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            %
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    %
			    default:
                {
                    var tout = typeof(TOut);
                    return @in => (TOut)Convert.ChangeType(@in, tout);
                }
		    }
#endregion
#else

#region Compute

            //#n is input, #10n is output
            switch (InfoOf<TIn>.NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<bool, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<bool, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<bool, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<bool, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<bool, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<bool, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<bool, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<bool, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<bool, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<bool, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<bool, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<bool, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Byte:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<byte, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<byte, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<byte, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<byte, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<byte, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<byte, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<byte, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<byte, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<byte, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<byte, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<byte, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<byte, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Int16:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<short, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<short, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<short, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<short, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<short, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<short, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<short, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<short, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<short, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<short, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<short, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<short, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.UInt16:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<ushort, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<ushort, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<ushort, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<ushort, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<ushort, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<ushort, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<ushort, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<ushort, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<ushort, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<ushort, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<ushort, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<ushort, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Int32:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<int, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<int, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<int, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<int, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<int, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<int, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<int, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<int, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<int, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<int, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<int, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<int, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.UInt32:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<uint, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<uint, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<uint, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<uint, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<uint, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<uint, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<uint, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<uint, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<uint, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<uint, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<uint, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<uint, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Int64:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<long, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<long, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<long, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<long, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<long, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<long, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<long, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<long, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<long, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<long, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<long, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<long, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.UInt64:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<ulong, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<ulong, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<ulong, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<ulong, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<ulong, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<ulong, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<ulong, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<ulong, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<ulong, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<ulong, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<ulong, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<ulong, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Char:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<char, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<char, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<char, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<char, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<char, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<char, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<char, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<char, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<char, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<char, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<char, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<char, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Double:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<double, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<double, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<double, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<double, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<double, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<double, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<double, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<double, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<double, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<double, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<double, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<double, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Single:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<float, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<float, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<float, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<float, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<float, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<float, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<float, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<float, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<float, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<float, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<float, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<float, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                case NPTypeCode.Decimal:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<decimal, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<decimal, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<decimal, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<decimal, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<decimal, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<decimal, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<decimal, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<decimal, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<decimal, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<decimal, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<decimal, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<decimal, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                    return CreateFallbackConverter<TIn, TOut>();
                        }
                    }
                default:
                    return CreateFallbackConverter<TIn, TOut>();
            }

#endregion

#endif
        }

#region ToScalar

#if _REGEN
#region Compute

		%foreach supported_dtypes,supported_dtypes_lowercase%
		[MethodImpl(Inline)]
        public static #2 To#1(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.#1 ? nd.GetAtIndex<#2>(0) : Converts.To#1(nd.GetAtIndex(0));
        }
		%
			    
#endregion
#else

#region Compute

        [MethodImpl(Inline)]
        public static bool ToBoolean(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Boolean ? nd.GetAtIndex<bool>(0) : Converts.ToBoolean(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static byte ToByte(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Byte ? nd.GetAtIndex<byte>(0) : Converts.ToByte(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static short ToInt16(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int16 ? nd.GetAtIndex<short>(0) : Converts.ToInt16(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static ushort ToUInt16(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt16 ? nd.GetAtIndex<ushort>(0) : Converts.ToUInt16(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static int ToInt32(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int32 ? nd.GetAtIndex<int>(0) : Converts.ToInt32(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static uint ToUInt32(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt32 ? nd.GetAtIndex<uint>(0) : Converts.ToUInt32(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static long ToInt64(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int64 ? nd.GetAtIndex<long>(0) : Converts.ToInt64(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static ulong ToUInt64(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt64 ? nd.GetAtIndex<ulong>(0) : Converts.ToUInt64(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static char ToChar(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Char ? nd.GetAtIndex<char>(0) : Converts.ToChar(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static double ToDouble(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Double ? nd.GetAtIndex<double>(0) : Converts.ToDouble(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static float ToSingle(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Single ? nd.GetAtIndex<float>(0) : Converts.ToSingle(nd.GetAtIndex(0));
        }

        [MethodImpl(Inline)]
        public static decimal ToDecimal(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Decimal ? nd.GetAtIndex<decimal>(0) : Converts.ToDecimal(nd.GetAtIndex(0));
        }

#endregion

#endif

#endregion
    }
}
