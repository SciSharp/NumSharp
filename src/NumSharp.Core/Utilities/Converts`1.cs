using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides various methods related to <see cref="System.Convert"/> based on give <typeparamref name="T"/>.
    /// </summary>
    public static class Converts<T>
    {
        static Converts()
        {
            if (typeof(T).GetTypeCode() == NPTypeCode.Empty)
                throw new NotSupportedException($"Unable to perform conversions in Converts<T> for type {typeof(T).Name}");
        }
        
        #region Cached Converters

#if _REGEN1
        #region Compute
		%foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="#1"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="#1"/></param>
        /// <returns>A <see cref="#1"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static #2 To#1(T obj) => _to#1(obj);
		private static readonly Func<T, #2> _to#1 = Converts.FindConverter<T, #2>();

		%
        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="String"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="String"/></param>
        /// <returns>A <see cref="String"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static String ToString(T obj) => _toString(obj);
		private static readonly Func<T, string> _toString = Converts.FindConverter<T, string>();
        #endregion
#else

        #region Compute
        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Boolean"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Boolean"/></param>
        /// <returns>A <see cref="Boolean"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool ToBoolean(T obj) => _toBoolean(obj);
		private static readonly Func<T, bool> _toBoolean = Converts.FindConverter<T, bool>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Byte"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Byte"/></param>
        /// <returns>A <see cref="Byte"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte ToByte(T obj) => _toByte(obj);
		private static readonly Func<T, byte> _toByte = Converts.FindConverter<T, byte>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Int32"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Int32"/></param>
        /// <returns>A <see cref="Int32"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int ToInt32(T obj) => _toInt32(obj);
		private static readonly Func<T, int> _toInt32 = Converts.FindConverter<T, int>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Int64"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Int64"/></param>
        /// <returns>A <see cref="Int64"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long ToInt64(T obj) => _toInt64(obj);
		private static readonly Func<T, long> _toInt64 = Converts.FindConverter<T, long>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Single"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Single"/></param>
        /// <returns>A <see cref="Single"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float ToSingle(T obj) => _toSingle(obj);
		private static readonly Func<T, float> _toSingle = Converts.FindConverter<T, float>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="Double"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="Double"/></param>
        /// <returns>A <see cref="Double"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double ToDouble(T obj) => _toDouble(obj);
		private static readonly Func<T, double> _toDouble = Converts.FindConverter<T, double>();

        /// <summary>
        ///     Converts <typeparamref name="T"/> to <see cref="String"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <see cref="String"/></param>
        /// <returns>A <see cref="String"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static String ToString(T obj) => _toString(obj);
		private static readonly Func<T, string> _toString = Converts.FindConverter<T, string>();
        #endregion
#endif

#if _REGEN
		#region Compute
		%foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Converts <see cref="#1"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="#1"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(#2 obj) => _from#1(obj);
		private static readonly Func<#2, T> _from#1 = Converts.FindConverter<#2, T>();

		%
        /// <summary>
        ///     Converts <see cref="String"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="String"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(string obj) => _fromString(obj);
        private static readonly Func<string, T> _fromString = Converts.FindConverter<string, T>();
		#endregion
#else

		#region Compute
        /// <summary>
        ///     Converts <see cref="Boolean"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Boolean"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(bool obj) => _fromBoolean(obj);
		private static readonly Func<bool, T> _fromBoolean = Converts.FindConverter<bool, T>();

        /// <summary>
        ///     Converts <see cref="Byte"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Byte"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(byte obj) => _fromByte(obj);
		private static readonly Func<byte, T> _fromByte = Converts.FindConverter<byte, T>();

        /// <summary>
        ///     Converts <see cref="Int16"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Int16"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(short obj) => _fromInt16(obj);
		private static readonly Func<short, T> _fromInt16 = Converts.FindConverter<short, T>();

        /// <summary>
        ///     Converts <see cref="UInt16"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="UInt16"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(ushort obj) => _fromUInt16(obj);
		private static readonly Func<ushort, T> _fromUInt16 = Converts.FindConverter<ushort, T>();

        /// <summary>
        ///     Converts <see cref="Int32"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Int32"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(int obj) => _fromInt32(obj);
		private static readonly Func<int, T> _fromInt32 = Converts.FindConverter<int, T>();

        /// <summary>
        ///     Converts <see cref="UInt32"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="UInt32"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(uint obj) => _fromUInt32(obj);
		private static readonly Func<uint, T> _fromUInt32 = Converts.FindConverter<uint, T>();

        /// <summary>
        ///     Converts <see cref="Int64"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Int64"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(long obj) => _fromInt64(obj);
		private static readonly Func<long, T> _fromInt64 = Converts.FindConverter<long, T>();

        /// <summary>
        ///     Converts <see cref="UInt64"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="UInt64"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(ulong obj) => _fromUInt64(obj);
		private static readonly Func<ulong, T> _fromUInt64 = Converts.FindConverter<ulong, T>();

        /// <summary>
        ///     Converts <see cref="Char"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Char"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(char obj) => _fromChar(obj);
		private static readonly Func<char, T> _fromChar = Converts.FindConverter<char, T>();

        /// <summary>
        ///     Converts <see cref="Double"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Double"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(double obj) => _fromDouble(obj);
		private static readonly Func<double, T> _fromDouble = Converts.FindConverter<double, T>();

        /// <summary>
        ///     Converts <see cref="Single"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Single"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(float obj) => _fromSingle(obj);
		private static readonly Func<float, T> _fromSingle = Converts.FindConverter<float, T>();

        /// <summary>
        ///     Converts <see cref="Decimal"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="Decimal"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(decimal obj) => _fromDecimal(obj);
		private static readonly Func<decimal, T> _fromDecimal = Converts.FindConverter<decimal, T>();

        /// <summary>
        ///     Converts <see cref="String"/> to <typeparamref name="T"/> using staticly cached <see cref="Converts.FindConverter{TIn,TOut}"/>.
        /// </summary>
        /// <param name="obj">The object to convert to <typeparamref name="T"/> from <see cref="String"/></param>
        /// <returns>A <typeparamref name="T"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static T From(string obj) => _fromString(obj);
        private static readonly Func<string, T> _fromString = Converts.FindConverter<string, T>();
		#endregion
#endif

        #endregion
    }
}
