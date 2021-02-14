using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides various methods related to <see cref="System.Convert"/>.
    /// </summary>
    public static partial class Converts
    {
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
        [MethodImpl((MethodImplOptions)512)]
        public static TOut ChangeType<TOut>(Object value)
        {
            if (value == null)
                return default;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (InfoOf<TOut>.NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    return (TOut)(object)((IConvertible)value).ToBoolean(CultureInfo.InvariantCulture);
                case NPTypeCode.Byte:
                    return (TOut)(object)((IConvertible)value).ToByte(CultureInfo.InvariantCulture);
                case NPTypeCode.Int32:
                    return (TOut)(object)((IConvertible)value).ToInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.Int64:
                    return (TOut)(object)((IConvertible)value).ToInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.Single:
                    return (TOut)(object)((IConvertible)value).ToSingle(CultureInfo.InvariantCulture);
                case NPTypeCode.Double:
                    return (TOut)(object)((IConvertible)value).ToDouble(CultureInfo.InvariantCulture);
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
        [MethodImpl((MethodImplOptions)512)]
        public static Object ChangeType(Object value, NPTypeCode typeCode)
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    return ((IConvertible)value).ToBoolean(CultureInfo.InvariantCulture);
                case NPTypeCode.Byte:
                    return ((IConvertible)value).ToByte(CultureInfo.InvariantCulture);
                case NPTypeCode.Int32:
                    return ((IConvertible)value).ToInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.Int64:
                    return ((IConvertible)value).ToInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.Single:
                    return ((IConvertible)value).ToSingle(CultureInfo.InvariantCulture);
                case NPTypeCode.Double:
                    return ((IConvertible)value).ToDouble(CultureInfo.InvariantCulture);
                case NPTypeCode.String:
                    return ((IConvertible)value).ToString(CultureInfo.InvariantCulture);
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
        [MethodImpl((MethodImplOptions)512)]
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
                case NPTypeCode.Byte:
                    return ((IConvertible)value).ToByte(provider);
                case NPTypeCode.Int32:
                    return ((IConvertible)value).ToInt32(provider);
                case NPTypeCode.Int64:
                    return ((IConvertible)value).ToInt64(provider);
                case NPTypeCode.Single:
                    return ((IConvertible)value).ToSingle(provider);
                case NPTypeCode.Double:
                    return ((IConvertible)value).ToDouble(provider);
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
#if _REGEN1
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
				            Func<#2, #102> ret = Converts.To#101;
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
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<bool, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<bool, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<bool, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<bool, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<bool, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    case NPTypeCode.Byte:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            case NPTypeCode.Boolean:
			            {
				            Func<byte, bool> ret = Converts.ToBoolean;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<byte, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<byte, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<byte, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<byte, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<byte, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    case NPTypeCode.Int32:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            case NPTypeCode.Boolean:
			            {
				            Func<int, bool> ret = Converts.ToBoolean;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<int, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<int, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<int, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<int, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<int, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    case NPTypeCode.Int64:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            case NPTypeCode.Boolean:
			            {
				            Func<long, bool> ret = Converts.ToBoolean;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<long, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<long, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<long, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<long, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<long, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    case NPTypeCode.Single:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            case NPTypeCode.Boolean:
			            {
				            Func<float, bool> ret = Converts.ToBoolean;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<float, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<float, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<float, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<float, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<float, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    case NPTypeCode.Double:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            case NPTypeCode.Boolean:
			            {
				            Func<double, bool> ret = Converts.ToBoolean;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Byte:
			            {
				            Func<double, byte> ret = Converts.ToByte;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int32:
			            {
				            Func<double, int> ret = Converts.ToInt32;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Int64:
			            {
				            Func<double, long> ret = Converts.ToInt64;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Single:
			            {
				            Func<double, float> ret = Converts.ToSingle;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            case NPTypeCode.Double:
			            {
				            Func<double, double> ret = Converts.ToDouble;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    default:
                {
                    var tout = typeof(TOut);
                    return @in => (TOut)Convert.ChangeType(@in, tout);
                }
		    }
            #endregion
#endif
        }

        #region ToScalar

#if _REGEN1
        #region Compute

		%foreach supported_dtypes,supported_dtypes_lowercase%
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool ToBoolean(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Boolean ? nd.GetAtIndex<bool>(0) : Converts.ToBoolean(nd.GetAtIndex(0));
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Byte ? nd.GetAtIndex<byte>(0) : Converts.ToByte(nd.GetAtIndex(0));
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int32 ? nd.GetAtIndex<int>(0) : Converts.ToInt32(nd.GetAtIndex(0));
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int64 ? nd.GetAtIndex<long>(0) : Converts.ToInt64(nd.GetAtIndex(0));
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Single ? nd.GetAtIndex<float>(0) : Converts.ToSingle(nd.GetAtIndex(0));
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Double ? nd.GetAtIndex<double>(0) : Converts.ToDouble(nd.GetAtIndex(0));
        }
			    
        #endregion
#endif

        #endregion
    }
}
