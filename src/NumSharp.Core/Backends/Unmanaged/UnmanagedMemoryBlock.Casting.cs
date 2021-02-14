using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static partial class UnmanagedMemoryBlock
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <param name="to">The type to cast this memory block to.</param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static IMemoryBlock CastTo(this IMemoryBlock source, NPTypeCode to)
        {
            switch (to)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return CastTo<#2>(source);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return CastTo<bool>(source);
	            case NPTypeCode.Byte: return CastTo<byte>(source);
	            case NPTypeCode.Int32: return CastTo<int>(source);
	            case NPTypeCode.Int64: return CastTo<long>(source);
	            case NPTypeCode.Single: return CastTo<float>(source);
	            case NPTypeCode.Double: return CastTo<double>(source);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static IMemoryBlock<TOut> CastTo<TOut>(this IMemoryBlock source) where TOut : unmanaged
        {
            switch (source.TypeCode)
            {
#if _REGEN
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1: return CastTo<#2, TOut>(source);
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean: return CastTo<bool, TOut>(source);
	                case NPTypeCode.Byte: return CastTo<byte, TOut>(source);
	                case NPTypeCode.Int32: return CastTo<int, TOut>(source);
	                case NPTypeCode.Int64: return CastTo<long, TOut>(source);
	                case NPTypeCode.Single: return CastTo<float, TOut>(source);
	                case NPTypeCode.Double: return CastTo<double, TOut>(source);
	                default:
		                throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static IMemoryBlock<TOut> CastTo<TIn, TOut>(this IMemoryBlock<TIn> source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedMemoryBlock<TOut>(source.Count);
                var src = source.Address;
                var dst = ret.Address;
                var len = source.Count;
                var convert = Converts.FindConverter<TIn, TOut>();
                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));

                if (source is IArraySlice)
                    return new ArraySlice<TOut>(ret);

                return ret;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static IMemoryBlock<TOut> CastTo<TIn, TOut>(this IMemoryBlock source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var len = source.Count;
                var ret = new UnmanagedMemoryBlock<TOut>(len);

                switch (InfoOf<TIn>.NPTypeCode)
                {
#if _REGEN
                    #region Compute
                    //#n is in, #10n is out
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1:
                    {
                        var src = (#2*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        %foreach supported_dtypes,supported_dtypes_lowercase%
	                        case NPTypeCode.#101:
                            {
                                var dst = (#102*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.To#101(*(src + i)));
                                break;
                            }
	                        %
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                %
	                default:
		                throw new NotSupportedException();
                    #endregion
#else
                    #region Compute
                    //#n is in, #10n is out
	                case NPTypeCode.Boolean:
                    {
                        var src = (bool*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Byte:
                    {
                        var src = (byte*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int32:
                    {
                        var src = (int*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int64:
                    {
                        var src = (long*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Single:
                    {
                        var src = (float*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Double:
                    {
                        var src = (double*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        case NPTypeCode.Boolean:
                            {
                                var dst = (bool*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                default:
		                throw new NotSupportedException();
                    #endregion
#endif
                }

                if (source is IArraySlice)
                    return new ArraySlice<TOut>(ret);

                return ret;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <param name="out"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static unsafe void CastTo(this IMemoryBlock source, IMemoryBlock @out, int? bytesOffset = null, int? countOffset = null)
        {
            var len = source.Count;
            var offset = countOffset ?? ((bytesOffset ?? 0) / @out.ItemLength);
            var ret = @out ?? throw new ArgumentNullException(nameof(@out));
            if (source.BytesLength + bytesOffset > ret.BytesLength)
                throw new ArgumentOutOfRangeException(nameof(@out), "The length of IMemoryBlock @out does not match source's length.");

            switch (source.TypeCode)
            {

#if _REGEN
                #region Compute
                //#n is in, #10n is out
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                {
                    var src = (#2*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    %foreach supported_dtypes,supported_dtypes_lowercase%
	                    case NPTypeCode.#101:
                        {
                            var dst = (#102*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.To#101(*(src + i)));
                            break;
                        }
	                    %
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            %
	            default:
		            throw new NotSupportedException();
                #endregion
#else
                #region Compute
                //#n is in, #10n is out
	            case NPTypeCode.Boolean:
                {
                    var src = (bool*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            case NPTypeCode.Byte:
                {
                    var src = (byte*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            case NPTypeCode.Int32:
                {
                    var src = (int*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            case NPTypeCode.Int64:
                {
                    var src = (long*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            case NPTypeCode.Single:
                {
                    var src = (float*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            case NPTypeCode.Double:
                {
                    var src = (double*)source.Address;
                    switch (ret.TypeCode)
                    {
	                    case NPTypeCode.Boolean:
                        {
                            var dst = (bool*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToBoolean(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Byte:
                        {
                            var dst = (byte*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToByte(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int32:
                        {
                            var dst = (int*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Int64:
                        {
                            var dst = (long*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Single:
                        {
                            var dst = (float*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                            break;
                        }
	                    case NPTypeCode.Double:
                        {
                            var dst = (double*)ret.Address + offset;
                            Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                            break;
                        }
	                    default:
		                    throw new NotSupportedException();
                    }

                    break;
                }
	            default:
		            throw new NotSupportedException();
                #endregion
#endif
            }
        }
    }
}
