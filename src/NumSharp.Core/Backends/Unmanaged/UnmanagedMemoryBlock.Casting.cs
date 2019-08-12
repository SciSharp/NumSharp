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
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
	                return Cast<#2>(source);
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Boolean:
                    return CastTo<bool>(source);
                case NPTypeCode.Byte:
                    return CastTo<byte>(source);
                case NPTypeCode.Int16:
                    return CastTo<short>(source);
                case NPTypeCode.UInt16:
                    return CastTo<ushort>(source);
                case NPTypeCode.Int32:
                    return CastTo<int>(source);
                case NPTypeCode.UInt32:
                    return CastTo<uint>(source);
                case NPTypeCode.Int64:
                    return CastTo<long>(source);
                case NPTypeCode.UInt64:
                    return CastTo<ulong>(source);
                case NPTypeCode.Char:
                    return CastTo<char>(source);
                case NPTypeCode.Double:
                    return CastTo<double>(source);
                case NPTypeCode.Single:
                    return CastTo<float>(source);
                case NPTypeCode.Decimal:
                    return CastTo<decimal>(source);
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
	                case NPTypeCode.#1: return Cast<#2, TOut>(source);
	                %
	                default:
		                throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return CastTo<bool, TOut>(source);
                case NPTypeCode.Byte: return CastTo<byte, TOut>(source);
                case NPTypeCode.Int16: return CastTo<short, TOut>(source);
                case NPTypeCode.UInt16: return CastTo<ushort, TOut>(source);
                case NPTypeCode.Int32: return CastTo<int, TOut>(source);
                case NPTypeCode.UInt32: return CastTo<uint, TOut>(source);
                case NPTypeCode.Int64: return CastTo<long, TOut>(source);
                case NPTypeCode.UInt64: return CastTo<ulong, TOut>(source);
                case NPTypeCode.Char: return CastTo<char, TOut>(source);
                case NPTypeCode.Double: return CastTo<double, TOut>(source);
                case NPTypeCode.Single: return CastTo<float, TOut>(source);
                case NPTypeCode.Decimal: return CastTo<decimal, TOut>(source);
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int16:
                    {
                        var src = (short*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt16:
                    {
                        var src = (ushort*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt32:
                    {
                        var src = (uint*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt64:
                    {
                        var src = (ulong*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Char:
                    {
                        var src = (char*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
                                break;
                            }
	                        default:
		                        throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Decimal:
                    {
                        var src = (decimal*)source.Address;
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
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt16(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt32(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToUInt64(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToChar(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDouble(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToSingle(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Parallel.For(0, len, i => *(dst + i) = Converts.ToDecimal(*(src + i)));
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
            //TODO! wth is going on here, the offsetting is not used below.
#if __REGEN
            //TODO* Nested regen code:
            switch (ret.TypeCode)
            {
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                {
                    var src = (#2*)source.Address;
		            for (int i = 0; i < len; i++)
                        *(dst + offset + i) = Convert.To___(*(src + i));
                    break;
                }
	            %
	            default:
		            throw new NotSupportedException();
            }
#else
#endif
            if (offset != 0) 
                switch (ret.TypeCode)
                {
#if _REGEN
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1:
                    {
                        var dst = (#2*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean:
                    {
                        var dst = (bool*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Byte:
                    {
                        var dst = (byte*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int16:
                    {
                        var dst = (short*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt16:
                    {
                        var dst = (ushort*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int32:
                    {
                        var dst = (int*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt32:
                    {
                        var dst = (uint*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int64:
                    {
                        var dst = (long*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt64:
                    {
                        var dst = (ulong*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Char:
                    {
                        var dst = (char*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Double:
                    {
                        var dst = (double*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Single:
                    {
                        var dst = (float*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Decimal:
                    {
                        var dst = (decimal*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + offset + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                default:
		                throw new NotSupportedException();
#endif
                }
            else
            {
#if __REGEN
                //TODO* Nested regen code:
                switch (ret.TypeCode)
                {
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1:
                    {
                        var src = (#2*)source.Address;
		                for (int i = 0; i < len; i++)
                            *(dst + i) = Convert.To___(*(src + i));
                        break;
                    }
	                %
	                default:
		                throw new NotSupportedException();
                }
#else
#endif

                switch (ret.TypeCode)
                {
#if _REGEN
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1:
                    {
                        var dst = (#2*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.To#1(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean:
                    {
                        var dst = (bool*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToBoolean(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Byte:
                    {
                        var dst = (byte*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToByte(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int16:
                    {
                        var dst = (short*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt16(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt16:
                    {
                        var dst = (ushort*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt16(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int32:
                    {
                        var dst = (int*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt32(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt32:
                    {
                        var dst = (uint*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt32(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Int64:
                    {
                        var dst = (long*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToInt64(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.UInt64:
                    {
                        var dst = (ulong*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToUInt64(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Char:
                    {
                        var dst = (char*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToChar(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Double:
                    {
                        var dst = (double*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDouble(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Single:
                    {
                        var dst = (float*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToSingle(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                case NPTypeCode.Decimal:
                    {
                        var dst = (decimal*)ret.Address;
                        switch (source.TypeCode)
                        {
                            case NPTypeCode.Boolean:
                            {
                                var src = (bool*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }                            
                            case NPTypeCode.Byte:
                            {
                                var src = (byte*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int16:
                            {
                                var src = (short*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt16:
                            {
                                var src = (ushort*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int32:
                            {
                                var src = (int*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt32:
                            {
                                var src = (uint*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Int64:
                            {
                                var src = (long*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.UInt64:
                            {
                                var src = (ulong*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Char:
                            {
                                var src = (char*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Double:
                            {
                                var src = (double*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Single:
                            {
                                var src = (float*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            case NPTypeCode.Decimal:
                            {
                                var src = (decimal*)source.Address;
                                for (int i = 0; i < len; i++)
                                    *(dst + i) = Convert.ToDecimal(*(src + i));
                                break;
                            }

                            default:
                                throw new NotSupportedException();
                        }

                        break;
                    }
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
