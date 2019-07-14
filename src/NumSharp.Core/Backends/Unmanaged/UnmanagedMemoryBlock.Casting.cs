using System;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static partial class UnmanagedMemoryBlock
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static IMemoryBlock Cast(this IMemoryBlock source, NPTypeCode to)
        {
            switch (to)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	                return Cast<#2>(source);
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                    return Cast<byte>(source);
                case NPTypeCode.Int16:
                    return Cast<short>(source);
                case NPTypeCode.UInt16:
                    return Cast<ushort>(source);
                case NPTypeCode.Int32:
                    return Cast<int>(source);
                case NPTypeCode.UInt32:
                    return Cast<uint>(source);
                case NPTypeCode.Int64:
                    return Cast<long>(source);
                case NPTypeCode.UInt64:
                    return Cast<ulong>(source);
                case NPTypeCode.Char:
                    return Cast<char>(source);
                case NPTypeCode.Double:
                    return Cast<double>(source);
                case NPTypeCode.Single:
                    return Cast<float>(source);
                case NPTypeCode.Decimal:
                    return Cast<decimal>(source);
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
        public static IMemoryBlock Cast<TOut>(this IMemoryBlock source) where TOut : unmanaged
        {
            switch (source.TypeCode)
            {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                case NPTypeCode.#1: return Cast<#2, TOut>(source);
	                %
	                default:
		                throw new NotSupportedException();
#else
                case NPTypeCode.Byte: return Cast<byte, TOut>(source);
                case NPTypeCode.Int16: return Cast<short, TOut>(source);
                case NPTypeCode.UInt16: return Cast<ushort, TOut>(source);
                case NPTypeCode.Int32: return Cast<int, TOut>(source);
                case NPTypeCode.UInt32: return Cast<uint, TOut>(source);
                case NPTypeCode.Int64: return Cast<long, TOut>(source);
                case NPTypeCode.UInt64: return Cast<ulong, TOut>(source);
                case NPTypeCode.Char: return Cast<char, TOut>(source);
                case NPTypeCode.Double: return Cast<double, TOut>(source);
                case NPTypeCode.Single: return Cast<float, TOut>(source);
                case NPTypeCode.Decimal: return Cast<decimal, TOut>(source);
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
        public static UnmanagedMemoryBlock<TOut> Cast<TIn, TOut>(this UnmanagedMemoryBlock<TIn> source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedMemoryBlock<TOut>(source.Count);
                var src = source.Address;
                var dst = ret.Address;
                var len = source.Count;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType((object)*(src + i), tc);
                } 

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
        public static UnmanagedMemoryBlock<TOut> Cast<TIn, TOut>(this IMemoryBlock source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var len = source.Count;
                var ret = new UnmanagedMemoryBlock<TOut>(len);

#if __REGEN
                //TODO* Nested regen code:
                switch (InfoOf<TOut>.NPTypeCode)
                {
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
                
                switch (InfoOf<TOut>.NPTypeCode)
                {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                case NPTypeCode.#1:
                    {
                        var dst = (#2*)ret.Address;
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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
                        switch (InfoOf<TIn>.NPTypeCode)
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

                return ret;
            }
        }
    }
}
