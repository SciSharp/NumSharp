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
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                case NPTypeCode.#1:
                    {
                        var src = (#2*)source.Address;
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
	                        %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                        case NPTypeCode.#101:
                            {
                                var dst = (#102*)ret.Address;
                                Func<#2, #102> convert = Convert.To#101;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<bool, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<bool, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<bool, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<bool, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<bool, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<bool, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<bool, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<bool, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<bool, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<bool, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<bool, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<bool, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<byte, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<byte, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<byte, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<byte, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<byte, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<byte, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<byte, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<byte, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<byte, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<byte, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<byte, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<byte, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<short, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<short, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<short, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<short, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<short, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<short, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<short, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<short, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<short, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<short, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<short, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<short, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<ushort, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<ushort, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<ushort, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<ushort, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<ushort, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<ushort, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<ushort, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<ushort, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<ushort, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<ushort, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<ushort, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<ushort, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<int, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<int, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<int, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<int, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<int, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<int, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<int, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<int, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<int, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<int, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<int, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<int, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<uint, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<uint, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<uint, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<uint, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<uint, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<uint, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<uint, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<uint, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<uint, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<uint, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<uint, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<uint, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<long, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<long, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<long, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<long, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<long, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<long, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<long, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<long, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<long, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<long, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<long, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<long, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<ulong, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<ulong, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<ulong, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<ulong, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<ulong, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<ulong, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<ulong, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<ulong, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<ulong, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<ulong, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<ulong, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<ulong, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<char, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<char, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<char, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<char, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<char, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<char, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<char, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<char, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<char, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<char, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<char, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<char, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<double, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<double, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<double, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<double, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<double, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<double, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<double, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<double, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<double, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<double, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<double, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<double, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<float, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<float, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<float, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<float, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<float, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<float, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<float, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<float, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<float, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<float, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<float, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<float, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
                                Func<decimal, bool> convert = Convert.ToBoolean;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Byte:
                            {
                                var dst = (byte*)ret.Address;
                                Func<decimal, byte> convert = Convert.ToByte;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int16:
                            {
                                var dst = (short*)ret.Address;
                                Func<decimal, short> convert = Convert.ToInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt16:
                            {
                                var dst = (ushort*)ret.Address;
                                Func<decimal, ushort> convert = Convert.ToUInt16;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int32:
                            {
                                var dst = (int*)ret.Address;
                                Func<decimal, int> convert = Convert.ToInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt32:
                            {
                                var dst = (uint*)ret.Address;
                                Func<decimal, uint> convert = Convert.ToUInt32;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Int64:
                            {
                                var dst = (long*)ret.Address;
                                Func<decimal, long> convert = Convert.ToInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.UInt64:
                            {
                                var dst = (ulong*)ret.Address;
                                Func<decimal, ulong> convert = Convert.ToUInt64;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Char:
                            {
                                var dst = (char*)ret.Address;
                                Func<decimal, char> convert = Convert.ToChar;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Double:
                            {
                                var dst = (double*)ret.Address;
                                Func<decimal, double> convert = Convert.ToDouble;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Single:
                            {
                                var dst = (float*)ret.Address;
                                Func<decimal, float> convert = Convert.ToSingle;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
                                break;
                            }
	                        case NPTypeCode.Decimal:
                            {
                                var dst = (decimal*)ret.Address;
                                Func<decimal, decimal> convert = Convert.ToDecimal;
                                Parallel.For(0, len, i => *(dst + i) = convert(*(src + i)));
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
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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

                switch (ret.TypeCode)
                {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
