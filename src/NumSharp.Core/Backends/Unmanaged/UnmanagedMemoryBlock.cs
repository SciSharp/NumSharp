using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Unmanaged
{
    public static class UnmanagedMemoryBlock
    {
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
                } //TODO! seperate class for NP!

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
                var src = (TIn*)source.Address;
                var dst = ret.Address;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType(*(src + i), tc);
                } //TODO! seperate class for NP!

                return ret;
            }
        }

        public static IMemoryBlock FromArray(Array arr, bool copy, Type elementType = null)
        {
            if (elementType == null || elementType.IsArray)
            {
                elementType = arr.GetType().GetElementType();

                // ReSharper disable once PossibleNullReferenceException
                while (elementType.IsArray)
                    elementType = elementType.GetElementType();
            }

            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
                    return UnmanagedMemoryBlock<#2>.FromArray((#2[])arr);
	            }

	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return UnmanagedMemoryBlock<byte>.FromArray((byte[])arr);
                }

                case NPTypeCode.Int16:
                {
                    return UnmanagedMemoryBlock<short>.FromArray((short[])arr);
                }

                case NPTypeCode.UInt16:
                {
                    return UnmanagedMemoryBlock<ushort>.FromArray((ushort[])arr);
                }

                case NPTypeCode.Int32:
                {
                    return UnmanagedMemoryBlock<int>.FromArray((int[])arr);
                }

                case NPTypeCode.UInt32:
                {
                    return UnmanagedMemoryBlock<uint>.FromArray((uint[])arr);
                }

                case NPTypeCode.Int64:
                {
                    return UnmanagedMemoryBlock<long>.FromArray((long[])arr);
                }

                case NPTypeCode.UInt64:
                {
                    return UnmanagedMemoryBlock<ulong>.FromArray((ulong[])arr);
                }

                case NPTypeCode.Char:
                {
                    return UnmanagedMemoryBlock<char>.FromArray((char[])arr);
                }

                case NPTypeCode.Double:
                {
                    return UnmanagedMemoryBlock<double>.FromArray((double[])arr);
                }

                case NPTypeCode.Single:
                {
                    return UnmanagedMemoryBlock<float>.FromArray((float[])arr);
                }

                case NPTypeCode.Decimal:
                {
                    return UnmanagedMemoryBlock<decimal>.FromArray((decimal[])arr);
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }
    }
}
