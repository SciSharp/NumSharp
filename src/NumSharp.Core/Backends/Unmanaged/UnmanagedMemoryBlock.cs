using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static partial class UnmanagedMemoryBlock
    {
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
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    return UnmanagedMemoryBlock<#2>.FromArray((#2[])arr);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
                    return UnmanagedMemoryBlock<bool>.FromArray((bool[])arr);
	            case NPTypeCode.Byte:
                    return UnmanagedMemoryBlock<byte>.FromArray((byte[])arr);
	            case NPTypeCode.Int16:
                    return UnmanagedMemoryBlock<short>.FromArray((short[])arr);
	            case NPTypeCode.UInt16:
                    return UnmanagedMemoryBlock<ushort>.FromArray((ushort[])arr);
	            case NPTypeCode.Int32:
                    return UnmanagedMemoryBlock<int>.FromArray((int[])arr);
	            case NPTypeCode.UInt32:
                    return UnmanagedMemoryBlock<uint>.FromArray((uint[])arr);
	            case NPTypeCode.Int64:
                    return UnmanagedMemoryBlock<long>.FromArray((long[])arr);
	            case NPTypeCode.UInt64:
                    return UnmanagedMemoryBlock<ulong>.FromArray((ulong[])arr);
	            case NPTypeCode.Char:
                    return UnmanagedMemoryBlock<char>.FromArray((char[])arr);
	            case NPTypeCode.Double:
                    return UnmanagedMemoryBlock<double>.FromArray((double[])arr);
	            case NPTypeCode.Single:
                    return UnmanagedMemoryBlock<float>.FromArray((float[])arr);
	            case NPTypeCode.Decimal:
                    return UnmanagedMemoryBlock<decimal>.FromArray((decimal[])arr);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IMemoryBlock Allocate(Type elementType, int count)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    return new UnmanagedMemoryBlock<#2>(count);
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                    return new UnmanagedMemoryBlock<byte>(count);
                case NPTypeCode.Int16:
                    return new UnmanagedMemoryBlock<short>(count);
                case NPTypeCode.UInt16:
                    return new UnmanagedMemoryBlock<ushort>(count);
                case NPTypeCode.Int32:
                    return new UnmanagedMemoryBlock<int>(count);
                case NPTypeCode.UInt32:
                    return new UnmanagedMemoryBlock<uint>(count);
                case NPTypeCode.Int64:
                    return new UnmanagedMemoryBlock<long>(count);
                case NPTypeCode.UInt64:
                    return new UnmanagedMemoryBlock<ulong>(count);
                case NPTypeCode.Char:
                    return new UnmanagedMemoryBlock<char>(count);
                case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count);
                case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count);
                case NPTypeCode.Decimal:
                    return new UnmanagedMemoryBlock<decimal>(count);
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        public static IMemoryBlock Allocate(Type elementType, int count, object fill)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    return new UnmanagedMemoryBlock<#2>(count, (#2)fill);
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                    return new UnmanagedMemoryBlock<byte>(count, (byte)fill);
                case NPTypeCode.Int16:
                    return new UnmanagedMemoryBlock<short>(count, (short)fill);
                case NPTypeCode.UInt16:
                    return new UnmanagedMemoryBlock<ushort>(count, (ushort)fill);
                case NPTypeCode.Int32:
                    return new UnmanagedMemoryBlock<int>(count, (int)fill);
                case NPTypeCode.UInt32:
                    return new UnmanagedMemoryBlock<uint>(count, (uint)fill);
                case NPTypeCode.Int64:
                    return new UnmanagedMemoryBlock<long>(count, (long)fill);
                case NPTypeCode.UInt64:
                    return new UnmanagedMemoryBlock<ulong>(count, (ulong)fill);
                case NPTypeCode.Char:
                    return new UnmanagedMemoryBlock<char>(count, (char)fill);
                case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count, (double)fill);
                case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count, (float)fill);
                case NPTypeCode.Decimal:
                    return new UnmanagedMemoryBlock<decimal>(count, (decimal)fill);
                default:
                    throw new NotSupportedException();
#endif
            }
        }
    }
}
