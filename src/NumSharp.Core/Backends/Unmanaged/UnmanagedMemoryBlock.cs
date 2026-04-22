using System;
using System.Numerics;

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
	            case NPTypeCode.SByte:
                    return UnmanagedMemoryBlock<sbyte>.FromArray((sbyte[])arr);
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
	            case NPTypeCode.Half:
                    return UnmanagedMemoryBlock<Half>.FromArray((Half[])arr);
	            case NPTypeCode.Double:
                    return UnmanagedMemoryBlock<double>.FromArray((double[])arr);
	            case NPTypeCode.Single:
                    return UnmanagedMemoryBlock<float>.FromArray((float[])arr);
	            case NPTypeCode.Decimal:
                    return UnmanagedMemoryBlock<decimal>.FromArray((decimal[])arr);
	            case NPTypeCode.Complex:
                    return UnmanagedMemoryBlock<Complex>.FromArray((Complex[])arr);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IMemoryBlock Allocate(Type elementType, long count)
        {
            switch (elementType.GetTypeCode())
            {
                case NPTypeCode.Boolean:
                    return new UnmanagedMemoryBlock<bool>(count);
                case NPTypeCode.SByte:
                    return new UnmanagedMemoryBlock<sbyte>(count);
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
                case NPTypeCode.Half:
                    return new UnmanagedMemoryBlock<Half>(count);
                case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count);
                case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count);
                case NPTypeCode.Decimal:
                    return new UnmanagedMemoryBlock<decimal>(count);
                case NPTypeCode.Complex:
                    return new UnmanagedMemoryBlock<Complex>(count);
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        /// Backwards-compatible overload accepting int count.
        /// </summary>
        public static IMemoryBlock Allocate(Type elementType, int count)
            => Allocate(elementType, (long)count);

        public static IMemoryBlock Allocate(Type elementType, long count, object fill)
        {
            // Route through Converts.ToXxx(object) dispatchers — handles all 15 dtypes
            // and cross-type fills (e.g. int -> Half, double -> Complex) with NumPy-parity
            // wrapping semantics. Direct boxing casts like (Half)fill throw InvalidCastException
            // unless `fill` is already the exact target type, which breaks fill=int on Half etc.
            switch (elementType.GetTypeCode())
            {
                case NPTypeCode.Boolean:
                    return new UnmanagedMemoryBlock<bool>(count, Utilities.Converts.ToBoolean(fill));
                case NPTypeCode.SByte:
                    return new UnmanagedMemoryBlock<sbyte>(count, Utilities.Converts.ToSByte(fill));
                case NPTypeCode.Byte:
                    return new UnmanagedMemoryBlock<byte>(count, Utilities.Converts.ToByte(fill));
                case NPTypeCode.Int16:
                    return new UnmanagedMemoryBlock<short>(count, Utilities.Converts.ToInt16(fill));
                case NPTypeCode.UInt16:
                    return new UnmanagedMemoryBlock<ushort>(count, Utilities.Converts.ToUInt16(fill));
                case NPTypeCode.Int32:
                    return new UnmanagedMemoryBlock<int>(count, Utilities.Converts.ToInt32(fill));
                case NPTypeCode.UInt32:
                    return new UnmanagedMemoryBlock<uint>(count, Utilities.Converts.ToUInt32(fill));
                case NPTypeCode.Int64:
                    return new UnmanagedMemoryBlock<long>(count, Utilities.Converts.ToInt64(fill));
                case NPTypeCode.UInt64:
                    return new UnmanagedMemoryBlock<ulong>(count, Utilities.Converts.ToUInt64(fill));
                case NPTypeCode.Char:
                    return new UnmanagedMemoryBlock<char>(count, Utilities.Converts.ToChar(fill));
                case NPTypeCode.Half:
                    return new UnmanagedMemoryBlock<Half>(count, Utilities.Converts.ToHalf(fill));
                case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count, Utilities.Converts.ToDouble(fill));
                case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count, Utilities.Converts.ToSingle(fill));
                case NPTypeCode.Decimal:
                    return new UnmanagedMemoryBlock<decimal>(count, Utilities.Converts.ToDecimal(fill));
                case NPTypeCode.Complex:
                    return new UnmanagedMemoryBlock<Complex>(count, Utilities.Converts.ToComplex(fill));
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        /// Backwards-compatible overload accepting int count.
        /// </summary>
        public static IMemoryBlock Allocate(Type elementType, int count, object fill)
            => Allocate(elementType, (long)count, fill);
    }
}
