using System;

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
#if _REGEN1
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
	            case NPTypeCode.Int32:
                    return UnmanagedMemoryBlock<int>.FromArray((int[])arr);
	            case NPTypeCode.Int64:
                    return UnmanagedMemoryBlock<long>.FromArray((long[])arr);
	            case NPTypeCode.Single:
                    return UnmanagedMemoryBlock<float>.FromArray((float[])arr);
	            case NPTypeCode.Double:
                    return UnmanagedMemoryBlock<double>.FromArray((double[])arr);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IMemoryBlock Allocate(Type elementType, int count)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    return new UnmanagedMemoryBlock<#2>(count);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
                    return new UnmanagedMemoryBlock<bool>(count);
	            case NPTypeCode.Byte:
                    return new UnmanagedMemoryBlock<byte>(count);
	            case NPTypeCode.Int32:
                    return new UnmanagedMemoryBlock<int>(count);
	            case NPTypeCode.Int64:
                    return new UnmanagedMemoryBlock<long>(count);
	            case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count);
	            case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IMemoryBlock Allocate(Type elementType, int count, object fill)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    return new UnmanagedMemoryBlock<#2>(count, (#2)fill);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
                    return new UnmanagedMemoryBlock<bool>(count, (bool)fill);
	            case NPTypeCode.Byte:
                    return new UnmanagedMemoryBlock<byte>(count, (byte)fill);
	            case NPTypeCode.Int32:
                    return new UnmanagedMemoryBlock<int>(count, (int)fill);
	            case NPTypeCode.Int64:
                    return new UnmanagedMemoryBlock<long>(count, (long)fill);
	            case NPTypeCode.Single:
                    return new UnmanagedMemoryBlock<float>(count, (float)fill);
	            case NPTypeCode.Double:
                    return new UnmanagedMemoryBlock<double>(count, (double)fill);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
