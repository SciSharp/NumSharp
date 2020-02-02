using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using NumSharp.Unmanaged.Memory;

namespace NumSharp.Backends.Unmanaged
{
    public static class ArraySlice
    {
        private static readonly StackedMemoryPool _buffer = ScalarMemoryPool.Instance;

        /// <summary>
        ///     Wrap a <see cref="val"/> inside <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="val">The value to wrap into an arrayslice.</param>
        /// <returns></returns>
        public static IArraySlice Scalar(object val)
        {
            switch (val.GetType().GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#1>(UnmanagedMemoryBlock<#1>.FromPool(_buffer)) {[0] = ((IConvertible)val).To#1(CultureInfo.InvariantCulture)};
	            %
	            default:
		            throw new NotSupportedException();
#else

	            case NPTypeCode.Boolean: return new ArraySlice<Boolean>(UnmanagedMemoryBlock<Boolean>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToBoolean(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Byte: return new ArraySlice<Byte>(UnmanagedMemoryBlock<Byte>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToByte(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Int32: return new ArraySlice<Int32>(UnmanagedMemoryBlock<Int32>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToInt32(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Int64: return new ArraySlice<Int64>(UnmanagedMemoryBlock<Int64>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToInt64(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Single: return new ArraySlice<Single>(UnmanagedMemoryBlock<Single>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToSingle(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Double: return new ArraySlice<Double>(UnmanagedMemoryBlock<Double>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToDouble(CultureInfo.InvariantCulture)};
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="val">The value to wrap into an arrayslice.</param>
        /// <param name="typeCode">The type expected to be returned</param>
        /// <returns></returns>
        public static IArraySlice Scalar(object val, NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#1>(UnmanagedMemoryBlock<#1>.FromPool(_buffer)) {[0] = ((IConvertible)val).To#1(CultureInfo.InvariantCulture)};
	            %
	            default:
		            throw new NotSupportedException();
#else

	            case NPTypeCode.Boolean: return new ArraySlice<Boolean>(UnmanagedMemoryBlock<Boolean>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToBoolean(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Byte: return new ArraySlice<Byte>(UnmanagedMemoryBlock<Byte>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToByte(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Int32: return new ArraySlice<Int32>(UnmanagedMemoryBlock<Int32>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToInt32(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Int64: return new ArraySlice<Int64>(UnmanagedMemoryBlock<Int64>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToInt64(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Single: return new ArraySlice<Single>(UnmanagedMemoryBlock<Single>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToSingle(CultureInfo.InvariantCulture)};
	            case NPTypeCode.Double: return new ArraySlice<Double>(UnmanagedMemoryBlock<Double>.FromPool(_buffer)) {[0] = ((IConvertible)val).ToDouble(CultureInfo.InvariantCulture)};
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static ArraySlice<T> Scalar<T>(T val) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromPool(_buffer)) {[0] = val};
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromArray<T>(T[,,,,,,,,,,,,,,,] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromBuffer<T>(byte[] arr, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromBuffer(arr, copy));
        }

        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArraySlice<T> FromPool<T>(StackedMemoryPool pool) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromPool(pool));
        }

        [MethodImpl((MethodImplOptions)768)]
        public static IArraySlice FromArray(Array arr, bool copy = false)
        {
            var elementType = arr.GetType().GetElementType();

            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(copy ? (#2[])arr.Clone() : (#2[])arr));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(copy ? (bool[])arr.Clone() : (bool[])arr));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(copy ? (byte[])arr.Clone() : (byte[])arr));
	            case NPTypeCode.Int32: return new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(copy ? (int[])arr.Clone() : (int[])arr));
	            case NPTypeCode.Int64: return new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(copy ? (long[])arr.Clone() : (long[])arr));
	            case NPTypeCode.Single: return new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(copy ? (float[])arr.Clone() : (float[])arr));
	            case NPTypeCode.Double: return new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(copy ? (double[])arr.Clone() : (double[])arr));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        public static IArraySlice FromMemoryBlock(IMemoryBlock block, bool copy = false)
        {
            if (block is IArraySlice arr)
                block = arr.MemoryBlock;

            var type = block.GetType();

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(UnmanagedMemoryBlock<>))

                switch (type.GetGenericArguments()[0].GetTypeCode())
                {
#if _REGEN1
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1: return new ArraySlice<#2>(copy ? ((UnmanagedMemoryBlock<#2>)block).Clone() : (UnmanagedMemoryBlock<#2>)block);
	                %
	                default:
		                throw new NotSupportedException();
#else

	                case NPTypeCode.Boolean: return new ArraySlice<bool>(copy ? ((UnmanagedMemoryBlock<bool>)block).Clone() : (UnmanagedMemoryBlock<bool>)block);
	                case NPTypeCode.Byte: return new ArraySlice<byte>(copy ? ((UnmanagedMemoryBlock<byte>)block).Clone() : (UnmanagedMemoryBlock<byte>)block);
	                case NPTypeCode.Int32: return new ArraySlice<int>(copy ? ((UnmanagedMemoryBlock<int>)block).Clone() : (UnmanagedMemoryBlock<int>)block);
	                case NPTypeCode.Int64: return new ArraySlice<long>(copy ? ((UnmanagedMemoryBlock<long>)block).Clone() : (UnmanagedMemoryBlock<long>)block);
	                case NPTypeCode.Single: return new ArraySlice<float>(copy ? ((UnmanagedMemoryBlock<float>)block).Clone() : (UnmanagedMemoryBlock<float>)block);
	                case NPTypeCode.Double: return new ArraySlice<double>(copy ? ((UnmanagedMemoryBlock<double>)block).Clone() : (UnmanagedMemoryBlock<double>)block);
	                default:
		                throw new NotSupportedException();
#endif
                }

            throw new NotSupportedException($"IMemoryBlock of type {block.GetType().Name} is not supported for ArraySlice.FromMemoryBlock(...)");
        }

#if _REGEN1
        %foreach supported_dtypes,supported_dtypes_lowercase%
        public static ArraySlice<#2> FromArray(#2[] #2s, bool copy = false) => new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(#2s, copy));
        %
#else
        public static ArraySlice<bool> FromArray(bool[] bools, bool copy = false) => new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(bools, copy));
        public static ArraySlice<byte> FromArray(byte[] bytes, bool copy = false) => new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(bytes, copy));
        public static ArraySlice<int> FromArray(int[] ints, bool copy = false) => new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(ints, copy));
        public static ArraySlice<long> FromArray(long[] longs, bool copy = false) => new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(longs, copy));
        public static ArraySlice<float> FromArray(float[] floats, bool copy = false) => new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(floats, copy));
        public static ArraySlice<double> FromArray(double[] doubles, bool copy = false) => new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(doubles, copy));
#endif

        public static IArraySlice Allocate(Type elementType, int count, object fill)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, ((IConvertible)fill).To#1(CultureInfo.InvariantCulture)));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, ((IConvertible)fill).ToBoolean(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, ((IConvertible)fill).ToByte(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, ((IConvertible)fill).ToInt32(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, ((IConvertible)fill).ToInt64(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, ((IConvertible)fill).ToSingle(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, ((IConvertible)fill).ToDouble(CultureInfo.InvariantCulture)));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(Type elementType, int count, bool fillDefault)
        {
            if (!fillDefault)
                return Allocate(elementType, count);

            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, default));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, default));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, default));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, default));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, default));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, default));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, default));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(Type elementType, int count)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(NPTypeCode typeCode, int count, object fill)
        {
            switch (typeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, ((IConvertible)fill).To#1(CultureInfo.InvariantCulture)));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, ((IConvertible)fill).ToBoolean(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, ((IConvertible)fill).ToByte(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, ((IConvertible)fill).ToInt32(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, ((IConvertible)fill).ToInt64(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, ((IConvertible)fill).ToSingle(CultureInfo.InvariantCulture)));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, ((IConvertible)fill).ToDouble(CultureInfo.InvariantCulture)));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(NPTypeCode typeCode, int count, bool fillDefault)
        {
            if (!fillDefault)
                return Allocate(typeCode, count);

            switch (typeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, default));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, default));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, default));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, default));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, default));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, default));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, default));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(NPTypeCode typeCode, int count)
        {
            switch (typeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Allocate an array filled filled with <paramref name="fill"/>.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <param name="fill">The item to fill the newly allocated memory with.</param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate<T>(int count, T fill) where T : unmanaged
            => new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count, fill));

        /// <summary>
        ///     Allocate an array filled with default value of <typeparamref name="T"/>.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <param name="fillDefault">Should the newly allocated memory be filled with the default of <typeparamref name="T"/></param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate<T>(int count, bool fillDefault) where T : unmanaged
            => !fillDefault ? Allocate<T>(count) : new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count, default(T)));

        /// <summary>
        ///     Allocate an array filled with noisy memory.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate<T>(int count) where T : unmanaged
            => new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count));

        /// <summary>
        ///     Wrap around a <paramref name="address"/> with given <paramref name="count"/> without claiming ownership of the address.
        /// </summary>
        /// <param name="address">The address at which the memory block starts</param>
        /// <param name="count">The count of items of type <typeparamref name="T"/> (not bytes count)</param>
        /// <returns>A wrapped memory block as <see cref="ArraySlice{T}"/></returns>
        public static unsafe ArraySlice<T> Wrap<T>(T* address, int count) where T : unmanaged => new ArraySlice<T>(new UnmanagedMemoryBlock<T>(address, count));

        /// <summary>
        ///     Wrap around a <paramref name="address"/> with given <paramref name="count"/> without claiming ownership of the address.
        /// </summary>
        /// <param name="address">The address at which the memory block starts</param>
        /// <param name="count">The count of items of type <typeparamref name="T"/> (not bytes count)</param>
        /// <returns>A wrapped memory block as <see cref="ArraySlice{T}"/></returns>
        public static unsafe ArraySlice<T> Wrap<T>(void* address, int count) where T : unmanaged => new ArraySlice<T>(new UnmanagedMemoryBlock<T>((T*)address, count));
    }
}
