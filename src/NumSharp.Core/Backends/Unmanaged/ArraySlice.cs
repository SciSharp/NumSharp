using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static class ArraySlice
    {
        private static readonly InternalBufferManager.PooledBufferManager _buffer = ScalarMemoryPool.Instance;

        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="UnmanagedByteStorage{T}"/>.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static IArraySlice Scalar(object val)
        {
            switch (val.GetType().GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#1>(UnmanagedMemoryBlock<#1>.FromPool(1, _buffer));
	            %
	            default:
		            throw new NotSupportedException();
#else

	            case NPTypeCode.Boolean: return new ArraySlice<Boolean>(UnmanagedMemoryBlock<Boolean>.FromPool(1, _buffer));
	            case NPTypeCode.Byte: return new ArraySlice<Byte>(UnmanagedMemoryBlock<Byte>.FromPool(1, _buffer));
	            case NPTypeCode.Int16: return new ArraySlice<Int16>(UnmanagedMemoryBlock<Int16>.FromPool(1, _buffer));
	            case NPTypeCode.UInt16: return new ArraySlice<UInt16>(UnmanagedMemoryBlock<UInt16>.FromPool(1, _buffer));
	            case NPTypeCode.Int32: return new ArraySlice<Int32>(UnmanagedMemoryBlock<Int32>.FromPool(1, _buffer));
	            case NPTypeCode.UInt32: return new ArraySlice<UInt32>(UnmanagedMemoryBlock<UInt32>.FromPool(1, _buffer));
	            case NPTypeCode.Int64: return new ArraySlice<Int64>(UnmanagedMemoryBlock<Int64>.FromPool(1, _buffer));
	            case NPTypeCode.UInt64: return new ArraySlice<UInt64>(UnmanagedMemoryBlock<UInt64>.FromPool(1, _buffer));
	            case NPTypeCode.Char: return new ArraySlice<Char>(UnmanagedMemoryBlock<Char>.FromPool(1, _buffer));
	            case NPTypeCode.Double: return new ArraySlice<Double>(UnmanagedMemoryBlock<Double>.FromPool(1, _buffer));
	            case NPTypeCode.Single: return new ArraySlice<Single>(UnmanagedMemoryBlock<Single>.FromPool(1, _buffer));
	            case NPTypeCode.Decimal: return new ArraySlice<Decimal>(UnmanagedMemoryBlock<Decimal>.FromPool(1, _buffer));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="UnmanagedByteStorage{T}"/>.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static ArraySlice<T> Scalar<T>(T val) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromPool(1, _buffer));
        }

        public static ArraySlice<T> FromArray<T>(T[] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(array, copy));
        }


        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromBuffer<T>(byte[] arr, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromBuffer(arr, copy));
        }

        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromPool<T>(int count, InternalBufferManager pool) where T : unmanaged
        {
            //TODO! Upgrade InternalBufferManager to use pre-pinned arrays.
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromPool(count, pool));
        }

        public static IArraySlice FromArray(Array arr, bool copy = false)
        {
            var elementType = arr.GetType().GetElementType();

            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(copy ? (#2[])arr.Clone() : (#2[])arr));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(copy ? (bool[])arr.Clone() : (bool[])arr));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(copy ? (byte[])arr.Clone() : (byte[])arr));
	            case NPTypeCode.Int16: return new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromArray(copy ? (short[])arr.Clone() : (short[])arr));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromArray(copy ? (ushort[])arr.Clone() : (ushort[])arr));
	            case NPTypeCode.Int32: return new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(copy ? (int[])arr.Clone() : (int[])arr));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromArray(copy ? (uint[])arr.Clone() : (uint[])arr));
	            case NPTypeCode.Int64: return new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(copy ? (long[])arr.Clone() : (long[])arr));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromArray(copy ? (ulong[])arr.Clone() : (ulong[])arr));
	            case NPTypeCode.Char: return new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromArray(copy ? (char[])arr.Clone() : (char[])arr));
	            case NPTypeCode.Double: return new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(copy ? (double[])arr.Clone() : (double[])arr));
	            case NPTypeCode.Single: return new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(copy ? (float[])arr.Clone() : (float[])arr));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromArray(copy ? (decimal[])arr.Clone() : (decimal[])arr));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice FromMemoryBlock(IMemoryBlock block, bool copy = false)
        {
            var type = block.GetType();
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(UnmanagedMemoryBlock<>))

                switch (type.GetGenericArguments()[0].GetTypeCode())
                {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                case NPTypeCode.#1: return new ArraySlice<#2>(copy ? ((UnmanagedMemoryBlock<#2>)block).Clone() : (UnmanagedMemoryBlock<#2>)block);
	                %
	                default:
		                throw new NotSupportedException();
#else

	                case NPTypeCode.Boolean: return new ArraySlice<bool>(copy ? ((UnmanagedMemoryBlock<bool>)block).Clone() : (UnmanagedMemoryBlock<bool>)block);
	                case NPTypeCode.Byte: return new ArraySlice<byte>(copy ? ((UnmanagedMemoryBlock<byte>)block).Clone() : (UnmanagedMemoryBlock<byte>)block);
	                case NPTypeCode.Int16: return new ArraySlice<short>(copy ? ((UnmanagedMemoryBlock<short>)block).Clone() : (UnmanagedMemoryBlock<short>)block);
	                case NPTypeCode.UInt16: return new ArraySlice<ushort>(copy ? ((UnmanagedMemoryBlock<ushort>)block).Clone() : (UnmanagedMemoryBlock<ushort>)block);
	                case NPTypeCode.Int32: return new ArraySlice<int>(copy ? ((UnmanagedMemoryBlock<int>)block).Clone() : (UnmanagedMemoryBlock<int>)block);
	                case NPTypeCode.UInt32: return new ArraySlice<uint>(copy ? ((UnmanagedMemoryBlock<uint>)block).Clone() : (UnmanagedMemoryBlock<uint>)block);
	                case NPTypeCode.Int64: return new ArraySlice<long>(copy ? ((UnmanagedMemoryBlock<long>)block).Clone() : (UnmanagedMemoryBlock<long>)block);
	                case NPTypeCode.UInt64: return new ArraySlice<ulong>(copy ? ((UnmanagedMemoryBlock<ulong>)block).Clone() : (UnmanagedMemoryBlock<ulong>)block);
	                case NPTypeCode.Char: return new ArraySlice<char>(copy ? ((UnmanagedMemoryBlock<char>)block).Clone() : (UnmanagedMemoryBlock<char>)block);
	                case NPTypeCode.Double: return new ArraySlice<double>(copy ? ((UnmanagedMemoryBlock<double>)block).Clone() : (UnmanagedMemoryBlock<double>)block);
	                case NPTypeCode.Single: return new ArraySlice<float>(copy ? ((UnmanagedMemoryBlock<float>)block).Clone() : (UnmanagedMemoryBlock<float>)block);
	                case NPTypeCode.Decimal: return new ArraySlice<decimal>(copy ? ((UnmanagedMemoryBlock<decimal>)block).Clone() : (UnmanagedMemoryBlock<decimal>)block);
	                default:
		                throw new NotSupportedException();
#endif
                }

            throw new NotSupportedException($"IMemoryBlock of type {block.GetType().Name} is not supported for ArraySlice.FromMemoryBlock(...)");
        }

#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        public static ArraySlice<#2> FromArray(#2[] #2s, bool copy = false) => new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(#2s, copy));
        %
#else
        public static ArraySlice<bool> FromArray(bool[] bools, bool copy = false) => new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(bools, copy));
        public static ArraySlice<byte> FromArray(byte[] bytes, bool copy = false) => new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(bytes, copy));
        public static ArraySlice<short> FromArray(short[] shorts, bool copy = false) => new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromArray(shorts, copy));
        public static ArraySlice<ushort> FromArray(ushort[] ushorts, bool copy = false) => new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromArray(ushorts, copy));
        public static ArraySlice<int> FromArray(int[] ints, bool copy = false) => new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(ints, copy));
        public static ArraySlice<uint> FromArray(uint[] uints, bool copy = false) => new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromArray(uints, copy));
        public static ArraySlice<long> FromArray(long[] longs, bool copy = false) => new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(longs, copy));
        public static ArraySlice<ulong> FromArray(ulong[] ulongs, bool copy = false) => new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromArray(ulongs, copy));
        public static ArraySlice<char> FromArray(char[] chars, bool copy = false) => new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromArray(chars, copy));
        public static ArraySlice<double> FromArray(double[] doubles, bool copy = false) => new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(doubles, copy));
        public static ArraySlice<float> FromArray(float[] floats, bool copy = false) => new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(floats, copy));
        public static ArraySlice<decimal> FromArray(decimal[] decimals, bool copy = false) => new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromArray(decimals, copy));
#endif

        public static IArraySlice Allocate(Type elementType, int count, object fill)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, (#2)fill));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, (bool)fill));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, (byte)fill));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count, (short)fill));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count, (ushort)fill));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, (int)fill));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count, (uint)fill));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, (long)fill));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count, (ulong)fill));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count, (char)fill));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, (double)fill));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, (float)fill));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count, (decimal)fill));
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
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, default));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, default));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, default));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count, default));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count, default));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, default));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count, default));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, default));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count, default));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count, default));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, default));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, default));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count, default));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(Type elementType, int count)
        {
            switch (elementType.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(NPTypeCode typeCode, int count, object fill)
        {
            switch (typeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, (#2)fill));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, (bool)fill));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, (byte)fill));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count, (short)fill));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count, (ushort)fill));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, (int)fill));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count, (uint)fill));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, (long)fill));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count, (ulong)fill));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count, (char)fill));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, (double)fill));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, (float)fill));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count, (decimal)fill));
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
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count, default));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, default));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count, default));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count, default));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count, default));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count, default));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count, default));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count, default));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count, default));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count, default));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count, default));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count, default));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count, default));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static IArraySlice Allocate(NPTypeCode typeCode, int count)
        {
            switch (typeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(count));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count));
	            case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(count));
	            case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>(count));
	            case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(count));
	            case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>(count));
	            case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(count));
	            case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>(count));
	            case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(count));
	            case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>(count));
	            case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>(count));
	            case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>(count));
	            case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(count));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }

    /// <summary>
    ///     <see cref="ArraySlice{T}"/> is similar to <see cref="Span{T}"/> but it can be moved around without having to follow `ref struct` rules.
    /// </summary>
    /// <typeparam name="T">The type that the <see cref="MemoryBlock"/> implements.</typeparam>
    public readonly unsafe struct ArraySlice<T> : IArraySlice, IMemoryBlock<T>, IEnumerable<T> where T : unmanaged
    {
        public static NPTypeCode TypeCode { get; } = InfoOf<T>.NPTypeCode;
        private static readonly InternalBufferManager.PooledBufferManager _buffer = ScalarMemoryPool.Instance;

        /// <summary>
        ///     The memory block this <see cref="ArraySlice{T}"/> is stored in.
        /// </summary>
        /// <remarks>If <see cref="IsSlice"/> is false then this slice represents the entire MemoryBlock.</remarks>
        public readonly UnmanagedMemoryBlock<T> MemoryBlock;

        public readonly T* Address;
        public readonly void* VoidAddress;

        public readonly int Count;

        /// <summary>
        ///     Is this <see cref="ArraySlice{T}"/> a smaller part/slice of an unmanaged allocation?
        /// </summary>
        public readonly bool IsSlice;

        /// A Span representing this slice.
        public Span<T> AsSpan
        {
            [MethodImpl((MethodImplOptions)768)] get => new Span<T>(Address, Count);
        }

        public int ItemLength
        {
            [MethodImpl((MethodImplOptions)768)] get => InfoOf<T>.Size;
        }

        #region Construction

        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock)
        {
            MemoryBlock = memoryBlock;
            IsSlice = false;
            VoidAddress = Address = MemoryBlock.Address;
            Count = MemoryBlock.Count;
        }

        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, Span<T> slice)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = slice.Length;
            VoidAddress = Address = (T*)Unsafe.AsPointer(ref slice.GetPinnableReference());
        }

        /// <summary>
        ///     Creates a sliced <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="memoryBlock"></param>
        /// <param name="offset">The offset in <typeparamref name="T"/> and not bytes - relative to the <paramref name="memoryBlock"/>.</param>
        /// <param name="count">The number of <typeparamref name="T"/> this slice should contain - relative to the <paramref name="memoryBlock"/></param>
        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, int offset, int count)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = count;
            VoidAddress = Address = (memoryBlock.Address + offset);
            //TODO! we should check that is does not exceed bounds.
        }

        #endregion

        #region Accessing

        /// A Span representing this slice.
        /// <remarks>Does not perform copy.</remarks>
        Span<T1> IArraySlice.AsSpan<T1>()
        {
            return new Span<T1>(VoidAddress, Count);
        }

        /// <param name="index"></param>
        /// <returns></returns>
        object IArraySlice.this[int index]
        {
            get => *(Address + index);
            set => *(Address + index) = (T)value;
        }

        /// <param name="index"></param>
        /// <returns></returns>
        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(Address + index);
            [MethodImpl((MethodImplOptions)768)] set => *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetIndex(int index)
        {
            return *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, object value)
        {
            *(Address + index) = (T)value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, T value)
        {
            *(Address + index) = value;
        }

        #region Implicit Interface

        [MethodImpl((MethodImplOptions)768)]
        TRet IArraySlice.GetIndex<TRet>(int index)
        {
            Debug.Assert(InfoOf<TRet>.Size == InfoOf<T>.Size);
            return *((TRet*)VoidAddress + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        void IArraySlice.SetIndex<TVal>(int index, TVal value)
        {
            Debug.Assert(InfoOf<TVal>.Size == InfoOf<T>.Size);
            *((TVal*)VoidAddress + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        object IArraySlice.GetIndex(int index)
        {
            return *(Address + index);
        }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        T* IMemoryBlock<T>.Address => Address;

        #endregion

        #endregion

        [MethodImpl((MethodImplOptions)768)]
        public bool Contains(T item)
        {
            int len = Count;
            for (var i = 0; i < len; i++) //TODO! Parallel.For?
            {
                if ((*(Address + i)).Equals(item)) return true;
            }

            return false;
        }

        /// <param name="value"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value)
        {
            AsSpan.Fill(value);
        }

        /// <summary>
        ///     Fills all indexes with <paramref name="value"/>.
        /// </summary>
        /// <param name="value"></param>
        void IArraySlice.Fill(object value)
        {
            Fill((T)value);
        }

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice IArraySlice.Slice(int start)
        {
            return Slice(start);
        }

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <param name="count">The number of items to slice (not bytes)</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice IArraySlice.Slice(int start, int count)
        {
            return Slice(start, count);
        }

        /// <param name="destination"></param>
        void IArraySlice.CopyTo<T1>(Span<T1> destination)
        {
            new Span<T1>(Address, Count).CopyTo(destination);
        }

        /// <summary>
        ///     Gets pinnable reference of the first item in the memory block storage.
        /// </summary>
        ref T1 IArraySlice.GetPinnableReference<T1>()
        {
            return ref (*(T1*)VoidAddress);
        }
        
        /// <param name="start"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start)
        {
            return new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start));
        }

        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start, int length)
        {
            return new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start, length));
        }

        /// <param name="destination"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination)
        {
            AsSpan.CopyTo(destination);
        }

        [EditorBrowsable(EditorBrowsableState.Never)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference()
        {
            return ref Unsafe.AsRef<T>(Address);
        }
        ArraySlice<T1> IArraySlice.Clone<T1>()
        {
            return new ArraySlice<T1>(UnmanagedMemoryBlock<T1>.Copy(Address, Count));
        }

        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Clone()
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));
        }

        [MethodImpl((MethodImplOptions)768)]
        IArraySlice IArraySlice.Clone()
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));
        }

        #region Explicit Interfaces

        object ICloneable.Clone()
        {
            return new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));
        }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe void* IMemoryBlock.Address
        {
            [MethodImpl((MethodImplOptions)768)] get => VoidAddress;
        }

        /// <summary>
        ///     How many items are stored in <see cref="IMemoryBlock.Address"/>?
        /// </summary>
        /// <remarks></remarks>
        int IMemoryBlock.Count
        {
            [MethodImpl((MethodImplOptions)768)] get => Count;
        }

        /// <summary>
        ///     The items with length of <see cref="IMemoryBlock.TypeCode"/> are present in <see cref="IMemoryBlock.Address"/>.
        /// </summary>
        /// <remarks>Calculated by <see cref="IMemoryBlock.Count"/>*<see cref="IMemoryBlock.ItemLength"/></remarks>
        int IMemoryBlock.BytesLength
        {
            [MethodImpl((MethodImplOptions)768)] get => InfoOf<T>.Size;
        }

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of the type stored inside this memory block.
        /// </summary>
        NPTypeCode IMemoryBlock.TypeCode
        {
            [MethodImpl((MethodImplOptions)768)] get => TypeCode;
        }

        IMemoryBlock IArraySlice.MemoryBlock
        {
            [MethodImpl((MethodImplOptions)768)] get => MemoryBlock;
        }

        #endregion

        /// <summary>
        ///     Performs dispose on the internal unmanaged memory block.<br></br>
        /// </summary>
        /// <remarks>
        ///     Dangerous because this <see cref="ArraySlice"/> might be a <see cref="IsSlice"/> therefore there might be other slices that point to current <see cref="MemoryBlock"/>.<br></br>
        ///     So releasing the <see cref="MemoryBlock"/> might cause memory corruption elsewhere.<br></br>
        ///     It is best to leave MemoryBlock to GC.
        /// </remarks>
        public void DangerousFree()
        {
            // ReSharper disable once ImpureMethodCallOnReadonlyValueField
            MemoryBlock.Free();
        }

        /// <summary>
        ///     Copies this <see cref="IArraySlice"/> contents into a new array.
        /// </summary>
        /// <returns></returns>
        Array IArraySlice.ToArray()
        {
            return ToArray();
        }

        [MethodImpl((MethodImplOptions)768)]
        public T[] ToArray()
        {
            return AsSpan.ToArray();
        }

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            return _enumerate().GetEnumerator();
        }

        private IEnumerable<T> _enumerate()
        {
            var len = this.Count;
            for (int i = 0; i < len; i++)
            {
                yield return this[i];
            }
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator"></see> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
