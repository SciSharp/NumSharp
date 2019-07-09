using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OOMath.MemoryPooling;

namespace NumSharp.Backends.Unmanaged
{
    public static class UnmanagedArray
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<TOut> Cast<TIn, TOut>(UnmanagedArray<TIn> source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedArray<TOut>(source._itemCounts);
                var src = source._itemBuffer;
                var dst = ret._itemBuffer;
                var len = source.Count;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType((object) *(src + i), tc);
                }

                return ret;
            }
        }
    }

    public interface IUnmanagedArray : ICollection
    {
        unsafe void* _itemBuffer { get; }
        int _itemCounts { get; }
    }

    public delegate void DisposeUnmanaged<T>(ref T addr, int length);

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct UnmanagedArray<T> : IEnumerable<T>, IEquatable<UnmanagedArray<T>>, ICollection<T>, IUnmanagedArray where T : unmanaged
    {
        const int AllocHGlobalAfter = int.MaxValue; //530

        public int _itemCounts;
        private Action _dispose;
        public T* _itemBuffer;
        public GCHandle? _gcHandle;

        [MethodImpl((MethodImplOptions)512)]
        public UnmanagedArray(int length)
        {
            var bytes = length * sizeof(T);
            if (false)
            {
                //bytes > AllocHGlobalAfter
                _itemBuffer = (T*)Marshal.AllocHGlobal(bytes).ToPointer();
                _gcHandle = null;
            }
            else
            {
                var handle = GCHandle.Alloc(new T[length], GCHandleType.Pinned);
                _gcHandle = handle;
                _itemBuffer = (T*)handle.AddrOfPinnedObject();
            }

            _dispose = null;
            _itemCounts = length;
        }

        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedArray(T* start, int length, Action dispose)
        {
            _itemCounts = length;
            _dispose = dispose;
            _itemBuffer = start;
            _gcHandle = null;
        }

        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedArray(int length, T fill) : this(length)
        {
            new Span<T>(_itemBuffer, length).Fill(fill);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> FromArray(T[] arr, bool copy)
        {
            if (!copy)
                return FromArray(arr);

            var ret = new UnmanagedArray<T>(arr.Length);
            new Span<T>(arr).CopyTo(ret.AsSpan());

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> FromArray(T[] arr)
        {
            var ret = new UnmanagedArray<T>();
            var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            ret._gcHandle = handle;
            ret._itemBuffer = (T*)handle.AddrOfPinnedObject();
            ret._itemCounts = arr.Length;

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> FromBuffer(byte[] arr)
        {
            var ret = new UnmanagedArray<T>();
            ret._itemCounts = arr.Length / sizeof(T);
            var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            ret._gcHandle = handle;
            ret._itemBuffer = (T*)handle.AddrOfPinnedObject();

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> FromPool(int length, InternalBufferManager manager)
        {
            //TODO! Upgrade InternalBufferManager to use pre-pinned arrays.
            var ret = new UnmanagedArray<T>();
            ret._itemCounts = length;
            var buffer = manager.TakeBuffer(length * sizeof(T));
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            ret._gcHandle = handle;
            ret._itemBuffer = (T*)handle.AddrOfPinnedObject();
            ret._dispose = () => manager.ReturnBuffer(buffer);
            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> FromStackalloc(int length, InternalBufferManager manager)
        {
            var ret = new UnmanagedArray<T>();
            ret._itemCounts = length;
            var buffer = manager.TakeBuffer(length * sizeof(T));
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            ret._gcHandle = handle;
            ret._itemBuffer = (T*)handle.AddrOfPinnedObject();
            ret._dispose = () => manager.ReturnBuffer(buffer);
            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> Copy(UnmanagedArray<T> source)
        {
            var itemCount = source._itemCounts;
            var len = itemCount * sizeof(T);
            var ret = new UnmanagedArray<T>(itemCount);
            Buffer.MemoryCopy(source._itemBuffer, ret._itemBuffer, len, len);
            //source.AsSpan().CopyTo(ret.AsSpan());
            return ret;
        }

        public int Count => _itemCounts;

        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(_itemBuffer + index);
            [MethodImpl((MethodImplOptions)768)] set => *(_itemBuffer + index) = value;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void Reallocate(int length, bool copyOldValues = false)
        {
            var ptr = Marshal.AllocHGlobal(length * sizeof(T)).ToPointer();
            if (copyOldValues)
            {
                var tptr = (T*)ptr;
                var bytes = Math.Min(_itemCounts, length) * sizeof(T);
                Buffer.MemoryCopy(_itemBuffer, tptr, bytes, bytes);
            }

            Free();
            _itemCounts = length;
            _itemBuffer = (T*)ptr;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void Reallocate(int length, bool copyOldValues, T fill)
        {
            var ptr = Marshal.AllocHGlobal(length * sizeof(T)).ToPointer();
            var tptr = (T*)ptr;
            if (copyOldValues)
            {
                var bytes = Math.Min(_itemCounts, length) * sizeof(T);
                Buffer.MemoryCopy(_itemBuffer, tptr, bytes, bytes);

                if (length > _itemCounts)
                {
                    new Span<T>(tptr + _itemCounts, length - _itemCounts).Fill(fill);
                }
            }
            else
            {
                new Span<T>(tptr, length).Fill(fill);
            }

            Free();
            _itemCounts = length;
            _itemBuffer = (T*)ptr;
        }

        [MethodImpl((MethodImplOptions)768)]
        private void Reallocate(int length, T fill)
        {
            var ptr = Marshal.AllocHGlobal(length * sizeof(T)).ToPointer();

            Free();
            _itemCounts = length;
            _itemBuffer = (T*)ptr;
            new Span<T>(ptr, _itemCounts).Fill(fill);
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetValue(int index)
        {
            return *(_itemBuffer + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public ref T GetRefTo(int index)
        {
            return ref *(_itemBuffer + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetValue(int index, ref T value)
        {
            *(_itemBuffer + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetValue(int index, T value)
        {
            *(_itemBuffer + index) = value;
        }

        public void Free()
        {
            if (_dispose != null)
            {
                _dispose();
                _dispose = null;
                return;
            }

            if (_gcHandle.HasValue)
            {
                _gcHandle.Value.Free();
                _gcHandle = null;
                _itemBuffer = null;
                return;
            }

            if (_itemBuffer != null)
            {
                Marshal.FreeHGlobal((IntPtr)_itemBuffer);
                _itemBuffer = null;
                return;
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        public IEnumerator<T> GetEnumerator()
        {
            for (var i = 0; i < _itemCounts; i++) yield return GetValue(i);
        }

        [MethodImpl((MethodImplOptions)768)]
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// NotSupported
        /// </summary>
        [MethodImpl((MethodImplOptions)512)]
        public void Clear()
        {
            throw new NotSupportedException();
        }

        [MethodImpl((MethodImplOptions)768)]
        public bool Contains(T item)
        {
            for (var i = 0; i < _itemCounts; i++)
            {
                if ((*(_itemBuffer + i)).Equals(item)) return true;
            }

            return false;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(T[] array, int arrayIndex)
        {
            for (var i = 0; i < _itemCounts; i++)
            {
                array[i + arrayIndex] = *(_itemBuffer + i);
            }
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(UnmanagedArray<T> array, int arrayIndex)
        {
            var length = _itemCounts - arrayIndex;
            Buffer.MemoryCopy(_itemBuffer + arrayIndex, array._itemBuffer, sizeof(T) * array._itemCounts, sizeof(T) * length);
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(T* array, int arrayIndex, int lengthToCopy)
        {
            Buffer.MemoryCopy(_itemBuffer + arrayIndex, array, sizeof(T) * lengthToCopy, sizeof(T) * lengthToCopy);
        }

        /// <summary>Copies the elements of the <see cref="T:System.Collections.ICollection" /> to an <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.</summary>
        /// <param name="array">The one-dimensional <see cref="T:System.Array" /> that is the destination of the elements copied from <see cref="T:System.Collections.ICollection" />. The <see cref="T:System.Array" /> must have zero-based indexing. </param>
        /// <param name="index">The zero-based index in <paramref name="array" /> at which copying begins. </param>
        /// <exception cref="T:System.ArgumentNullException">
        /// <paramref name="array" /> is <see langword="null" />. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">
        /// <paramref name="index" /> is less than zero. </exception>
        /// <exception cref="T:System.ArgumentException">
        /// <paramref name="array" /> is multidimensional.-or- The number of elements in the source <see cref="T:System.Collections.ICollection" /> is greater than the available space from <paramref name="index" /> to the end of the destination <paramref name="array" />.-or-The type of the source <see cref="T:System.Collections.ICollection" /> cannot be cast automatically to the type of the destination <paramref name="array" />.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(Array array, int index)
        {
            for (var i = 0; i < _itemCounts; i++)
            {
                array.SetValue((object) *(_itemBuffer + i), i + index);
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        public Span<T> AsSpan() => new Span<T>(_itemBuffer, _itemCounts);

        #region Explicit Implementations

        /// <summary>Gets a value indicating whether access to the <see cref="T:System.Collections.ICollection" /> is synchronized (thread safe).</summary>
        /// <returns>
        /// <see langword="true" /> if access to the <see cref="T:System.Collections.ICollection" /> is synchronized (thread safe); otherwise, <see langword="false" />.</returns>
        bool ICollection.IsSynchronized => false;

        bool ICollection<T>.IsReadOnly => false;

        /// <summary>Gets an object that can be used to synchronize access to the <see cref="T:System.Collections.ICollection" />.</summary>
        /// <returns>An object that can be used to synchronize access to the <see cref="T:System.Collections.ICollection" />.</returns>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public object SyncRoot => throw new NotSupportedException();

        /// <summary>
        /// NotSupported
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)512)]
        bool ICollection<T>.Remove(T item)
        {
            throw new NotSupportedException();
        }

        /// <summary>
        /// NotSupported
        /// </summary>
        /// <param name="item"></param>
        [MethodImpl((MethodImplOptions)512)]
        void ICollection<T>.Add(T item)
        {
            throw new NotSupportedException();
        }


        [EditorBrowsable(EditorBrowsableState.Never)]
        public ref T GetPinnableReference()
        {
            return ref *_itemBuffer;
        }

        #endregion

        #region Equality

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>
        /// <see langword="true" /> if the current object is equal to the <paramref name="other" /> parameter; otherwise, <see langword="false" />.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public bool Equals(UnmanagedArray<T> other)
        {
            return _itemCounts == other._itemCounts && _itemBuffer == other._itemBuffer;
        }

        /// <summary>Indicates whether this instance and a specified object are equal.</summary>
        /// <param name="obj">The object to compare with the current instance. </param>
        /// <returns>
        /// <see langword="true" /> if <paramref name="obj" /> and this instance are the same type and represent the same value; otherwise, <see langword="false" />. </returns>
        [MethodImpl((MethodImplOptions)768)]
        public override bool Equals(object obj)
        {
            return obj is UnmanagedArray<T> other && Equals(other);
        }

        /// <summary>Returns the hash code for this instance.</summary>
        /// <returns>A 32-bit signed integer that is the hash code for this instance.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public override int GetHashCode()
        {
            unchecked
            {
                return (_itemCounts * 397) ^ unchecked((int)(long)_itemBuffer);
            }
        }

        /// <summary>Returns a value that indicates whether the values of two <see cref="T:NumSharp.Backends.Unmanaged.UnmanagedArray`1" /> objects are equal.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public static bool operator ==(UnmanagedArray<T> left, UnmanagedArray<T> right)
        {
            return left.Equals(right);
        }

        /// <summary>Returns a value that indicates whether two <see cref="T:NumSharp.Backends.Unmanaged.UnmanagedArray`1" /> objects have different values.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public static bool operator !=(UnmanagedArray<T> left, UnmanagedArray<T> right)
        {
            return !left.Equals(right);
        }

        #endregion

        unsafe void* IUnmanagedArray._itemBuffer => _itemBuffer;
        int IUnmanagedArray._itemCounts => _itemCounts;
    }
}
