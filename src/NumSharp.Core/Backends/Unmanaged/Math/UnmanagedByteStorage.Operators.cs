using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Unmanaged
{
    public partial class UnmanagedByteStorage<T>
    {
        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<TTo> Cast<TFrom, TTo>(ref TFrom addr, int length)
        {
            return new Span<TTo>(Unsafe.AsPointer(ref addr), checked(length * Unsafe.SizeOf<TFrom>()) / Unsafe.SizeOf<TTo>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<TTo> Cast<TFrom, TTo>(TFrom* addr, int length) where TFrom : unmanaged
        {
            return new Span<TTo>(addr, checked(length * Unsafe.SizeOf<TFrom>()) / Unsafe.SizeOf<TTo>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<TTo> Cast<TFrom, TTo>(IntPtr addr, int length)
        {
            return new Span<TTo>(addr.ToPointer(), checked(length * Unsafe.SizeOf<TFrom>()) / Unsafe.SizeOf<TTo>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<TTo> Cast<TFrom, TTo>(UnmanagedByteStorage<T> unmanagedByteStorage)
        {
            return new Span<TTo>(unmanagedByteStorage.Address, checked(unmanagedByteStorage._array.Length * Unsafe.SizeOf<TFrom>()) / Unsafe.SizeOf<TTo>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<Vector<T>> CastVector<T>(ref T addr, int length) where T : struct
        {
            return new Span<Vector<T>>(Unsafe.AsPointer(ref addr), checked(length * Unsafe.SizeOf<T>()) / Unsafe.SizeOf<Vector<T>>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<Vector<T>> CastVector<T>(T* addr, int length) where T : unmanaged
        {
            return new Span<Vector<T>>(addr, checked(length * Unsafe.SizeOf<T>()) / Unsafe.SizeOf<Vector<T>>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<Vector<T>> CastVector<T>(IntPtr addr, int length) where T : struct
        {
            return new Span<Vector<T>>(addr.ToPointer(), checked(length * Unsafe.SizeOf<T>()) / Unsafe.SizeOf<Vector<T>>());
        }

        [MethodImpl((MethodImplOptions)768)]
        private static unsafe Span<Vector<T>> CastVector<T>(UnmanagedByteStorage<T> unmanagedByteStorage) where T : unmanaged
        {
            return new Span<Vector<T>>(unmanagedByteStorage.Address, checked(unmanagedByteStorage._array.Length * Unsafe.SizeOf<T>()) / Unsafe.SizeOf<Vector<T>>());
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal sealed class Pin<T>
    {
        public T Data;
    }
}
