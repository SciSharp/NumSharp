// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// Simplified for NumSharp - only supports unmanaged types (no reference type handling)

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Provides low-level memory copy operations for unmanaged types.
    /// </summary>
    public static partial class UnmanagedBuffer
    {
        /// <summary>
        /// Copies bytes from source to destination using unmanaged memory copy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [CLSCompliant(false)]
        public static unsafe void MemoryCopy(void* source, void* destination, long destinationSizeInBytes, long sourceBytesToCopy)
        {
            if (sourceBytesToCopy > destinationSizeInBytes)
            {
                ThrowHelper.ThrowArgumentOutOfRangeException(ExceptionArgument.sourceBytesToCopy);
            }

            Memmove(ref *(byte*)destination, ref *(byte*)source, checked((nuint)sourceBytesToCopy));
        }

        /// <summary>
        /// Copies bytes from source to destination using unmanaged memory copy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [CLSCompliant(false)]
        public static unsafe void MemoryCopy(void* source, void* destination, ulong destinationSizeInBytes, ulong sourceBytesToCopy)
        {
            if (sourceBytesToCopy > destinationSizeInBytes)
            {
                ThrowHelper.ThrowArgumentOutOfRangeException(ExceptionArgument.sourceBytesToCopy);
            }

            Memmove(ref *(byte*)destination, ref *(byte*)source, checked((nuint)sourceBytesToCopy));
        }

        /// <summary>
        /// Copies elements from source to destination.
        /// For unmanaged types only - does not handle reference types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Memmove<T>(ref T destination, ref T source, nuint elementCount) where T : unmanaged
        {
            UnmanagedSpanHelpers.Memmove(
                ref Unsafe.As<T, byte>(ref destination),
                ref Unsafe.As<T, byte>(ref source),
                elementCount * (nuint)sizeof(T));
        }

        /// <summary>
        /// Copies elements from source to destination with ulong element count.
        /// For unmanaged types only - does not handle reference types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Memmove<T>(ref T destination, ref T source, ulong elementCount) where T : unmanaged
        {
            UnmanagedSpanHelpers.Memmove(
                ref Unsafe.As<T, byte>(ref destination),
                ref Unsafe.As<T, byte>(ref source),
                checked((nuint)(elementCount * (ulong)sizeof(T))));
        }
    }
}
