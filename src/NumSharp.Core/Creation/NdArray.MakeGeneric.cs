using System;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Creates an independent typed view (alias) over this array's data without reallocating.
        /// </summary>
        /// <remarks>
        ///     The returned <see cref="NDArray{T}"/> shares the same underlying memory block but carries
        ///     its <b>own</b> shape metadata (via <see cref="UnmanagedStorage.Alias()"/>), so reshaping,
        ///     expanding or otherwise mutating its shape does NOT propagate back to this array — matching
        ///     NumPy's <c>ndarray.view()</c> semantics. Element writes still affect the shared data.
        ///     A fresh header is produced on every call, exactly like NumPy's <c>view()</c>.
        /// </remarks>
        /// <typeparam name="T">The type of the generic; must equal <see cref="dtype"/>.</typeparam>
        /// <returns>An independent typed view sharing this NDArray's data.</returns>
        /// <exception cref="ArgumentException">When <typeparamref name="T"/> != <see cref="dtype"/></exception>
        public NDArray<T> MakeGeneric<T>() where T : unmanaged
        {
            return new NDArray<T>(Storage.Alias());
        }

        /// <summary>
        ///     Tries to cast to <see cref="NDArray{T}"/>; if that fails but the dtype already matches,
        ///     wraps the existing storage. Returns <c>null</c> when <typeparamref name="T"/> != <see cref="dtype"/>
        ///     (try-cast / <c>as</c> semantics — never throws).
        /// </summary>
        /// <remarks>
        ///     The zero-data-alloc fast path returns <c>this</c> when it is already an <see cref="NDArray{T}"/>.
        ///     Otherwise it wraps the same storage; this is intended for freshly-produced engine results
        ///     (e.g. comparison outputs), so the wrapped storage is not aliased.
        /// </remarks>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version, or <c>null</c> when <typeparamref name="T"/> != <see cref="dtype"/>.</returns>
        public NDArray<T> AsGeneric<T>() where T : unmanaged
        {
            if (typeof(T) != dtype)
                return null;

            return this as NDArray<T> ?? new NDArray<T>(Storage);
        }

        /// <summary>
        ///     When the dtype already matches, returns an independent typed view (alias) sharing this
        ///     array's data; otherwise converts the storage to <typeparamref name="T"/> via the
        ///     NDIter-backed <see cref="UnmanagedStorage.Cast{T}"/> (a fresh, owned copy). Never throws
        ///     on dtype mismatch.
        /// </summary>
        /// <remarks>
        ///     The matching branch aliases (see <see cref="MakeGeneric{T}"/>) so a later reshape of the
        ///     result does not mutate this array's shape; the converting branch already owns fresh storage.
        /// </remarks>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version, sharing data when the dtype matches.</returns>
        public NDArray<T> AsOrMakeGeneric<T>() where T : unmanaged
        {
            if (typeof(T) != dtype)
            {
                var converted = Storage.Cast<T>();
                return new NDArray<T>(converted);
            }

            return new NDArray<T>(Storage.Alias());
        }
    }
}
