using System;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Creates an alias without reallocating data.
        /// </summary>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version.</returns>
        /// <exception cref="InvalidOperationException">When <typeparamref name="T"/> != <see cref="dtype"/></exception>
        public NDArray<T> MakeGeneric<T>() where T : unmanaged
        {
            return new NDArray<T>(Storage);
        }

        /// <summary>
        ///     Tries to cast to <see cref="NDArray{T}"/>, otherwise creates an alias without reallocating data.
        /// </summary>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version.</returns>
        /// <exception cref="InvalidOperationException">When <typeparamref name="T"/> != <see cref="dtype"/></exception>
        public NDArray<T> AsGeneric<T>() where T : unmanaged
        {
            if (typeof(T) != dtype)
                return null;
            
            return this as NDArray<T> ?? new NDArray<T>(Storage);
        }

        /// <summary>
        ///     Tries to cast to <see cref="NDArray{T}"/>, otherwise calls <see cref="NDArray{T}.astype"/>.
        /// </summary>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version.</returns>
        /// <exception cref="InvalidOperationException">When <typeparamref name="T"/> != <see cref="dtype"/></exception>
        public NDArray<T> AsOrMakeGeneric<T>() where T : unmanaged
        {
            if (typeof(T) != dtype)
                return new NDArray<T>(this.astype(InfoOf<T>.NPTypeCode, copy: true));

            return new NDArray<T>(Storage);
        }
    }
}
