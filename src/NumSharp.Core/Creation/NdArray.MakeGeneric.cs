using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray<T> MakeGeneric<T>() where T : unmanaged
        {
            return new NDArray<T>(Storage);
        }

        /// <summary>
        ///     Creates an identical copy without reallocating data.
        /// </summary>
        /// <typeparam name="T">The type of the generic</typeparam>
        /// <returns>This NDArray as a generic version.</returns>
        /// <exception cref="InvalidOperationException">When <typeparamref name="T"/> != <see cref="dtype"/></exception>
        public NDArray<T> AsGeneric<T>() where T : unmanaged
        {
            if (typeof(T) != dtype)
                throw new InvalidOperationException($"Given constraint type {typeof(T).Name} does not match dtype {dtype.Name}. If you intended to cast if necessary then use MakeGeneric.");
            if (this is NDArray<T> ret)
                return ret;

            return MakeGeneric<T>();
        }
    }
}
