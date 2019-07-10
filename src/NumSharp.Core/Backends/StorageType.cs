using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public enum StorageType
    {
        /// <summary>
        ///     <see cref="UnmanagedByteStorage{T}"/>
        /// </summary>
        Unmanaged = 0,

        /// <summary>
        ///     <see cref="TypedArrayStorage"/>
        /// </summary>
        TypedArray = 1,
    }
}
