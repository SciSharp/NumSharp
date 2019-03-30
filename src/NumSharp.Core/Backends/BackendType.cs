using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public enum BackendType
    {
        /// <summary>
        /// Managed Array
        /// </summary>
        ManagedArray = 1,

        /// <summary>
        /// Managed SIMD
        /// </summary>
        VectorT = 2
    }
}
