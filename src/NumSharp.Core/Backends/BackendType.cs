using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public enum BackendType
    {
        Default = 1,

        /// <summary>
        /// Managed SIMD
        /// </summary>
        SIMD = 2,

        ArrayFire = 4
    }
}
