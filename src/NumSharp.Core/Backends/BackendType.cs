using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public enum BackendType
    {
        MKL = 1,

        CUDA = 2,

        /// <summary>
        /// Managed SIMD
        /// </summary>
        SIMD = 3,

        ArrayFire = 4
    }
}
