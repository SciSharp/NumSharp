using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace NumSharp.Backends
{
    /// <summary>
    ///     Default Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    public partial class DefaultEngine : TensorEngine
    {
        /// <summary>
        ///     The threshold atwhich after n-items in an array, computation will use Parallel.For
        /// </summary>
        public const int ParallelLimit = 84999;
    }
}
