using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    ///     Default Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    public partial class DefaultEngine : TensorEngine
    {
        /// <summary>
        ///     The threshold at which after n-items in an array, computation will use Parallel.For
        /// </summary>
        public const int ParallelAbove = 84999;

        /// <summary>
        ///     The kernel provider for IL-generated kernels.
        ///     Abstracts kernel generation to enable future backends (CUDA, Vulkan).
        /// </summary>
        protected readonly IKernelProvider KernelProvider = ILKernelGenerator.Instance;
    }
}
