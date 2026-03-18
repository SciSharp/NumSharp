using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    ///     Default Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    public partial class DefaultEngine : TensorEngine
    {
        // KernelProvider field removed - DefaultEngine calls ILKernelGenerator.Instance directly

        /// <summary>
        ///     Default kernel provider for static access (np.all, np.any, masking, etc.).
        ///     TODO: Remove this once np.all/np.any/masking are routed through TensorEngine.
        /// </summary>
        public static IKernelProvider DefaultKernelProvider { get; } = ILKernelGenerator.Instance;
    }
}
