namespace NumSharp.Backends
{
    /// <summary>
    ///     Default Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    /// <remarks>
    /// DefaultEngine is the pure C# implementation of TensorEngine.
    /// It uses ILKernelGenerator internally for SIMD-optimized kernels.
    /// All computation on NDArray should go through TensorEngine methods.
    /// </remarks>
    public partial class DefaultEngine : TensorEngine
    {
        // ILKernelGenerator is used directly in DefaultEngine partial files
        // (DefaultEngine.BinaryOp.cs, DefaultEngine.UnaryOp.cs, etc.)
        // No public kernel access - all computation goes through TensorEngine methods
    }
}
