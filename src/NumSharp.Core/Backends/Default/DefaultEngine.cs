namespace NumSharp.Backends
{
    /// <summary>
    ///     Default Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    /// <remarks>
    /// DefaultEngine is the pure C# implementation of TensorEngine.
    /// It uses DirectILKernelGenerator internally for SIMD-optimized kernels.
    /// All computation on NDArray should go through TensorEngine methods.
    /// Its results bind the instance that computed them (<see cref="TensorEngine.NewArray(DType, Shape, bool)"/>), so a
    /// subclass assigned to an array, or to <see cref="TensorEngine.Default"/>, also computes the operations on those
    /// results.
    /// </remarks>
    public partial class DefaultEngine : TensorEngine
    {
        // DirectILKernelGenerator is used directly in DefaultEngine partial files
        // (DefaultEngine.BinaryOp.cs, DefaultEngine.UnaryOp.cs, etc.)
        // No public kernel access - all computation goes through TensorEngine methods

        /// <summary>
        ///     The <see cref="DefaultEngine"/> that runs DefaultEngine's own statements for <paramref name="a"/> where a facade
        ///     calls them directly instead of a <see cref="TensorEngine"/> member (np.quantile / np.percentile's
        ///     numpy.char.chararray route, which needs a flag no member takes): <paramref name="a"/>'s engine when it is a
        ///     DefaultEngine (a subclass included), so the results bind it as the member's would, else the built-in one.
        /// </summary>
        /// <param name="a">The operand whose engine computes the call; not null.</param>
        /// <returns>The operand's engine, or <see cref="BackendFactory.BuiltIn"/> for an engine that does not derive from
        ///     DefaultEngine (whose own members cannot run these statements).</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> is null (the facades check it first).</exception>
        internal static DefaultEngine Of(NDArray a) => a.TensorEngine as DefaultEngine ?? BackendFactory.BuiltIn;
    }
}
