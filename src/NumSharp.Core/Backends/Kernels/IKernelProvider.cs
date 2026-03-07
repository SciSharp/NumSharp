namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Interface for kernel providers (IL, CUDA, Vulkan, etc.)
    /// Defines the contract for generating and caching computational kernels.
    /// </summary>
    /// <remarks>
    /// Kernel providers are responsible for:
    /// - Generating optimized kernels for specific operation/type combinations
    /// - Caching generated kernels to avoid repeated compilation
    /// - Detecting and utilizing hardware capabilities (SIMD, GPU)
    ///
    /// The IL provider (ILKernelGenerator) is the default implementation,
    /// using System.Reflection.Emit to generate SIMD-optimized kernels at runtime.
    /// Future providers could target CUDA, Vulkan, or other backends.
    /// </remarks>
    public interface IKernelProvider
    {
        /// <summary>Provider name for diagnostics (e.g., "IL", "CUDA").</summary>
        string Name { get; }

        /// <summary>Whether this provider is enabled.</summary>
        bool Enabled { get; set; }

        /// <summary>SIMD vector width in bits (128, 256, 512, or 0 for none).</summary>
        int VectorBits { get; }

        /// <summary>Check if type supports SIMD on this provider.</summary>
        /// <param name="type">The NPTypeCode to check.</param>
        /// <returns>True if SIMD operations are supported for this type.</returns>
        bool CanUseSimd(NPTypeCode type);

        // ===================
        // Binary Operations
        // ===================

        /// <summary>
        /// Get contiguous same-type binary kernel.
        /// Used for fast-path when both operands are contiguous with identical type.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="op">The binary operation to perform.</param>
        /// <returns>The kernel delegate, or null if not supported.</returns>
        ContiguousKernel<T>? GetContiguousKernel<T>(BinaryOp op) where T : unmanaged;

        /// <summary>
        /// Get mixed-type binary kernel.
        /// Handles operations where LHS, RHS, and result may have different types.
        /// </summary>
        /// <param name="key">The kernel key specifying types, operation, and execution path.</param>
        /// <returns>The kernel delegate, or null if not supported.</returns>
        MixedTypeKernel? GetMixedTypeKernel(MixedTypeKernelKey key);

        // ===================
        // Unary Operations
        // ===================

        /// <summary>
        /// Get unary kernel for array operations.
        /// </summary>
        /// <param name="key">The kernel key specifying input/output types and operation.</param>
        /// <returns>The kernel delegate, or null if not supported.</returns>
        UnaryKernel? GetUnaryKernel(UnaryKernelKey key);

        /// <summary>
        /// Get unary scalar function for general path.
        /// Used for element-by-element operations in broadcasting scenarios.
        /// </summary>
        /// <typeparam name="TIn">Input value type.</typeparam>
        /// <typeparam name="TOut">Output value type.</typeparam>
        /// <param name="op">The unary operation to perform.</param>
        /// <returns>The scalar function delegate, or null if not supported.</returns>
        UnaryScalar<TIn, TOut>? GetUnaryScalar<TIn, TOut>(UnaryOp op);

        // ===================
        // Reduction Operations
        // ===================

        /// <summary>
        /// Get element-wise reduction kernel.
        /// Reduces all elements to a single scalar value.
        /// </summary>
        /// <typeparam name="TResult">The result/accumulator type.</typeparam>
        /// <param name="key">The kernel key specifying input type, accumulator type, and operation.</param>
        /// <returns>The kernel delegate, or null if not supported.</returns>
        TypedElementReductionKernel<TResult>? GetElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged;

        // ===================
        // Comparison Operations
        // ===================

        /// <summary>
        /// Get comparison kernel.
        /// Performs element-wise comparison returning boolean array.
        /// </summary>
        /// <param name="key">The kernel key specifying LHS/RHS types, operation, and execution path.</param>
        /// <returns>The kernel delegate, or null if not supported.</returns>
        ComparisonKernel? GetComparisonKernel(ComparisonKernelKey key);

        /// <summary>
        /// Get comparison scalar function for general path.
        /// Used for element-by-element comparisons in broadcasting scenarios.
        /// </summary>
        /// <typeparam name="TLhs">Left operand type.</typeparam>
        /// <typeparam name="TRhs">Right operand type.</typeparam>
        /// <param name="op">The comparison operation to perform.</param>
        /// <returns>The scalar function delegate, or null if not supported.</returns>
        ComparisonScalar<TLhs, TRhs>? GetComparisonScalar<TLhs, TRhs>(ComparisonOp op);

        // ===================
        // Cache Management
        // ===================

        /// <summary>Clear all cached kernels.</summary>
        void Clear();

        /// <summary>Number of cached kernels.</summary>
        int CacheCount { get; }
    }
}
