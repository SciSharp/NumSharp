using System;

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

        /// <summary>
        /// Get unary scalar delegate with runtime type dispatch.
        /// Returns Func&lt;TIn, TOut&gt; as Delegate for type-erased scenarios.
        /// </summary>
        Delegate? GetUnaryScalarDelegate(UnaryScalarKernelKey key);

        /// <summary>
        /// Get binary scalar delegate with runtime type dispatch.
        /// Returns Func&lt;TLhs, TRhs, TResult&gt; as Delegate for type-erased scenarios.
        /// </summary>
        Delegate? GetBinaryScalarDelegate(BinaryScalarKernelKey key);

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

        /// <summary>
        /// Get comparison scalar delegate with runtime type dispatch.
        /// Returns Func&lt;TLhs, TRhs, bool&gt; as Delegate for type-erased scenarios.
        /// </summary>
        Delegate? GetComparisonScalarDelegate(ComparisonScalarKernelKey key);

        // ===================
        // SIMD Helper Operations
        // ===================

        /// <summary>
        /// Test whether all array elements evaluate to true (non-zero).
        /// Uses SIMD with early-exit for contiguous arrays.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="data">Pointer to contiguous array data.</param>
        /// <param name="size">Number of elements.</param>
        /// <returns>True if all elements are non-zero.</returns>
        unsafe bool All<T>(T* data, int size) where T : unmanaged;

        /// <summary>
        /// Test whether any array element evaluates to true (non-zero).
        /// Uses SIMD with early-exit for contiguous arrays.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="data">Pointer to contiguous array data.</param>
        /// <param name="size">Number of elements.</param>
        /// <returns>True if any element is non-zero.</returns>
        unsafe bool Any<T>(T* data, int size) where T : unmanaged;

        /// <summary>
        /// Find indices of all non-zero elements.
        /// Uses SIMD for efficient scanning of contiguous arrays.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="data">Pointer to contiguous array data.</param>
        /// <param name="size">Number of elements.</param>
        /// <param name="indices">Output list to populate with non-zero indices.</param>
        unsafe void FindNonZero<T>(T* data, int size, System.Collections.Generic.List<int> indices) where T : unmanaged;

        /// <summary>
        /// Convert flat (linear) indices to per-dimension coordinate arrays.
        /// </summary>
        /// <param name="flatIndices">List of flat indices.</param>
        /// <param name="shape">Shape of the array.</param>
        /// <returns>Array of NDArray&lt;int&gt;, one per dimension.</returns>
        NumSharp.Generic.NDArray<int>[] ConvertFlatToCoordinates(System.Collections.Generic.List<int> flatIndices, int[] shape);

        /// <summary>
        /// Count the number of true values in a boolean array.
        /// Uses SIMD for efficient counting.
        /// </summary>
        /// <param name="data">Pointer to boolean array.</param>
        /// <param name="size">Number of elements.</param>
        /// <returns>Count of true values.</returns>
        unsafe int CountTrue(bool* data, int size);

        /// <summary>
        /// Copy elements from source to destination where mask is true.
        /// Uses SIMD for efficient mask scanning.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="src">Source array pointer.</param>
        /// <param name="mask">Boolean mask array pointer.</param>
        /// <param name="dest">Destination array pointer (must have capacity for all true elements).</param>
        /// <param name="size">Number of elements in source and mask.</param>
        /// <returns>Number of elements copied.</returns>
        unsafe int CopyMasked<T>(T* src, bool* mask, T* dest, int size) where T : unmanaged;

        /// <summary>
        /// Compute variance of a contiguous array.
        /// Uses SIMD-optimized two-pass algorithm for float/double.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="data">Pointer to contiguous array data.</param>
        /// <param name="size">Number of elements.</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population, 1 for sample).</param>
        /// <returns>Variance as double.</returns>
        unsafe double Variance<T>(T* data, int size, int ddof = 0) where T : unmanaged;

        /// <summary>
        /// Compute standard deviation of a contiguous array.
        /// Uses SIMD-optimized two-pass algorithm for float/double.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
        /// <param name="data">Pointer to contiguous array data.</param>
        /// <param name="size">Number of elements.</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population, 1 for sample).</param>
        /// <returns>Standard deviation as double.</returns>
        unsafe double StandardDeviation<T>(T* data, int size, int ddof = 0) where T : unmanaged;

        /// <summary>
        /// NaN-aware sum: sums all non-NaN values (NaN treated as 0).
        /// </summary>
        unsafe float NanSumFloat(float* data, int size);

        /// <summary>
        /// NaN-aware sum: sums all non-NaN values (NaN treated as 0).
        /// </summary>
        unsafe double NanSumDouble(double* data, int size);

        /// <summary>
        /// NaN-aware product: multiplies all non-NaN values (NaN treated as 1).
        /// </summary>
        unsafe float NanProdFloat(float* data, int size);

        /// <summary>
        /// NaN-aware product: multiplies all non-NaN values (NaN treated as 1).
        /// </summary>
        unsafe double NanProdDouble(double* data, int size);

        /// <summary>
        /// NaN-aware minimum: finds minimum ignoring NaN values.
        /// Returns NaN if all values are NaN.
        /// </summary>
        unsafe float NanMinFloat(float* data, int size);

        /// <summary>
        /// NaN-aware minimum: finds minimum ignoring NaN values.
        /// Returns NaN if all values are NaN.
        /// </summary>
        unsafe double NanMinDouble(double* data, int size);

        /// <summary>
        /// NaN-aware maximum: finds maximum ignoring NaN values.
        /// Returns NaN if all values are NaN.
        /// </summary>
        unsafe float NanMaxFloat(float* data, int size);

        /// <summary>
        /// NaN-aware maximum: finds maximum ignoring NaN values.
        /// Returns NaN if all values are NaN.
        /// </summary>
        unsafe double NanMaxDouble(double* data, int size);

        // ===================
        // Cache Management
        // ===================

        /// <summary>Clear all cached kernels.</summary>
        void Clear();

        /// <summary>Number of cached kernels.</summary>
        int CacheCount { get; }
    }
}
