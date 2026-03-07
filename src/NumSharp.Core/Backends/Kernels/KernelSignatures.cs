namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // KernelSignatures.cs - Centralized delegate type definitions for kernel system
    // =============================================================================
    //
    // This file defines delegate types used by IKernelProvider implementations.
    // Some delegates have existing definitions in other files (noted below) and
    // will be consolidated in future refactoring phases.
    //
    // EXISTING DELEGATES (defined elsewhere, DO NOT duplicate):
    //   - BinaryKernel<T>, MixedTypeKernel, UnaryKernel, ComparisonKernel in BinaryKernel.cs
    //   - ElementReductionKernel, TypedElementReductionKernel<T>, AxisReductionKernel,
    //     CumulativeKernel, CumulativeAxisKernel in ReductionKernel.cs
    //
    // SHARED DELEGATES (defined here):
    //   - ContiguousKernel<T> - contiguous same-type binary operations
    //   - UnaryKernelStrided<TIn, TOut> - strided unary operations
    //   - UnaryScalar<TIn, TOut> - single-value unary function
    //   - BinaryScalar<TLhs, TRhs, TOut> - single-value binary function
    //   - ComparisonScalar<TLhs, TRhs> - single-value comparison function
    //   - TypedAxisReductionKernel<T> - typed axis reduction
    //   - SimpleReductionKernel<T> - simplified contiguous reduction
    // =============================================================================

    // ===================
    // Contiguous Binary Kernel
    // ===================

    /// <summary>
    /// Delegate for contiguous (SimdFull) binary operations.
    /// Simplified signature - no strides needed since both arrays are contiguous.
    /// </summary>
    /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
    /// <param name="lhs">Pointer to left operand data.</param>
    /// <param name="rhs">Pointer to right operand data.</param>
    /// <param name="result">Pointer to output data.</param>
    /// <param name="count">Number of elements to process.</param>
    public unsafe delegate void ContiguousKernel<T>(T* lhs, T* rhs, T* result, int count) where T : unmanaged;

    // ===================
    // Strided Unary Kernel
    // ===================

    /// <summary>
    /// Strided unary operation kernel with explicit offset and stride parameters.
    /// Used when input/output arrays are not contiguous in memory.
    /// </summary>
    /// <typeparam name="TIn">Input element type</typeparam>
    /// <typeparam name="TOut">Output element type</typeparam>
    /// <param name="input">Pointer to input data</param>
    /// <param name="inOffset">Starting offset in input (element units)</param>
    /// <param name="inStride">Stride between input elements (element units)</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="outOffset">Starting offset in output (element units)</param>
    /// <param name="outStride">Stride between output elements (element units)</param>
    /// <param name="count">Number of elements to process</param>
    public unsafe delegate void UnaryKernelStrided<TIn, TOut>(
        TIn* input, int inOffset, int inStride,
        TOut* output, int outOffset, int outStride,
        int count) where TIn : unmanaged where TOut : unmanaged;

    // ===================
    // Scalar Delegates (for broadcasting and element-wise operations)
    // ===================

    /// <summary>
    /// Unary scalar function delegate.
    /// Used for element-by-element operations in broadcasting and general paths.
    /// </summary>
    /// <typeparam name="TIn">Input value type</typeparam>
    /// <typeparam name="TOut">Output value type</typeparam>
    /// <param name="value">Input value</param>
    /// <returns>Transformed output value</returns>
    public delegate TOut UnaryScalar<in TIn, out TOut>(TIn value);

    /// <summary>
    /// Binary scalar function delegate.
    /// Used for element-by-element binary operations in broadcasting and general paths.
    /// </summary>
    /// <typeparam name="TLhs">Left operand type</typeparam>
    /// <typeparam name="TRhs">Right operand type</typeparam>
    /// <typeparam name="TOut">Result type</typeparam>
    /// <param name="lhs">Left operand value</param>
    /// <param name="rhs">Right operand value</param>
    /// <returns>Result of binary operation</returns>
    public delegate TOut BinaryScalar<in TLhs, in TRhs, out TOut>(TLhs lhs, TRhs rhs);

    /// <summary>
    /// Comparison scalar function delegate.
    /// Used for element-by-element comparisons returning boolean.
    /// </summary>
    /// <typeparam name="TLhs">Left operand type</typeparam>
    /// <typeparam name="TRhs">Right operand type</typeparam>
    /// <param name="lhs">Left operand value</param>
    /// <param name="rhs">Right operand value</param>
    /// <returns>True if comparison holds, false otherwise</returns>
    public delegate bool ComparisonScalar<in TLhs, in TRhs>(TLhs lhs, TRhs rhs);

    // ===================
    // Typed Reduction Delegates
    // ===================

    /// <summary>
    /// Typed axis reduction kernel with generic element type.
    /// Reduces along a specific axis with strongly-typed pointers.
    /// </summary>
    /// <typeparam name="T">Element type</typeparam>
    /// <param name="input">Pointer to input data</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="outerSize">Product of dimensions before the reduction axis</param>
    /// <param name="axisSize">Size of the axis being reduced</param>
    /// <param name="innerSize">Product of dimensions after the reduction axis</param>
    public unsafe delegate void TypedAxisReductionKernel<T>(
        T* input, T* output,
        int outerSize, int axisSize, int innerSize) where T : unmanaged;

    /// <summary>
    /// Simple contiguous reduction kernel returning single value.
    /// Used for full-array reductions when input is contiguous.
    /// </summary>
    /// <typeparam name="T">Element type</typeparam>
    /// <param name="input">Pointer to contiguous input data</param>
    /// <param name="count">Number of elements</param>
    /// <returns>Reduced value</returns>
    public unsafe delegate T SimpleReductionKernel<T>(T* input, int count) where T : unmanaged;
}
