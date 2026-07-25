namespace NumSharp.Backends
{
    /// <summary>
    ///     An external BLAS a <see cref="TensorEngine"/> may delegate its matrix products to.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp is 100 % managed C#: it has no native dependency and computes every matrix
    ///     product with its own SIMD kernels. This interface is the ONE place an optional package
    ///     can offer an alternative — it is a seam, not a requirement. With
    ///     <see cref="TensorEngine.Blas"/> left null (the default) nothing here is ever consulted.
    ///     </para>
    ///     <para>
    ///     Both members are <c>Try</c>-shaped on purpose: a backend answers only for the operand
    ///     combinations it actually implements (an external CBLAS covers <c>float32</c>/
    ///     <c>float64</c> and nothing else) and returns false for the rest, which the engine then
    ///     computes with its own kernels. So installing a backend can change WHICH implementation
    ///     runs, never WHETHER NumSharp can compute the product.
    ///     </para>
    ///     <para>
    ///     The reason to want one is not only speed. Two correct matrix products that sum in
    ///     different orders give different bits, so a workload that must agree with another stack
    ///     to the last bit — e.g. training a network twice and byte-comparing the weights — has to
    ///     call that stack's own BLAS. See <c>NumSharp.Interop.BLAS</c> and
    ///     <c>docs/GEMM_PARITY.md</c>.
    ///     </para>
    /// </remarks>
    public interface IBlasBackend
    {
        /// <summary>
        ///     A one-line description of the underlying library (path, symbols, thread count),
        ///     for diagnostics. Never null.
        /// </summary>
        string Info { get; }

        /// <summary>
        ///     Computes <c>np.dot(left, right)</c>, allocating the result.
        /// </summary>
        /// <param name="result">The product, when this returns true; otherwise null.</param>
        /// <returns>
        ///     False when this backend does not serve these operands — the caller must then fall
        ///     back to its own kernels, and <paramref name="result"/> is meaningless.
        /// </returns>
        /// <remarks>
        ///     <c>dot</c> is a separate entry point from <see cref="TryMatMul2D"/> because the two
        ///     are not the same operation for every shape: NumPy implements them with two different
        ///     dispatchers, which disagree on some non-contiguous operands, and a backend claiming
        ///     bit-parity has to reproduce that.
        /// </remarks>
        bool TryDot(NDArray left, NDArray right, out NDArray result);

        /// <summary>
        ///     Computes the 2-D matrix product <c>left @ right</c> INTO <paramref name="result"/> —
        ///     the core of <c>np.matmul</c>, of <c>@</c>, and of every element of a stacked product.
        /// </summary>
        /// <param name="result">
        ///     A pre-allocated <c>(M, N)</c> array of the promoted dtype. Written in full when this
        ///     returns true; left untouched when it returns false.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    }
}
