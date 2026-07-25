using System;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.Blas
{
    /// <summary>
    ///     The <see cref="IBlasBackend"/> this package installs: NumSharp's matrix products computed
    ///     by an external CBLAS library, through a route-for-route port of NumPy's two
    ///     matrix-product dispatchers.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It answers for <c>float32</c> and <c>float64</c> only, and returns false for everything
    ///     else so the engine keeps computing it with its own managed kernels — integer and bool
    ///     products do not need a BLAS anyway (modular integer addition is associative, so summation
    ///     order cannot change the result).
    ///     </para>
    ///     <para>
    ///     <b>Why the two entry points are ported separately.</b> <c>np.dot</c> and <c>np.matmul</c>
    ///     are not the same C code in NumPy: <c>cblas_matrixproduct</c> (+ the N-D <c>dotfunc</c>
    ///     tail) versus the <c>@TYPE@_matmul</c> gufunc. They agree bit-for-bit on nearly every
    ///     input, but pick different routes when an operand is not blasable — a stride-2 matrix
    ///     times a vector gets gemv-on-a-copy from one and the portable loop from the other, and
    ///     278/300 elements differ. A backend claiming bit-parity must reproduce that split, which
    ///     is why <see cref="IBlasBackend"/> has both <see cref="TryDot"/> and
    ///     <see cref="TryMatMul2D"/> rather than one matrix-product method.
    ///     </para>
    /// </remarks>
    public sealed class OpenBlasBackend : IBlasBackend
    {
        /// <inheritdoc/>
        public string Info
        {
            get
            {
                if (!CBlasNative.IsLoaded)
                    return "<no CBLAS library loaded>";

                var config = CBlasNative.GetConfig();
                return $"{CBlasNative.LibraryPath} [symbols {CBlasNative.SymbolScheme}, " +
                       $"{(CBlasNative.IsIlp64 ? "ILP64" : "LP64")}, threads {CBlasNative.GetNumThreads()}]" +
                       (config == null ? string.Empty : " " + config);
            }
        }

        /// <inheritdoc/>
        public bool TryDot(NDArray left, NDArray right, out NDArray result)
            => BlasParity.TryDot(left, right, out result);

        /// <inheritdoc/>
        public bool TryMatMul2D(NDArray left, NDArray right, NDArray result)
            => BlasParity.TryMatmul2D(left, right, result);

        /// <summary>The loaded library's own description, for diagnostics.</summary>
        public override string ToString() => "OpenBlasBackend " + Info;
    }
}
