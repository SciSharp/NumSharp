using System;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.Blas
{
    /// <summary>
    ///     A <see cref="DefaultEngine"/> whose matrix products are computed by an external CBLAS
    ///     library instead of NumSharp's managed SIMD GEMM — the engine this package installs.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It overrides exactly two members, and everything else — every other op, dtype, kernel and
    ///     iterator — stays the built-in managed code it inherits:
    ///     </para>
    ///     <list type="bullet">
    ///       <item><see cref="Dot"/> — mirrors NumPy's <c>cblas_matrixproduct</c> (+ the N-D
    ///             <c>dotfunc</c> tail), the dispatcher behind <c>np.dot</c>.</item>
    ///       <item><see cref="MultiplyMatrix"/> — mirrors NumPy's <c>@TYPE@_matmul</c> gufunc, the
    ///             2-D core reached by <c>np.matmul</c>, by <c>@</c>, and by every element of a
    ///             stacked/batched product.</item>
    ///     </list>
    ///     <para>
    ///     Anything the port cannot service — any dtype other than <c>float32</c>/<c>float64</c>, or
    ///     a shape combination outside the ported surface — falls straight through to
    ///     <c>base</c>, so installing this engine can only ever change which of the two computes a
    ///     float matrix product, never whether NumSharp can compute one.
    ///     </para>
    ///     <para>
    ///     <b>Why it exists.</b> Both answers are correct floating-point products, but they sum in
    ///     different orders — NumSharp's managed GEMM differed from NumPy on 94.5 % of the elements
    ///     of a <c>(128,784)@(784,128)</c> float32 product. NumPy's bits ARE its BLAS's bits (for
    ///     f32/f64 mat@mat it always calls cblas, copying non-blasable operands rather than taking
    ///     its own portable loop), and OpenBLAS' multi-accumulator micro-kernels are not reproducible
    ///     by any portable algorithm. Calling the same library through the same routes is the only
    ///     way to agree to the last bit. See <c>docs/GEMM_PARITY.md</c>.
    ///     </para>
    /// </remarks>
    public class BlasEngine : DefaultEngine
    {
        /// <summary>Describes the CBLAS library backing this engine (path, symbols, threads).</summary>
        public string LibraryInfo => Blas.Info;

        /// <inheritdoc/>
        public override NDArray Dot(NDArray left, NDArray right)
        {
            // np.dot has its OWN dispatcher in NumPy (cblas_matrixproduct + the dotfunc iterator
            // tail), which is NOT matmul's — the two disagree on e.g. a strided matrix times a
            // vector — so it is consulted here, ahead of every shape branch in the base method.
            NDArray result;
            if (BlasParity.TryDot(left, right, out result))
                return result;

            return base.Dot(left, right);
        }

        /// <inheritdoc/>
        protected override NDArray MultiplyMatrix(NDArray left, NDArray right, NDArray @out = null)
        {
            // The base method allocates the result and picks a kernel; the parity port has to own
            // the whole product to reproduce the bits, so it runs first and writes into the same
            // output the base method would have returned.
            var result = @out;
            if (result is null)
            {
                if (left.Shape.NDim == 2 && right.Shape.NDim == 2 && left.shape[1] == right.shape[0])
                {
                    var resultType = np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode);
                    if (BlasParity.IsSupported(resultType))
                        result = new NDArray(resultType, Shape.Matrix(left.shape[0], right.shape[1]));
                }

                if (result is null)
                    return base.MultiplyMatrix(left, right, null);
            }

            if (BlasParity.TryMatmul2D(left, right, result))
                return result;

            return base.MultiplyMatrix(left, right, @out);
        }
    }
}
