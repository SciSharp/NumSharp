using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     <c>np.linalg.cholesky</c> — a route-for-route port of NumPy 2.4.2's <c>cholesky_lo</c> /
    ///     <c>cholesky_up</c> gufuncs (<c>umath_linalg.cpp</c>) calling the SAME LAPACK <c>potrf</c> the
    ///     bundled scipy-openblas ships.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like all of NumPy's linalg this computes in double precision regardless of input width: a
    ///     float32 operand is upcast to float64 (exactly), factorised with <c>dpotrf</c>, and the result
    ///     cast back to float32 — bit-identical to NumPy's <c>gufunc(a).astype(float32)</c>. Only the
    ///     double and cdouble routines are ever called.
    ///     </para>
    ///     <para>
    ///     Each 2-D matrix is copied into a fresh COLUMN-MAJOR buffer read through the operand's own
    ///     strides (NumPy's <c>linearize_matrix</c> with the swapped steps), so every layout works and
    ///     the operand is never mutated. <c>potrf</c> reads only the lower (<c>upper=false</c>) or upper
    ///     triangle; the opposite triangle of the result is then zeroed, exactly as NumPy's
    ///     <c>zero_upper_triangle</c> / <c>zero_lower_triangle</c>. Stacks loop element by element with
    ///     the scratch buffer hoisted once, the way the gufunc's outer loop does.
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>Parity entry point for <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).</summary>
        internal static bool TryCholesky(NDArray a, bool upper, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    result = CholeskyCore<double, DoubleLapack>(a, upper);
                    return true;
                case NPTypeCode.Complex:
                    result = CholeskyCore<Complex, ComplexLapack>(a, upper);
                    return true;
                case NPTypeCode.Single:
                    result = CholeskyCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), upper)
                        .astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        private static NDArray CholeskyCore<T, TOps>(NDArray a, bool upper)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1]; // square: rows == cols == m (validated by AssertStackedSquare)

            // fillZeros:false: DelinearizeTriangle writes every cell (factor + default), so pre-zeroing is wasted.
            var result = new NDArray(InfoOf<T>.NPTypeCode, new Shape((long[])shape.dimensions.Clone()), fillZeros: false);
            if (result.size == 0)
                return result; // (…,0,0) → empty factor, matching NumPy

            int nb = nd - 2;
            long sr = shape.strides[nd - 2], sc = shape.strides[nd - 1];
            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            long lda = Math.Max(m, 1);
            long block = m * m;
            byte uplo = upper ? (byte)'U' : (byte)'L';
            T* pa = (T*)a.Address;
            T* po = (T*)result.Address + result.Shape.offset;
            T* buf = Alloc<T>(m * m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize(pa + aoff, sr, sc, buf, m, m);
                    long info = ops.Potrf(uplo, m, buf, lda);
                    if (info > 0)
                        throw new LinAlgError("Matrix is not positive definite");

                    DelinearizeTriangle(buf, m, lda, upper, po + e * block);
                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(buf);
            }

            return result;
        }

        /// <summary>
        ///     Copies a column-major factor into a contiguous row-major block, keeping only the kept
        ///     triangle and zeroing the other — NumPy's <c>zero_*_triangle</c> then <c>delinearize</c>
        ///     fused into one pass. Lower keeps <c>c &lt;= r</c>, upper keeps <c>c &gt;= r</c>.
        /// </summary>
        private static void DelinearizeTriangle<T>(T* colSrc, long n, long lda, bool upper, T* dstRowMajor)
            where T : unmanaged
        {
            for (long r = 0; r < n; r++)
            {
                T* row = dstRowMajor + r * n;
                for (long c = 0; c < n; c++)
                {
                    bool keep = upper ? c >= r : c <= r;
                    row[c] = keep ? colSrc[r + c * lda] : default;
                }
            }
        }
    }
}
