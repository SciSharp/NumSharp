using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The LU-based <c>np.linalg</c> factorisations (<c>solve</c>/<c>inv</c>/<c>det</c>/
    ///     <c>slogdet</c>), a route-for-route port of NumPy 2.4.2's <c>umath_linalg.cpp</c> gufuncs
    ///     calling the SAME LAPACK the bundled scipy-openblas ships.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy's linalg is a "lite" wrapper that ALWAYS computes in double precision: its
    ///     <c>_commonType</c> resolves the gufunc signature to <c>'d'</c> (or <c>'D'</c> for complex)
    ///     regardless of input width, so a float32 operand is upcast to float64 (exactly),
    ///     factorised with <c>dgesv</c>/<c>dgetrf</c>, and the result cast back to float32. This
    ///     backend reproduces that exactly — the float32 path is <c>astype(double)</c> → the double
    ///     core → <c>astype(single)</c>, which is bit-identical to what NumPy does. Hence only the
    ///     double and cdouble LAPACK routines are ever called.
    ///     </para>
    ///     <para>
    ///     Each 2-D matrix is copied into a fresh COLUMN-MAJOR buffer (LAPACK is Fortran) read through
    ///     the operand's own strides, so every layout — C/F/strided/transposed/reversed/broadcast —
    ///     works; the operand is never mutated. This mirrors NumPy's <c>linearize_matrix</c> with the
    ///     swapped steps that puts the matrix in Fortran order. Stacks are looped element by element,
    ///     the way the gufunc's outer loop does, with the scratch buffers hoisted once per call.
    ///     </para>
    ///     <para>
    ///     Every result buffer is allocated <c>fillZeros:false</c>: the delinearize / per-element
    ///     write below fills every cell, so pre-zeroing is wasted work and, before the LAPACK call,
    ///     faults the fresh buffer's demand-zero pages (the same lever as the matmul/dot fix).
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>
        ///     The per-dtype LAPACK operations, as a struct type argument so the shared cores JIT into
        ///     monomorphic code — the same pattern as <see cref="IBlasType{T}"/> for the products.
        ///     Only double and Complex (cdouble) instantiate: NumPy computes float32 linalg in double.
        /// </summary>
        internal interface ILapackType<T> where T : unmanaged
        {
            T One { get; }
            T Zero { get; }
            T MinusOne { get; }

            /// <summary>Column-major LU factorisation in place. Returns LAPACK <c>info</c>.</summary>
            long Getrf(long m, long n, T* a, long lda, void* ipiv);

            /// <summary>Column-major solve <c>A X = B</c> in place (B ← X). Returns LAPACK <c>info</c>.</summary>
            long Gesv(long n, long nrhs, T* a, long lda, void* ipiv, T* b, long ldb);

            /// <summary>
            ///     Folds the LU factor's diagonal (column-major, <c>m</c> entries at stride
            ///     <c>m+1</c>) into a running sign and log|det|, exactly as NumPy's
            ///     <c>slogdet_from_factored_diagonal</c>.
            /// </summary>
            void FoldDiagonal(T* factored, long m, ref T sign, out double logabsdet);

            /// <summary><c>sign * exp(logabsdet)</c> — NumPy's <c>det_from_slogdet</c>.</summary>
            T DetFromSlogdet(T sign, double logabsdet);

            /// <summary>Column-major Cholesky factorisation in place. Returns LAPACK <c>info</c>.</summary>
            long Potrf(byte uplo, long n, T* a, long lda);

            /// <summary>
            ///     Column-major QR factorisation in place (LAPACK <c>geqrf</c>): <paramref name="a"/> is
            ///     overwritten with the packed R + Householder reflectors, <paramref name="tau"/> filled.
            ///     Call with <paramref name="lwork"/> == -1 to have <paramref name="work"/>[0] receive the
            ///     optimal workspace size. Returns LAPACK <c>info</c>.
            /// </summary>
            long Geqrf(long m, long n, T* a, long lda, T* tau, T* work, long lwork);

            /// <summary>
            ///     Generates the QR factor Q from <c>geqrf</c>'s reflectors (LAPACK <c>orgqr</c>/
            ///     <c>ungqr</c>), in place in <paramref name="a"/>. Same workspace-query protocol as
            ///     <see cref="Geqrf"/>. Returns LAPACK <c>info</c>.
            /// </summary>
            long Orgqr(long m, long n, long k, T* a, long lda, T* tau, T* work, long lwork);

            /// <summary>The optimal workspace size a <c>geqrf</c>/<c>orgqr</c> query left in <c>work[0]</c>.</summary>
            long OptimalLwork(T* work);

            /// <summary>
            ///     Column-major SVD (LAPACK <c>gesdd</c>): <paramref name="jobz"/> is 'N'/'S'/'A',
            ///     <paramref name="a"/> is overwritten, and the singular values <paramref name="s"/> —
            ///     ALWAYS the real basetype (double) — plus <paramref name="u"/>/<paramref name="vt"/>
            ///     are filled. <paramref name="rwork"/> is used only by the complex routine (null for
            ///     real). Same workspace-query protocol as <see cref="Geqrf"/>. Returns LAPACK <c>info</c>.
            /// </summary>
            long Gesdd(byte jobz, long m, long n, T* a, long lda, double* s, T* u, long ldu, T* vt, long ldvt,
                T* work, long lwork, double* rwork, void* iwork);

            /// <summary>
            ///     Column-major minimum-norm least squares (LAPACK <c>gelsd</c>): solves
            ///     <c>min ||A x - B||</c> in place (B ← X), filling the singular values
            ///     <paramref name="s"/> (real basetype) and the effective <paramref name="rank"/>.
            ///     <paramref name="rwork"/> is used only by the complex routine. The workspace query
            ///     (<paramref name="lwork"/> == -1) writes both <c>work[0]</c> and <c>iwork[0]</c>.
            ///     Returns LAPACK <c>info</c>.
            /// </summary>
            long Gelsd(long m, long n, long nrhs, T* a, long lda, T* b, long ldb, double* s, double rcond,
                out long rank, T* work, long lwork, double* rwork, void* iwork);

            /// <summary>
            ///     Squared magnitude <c>|v|²</c> as a real — NumPy's <c>abs2</c>, used to form
            ///     <c>lstsq</c>'s residuals as the squared 2-norm of each excess column.
            /// </summary>
            double Abs2(T v);

            /// <summary>
            ///     Column-major symmetric/Hermitian eigendecomposition (LAPACK <c>syevd</c>/<c>heevd</c>):
            ///     <paramref name="a"/> is overwritten with the orthonormal eigenvectors (columns) when
            ///     <paramref name="jobz"/> is 'V', and the eigenvalues — ALWAYS the real basetype
            ///     (double), in ASCENDING order — fill <paramref name="w"/>. <paramref name="rwork"/> is
            ///     used only by the complex routine (null for real). Same workspace-query protocol as
            ///     <see cref="Geqrf"/>, extended: <paramref name="lwork"/>/<paramref name="lrwork"/>/
            ///     <paramref name="liwork"/> == -1 writes <c>work[0]</c>/<c>rwork[0]</c>/<c>iwork[0]</c>.
            ///     Returns LAPACK <c>info</c>.
            /// </summary>
            long Syevd(byte jobz, byte uplo, long n, T* a, long lda, double* w, T* work, long lwork,
                double* rwork, long lrwork, void* iwork, long liwork);

            /// <summary>
            ///     Column-major general eigendecomposition (LAPACK <c>geev</c>), producing ALWAYS-COMPLEX
            ///     outputs: eigenvalues <paramref name="w"/> (n) and — when <paramref name="jobvr"/> is
            ///     'V' — the right eigenvectors <paramref name="vr"/> (column-major n×n, LD = n). The real
            ///     routine (<c>dgeev</c>) writes split real parts into the caller-hoisted
            ///     <paramref name="wr"/>/<paramref name="wi"/>/<paramref name="vrr"/> scratch and this
            ///     method assembles the complex result exactly as NumPy's <c>mk_complex_array</c> /
            ///     <c>mk_geev_complex_eigenvectors</c>; the complex routine (<c>zgeev</c>) writes
            ///     <paramref name="w"/>/<paramref name="vr"/> directly and uses <paramref name="rwork"/>
            ///     (2n). Left eigenvectors are never computed (jobvl='N'). A workspace query
            ///     (<paramref name="lwork"/> == -1) writes <c>work[0]</c> and SKIPS the assembly. Returns
            ///     LAPACK <c>info</c>.
            /// </summary>
            long Geev(byte jobvr, long n, T* a, long lda, Complex* w, Complex* vr, long ldvr,
                T* work, long lwork, double* wr, double* wi, T* vrr, double* rwork);
        }

        /// <summary>float64 LAPACK bindings — NumPy's real <c>'d'</c> signature.</summary>
        internal readonly struct DoubleLapack : ILapackType<double>
        {
            public double One => 1.0;
            public double Zero => 0.0;
            public double MinusOne => -1.0;

            public long Getrf(long m, long n, double* a, long lda, void* ipiv)
                => OpenBlasNative.Dgetrf(m, n, a, lda, ipiv);

            public long Gesv(long n, long nrhs, double* a, long lda, void* ipiv, double* b, long ldb)
                => OpenBlasNative.Dgesv(n, nrhs, a, lda, ipiv, b, ldb);

            public void FoldDiagonal(double* factored, long m, ref double sign, out double logabsdet)
            {
                double s = sign, acc = 0.0;
                for (long i = 0; i < m; i++)
                {
                    double d = factored[i * (m + 1)];
                    if (d < 0.0)
                    {
                        s = -s;
                        d = -d;
                    }

                    acc += Math.Log(d);
                }

                sign = s;
                logabsdet = acc;
            }

            public double DetFromSlogdet(double sign, double logabsdet) => sign * Math.Exp(logabsdet);

            public long Potrf(byte uplo, long n, double* a, long lda)
                => OpenBlasNative.Dpotrf(uplo, n, a, lda);

            public long Geqrf(long m, long n, double* a, long lda, double* tau, double* work, long lwork)
                => OpenBlasNative.Dgeqrf(m, n, a, lda, tau, work, lwork);

            public long Orgqr(long m, long n, long k, double* a, long lda, double* tau, double* work, long lwork)
                => OpenBlasNative.Dorgqr(m, n, k, a, lda, tau, work, lwork);

            public long OptimalLwork(double* work) => (long)work[0];

            // dgesdd/dgelsd have no RWORK (real routines) — the parameter is ignored.
            public long Gesdd(byte jobz, long m, long n, double* a, long lda, double* s, double* u, long ldu,
                double* vt, long ldvt, double* work, long lwork, double* rwork, void* iwork)
                => OpenBlasNative.Dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork);

            public long Gelsd(long m, long n, long nrhs, double* a, long lda, double* b, long ldb, double* s,
                double rcond, out long rank, double* work, long lwork, double* rwork, void* iwork)
                => OpenBlasNative.Dgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, out rank, work, lwork, iwork);

            public double Abs2(double v) => v * v;

            // dsyevd has no RWORK (real routine) — the parameter is ignored.
            public long Syevd(byte jobz, byte uplo, long n, double* a, long lda, double* w, double* work,
                long lwork, double* rwork, long lrwork, void* iwork, long liwork)
                => OpenBlasNative.Dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork);

            public long Geev(byte jobvr, long n, double* a, long lda, Complex* w, Complex* vr, long ldvr,
                double* work, long lwork, double* wr, double* wi, double* vrr, double* rwork)
            {
                // jobvl='N' — left vectors are never computed, so VL is not referenced; alias VR's buffer
                // for it exactly as NumPy does (vlr == vrr when jobvl != 'V').
                long info = OpenBlasNative.Dgeev((byte)'N', jobvr, n, a, lda, wr, wi, vrr, ldvr, vrr, ldvr, work, lwork);
                if (lwork < 0 || info != 0)
                    return info; // workspace query (nothing computed) or failure — no assembly

                // process_geev_results (scalar_trait): translate dgeev's real WR/WI/VRR into the
                // always-complex W/VR the gufunc returns.
                for (long i = 0; i < n; i++)
                    w[i] = new Complex(wr[i], wi[i]); // mk_complex_array
                if (jobvr == (byte)'V')
                    AssembleGeevEigenvectors(vr, vrr, wi, n); // mk_geev_complex_eigenvectors

                return info;
            }
        }

        /// <summary>complex128 LAPACK bindings — NumPy's <c>'D'</c> signature (cdouble).</summary>
        internal readonly struct ComplexLapack : ILapackType<Complex>
        {
            public Complex One => Complex.One;
            public Complex Zero => Complex.Zero;
            public Complex MinusOne => new Complex(-1.0, 0.0);

            public long Getrf(long m, long n, Complex* a, long lda, void* ipiv)
                => OpenBlasNative.Zgetrf(m, n, (double*)a, lda, ipiv);

            public long Gesv(long n, long nrhs, Complex* a, long lda, void* ipiv, Complex* b, long ldb)
                => OpenBlasNative.Zgesv(n, nrhs, (double*)a, lda, ipiv, (double*)b, ldb);

            public void FoldDiagonal(Complex* factored, long m, ref Complex sign, out double logabsdet)
            {
                // NumPy: abs = npy_cabs(z); sign *= z/abs; logdet += log(abs). The product is its own
                // `mult` helper (the textbook complex product), spelled out here so the rounding
                // matches rather than trusting an operator's formula choice.
                double sr = sign.Real, si = sign.Imaginary, acc = 0.0;
                for (long i = 0; i < m; i++)
                {
                    Complex z = factored[i * (m + 1)];
                    // NumPy's npy_cabs is a straight `hypot(re, im)` on the CRT; Complex.Abs uses
                    // .NET's own scaled hypot and differs by ~1 ULP, which shows in det/slogdet.
                    double abs = OpenBlasNative.Hypot(z.Real, z.Imaginary);
                    double er = z.Real / abs, ei = z.Imaginary / abs;
                    double nr = sr * er - si * ei;
                    double ni = sr * ei + si * er;
                    sr = nr;
                    si = ni;
                    acc += Math.Log(abs);
                }

                sign = new Complex(sr, si);
                logabsdet = acc;
            }

            public Complex DetFromSlogdet(Complex sign, double logabsdet)
            {
                // mult(sign, (exp(logabsdet), 0)) — the imaginary factor is zero.
                double e = Math.Exp(logabsdet);
                return new Complex(sign.Real * e, sign.Imaginary * e);
            }

            // complex*16 matrices/vectors are interleaved [Real, Imaginary] doubles — the exact
            // System.Numerics.Complex layout — so a Complex* casts to the double* LAPACK wants.
            public long Potrf(byte uplo, long n, Complex* a, long lda)
                => OpenBlasNative.Zpotrf(uplo, n, (double*)a, lda);

            public long Geqrf(long m, long n, Complex* a, long lda, Complex* tau, Complex* work, long lwork)
                => OpenBlasNative.Zgeqrf(m, n, (double*)a, lda, (double*)tau, (double*)work, lwork);

            public long Orgqr(long m, long n, long k, Complex* a, long lda, Complex* tau, Complex* work, long lwork)
                => OpenBlasNative.Zungqr(m, n, k, (double*)a, lda, (double*)tau, (double*)work, lwork);

            public long OptimalLwork(Complex* work) => (long)work[0].Real;

            // S/RWORK are the real basetype (double); the interleaved-double cast is the same as above.
            public long Gesdd(byte jobz, long m, long n, Complex* a, long lda, double* s, Complex* u, long ldu,
                Complex* vt, long ldvt, Complex* work, long lwork, double* rwork, void* iwork)
                => OpenBlasNative.Zgesdd(jobz, m, n, (double*)a, lda, s, (double*)u, ldu, (double*)vt, ldvt,
                    (double*)work, lwork, rwork, iwork);

            public long Gelsd(long m, long n, long nrhs, Complex* a, long lda, Complex* b, long ldb, double* s,
                double rcond, out long rank, Complex* work, long lwork, double* rwork, void* iwork)
                => OpenBlasNative.Zgelsd(m, n, nrhs, (double*)a, lda, (double*)b, ldb, s, rcond, out rank,
                    (double*)work, lwork, rwork, iwork);

            public double Abs2(Complex v) => v.Real * v.Real + v.Imaginary * v.Imaginary;

            // zheevd's eigenvalue vector W and RWORK are the real basetype (double); the interleaved-double
            // cast is the same as the other complex routines.
            public long Syevd(byte jobz, byte uplo, long n, Complex* a, long lda, double* w, Complex* work,
                long lwork, double* rwork, long lrwork, void* iwork, long liwork)
                => OpenBlasNative.Zheevd(jobz, uplo, n, (double*)a, lda, w, (double*)work, lwork, rwork, lrwork, iwork, liwork);

            public long Geev(byte jobvr, long n, Complex* a, long lda, Complex* w, Complex* vr, long ldvr,
                Complex* work, long lwork, double* wr, double* wi, Complex* vrr, double* rwork)
                // zgeev writes W and VR (complex) directly — no real->complex assembly, so the wr/wi/vrr
                // scratch is unused here. VL is not referenced (jobvl='N'); alias VR's buffer as NumPy does.
                => OpenBlasNative.Zgeev((byte)'N', jobvr, n, (double*)a, lda, (double*)w, (double*)vr, ldvr,
                    (double*)vr, ldvr, (double*)work, lwork, rwork);
        }

        // ----------------------------------------------------------------------------------------
        //  Entry points (dtype dispatch). The engine only ever hands these Single/Double/Complex —
        //  linalg's CommonType rejects Half/Decimal/Char up front and widens int/bool to Double.
        // ----------------------------------------------------------------------------------------

        /// <summary>Parity entry point for <c>np.linalg.inv</c> (LAPACK <c>gesv</c> vs the identity).</summary>
        internal static bool TryInv(NDArray a, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    result = InvCore<double, DoubleLapack>(a);
                    return true;
                case NPTypeCode.Complex:
                    result = InvCore<Complex, ComplexLapack>(a);
                    return true;
                case NPTypeCode.Single:
                    result = InvCore<double, DoubleLapack>(a.astype(NPTypeCode.Double)).astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>Parity entry point for <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
        internal static bool TrySolve(NDArray a, NDArray b, bool oneDimensionalRhs, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    result = SolveCore<double, DoubleLapack>(a, b, oneDimensionalRhs);
                    return true;
                case NPTypeCode.Complex:
                    result = SolveCore<Complex, ComplexLapack>(a, b, oneDimensionalRhs);
                    return true;
                case NPTypeCode.Single:
                    result = SolveCore<double, DoubleLapack>(
                        a.astype(NPTypeCode.Double), b.astype(NPTypeCode.Double), oneDimensionalRhs)
                        .astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>Parity entry point for <c>np.linalg.det</c> (LAPACK <c>getrf</c>).</summary>
        internal static bool TryDet(NDArray a, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    result = DetSlogdetCore<double, DoubleLapack>(a, wantDet: true).det;
                    return true;
                case NPTypeCode.Complex:
                    result = DetSlogdetCore<Complex, ComplexLapack>(a, wantDet: true).det;
                    return true;
                case NPTypeCode.Single:
                    result = DetSlogdetCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), wantDet: true)
                        .det.astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>Parity entry point for <c>np.linalg.slogdet</c> (LAPACK <c>getrf</c>).</summary>
        internal static bool TrySlogdet(NDArray a, out NDArray sign, out NDArray logabsdet)
        {
            sign = null;
            logabsdet = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (_, sign, logabsdet) = DetSlogdetCore<double, DoubleLapack>(a, wantDet: false);
                    return true;
                case NPTypeCode.Complex:
                    // sign is complex128, logabsdet is the real basetype (float64) — NumPy's real_t.
                    (_, sign, logabsdet) = DetSlogdetCore<Complex, ComplexLapack>(a, wantDet: false);
                    return true;
                case NPTypeCode.Single:
                    (_, var s, var l) = DetSlogdetCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), wantDet: false);
                    sign = s.astype(NPTypeCode.Single);       // result_t = single
                    logabsdet = l.astype(NPTypeCode.Single);  // real_t   = single
                    return true;
                default:
                    return false;
            }
        }

        // ----------------------------------------------------------------------------------------
        //  Generic cores (T ∈ {double, Complex}).
        // ----------------------------------------------------------------------------------------

        private static NDArray InvCore<T, TOps>(NDArray a)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1];

            // fillZeros:false: Delinearize writes every cell, so pre-zeroing is wasted (see the file header note).
            var result = new NDArray(InfoOf<T>.NPTypeCode, new Shape((long[])shape.dimensions.Clone()), fillZeros: false);
            if (result.size == 0)
                return result; // (…,0,0) → empty inverse, matching NumPy

            int nb = nd - 2;
            long sr = shape.strides[nd - 2], sc = shape.strides[nd - 1];
            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            long block = m * m;
            T* pa = (T*)a.Address;
            T* po = (T*)result.Address + result.Shape.offset;
            T* bufA = Alloc<T>(m * m);
            T* bufB = Alloc<T>(m * m);
            void* ipiv = AllocIpiv(m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize(pa + aoff, sr, sc, bufA, m, m);
                    IdentityColMajor(bufB, m, ops.One);
                    long info = ops.Gesv(m, m, bufA, m, ipiv, bufB, m);
                    if (info > 0)
                        throw new LinAlgError("Singular matrix");

                    Delinearize(bufB, m, m, po + e * block);
                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(bufB);
                FreeIpiv(ipiv);
            }

            return result;
        }

        private static NDArray SolveCore<T, TOps>(NDArray a, NDArray b, bool oneDim)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var sa = a.Shape;
            var sb = b.Shape;
            int nda = sa.NDim, ndb = sb.NDim;
            long m = sa.dimensions[nda - 1];

            int aNb = nda - 2;
            long aSr = sa.strides[nda - 2], aSc = sa.strides[nda - 1];

            // RHS core: a bare vector (solve1: (m)->(m)) or a matrix column block (solve: (m,k)->(m,k)).
            long k;
            int bNb;
            long bSr, bSc;
            long[] trailing;
            if (oneDim)
            {
                k = 1;
                bNb = 0;                 // b is exactly (m,), it has no batch dims
                bSr = sb.strides[0];
                bSc = 0;
                trailing = new[] { m };
            }
            else
            {
                k = sb.dimensions[ndb - 1];
                bNb = ndb - 2;
                bSr = sb.strides[ndb - 2];
                bSc = sb.strides[ndb - 1];
                trailing = new[] { m, k };
            }

            if (!TryBroadcastBatch(sa.dimensions, sa.strides, aNb, sb.dimensions, sb.strides, bNb,
                    out var batchShape, out var aBatchStr, out var bBatchStr))
            {
                throw new IncorrectShapeException(
                    $"np.linalg.solve: the leading dimensions of a {DimsText(sa.dimensions, aNb)} and " +
                    $"b {DimsText(sb.dimensions, bNb)} could not be broadcast together.");
            }

            int nb = batchShape.Length;
            var outDims = new long[nb + trailing.Length];
            long count = 1;
            for (int i = 0; i < nb; i++)
            {
                outDims[i] = batchShape[i];
                count *= batchShape[i];
            }

            for (int i = 0; i < trailing.Length; i++)
                outDims[nb + i] = trailing[i];

            var result = new NDArray(InfoOf<T>.NPTypeCode, new Shape(outDims), fillZeros: false);
            if (result.size == 0)
                return result;

            long block = m * k;
            T* pa = (T*)a.Address;
            T* pb = (T*)b.Address;
            T* po = (T*)result.Address + result.Shape.offset;
            T* bufA = Alloc<T>(m * m);
            T* bufB = Alloc<T>(m * k);
            void* ipiv = AllocIpiv(m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = sa.offset, boff = sb.offset;
                    for (int i = 0; i < nb; i++)
                    {
                        aoff += coord[i] * aBatchStr[i];
                        boff += coord[i] * bBatchStr[i];
                    }

                    Linearize(pa + aoff, aSr, aSc, bufA, m, m);
                    Linearize(pb + boff, bSr, bSc, bufB, m, k);
                    long info = ops.Gesv(m, k, bufA, m, ipiv, bufB, m);
                    if (info > 0)
                        throw new LinAlgError("Singular matrix");

                    Delinearize(bufB, m, k, po + e * block);
                    AdvanceCoord(coord, batchShape, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(bufB);
                FreeIpiv(ipiv);
            }

            return result;
        }

        private static (NDArray det, NDArray sign, NDArray logdet) DetSlogdetCore<T, TOps>(NDArray a, bool wantDet)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1];
            int nb = nd - 2;

            var batchDims = new long[nb];
            long count = 1;
            for (int i = 0; i < nb; i++)
            {
                batchDims[i] = shape.dimensions[i];
                count *= batchDims[i];
            }

            var batchShape = nb == 0 ? Shape.Scalar : new Shape(batchDims);

            NDArray det = null, sign = null, logdet = null;
            T* pdet = null, psign = null;
            double* plog = null;
            if (wantDet)
            {
                det = new NDArray(InfoOf<T>.NPTypeCode, batchShape, fillZeros: false);
                pdet = (T*)det.Address + det.Shape.offset;
            }
            else
            {
                sign = new NDArray(InfoOf<T>.NPTypeCode, batchShape, fillZeros: false);
                logdet = new NDArray(NPTypeCode.Double, batchShape, fillZeros: false);
                psign = (T*)sign.Address + sign.Shape.offset;
                plog = (double*)logdet.Address + logdet.Shape.offset;
            }

            long sr = shape.strides[nd - 2], sc = shape.strides[nd - 1];
            T* pa = (T*)a.Address;
            T* bufA = Alloc<T>(Math.Max(m * m, 1));
            void* ipiv = AllocIpiv(Math.Max(m, 1));
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize(pa + aoff, sr, sc, bufA, m, m);
                    FactorAndFold<T, TOps>(bufA, m, ipiv, out T s, out double ld);
                    if (wantDet)
                        pdet[e] = ops.DetFromSlogdet(s, ld);
                    else
                    {
                        psign[e] = s;
                        plog[e] = ld;
                    }

                    AdvanceCoord(coord, batchDims, nb);
                }
            }
            finally
            {
                Free(bufA);
                FreeIpiv(ipiv);
            }

            return (det, sign, logdet);
        }

        /// <summary>
        ///     One matrix's <c>getrf</c> + sign/log|det| fold — NumPy's <c>slogdet_single_element</c>.
        ///     An empty matrix is the empty product (sign 1, logdet 0); a singular one (info &gt; 0)
        ///     is (sign 0, logdet -inf), which det/slogdet report rather than raising.
        /// </summary>
        private static void FactorAndFold<T, TOps>(T* colBuf, long m, void* ipiv, out T sign, out double logdet)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            if (m == 0)
            {
                sign = ops.One;
                logdet = 0.0;
                return;
            }

            long info = ops.Getrf(m, m, colBuf, m, ipiv);
            if (info != 0)
            {
                sign = ops.Zero;
                logdet = double.NegativeInfinity;
                return;
            }

            int changes = 0;
            for (long i = 0; i < m; i++)
                if (OpenBlasNative.ReadPivot(ipiv, i) != i + 1) // fortran pivots are 1-based
                    changes++;

            sign = (changes & 1) == 1 ? ops.MinusOne : ops.One;
            ops.FoldDiagonal(colBuf, m, ref sign, out logdet);
        }

        // ----------------------------------------------------------------------------------------
        //  Small helpers.
        // ----------------------------------------------------------------------------------------

        /// <summary>Copies a strided row-major matrix into a column-major buffer (LD = rows).</summary>
        private static void Linearize<T>(T* src, long sr, long sc, T* dst, long rows, long cols)
            where T : unmanaged
        {
            for (long c = 0; c < cols; c++)
            {
                T* col = dst + c * rows;
                T* s = src + c * sc;
                for (long r = 0; r < rows; r++)
                    col[r] = s[r * sr];
            }
        }

        /// <summary>Copies a column-major buffer (LD = rows) into a contiguous row-major block.</summary>
        private static void Delinearize<T>(T* colSrc, long rows, long cols, T* dstRowMajor)
            where T : unmanaged
        {
            for (long r = 0; r < rows; r++)
            {
                T* row = dstRowMajor + r * cols;
                for (long c = 0; c < cols; c++)
                    row[c] = colSrc[c * rows + r];
            }
        }

        /// <summary>Writes a column-major identity into <paramref name="buf"/> (NumPy's identity_matrix).</summary>
        private static void IdentityColMajor<T>(T* buf, long n, T one) where T : unmanaged
        {
            Zero(buf, n * n);
            for (long i = 0; i < n; i++)
                buf[i * (n + 1)] = one;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AdvanceCoord(long[] coord, long[] dims, int nb)
        {
            for (int i = nb - 1; i >= 0; i--)
            {
                if (++coord[i] < dims[i])
                    return;
                coord[i] = 0;
            }
        }

        /// <summary>
        ///     Right-aligned broadcast of two operands' BATCH (leading) dimensions, producing the
        ///     batch shape and each operand's per-axis batch stride (0 where it broadcasts). False on
        ///     an incompatible pair.
        /// </summary>
        private static bool TryBroadcastBatch(
            long[] aDims, long[] aStr, int aNb,
            long[] bDims, long[] bStr, int bNb,
            out long[] batchShape, out long[] aBatchStr, out long[] bBatchStr)
        {
            int nb = Math.Max(aNb, bNb);
            batchShape = new long[nb];
            aBatchStr = new long[nb];
            bBatchStr = new long[nb];
            for (int i = 0; i < nb; i++)
            {
                int ai = aNb - nb + i;
                int bi = bNb - nb + i;
                long ad = ai >= 0 ? aDims[ai] : 1;
                long bd = bi >= 0 ? bDims[bi] : 1;

                long dim;
                if (ad == bd) dim = ad;
                else if (ad == 1) dim = bd;
                else if (bd == 1) dim = ad;
                else return false;

                batchShape[i] = dim;
                aBatchStr[i] = (ai >= 0 && ad != 1) ? aStr[ai] : 0;
                bBatchStr[i] = (bi >= 0 && bd != 1) ? bStr[bi] : 0;
            }

            return true;
        }

        private static string DimsText(long[] dims, int nb)
            => "(" + string.Join(",", new ArraySegment<long>(dims, 0, nb)) + ")";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void* AllocIpiv(long m)
            => NativeMemory.Alloc((nuint)Math.Max(m, 1), (nuint)OpenBlasNative.FortranIntSize);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FreeIpiv(void* p)
        {
            if (p != null)
                NativeMemory.Free(p);
        }
    }
}
