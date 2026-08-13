using System;
using NumSharp;
using NumSharp.Backends;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     Opt-in byte-parity backend for <c>np.dot</c> / <c>np.matmul</c>: a route-for-route port of
    ///     NumPy 2.4.2's matrix-product dispatchers that calls the SAME CBLAS binary NumPy calls.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy does not compute a matrix product with any portable algorithm you can re-derive.
    ///     For float32/float64 it hands the work to cblas, and scipy-openblas' <c>sgemm</c> sums each
    ///     dot product in an arch-specific multi-accumulator register scheme whose bits match neither
    ///     a sequential mul+add chain nor a sequential FMA chain, and which changes with the BLAS
    ///     thread count. The only way to reproduce those bits is to call that binary — which is what
    ///     this backend does. It is off by default; NumSharp's own SIMD GEMM stays the fast path.
    ///     </para>
    ///     <para>
    ///     Ported from <c>numpy/_core/src/umath/matmul.c.src</c> (<c>@TYPE@_matmul</c>, the gufunc
    ///     behind <c>np.matmul</c> and <c>@</c>) and <c>numpy/_core/src/common/cblasfuncs.c</c>
    ///     (<c>cblas_matrixproduct</c>, behind <c>np.dot</c>) — two genuinely different dispatchers
    ///     that agree bit-for-bit on nearly every input but not all of them, so both are mirrored.
    ///     </para>
    ///     <para>
    ///     One deliberate change of units: NumPy's strides are in BYTES, NumSharp's in ELEMENTS.
    ///     Every ported predicate below works in elements, which is the same logic with
    ///     <c>itemsize == 1</c> (NumPy's <c>stride % itemsize</c> guards are then vacuous — an
    ///     element stride is aligned by construction).
    ///     </para>
    /// </remarks>
    internal static unsafe partial class BlasParity
    {
        /// <summary>
        ///     A 2-D operand as the dispatchers see it: base pointer plus per-axis element strides.
        ///     Mirrors the <c>(ip, is_m, is_n, dm, dn)</c> argument groups of matmul.c.src.
        /// </summary>
        internal struct Mat<T> where T : unmanaged
        {
            public T* Data;
            public long S0;
            public long S1;
            public long D0;
            public long D1;
        }

        /// <summary>
        ///     Per-dtype cblas entry points, as a struct type argument so the shared dispatchers below
        ///     JIT into monomorphic code — the C# equivalent of NumPy's <c>@prefix@</c> template
        ///     expansion (one source, one instantiation per dtype).
        /// </summary>
        internal interface IBlasType<T> where T : unmanaged
        {
            void Gemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
                long m, long n, long k, T* a, long lda, T* b, long ldb, T* c, long ldc);

            void Gemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
                T* a, long lda, T* x, long incX, T* y, long incY);

            void Syrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
                long n, long k, T* a, long lda, T* c, long ldc);

            void Axpy(long n, T alpha, T* x, long incX, T* y, long incY);

            /// <summary>
            ///     NumPy's <c>@name@_dot</c> from arraytypes.c.src: chunked cblas <c>?dot</c> summed in
            ///     a <c>double</c> "for stability", or a sequential same-precision loop when a stride
            ///     is negative / not a whole number of elements (<c>blas_stride</c> returned 0).
            /// </summary>
            void Dot(T* ip1, long is1, T* ip2, long is2, T* op, long n);
        }

        /// <summary>float32 cblas bindings — NumPy's <c>#prefix = s#</c> expansion.</summary>
        internal readonly struct SingleBlas : IBlasType<float>
        {
            public void Gemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
                long m, long n, long k, float* a, long lda, float* b, long ldb, float* c, long ldc)
                => CBlasNative.Sgemm(order, transA, transB, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);

            public void Gemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
                float* a, long lda, float* x, long incX, float* y, long incY)
                => CBlasNative.Sgemv(order, trans, m, n, 1.0f, a, lda, x, incX, 0.0f, y, incY);

            public void Syrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
                long n, long k, float* a, long lda, float* c, long ldc)
                => CBlasNative.Ssyrk(order, uplo, trans, n, k, 1.0f, a, lda, 0.0f, c, ldc);

            public void Axpy(long n, float alpha, float* x, long incX, float* y, long incY)
                => CBlasNative.Saxpy(n, alpha, x, incX, y, incY);

            public void Dot(float* ip1, long is1, float* ip2, long is2, float* op, long n)
            {
                long is1b = BlasStride(is1), is2b = BlasStride(is2);
                if (is1b != 0 && is2b != 0)
                {
                    double sum = 0.0; // double for stability
                    long chunkMax = CBlasNative.CBlasChunk;
                    while (n > 0)
                    {
                        long chunk = n < chunkMax ? n : chunkMax;
                        sum += CBlasNative.Sdot(chunk, ip1, is1b, ip2, is2b);
                        ip1 += chunk * is1;
                        ip2 += chunk * is2;
                        n -= chunk;
                    }

                    *op = (float)sum;
                }
                else
                {
                    float sum = 0f;
                    for (long i = 0; i < n; i++, ip1 += is1, ip2 += is2)
                        sum += *ip1 * *ip2;
                    *op = sum;
                }
            }
        }

        /// <summary>float64 cblas bindings — NumPy's <c>#prefix = d#</c> expansion.</summary>
        internal readonly struct DoubleBlas : IBlasType<double>
        {
            public void Gemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
                long m, long n, long k, double* a, long lda, double* b, long ldb, double* c, long ldc)
                => CBlasNative.Dgemm(order, transA, transB, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);

            public void Gemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
                double* a, long lda, double* x, long incX, double* y, long incY)
                => CBlasNative.Dgemv(order, trans, m, n, 1.0, a, lda, x, incX, 0.0, y, incY);

            public void Syrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
                long n, long k, double* a, long lda, double* c, long ldc)
                => CBlasNative.Dsyrk(order, uplo, trans, n, k, 1.0, a, lda, 0.0, c, ldc);

            public void Axpy(long n, double alpha, double* x, long incX, double* y, long incY)
                => CBlasNative.Daxpy(n, alpha, x, incX, y, incY);

            public void Dot(double* ip1, long is1, double* ip2, long is2, double* op, long n)
            {
                long is1b = BlasStride(is1), is2b = BlasStride(is2);
                if (is1b != 0 && is2b != 0)
                {
                    double sum = 0.0;
                    long chunkMax = CBlasNative.CBlasChunk;
                    while (n > 0)
                    {
                        long chunk = n < chunkMax ? n : chunkMax;
                        sum += CBlasNative.Ddot(chunk, ip1, is1b, ip2, is2b);
                        ip1 += chunk * is1;
                        ip2 += chunk * is2;
                        n -= chunk;
                    }

                    *op = sum;
                }
                else
                {
                    double sum = 0.0;
                    for (long i = 0; i < n; i++, ip1 += is1, ip2 += is2)
                        sum += *ip1 * *ip2;
                    *op = sum;
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>blas_stride</c> (npy_cblas.h): the BLAS <c>inc</c> for a stride, or 0 when
        ///     BLAS cannot express it (negative or not a whole number of elements) and the caller
        ///     must fall back. In element units the divisibility test is vacuous.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long BlasStride(long elementStride)
            => elementStride > 0 && elementStride <= CBlasNative.BlasMaxSize ? elementStride : 0;

        /// <summary>
        ///     NumPy's <c>is_blasable2d</c> (matmul.c.src), in element units:
        ///     "1. Strides must not alias or overlap. 2. The faster (second) axis must be contiguous.
        ///     3. The slower (first) axis stride, in unit steps, must be larger than the faster axis
        ///     dimension."
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBlasable2d(long stride1, long stride2, long d1, long d2)
        {
            if (stride2 != 1)
                return false;

            return stride1 >= d2 && stride1 <= CBlasNative.BlasMaxSize;
        }

        /// <summary>
        ///     Port of <c>@name@_matrix_copy</c> (matmul.c.src) — the copy NumPy makes when an operand
        ///     is not blasable, walking the source with its real strides.
        /// </summary>
        internal static void MatrixCopy<T>(bool transpose, T* ip, long isM, long isN,
            T* op, long osM, long osN, long dm, long dn) where T : unmanaged
        {
            long m, n, ib, ob;

            if (transpose)
            {
                ib = isM * dm;
                ob = osM * dm;

                for (n = 0; n < dn; n++)
                {
                    for (m = 0; m < dm; m++)
                    {
                        *op = *ip;
                        ip += isM;
                        op += osM;
                    }

                    ip += isN - ib;
                    op += osN - ob;
                }

                return;
            }

            ib = isN * dn;
            ob = osN * dn;

            for (m = 0; m < dm; m++)
            {
                for (n = 0; n < dn; n++)
                {
                    *op = *ip;
                    ip += isN;
                    op += osN;
                }

                ip += isM - ib;
                op += osM - ob;
            }
        }

        /// <summary>
        ///     Port of <c>@TYPE@_matmul_inner_noblas</c> (matmul.c.src) — the portable C loop NumPy
        ///     keeps for zero-sized, oversized and 1-in-a-core-dimension cases:
        ///     <c>*(typ*)op += val1 * val2</c>, ascending k, accumulating straight into the output.
        /// </summary>
        internal static void MatmulInnerNoBlas<T>(T* ip1, long is1M, long is1N,
            T* ip2, long is2N, long is2P, T* op, long osM, long osP, long dm, long dn, long dp)
            where T : unmanaged, INumberBase<T>
        {
            long ib1N = is1N * dn, ib2N = is2N * dn, ib2P = is2P * dp, obP = osP * dp;

            for (long m = 0; m < dm; m++)
            {
                for (long p = 0; p < dp; p++)
                {
                    *op = T.Zero;
                    for (long n = 0; n < dn; n++)
                    {
                        T val1 = *ip1;
                        T val2 = *ip2;
                        *op += val1 * val2;
                        ip2 += is2N;
                        ip1 += is1N;
                    }

                    ip1 -= ib1N;
                    ip2 -= ib2N;
                    op += osP;
                    ip2 += is2P;
                }

                op -= obP;
                ip2 -= ib2P;
                ip1 += is1M;
                op += osM;
            }
        }

        /// <summary>Allocates a native scratch buffer for <paramref name="count"/> elements.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static T* Alloc<T>(long count) where T : unmanaged
            => count <= 0 ? null : (T*)NativeMemory.Alloc((nuint)count, (nuint)sizeof(T));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Free<T>(T* p) where T : unmanaged
        {
            if (p != null)
                NativeMemory.Free(p);
        }

        /// <summary>Zeroes <paramref name="count"/> elements — NumPy's output <c>memset</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Zero<T>(T* p, long count) where T : unmanaged
        {
            if (count > 0)
                NativeMemory.Clear(p, (nuint)count * (nuint)sizeof(T));
        }

        /// <summary>Whether the parity backend can service this dtype pair at all.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSupported(NPTypeCode common)
            => common == NPTypeCode.Single || common == NPTypeCode.Double;

        /// <summary>
        ///     Casts an operand to the common dtype when needed. NumPy does this up front with
        ///     <c>PyArray_FromAny(op, typec, …)</c>, which likewise yields a fresh C-contiguous array.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray AsCommon(NDArray a, NPTypeCode common)
            => a.GetTypeCode == common ? a : a.astype(common, copy: true, order: 'C');

        /// <summary>Builds the <see cref="Mat{T}"/> view of a 1-D or 2-D operand (element units).</summary>
        internal static Mat<T> AsMat<T>(NDArray a) where T : unmanaged
        {
            var shape = a.Shape;
            var mat = new Mat<T> { Data = (T*)a.Address + shape.offset };
            if (shape.NDim == 0)
            {
                mat.D0 = 1; mat.S0 = 1; mat.D1 = 1; mat.S1 = 1;
            }
            else if (shape.NDim == 1)
            {
                mat.D0 = shape.dimensions[0]; mat.S0 = shape.strides[0];
                mat.D1 = 1; mat.S1 = 1;
            }
            else
            {
                mat.D0 = shape.dimensions[0]; mat.S0 = shape.strides[0];
                mat.D1 = shape.dimensions[1]; mat.S1 = shape.strides[1];
            }

            return mat;
        }
    }
}
