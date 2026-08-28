using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    // ============================ Symmetric Gram (syrk) kernel ============================
    // A managed matrix-product path for np.cov / np.corrcoef, whose core is
    //
    //     C = dot(Xc, Xc.T)            (unweighted)
    //     C = dot(Xc, (Xc*w).T)        (weighted)
    //
    // where Xc is (M variables, K observations), C-contiguous. This is a SYMMETRIC Gram
    // matrix (a syrk): C[i,j] = C[j,i]. Routing it through the general GEMM is pathological
    // for the shape cov produces — M and N are tiny (the variable count, usually 2), K is
    // huge (the observations), and the second operand Xc.T is a TRANSPOSED (strided) view.
    // The general GEMM's inner SIMD loop is over N (= M = 2), far below a vector width, and
    // every second-operand access is strided; measured ~553 µs for the (2,100000)@(100000,2)
    // product that cov(a,b) at N=100000 forms.
    //
    // This kernel instead reads the two C-contiguous rows of each variable pair with a
    // 4×-unrolled SIMD dot over the K axis, and exploits symmetry (only the upper triangle
    // is computed, then mirrored) — measured ~53 µs for the same product, ~10× the general
    // GEMM, and bit-identical to it for the small observation counts the oracle pins (there
    // K < one unrolled step, so the whole dot is the sequential scalar tail).
    //
    // Scope: real Single / Double (cov's default result dtype is >= float64; complex128,
    // float16 and any dtype the vector path can't take fall back to np.dot, exactly as
    // before). Gated to a small variable count by the caller — for many variables the
    // pairwise re-reads exceed cache and the general GEMM's blocking wins again.
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Computes the symmetric Gram matrix <c>G = A @ B^T</c> for row-major
        ///     <c>A</c>, <c>B</c> each shaped <c>(m, k)</c>, into the <c>(m, m)</c> buffer
        ///     <c>g</c>. The result is symmetric (<c>g[i,j] == g[j,i]</c>); the upper
        ///     triangle is computed and mirrored. For the unweighted covariance
        ///     <c>a == b</c> (the same buffer); the weighted form passes <c>b = Xc*w</c>.
        /// </summary>
        public unsafe delegate void GramKernel(void* a, void* b, void* g, int m, long k);

        /// <summary>Cache of Gram kernels keyed by element dtype (Single / Double only).</summary>
        internal static readonly ConcurrentDictionary<NPTypeCode, GramKernel> _gramCache = new();

        /// <summary>
        ///     Get the symmetric-Gram kernel for <paramref name="dt"/>, or <c>null</c> when
        ///     the dtype is outside scope (only Single / Double are served — cov's real
        ///     result dtypes), IL/SIMD is unavailable, or Vector256 is not hardware
        ///     accelerated. A <c>null</c> return routes the caller back to <c>np.dot</c>.
        /// </summary>
        public static GramKernel GetGramKernel(NPTypeCode dt)
        {
            if (!Enabled)
                return null;
            if (dt != NPTypeCode.Single && dt != NPTypeCode.Double)
                return null; // complex / half / decimal covariances fall back to np.dot
            if (!Vector256.IsHardwareAccelerated)
                return null;

            try
            {
                return _gramCache.GetOrAdd(dt, static d =>
                    GetGenericHelper(nameof(GramHelperSameType), GetClrType(d))
                        .CreateDelegate<GramKernel>());
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[ILKernel] GetGramKernel({dt}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Generic symmetric-Gram driver: upper-triangle pairwise SIMD dot + mirror.
        ///     Instantiated per element type (Single / Double) and bound to
        ///     <see cref="GramKernel"/> via <c>CreateDelegate</c> — the same "generic helper
        ///     called through a cached delegate" shape as <c>CumSumHelperSameType</c>. The
        ///     <c>void*</c> parameters keep the closed method's signature dtype-independent so
        ///     one delegate type serves every instantiation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void GramHelperSameType<T>(void* ap, void* bp, void* gp, int m, long k)
            where T : unmanaged, INumber<T>
        {
            T* a = (T*)ap;
            T* b = (T*)bp;
            T* g = (T*)gp;

            for (int i = 0; i < m; i++)
            {
                T* ai = a + (long)i * k;
                long ig = (long)i * m;
                for (int j = i; j < m; j++)
                {
                    T s = GramDot<T>(ai, b + (long)j * k, k);
                    g[ig + j] = s;          // upper (and diagonal)
                    g[(long)j * m + i] = s; // mirror to lower
                }
            }
        }

        /// <summary>
        ///     Fused 4×-unrolled SIMD dot of two contiguous vectors — the same shape as
        ///     <see cref="SimdDot"/> but generic over the (SIMD-capable) element type.
        ///     For counts below one unrolled step (the small observation counts the cov
        ///     oracle pins) the SIMD loop is skipped entirely and the sequential scalar tail
        ///     produces the result — bit-identical to a naive left-to-right dot.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe T GramDot<T>(T* a, T* b, long n) where T : unmanaged, INumber<T>
        {
            int w = Vector256<T>.Count;
            long step = (long)w * 4;
            var acc0 = Vector256<T>.Zero;
            var acc1 = acc0;
            var acc2 = acc0;
            var acc3 = acc0;

            long i = 0;
            long lim = n - (n % step);
            for (; i < lim; i += step)
            {
                acc0 += Vector256.Load(a + i) * Vector256.Load(b + i);
                acc1 += Vector256.Load(a + i + w) * Vector256.Load(b + i + w);
                acc2 += Vector256.Load(a + i + 2 * w) * Vector256.Load(b + i + 2 * w);
                acc3 += Vector256.Load(a + i + 3 * w) * Vector256.Load(b + i + 3 * w);
            }

            T sum = Vector256.Sum((acc0 + acc1) + (acc2 + acc3));
            for (; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }
    }
}
