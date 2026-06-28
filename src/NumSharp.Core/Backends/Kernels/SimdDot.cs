using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// SIMD fused multiply-accumulate dot product for contiguous float / double vectors.
    /// Computes <c>sum(a[i] * b[i])</c> in a single pass — no temporary product array
    /// (contrast with <c>left * right</c> followed by <c>ReduceAdd</c>, which materializes
    /// an n-element temp and walks the data twice).
    ///
    /// Four independent Vector256 accumulators give the out-of-order core enough
    /// instruction-level parallelism to hide FMA latency; a scalar tail handles the
    /// remainder. Accumulation type matches the element type (double in double, float in
    /// float) so the result dtype mirrors NumPy's <c>np.dot</c>.
    ///
    /// Callers route only contiguous (stride == 1) same-type operands here; strided views
    /// take a scalar strided loop, and non-float dtypes take the INumber&lt;T&gt; path —
    /// both in <c>Default.Dot.Fused.cs</c>.
    /// </summary>
    public static class SimdDot
    {
        /// <summary>Fused dot of two contiguous double vectors of length <paramref name="n"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double DotDouble(double* a, double* b, long n)
        {
            long i = 0;
            var a0 = Vector256<double>.Zero; var a1 = Vector256<double>.Zero;
            var a2 = Vector256<double>.Zero; var a3 = Vector256<double>.Zero;
            long limit = n - (n & 15); // 4 vectors x 4 doubles = 16 elements per unrolled step
            if (Fma.IsSupported)
            {
                for (; i < limit; i += 16)
                {
                    a0 = Fma.MultiplyAdd(Vector256.Load(a + i),      Vector256.Load(b + i),      a0);
                    a1 = Fma.MultiplyAdd(Vector256.Load(a + i + 4),  Vector256.Load(b + i + 4),  a1);
                    a2 = Fma.MultiplyAdd(Vector256.Load(a + i + 8),  Vector256.Load(b + i + 8),  a2);
                    a3 = Fma.MultiplyAdd(Vector256.Load(a + i + 12), Vector256.Load(b + i + 12), a3);
                }
            }
            else
            {
                for (; i < limit; i += 16)
                {
                    a0 = Vector256.Add(a0, Vector256.Multiply(Vector256.Load(a + i),      Vector256.Load(b + i)));
                    a1 = Vector256.Add(a1, Vector256.Multiply(Vector256.Load(a + i + 4),  Vector256.Load(b + i + 4)));
                    a2 = Vector256.Add(a2, Vector256.Multiply(Vector256.Load(a + i + 8),  Vector256.Load(b + i + 8)));
                    a3 = Vector256.Add(a3, Vector256.Multiply(Vector256.Load(a + i + 12), Vector256.Load(b + i + 12)));
                }
            }
            double sum = Vector256.Sum(Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)));
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        /// <summary>Fused dot of two contiguous float vectors of length <paramref name="n"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe float DotFloat(float* a, float* b, long n)
        {
            long i = 0;
            var a0 = Vector256<float>.Zero; var a1 = Vector256<float>.Zero;
            var a2 = Vector256<float>.Zero; var a3 = Vector256<float>.Zero;
            long limit = n - (n & 31); // 4 vectors x 8 floats = 32 elements per unrolled step
            if (Fma.IsSupported)
            {
                for (; i < limit; i += 32)
                {
                    a0 = Fma.MultiplyAdd(Vector256.Load(a + i),      Vector256.Load(b + i),      a0);
                    a1 = Fma.MultiplyAdd(Vector256.Load(a + i + 8),  Vector256.Load(b + i + 8),  a1);
                    a2 = Fma.MultiplyAdd(Vector256.Load(a + i + 16), Vector256.Load(b + i + 16), a2);
                    a3 = Fma.MultiplyAdd(Vector256.Load(a + i + 24), Vector256.Load(b + i + 24), a3);
                }
            }
            else
            {
                for (; i < limit; i += 32)
                {
                    a0 = Vector256.Add(a0, Vector256.Multiply(Vector256.Load(a + i),      Vector256.Load(b + i)));
                    a1 = Vector256.Add(a1, Vector256.Multiply(Vector256.Load(a + i + 8),  Vector256.Load(b + i + 8)));
                    a2 = Vector256.Add(a2, Vector256.Multiply(Vector256.Load(a + i + 16), Vector256.Load(b + i + 16)));
                    a3 = Vector256.Add(a3, Vector256.Multiply(Vector256.Load(a + i + 24), Vector256.Load(b + i + 24)));
                }
            }
            float sum = Vector256.Sum(Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)));
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
    }
}
