using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Vector-width driver for <see cref="NDHeavisideMath"/>. NumPy's <c>heaviside</c> is a SCALAR
    /// <c>BINARY_LOOP</c> (no SIMD on any platform), so vectorizing the step — which is nothing but two
    /// compares and three selects — beats NumPy several-fold while staying BIT-identical to the scalar
    /// kernel lane-for-lane (every branch is expressible as a mask+select, so lane <c>i</c> equals
    /// <c>Heaviside(x[i], h0[i])</c>; verified 0-diff vs NumPy 2.4.2 over the adversarial ±0/±inf/±NaN grid
    /// for float32 and float64).
    ///
    /// <para><b>The branchless composition and why its ORDER is exact.</b> The three <c>ConditionalSelect</c>s
    /// reproduce NumPy's four-way branch (NaN → 0==x → x&lt;0 → else):
    /// <list type="number">
    /// <item><c>res = (x &gt; 0) ? 1 : 0</c> — the positive/negative split; a NaN lane fails <c>&gt; 0</c> so
    /// it provisionally lands on 0, and the <c>x == 0</c> lane also lands on 0 here, both corrected below.</item>
    /// <item><c>res = (x == 0) ? h0 : res</c> — overlays <c>h0</c> where <c>x</c> is <c>±0.0</c> (IEEE
    /// <c>-0.0 == 0.0</c>); a NaN lane fails <c>== 0</c> so it is untouched.</item>
    /// <item><c>res = (x == x) ? res : NaN</c> — the only lanes where <c>x == x</c> is false are the NaN
    /// lanes, which take the POSITIVE canonical NaN; this is LAST so it overrides the provisional 0 a NaN
    /// <c>x</c> received in step 1, and it uses the non-NaN mask directly (no bitwise complement).</item>
    /// </list>
    /// Because <see cref="Vector.GreaterThan{T}(Vector{T},Vector{T})"/> and
    /// <see cref="Vector.Equals{T}(Vector{T},Vector{T})"/> both return false for a NaN operand, and
    /// <see cref="Vector.ConditionalSelect{T}(Vector{T},Vector{T},Vector{T})"/> is a per-bit blend, the
    /// composition is exactly the scalar branch — including the <c>h0</c>-passthrough at <c>x == 0</c>
    /// (a NaN or <c>-0.0</c> <c>h0</c> keeps its own bits) and the positive-canonical NaN at a NaN
    /// <c>x</c> (its sign discarded).</para>
    ///
    /// <para>float32 lanes are handled at single precision directly (no widen-to-double as
    /// <see cref="NDHypotMath"/> needs) because heaviside's outputs are exact and require no rounding.
    /// Width adapts through <see cref="Vector{T}"/> (V128/V256/V512 baked at startup).</para>
    /// </summary>
    public static partial class NDHeavisideMath
    {
        /// <summary>Whether the vectorized heaviside fast path can run (hardware-accelerated
        /// <see cref="Vector{T}"/>). When false, callers use the scalar kernels.</summary>
        public static bool SimdAvailable => Vector.IsHardwareAccelerated;

        /// <summary>The branchless step for one float64 vector (see the class remarks for the ordering).</summary>
        /// <param name="vx">The step arguments.</param>
        /// <param name="vh">The <c>x == 0</c> fills for these lanes.</param>
        /// <returns>The per-lane heaviside result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<double> StepD(Vector<double> vx, Vector<double> vh)
        {
            var res = Vector.ConditionalSelect(Vector.GreaterThan(vx, Vector<double>.Zero), OneD, Vector<double>.Zero);
            res = Vector.ConditionalSelect(Vector.Equals(vx, Vector<double>.Zero), vh, res);
            return Vector.ConditionalSelect(Vector.Equals(vx, vx), res, NanVecD);   // NaN lanes -> +NaN
        }

        /// <summary>The branchless step for one float32 vector (single precision; see class remarks).</summary>
        /// <param name="vx">The step arguments.</param>
        /// <param name="vh">The <c>x == 0</c> fills for these lanes.</param>
        /// <returns>The per-lane heaviside result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> StepF(Vector<float> vx, Vector<float> vh)
        {
            var res = Vector.ConditionalSelect(Vector.GreaterThan(vx, Vector<float>.Zero), OneF, Vector<float>.Zero);
            res = Vector.ConditionalSelect(Vector.Equals(vx, Vector<float>.Zero), vh, res);
            return Vector.ConditionalSelect(Vector.Equals(vx, vx), res, NanVecF);
        }

        // Constant vectors hoisted once (Vector<T> ctor from a scalar is cheap, but naming them keeps the
        // Step bodies readable). NaN is the POSITIVE canonical NaN so a NaN x yields NPY_NAN, never .NET's
        // negative NaN.
        private static readonly Vector<double> OneD = new Vector<double>(1.0);
        private static readonly Vector<float> OneF = new Vector<float>(1f);
        private static readonly Vector<double> NanVecD = new Vector<double>(NanD);
        private static readonly Vector<float> NanVecF = new Vector<float>(NanF);

        // =====================================================================
        // float64
        // =====================================================================

        /// <summary>heaviside over two contiguous float64 arrays: <c>r[i] = heaviside(x[i], h0[i])</c>.</summary>
        /// <param name="x">Contiguous step arguments.</param>
        /// <param name="h0">Contiguous <c>x == 0</c> fills.</param>
        /// <param name="r">Contiguous output (may alias neither operand for the vector path).</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideContiguousF64(double* x, double* h0, double* r, long n)
        {
            int w = Vector<double>.Count;
            long i = 0;
            for (; i <= n - w; i += w)
                StepD(Unsafe.ReadUnaligned<Vector<double>>(x + i), Unsafe.ReadUnaligned<Vector<double>>(h0 + i))
                    .CopyTo(new Span<double>(r + i, w));
            for (; i < n; i++) r[i] = Heaviside(x[i], h0[i]);   // scalar tail (bit-identical to the vector body)
        }

        /// <summary>heaviside with a broadcast fill: <c>r[i] = heaviside(x[i], h0)</c> (the common
        /// <c>np.heaviside(arr, 0.5)</c> shape). <paramref name="h0"/> is loaded once into a vector.</summary>
        /// <param name="x">Contiguous step arguments.</param>
        /// <param name="h0">The single fill value (broadcast).</param>
        /// <param name="r">Contiguous output.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideH0ScalarF64(double* x, double h0, double* r, long n)
        {
            int w = Vector<double>.Count;
            var vh = new Vector<double>(h0);
            long i = 0;
            for (; i <= n - w; i += w)
                StepD(Unsafe.ReadUnaligned<Vector<double>>(x + i), vh).CopyTo(new Span<double>(r + i, w));
            for (; i < n; i++) r[i] = Heaviside(x[i], h0);
        }

        /// <summary>heaviside with a broadcast step argument: <c>r[i] = heaviside(x, h0[i])</c>. Because a
        /// single <paramref name="x"/> selects ONE branch for the whole array, this collapses to a fill
        /// (negative → 0, positive → 1, NaN → NaN) or, at <c>x == 0</c>, a straight copy of the
        /// <paramref name="h0"/> bits — so it never even runs the select chain.</summary>
        /// <param name="x">The single step argument (broadcast).</param>
        /// <param name="h0">Contiguous fills (only read when <paramref name="x"/> is exactly 0).</param>
        /// <param name="r">Contiguous output.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideXScalarF64(double x, double* h0, double* r, long n)
        {
            // x == 0 (incl. -0.0): every output is that lane's h0 bits -> memcpy, no arithmetic.
            if (x == 0.0) { Buffer.MemoryCopy(h0, r, n * sizeof(double), n * sizeof(double)); return; }
            double fill = double.IsNaN(x) ? NanD : (x < 0.0 ? 0.0 : 1.0);
            var v = new Vector<double>(fill);
            int w = Vector<double>.Count;
            long i = 0;
            for (; i <= n - w; i += w) v.CopyTo(new Span<double>(r + i, w));
            for (; i < n; i++) r[i] = fill;
        }

        // =====================================================================
        // float32
        // =====================================================================

        /// <summary>heaviside over two contiguous float32 arrays: <c>r[i] = heaviside(x[i], h0[i])</c>.</summary>
        /// <param name="x">Contiguous step arguments.</param>
        /// <param name="h0">Contiguous <c>x == 0</c> fills.</param>
        /// <param name="r">Contiguous output.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideContiguousF32(float* x, float* h0, float* r, long n)
        {
            int w = Vector<float>.Count;
            long i = 0;
            for (; i <= n - w; i += w)
                StepF(Unsafe.ReadUnaligned<Vector<float>>(x + i), Unsafe.ReadUnaligned<Vector<float>>(h0 + i))
                    .CopyTo(new Span<float>(r + i, w));
            for (; i < n; i++) r[i] = HeavisideF(x[i], h0[i]);
        }

        /// <summary>heaviside with a broadcast fill: <c>r[i] = heaviside(x[i], h0)</c> (float32).</summary>
        /// <param name="x">Contiguous step arguments.</param>
        /// <param name="h0">The single fill value (broadcast).</param>
        /// <param name="r">Contiguous output.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideH0ScalarF32(float* x, float h0, float* r, long n)
        {
            int w = Vector<float>.Count;
            var vh = new Vector<float>(h0);
            long i = 0;
            for (; i <= n - w; i += w)
                StepF(Unsafe.ReadUnaligned<Vector<float>>(x + i), vh).CopyTo(new Span<float>(r + i, w));
            for (; i < n; i++) r[i] = HeavisideF(x[i], h0);
        }

        /// <summary>heaviside with a broadcast step argument: <c>r[i] = heaviside(x, h0[i])</c> (float32).
        /// Collapses to a fill or an <c>h0</c> copy — see the float64 twin.</summary>
        /// <param name="x">The single step argument (broadcast).</param>
        /// <param name="h0">Contiguous fills (only read when <paramref name="x"/> is exactly 0).</param>
        /// <param name="r">Contiguous output.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void HeavisideXScalarF32(float x, float* h0, float* r, long n)
        {
            if (x == 0f) { Buffer.MemoryCopy(h0, r, n * sizeof(float), n * sizeof(float)); return; }
            float fill = float.IsNaN(x) ? NanF : (x < 0f ? 0f : 1f);
            var v = new Vector<float>(fill);
            int w = Vector<float>.Count;
            long i = 0;
            for (; i <= n - w; i += w) v.CopyTo(new Span<float>(r + i, w));
            for (; i < n; i++) r[i] = fill;
        }
    }
}
