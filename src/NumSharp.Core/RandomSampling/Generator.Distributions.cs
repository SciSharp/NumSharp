using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // Bit-exact log1p: Math.Log matches ucrtbase log on win-amd64 (NumPy's npy_log = log), and
        // the Kahan/Goldberg correction recovers the log1p precision the raw Math.Log(1+x) loses.
        // NumPy's npy_log1p is `#define npy_log1p log1p` (the CRT), so this reproduces it bit-for-bit
        // (verified 0-diff over 300k values), which is what keeps the ziggurat tail byte-exact.
        //
        // Bit-for-bit for x > -1 — the only domain any caller reaches (-U of a uniform in [0, 1), a probability its
        // parameter check keeps below 1) — and NOT at the edge (probed against np.log1p, 2026-10-01): x = -1 returns NaN
        // where the CRT's pole gives -inf (u = 0 turns the correction into -inf - 0/0), and x < -1 returns .NET's NEGATIVE
        // NaN (Math.Log's) where the CRT gives 0x7ff8000000000000. Consequence: an UNCHECKED p = 1 makes the Generator's
        // logseries return 2 (r = NaN) where NumPy's loops forever. Deliberately not branched: the u <= 0 test measured
        // 4.0-5.7% slower on the inversion fill's per-element loop (in-process interleaved A/B, 10M doubles), a path only
        // ~1.2x NumPy, for inputs no validated call produces.
        internal static double Log1p(double x)
        {
            double u = 1.0 + x;
            if (u == 1.0)
                return x;
            double y = Math.Log(u);
            if (u > 2.0)
                return y;
            return y - ((u - 1.0) - x) / u;
        }

        // ---- standard normal (ziggurat) : numpy random_standard_normal ----
        //
        // Each ziggurat sampler has ONE implementation, over a DrawBuffer (see DrawBuffer.cs): the per-draw form wraps
        // it in a one-word buffer (every word drawn on demand, exactly NumPy's call sequence), and the bulk fills give
        // it a read-ahead buffer — which cannot change the stream, because every output consumes at least one draw.

        /// <summary>One standard normal (NumPy's <c>random_standard_normal</c>), drawn straight from the bit generator.</summary>
        /// <returns>The draw.</returns>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        internal unsafe double NextStandardNormal()
        {
            ulong word;
            var src = new DrawBuffer64(_bitGenerator, &word, 1);
            return StandardNormal(ref src);
        }

        /// <summary>
        ///     NumPy's <c>random_standard_normal</c> ziggurat: one 64-bit draw decides index, sign and the candidate
        ///     (99.3% accepted outright); the rare wedge/tail cases draw doubles.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <returns>The draw.</returns>
        /// <remarks>
        ///     Split for speed without changing a bit: the accept-outright path is small enough to inline into the bulk
        ///     fill, and the wedge/tail continuation (<see cref="StandardNormalUnlikely"/>) stays out of line. The sign is
        ///     applied by flipping the IEEE sign bit — exactly what <c>x = -x</c> does, for every value — because the sign
        ///     bit is a coin flip, and as a branch it mispredicted half the time (measured: ~40% of the sampler's cost).
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static double StandardNormal(ref DrawBuffer64 src)
        {
            if (NormalAccept(src.NextUInt64(), out int idx, out ulong rabs, out double x))
                return x; // 99.3% of the time
            return StandardNormalUnlikely(ref src, idx, rabs, x);
        }

        /// <summary>
        ///     The ziggurat's accept-outright step for one 64-bit draw: the layer index, the signed candidate, and whether it
        ///     falls inside the layer's rectangle (shared by <see cref="StandardNormal"/> and the bulk fill's inline loop).
        /// </summary>
        /// <param name="r">The draw.</param>
        /// <param name="idx">The layer index (low 8 bits).</param>
        /// <param name="rabs">The 52-bit magnitude.</param>
        /// <param name="x">The signed candidate.</param>
        /// <returns>True when the candidate is accepted outright.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static bool NormalAccept(ulong r, out int idx, out ulong rabs, out double x)
        {
            idx = (int)(r & 0xff);
            r >>= 8;
            rabs = (r >> 1) & 0x000fffffffffffffUL;
            // rabs < 2**52 converts exactly through the SIGNED conversion — one cvtsi2sd, where a ulong -> double
            // conversion without AVX-512 is a multi-instruction sequence on the hot path.
            x = (long)rabs * ZigguratTables.wi_double[idx];
            x = BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(x) ^ (long)((r & 0x1) << 63));
            return rabs < ZigguratTables.ki_double[idx];
        }

        /// <summary>
        ///     The rest of NumPy's <c>random_standard_normal</c> loop body after a candidate missed the rectangle: the
        ///     tail (<c>idx == 0</c>) or the wedge test, and — when the wedge rejects — a fresh draw (the loop's next pass).
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="idx">The layer index of the rejected candidate.</param>
        /// <param name="rabs">Its 52-bit magnitude (bit 8 decides the tail's sign).</param>
        /// <param name="x">The signed candidate.</param>
        /// <returns>The draw.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static double StandardNormalUnlikely(ref DrawBuffer64 src, int idx, ulong rabs, double x)
        {
            if (idx == 0)
            {
                for (;;)
                {
                    double xx = -ZigguratTables.ziggurat_nor_inv_r * Log1p(-src.NextDouble());
                    double yy = -Log1p(-src.NextDouble());
                    if (yy + yy > xx * xx)
                        return ((rabs >> 8) & 0x1) != 0
                            ? -(ZigguratTables.ziggurat_nor_r + xx)
                            : ZigguratTables.ziggurat_nor_r + xx;
                }
            }
            if (((ZigguratTables.fi_double[idx - 1] - ZigguratTables.fi_double[idx]) * src.NextDouble()
                 + ZigguratTables.fi_double[idx]) < Math.Exp(-0.5 * x * x))
                return x;
            return StandardNormal(ref src);
        }

        // ---- standard exponential (ziggurat) : numpy random_standard_exponential ----

        /// <summary>One standard exponential (NumPy's <c>random_standard_exponential</c>), drawn straight from the bit generator.</summary>
        /// <returns>The draw.</returns>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        internal unsafe double NextStandardExponential()
        {
            ulong word;
            var src = new DrawBuffer64(_bitGenerator, &word, 1);
            return StandardExponential(ref src);
        }

        /// <summary>
        ///     NumPy's <c>random_standard_exponential</c> ziggurat (98.9% accepted outright); its
        ///     <c>standard_exponential_unlikely</c> tail/wedge cases live in <see cref="StandardExponentialUnlikely"/>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <returns>The draw.</returns>
        /// <remarks>The accept-outright path is inlined into the bulk fill; the continuation stays out of line.</remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static double StandardExponential(ref DrawBuffer64 src)
        {
            if (ExponentialAccept(src.NextUInt64(), out int idx, out double x))
                return x; // 98.9% of the time
            return StandardExponentialUnlikely(ref src, idx, x);
        }

        /// <summary>The exponential ziggurat's accept-outright step for one 64-bit draw.</summary>
        /// <param name="ri">The draw.</param>
        /// <param name="idx">The layer index.</param>
        /// <param name="x">The candidate.</param>
        /// <returns>True when the candidate is accepted outright.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static bool ExponentialAccept(ulong ri, out int idx, out double x)
        {
            ri >>= 3;
            idx = (int)(ri & 0xFF);
            ri >>= 8;
            x = (long)ri * ZigguratTables.we_double[idx]; // ri < 2**53: exact through the signed conversion
            return ri < ZigguratTables.ke_double[idx];
        }

        /// <summary>
        ///     NumPy's <c>standard_exponential_unlikely</c>: the tail (<c>idx == 0</c>), the wedge test, or — when the wedge
        ///     rejects — a fresh <c>random_standard_exponential</c> (NumPy's own tail call).
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="idx">The layer index of the rejected candidate.</param>
        /// <param name="x">The candidate.</param>
        /// <returns>The draw.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static double StandardExponentialUnlikely(ref DrawBuffer64 src, int idx, double x)
        {
            if (idx == 0)
                return ZigguratTables.ziggurat_exp_r - Log1p(-src.NextDouble());
            if ((ZigguratTables.fe_double[idx - 1] - ZigguratTables.fe_double[idx]) * src.NextDouble()
                + ZigguratTables.fe_double[idx] < Math.Exp(-x))
                return x;
            return StandardExponential(ref src);
        }

        // numpy random_standard_exponential_inv : the method='inv' inverse-CDF sampler.
        internal double NextStandardExponentialInv()
        {
            return -Log1p(-_bitGenerator.NextDouble());
        }

        // float32 Kahan log1p, mirroring the double form (MathF.Log ~ ucrtbase logf on win-amd64).
        internal static float Log1pF(float x)
        {
            float u = 1.0f + x;
            if (u == 1.0f)
                return x;
            float y = MathF.Log(u);
            if (u > 2.0f)
                return y;
            return y - ((u - 1.0f) - x) / u;
        }

        // ---- float32 ziggurat : numpy random_standard_normal_f / _exponential_f ----

        /// <summary>One float32 standard normal (NumPy's <c>random_standard_normal_f</c>), drawn straight from the bit generator.</summary>
        /// <returns>The draw.</returns>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        internal unsafe float NextStandardNormalF()
        {
            uint word;
            var src = new DrawBuffer32(_bitGenerator, &word, 1);
            return StandardNormalF(ref src);
        }

        /// <summary>NumPy's <c>random_standard_normal_f</c> ziggurat over 32-bit draws.</summary>
        /// <param name="src">The draw source.</param>
        /// <returns>The draw.</returns>
        /// <remarks>Split and sign-flipped branchlessly like <see cref="StandardNormal"/>, for the same reasons.</remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static float StandardNormalF(ref DrawBuffer32 src)
        {
            if (NormalAcceptF(src.NextUInt32(), out int idx, out uint rabs, out float x))
                return x;
            return StandardNormalUnlikelyF(ref src, idx, rabs, x);
        }

        /// <summary>The float32 normal ziggurat's accept-outright step for one 32-bit draw.</summary>
        /// <param name="r">The draw.</param>
        /// <param name="idx">The layer index.</param>
        /// <param name="rabs">The 23-bit magnitude.</param>
        /// <param name="x">The signed candidate.</param>
        /// <returns>True when the candidate is accepted outright.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static bool NormalAcceptF(uint r, out int idx, out uint rabs, out float x)
        {
            idx = (int)(r & 0xff);
            rabs = (r >> 9) & 0x007fffff; // 23-bit mantissa mask
            x = (int)rabs * ZigguratTables.wi_float[idx]; // rabs < 2**23: exact through the signed conversion
            x = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(x) ^ (int)(((r >> 8) & 0x1) << 31));
            return rabs < ZigguratTables.ki_float[idx];
        }

        /// <summary>The tail / wedge continuation of NumPy's <c>random_standard_normal_f</c> (a rejected wedge draws afresh).</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="idx">The layer index of the rejected candidate.</param>
        /// <param name="rabs">Its 23-bit magnitude (bit 8 decides the tail's sign).</param>
        /// <param name="x">The signed candidate.</param>
        /// <returns>The draw.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static float StandardNormalUnlikelyF(ref DrawBuffer32 src, int idx, uint rabs, float x)
        {
            if (idx == 0)
            {
                for (;;)
                {
                    float xx = -ZigguratTables.ziggurat_nor_inv_r_f * Log1pF(-src.NextFloat());
                    float yy = -Log1pF(-src.NextFloat());
                    if (yy + yy > xx * xx)
                        return ((rabs >> 8) & 0x1) != 0
                            ? -(ZigguratTables.ziggurat_nor_r_f + xx)
                            : ZigguratTables.ziggurat_nor_r_f + xx;
                }
            }
            // NumPy uses double exp here (comparison promotes the float LHS to double).
            if (((ZigguratTables.fi_float[idx - 1] - ZigguratTables.fi_float[idx]) * src.NextFloat()
                 + ZigguratTables.fi_float[idx]) < Math.Exp(-0.5 * x * x))
                return x;
            return StandardNormalF(ref src);
        }

        /// <summary>One float32 standard exponential (NumPy's <c>random_standard_exponential_f</c>), drawn straight from the bit generator.</summary>
        /// <returns>The draw.</returns>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        internal unsafe float NextStandardExponentialF()
        {
            uint word;
            var src = new DrawBuffer32(_bitGenerator, &word, 1);
            return StandardExponentialF(ref src);
        }

        /// <summary>NumPy's <c>random_standard_exponential_f</c> ziggurat over 32-bit draws.</summary>
        /// <param name="src">The draw source.</param>
        /// <returns>The draw.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static float StandardExponentialF(ref DrawBuffer32 src)
        {
            if (ExponentialAcceptF(src.NextUInt32(), out int idx, out float x))
                return x;
            return StandardExponentialUnlikelyF(ref src, idx, x);
        }

        /// <summary>The float32 exponential ziggurat's accept-outright step for one 32-bit draw.</summary>
        /// <param name="ri">The draw.</param>
        /// <param name="idx">The layer index.</param>
        /// <param name="x">The candidate.</param>
        /// <returns>True when the candidate is accepted outright.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static bool ExponentialAcceptF(uint ri, out int idx, out float x)
        {
            ri >>= 1;
            idx = (int)(ri & 0xFF);
            ri >>= 8;
            x = (int)ri * ZigguratTables.we_float[idx]; // ri < 2**23: exact through the signed conversion
            return ri < ZigguratTables.ke_float[idx];
        }

        /// <summary>NumPy's <c>standard_exponential_unlikely_f</c>: tail, wedge, or a fresh draw.</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="idx">The layer index of the rejected candidate.</param>
        /// <param name="x">The candidate.</param>
        /// <returns>The draw.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static float StandardExponentialUnlikelyF(ref DrawBuffer32 src, int idx, float x)
        {
            if (idx == 0)
                return ZigguratTables.ziggurat_exp_r_f - Log1pF(-src.NextFloat());
            if ((ZigguratTables.fe_float[idx - 1] - ZigguratTables.fe_float[idx]) * src.NextFloat()
                + ZigguratTables.fe_float[idx] < MathF.Exp(-x)) // NumPy uses expf here
                return x;
            return StandardExponentialF(ref src);
        }

        // numpy random_standard_exponential_inv_fill_f : double log1p of the float draw, stored as float.
        internal float NextStandardExponentialInvF()
        {
            return (float)(-Log1p(-(double)_bitGenerator.NextFloat()));
        }

        // numpy random_standard_gamma_f (float32 gamma sampler).
        internal float NextStandardGammaF(float shape)
        {
            if (shape == 1.0f)
                return NextStandardExponentialF();
            if (shape == 0.0f)
                return 0.0f;
            if (shape < 1.0f)
            {
                for (;;)
                {
                    float U = _bitGenerator.NextFloat();
                    float V = NextStandardExponentialF();
                    if (U <= 1.0f - shape)
                    {
                        float X = MathF.Pow(U, 1.0f / shape);
                        if (X <= V)
                            return X;
                    }
                    else
                    {
                        float Y = -MathF.Log((1.0f - U) / shape);
                        float X = MathF.Pow(1.0f - shape + shape * Y, 1.0f / shape);
                        if (X <= V + Y)
                            return X;
                    }
                }
            }
            else
            {
                float b = shape - 1.0f / 3.0f;
                float c = 1.0f / MathF.Sqrt(9.0f * b);
                for (;;)
                {
                    float X, V;
                    do
                    {
                        X = NextStandardNormalF();
                        V = 1.0f + c * X;
                    } while (V <= 0.0f);

                    V = V * V * V;
                    float U = _bitGenerator.NextFloat();
                    if (U < 1.0f - 0.0331f * (X * X) * (X * X))
                        return b * V;
                    if (MathF.Log(U) < 0.5f * X * X + b * (1.0f - V + MathF.Log(V)))
                        return b * V;
                }
            }
        }

        // ---- standard gamma : numpy random_standard_gamma ----

        internal double NextStandardGamma(double shape)
        {
            if (shape == 1.0)
                return NextStandardExponential();
            if (shape == 0.0)
                return 0.0;
            if (shape < 1.0)
            {
                for (;;)
                {
                    double U = _bitGenerator.NextDouble();
                    double V = NextStandardExponential();
                    if (U <= 1.0 - shape)
                    {
                        double X = Math.Pow(U, 1.0 / shape);
                        if (X <= V)
                            return X;
                    }
                    else
                    {
                        double Y = -Math.Log((1.0 - U) / shape);
                        double X = Math.Pow(1.0 - shape + shape * Y, 1.0 / shape);
                        if (X <= V + Y)
                            return X;
                    }
                }
            }
            else
            {
                double b = shape - 1.0 / 3.0;
                double c = 1.0 / Math.Sqrt(9.0 * b);
                for (;;)
                {
                    double X, V;
                    do
                    {
                        X = NextStandardNormal();
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    double U = _bitGenerator.NextDouble();
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                        return b * V;
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V)))
                        return b * V;
                }
            }
        }
    }
}
