using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     NumPy's dot functions for float32, float64 and complex128 (<c>FLOAT_dot</c>, <c>DOUBLE_dot</c>,
    ///     <c>CDOUBLE_dot</c> in arraytypes.c.src) with the BLAS arithmetic NumPy 2.4.2's wheel runs under them:
    ///     scipy-openblas' <c>cblas_sdot</c> / <c>cblas_ddot</c> / <c>cblas_zdotu_sub</c> (OpenBLAS kernel/x86_64 sdot.c,
    ///     ddot.c and zdot.c with their Haswell microkernels) at one thread. Every np.dot, np.correlate and np.convolve
    ///     call site that NumPy routes through a dtype's <c>dotfunc</c> computes here, so they give NumPy's bits.
    /// </summary>
    /// <remarks>
    ///     <para><b>Which bits these are.</b> The Haswell kernels are what OpenBLAS dispatches on an AVX2 host (probed on
    ///     an i9-13900K, 2026-10-30 and 2026-11-05: every float32 / float64 / complex128 result of np.dot over the three
    ///     stride classes, 1-D and N-D, lengths 1 to 1,000). NumPy on a host that dispatches another kernel (AVX-512's
    ///     SkylakeX for sdot / ddot), or with OpenBLAS threads above one on a float64 or complex dot longer than 10,000
    ///     terms, gives other bits; the results here stay within the usual reordering bound of those.</para>
    ///     <para><b>Same bits on every host.</b> The vector kernels need AVX and FMA; without them the same lanes, the
    ///     same fused multiply-adds (<see cref="MathF.FusedMultiplyAdd"/> / <see cref="Math.FusedMultiplyAdd"/>, both
    ///     correctly rounded) and the same reduction tree run in scalar code, slower but byte-identical, so an Arm64 or
    ///     pre-Haswell host gives the AVX2 host's answers.</para>
    ///     <para><b>Strides</b> are in elements and may have any sign. NumPy's <c>blas_stride</c> hands cblas a positive
    ///     stride that is a multiple of the item size (always true for an element stride) and fits a CBLAS integer
    ///     (scipy-openblas64's are 64-bit, so NumPy's chunking by <c>NPY_CBLAS_CHUNK</c> never splits a call); a zero or
    ///     negative stride takes the dot function's own loop.</para>
    ///     <para><b>Three functions per dtype, one per stride class.</b> Every output of one np.dot or np.correlate call
    ///     has the same pair of strides, so a caller that makes many dots picks the class once (<see cref="FloatDotFor"/>,
    ///     <see cref="DoubleDotFor"/>, <see cref="ComplexDotFor"/>) and calls a small function per output; a single dot
    ///     goes through the dispatching <see cref="FloatDot"/> / <see cref="DoubleDot"/> / <see cref="ComplexDot"/>. All
    ///     three per-class functions of a dtype share its signature, so any of them can stand behind one function
    ///     pointer.</para>
    ///     <para><b>NaN.</b> A single NaN keeps its payload, and a NaN that arises inside the sum (<c>inf * 0</c>,
    ///     <c>inf - inf</c>) is x86's default NaN, as in NumPy. When two different NaNs meet in one dot, which one
    ///     survives follows the operand order RyuJIT gives each add and fused multiply-add, not OpenBLAS's assembly order
    ///     (the library's other NaN-order cases are documented the same way).</para>
    ///     <para><b>Never -0.0.</b> Every sum below starts at +0.0 and every vector accumulator at a zeroed register, and
    ///     in round-to-nearest <c>x + (+0.0)</c> is -0.0 only for <c>x = -0.0</c>, which the start excludes; so the zeroed
    ///     double that FLOAT_dot / DOUBLE_dot / CDOUBLE_dot add cblas' result into changes nothing but is kept as NumPy
    ///     writes it.</para>
    /// </remarks>
    internal static class NDDot
    {
        // ------------------------------------------------------------------------------------------------------------
        //  float32: FLOAT_dot over cblas_sdot
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>FLOAT_dot</c> for one dot: two positive strides go to <c>cblas_sdot</c> (unit strides
        ///     <see cref="FloatDotUnit"/>, others <see cref="FloatDotStrided"/>), any other stride to its own loop
        ///     (<see cref="FloatDotOwn"/>).
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>, cblas' x).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>, cblas' y).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms (none summed when zero or negative).</param>
        /// <returns>The dot product NumPy's float32 dotfunc writes.</returns>
        /// <remarks>A caller making many dots with one pair of strides should take <see cref="FloatDotFor"/> once
        ///     instead: this dispatch costs a branch and a call per dot.</remarks>
        internal static unsafe float FloatDot(float* ip1, long is1, float* ip2, long is2, long n)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? FloatDotUnit(ip1, 1, ip2, 1, n) : FloatDotStrided(ip1, is1, ip2, is2, n);
            return FloatDotOwn(ip1, is1, ip2, is2, n);
        }

        /// <summary>
        ///     The float32 dot function for one pair of strides, chosen as <see cref="FloatDot"/> chooses: the function
        ///     every dot of a call with these strides runs.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="FloatDotUnit"/>, <see cref="FloatDotStrided"/> or <see cref="FloatDotOwn"/>.</returns>
        internal static unsafe delegate*<float*, long, float*, long, long, float> FloatDotFor(long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &FloatDotUnit : &FloatDotStrided;
            return &FloatDotOwn;
        }

        /// <summary>
        ///     FLOAT_dot over <c>cblas_sdot</c> with two unit strides (sdot.c): the Haswell vector kernel over the first
        ///     <c>n &amp; -32</c> terms (<see cref="SdotKernel16"/>, a float32 result), the remaining terms' float32 products
        ///     summed into a double from +0, the kernel's result added last, the double rounded to float32, and that added
        ///     into FLOAT_dot's zeroed double and rounded again.
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand), unit stride.</param>
        /// <param name="is1">Ignored: 1 (the shared dot-function signature).</param>
        /// <param name="ip2">cblas' y (NumPy's second operand), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float32 dotfunc writes.</returns>
        /// <remarks>Below 32 terms that is the scalar loop alone (probed: 960 of 960 N-D results per length from 3 to 31
        ///     terms, 200 of 200 1-D); from 32 terms, the kernel (200 of 200 1-D results from 32 to 1,000 terms).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe float FloatDotUnit(float* ip1, long is1, float* ip2, long is2, long n)
        {
            long n1 = n > 0 ? n & -32 : 0;   // a negative count sums nothing (n & -32 would be negative)
            // sdot.c's `double dot = 0.0`: the tail's float32 products (y * x, rounded) widened into it, in order.
            double dot = 0.0;
            for (long i = n1; i < n; i++)
                dot += ip2[i] * ip1[i];
            if (n1 > 0)
                dot += SdotKernel16(n1, ip1, ip2);   // `dot += mydot`: the kernel's float32 sum comes last
            double sum = 0.0;                         // FLOAT_dot's `double sum = 0.; sum += cblas_sdot(...)`
            sum += (float)dot;
            return (float)sum;
        }

        /// <summary>
        ///     FLOAT_dot over <c>cblas_sdot</c> with positive strides that are not both 1 (sdot.c): pairs of float32
        ///     products added in float32, each pair widened into a double from +0, an odd last product added alone, the
        ///     double rounded to float32 and passed through FLOAT_dot's zeroed double.
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand).</param>
        /// <param name="is1">Its element stride (positive).</param>
        /// <param name="ip2">cblas' y (NumPy's second operand).</param>
        /// <param name="is2">Its element stride (positive).</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float32 dotfunc writes.</returns>
        /// <remarks>Probed against two variants that contract one product of a pair into a fused multiply-add: neither
        ///     matched, the plain pairs did (960 of 960 N-D results per length).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        internal static unsafe float FloatDotStrided(float* ip1, long is1, float* ip2, long is2, long n)
        {
            double d = 0.0;
            long m = n & -2;
            long t = 0;
            long s1 = 2 * is1, s2 = 2 * is2;
            for (; t < m; t += 2, ip1 += s1, ip2 += s2)
                d += ip2[0] * ip1[0] + ip2[is2] * ip1[is1];   // the pair added in float32, then widened into the double
            for (; t < n; t++, ip1 += is1, ip2 += is2)
                d += ip2[0] * ip1[0];
            double sum = 0.0;
            sum += (float)d;
            return (float)sum;
        }

        /// <summary>
        ///     FLOAT_dot's own loop, for a stride cblas is not handed (zero or negative): a float32 sum of float32
        ///     products from +0, in order.
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float32 dotfunc writes.</returns>
        /// <remarks>NumPy compiles <c>sum += ip1r * ip2r</c> without contraction (probed: N-D np.dot over a reversed run,
        ///     960 of 960 results per length from 3 to 64 terms; the fused variant matched 614 to 751), and RyuJIT never
        ///     fuses a multiply and an add, so the C# below is that arithmetic.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe float FloatDotOwn(float* ip1, long is1, float* ip2, long is2, long n)
        {
            float s = 0f;
            for (long i = 0; i < n; i++, ip1 += is1, ip2 += is2)
                s += *ip1 * *ip2;
            return s;
        }

        /// <summary>
        ///     The float32 row function for one pair of strides, or null where the per-output function
        ///     (<see cref="FloatDotFor"/>) is the faster choice: a row computes <c>o[j] = FLOAT_dot(x, y + j * yStep)</c>
        ///     for consecutive outputs that share their first operand.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="FloatDotUnitRow"/> (two unit strides), <see cref="FloatDotOwnRow"/> (a stride cblas is not
        ///     handed), or null (other positive strides: sdot's pairs already keep two products in flight).</returns>
        internal static unsafe delegate*<float*, long, float*, long, long, float*, long, long, void> FloatDotRowFor(
            long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &FloatDotUnitRow : null;
            return &FloatDotOwnRow;
        }

        /// <summary>
        ///     <see cref="FloatDotUnit"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = FLOAT_dot(x, y + j * yStep)</c>, two unit-stride runs each, four outputs at a time with their
        ///     sequential double tails side by side; every output keeps sdot's own order, so the bits are
        ///     <see cref="FloatDotUnit"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (cblas' x), unit stride.</param>
        /// <param name="is1">Ignored: 1 (the walk's row signature).</param>
        /// <param name="y">The first output's second operand (cblas' y), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives +0.</param>
        /// <remarks>Why: below 32 terms sdot is one chain of dependent double adds per output; four chains in flight
        ///     cover the add latency one output alone waits on.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void FloatDotUnitRow(float* x, long is1, float* y, long is2, long yStep, float* o,
            long count, long n)
        {
            long n1 = n > 0 ? n & -32 : 0;
            long j = 0;
            for (; j + 4 <= count; j += 4, y += 4 * yStep)
            {
                float* y0 = y;
                float* y1 = y0 + yStep;
                float* y2 = y1 + yStep;
                float* y3 = y2 + yStep;
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
                for (long i = n1; i < n; i++)
                {
                    float xi = x[i];
                    d0 += y0[i] * xi;   // each a float32 product, then widened into its double sum
                    d1 += y1[i] * xi;
                    d2 += y2[i] * xi;
                    d3 += y3[i] * xi;
                }

                if (n1 > 0)
                {
                    // `dot += mydot` per output, after its tail, as sdot.c orders it.
                    d0 += SdotKernel16(n1, x, y0);
                    d1 += SdotKernel16(n1, x, y1);
                    d2 += SdotKernel16(n1, x, y2);
                    d3 += SdotKernel16(n1, x, y3);
                }

                // sdot returns (float)dot; FLOAT_dot adds that into a zeroed double and rounds again.
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                s0 += (float)d0;
                s1 += (float)d1;
                s2 += (float)d2;
                s3 += (float)d3;
                o[j] = (float)s0;
                o[j + 1] = (float)s1;
                o[j + 2] = (float)s2;
                o[j + 3] = (float)s3;
            }

            for (; j < count; j++, y += yStep)
                o[j] = FloatDotUnit(x, 1, y, 1, n);
        }

        /// <summary>
        ///     <see cref="FloatDotOwn"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = FLOAT_dot(x, y + j * yStep)</c> with a stride cblas is not handed, four outputs at a time with
        ///     their float32 sums side by side; every output keeps the own loop's order, so the bits are
        ///     <see cref="FloatDotOwn"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="y">The first output's second operand (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives +0.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void FloatDotOwnRow(float* x, long is1, float* y, long is2, long yStep, float* o,
            long count, long n)
        {
            long j = 0;
            for (; j + 4 <= count; j += 4, y += 4 * yStep)
            {
                float* y0 = y;
                float* y1 = y0 + yStep;
                float* y2 = y1 + yStep;
                float* y3 = y2 + yStep;
                float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
                long xo = 0, yo = 0;
                for (long i = 0; i < n; i++, xo += is1, yo += is2)
                {
                    float xv = x[xo];
                    s0 += xv * y0[yo];   // FLOAT_dot's `sum += ip1r * ip2r`, per output
                    s1 += xv * y1[yo];
                    s2 += xv * y2[yo];
                    s3 += xv * y3[yo];
                }
                o[j] = s0;
                o[j + 1] = s1;
                o[j + 2] = s2;
                o[j + 3] = s3;
            }

            for (; j < count; j++, y += yStep)
                o[j] = FloatDotOwn(x, is1, y, is2, n);
        }

        // ------------------------------------------------------------------------------------------------------------
        //  float64: DOUBLE_dot over cblas_ddot
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The float64 row function for one pair of strides, or null where the per-output function
        ///     (<see cref="DoubleDotFor"/>) is the faster choice: a row computes <c>o[j] = DOUBLE_dot(x, y + j * yStep)</c>
        ///     for consecutive outputs that share their first operand.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="DoubleDotUnitRow"/> (two unit strides), <see cref="DoubleDotOwnRow"/> (a stride cblas is
        ///     not handed), or null (other positive strides: ddot's two sums already keep two adds in flight).</returns>
        internal static unsafe delegate*<double*, long, double*, long, long, double*, long, long, void> DoubleDotRowFor(
            long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &DoubleDotUnitRow : null;
            return &DoubleDotOwnRow;
        }

        /// <summary>
        ///     <see cref="DoubleDotOwn"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = DOUBLE_dot(x, y + j * yStep)</c> with a stride cblas is not handed, four outputs at a time with
        ///     their double sums side by side; every output keeps the own loop's order, so the bits are
        ///     <see cref="DoubleDotOwn"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="y">The first output's second operand (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives +0.</param>
        /// <remarks>Why: one output's own loop is a chain of dependent adds, and a single 16-term chain measured 0.83x
        ///     to 1.04x of the former loop from run to run with identical machine code (its loop exit placed against the
        ///     branch predictor); four chains in flight make the row faster by more than that placement noise.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void DoubleDotOwnRow(double* x, long is1, double* y, long is2, long yStep, double* o,
            long count, long n)
        {
            long j = 0;
            for (; j + 4 <= count; j += 4, y += 4 * yStep)
            {
                double* y0 = y;
                double* y1 = y0 + yStep;
                double* y2 = y1 + yStep;
                double* y3 = y2 + yStep;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                long xo = 0, yo = 0;
                for (long i = 0; i < n; i++, xo += is1, yo += is2)
                {
                    double xv = x[xo];
                    s0 += xv * y0[yo];   // DOUBLE_dot's `sum += (*ip1) * (*ip2)`, per output
                    s1 += xv * y1[yo];
                    s2 += xv * y2[yo];
                    s3 += xv * y3[yo];
                }
                o[j] = s0;
                o[j + 1] = s1;
                o[j + 2] = s2;
                o[j + 3] = s3;
            }

            for (; j < count; j++, y += yStep)
                o[j] = DoubleDotOwn(x, is1, y, is2, n);
        }

        /// <summary>
        ///     NumPy's <c>DOUBLE_dot</c> for one dot: two positive strides go to <c>cblas_ddot</c> (unit strides
        ///     <see cref="DoubleDotUnit"/>, others <see cref="DoubleDotStrided"/>), any other stride to its own loop
        ///     (<see cref="DoubleDotOwn"/>).
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>, cblas' x).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>, cblas' y).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms (none summed when zero or negative).</param>
        /// <returns>The dot product NumPy's float64 dotfunc writes.</returns>
        /// <remarks>A caller making many dots with one pair of strides should take <see cref="DoubleDotFor"/> once.</remarks>
        internal static unsafe double DoubleDot(double* ip1, long is1, double* ip2, long is2, long n)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? DoubleDotUnit(ip1, 1, ip2, 1, n) : DoubleDotStrided(ip1, is1, ip2, is2, n);
            return DoubleDotOwn(ip1, is1, ip2, is2, n);
        }

        /// <summary>
        ///     The float64 dot function for one pair of strides, chosen as <see cref="DoubleDot"/> chooses.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="DoubleDotUnit"/>, <see cref="DoubleDotStrided"/> or <see cref="DoubleDotOwn"/>.</returns>
        internal static unsafe delegate*<double*, long, double*, long, long, double> DoubleDotFor(long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &DoubleDotUnit : &DoubleDotStrided;
            return &DoubleDotOwn;
        }

        /// <summary>
        ///     DOUBLE_dot over <c>cblas_ddot</c> with two unit strides (ddot.c): the Haswell vector kernel over the first
        ///     <c>n &amp; -16</c> terms (<see cref="DdotKernel8"/>), whose result the sum starts from, then the remaining
        ///     terms added in order with separate multiplies and adds, and that added into DOUBLE_dot's zeroed double.
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand), unit stride.</param>
        /// <param name="is1">Ignored: 1.</param>
        /// <param name="ip2">cblas' y (NumPy's second operand), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float64 dotfunc writes.</returns>
        /// <remarks>Below 16 terms that is a plain sequential sum. Probed exact on every result: 1-D 5 to 1,000 terms,
        ///     N-D 3 to 257.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe double DoubleDotUnit(double* ip1, long is1, double* ip2, long is2, long n)
        {
            long n1 = n > 0 ? n & -16 : 0;
            // The kernel STORES its result into dot (vmovsd), it does not add it: the sum starts as that value.
            double dot = n1 > 0 ? DdotKernel8(n1, ip1, ip2) : 0.0;
            for (long i = n1; i < n; i++)
                dot += ip2[i] * ip1[i];
            double sum = 0.0;   // DOUBLE_dot's `double sum = 0.; sum += cblas_ddot(...)`
            sum += dot;
            return sum;
        }

        /// <summary>
        ///     <see cref="DoubleDotUnit"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = DOUBLE_dot(x, y + j * yStep)</c> for <c>j</c> in <c>[0, count)</c>, two unit-stride runs each. Four
        ///     outputs at a time run their sequential tails side by side, so the tails' dependent adds overlap instead of
        ///     waiting on each other; every output keeps NumPy's own order of operations, so the bits are
        ///     <see cref="DoubleDotUnit"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (cblas' x), unit stride.</param>
        /// <param name="is1">Ignored: 1 (the walk's row signature).</param>
        /// <param name="y">The first output's second operand (cblas' y), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives +0.</param>
        /// <remarks>
        ///     Why: from 17 to 31 terms ddot runs its 16-term vector kernel and then up to 15 terms one after another, a
        ///     chain of dependent adds that one dot cannot shorten (NumPy's order is the point). The former one-accumulator
        ///     loop was faster per output there (N-D np.dot at 31 terms: 0.80x of it one output at a time); four tails in
        ///     flight cover that latency.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void DoubleDotUnitRow(double* x, long is1, double* y, long is2, long yStep, double* o,
            long count, long n)
        {
            long n1 = n > 0 ? n & -16 : 0;
            long j = 0;
            for (; j + 4 <= count; j += 4, y += 4 * yStep)
            {
                double* y0 = y;
                double* y1 = y0 + yStep;
                double* y2 = y1 + yStep;
                double* y3 = y2 + yStep;
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
                if (n1 > 0)
                {
                    // Four independent kernel calls: nothing orders them, so the core overlaps their reductions.
                    d0 = DdotKernel8(n1, x, y0);
                    d1 = DdotKernel8(n1, x, y1);
                    d2 = DdotKernel8(n1, x, y2);
                    d3 = DdotKernel8(n1, x, y3);
                }

                for (long i = n1; i < n; i++)
                {
                    double xi = x[i];
                    d0 += y0[i] * xi;
                    d1 += y1[i] * xi;
                    d2 += y2[i] * xi;
                    d3 += y3[i] * xi;
                }

                // DOUBLE_dot's `double sum = 0.; sum += cblas_ddot(...)` per output.
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                s0 += d0;
                s1 += d1;
                s2 += d2;
                s3 += d3;
                o[j] = s0;
                o[j + 1] = s1;
                o[j + 2] = s2;
                o[j + 3] = s3;
            }

            for (; j < count; j++, y += yStep)
                o[j] = DoubleDotUnit(x, 1, y, 1, n);
        }

        /// <summary>
        ///     DOUBLE_dot over <c>cblas_ddot</c> with positive strides that are not both 1 (ddot.c): four products a
        ///     group added as <c>t1 += m1 + m3; t2 += m2 + m4</c>, the remainder into <c>t1</c>, then <c>t1 + t2</c>,
        ///     added into DOUBLE_dot's zeroed double.
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand).</param>
        /// <param name="is1">Its element stride (positive).</param>
        /// <param name="ip2">cblas' y (NumPy's second operand).</param>
        /// <param name="is2">Its element stride (positive).</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float64 dotfunc writes.</returns>
        /// <remarks>Probed against NumPy 2.4.2 (a unit-stride run against a stride-3 one, 3 to 100 terms): 84 of 84
        ///     results, where a sequential sum matched 29; stepped 1-D runs 200 of 200.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        internal static unsafe double DoubleDotStrided(double* ip1, long is1, double* ip2, long is2, long n)
        {
            double t1 = 0.0, t2 = 0.0;
            long sx2 = 2 * is1, sx3 = 3 * is1, sx4 = 4 * is1, sy2 = 2 * is2, sy3 = 3 * is2, sy4 = 4 * is2;
            long t = 0;
            long n4 = n & -4;
            for (; t < n4; t += 4, ip1 += sx4, ip2 += sy4)
            {
                double m1 = ip2[0] * ip1[0];
                double m2 = ip2[is2] * ip1[is1];
                double m3 = ip2[sy2] * ip1[sx2];
                double m4 = ip2[sy3] * ip1[sx3];
                t1 += m1 + m3;
                t2 += m2 + m4;
            }
            for (; t < n; t++, ip1 += is1, ip2 += is2)
                t1 += ip2[0] * ip1[0];
            double sum = 0.0;
            sum += t1 + t2;
            return sum;
        }

        /// <summary>
        ///     DOUBLE_dot's own loop, for a stride cblas is not handed (zero or negative): a sequential double sum from
        ///     +0.
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms; zero or negative gives +0.</param>
        /// <returns>The dot product NumPy's float64 dotfunc writes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe double DoubleDotOwn(double* ip1, long is1, double* ip2, long is2, long n)
        {
            double s = 0.0;
            for (long i = 0; i < n; i++, ip1 += is1, ip2 += is2)
                s += *ip1 * *ip2;
            return s;
        }

        // ------------------------------------------------------------------------------------------------------------
        //  complex128: CDOUBLE_dot over cblas_zdotu_sub
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>CDOUBLE_dot</c> for one dot, no conjugation: two positive strides go to <c>cblas_zdotu_sub</c>
        ///     (unit strides <see cref="ComplexDotUnit"/>, others <see cref="ComplexDotStrided"/>), any other stride to its
        ///     own loop (<see cref="ComplexDotOwn"/>).
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>, cblas' x).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>, cblas' y).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms (none summed when zero or negative).</param>
        /// <returns>The dot product NumPy's complex128 dotfunc writes.</returns>
        /// <remarks>A caller making many dots with one pair of strides should take <see cref="ComplexDotFor"/> once.</remarks>
        internal static unsafe Complex ComplexDot(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? ComplexDotUnit(ip1, 1, ip2, 1, n) : ComplexDotStrided(ip1, is1, ip2, is2, n);
            return ComplexDotOwn(ip1, is1, ip2, is2, n);
        }

        /// <summary>
        ///     The complex128 dot function for one pair of strides, chosen as <see cref="ComplexDot"/> chooses.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="ComplexDotUnit"/>, <see cref="ComplexDotStrided"/> or <see cref="ComplexDotOwn"/>.</returns>
        internal static unsafe delegate*<Complex*, long, Complex*, long, long, Complex> ComplexDotFor(long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &ComplexDotUnit : &ComplexDotStrided;
            return &ComplexDotOwn;
        }

        /// <summary>
        ///     CDOUBLE_dot over <c>cblas_zdotu_sub</c> with two unit strides (zdot.c): the Haswell vector kernel
        ///     (<see cref="ZdotKernel8"/>) fills four double sums (<c>Σ xr·yr</c>, <c>Σ xi·yi</c>, <c>Σ xr·yi</c>,
        ///     <c>Σ xi·yr</c>) over the first <c>n &amp; -8</c> terms, the remaining terms continue them in order, and zdotu's
        ///     C99 result reaches CDOUBLE_dot's zeroed pair (<see cref="ZdotuResult"/>).
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand), unit stride.</param>
        /// <param name="is1">Ignored: 1.</param>
        /// <param name="ip2">cblas' y (NumPy's second operand, not conjugated), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="n">The number of terms; zero or negative gives (+0, +0).</param>
        /// <returns>The dot product NumPy's complex128 dotfunc writes.</returns>
        /// <remarks>Probed 2026-11-05 against NumPy 2.4.2: 200 of 200 1-D results at each of 16 lengths from 1 to
        ///     1,000, 960 of 960 N-D results at 7 to 64 terms. The four sums alone, the model before, matched 0 to 118 of
        ///     200 from 8 terms on.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Complex ComplexDotUnit(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            double* xp = (double*)ip1;
            double* yp = (double*)ip2;
            double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
            long i = 0;
            long n1 = n > 0 ? n & -8 : 0;
            if (n1 > 0)
            {
                // The kernel STORES the four sums (vmovups), so they start as its values; it returns them by value so
                // d0..d3 stay in registers (an out-parameter would make every tail add a store and a reload).
                Vector256<double> k = ZdotKernel8(n1, xp, yp);
                d0 = k.GetElement(0);
                d1 = k.GetElement(1);
                d2 = k.GetElement(2);
                d3 = k.GetElement(3);
                i = n1;
                xp += 2 * n1;
                yp += 2 * n1;
            }

            for (; i < n; i++, xp += 2, yp += 2)
            {
                d0 += xp[0] * yp[0];
                d1 += xp[1] * yp[1];
                d2 += xp[0] * yp[1];
                d3 += xp[1] * yp[0];
            }
            return ZdotuResult(d0 - d1, d2 + d3);
        }

        /// <summary>
        ///     CDOUBLE_dot over <c>cblas_zdotu_sub</c> with positive strides that are not both 1 (zdot.c): the four double
        ///     sums over every term in order, and zdotu's C99 result reaching CDOUBLE_dot's zeroed pair
        ///     (<see cref="ZdotuResult"/>).
        /// </summary>
        /// <param name="ip1">cblas' x (NumPy's first operand).</param>
        /// <param name="is1">Its element stride (positive).</param>
        /// <param name="ip2">cblas' y (NumPy's second operand, not conjugated).</param>
        /// <param name="is2">Its element stride (positive).</param>
        /// <param name="n">The number of terms; zero or negative gives (+0, +0).</param>
        /// <returns>The dot product NumPy's complex128 dotfunc writes.</returns>
        /// <remarks>Probed against NumPy 2.4.2: a strided pair of 3 to 100 terms matches on every result (84 of 84); the
        ///     naive per-term complex product on 6 of them.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Complex ComplexDotStrided(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            double* xp = (double*)ip1;
            double* yp = (double*)ip2;
            long sx = 2 * is1, sy = 2 * is2;   // the strides in doubles, as zdot.c doubles them
            double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (long i = 0; i < n; i++, xp += sx, yp += sy)
            {
                d0 += xp[0] * yp[0];
                d1 += xp[1] * yp[1];
                d2 += xp[0] * yp[1];
                d3 += xp[1] * yp[0];
            }
            return ZdotuResult(d0 - d1, d2 + d3);
        }

        /// <summary>
        ///     CDOUBLE_dot's own loop, for a stride cblas is not handed (zero or negative):
        ///     <c>sumr += ar*br - ai*bi; sumi += ar*bi + ai*br</c> from +0. It builds no C99 complex, so an infinite
        ///     imaginary part leaves the real part alone here.
        /// </summary>
        /// <param name="ip1">The first operand's element 0 (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="ip2">The second operand's element 0 (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="n">The number of terms; zero or negative gives (+0, +0).</param>
        /// <returns>The dot product NumPy's complex128 dotfunc writes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Complex ComplexDotOwn(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            double* xp = (double*)ip1;
            double* yp = (double*)ip2;
            long sx = 2 * is1, sy = 2 * is2;
            double sumr = 0.0, sumi = 0.0;
            for (long i = 0; i < n; i++, xp += sx, yp += sy)
            {
                double ar = xp[0], ai = xp[1], br = yp[0], bi = yp[1];
                sumr += ar * br - ai * bi;
                sumi += ar * bi + ai * br;
            }
            return new Complex(sumr, sumi);
        }

        /// <summary>
        ///     The complex128 row function for one pair of strides, or null where the per-output function
        ///     (<see cref="ComplexDotFor"/>) is the faster choice: a row computes <c>o[j] = CDOUBLE_dot(x, y + j * yStep)</c>
        ///     for consecutive outputs that share their first operand.
        /// </summary>
        /// <param name="is1">The first operand's element stride.</param>
        /// <param name="is2">The second operand's element stride.</param>
        /// <returns><see cref="ComplexDotUnitRow"/> (two unit strides), <see cref="ComplexDotOwnRow"/> (a stride cblas is
        ///     not handed), or null (other positive strides: zdotu's four sums already keep four adds in flight).</returns>
        internal static unsafe delegate*<Complex*, long, Complex*, long, long, Complex*, long, long, void> ComplexDotRowFor(
            long is1, long is2)
        {
            if (is1 > 0 && is2 > 0)
                return is1 == 1 && is2 == 1 ? &ComplexDotUnitRow : null;
            return &ComplexDotOwnRow;
        }

        /// <summary>
        ///     <see cref="ComplexDotUnit"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = CDOUBLE_dot(x, y + j * yStep)</c>, two unit-stride runs each, two outputs at a time with their
        ///     four-sum tails side by side; every output keeps zdotu's own order, so the bits are
        ///     <see cref="ComplexDotUnit"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (cblas' x), unit stride.</param>
        /// <param name="is1">Ignored: 1 (the walk's row signature).</param>
        /// <param name="y">The first output's second operand (cblas' y), unit stride.</param>
        /// <param name="is2">Ignored: 1.</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives (+0, +0).</param>
        /// <remarks>Two outputs, not four: their eight sums and two operands fill the x64 vector registers, and a
        ///     third output would spill. A short complex dot is mostly fixed cost, which the row also drops (one call per
        ///     row, not per output).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void ComplexDotUnitRow(Complex* x, long is1, Complex* y, long is2, long yStep, Complex* o,
            long count, long n)
        {
            long n1 = n > 0 ? n & -8 : 0;
            double* xp = (double*)x;
            long j = 0;
            for (; j + 2 <= count; j += 2, y += 2 * yStep)
            {
                double* ya = (double*)y;
                double* yb = (double*)(y + yStep);
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                double b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
                if (n1 > 0)
                {
                    // Each output's kernel STORES its four sums, which its tail then continues.
                    Vector256<double> ka = ZdotKernel8(n1, xp, ya);
                    Vector256<double> kb = ZdotKernel8(n1, xp, yb);
                    a0 = ka.GetElement(0); a1 = ka.GetElement(1); a2 = ka.GetElement(2); a3 = ka.GetElement(3);
                    b0 = kb.GetElement(0); b1 = kb.GetElement(1); b2 = kb.GetElement(2); b3 = kb.GetElement(3);
                }

                for (long i = 2 * n1; i < 2 * n; i += 2)
                {
                    double xr = xp[i], xi = xp[i + 1];
                    a0 += xr * ya[i];
                    a1 += xi * ya[i + 1];
                    a2 += xr * ya[i + 1];
                    a3 += xi * ya[i];
                    b0 += xr * yb[i];
                    b1 += xi * yb[i + 1];
                    b2 += xr * yb[i + 1];
                    b3 += xi * yb[i];
                }
                o[j] = ZdotuResult(a0 - a1, a2 + a3);
                o[j + 1] = ZdotuResult(b0 - b1, b2 + b3);
            }

            for (; j < count; j++, y += yStep)
                o[j] = ComplexDotUnit(x, 1, y, 1, n);
        }

        /// <summary>
        ///     <see cref="ComplexDotOwn"/> for a row of outputs that share their first operand:
        ///     <c>o[j] = CDOUBLE_dot(x, y + j * yStep)</c> with a stride cblas is not handed, four outputs at a time with
        ///     their two sums each side by side; every output keeps the own loop's order, so the bits are
        ///     <see cref="ComplexDotOwn"/>'s.
        /// </summary>
        /// <param name="x">The shared first operand (NumPy's <c>ip1</c>).</param>
        /// <param name="is1">Its element stride (any sign).</param>
        /// <param name="y">The first output's second operand (<c>ip2</c>).</param>
        /// <param name="is2">Its element stride (any sign).</param>
        /// <param name="yStep">The element distance from one output's second operand to the next one's.</param>
        /// <param name="o">The first output; the row's outputs are consecutive elements.</param>
        /// <param name="count">The number of outputs.</param>
        /// <param name="n">The terms of each dot; zero or negative gives (+0, +0).</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void ComplexDotOwnRow(Complex* x, long is1, Complex* y, long is2, long yStep, Complex* o,
            long count, long n)
        {
            double* xp = (double*)x;
            long sx = 2 * is1, sy = 2 * is2;
            long j = 0;
            for (; j + 4 <= count; j += 4, y += 4 * yStep)
            {
                double* y0 = (double*)y;
                double* y1 = (double*)(y + yStep);
                double* y2 = (double*)(y + 2 * yStep);
                double* y3 = (double*)(y + 3 * yStep);
                double r0 = 0.0, i0 = 0.0, r1 = 0.0, i1 = 0.0, r2 = 0.0, i2 = 0.0, r3 = 0.0, i3 = 0.0;
                long xo = 0, yo = 0;
                for (long t = 0; t < n; t++, xo += sx, yo += sy)
                {
                    double ar = xp[xo], ai = xp[xo + 1];
                    // CDOUBLE_dot's `sumr += ar*br - ai*bi; sumi += ar*bi + ai*br`, per output
                    r0 += ar * y0[yo] - ai * y0[yo + 1];
                    i0 += ar * y0[yo + 1] + ai * y0[yo];
                    r1 += ar * y1[yo] - ai * y1[yo + 1];
                    i1 += ar * y1[yo + 1] + ai * y1[yo];
                    r2 += ar * y2[yo] - ai * y2[yo + 1];
                    i2 += ar * y2[yo + 1] + ai * y2[yo];
                    r3 += ar * y3[yo] - ai * y3[yo + 1];
                    i3 += ar * y3[yo + 1] + ai * y3[yo];
                }
                o[j] = new Complex(r0, i0);
                o[j + 1] = new Complex(r1, i1);
                o[j + 2] = new Complex(r2, i2);
                o[j + 3] = new Complex(r3, i3);
            }

            for (; j < count; j++, y += yStep)
                o[j] = ComplexDotOwn(x, is1, y, is2, n);
        }

        /// <summary>
        ///     zdotu's result as CDOUBLE_dot receives it, added into CDOUBLE_dot's zeroed double pair: scipy-openblas is
        ///     built with C99 complex, so zdot_compute returns <c>re + im * _Complex_I</c>, and C's real-by-complex product
        ///     <c>im * (0 + 1i)</c> is <c>(im*0) + (im*1)i</c>: the real part becomes <c>re + im*0</c>, which an infinite or
        ///     NaN imaginary part poisons exactly as in NumPy.
        /// </summary>
        /// <param name="re">zdotu's real part, <c>Σ ar·br − Σ ai·bi</c>.</param>
        /// <param name="im">zdotu's imaginary part, <c>Σ ar·bi + Σ ai·br</c>.</param>
        /// <returns>The value NumPy's complex128 dotfunc writes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex ZdotuResult(double re, double im)
        {
            double sum0 = 0.0, sum1 = 0.0;   // CDOUBLE_dot's `sum += tmp` over its one chunk
            sum0 += re + im * 0.0;
            sum1 += im;
            return new Complex(sum0, sum1);
        }

        // ------------------------------------------------------------------------------------------------------------
        //  OpenBLAS's Haswell microkernels
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     OpenBLAS's Haswell <c>sdot_kernel_16</c> (sdot_microk_haswell-2.c) over <paramref name="n1"/> terms, a
        ///     positive multiple of 32: four eight-lane float32 accumulators of fused multiply-adds, each folded to four
        ///     lanes (upper half + lower half), the four added as <c>(t3 + t2) + (t1 + t0)</c>, then two horizontal adds.
        /// </summary>
        /// <param name="n1">The terms the kernel takes (a positive multiple of 32).</param>
        /// <param name="x">cblas' x, unit stride.</param>
        /// <param name="y">cblas' y, unit stride.</param>
        /// <returns>The kernel's float32 sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float SdotKernel16(long n1, float* x, float* y)
        {
            if (Fma.IsSupported)
            {
                var a4 = Vector256<float>.Zero;
                var a5 = Vector256<float>.Zero;
                var a6 = Vector256<float>.Zero;
                var a7 = Vector256<float>.Zero;
                for (long i = 0; i < n1; i += 32)
                {
                    a4 = Fma.MultiplyAdd(Vector256.Load(x + i), Vector256.Load(y + i), a4);
                    a5 = Fma.MultiplyAdd(Vector256.Load(x + i + 8), Vector256.Load(y + i + 8), a5);
                    a6 = Fma.MultiplyAdd(Vector256.Load(x + i + 16), Vector256.Load(y + i + 16), a6);
                    a7 = Fma.MultiplyAdd(Vector256.Load(x + i + 24), Vector256.Load(y + i + 24), a7);
                }

                var t4 = a4.GetUpper() + a4.GetLower();
                var t5 = a5.GetUpper() + a5.GetLower();
                var t6 = a6.GetUpper() + a6.GetLower();
                var t7 = a7.GetUpper() + a7.GetLower();
                var w = (t7 + t6) + (t5 + t4);
                // Fma implies AVX, hence SSE3's horizontal add: lane 0 = (w1 + w0) + (w3 + w2) after the second pass.
                var h = Sse3.HorizontalAdd(w, w);
                return Sse3.HorizontalAdd(h, h).ToScalar();
            }

            // The same lanes, fused multiply-adds and reduction tree in scalar code (no AVX / FMA on this host).
            float* acc = stackalloc float[32];
            for (int l = 0; l < 32; l++)
                acc[l] = 0f;
            for (long i = 0; i < n1; i += 32)
                for (int l = 0; l < 32; l++)
                    acc[l] = MathF.FusedMultiplyAdd(x[i + l], y[i + l], acc[l]);

            float* tt = stackalloc float[16];   // tt[4k + L]: accumulator k folded, lane L
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    tt[4 * k + l] = acc[8 * k + 4 + l] + acc[8 * k + l];
            float w0 = (tt[12] + tt[8]) + (tt[4] + tt[0]);
            float w1 = (tt[13] + tt[9]) + (tt[5] + tt[1]);
            float w2 = (tt[14] + tt[10]) + (tt[6] + tt[2]);
            float w3 = (tt[15] + tt[11]) + (tt[7] + tt[3]);
            return (w1 + w0) + (w3 + w2);
        }

        /// <summary>
        ///     OpenBLAS's Haswell <c>ddot_kernel_8</c> (ddot_microk_haswell-2.c) over <paramref name="n1"/> terms, a
        ///     positive multiple of 16: four four-lane double accumulators of fused multiply-adds, each folded to two
        ///     lanes (upper half + lower half), the four added as <c>(t3 + t2) + (t1 + t0)</c>, then one horizontal add.
        /// </summary>
        /// <param name="n1">The terms the kernel takes (a positive multiple of 16).</param>
        /// <param name="x">cblas' x, unit stride.</param>
        /// <param name="y">cblas' y, unit stride.</param>
        /// <returns>The kernel's double sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DdotKernel8(long n1, double* x, double* y)
        {
            if (Fma.IsSupported)
            {
                var a4 = Vector256<double>.Zero;
                var a5 = Vector256<double>.Zero;
                var a6 = Vector256<double>.Zero;
                var a7 = Vector256<double>.Zero;
                for (long i = 0; i < n1; i += 16)
                {
                    a4 = Fma.MultiplyAdd(Vector256.Load(x + i), Vector256.Load(y + i), a4);
                    a5 = Fma.MultiplyAdd(Vector256.Load(x + i + 4), Vector256.Load(y + i + 4), a5);
                    a6 = Fma.MultiplyAdd(Vector256.Load(x + i + 8), Vector256.Load(y + i + 8), a6);
                    a7 = Fma.MultiplyAdd(Vector256.Load(x + i + 12), Vector256.Load(y + i + 12), a7);
                }

                var t4 = a4.GetUpper() + a4.GetLower();
                var t5 = a5.GetUpper() + a5.GetLower();
                var t6 = a6.GetUpper() + a6.GetLower();
                var t7 = a7.GetUpper() + a7.GetLower();
                var w = (t7 + t6) + (t5 + t4);
                return Sse3.HorizontalAdd(w, w).ToScalar();
            }

            double* acc = stackalloc double[16];
            for (int l = 0; l < 16; l++)
                acc[l] = 0.0;
            for (long i = 0; i < n1; i += 16)
                for (int l = 0; l < 16; l++)
                    acc[l] = Math.FusedMultiplyAdd(x[i + l], y[i + l], acc[l]);

            // acc[4k + L]: accumulator k, lane L; folded lanes are (L2 + L0, L3 + L1).
            double w0 = ((acc[14] + acc[12]) + (acc[10] + acc[8])) + ((acc[6] + acc[4]) + (acc[2] + acc[0]));
            double w1 = ((acc[15] + acc[13]) + (acc[11] + acc[9])) + ((acc[7] + acc[5]) + (acc[3] + acc[1]));
            return w1 + w0;
        }

        /// <summary>
        ///     OpenBLAS's Haswell <c>zdot_kernel_8</c> (zdot_microk_haswell-2.c) over <paramref name="n1"/> complex terms, a
        ///     positive multiple of 8: eight four-lane double accumulators, four of <c>x · y</c> (lanes <c>xr·yr</c>,
        ///     <c>xi·yi</c> of two terms) and four of <c>x · swap(y)</c> (lanes <c>xr·yi</c>, <c>xi·yr</c>), added as
        ///     <c>(a3 + a2) + (a1 + a0)</c> and <c>(a7 + a6) + (a5 + a4)</c>, each then folded upper half + lower half.
        /// </summary>
        /// <param name="n1">The complex terms the kernel takes (a positive multiple of 8).</param>
        /// <param name="x">cblas' x as interleaved doubles, unit complex stride.</param>
        /// <param name="y">cblas' y as interleaved doubles, unit complex stride.</param>
        /// <returns>The kernel's four sums in lanes 0 to 3: <c>Σ xr·yr</c>, <c>Σ xi·yi</c>, <c>Σ xr·yi</c>,
        ///     <c>Σ xi·yr</c> (returned by value, so the caller's sums stay in registers).</returns>
        /// <remarks>The kernel has a second loop for more than 1,280 terms that only adds prefetches; its arithmetic is
        ///     the same.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe Vector256<double> ZdotKernel8(long n1, double* x, double* y)
        {
            long doubles = 2 * n1;
            if (Fma.IsSupported)
            {
                var a0 = Vector256<double>.Zero;
                var a1 = Vector256<double>.Zero;
                var a2 = Vector256<double>.Zero;
                var a3 = Vector256<double>.Zero;
                var a4 = Vector256<double>.Zero;
                var a5 = Vector256<double>.Zero;
                var a6 = Vector256<double>.Zero;
                var a7 = Vector256<double>.Zero;
                for (long i = 0; i < doubles; i += 16)
                {
                    var x8 = Vector256.Load(x + i);
                    var x9 = Vector256.Load(x + i + 4);
                    var x10 = Vector256.Load(x + i + 8);
                    var x11 = Vector256.Load(x + i + 12);
                    var y12 = Vector256.Load(y + i);
                    var y13 = Vector256.Load(y + i + 4);
                    var y14 = Vector256.Load(y + i + 8);
                    var y15 = Vector256.Load(y + i + 12);
                    a0 = Fma.MultiplyAdd(x8, y12, a0);
                    a1 = Fma.MultiplyAdd(x9, y13, a1);
                    a2 = Fma.MultiplyAdd(x10, y14, a2);
                    a3 = Fma.MultiplyAdd(x11, y15, a3);
                    // vpermilpd $0x05: each complex's two parts swapped, so the next four sums are xr·yi and xi·yr.
                    a4 = Fma.MultiplyAdd(x8, Avx.Permute(y12, 0x05), a4);
                    a5 = Fma.MultiplyAdd(x9, Avx.Permute(y13, 0x05), a5);
                    a6 = Fma.MultiplyAdd(x10, Avx.Permute(y14, 0x05), a6);
                    a7 = Fma.MultiplyAdd(x11, Avx.Permute(y15, 0x05), a7);
                }

                var r0 = (a3 + a2) + (a1 + a0);
                var r4 = (a7 + a6) + (a5 + a4);
                return Vector256.Create(r0.GetUpper() + r0.GetLower(), r4.GetUpper() + r4.GetLower());
            }

            // acc[4k + L]: accumulator k (0-3: x·y, 4-7: x·swap(y)), lane L; the same fused multiply-adds in scalar code.
            double* acc = stackalloc double[32];
            for (int l = 0; l < 32; l++)
                acc[l] = 0.0;
            for (long i = 0; i < doubles; i += 16)
            {
                for (int k = 0; k < 4; k++)
                {
                    double* xs = x + i + 4 * k;
                    double* ys = y + i + 4 * k;
                    for (int l = 0; l < 4; l++)
                    {
                        acc[4 * k + l] = Math.FusedMultiplyAdd(xs[l], ys[l], acc[4 * k + l]);
                        acc[16 + 4 * k + l] = Math.FusedMultiplyAdd(xs[l], ys[l ^ 1], acc[16 + 4 * k + l]);
                    }
                }
            }

            double* r = stackalloc double[8];   // r[L] = lane L of (a3 + a2) + (a1 + a0); r[4 + L] the same of a4..a7
            for (int l = 0; l < 4; l++)
            {
                r[l] = (acc[12 + l] + acc[8 + l]) + (acc[4 + l] + acc[l]);
                r[4 + l] = (acc[28 + l] + acc[24 + l]) + (acc[20 + l] + acc[16 + l]);
            }
            return Vector256.Create(r[2] + r[0], r[3] + r[1], r[6] + r[4], r[7] + r[5]);
        }
    }
}
