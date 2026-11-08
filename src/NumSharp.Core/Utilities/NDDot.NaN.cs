using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     NumPy's dot functions again, for the outputs that come out NaN: which NaN NumPy 2.4.2's binaries return when
    ///     several meet in one dot (an operand's NaN, or the default NaN of an invalid operation such as <c>inf - inf</c>
    ///     or <c>0 * inf</c>). The fast functions compute every NaN where NumPy does, but RyuJIT orders the operands of
    ///     each add and fused multiply-add as it likes, so their NaN can be another of the ones that met; a per-output
    ///     function whose result is NaN calls the model of its class here, which repeats the computation with every
    ///     operation choosing its NaN explicitly, and the callers of the row functions scan their result
    ///     (<see cref="AnyNaN(double*, long)"/>) and send each NaN output through the per-output function.
    /// </summary>
    /// <remarks>
    ///     <para><b>The rules</b> (probed on an i9-13900K against NumPy 2.4.2 / scipy-openblas, NaNs of distinct payloads
    ///     and signs planted at every operation site; then 68,850 of 68,850 random results of np.dot, np.vdot and
    ///     np.vecdot over float32, float64 and complex128, unit, positive and other strides, 1 to 100 terms):</para>
    ///     <list type="bullet">
    ///         <item>an SSE / AVX add, subtract or multiply returns its first source's NaN, else its second's, quieted;
    ///         with no NaN source a NaN result is x86's default NaN (negative, payload 0);</item>
    ///         <item>a fused multiply-add <c>a·b + c</c> returns <c>a</c>'s NaN, else <c>b</c>'s, else <c>c</c>'s - the
    ///         arithmetic order, whichever register the 231 / 213 encoding names first (the addend loses to both
    ///         multiplicands);</item>
    ///         <item>a horizontal add returns its even lane's NaN first;</item>
    ///         <item>a NaN keeps its payload and sign through float32 / float64 conversions (the top 22 payload bits
    ///         from float64 to float32).</item>
    ///     </list>
    ///     <para>Which source is first is fixed by the assembly of the Haswell microkernels and by the compilers of
    ///     everything else: GCC built scipy-openblas' C loops (<c>dot += y[i] * x[i]</c> keeps y first and the sum first;
    ///     ddot's strided result is <c>temp2 + temp1</c>; zdot's fourth product is <c>y · x</c>), MSVC built NumPy's own
    ///     loops (FLOAT_dot's is unrolled by four, the first product of each group <c>x · y</c> and the other three
    ///     <c>y · x</c>; CDOUBLE_dot's products take <c>ip2</c> first; CDOUBLE_dotc's and CDOUBLE_vdot's add
    ///     <c>p2 + p1</c>). Each model below names its own.</para>
    ///     <para><b>Cost.</b> A per-output function checks its result once (a compare and a branch that is not taken, the
    ///     model's call in tail position so the fast path keeps no frame); a row function's caller pays one vector pass
    ///     over the finished result instead, because a branch inside the rows' four-output loops took their registers.
    ///     A result that is not NaN does not depend on the order, because IEEE addition and multiplication of non-NaN
    ///     values commute. The models are scalar and run only for NaN outputs.</para>
    ///     <para><b>Not covered</b> (measured slower with a check, so D23 keeps their code): np.vecdot's complex128
    ///     vector walker for a core of one element (<c>ComplexcOne.Many</c>, 0.84x with a per-vector check), and
    ///     np.correlate / np.convolve's float32 / float64 kernels and their blocked long-kernel routes, whose values are
    ///     already bounded rather than NumPy's (PA3a).</para>
    ///     <para><b>Every host.</b> The models write x86's default NaN explicitly, so a host whose invalid operations make
    ///     another NaN (Arm64's is positive) still returns NumPy's x86 bits for every NaN output.</para>
    /// </remarks>
    internal static partial class NDDot
    {
        // ------------------------------------------------------------------------------------------------------------
        //  x86's NaN rules, one operation at a time
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>x86's default float64 NaN (sign set, quiet bit set, payload 0): what an invalid operation without a
        ///     NaN source returns on an x86 host.</summary>
        private const ulong DefaultNaNBits = 0xFFF8_0000_0000_0000UL;

        /// <summary>x86's default float32 NaN.</summary>
        private const uint DefaultNaNBitsSingle = 0xFFC0_0000U;

        /// <summary>A NaN quieted as x86 quiets a signalling one: the quiet bit set, sign and payload kept.</summary>
        /// <param name="v">A NaN.</param>
        /// <returns><paramref name="v"/> with its quiet bit set.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Quiet(double v) => BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(v) | 0x0008_0000_0000_0000UL);

        /// <summary>A float32 NaN quieted as x86 quiets a signalling one.</summary>
        /// <param name="v">A NaN.</param>
        /// <returns><paramref name="v"/> with its quiet bit set.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Quiet(float v) => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v) | 0x0040_0000U);

        /// <summary>
        ///     The result of an operation none of whose sources was NaN: a NaN there came from an invalid operation, and
        ///     is x86's default NaN on every host.
        /// </summary>
        /// <param name="r">The computed result.</param>
        /// <returns><paramref name="r"/>, or x86's default NaN in place of any NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Defaulted(double r) => double.IsNaN(r) ? BitConverter.UInt64BitsToDouble(DefaultNaNBits) : r;

        /// <summary>The float32 form of <see cref="Defaulted(double)"/>.</summary>
        /// <param name="r">The computed result.</param>
        /// <returns><paramref name="r"/>, or x86's default float32 NaN in place of any NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Defaulted(float r) => float.IsNaN(r) ? BitConverter.UInt32BitsToSingle(DefaultNaNBitsSingle) : r;

        /// <summary><c>a + b</c> as an x86 add whose first source is <paramref name="a"/>.</summary>
        /// <param name="a">The first source.</param>
        /// <param name="b">The second source.</param>
        /// <returns>The sum, <paramref name="a"/>'s NaN, else <paramref name="b"/>'s, else the default NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double NaNAdd(double a, double b)
            => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : Defaulted(a + b);

        /// <summary><c>a - b</c> as an x86 subtract (the first source is always <paramref name="a"/>).</summary>
        /// <param name="a">The minuend, the first source.</param>
        /// <param name="b">The subtrahend.</param>
        /// <returns>The difference, <paramref name="a"/>'s NaN, else <paramref name="b"/>'s (its sign kept: a NaN is
        ///     not negated), else the default NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double NaNSub(double a, double b)
            => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : Defaulted(a - b);

        /// <summary><c>a · b</c> as an x86 multiply whose first source is <paramref name="a"/>.</summary>
        /// <param name="a">The first source.</param>
        /// <param name="b">The second source.</param>
        /// <returns>The product, <paramref name="a"/>'s NaN, else <paramref name="b"/>'s, else the default NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double NaNMul(double a, double b)
            => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : Defaulted(a * b);

        /// <summary><c>a · b + c</c> as an x86 fused multiply-add: the multiplicands' NaNs before the addend's.</summary>
        /// <param name="a">The first multiplicand (vfmadd231's second register, the 213 form's too).</param>
        /// <param name="b">The second multiplicand.</param>
        /// <param name="c">The addend.</param>
        /// <returns>The correctly rounded result, the first NaN of <paramref name="a"/>, <paramref name="b"/>,
        ///     <paramref name="c"/>, else the default NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double NaNFma(double a, double b, double c)
            => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : double.IsNaN(c) ? Quiet(c)
                : Defaulted(Math.FusedMultiplyAdd(a, b, c));

        /// <summary>The float32 form of <see cref="NaNAdd(double, double)"/>.</summary>
        /// <param name="a">The first source.</param>
        /// <param name="b">The second source.</param>
        /// <returns>The float32 sum with x86's NaN choice.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float NaNAdd(float a, float b)
            => float.IsNaN(a) ? Quiet(a) : float.IsNaN(b) ? Quiet(b) : Defaulted(a + b);

        /// <summary>The float32 form of <see cref="NaNMul(double, double)"/>.</summary>
        /// <param name="a">The first source.</param>
        /// <param name="b">The second source.</param>
        /// <returns>The float32 product with x86's NaN choice.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float NaNMul(float a, float b)
            => float.IsNaN(a) ? Quiet(a) : float.IsNaN(b) ? Quiet(b) : Defaulted(a * b);

        /// <summary>The float32 form of <see cref="NaNFma(double, double, double)"/>.</summary>
        /// <param name="a">The first multiplicand.</param>
        /// <param name="b">The second multiplicand.</param>
        /// <param name="c">The addend.</param>
        /// <returns>The correctly rounded float32 result with x86's NaN choice.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float NaNFma(float a, float b, float c)
            => float.IsNaN(a) ? Quiet(a) : float.IsNaN(b) ? Quiet(b) : float.IsNaN(c) ? Quiet(c)
                : Defaulted(MathF.FusedMultiplyAdd(a, b, c));

        /// <summary>Whether either part of a complex value is NaN: the test a complex fast function's result takes.</summary>
        /// <param name="v">The value.</param>
        /// <returns>True when the real or the imaginary part is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool HasNaN(Complex v) => double.IsNaN(v.Real) || double.IsNaN(v.Imaginary);

        /// <summary>
        ///     Whether any of <paramref name="count"/> consecutive float32 values is NaN: the check a row function's caller
        ///     runs over the result once the rows have filled it (the N-D walk, np.vecdot's pair-row walks, np.correlate's
        ///     complex middle). Why there: a NaN branch inside a row's four-output loop puts a call there, and the values
        ///     live across it took the loop's registers (np.vecdot of reversed rows measured 0.74-0.79x, short unit rows of
        ///     3-term dots 0.86x even with a scan per row); this one vector pass costs about a quarter of a cycle per
        ///     output and calls nothing.
        /// </summary>
        /// <param name="p">The first value.</param>
        /// <param name="count">The number of values.</param>
        /// <returns>True when one is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe bool AnyNaN(float* p, long count)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                // vcmpunordps sets a lane where EITHER operand's lane is NaN, so one compare tests two vectors; two
                // independent OR accumulators over four loads a step (half a uop per value: one compare and AND per
                // vector measured 0.14 ns a value, 7% of np.convolve with a 3-term complex kernel).
                var m0 = Vector256<float>.Zero;
                var m1 = Vector256<float>.Zero;
                for (; i + 32 <= count; i += 32)
                {
                    m0 = Avx.Or(m0, Avx.CompareUnordered(Avx.LoadVector256(p + i), Avx.LoadVector256(p + i + 8)));
                    m1 = Avx.Or(m1, Avx.CompareUnordered(Avx.LoadVector256(p + i + 16), Avx.LoadVector256(p + i + 24)));
                }
                for (; i + 8 <= count; i += 8)
                {
                    var v = Avx.LoadVector256(p + i);
                    m0 = Avx.Or(m0, Avx.CompareUnordered(v, v));
                }
                // The mask's sign bits, not a float compare: an all-ones lane is itself a NaN.
                if (Vector256.ExtractMostSignificantBits(Avx.Or(m0, m1)) != 0u)
                    return true;
            }
            for (; i < count; i++)
                if (float.IsNaN(p[i]))
                    return true;
            return false;
        }

        /// <summary>The float64 form of <see cref="AnyNaN(float*, long)"/> (a complex row passes its parts: twice the
        ///     count).</summary>
        /// <param name="p">The first value.</param>
        /// <param name="count">The number of values.</param>
        /// <returns>True when one is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe bool AnyNaN(double* p, long count)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                // One vcmpunordpd tests two vectors; two OR accumulators over four loads a step (see the float32 form).
                var m0 = Vector256<double>.Zero;
                var m1 = Vector256<double>.Zero;
                for (; i + 16 <= count; i += 16)
                {
                    m0 = Avx.Or(m0, Avx.CompareUnordered(Avx.LoadVector256(p + i), Avx.LoadVector256(p + i + 4)));
                    m1 = Avx.Or(m1, Avx.CompareUnordered(Avx.LoadVector256(p + i + 8), Avx.LoadVector256(p + i + 12)));
                }
                for (; i + 4 <= count; i += 4)
                {
                    var v = Avx.LoadVector256(p + i);
                    m0 = Avx.Or(m0, Avx.CompareUnordered(v, v));
                }
                if (Vector256.ExtractMostSignificantBits(Avx.Or(m0, m1)) != 0u)
                    return true;
            }
            for (; i < count; i++)
                if (double.IsNaN(p[i]))
                    return true;
            return false;
        }

        // ------------------------------------------------------------------------------------------------------------
        //  float32: FLOAT_dot
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <see cref="FloatDotUnit"/> / <see cref="FloatDotUnitInOrder"/> with NumPy's NaN choice: sdot.c's unit-stride
        ///     branch - the Haswell kernel's 32 lanes of <c>fma(x, y, acc)</c> (vfmadd231ps with x in the second register),
        ///     each accumulator folded upper half first, <c>t5 + t4</c>, <c>t7 + t6</c>, <c>t76 + t54</c>, two horizontal
        ///     adds; the tail's <c>dot += y[i] * x[i]</c> in double (y first, the sum first); <c>dot += mydot</c>.
        /// </summary>
        /// <param name="x">cblas' x: logical element 0.</param>
        /// <param name="sx">Its element stride (any sign: a reversed or broadcast run is read in its logical order).</param>
        /// <param name="y">cblas' y: logical element 0.</param>
        /// <param name="sy">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's FLOAT_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe float FloatUnitNaN(float* x, long sx, float* y, long sy, long n)
        {
            long n1 = n > 0 ? n & -32 : 0;
            float mydot = 0f;
            if (n1 > 0)
            {
                float* acc = stackalloc float[32];   // acc[8k + L]: accumulator k (ymm4 + k), lane L
                for (int l = 0; l < 32; l++)
                    acc[l] = 0f;
                for (long i = 0; i < n1; i += 32)
                    for (int l = 0; l < 32; l++)
                        acc[l] = NaNFma(x[(i + l) * sx], y[(i + l) * sy], acc[l]);

                float* t = stackalloc float[16];   // t[4k + L]: accumulator k folded, `vaddps lower, upper` (upper first)
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++)
                        t[4 * k + l] = NaNAdd(acc[8 * k + 4 + l], acc[8 * k + l]);
                float* w = stackalloc float[4];
                for (int l = 0; l < 4; l++)
                {
                    float t54 = NaNAdd(t[4 + l], t[l]);
                    float t76 = NaNAdd(t[12 + l], t[8 + l]);
                    w[l] = NaNAdd(t76, t54);
                }
                // vhaddps twice: lane 0 + lane 1 (even lane first), then the two pair sums.
                mydot = NaNAdd(NaNAdd(w[0], w[1]), NaNAdd(w[2], w[3]));
            }

            double dot = 0.0;
            for (long i = n1; i < n; i++)
                dot = NaNAdd(dot, (double)NaNMul(y[i * sy], x[i * sx]));
            dot = NaNAdd(dot, (double)mydot);
            // cblas_sdot returns (float)dot; FLOAT_dot adds it into its zeroed double and rounds again.
            return (float)NaNAdd(0.0, (double)(float)dot);
        }

        /// <summary>
        ///     <see cref="FloatDotStrided"/> with NumPy's NaN choice: sdot.c's strided pairs
        ///     <c>dot += y[i]*x[i] + y[i+1]*x[i+1]</c> (each product y first, the pair's first product first, the sum
        ///     first), an odd last term alone.
        /// </summary>
        /// <param name="x">cblas' x: element 0.</param>
        /// <param name="sx">Its element stride.</param>
        /// <param name="y">cblas' y: element 0.</param>
        /// <param name="sy">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's FLOAT_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe float FloatStridedNaN(float* x, long sx, float* y, long sy, long n)
        {
            double dot = 0.0;
            long m = n & -2;
            long i = 0;
            for (; i < m; i += 2)
                dot = NaNAdd(dot, (double)NaNAdd(NaNMul(y[i * sy], x[i * sx]), NaNMul(y[(i + 1) * sy], x[(i + 1) * sx])));
            for (; i < n; i++)
                dot = NaNAdd(dot, (double)NaNMul(y[i * sy], x[i * sx]));
            return (float)NaNAdd(0.0, (double)(float)dot);
        }

        /// <summary>
        ///     <see cref="FloatDotOwn"/> with NumPy's NaN choice: FLOAT_dot's own loop as MSVC compiled it, unrolled by
        ///     four over the first <c>n &amp; -4</c> terms - each group's first product <c>ip1 · ip2</c>, the other three
        ///     <c>ip2 · ip1</c> - the remaining terms <c>ip1 · ip2</c>, the sum first in every add.
        /// </summary>
        /// <param name="ip1">The first operand's element 0.</param>
        /// <param name="is1">Its element stride.</param>
        /// <param name="ip2">The second operand's element 0.</param>
        /// <param name="is2">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's FLOAT_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe float FloatOwnNaN(float* ip1, long is1, float* ip2, long is2, long n)
        {
            float s = 0f;
            long n4 = n & -4;
            for (long i = 0; i < n; i++)
            {
                float a = ip1[i * is1], b = ip2[i * is2];
                s = NaNAdd(s, i < n4 && (i & 3) != 0 ? NaNMul(b, a) : NaNMul(a, b));
            }
            return s;
        }

        // ------------------------------------------------------------------------------------------------------------
        //  float64: DOUBLE_dot
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <see cref="DoubleDotUnit"/> / <see cref="DoubleDotUnitInOrder"/> with NumPy's NaN choice: ddot.c's
        ///     unit-stride branch - the Haswell kernel's 16 lanes of <c>fma(x, y, acc)</c>, each accumulator folded upper
        ///     half first, <c>t5 + t4</c>, <c>t7 + t6</c>, <c>t76 + t54</c>, a horizontal add (lane 0 first); then the
        ///     tail's <c>dot += y[i] * x[i]</c> (y first, the sum first).
        /// </summary>
        /// <param name="x">cblas' x: logical element 0.</param>
        /// <param name="sx">Its element stride (any sign).</param>
        /// <param name="y">cblas' y: logical element 0.</param>
        /// <param name="sy">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's DOUBLE_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe double DoubleUnitNaN(double* x, long sx, double* y, long sy, long n)
        {
            long n1 = n > 0 ? n & -16 : 0;
            double dot = 0.0;
            if (n1 > 0)
            {
                double* acc = stackalloc double[16];   // acc[4k + L]: accumulator k (ymm4 + k), lane L
                for (int l = 0; l < 16; l++)
                    acc[l] = 0.0;
                for (long i = 0; i < n1; i += 16)
                    for (int l = 0; l < 16; l++)
                        acc[l] = NaNFma(x[(i + l) * sx], y[(i + l) * sy], acc[l]);

                double* t = stackalloc double[8];   // t[2k + L]: accumulator k folded, upper half first
                for (int k = 0; k < 4; k++)
                {
                    t[2 * k] = NaNAdd(acc[4 * k + 2], acc[4 * k]);
                    t[2 * k + 1] = NaNAdd(acc[4 * k + 3], acc[4 * k + 1]);
                }
                double w0 = NaNAdd(NaNAdd(t[6], t[4]), NaNAdd(t[2], t[0]));
                double w1 = NaNAdd(NaNAdd(t[7], t[5]), NaNAdd(t[3], t[1]));
                dot = NaNAdd(w0, w1);   // the kernel STORES this into dot
            }

            for (long i = n1; i < n; i++)
                dot = NaNAdd(dot, NaNMul(y[i * sy], x[i * sx]));
            return NaNAdd(0.0, dot);
        }

        /// <summary>
        ///     <see cref="DoubleDotStrided"/> with NumPy's NaN choice: ddot.c's strided loop, <c>m = y · x</c> per term,
        ///     <c>t1 += m1 + m3; t2 += m2 + m4</c> (the sum first, the earlier product first), the remainder into
        ///     <c>t1</c>, and the result <c>t2 + t1</c> as GCC ordered it.
        /// </summary>
        /// <param name="x">cblas' x: element 0.</param>
        /// <param name="sx">Its element stride.</param>
        /// <param name="y">cblas' y: element 0.</param>
        /// <param name="sy">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's DOUBLE_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe double DoubleStridedNaN(double* x, long sx, double* y, long sy, long n)
        {
            double t1 = 0.0, t2 = 0.0;
            long n4 = n & -4;
            long i = 0;
            for (; i < n4; i += 4)
            {
                double m1 = NaNMul(y[i * sy], x[i * sx]);
                double m2 = NaNMul(y[(i + 1) * sy], x[(i + 1) * sx]);
                double m3 = NaNMul(y[(i + 2) * sy], x[(i + 2) * sx]);
                double m4 = NaNMul(y[(i + 3) * sy], x[(i + 3) * sx]);
                t1 = NaNAdd(t1, NaNAdd(m1, m3));
                t2 = NaNAdd(t2, NaNAdd(m2, m4));
            }
            for (; i < n; i++)
                t1 = NaNAdd(t1, NaNMul(y[i * sy], x[i * sx]));
            return NaNAdd(0.0, NaNAdd(t2, t1));
        }

        /// <summary>
        ///     <see cref="DoubleDotOwn"/> with NumPy's NaN choice: DOUBLE_dot's own loop, <c>sum += ip1 · ip2</c>
        ///     (<c>ip1</c> first, the sum first).
        /// </summary>
        /// <param name="ip1">The first operand's element 0.</param>
        /// <param name="is1">Its element stride.</param>
        /// <param name="ip2">The second operand's element 0.</param>
        /// <param name="is2">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's DOUBLE_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe double DoubleOwnNaN(double* ip1, long is1, double* ip2, long is2, long n)
        {
            double s = 0.0;
            for (long i = 0; i < n; i++)
                s = NaNAdd(s, NaNMul(ip1[i * is1], ip2[i * is2]));
            return s;
        }

        // ------------------------------------------------------------------------------------------------------------
        //  complex128: CDOUBLE_dot (zdotu) and CDOUBLE_dotc / CDOUBLE_vdot (zdotc)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     zdot_compute's four sums with NumPy's NaN choice, <c>d0 = Σ xr·yr</c>, <c>d1 = Σ xi·yi</c>,
        ///     <c>d2 = Σ xr·yi</c>, <c>d3 = Σ xi·yr</c>: for two unit strides the Haswell kernel over the first
        ///     <c>n &amp; -8</c> terms - <c>fma(y, x, acc)</c> (vfmadd231pd with y in the second register) into four
        ///     accumulators of <c>x · y</c> and four of <c>x · swap(y)</c>, reduced <c>a1 + a0</c>, <c>a3 + a2</c>,
        ///     <c>a32 + a10</c> and folded upper half first - then the C loop over the rest; for other positive strides
        ///     the C loop over every term. The C loop's products are <c>x · y</c> but the fourth, <c>yr · xi</c>, and every
        ///     sum comes first in its add.
        /// </summary>
        /// <param name="x">cblas' x: logical element 0's real part (interleaved doubles).</param>
        /// <param name="sx">Its stride in complex elements (any sign).</param>
        /// <param name="y">cblas' y: logical element 0's real part.</param>
        /// <param name="sy">Its stride in complex elements.</param>
        /// <param name="n">The number of terms.</param>
        /// <param name="unit">Whether cblas saw two unit strides (the kernel runs).</param>
        /// <param name="d">The four sums, written.</param>
        private static unsafe void ZdotSumsNaN(double* x, long sx, double* y, long sy, long n, bool unit, double* d)
        {
            d[0] = d[1] = d[2] = d[3] = 0.0;
            long c0 = 0;
            if (unit && n >= 8)
            {
                long n1 = n & -8;
                double* acc = stackalloc double[32];   // acc[4k + L]: accumulator k (ymm0..ymm7), lane L
                for (int l = 0; l < 32; l++)
                    acc[l] = 0.0;
                for (long b = 0; b < n1; b += 8)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        for (int h = 0; h < 2; h++)
                        {
                            // Accumulator k holds terms b + 2k (lanes 0, 1) and b + 2k + 1 (lanes 2, 3).
                            double* xs = x + 2 * (b + 2 * k + h) * sx;
                            double* ys = y + 2 * (b + 2 * k + h) * sy;
                            int l = 4 * k + 2 * h;
                            acc[l] = NaNFma(ys[0], xs[0], acc[l]);
                            acc[l + 1] = NaNFma(ys[1], xs[1], acc[l + 1]);
                            acc[16 + l] = NaNFma(ys[1], xs[0], acc[16 + l]);       // swap(y): yi · xr
                            acc[16 + l + 1] = NaNFma(ys[0], xs[1], acc[16 + l + 1]); // yr · xi
                        }
                    }
                }

                for (int g = 0; g < 2; g++)
                {
                    double* a = acc + 16 * g;   // g = 0: x·y (d0, d1); g = 1: x·swap(y) (d2, d3)
                    double r0 = NaNAdd(NaNAdd(a[12], a[8]), NaNAdd(a[4], a[0]));
                    double r1 = NaNAdd(NaNAdd(a[13], a[9]), NaNAdd(a[5], a[1]));
                    double r2 = NaNAdd(NaNAdd(a[14], a[10]), NaNAdd(a[6], a[2]));
                    double r3 = NaNAdd(NaNAdd(a[15], a[11]), NaNAdd(a[7], a[3]));
                    d[2 * g] = NaNAdd(r2, r0);       // vextractf128 + vaddpd: the upper half first
                    d[2 * g + 1] = NaNAdd(r3, r1);
                }
                c0 = n1;
            }

            for (long c = c0; c < n; c++)
            {
                double* xs = x + 2 * c * sx;
                double* ys = y + 2 * c * sy;
                d[0] = NaNAdd(d[0], NaNMul(xs[0], ys[0]));
                d[1] = NaNAdd(d[1], NaNMul(xs[1], ys[1]));
                d[2] = NaNAdd(d[2], NaNMul(xs[0], ys[1]));
                d[3] = NaNAdd(d[3], NaNMul(ys[0], xs[1]));
            }
        }

        /// <summary>
        ///     <see cref="ZdotuResult"/> with NumPy's NaN choice: the C99 receipt <c>re + im · I</c> - the real part
        ///     <c>re + im·0</c>, the imaginary part <c>im·1</c> - added into the dot function's zeroed pair.
        /// </summary>
        /// <param name="re">zdot's real part.</param>
        /// <param name="im">zdot's imaginary part.</param>
        /// <returns>The value NumPy's complex128 dot function writes.</returns>
        private static Complex ZdotResultNaN(double re, double im)
            => new Complex(NaNAdd(0.0, NaNAdd(re, NaNMul(im, 0.0))), NaNAdd(0.0, NaNMul(im, 1.0)));

        /// <summary>
        ///     <see cref="ComplexDotUnit"/>, <see cref="ComplexDotUnitInOrder"/>, <see cref="ComplexDotStrided"/> and their
        ///     zdotc twins with NumPy's NaN choice: the four sums (<see cref="ZdotSumsNaN"/>), combined as zdotu
        ///     (<c>d0 − d1</c>, <c>d2 + d3</c>) or zdotc (<c>d0 + d1</c>, <c>d2 − d3</c>), received through the C99 result.
        /// </summary>
        /// <param name="x">cblas' x: logical element 0.</param>
        /// <param name="sx">Its element stride (any sign).</param>
        /// <param name="y">cblas' y: logical element 0.</param>
        /// <param name="sy">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <param name="unit">Whether cblas saw two unit strides.</param>
        /// <param name="conjugate">zdotc (np.vdot, np.vecdot) rather than zdotu (np.dot, np.correlate, np.convolve).</param>
        /// <returns>The value NumPy's dot function writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe Complex ComplexBlasNaN(Complex* x, long sx, Complex* y, long sy, long n, bool unit,
            bool conjugate)
        {
            double* d = stackalloc double[4];
            ZdotSumsNaN((double*)x, sx, (double*)y, sy, n, unit, d);
            return conjugate
                ? ZdotResultNaN(NaNAdd(d[0], d[1]), NaNSub(d[2], d[3]))
                : ZdotResultNaN(NaNSub(d[0], d[1]), NaNAdd(d[2], d[3]));
        }

        /// <summary>
        ///     <see cref="ComplexDotOwn"/> with NumPy's NaN choice: CDOUBLE_dot's own loop as MSVC compiled it - every
        ///     product <c>ip2</c> part first, <c>sumr += p1 − p2</c> and <c>sumi += p4 + p3</c> (the sums first).
        /// </summary>
        /// <param name="ip1">The first operand's element 0.</param>
        /// <param name="is1">Its element stride.</param>
        /// <param name="ip2">The second operand's element 0.</param>
        /// <param name="is2">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's CDOUBLE_dot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe Complex ComplexOwnNaN(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            double sumr = 0.0, sumi = 0.0;
            for (long i = 0; i < n; i++)
            {
                double* a = (double*)(ip1 + i * is1);
                double* b = (double*)(ip2 + i * is2);
                double p1 = NaNMul(b[0], a[0]), p2 = NaNMul(b[1], a[1]);   // ar·br, ai·bi
                double p3 = NaNMul(b[1], a[0]), p4 = NaNMul(b[0], a[1]);   // ar·bi, ai·br
                sumr = NaNAdd(sumr, NaNSub(p1, p2));
                sumi = NaNAdd(sumi, NaNAdd(p4, p3));
            }
            return new Complex(sumr, sumi);
        }

        /// <summary>
        ///     <see cref="ComplexDotcOwn"/> with NumPy's NaN choice: CDOUBLE_dotc's (matmul.c.src) and CDOUBLE_vdot's
        ///     (vdot.c) own loop, which MSVC compiled alike - every product <c>ip2</c> part first,
        ///     <c>sumr += p2 + p1</c> and <c>sumi += p3 − p4</c> (the sums first).
        /// </summary>
        /// <param name="ip1">The conjugated operand's element 0.</param>
        /// <param name="is1">Its element stride.</param>
        /// <param name="ip2">The second operand's element 0.</param>
        /// <param name="is2">Its element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The value NumPy's CDOUBLE_dotc / CDOUBLE_vdot writes, NaN bits included.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static unsafe Complex ComplexDotcOwnNaN(Complex* ip1, long is1, Complex* ip2, long is2, long n)
        {
            double sumr = 0.0, sumi = 0.0;
            for (long i = 0; i < n; i++)
            {
                double* a = (double*)(ip1 + i * is1);
                double* b = (double*)(ip2 + i * is2);
                double p1 = NaNMul(b[0], a[0]), p2 = NaNMul(b[1], a[1]);   // ar·br, ai·bi
                double p3 = NaNMul(b[1], a[0]), p4 = NaNMul(b[0], a[1]);   // ar·bi, ai·br
                sumr = NaNAdd(sumr, NaNAdd(p2, p1));
                sumi = NaNAdd(sumi, NaNSub(p3, p4));
            }
            return new Complex(sumr, sumi);
        }

        // ------------------------------------------------------------------------------------------------------------
        //  One-term own loops (np.vecdot's core of one element) and the plain products of np.dot's scalar branch
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>FLOAT_dot's own loop for one term with NumPy's NaN choice: <c>0 + ip1 · ip2</c> (<c>ip1</c> first; a
        ///     single term is no unrolled group).</summary>
        /// <param name="a">The first operand's element.</param>
        /// <param name="b">The second operand's element.</param>
        /// <returns>The value NumPy writes, NaN bits included.</returns>
        internal static float FloatOneNaN(float a, float b) => NaNAdd(0f, NaNMul(a, b));

        /// <summary>DOUBLE_dot's own loop for one term with NumPy's NaN choice: <c>0 + ip1 · ip2</c>.</summary>
        /// <param name="a">The first operand's element.</param>
        /// <param name="b">The second operand's element.</param>
        /// <returns>The value NumPy writes, NaN bits included.</returns>
        internal static double DoubleOneNaN(double a, double b) => NaNAdd(0.0, NaNMul(a, b));

        /// <summary>
        ///     cblas_matrixproduct's plain product for one output, <c>*ap2 · *ap1</c> (ap2 the operand its scalar test
        ///     swapped in), with NumPy's NaN choice: MSVC kept <c>ap2</c> first for float64 and put <c>ap1</c> first for
        ///     float32.
        /// </summary>
        /// <param name="ap2">The value <c>ap2</c> holds.</param>
        /// <param name="ap1">The value <c>ap1</c> holds.</param>
        /// <returns>The product NumPy writes, NaN bits included.</returns>
        internal static double ScalarProductNaN(double ap2, double ap1) => NaNMul(ap2, ap1);

        /// <summary>The float32 form of <see cref="ScalarProductNaN(double, double)"/>: <c>ap1</c> first.</summary>
        /// <param name="ap2">The value <c>ap2</c> holds.</param>
        /// <param name="ap1">The value <c>ap1</c> holds.</param>
        /// <returns>The product NumPy writes, NaN bits included.</returns>
        internal static float ScalarProductNaN(float ap2, float ap1) => NaNMul(ap1, ap2);

        /// <summary>
        ///     cblas_matrixproduct's complex plain product for one output with NumPy's NaN choice:
        ///     <c>(p1r·p2r − p1i·p2i, p1r·p2i + p2r·p1i)</c> where <c>p1 = *ap2</c> and <c>p2 = *ap1</c> - the last
        ///     product <c>p2r</c> first, as MSVC ordered it.
        /// </summary>
        /// <param name="p1">The value <c>ap2</c> holds (<c>ptr1</c> in cblasfuncs.c).</param>
        /// <param name="p2">The value <c>ap1</c> holds (<c>ptr2</c>).</param>
        /// <returns>The product NumPy writes, NaN bits included.</returns>
        internal static Complex ScalarProductNaN(Complex p1, Complex p2)
            => new Complex(NaNSub(NaNMul(p1.Real, p2.Real), NaNMul(p1.Imaginary, p2.Imaginary)),
                NaNAdd(NaNMul(p1.Real, p2.Imaginary), NaNMul(p2.Real, p1.Imaginary)));

        /// <summary>
        ///     One output of np.dot's complex scalar branch with a NaN in the scalar, <c>y = 0 + a · x</c> as zaxpy
        ///     computes it: the Haswell kernel's <c>fma(xi, ±ai, fma(ar, xr, 0))</c> / <c>fma(xr, ai, fma(ar, xi, 0))</c>
        ///     (the 213 form puts <c>ar</c> first, <c>±ai</c> is <c>ai · ∓1</c>, which keeps a NaN's sign), or the C loop's
        ///     <c>0 + ((ar·xr) − (ai·xi))</c> / <c>0 + ((ar·xi) + (ai·xr))</c>, the scalar's part first in every product.
        /// </summary>
        /// <param name="a">The scalar.</param>
        /// <param name="x">The element.</param>
        /// <param name="kernel">Whether NumPy's zaxpy call computes this output in its vector kernel.</param>
        /// <returns>The value NumPy writes, NaN bits included.</returns>
        internal static Complex ZaxpyNaN(Complex a, Complex x, bool kernel)
        {
            double ar = a.Real, ai = a.Imaginary, xr = x.Real, xi = x.Imaginary;
            if (kernel)
            {
                double nai = NaNMul(ai, -1.0);   // vmulpd by mvec = {-1, 1, -1, 1}: a NaN is returned as it is
                return new Complex(NaNFma(xi, nai, NaNFma(ar, xr, 0.0)), NaNFma(xr, NaNMul(ai, 1.0), NaNFma(ar, xi, 0.0)));
            }
            return new Complex(NaNAdd(0.0, NaNSub(NaNMul(ar, xr), NaNMul(ai, xi))),
                NaNAdd(0.0, NaNAdd(NaNMul(ar, xi), NaNMul(ai, xr))));
        }

        /// <summary>One output of np.dot's real scalar branch with a NaN scalar: the scalar's NaN, quieted (both the
        ///     axpy kernel's <c>fma(a, x, 0)</c> and its C loop's <c>0 + a·x</c> put the scalar first).</summary>
        /// <param name="a">The scalar (NaN).</param>
        /// <returns>The value NumPy writes for every output.</returns>
        internal static double AxpyNaNScalar(double a) => Quiet(a);

        /// <summary>The float32 form of <see cref="AxpyNaNScalar(double)"/>.</summary>
        /// <param name="a">The scalar (NaN).</param>
        /// <returns>The value NumPy writes for every output.</returns>
        internal static float AxpyNaNScalar(float a) => Quiet(a);
    }
}
