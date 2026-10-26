using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     NumPy's float16 dot-product sum, NaN for NaN: <c>HALF_dot</c> (arraytypes.c.src, behind np.dot, np.vecdot,
    ///     np.convolve and np.correlate) and the HALF loop of <c>matmul_inner_noblas</c> (matmul.c.src, behind np.matmul,
    ///     np.matvec and np.vecmat) widen each pair to float32, multiply there, add the products into a float32 sum from +0
    ///     in order, and narrow the sum to float16 once. NumSharp's float16 kernels do the same arithmetic, so they agree
    ///     with NumPy on every result that is not NaN; which NaN a NaN result carries (its sign and payload) depends on the
    ///     order of the operands of each multiply and add, and that order is what this class encodes.
    /// </summary>
    /// <remarks>
    ///     <para><b>NumPy's order</b> (probed 2.4.2 on win-amd64 with NaNs of distinct signs and payloads, identical for
    ///     np.matmul, np.dot, np.vecdot, np.convolve and np.correlate): in a product of two NaNs the SECOND factor's NaN
    ///     survives (MSVC compiles <c>a * b</c> with <c>b</c> as the first source of <c>mulss</c>, which returns its first
    ///     source's NaN), and in <c>sum += product</c> a NaN sum survives a NaN product (<c>addss sum, product</c>), so the
    ///     first NaN of the sequence wins. A NaN that arises inside the sum (<c>inf * 0</c>, <c>inf - inf</c>) is x86's
    ///     default NaN, negative.</para>
    ///     <para><b>Why it is spelled out.</b> RyuJIT orders the operands of a commutative operation as it likes (it puts a
    ///     memory operand second, for instance), so <c>acc += a * b</c> in C# says nothing about which NaN survives: the
    ///     matmul and dot kernels kept the product's NaN in the sum, the sliding kernels the later NaN. The two helpers
    ///     below cannot be reordered into another answer: <see cref="Multiply"/> returns a NaN <c>y</c> through
    ///     <c>y + y</c> (both operands the same NaN), and <see cref="Add"/> leaves a NaN sum untouched.</para>
    ///     <para><b>How the kernels use it.</b> They keep their fast loops and, for each output whose float32 sum is NaN,
    ///     recompute that output with <see cref="Sum(Half*, long, Half*, long, long)"/>: a sum that is not NaN does not
    ///     depend on the operand order, because a float32 multiply and add of non-NaN values are commutative.</para>
    /// </remarks>
    internal static class NDHalfDot
    {
        /// <summary>
        ///     The product NumPy's float16 loops form, <c>x * y</c> in float32, with NumPy's NaN priority: when
        ///     <paramref name="y"/> is NaN the answer is <paramref name="y"/> (quieted), whatever <paramref name="x"/> is.
        /// </summary>
        /// <param name="x">The first factor (NumPy's <c>ip1</c> / <c>val1</c>), widened to float32.</param>
        /// <param name="y">The second factor (<c>ip2</c> / <c>val2</c>), widened to float32.</param>
        /// <returns>The product; with a NaN factor, the NaN NumPy's product carries.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float Multiply(float x, float y) => float.IsNaN(y) ? y + y : x * y;

        /// <summary>
        ///     One step of NumPy's float32 sum, <c>sum + product</c>, with NumPy's NaN priority: a NaN
        ///     <paramref name="sum"/> stays, so the first NaN of the sequence decides the result.
        /// </summary>
        /// <param name="sum">The running sum.</param>
        /// <param name="product">The next product (<see cref="Multiply"/>).</param>
        /// <returns>The new running sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float Add(float sum, float product) => float.IsNaN(sum) ? sum : sum + product;

        /// <summary>
        ///     NumPy's float16 dot-product sum of two strided float16 vectors, in float32, NaN priorities included; the
        ///     caller narrows it to float16 (NumPy narrows once).
        /// </summary>
        /// <param name="a">The first operand's element 0 (NumPy's <c>ip1</c>).</param>
        /// <param name="sa">Its element stride (any sign).</param>
        /// <param name="b">The second operand's element 0 (<c>ip2</c>).</param>
        /// <param name="sb">Its element stride (any sign).</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The float32 sum.</returns>
        internal static unsafe float Sum(Half* a, long sa, Half* b, long sb, long n)
        {
            float s = 0f;
            for (long t = 0; t < n; t++)
                s = Add(s, Multiply((float)a[t * sa], (float)b[t * sb]));
            return s;
        }
    }
}
