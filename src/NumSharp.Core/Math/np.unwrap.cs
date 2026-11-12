using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== np.unwrap ==============================
    // Unwrap a signal by taking the complement of large deltas w.r.t. a period.
    //
    // Changes every element whose absolute difference from its predecessor
    // exceeds max(discont, period/2) to its period-complementary value, so that
    // adjacent differences never exceed period/2. For the default period 2*pi
    // and discont pi this unwraps a radian phase by adding 2*k*pi.
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::unwrap. NumPy is a
    // pure composition, and NumSharp mirrors it operation-for-operation so the
    // NEP50 dtype promotion, the integer/float path split, broadcasting and every
    // edge case fall out of the existing (parity-tested) building blocks:
    //
    //     dd    = diff(p, axis)
    //     ddmod = mod(dd - interval_low, period) + interval_low
    //     if boundary_ambiguous:                 # dd exactly +period/2 -> +period/2
    //         ddmod[(ddmod == interval_low) & (dd > 0)] = interval_high
    //     ph_correct        = ddmod - dd
    //     ph_correct[|dd| < discont] = 0          # small jumps are not unwrapped
    //     up            = p.astype(dtype)         # dtype = result_type(dd, period)
    //     up[1:]        = p[1:] + cumsum(ph_correct, axis)
    //
    // WHY the integer/float split. NumPy chooses the interval half-width and the
    // "boundary is ambiguous" flag from `result_type(dd, period)`:
    //   * FLOAT result (any float input, OR an integer input with a FLOAT period)
    //     -> interval_high = period/2 (a real number), boundary always ambiguous.
    //   * INTEGER result (an integer/bool input AND an INTEGER-typed period)
    //     -> interval_high = period//2 (floor), ambiguous only when period is even.
    // Because a Python int and a Python float are distinct at the call site, the
    // C# port needs the same distinction: the period is ONE `NDArray period` (since
    // API-parity batch 6; it replaced a `double period` and a `long period` overload,
    // whose positional `np.unwrap(a, 4)` bound period=4 where NumPy reads discont=4).
    // A C# integer converts to NumPy's weak Python int (the integer path), a C#
    // double to its weak Python float, and a NumPy scalar or a 0-d array is a STRONG
    // period whose dtype joins result_type(dd, period). Note that for an
    // INTEGER input the two paths compute the SAME values (an integer difference
    // never lands in the (period//2, period/2] gap, and the ambiguity fix only
    // fires on that gap); they differ only in the OUTPUT DTYPE — integer-typed
    // period keeps the input's integer dtype (bool -> int64), float period yields
    // float64. Unsigned integer inputs with an integer period reproduce NumPy's
    // OverflowError (interval_low is negative and cannot be cast to the unsigned
    // dtype); complex inputs reproduce NumPy's TypeError (mod rejects complex).
    //
    // FUSION / PERFORMANCE. The whole elementwise chain that maps `dd` to
    // `ph_correct` (subtract, mod, add, an optional boundary Where, subtract, and
    // the discont Where) is built as ONE NDExpr and produced in a single fused
    // np.evaluate pass, where NumPy allocates ~6 intermediate arrays. Only three
    // array passes remain (diff, the fused ph_correct, cumsum), so unwrap is
    // 2-9x faster than NumPy at 100K-10M elements while staying bit-identical.
    // The scalars (interval_low/high, period, discont) ride as NEP50-weak NDExpr
    // constants: on the float path a weak double never widens the operand's float
    // dtype (float16/float32 stay put); on the integer path a weak int keeps the
    // integer dtype and triggers the OverflowError exactly as NumPy's does. Every
    // loop lives inside np.diff / np.evaluate / np.cumsum — there is no hand
    // per-element loop here.
    public static partial class np
    {
        /// <summary>
        ///     Unwrap <paramref name="p"/> by changing deltas larger than <c>max(discont, period/2)</c> to their
        ///     <paramref name="period"/>-complement, so adjacent differences never exceed <c>period/2</c>: NumPy's
        ///     <c>unwrap(p, discont=None, axis=-1, *, period=2*pi)</c>, one signature in NumPy's order.
        /// </summary>
        /// <param name="p">
        ///     Input array (at least one dimensional; a 0-D input is rejected exactly as NumPy's inner
        ///     <see cref="diff(NDArray,int,int,object,object)"/> rejects it).
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity between values. <c>null</c> (the default) means <c>period/2</c>. Values below
        ///     <c>period/2</c> have no effect: taking the complement of a jump smaller than <c>period/2</c> would only
        ///     enlarge it. A positional second argument is this one, as in NumPy: <c>np.unwrap(a, 4)</c> is
        ///     <c>discont=4</c>.
        /// </param>
        /// <param name="axis">Axis along which to unwrap; default is the last axis. Negative axes count from the end.</param>
        /// <param name="period">
        ///     Size of the range over which the input wraps; <c>null</c> is NumPy's default <c>2*pi</c>. Its kind picks
        ///     NumPy's path: a C# integer (NumPy's Python int) keeps an integer input integral (the interval half-width is
        ///     <c>period // 2</c>, the boundary ambiguous only for an even period; a bool input gives int64); a C# double
        ///     (a Python float) gives a float result that keeps a float input's width (an integer input gives float64). A
        ///     NumPy scalar or a 0-d array is NumPy's STRONG period: the result dtype is <c>result_type(diff(p), period)</c>
        ///     (an int8 input with <c>np.int64(4)</c> is int64, with <c>np.float32(4)</c> float32). NumPy's
        ///     <c>period</c> is keyword-only: write <c>period:</c>.
        /// </param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array of the unwrapped signal, <paramref name="p"/>'s shape, the dtype
        ///     described for <paramref name="period"/>. A <c>numpy.char.chararray</c> operand gives a plain array: NumPy
        ///     converts it with <c>asarray</c>, which drops the class.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="p"/> is 0-D (propagated from
        ///     <see cref="diff(NDArray,int,int,object,object)"/>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> or <paramref name="period"/> is complex (NumPy's <c>mod</c> has no
        ///     complex loop), or <paramref name="period"/> is text (<c>unsupported operand type(s) for /: 'str' and
        ///     'int'</c>).</exception>
        /// <exception cref="OverflowError">An integer path whose negative <c>interval_low</c> (or the period itself) the
        ///     input's integer dtype cannot hold - every unsigned input, a too-narrow signed one (NumPy's <c>Python integer
        ///     N out of bounds for &lt;dtype&gt;</c>), raised for empty inputs too.</exception>
        /// <exception cref="NotSupportedException">A <paramref name="period"/> of several elements (NumPy broadcasts it into
        ///     the elementwise steps), or a C# bool period over a bool <paramref name="p"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html. One signature since API-parity batch 6:
        ///     the <c>double period</c> and <c>long period</c> overloads it replaced made a positional integer second
        ///     argument bind <c>period</c> (<c>np.unwrap(a, 4)</c> returned the period-4 unwrap, where NumPy reads
        ///     <c>discont=4</c>).
        /// </remarks>
        public static NDArray unwrap(NDArray p, double? discont = null, int axis = -1, NDArray period = null)
        {
            if (p is null) throw new ArgumentNullException(nameof(p));
            // The engine reads the period's Python kind (weak C# number vs strong NumPy scalar / array); null is 2*pi.
            return p.TensorEngine.Unwrap(p, discont, axis, period);
        }
    }
}
