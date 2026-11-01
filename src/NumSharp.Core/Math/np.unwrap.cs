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
    // C# port needs the same distinction, which is why there are two overloads:
    // the primary `double period` (the NumPy default and every float use) and a
    // `long period` overload for the integer-preserving path. Note that for an
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
        ///     Unwrap <paramref name="p"/> by changing deltas larger than
        ///     <c>max(discont, period/2)</c> to their <paramref name="period"/>-complement,
        ///     so adjacent differences never exceed <c>period/2</c>. This is the
        ///     FLOAT-period overload — the NumPy default and the form used for radian
        ///     phase unwrapping.
        /// </summary>
        /// <param name="p">
        ///     Input array (at least one dimensional; a 0-D input is rejected exactly
        ///     as NumPy's inner <see cref="diff(NDArray,int,int,object,object)"/> rejects it).
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity between values. <c>null</c> (the default) means
        ///     <c>period/2</c>. Values below <c>period/2</c> have no effect: taking the
        ///     complement of a jump smaller than <c>period/2</c> would only enlarge it,
        ///     so a smaller <paramref name="discont"/> is clamped up to <c>period/2</c>.
        ///     To change behaviour, pass a <paramref name="discont"/> LARGER than
        ///     <c>period/2</c>.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to unwrap; default is the last axis. Negative axes
        ///     count from the end.
        /// </param>
        /// <param name="period">
        ///     Size of the range over which the input wraps (default <c>2*pi</c>).
        ///     This overload treats the period as a real number (NumPy's weak float
        ///     scalar), so the result is always float — a float input keeps its width
        ///     (float16/float32/float64) and an integer/bool input yields float64. To
        ///     get the integer-preserving behaviour of NumPy's <c>period=&lt;int&gt;</c>,
        ///     call the <see cref="unwrap(NDArray,long,System.Nullable{double},int)"/>
        ///     overload (e.g. <c>np.unwrap(a, period: 4)</c>).
        /// </param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array of the unwrapped signal. The
        ///     shape matches <paramref name="p"/>; the dtype is the float
        ///     <c>result_type(diff(p), period)</c> described above.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="p"/> is 0-D (propagated from <see cref="diff(NDArray,int,int,object,object)"/>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is complex (NumPy's <c>mod</c> has no complex loop).</exception>
        /// <remarks>
        ///     NumPy's <c>period</c> is keyword-only, so ported code always writes
        ///     <c>period:</c>. Beware the one C# resolution wrinkle: a bare positional
        ///     integer second argument — <c>np.unwrap(a, 4)</c> — binds the
        ///     <see cref="unwrap(NDArray,long,System.Nullable{double},int)"/> overload
        ///     (period = 4), NOT <paramref name="discont"/> = 4. To set
        ///     <paramref name="discont"/> positionally, pass a double (<c>np.unwrap(a, 4.0)</c>)
        ///     or name it (<c>np.unwrap(a, discont: 4)</c>).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html
        /// </remarks>
        public static NDArray unwrap(NDArray p, double? discont = null, int axis = -1, double period = 2 * Math.PI)
        {
            if (p is null) throw new ArgumentNullException(nameof(p));
            return NDMatrixBoundary.Sees(p)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((p, discont, axis, period), static s => unwrap(s.p, s.discont, s.axis, s.period))
            : NDCharArrayBoundary.Sees(p)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((p, discont, axis, period), static s => unwrap(s.p, s.discont, s.axis, s.period))
            : p.TensorEngine.Unwrap(p, discont, axis, period);
        }

        /// <summary>
        ///     Unwrap <paramref name="p"/> with an INTEGER-typed <paramref name="period"/>,
        ///     the integer-preserving form of NumPy's <c>np.unwrap(a, period=&lt;int&gt;)</c>.
        ///     Use this (rather than the <c>double</c> overload) when you want NumPy's
        ///     integer output dtype — e.g. <c>np.unwrap(new[]{0,1,2,-1,0}, period: 4)</c>
        ///     returns <c>[0,1,2,3,4]</c> as an integer array.
        /// </summary>
        /// <param name="p">Input array (at least one dimensional).</param>
        /// <param name="period">
        ///     Size of the wrap range as an INTEGER. With an integer/bool input this
        ///     selects NumPy's integer code path: the interval half-width is
        ///     <c>period/2</c> floored, the boundary is ambiguous only when
        ///     <paramref name="period"/> is even, and the OUTPUT keeps the input's
        ///     integer dtype (a bool input yields int64). With a FLOAT input the result
        ///     is still float (an integer period cannot make a float signal integral),
        ///     identical to the double overload.
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity; <c>null</c> (default) means <c>period/2.0</c>
        ///     (a real number, matching NumPy's Python <c>period/2</c> true division
        ///     even on the integer path). See the double overload for semantics.
        /// </param>
        /// <param name="axis">Axis along which to unwrap; default is the last axis.</param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array. Integer/bool inputs keep their
        ///     integer dtype (bool -> int64); float inputs stay float.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="p"/> is 0-D.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is complex.</exception>
        /// <exception cref="OverflowException">
        ///     <paramref name="p"/> is an unsigned integer (or char) — or a signed
        ///     integer too narrow for the derived interval — so the negative
        ///     <c>interval_low</c> (or <paramref name="period"/> itself) cannot be
        ///     represented in the input dtype. This reproduces NumPy's OverflowError
        ///     ("Python integer N out of bounds for &lt;dtype&gt;"), which fires
        ///     regardless of array size (empty inputs included).
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html</remarks>
        public static NDArray unwrap(NDArray p, long period, double? discont = null, int axis = -1)
        {
            if (p is null) throw new ArgumentNullException(nameof(p));
            return NDMatrixBoundary.Sees(p)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((p, period, discont, axis), static s => unwrap(s.p, s.period, s.discont, s.axis))
            : NDCharArrayBoundary.Sees(p)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((p, period, discont, axis), static s => unwrap(s.p, s.period, s.discont, s.axis))
            : p.TensorEngine.Unwrap(p, discont, axis, (object)period);
        }
    }
}
