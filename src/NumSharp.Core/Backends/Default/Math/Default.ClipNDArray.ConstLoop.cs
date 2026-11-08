using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // Which of clip's two float loops NumPy runs
        // =============================================================================
        //
        // NumPy's clip ufunc loop (numpy/_core/src/umath/clip.cpp, _npy_clip) tests the bound
        // STEPS of each inner-loop call: when both are 0 it runs _npy_clip_const_minmax_, and for
        // float32/float64 that fast path is a compare-select (x < lo ? lo : x, then x > hi ? hi : x;
        // a NaN bound fills the call with that NaN). Otherwise it runs _NPY_MIN(_NPY_MAX(x, lo), hi).
        // The two disagree on ties: np.clip(-0.0, 0.0, 1.0) keeps -0.0 on the const path and returns
        // +0.0 on the general one. They also disagree on the NaN a NaN bound produces.
        //
        // The steps are decided far from the loop: by the ufunc's trivial-loop shortcut, or by
        // NpyIter's axis order, coalescing and buffering cost model. A bound that varies along
        // the result can still reach the loop with stride 0 (a (2,1) bound against a (2,3)
        // array, when NpyIter does not buffer it), and a 0-d bound copied into a buffer still has
        // stride 0. The port of that decision (check_for_trivial_loop, try_trivial_single_output_loop
        // and NpyIter's axis order, flips, COPY_IF_OVERLAP, coalescing and npyiter_find_buffering_setup)
        // lives in DefaultEngine.UfuncInnerSteps.cs, shared with np.power, whose float loops branch on
        // the exponent's step the same way; this file asks it for the two bound steps.
        // Validated against live NumPy 2.4.2: 70,000 random layout / dtype / out= / where= /
        // overlap cases, every one deciding the same loop (the -0.0-vs-+0.0 tie reveals it).
        // =============================================================================

        /// <summary>
        /// Whether NumPy 2.4.2 runs clip's CONST float loop (both bound steps 0) for this call, rather than the
        /// general <c>MIN(MAX(x, lo), hi)</c> loop. Only meaningful for a two-bound clip in a float32/float64 loop;
        /// the other dtypes compute the same result either way.
        /// </summary>
        /// <param name="a">The array being clipped, as the caller passed it (before any NumSharp cast).</param>
        /// <param name="lo">The lower bound as passed. A 0-d bound counts as a Python scalar (NumSharp cannot tell a
        /// C# literal from a 0-d array): NumPy converts it into a fresh 0-d array of the loop dtype.</param>
        /// <param name="hi">The upper bound as passed, with the same 0-d rule.</param>
        /// <param name="out">The out= array, or null when NumPy allocates the result.</param>
        /// <param name="where">The where= mask, or null. A mask disables the trivial loop.</param>
        /// <param name="loop">The resolved loop dtype.</param>
        /// <param name="dims">The broadcast iteration shape (inputs, out and mask together).</param>
        /// <returns>True when the const loop runs, so ties keep the element and a NaN bound fills with itself.</returns>
        /// <remarks>
        /// <para>The result is uniform over the call: the trivial loop is one inner-loop call, and NpyIter gives every
        /// inner-loop call of an elementwise iteration the same strides (no reduce operand exists here).</para>
        /// <para>Strides are compared in BYTES, as NumPy does; the copies a step creates are modelled as fresh,
        /// contiguous arrays of the loop dtype that cannot overlap anything
        /// (<see cref="NumPyUfuncInputSteps"/>).</para>
        /// </remarks>
        internal static bool NumPyClipRunsConstLoop(NDArray a, NDArray lo, NDArray hi, NDArray @out, NDArray where,
                                                   NPTypeCode loop, long[] dims)
        {
            // Two scalar bounds: every route gives them stride 0 (the trivial loop's 0-d rule, NpyIter's broadcast
            // and its single-stride buffering). The common case decides here.
            if (lo.ndim == 0 && hi.ndim == 0)
                return true;

            // The array is read as given (a 0-d array stays an array); a 0-d bound is a Python scalar.
            var ops = new UfuncLoopOperand[3]
            {
                UfuncLoopOperand.Of(a),
                UfuncLoopOperand.OfInput(lo, loop),
                UfuncLoopOperand.OfInput(hi, loop),
            };

            // The trivial-loop shortcut or NpyIter, exactly as NumPy decides (DefaultEngine.UfuncInnerSteps.cs).
            Span<long> steps = stackalloc long[3];
            NumPyUfuncInputSteps(ops, @out, where, loop, dims, steps);
            return steps[1] == 0 && steps[2] == 0;
        }
    }
}
