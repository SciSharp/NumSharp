using System;
using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.mean: ONE overload, NumPy's whole signature
    //     mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
    // (np-layer dtype plan, P10 item 1, decision D10; plan docs/plans/api-parameter-parity.md, item B3). The engine member is
    // TensorEngine.ReduceMean (NumPy's _methods._mean), which takes every keyword.
    //
    // NumSharp's earlier forms — (a), (a, int axis), (a, bool keepdims), (a, axis, dtype, keepdims), (a, axis, keepdims) —
    // are gone (decision D19). dtype= is reachable without an axis now (np.mean(a, dtype: np.float32) did not compile).
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Compute the arithmetic mean along the specified axis — NumPy's <c>np.mean(a, axis=None, dtype=None, out=None,
        ///     keepdims=&lt;no value&gt;, *, where=&lt;no value&gt;)</c> (<c>_methods._mean</c>): the sum divided by the count of
        ///     the summed elements. This is the only overload: NumPy's whole signature, with NumPy's names, order and
        ///     defaults.
        /// </summary>
        /// <param name="a">Array containing numbers whose mean is desired.</param>
        /// <param name="axis">The axis to average over (negative counts from the end); <c>null</c> averages every element.
        ///     Unlike np.sum, a 0-d array takes no axis (np.mean normalizes it first), unless <paramref name="where"/> is a
        ///     mask. A tuple (<c>(0, 2)</c>) reduces every axis it names together, in one pass (NumPy's tuple axis; the
        ///     empty tuple reduces none, and a tuple item gets no 0-d allowance).</param>
        /// <param name="dtype">The dtype the sum is computed in and the mean returned in. <c>null</c> is NumPy's default:
        ///     float64 for bool and integer input, float32 for float16 (with a float16 result), the input's own dtype
        ///     otherwise.</param>
        /// <param name="out">The array the mean is written into and returned (the result's shape). NumPy stores the SUM into
        ///     it and divides it in place, both with an unsafe cast, so an integer out holds truncated values. Footgun: the
        ///     FOURTH positional parameter, as in NumPy; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: each output averages its selected elements
        ///     only (divided by their count, so an output with none selected is NaN in a float loop). The C# <c>true</c>
        ///     literal is NumPy's default. <c>null</c> is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a mean
        ///     over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, or a text out's missing
        ///     division. Also a tuple that names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not
        ///     broadcast.</exception>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; NumPy's text refusals; a
        ///     <c>numpy.matrix</c> operand with <paramref name="keepdims"/> or <paramref name="where"/>
        ///     (<c>matrix.mean() got an unexpected keyword argument 'keepdims'</c>).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.mean.html</remarks>
        public static NDArray mean(NDArray a, NDAxis? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // The whole call is the engine's: TensorEngine.ReduceMean of a's engine takes every keyword.
            return a.TensorEngine.ReduceMean(a, axis, keepdims, dtype, @out, @where);
        }
    }
}
