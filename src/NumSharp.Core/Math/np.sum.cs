using System;
using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.sum: ONE overload, NumPy's whole signature
    //     sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    // (np-layer dtype plan, P10 item 1, decision D10; plan docs/plans/api-parameter-parity.md, item B1). The engine member
    // is TensorEngine.ReduceAdd, which takes every keyword (np.add.reduce IS np.sum's whole call).
    //
    // NumSharp's earlier forms — (a), (a, int axis), (a, bool keepdims), (a, axis, keepdims), (a, dtype), (a, axis, dtype),
    // (a, axis, keepdims, dtype) — are gone (decision D19: removed, not obsoleted). Positional arguments now follow NumPy:
    // the third is dtype and the fourth out, so keepdims, initial and where are written by name, as NumPy callers write them.
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Sum of array elements over a given axis — NumPy's <c>np.sum(a, axis=None, dtype=None, out=None,
        ///     keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e. <c>np.add.reduce(a, axis,
        ///     dtype, out, keepdims=keepdims, initial=initial, where=where)</c>. This is the only overload: NumPy's whole
        ///     signature, with NumPy's names, order and defaults.
        /// </summary>
        /// <param name="a">Elements to sum (any dtype; <c>StringDType</c> concatenates, fixed-width text is refused).</param>
        /// <param name="axis">The axis to sum over (negative counts from the end); <c>null</c> sums every element. A 0-d
        ///     array accepts 0 and -1 as every axis (ufunc.reduce's rule).</param>
        /// <param name="dtype">The accumulator and result dtype (a C# <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or
        ///     a NumPy dtype string converts implicitly): every element is cast to it before it is added. <c>null</c> is
        ///     NumPy's default, NEP 50's accumulator — int64 for bool and the signed integers narrower than 64 bits, uint64
        ///     for the unsigned ones, the input's own dtype otherwise — or, with <paramref name="out"/>,
        ///     <c>promote_types(out, a)</c> (no small-integer widening: an int8 sum into an int8 out wraps in int8).</param>
        /// <param name="out">The array the sum is written into and returned. It must have the result's shape; its dtype may
        ///     differ (the loop's values are cast into it unsafely). Footgun: this is the FOURTH positional parameter, as in
        ///     NumPy, so a positional bool there (<c>np.sum(a, 0, null, true)</c>) is a C# scalar passed as out and raises
        ///     NumPy's TypeError; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions, so the result broadcasts against
        ///     <paramref name="a"/>.</param>
        /// <param name="initial">The value every sum starts from instead of 0, converted into the loop dtype the way NumPy's
        ///     element setter converts a Python value: a C# integer is a Python int (range-checked, never wrapped:
        ///     <c>initial: 1000</c> into an int8 loop raises), a <c>double</c> a Python float (truncated into an integer
        ///     loop), a 0-d <see cref="NDArray"/> is cast. An empty sum is the initial value. <c>null</c> is NumPy's
        ///     default, the identity 0 (NumPy's <c>initial=None</c>, which starts from each output's first element, has no C#
        ///     spelling).</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: only the selected elements are summed, and
        ///     an output with none selected is the start value. The C# <c>true</c> literal is NumPy's default
        ///     <c>where=True</c>; <c>false</c> selects nothing. <c>null</c> is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a sum
        ///     over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar converted to a 0-d array ("return arrays must
        ///     be of ArrayType"); <paramref name="initial"/> is complex for a real loop or of an unconvertible type; NumPy's
        ///     text refusals; a <c>numpy.matrix</c> operand with <paramref name="keepdims"/>, <paramref name="initial"/> or
        ///     <paramref name="where"/> (NumPy calls <c>matrix.sum(axis, dtype, out)</c>, which has none of them:
        ///     <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c>).</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes than
        ///     <paramref name="a"/>; an unconvertible <paramref name="initial"/> (a float NaN into an integer loop, a
        ///     sequence).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/>.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop ("Python integer 1000
        ///     out of bounds for int8").</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>:
        ///     NumPy's matrix method computes into it and then calls <c>out._collapse</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.sum.html. A float sum is NumPy's add.reduce bit for
        ///     bit: the result starts at the identity (+0.0, so a sum of only -0.0 is +0.0) or the initial value, and each
        ///     inner-loop call of NumPy's iterator adds its pairwise sum. With <paramref name="initial"/> or
        ///     <paramref name="where"/> the call runs NumPy's reduction iterator itself; without them the engine's own
        ///     schedules, which give the same bits.
        /// </remarks>
        public static NDArray sum(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (NDMatrixBoundary.Sees(a))
            {
                // NumPy hands a numpy.matrix to matrix.sum(axis, dtype, out) with every keyword the caller passed, in its
                // signature's order; the method takes none of them, so the first one passed is its TypeError.
                if (keepdims)
                    throw NDMatrixBoundary.KeywordError("sum", "keepdims");
                if (initial is not null)
                    throw NDMatrixBoundary.KeywordError("sum", "initial");
                if (@where is not null)
                    throw NDMatrixBoundary.KeywordError("sum", "where");
                if (@out is not null)
                    return NDMatrixBoundary.ReducedInto(a, @out, (axis, dtype),
                        static (m, o, s) => m.sum(s.axis, s.dtype, o),
                        static (m, o, s) => m.TensorEngine.ReduceAdd(m, s.axis, keepdims: true, s.dtype, o));
                // a row, a column or the scalar, as NumPy's matrix method returns
                return NDMatrixBoundary.Reduced((a, axis, dtype), static s => sum(s.a, s.axis, s.dtype), axis, false, "sum");
            }

            // The whole call is the engine's: TensorEngine.ReduceAdd of a's engine takes every keyword.
            return a.TensorEngine.ReduceAdd(a, axis, keepdims, dtype, @out, initial, @where);
        }
    }
}
