using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative sum of the elements along a given axis (NumPy 2.1's Array-API alternative to
        ///     <see cref="cumsum"/>): never flattens an N-D input, and can prepend the sum's identity, zero.
        /// </summary>
        /// <param name="x">Input array (NumPy's positional-only <c>x</c>); a 0-d input is read as 1-D (NumPy's
        ///     <c>atleast_1d</c>).</param>
        /// <param name="axis">Axis along which the cumulative sum is computed (negative counts from the end). <c>null</c>
        ///     (NumPy's <c>None</c>, the default) is axis 0 for a 1-D input and an error for an input of two or more
        ///     dimensions: unlike <see cref="cumsum"/>, this function never flattens.</param>
        /// <param name="dtype">Type of the returned array and of the accumulator; <c>null</c> follows
        ///     <see cref="cumsum"/>'s rule (an integer narrower than int64 sums as int64, uint8 as uint64, bool as
        ///     int64).</param>
        /// <param name="out">Alternate output array; with <paramref name="include_initial"/> it must be one longer along
        ///     <paramref name="axis"/> than the input. It is returned.</param>
        /// <param name="include_initial">Prepend the identity (zero) along <paramref name="axis"/>, so the result is one
        ///     longer there: <c>[0, x0, x0 + x1, ...]</c>.</param>
        /// <returns>The cumulative sums, a new array unless <paramref name="out"/> was given (then <paramref name="out"/>
        ///     itself).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> is <c>null</c> and the input has two or more dimensions
        ///     ("For arrays which have more than one dimension ``axis`` argument is required."), or
        ///     <paramref name="out"/>'s shape does not fit (NumPy's iterator text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for the (at least 1-D) input.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of bounds while <paramref name="out"/> and
        ///     <paramref name="include_initial"/> are both given: NumPy indexes its slice list with the axis before it
        ///     accumulates ("list assignment index out of range").</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.cumulative_sum.html
        ///     <para>NumPy's signature <c>cumulative_sum(x, /, *, axis=None, dtype=None, out=None, include_initial=False)</c>;
        ///     C# has no keyword-only parameters, so the keywords may also be passed positionally, in NumPy's order. The body
        ///     is NumPy's <c>_cumulative_func</c> statement for statement (<see cref="CumulativeFunc"/>).</para>
        /// </remarks>
        public static NDArray cumulative_sum(NDArray x, int? axis = null, DType dtype = null, NDArray @out = null,
            bool include_initial = false)
            => CumulativeFunc(x, add: true, axis, dtype, @out, include_initial);

        /// <summary>
        ///     NumPy's <c>_cumulative_func(x, func, axis, dtype, out, include_initial)</c> (numpy/_core/fromnumeric.py), the
        ///     shared body of <see cref="cumulative_sum"/> and <see cref="cumulative_prod"/>: <c>func.accumulate</c> is
        ///     <see cref="cumsum"/> (<c>np.add.accumulate</c>) or <see cref="cumprod"/> (<c>np.multiply.accumulate</c>),
        ///     always called with an integer axis, so it never takes their flattening branch.
        /// </summary>
        /// <param name="x">The input; read as at least 1-D.</param>
        /// <param name="add">True for the sum (identity 0), false for the product (identity 1).</param>
        /// <param name="axis">The axis, or <c>null</c> (axis 0 of a 1-D input, an error otherwise).</param>
        /// <param name="dtype">The accumulator / result dtype, or <c>null</c> for the accumulate's own rule.</param>
        /// <param name="out">The output array, or <c>null</c>.</param>
        /// <param name="include_initial">Prepend the identity along the axis.</param>
        /// <returns>The result, or <paramref name="out"/> itself.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A missing axis for an N-D input, or an ill-shaped <paramref name="out"/>.</exception>
        /// <exception cref="AxisError">An out-of-bounds axis reaching the accumulate.</exception>
        /// <exception cref="IndexError">An out-of-bounds axis on the <paramref name="out"/> + include branch.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the at-least-1-D view, the slice of <paramref name="out"/>, the
        ///     accumulate's result before the identity is prepended and the identity block are released on return; the
        ///     returned array and the caller's arrays (built before the scope opened) are not.
        ///     <para>The public functions' computation for an array subclass operand: its routes reach the operand's class by
        ///     different mechanisms (the accumulate's ufunc wrap, concat's <c>PyArray_GetSubType</c>, out itself), so the
        ///     answer is <see cref="NDSubclassRules.Cumulative"/>'s, applied to the result computed on the plain
        ///     views.</para></remarks>
        [NDScoped]
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.Cumulative))]
        internal static NDArray CumulativeFunc(NDArray x, bool add, int? axis, DType dtype, NDArray @out, bool include_initial)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));

            // x = np.atleast_1d(x): a 0-d input accumulates as a one-element 1-D array (cumulative_sum(5) is [5]).
            var x1 = atleast_1d(x);
            int ndim = x1.ndim;
            int identity = add ? 0 : 1;

            int ax;
            if (axis is null)
            {
                // No flattening, unlike np.cumsum: the axis is required once there is more than one.
                if (ndim >= 2)
                    throw new ValueError("For arrays which have more than one dimension ``axis`` argument is required.");
                ax = 0;
            }
            else
            {
                ax = axis.Value;
            }

            if (@out is not null && include_initial)
            {
                // item = [slice(None)] * x_ndim; item[axis] = slice(1, None). Python's list assignment accepts -ndim..ndim-1
                // and raises IndexError beyond, BEFORE the accumulate would have raised its AxisError.
                if (ax < -ndim || ax >= ndim)
                    throw new IndexError("list assignment index out of range");
                int slot = ax < 0 ? ax + ndim : ax;
                var item = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    item[i] = Slice.All;
                item[slot] = new Slice(1, null);

                // func.accumulate(x, axis=axis, dtype=dtype, out=out[tuple(item)]): the sums land after the first slot.
                var tail = @out[item];
                if (add)
                    cumsum(x1, ax, dtype, tail);
                else
                    cumprod(x1, ax, dtype, tail);

                // item[axis] = 0; out[tuple(item)] = func.identity: the identity as the Python int it is (weak), cast into
                // out's dtype by the assignment.
                item[slot] = Slice.Index(0);
                @out[item] = identity;
                return @out;
            }

            // res = func.accumulate(x, axis=axis, dtype=dtype, out=out)
            var res = add ? cumsum(x1, ax, dtype, @out) : cumprod(x1, ax, dtype, @out);
            if (include_initial)
            {
                // initial_shape = list(x.shape); initial_shape[axis] = 1 (the accumulate already validated the axis).
                var initialDims = (long[])x1.Shape.dimensions.Clone();
                initialDims[ax < 0 ? ax + ndim : ax] = 1;

                // np.full_like(res, func.identity, shape=initial_shape) is NumPy's empty_like(res, shape=...) followed by
                // copyto(..., casting='unsafe'): res's dtype and class, laid out like res.
                var initial = empty_like(res, shape: new Shape(initialDims, copy: false));
                copyto(initial, (NDArray)identity, casting: "unsafe");

                // res = np.concat([initial, res], axis=axis)
                res = concat(new[] { initial, res }, axis: ax);
            }

            return res;
        }
    }
}
