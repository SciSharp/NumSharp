using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Applies <paramref name="func1d"/> to each 1-D slice of <paramref name="arr"/> taken
        ///     along <paramref name="axis"/>, then reassembles the results. The FIRST slice's result
        ///     fixes the output dtype and the block of new dimensions that REPLACE <paramref name="axis"/>:
        ///     a scalar-returning func drops the axis (one fewer dimension), a 1-D-returning func
        ///     keeps the rank, a higher-D func inserts those dimensions in the axis' place. So the
        ///     output shape is <c>arr.shape[:axis] + func_result.shape + arr.shape[axis+1:]</c>.
        /// </summary>
        /// <param name="func1d">
        ///     Applied to each 1-D slice; must accept a 1-D <see cref="NDArray"/> and return an
        ///     <see cref="NDArray"/> (a scalar result is a 0-d array). Called once per slice in
        ///     logical C-order over the non-axis dimensions. Every call is expected to return the same
        ///     shape and dtype as the first — a differing result is broadcast and unsafe-cast into the
        ///     first result's slot (NumPy's <c>buff[ind] = res</c> semantics), so a shape that cannot
        ///     broadcast raises.
        /// </param>
        /// <param name="axis">
        ///     Axis of <paramref name="arr"/> to slice along; negative counts from the end. The slice
        ///     handed to <paramref name="func1d"/> is one-dimensional with length <c>arr.shape[axis]</c>.
        /// </param>
        /// <param name="arr">Input array, of any rank (≥ 1) and any memory layout.</param>
        /// <returns>
        ///     A freshly allocated array holding the reassembled results, with <paramref name="func1d"/>'s
        ///     output dtype. The result is a transposed view over a fresh contiguous buffer (as in
        ///     NumPy), so it owns its data.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="func1d"/> or <paramref name="arr"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="arr"/>'s rank (reports the original axis).</exception>
        /// <exception cref="ValueError">
        ///     Any non-axis (iteration) dimension is zero — <c>"Cannot apply_along_axis when any
        ///     iteration dimensions are 0"</c>, verbatim with NumPy — because there is then no first
        ///     slice to determine the result shape.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
        ///     <para><b>Ownership.</b> Every array <paramref name="func1d"/> returns — and every slice handed to
        ///     it — stays the CALLER's: this method copies each result into its own buffer and never disposes
        ///     one, because a callback may return an array it keeps using (the slice itself, a held constant).
        ///     A <paramref name="func1d"/> that ALLOCATES a fresh result per slice therefore leaves those to a
        ///     GC — run the call inside an <see cref="NDScope"/> (yielding the result) when deterministic
        ///     release matters. This method's own buffer and views are released before it returns.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray apply_along_axis(Func<NDArray, NDArray> func1d, int axis, NDArray arr)
        {
            // a numpy.matrix operand: NumPy wraps the assembled result in the operand's class (conv.wrap). func1d gets a
            // plain 1-d slice here, where NumPy hands it the 1 x N matrix row its matrix indexing returns.
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((func1d, axis, arr), static s => apply_along_axis(s.func1d, s.axis, s.arr));
            if (func1d is null)
                throw new ArgumentNullException(nameof(func1d));
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));

            // NumPy's numpy.apply_along_axis: the statements are the array object's (NDArray.ShapeBase.cs)
            return NDArray.ApplyAlongAxis(func1d, axis, arr);
        }

        /// <summary>
        ///     The <c>*args</c> overload — forwards extra positional arguments to every
        ///     <paramref name="func1d"/> call, mirroring NumPy's <c>apply_along_axis(func1d, axis,
        ///     arr, *args, **kwargs)</c>. Wraps the two-argument delegate into the single-argument
        ///     form; a call with no extra arguments passes an empty array, exactly as NumPy passes an
        ///     empty <c>*args</c>.
        /// </summary>
        /// <param name="func1d">Applied as <c>func1d(slice, args)</c> to each 1-D slice.</param>
        /// <param name="axis">Axis to slice along; negative counts from the end.</param>
        /// <param name="arr">Input array.</param>
        /// <param name="args">Extra positional arguments forwarded to every <paramref name="func1d"/> call.</param>
        /// <returns>See the primary overload.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="func1d"/> or <paramref name="arr"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="arr"/>'s rank.</exception>
        /// <exception cref="ValueError">Any iteration dimension is zero.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray apply_along_axis(Func<NDArray, object[], NDArray> func1d, int axis, NDArray arr, params object[] args)
        {
            // a numpy.matrix operand: NumPy wraps the assembled result in the operand's class (conv.wrap). func1d gets a
            // plain 1-d slice here, where NumPy hands it the 1 x N matrix row its matrix indexing returns.
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((func1d, axis, arr, args), static s => apply_along_axis(s.func1d, s.axis, s.arr, s.args));
            if (func1d is null)
                throw new ArgumentNullException(nameof(func1d));

            // args is never null for a params call (it is at worst an empty array), so the captured
            // closure hands func1d the same object every slice — no per-call allocation.
            return apply_along_axis(slice => func1d(slice, args), axis, arr);
        }
    }
}
