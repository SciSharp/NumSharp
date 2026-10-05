using System;
using System.Text;

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumPy's reduce prologue: the argument checks every <c>ufunc.reduce</c>-based reduction runs before it reads a
    ///     single element (<c>np.all</c>, <c>np.any</c>, <c>np.max</c>, <c>np.sum</c>, … and their <c>ndarray</c> methods),
    ///     in NumPy's order and with NumPy's texts.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A port of three NumPy 2.4.2 functions, which run in this order:
    ///     <list type="number">
    ///     <item><c>PyUFunc_GenericReduction</c> (<c>umath/ufunc_object.c</c>) parses the arguments in signature
    ///     order. Its converters raise first: a read-only <c>out=</c> ("output array is read-only"), then a
    ///     <c>where=</c> mask that does not cast to bool under the <c>safe</c> rule, then the axis
    ///     (<c>AxisError</c>; a 0-d input accepts axis 0 and -1 as "no axis"), then a repeated axis.</item>
    ///     <item><c>PyUFunc_ReduceWrapper</c> (<c>umath/reduction.c</c>) maps the result's axes onto the input's
    ///     (<c>result_axes</c>) and checks the rank of <c>out=</c> ("…has the wrong number of dimensions…").</item>
    ///     <item><c>npyiter_fill_axisdata</c> (<c>multiarray/nditer_constr.c</c>) builds the iterator over
    ///     <c>[out, input, where]</c>. Its first pass broadcasts the shapes (the "remapped shapes" text, and
    ///     "input operand has more dimensions…" for a mask with more axes than the input); its second pass walks
    ///     the axes innermost first and rejects a stretched input (the input is <c>NO_BROADCAST</c>), a kept reduced
    ///     axis of <c>out=</c> whose length is not 1, and an <c>out=</c> axis of length 1 that would have to
    ///     broadcast (the reduction iterator has no <c>REDUCE_OK</c>, so that is "requires a reduction … not
    ///     enabled").</item>
    ///     </list>
    ///     </para>
    ///     <para>
    ///     Exception types follow the house mapping: NumPy's <c>ValueError</c> is <see cref="ValueError"/>, except a
    ///     broadcast failure, which is <see cref="IncorrectShapeException"/> like every other broadcast error in
    ///     NumSharp; NumPy's <c>TypeError</c> for the mask stays the <see cref="ArgumentException"/> the ufunc
    ///     <c>where=</c> path raises (same text). The texts are NumPy's, character for character.
    ///     </para>
    ///     <para>
    ///     The checks read only shapes, so they cost O(ndim) and run before any allocation. A reduction that passes
    ///     them may rely on: the mask broadcasts TO the input's shape (never the reverse), and <c>out=</c> has exactly
    ///     the result's shape (with <c>keepdims</c>, a length-1 axis in each reduced position).
    ///     </para>
    /// </remarks>
    internal static class ReductionArguments
    {
        /// <summary>
        ///     Runs NumPy's whole reduce prologue for a single (or no) reduction axis and returns the reduced-axis flags.
        /// </summary>
        /// <param name="input">The array being reduced.</param>
        /// <param name="axis">The reduction axis, or <c>null</c> for NumPy's <c>axis=None</c> (every axis).</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="out">The caller's output array, or <c>null</c> when the result is allocated.</param>
        /// <param name="where">The caller's mask, or <c>null</c> for NumPy's <c>where=True</c>.</param>
        /// <param name="funcname">The ufunc name NumPy prints (<c>logical_and</c> for <c>np.all</c>,
        ///     <c>maximum</c> for <c>np.max</c>, …).</param>
        /// <returns>One flag per input axis: <c>true</c> where the axis is reduced. A 0-d input returns an empty array.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only, has the wrong rank, or has a length that
        ///     cannot hold the result.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="input"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against the input.</exception>
        internal static bool[] Validate(NDArray input, int? axis, bool keepdims, NDArray @out, NDArray @where, string funcname)
        {
            ValidateParsedArguments(@out, @where);
            bool[] flags = AxisFlags(input.ndim, axis);
            ValidateOperands(input, flags, keepdims, @out, @where, funcname);
            return flags;
        }

        /// <summary>
        ///     Runs NumPy's whole reduce prologue for a tuple of reduction axes and returns the reduced-axis flags.
        /// </summary>
        /// <param name="input">The array being reduced.</param>
        /// <param name="axes">The reduction axes (NumPy's tuple <c>axis</c>); an empty array reduces nothing.</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="out">The caller's output array, or <c>null</c> when the result is allocated.</param>
        /// <param name="where">The caller's mask, or <c>null</c> for NumPy's <c>where=True</c>.</param>
        /// <param name="funcname">The ufunc name NumPy prints in its messages.</param>
        /// <returns>One flag per input axis: <c>true</c> where the axis is reduced.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="axes"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or misshapen, or an axis repeats
        ///     ("duplicate value in 'axis'").</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError">An axis is out of bounds for <paramref name="input"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against the input.</exception>
        internal static bool[] Validate(NDArray input, int[] axes, bool keepdims, NDArray @out, NDArray @where, string funcname)
        {
            if (axes is null)
                throw new ArgumentNullException(nameof(axes));
            ValidateParsedArguments(@out, @where);
            bool[] flags = AxisFlags(input.ndim, axes);
            ValidateOperands(input, flags, keepdims, @out, @where, funcname);
            return flags;
        }

        /// <summary>
        ///     The two argument converters of <c>PyUFunc_GenericReduction</c> that can raise, in their parsing order:
        ///     <c>out=</c> (<c>_set_out_array</c>) before <c>where=</c> (<c>_wheremask_converter</c>).
        /// </summary>
        /// <param name="out">The caller's output array, or <c>null</c>.</param>
        /// <param name="where">The caller's mask, or <c>null</c>.</param>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (NumPy checks it before the mask, the axis
        ///     and every shape, probed 2.4.2).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        internal static void ValidateParsedArguments(NDArray @out, NDArray @where)
        {
            if (@out is not null && !@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            DefaultEngine.ValidateWhereMask(@where);
        }

        /// <summary>
        ///     NumPy's conversion of an integer (or absent) <c>axis</c> into reduction flags.
        /// </summary>
        /// <param name="ndim">The input's rank.</param>
        /// <param name="axis">The axis, or <c>null</c> for every axis.</param>
        /// <returns>The reduced-axis flags.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <remarks>
        ///     A 0-d input accepts axis 0 and -1 as "no reduction axis", a special case NumPy keeps for backward
        ///     compatibility of <c>np.sum(scalar, axis=0)</c> (<c>PyUFunc_GenericReduction</c>).
        /// </remarks>
        internal static bool[] AxisFlags(int ndim, int? axis)
        {
            var flags = new bool[ndim];
            if (axis is null)
            {
                for (int i = 0; i < ndim; i++)
                    flags[i] = true;
                return flags;
            }

            if (ndim == 0 && (axis.Value == 0 || axis.Value == -1))
                return flags;
            flags[DefaultEngine.NormalizeAxis(axis.Value, ndim)] = true;
            return flags;
        }

        /// <summary>
        ///     NumPy's conversion of a tuple <c>axis</c> into reduction flags: every axis normalized first, then the
        ///     repeat check of <c>PyUFunc_Reduce</c>.
        /// </summary>
        /// <param name="ndim">The input's rank.</param>
        /// <param name="axes">The axes.</param>
        /// <returns>The reduced-axis flags.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">An axis appears twice ("duplicate value in 'axis'").</exception>
        internal static bool[] AxisFlags(int ndim, int[] axes)
        {
            // NumPy normalizes the whole tuple (check_and_adjust_axis per item) before it looks for repeats, so an
            // out-of-bounds axis wins over a repeated one.
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                normalized[i] = DefaultEngine.NormalizeAxis(axes[i], ndim);

            var flags = new bool[ndim];
            foreach (int ax in normalized)
            {
                if (flags[ax])
                    throw new ValueError("duplicate value in 'axis'");
                flags[ax] = true;
            }

            return flags;
        }

        /// <summary>
        ///     The shape checks of <c>PyUFunc_ReduceWrapper</c> and the reduction iterator it builds over
        ///     <c>[out, input, where]</c>.
        /// </summary>
        /// <param name="input">The array being reduced.</param>
        /// <param name="axisFlags">The reduced-axis flags (one per input axis).</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="out">The caller's output array, or <c>null</c>.</param>
        /// <param name="where">The caller's mask, or <c>null</c>.</param>
        /// <param name="funcname">The ufunc name NumPy prints in the rank message.</param>
        /// <exception cref="ValueError"><paramref name="out"/> has the wrong rank, a kept reduced axis of
        ///     <paramref name="out"/> is not 1 long, an axis of <paramref name="out"/> is 1 long where the result is longer,
        ///     the input would have to stretch, or the mask has more axes than the input.</exception>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast (NumPy's "remapped shapes" text).</exception>
        internal static void ValidateOperands(NDArray input, bool[] axisFlags, bool keepdims, NDArray @out, NDArray @where, string funcname)
        {
            int ndim = input.ndim;
            long[] inShape = input.shape;

            // result_axes: the out axis each input axis maps to (-1 for a reduced axis that keepdims drops), and
            // whether the input axis is a reduction axis (NPY_ITER_REDUCTION_AXIS).
            var resultAxes = new int[ndim];
            int curr = 0;
            for (int i = 0; i < ndim; i++)
                resultAxes[i] = axisFlags[i] && !keepdims ? -1 : curr++;

            if (@out is not null && curr != @out.ndim)
            {
                // "NpyIter does not raise a good error message in this common case" (reduction.c): the wrapper's own text.
                throw new ValueError(
                    $"output parameter for reduction operation {funcname} has the wrong number of dimensions: " +
                    $"Found {@out.ndim} but expected {curr}" +
                    (keepdims ? " (must match the operand's when keepdims=True)" : ""));
            }

            long[] outShape = @out?.shape;
            long[] whereShape = @where?.shape;

            // Pass 1 (npyiter_fill_axisdata): broadcast every operand's shape onto the iterator's, in operand order.
            var bshape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                bshape[i] = 1;

            if (outShape is not null)
            {
                for (int idim = 0; idim < ndim; idim++)
                {
                    int i = resultAxes[idim];
                    if (i >= 0 && !Join(bshape, idim, outShape[i]))
                        throw BroadcastError(input, @out, @where, resultAxes);
                }
            }

            for (int idim = 0; idim < ndim; idim++)
            {
                if (!Join(bshape, idim, inShape[idim]))
                    throw BroadcastError(input, @out, @where, resultAxes);
            }

            if (whereShape is not null)
            {
                int wdim = whereShape.Length;
                if (wdim > ndim)
                    throw new ValueError("input operand has more dimensions than allowed by the axis remapping");
                for (int k = 0; k < wdim; k++)
                {
                    if (!Join(bshape, k + ndim - wdim, whereShape[k]))
                        throw BroadcastError(input, @out, @where, resultAxes);
                }
            }

            // Pass 2: the axes innermost first (NpyIter stores them reversed), each axis checking the operands in
            // order. The first offending (axis, operand) pair raises.
            for (int idim = 0; idim < ndim; idim++)
            {
                int axis = ndim - idim - 1;
                long bs = bshape[axis];

                if (outShape is not null)
                {
                    int i = resultAxes[axis];
                    if (axisFlags[axis])
                    {
                        // An explicit reduction axis: kept (keepdims) it must be exactly 1 long.
                        if (i >= 0 && outShape[i] != 1)
                            throw new ValueError(
                                $"operand was set up as a reduction along axis {i}, but the length of the axis is {outShape[i]} (it has to be 1)");
                    }
                    else if (bs != 1 && outShape[i] == 1)
                    {
                        // out would broadcast (a write-side reduction) along a kept axis; the reduction iterator does not
                        // enable REDUCE_OK, so NumPy refuses it (npyiter_check_reduce_ok_and_set_flags, dim = out's axis).
                        throw new ValueError(
                            $"output operand requires a reduction along dimension {i}, but the reduction is not enabled. " +
                            "The dimension size of 1 does not match the expected output shape.");
                    }
                }

                // The input is NO_BROADCAST: it may not stretch along any axis.
                if (bs != 1 && inShape[axis] == 1)
                {
                    throw new ValueError(
                        $"non-broadcastable operand with shape {Shape.ConvertShapeToString(inShape)} doesn't match the broadcast shape {Shape.ConvertShapeToString(bshape)}");
                }

                // The mask is read-only and may broadcast freely; nothing to check.
            }
        }

        /// <summary>
        ///     One broadcast step of <c>npyiter_fill_axisdata</c>'s first pass: a length-1 iterator axis takes the
        ///     operand's length; otherwise the operand must match it or be 1 long.
        /// </summary>
        /// <param name="bshape">The iterator shape being built (updated in place).</param>
        /// <param name="idim">The iterator axis.</param>
        /// <param name="length">The operand's length on that axis.</param>
        /// <returns><c>false</c> when the lengths conflict.</returns>
        private static bool Join(long[] bshape, int idim, long length)
        {
            if (bshape[idim] == 1)
            {
                bshape[idim] = length;
                return true;
            }

            return bshape[idim] == length || length == 1;
        }

        /// <summary>
        ///     NumPy's broadcast failure for an iterator with axis remapping: every present operand's shape in operand
        ///     order, <c>out=</c> followed by <c>-&gt;</c> and its remapped shape (a dropped reduced axis reads
        ///     <c>newaxis</c>), each shape followed by one space.
        /// </summary>
        /// <param name="input">The array being reduced.</param>
        /// <param name="out">The caller's output array, or <c>null</c> (an allocated result is not printed).</param>
        /// <param name="where">The caller's mask, or <c>null</c>.</param>
        /// <param name="resultAxes">The out axis per input axis (-1 for a dropped reduced axis).</param>
        /// <returns>The exception to throw.</returns>
        private static IncorrectShapeException BroadcastError(NDArray input, NDArray @out, NDArray @where, int[] resultAxes)
        {
            var sb = new StringBuilder("operands could not be broadcast together with remapped shapes [original->remapped]: ");
            if (@out is not null)
            {
                long[] outShape = @out.shape;
                var remapped = new long[resultAxes.Length];
                for (int idim = 0; idim < resultAxes.Length; idim++)
                {
                    int i = resultAxes[idim];
                    remapped[idim] = i >= 0 && i < outShape.Length ? outShape[i] : -1;
                }

                sb.Append(Shape.ConvertShapeToString(outShape)).Append("->").Append(Shape.ConvertShapeToString(remapped)).Append(' ');
            }

            sb.Append(Shape.ConvertShapeToString(input.shape)).Append(' ');
            if (@where is not null)
                sb.Append(Shape.ConvertShapeToString(@where.shape)).Append(' ');
            return new IncorrectShapeException(sb.ToString());
        }
    }
}
