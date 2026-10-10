using System;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether any array element evaluates to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if any element evaluates to True (non-zero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static bool any(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Any(a);
        }

        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="axis">Axis along which a logical OR reduction is performed.</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        [NDScoped] // boundary for the engine reduction tree (reshape/chained-axis temps); the reduced result is yielded
        public static NDArray<bool> any(NDArray nd, int axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
            {
                var reduced = defaultEngine.Any(nd, axis, keepdims);
                reduced.MarkReductionScalar(); // 1-D input reduced to 0-d -> numpy scalar (read-only)
                return reduced;
            }

            var result = nd.TensorEngine.Any(nd, axis);
            if (keepdims && nd.ndim > 0)
            {
                axis = DefaultEngine.NormalizeAxis(axis, nd.ndim);
                var dims = (long[])nd.Shape.dimensions.Clone();
                dims[axis] = 1;
                result.Storage.Reshape(Shape.Adopt(dims));
            }

            return result;
        }

        /// <summary>
        ///     Test whether any array element along the given axes evaluates to True.
        ///     Multiple axes can be specified by passing an array of ints.
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="axis">Tuple of axes along which a logical OR reduction is performed.
        ///     An empty array returns the input cast to bool (no reduction).</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        // See the tuple-axis out=/where= overload: a bare null axis goes to the int? overload instead.
        [OverloadResolutionPriority(-1)]
        [NDScoped] // boundary for the engine multi-axis tree (axis-run reshape view, chained single-axis intermediates)
        public static NDArray<bool> any(NDArray nd, int[] axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
            {
                var reduced = defaultEngine.Any(nd, axis, keepdims);
                reduced.MarkReductionScalar(); // every axis reduced away -> numpy scalar (read-only)
                return reduced;
            }

            if (axis.Length == 0)
                return DefaultEngine.CastToBoolPreservingShape(nd);

            var normalized = axis.Select(a => DefaultEngine.NormalizeAxis(a, nd.ndim)).ToArray();
            Array.Sort(normalized);

            NDArray<bool> result = null;
            NDArray current = nd;
            for (int i = normalized.Length - 1; i >= 0; i--)
            {
                result = any(current, normalized[i], keepdims: true);
                current = result;
            }

            if (!keepdims)
                result = SqueezeAxes(result, normalized);

            return result;
        }

        /// <summary>
        ///     Test whether any array element evaluates to True, optionally keeping reduced dimensions.
        ///     Reduces over all axes (axis = None semantics).
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="keepdims">If True, the result has all dimensions as size 1 (broadcast-compatible
        ///     with the input). Otherwise the result is a 0-d array.</param>
        /// <returns>A new boolean ndarray.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray<bool> any(NDArray nd, bool keepdims)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            bool scalar = any(nd);
            if (!keepdims || nd.ndim == 0)
                return NDArray.Scalar(scalar).MakeGeneric<bool>();

            var dims = new long[nd.ndim];
            for (int i = 0; i < dims.Length; i++) dims[i] = 1;
            var result = NDArray.Scalar(scalar).MakeGeneric<bool>();
            result.Storage.Reshape(Shape.Adopt(dims));
            return result;
        }

        /// <summary>
        ///     Test whether any array element along the given axis evaluates to True, with optional
        ///     out= destination and where= mask. Matches NumPy 2.x:
        ///     <c>any(a, axis=None, out=None, keepdims=False, *, where=True)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which to reduce. Pass <c>null</c> for axis=None (all axes).</param>
        /// <param name="out">Destination array. Its dtype is preserved. Pass <c>null</c> to allocate fresh.</param>
        /// <param name="keepdims">If True, the reduced axes are left as size-one dimensions.</param>
        /// <param name="where">A bool mask that broadcasts TO <paramref name="a"/>'s shape. Elements where it is
        ///     False are left out of the reduction (they count as False, the identity of <c>any</c>).
        ///     A mask of any other dtype raises, as in NumPy. Pass <c>null</c> for no mask.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only ("output array is read-only", checked
        ///     first), has the wrong rank, or cannot hold the result (NumPy's reduction-iterator texts), or
        ///     <paramref name="where"/> has more axes than <paramref name="a"/> or would make <paramref name="a"/>
        ///     stretch.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's <c>TypeError</c>
        ///     text: the mask must cast to bool under the <c>safe</c> rule).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not
        ///     broadcast against <paramref name="a"/>.</exception>
        /// <remarks>
        ///     A bare <c>null</c> axis binds this overload (NumPy's <c>axis=None</c>): the <c>int[]</c>-axis overloads
        ///     carry a lower <c>OverloadResolutionPriority</c> for exactly that literal.
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.any.html</para>
        /// </remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        [NDScoped] // reclaims the where-mask effective array and the reduced temp on the out= path
        public static NDArray any(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims, @where), static s => any(s.a, s.axis, s.@out, s.keepdims, s.@where), axis, keepdims, "any");
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy's reduce prologue, in NumPy's order (read-only out, mask dtype, axis, out rank, iterator
            // shapes), before any work. Without out= and where= it can raise nothing the reduction below does
            // not raise itself, so the plain call skips it.
            if (@out is not null || @where is not null)
                ReductionArguments.Validate(a, axis, keepdims, @out, @where, "logical_or");

            NDArray<bool> reduced = ReduceAnyWithWhere(a, axis, keepdims, @where);
            // A fresh 0-d result is a numpy SCALAR (np.bool_) at the boundary: read-only
            // (PyArray_Return semantics); out= returns the writeable out array itself.
            return @out is null ? reduced.MarkReductionScalar() : WriteToOut(reduced, @out);
        }

        /// <summary>
        ///     Tuple-axis variant of the full <c>np.any</c> overload (with <paramref name="out"/> and <paramref name="where"/>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axes along which to reduce.</param>
        /// <param name="out">Destination array. Its dtype is preserved.</param>
        /// <param name="keepdims">If True, reduced axes are left as size-one dimensions.</param>
        /// <param name="where">A bool mask that broadcasts TO <paramref name="a"/>'s shape; any other dtype
        ///     raises.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only ("output array is read-only", checked
        ///     first), has the wrong rank, or cannot hold the result (NumPy's reduction-iterator texts), or
        ///     <paramref name="where"/> has more axes than <paramref name="a"/> or would make <paramref name="a"/>
        ///     stretch, or an axis repeats ("duplicate value in 'axis'").</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's <c>TypeError</c>
        ///     text: the mask must cast to bool under the <c>safe</c> rule).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="axis"/> is <c>null</c>.</exception>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not
        ///     broadcast against <paramref name="a"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        // A bare null axis fits int? and int[] equally well (CS0121 for np.any(a, null, null, false, w)); the
        // lower priority hands that literal to the int? overload (NumPy's axis=None). An int[] argument fits only
        // the int[] overloads, so they still bind it.
        [OverloadResolutionPriority(-1)]
        [NDScoped] // reclaims the where-mask effective array and the reduced temp on the out= path
        public static NDArray any(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            // NumPy's reduce prologue for a tuple axis (see the int-axis overload).
            if (@out is not null || @where is not null)
                ReductionArguments.Validate(a, axis, keepdims, @out, @where, "logical_or");

            NDArray effective = @where is null ? a : ApplyWhereForAny(a, @where);
            NDArray<bool> reduced = any(effective, axis, keepdims);
            return @out is null ? reduced : WriteToOut(reduced, @out);
        }

        private static NDArray<bool> ReduceAnyWithWhere(NDArray a, int? axis, bool keepdims, NDArray @where)
        {
            NDArray effective = @where is null ? a : ApplyWhereForAny(a, @where);

            if (!axis.HasValue)
                return any(effective, keepdims);

            return any(effective, axis.Value, keepdims);
        }
    }
}
