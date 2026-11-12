using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether any array element evaluates to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if any element evaluates to True (non-zero); False for an empty array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF (NumPy's bool cast of the
        ///     text).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html — the reduction is the engine's
        ///     (<c>TensorEngine.Any</c>). The C# <see cref="bool"/> of NumPy's <c>np.any(a)</c> (its <c>np.bool</c>
        ///     scalar), kept beside the axis signature below as C# programs read it (<c>if (np.any(mask))</c>); with
        ///     any argument after <paramref name="a"/> C# binds the axis signature, whose answer is an array.</remarks>
        public static bool any(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Any(a);
        }

        /// <summary>
        ///     NumPy's <c>np.any(a, axis=None, out=None, keepdims=False, *, where=True)</c>: whether any element evaluates to True
        ///     along the given axes (every axis by default), as NumPy's
        ///     <c>np.logical_or.reduce(a, axis, bool, out, keepdims=keepdims, where=where)</c>.
        /// </summary>
        /// <param name="a">Input array (any dtype; a text element is true when non-empty).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces every axis; one axis
        ///     (<c>0</c>, <c>-1</c>) reduces that axis; a tuple (<c>(0, 2)</c> or <c>new[] { 0, 2 }</c>) reduces each of its
        ///     axes, and the empty tuple reduces none (the truth values in <paramref name="a"/>'s shape). Negative axes
        ///     count from the end. A 0-d <paramref name="a"/> takes the integer axis 0 or -1 (the ufunc reduction's
        ///     special case) and the empty tuple.</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape and ANY dtype that receives the result
        ///     with unsafe casting (an int <paramref name="out"/> stores 0 / 1), and is returned; null allocates a fresh bool
        ///     array. A C# scalar here (<c>np.any(a, 0, true)</c>) is NumPy's <c>TypeError: return arrays must be of
        ///     ArrayType</c>: write <c>keepdims: true</c>.</param>
        /// <param name="keepdims">If True, the reduced axes are left as dimensions with size one, so the result broadcasts
        ///     against <paramref name="a"/>.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask that broadcasts TO <paramref name="a"/>'s shape. Elements
        ///     where it is False are left out of the reduction (they count as False, the identity of
        ///     <c>any</c>). A mask of any other dtype raises, as in NumPy. Null for no mask.</param>
        /// <returns><paramref name="out"/> itself when given (of its own dtype, which is why the declared type is
        ///     <see cref="NDArray"/>); otherwise a fresh bool array (an <c>NDArray&lt;bool&gt;</c> instance:
        ///     <c>AsGeneric&lt;bool&gt;()</c> gives the typed view), read-only when 0-d (NumPy's <c>np.bool</c>
        ///     scalar). A <c>numpy.matrix</c> operand gives NumPy's <c>matrix.any</c> answer: the scalar without an
        ///     axis, otherwise a matrix that keeps the reduced axes with length one.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only ("output array is read-only", checked
        ///     first), has the wrong rank, or cannot hold the result (NumPy's reduction-iterator texts);
        ///     <paramref name="where"/> has more axes than <paramref name="a"/> or would make <paramref name="a"/>
        ///     stretch; or a tuple names an axis twice ("duplicate value in 'axis'", after every axis' range).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's <c>TypeError</c>
        ///     text: the mask must cast to bool under the <c>safe</c> rule).</exception>
        /// <exception cref="AxisError">An axis is out of bounds (the first one in order, as given).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not
        ///     broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar (<c>return arrays must be of
        ///     ArrayType</c>); a <c>numpy.matrix</c> operand with <c>keepdims: true</c> or a mask: NumPy calls the matrix
        ///     method, which has no such keyword (<c>matrix.any() got an unexpected keyword argument 'keepdims'</c>), or
        ///     with an <paramref name="out"/> (<c>Cannot construct a dtype from an array</c>).</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.any.html
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     <c>out=</c> overloads and the three typed <c>NDArray&lt;bool&gt;</c> forms (<c>(nd, int axis, keepdims)</c>,
        ///     <c>(nd, int[] axis, keepdims)</c>, <c>(nd, bool keepdims)</c>). The typed forms could not return NumPy's
        ///     answer for an <paramref name="out"/> of another dtype or for a matrix operand (they returned a plain
        ///     array); <c>np.any(a, null)</c> was ambiguous (CS0121). A former <c>np.any(a, 0, true)</c> (keepdims)
        ///     now binds <paramref name="out"/> and raises NumPy's TypeError: name the keyword.</para>
        /// </remarks>
        [NDScoped] // the boundary of the matrix rule and the engine call (the engine reclaims its own temporaries)
        public static NDArray any(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // np.any IS np.logical_or.reduce(a, axis, bool, out, keepdims=, where=), and NumPy's whole call (the reduce
            // prologue, the text truth route, the masked reduction, the out= store) is the engine's. A tuple takes the
            // engine's tuple member (every axis' range, then a repeat); None and an integer its
            // single-axis member (a 0-d operand accepts the integer 0 / -1). Both members declare the
            // same matrix dispatch, so a matrix operand reaches matrix.any with the axis as given.
            return axis is { IsTuple: true } tuple
                ? a.TensorEngine.Any(a, tuple.ToArray(), @out, keepdims, @where)
                : a.TensorEngine.Any(a, axis is { } single ? single[0] : (int?)null, @out, keepdims, @where);
        }
    }
}
