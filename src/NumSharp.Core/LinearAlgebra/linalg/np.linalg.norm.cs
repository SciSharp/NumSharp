using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Matrix or vector norm.
            /// </summary>
            /// <param name="x">Input array. Integer and bool operands are computed in float64.</param>
            /// <param name="ord">
            ///     The order: <c>null</c> (Frobenius for a matrix, 2-norm for a vector), a number
            ///     (including <c>double.PositiveInfinity</c>), or one of <c>"fro"</c>, <c>"f"</c>,
            ///     <c>"nuc"</c>.
            /// </param>
            /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (the whole array: a vector norm of a 1-D input, a matrix
            ///     norm of a 2-D one), one axis (NumPy's 1-tuple: a vector norm along it), or a tuple of one or two axes (a
            ///     vector or a matrix norm). Any other length is NumPy's ValueError.</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <returns>The norm: a 0-d array for a whole-array norm, else the reduced array. An EMPTY operand's
            ///     inf-, 1- and 2-norms are 0.0 (NumPy 2.x reduces those with <c>initial=0</c>).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="ValueError">An order that is invalid for the number of reduced axes, duplicate axes
            ///     (<c>Duplicate axes given.</c>), no axis or more than two (<c>Improper number of dimensions to
            ///     norm.</c>), or a min order (-1, -inf, -2) of an empty operand.</exception>
            /// <exception cref="ValueError">A text element that does not parse as a float (NumPy's <c>x.astype(float)</c>:
            ///     <c>could not convert string to float: np.str_('x')</c>).</exception>
            /// <exception cref="OpenBlasMissingBackendException">A matrix order 2, -2 or <c>"nuc"</c> over a non-empty
            ///     matrix with no matrix backend installed (an empty matrix or a zero-size batch needs none).</exception>
            /// <exception cref="AxisError">An axis out of range.</exception>
            /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            ///     <para>
            ///     <b>Every order works WITHOUT a matrix backend except three</b> — matrix
            ///     <c>ord</c> 2, -2 and <c>"nuc"</c>, which are defined through singular values and so
            ///     raise <see cref="OpenBlasMissingBackendException"/>. The rest are reductions
            ///     (<see cref="np.abs(NDArray, NDArray, NDArray, DType)"/>, <see cref="np.sum(NDArray, NDAxis?, DType, NDArray, bool, object, NDArray)"/>, <see cref="np.amax(NDArray, NDAxis?, NDArray, bool, object, NDArray)"/>,
            ///     <see cref="np.power(NDArray, NDArray)"/>) and compute normally.
            ///     </para>
            ///     <para>
            ///     How many axes are being reduced decides which vocabulary of orders applies, which
            ///     is why <c>norm(vector, "fro")</c> and <c>norm(matrix, 3)</c> both raise but with
            ///     DIFFERENT messages, and a 3-D operand with an explicit <c>ord</c> and no
            ///     <c>axis</c> raises a third ("Improper number of dimensions to norm.").
            ///     </para>
            ///     <para>One signature for NumPy's int-or-tuple <c>axis</c> since API-parity batch 6 (<c>norm(x, null, null,
            ///     false)</c> was ambiguous between the int? and int[] forms); the <c>int</c> and <c>int[]</c> overloads forward
            ///     here. NumPy turns an int into the 1-tuple <c>(axis,)</c>, so both reach the engine as one array.</para>
            /// </remarks>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static NDArray norm(NDArray x, object ord = null, NDAxis? axis = null, bool keepdims = false)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                return x.TensorEngine.Linalg.Norm(x, ord, axis?.ToArray(), keepdims);
            }

            /// <summary>
            ///     <see cref="norm(NDArray, object, NDAxis?, bool)"/> with one <c>int</c> axis: the overload a C# integer binds
            ///     (an <c>ord</c> must be given positionally or by name with it), forwarding to the NDAxis? form.
            /// </summary>
            /// <param name="x">Input array.</param>
            /// <param name="ord">The order (see the NDAxis? form; <c>null</c> for NumPy's default).</param>
            /// <param name="axis">The axis of a vector norm (a negative value counts from the end).</param>
            /// <param name="keepdims">Leave the reduced axis in the result with length 1.</param>
            /// <returns>The norm.</returns>
            /// <inheritdoc cref="norm(NDArray, object, NDAxis?, bool)"/>
            public static NDArray norm(NDArray x, object ord, int axis, bool keepdims = false)
                => norm(x, ord, (NDAxis?)axis, keepdims);

            /// <summary>
            ///     <see cref="norm(NDArray, object, NDAxis?, bool)"/> with the axes as an <c>int[]</c>: the overload an
            ///     <c>int[]</c> and a bare <c>null</c> bind, forwarding to the NDAxis? form (a null array is None, any other
            ///     array NumPy's tuple).
            /// </summary>
            /// <param name="x">Input array.</param>
            /// <param name="ord">The order (see the NDAxis? form).</param>
            /// <param name="axis">The axes as NumPy's tuple (one for a vector norm, two for a matrix norm), or <c>null</c>.</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <returns>The norm.</returns>
            /// <inheritdoc cref="norm(NDArray, object, NDAxis?, bool)"/>
            public static NDArray norm(NDArray x, object ord, int[] axis, bool keepdims = false)
                => norm(x, ord, (NDAxis?)axis, keepdims);

            /// <summary>
            ///     The Array-API vector norm — <c>axis</c> defaults to flattening and <c>ord</c> to 2.
            /// </summary>
            /// <param name="x">Input array. Integer and bool operands are computed in float64.</param>
            /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (the flattened array), one axis, or a tuple of any number of
            ///     axes (negative values count from the end; <c>()</c> reduces nothing: the elementwise magnitude).</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <param name="ord">
            ///     The order. <c>null</c> means NumPy's default of <b>2</b> — C# cannot spell a
            ///     non-null default for an <c>object</c> parameter, so the sentinel carries it.
            /// </param>
            /// <returns>The vector norm over the axes; with <paramref name="keepdims"/> a view of it (NumPy's reshape, which
            ///     does not own its data).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis).</exception>
            /// <exception cref="ValueError">An axis appears twice ("repeated axis"), a text order, or -inf of an empty
            ///     operand.</exception>
            /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html
            ///     <para>
            ///     An integer axis is NOT the one-element tuple: NumPy hands an int to norm as it is, while its tuple branch,
            ///     statement for statement, is <c>normalize_axis_tuple(axis, x.ndim)</c>, the reduced axes moved to the front
            ///     (<c>transpose(x, axis + rest)</c>, the axes AS GIVEN) and merged into one
            ///     (<c>reshape((prod(shape[axis]), *shape[rest]))</c>), then norm's ONE-axis branch over axis 0 - a vector
            ///     norm whatever the tuple's length. Both reach the vector branch, in different reduction layouts, so the
            ///     engine keeps them apart. Handing a 2-tuple to <see cref="norm(NDArray, object, NDAxis?, bool)"/> directly
            ///     would reach its MATRIX branch instead (the largest singular value, not <c>sqrt(sum(|x|^2))</c>).
            ///     </para>
            ///     <para>One signature for NumPy's int-or-tuple <c>axis</c> since API-parity batch 6; the <c>int</c> and
            ///     <c>int[]</c> overloads forward here.</para>
            /// </remarks>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static NDArray vector_norm(NDArray x, NDAxis? axis = null, bool keepdims = false, object ord = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                // The engine's two members are NumPy's two branches: an int axis, or a tuple (None = the flattened array,
                // the engine's null array).
                if (axis is { IsTuple: false } single)
                    return x.TensorEngine.Linalg.VectorNorm(x, single[0], keepdims, ord);
                return x.TensorEngine.Linalg.VectorNorm(x, axis?.ToArray(), keepdims, ord);
            }

            /// <summary>
            ///     <see cref="vector_norm(NDArray, NDAxis?, bool, object)"/> with one <c>int</c> axis: the overload a C#
            ///     integer binds, forwarding to the NDAxis? form (NumPy's integer branch: norm along that axis).
            /// </summary>
            /// <param name="x">Input array.</param>
            /// <param name="axis">The reduced axis (a negative value counts from the end).</param>
            /// <param name="keepdims">Leave the reduced axis in the result with length 1.</param>
            /// <param name="ord">The vector order; <c>null</c> means NumPy's default of 2.</param>
            /// <returns>The norm along <paramref name="axis"/>.</returns>
            /// <inheritdoc cref="vector_norm(NDArray, NDAxis?, bool, object)"/>
            public static NDArray vector_norm(NDArray x, int axis, bool keepdims = false, object ord = null)
                => vector_norm(x, (NDAxis?)axis, keepdims, ord);

            /// <summary>
            ///     <see cref="vector_norm(NDArray, NDAxis?, bool, object)"/> with the axes as an <c>int[]</c>: the overload an
            ///     <c>int[]</c> and a bare <c>null</c> bind, forwarding to the NDAxis? form (a null array is the flattened
            ///     array, any other array NumPy's tuple branch).
            /// </summary>
            /// <param name="x">Input array.</param>
            /// <param name="axis">The reduced axes as NumPy's tuple, or <c>null</c> for the flattened array.</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <param name="ord">The vector order; <c>null</c> means NumPy's default of 2.</param>
            /// <returns>The vector norm over the axes.</returns>
            /// <inheritdoc cref="vector_norm(NDArray, NDAxis?, bool, object)"/>
            public static NDArray vector_norm(NDArray x, int[] axis, bool keepdims = false, object ord = null)
                => vector_norm(x, (NDAxis?)axis, keepdims, ord);

            /// <summary>
            ///     The Array-API matrix norm — always reduces the LAST TWO axes, and defaults to
            ///     Frobenius.
            /// </summary>
            /// <param name="x">The matrix, or a stack of them (the last two axes).</param>
            /// <param name="keepdims">Leave the two reduced axes in the result with length 1.</param>
            /// <param name="ord">
            ///     The order. <c>null</c> means NumPy's default of <c>"fro"</c> — C# cannot spell a
            ///     non-null default for an <c>object</c> parameter, so the sentinel carries it.
            /// </param>
            /// <returns>The matrix norm over the last two axes.</returns>
            /// <exception cref="ValueError">An invalid matrix order, or a min order of an empty matrix.</exception>
            /// <exception cref="AxisError">An operand of fewer than two axes.</exception>
            /// <exception cref="OpenBlasMissingBackendException">An order 2, -2 or <c>"nuc"</c> over a non-empty matrix
            ///     with no matrix backend installed.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_norm.html</remarks>
            /// <remarks>
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.
            /// </remarks>
            public static NDArray matrix_norm(NDArray x, bool keepdims = false, object ord = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                return x.TensorEngine.Linalg.MatrixNorm(x, keepdims, ord);
            }
        }
    }
}
