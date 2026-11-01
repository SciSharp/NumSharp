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
            /// <param name="axis">The axis to reduce. <c>null</c> flattens.</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <returns>The norm: a 0-d array for a whole-array norm, else the reduced array. An EMPTY operand's
            ///     inf-, 1- and 2-norms are 0.0 (NumPy 2.x reduces those with <c>initial=0</c>).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="ValueError">An order that is invalid for the number of reduced axes, duplicate axes, more
            ///     than two reduced axes, or a min order (-1, -inf, -2) of an empty operand.</exception>
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
            ///     (<see cref="np.abs(NDArray)"/>, <see cref="np.sum(NDArray)"/>, <see cref="np.amax"/>,
            ///     <see cref="np.power(NDArray, NDArray)"/>) and compute normally.
            ///     </para>
            ///     <para>
            ///     How many axes are being reduced decides which vocabulary of orders applies, which
            ///     is why <c>norm(vector, "fro")</c> and <c>norm(matrix, 3)</c> both raise but with
            ///     DIFFERENT messages, and a 3-D operand with an explicit <c>ord</c> and no
            ///     <c>axis</c> raises a third ("Improper number of dimensions to norm.").
            ///     </para>
            /// </remarks>
            public static NDArray norm(NDArray x, object ord = null, int? axis = null, bool keepdims = false)
                => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
                ? NDMatrixBoundary.Plain((x, ord, axis, keepdims), static s => norm(s.x, s.ord, s.axis, s.keepdims))
                : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
                ? NDCharArrayBoundary.Plain((x, ord, axis, keepdims), static s => norm(s.x, s.ord, s.axis, s.keepdims))
                : norm(x, ord, axis is null ? null : new[] {axis.Value}, keepdims);

            /// <summary>
            ///     Matrix or vector norm reduced over an explicit axis tuple.
            /// </summary>
            /// <inheritdoc cref="norm(NDArray, object, int?, bool)"/>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static NDArray norm(NDArray x, object ord, int[] axis, bool keepdims = false)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
                if (NDMatrixBoundary.Sees(x))
                    return NDMatrixBoundary.Plain((x, ord, axis, keepdims), static s => norm(s.x, s.ord, s.axis, s.keepdims));
                // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
                if (NDCharArrayBoundary.Sees(x))
                    return NDCharArrayBoundary.Plain((x, ord, axis, keepdims), static s => norm(s.x, s.ord, s.axis, s.keepdims));
                return x.TensorEngine.Linalg.Norm(x, ord, axis, keepdims);
            }

            /// <summary>
            ///     The Array-API vector norm reduced over a single axis: NumPy's <c>norm(x, axis=axis, ord=ord)</c>, the
            ///     one-axis branch of <see cref="norm(NDArray, object, int[], bool)"/>.
            /// </summary>
            /// <param name="x">Input array. Integer and bool operands are computed in float64.</param>
            /// <param name="axis">The reduced axis (negative values count from the end).</param>
            /// <param name="keepdims">Leave the reduced axis in the result with length 1.</param>
            /// <param name="ord">The vector order; <c>null</c> means NumPy's default of 2.</param>
            /// <returns>The norm along <paramref name="axis"/>; with <paramref name="keepdims"/> a view of it (NumPy's
            ///     reshape, which does not own its data).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="AxisError"><paramref name="axis"/> is out of range (a 0-d input has no axis).</exception>
            /// <exception cref="ValueError">A text order, or -inf of an empty axis.</exception>
            /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <remarks>
            ///     An integer axis is NOT the one-element tuple: NumPy hands it to norm as it is, while a tuple is moved to
            ///     the front and merged first (see the <c>int[]</c> overload). Both reach the vector branch, but in different
            ///     reduction layouts, so they are kept apart.
            /// </remarks>
            public static NDArray vector_norm(NDArray x, int axis, bool keepdims = false, object ord = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                return NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
                ? NDMatrixBoundary.Plain((x, axis, keepdims, ord), static s => vector_norm(s.x, s.axis, s.keepdims, s.ord))
                : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
                ? NDCharArrayBoundary.Plain((x, axis, keepdims, ord), static s => vector_norm(s.x, s.axis, s.keepdims, s.ord))
                : x.TensorEngine.Linalg.VectorNorm(x, axis, keepdims, ord);
            }

            /// <summary>
            ///     The Array-API vector norm — <c>axis</c> defaults to flattening and <c>ord</c> to 2.
            /// </summary>
            /// <param name="ord">
            ///     The order. <c>null</c> means NumPy's default of <b>2</b> — C# cannot spell a
            ///     non-null default for an <c>object</c> parameter, so the sentinel carries it.
            /// </param>
            /// <param name="x">Input array. Integer and bool operands are computed in float64.</param>
            /// <param name="axis">The reduced axes, any number of them (negative values count from the end), or null for
            ///     the flattened array. An empty tuple reduces nothing (the elementwise magnitude).</param>
            /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
            /// <returns>The vector norm over the axes.
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="AxisError">An axis is out of range.</exception>
            /// <exception cref="ValueError">An axis appears twice ("repeated axis"), a text order, or -inf of an empty
            ///     operand.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html
            ///     <para>
            ///     NumPy's tuple branch, statement for statement: <c>normalize_axis_tuple(axis, x.ndim)</c>, the reduced axes
            ///     moved to the front (<c>transpose(x, axis + rest)</c>, the axes AS GIVEN) and merged into one
            ///     (<c>reshape((prod(shape[axis]), *shape[rest]))</c>), then norm's ONE-axis branch over axis 0 - a vector
            ///     norm whatever the tuple's length. Handing a 2-tuple to <see cref="norm(NDArray, object, int[], bool)"/>
            ///     directly (as this method once did) reached its MATRIX branch instead: the default order then computed the
            ///     largest singular value, not <c>sqrt(sum(|x|^2))</c>, and a 3-tuple raised "Improper number of dimensions
            ///     to norm.".
            ///     </para>
            /// </remarks>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static NDArray vector_norm(NDArray x, int[] axis = null, bool keepdims = false, object ord = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
                if (NDMatrixBoundary.Sees(x))
                    return NDMatrixBoundary.Plain((x, axis, keepdims, ord), static s => vector_norm(s.x, s.axis, s.keepdims, s.ord));
                // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
                if (NDCharArrayBoundary.Sees(x))
                    return NDCharArrayBoundary.Plain((x, axis, keepdims, ord), static s => vector_norm(s.x, s.axis, s.keepdims, s.ord));
                return x.TensorEngine.Linalg.VectorNorm(x, axis, keepdims, ord);
            }

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
                return NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
                ? NDMatrixBoundary.Plain((x, keepdims, ord), static s => matrix_norm(s.x, s.keepdims, s.ord))
                : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
                ? NDCharArrayBoundary.Plain((x, keepdims, ord), static s => matrix_norm(s.x, s.keepdims, s.ord))
                : x.TensorEngine.Linalg.MatrixNorm(x, keepdims, ord);
            }
        }
    }
}
