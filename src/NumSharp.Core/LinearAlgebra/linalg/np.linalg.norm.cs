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
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            ///     <para>
            ///     <b>Every order works WITHOUT a LAPACK backend except three</b> — matrix
            ///     <c>ord</c> 2, -2 and <c>"nuc"</c>, which are defined through singular values and so
            ///     raise <see cref="NotSupportedException"/>. The rest are reductions
            ///     (<see cref="np.abs"/>, <see cref="np.sum"/>, <see cref="np.amax"/>,
            ///     <see cref="np.power"/>) and compute normally.
            ///     </para>
            ///     <para>
            ///     How many axes are being reduced decides which vocabulary of orders applies, which
            ///     is why <c>norm(vector, "fro")</c> and <c>norm(matrix, 3)</c> both raise but with
            ///     DIFFERENT messages, and a 3-D operand with an explicit <c>ord</c> and no
            ///     <c>axis</c> raises a third ("Improper number of dimensions to norm.").
            ///     </para>
            /// </remarks>
            public static NDArray norm(NDArray x, object ord = null, int? axis = null, bool keepdims = false)
                => norm(x, ord, axis is null ? null : new[] {axis.Value}, keepdims);

            /// <summary>
            ///     Matrix or vector norm reduced over an explicit axis tuple.
            /// </summary>
            /// <inheritdoc cref="norm(NDArray, object, int?, bool)"/>
            public static NDArray norm(NDArray x, object ord, int[] axis, bool keepdims = false)
            {
                var a = ToFloating(x);
                int nd = a.ndim;

                // NumPy's fast path: with no axis and the default order, the whole array is one
                // vector — and it reaches it through `x.dot(x)`, so this mirrors that call rather
                // than composing a reduction with a different summation order.
                if (axis is null &&
                    (ord is null
                     || (IsText(ord, out var word) && (word == "f" || word == "fro") && nd == 2)
                     || (IsNumber(ord, out double two) && two == 2d && nd == 1)))
                {
                    var flat = np.ravel(a);
                    var squared = a.typecode == NPTypeCode.Complex
                        ? np.sum(AbsSquared(flat))
                        : np.dot(flat, flat);
                    var whole = np.sqrt(squared);
                    return keepdims ? np.reshape(whole, Ones(nd)) : whole;
                }

                int[] axes = axis ?? AllAxes(nd);

                if (axes.Length == 1)
                    return VectorNorm(a, ord, NormalizeAxis(axes[0], nd), keepdims);
                if (axes.Length == 2)
                    return MatrixNorm(a, ord, axes, nd, keepdims);

                throw new ValueError("Improper number of dimensions to norm.");
            }

            /// <summary>
            ///     The Array-API vector norm reduced over a single axis.
            /// </summary>
            /// <inheritdoc cref="vector_norm(NDArray, int[], bool, object)"/>
            public static NDArray vector_norm(NDArray x, int axis, bool keepdims = false, object ord = null)
                => vector_norm(x, new[] {axis}, keepdims, ord);

            /// <summary>
            ///     The Array-API vector norm — <c>axis</c> defaults to flattening and <c>ord</c> to 2.
            /// </summary>
            /// <param name="ord">
            ///     The order. <c>null</c> means NumPy's default of <b>2</b> — C# cannot spell a
            ///     non-null default for an <c>object</c> parameter, so the sentinel carries it.
            /// </param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html</remarks>
            public static NDArray vector_norm(NDArray x, int[] axis = null, bool keepdims = false, object ord = null)
            {
                object order = ord ?? 2;
                if (axis is not null)
                    return norm(x, order, axis, keepdims);

                // Flatten first, then reduce — and restore the rank afterwards when keepdims asks,
                // because the ravel has already erased it.
                var flattened = norm(np.ravel(ToFloating(x)), order, new[] {0}, false);
                return keepdims ? np.reshape(flattened, Ones(x.ndim)) : flattened;
            }

            /// <summary>
            ///     The Array-API matrix norm — always reduces the LAST TWO axes, and defaults to
            ///     Frobenius.
            /// </summary>
            /// <param name="ord">
            ///     The order. <c>null</c> means NumPy's default of <c>"fro"</c> — C# cannot spell a
            ///     non-null default for an <c>object</c> parameter, so the sentinel carries it.
            /// </param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_norm.html</remarks>
            public static NDArray matrix_norm(NDArray x, bool keepdims = false, object ord = null)
                => norm(x, ord ?? "fro", new[] {-2, -1}, keepdims);

            #region internals

            private static NDArray VectorNorm(NDArray a, object ord, int axis, bool keepdims)
            {
                if (ord is null || (IsNumber(ord, out double n) && n == 2d))
                    return np.sqrt(np.sum(AbsSquared(a), axis, keepdims));

                if (IsText(ord, out var word))
                    throw new ValueError($"Invalid norm order '{word}' for vectors");

                double order = AsNumber(ord);

                if (double.IsPositiveInfinity(order))
                    return np.amax(np.abs(a), axis, keepdims: keepdims);
                if (double.IsNegativeInfinity(order))
                    return np.amin(np.abs(a), axis, keepdims: keepdims);
                if (order == 0d)
                {
                    // NumPy counts non-zeros in `x.real.dtype`, NOT `x.dtype` — so a COMPLEX operand's
                    // zero-"norm" is float64, not complex. For every real dtype the real component's
                    // dtype is the dtype itself, so only Complex needs the substitution.
                    var countType = a.typecode == NPTypeCode.Complex ? NPTypeCode.Double : a.typecode;
                    return np.sum(np.not_equal(a, NDArray.Scalar(0)).astype(countType), axis, keepdims);
                }
                if (order == 1d)
                    return np.sum(np.abs(a), axis, keepdims);

                var raised = np.power(np.abs(a), NDArray.Scalar(order));
                var reduced = np.sum(raised, axis, keepdims);
                return np.power(reduced, NDArray.Scalar(1d / order));
            }

            private static NDArray MatrixNorm(NDArray a, object ord, int[] axes, int nd, bool keepdims)
            {
                int rowAxis = NormalizeAxis(axes[0], nd);
                int colAxis = NormalizeAxis(axes[1], nd);
                if (rowAxis == colAxis)
                    throw new ValueError("Duplicate axes given.");

                NDArray result;

                if (IsText(ord, out var word))
                {
                    switch (word)
                    {
                        case "f":
                        case "fro":
                            result = FrobeniusOverBoth(a, rowAxis, colAxis);
                            break;
                        case "nuc":
                            result = SingularValueNorm(a, rowAxis, colAxis, "nuc");
                            break;
                        default:
                            throw new ValueError("Invalid norm order for matrices.");
                    }
                }
                else if (ord is null)
                {
                    result = FrobeniusOverBoth(a, rowAxis, colAxis);
                }
                else
                {
                    double order = AsNumber(ord);

                    // NumPy reduces one axis FIRST and then shifts the surviving axis index down if
                    // the reduced one sat before it — so these pairs are not interchangeable.
                    if (order == 2d || order == -2d)
                        result = SingularValueNorm(a, rowAxis, colAxis, order == 2d ? "2" : "-2");
                    else if (order == 1d)
                        result = np.amax(np.sum(np.abs(a), rowAxis), Shift(colAxis, rowAxis));
                    else if (order == -1d)
                        result = np.amin(np.sum(np.abs(a), rowAxis), Shift(colAxis, rowAxis));
                    else if (double.IsPositiveInfinity(order))
                        result = np.amax(np.sum(np.abs(a), colAxis), Shift(rowAxis, colAxis));
                    else if (double.IsNegativeInfinity(order))
                        result = np.amin(np.sum(np.abs(a), colAxis), Shift(rowAxis, colAxis));
                    else
                        throw new ValueError("Invalid norm order for matrices.");
                }

                if (!keepdims)
                    return result;

                // Put the two reduced axes back as length-1, lowest index first so the second
                // insertion is not displaced by the first.
                int first = rowAxis < colAxis ? rowAxis : colAxis;
                int second = rowAxis < colAxis ? colAxis : rowAxis;
                return np.expand_dims(np.expand_dims(result, first), second);
            }

            private static NDArray FrobeniusOverBoth(NDArray a, int rowAxis, int colAxis)
            {
                var squared = AbsSquared(a);
                var once = np.sum(squared, rowAxis);
                return np.sqrt(np.sum(once, Shift(colAxis, rowAxis)));
            }

            /// <summary>An axis index after a LOWER-numbered axis has been reduced away.</summary>
            private static int Shift(int axis, int removed) => axis > removed ? axis - 1 : axis;

            private static NDArray SingularValueNorm(NDArray a, int rowAxis, int colAxis, string which)
            {
                // NumPy's _multi_svd_norm: move the two matrix axes to the end, take the singular values
                // over them, then reduce — 2 → largest, -2 → smallest, 'nuc' → sum of the singulars.
                var y = np.moveaxis(a, new[] {rowAxis, colAxis}, new[] {-2, -1});
                var s = svdvals(y);
                return which switch
                {
                    "2" => np.amax(s, -1),
                    "-2" => np.amin(s, -1),
                    _ => np.sum(s, -1), // "nuc"
                };
            }

            /// <summary>
            ///     NumPy computes norms in floating point — a bool or integer operand is cast up
            ///     before anything else happens, which is why <c>norm(arange(3))</c> is float64.
            /// </summary>
            private static NDArray ToFloating(NDArray x)
                => x.typecode switch
                {
                    NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double
                        or NPTypeCode.Decimal or NPTypeCode.Complex => x,
                    _ => x.TensorEngine.Cast(x, NPTypeCode.Double, copy: false)
                };

            private static int[] AllAxes(int nd)
            {
                var all = new int[nd];
                for (int i = 0; i < nd; i++)
                    all[i] = i;
                return all;
            }

            private static int[] Ones(int nd)
            {
                var ones = new int[nd];
                for (int i = 0; i < nd; i++)
                    ones[i] = 1;
                return ones;
            }

            private static int NormalizeAxis(int axis, int nd)
            {
                int resolved = axis < 0 ? axis + nd : axis;
                if (resolved < 0 || resolved >= nd)
                    throw new AxisOutOfRangeException($"axis {axis} is out of bounds for array of dimension {nd}");
                return resolved;
            }

            private static bool IsText(object ord, out string word)
            {
                word = ord as string;
                return word is not null;
            }

            private static bool IsNumber(object ord, out double value)
            {
                value = 0;
                if (ord is null || ord is string)
                    return false;
                value = AsNumber(ord);
                return true;
            }

            private static double AsNumber(object ord)
            {
                try
                {
                    return Convert.ToDouble(ord, System.Globalization.CultureInfo.InvariantCulture);
                }
                catch (Exception e) when (e is FormatException or InvalidCastException or OverflowException)
                {
                    throw new TypeError("'ord' must be None, a number or one of 'fro', 'f', 'nuc'");
                }
            }

            #endregion
        }
    }
}
