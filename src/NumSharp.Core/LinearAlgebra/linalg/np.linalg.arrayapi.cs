using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Matrix product — the Array-API spelling of <see cref="np.matmul"/>.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.matmul.html
            ///     <para>
            ///     This one really is the same operation as its main-namespace twin, unlike the rest
            ///     of the forms in this file.
            ///     </para>
            /// </remarks>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray matmul(NDArray x1, NDArray x2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Matmul(x1, x2);
            }

            /// <summary>
            ///     Outer product of two VECTORS.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html
            ///     <para>
            ///     Not a synonym for <see cref="np.outer"/>: that one flattens whatever it is given,
            ///     while this rejects anything but 1-D. <c>np.outer(ones((2,3)), ones(3))</c> is a
            ///     <c>(6,3)</c> array; <c>np.linalg.outer</c> of the same pair raises.
            ///     </para>
            /// </remarks>
            /// <param name="x1">The first vector.</param>
            /// <param name="x2">The second vector.</param>
            /// <returns>The <c>(len(x1), len(x2))</c> outer product, a fresh array.</returns>
            /// <exception cref="ValueError">Either operand is not one-dimensional.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray outer(NDArray x1, NDArray x2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Outer(x1, x2);
            }

            /// <summary>
            ///     Tensor contraction — the Array-API spelling of <see cref="np.tensordot(NDArray, NDArray, int)"/>.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html</remarks>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray tensordot(NDArray x1, NDArray x2, int axes = 2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Tensordot(x1, x2, axes);
            }

            /// <summary>
            ///     Tensor contraction pairing ONE axis of <paramref name="x1"/> with one of <paramref name="x2"/> — NumPy's
            ///     <c>np.linalg.tensordot(x1, x2, axes=(1, 0))</c>, the Array-API spelling of
            ///     <see cref="np.tensordot(NDArray, NDArray, ValueTuple{int, int})"/>.
            /// </summary>
            /// <param name="x1">The first tensor.</param>
            /// <param name="x2">The second tensor.</param>
            /// <param name="axes">The axis of <paramref name="x1"/> and the axis of <paramref name="x2"/> to contract (NumPy's
            ///     <c>axes_a, axes_b = axes</c> makes each a one-element list).</param>
            /// <returns>A new array of <paramref name="x1"/>'s kept axes followed by <paramref name="x2"/>'s.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            /// <exception cref="IndexError">An axis out of range (<c>tuple index out of range</c>).</exception>
            /// <exception cref="ValueError">Extents that differ (<c>shape-mismatch for sum</c>).</exception>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html — NumPy's axes is
            ///     keyword-only there; C# cannot refuse a positional pair, so <c>tensordot(x1, x2, (1, 0))</c> also binds.
            ///     Added in API-parity batch 7 (np.tensordot had the pair already).</remarks>
            public static NDArray tensordot(NDArray x1, NDArray x2, (int AxisA, int AxisB) axes)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Tensordot(x1, x2, new[] { axes.AxisA }, new[] { axes.AxisB });
            }

            /// <summary>
            ///     Tensor contraction pairing a list of axes of <paramref name="x1"/> with a list of axes of
            ///     <paramref name="x2"/>, element by element — NumPy's <c>np.linalg.tensordot(x1, x2, axes=([1, 0], [0, 1]))</c>,
            ///     the pair as ONE argument: <c>np.linalg.tensordot(x1, x2, axes: (new[] { 1, 0 }, new[] { 0, 1 }))</c>.
            /// </summary>
            /// <param name="x1">The first tensor.</param>
            /// <param name="x2">The second tensor.</param>
            /// <param name="axes">NumPy's pair of axis lists: <c>AxesA</c> of <paramref name="x1"/>, <c>AxesB</c> of
            ///     <paramref name="x2"/>, paired element by element.</param>
            /// <returns>A new array of <paramref name="x1"/>'s kept axes followed by <paramref name="x2"/>'s.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            /// <exception cref="ValueError"><c>duplicate axes are not allowed in tensordot</c> (checked first), or
            ///     <c>shape-mismatch for sum</c> for unequal list lengths and mismatched extents.</exception>
            /// <exception cref="IndexError">A contraction axis out of range (<c>tuple index out of range</c>).</exception>
            /// <exception cref="System.NullReferenceException">An axis list is null.</exception>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html — the statements are
            ///     the Linalg engine's <c>Tensordot(x1, x2, int[], int[])</c>. Until API-parity batch 7 the pair was split into
            ///     two parameters, <c>tensordot(x1, x2, int[] axesA, int[] axesB)</c>, a four-argument call NumPy refuses
            ///     (<c>tensordot() takes 2 positional arguments but 4 were given</c>).</remarks>
            public static NDArray tensordot(NDArray x1, NDArray x2, (int[] AxesA, int[] AxesB) axes)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Tensordot(x1, x2, axes.AxesA, axes.AxesB);
            }

            /// <summary>
            ///     Sum along the diagonals of the LAST TWO axes.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.trace.html
            ///     <para>
            ///     <b>The axes differ from <see cref="np.trace"/>'s</b>, which defaults to the FIRST
            ///     two — so on a <c>(2,3,3)</c> stack this returns shape <c>(2,)</c> (one trace per
            ///     matrix) where <c>np.trace</c> returns <c>(3,)</c>. Neither is wrong; they are
            ///     different functions.
            ///     </para>
            /// </remarks>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            public static NDArray trace(NDArray x, int offset = 0, DType dtype = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                return x.TensorEngine.Linalg.Trace(x, offset, dtype);
            }

            /// <summary>
            ///     Diagonals of the LAST TWO axes.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.diagonal.html
            ///     <para>
            ///     As with <see cref="trace"/>, the axes differ from <see cref="np.diagonal"/>'s
            ///     first-two default: a <c>(2,3,3)</c> stack gives <c>(2,3)</c> here and
            ///     <c>(3,2)</c> there.
            ///     </para>
            /// </remarks>
            public static NDArray diagonal(NDArray x, int offset = 0)
                => np.diagonal(x, offset, -2, -1);

            /// <summary>
            ///     Transposes the last two axes — the Array-API spelling of
            ///     <see cref="np.matrix_transpose"/>. An O(1) view.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_transpose.html</remarks>
            public static NDArray matrix_transpose(NDArray x) => np.matrix_transpose(x);

            /// <summary>
            ///     Vector dot product over <paramref name="axis"/>, conjugating the first operand.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.vecdot.html</remarks>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray vecdot(NDArray x1, NDArray x2, int axis = -1)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Vecdot(x1, x2, axis);
            }

            /// <summary>
            ///     Cross product of 3-element vectors.
            /// </summary>
            /// <param name="axis">The axis holding the three components. Defaults to the last.</param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.cross.html
            ///     <para>
            ///     <b>Stricter than <see cref="np.cross"/>:</b> the Array-API form accepts 3-vectors
            ///     ONLY, over a single <paramref name="axis"/>, where the main-namespace
            ///     <see cref="np.cross"/> also takes the 2-vector form (deprecated in NumPy 2.0) and a
            ///     separate <c>axisa</c>/<c>axisb</c>/<c>axisc</c> per operand. Past the length check it IS
            ///     <see cref="np.cross"/> (NumPy's <c>_core_cross(x1, x2, axis=axis)</c>): the dtypes promote
            ///     strongly (<c>promote_types</c>), and the result is a view of a fresh C-ordered array whose
            ///     component axis is moved to <paramref name="axis"/>.
            ///     </para>
            /// </remarks>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <returns>The cross product, the component axis at <paramref name="axis"/> of the broadcast result.
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="IndexError"><paramref name="axis"/> is out of range for either operand ("tuple index out of
            ///     range": NumPy indexes <c>x1.shape[axis]</c> before anything else).</exception>
            /// <exception cref="ValueError">The component axis of either operand does not have length 3, or the other axes
            ///     do not broadcast.</exception>
            /// <exception cref="TypeError">Either operand holds text (the multiply has no string loop).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static NDArray cross(NDArray x1, NDArray x2, int axis = -1)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Cross(x1, x2, axis);
            }
        }
    }
}
