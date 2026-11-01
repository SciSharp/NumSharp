namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Tensor contraction over the given axes — a sum product over the last
        ///     <paramref name="axes"/> axes of <paramref name="a"/> and the first
        ///     <paramref name="axes"/> of <paramref name="b"/>.
        /// </summary>
        /// <param name="axes">
        ///     How many trailing axes of <c>a</c> to contract against leading axes of <c>b</c>.
        ///     <c>0</c> gives the outer (tensor) product; <c>1</c> is <see cref="dot"/>; <c>2</c>
        ///     (the default) is the double contraction. A NEGATIVE count contracts nothing — NumPy
        ///     forms <c>range(-axes, 0)</c>, which is empty for any <c>axes &lt;= 0</c>.
        /// </param>
        /// <param name="a">The first tensor (not null).</param>
        /// <param name="b">The second tensor (not null).</param>
        /// <returns>A new array of <paramref name="a"/>'s kept axes followed by <paramref name="b"/>'s.</returns>
        /// <exception cref="IndexError">More axes than an operand has (<c>tuple index out of range</c>).</exception>
        /// <exception cref="ValueError">Contracted extents that differ (<c>shape-mismatch for sum</c>), or the final dot's
        ///     refusals of text.</exception>
        /// <exception cref="TypeError">The final dot's refusals of text operands.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand beside a number (NumPy's object
        ///     product).</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html</remarks>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class. The statements are
        ///     <see cref="TensorEngine.Tensordot(NDArray, NDArray, int)"/>, on <paramref name="a"/>'s engine.
        /// </remarks>
        public static NDArray tensordot(NDArray a, NDArray b, int axes = 2)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a, b))
                return NDMatrixBoundary.Plain((a, b, axes), static s => tensordot(s.a, s.b, s.axes));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a, b))
                return NDCharArrayBoundary.Plain((a, b, axes), static s => tensordot(s.a, s.b, s.axes));
            // NumPy's body (the two axis lists from the count, the axis checks, the final dot) is the engine's.
            return a.TensorEngine.Tensordot(a, b, axes);
        }

        /// <summary>
        ///     Tensor contraction pairing one axis of <paramref name="a"/> with one of
        ///     <paramref name="b"/> — NumPy's <c>axes=(int, int)</c> spelling.
        /// </summary>
        /// <param name="a">The first tensor (not null).</param>
        /// <param name="b">The second tensor (not null).</param>
        /// <param name="axes">The axis of <paramref name="a"/> and the axis of <paramref name="b"/> to contract.</param>
        /// <returns>A new array of <paramref name="a"/>'s kept axes followed by <paramref name="b"/>'s.</returns>
        /// <exception cref="IndexError">An axis out of range (<c>tuple index out of range</c>).</exception>
        /// <exception cref="ValueError">Extents that differ (<c>shape-mismatch for sum</c>), or the final dot's refusals of
        ///     text.</exception>
        /// <exception cref="TypeError">The final dot's refusals of text operands.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand beside a number.</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class. NumPy's <c>axes_a, axes_b = axes</c> with two ints makes
        ///     the one-element lists <c>[axes_a]</c> / <c>[axes_b]</c>, so the pair calls
        ///     <see cref="TensorEngine.Tensordot(NDArray, NDArray, int[], int[])"/>, on <paramref name="a"/>'s engine.
        /// </remarks>
        public static NDArray tensordot(NDArray a, NDArray b, (int AxisA, int AxisB) axes)
            => NDMatrixBoundary.Sees(a, b)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((a, b, axes), static s => tensordot(s.a, s.b, s.axes))
            : NDCharArrayBoundary.Sees(a, b)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((a, b, axes), static s => tensordot(s.a, s.b, s.axes))
            : a.TensorEngine.Tensordot(a, b, new[] {axes.AxisA}, new[] {axes.AxisB});

        /// <summary>
        ///     Tensor contraction pairing <paramref name="axesA"/> of <paramref name="a"/> with
        ///     <paramref name="axesB"/> of <paramref name="b"/>, element by element — NumPy's
        ///     <c>axes=(list, list)</c> spelling.
        /// </summary>
        /// <exception cref="ValueError">
        ///     <c>"duplicate axes are not allowed in tensordot"</c> when either list repeats a raw
        ///     axis value (checked first, ahead of everything), or <c>"shape-mismatch for sum"</c> for
        ///     unequal list lengths and mismatched contracted extents alike.
        /// </exception>
        /// <exception cref="IndexError">
        ///     <c>"tuple index out of range"</c> when a contraction axis is out of range — NumPy
        ///     indexes the shape tuple with the raw axis, so this is an index error, not a shape
        ///     mismatch. A mismatched extent found earlier in the list settles as the shape mismatch
        ///     first, hiding a later out-of-range axis.
        /// </exception>
        /// <exception cref="TypeError">The final dot's refusals of text operands.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand beside a number (NumPy's object
        ///     product).</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/>, <paramref name="b"/> or an axis list is
        ///     null.</exception>
        /// <param name="a">The first tensor (not null).</param>
        /// <param name="b">The second tensor (not null).</param>
        /// <param name="axesA">The axes of <paramref name="a"/> to contract, in pairing order.</param>
        /// <param name="axesB">The axes of <paramref name="b"/> to contract, paired element by element with
        ///     <paramref name="axesA"/>.</param>
        /// <returns>A new array of <paramref name="a"/>'s kept axes followed by <paramref name="b"/>'s.</returns>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class. The statements are
        ///     <see cref="TensorEngine.Tensordot(NDArray, NDArray, int[], int[])"/>, on <paramref name="a"/>'s engine.
        /// </remarks>
        public static NDArray tensordot(NDArray a, NDArray b, int[] axesA, int[] axesB)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a, b))
                return NDMatrixBoundary.Plain((a, b, axesA, axesB), static s => tensordot(s.a, s.b, s.axesA, s.axesB));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a, b))
                return NDCharArrayBoundary.Plain((a, b, axesA, axesB), static s => tensordot(s.a, s.b, s.axesA, s.axesB));
            // NumPy's body (the axis checks, the two transposes and reshapes, the final dot) is the engine's.
            return a.TensorEngine.Tensordot(a, b, axesA, axesB);
        }
    }
}
