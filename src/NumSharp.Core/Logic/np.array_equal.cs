using NumSharp.Backends.Kernels;

namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>array_equal</c> when an operand holds strings, statement by statement: a shape mismatch is False;
        ///     without <c>equal_nan</c>, <c>(a1 == a2).all()</c> — the <c>==</c> operator answers all-False for a pair
        ///     <c>equal</c> has no loop for (so the result is True only for empty arrays), and two strings of one class compare
        ///     their text (an empty pair compares nothing: True); with <c>equal_nan</c>, the same object is True
        ///     (<c>a1 is a2</c>), and otherwise <c>isnan</c> meets the text and refuses it.
        /// </summary>
        /// <param name="a">The first array (not null).</param>
        /// <param name="b">The second array (not null).</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>.</param>
        /// <returns>NumPy's answer.</returns>
        /// <exception cref="TypeError"><paramref name="equal_nan"/> with two different arrays of one shape (<c>isnan</c> has
        ///     no string loop).</exception>
        private static bool ArrayEqualText(NDArray a, NDArray b, bool equal_nan)
        {
            var da = a.Shape.dimensions;
            var db = b.Shape.dimensions;
            if (da.Length != db.Length)
                return false;
            for (int d = 0; d < da.Length; d++)
                if (da[d] != db[d])
                    return false;
            if (!equal_nan)
            {
                if (!Backends.Strings.NDStringUfuncs.ComparesText(a.dtype, b.dtype) || a.size == 0)
                    return a.size == 0;
                // (a1 == a2).all(): two strings of one class compare their text (the equal kernel), then the bool all().
                using var equal = a == b;
                return np.all(equal);
            }
            if (ReferenceEquals(a, b))
                return true;
            // _dtype_cannot_hold_nan is False for a string dtype, so NumPy calls isnan(a1), isnan(a2): the text one refuses.
            throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("isnan");
        }

        /// <summary>
        ///     True if two arrays have the same shape and all elements are equal, False otherwise.
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <param name="equal_nan">
        ///     When <c>false</c> (the NumPy default) NaN never compares equal, so an array that holds a NaN
        ///     is NOT equal even to itself — <c>np.array_equal(a, a)</c> is <c>false</c> whenever <paramref name="a"/>
        ///     contains a NaN. When <c>true</c>, NaNs at matching positions are treated as equal; for a complex
        ///     dtype a value counts as NaN when EITHER its real or imaginary component is NaN.
        /// </param>
        /// <returns>True if the arrays have identical shape and every element is equal.</returns>
        /// <remarks>
        ///     The shapes must match EXACTLY — unlike <see cref="array_equiv(NDArray,NDArray)"/> this performs
        ///     NO broadcasting. Mirrors <c>numpy.array_equal(a1, a2, equal_nan=False)</c> (NumPy 2.4.2).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        /// </remarks>
        /// <exception cref="TypeError">A string operand with <paramref name="equal_nan"/> and two different arrays of one shape
        ///     (NumPy's <c>isnan</c> has no string loop). Without it, two strings of one class compare their text and other
        ///     string pairs are answered from shape and dtype (a shape mismatch, or no <c>equal</c> loop, is False; empty
        ///     arrays are True).</exception>
        public static bool array_equal(NDArray a, NDArray b, bool equal_nan = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // A null "array" cannot equal a real one; two nulls are trivially equal (mirrors the
            // NDArray == null convention). Guarding here also keeps a.array_equal(...) from throwing.
            if (a is null)
                return b is null;
            // Text: NumPy's statements decide by shape and dtype until (a1 == a2).all() compares text (the next phase).
            if (b is not null && (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(b)))
                return ArrayEqualText(a, b, equal_nan);

            // Fused fast path for the common default (equal_nan=false): when both operands are the same
            // shape, same dtype and dense-contiguous in the same order, ONE early-exiting SIMD pass
            // (EqualityScan) replaces the full bool temp + separate all() reduction — ~2x at cache-resident
            // sizes and instant on an early mismatch. Its Vector.EqualsAll gives NaN != NaN and
            // -0.0 == +0.0, so the result is identical to the composition. TryAllEqual returns false for
            // every ineligible case (broadcast/mixed-dtype/strided/Half/Decimal), and equal_nan=true keeps
            // the instance path with its NaN-position logic.
            if (!equal_nan && b is not null && EqualityScan.TryAllEqual(a, b, out bool fused))
                return fused;

            return a.array_equal(b, equal_nan);
        }
    }
}
