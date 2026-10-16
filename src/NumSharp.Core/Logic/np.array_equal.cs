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
        ///     NumPy's <c>array_equal</c> when an operand is NEP 55 <c>StringDType</c>, statement by statement: a shape
        ///     mismatch is False; without <c>equal_nan</c>, <c>(a1 == a2).all()</c> (<see cref="AllEqualVString"/>); with it,
        ///     the same object is True, and otherwise <c>isnan</c> runs on both (only bool and the signed integers are NumPy's
        ///     no-NaN dtypes, so a <c>StringDType</c> operand always reaches it: True at a NaN-like NA, while <c>S</c> / <c>U</c>
        ///     text refuses it), the NaN positions must agree, and the rest compare with <c>==</c>.
        /// </summary>
        /// <param name="a">The first array (not null).</param>
        /// <param name="b">The second array (not null).</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>.</param>
        /// <returns>NumPy's answer.</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ (the comparison's <c>Cannot find
        ///     a compatible null string value for null strings 'nan' and 'None'</c>); with <paramref name="equal_nan"/>, an
        ///     <c>S</c> / <c>U</c> operand (<c>isnan</c> has no string loop).</exception>
        /// <remarks>
        ///     <c>==</c> over <c>StringDType</c> is NumPy's <c>NpyString_cmp</c>: <c>'a\0c'</c> equals <c>'a\0b'</c> (the bytes
        ///     after a NUL both share are not compared), a NaN-like NA equals nothing (itself included — only
        ///     <paramref name="equal_nan"/> matches NA positions), a None-like NA reads as <c>''</c>.
        /// </remarks>
        private static bool ArrayEqualVString(NDArray a, NDArray b, bool equal_nan)
        {
            var da = a.Shape.dimensions;
            var db = b.Shape.dimensions;
            if (da.Length != db.Length)
                return false;
            for (int d = 0; d < da.Length; d++)
                if (da[d] != db[d])
                    return false;
            if (!equal_nan)
                return AllEqualVString(a, b);
            if (ReferenceEquals(a, b))
                return true;
            // `a1nan, a2nan = isnan(a1), isnan(a2)` — in this order, so an S / U first operand is the one refused.
            using var aNaN = np.isnan(a);
            using var bNaN = np.isnan(b);
            using (var samePlaces = np.equal(aNaN, bNaN))
                if (!np.all(samePlaces))
                    return false;
            // `(a1[~a1nan] == a2[~a1nan]).all()`: both selected by the FIRST operand's NaN positions.
            using var keep = np.logical_not(aNaN);
            using var aRest = a[keep];
            using var bRest = b[keep];
            return AllEqualVString(aRest, bRest);
        }

        /// <summary>
        ///     <c>bool((a1 == a2).all())</c> with a <c>StringDType</c> operand: the <c>StringDType</c> comparison (NA rules
        ///     and broadcasting included) when <c>equal</c> has a loop for the pair — <c>StringDType</c> with
        ///     <c>StringDType</c> or <c>U</c> — and otherwise NumPy's all-False array (<c>S</c>, numbers, bool), which is
        ///     True only when the broadcast is empty.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand (shapes already known to broadcast).</param>
        /// <returns>Whether every element compared equal.</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ.</exception>
        internal static bool AllEqualVString(NDArray a, NDArray b)
        {
            static bool Compares(NDArray x) => Backends.Strings.NDStringUfuncs.IsVString(x)
                                                || (x.typecode == NPTypeCode.String && x.dtype.kind == 'U');
            if (!(Compares(a) && Compares(b)))
                return Shape.ResolveReturnShape(a.Shape, b.Shape).size == 0;
            using var equal = np.equal(a, b);
            return np.all(equal);
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
        /// <exception cref="TypeError">A fixed-width string operand with <paramref name="equal_nan"/> and two different arrays
        ///     of one shape (NumPy's <c>isnan</c> has no string loop). Without it, two strings of one class compare their text
        ///     and other string pairs are answered from shape and dtype (a shape mismatch, or no <c>equal</c> loop, is False;
        ///     empty arrays are True). A <c>StringDType</c> operand compares through its own <c>==</c> (NumPy's NA rules: two
        ///     different NA objects raise its TypeError) and, with <paramref name="equal_nan"/>, its <c>isnan</c> (a NaN-like
        ///     NA is a NaN).</exception>
        public static bool array_equal(NDArray a, NDArray b, bool equal_nan = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // A null "array" cannot equal a real one; two nulls are trivially equal (mirrors the
            // NDArray == null convention). Guarding here also keeps a.array_equal(...) from throwing.
            if (a is null)
                return b is null;
            // StringDType first (IsString refuses it): NumPy's statements over the StringDType comparison and isnan.
            if (b is not null && (Backends.Strings.NDStringUfuncs.IsVString(a) || Backends.Strings.NDStringUfuncs.IsVString(b)))
                return ArrayEqualVString(a, b, equal_nan);
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
