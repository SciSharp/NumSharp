using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.array_equal</c> and <c>numpy.array_equiv</c> (<c>numpy/_core/numeric.py</c>), statement for
    ///     statement: the null convention, the shape (or broadcast) check, <c>bool((a1 == a2).all())</c>, and
    ///     array_equal's <c>equal_nan</c> statements (the same object, the no-NaN dtypes, <c>isnan</c> at the same places);
    ///     over text and NEP 55 <c>StringDType</c>, NumPy's statements up to the comparison or the <c>isnan</c> that answers
    ///     or refuses them; for the common case, a fused early-exiting equality scan with the composition's answer.
    /// </summary>
    /// <remarks>
    ///     An internal operation class (the np-layer dtype plan's D1: NumPy writes both functions in Python), moved from the
    ///     public <c>np.array_equal</c> / <c>np.array_equiv</c> and the member <see cref="NDArray.array_equal(NDArray, bool)"/>
    ///     by P2; the facades keep the matrix boundary's plain region and one call, the member one call. It sits beside the
    ///     facades, not under <c>Backends/</c>, because its statements call the public np.* functions NumPy's call
    ///     (np.all, np.isnan, np.equal, np.logical_not, np.are_broadcastable), which ratchet R2 counts under
    ///     <c>Backends/</c>.
    /// </remarks>
    internal static class ArrayEqualEngine
    {
        /// <summary>
        ///     <c>array_equal(a, b, equal_nan)</c> after the facade's plain region: the null convention, NumPy's statements
        ///     for a <c>StringDType</c> or fixed-width text operand, the fused equality scan for the common default, and the
        ///     numeric statements (<see cref="ArrayEqualNumeric"/>).
        /// </summary>
        /// <param name="a">The first array, or null.</param>
        /// <param name="b">The second array, or null.</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>: NaNs at the same places count as equal.</param>
        /// <returns>NumPy's answer: the shapes equal and every element equal. Two nulls are equal, a null and an array are
        ///     not.</returns>
        /// <exception cref="TypeError">A fixed-width string operand with <paramref name="equal_nan"/> and two different arrays
        ///     of one shape (NumPy's <c>isnan</c> has no string loop); two <c>StringDType</c> operands whose NA objects
        ///     differ.</exception>
        internal static bool ArrayEqual(NDArray a, NDArray b, bool equal_nan)
        {
            // A null "array" cannot equal a real one; two nulls are trivially equal (mirrors the
            // NDArray == null convention). Guarding here also keeps the numeric statements from reading a null a's shape.
            if (a is null)
                return b is null;
            // StringDType first (IsString refuses it): NumPy's statements over the StringDType comparison and isnan.
            if (b is not null && (Backends.Strings.NDStringUfuncs.IsVString(a) || Backends.Strings.NDStringUfuncs.IsVString(b)))
                return ArrayEqualVString(a, b, equal_nan);
            // Text: NumPy's statements decide by shape and dtype until (a1 == a2).all() compares the text.
            if (b is not null && (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(b)))
                return ArrayEqualText(a, b, equal_nan);

            // Fused fast path for the common default (equal_nan=false): when both operands are the same
            // shape, same dtype and dense-contiguous in the same order, ONE early-exiting SIMD pass
            // (EqualityScan) replaces the full bool temp + separate all() reduction — ~2x at cache-resident
            // sizes and instant on an early mismatch. Its Vector.EqualsAll gives NaN != NaN and
            // -0.0 == +0.0, so the result is identical to the composition. TryAllEqual returns false for
            // every ineligible case (broadcast/mixed-dtype/strided/Half/Decimal), and equal_nan=true keeps
            // the numeric statements with their NaN-position logic.
            if (!equal_nan && b is not null && EqualityScan.TryAllEqual(a, b, out bool fused))
                return fused;

            return ArrayEqualNumeric(a, b, equal_nan);
        }

        /// <summary>
        ///     <c>array_equiv(a1, a2)</c> after the facade's plain region: NumPy's <c>multiarray.broadcast(a1, a2)</c> check (a
        ///     failure is "not equivalent"), then <c>bool((a1 == a2).all())</c> through the <c>StringDType</c> comparison, the
        ///     text comparison, the fused equality scan or the broadcasting <c>==</c> operator.
        /// </summary>
        /// <param name="a1">The first array, or null.</param>
        /// <param name="a2">The second array, or null.</param>
        /// <returns>True when the shapes broadcast together and every element compares equal (NaN never does). Two nulls are
        ///     equivalent, a null and an array are not.</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ (NumPy's comparison refuses the
        ///     pair).</exception>
        internal static bool ArrayEquiv(NDArray a1, NDArray a2)
        {
            // A null "array" cannot broadcast against or equal a real one; two nulls are trivially
            // equivalent (mirrors the NDArray == null convention).
            if (a1 is null || a2 is null)
                return a1 is null && a2 is null;

            // NumPy attempts multiarray.broadcast(a1, a2) and treats a failure as "not equivalent".
            // Non-broadcastable shapes short-circuit to False before any element comparison.
            if (!np.are_broadcastable(a1, a2))
                return false;

            // StringDType first (IsString refuses it): (a1 == a2).all() through the StringDType comparison, or NumPy's
            // all-False array for a pair equal has no loop for (S, numbers, bool).
            if (Backends.Strings.NDStringUfuncs.IsVString(a1) || Backends.Strings.NDStringUfuncs.IsVString(a2))
                return AllEqualVString(a1, a2);

            // Text: (a1 == a2).all() — a pair equal has no loop for gets the operator's all-False array, so only an empty
            // broadcast is equivalent; two strings of one class compare their text (the equal kernel, broadcasting).
            if (Backends.Strings.NDStringUfuncs.IsString(a1) || Backends.Strings.NDStringUfuncs.IsString(a2))
            {
                bool empty = Shape.ResolveReturnShape(a1.Shape, a2.Shape).size == 0;
                if (empty || !Backends.Strings.NDStringUfuncs.ComparesText(a1.dtype, a2.dtype))
                    return empty;
                using var textEqual = a1 == a2;
                return np.all(textEqual);
            }

            // Equal-shape fast path: when the two operands are already the same shape (no broadcast
            // needed), same dtype and dense-contiguous, one early-exiting SIMD pass (EqualityScan)
            // replaces the bool temp + all(). Bit-identical to the composition (Vector.EqualsAll gives
            // NaN != NaN / -0.0 == +0.0). A genuine broadcast (different shapes) returns false here and
            // takes the composition below, where the == operator stretches the size-1 axes.
            if (EqualityScan.TryAllEqual(a1, a2, out bool fused))
                return fused;

            // The == operator broadcasts the operands to their common shape; all() then reduces to a
            // single bool. `using` reclaims the comparison result (all() only reads it).
            using var eq = a1 == a2;
            return np.all(eq);
        }

        /// <summary>
        ///     array_equal's numeric statements: the shape check, then <c>all(a == b)</c> without <c>equal_nan</c>; with it, the
        ///     same object is True, two dtypes that cannot hold NaN compare plainly, and otherwise the NaN positions must agree
        ///     and every other element must be equal.
        /// </summary>
        /// <param name="a">The first array (not null).</param>
        /// <param name="b">The second array, or null (never equal).</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>.</param>
        /// <returns>True if the arrays have identical shapes and every element is equal (NaNs at the same places count as
        ///     equal under <paramref name="equal_nan"/>).</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ (the <c>==</c> comparison
        ///     refuses them; reached through the member <see cref="NDArray.array_equal(NDArray, bool)"/>, which hands any
        ///     operand here).</exception>
        /// <remarks>
        ///     Shape must match EXACTLY (no broadcasting; see <see cref="ArrayEquiv"/> for the broadcasting variant), and the
        ///     shape check runs FIRST. The default path is <c>all(a == b)</c> — deliberately WITHOUT a same-reference
        ///     short-circuit, because NaN ≠ NaN means an array with a NaN is not equal to itself here; a same-reference
        ///     short-circuit is only valid under <paramref name="equal_nan"/>. Dtypes that cannot hold NaN (bool / all
        ///     integer widths / char / decimal, and text) skip the NaN machinery: their <c>isnan</c> is uniformly false for
        ///     the numbers, so the plain comparison yields the identical result NumPy would (this also sidesteps calling
        ///     <c>isnan</c> on Char/Decimal, which have no NumPy analog). The member reaches this method with a text operand
        ///     as well, which <see cref="ArrayEqual"/> routes elsewhere: with <paramref name="equal_nan"/> such a pair is
        ///     compared here, where NumPy's <c>isnan</c> refuses it.
        ///     Scoped by <c>[NDScoped]</c>: every comparison/isnan mask the method builds is released on return
        ///     (it returns a <c>bool</c>, so nothing is yielded), and the operands, built before the scope
        ///     opened, are never touched.
        /// </remarks>
        [NDScoped]
        internal static bool ArrayEqualNumeric(NDArray a, NDArray b, bool equal_nan)
        {
            // A null operand can never match a real array.
            if (b is null)
                return false;

            // NumPy checks shape equality before anything else; differing shapes are never equal
            // (array_equal does NOT broadcast — that is array_equiv's job).
            if (a.Shape != b.Shape)
                return false;

            // Default (equal_nan == false): pure element-wise equality reduced with all(). NaN != NaN,
            // so an array containing NaN is not equal even to itself — which is exactly why NO
            // reference/storage short-circuit is taken here (one would wrongly report such arrays equal).
            if (!equal_nan)
            {
                var eq = a == b;
                return np.all(eq);
            }

            // equal_nan == true from here on.
            // Same object: NaN compares equal to itself, so an array always equals itself.
            if (ReferenceEquals(a, b))
                return true;

            // Dtypes that cannot hold NaN take the plain comparison: their isnan is all-false, so the
            // NaN-aware branch below would reduce to exactly this. Result-identical to NumPy's structure.
            if (!CanHoldNaN(a) && !CanHoldNaN(b))
            {
                var eq = a == b;
                return np.all(eq);
            }

            // NaN-aware path: NaN must occur at the SAME positions in both operands...
            using var plain = NDMatrixBoundary.PlainFor(a);   // the member keeps the plain-array meaning for a matrix object
            var a1nan = np.isnan(a);
            var a2nan = np.isnan(b);
            var nanPositionsEq = a1nan == a2nan;
            if (!np.all(nanPositionsEq))
                return false;

            // ...and every finite slot must be equal, while a NaN slot passes (a1nan is true there, and
            // the positions were just proven to coincide). This `all((a == b) | isnan(a))` form is
            // algebraically identical to NumPy's `all(a[~a1nan] == b[~a1nan])` once the positions
            // agree, but needs no gather/copy and handles empty / 0-d inputs uniformly.
            var elementsEq = a == b;
            var okOrNan = elementsEq | a1nan;
            return np.all(okOrNan);
        }

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
        ///     Whether a dtype's values can be NaN: the float family plus complex. Every other NumSharp dtype — bool, the
        ///     signed/unsigned integer widths, char, decimal and text — cannot, so <see cref="ArrayEqualNumeric"/>'s
        ///     <c>equal_nan</c> path skips <c>isnan</c> for them.
        /// </summary>
        /// <param name="a">The array whose dtype is asked about.</param>
        /// <returns>True for float16, float32, float64 and complex128.</returns>
        /// <remarks>
        ///     Wider than NumPy's <c>_dtype_cannot_hold_nan</c> set (bool and the signed integers): it adds the unsigned
        ///     integers, whose <c>isnan</c> NumPy computes as all False (so its NaN branch reduces to the same plain
        ///     comparison), char and decimal (no NumPy dtype), and text, whose <c>isnan</c> NumPy refuses
        ///     (<see cref="ArrayEqual"/> sends text to its own statements before it gets here).
        /// </remarks>
        private static bool CanHoldNaN(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    return true;
                default:
                    return false;
            }
        }
    }
}
