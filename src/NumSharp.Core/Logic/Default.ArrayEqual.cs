using NumSharp.Backends.Kernels;

// =============================================================================
// Default.ArrayEqual.cs - DefaultEngine's np.array_equal and np.array_equiv (numpy/_core/numeric.py)
// =============================================================================
//
// DefaultEngine's overrides of TensorEngine.ArrayEqual and TensorEngine.ArrayEquiv (TensorEngine.Numeric.cs): NumPy's
// statements for every dtype - the null convention, the shape (or broadcast) check, `bool((a1 == a2).all())`, and
// array_equal's equal_nan statements (the same object, the dtypes that cannot hold NaN, isnan at the same places). Text
// and NEP 55 StringDType take the same statements, answered by the == operator and np.isnan as NumPy's are. Two shortcuts
// give a statement's answer faster: the fused early-exiting equality scan, and the shapes' answer for a pair == has no
// loop for.
//
// History: the np-layer dtype plan's P2 moved these statements from the public np.array_equal / np.array_equiv and the
// member NDArray.array_equal into the internal operation class ArrayEqualEngine; its phase PE (E1) made that class this
// DefaultEngine partial. The facades and the member keep the matrix boundary's plain region and one call. The file sits
// beside the facades, not under Backends/, because its statements call the public np.* functions NumPy's call (np.all,
// np.isnan, np.are_broadcastable), which ratchet R2 counts under Backends/.
//
// No scope: every array these statements make is disposed by a `using` before the answer (a bool) is returned.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.ArrayEqual"/>: <c>array_equal(a1, a2, equal_nan)</c> after its callers'
        ///     plain region (np.array_equal's facade and the member <see cref="NDArray.array_equal(NDArray, bool)"/>): the
        ///     null convention, then NumPy's statements — the shape check; without <c>equal_nan</c>,
        ///     <c>bool((a1 == a2).all())</c>; with it, the same object is True, two dtypes that cannot hold NaN compare
        ///     plainly, and otherwise <c>isnan</c> runs on both, the NaN positions must agree and every other element must be
        ///     equal.
        /// </summary>
        /// <param name="a1">The first array, or null.</param>
        /// <param name="a2">The second array, or null.</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>: NaNs at the same places count as equal.</param>
        /// <returns>NumPy's answer: the shapes equal and every element equal. Two nulls are equal, a null and an array are
        ///     not.</returns>
        /// <exception cref="TypeError">With <paramref name="equal_nan"/>, a fixed-width string operand and two different
        ///     arrays of one shape (NumPy's <c>isnan</c> has no string loop); two <c>StringDType</c> operands whose NA objects
        ///     differ (the comparison's <c>Cannot find a compatible null string value for null strings 'nan' and
        ///     'None'</c>).</exception>
        /// <remarks>
        ///     Text and <c>StringDType</c> run these statements as they are: <c>==</c> compares two strings of one class and a
        ///     <c>StringDType</c> operand with a <c>StringDType</c> or <c>U</c> one (<c>NpyString_cmp</c>: <c>'a\0c'</c>
        ///     equals <c>'a\0b'</c>, the bytes after a NUL both share are not compared; a NaN-like NA equals nothing, itself
        ///     included; a None-like NA reads as <c>''</c>), and answers all-False for any other pair with a string operand;
        ///     <c>isnan</c> refuses <c>S</c> / <c>U</c> text and is True at a <c>StringDType</c> NaN-like NA. The default
        ///     path is <c>all(a1 == a2)</c> deliberately WITHOUT a same-reference short-circuit: NaN ≠ NaN, so an array
        ///     holding a NaN is not equal to itself; the short-circuit is valid only under <paramref name="equal_nan"/>,
        ///     where NumPy takes it. No scope: every array made here is disposed before the answer is returned.
        /// </remarks>
        public override bool ArrayEqual(NDArray a1, NDArray a2, bool equal_nan = false)
        {
            // A null "array" cannot equal a real one; two nulls are trivially equal (mirrors the NDArray == null
            // convention). Guarding here also keeps the statements below from reading a null operand's shape.
            if (a1 is null)
                return a2 is null;
            if (a2 is null)
                return false;

            // `if a1.shape != a2.shape: return False`, before anything else: array_equal does NOT broadcast (that is
            // array_equiv's job).
            if (a1.Shape != a2.Shape)
                return false;

            // `if not equal_nan: return builtins.bool((asanyarray(a1 == a2)).all())`
            if (!equal_nan)
                return AllEqual(a1, a2);

            // `if a1 is a2: return True`: NaN compares equal to itself under equal_nan, so an array equals itself.
            if (ReferenceEquals(a1, a2))
                return true;

            // `if cannot_have_nan: return builtins.bool(asarray(a1 == a2).all())`: the isnan statements below would reduce
            // to exactly this, their masks all False.
            if (!CanHoldNaN(a1) && !CanHoldNaN(a2))
                return AllEqual(a1, a2);

            // `a1nan, a2nan = isnan(a1), isnan(a2)`, in this order, so an S / U first operand is the one refused (isnan has
            // no string loop); a StringDType operand's isnan is True at a NaN-like NA. The callers' plain region covers a
            // matrix or chararray operand here.
            using var a1nan = np.isnan(a1);
            using var a2nan = np.isnan(a2);
            // `if not (a1nan == a2nan).all(): return False`: NaN must occur at the SAME positions in both operands...
            using (var samePlaces = a1nan == a2nan)
                if (!np.all(samePlaces))
                    return false;

            // `return builtins.bool((a1[~a1nan] == a2[~a1nan]).all())`: ...and every other slot must be equal. The form
            // `all((a1 == a2) | a1nan)` lets a NaN slot pass (a1nan is true there, and the positions were just proven to
            // coincide): identical once the positions agree, but it needs no gather or copy and handles empty / 0-d inputs
            // uniformly. A pair == has no loop for gives its all-False answer here as at the gathered slots, and two
            // StringDType operands with different NA objects are refused as NumPy refuses the gathered ones.
            using var elementsEq = a1 == a2;
            using var okOrNan = elementsEq | a1nan;
            return np.all(okOrNan);
        }

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.ArrayEquiv"/>: <c>array_equiv(a1, a2)</c> after the facade's plain
        ///     region: NumPy's <c>multiarray.broadcast(a1, a2)</c> check (a failure is "not equivalent"), then
        ///     <c>bool((a1 == a2).all())</c> over the broadcast operands.
        /// </summary>
        /// <param name="a1">The first array, or null.</param>
        /// <param name="a2">The second array, or null.</param>
        /// <returns>True when the shapes broadcast together and every element compares equal (NaN never does). Two nulls are
        ///     equivalent, a null and an array are not.</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ (NumPy's comparison refuses the
        ///     pair).</exception>
        /// <remarks>No scope: the one comparison array is disposed before the answer is returned.</remarks>
        public override bool ArrayEquiv(NDArray a1, NDArray a2)
        {
            // A null "array" cannot broadcast against or equal a real one; two nulls are trivially
            // equivalent (mirrors the NDArray == null convention).
            if (a1 is null || a2 is null)
                return a1 is null && a2 is null;

            // NumPy attempts multiarray.broadcast(a1, a2) and treats a failure as "not equivalent".
            // Non-broadcastable shapes short-circuit to False before any element comparison.
            if (!np.are_broadcastable(a1, a2))
                return false;

            // `return builtins.bool(asanyarray(a1 == a2).all())`
            return AllEqual(a1, a2);
        }

        /// <summary>
        ///     <c>bool((a1 == a2).all())</c> for two operands whose shapes broadcast (array_equal's are one shape): the shapes'
        ///     answer for a pair <c>==</c> has no loop for, the fused equality scan for two dense operands of one numeric
        ///     dtype and shape, and otherwise the <c>==</c> operator reduced by <c>all()</c>.
        /// </summary>
        /// <param name="a">The first operand (not null).</param>
        /// <param name="b">The second operand (not null; its shape broadcasts with <paramref name="a"/>'s).</param>
        /// <returns>Whether every element of the broadcast comparison is True (NaN never compares equal).</returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ.</exception>
        /// <remarks>
        ///     Both shortcuts give the statement's answer without its bool temporary (the plan's C11: where NumPy's composition
        ///     is slower, the faster path stays). For a pair with a string operand that <c>==</c> has no loop for (fixed-width
        ///     text against numbers or the other string class; <c>StringDType</c> against <c>S</c>, numbers or bool), NumPy's
        ///     operator answers False everywhere (<c>_failed_comparison_workaround</c>), so <c>all()</c> of it is True only for
        ///     an empty broadcast, which the shapes tell. The fused scan (<see cref="EqualityScan"/>: one early-exiting SIMD
        ///     pass, about 2x at cache-resident sizes and instant on an early mismatch, with the comparison's NaN != NaN and
        ///     -0.0 == +0.0) declines every other case: broadcasting, mixed dtypes, strided or mixed C / F layouts, Half,
        ///     Decimal and text.
        /// </remarks>
        private static bool AllEqual(NDArray a, NDArray b)
        {
            // The operators' own test (DefaultEngine.RichCompareEquality): only a pair with a string operand can lack a loop.
            if ((a.typecode == NPTypeCode.String || b.typecode == NPTypeCode.String)
                && !Backends.Strings.NDStringUfuncs.EqualHasStringLoop(a.dtype, b.dtype))
                return Shape.ResolveReturnShape(a.Shape, b.Shape).size == 0;

            if (EqualityScan.TryAllEqual(a, b, out bool fused))
                return fused;

            // The == operator broadcasts the operands to their common shape; all() then reduces to a
            // single bool. `using` reclaims the comparison result (all() only reads it).
            using var eq = a == b;
            return np.all(eq);
        }

        /// <summary>
        ///     Whether array_equal's <c>equal_nan</c> statements run <c>isnan</c> over a dtype: the float family, complex, and
        ///     text (NumPy's <c>isnan</c> refuses <c>S</c> / <c>U</c> text and finds a <c>StringDType</c> NaN-like NA). The
        ///     other NumSharp dtypes — bool, the signed and unsigned integer widths, char and decimal — cannot hold NaN, so the
        ///     statements compare them plainly.
        /// </summary>
        /// <param name="a">The array whose dtype is asked about.</param>
        /// <returns>True for float16, float32, float64, complex128 and text.</returns>
        /// <remarks>
        ///     NumPy's <c>_dtype_cannot_hold_nan</c> set is bool and the signed integers. NumSharp's also holds the unsigned
        ///     integers, whose <c>isnan</c> NumPy computes as all False (its NaN branch reduces to the same plain comparison),
        ///     and char and decimal, which have no NumPy dtype; skipping <c>isnan</c> for them costs no answer and saves its two
        ///     masks.
        /// </remarks>
        private static bool CanHoldNaN(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                // Text reaches isnan as in NumPy: _dtype_cannot_hold_nan is False for a string dtype.
                case NPTypeCode.String:
                    return true;
                default:
                    return false;
            }
        }
    }
}
