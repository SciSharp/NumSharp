using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Determines if NDArray data is same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            unsafe
            {
                if (obj is null)
                    return false;

                if (ReferenceEquals(this, obj))
                    return true;

                // Using this comparison allows less restrictive semantics,
                // like comparing a scalar to an array
                // we can use unmanaged access because the result of == op is never a slice.
                // The comparison array is this method's own temporary: released here, or every Equals call on two
                // distinct arrays kept one pooled bool buffer until a finalizer ran.
                using var results = (this == obj);
                var len = results.size;
                var addr = results.Address;

                for (long i = 0; i < len; i++)
                    if (!addr[i])
                        return false;

                return true;
            }
        }

        /// <summary>
        ///     NumPy's <c>a == b</c> (<c>array_richcompare</c>): the elementwise equal ufunc over the broadcast operands, as a
        ///     typed bool array — NOT a reference or value-equality test. Where the ufunc has no loop for a string operand
        ///     (text against numbers, two string classes that do not compare) the answer is False everywhere instead of an
        ///     error, as NumPy's operator gives it.
        /// </summary>
        /// <param name="lhs">The left operand; null compares as C# null (see remarks).</param>
        /// <param name="rhs">The right operand; null compares as C# null.</param>
        /// <returns>
        ///     The bool comparison in the broadcast shape; a 0-d bool for a null operand; an empty bool array of
        ///     <paramref name="lhs"/>'s shape when <paramref name="lhs"/> is uninitialized (has no shape at all).
        /// </returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        /// <remarks>
        ///     <para>Footgun: <c>if (a == b)</c> does not compile into an equality test — the result is an array. Use
        ///     <see cref="array_equal(NDArray, bool)"/> (or <c>np.all(a == b)</c>) for one bool, and <see cref="object.ReferenceEquals"/>
        ///     for identity.</para>
        ///     <para>The null answers are C#'s protocol, not NumPy's (Python's <c>None</c> becomes an object array there):
        ///     two nulls are a 0-d True, one null a 0-d False. They and the uninitialized-array answer stay here; the
        ///     comparison itself is <see cref="Backends.DefaultEngine.RichCompareEquality"/>, the engine's (the np-layer
        ///     dtype plan, P1c).</para>
        ///     <para>Not scoped: the engine returns a bool-typed result that <c>AsGeneric</c> passes through, so this hot path
        ///     mints no temporary to reclaim. On the rare null and uninitialized answers MakeGeneric aliases a fresh untyped
        ///     array, which is left to the finalizer, as before.</para>
        /// </remarks>
        public static NDArray<bool> operator ==(NDArray lhs, NDArray rhs)
        {
            if (lhs is null && rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // An uninitialized array (no shape at all) compares to an empty result. A zero-size array goes to the ufunc,
            // which broadcasts it with the other operand as NumPy does: (0,3) against (2,1,3) is (2,0,3), and (0,) against
            // (3,) raises NumPy's broadcast error (this branch used to answer the left operand's shape for both).
            if (lhs.Shape.IsEmpty)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            // NumPy's array_richcompare: the equal ufunc, or _failed_comparison_workaround's False constant where it has
            // no loop for a string operand — the engine's (DefaultEngine.RichCompareEquality).
            return Backends.DefaultEngine.RichCompareEquality(lhs, rhs, notEqual: false);
        }

        /// <summary>
        /// Element-wise equal comparison with scalar (==).
        /// </summary>
        // Scope: np.asanyarray(rhs) mints a temp for a scalar/array-like operand, and the
        // null/empty early returns strand an untyped Scalar/empty wrapper under MakeGeneric —
        // [NDScoped] reclaims both while yielding the typed result (an NDArray-input passthrough
        // is never tracked, rule R2). The NDArray×NDArray operator above owns no temp (the engine
        // returns a bool-typed result AsGeneric passes through) and stays unscoped on that hot path.
        [NDScoped]
        public static NDArray<bool> operator ==(NDArray lhs, object rhs)
        {
            if (rhs is null)
                return Scalar<bool>(ReferenceEquals(lhs, null)).MakeGeneric<bool>();

            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // An uninitialized array (no shape at all) compares to an empty result. A zero-size array goes to the ufunc,
            // which broadcasts it with the other operand as NumPy does: (0,3) against (2,1,3) is (2,0,3), and (0,) against
            // (3,) raises NumPy's broadcast error (this branch used to answer the left operand's shape for both).
            if (lhs.Shape.IsEmpty)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            return lhs == np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise equal comparison with scalar on left (==).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator ==(object lhs, NDArray rhs)
        {
            if (lhs is null)
                return Scalar<bool>(ReferenceEquals(rhs, null)).MakeGeneric<bool>();

            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) == rhs;
        }

        /// <summary>
        ///     This array <c>==</c> <paramref name="np2"/> elementwise, through the <c>==</c> operator (NumPy's
        ///     <c>numpy.equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[,
        ///     signature, extobj])</c>, the <c>&lt;ufunc 'equal'&gt;</c>, with the operator's answers for a pair it has no
        ///     loop for).
        /// </summary>
        /// <param name="np2">The array to compare with (broadcast against this one).</param>
        /// <returns>A bool array of the broadcast shape: True where the elements are equal (NaN never is).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        private NDArray<bool> equal(NDArray np2)
        {
            return this == np2;
        }

        /// <summary>
        ///     True if this array and <paramref name="rhs"/> have the same shape and all elements are equal: NumSharp's
        ///     member for <c>np.array_equal(this, rhs, equal_nan)</c> (NumPy has no ndarray method of that name), with the
        ///     function's answers for every dtype, text and <c>StringDType</c> included.
        /// </summary>
        /// <param name="rhs">The array to compare against; null is never equal.</param>
        /// <param name="equal_nan">
        ///     When <c>false</c> (the NumPy default) NaN never compares equal — an array holding a NaN is NOT
        ///     equal even to itself, so <c>a.array_equal(a)</c> is <c>false</c> when <c>a</c> contains a NaN.
        ///     When <c>true</c>, NaNs at matching positions are treated as equal, and for a complex dtype a
        ///     value counts as NaN when EITHER component is NaN; a NaN-like NA of <c>StringDType</c> is a NaN.
        /// </param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>
        ///     Faithful port of NumPy 2.4.2's <c>array_equal</c>. Shape must match EXACTLY (no broadcasting;
        ///     see <see cref="np.array_equiv(NDArray,NDArray)"/> for the broadcasting variant), and the shape
        ///     check runs FIRST. The default path is <c>all(this == rhs)</c> — deliberately WITHOUT a
        ///     same-reference short-circuit, because NaN ≠ NaN means an array with a NaN is not equal to itself
        ///     here; a same-reference short-circuit is only valid under <paramref name="equal_nan"/>. Dtypes
        ///     that cannot hold NaN (bool / all integer widths / char / decimal) skip the NaN machinery: their
        ///     <c>isnan</c> is uniformly false, so the fast path yields the identical result NumPy would (this
        ///     also sidesteps calling <c>isnan</c> on Char/Decimal, which have no NumPy analog). Footgun: a
        ///     fixed-width text operand cannot take <paramref name="equal_nan"/> — NumPy's <c>isnan</c> refuses
        ///     it unless both operands are the same object (or the shapes differ).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        /// </remarks>
        /// <exception cref="TypeError">With <paramref name="equal_nan"/>, a fixed-width string operand and two different
        ///     arrays of one shape (NumPy's <c>isnan</c> has no string loop); two <c>StringDType</c> operands whose NA
        ///     objects differ (their comparison refuses the pair).</exception>
        public bool array_equal(NDArray rhs, bool equal_nan = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(this, rhs);   // the member keeps the plain-array meaning for a matrix object
            // np.array_equal's statements, in ArrayEqualEngine (Logic/ArrayEqualEngine.cs): the text and StringDType
            // routes before the numeric ones, so the member answers as the function does.
            return ArrayEqualEngine.ArrayEqual(this, rhs, equal_nan);
        }

    }
}
