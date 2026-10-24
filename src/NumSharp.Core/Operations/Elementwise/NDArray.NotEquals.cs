using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>a != b</c> (<c>array_richcompare</c>): the elementwise not_equal ufunc over the broadcast operands,
        ///     as a typed bool array — NOT a reference or value-inequality test. Where the ufunc has no loop for a string
        ///     operand (text against numbers, two string classes that do not compare) the answer is True everywhere instead
        ///     of an error, as NumPy's operator gives it.
        /// </summary>
        /// <param name="lhs">The left operand; null compares as C# null (see remarks).</param>
        /// <param name="rhs">The right operand; null compares as C# null.</param>
        /// <returns>
        ///     The bool comparison in the broadcast shape; a 0-d bool for a null operand; an empty bool array of
        ///     <paramref name="lhs"/>'s shape when <paramref name="lhs"/> is uninitialized (has no shape at all).
        /// </returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        /// <remarks>
        ///     <para>Footgun: <c>if (a != b)</c> does not compile into an inequality test — the result is an array. Use
        ///     <c>!a.array_equal(b)</c> (or <c>np.any(a != b)</c>) for one bool, and <see cref="object.ReferenceEquals"/>
        ///     for identity.</para>
        ///     <para>The null answers are C#'s protocol, not NumPy's: two nulls are a 0-d False, one null a 0-d True. They and
        ///     the uninitialized-array answer stay here; the comparison itself is
        ///     <see cref="Backends.DefaultEngine.RichCompareEquality"/>, the engine's (the np-layer dtype plan, P1c).</para>
        ///     <para>Not scoped: the engine returns a bool-typed result that <c>AsGeneric</c> passes through, so this hot path
        ///     mints no temporary to reclaim (see the <c>==</c> operator).</para>
        /// </remarks>
        public static NDArray<bool> operator !=(NDArray lhs, NDArray rhs)
        {
            if (lhs is null && rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            if (lhs is null || rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            // An uninitialized array (no shape at all) compares to an empty result. A zero-size array goes to the ufunc,
            // which broadcasts it with the other operand as NumPy does: (0,3) against (2,1,3) is (2,0,3), and (0,) against
            // (3,) raises NumPy's broadcast error (this branch used to answer the left operand's shape for both).
            if (lhs.Shape.IsEmpty)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            // NumPy's array_richcompare: the not_equal ufunc, or _failed_comparison_workaround's True constant where it
            // has no loop for a string operand — the engine's (DefaultEngine.RichCompareEquality).
            return Backends.DefaultEngine.RichCompareEquality(lhs, rhs, notEqual: true);
        }

        /// <summary>
        /// Element-wise not-equal comparison with scalar (!=).
        /// </summary>
        // Scope: reclaims the np.asanyarray temp and the null/empty MakeGeneric strands while
        // yielding the typed result (see NDArray.Equals.cs). The NDArray×NDArray operator above
        // owns no temp and stays unscoped on its hot path.
        [NDScoped]
        public static NDArray<bool> operator !=(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(rhs != null).MakeGeneric<bool>();

            if (rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            return lhs != np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise not-equal comparison with scalar on left (!=).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator !=(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(lhs != null).MakeGeneric<bool>();

            if (lhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            return np.asanyarray(lhs) != rhs;
        }
    }
}
