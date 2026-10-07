using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise not-equal comparison (!=).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
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

            // A string operand: two strings of one class compare their text (the next phase of the string support). For
            // any other pair the not_equal ufunc has no loop, and NumPy's `!=` operator answers
            // True everywhere instead of raising (_failed_comparison_workaround): a constant of the broadcast shape.
            if (lhs.typecode == NPTypeCode.String || rhs.typecode == NPTypeCode.String)
            {
                var constant = Backends.Strings.NDStringUfuncs.FailedComparison(lhs, rhs, notEqual: true);
                if (constant is not null)
                    return constant;
                // Two strings of one class: the not_equal ufunc compares their text (the next phase), except for an empty
                // broadcast result, which compares nothing — the engine's string guard returns that result and raises
                // the next-phase exception otherwise.
            }

            return lhs.TensorEngine.NotEqual(lhs, rhs).AsGeneric<bool>();
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
