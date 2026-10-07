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

            // NumPy: empty array comparison returns empty array, not scalar
            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            // A string operand: two strings of one class compare their text (the next phase of the string support). For
            // any other pair the not_equal ufunc has no loop, and NumPy's `!=` operator answers
            // True everywhere instead of raising (_failed_comparison_workaround): a constant of the broadcast shape.
            if (lhs.typecode == NPTypeCode.String || rhs.typecode == NPTypeCode.String)
            {
                var constant = Backends.Strings.NDStringUfuncs.FailedComparison(lhs, rhs, notEqual: true);
                if (constant is null)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("not_equal", lhs);
                return constant;
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
