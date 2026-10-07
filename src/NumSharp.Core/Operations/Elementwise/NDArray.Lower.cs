using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise less-than comparison (&lt;).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator <(NDArray lhs, NDArray rhs)
        {
            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // An uninitialized array (no shape at all) compares to an empty result. A zero-size array goes to the ufunc,
            // which broadcasts it with the other operand as NumPy does: (0,3) against (2,1,3) is (2,0,3), and (0,) against
            // (3,) raises NumPy's broadcast error (this branch used to answer the left operand's shape for both).
            if (lhs.Shape.IsEmpty)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            return lhs.TensorEngine.Less(lhs, rhs).AsGeneric<bool>();
        }

        /// <summary>
        /// Element-wise less-than comparison with scalar (&lt;).
        /// </summary>
        // Scope: reclaims the np.asanyarray temp and the null-branch MakeGeneric strand while
        // yielding the typed result (see NDArray.Equals.cs). The NDArray×NDArray operator above
        // owns no temp and stays unscoped on its hot path.
        [NDScoped]
        public static NDArray<bool> operator <(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs < np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise less-than comparison with scalar on left (&lt;).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator <(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) < rhs;
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison (&lt;=).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator <=(NDArray lhs, NDArray rhs)
        {
            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // An uninitialized array (no shape at all) compares to an empty result. A zero-size array goes to the ufunc,
            // which broadcasts it with the other operand as NumPy does: (0,3) against (2,1,3) is (2,0,3), and (0,) against
            // (3,) raises NumPy's broadcast error (this branch used to answer the left operand's shape for both).
            if (lhs.Shape.IsEmpty)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            return lhs.TensorEngine.LessEqual(lhs, rhs).AsGeneric<bool>();
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison with scalar (&lt;=).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator <=(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs <= np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison with scalar on left (&lt;=).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator <=(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) <= rhs;
        }
    }
}
